"""
demo.py
=======
Markov Deterioration Hazard Model x FedAvg Demo  (500-client edition)

Scenario:
  - 500 municipalities each hold bridge inspection records (data stays local)
  - Each round: CLIENT_FRACTION of clients are randomly sampled (Partial Participation)
  - Client variation: region type, bridge count, data quality, local beta noise
  - FedAvg trains for N_ROUNDS rounds
  - Global model outputs benchmark transition probabilities
  - Learning curves + heatmaps saved to output/

Usage:
    python demo.py
"""

from __future__ import annotations

import os
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.data_utils import generate_synthetic_data
from src.client import FedAvgClient
from src.server import FedAvgServer


# ======================== Config ===================================== #
N_CLIENTS       = 4000   # total registered clients
CLIENT_FRACTION = 0.10   # fraction sampled per round (400 clients / round)
N_ROUNDS        = 50     # federated rounds
SERVER_LR       = 0.05
N_LOCAL_STEPS   = 3
LOCAL_LR        = 0.01
BATCH_SIZE      = 32
RANDOM_SEED     = 2024

# Ground-truth betas (population mean — each client perturbs slightly)
TRUE_BETAS = np.array([
    [-2.0,  0.5, -0.3,  0.10],   # 0->1
    [-4.0,  0.3, -0.5,  0.05],   # 0->2
    [-2.5,  0.4, -0.4,  0.08],   # 1->2
])

# Region types: (label, sea_distance_range_km, bridge_count_range, beta_noise_scale)
REGION_TYPES = [
    ("coastal",   (0.2,  5.0),  (10,  80),  0.20),  # 30% — near sea, salt damage
    ("riverside", (5.0, 20.0),  ( 5,  50),  0.15),  # 30% — moderate environment
    ("inland",    (15., 60.0),  ( 3,  30),  0.10),  # 40% — far from sea, mild
]
REGION_PROBS = [0.30, 0.30, 0.40]

TRANSITION_LABELS = ["0->1 (good->minor)", "0->2 (good->severe)", "1->2 (minor->severe)"]
COEF_LABELS = ["b0", "b1(age)", "b2(sea)", "b3(area)"]

OUTPUT_DIR = os.path.join(ROOT, "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ======================== Data generation ============================ #
def _perturb_betas(base: np.ndarray, noise_scale: float, rng: np.random.Generator) -> np.ndarray:
    """Add Gaussian noise to base betas to simulate local heterogeneity."""
    return base + rng.normal(0.0, noise_scale, size=base.shape)


def create_clients(n_clients: int) -> list[FedAvgClient]:
    """
    Generate n_clients with varied attributes:
      - Region type  (coastal / riverside / inland)
      - Bridge count (heavy-tailed: log-normal)
      - n_inspections (2-5, data quality variation)
      - local beta noise (heterogeneous deterioration environment)
    Only a compact summary is printed (not one line per client).
    """
    clients = []
    rng = np.random.default_rng(RANDOM_SEED)
    region_counts = {r[0]: 0 for r in REGION_TYPES}
    skipped = 0

    for uid in range(n_clients):
        user_id = f"U{uid:04d}"

        # --- region assignment ---
        r_idx = rng.choice(len(REGION_TYPES), p=REGION_PROBS)
        r_label, sea_range, bridge_range, noise_scale = REGION_TYPES[r_idx]

        # bridge count: log-normal for heavy-tail (some large, many small)
        lo, hi = bridge_range
        n_bridges = int(np.clip(
            np.exp(rng.normal(np.log((lo + hi) / 2), 0.5)),
            lo, hi
        ))

        # data quality: inspections per bridge (2-5)
        n_insp = int(rng.integers(2, 6))

        # member count (1-3)
        n_members = int(rng.integers(1, 4))

        # local beta perturbation
        local_betas = _perturb_betas(TRUE_BETAS, noise_scale, rng)

        df = generate_synthetic_data(
            user_id=user_id,
            n_bridges=n_bridges,
            n_members=n_members,
            n_inspections=n_insp,
            true_betas=local_betas,
            sea_distance_range=sea_range,
            random_seed=uid * 13 + 7,
        )

        try:
            client = FedAvgClient(
                user_id=user_id,
                df=df,
                batch_size=BATCH_SIZE,
                n_local_steps=N_LOCAL_STEPS,
                lr=LOCAL_LR,
            )
            clients.append(client)
            region_counts[r_label] += 1
        except ValueError:
            skipped += 1

    # --- summary printout ---
    n_samples_list = [c.n_samples for c in clients]
    print(f"  Total clients   : {len(clients)}  (skipped: {skipped})")
    print(f"  Region breakdown: "
          + "  ".join(f"{k}={v}" for k, v in region_counts.items()))
    print(f"  Transition pairs: "
          f"min={min(n_samples_list)}  "
          f"median={int(np.median(n_samples_list))}  "
          f"max={max(n_samples_list)}  "
          f"total={sum(n_samples_list):,}")
    return clients


# ======================== FL loop ==================================== #
def run_federated_learning(
    clients: list[FedAvgClient],
    server: FedAvgServer,
    n_rounds: int,
    client_fraction: float = 1.0,
    seed: int = 0,
) -> None:
    """
    FedAvg main loop with Partial Participation.

    Each round, `client_fraction` of all clients are randomly sampled
    to compute local gradients.  This simulates realistic FL where not
    every municipality responds every round.
    """
    n_participants = max(1, int(len(clients) * client_fraction))
    rng = np.random.default_rng(seed)

    print(f"\n{'='*65}")
    print(f"  FedAvg training : {len(clients)} clients  x  {n_rounds} rounds")
    print(f"  Per-round sample: {n_participants} clients "
          f"({client_fraction*100:.0f}% partial participation)")
    print(f"{'='*65}")

    for r in range(n_rounds):
        global_beta = server.broadcast()

        # --- partial participation: sample clients for this round ---
        indices = rng.choice(len(clients), size=n_participants, replace=False)
        selected = [clients[i] for i in indices]

        updates = [c.local_gradient(global_beta) for c in selected]
        log = server.aggregate_and_update(updates, round_idx=r + 1)

        if (r + 1) % 5 == 0 or r == 0:
            print(
                f"  Round {log.round_idx:>3d} | "
                f"Participants={log.n_clients:>3d} | "
                f"Samples={log.n_total_samples:>6d} | "
                f"Avg NLL={log.avg_nll:.4f} | "
                f"||grad||={log.grad_norm:.6f}"
            )

    print(f"\nTraining complete ({n_rounds} rounds)")


# ======================== Result display ============================= #
def print_beta_comparison(server: FedAvgServer) -> None:
    """Compare true beta vs. learned beta."""
    learned = server.global_beta.reshape(3, 4).numpy()

    print(f"\n{'-'*75}")
    print("Beta parameter comparison (true vs. learned)")
    print(f"{'-'*75}")
    print(f"{'Transition':>22}  {'Coef':>10}  {'True':>10}  {'Learned':>10}  {'Error':>10}")
    print(f"{'-'*75}")

    for t_idx, t_label in enumerate(TRANSITION_LABELS):
        for c_idx, c_label in enumerate(COEF_LABELS):
            true_val = TRUE_BETAS[t_idx, c_idx]
            learned_val = learned[t_idx, c_idx]
            error = learned_val - true_val
            marker = " <--" if abs(error) > 0.5 else ""
            print(
                f"{t_label[:22]:>22}  "
                f"{c_label:>10}  "
                f"{true_val:>10.4f}  "
                f"{learned_val:>10.4f}  "
                f"{error:>+10.4f}{marker}"
            )
    print(f"{'-'*75}\n")


def print_benchmark_transition_probs(server: FedAvgServer) -> None:
    """Output benchmark transition probabilities for representative scenarios."""
    print("Benchmark transition probabilities (global model)")
    print(f"{'-'*60}")

    scenarios = [
        {
            "label": "Young bridge, far from sea, small  (z=[0.2, 0.8, 0.1])",
            "z": torch.tensor([0.2, 0.8, 0.1]),
            "delta_t": 3.0,
        },
        {
            "label": "Mid-age bridge, near sea, medium   (z=[0.5, 0.3, 0.5])",
            "z": torch.tensor([0.5, 0.3, 0.5]),
            "delta_t": 3.0,
        },
        {
            "label": "Old bridge, near sea, large         (z=[0.9, 0.1, 0.9])",
            "z": torch.tensor([0.9, 0.1, 0.9]),
            "delta_t": 3.0,
        },
    ]

    for sc in scenarios:
        print(f"\n  Scenario: {sc['label']}  dt={sc['delta_t']}yr")
        for state_from in [0, 1]:
            probs = server.benchmark_transition_prob(
                state_from=state_from,
                z_normalized=sc["z"],
                delta_t=sc["delta_t"],
            )
            prob_str = "  ".join(f"->{k}: {v:.4f}" for k, v in sorted(probs.items()))
            state_name = ["good", "minor"][state_from]
            print(f"    State {state_from} ({state_name}): {prob_str}")

    print()


# ======================== Visualization ============================== #
def plot_learning_curves(server: FedAvgServer, save_path: str) -> None:
    """Plot NLL learning curve and beta convergence."""
    history = server.history
    rounds = [h.round_idx for h in history]
    nlls = [h.avg_nll for h in history]
    gnorms = [h.grad_norm for h in history]
    betas_over_rounds = np.array(
        [h.beta_snapshot.numpy().flatten() for h in history]
    )  # (n_rounds, 12)

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    fig.suptitle(
        f"FedAvg: Markov Hazard Model  [{N_CLIENTS} clients, {int(CLIENT_FRACTION*100)}% participation/round]",
        fontsize=12, fontweight="bold"
    )

    # NLL
    ax = axes[0, 0]
    ax.plot(rounds, nlls, "b-o", markersize=3, linewidth=1.5, label="Avg NLL")
    ax.set_xlabel("Round")
    ax.set_ylabel("Avg NLL")
    ax.set_title("Negative Log-Likelihood Convergence")
    ax.grid(alpha=0.3)
    ax.legend()

    # Gradient norm
    ax = axes[0, 1]
    ax.semilogy(rounds, gnorms, "r-s", markersize=3, linewidth=1.5)
    ax.set_xlabel("Round")
    ax.set_ylabel("||grad|| (log scale)")
    ax.set_title("Aggregated Gradient Norm")
    ax.grid(alpha=0.3)

    # Beta convergence per transition
    panel_map = {0: axes[1, 0], 1: axes[1, 0], 2: axes[1, 1]}
    for t_idx, t_label in enumerate(TRANSITION_LABELS):
        ax = panel_map[t_idx]
        for c_idx, c_label in enumerate(COEF_LABELS):
            param_idx = t_idx * 4 + c_idx
            ax.plot(
                rounds,
                betas_over_rounds[:, param_idx],
                label=f"{c_label} ({t_label[:4]})",
                linewidth=1.3,
            )
            ax.axhline(
                TRUE_BETAS[t_idx, c_idx],
                linestyle="--",
                alpha=0.35,
                color=f"C{c_idx}",
            )

    for ax, title in zip(
        [axes[1, 0], axes[1, 1]],
        ["Beta convergence: 0->1 / 0->2", "Beta convergence: 1->2"],
    ):
        ax.set_xlabel("Round")
        ax.set_ylabel("Beta value")
        ax.set_title(title)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7)

    plt.tight_layout()
    plt.savefig(save_path, dpi=120, bbox_inches="tight")
    print(f"Learning curves saved: {save_path}")


def plot_benchmark_heatmap(server: FedAvgServer, save_path: str) -> None:
    """
    3x3 grid heatmap showing how each pair of covariates affects
    each specific transition probability.

    Rows    : transitions  0->1 | 0->2 | 1->2
    Columns : covariate pairs  (z1,z2) | (z1,z3) | (z2,z3)
              (the remaining covariate is fixed at 0.5)
    dt = 3.0 yr for all cells.
    """
    GRID = 30       # resolution per axis
    DT = 3.0
    MID = 0.5       # fixed value for the 3rd covariate
    vs = np.linspace(0.0, 1.0, GRID)

    # --- transition spec: (state_from, target_state, row_label) ---
    transitions = [
        (0, 1, "0->1  (good -> minor)"),
        (0, 2, "0->2  (good -> severe)"),
        (1, 2, "1->2  (minor -> severe)"),
    ]

    # --- covariate-pair spec: (x_idx, y_idx, x_label, y_label, fixed_idx) ---
    pairs = [
        (0, 1, "z1: age", "z2: sea dist", 2),
        (0, 2, "z1: age", "z3: area",     1),
        (1, 2, "z2: sea dist", "z3: area", 0),
    ]
    pair_labels = ["z1 x z2\n(area=0.5)", "z1 x z3\n(sea=0.5)", "z2 x z3\n(age=0.5)"]

    # --- pre-compute all 9 matrices ---
    matrices = {}   # (row, col) -> ndarray(GRID, GRID)
    for ri, (sf, tj, _) in enumerate(transitions):
        for ci, (xi, yi, xl, yl, fi) in enumerate(pairs):
            mat = np.zeros((GRID, GRID))
            for iy, vy in enumerate(vs):
                for ix, vx in enumerate(vs):
                    z_vals = [MID, MID, MID]
                    z_vals[xi] = vx
                    z_vals[yi] = vy
                    z = torch.tensor(z_vals, dtype=torch.float32)
                    probs = server.benchmark_transition_prob(
                        state_from=sf, z_normalized=z, delta_t=DT
                    )
                    mat[iy, ix] = probs.get(tj, 0.0)
            matrices[(ri, ci)] = mat

    # --- compute per-row vmax so each row shares the same scale ---
    row_vmax = {}
    for ri in range(3):
        row_vmax[ri] = max(matrices[(ri, ci)].max() for ci in range(3))
        row_vmax[ri] = max(row_vmax[ri], 1e-6)   # avoid zero scale

    # --- plot ---
    fig, axes = plt.subplots(3, 3, figsize=(13, 11))
    fig.suptitle(
        f"Benchmark Transition Probability Heatmaps  (dt={DT} yr)\n"
        "Rows: transition | Columns: covariate pair (3rd var fixed at 0.5)",
        fontsize=12, fontweight="bold"
    )

    for ri, (sf, tj, row_label) in enumerate(transitions):
        for ci, (xi, yi, xl, yl, fi) in enumerate(pairs):
            ax = axes[ri, ci]
            mat = matrices[(ri, ci)]
            im = ax.imshow(
                mat,
                origin="lower",
                aspect="auto",
                extent=[0, 1, 0, 1],
                cmap="YlOrRd",
                vmin=0,
                vmax=row_vmax[ri],
            )
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04,
                         label=f"P({sf}->{tj})" if ci == 2 else "")

            # contour for readability
            levels = np.linspace(0, row_vmax[ri], 6)[1:]
            ax.contour(
                np.linspace(0, 1, GRID),
                np.linspace(0, 1, GRID),
                mat,
                levels=levels,
                colors="white",
                linewidths=0.6,
                alpha=0.6,
            )

            ax.set_xlabel(xl, fontsize=8)
            ax.set_ylabel(yl, fontsize=8)
            ax.tick_params(labelsize=7)

            # column header on top row
            if ri == 0:
                ax.set_title(pair_labels[ci], fontsize=9)

            # row label on left column
            if ci == 0:
                ax.set_ylabel(f"{row_label}\n\n{yl}", fontsize=8)

    plt.tight_layout()
    plt.savefig(save_path, dpi=130, bbox_inches="tight")
    print(f"Heatmap saved: {save_path}")


# ======================== Main ======================================= #
def main() -> None:
    print("=" * 65)
    print(f"  Markov Deterioration Hazard x FedAvg Demo  [{N_CLIENTS}-client]")
    print("  Bridge Inspection Records - Deterioration Transition Estimation")
    print("=" * 65)

    # 1. Create clients
    print(f"\n[Step 1] Generate synthetic data ({N_CLIENTS} municipalities)")
    clients = create_clients(N_CLIENTS)

    # 2. Initialize server
    server = FedAvgServer(
        lr=SERVER_LR,
        momentum=0.9,
        grad_clip_norm=1.0,
    )

    # 3. FedAvg training with partial participation
    print(f"\n[Step 2] FedAvg Training")
    run_federated_learning(
        clients,
        server,
        n_rounds=N_ROUNDS,
        client_fraction=CLIENT_FRACTION,
        seed=RANDOM_SEED,
    )

    # 4. Results
    print(f"\n[Step 3] Results")
    server.print_beta_table()
    print_beta_comparison(server)
    print_benchmark_transition_probs(server)

    # 5. Visualization
    print(f"\n[Step 4] Visualization")
    plot_learning_curves(
        server,
        save_path=os.path.join(OUTPUT_DIR, "learning_curves.png"),
    )
    plot_benchmark_heatmap(
        server,
        save_path=os.path.join(OUTPUT_DIR, "benchmark_heatmap.png"),
    )

    # 6. Checkpoint
    server.save_checkpoint(os.path.join(OUTPUT_DIR, "global_model.pt"))

    print("\n[Done] All outputs saved to output/")
    print("=" * 65)


if __name__ == "__main__":
    main()
