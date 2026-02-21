"""
compare_scales.py
=================
Run FedAvg experiments for N_CLIENTS in [500, 2000, 4000] under identical settings
and produce a single 2x2 comparison figure saved to output/scale_comparison.png.

Panels
------
  [0,0]  NLL convergence curves  (3 lines)
  [0,1]  Aggregated gradient-norm convergence  (log scale, 3 lines)
  [1,0]  Beta estimation error  (MAE per transition, grouped bar)
  [1,1]  Final-round convergence metrics  (bar: NLL and grad-norm side-by-side)

Usage
-----
    python compare_scales.py
"""

from __future__ import annotations

import os
import sys
import time

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

# ======================== Shared config ============================== #
SCALES          = [500, 2000, 4000]
CLIENT_FRACTION = 0.10
N_ROUNDS        = 50
SERVER_LR       = 0.05
N_LOCAL_STEPS   = 3
LOCAL_LR        = 0.01
BATCH_SIZE      = 32
RANDOM_SEED     = 2024

TRUE_BETAS = np.array([
    [-2.0,  0.5, -0.3,  0.10],   # 0->1
    [-4.0,  0.3, -0.5,  0.05],   # 0->2
    [-2.5,  0.4, -0.4,  0.08],   # 1->2
])

REGION_TYPES = [
    ("coastal",   (0.2,  5.0),  (10, 80),  0.20),
    ("riverside", (5.0, 20.0),  ( 5, 50),  0.15),
    ("inland",    (15., 60.0),  ( 3, 30),  0.10),
]
REGION_PROBS = [0.30, 0.30, 0.40]

TRANSITION_LABELS = ["0->1", "0->2", "1->2"]
COEF_LABELS       = ["b0", "b1(age)", "b2(sea)", "b3(area)"]

OUTPUT_DIR = os.path.join(ROOT, "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ======================== Client factory ============================= #
def _perturb_betas(base: np.ndarray, noise_scale: float,
                   rng: np.random.Generator) -> np.ndarray:
    return base + rng.normal(0.0, noise_scale, size=base.shape)


def create_clients(n_clients: int) -> list[FedAvgClient]:
    clients = []
    rng = np.random.default_rng(RANDOM_SEED)
    skipped = 0

    for uid in range(n_clients):
        user_id = f"U{uid:04d}"

        r_idx = rng.choice(len(REGION_TYPES), p=REGION_PROBS)
        r_label, sea_range, bridge_range, noise_scale = REGION_TYPES[r_idx]

        lo, hi = bridge_range
        n_bridges = int(np.clip(
            np.exp(rng.normal(np.log((lo + hi) / 2), 0.5)),
            lo, hi
        ))

        n_insp     = int(rng.integers(2, 6))
        n_members  = int(rng.integers(1, 4))
        local_betas = _perturb_betas(TRUE_BETAS, noise_scale, rng)

        try:
            df = generate_synthetic_data(
                user_id=user_id,
                n_bridges=n_bridges,
                n_members=n_members,
                n_inspections=n_insp,
                true_betas=local_betas,
                sea_distance_range=sea_range,
                random_seed=uid * 13 + 7,
            )
        except Exception:
            skipped += 1
            continue

        try:
            client = FedAvgClient(
                user_id=user_id,
                df=df,
                batch_size=BATCH_SIZE,
                n_local_steps=N_LOCAL_STEPS,
                lr=LOCAL_LR,
            )
            clients.append(client)
        except ValueError:
            skipped += 1

    print(f"  Created {len(clients)} clients  (skipped {skipped})")
    return clients


# ======================== FedAvg loop ================================ #
def run_federated_learning(
    clients: list[FedAvgClient],
    server: FedAvgServer,
    n_rounds: int,
    client_fraction: float,
    seed: int,
) -> None:
    rng = np.random.default_rng(seed)
    n_sample = max(1, int(len(clients) * client_fraction))

    for r in range(n_rounds):
        global_beta = server.broadcast()
        indices = rng.choice(len(clients), size=n_sample, replace=False)
        sampled_clients = [clients[i] for i in indices]

        updates = [c.local_gradient(global_beta) for c in sampled_clients]
        server.aggregate_and_update(updates, round_idx=r + 1)


# ======================== Single experiment ========================== #
def run_experiment(n_clients: int) -> dict:
    """
    Run one complete FedAvg experiment and return result dict with:
      - rounds, nlls, gnorms  (lists, length = N_ROUNDS)
      - learned_betas         (np.ndarray shape 3x4)
      - elapsed_sec           (float)
    """
    print(f"\n{'='*55}")
    print(f"  Experiment: N_CLIENTS = {n_clients}")
    print(f"  Sampled per round: {max(1, int(n_clients * CLIENT_FRACTION))}")
    print(f"{'='*55}")

    t0 = time.perf_counter()

    clients = create_clients(n_clients)
    server = FedAvgServer(lr=SERVER_LR, momentum=0.9, grad_clip_norm=1.0)

    run_federated_learning(
        clients, server,
        n_rounds=N_ROUNDS,
        client_fraction=CLIENT_FRACTION,
        seed=RANDOM_SEED,
    )

    elapsed = time.perf_counter() - t0

    history = server.history
    rounds  = [h.round_idx   for h in history]
    nlls    = [h.avg_nll     for h in history]
    gnorms  = [h.grad_norm   for h in history]

    learned_flat = server.global_beta.numpy()   # (12,)
    learned_betas = learned_flat.reshape(3, 4)

    final_nll   = nlls[-1]   if nlls   else float("nan")
    final_gnorm = gnorms[-1] if gnorms else float("nan")
    print(f"  Final NLL={final_nll:.4f}  ||grad||={final_gnorm:.4f}"
          f"  elapsed={elapsed:.1f}s")

    return {
        "n_clients":     n_clients,
        "rounds":        rounds,
        "nlls":          nlls,
        "gnorms":        gnorms,
        "learned_betas": learned_betas,
        "elapsed_sec":   elapsed,
    }


# ======================== Comparison figure ========================== #
def plot_comparison(results: list[dict], save_path: str) -> None:
    """
    2x2 comparison figure:
      [0,0]  NLL convergence curves
      [0,1]  Gradient-norm convergence (log scale)
      [1,0]  Beta MAE per transition  (grouped bar, 3 scales x 3 transitions)
      [1,1]  Final-round summary bar  (NLL left axis, grad-norm right axis)
    """
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]   # blue / orange / green
    markers = ["o", "s", "^"]
    style_kw = dict(markersize=3, linewidth=1.5, markevery=5)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(
        "FedAvg Scale Comparison: 500 / 2 000 / 4 000 Clients\n"
        f"(10% partial participation, {N_ROUNDS} rounds, K={N_LOCAL_STEPS} local steps)",
        fontsize=12, fontweight="bold"
    )

    # ---- [0,0] NLL convergence ----
    ax = axes[0, 0]
    for res, c, m in zip(results, colors, markers):
        label = f"{res['n_clients']:,} clients"
        ax.plot(res["rounds"], res["nlls"], color=c, marker=m,
                label=label, **style_kw)
    ax.set_xlabel("Round")
    ax.set_ylabel("Avg NLL")
    ax.set_title("NLL Convergence")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    # ---- [0,1] Gradient norm (log) ----
    ax = axes[0, 1]
    for res, c, m in zip(results, colors, markers):
        label = f"{res['n_clients']:,} clients"
        ax.semilogy(res["rounds"], res["gnorms"], color=c, marker=m,
                    label=label, **style_kw)
    ax.set_xlabel("Round")
    ax.set_ylabel("||grad|| (log scale)")
    ax.set_title("Aggregated Gradient Norm")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3, which="both")

    # ---- [1,0] Beta MAE per transition (grouped bar) ----
    ax = axes[1, 0]
    n_transitions = len(TRANSITION_LABELS)
    bar_w = 0.22
    x = np.arange(n_transitions)

    for k, (res, c) in enumerate(zip(results, colors)):
        mae_per_transition = np.mean(
            np.abs(res["learned_betas"] - TRUE_BETAS), axis=1
        )   # shape (3,)
        offset = (k - 1) * bar_w
        bars = ax.bar(x + offset, mae_per_transition, width=bar_w,
                      color=c, alpha=0.85,
                      label=f"{res['n_clients']:,} clients")
        for bar, val in zip(bars, mae_per_transition):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.003,
                f"{val:.3f}",
                ha="center", va="bottom", fontsize=7
            )

    ax.set_xticks(x)
    ax.set_xticklabels(TRANSITION_LABELS)
    ax.set_ylabel("Mean Absolute Error (beta)")
    ax.set_title("Beta Estimation Error per Transition\n(MAE from population-mean true beta)")
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    ax.set_ylim(bottom=0)

    # ---- [1,1] Final-round summary (dual y-axis bar) ----
    ax = axes[1, 1]
    ax2 = ax.twinx()

    scale_labels  = [f"{r['n_clients']:,}" for r in results]
    final_nlls    = [r["nlls"][-1]   for r in results]
    final_gnorms  = [r["gnorms"][-1] for r in results]
    elapsed_times = [r["elapsed_sec"] for r in results]

    x_pos = np.arange(len(results))
    bw = 0.35

    b1 = ax.bar(x_pos - bw / 2, final_nlls,    width=bw,
                color=colors, alpha=0.75, label="Final NLL")
    b2 = ax2.bar(x_pos + bw / 2, final_gnorms, width=bw,
                 color=colors, alpha=0.40, hatch="//", label="Final ||grad||")

    # annotate bars
    for bar, val in zip(b1, final_nlls):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.002,
                f"{val:.3f}", ha="center", va="bottom", fontsize=8)
    for bar, val in zip(b2, final_gnorms):
        ax2.text(bar.get_x() + bar.get_width() / 2,
                 bar.get_height() + 0.002,
                 f"{val:.3f}", ha="center", va="bottom", fontsize=8)

    # elapsed annotation below x-axis
    for i, (pos, t) in enumerate(zip(x_pos, elapsed_times)):
        ax.text(pos, -0.015, f"({t:.0f}s)",
                ha="center", va="top",
                transform=ax.get_xaxis_transform(),
                fontsize=7, color="gray")

    ax.set_xticks(x_pos)
    ax.set_xticklabels([f"{lb}\nclients" for lb in scale_labels])
    ax.set_ylabel("Final Avg NLL", color="black")
    ax2.set_ylabel("Final ||grad||", color="gray")
    ax2.tick_params(axis="y", colors="gray")
    ax.set_title("Final Round: NLL and Gradient Norm\n(wall-clock time in parentheses)")
    ax.yaxis.grid(alpha=0.3)

    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, fontsize=9, loc="upper right")

    plt.tight_layout()
    plt.savefig(save_path, dpi=130, bbox_inches="tight")
    print(f"\nComparison figure saved: {save_path}")


# ======================== Main ======================================= #
def main() -> None:
    print("=" * 55)
    print("  FedAvg Scale Comparison")
    print(f"  Scales: {SCALES}")
    print(f"  Rounds: {N_ROUNDS}  |  Local steps K={N_LOCAL_STEPS}")
    print("=" * 55)

    results = []
    for n_clients in SCALES:
        res = run_experiment(n_clients)
        results.append(res)

    save_path = os.path.join(OUTPUT_DIR, "scale_comparison.png")
    plot_comparison(results, save_path)

    # Print summary table
    print("\n" + "=" * 60)
    print(f"  {'N_CLIENTS':>10}  {'Final NLL':>10}  {'||grad||':>10}  {'Time(s)':>8}")
    print(f"  {'-'*55}")
    for r in results:
        print(f"  {r['n_clients']:>10,}  {r['nlls'][-1]:>10.4f}"
              f"  {r['gnorms'][-1]:>10.4f}  {r['elapsed_sec']:>8.1f}")
    print("=" * 60)
    print("\n[Done] output/scale_comparison.png")


if __name__ == "__main__":
    main()
