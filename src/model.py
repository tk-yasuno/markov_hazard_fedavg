"""
model.py
========
CTMC マルコフ劣化ハザードモデル（PyTorch実装）

モデル仕様
──────────────────────────────────────────────────────
状態:          s ∈ {0, 1, 2}   (0=健全, 1=軽微, 2=要補修)
許容遷移:      0→1, 0→2, 1→2  (劣化方向のみ。2は吸収状態)
説明変数:      z = (z1, z2, z3) 標準化済み
ハザード:      λ_{ij}(z) = exp(β₀ + β₁z₁ + β₂z₂ + β₃z₃)

パラメータ次元: 3遷移 × 4係数 = 12次元
  β行列 shape: (3, 4)
    行0: (0→1) の [β₀, β₁, β₂, β₃]
    行1: (0→2) の [β₀, β₁, β₂, β₃]
    行2: (1→2) の [β₀, β₁, β₂, β₃]
──────────────────────────────────────────────────────
"""

from __future__ import annotations

import torch
import torch.nn as nn

# 許容遷移インデックス: (from_state, to_state)
ALLOWED_TRANSITIONS = [(0, 1), (0, 2), (1, 2)]
N_TRANSITIONS = len(ALLOWED_TRANSITIONS)  # 3
N_COVARIATES = 3                          # z1, z2, z3
N_BETA_COLS = N_COVARIATES + 1            # β₀ + β₁ + β₂ + β₃ = 4
N_PARAMS = N_TRANSITIONS * N_BETA_COLS   # 12次元

# 状態ごとに関連する遷移インデックスのマップ
# STATE_TRANSITIONS[i] = [(trans_idx, j), ...]
STATE_TRANSITIONS: dict[int, list[tuple[int, int]]] = {0: [], 1: [], 2: []}
for _t_idx, (_i, _j) in enumerate(ALLOWED_TRANSITIONS):
    STATE_TRANSITIONS[_i].append((_t_idx, _j))


class MarkovHazardModel(nn.Module):
    """
    連続時間マルコフ劣化ハザードモデル。

    パラメータ
    ----------
    beta : nn.Parameter, shape (N_TRANSITIONS=3, N_BETA_COLS=4)
        β 行列。フラット化すると 12次元ベクトル。
    """

    def __init__(self, init_beta: torch.Tensor | None = None):
        super().__init__()
        if init_beta is None:
            # ゼロ初期化（β₀ = 0 → λ = 1.0 が起点）
            init_beta = torch.zeros(N_TRANSITIONS, N_BETA_COLS)
        self.beta = nn.Parameter(init_beta.clone().float())

    # -------------------------------------------------------------- #
    # ハザード計算                                                     #
    # -------------------------------------------------------------- #
    def hazard(self, z: torch.Tensor) -> torch.Tensor:
        """
        全許容遷移のハザードを計算する。

        Parameters
        ----------
        z : shape (..., N_COVARIATES=3)

        Returns
        -------
        lam : shape (..., N_TRANSITIONS=3)
            lam[..., t] = λ_{ij}(z) for t-th transition
        """
        # z を拡張して (N_TRANSITIONS, N_BETA_COLS) の beta と内積
        # z: (..., 3) → (..., 3, 1) に対して beta: (3,4)
        # design vector: [1, z1, z2, z3] shape (..., 4)
        ones = torch.ones(*z.shape[:-1], 1, device=z.device, dtype=z.dtype)
        design = torch.cat([ones, z], dim=-1)  # (..., 4)

        # λ_{ij}(z) = exp(β_row · design)
        # beta: (3, 4), design: (..., 4) → (..., 3)
        log_lam = design @ self.beta.T  # (..., 3)
        lam = torch.exp(log_lam)        # (..., 3)
        return lam

    def total_hazard(self, state: torch.Tensor, lam: torch.Tensor) -> torch.Tensor:
        """
        状態 i における総ハザード Λ_i(z) = Σ_{j} λ_{ij}(z)

        Parameters
        ----------
        state : shape (B,) — 現在状態（整数）
        lam   : shape (B, 3) — 全遷移ハザード

        Returns
        -------
        Lambda : shape (B,) — 総ハザード
        """
        B = state.shape[0]
        Lambda = torch.zeros(B, device=lam.device, dtype=lam.dtype)

        for t_idx, (i, _j) in enumerate(ALLOWED_TRANSITIONS):
            mask = (state == i)
            Lambda = Lambda + lam[:, t_idx] * mask.float()

        return Lambda

    # -------------------------------------------------------------- #
    # 遷移確率（参照用・予測用）                                       #
    # -------------------------------------------------------------- #
    def transition_prob(
        self,
        state_from: int,
        z: torch.Tensor,
        delta_t: float | torch.Tensor,
    ) -> dict[int, torch.Tensor]:
        """
        状態 i から Δt 後の各状態への遷移確率を計算。

        Parameters
        ----------
        state_from : 現在状態 (0, 1, 2)
        z          : shape (..., 3) 標準化済み説明変数
        delta_t    : 点検間隔 (スカラー or ...)

        Returns
        -------
        probs : dict  {next_state: probability tensor shape (...)}
        """
        if state_from == 2:
            return {2: torch.ones(z.shape[:-1], device=z.device)}

        lam = self.hazard(z)  # (..., 3)

        # 状態 i に関連するハザードの和
        related = STATE_TRANSITIONS[state_from]  # [(t_idx, j), ...]
        Lambda_i = sum(lam[..., t_idx] for t_idx, _ in related)  # (...)

        p_stay = torch.exp(-Lambda_i * delta_t)

        probs: dict[int, torch.Tensor] = {state_from: p_stay}
        for t_idx, j in related:
            lam_ij = lam[..., t_idx]
            p_move = (lam_ij / Lambda_i) * (1.0 - p_stay)
            probs[j] = p_move

        return probs

    # -------------------------------------------------------------- #
    # フラット化 / アンフラット化                                      #
    # -------------------------------------------------------------- #
    def get_flat_params(self) -> torch.Tensor:
        """β を 12次元フラットベクトルで返す"""
        return self.beta.detach().reshape(-1)

    def set_flat_params(self, flat: torch.Tensor) -> None:
        """12次元フラットベクトルで β を上書き"""
        with torch.no_grad():
            self.beta.copy_(flat.reshape(N_TRANSITIONS, N_BETA_COLS))

    def get_flat_grad(self) -> torch.Tensor | None:
        """β の勾配を 12次元フラットベクトルで返す"""
        if self.beta.grad is None:
            return None
        return self.beta.grad.detach().reshape(-1)

    def zero_grad_beta(self) -> None:
        """β の勾配をリセット"""
        if self.beta.grad is not None:
            self.beta.grad.zero_()
