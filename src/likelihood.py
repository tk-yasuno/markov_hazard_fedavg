"""
likelihood.py
=============
CTMC 対数尤度の実装（PyTorch autograd 対応）

1観測ペア (s_t=i, s_{t+1}=k, Δt, z) に対する対数尤度
────────────────────────────────────────────────────────
ケース1：劣化なし (k = i)
    log L = -Λ_i(z) · Δt

ケース2：劣化あり (k ≠ i, 許容遷移)
    log L = log λ_{ik}(z) − log Λ_i(z) + log(1 − exp(−Λ_i(z)·Δt))
────────────────────────────────────────────────────────
"""

from __future__ import annotations

import torch

from src.model import (
    ALLOWED_TRANSITIONS,
    STATE_TRANSITIONS,
    MarkovHazardModel,
)

EPS = 1e-38  # ゼロ除算・log(0) 防止


def log_likelihood_batch(
    model: MarkovHazardModel,
    state_from: torch.Tensor,  # (B,) int
    state_to: torch.Tensor,    # (B,) int
    delta_t: torch.Tensor,     # (B,) float
    z: torch.Tensor,           # (B, 3) float
) -> torch.Tensor:
    """
    ミニバッチ対数尤度の合計を返す（最大化目標、∴ loss は符号反転して使う）。

    Parameters
    ----------
    model      : MarkovHazardModel
    state_from : 現在状態ベクトル (B,)
    state_to   : 次状態ベクトル   (B,)
    delta_t    : 点検間隔ベクトル (B,)
    z          : 標準化済み説明変数 (B, 3)

    Returns
    -------
    log_lik : スカラー Tensor（sum over batch）
    """
    # 全遷移ハザード: (B, 3)
    lam = model.hazard(z)

    # 状態ごとの総ハザード Λ_i(z): (B,)
    Lambda = model.total_hazard(state_from, lam)

    B = state_from.shape[0]
    log_lik = torch.zeros(B, device=z.device, dtype=z.dtype)

    for b_idx in range(B):
        i = state_from[b_idx].item()
        k = state_to[b_idx].item()
        dt = delta_t[b_idx]
        Lambda_i = Lambda[b_idx]

        if i == 2:
            # 吸収状態: 常に log L = 0
            continue

        if k == i:
            # ケース1: 留まる（劣化なし）
            log_lik[b_idx] = -Lambda_i * dt

        else:
            # ケース2: 劣化遷移 (i → k, k ≠ i)
            # 許容遷移かチェック
            t_idx = None
            for _t_idx, (ai, aj) in enumerate(ALLOWED_TRANSITIONS):
                if ai == i and aj == k:
                    t_idx = _t_idx
                    break

            if t_idx is None:
                # 許容されない遷移（データエラー等）→ 非常に低い確率
                log_lik[b_idx] = torch.tensor(-1e6, dtype=z.dtype, device=z.device)
                continue

            lam_ik = lam[b_idx, t_idx]
            exp_term = torch.exp(-Lambda_i * dt)
            one_minus_exp = 1.0 - exp_term

            # 数値安定化
            lam_ik_safe = lam_ik.clamp(min=EPS)
            Lambda_i_safe = Lambda_i.clamp(min=EPS)
            one_minus_exp_safe = one_minus_exp.clamp(min=EPS)

            log_lik[b_idx] = (
                torch.log(lam_ik_safe)
                - torch.log(Lambda_i_safe)
                + torch.log(one_minus_exp_safe)
            )

    return log_lik.sum()


def log_likelihood_batch_vectorized(
    model: MarkovHazardModel,
    state_from: torch.Tensor,  # (B,) int
    state_to: torch.Tensor,    # (B,) int
    delta_t: torch.Tensor,     # (B,) float
    z: torch.Tensor,           # (B, 3) float
) -> torch.Tensor:
    """
    ベクトル化版の対数尤度計算（高速・勾配フロー良好）。

    ループを排除し、mask 演算で処理する。
    """
    lam = model.hazard(z)  # (B, 3)

    # 総ハザード Λ_i (B,)
    Lambda = torch.zeros(state_from.shape[0], device=z.device, dtype=z.dtype)
    for t_idx, (i, _j) in enumerate(ALLOWED_TRANSITIONS):
        mask_i = (state_from == i).float()
        Lambda = Lambda + lam[:, t_idx] * mask_i

    exp_term = torch.exp(-Lambda * delta_t)           # (B,)
    one_minus_exp = (1.0 - exp_term).clamp(min=EPS)  # (B,)

    # ケース1: k == i (留まる)
    stay_mask = (state_from == state_to).float()

    # ケース2: k != i (遷移する) — 遷移ペアごとに処理
    log_lik_stay = -Lambda * delta_t                 # ケース1の寄与

    log_lik_move = torch.full_like(log_lik_stay, 0.0)

    for t_idx, (i, j) in enumerate(ALLOWED_TRANSITIONS):
        transition_mask = ((state_from == i) & (state_to == j)).float()  # (B,)
        lam_ij = lam[:, t_idx].clamp(min=EPS)
        Lambda_safe = Lambda.clamp(min=EPS)

        log_lik_ij = (
            torch.log(lam_ij)
            - torch.log(Lambda_safe)
            + torch.log(one_minus_exp)
        )
        log_lik_move = log_lik_move + log_lik_ij * transition_mask

    # 吸収状態 (i=2) の寄与は 0
    absorb_mask = (state_from == 2).float()

    log_lik = (
        log_lik_stay * stay_mask * (1.0 - absorb_mask)
        + log_lik_move * (1.0 - stay_mask) * (1.0 - absorb_mask)
    )

    return log_lik.sum()


def compute_nll_and_grad(
    model: MarkovHazardModel,
    state_from: torch.Tensor,
    state_to: torch.Tensor,
    delta_t: torch.Tensor,
    z: torch.Tensor,
) -> tuple[float, torch.Tensor]:
    """
    Negative Log-Likelihood (NLL) と β の勾配（12次元）を返す。

    FedAvg クライアントはこの関数を呼んで勾配を収集する。

    Returns
    -------
    nll_value : float  サンプル平均 NLL
    grad      : shape (12,) Tensor — β に対する ∂NLL/∂β
    """
    model.zero_grad_beta()

    n = state_from.shape[0]
    log_lik = log_likelihood_batch_vectorized(
        model, state_from, state_to, delta_t, z
    )
    nll = -log_lik / n  # サンプル平均

    nll.backward()

    grad = model.get_flat_grad()
    assert grad is not None, "Gradient has not been computed."

    return nll.item(), grad.clone()
