"""
server.py
=========
FedAvg サーバ実装

サーバの役割:
  1. グローバルモデル β (12次元) を保持・配信
  2. 各クライアントからの勾配 g_u とサンプル数 n_u を受信
  3. サンプル数加重平均で集約：
         ḡ = Σ(n_u · g_u) / Σ(n_u)
  4. グローバルパラメータを更新：
         β_new = β_old − η · ḡ
  5. ラウンドごとのログを記録・集計統計を出力

サーバは生データには一切アクセスしない（プライバシー保護の前提）。
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import List, Optional

import torch

from src.client import ClientUpdate
from src.model import N_PARAMS, MarkovHazardModel, ALLOWED_TRANSITIONS


# ------------------------------------------------------------------ #
# ラウンドログ                                                         #
# ------------------------------------------------------------------ #
@dataclass
class RoundLog:
    """1フェデレーテッドラウンドの統計"""
    round_idx: int
    n_clients: int
    n_total_samples: int
    avg_nll: float
    grad_norm: float
    elapsed_sec: float
    beta_snapshot: torch.Tensor  # shape (3, 4)


# ------------------------------------------------------------------ #
# FedAvg サーバクラス                                                  #
# ------------------------------------------------------------------ #
class FedAvgServer:
    """
    FedAvg グローバルアグリゲータ。

    Parameters
    ----------
    lr             : グローバル学習率 η
    init_beta      : 初期パラメータ shape (3,4) or (12,)。None でゼロ初期化
    momentum       : SGD モメンタム係数 (0 なら純粋な勾配降下)
    grad_clip_norm : 集約勾配のクリッピング閾値（0 なら無効）
    """

    def __init__(
        self,
        lr: float = 0.1,
        init_beta: Optional[torch.Tensor] = None,
        momentum: float = 0.0,
        grad_clip_norm: float = 0.0,
    ):
        self.lr = lr
        self.momentum = momentum
        self.grad_clip_norm = grad_clip_norm

        if init_beta is None:
            self._beta = torch.zeros(N_PARAMS)
        else:
            self._beta = init_beta.reshape(-1).clone().float()

        # モメンタムバッファ
        self._velocity = torch.zeros(N_PARAMS)

        # ログ
        self.history: List[RoundLog] = []

    # -------------------------------------------------------------- #
    # グローバルパラメータ配信                                         #
    # -------------------------------------------------------------- #
    @property
    def global_beta(self) -> torch.Tensor:
        """現在のグローバルパラメータ（12次元フラット）"""
        return self._beta.clone()

    def broadcast(self) -> torch.Tensor:
        """クライアントへ配信するパラメータのコピーを返す"""
        return self._beta.clone()

    # -------------------------------------------------------------- #
    # 集約・更新                                                       #
    # -------------------------------------------------------------- #
    def aggregate_and_update(
        self, updates: List[ClientUpdate], round_idx: int = 0
    ) -> RoundLog:
        """
        クライアント更新を受け取り、グローバルモデルを更新する。

        Parameters
        ----------
        updates   : 各クライアントからの ClientUpdate リスト
        round_idx : 現在のラウンド番号（ログ用）

        Returns
        -------
        RoundLog
        """
        t0 = time.perf_counter()

        if len(updates) == 0:
            raise ValueError("No client updates received.")

        # サンプル数加重平均勾配
        total_n = sum(u.n_samples for u in updates)
        g_bar = torch.zeros(N_PARAMS)

        for u in updates:
            g_bar += u.gradient.cpu() * u.n_samples

        g_bar = g_bar / total_n

        # 勾配クリッピング
        if self.grad_clip_norm > 0:
            gnorm = g_bar.norm()
            if gnorm > self.grad_clip_norm:
                g_bar = g_bar * (self.grad_clip_norm / gnorm)

        # モメンタム更新
        if self.momentum > 0:
            self._velocity = self.momentum * self._velocity + g_bar
            update_dir = self._velocity
        else:
            update_dir = g_bar

        # パラメータ更新
        self._beta = self._beta - self.lr * update_dir

        # ログ集計
        avg_nll = sum(u.nll * u.n_samples for u in updates) / total_n
        grad_norm = g_bar.norm().item()
        elapsed = time.perf_counter() - t0

        log = RoundLog(
            round_idx=round_idx,
            n_clients=len(updates),
            n_total_samples=total_n,
            avg_nll=avg_nll,
            grad_norm=grad_norm,
            elapsed_sec=elapsed,
            beta_snapshot=self._beta.reshape(3, 4).clone(),
        )
        self.history.append(log)
        return log

    # -------------------------------------------------------------- #
    # ベンチマーク：遷移確率の推計（サーバ側で計算）                   #
    # -------------------------------------------------------------- #
    def benchmark_transition_prob(
        self,
        state_from: int,
        z_normalized: torch.Tensor,
        delta_t: float,
    ) -> dict[int, float]:
        """
        グローバルモデルによる遷移確率ベンチマーク推計。

        Parameters
        ----------
        state_from    : 現在状態 (0, 1, 2)
        z_normalized  : shape (3,) 標準化済み説明変数
        delta_t       : 点検間隔 (年)

        Returns
        -------
        {next_state: probability}
        """
        model = MarkovHazardModel()
        model.set_flat_params(self._beta)
        model.eval()

        with torch.no_grad():
            probs = model.transition_prob(state_from, z_normalized, delta_t)

        return {k: float(v.item()) for k, v in probs.items()}

    # -------------------------------------------------------------- #
    # 学習曲線・パラメータ表示                                         #
    # -------------------------------------------------------------- #
    def print_history(self) -> None:
        """Print training history for all rounds."""
        print(f"\n{'-'*65}")
        print(f"{'Round':>6} {'Clients':>8} {'Samples':>9} {'Avg NLL':>10} {'||grad||':>10}")
        print(f"{'-'*65}")
        for log in self.history:
            print(
                f"{log.round_idx:>6d}"
                f"{log.n_clients:>9d}"
                f"{log.n_total_samples:>10d}"
                f"{log.avg_nll:>11.4f}"
                f"{log.grad_norm:>11.6f}"
            )
        print(f"{'-'*65}\n")

    def print_beta_table(self) -> None:
        """Print current global beta parameters by transition."""
        beta_mat = self._beta.reshape(3, 4)
        transition_labels = ["0->1", "0->2", "1->2"]
        coef_labels = ["b0", "b1(age)", "b2(sea)", "b3(area)"]

        print("\nGlobal beta parameters (current)")
        print(f"{'Trans':>6}", end="")
        for c in coef_labels:
            print(f"{c:>12}", end="")
        print()
        print("-" * (6 + 12 * 4))

        for t_idx, label in enumerate(transition_labels):
            print(f"{label:>6}", end="")
            for col in range(4):
                print(f"{beta_mat[t_idx, col].item():>12.4f}", end="")
            print()
        print()

    def save_checkpoint(self, path: str) -> None:
        """Save beta parameters to file."""
        torch.save({"beta": self._beta, "history": self.history}, path)
        print(f"Checkpoint saved: {path}")

    def load_checkpoint(self, path: str) -> None:
        """Load beta parameters from checkpoint."""
        ckpt = torch.load(path, map_location="cpu")
        self._beta = ckpt["beta"]
        self.history = ckpt.get("history", [])
        print(f"Checkpoint loaded: {path}")
