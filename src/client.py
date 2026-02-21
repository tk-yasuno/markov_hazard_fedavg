"""
client.py
=========
FedAvg クライアント実装

1ユーザーが担当する処理:
  1. サーバから受け取った β (12次元) をローカルモデルにロード
  2. ローカルデータを使い、ミニバッチ NLL 勾配を計算
  3. 勾配ベクトル（12次元）とサンプル数 n_u をサーバへ送信

FedAvg の場合、クライアントは複数ローカルステップを行うことも可能。
MVP では、全データを1エポック回して平均勾配をサーバに返す。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

from src.data_utils import (
    extract_transition_pairs,
    generate_synthetic_data,
    normalize_local,
)
from src.likelihood import compute_nll_and_grad, log_likelihood_batch_vectorized
from src.model import MarkovHazardModel, N_PARAMS


# ------------------------------------------------------------------ #
# クライアント通信メッセージの型定義                                   #
# ------------------------------------------------------------------ #
@dataclass
class ClientUpdate:
    """クライアント → サーバ の送信内容"""
    user_id: str
    gradient: torch.Tensor   # shape (12,) 平均勾配 ∂NLL/∂β
    n_samples: int            # 有効サンプル数（重み付き平均用）
    nll: float                # 平均 NLL（ログ記録用）


# ------------------------------------------------------------------ #
# FedAvg クライアントクラス                                            #
# ------------------------------------------------------------------ #
class FedAvgClient:
    """
    FedAvg クライアント（橋梁点検データ 1ユーザー分）。

    Parameters
    ----------
    user_id    : ユーザ識別子
    df         : inspection_records スキーマの DataFrame
    batch_size : ミニバッチサイズ
    n_local_steps : ローカル SGD ステップ数
                    1 → FedSGD（純粋な1ステップ勾配）
                    > 1 → FedAvg（ローカルに複数ステップ）
    lr         : ローカル SGD の学習率（FedAvg モード時のみ使用）
    """

    def __init__(
        self,
        user_id: str,
        df: pd.DataFrame,
        batch_size: int = 64,
        n_local_steps: int = 1,
        lr: float = 0.01,
        device: str = "cpu",
    ):
        self.user_id = user_id
        self.batch_size = batch_size
        self.n_local_steps = n_local_steps
        self.lr = lr
        self.device = torch.device(device)

        # 前処理: 標準化 → 遷移ペア抽出
        df_norm, self.norm_stats = normalize_local(df)
        pairs = extract_transition_pairs(df_norm)

        if len(pairs) == 0:
            raise ValueError(f"[{user_id}] No valid transition pairs found.")

        self.n_samples = len(pairs)

        # Tensor 化
        self.state_from = torch.tensor(
            pairs["state_from"].values, dtype=torch.long, device=self.device
        )
        self.state_to = torch.tensor(
            pairs["state_to"].values, dtype=torch.long, device=self.device
        )
        self.delta_t = torch.tensor(
            pairs["delta_t"].values, dtype=torch.float32, device=self.device
        )
        self.z = torch.tensor(
            pairs[["z1", "z2", "z3"]].values, dtype=torch.float32, device=self.device
        )

        # DataLoader（ミニバッチ用）
        dataset = TensorDataset(
            self.state_from, self.state_to, self.delta_t, self.z
        )
        self.loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # -------------------------------------------------------------- #
    # グローバルパラメータを受け取って勾配 or 更新後パラメータを返す  #
    # -------------------------------------------------------------- #
    def local_gradient(self, global_beta: torch.Tensor) -> ClientUpdate:
        """
        FedSGD モード（n_local_steps=1）:
          全データの平均勾配 ∂NLL/∂β を返す。

        FedAvg モード（n_local_steps>1）:
          ローカル SGD n ステップ後の ⊿β（疑似勾配）を返す。

        Parameters
        ----------
        global_beta : shape (12,) or (3,4) サーバのパラメータ

        Returns
        -------
        ClientUpdate
        """
        model = MarkovHazardModel().to(self.device)
        model.set_flat_params(global_beta.to(self.device))

        if self.n_local_steps == 1:
            return self._one_shot_gradient(model)
        else:
            return self._local_sgd(model)

    def _one_shot_gradient(self, model: MarkovHazardModel) -> ClientUpdate:
        """全データを1パスして平均勾配を計算（FedSGD）"""
        total_grad = torch.zeros(N_PARAMS, device=self.device)
        total_nll = 0.0

        for sf, st, dt, z in self.loader:
            nll_val, grad = compute_nll_and_grad(model, sf, st, dt, z)
            total_grad += grad * sf.shape[0]
            total_nll += nll_val * sf.shape[0]

        avg_grad = total_grad / self.n_samples
        avg_nll = total_nll / self.n_samples

        return ClientUpdate(
            user_id=self.user_id,
            gradient=avg_grad,
            n_samples=self.n_samples,
            nll=avg_nll,
        )

    def _local_sgd(self, model: MarkovHazardModel) -> ClientUpdate:
        """
        複数ローカルステップ SGD を行い、疑似勾配として
        (β_init − β_local) / lr を返す（FedAvg 標準近似）。
        """
        beta_init = model.get_flat_params().clone()
        optimizer = torch.optim.SGD(model.parameters(), lr=self.lr)

        step = 0
        total_nll = 0.0
        n_batches = 0

        while step < self.n_local_steps:
            for sf, st, dt, z in self.loader:
                optimizer.zero_grad()
                n = sf.shape[0]
                log_lik = log_likelihood_batch_vectorized(model, sf, st, dt, z)
                nll = -log_lik / n
                nll.backward()
                optimizer.step()

                total_nll += nll.item()
                n_batches += 1
                step += 1
                if step >= self.n_local_steps:
                    break

        beta_local = model.get_flat_params()
        # 疑似勾配: 初期値と更新後の差 / lr を返す
        pseudo_grad = (beta_init - beta_local) / self.lr
        avg_nll = total_nll / n_batches if n_batches > 0 else 0.0

        return ClientUpdate(
            user_id=self.user_id,
            gradient=pseudo_grad,
            n_samples=self.n_samples,
            nll=avg_nll,
        )

    # -------------------------------------------------------------- #
    # 推論: 遷移確率を計算して返す（ベンチマーク利用時）              #
    # -------------------------------------------------------------- #
    def predict_transition_prob(
        self,
        global_beta: torch.Tensor,
        state_from: int,
        z_raw: np.ndarray,
        delta_t: float,
    ) -> dict[int, float]:
        """
        グローバルモデルによる遷移確率を計算（ローカル標準化を適用）。

        Parameters
        ----------
        global_beta : shape (12,)
        state_from  : 現在状態
        z_raw       : [age_years, sea_distance_km, deck_area_m2] の生値
        delta_t     : 点検間隔 (年)

        Returns
        -------
        {next_state: probability}
        """
        model = MarkovHazardModel().to(self.device)
        model.set_flat_params(global_beta.to(self.device))
        model.eval()

        # ローカル標準化スケールを適用
        z_norm = np.array([
            z_raw[0] / self.norm_stats.max_age,
            z_raw[1] / self.norm_stats.max_sea,
            z_raw[2] / self.norm_stats.max_area,
        ])
        z_tensor = torch.tensor(z_norm, dtype=torch.float32, device=self.device)

        with torch.no_grad():
            probs = model.transition_prob(state_from, z_tensor, delta_t)

        return {k: float(v.item()) for k, v in probs.items()}
