"""
data_utils.py
=============
データスキーマ定義・合成データ生成・前処理ユーティリティ

inspection_records スキーマ
─────────────────────────────────────────
user_id          : ユーザ識別子
bridge_id        : 橋梁ID
member_id        : 部材ID
damage_type      : 損傷種別 (crack_conc / rebar_exposure / steel_paint)
inspection_date  : 点検日
state            : 損傷状態 (0=健全, 1=軽微, 2=要補修)
delta_t          : 前回点検からの経過年数
x1_age_years     : 経過年数（橋齢）
x2_sea_distance_km : 海岸からの距離 (km)
x3_deck_area_m2  : 橋面積 (m²)
─────────────────────────────────────────
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import List, Optional


# ------------------------------------------------------------------ #
# 1. スキーマ定義                                                      #
# ------------------------------------------------------------------ #
INSPECTION_COLUMNS = [
    "user_id",
    "bridge_id",
    "member_id",
    "damage_type",
    "inspection_date",
    "state",
    "delta_t",
    "x1_age_years",
    "x2_sea_distance_km",
    "x3_deck_area_m2",
]

DAMAGE_TYPES = ["crack_conc", "rebar_exposure", "steel_paint"]

# 許容遷移ペア（劣化方向のみ）
ALLOWED_TRANSITIONS = [(0, 1), (0, 2), (1, 2)]
ABSORBING_STATE = 2


# ------------------------------------------------------------------ #
# 2. 合成データ生成                                                    #
# ------------------------------------------------------------------ #
def _sample_transition(
    state: int,
    delta_t: float,
    true_betas: np.ndarray,
    z: np.ndarray,
    rng: np.random.Generator,
) -> int:
    """
    真のパラメータ true_betas に基づいてCTMCで次状態をサンプリング。

    Parameters
    ----------
    state      : 現在状態 (0 or 1 or 2)
    delta_t    : 点検間隔 (年)
    true_betas : shape (3, 4) — 各行が遷移ペア (01,02,12) の β
    z          : shape (3,) — 標準化済み説明変数
    rng        : 乱数生成器
    """
    if state == ABSORBING_STATE:
        return ABSORBING_STATE

    # ハザード計算: λ_ij = exp(β_0 + β_1 z1 + β_2 z2 + β_3 z3)
    lam = {}
    for idx, (i, j) in enumerate(ALLOWED_TRANSITIONS):
        if i == state:
            beta = true_betas[idx]  # shape (4,)
            lam[(i, j)] = np.exp(beta[0] + beta[1:] @ z)

    if not lam:
        return state  # 吸収状態なら留まる

    Lambda_i = sum(lam.values())
    p_stay = np.exp(-Lambda_i * delta_t)

    u = rng.uniform()
    if u < p_stay:
        return state  # 留まる

    # どの遷移先かをハザード比で決定
    pairs = list(lam.keys())
    rates = np.array([lam[p] for p in pairs])
    probs = rates / rates.sum()
    chosen = rng.choice(len(pairs), p=probs)
    return pairs[chosen][1]


def generate_synthetic_data(
    user_id: str,
    n_bridges: int = 50,
    n_members: int = 2,
    n_inspections: int = 4,
    true_betas: Optional[np.ndarray] = None,
    sea_distance_range: tuple = (0.5, 30.0),
    deck_area_range: tuple = (100.0, 2000.0),
    delta_t_range: tuple = (2.0, 5.0),
    random_seed: int = 42,
) -> pd.DataFrame:
    """
    1ユーザー分の合成点検記録を生成する。

    Parameters
    ----------
    user_id          : ユーザID文字列
    n_bridges        : 橋梁数
    n_members        : 橋梁あたり部材数
    n_inspections    : 部材あたり点検回数（最大）
    true_betas       : shape (3, 4) の真のβ。None なら乱数で設定
    sea_distance_range : 海岸距離 [km] の (min, max)
    deck_area_range  : 橋面積 [m²] の (min, max)
    delta_t_range    : 点検間隔 [年] の (min, max)
    random_seed      : シード

    Returns
    -------
    pd.DataFrame  inspection_records スキーマに準拠
    """
    rng = np.random.default_rng(random_seed)

    if true_betas is None:
        # デフォルトの真値（劣化しやすい方向）
        # 行: (0→1), (0→2), (1→2) / 列: β0, β1(age), β2(sea), β3(area)
        true_betas = np.array([
            [-2.0, 0.5, -0.3, 0.1],   # 0→1
            [-4.0, 0.3, -0.5, 0.05],  # 0→2
            [-2.5, 0.4, -0.4, 0.08],  # 1→2
        ])

    records = []
    base_year = 2000

    for b in range(n_bridges):
        bridge_id = f"B{user_id}_{b:03d}"
        sea_dist = rng.uniform(*sea_distance_range)
        deck_area = rng.uniform(*deck_area_range)
        bridge_age_at_start = rng.integers(1, 30)  # 供用開始から最初点検までの年数
        damage_type = rng.choice(DAMAGE_TYPES)

        for m in range(n_members):
            member_id = f"M{m:02d}"
            state = 0  # 初期状態：健全
            current_age = bridge_age_at_start + rng.integers(0, 5)
            inspection_year = base_year + rng.integers(0, 10)

            for insp in range(n_inspections):
                # 標準化（ローカル最大を1として後で割るが、ここでは生値を保存）
                # ただしサンプリング時は事前に「代表スケール」で割る（固定値）
                z = np.array([
                    current_age / 50.0,       # age scale: 50年
                    sea_dist / 30.0,          # sea scale: 30km
                    deck_area / 2000.0,       # area scale: 2000m²
                ])

                if insp == 0:
                    # 初回点検：状態を記録するが delta_t は NaN（前回なし）
                    records.append({
                        "user_id": user_id,
                        "bridge_id": bridge_id,
                        "member_id": member_id,
                        "damage_type": damage_type,
                        "inspection_date": f"{inspection_year}-04-01",
                        "state": state,
                        "delta_t": np.nan,
                        "x1_age_years": current_age,
                        "x2_sea_distance_km": sea_dist,
                        "x3_deck_area_m2": deck_area,
                    })
                    prev_state = state
                    prev_age = current_age
                    prev_year = inspection_year
                else:
                    delta_t = rng.uniform(*delta_t_range)
                    current_age = prev_age + delta_t
                    inspection_year = prev_year + int(np.round(delta_t))

                    # 次状態をサンプリング
                    next_state = _sample_transition(
                        prev_state, delta_t, true_betas, z, rng
                    )

                    records.append({
                        "user_id": user_id,
                        "bridge_id": bridge_id,
                        "member_id": member_id,
                        "damage_type": damage_type,
                        "inspection_date": f"{inspection_year}-04-01",
                        "state": next_state,
                        "delta_t": delta_t,
                        "x1_age_years": current_age,
                        "x2_sea_distance_km": sea_dist,
                        "x3_deck_area_m2": deck_area,
                    })

                    prev_state = next_state
                    prev_age = current_age
                    prev_year = inspection_year

                    if next_state == ABSORBING_STATE:
                        break  # 吸収状態：以降の点検なし

    df = pd.DataFrame(records, columns=INSPECTION_COLUMNS)
    return df


# ------------------------------------------------------------------ #
# 3. 前処理：標準化 + 学習用ペア抽出                                   #
# ------------------------------------------------------------------ #
@dataclass
class NormStats:
    """標準化スケール（ローカル最大値）"""
    max_age: float
    max_sea: float
    max_area: float


def normalize_local(df: pd.DataFrame) -> tuple[pd.DataFrame, NormStats]:
    """
    ローカル最大値で説明変数を標準化する（MVP: 各ユーザのデータで計算）。

    Returns
    -------
    df_norm  : 標準化済みカラム (z1, z2, z3) を付与したDataFrame
    stats    : 使用した最大値（将来の推論時に再利用）
    """
    max_age = df["x1_age_years"].max()
    max_sea = df["x2_sea_distance_km"].max()
    max_area = df["x3_deck_area_m2"].max()

    df = df.copy()
    df["z1"] = df["x1_age_years"] / max_age
    df["z2"] = df["x2_sea_distance_km"] / max_sea
    df["z3"] = df["x3_deck_area_m2"] / max_area

    stats = NormStats(max_age=max_age, max_sea=max_sea, max_area=max_area)
    return df, stats


def extract_transition_pairs(df: pd.DataFrame) -> pd.DataFrame:
    """
    点検記録から「連続する2時点ペア（s_t, s_{t+1}, delta_t, z）」を抽出する。
    delta_t が NaN（初回点検）の行は除外。

    Returns
    -------
    pd.DataFrame with columns:
        state_from, state_to, delta_t, z1, z2, z3
    """
    # delta_t が有効な行のみ（2回目以降の点検）
    pairs = df.dropna(subset=["delta_t"]).copy()
    pairs = pairs.rename(columns={"state": "state_to"})

    # 前時点の状態は同 bridge & member の1行前
    df_sorted = df.sort_values(["bridge_id", "member_id", "inspection_date"])
    prev_state = df_sorted.groupby(["bridge_id", "member_id"])["state"].shift(0)

    # 前行の state を「state_from」として結合
    pairs_sorted = pairs.sort_values(["bridge_id", "member_id", "inspection_date"])
    from_states = (
        df_sorted.groupby(["bridge_id", "member_id"])["state"]
        .shift(1)
        .reindex(pairs_sorted.index)
    )
    pairs_sorted = pairs_sorted.copy()
    pairs_sorted["state_from"] = from_states.values

    # delta_t > 0 かつ state_from が有効なもののみ
    pairs_sorted = pairs_sorted.dropna(subset=["state_from"])
    pairs_sorted["state_from"] = pairs_sorted["state_from"].astype(int)
    pairs_sorted = pairs_sorted[pairs_sorted["delta_t"] > 0]

    return pairs_sorted[["state_from", "state_to", "delta_t", "z1", "z2", "z3"]].reset_index(drop=True)
