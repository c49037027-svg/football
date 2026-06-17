"""國際賽資料：下載、正規化、Elo 評分、進球分鐘分布。

資料來源（GitHub 公開）：martj42/international_results
  - results.csv     ：1872 至今所有國際賽（含中立場 neutral 標記、賽事 tournament）
  - goalscorers.csv ：每球的進球分鐘，用來估上/下半場進球分布

由於 FIFA 官方排名沒有穩定的公開 CSV，這裡用國際賽結果自算
**World Football Elo**（與 FIFA 排名高度相關、且對勝負更具預測力）作為「實力排名」，
並可選擇性匯入官方 FIFA 排名（見 load_fifa_ranking）。
"""
from __future__ import annotations

from io import StringIO
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from ..data import schema as S

BASE = "https://raw.githubusercontent.com/martj42/international_results/master"
RESULTS_URL = f"{BASE}/results.csv"
GOALSCORERS_URL = f"{BASE}/goalscorers.csv"

# 賽事重要性權重（影響 Elo 的 K 值）。越重要的賽事，賽果對評分影響越大。
TOURNAMENT_WEIGHT = {
    "FIFA World Cup": 60,
    "FIFA World Cup qualification": 40,
    "UEFA Euro": 50,
    "UEFA Euro qualification": 35,
    "Copa América": 50,
    "African Cup of Nations": 40,
    "AFC Asian Cup": 40,
    "UEFA Nations League": 40,
    "Confederations Cup": 45,
    "Friendly": 20,
}
DEFAULT_WEIGHT = 30


def _download(url: str, timeout: float = 60.0) -> pd.DataFrame:
    r = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=timeout)
    r.raise_for_status()
    return pd.read_csv(StringIO(r.text))


def normalize_results(raw: pd.DataFrame) -> pd.DataFrame:
    """results.csv → 內部格式（含 neutral, tournament）。"""
    df = pd.DataFrame({
        S.DATE: pd.to_datetime(raw["date"], errors="coerce"),
        S.HOME: raw["home_team"].astype(str),
        S.AWAY: raw["away_team"].astype(str),
        S.HOME_GOALS: pd.to_numeric(raw["home_score"], errors="coerce"),
        S.AWAY_GOALS: pd.to_numeric(raw["away_score"], errors="coerce"),
        S.NEUTRAL: raw["neutral"].astype(bool) if "neutral" in raw else False,
        S.TOURNAMENT: raw["tournament"].astype(str) if "tournament" in raw else "Friendly",
    })
    df = df.dropna(subset=[S.DATE, S.HOME_GOALS, S.AWAY_GOALS])
    df[S.HOME_GOALS] = df[S.HOME_GOALS].astype(int)
    df[S.AWAY_GOALS] = df[S.AWAY_GOALS].astype(int)
    return df.sort_values(S.DATE).reset_index(drop=True)


def fetch_international(out_path: str | Path | None = None,
                       since: str | None = "2006-01-01") -> pd.DataFrame:
    """下載並正規化國際賽結果。since 可限制起始日期以加速。"""
    df = normalize_results(_download(RESULTS_URL))
    if since:
        df = df[df[S.DATE] >= pd.Timestamp(since)].reset_index(drop=True)
    if out_path is not None:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_path, index=False)
        print(f"[ok] 已存 {len(df)} 場國際賽到 {out_path}")
    return df


def compute_elo(df: pd.DataFrame, base: float = 1500.0, home_field: float = 60.0,
                annotate: bool = True) -> tuple[pd.DataFrame, dict[str, float]]:
    """以 World Football Elo 演算法計算評分。

    - 期望勝率 We = 1 / (1 + 10^(-(dr)/400))，dr 含主場加成（中立場為 0）。
    - K = 賽事權重 × 進球差調整（大勝加成）。
    - 結果 W：勝 1 / 平 0.5 / 負 0。
    annotate=True 會在每場標上「賽前」雙方 Elo（home_elo/away_elo），供模型當特徵
    （這是賽前資訊，不會造成前視偏誤）。
    回傳 (annotated_df, 最終各隊 Elo)。
    """
    elo: dict[str, float] = {}
    he, ae = np.full(len(df), base), np.full(len(df), base)

    def get(t: str) -> float:
        return elo.get(t, base)

    for i, r in enumerate(df.itertuples(index=False)):
        h, a = getattr(r, S.HOME), getattr(r, S.AWAY)
        hg, ag = getattr(r, S.HOME_GOALS), getattr(r, S.AWAY_GOALS)
        neutral = bool(getattr(r, S.NEUTRAL, False))
        tour = str(getattr(r, S.TOURNAMENT, "Friendly"))

        rh, ra = get(h), get(a)
        he[i], ae[i] = rh, ra  # 賽前 Elo

        adv = 0.0 if neutral else home_field
        we_h = 1.0 / (1.0 + 10 ** (-((rh + adv) - ra) / 400.0))
        w_h = 1.0 if hg > ag else 0.5 if hg == ag else 0.0

        k = TOURNAMENT_WEIGHT.get(tour, DEFAULT_WEIGHT)
        gd = abs(hg - ag)
        g_mult = 1.0 if gd <= 1 else (1.5 if gd == 2 else (1.75 + (gd - 3) / 8.0))
        delta = k * g_mult * (w_h - we_h)
        elo[h] = rh + delta
        elo[a] = ra - delta

    if annotate:
        df = df.copy()
        df[S.HOME_ELO] = he
        df[S.AWAY_ELO] = ae
    return df, elo


def goal_minute_split(goalscorers: pd.DataFrame | None = None,
                      timeout: float = 60.0) -> float:
    """估「上半場進球佔比」。沒給資料就下載 goalscorers.csv。

    回傳 first_half_fraction（典型約 0.45，下半場進球略多）。
    """
    if goalscorers is None:
        try:
            goalscorers = _download(GOALSCORERS_URL, timeout=timeout)
        except Exception:  # noqa: BLE001
            return 0.45  # 取不到就用文獻典型值
    minute = pd.to_numeric(goalscorers.get("minute"), errors="coerce").dropna()
    if len(minute) == 0:
        return 0.45
    first = (minute <= 45).mean()
    return float(np.clip(first, 0.30, 0.60))


def load_fifa_ranking(path: str | Path) -> dict[str, float]:
    """選用：匯入官方 FIFA 排名 CSV（欄位：team, points 或 rank）。

    回傳 {team: points}。若只有名次，轉成遞減分數。可用來取代/補充自算 Elo。
    """
    df = pd.read_csv(path)
    cols = {c.lower(): c for c in df.columns}
    team_col = cols.get("team") or cols.get("country") or list(df.columns)[0]
    if "points" in cols:
        return dict(zip(df[team_col].astype(str), pd.to_numeric(df[cols["points"]], errors="coerce")))
    if "rank" in cols:
        ranks = pd.to_numeric(df[cols["rank"]], errors="coerce")
        pts = 2000 - ranks * 5  # 名次轉粗略分數
        return dict(zip(df[team_col].astype(str), pts))
    raise ValueError("FIFA 排名 CSV 需含 points 或 rank 欄位")
