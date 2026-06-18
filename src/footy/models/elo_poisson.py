"""Elo–Poisson 模型（純 Elo 驅動，無每隊攻防參數）。

參考公開方法論（djyylive 等）：球隊強度完全來自賽前 Elo，不為每隊擬合 attack/defence，
因此參數極少（不會過擬合小樣本國家隊），對國際賽特別穩健。

  λ_home = (μ/2) · exp(+s · Δr)
  λ_away = (μ/2) · exp(−s · Δr)
  Δr     = (Elo_home − Elo_away) + hfa·(非中立場 ? 1 : 0)

其中 μ=場均總進球、s=Elo 敏感度（等價於 β·收縮係數）、hfa=主場優勢(Elo 點)、
ρ=Dixon–Coles 低比分修正。全部由加權 MLE 在歷史資料上擬合。
"""
from __future__ import annotations

import pickle
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from ..data import schema as S
from .dixon_coles import _half_life_to_xi, _tau, score_matrix_from_rates


@dataclass
class EloPoissonModel:
    mu: float                 # 場均總進球
    s: float                  # Elo 敏感度（每 Elo 點）
    hfa: float                # 主場優勢（Elo 點）
    rho: float
    team_elo: dict[str, float] = field(default_factory=dict)
    max_goals: int = 10
    trained_until: "pd.Timestamp | None" = None

    @property
    def attack(self):  # 相容介面：讓既有程式用 `in model.attack` 判斷球隊
        return self.team_elo

    @property
    def teams(self):
        return list(self.team_elo.keys())

    def _elo(self, t: str) -> float:
        if self.team_elo:
            return self.team_elo.get(t, sum(self.team_elo.values()) / len(self.team_elo))
        return 1500.0

    def expected_goals(self, home: str, away: str,
                       neutral: bool = False) -> tuple[float, float]:
        dr = (self._elo(home) - self._elo(away)) + (0.0 if neutral else self.hfa)
        lam = (self.mu / 2.0) * np.exp(self.s * dr)
        mu_a = (self.mu / 2.0) * np.exp(-self.s * dr)
        return float(np.clip(lam, 1e-6, 30)), float(np.clip(mu_a, 1e-6, 30))

    def score_matrix(self, home: str, away: str, lam=None, mu=None,
                     neutral: bool = False) -> np.ndarray:
        if lam is None or mu is None:
            lam, mu = self.expected_goals(home, away, neutral=neutral)
        return score_matrix_from_rates(lam, mu, self.rho, self.max_goals)

    def save(self, path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path):
        with open(path, "rb") as f:
            return pickle.load(f)


def fit_elo_poisson(df: pd.DataFrame, half_life_days: float = 540.0,
                    max_goals: int = 10, reference_date=None,
                    team_elo: dict | None = None) -> EloPoissonModel:
    """加權 MLE 擬合 s, hfa, rho（μ 取訓練集場均總進球）。

    df 需含 home_elo/away_elo（賽前 Elo）、neutral、進球與日期。
    team_elo：各隊最新 Elo（推論用）；未給則由資料末尾推得。
    """
    df = df.dropna(subset=[S.HOME, S.AWAY, S.HOME_GOALS, S.AWAY_GOALS,
                           S.HOME_ELO, S.AWAY_ELO]).copy()
    if reference_date is None:
        reference_date = df[S.DATE].max()
    df = df[df[S.DATE] <= reference_date]

    hg = df[S.HOME_GOALS].to_numpy(float)
    ag = df[S.AWAY_GOALS].to_numpy(float)
    delo = (pd.to_numeric(df[S.HOME_ELO]) - pd.to_numeric(df[S.AWAY_ELO])).to_numpy(float)
    host = (~df[S.NEUTRAL].astype(bool)).to_numpy(float) if S.NEUTRAL in df else np.ones(len(df))
    xi = _half_life_to_xi(half_life_days)
    w = np.exp(-xi * np.clip((reference_date - df[S.DATE]).dt.days.to_numpy(float), 0, None))
    mu = float((hg + ag).mean()) if len(df) else 2.6

    from scipy.special import gammaln

    def nll(p):
        s, hfa, rho = p
        dr = delo + hfa * host
        lam = np.clip((mu / 2.0) * np.exp(s * dr), 1e-8, 30)
        mua = np.clip((mu / 2.0) * np.exp(-s * dr), 1e-8, 30)
        logp = (hg * np.log(lam) - lam - gammaln(hg + 1)
                + ag * np.log(mua) - mua - gammaln(ag + 1))
        tau = _tau(hg, ag, lam, mua, rho)
        if np.any(tau <= 0):
            return 1e10
        return -np.sum(w * (logp + np.log(tau)))

    res = minimize(nll, np.array([0.003, 60.0, -0.05]), method="L-BFGS-B",
                   bounds=[(1e-5, 0.02), (-50, 200), (-0.2, 0.2)],
                   options={"maxiter": 500})
    s, hfa, rho = res.x

    if team_elo is None:
        team_elo = {}
        for r in df.itertuples(index=False):
            team_elo[getattr(r, S.HOME)] = float(getattr(r, S.HOME_ELO))
            team_elo[getattr(r, S.AWAY)] = float(getattr(r, S.AWAY_ELO))

    return EloPoissonModel(mu=mu, s=float(s), hfa=float(hfa), rho=float(rho),
                           team_elo=team_elo, max_goals=max_goals,
                           trained_until=reference_date)
