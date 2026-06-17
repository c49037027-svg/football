"""Dixon–Coles 模型（帶時間衰減的雙變量 Poisson）。

參考：Dixon & Coles (1997), "Modelling Association Football Scores and
Inefficiencies in the Football Betting Market".

每隊有「攻擊強度」alpha_i 與「防守強度」beta_i，外加全聯賽共用的主場優勢 gamma。
主隊預期進球 lambda = exp(alpha_home - beta_away + gamma)
客隊預期進球 mu     = exp(alpha_away - beta_home)

進球數近似 Poisson，但 0-0/1-0/0-1/1-1 這些低比分用 Dixon–Coles 的
tau(x, y; lambda, mu, rho) 修正項調整其相關性（rho 通常為小負數）。

並用時間衰減 weight = exp(-xi * t)（t 為距今天數）讓近期比賽權重更高。
為了可辨識性，限制所有 attack 參數平均為 0。
"""
from __future__ import annotations

import pickle
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from ..data import schema as S


def _tau(home_goals: np.ndarray, away_goals: np.ndarray,
         lam: np.ndarray, mu: np.ndarray, rho: float) -> np.ndarray:
    """Dixon–Coles 低比分修正項，向量化。"""
    out = np.ones_like(lam, dtype=float)
    h0 = home_goals == 0
    h1 = home_goals == 1
    a0 = away_goals == 0
    a1 = away_goals == 1
    out = np.where(h0 & a0, 1.0 - lam * mu * rho, out)
    out = np.where(h0 & a1, 1.0 + lam * rho, out)
    out = np.where(h1 & a0, 1.0 + mu * rho, out)
    out = np.where(h1 & a1, 1.0 - rho, out)
    return out


def _half_life_to_xi(half_life_days: float) -> float:
    """半衰期（天）轉成衰減率 xi，使 weight 在 half_life 天時為 0.5。"""
    if half_life_days <= 0:
        return 0.0
    return np.log(2.0) / half_life_days


@dataclass
class DixonColesModel:
    teams: list[str]
    attack: dict[str, float]
    defence: dict[str, float]
    home_adv: float
    rho: float
    max_goals: int = 10
    # 訓練資料的最後日期，用於走地/初盤推論時計算「距今天數」（此處僅備查）。
    trained_until: "pd.Timestamp | None" = None
    # Elo 擴充：全域係數與每隊最新 Elo（推論時用）。elo_coef=0 代表未用 Elo。
    elo_coef: float = 0.0
    team_elo: dict[str, float] = field(default_factory=dict)

    # ---------- 推論 ----------
    def expected_goals(self, home: str, away: str,
                       neutral: bool = False) -> tuple[float, float]:
        """回傳 (lambda_home, mu_away)：雙方預期進球數。

        neutral=True 時不套用主場優勢（國際賽中立場地，如世界盃）。
        """
        if home not in self.attack or away not in self.attack:
            raise KeyError(f"模型未包含球隊：{home} 或 {away}")
        elo_term = 0.0
        if self.elo_coef and self.team_elo:
            mean_elo = sum(self.team_elo.values()) / len(self.team_elo)
            eh = self.team_elo.get(home, mean_elo)
            ea = self.team_elo.get(away, mean_elo)
            elo_term = self.elo_coef * (eh - ea) / 400.0
        ha = 0.0 if neutral else self.home_adv
        lam = np.exp(self.attack[home] - self.defence[away] + ha + elo_term)
        mu = np.exp(self.attack[away] - self.defence[home] - elo_term)
        return float(lam), float(mu)

    def score_matrix(self, home: str, away: str,
                     lam: float | None = None, mu: float | None = None,
                     neutral: bool = False) -> np.ndarray:
        """回傳 (max_goals+1) x (max_goals+1) 的比分機率矩陣 P[h, a]。

        可直接傳入 lam/mu（走地時用調整後的預期進球），否則用 pre-match 估計。
        """
        if lam is None or mu is None:
            lam, mu = self.expected_goals(home, away, neutral=neutral)
        return score_matrix_from_rates(lam, mu, self.rho, self.max_goals)

    # ---------- 序列化 ----------
    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path: str | Path) -> "DixonColesModel":
        with open(path, "rb") as f:
            return pickle.load(f)


def score_matrix_from_rates(lam: float, mu: float, rho: float, max_goals: int) -> np.ndarray:
    """由預期進球率與 rho 直接建構比分機率矩陣（含 DC 低分修正並重新正規化）。"""
    from scipy.stats import poisson

    h = np.arange(max_goals + 1)
    home_p = poisson.pmf(h, lam)
    away_p = poisson.pmf(h, mu)
    mat = np.outer(home_p, away_p)

    # 套用低比分修正
    corr = np.ones((2, 2))
    corr[0, 0] = 1.0 - lam * mu * rho
    corr[0, 1] = 1.0 + lam * rho
    corr[1, 0] = 1.0 + mu * rho
    corr[1, 1] = 1.0 - rho
    # 修正項必須為正，否則該組參數無效（在最佳化時會被懲罰）
    corr = np.clip(corr, 1e-10, None)
    mat[:2, :2] *= corr

    total = mat.sum()
    if total <= 0:
        raise ValueError("比分矩陣總和非正，參數無效。")
    return mat / total


def fit(df: pd.DataFrame, half_life_days: float = 180.0, max_goals: int = 10,
        rho_init: float = -0.05, reference_date: "pd.Timestamp | None" = None,
        xg_weight: float = 0.0, use_elo: bool = False, reg: float = 0.0,
        verbose: bool = False) -> DixonColesModel:
    """以最大概似估計擬合 Dixon–Coles 模型。

    df 需含內部欄位：date, home, away, home_goals, away_goals。
    reference_date：時間衰減的基準（預設為資料中最後一天）。**晚於此日期的比賽會被
        剔除**，避免用到「未來」資料造成洩漏（look-ahead）。
    xg_weight：若資料含 home_xg/away_xg，用 (1-w)*goals + w*xG 當建模目標。
        xG 通常比實際進球更穩定、雜訊更低。注意：xG 為連續值，Poisson 概似用
        gammaln 仍可定義；但 Dixon–Coles 的低比分 tau 修正只對「整數低比分」有意義，
        因此 xg_weight>0 時自動關閉 tau 修正（rho 固定為 0）。
    reg：L2 正則化強度，對攻防參數施加 reg*Σ(attack²+defence²) 懲罰，把弱資料的隊
        往聯盟平均收縮，降低過擬合。
    """
    df = df.dropna(subset=[S.HOME, S.AWAY, S.HOME_GOALS, S.AWAY_GOALS]).copy()
    if reference_date is None:
        reference_date = df[S.DATE].max()
    else:
        # 防洩漏：只用基準日（含）之前的比賽
        df = df[df[S.DATE] <= reference_date].copy()

    teams = sorted(set(df[S.HOME]) | set(df[S.AWAY]))
    n = len(teams)
    idx = {t: i for i, t in enumerate(teams)}

    hi = df[S.HOME].map(idx).to_numpy()
    ai = df[S.AWAY].map(idx).to_numpy()
    hg_actual = df[S.HOME_GOALS].to_numpy(dtype=float)
    ag_actual = df[S.AWAY_GOALS].to_numpy(dtype=float)

    use_xg = xg_weight > 0 and S.HOME_XG in df.columns and S.AWAY_XG in df.columns
    if use_xg:
        hxg = pd.to_numeric(df[S.HOME_XG], errors="coerce").fillna(df[S.HOME_GOALS]).to_numpy(dtype=float)
        axg = pd.to_numeric(df[S.AWAY_XG], errors="coerce").fillna(df[S.AWAY_GOALS]).to_numpy(dtype=float)
        hg = (1 - xg_weight) * hg_actual + xg_weight * hxg
        ag = (1 - xg_weight) * ag_actual + xg_weight * axg
        disable_tau = True  # 連續目標下 tau 無定義
    else:
        hg = hg_actual
        ag = ag_actual
        disable_tau = False

    xi = _half_life_to_xi(half_life_days)
    age_days = (reference_date - df[S.DATE]).dt.days.to_numpy(dtype=float)
    weights = np.exp(-xi * np.clip(age_days, 0, None))

    # 中立場地遮罩：中立場（neutral=True）不套用主場優勢。
    if S.NEUTRAL in df.columns:
        home_mask = (~df[S.NEUTRAL].astype(bool)).to_numpy(dtype=float)
    else:
        home_mask = np.ones(len(df))

    # Elo：若啟用且資料含 Elo 欄位，準備每場的 (Elo_home - Elo_away)/400。
    has_elo = use_elo and S.HOME_ELO in df.columns and S.AWAY_ELO in df.columns
    if has_elo:
        eh = pd.to_numeric(df[S.HOME_ELO], errors="coerce")
        ea = pd.to_numeric(df[S.AWAY_ELO], errors="coerce")
        elo_diff = ((eh - ea) / 400.0).fillna(0.0).to_numpy(dtype=float)
    else:
        elo_diff = np.zeros(len(df))

    # 參數向量：[attack(n-1), defence(n), home_adv, rho, (elo_coef 若啟用)]
    # attack 平均固定為 0（以最後一隊 = -sum(其餘) 表示，少一個自由參數）。
    def unpack(params: np.ndarray):
        att = np.empty(n)
        att[: n - 1] = params[: n - 1]
        att[n - 1] = -att[: n - 1].sum()  # 約束：sum(attack)=0
        defence = params[n - 1 : 2 * n - 1]
        home_adv = params[2 * n - 1]
        rho = params[2 * n]
        elo_coef = params[2 * n + 1] if has_elo else 0.0
        return att, defence, home_adv, rho, elo_coef

    def neg_log_likelihood(params: np.ndarray) -> float:
        att, dfc, home_adv, rho, elo_coef = unpack(params)
        lam = np.exp(att[hi] - dfc[ai] + home_adv * home_mask + elo_coef * elo_diff)
        mu = np.exp(att[ai] - dfc[hi] - elo_coef * elo_diff)
        # 數值保護
        lam = np.clip(lam, 1e-8, 30.0)
        mu = np.clip(mu, 1e-8, 30.0)

        # log Poisson pmf（用 gammaln 算 log(k!)）
        from scipy.special import gammaln

        log_p = (hg * np.log(lam) - lam - gammaln(hg + 1)
                 + ag * np.log(mu) - mu - gammaln(ag + 1))
        # L2 正則化：把攻防參數往 0（聯盟平均）收縮
        penalty = reg * (np.sum(att ** 2) + np.sum(dfc ** 2)) if reg > 0 else 0.0
        if disable_tau:
            ll = weights * log_p
            return -ll.sum() + penalty
        tau = _tau(hg, ag, lam, mu, rho)
        if np.any(tau <= 0):
            return 1e10  # 無效參數重罰
        ll = weights * (log_p + np.log(tau))
        return -ll.sum() + penalty

    # 初始值
    x0_parts = [
        np.zeros(n - 1),          # attack
        np.zeros(n),              # defence
        np.array([0.25]),         # home_adv（典型約 0.2~0.4）
        np.array([rho_init]),     # rho
    ]
    bounds = ([(-3, 3)] * (n - 1)
              + [(-3, 3)] * n
              + [(-1, 1)]       # home_adv
              + [(-0.2, 0.2)])  # rho 通常很小
    if has_elo:
        x0_parts.append(np.array([0.5]))  # elo_coef 初始值
        bounds.append((-5, 5))
    x0 = np.concatenate(x0_parts)

    res = minimize(
        neg_log_likelihood, x0, method="L-BFGS-B", bounds=bounds,
        options={"maxiter": 2000, "maxfun": 20000, "ftol": 1e-9},
    )
    att, dfc, home_adv, rho, elo_coef = unpack(res.x)
    if verbose:
        extra = f" elo_coef={elo_coef:.3f}" if has_elo else ""
        print(f"[fit] 收斂={res.success} NLL={res.fun:.2f} 球隊數={n} 比賽數={len(df)}{extra}")
    if disable_tau:
        rho = 0.0  # xG 模式未使用 tau，rho 無意義，存 0 避免推論時誤用

    # 每隊最新 Elo（推論時用）：取訓練資料中該隊最後一次出現的 Elo
    team_elo: dict[str, float] = {}
    if has_elo:
        for _, r in df.iterrows():
            he = r.get(S.HOME_ELO)
            ae = r.get(S.AWAY_ELO)
            if pd.notna(he):
                team_elo[r[S.HOME]] = float(he)
            if pd.notna(ae):
                team_elo[r[S.AWAY]] = float(ae)

    return DixonColesModel(
        teams=teams,
        attack={t: float(att[idx[t]]) for t in teams},
        defence={t: float(dfc[idx[t]]) for t in teams},
        home_adv=float(home_adv),
        rho=float(rho),
        max_goals=max_goals,
        trained_until=reference_date,
        elo_coef=float(elo_coef),
        team_elo=team_elo,
    )
