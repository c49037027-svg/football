"""測試共用：產生合成的歷史賽果（不需網路）。"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from footy.data import schema as S


def make_synthetic_matches(n_teams: int = 10, n_rounds: int = 30,
                           seed: int = 0) -> pd.DataFrame:
    """用已知攻防強度生成 Poisson 比分，讓模型有可學的訊號。"""
    rng = np.random.default_rng(seed)
    teams = [f"T{i:02d}" for i in range(n_teams)]
    attack = {t: rng.normal(0, 0.3) for t in teams}
    defence = {t: rng.normal(0, 0.3) for t in teams}
    home_adv = 0.25

    rows = []
    start = pd.Timestamp("2022-08-01")
    day = 0
    for r in range(n_rounds):
        rng.shuffle(teams)
        for i in range(0, n_teams - 1, 2):
            h, a = teams[i], teams[i + 1]
            lam = np.exp(attack[h] - defence[a] + home_adv)
            mu = np.exp(attack[a] - defence[h])
            hg = rng.poisson(lam)
            ag = rng.poisson(mu)
            rows.append({
                S.DATE: start + pd.Timedelta(days=day),
                S.HOME: h, S.AWAY: a,
                S.HOME_GOALS: int(hg), S.AWAY_GOALS: int(ag),
                # xG：以真實 lambda/mu 加噪音，模擬 xG 較進球更平滑
                S.HOME_XG: round(float(lam + rng.normal(0, 0.2)), 2),
                S.AWAY_XG: round(float(mu + rng.normal(0, 0.2)), 2),
                # 給個粗略的「市場」賠率（公平機率加 5% vig），回測會用到
                S.ODDS_HOME: round(1.0 / max(0.05, 0.40) * 1.05, 2),
                S.ODDS_DRAW: round(1.0 / max(0.05, 0.27) * 1.05, 2),
                S.ODDS_AWAY: round(1.0 / max(0.05, 0.33) * 1.05, 2),
                # 開盤賠率（略不同於收盤，供 CLV 測試）
                S.ODDS_HOME_OPEN: round(1.0 / max(0.05, 0.40) * 1.05 * 1.03, 2),
                S.ODDS_DRAW_OPEN: round(1.0 / max(0.05, 0.27) * 1.05 * 1.03, 2),
                S.ODDS_AWAY_OPEN: round(1.0 / max(0.05, 0.33) * 1.05 * 1.03, 2),
            })
            day += 3
    return pd.DataFrame(rows).sort_values(S.DATE).reset_index(drop=True)


@pytest.fixture
def synthetic_df():
    return make_synthetic_matches()
