"""Elo–Poisson 模型測試。"""
import pandas as pd

from footy.data import schema as S
from footy.models.elo_poisson import EloPoissonModel, fit_elo_poisson


def _intl_df():
    rng = __import__("numpy").random.default_rng(0)
    teams = ["A", "B", "C", "D", "E", "F"]
    elo = {t: 1400 + i * 80 for i, t in enumerate(teams)}  # F 最強
    rows = []
    start = pd.Timestamp("2016-01-01")
    for r in range(60):
        rng.shuffle(teams)
        for i in range(0, len(teams) - 1, 2):
            h, a = teams[i], teams[i + 1]
            import numpy as np
            dr = (elo[h] - elo[a]) / 400.0 + 0.2
            lam, mu = 1.4 * np.exp(0.4 * dr), 1.4 * np.exp(-0.4 * dr)
            rows.append({S.DATE: start + pd.Timedelta(days=r * 7),
                         S.HOME: h, S.AWAY: a,
                         S.HOME_GOALS: int(rng.poisson(lam)),
                         S.AWAY_GOALS: int(rng.poisson(mu)),
                         S.HOME_ELO: elo[h], S.AWAY_ELO: elo[a],
                         S.NEUTRAL: bool(r % 2)})
    return pd.DataFrame(rows).sort_values(S.DATE).reset_index(drop=True)


def test_fit_elo_poisson_learns_sensitivity():
    df = _intl_df()
    m = fit_elo_poisson(df, half_life_days=10_000)
    assert m.s > 0                      # Elo 差越大、進球差越大 → 正敏感度
    assert m.mu > 0 and -0.2 <= m.rho <= 0.2
    assert len(m.team_elo) == 6


def test_elo_poisson_stronger_team_scores_more():
    df = _intl_df()
    m = fit_elo_poisson(df, half_life_days=10_000)
    teams = sorted(m.team_elo, key=lambda t: m.team_elo[t])
    weak, strong = teams[0], teams[-1]
    lam, mu = m.expected_goals(strong, weak, neutral=True)
    assert lam > mu                     # 強隊（主）預期進球高於弱隊
    mat = m.score_matrix(strong, weak, neutral=True)
    assert abs(mat.sum() - 1.0) < 1e-9


def test_elo_poisson_neutral_removes_hfa():
    df = _intl_df()
    m = fit_elo_poisson(df, half_life_days=10_000)
    h, a = m.teams[0], m.teams[1]
    lam_home, _ = m.expected_goals(h, a, neutral=False)
    lam_neu, _ = m.expected_goals(h, a, neutral=True)
    assert lam_home >= lam_neu          # 非中立場有主場加成
