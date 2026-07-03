"""MLB 模組測試：schedule 解析、錢線/大小/讓分數學、端到端。"""
import numpy as np
import pandas as pd
import pytest

from footy import mlb
from footy.models import dixon_coles as dc

FIXTURE = {
    "dates": [{
        "date": "2026-07-01",
        "games": [
            {   # 已完賽例行賽 → 收
                "gameType": "R", "officialDate": "2026-07-01",
                "status": {"abstractGameState": "Final"},
                "teams": {
                    "home": {"team": {"name": "New York Yankees"}, "score": 5,
                             "probablePitcher": {"fullName": "G. Cole"}},
                    "away": {"team": {"name": "Boston Red Sox"}, "score": 3},
                },
            },
            {   # 未開打 → finals_only 時略過
                "gameType": "R", "officialDate": "2026-07-01",
                "status": {"abstractGameState": "Preview"},
                "teams": {
                    "home": {"team": {"name": "Los Angeles Dodgers"}},
                    "away": {"team": {"name": "San Diego Padres"}},
                },
            },
            {   # 熱身賽 → 永遠略過
                "gameType": "S", "officialDate": "2026-07-01",
                "status": {"abstractGameState": "Final"},
                "teams": {
                    "home": {"team": {"name": "Springville"}, "score": 1},
                    "away": {"team": {"name": "Testtown"}, "score": 0},
                },
            },
        ],
    }],
}


def test_parse_schedule_finals_only():
    rows = mlb.parse_schedule(FIXTURE, finals_only=True)
    assert len(rows) == 1
    r = rows[0]
    assert r["home"] == "New York Yankees" and r["home_goals"] == 5
    assert r["away"] == "Boston Red Sox" and r["away_goals"] == 3
    assert r["home_pitcher"] == "G. Cole"


def test_parse_schedule_all_games():
    rows = mlb.parse_schedule(FIXTURE, finals_only=False)
    assert len(rows) == 2  # 熱身賽仍排除
    assert {r["status"] for r in rows} == {"Final", "Preview"}


def test_moneyline_symmetric_matrix():
    # 對稱矩陣 → 五五波；對角線（延長賽）各半分
    n = 6
    mat = np.ones((n, n)) / (n * n)
    p_h, p_a = mlb.moneyline(mat)
    assert abs(p_h - 0.5) < 1e-9 and abs(p_h + p_a - 1.0) < 1e-9


def test_moneyline_favours_stronger():
    # 質量集中在主隊多分 → 主勝率高
    mat = np.zeros((6, 6))
    mat[4, 1] = 0.7
    mat[2, 2] = 0.3   # 平手部分各半分
    p_h, p_a = mlb.moneyline(mat)
    assert abs(p_h - 0.85) < 1e-9


def _mlb_model():
    """合成 MLB 資料：得分 ~Poisson(4.5)，A 隊明顯較強。"""
    rng = np.random.default_rng(7)
    teams = [f"Team{i}" for i in range(6)]
    strength = {t: 0.25 if t == "Team0" else (-0.25 if t == "Team5" else 0.0)
                for t in teams}
    rows = []
    day = pd.Timestamp("2025-04-01")
    for rnd in range(60):
        for i in range(0, 6, 2):
            h, a = teams[(i + rnd) % 6], teams[(i + rnd + 1) % 6]
            lam = 4.5 * np.exp(strength[h] - strength[a] * 0.5 + 0.05)
            mu = 4.2 * np.exp(strength[a] - strength[h] * 0.5)
            rows.append(dict(date=day + pd.Timedelta(days=rnd), home=h, away=a,
                             home_goals=int(rng.poisson(lam)),
                             away_goals=int(rng.poisson(mu))))
    df = pd.DataFrame(rows)
    return dc.fit(df, half_life_days=10_000, max_goals=20, rho_init=0.0, reg=0.2)


def test_analyze_game_end_to_end():
    model = _mlb_model()
    m = mlb.analyze_game(model, "Team0", "Team5", total_line=8.5, run_line=-1.5)
    # 機率健全性
    assert abs(m.p_home + m.p_away - 1.0) < 1e-9
    assert abs(m.p_over + m.p_under - 1.0) < 1e-6   # 8.5 無走盤
    assert 0.0 <= m.p_cover_home <= 1.0
    # 強隊在家對弱隊 → 錢線應明顯 > 50%
    assert m.p_home > 0.55
    # 公平賠率互為機率倒數
    assert abs(m.ml_home_odds - round(1 / m.p_home, 2)) < 0.02
    # 期望得分在棒球合理範圍
    assert 2.0 < m.exp_home < 9.0 and 2.0 < m.exp_away < 9.0
    assert len(m.top_scores) == 4


def test_zh_mlb():
    assert mlb.zh_mlb("New York Yankees") == "洋基"
    assert mlb.zh_mlb("Unknown Club") == "Unknown Club"


def test_cli_wiring():
    from click.testing import CliRunner
    from footy.cli import cli
    out = CliRunner().invoke(cli, ["mlb", "--help"])
    assert out.exit_code == 0
    for cmd in ("fetch", "train", "analyze", "today"):
        assert cmd in out.output
