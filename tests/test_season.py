"""整季蒙地卡羅模擬測試。"""
import numpy as np

from footy import report, season
from footy.models import dixon_coles as dc


def test_round_robin_count():
    teams = ["A", "B", "C", "D"]
    fx = season.round_robin_fixtures(teams)
    # 雙循環：n*(n-1) 場
    assert len(fx) == 4 * 3
    # 不含自己對自己
    assert all(h != a for h, a in fx)


def test_standings_from_matches(synthetic_df):
    table = season.standings_from_matches(synthetic_df)
    # 每隊已踢場數 > 0，積分非負
    assert all(st.played > 0 for st in table.values())
    assert all(st.points >= 0 for st in table.values())
    # 積分上限檢查：不超過 3*played
    assert all(st.points <= 3 * st.played for st in table.values())


def test_simulate_season_probabilities_sum(synthetic_df):
    model = dc.fit(synthetic_df, half_life_days=10_000)
    teams = model.teams
    sim = season.simulate_season(model, teams, n_sims=2000, seed=1)
    # 冠軍機率總和 ≈ 1
    total_title = sum(r.title_pct for r in sim.results.values())
    assert abs(total_title - 1.0) < 1e-9
    # 每隊名次分布加總 ≈ 1
    for r in sim.results.values():
        assert abs(r.position_dist.sum() - 1.0) < 1e-9
    # 名次落在 1..n
    n = len(teams)
    assert all(1 <= r.exp_position <= n for r in sim.results.values())
    # 降級機率總和 ≈ 降級名額（每次模擬剛好有 relegation_spots 隊降級）
    total_releg = sum(r.relegation_pct for r in sim.results.values())
    assert abs(total_releg - sim.relegation_spots) < 1e-9


def test_stronger_team_has_higher_title_odds(synthetic_df):
    model = dc.fit(synthetic_df, half_life_days=10_000)
    teams = model.teams
    sim = season.simulate_season(model, teams, n_sims=3000, seed=2)
    # 攻擊最強、防守最好的隊，奪冠機率應高於最弱隊
    strength = {t: model.attack[t] + model.defence[t] for t in teams}
    best = max(strength, key=strength.get)
    worst = min(strength, key=strength.get)
    assert sim.results[best].title_pct >= sim.results[worst].title_pct


def test_start_standings_carry_over(synthetic_df):
    model = dc.fit(synthetic_df, half_life_days=10_000)
    teams = model.teams
    # 給某隊巨大領先，沒有剩餘賽程 -> 必奪冠
    lead = {t: season.TeamStanding(team=t, points=0) for t in teams}
    lead[teams[0]].points = 100
    sim = season.simulate_season(model, teams, fixtures=[], start_standings=lead,
                                 n_sims=500, seed=3)
    assert sim.results[teams[0]].title_pct == 1.0


def test_season_render(synthetic_df):
    model = dc.fit(synthetic_df, half_life_days=10_000)
    sim = season.simulate_season(model, model.teams, n_sims=500, seed=4)
    con = report.render_season_console(sim)
    assert "冠軍%" in con
    htmls = report.render_season_html(sim, "測試季")
    assert "<!doctype html>" in htmls and "降級" in htmls
    # 表格依冠軍機率排序：第一列冠軍% >= 最後一列
    df = sim.table()
    assert df.iloc[0]["冠軍%"] >= df.iloc[-1]["冠軍%"]
