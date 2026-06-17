"""國際賽 Elo、計數模型、單場分析與渲染測試。"""
import numpy as np
import pandas as pd

from footy import analysis, counts, report
from footy.data import schema as S
from footy.intl import data as intl
from footy.models import dixon_coles as dc


def _intl_df():
    """小型國際賽合成資料（含 neutral / tournament）。"""
    rng = np.random.default_rng(0)
    teams = ["A", "B", "C", "D", "E", "F"]
    strength = {t: i * 0.3 for i, t in enumerate(teams)}  # F 最強
    rows = []
    start = pd.Timestamp("2018-01-01")
    for r in range(40):
        rng.shuffle(teams)
        for i in range(0, len(teams) - 1, 2):
            h, a = teams[i], teams[i + 1]
            lam = np.exp(strength[h] - strength[a] + 0.2)
            mu = np.exp(strength[a] - strength[h])
            rows.append({
                S.DATE: start + pd.Timedelta(days=r * 7),
                S.HOME: h, S.AWAY: a,
                S.HOME_GOALS: int(rng.poisson(lam)), S.AWAY_GOALS: int(rng.poisson(mu)),
                S.NEUTRAL: bool(r % 2), S.TOURNAMENT: "Friendly",
            })
    return pd.DataFrame(rows).sort_values(S.DATE).reset_index(drop=True)


# ---------- Elo ----------
def test_compute_elo_ranks_stronger_higher():
    df = _intl_df()
    annotated, elo = intl.compute_elo(df)
    assert S.HOME_ELO in annotated.columns and S.AWAY_ELO in annotated.columns
    # 最強隊 F 的 Elo 應高於最弱隊 A
    assert elo["F"] > elo["A"]
    # 賽前 Elo 不應有 NaN
    assert annotated[S.HOME_ELO].notna().all()


def test_normalize_results_schema():
    raw = pd.DataFrame({
        "date": ["2022-11-20"], "home_team": ["Qatar"], "away_team": ["Ecuador"],
        "home_score": [0], "away_score": [2], "tournament": ["FIFA World Cup"],
        "city": ["Al Khor"], "country": ["Qatar"], "neutral": [False],
    })
    df = intl.normalize_results(raw)
    assert df.iloc[0][S.HOME] == "Qatar" and df.iloc[0][S.AWAY_GOALS] == 2


# ---------- 計數模型 ----------
def test_corners_and_cards_estimates():
    c = counts.estimate_corners(0.3, -0.1)
    assert c.total > 0 and c.home > c.away  # 強隊角球較多
    assert 0.5 <= c.confidence <= 1.0
    assert c.recommend in ("買大", "買小")
    k = counts.estimate_cards(0.0, 0.0, knockout=True)
    k0 = counts.estimate_cards(0.0, 0.0, knockout=False)
    assert k.total >= k0.total  # 淘汰賽黃牌加成


# ---------- 中立場模型 ----------
def test_neutral_removes_home_advantage():
    df = _intl_df()
    annotated, _ = intl.compute_elo(df)
    model = dc.fit(annotated, half_life_days=10_000, use_elo=True)
    h, a = model.teams[0], model.teams[1]
    lam_home, _ = model.expected_goals(h, a, neutral=False)
    lam_neu, _ = model.expected_goals(h, a, neutral=True)
    # 中立場主隊預期進球應較低（少了主場優勢）
    assert lam_neu <= lam_home


# ---------- 單場分析 ----------
def test_analyze_all_panels(synthetic_df):
    model = dc.fit(synthetic_df, half_life_days=10_000)
    h, a = model.teams[0], model.teams[1]
    res = analysis.analyze(model, h, a, history=synthetic_df, neutral=True,
                           n_sims=5000, seed=1)
    # 1X2 合理
    assert abs(res.p_home + res.p_draw + res.p_away - 1.0) < 1e-9
    # 上半場機率合理且總和≈1
    assert abs(res.fh_home + res.fh_draw + res.fh_away - 1.0) < 0.02
    # 上半場進球機率應低於全場（大1.5 上半場 < 大1.5 全場）
    assert res.fh_over[1.5] < res.over_under[1.5]["over"] + 0.02
    # 各面板存在
    assert 1.5 in res.over_under and 2.5 in res.over_under and 3.5 in res.over_under
    assert res.corners.total > 0 and res.cards.total > 0
    assert res.ah_reco
    assert 0 <= res.btts_yes <= 1


def test_analyze_render(synthetic_df):
    model = dc.fit(synthetic_df, half_life_days=10_000)
    h, a = model.teams[0], model.teams[1]
    res = analysis.analyze(model, h, a, history=synthetic_df, n_sims=3000)
    con = report.render_analysis_console(res)
    assert "AI 預測比分" in con and "上半場" in con
    htmls = report.render_analysis_html(res, "測試分析")
    assert "<!doctype html>" in htmls
    for label in ("大小球", "兩隊都進球", "亞盤讓球", "角球預測", "黃牌預測", "上半場走向", "影響因子"):
        assert label in htmls
