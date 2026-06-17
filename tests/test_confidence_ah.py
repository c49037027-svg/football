"""可信度（決斷度×資料支撐）與指定盤口讓球線測試。"""
from footy import analysis, report
from footy.models import dixon_coles as dc


def test_conf_tier_not_just_probability():
    # 同樣 60% 機率，資料充足 vs 資料稀少 → 可信度星等不同（非循環）
    s_full = report._conf_tier(0.60, support=1.0)[0]
    s_thin = report._conf_tier(0.60, support=0.3)[0]
    assert s_thin < s_full
    # 60% 決斷度低 → 星等不應是滿星（避免「高機率＝高信心」的循環）
    assert s_full < 5
    # 壓倒性 90% + 充足資料 → 高星
    assert report._conf_tier(0.90, 1.0)[0] >= 4


def test_ah_line_override_picks_better_side(synthetic_df):
    model = dc.fit(synthetic_df, half_life_days=10_000)
    # 找出有讓步的一組（攻擊最強 vs 最弱）
    strength = {t: model.attack[t] - model.defence[t] for t in model.teams}
    h = max(strength, key=strength.get)
    a = min(strength, key=strength.get)
    base = analysis.analyze(model, h, a, neutral=True, n_sims=4000)
    # 指定一條比模型線更深的盤口（主隊讓更多）→ 模型可能改推薦受讓方
    deep = analysis.analyze(model, h, a, neutral=True, n_sims=4000,
                            ah_line_override=base.ah_line - 1.0)
    assert deep.ah_line == base.ah_line - 1.0
    # 兩邊公平賠率仍為合法數字
    assert deep.ah_home_odds > 1.0 and deep.ah_away_odds > 1.0


def test_data_support_thin_for_unknown_pairs(synthetic_df):
    model = dc.fit(synthetic_df, half_life_days=10_000)
    h, a = model.teams[0], model.teams[1]
    # 無歷史 → 預設中性支撐
    a0 = analysis.analyze(model, h, a, history=None, neutral=True, n_sims=2000)
    assert 0.0 <= a0.data_support <= 1.0
    # 有歷史 → 介於 0.3~1.0
    a1 = analysis.analyze(model, h, a, history=synthetic_df, neutral=True, n_sims=2000)
    assert 0.3 <= a1.data_support <= 1.0


def test_formation_fields_on_analysis(synthetic_df):
    model = dc.fit(synthetic_df, half_life_days=10_000)
    h, a = model.teams[0], model.teams[1]
    res = analysis.analyze(model, h, a, neutral=True, n_sims=2000,
                           home_formation="4-3-3", away_formation="5-4-1")
    assert res.home_formation == "4-3-3" and res.away_formation == "5-4-1"
    html = report.render_analysis_html(res, "t")
    assert "陣型" in html and "4-3-3" in html and "5-4-1" in html
    assert "可信度" in html
