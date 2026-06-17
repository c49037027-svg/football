"""預測內容與渲染測試。"""
from footy import report
from footy.models import dixon_coles as dc
from footy.models import markets
from footy.models.dixon_coles import score_matrix_from_rates
from footy.predict import predict_fixtures, predict_match

import pandas as pd


def test_btts_and_correct_score():
    mat = score_matrix_from_rates(1.5, 1.2, -0.03, 10)
    bt = markets.btts(mat)
    assert abs(bt["yes"] + bt["no"] - 1.0) < 1e-9
    cs = markets.correct_score(mat, top_n=5)
    assert len(cs) == 5
    # 機率由高到低
    assert cs[0][1] >= cs[-1][1]
    # 最可能比分與 correct_score 第一個一致
    assert markets.most_likely_score(mat) == cs[0][0]


def test_expected_goals_from_matrix_close_to_rates():
    mat = score_matrix_from_rates(1.6, 1.0, 0.0, 12)
    eh, ea = markets.expected_goals_from_matrix(mat)
    assert abs(eh - 1.6) < 0.05 and abs(ea - 1.0) < 0.05


def test_predict_match_fields(synthetic_df):
    model = dc.fit(synthetic_df, half_life_days=10_000)
    h, a = model.teams[0], model.teams[1]
    p = predict_match(model, h, a, history=synthetic_df)
    assert abs(p.p_home + p.p_draw + p.p_away - 1.0) < 1e-9
    assert len(p.correct_scores) == 5
    assert 2.5 in p.over_under
    assert p.tip  # 有產生 tip
    assert set(p.home_form) <= set("WDL")  # 狀態字串只含 WDL


def test_render_outputs(synthetic_df):
    model = dc.fit(synthetic_df, half_life_days=10_000)
    fx = pd.DataFrame([{"home": model.teams[0], "away": model.teams[1]},
                       {"home": model.teams[2], "away": model.teams[3]}])
    preds = predict_fixtures(model, fx, history=synthetic_df)
    assert len(preds) == 2
    con = report.render_console(preds[0])
    assert "1X2" in con
    md = report.render_markdown(preds)
    assert "## " in md and "BTTS" in md
    htmls = report.render_html(preds, "測試")
    assert "<!doctype html>" in htmls and "正確比分" in htmls
