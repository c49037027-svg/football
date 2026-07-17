"""手動走地測試：表單頁渲染、足球/MLB 手動輸入 → 結果卡片。"""
import numpy as np
import pandas as pd

from footy import manual_live
from footy.models import dixon_coles as dc


def _foot_model():
    rows = []
    d = pd.Timestamp("2025-01-01")
    seq = [("Alpha", "Beta", 2, 0), ("Beta", "Alpha", 1, 1),
           ("Alpha", "Beta", 3, 1), ("Beta", "Alpha", 0, 2)] * 12
    for i, (h, a, hg, ag) in enumerate(seq):
        rows.append({"date": (d + pd.Timedelta(days=i)).date().isoformat(),
                     "home": h, "away": a, "home_goals": hg, "away_goals": ag})
    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"])
    return dc.fit(df, half_life_days=1e9, max_goals=10, reg=0.5)


def test_foot_manual_result_and_directions():
    m = _foot_model()
    # 無輸入 → 空
    assert manual_live.foot_manual_result(m, {}) == ""
    # 領先 70' → 主勝機率高、卡片含公平賠率表
    html = manual_live.foot_manual_result(
        m, {"home": "Alpha", "away": "Beta", "minute": "70", "hg": "1", "ag": "0"})
    assert "大小線" in html and "主勝" in html
    # 同隊/不在模型 → 錯誤訊息
    assert "不能相同" in manual_live.foot_manual_result(
        m, {"home": "Alpha", "away": "Alpha"})
    assert "不在模型" in manual_live.foot_manual_result(
        m, {"home": "Alpha", "away": "Nobody"})
    # 紅牌方向：主隊吃紅牌 → 主勝下降
    from footy import foot_live
    base = foot_live.live_probs(m, "Alpha", "Beta", 60, 0, 0)
    red = foot_live.live_probs(m, "Alpha", "Beta", 60, 0, 0, home_red=1)
    assert red["p_home"] < base["p_home"]
    # 非法輸入被夾住不炸
    ok = manual_live.foot_manual_result(
        m, {"home": "Alpha", "away": "Beta", "minute": "999", "hg": "-3", "ag": "abc"})
    assert "大小線" in ok


def test_render_manual_page_form_and_result(tmp_path):
    m = _foot_model()
    # 無輸入：只有表單
    page = manual_live.render_manual_page(m, {}, mlb_model_path=str(tmp_path / "no.pkl"))
    assert "手動走地" in page and "sport" in page and "中立場" in page
    assert "分析結果" not in page
    # 足球輸入：結果在頁上、表單保留選擇
    page2 = manual_live.render_manual_page(
        m, {"sport": "foot", "home": "Alpha", "away": "Beta",
            "minute": "70", "hg": "1", "ag": "0", "neutral": "1"},
        mlb_model_path=str(tmp_path / "no.pkl"))
    assert "分析結果" in page2 and "大小線" in page2
    assert 'value="Alpha" selected' in page2
    # MLB 模型缺失：頁面仍可用（表單顯示未載入），選 mlb 時回錯誤卡
    page3 = manual_live.render_manual_page(
        m, {"sport": "mlb", "mhome": "X", "maway": "Y"},
        mlb_model_path=str(tmp_path / "no.pkl"))
    assert "MLB 模型未載入" in page3


def test_mlb_manual_result(tmp_path):
    # 用足球式 DC 模型充當 MLB 模型（介面相同：attack/expected_goals）
    m = _foot_model()
    mp = tmp_path / "mlb.pkl"
    m.save(mp)
    csv = tmp_path / "mlb.csv"
    csv.write_text("date,home,away,home_goals,away_goals,game_pk\n"
                   "2025-05-01,Alpha,Beta,4,3,1\n2025-05-02,Beta,Alpha,2,5,2\n")
    html = manual_live.mlb_manual_result(
        {"mhome": "Alpha", "maway": "Beta", "inning": "7", "half": "bottom",
         "outs": "2", "b2": "1", "hs": "4", "as": "3"},
        model_path=str(mp), data_path=str(csv))
    assert "7局下" in html and "2出局" in html and "二" in html
    assert "主勝" in html and "大小線" in html
    assert manual_live.mlb_manual_result({}, str(mp), str(csv)) == ""
