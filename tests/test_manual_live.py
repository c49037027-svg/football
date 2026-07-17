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


def _nba_model():
    from footy import nba
    import numpy as np
    rng = np.random.default_rng(0)
    teams = {"Strong": (118, 106), "Weak": (105, 119), "Mid A": (112, 112)}
    rows, names = [], list(teams)
    d = pd.Timestamp("2025-01-01")
    for r in range(30):
        for i in range(len(names)):
            for j in range(len(names)):
                if i == j:
                    continue
                h, a = names[i], names[j]
                rows.append({"date": (d + pd.Timedelta(days=r)).date().isoformat(),
                             "home": h, "away": a,
                             "home_goals": int(rng.normal((teams[h][0] + teams[a][1]) / 2 + 1.5, 10)),
                             "away_goals": int(rng.normal((teams[a][0] + teams[h][1]) / 2 - 1.5, 10))})
    return nba.fit_ratings(pd.DataFrame(rows), half_life_days=1e9, reg=5.0)


def test_nba_manual_result_directions(tmp_path):
    m = _nba_model()
    mp = tmp_path / "nba.pkl"
    m.save(mp)
    q = {"nhome": "Strong", "naway": "Weak", "quarter": "4", "qmin": "5",
         "nhs": "100", "nas": "90"}
    html = manual_live.nba_manual_result(q, model_path=str(mp))
    assert "第4節" in html and "主勝" in html and "未經走地資料回測" in html
    # 領先 10 分剩 5 分鐘 → 主勝極高
    q2 = dict(q, nhs="90", nas="100")
    html2 = manual_live.nba_manual_result(q2, model_path=str(mp))
    import re
    def p_home(h):
        return float(re.search(r"主勝 <b>([\d.]+)%", h).group(1))
    assert p_home(html) > 90 and p_home(html2) < 20
    # 開賽（第1節剩12分、0-0）≈ 賽前機率
    from footy import nba as _nba
    q3 = {"nhome": "Strong", "naway": "Weak", "quarter": "1", "qmin": "12"}
    pre = _nba.analyze_game(m, "Strong", "Weak").p_home
    assert abs(p_home(manual_live.nba_manual_result(q3, model_path=str(mp))) / 100 - pre) < 0.02
    assert manual_live.nba_manual_result({}, model_path=str(mp)) == ""


def test_edge_card_ev(tmp_path):
    m = _foot_model()
    # 莊家賠率高於公平價 → 正 EV;低於 → 負 EV
    q = {"home": "Alpha", "away": "Beta", "minute": "60", "hg": "1", "ag": "0",
         "f_oh": "9.99", "f_oa": "1.01"}
    html = manual_live.foot_manual_result(m, q)
    assert "盤口對照" in html and "+" in html
    assert 'class="pos"' in html.replace("'", '"') or "class='pos'" in html
    # 沒填賠率 → 無盤口卡
    html2 = manual_live.foot_manual_result(
        m, {"home": "Alpha", "away": "Beta", "minute": "60", "hg": "1", "ag": "0"})
    assert "盤口對照" not in html2
    # 大小線:填 2.75 會被四捨五入到 .5 線(3.0→3.5 不會;2.75→3.0→3.5)
    q3 = dict(q, f_otl="2.5", f_oto="2.2", f_otu="1.6")
    html3 = manual_live.foot_manual_result(m, q3)
    assert "大 2.5" in html3 and "小 2.5" in html3
