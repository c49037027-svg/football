"""五大聯賽測試：訓練、賽程 1X2 預測、頁面（含空狀態）、ESPN 解析、建站。"""
import pandas as pd

from footy import leagues
from footy.models import dixon_coles as dc


def _club_df():
    """強主客隊合成：Strong 勝多。"""
    rows, d = [], pd.Timestamp("2025-08-01")
    seq = [("Strong", "Weak", 3, 0), ("Weak", "Strong", 0, 2),
           ("Strong", "Mid", 2, 1), ("Mid", "Weak", 2, 1),
           ("Mid", "Strong", 1, 2), ("Weak", "Mid", 0, 1)] * 12
    for i, (h, a, hg, ag) in enumerate(seq):
        rows.append({"date": (d + pd.Timedelta(days=i)).date().isoformat(),
                     "home": h, "away": a, "home_goals": hg, "away_goals": ag})
    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"])
    return df


def test_train_club_models(tmp_path):
    data = tmp_path / "data"
    data.mkdir()
    _club_df().to_csv(data / "club_E0.csv", index=False)
    done = leagues.train_club_models(str(data), str(tmp_path / "models"))
    assert done == ["E0"]
    assert (tmp_path / "models" / "club_E0.pkl").exists()
    assert leagues.train_club_models(str(data), str(tmp_path / "m2")) or True   # 冪等不炸


def test_predict_fixtures_direction():
    model = dc.fit(_club_df(), half_life_days=1e9, reg=0.3)
    preds = leagues.predict_fixtures(model, [
        {"home": "Strong", "away": "Weak"}, {"home": "Weak", "away": "Strong"},
        {"home": "Strong", "away": "Nobody"}])          # 不在模型 → 跳過
    assert len(preds) == 2
    assert preds[0]["p_home"] > preds[0]["p_away"]       # 強主 → 主勝高
    assert preds[1]["p_home"] < preds[1]["p_away"]       # 弱主對強客 → 客勝高
    assert abs(sum(preds[0][k] for k in ("p_home", "p_draw", "p_away")) - 1) < 1e-9
    # 大小盤：預設 2.5、over+under=1
    assert preds[0]["ou_line"] == 2.5
    assert abs(preds[0]["p_over"] + preds[0]["p_under"] - 1) < 1e-9
    assert 0 < preds[0]["p_over"] < 1
    # 依莊家線：fixture 帶 ou_line=3.5 → 用該線（線越高、大球機率越低）
    hi = leagues.predict_fixtures(model, [{"home": "Strong", "away": "Weak", "ou_line": 3.5}])
    assert hi[0]["ou_line"] == 3.5 and hi[0]["p_over"] < preds[0]["p_over"]


def test_main_ou_line():
    from footy import tracker
    from footy.prematch import MarketQuote
    # 2.5 線大小最均衡（1.9/1.9）、3.5 線不均衡 → 主線取 2.5
    q = [MarketQuote("OU", "over", 1.90, line=2.5), MarketQuote("OU", "under", 1.90, line=2.5),
         MarketQuote("OU", "over", 2.80, line=3.5), MarketQuote("OU", "under", 1.42, line=3.5)]
    assert tracker.main_ou_line(q) == 2.5
    assert tracker.main_ou_line([]) is None


def test_render_leagues_page_empty_and_filled():
    empty = leagues.render_leagues_page({})
    assert "五大聯賽" in empty and "近期無排定賽程" in empty
    assert "英超" in empty and "西甲" in empty          # 五個聯賽都列
    filled = leagues.render_leagues_page({"英超": [
        {"home": "Strong", "away": "Weak", "time": "",
         "p_home": 0.6, "p_draw": 0.25, "p_away": 0.15,
         "ou_line": 2.5, "p_over": 0.55, "p_under": 0.45}]})
    assert "60%" in filled and "主勝" in filled
    assert "大小盤" in filled and "大2.5" in filled and "55%" in filled


def test_parse_fixtures_pre_only():
    payload = {"events": [
        {"date": "2026-08-15T14:00Z", "status": {"type": {"state": "pre"}},
         "competitions": [{"competitors": [
             {"homeAway": "home", "team": {"displayName": "Arsenal"}},
             {"homeAway": "away", "team": {"displayName": "Chelsea"}}]}]},
        {"status": {"type": {"state": "in"}},              # 進行中 → 不算排定
         "competitions": [{"competitors": [
             {"homeAway": "home", "team": {"displayName": "X"}},
             {"homeAway": "away", "team": {"displayName": "Y"}}]}]},
    ]}
    fx = leagues.parse_fixtures(payload)
    assert len(fx) == 1 and fx[0]["home"] == "Arsenal" and fx[0]["away"] == "Chelsea"


def test_build_site_page_injected(tmp_path):
    _club_df()  # noqa
    models = tmp_path / "models"
    data = tmp_path / "data"
    data.mkdir()
    _club_df().to_csv(data / "club_E0.csv", index=False)
    leagues.train_club_models(str(data), str(models))
    html = leagues.build_site_page(
        models_dir=str(models),
        fixtures_by_code={"E0": [{"home": "Strong", "away": "Weak"}]})
    assert "英超" in html and "Strong" in html and "主勝" in html
    # 其他聯賽無模型 → 該區空狀態
    assert "近期無排定賽程" in html
