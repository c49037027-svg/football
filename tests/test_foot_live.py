"""足球走地測試：修正參數行為、ESPN 解析、快照與渲染。"""
import numpy as np
import pandas as pd

from footy import foot_live
from footy.models import dixon_coles as dc


def _model():
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


def test_live_probs_consistency_and_directions():
    m = _model()
    # 開賽 0'：應接近賽前(溫度縮放造成的微小差異內)
    live0 = foot_live.live_probs(m, "Alpha", "Beta", 0, 0, 0)
    mat = m.score_matrix("Alpha", "Beta")
    pre_h = float(np.tril(mat, -1).sum() / mat.sum())
    assert abs(live0["p_home"] - pre_h) < 0.03
    assert abs(live0["p_home"] + live0["p_draw"] + live0["p_away"] - 1) < 1e-9
    # 領先 → 勝率升;越晚領先越穩
    lead60 = foot_live.live_probs(m, "Alpha", "Beta", 60, 1, 0)
    lead85 = foot_live.live_probs(m, "Alpha", "Beta", 85, 1, 0)
    assert lead60["p_home"] > live0["p_home"] + 0.1
    assert lead85["p_home"] > lead60["p_home"]
    # 85' 平手 → 和局機率很高
    tied85 = foot_live.live_probs(m, "Alpha", "Beta", 85, 1, 1)
    assert tied85["p_draw"] > 0.55
    # 紅牌:主隊少一人 → 主勝率下降
    red = foot_live.live_probs(m, "Alpha", "Beta", 60, 1, 0, home_red=1)
    assert red["p_home"] < lead60["p_home"]
    # 大小:當前總球 1,大 1.5 機率 = 再進 ≥1 球
    p_o = foot_live.p_over(lead60, 1.5)
    assert 0.1 < p_o < 0.9
    assert foot_live.p_over(lead60, 0.5) == 1.0     # 已 1 球,必過 0.5


def test_rate_scale_schedule():
    assert foot_live._rate_scale(0) == 1.0
    assert abs(foot_live._rate_scale(45) - foot_live.SECOND_HALF_SCALE) < 1e-9
    assert foot_live._rate_scale(80) == foot_live._rate_scale(45)


def test_parse_espn_soccer():
    payload = {"events": [
        {"id": "1", "status": {"displayClock": "67'", "type": {"state": "in", "name": "STATUS_IN_PROGRESS"}},
         "season": {"slug": "fifa-world-cup"},
         "competitions": [{"competitors": [
             {"homeAway": "home", "score": "2", "team": {"displayName": "Japan"}},
             {"homeAway": "away", "score": "1", "team": {"displayName": "USA"}}]}]},
        {"id": "2", "status": {"displayClock": "45'+3'", "type": {"state": "in", "name": "STATUS_HALFTIME"}},
         "competitions": [{"competitors": [
             {"homeAway": "home", "score": "0", "team": {"displayName": "Türkiye"}},
             {"homeAway": "away", "score": "0", "team": {"displayName": "Brazil"}}]}]},
        {"id": "3", "status": {"type": {"state": "post"}},
         "competitions": [{"competitors": []}]},        # 已完賽 → 排除
    ]}
    rows = foot_live.parse_espn_soccer(payload)
    assert len(rows) == 2
    g = rows[0]
    assert (g["home"], g["away"], g["home_goals"], g["away_goals"]) == ("Japan", "United States", 2, 1)
    assert g["minute"] == 67 and g["phase"] == "in"
    assert rows[1]["home"] == "Turkey" and rows[1]["phase"] == "ht" and rows[1]["minute"] == 45


def test_live_snapshot_and_section(tmp_path):
    m = _model()
    mp = tmp_path / "intl.pkl"
    m.save(mp)
    matches = [
        {"home": "Alpha", "away": "Beta", "home_goals": 1, "away_goals": 0,
         "minute": 60, "phase": "in", "league": "test"},
        {"home": "Nobody", "away": "Beta", "home_goals": 0, "away_goals": 0,
         "minute": 10, "phase": "in", "league": "test"},   # 不在模型 → skipped
    ]
    snap = foot_live.live_snapshot(str(mp), matches=matches)
    assert len(snap["rows"]) == 1 and len(snap["skipped"]) == 1
    r = snap["rows"][0]
    assert r["p"]["p_home"] > 0.5
    assert r["fair"]["home_odds"] < r["fair"]["away_odds"]
    lines = [ln for ln, _ in r["fair"]["over_lines"]]
    assert lines == [1.5, 2.5, 3.5]                 # 當前 1 球 → 由 1.5 起三條
    html_out = foot_live.render_live_section(snap)
    assert "足球走地" in html_out and "60" in html_out
    assert "大小線" in html_out and "預期總球" in html_out
    empty = foot_live.render_live_section({"rows": [], "skipped": []})
    assert "無進行中的比賽" in empty
