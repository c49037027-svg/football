"""走地引擎測試：開賽=賽前一致、方向性、終局規則、狀態解析、邊界還原。"""
import numpy as np

from footy import mlb_live
from footy.mlb_live import LiveState, boundary_states_from_innings, parse_linescore_state, simulate


def _sim(state, lh=4.6, la=4.3, **kw):
    return mlb_live.simulate(state, lh, la, k=3.0, n_sims=30000, seed=1, **kw)


def test_game_start_matches_pregame_model():
    """開賽瞬間（1 上 0 出局無人）應重現賽前 NB 模型的勝率/總分。"""
    from footy.mlb import nb_score_matrix, moneyline
    lh, la = 4.8, 4.2
    r = _sim(LiveState(1, "top", 0, "000", 0, 0), lh, la)
    mat = nb_score_matrix(lh, la, 3.0, 20)
    p_pre, _ = moneyline(mat)
    assert abs(r["p_home"] - p_pre) < 0.02          # 蒙地卡羅誤差內
    assert abs(r["exp_total"] - (lh + la)) < 0.35   # 9下截斷會略降總分
    assert r["exp_home"] > r["exp_away"]


def test_score_and_inning_direction():
    base = _sim(LiveState(5, "top", 0, "000", 3, 3))
    lead = _sim(LiveState(5, "top", 0, "000", 5, 3))
    late = _sim(LiveState(8, "top", 0, "000", 5, 3))
    assert lead["p_home"] > base["p_home"] + 0.15   # 領先 2 分勝率大增
    assert late["p_home"] > lead["p_home"]          # 越晚領先越穩


def test_endgame_rules():
    # 9 下開始主隊領先 → 其實已結束；勝率 ≈ 1
    r = _sim(LiveState(9, "bottom", 0, "000", 4, 3))
    assert r["p_home"] > 0.995
    # 9 上開始平手 → 勝率貼近五五波，主隊靠得分率優勢微領先
    # （「後攻」本身在得分交換模型中不改變勝率；實證平手進 9 局主隊 ~51-52%）
    r2 = _sim(LiveState(9, "top", 0, "000", 3, 3))
    assert 0.50 < r2["p_home"] < 0.58
    # 9 上客隊領先 5 分 → 主隊勝率很低
    r3 = _sim(LiveState(9, "top", 0, "000", 0, 5))
    assert r3["p_home"] < 0.03


def test_bases_outs_re24_effect():
    # 同分同局：滿壘無出局的進攻方明顯提升該隊表現
    neutral = _sim(LiveState(7, "bottom", 0, "000", 2, 2))
    loaded = _sim(LiveState(7, "bottom", 0, "111", 2, 2))
    two_out = _sim(LiveState(7, "bottom", 2, "000", 2, 2))
    assert loaded["p_home"] > neutral["p_home"] + 0.08
    assert two_out["p_home"] < neutral["p_home"]
    assert loaded["exp_total"] > neutral["exp_total"] + 1.0


def test_p_over_monotone():
    r = _sim(LiveState(3, "top", 0, "000", 2, 1))
    assert mlb_live.p_over(r, 6.5) > mlb_live.p_over(r, 9.5)
    assert 0.0 <= mlb_live.p_over(r, 30.5) < 0.02


def test_parse_linescore_state():
    ls = {"currentInning": 6, "inningState": "Bottom", "outs": 1,
          "teams": {"home": {"runs": 3}, "away": {"runs": 5}},
          "offense": {"first": {"id": 1}, "third": {"id": 2}}}
    s = parse_linescore_state(ls)
    assert (s.inning, s.half, s.outs, s.bases) == (6, "bottom", 1, "101")
    assert (s.home_score, s.away_score) == (3, 5)
    # Middle of 7th = 7 下開始
    s2 = parse_linescore_state({"currentInning": 7, "inningState": "Middle",
                                "outs": 3, "teams": {"home": {"runs": 0},
                                                     "away": {"runs": 0}}})
    assert (s2.inning, s2.half, s2.outs, s2.bases) == (7, "bottom", 0, "000")
    # End of 7th = 8 上開始
    s3 = parse_linescore_state({"currentInning": 7, "inningState": "End",
                                "teams": {"home": {"runs": 2}, "away": {"runs": 1}}})
    assert (s3.inning, s3.half) == (8, "top")
    assert parse_linescore_state({}) is None
    # outs=3 過渡窗口（第三出局後、狀態翻 Middle 前）：視為下個半局開局
    s4 = parse_linescore_state({"currentInning": 5, "inningState": "Top",
                                "outs": 3, "offense": {"first": {"id": 9}},
                                "teams": {"home": {"runs": 1}, "away": {"runs": 4}}})
    assert (s4.inning, s4.half, s4.outs, s4.bases) == (5, "bottom", 0, "000")
    s5 = parse_linescore_state({"currentInning": 5, "inningState": "Bottom",
                                "outs": 3, "teams": {"home": {"runs": 1},
                                                     "away": {"runs": 4}}})
    assert (s5.inning, s5.half, s5.outs) == (6, "top", 0)
    # 進行中但比分缺失 ≠ 0-0：不出價
    assert parse_linescore_state({"currentInning": 3, "inningState": "Top",
                                  "teams": {"home": {}, "away": {"runs": 2}}}) is None


def test_boundary_states_from_innings():
    innings = [
        {"num": 1, "away": {"runs": 1}, "home": {"runs": 0}},
        {"num": 2, "away": {"runs": 0}, "home": {"runs": 3}},
        {"num": 3, "away": {"runs": 2}, "home": {}},   # 9局前不會發生,防呆:home 缺 runs → 停
    ]
    bs = boundary_states_from_innings(innings, 3, 3)
    labels = [b["label"] for b in bs]
    assert labels[:4] == ["T1", "B1", "T2", "B2"]
    b2 = next(b for b in bs if b["label"] == "B2")
    assert (b2["state"].home_score, b2["state"].away_score) == (0, 1)
    t3 = next(b for b in bs if b["label"] == "T3")
    assert (t3["state"].home_score, t3["state"].away_score) == (3, 1)
    # 9 下沒打（主隊領先）→ 不產生 B9
    innings9 = [{"num": i, "away": {"runs": 0}, "home": {"runs": 1 if i == 1 else 0}}
                for i in range(1, 9)]
    innings9.append({"num": 9, "away": {"runs": 0}, "home": {"runs": None}})
    bs9 = boundary_states_from_innings(innings9, 1, 0)
    assert "T9" in [b["label"] for b in bs9]
    assert "B9" not in [b["label"] for b in bs9]


def test_walkoff_truncation_totals():
    """9 下平手時打 → 總分期望應高於主隊已領先(9下不打)的鏡像情境。"""
    tied = _sim(LiveState(9, "top", 0, "000", 4, 4))
    # 9 上結束平手時,還有 9 下(可能延長) → 期望總分至少 +0.4
    assert tied["exp_total"] > 8.0 + 0.4


def test_live_snapshot_and_page(tmp_path):
    """live_snapshot + render_live_page:注入假 schedule payload,不打網路。"""
    import pandas as pd
    from footy.models import dixon_coles as dc

    # 迷你模型(兩隊)與資料檔(球場係數/離散度用)
    rows = []
    d = pd.Timestamp("2026-06-01")
    rng_scores = [(5, 3), (4, 2), (6, 5), (3, 4), (5, 2), (4, 4)] * 10
    for i, (h, a) in enumerate(rng_scores):
        rows.append({"date": (d + pd.Timedelta(days=i)).date().isoformat(),
                     "home": "New York Yankees", "away": "Boston Red Sox",
                     "home_goals": h, "away_goals": a})
        rows.append({"date": (d + pd.Timedelta(days=i)).date().isoformat(),
                     "home": "Boston Red Sox", "away": "New York Yankees",
                     "home_goals": a, "away_goals": h})
    df = pd.DataFrame(rows)
    csv = tmp_path / "mlb.csv"
    df.to_csv(csv, index=False)
    df["date"] = pd.to_datetime(df["date"])
    model = dc.fit(df, half_life_days=1e9, max_goals=20, rho_init=0.0, reg=0.3)
    mp = tmp_path / "mlb.pkl"
    model.save(mp)
    payload = {"dates": [{"date": "2026-07-12", "games": [
        {"gamePk": 111, "gameType": "R", "officialDate": "2026-07-12",
         "gameDate": "2026-07-12T23:00:00Z",
         "status": {"abstractGameState": "Live"},
         "teams": {"home": {"team": {"name": "New York Yankees"}, "score": 3},
                   "away": {"team": {"name": "Boston Red Sox"}, "score": 2}},
         "linescore": {"currentInning": 6, "inningState": "Bottom", "outs": 1,
                       "teams": {"home": {"runs": 3}, "away": {"runs": 2}},
                       "offense": {"second": {"id": 9}}}},
        {"gamePk": 222, "gameType": "R", "officialDate": "2026-07-12",
         "gameDate": "2026-07-12T23:00:00Z",
         "status": {"abstractGameState": "Preview"},
         "teams": {"home": {"team": {"name": "New York Yankees"}, "score": 0},
                   "away": {"team": {"name": "Boston Red Sox"}, "score": 0}},
         "linescore": {}},
    ]}]}
    snap = mlb_live.live_snapshot(str(mp), str(csv), date="2026-07-12",
                                  n_sims=5000, schedule_payload=payload)
    assert len(snap["rows"]) == 1 and len(snap["others"]) == 1
    r = snap["rows"][0]
    assert r["p_home"] > 0.5                 # 6 局下領先 1 分且強隊 → 主勝率過半
    assert r["fair"]["home_odds"] < r["fair"]["away_odds"]
    lines = [ln for ln, _ in r["fair"]["over_lines"]]
    assert len(lines) == 3 and all(ln % 1 == 0.5 for ln in lines)
    html_out = mlb_live.render_live_page(snap)
    assert "MLB 走地" in html_out and "洋基" in html_out and "紅襪" in html_out
    assert "6局下" in html_out and "公平賠率" in html_out
    # 無比賽時的降級頁
    empty = mlb_live.render_live_page({"date": "2026-07-12", "rows": [], "others": []})
    assert "目前無進行中的比賽" in empty


def test_simulate_extra_inning_top_plays_bottom_half():
    """延長賽上半局狀態：主隊該局下半必須被模擬（回歸：曾被 while inn<=9 跳過）。"""
    st = mlb_live.LiveState(inning=10, half="top", outs=0, bases="010",
                            home_score=5, away_score=5)
    r = mlb_live.simulate(st, 4.5, 4.5, seed=1)
    assert 0.40 < r["p_home"] < 0.60          # 平手 10 上 ≈ 五五波
    st2 = mlb_live.LiveState(inning=10, half="top", outs=1, bases="000",
                             home_score=5, away_score=6)
    r2 = mlb_live.simulate(st2, 4.5, 4.5, seed=1)
    assert 0.15 < r2["p_home"] < 0.40         # 落後 1 分仍有幽靈跑者反攻，不趨近 0
