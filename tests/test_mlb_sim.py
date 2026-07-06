"""建構式模擬引擎（log5 + 規則引擎 + 蒙地卡羅）測試。純函式為主,可離線驗證。"""
import numpy as np

from footy import mlb_sim as sim


# ---------------- log5 ----------------
def test_log5_identity_and_shift():
    lg = dict(sim.LEAGUE_RATES)
    # 打者=投手=聯盟 → 對戰率回聯盟率
    m = sim.log5_matchup(lg, lg, lg)
    for e in sim.CATS:
        assert abs(m[e] - lg[e]) < 1e-9
    assert abs(sum(m.values()) - 1.0) < 1e-9
    # 強打者(HR 加倍)對中性投手 → HR 機率上升、機率仍正規化
    slug = dict(lg)
    slug["hr"] = lg["hr"] * 2
    m2 = sim.log5_matchup(slug, lg, lg)
    assert m2["hr"] > m["hr"]
    assert abs(sum(m2.values()) - 1.0) < 1e-9


# ---------------- 規則引擎 ----------------
def test_advance_rules():
    # 全壘打清壘 + 全數得分
    assert sim.advance(True, True, True, 0, "hr") == (False, False, False, 0, 4)
    assert sim.advance(False, False, False, 0, "hr") == (False, False, False, 0, 1)
    # 滿壘保送擠回一分,壘包仍滿
    assert sim.advance(True, True, True, 0, "bb") == (True, True, True, 0, 1)
    # 一壘有人保送 → 推進到二壘,三壘不受影響
    assert sim.advance(True, False, False, 0, "bb") == (True, True, False, 0, 0)
    # 一壘空的保送 → 無人被迫,打者上一壘
    assert sim.advance(False, True, True, 0, "bb") == (True, True, True, 0, 0)
    # 一安:三壘跑者回本壘,打者上一壘
    assert sim.advance(False, False, True, 0, "1b") == (True, False, False, 0, 1)
    # 二安:二三壘跑者皆回,打者上二壘
    assert sim.advance(False, True, True, 0, "2b") == (False, True, False, 0, 2)
    # 出局累加
    assert sim.advance(True, False, False, 0, "out") == (True, False, False, 1, 0)


# ---------------- 事件率換算 ----------------
def test_rates_from_batting():
    line = {"pa": 600, "h": 150, "2b": 30, "3b": 3, "hr": 25, "bb": 60, "hbp": 6}
    r = sim.rates_from_batting(line)
    assert abs(sum(r.values()) - 1.0) < 1e-9
    assert r["hr"] > 0 and r["out"] > 0.5
    # 小樣本回聯盟均值
    assert sim.rates_from_batting({"pa": 0}) == sim.LEAGUE_RATES


def test_rates_from_pitching_bf_fallback():
    r = sim.rates_from_pitching({"ip": 180, "h": 150, "hr": 18, "bb": 45})
    assert abs(sum(r.values()) - 1.0) < 1e-9
    assert r["out"] > 0.6


# ---------------- 蒙地卡羅整場 ----------------
def _league_team():
    return [dict(sim.LEAGUE_RATES) for _ in range(9)]


def test_simulate_matrix_normalized_and_sane():
    mat = sim.simulate_matrix(_league_team(), _league_team(),
                              dict(sim.LEAGUE_RATES), dict(sim.LEAGUE_RATES),
                              n_sims=2000, seed=1)
    assert abs(mat.sum() - 1.0) < 1e-9
    ks = np.arange(mat.shape[0])
    mean_home = (mat.sum(axis=1) * ks).sum()
    # 聯盟均隊每場得分落在合理帶（約 3.5~6）
    assert 3.0 < mean_home < 6.5


def test_strong_lineup_beats_weak():
    from footy import mlb
    lg = dict(sim.LEAGUE_RATES)
    strong = [{"bb": 0.11, "1b": 0.16, "2b": 0.06, "3b": 0.005,
               "hr": 0.055, "out": 0.610} for _ in range(9)]
    weak = [{"bb": 0.06, "1b": 0.11, "2b": 0.03, "3b": 0.002,
             "hr": 0.015, "out": 0.783} for _ in range(9)]
    mat = sim.simulate_matrix(strong, weak, lg, lg, n_sims=3000, seed=2)
    p_h, p_a = mlb.moneyline(mat)
    assert p_h > 0.6                       # 強打線主隊明顯較可能贏


def test_analyze_game_sim_plugs_into_picks():
    from footy import mlb
    lg = dict(sim.LEAGUE_RATES)
    m = sim.analyze_game_sim(_league_team(), _league_team(), lg, lg,
                             n_sims=1500, seed=3)
    assert abs(m.p_home + m.p_away - 1.0) < 1e-9
    assert abs(m.p_over + m.p_under - 1.0) < 1e-6
    picks = mlb.picks_for_game(m)          # 沿用既有推薦流程
    assert any(p["market"] == "1X2" for p in picks)


# ---------------- 資料層解析 ----------------
def test_parse_batting_and_lineups():
    payload = {"stats": [{"splits": [
        {"player": {"id": 1, "fullName": "Bat One"}, "team": {"name": "Team A"},
         "stat": {"plateAppearances": 600, "hits": 150, "doubles": 30,
                  "triples": 3, "homeRuns": 25, "baseOnBalls": 60,
                  "hitByPitch": 5}}]}]}
    rows = sim.parse_batting_stats(payload)
    assert rows[0]["pa"] == 600 and rows[0]["hr"] == 25
    r = sim.rates_from_batting(rows[0])
    assert abs(sum(r.values()) - 1.0) < 1e-9
    box = {"teams": {"home": {"battingOrder": [11, 12, 13]},
                     "away": {"battingOrder": []}}}
    lu = sim.parse_lineups(box)
    assert lu["home"] == [11, 12, 13] and lu["away"] == []


# ---------------- Phase 3 回測骨架 ----------------
def test_score_market_and_logloss():
    # 完美預測 → logloss≈0；亂猜 0.5 → logloss≈ln2
    perfect = [(0.999999, 1.0), (0.000001, 0.0)]
    coin = [(0.5, 1.0), (0.5, 0.0)]
    sp = sim.score_market(perfect)
    sc = sim.score_market(coin)
    assert sp["logloss"] < 1e-4 and sp["n"] == 2
    assert abs(sc["logloss"] - np.log(2)) < 1e-6
    assert abs(sc["brier"] - 0.25) < 1e-9
    assert sim.score_market([])["n"] == 0


def test_build_game_record_coverage():
    lg = dict(sim.LEAGUE_RATES)
    bat = {1: dict(lg), 2: dict(lg)}
    result = {"home": "A", "away": "B", "home_goals": 5, "away_goals": 3,
              "home_pitcher_id": 100, "away_pitcher_id": 200}
    # 有打序
    g = sim.build_game_record(result, {"home": [1, 2, 1, 2, 1, 2, 1, 2, 1],
                                       "away": [2, 1, 2, 1, 2, 1, 2, 1, 2]},
                              bat, {}, lg)
    assert len(g["home_lineup"]) == 9 and g["home_score"] == 5
    assert g["away_pitcher"] == lg           # 查無投手 → 聯盟率
    # 無打序 → 空 lineup → sim 略過
    g2 = sim.build_game_record(result, {"home": [], "away": []}, bat, {}, lg)
    assert g2["home_lineup"] == []
    assert sim.sim_predictor()(({**g2, "_total_line": 8.5})) is None


def test_compare_backtest_end_to_end():
    from test_mlb import _mlb_model
    model = _mlb_model()
    lg = dict(sim.LEAGUE_RATES)
    strong = [{"bb": 0.11, "1b": 0.16, "2b": 0.06, "3b": 0.005,
               "hr": 0.055, "out": 0.610} for _ in range(9)]
    lgteam = [dict(lg) for _ in range(9)]
    games = []
    for hs, as_ in [(6, 2), (5, 4), (3, 1), (2, 7), (8, 3)]:
        games.append(sim.build_game_record(
            {"home": "Team0", "away": "Team5", "home_goals": hs, "away_goals": as_,
             "home_pitcher_id": None, "away_pitcher_id": None},
            {"home": [1] * 9, "away": [2] * 9},
            {1: strong[0], 2: lgteam[0]}, {}, lg))
    res = sim.compare_backtest(games, {"sim": sim.sim_predictor(n_sims=800, seed=1),
                                       "nb": sim.nb_predictor(model)})
    assert res["sim"]["ml"]["n"] == 5 and res["nb"]["ml"]["n"] == 5
    assert res["sim"]["ou"]["logloss"] is not None
    assert isinstance(sim.format_backtest(res), str)
    assert "錢線" in sim.format_backtest(res)
