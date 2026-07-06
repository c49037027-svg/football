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


# ---------------- 打線係數（把打者資訊接進 NB） ----------------
def test_offensive_rate_and_lineup_book():
    lg = dict(sim.LEAGUE_RATES)
    # 強打（多 HR/安打）每打席產能 > 弱打
    strong = {"bb": 0.11, "1b": 0.16, "2b": 0.06, "3b": 0.005, "hr": 0.055, "out": 0.610}
    weak = {"bb": 0.06, "1b": 0.11, "2b": 0.03, "3b": 0.002, "hr": 0.015, "out": 0.783}
    assert sim.offensive_rate(strong) > sim.offensive_rate(weak)
    # LineupBook：兩隊各 3 名（A 隊強、B 隊弱）
    rows = ([{"id": i, "team": "A", "pa": 500, "h": 150, "2b": 35, "3b": 3,
              "hr": 30, "bb": 55, "hbp": 5} for i in range(1, 4)]
            + [{"id": i, "team": "B", "pa": 500, "h": 120, "2b": 20, "3b": 1,
                "hr": 10, "bb": 40, "hbp": 3} for i in range(4, 7)])
    book = sim.LineupBook(rows, lg)
    # 該隊全先發上場 → 係數≈1（等於隊平均）
    f_team = book.factor([1, 2, 3], team="A")
    assert 0.97 < f_team < 1.03
    # 只派隊內最強一人、基準用聯盟均值 → >1；夾在 [clip] 混權重後仍 >1
    f_vs_league = book.factor([1], team=None)
    assert f_vs_league > 1.0
    # 查無球員 → 中性 1.0
    assert book.factor([999], team="A") == 1.0
    assert book.factor([], team="A") == 1.0


def test_nb_lineup_predictor_shifts_probs():
    from test_mlb import _mlb_model
    model = _mlb_model()
    lg = dict(sim.LEAGUE_RATES)
    strong = {"pa": 500, "h": 150, "2b": 35, "3b": 3, "hr": 30, "bb": 55, "hbp": 5}
    avg = {"pa": 500, "h": 130, "2b": 25, "3b": 2, "hr": 18, "bb": 45, "hbp": 4}
    weak = {"pa": 500, "h": 115, "2b": 18, "3b": 1, "hr": 9, "bb": 38, "hbp": 3}
    # 每隊各有強弱差；派出的一人才能偏離隊平均
    rows = ([{"id": 1, "team": "Team0", **strong},
             {"id": 2, "team": "Team0", **avg}, {"id": 3, "team": "Team0", **avg}]
            + [{"id": 4, "team": "Team5", **weak},
               {"id": 5, "team": "Team5", **avg}, {"id": 6, "team": "Team5", **avg}])
    book = sim.LineupBook(rows, lg)
    assert book.factor([1], team="Team0") > 1.0    # 派隊內最強 → >隊平均
    assert book.factor([4], team="Team5") < 1.0    # 派隊內最弱 → <隊平均
    g = sim.build_game_record(
        {"home": "Team0", "away": "Team5", "home_goals": 5, "away_goals": 3,
         "home_pitcher_id": None, "away_pitcher_id": None},
        {"home": [1], "away": [4]},   # 主派最強、客派最弱
        {}, {}, lg)
    g["_total_line"] = 8.5
    base = sim.nb_predictor(model)(g)
    boosted = sim.nb_lineup_predictor(model, book)(g)
    # 主隊打線變強、客隊變弱 → 主勝機率上升
    assert boosted["p_home"] > base["p_home"]


# ---------------- 天氣（大小盤訊號） ----------------
def test_parse_weather_and_factor():
    box = {"info": [
        {"label": "Weather", "value": "88 degrees, Sunny."},
        {"label": "Wind", "value": "15 mph, Out To CF."},
        {"label": "First pitch", "value": "7:05 PM."}]}
    w = sim.parse_weather(box)
    assert w["temp"] == 88.0 and w["wind_speed"] == 15.0 and w["wind_sign"] == 1
    # 熱 + 吹出去 → 總分環境放大
    assert sim.weather_total_factor(w) > 1.0
    # 冷 + 吹進來 → 壓低
    cold_in = sim.parse_weather({"info": [
        {"label": "Weather", "value": "48 degrees, Cloudy."},
        {"label": "Wind", "value": "18 mph, In From LF."}]})
    assert cold_in["wind_sign"] == -1
    assert sim.weather_total_factor(cold_in) < 1.0
    # 橫風 / 無資料 → 中性
    cross = sim.parse_weather({"info": [{"label": "Wind", "value": "10 mph, L To R."}]})
    assert cross["wind_sign"] == 0
    assert sim.weather_total_factor(None) == 1.0
    assert sim.weather_total_factor({"temp": 70.0, "wind_speed": 0.0, "wind_sign": 0}) == 1.0


def test_nb_weather_predictor_shifts_totals():
    from test_mlb import _mlb_model
    model = _mlb_model()
    lg = dict(sim.LEAGUE_RATES)
    hot = {"temp": 95.0, "wind_speed": 18.0, "wind_sign": 1}     # 助攻天氣
    g = sim.build_game_record(
        {"home": "Team0", "away": "Team1", "home_goals": 5, "away_goals": 4,
         "home_pitcher_id": None, "away_pitcher_id": None},
        {"home": [], "away": []}, {}, {}, lg, weather=hot)
    g["_total_line"] = 8.5
    p_base = sim.nb_predictor(model)(g)
    p_wx = sim.nb_weather_predictor(model)(g)
    assert p_wx["p_over"] > p_base["p_over"]     # 助攻天氣 → 大分機率上升
