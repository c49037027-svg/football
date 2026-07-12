"""NBA 模組測試：解析（fixture）、評分模型方向、市場機率、建站頁面。"""
import math

import pandas as pd
import pytest

from footy import nba


# ---------------- 解析 ----------------
def _sched_payload():
    def game(gid, status, home_city, home_name, hs, away_city, away_name, as_, iso):
        return {"gameId": gid, "gameStatus": status, "gameStatusText": "Final" if status == 3 else "7:00 pm ET",
                "gameDateTimeUTC": iso,
                "homeTeam": {"teamCity": home_city, "teamName": home_name, "score": hs},
                "awayTeam": {"teamCity": away_city, "teamName": away_name, "score": as_}}
    return {"leagueSchedule": {"gameDates": [
        {"gameDate": "10/21/2025 00:00:00", "games": [
            game("0022500001", 3, "Boston", "Celtics", 120, "New York", "Knicks", 112,
                 "2025-10-22T00:00:00Z"),
            game("0012500001", 3, "Miami", "Heat", 100, "Orlando", "Magic", 99,
                 "2025-10-22T00:00:00Z"),          # 熱身賽（001）應排除
        ]},
        {"gameDate": "10/22/2025 00:00:00", "games": [
            game("0022500002", 1, "Los Angeles", "Lakers", 0, "Golden State", "Warriors", 0,
                 "2025-10-23T02:00:00Z"),           # 未開打
        ]},
    ]}}


def test_parse_schedule_v2_finals_and_types():
    rows = nba.parse_schedule_v2(_sched_payload(), finals_only=True)
    assert len(rows) == 1
    g = rows[0]
    assert g["home"] == "Boston Celtics" and g["away"] == "New York Knicks"
    assert g["home_goals"] == 120 and g["away_goals"] == 112
    assert g["game_pk"] == 22500001
    assert g["date"] == "2025-10-21"       # UTC 00:00 → 美東前一天
    allg = nba.parse_schedule_v2(_sched_payload(), finals_only=False)
    assert len(allg) == 2                   # 含未開打，仍排除熱身賽
    assert allg[1]["home_goals"] is None


def test_parse_leaguegamelog_pairs_home_away():
    payload = {"resultSets": [{
        "headers": ["GAME_ID", "GAME_DATE", "TEAM_NAME", "MATCHUP", "PTS"],
        "rowSet": [
            ["0022400001", "2024-10-22", "Boston Celtics", "BOS vs. NYK", 132],
            ["0022400001", "2024-10-22", "New York Knicks", "NYK @ BOS", 109],
            ["0022400002", "2024-10-22", "Los Angeles Clippers", "LAC vs. PHX", 113],
            ["0022400002", "2024-10-22", "Phoenix Suns", "PHX @ LAC", 116],
        ]}]}
    rows = nba.parse_leaguegamelog(payload)
    assert len(rows) == 2
    g = rows[0]
    assert (g["home"], g["away"], g["home_goals"], g["away_goals"]) == \
        ("Boston Celtics", "New York Knicks", 132, 109)
    assert rows[1]["home"] == "LA Clippers"      # 異名統一


def test_zh_nba():
    assert nba.zh_nba("Boston Celtics") == "塞爾提克"
    assert nba.zh_nba("LA Clippers") == "快艇"
    assert len({v for k, v in nba.NBA_ZH.items() if k != "Los Angeles Clippers"}) == 30


# ---------------- 評分模型 ----------------
def _synth_df(n_rounds=40, seed=0):
    """三強一弱四隊合成資料：Strong 得分多失分少。"""
    import numpy as np
    rng = np.random.default_rng(seed)
    teams = {"Strong": (118, 106), "Mid A": (112, 112), "Mid B": (111, 113),
             "Weak": (105, 119)}   # (攻, 防=平均失分)
    rows = []
    names = list(teams)
    d = pd.Timestamp("2025-01-01")
    for r in range(n_rounds):
        for i in range(len(names)):
            for j in range(len(names)):
                if i == j:
                    continue
                h, a = names[i], names[j]
                mu_h = (teams[h][0] + teams[a][1]) / 2 + 1.5
                mu_a = (teams[a][0] + teams[h][1]) / 2 - 1.5
                rows.append({"date": (d + pd.Timedelta(days=r)).date().isoformat(),
                             "home": h, "away": a,
                             "home_goals": int(rng.normal(mu_h, 10)),
                             "away_goals": int(rng.normal(mu_a, 10))})
    return pd.DataFrame(rows)


def test_fit_ratings_directions():
    df = _synth_df()
    model = nba.fit_ratings(df, half_life_days=1e9, reg=5.0)
    assert model.home_adv > 0                       # 主場優勢方向
    assert model.off["Strong"] > model.off["Weak"]  # 強隊攻高
    assert model.deff["Strong"] > model.deff["Weak"]  # 強隊防好（正=壓低對手）
    assert 8.0 < model.sigma_margin < 25.0
    mu_h, mu_a = model.expected_points("Strong", "Weak")
    assert mu_h > mu_a                              # 強隊主場預期贏
    assert 90 < mu_a < mu_h < 135


def test_analyze_game_probabilities():
    df = _synth_df()
    model = nba.fit_ratings(df, half_life_days=1e9, reg=5.0)
    m = nba.analyze_game(model, "Strong", "Weak")
    assert m.p_home > 0.60                          # 強對弱主勝機率高
    assert abs(m.p_home + m.p_away - 1) < 1e-9
    assert abs(m.p_over + m.p_under - 1) < 1e-9
    # 模型自取線 → 兩側都接近五五波（線在期望附近）
    assert 0.4 < m.p_over < 0.6 and 0.35 < m.p_cover_home < 0.65
    assert m.total_line % 0.5 == 0 and m.total_line % 1 != 0   # .5 線
    # 給定市場線：大分線提高 → p_over 下降
    m_hi = nba.analyze_game(model, "Strong", "Weak", total_line=m.total_line + 10)
    assert m_hi.p_over < m.p_over
    # 主隊讓更多分 → 過盤機率下降
    m_deep = nba.analyze_game(model, "Strong", "Weak", run_line=m.run_line - 6)
    assert m_deep.p_cover_home < m.p_cover_home
    # 錢線與常態一致
    mu_m = m.exp_home - m.exp_away
    assert m.p_home == pytest.approx(
        0.5 * (1 + math.erf(mu_m / model.sigma_margin / math.sqrt(2))))


def test_team_power_order_and_roundtrip(tmp_path):
    df = _synth_df()
    model = nba.fit_ratings(df, half_life_days=1e9, reg=5.0)
    power = nba.team_power(model)
    assert power[0]["team"] == "Strong" and power[-1]["team"] == "Weak"
    p = tmp_path / "nba.pkl"
    model.save(p)
    m2 = nba.NBAModel.load(p)
    assert m2.expected_points("Strong", "Weak") == model.expected_points("Strong", "Weak")


def test_load_with_history_and_csv(tmp_path):
    cur = tmp_path / "nba.csv"
    hist = tmp_path / "nba_hist.csv"
    nba.write_games_csv([{"date": "2025-10-21", "home": "Boston Celtics",
                          "away": "New York Knicks", "home_goals": 120,
                          "away_goals": 112, "game_pk": 22500001}], cur)
    hist.write_text("date,home,away,home_goals,away_goals,game_pk\n"
                    "2024-10-22,Los Angeles Clippers,Phoenix Suns,113,116,22400002\n"
                    "2025-10-21,Boston Celtics,New York Knicks,120,112,22500001\n")
    df = nba.load_with_history(cur, hist)
    assert len(df) == 2                              # 去重
    assert "LA Clippers" in set(df["home"])          # 異名統一


# ---------------- 訊號 + 建站 ----------------
def test_bet_signals_reuse_and_site_page(tmp_path, monkeypatch):
    df = _synth_df()
    model = nba.fit_ratings(df, half_life_days=1e9, reg=5.0)
    from footy import mlb
    m = nba.analyze_game(model, "Strong", "Weak")
    sig = mlb.bet_signals(m, None)
    assert set(sig) == {"1X2", "OU", "AH"}
    assert sig["1X2"]["side"] == "home" and sig["1X2"]["odds"] is None
    picks = mlb.picks_for_game(m, None)
    assert any(p["market"] == "1X2" for p in picks)
    # 建站：塞一場「今天」的比賽（用假 schedule payload，不打網路）
    model_path = tmp_path / "nba.pkl"
    model.save(model_path)
    date = mlb.us_today()
    payload = {"leagueSchedule": {"gameDates": [{"games": [{
        "gameId": "0022599999", "gameStatus": 1, "gameStatusText": "7:00 pm ET",
        "gameDateTimeUTC": f"{date}T23:30:00Z",
        "homeTeam": {"teamCity": "", "teamName": "Strong", "score": 0},
        "awayTeam": {"teamCity": "", "teamName": "Weak", "score": 0},
    }]}]}}
    html_out = nba.build_site_page(model_path=str(model_path),
                                   ledger_path=str(tmp_path / "nba_bets.csv"),
                                   with_odds=False, schedule_payload=payload,
                                   date=date)
    assert "NBA 今日預測" in html_out and "🏀" in html_out
    assert "Strong" in html_out                      # 卡片有出現
    assert "nba_perf.html" in html_out               # 績效連結
    # 帳本有記到模型推薦
    led = tmp_path / "nba_bets.csv"
    assert led.exists() and "22599999" in led.read_text()


def test_site_page_offseason_note(tmp_path):
    df = _synth_df()
    model = nba.fit_ratings(df, half_life_days=1e9, reg=5.0)
    model_path = tmp_path / "nba.pkl"
    model.save(model_path)
    html_out = nba.build_site_page(model_path=str(model_path),
                                   ledger_path=str(tmp_path / "l.csv"),
                                   with_odds=False,
                                   schedule_payload={"leagueSchedule": {"gameDates": []}},
                                   date="2026-08-15")
    assert "休賽季" in html_out


def test_parse_espn_scoreboard_and_months():
    payload = {"events": [
        {"id": "401585601", "date": "2024-10-23T23:30Z",
         "season": {"type": 2},
         "status": {"type": {"completed": True}},
         "competitions": [{"competitors": [
             {"homeAway": "home", "score": "132",
              "team": {"displayName": "Boston Celtics"}},
             {"homeAway": "away", "score": "109",
              "team": {"displayName": "New York Knicks"}}]}]},
        {"id": "401585602", "date": "2024-10-15T23:00Z",
         "season": {"type": 1},          # 熱身賽 → 排除
         "status": {"type": {"completed": True}},
         "competitions": [{"competitors": [
             {"homeAway": "home", "score": "100",
              "team": {"displayName": "Miami Heat"}},
             {"homeAway": "away", "score": "99",
              "team": {"displayName": "Orlando Magic"}}]}]},
        {"id": "401585603", "date": "2024-10-24T23:00Z",
         "season": {"type": 2},
         "status": {"type": {"completed": False}},   # 未完賽 → 排除
         "competitions": [{"competitors": [
             {"homeAway": "home", "score": "0",
              "team": {"displayName": "LA Clippers"}},
             {"homeAway": "away", "score": "0",
              "team": {"displayName": "Phoenix Suns"}}]}]},
    ]}
    rows = nba.parse_espn_scoreboard(payload)
    assert len(rows) == 1
    g = rows[0]
    assert (g["home"], g["away"], g["home_goals"], g["away_goals"]) == \
        ("Boston Celtics", "New York Knicks", 132, 109)
    assert g["date"] == "2024-10-23"
    months = nba.season_months("2024-25")
    assert months[0] == ("20241001", "20241031")
    assert months[-1] == ("20250601", "20250630")
    assert len(months) == 9


def test_parse_espn_uncompleted_and_window_fields():
    payload = {"events": [
        {"id": "401600001", "date": "2026-10-21T23:30Z",
         "season": {"type": 2},
         "status": {"type": {"completed": False, "shortDetail": "7:30 PM ET"}},
         "competitions": [{"competitors": [
             {"homeAway": "home", "score": "0",
              "team": {"displayName": "Boston Celtics"}},
             {"homeAway": "away", "score": "0",
              "team": {"displayName": "New York Knicks"}}]}]},
    ]}
    assert nba.parse_espn_scoreboard(payload) == []          # 預設只留已完賽
    rows = nba.parse_espn_scoreboard(payload, completed_only=False)
    assert len(rows) == 1
    g = rows[0]
    assert g["home_goals"] is None and g["status"] == "7:30 PM ET"
    assert g["game_date_iso"] == "2026-10-21T23:30Z"
    assert g["date"] == "2026-10-21"
    s = nba.current_season()
    assert len(s) == 7 and s[4] == "-"
