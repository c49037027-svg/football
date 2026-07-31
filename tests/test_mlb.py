"""MLB 模組測試：schedule 解析、錢線/大小/讓分數學、端到端。"""
import numpy as np
import pandas as pd
import pytest

from footy import mlb
from footy.models import dixon_coles as dc

FIXTURE = {
    "dates": [{
        "date": "2026-07-01",
        "games": [
            {   # 已完賽例行賽 → 收
                "gameType": "R", "officialDate": "2026-07-01",
                "status": {"abstractGameState": "Final"},
                "teams": {
                    "home": {"team": {"name": "New York Yankees"}, "score": 5,
                             "probablePitcher": {"fullName": "G. Cole"}},
                    "away": {"team": {"name": "Boston Red Sox"}, "score": 3},
                },
            },
            {   # 未開打 → finals_only 時略過
                "gameType": "R", "officialDate": "2026-07-01",
                "status": {"abstractGameState": "Preview"},
                "teams": {
                    "home": {"team": {"name": "Los Angeles Dodgers"}},
                    "away": {"team": {"name": "San Diego Padres"}},
                },
            },
            {   # 熱身賽 → 永遠略過
                "gameType": "S", "officialDate": "2026-07-01",
                "status": {"abstractGameState": "Final"},
                "teams": {
                    "home": {"team": {"name": "Springville"}, "score": 1},
                    "away": {"team": {"name": "Testtown"}, "score": 0},
                },
            },
        ],
    }],
}


def test_parse_schedule_finals_only():
    rows = mlb.parse_schedule(FIXTURE, finals_only=True)
    assert len(rows) == 1
    r = rows[0]
    assert r["home"] == "New York Yankees" and r["home_goals"] == 5
    assert r["away"] == "Boston Red Sox" and r["away_goals"] == 3
    assert r["home_pitcher"] == "G. Cole"


def test_parse_schedule_all_games():
    rows = mlb.parse_schedule(FIXTURE, finals_only=False)
    assert len(rows) == 2  # 熱身賽仍排除
    assert {r["status"] for r in rows} == {"Final", "Preview"}


def test_moneyline_symmetric_matrix():
    # 對稱矩陣 → 五五波；對角線（延長賽）各半分
    n = 6
    mat = np.ones((n, n)) / (n * n)
    p_h, p_a = mlb.moneyline(mat)
    assert abs(p_h - 0.5) < 1e-9 and abs(p_h + p_a - 1.0) < 1e-9


def test_moneyline_favours_stronger():
    # 質量集中在主隊多分 → 主勝率高
    mat = np.zeros((6, 6))
    mat[4, 1] = 0.7
    mat[2, 2] = 0.3   # 平手部分各半分
    p_h, p_a = mlb.moneyline(mat)
    assert abs(p_h - 0.85) < 1e-9


def _mlb_model():
    """合成 MLB 資料：得分 ~Poisson(4.5)，A 隊明顯較強。"""
    rng = np.random.default_rng(7)
    teams = [f"Team{i}" for i in range(6)]
    strength = {t: 0.25 if t == "Team0" else (-0.25 if t == "Team5" else 0.0)
                for t in teams}
    rows = []
    day = pd.Timestamp("2025-04-01")
    for rnd in range(60):
        for i in range(0, 6, 2):
            h, a = teams[(i + rnd) % 6], teams[(i + rnd + 1) % 6]
            lam = 4.5 * np.exp(strength[h] - strength[a] * 0.5 + 0.05)
            mu = 4.2 * np.exp(strength[a] - strength[h] * 0.5)
            rows.append(dict(date=day + pd.Timedelta(days=rnd), home=h, away=a,
                             home_goals=int(rng.poisson(lam)),
                             away_goals=int(rng.poisson(mu))))
    df = pd.DataFrame(rows)
    return dc.fit(df, half_life_days=10_000, max_goals=20, rho_init=0.0, reg=0.2)


def test_analyze_game_end_to_end():
    model = _mlb_model()
    m = mlb.analyze_game(model, "Team0", "Team5", total_line=8.5, run_line=-1.5)
    # 機率健全性
    assert abs(m.p_home + m.p_away - 1.0) < 1e-9
    assert abs(m.p_over + m.p_under - 1.0) < 1e-6   # 8.5 無走盤
    assert 0.0 <= m.p_cover_home <= 1.0
    # 強隊在家對弱隊 → 錢線應明顯 > 50%
    assert m.p_home > 0.55
    # 公平賠率互為機率倒數
    assert abs(m.ml_home_odds - round(1 / m.p_home, 2)) < 0.02
    # 期望得分在棒球合理範圍
    assert 2.0 < m.exp_home < 9.0 and 2.0 < m.exp_away < 9.0
    assert len(m.top_scores) == 4


PITCHER_FIXTURE = {
    "stats": [{
        "splits": [
            # 王牌：180 局失 50 分 → RA/9 = 2.5
            {"player": {"id": 1, "fullName": "Ace Man"},
             "team": {"name": "New York Yankees"},
             "stat": {"inningsPitched": "180.0", "runs": 50, "gamesStarted": 30}},
            # 普通先發：150 局失 75 分 → RA/9 = 4.5
            {"player": {"id": 2, "fullName": "Avg Joe"},
             "team": {"name": "New York Yankees"},
             "stat": {"inningsPitched": "150.0", "runs": 75, "gamesStarted": 28}},
            # 爛先發：120 局失 90 分 → RA/9 = 6.75
            {"player": {"id": 3, "fullName": "Bad Luck"},
             "team": {"name": "Boston Red Sox"},
             "stat": {"inningsPitched": "120.0", "runs": 90, "gamesStarted": 25}},
            # 小樣本新秀：6 局失 0 分（收縮後不該變成怪物）
            {"player": {"id": 4, "fullName": "Tiny Sample"},
             "team": {"name": "Boston Red Sox"},
             "stat": {"inningsPitched": "6.0", "runs": 0, "gamesStarted": 1}},
            # 牛棚（gs=0，不進隊先發平均）
            {"player": {"id": 5, "fullName": "Pen Guy"},
             "team": {"name": "Boston Red Sox"},
             "stat": {"inningsPitched": "60.1", "runs": 30, "gamesStarted": 0}},
        ],
    }],
}


def test_ip_to_float():
    assert abs(mlb.ip_to_float("123.2") - (123 + 2 / 3)) < 1e-9
    assert abs(mlb.ip_to_float("45.1") - (45 + 1 / 3)) < 1e-9
    assert mlb.ip_to_float("88") == 88.0
    assert mlb.ip_to_float("") == 0.0 and mlb.ip_to_float(None) == 0.0


def test_parse_pitcher_stats():
    rows = mlb.parse_pitcher_stats(PITCHER_FIXTURE)
    assert len(rows) == 5
    ace = next(r for r in rows if r["id"] == 1)
    assert ace["name"] == "Ace Man" and ace["ip"] == 180.0 and ace["runs"] == 50


def test_pitcher_factor_direction_and_clip():
    book = mlb.PitcherBook(mlb.parse_pitcher_stats(PITCHER_FIXTURE))
    f_ace, _ = book.factor(1)      # 王牌 → 壓低對手得分
    f_bad, _ = book.factor(3)      # 爛投 → 放大對手得分
    f_none, note = book.factor(999)
    assert f_ace < 1.0 < f_bad
    assert f_none == 1.0 and "無評分" in note
    # 係數混入 0.6 權重且原始值有夾限 → 落在 [0.82, 1.21] 內
    assert 0.7 * 0.6 + 0.4 <= f_ace and f_bad <= 1.35 * 0.6 + 0.4
    # 姓名查詢也通
    f_name, _ = book.factor("Ace Man")
    assert abs(f_name - f_ace) < 1e-12


def test_pitcher_shrinkage_small_sample():
    book = mlb.PitcherBook(mlb.parse_pitcher_stats(PITCHER_FIXTURE))
    # 6 局 0 失分的新秀：收縮後 RA/9 應接近聯盟平均，不會是 0
    rookie = book.by_id[4]
    shrunk = book._shrunk_ra9(rookie)
    assert shrunk > book.league_ra9 * 0.7


def test_pitcher_factor_shifts_moneyline():
    model = _mlb_model()
    base = mlb.analyze_game(model, "Team0", "Team1")
    # 客隊派王牌（客先發係數 <1 → 主隊得分被壓）→ 主勝率下降、總分小盤機率升
    strong_away = mlb.analyze_game(model, "Team0", "Team1", away_pitcher_factor=0.85)
    assert strong_away.p_home < base.p_home
    assert strong_away.p_under > base.p_under
    # 主隊派王牌 → 主勝率上升
    strong_home = mlb.analyze_game(model, "Team0", "Team1", home_pitcher_factor=0.85)
    assert strong_home.p_home > base.p_home


def test_pitcher_book_csv_roundtrip(tmp_path):
    import csv as _csv
    rows = mlb.parse_pitcher_stats(PITCHER_FIXTURE)
    p = tmp_path / "pitchers.csv"
    with p.open("w", newline="") as f:
        w = _csv.DictWriter(f, fieldnames=["id", "name", "team", "ip", "runs",
                                           "gs", "so", "bb", "hr"])
        w.writeheader()
        w.writerows(rows)
    book = mlb.PitcherBook.load_csv(p)
    f1, _ = book.factor(1)
    f2, _ = mlb.PitcherBook(rows).factor(1)
    assert abs(f1 - f2) < 1e-12


def test_zh_mlb():
    assert mlb.zh_mlb("New York Yankees") == "洋基"
    assert mlb.zh_mlb("Unknown Club") == "Unknown Club"


def test_nb_matrix_vs_poisson():
    # k=None → Poisson；k 小 → 尾端更厚（總分>12 機率更高）、平均不變
    lam, mu = 4.6, 4.2
    pois = mlb.nb_score_matrix(lam, mu, None, 20)
    nb = mlb.nb_score_matrix(lam, mu, 3.5, 20)
    ks = np.arange(21)
    assert abs((pois * ks[:, None]).sum() - lam) < 0.02       # 平均保留
    assert abs((nb * ks[:, None]).sum() - lam) < 0.05
    def p_total_over(mat, line):
        t = np.add.outer(ks, ks)
        return float(mat[t > line].sum())
    assert p_total_over(nb, 12.5) > p_total_over(pois, 12.5)  # 尾端更厚
    # 錢線：分布變寬 → 強隊機率往 5 成收
    p_pois, _ = mlb.moneyline(mlb.nb_score_matrix(6.0, 3.0, None, 20))
    p_nb, _ = mlb.moneyline(mlb.nb_score_matrix(6.0, 3.0, 3.5, 20))
    assert 0.5 < p_nb < p_pois


def test_dispersion_from_df():
    rng = np.random.default_rng(1)
    m, k = 4.5, 3.5
    # 由 NB(m, k) 取樣 → 動差法估回的 k 應接近
    p = k / (k + m)
    from scipy import stats
    runs = stats.nbinom.rvs(k, p, size=20000, random_state=rng)
    df = pd.DataFrame({"home_goals": runs[:10000], "away_goals": runs[10000:],
                       "home": "A", "away": "B"})
    est = mlb.dispersion_from_df(df)
    assert est is not None and 2.8 < est < 4.4
    # 純 Poisson 資料 → 無過度離散 → None 或很大
    pois = stats.poisson.rvs(4.5, size=20000, random_state=rng)
    df2 = pd.DataFrame({"home_goals": pois[:10000], "away_goals": pois[10000:],
                        "home": "A", "away": "B"})
    est2 = mlb.dispersion_from_df(df2)
    assert est2 is None or est2 > 15


def test_analyze_game_dispersion_shrinks_favourite():
    model = _mlb_model()
    pois = mlb.analyze_game(model, "Team0", "Team5")
    nb = mlb.analyze_game(model, "Team0", "Team5", dispersion=3.3)
    assert 0.5 < nb.p_home < pois.p_home


def test_fip_blend():
    base = dict(team="X", ip=150.0, runs=75, gs=28)   # RA/9 = 4.5
    rows = [
        {"id": 1, "name": "K Machine", **base, "so": 200, "bb": 25, "hr": 10},
        {"id": 2, "name": "Contact Luck", **base, "so": 80, "bb": 70, "hr": 28},
        {"id": 3, "name": "No FIP Data", **base, "so": 0, "bb": 0, "hr": 0},
    ]
    book = mlb.PitcherBook(rows)
    # 相同 RA/9：FIP 好的評分應優於 FIP 差的
    r1, r2 = book._rating(rows[0]), book._rating(rows[1])
    assert r1 < r2
    # 缺 K/BB/HR → 退回純 RA/9
    assert abs(book._rating(rows[2]) - book._shrunk_ra9(rows[2])) < 1e-9


def test_evaluate_smoke():
    """回測函式在小合成資料上能跑且 NB 不劣於 Poisson 太多。"""
    rng = np.random.default_rng(3)
    teams = [f"T{i}" for i in range(4)]
    rows = []
    day = pd.Timestamp("2025-04-01")
    from scipy import stats as st
    for rnd in range(120):
        for i in range(0, 4, 2):
            h, a = teams[(i + rnd) % 4], teams[(i + rnd + 1) % 4]
            k, m1, m2 = 3.5, 4.6, 4.2
            rows.append(dict(date=(day + pd.Timedelta(days=rnd)).date().isoformat(),
                             home=h, away=a,
                             home_goals=int(st.nbinom.rvs(k, k / (k + m1), random_state=rng)),
                             away_goals=int(st.nbinom.rvs(k, k / (k + m2), random_state=rng))))
    df = pd.DataFrame(rows)
    res = mlb.evaluate(df, cut="2025-07-01", ks=[None, 3.5])
    assert res["n_test"] > 30 and len(res["rows"]) == 2
    pois, nb = res["rows"][0], res["rows"][1]
    assert nb["ou_logloss"] <= pois["ou_logloss"] + 0.01   # NB 資料上 NB 應不劣


def test_park_factors():
    rows = []
    # Team0 主場高得分（場均 12），Team1 正常（9），各 40 場
    for i in range(40):
        rows.append(dict(home="Team0", away="TeamX", home_goals=7, away_goals=5))
        rows.append(dict(home="Team1", away="TeamX", home_goals=5, away_goals=4))
    df = pd.DataFrame(rows)
    pf = mlb.park_factors(df)
    assert pf["Team0"] > 1.0 > pf["Team1"] * 1.001 or pf["Team0"] > pf["Team1"]
    assert 0.90 <= pf["Team0"] <= 1.12  # 夾限
    # park factor 放大總分 → 大盤機率升
    model = _mlb_model()
    base = mlb.analyze_game(model, "Team0", "Team1")
    juiced = mlb.analyze_game(model, "Team0", "Team1", park_factor=1.10)
    assert juiced.p_over > base.p_over


def test_team_power():
    model = _mlb_model()
    power = mlb.team_power(model)
    assert len(power) == 6
    assert power[0]["diff"] >= power[-1]["diff"]      # 依淨值排序
    assert power[0]["team"] == "Team0"                 # 合成資料裡最強
    assert all(1.0 < p["rf"] < 10.0 for p in power)


def test_picks_and_ledger(tmp_path):
    model = _mlb_model()
    m = mlb.analyze_game(model, "Team0", "Team5")
    picks = mlb.picks_for_game(m)
    mks = {p["market"] for p in picks}
    assert "1X2" in mks and "AH" in mks               # 錢線+讓分必有
    ml = next(p for p in picks if p["market"] == "1X2")
    assert ml["selection"] == "home"                  # 強隊在家 → 推主
    led = tmp_path / "mlb_bets.csv"
    game = {"game_pk": 777001, "home": "Team0", "away": "Team5"}
    n = mlb.log_picks(led, "2026-07-03", game, picks)
    assert n == len(picks)
    assert mlb.log_picks(led, "2026-07-03", game, picks) == 0   # 不重複
    # 結算：主隊 6-2 贏 → 錢線過
    from footy import tracker
    settled = tracker.settle(led, {777001: (6, 2)})
    assert settled == n
    txt = mlb.summary_text(led)
    assert txt and "MLB 推薦戰績" in txt and "錢線" in txt


def test_render_mlb_page():
    from footy import report
    model = _mlb_model()
    m = mlb.analyze_game(model, "Team0", "Team5")
    # 錢線給一個「買」訊號（賠率 1.5，模型看好主 → +EV），供 @1.5/買/TOP5 檢查
    sig = {"1X2": {"side": "home", "odds": 1.5, "edge": 0.12,
                   "verdict": "買", "p": m.p_home},
           "OU": {"side": "over" if m.p_over >= m.p_under else "under",
                  "odds": None, "edge": None, "verdict": None, "p": m.p_over},
           "AH": {"side": "home", "odds": None, "edge": None,
                  "verdict": None, "p": m.p_cover_home}}
    rows = [{"game": {"home": "Team0", "away": "Team5", "home_pitcher": "P One",
                      "away_pitcher": "P Two",
                      "game_date_iso": "2026-07-03T23:05:00Z"},
             "m": m, "pf": 1.08, "hp_note": "RA/9 3.00", "ap_note": "",
             "signals": sig, "best_edge": 0.12, "time": mlb.taipei_time(
                 "2026-07-03T23:05:00Z"),
             "status": "Preview"}]
    power = mlb.team_power(model)
    page = report.render_mlb_page(rows, date="2026-07-03", power=power,
                                  track_text="MLB 推薦戰績｜1 勝 0 敗")
    assert "MLB 今日預測" in page and "錢線" in page and "大小" in page
    assert "球場 1.08" in page and "戰力表" in page and "MLB 推薦戰績" in page
    assert "@1.5" in page                              # 有 odds → 顯示賠率
    assert "class='mgrid'" in page                     # 卡片格線排版
    assert "買" in page and "TOP 5" in page             # 買訊號 + 最推薦區塊
    assert "07/04" in page                             # 台北時間（+8 跨日）
    assert "mlb_perf.html" in page                     # 連到績效頁
    # 空狀態
    empty = report.render_mlb_page([], date="2026-07-03", note="尚未訓練")
    assert "今日無可預測比賽" in empty and "尚未訓練" in empty


def test_bet_signals_and_perf_page():
    from footy import report
    from footy.live.feed import MarketQuote
    model = _mlb_model()
    m = mlb.analyze_game(model, "Team0", "Team5")
    # 錢線給主隊很甜的賠率 → 融合後仍 +EV → 買
    quotes = [MarketQuote("1X2", "home", 2.2), MarketQuote("1X2", "away", 1.7)]
    sig = mlb.bet_signals(m, quotes)
    assert sig["1X2"]["side"] == "home"
    assert sig["1X2"]["odds"] == 2.2
    assert sig["1X2"]["verdict"] in ("買", "觀望")
    # 無盤口 → verdict=None
    assert mlb.bet_signals(m, None)["1X2"]["verdict"] is None
    be = mlb.best_edge(sig)
    assert be is None or isinstance(be, float)
    # 台北時間
    assert mlb.taipei_time("2026-07-03T23:05:00Z").startswith("週")
    assert mlb.taipei_time(None) == ""
    # 績效頁三態
    empty = report.render_mlb_perf_page([], pending=[])
    assert "尚無收益紀錄" in empty and "mlb.html" in empty
    pend = report.render_mlb_perf_page([], pending=[
        {"date": "2026-07-03", "home": "Team0", "away": "Team5", "market": "1X2",
         "selection": "home", "line": "", "odds": 2.2, "edge": 0.08}])
    assert "待結算" in pend and "錢線" in pend
    hist = [{"date": "2026-07-01", "match_num": 1, "home": "Team0", "away": "Team5",
             "market": "1X2", "selection": "home", "line": "", "odds": 1.9,
             "close_odds": 1.8, "clv": 0.056, "result": "win", "pl": 0.9,
             "cum_pl": 0.9, "n": 1, "cum_roi": 0.9, "cum_clv": 0.056,
             "beat_rate": 1.0}]
    full = report.render_mlb_perf_page(hist, track_text="MLB 推薦戰績｜1 勝 0 敗")
    assert "過" in full and "錢線" in full and "累積收益" in full


def test_cli_wiring():
    from click.testing import CliRunner
    from footy.cli import cli
    out = CliRunner().invoke(cli, ["mlb", "--help"])
    assert out.exit_code == 0
    for cmd in ("fetch", "train", "analyze", "today"):
        assert cmd in out.output


def test_pitcher_gamelog_and_formbook():
    payload = {"stats": [{"splits": [
        {"date": "2025-04-05", "stat": {"inningsPitched": "6.0", "runs": 1,
         "strikeOuts": 8, "baseOnBalls": 1, "homeRuns": 0}},
        {"date": "2025-04-17", "stat": {"inningsPitched": "7.0", "runs": 0,
         "strikeOuts": 10, "baseOnBalls": 0, "homeRuns": 0}}]}]}
    log = mlb.parse_pitcher_gamelog(payload)
    assert len(log) == 2 and log[0]["date"] == "2025-04-05"
    # 兩位「整季相同、順序相反」的投手：A 近況好、B 近況差
    a = [{"date": "2025-04-01", "ip": 5.0, "r": 6, "so": 3, "bb": 3, "hr": 2},   # 早爛
         {"date": "2025-04-20", "ip": 5.0, "r": 0, "so": 8, "bb": 0, "hr": 0}]   # 近好
    b = [{"date": "2025-04-01", "ip": 5.0, "r": 0, "so": 8, "bb": 0, "hr": 0},   # 早好
         {"date": "2025-04-20", "ip": 5.0, "r": 6, "so": 3, "bb": 3, "hr": 2}]   # 近爛
    book = mlb.PitcherFormBook({11: a, 22: b})
    as_of = "2025-04-25"
    # 季版（halflife 極大）：兩人整季相同 → 係數相同
    assert abs(book.factor(11, as_of, halflife=1e9)
               - book.factor(22, as_of, halflife=1e9)) < 1e-9
    # 近況版（halflife 小）：A 近況好 → 係數較低（更壓制對手）；B 反之
    assert book.factor(11, as_of, halflife=1.0) < book.factor(22, as_of, halflife=1.0)
    # point-in-time：只看 as_of 之前
    f_early = book.factor(11, as_of="2025-04-10", halflife=1.0)   # 只有早爛那場
    f_late = book.factor(11, as_of="2025-04-25", halflife=1.0)    # 含近好那場
    assert f_late < f_early
    # 查無投手 / 無 as_of 之前出賽 → 中性
    assert book.factor(999, as_of="2025-04-25") == 1.0
    assert book.factor(11, as_of="2025-01-01") == 1.0


def test_pitcher_formbook_team_baseline():
    # 同隊三人：ace 強、兩位平庸 → ace 相對隊平均應 <1，平庸 ≈1
    ace = [{"date": "2025-04-01", "ip": 7.0, "r": 1, "so": 9, "bb": 1, "hr": 0},
           {"date": "2025-04-08", "ip": 7.0, "r": 1, "so": 8, "bb": 0, "hr": 0}]
    mid1 = [{"date": "2025-04-02", "ip": 5.0, "r": 4, "so": 4, "bb": 3, "hr": 1},
            {"date": "2025-04-09", "ip": 5.0, "r": 5, "so": 3, "bb": 2, "hr": 2}]
    mid2 = [{"date": "2025-04-03", "ip": 5.0, "r": 5, "so": 3, "bb": 3, "hr": 2},
            {"date": "2025-04-10", "ip": 5.0, "r": 4, "so": 4, "bb": 2, "hr": 1}]
    logs = {1: ace, 2: mid1, 3: mid2}
    pid_team = {1: "Aces", 2: "Aces", 3: "Aces"}
    team = mlb.PitcherFormBook(logs, pid_team=pid_team)
    league = mlb.PitcherFormBook(logs)   # 無隊基準（聯盟）
    as_of = "2025-04-15"
    assert team.team_base and "Aces" in team.team_base
    # 隊基準下：ace 優於隊平均 → 係數 <1 且明顯低於平庸隊友
    f_ace = team.factor(1, as_of, halflife=1e9)
    f_mid = team.factor(2, as_of, halflife=1e9)
    assert f_ace < 1.0 < f_mid            # ace 壓制、平庸放大（相對隊平均）
    assert f_ace < f_mid
    # 隊基準 vs 聯盟基準應不同（基準不同）
    assert f_ace != league.factor(1, as_of, halflife=1e9)


def test_market_confidence_and_top5_weighting(tmp_path):
    import csv as _csv
    from footy import report
    led = tmp_path / "mlb_bets.csv"
    fields = ["date", "match_num", "home", "away", "market", "selection",
              "line", "odds", "edge", "source", "close_odds", "result", "pl"]
    rows = []
    # 讓分(AH) 8勝2敗；錢線(1X2) 3勝7敗 → AH 乘數應 >1、1X2 <1
    for i in range(10):
        rows.append(dict(date="2026-07-01", match_num=1000 + i, home="A", away="B",
                         market="AH", selection="主", line=-1.5, odds=1.9, edge=0.05,
                         source="market", close_odds=1.9,
                         result="win" if i < 8 else "loss", pl=0.9 if i < 8 else -1.0))
        rows.append(dict(date="2026-07-01", match_num=2000 + i, home="A", away="B",
                         market="1X2", selection="home", line="", odds=1.9, edge=0.05,
                         source="market", close_odds=1.9,
                         result="win" if i < 3 else "loss", pl=0.9 if i < 3 else -1.0))
    with led.open("w", newline="") as f:
        w = _csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(rows)
    mc = mlb.market_confidence(str(led))
    assert mc["mult"]["AH"] > 1.05 and mc["mult"]["1X2"] < 0.95   # 方向正確
    assert 0.7 <= mc["mult"]["1X2"] and mc["mult"]["AH"] <= 1.3   # 夾限
    assert abs(mc["winrate"]["AH"] - 0.8) < 1e-9

    # 排序：錢線 edge 0.06 vs 讓分 edge 0.05；勝率加權後讓分應排前
    model = _mlb_model()
    m = mlb.analyze_game(model, "Team0", "Team5")

    def mkrow(mk, edge):
        sig = {"1X2": {"verdict": None}, "OU": {"verdict": None}, "AH": {"verdict": None}}
        sd = {"1X2": "home", "OU": "over", "AH": "home"}[mk]
        sig[mk] = {"side": sd, "odds": 1.95, "edge": edge, "verdict": "買", "p": 0.55}
        return {"game": {"home": "Team0", "away": "Team5", "home_pitcher": "",
                         "away_pitcher": "", "game_date_iso": "2026-07-06T23:05:00Z"},
                "m": m, "pf": 1.0, "wx": None, "wf": 1.0, "hp_note": "", "ap_note": "",
                "signals": sig, "best_edge": edge, "time": "19:00", "status": ""}
    rows2 = [mkrow("1X2", 0.06), mkrow("AH", 0.05)]
    import re
    def _top5_sec(page):
        """只取 edge TOP5 區塊（頁面上方另有「今日過盤預測」清單，其表頭也含盤口名）。"""
        return page.split("有 edge 的 TOP 5")[1].split("</table>")[0]
    p_plain = _top5_sec(report.render_mlb_page(rows2, date="d", mkt_conf=None))
    p_adj = _top5_sec(report.render_mlb_page(rows2, date="d", mkt_conf=mc))
    assert re.findall(r"錢線|讓分", p_plain)[0] == "錢線"   # 純 edge → 錢線先
    assert re.findall(r"錢線|讓分", p_adj)[0] == "讓分"     # 勝率加權 → 讓分先


def test_bullpen_book_directions_and_pit():
    # 兩隊整季牛棚相同(季 RA/9 ≈ 4.5)，但 A 隊近期爆、B 隊近期神
    def mk_games(team, opp, allowed_seq, start_day=1):
        out = []
        for i, a in enumerate(allowed_seq):
            out.append({"date": f"2025-06-{start_day + i:02d}", "home": team,
                        "away": opp, "home_goals": 4, "away_goals": a})
        return out
    # 先發每場 6 局失 2 分 → 牛棚 3 局失 (allowed-2) 分
    seq_a = [3] * 10 + [6, 6, 6, 6]      # A：季常 1 分/3局，近期 4 分/3局(爆)
    seq_b = [6] * 4 + [3] * 6 + [2, 2, 2, 2]   # B：近期 0 分/3局(神)
    games = mk_games("Alpha", "X", seq_a) + mk_games("Beta", "X", seq_b)
    logs, pid_team = {}, {}
    pid = 100
    for team, seq in (("Alpha", seq_a), ("Beta", seq_b)):
        rows = [{"date": f"2025-06-{i + 1:02d}", "ip": 6.0, "r": 2,
                 "so": 5, "bb": 2, "hr": 0, "gs": 1} for i in range(len(seq))]
        logs[pid] = rows
        pid_team[pid] = team
        pid += 1
    bp = mlb.BullpenBook(games, logs, pid_team, prior_ip=10.0, halflife_days=4.0)
    as_of = "2025-06-20"
    fa, fb = bp.factor("Alpha", as_of), bp.factor("Beta", as_of)
    assert fa > 1.0 > fb                     # 近期爆 → 放大對手得分；近期神 → 壓低
    # point-in-time：更早的 as_of 不看之後的比賽
    fa_early = bp.factor("Alpha", "2025-06-08")   # 只看到穩定期
    assert abs(fa_early - 1.0) < abs(fa - 1.0)
    # 查無隊 / 樣本不足 → 中性
    assert bp.factor("Nobody", as_of) == 1.0
    assert bp.factor("Alpha", "2025-06-02") == 1.0
    # 夾限：係數落在 1 ± BP_SHARE*(CLIP-1) 內
    lo = 1.0 + mlb.BullpenBook.BP_SHARE * (mlb.BullpenBook.CLIP[0] - 1.0)
    hi = 1.0 + mlb.BullpenBook.BP_SHARE * (mlb.BullpenBook.CLIP[1] - 1.0)
    assert lo <= fb < 1.0 < fa <= hi


def test_bullpen_book_skips_partial_doubleheader():
    # 同日雙重賽只有 1 筆先發紀錄 → 該隊日必須整日略過（避免錯算牛棚失分）
    games = [{"date": "2025-06-01", "home": "DH", "away": "X",
              "home_goals": 4, "away_goals": 3},
             {"date": "2025-06-01", "home": "DH", "away": "X",
              "home_goals": 2, "away_goals": 8}]
    logs = {7: [{"date": "2025-06-01", "ip": 6.0, "r": 2, "so": 5, "bb": 1,
                 "hr": 0, "gs": 1}]}
    bp = mlb.BullpenBook(games, logs, {7: "DH"})
    assert "DH" not in bp.team_games
    # 補齊第二筆 → 納入，且牛棚失分 = 總失分 11 − 先發 2+3 = 6、局數 18−12=6
    logs[8] = [{"date": "2025-06-01", "ip": 6.0, "r": 3, "so": 4, "bb": 2,
                "hr": 1, "gs": 1}]
    bp2 = mlb.BullpenBook(games, logs, {7: "DH", 8: "DH"})
    assert bp2.team_games["DH"] == [("2025-06-01", 6.0, 6.0)]


def test_bullpen_relief_appearance_excluded():
    # gs=0 的出賽（假先發/長中繼）不得算進先發合計
    games = [{"date": "2025-06-01", "home": "T", "away": "X",
              "home_goals": 4, "away_goals": 5}]
    logs = {1: [{"date": "2025-06-01", "ip": 5.0, "r": 2, "so": 4, "bb": 1,
                 "hr": 0, "gs": 1}],
            2: [{"date": "2025-06-01", "ip": 2.0, "r": 1, "so": 2, "bb": 0,
                 "hr": 0, "gs": 0}]}   # 後援
    bp = mlb.BullpenBook(games, logs, {1: "T", 2: "T"})
    # 牛棚失分 = 5 − 先發 2 = 3；局數 = 9 − 5 = 4（後援那 2 局屬牛棚，不扣）
    assert bp.team_games["T"] == [("2025-06-01", 3.0, 4.0)]


def test_bullpen_predictor_shifts_totals():
    from footy import mlb_sim
    model = _mlb_model()
    games = []
    for i in range(20):
        games.append({"date": f"2025-06-{i + 1:02d}", "home": "Team0",
                      "away": "Team5", "home_goals": 4,
                      "away_goals": 3 if i < 14 else 8})   # Team0 牛棚近期爆
    logs = {9: [{"date": f"2025-06-{i + 1:02d}", "ip": 6.0, "r": 2, "so": 5,
                 "bb": 1, "hr": 0, "gs": 1} for i in range(20)]}
    bp = mlb.BullpenBook(games, logs, {9: "Team0"}, prior_ip=10.0, halflife_days=4.0)
    g = {"home": "Team0", "away": "Team5", "date": "2025-06-25",
         "_total_line": 8.5, "run_line": -1.5, "park": 1.0}
    base = mlb_sim.nb_predictor(model)(g)
    with_bp = mlb_sim.nb_bullpen_predictor(model, bp)(g)
    assert with_bp["p_over"] > base["p_over"]   # 主隊牛棚爆 → 客隊得分↑ → 大分機率↑


def test_load_with_history(tmp_path):
    cur = tmp_path / "mlb.csv"
    hist = tmp_path / "mlb_hist.csv"
    cur.write_text("date,home,away,home_goals,away_goals\n"
                   "2025-04-01,Cleveland Guardians,Athletics,3,2\n")
    hist.write_text("date,home,away,home_goals,away_goals\n"
                    "2021-05-01,Cleveland Indians,Oakland Athletics,4,1\n"
                    "2025-04-01,Cleveland Guardians,Athletics,3,2\n")   # 重複列
    df = mlb.load_with_history(cur, hist)
    assert len(df) == 2                                  # 去重
    assert set(df["home"]) == {"Cleveland Guardians"}    # 舊名 → 現名
    assert set(df["away"]) == {"Athletics"}
    assert df.iloc[0]["date"].year == 2021               # 依日排序
    # 無歷史檔 → 只載現行
    df2 = mlb.load_with_history(cur, tmp_path / "nope.csv")
    assert len(df2) == 1


def test_pure_picks_and_dual_ledger(tmp_path):
    """雙帳本：pure_picks 照盤口(市場線)全盤口、不帶 edge、路徑推導正確。"""
    from footy import mlb
    assert mlb._derive_model_ledger("data/mlb_bets.csv") == "data/mlb_model_bets.csv"
    assert mlb._derive_model_ledger("x/nba_bets.csv") == "x/nba_model_bets.csv"

    picks = [{"market": "1X2", "selection": "home", "line": "", "odds": 1.9, "edge": 0.02},
             {"market": "OU", "selection": "小", "line": 8.5, "odds": None},
             {"market": "AH", "selection": "主", "line": -1.5, "odds": 1.85, "edge": -0.01}]
    cp = mlb.pure_picks(picks)
    assert len(cp) == 3                              # 全盤口都記
    assert all("edge" not in p for p in cp)          # 賠率/edge 不參與
    assert cp[1]["line"] == 8.5 and cp[2]["line"] == -1.5   # 線照盤口
    assert cp[0]["odds"] == 1.9                      # 賠率保留供結算 ROI
    lp = tmp_path / "m_model_bets.csv"
    n = mlb.log_picks(str(lp), "2026-07-18",
                      {"game_pk": 1, "home": "A", "away": "B"}, cp)
    assert n == 3 and lp.exists()


def test_ou_recommend_disabled_by_default(monkeypatch):
    """OU 停買:帳本實證 edge 反向(43% 勝率/edge>0 只贏 33%),預設不出「買」。"""
    from footy import mlb
    from footy.prematch import MarketQuote

    class M:
        p_home, p_away = 0.52, 0.48
        p_over, p_under = 0.62, 0.38          # 模型強烈看大
        p_cover_home = 0.50
        total_line, run_line = 8.5, -1.5
    quotes = [MarketQuote("OU", "over", 1.90, line=8.5),   # 融合後 edge ≈ +6%（在風控範圍內）
              MarketQuote("OU", "under", 1.90, line=8.5)]
    sig = mlb.bet_signals(M(), quotes)
    assert sig["OU"]["edge"] is not None and sig["OU"]["edge"] > 0
    assert sig["OU"]["verdict"] == "觀望"       # 有正 edge 也不買（閘門）
    # 研究用環境變數可強開
    monkeypatch.setenv("FOOTY_OU_RECOMMEND", "1")
    sig2 = mlb.bet_signals(M(), quotes)
    assert sig2["OU"]["verdict"] == "買"
