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
        w = _csv.DictWriter(f, fieldnames=["id", "name", "team", "ip", "runs", "gs"])
        w.writeheader()
        w.writerows(rows)
    book = mlb.PitcherBook.load_csv(p)
    f1, _ = book.factor(1)
    f2, _ = mlb.PitcherBook(rows).factor(1)
    assert abs(f1 - f2) < 1e-12


def test_zh_mlb():
    assert mlb.zh_mlb("New York Yankees") == "洋基"
    assert mlb.zh_mlb("Unknown Club") == "Unknown Club"


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
    rows = [{"game": {"home": "Team0", "away": "Team5", "home_pitcher": "P One",
                      "away_pitcher": "P Two"},
             "m": m, "pf": 1.08, "hp_note": "RA/9 3.00", "ap_note": "",
             "ml_odds": {"home": 1.5, "away": 2.8}, "ou_odds": {}, "rl_odds": {},
             "status": "Preview"}]
    power = mlb.team_power(model)
    page = report.render_mlb_page(rows, date="2026-07-03", power=power,
                                  track_text="MLB 推薦戰績｜1 勝 0 敗")
    assert "MLB 今日預測" in page and "錢線" in page and "球場因子 1.08" in page
    assert "戰力表" in page and "MLB 推薦戰績" in page
    assert "edge" in page                              # 有 odds → 顯示 edge
    # 空狀態
    empty = report.render_mlb_page([], date="2026-07-03", note="尚未訓練")
    assert "今日無可預測比賽" in empty and "尚未訓練" in empty


def test_cli_wiring():
    from click.testing import CliRunner
    from footy.cli import cli
    out = CliRunner().invoke(cli, ["mlb", "--help"])
    assert out.exit_code == 0
    for cmd in ("fetch", "train", "analyze", "today"):
        assert cmd in out.output
