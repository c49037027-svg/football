"""命令列入口。

子命令：
  fetch-data     下載 football-data.co.uk 歷史資料
  train          訓練 Dixon–Coles 模型並存檔
  scan-prematch  對即將開賽盤口做初盤價值掃描
  backtest       walk-forward 回測驗證是否 +EV
  live           走地實時盯盤（預設用模擬 feed）
"""
from __future__ import annotations

import click
import pandas as pd

from .config import Config
from .data import loader
from .models import dixon_coles as dc


@click.group()
@click.option("--config", "config_path", default=None, help="YAML 設定檔路徑")
@click.pass_context
def cli(ctx, config_path):
    """Footy — 足球 +EV 預測與實時盯盤（研究/提示用途，不自動下注）。"""
    ctx.ensure_object(dict)
    ctx.obj["cfg"] = Config.load(config_path)


@cli.command("fetch-data")
@click.option("--league", required=True, help="聯賽代碼，如 E0/SP1/I1/D1/F1")
@click.option("--seasons", required=True, multiple=True, type=int,
              help="球季起始年，可多個，如 --seasons 2021 2022 2023")
@click.option("--out", default=None, help="輸出 CSV 路徑（預設 data/<league>.csv）")
def fetch_data(league, seasons, out):
    out = out or f"data/{league}.csv"
    loader.fetch(league, list(seasons), out_path=out)


@cli.command("fetch-github")
@click.option("--league", required=True, help="聯賽代碼，如 E0/SP1/I1/D1/F1")
@click.option("--out", default=None, help="輸出 CSV 路徑（預設 data/<league>.csv）")
def fetch_github(league, out):
    """從 GitHub 鏡像下載含賠率的歷史資料（官方站被網路政策擋住時使用）。"""
    out = out or f"data/{league}.csv"
    loader.fetch_github(league, out_path=out)


@cli.command("fetch-xg")
@click.option("--league", required=True, help="聯賽代碼 E0/SP1/I1/D1/F1")
@click.option("--seasons", multiple=True, type=int, required=True,
              help="球季起始年（可多個），如 --seasons 2022 --seasons 2023")
@click.option("--out", default=None, help="輸出 CSV（預設 data/xg_<league>.csv）")
def fetch_xg(league, seasons, out):
    """抓 understat 真實 xG（每場 h/a xG），存成含 home_xg/away_xg 的訓練 CSV。

    需連 understat（沙箱擋，跑在 Actions/Render）。五大聯賽遷移用；比射正代理準。
    """
    from .data import understat
    out = out or f"data/xg_{league}.csv"
    understat.build_league_csv(league, list(seasons), out_path=out)


@cli.command("train")
@click.option("--data", "data_path", required=True, help="歷史資料 CSV")
@click.option("--out", default=None, help="模型輸出路徑（預設 models/<name>.pkl）")
@click.option("--half-life", default=None, type=float, help="時間衰減半衰期（天）")
@click.option("--xg-weight", default=None, type=float, help="xG 混合權重 0~1（需資料含 xG）")
@click.option("--use-elo", is_flag=True, default=False, help="把賽前 Elo 當特徵（需資料含 Elo）")
@click.option("--reg", default=None, type=float, help="L2 正則化強度（往聯盟平均收縮）")
@click.pass_context
def train(ctx, data_path, out, half_life, xg_weight, use_elo, reg):
    cfg: Config = ctx.obj["cfg"]
    if half_life is not None:
        cfg.model.half_life_days = half_life
    if xg_weight is not None:
        cfg.model.xg_weight = xg_weight
    if use_elo:
        cfg.model.use_elo = True
    if reg is not None:
        cfg.model.reg = reg
    df = loader.load_csv(data_path)
    click.echo(f"[train] 載入 {len(df)} 場比賽，開始擬合"
               f"（half_life={cfg.model.half_life_days}天, xg_weight={cfg.model.xg_weight}, "
               f"reg={cfg.model.reg}）…")
    model = dc.fit(df, half_life_days=cfg.model.half_life_days,
                   max_goals=cfg.model.max_goals, rho_init=cfg.model.rho_init,
                   xg_weight=cfg.model.xg_weight, use_elo=cfg.model.use_elo,
                   reg=cfg.model.reg, verbose=True)
    out = out or f"models/{_stem(data_path)}.pkl"
    model.save(out)
    click.echo(f"[ok] 模型已存：{out}（{len(model.teams)} 隊，主場優勢={model.home_adv:.3f}，rho={model.rho:.3f}）")


@cli.command("scan-prematch")
@click.option("--model", "model_path", required=True)
@click.option("--fixtures", required=True, help="即將開賽盤口 CSV")
@click.option("--adjustments", default=None, help="傷停/輪休手動調整 CSV（選用）")
@click.pass_context
def scan_prematch(ctx, model_path, fixtures, adjustments):
    from . import prematch
    cfg: Config = ctx.obj["cfg"]
    model = dc.DixonColesModel.load(model_path)
    fx = pd.read_csv(fixtures)
    adj = None
    if adjustments:
        from .context import load_adjustments_csv
        adj = load_adjustments_csv(adjustments)
        click.echo(f"[ctx] 已載入 {len(adj)} 場情境調整")
    rows = prematch.scan(model, fx, cfg, adjustments=adj)
    if not rows:
        click.echo("沒有找到符合門檻的 value 下注。")
        return
    click.echo(f"\n找到 {len(rows)} 個 value 下注建議：\n")
    for r in rows:
        click.echo(
            f"  {r['match']:<28} {r['market']:>8} {r['selection']:<6} @ {r['odds']:>5} "
            f"| edge={r['edge']:+.1%} EV={r['ev']:+.3f} 下注={r['suggested_stake']:.2f} ({r['risk_note']})"
        )


@cli.command("predict")
@click.option("--model", "model_path", required=True)
@click.option("--fixtures", required=True, help="賽程 CSV（需含 home, away 欄位）")
@click.option("--history", "history_path", default=None,
              help="歷史資料 CSV（用於近期狀態/H2H，選用）")
@click.option("--adjustments", default=None, help="傷停/輪休手動調整 CSV（選用）")
@click.option("--html", "html_out", default=None, help="輸出 HTML 預測頁路徑")
@click.option("--md", "md_out", default=None, help="輸出 Markdown 路徑")
@click.option("--title", default="今日足球預測", help="頁面標題")
@click.pass_context
def predict(ctx, model_path, fixtures, history_path, adjustments, html_out, md_out, title):
    """產生像預測站那樣的每場比賽預測內容（1X2/比分/大小球/BTTS/狀態）。"""
    from . import predict as predmod
    from . import report
    model = dc.DixonColesModel.load(model_path)
    fx = pd.read_csv(fixtures)
    hist = loader.load_csv(history_path) if history_path else None
    adj = None
    if adjustments:
        from .context import load_adjustments_csv
        adj = load_adjustments_csv(adjustments)
    preds = predmod.predict_fixtures(model, fx, history=hist, adjustments=adj)
    if not preds:
        click.echo("沒有可預測的比賽（球隊不在模型中？）。")
        return
    for p in preds:
        click.echo(report.render_console(p))
        click.echo("")
    if md_out:
        from pathlib import Path
        Path(md_out).parent.mkdir(parents=True, exist_ok=True)
        Path(md_out).write_text(report.render_markdown(preds, title), encoding="utf-8")
        click.echo(f"[ok] 已輸出 Markdown：{md_out}")
    if html_out:
        report.write_html(preds, html_out, title)
        click.echo(f"[ok] 已輸出 HTML：{html_out}")


@cli.command("fetch-intl")
@click.option("--out", default="data/intl.csv", help="輸出 CSV 路徑")
@click.option("--since", default="2006-01-01", help="只取此日期後的國際賽")
def fetch_intl(out, since):
    """下載國際賽結果（martj42 公開資料）並計算 Elo 評分。"""
    from .intl import data as intl
    df = intl.fetch_international(out_path=None, since=since)
    df, elo = intl.compute_elo(df)
    from pathlib import Path
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    top = sorted(elo.items(), key=lambda x: x[1], reverse=True)[:10]
    click.echo(f"[ok] 已存 {len(df)} 場到 {out}（含賽前 Elo）")
    click.echo("Elo 前十：" + ", ".join(f"{t} {r:.0f}" for t, r in top))


@cli.command("analyze")
@click.option("--model", "model_path", required=True)
@click.option("--home", required=True)
@click.option("--away", required=True)
@click.option("--history", "history_path", default=None, help="歷史賽果 CSV（狀態/H2H）")
@click.option("--neutral/--home-away", default=True, help="中立場（世界盃預設中立）")
@click.option("--knockout", is_flag=True, default=False, help="淘汰賽（黃牌加成）")
@click.option("--n-sims", default=50000, type=int)
@click.option("--home-formation", default=None, help="主隊陣型，如 4-3-3（手動輸入）")
@click.option("--away-formation", default=None, help="客隊陣型，如 5-3-2（手動輸入）")
@click.option("--home-missing", default=0, type=int, help="主隊缺陣主力人數")
@click.option("--away-missing", default=0, type=int, help="客隊缺陣主力人數")
@click.option("--ah-line", default=None, type=float,
              help="指定盤口讓球線（主隊視角，如 -1.5），模型在此線上評估買哪邊")
@click.option("--html", "html_out", default=None, help="輸出分析 HTML")
@click.option("--title", default=None, help="頁面標題")
@click.pass_context
def analyze(ctx, model_path, home, away, history_path, neutral, knockout,
            n_sims, home_formation, away_formation, home_missing, away_missing,
            ah_line, html_out, title):
    """世界盃單場深度分析（比分/大小/BTTS/亞盤/角球/黃牌/上半場/因子）。"""
    from . import analysis, report, context
    model = dc.DixonColesModel.load(model_path)
    if home not in model.attack or away not in model.attack:
        raise click.ClickException(f"模型未包含 {home} 或 {away}")
    hist = loader.load_csv(history_path) if history_path else None
    # 缺陣調整（陣型由 analyze 內部依 home/away_formation 套用，不在此重複）
    adj = (context.injuries_to_adjustment(home_missing, away_missing)
           if (home_missing or away_missing) else None)
    a = analysis.analyze(model, home, away, history=hist, neutral=neutral,
                         knockout=knockout, n_sims=n_sims, adjustment=adj,
                         home_formation=home_formation or "",
                         away_formation=away_formation or "",
                         ah_line_override=ah_line)
    if home_missing or away_missing:
        a.player_note_home = f"缺主力 {home_missing} 人" if home_missing else "主力盡出"
        a.player_note_away = f"缺主力 {away_missing} 人" if away_missing else "主力盡出"
    click.echo(report.render_analysis_console(a))
    if html_out:
        report.write_analysis_html(a, html_out, title or f"{home} vs {away} 分析")
        click.echo(f"[ok] 已輸出 HTML：{html_out}")


@cli.command("track")
@click.option("--model", "model_path", default="models/intl.pkl")
@click.option("--schedule", default="data/wc2026.json", help="賽程 JSON（取賽果結算）")
@click.option("--ledger", default="data/bets.csv", help="戰績帳本 CSV")
@click.option("--odds/--no-odds", default=True, help="抓真實盤口記 +EV 推薦與 ROI/CLV（需 ODDS_API_KEY）")
def track(model_path, schedule, ledger, odds):
    """記錄各推薦項目、用賽果結算過/沒過；有盤口時另算實際 ROI 與 CLV。"""
    from . import tracker, worldcup as wc
    model = dc.DixonColesModel.load(model_path)
    _, matches, _ = wc.parse_wc_json(schedule)
    odds_index = None
    if odds:
        try:
            from .live.providers import fetch_wc_odds
            odds_index = fetch_wc_odds(matches)
            click.echo(f"[track] 已抓盤口：{len(odds_index)} 場有 +EV 候選")
        except Exception as e:  # noqa: BLE001
            click.echo(f"[track] 無真實盤口（改記勝率自我校驗）：{e}")
    s = tracker.prepare(matches, model, ledger, odds_index=odds_index)
    click.echo(s.text())


@cli.command("backfill")
@click.option("--model", "model_path", default="models/intl.pkl")
@click.option("--schedule", default="data/wc2026.json", help="賽程 JSON")
@click.option("--ledger", default="data/bets.csv", help="戰績帳本 CSV")
def backfill(model_path, schedule, ledger):
    """整屆回填：把開賽至今所有已踢比賽補記模型推薦並結算（只計勝率，非盤口損益）。"""
    from . import tracker, worldcup as wc
    model = dc.DixonColesModel.load(model_path)
    _, matches, _ = wc.parse_wc_json(schedule)
    n = tracker.backfill_played(matches, model, ledger)
    click.echo(f"[backfill] 回填 {n} 筆已踢比賽的模型推薦（已結算）")
    click.echo(tracker.summary(ledger).text())


@cli.command("rebuild-ledger")
@click.option("--ledger", default="data/bets.csv", help="戰績帳本 CSV")
@click.option("--snap", default="data/odds_log.csv", help="賠率快照 CSV")
def rebuild_ledger(ledger, snap):
    """用賠率快照重建真實盤口戰績（模型推薦+快照賠率，已踢者結算收益）。"""
    from . import tracker
    n = tracker.rebuild_market_from_snapshots(ledger, snap)
    click.echo(f"[rebuild] 由快照重建 {n} 筆真實盤口推薦")
    click.echo(tracker.summary(ledger).text())


@cli.command("tune-blend")
@click.option("--snap", default="data/odds_log.csv", help="賠率快照（由 track/wc-site 累積）")
def tune_blend(snap):
    """用歷史賠率快照回測各融合權重的 ROI/CLV，建議 BLEND_WEIGHT。"""
    from . import tracker
    res = tracker.tune_weight(snap)
    if not res["rows"]:
        click.echo(f"[tune] {snap} 尚無『已結算』快照——需先累積有真實盤口的"
                   "已踢比賽。設好 ODDS_API_KEY 跑幾天 track/wc-site 後再試。")
        return
    click.echo(f"回測 {res['n_matches']} 個決策單位：")
    click.echo(f"{'權重':>4} {'注數':>4} {'損益':>8} {'ROI':>8} {'CLV':>8}")
    for r in res["rows"]:
        if r["n_bets"] == 0:
            continue
        star = " ★" if res["best_roi"] and abs(r["weight"] - res["best_roi"]["weight"]) < 1e-9 else ""
        click.echo(f"{r['weight']:>4.1f} {r['n_bets']:>4} {r['pl']:>+8.2f} "
                   f"{r['roi']:>+8.1%} {r['clv']:>+8.1%}{star}")
    b = res["best_roi"]
    if b:
        click.echo(f"\n建議：BLEND_WEIGHT={b['weight']:.1f}（ROI {b['roi']:+.1%}）。"
                   "樣本少時 CLV 比 ROI 可靠；以環境變數設定即可生效。")


@cli.group("agent")
def agent():
    """AI agents（賽前分析/辯論/風控/賽後檢討/新聞抽取）。需設 ANTHROPIC_API_KEY。"""


def _analyze_one(model_path, schedule, home, away):
    from . import analysis, worldcup as wc
    from .models import dixon_coles as dc
    model = dc.DixonColesModel.load(model_path)
    if home not in model.attack or away not in model.attack:
        raise click.ClickException(f"球隊不在模型中：{home} / {away}")
    hist = None
    try:
        from .data import loader
        hist = loader.load_csv("data/intl.csv")
    except Exception:  # noqa: BLE001
        pass
    return analysis.analyze(model, home, away, history=hist, neutral=True, n_sims=10000)


@agent.command("check")
@click.option("--live/--no-live", default=False, help="實際呼叫一次確認 key/model 可用")
def agent_check(live):
    """印出目前 LLM 設定；--live 會真的呼叫一次 Claude 驗證連線。"""
    from .agents import llm
    c = llm.config()
    click.echo(f"provider：Anthropic Claude　model：{c['model']}　"
               f"金鑰：{'已設定' if c['key'] else '未設定（agent 會略過）'}")
    if not live:
        return
    if not c["key"]:
        click.echo("[ai-check] 未設金鑰，略過實測。")
        return
    try:
        out = llm.complete("用繁體中文回覆「可用」兩個字。", max_tokens=80, timeout=30)
        status = "PASS" if out else "PASS(但回覆為空)"
        click.echo(f"[ai-check] {status}（model={c['model']}）回覆：{out[:60]}")
    except Exception as e:  # noqa: BLE001
        click.echo(f"[ai-check] FAIL：{str(e)[:300]}")


@agent.command("preview")
@click.option("--model", "model_path", default="models/intl.pkl")
@click.option("--schedule", default="data/wc2026.json")
@click.argument("home")
@click.argument("away")
def agent_preview(model_path, schedule, home, away):
    """賽前分析：footy agent preview Brazil Croatia"""
    from .agents import roles
    a = _analyze_one(model_path, schedule, home, away)
    out = roles.preview(a)
    click.echo(out or "（未設金鑰或產生失敗）")


@agent.command("debate")
@click.option("--model", "model_path", default="models/intl.pkl")
@click.option("--schedule", default="data/wc2026.json")
@click.argument("home")
@click.argument("away")
def agent_debate(model_path, schedule, home, away):
    """多代理辯論 + 裁判綜合。"""
    from .agents import roles
    a = _analyze_one(model_path, schedule, home, away)
    res = roles.debate(a)
    if not res:
        click.echo("（未設金鑰或產生失敗）")
        return
    for x in res["analysts"]:
        click.echo(f"[{x['role']}] {x['view']}")
    v = res.get("verdict") or {}
    click.echo(f"\n裁判：{v.get('lean','?')}（信心 {v.get('confidence','?')}）"
               f"— {v.get('summary','')}")


@agent.command("risk")
@click.option("--ledger", default="data/bets.csv")
def agent_risk(ledger):
    """風控：檢視待結算的推薦清單。"""
    from . import tracker
    from .agents import roles
    df = tracker.load_ledger(ledger)
    pend = df[df["result"] == "pending"].to_dict("records")
    out = roles.risk_review(pend)
    click.echo(out or "（未設金鑰、無待結算推薦或產生失敗）")


@agent.command("postmortem")
@click.option("--ledger", default="data/bets.csv")
@click.option("--n", default=40, type=int, help="取最近 N 筆已結算")
def agent_postmortem(ledger, n):
    """賽後檢討：分析最近已結算的推薦結果。"""
    from . import tracker
    from .agents import roles
    df = tracker.load_ledger(ledger)
    rows = df[df["result"].isin(["win", "loss", "push"])].tail(n).to_dict("records")
    out = roles.postmortem(rows)
    click.echo(out or "（未設金鑰、無已結算紀錄或產生失敗）")


@agent.command("news")
@click.option("--model", "model_path", default="models/intl.pkl")
@click.option("--schedule", default="data/wc2026.json")
@click.option("--file", "news_file", required=True, help="新聞文字檔（不上網，只讀此檔）")
@click.argument("home")
@click.argument("away")
def agent_news(model_path, schedule, news_file, home, away):
    """從新聞文字抽出缺陣/陣型，並用模型重算（套用調整）。"""
    from pathlib import Path

    from . import analysis, context
    from .agents import roles
    from .models import dixon_coles as dc
    text = Path(news_file).read_text(encoding="utf-8")
    info = roles.extract_news(home, away, text)
    if not info:
        click.echo("（未設金鑰、新聞空白或產生失敗）")
        return
    click.echo(f"抽取結果：{info}")
    hm = int((info.get("home") or {}).get("missing", 0) or 0)
    am = int((info.get("away") or {}).get("missing", 0) or 0)
    hf = (info.get("home") or {}).get("formation", "") or ""
    af = (info.get("away") or {}).get("formation", "") or ""
    model = dc.DixonColesModel.load(model_path)
    adj = context.injuries_to_adjustment(hm, am) if (hm or am) else None
    a = analysis.analyze(model, home, away, neutral=True, n_sims=10000,
                         adjustment=adj, home_formation=hf, away_formation=af)
    click.echo(f"套用後 1X2：主 {a.odds_home:.0%} / 和 {a.odds_draw:.0%} / 客 {a.odds_away:.0%}"
               f"；讓盤 {a.ah_line:+g}")


@cli.group("mlb")
def mlb_group():
    """MLB 美國職棒預測（錢線/讓分/大小）。資料源 statsapi.mlb.com（免費）。"""


@cli.group("nba")
def nba_group():
    """NBA 美國職籃預測（錢線/讓分/大小）。資料源 nba.com（免費）。"""


@nba_group.command("fetch")
@click.option("--out", default="data/nba.csv")
def nba_fetch(out):
    """下載本季賽果（cdn.nba.com 賽程，一次呼叫）。沙箱擋，需在 Actions 跑。"""
    from . import nba
    rows = nba.parse_schedule_v2(nba.fetch_schedule(), finals_only=True)
    n = nba.write_games_csv(rows, out)
    d = f"{rows[0]['date']}→{rows[-1]['date']}" if rows else "無"
    click.echo(f"[nba] 已存 {n} 場到 {out}（{d}）")


@nba_group.command("fetch-history")
@click.option("--seasons", multiple=True, required=True,
              help="球季，可多個：--seasons 2021-22 --seasons 2022-23 …")
@click.option("--out", default="data/nba_hist.csv")
def nba_fetch_history(seasons, out):
    """下載歷史賽季（ESPN scoreboard 逐月抓；stats.nba.com 擋雲端 IP 不可用）。"""
    import time as _t

    from . import nba
    rows, seen = [], set()
    for s in seasons:
        n_season = 0
        for a, b in nba.season_months(s):
            try:
                got = nba.fetch_espn_range(a, b)
            except Exception as e:  # noqa: BLE001
                click.echo(f"[nba] {s} {a[:6]} 失敗：{e}")
                continue
            for g in got:
                key = (g["date"], g["home"], g["away"])
                if key in seen:
                    continue
                seen.add(key)
                rows.append(g)
                n_season += 1
            _t.sleep(0.3)
        click.echo(f"[nba] {s}：{n_season} 場")
    rows.sort(key=lambda r: r["date"])
    n = nba.write_games_csv(rows, out)
    click.echo(f"[nba] 共 {n} 場 → {out}")
    if n < 500 * len(seasons):
        raise click.ClickException(f"場數異常偏低（{n}），資料源可能被擋，不提交。")


@nba_group.command("train")
@click.option("--data", "data_path", default="data/nba.csv")
@click.option("--hist", "hist_path", default="data/nba_hist.csv",
              help="歷史賽季 CSV（存在則合併；傳空字串停用）")
@click.option("--out", default="models/nba.pkl")
@click.option("--half-life", default=90.0, type=float,
              help="時間衰減半衰期（天）。回測：兩窗口 80-100 天最佳（NBA 要看近況）")
@click.option("--reg", default=3.0, type=float, help="攻防評分 L2 收縮強度（回測 reg=3 最佳）")
def nba_train(data_path, hist_path, out, half_life, reg):
    """訓練攻防評分模型（加權嶺回歸 + 常態殘差）。"""
    from . import nba
    df = nba.load_with_history(data_path, hist_path or None)
    click.echo(f"[nba] 載入 {len(df)} 場（{df['date'].min().date()}→"
               f"{df['date'].max().date()}），擬合中（half_life={half_life}天, reg={reg}）…")
    model = nba.fit_ratings(df, half_life_days=half_life, reg=reg)
    model.save(out)
    click.echo(f"[ok] 模型已存：{out}（{len(model.teams)} 隊，主場優勢="
               f"{model.home_adv:+.2f} 分，σ分差={model.sigma_margin:.1f}"
               f"，σ總分={model.sigma_total:.1f}）")


@nba_group.command("today")
@click.option("--model", "model_path", default="models/nba.pkl")
@click.option("--date", default=None, help="美東賽程日 YYYY-MM-DD（預設今天）")
def nba_today(model_path, date):
    """列出今日各場模型看法（無盤口版；網站版含盤口）。"""
    from . import mlb, nba
    model = nba.NBAModel.load(model_path)
    date = date or mlb.us_today()
    sched = nba.parse_schedule_v2(nba.fetch_schedule(), finals_only=False)
    games = [g for g in sched if g["date"] == date
             and g["home"] in model.off and g["away"] in model.off]
    if not games:
        click.echo(f"{date} 無 NBA 賽事（或休賽季）。")
        return
    for g in games:
        m = nba.analyze_game(model, g["home"], g["away"])
        hz, az = nba.zh_nba(g["home"]), nba.zh_nba(g["away"])
        click.echo(f"{az} @ {hz}｜預期 {m.exp_away:.0f}–{m.exp_home:.0f}"
                   f"｜主勝 {m.p_home:.0%}｜大小 {m.total_line:g} 大 {m.p_over:.0%}"
                   f"｜讓分 {m.run_line:+g} 主過盤 {m.p_cover_home:.0%}")


@nba_group.command("backtest")
@click.option("--data", "data_path", default="data/nba.csv")
@click.option("--hist", "hist_path", default="data/nba_hist.csv")
@click.option("--cut", required=True, help="測試起日（之前訓練，walk-forward）")
@click.option("--end", default="9999-12-31", help="測試迄日")
@click.option("--half-life", "half_lives", multiple=True, type=float,
              default=(60.0, 90.0, 140.0, 300.0), help="半衰期網格")
@click.option("--reg", default=3.0, type=float)
def nba_backtest(data_path, hist_path, cut, end, half_lives, reg):
    """walk-forward 回測：錢線/大小/讓分 log-loss（線用模型預期最近 .5，公平對比）。"""
    import numpy as np
    import pandas as pd

    from . import nba
    df = nba.load_with_history(data_path, hist_path or None)
    cut_ts, end_ts = pd.Timestamp(cut), pd.Timestamp(end)
    test = df[(df["date"] >= cut_ts) & (df["date"] <= end_ts)]
    train_all = df[df["date"] < cut_ts]
    click.echo(f"[nba-bt] 訓練 {len(train_all)} 場（<{cut}）｜測試 {len(test)} 場")
    eps = 1e-12
    # 大小/讓分若用「模型自取線」評估會恆為五五波（無資訊），故用不依賴盤口線的
    # 分差/總分 MAE + 錢線 log-loss/準確率/校準來比。
    print(f"{'半衰期':>8} | 錢線LL | 準確率 | 分差MAE | 總分MAE | 平均P(主) vs 實際 | n")
    for hl in half_lives:
        model = nba.fit_ratings(train_all, half_life_days=hl, reg=reg,
                                reference_date=cut_ts)
        ml = mae_m = mae_t = cal_p = cal_y = 0.0
        acc = n = 0
        for r in test.itertuples():
            if r.home not in model.off or r.away not in model.off:
                continue
            m = nba.analyze_game(model, r.home, r.away)
            hg, ag = int(r.home_goals), int(r.away_goals)
            y = 1.0 if hg > ag else 0.0
            ml -= y * np.log(max(m.p_home, eps)) + (1 - y) * np.log(max(1 - m.p_home, eps))
            acc += (m.p_home >= 0.5) == (y == 1.0)
            mae_m += abs((hg - ag) - (m.exp_home - m.exp_away))
            mae_t += abs((hg + ag) - (m.exp_home + m.exp_away))
            cal_p += m.p_home; cal_y += y
            n += 1
        name = "等權" if hl >= 1e6 else f"{hl:.0f}天"
        print(f"{name:>8} | {ml/n:.4f} | {acc/n:.1%} | {mae_m/n:.2f} | {mae_t/n:.2f}"
              f" | {cal_p/n:.3f} vs {cal_y/n:.3f} | {n}")


@mlb_group.command("fetch")
@click.option("--seasons", multiple=True, type=int, required=True,
              help="球季年份，可多個：--seasons 2024 --seasons 2025 --seasons 2026")
@click.option("--out", default="data/mlb.csv")
def mlb_fetch(seasons, out):
    """下載球季賽果（例行賽+季後賽）。沙箱擋 statsapi，需在 Render/Actions 跑。"""
    from . import mlb
    n = mlb.fetch_seasons(list(seasons), out)
    click.echo(f"[mlb] 已存 {n} 場到 {out}")


@mlb_group.command("train")
@click.option("--data", "data_path", default="data/mlb.csv")
@click.option("--hist", "hist_path", default="data/mlb_hist.csv",
              help="歷史賽季 CSV（存在則合併訓練；傳空字串停用）")
@click.option("--out", default="models/mlb.pkl")
@click.option("--half-life", default=365.0, type=float,
              help="時間衰減半衰期（天）。回測：全史+365 天四窗口勝 2 季+120 天")
@click.option("--reg", default=0.3, type=float, help="L2 正則化（往聯盟平均收縮）")
def mlb_train(data_path, hist_path, out, half_life, reg):
    """以得分（runs）訓練 Dixon–Coles（max_goals=20、無低比分修正）。

    隊伍強度用「全史+半衰期」訓練；球場係數/離散度另在建站時以近 2 季
    （data/mlb.csv）估計——回測顯示結構參數用短窗口較準（FINDINGS）。
    """
    from . import mlb
    df = mlb.load_with_history(data_path, hist_path or None)
    click.echo(f"[mlb] 載入 {len(df)} 場（{df['date'].min().date()}→"
               f"{df['date'].max().date()}），擬合中（half_life={half_life}天, reg={reg}）…")
    model = dc.fit(df, half_life_days=half_life, max_goals=20, rho_init=0.0,
                   reg=reg, verbose=True)
    model.save(out)
    click.echo(f"[ok] 模型已存：{out}（{len(model.teams)} 隊，主場優勢={model.home_adv:.3f}）")


@mlb_group.command("fetch-pitchers")
@click.option("--season", default=2026, type=int)
@click.option("--out", default="data/mlb_pitchers.csv")
def mlb_fetch_pitchers(season, out):
    """下載整季投手數據（一次呼叫），供先發投手評分。需在 Render/Actions 跑。"""
    from . import mlb
    rows = mlb.fetch_pitchers(season, out_path=out)
    click.echo(f"[mlb] 已存 {len(rows)} 位投手到 {out}")


def _load_pitcher_book(path):
    from pathlib import Path

    from . import mlb
    if path and Path(path).exists():
        return mlb.PitcherBook.load_csv(path)
    return None


@mlb_group.command("analyze")
@click.option("--model", "model_path", default="models/mlb.pkl")
@click.option("--total-line", default=8.5, type=float, help="大小分線")
@click.option("--run-line", default=-1.5, type=float, help="讓分線（主隊視角）")
@click.option("--pitchers", "pitchers_path", default="data/mlb_pitchers.csv",
              help="投手數據 CSV（mlb fetch-pitchers 產出；檔案不存在則不調整）")
@click.option("--data", "data_path", default="data/mlb.csv",
              help="訓練資料 CSV（估球場因子與負二項離散度；不存在則不調整）")
@click.option("--home-pitcher", default=None, help="主隊先發姓名（英文全名）")
@click.option("--away-pitcher", default=None, help="客隊先發姓名（英文全名）")
@click.argument("home")
@click.argument("away")
def mlb_analyze(model_path, total_line, run_line, pitchers_path, data_path,
                home_pitcher, away_pitcher, home, away):
    """單場分析：footy mlb analyze "New York Yankees" "Boston Red Sox" """
    from . import mlb
    model = dc.DixonColesModel.load(model_path)
    if home not in model.attack or away not in model.attack:
        raise click.ClickException(f"球隊不在模型中（需完整隊名，如 New York Yankees）")
    hf = af = 1.0
    book = _load_pitcher_book(pitchers_path) if (home_pitcher or away_pitcher) else None
    notes = []
    if book:
        if home_pitcher:
            hf, n = book.factor(home_pitcher)
            notes.append(f"主先發 {home_pitcher}：{n}")
        if away_pitcher:
            af, n = book.factor(away_pitcher)
            notes.append(f"客先發 {away_pitcher}：{n}")
    elif home_pitcher or away_pitcher:
        notes.append(f"（找不到投手數據 {pitchers_path}，未調整——先跑 mlb fetch-pitchers）")
    pf = mlb.park_factors_from_csv(data_path).get(home, 1.0)
    disp = mlb.dispersion_from_csv(data_path)
    m = mlb.analyze_game(model, home, away, total_line=total_line, run_line=run_line,
                         home_pitcher_factor=hf, away_pitcher_factor=af,
                         park_factor=pf, dispersion=disp)
    hz, az = mlb.zh_mlb(home), mlb.zh_mlb(away)
    click.echo(f"\n{hz}(主) vs {az}")
    for n in notes:
        click.echo(f"  {n}")
    click.echo(f"  期望得分     {m.exp_home:.2f} : {m.exp_away:.2f}")
    click.echo(f"  錢線         {hz} {m.p_home:.1%}（公平賠率 {m.ml_home_odds}）"
               f" / {az} {m.p_away:.1%}（{m.ml_away_odds}）")
    click.echo(f"  讓分 {m.run_line:+g}    {hz} 過盤 {m.p_cover_home:.1%}")
    click.echo(f"  大小 {m.total_line}    大 {m.p_over:.1%} / 小 {m.p_under:.1%}")
    tops = "、".join(f"{h}-{a} {p:.1%}" for (h, a), p in m.top_scores)
    click.echo(f"  最可能比分   {tops}")
    click.echo("  ⚠️ 供研究參考、非投注建議。")


@mlb_group.command("eval")
@click.option("--data", "data_path", default="data/mlb.csv")
@click.option("--cut", default="2026-05-01", help="切分日：之前訓練、之後評估（無前視）")
def mlb_eval(data_path, cut):
    """回測：比較 Poisson vs 負二項（不同離散度 k）的錢線/大小盤預測品質。"""
    import pandas as pd

    from . import mlb
    df = pd.read_csv(data_path)
    res = mlb.evaluate(df, cut)
    click.echo(f"訓練 {res['n_train']} 場｜測試 {res['n_test']} 場｜"
               f"動差法 k={res['k_mom']:.2f}" if res['k_mom'] else "無過度離散")
    click.echo(f"{'k':>8} {'錢線LL':>8} {'大小LL':>8} {'大小Brier':>9} "
               f"{'平均P(主)':>9} {'實際主勝率':>10}")
    for r in res["rows"]:
        ks = "Poisson" if r["k"] is None else f"{r['k']:.2f}"
        click.echo(f"{ks:>8} {r['ml_logloss']:>8.4f} {r['ou_logloss']:>8.4f} "
                   f"{r['ou_brier']:>9.4f} {r['mean_p_home']:>9.3f} "
                   f"{r['home_win_rate']:>10.3f}")


@mlb_group.command("today")
@click.option("--model", "model_path", default="models/mlb.pkl")
@click.option("--date", default=None, help="日期 YYYY-MM-DD（預設今天）")
@click.option("--odds/--no-odds", default=True, help="抓真實盤口比對（需 ODDS_API_KEY）")
@click.option("--pitchers", "pitchers_path", default="data/mlb_pitchers.csv",
              help="投手數據 CSV（存在則自動套用先發評分）")
@click.option("--data", "data_path", default="data/mlb.csv",
              help="訓練資料 CSV（估球場因子與離散度）")
def mlb_today(model_path, date, odds, pitchers_path, data_path):
    """今日賽程逐場預測（自動套用預告先發評分），可比對真實盤口。"""
    import datetime as _dt
    from . import mlb
    date = date or _dt.date.today().isoformat()
    model = dc.DixonColesModel.load(model_path)
    try:
        games = mlb.fetch_today(date)
    except Exception as e:  # noqa: BLE001
        raise click.ClickException(f"抓賽程失敗（沙箱擋 statsapi？需在 Render/Actions 跑）：{e}")
    if not games:
        click.echo(f"{date} 無比賽。")
        return
    pf_map = mlb.park_factors_from_csv(data_path)
    disp = mlb.dispersion_from_csv(data_path)
    book = _load_pitcher_book(pitchers_path)
    if book:
        click.echo(f"[mlb] 先發投手評分：{len(book.rows)} 位（{pitchers_path}）")
    else:
        click.echo("[mlb] 無投手數據，未做先發調整（跑 mlb fetch-pitchers 可啟用）")
    gobjs = [mlb._Game(i + 1, g["home"], g["away"]) for i, g in enumerate(games)]
    odds_index = {}
    if odds:
        try:
            odds_index = mlb.fetch_mlb_odds(gobjs)
            click.echo(f"[mlb] 已抓盤口：{len(odds_index)} 場")
        except Exception as e:  # noqa: BLE001
            click.echo(f"[mlb] 無盤口（{e}），僅顯示模型預測")
    for i, g in enumerate(games):
        h, a = g["home"], g["away"]
        hz, az = mlb.zh_mlb(h), mlb.zh_mlb(a)
        line = f"{date} {az} @ {hz}"
        if g.get("away_pitcher") or g.get("home_pitcher"):
            line += f"（先發 {g.get('away_pitcher') or '?'} vs {g.get('home_pitcher') or '?'}）"
        click.echo("\n" + line)
        if h not in model.attack or a not in model.attack:
            click.echo("  （球隊不在模型中，先跑 mlb fetch + mlb train）")
            continue
        hf = af = 1.0
        if book:
            hf, hn = book.factor(g.get("home_pitcher_id") or g.get("home_pitcher"))
            af, an = book.factor(g.get("away_pitcher_id") or g.get("away_pitcher"))
            if g.get("home_pitcher") or g.get("away_pitcher"):
                click.echo(f"  先發評分：主 {hn}｜客 {an}")
        m = mlb.analyze_game(model, h, a,
                             home_pitcher_factor=hf, away_pitcher_factor=af,
                             park_factor=pf_map.get(h, 1.0), dispersion=disp)
        click.echo(f"  模型：{hz} 勝 {m.p_home:.1%}｜大小8.5 大 {m.p_over:.1%}"
                   f"｜{hz} -1.5 過盤 {m.p_cover_home:.1%}")
        quotes = odds_index.get(i + 1)
        if quotes:
            ml = {q.selection: q.odds for q in quotes if q.market == "1X2"}
            if "home" in ml:
                edge_h = m.p_home * ml["home"] - 1
                edge_a = m.p_away * ml.get("away", 0) - 1 if ml.get("away") else None
                s = f"  盤口：{hz} @{ml['home']}（edge {edge_h:+.1%}）"
                if edge_a is not None:
                    s += f" / {az} @{ml['away']}（edge {edge_a:+.1%}）"
                click.echo(s)


@mlb_group.command("backtest-sim")
@click.option("--data", "data_path", default="data/mlb.csv",
              help="賽果 CSV（訓練 NB 模型 + 估球場/離散度）")
@click.option("--start", required=True, help="測試起日 YYYY-MM-DD（之前訓練 NB、當作 walk-forward 切分）")
@click.option("--end", required=True, help="測試迄日 YYYY-MM-DD")
@click.option("--rates-season", type=int, required=True,
              help="事件率用的『前一季』年份（零洩漏 point-in-time 近似）")
@click.option("--n-sims", default=3000, type=int)
@click.option("--max-games", default=0, type=int, help="限制場數（0=全部；除錯用）")
@click.option("--with-sim/--no-sim", default=False,
              help="是否也跑 event-sim（慢；已知輸 NB，預設關）")
def mlb_backtest_sim(data_path, start, end, rates_season, n_sims, max_games, with_sim):
    """Phase 3：把打者資訊接進 NB（NB+打線）vs 現行 NB 的樣本外比對，選配 event-sim。

    流程：以前一季打者/投手事件率為 point-in-time 近似 → 抓 [start,end] 每場實際打序
    與賽果 → NB 只用 start 之前的賽果訓練（walk-forward）→ 對同一批賽果比較
    「NB」「NB+打線（今日打序相對隊平均微調 λ）」的錢線/大小 log-loss。
    只有 NB+打線 vs NB 涵蓋數相同、可直接比；--with-sim 另加 event-sim（覆蓋不同）。
    """
    import pandas as pd

    from . import mlb, mlb_sim
    # 1) NB 模型：只用切分日之前的賽果訓練（無前視）。
    #    鏡射生產配置：強度=全史+半衰期 365；球場係數/離散度=近 2 季（FINDINGS）
    df = mlb.load_with_history(data_path)
    cut = pd.Timestamp(start)
    train = df[df["date"] < cut]
    if train.empty:
        raise click.ClickException("切分日之前無訓練資料。")
    model = dc.fit(train, half_life_days=365, max_goals=20, rho_init=0.0,
                   reg=0.3, reference_date=cut)
    struct = train[train["date"] >= cut - pd.Timedelta(days=731)]
    disp = mlb.dispersion_from_df(struct)
    pf_map = mlb.park_factors(struct) if len(struct) else {}
    click.echo(f"[bt] NB 訓練 {len(train)} 場（<{start}，hl=365）｜"
               f"結構參數近 2 季 {len(struct)} 場｜k={disp:.2f}" if disp else "[bt] 無過度離散")

    # 2) 前一季事件率（point-in-time 近似）
    try:
        bat_lines = mlb_sim.fetch_batting(rates_season)
        pit_lines = mlb_sim.fetch_pitching_events(rates_season)
    except Exception as e:  # noqa: BLE001
        raise click.ClickException(f"抓 {rates_season} 季事件率失敗（沙箱擋 statsapi？）：{e}")
    league = mlb_sim.league_rates(bat_lines)
    bat = {int(r["id"]): mlb_sim.rates_from_batting(r) for r in bat_lines}
    pit = {int(r["id"]): mlb_sim.rates_from_pitching(r) for r in pit_lines}
    click.echo(f"[bt] 前一季事件率：打者 {len(bat)} 位、投手 {len(pit)} 位")

    # 3) 抓測試區間賽果 + 每場打序
    finals = mlb.fetch_games(start, end)
    if max_games:
        finals = finals[:max_games]
    games, no_lineup, no_wx = [], 0, 0
    for gr in finals:
        pk = gr.get("game_pk")
        lu = {"home": [], "away": []}
        wx = None
        if pk:
            try:
                box = mlb_sim.fetch_boxscore(int(pk))   # 一次請求兼取打序與天氣
                lu = mlb_sim.parse_lineups(box)
                wx = mlb_sim.parse_weather(box)
            except Exception:  # noqa: BLE001
                lu, wx = {"home": [], "away": []}, None
        if not lu["home"]:
            no_lineup += 1
        if not wx or (wx.get("temp") is None and not wx.get("wind_speed")):
            no_wx += 1
        games.append(mlb_sim.build_game_record(
            gr, lu, bat, pit, league,
            park=pf_map.get(gr.get("home"), 1.0), weather=wx))
    click.echo(f"[bt] 測試 {len(games)} 場（{no_lineup} 無打序、{no_wx} 無天氣 → 各自略過）")

    # 3b) 抓當季先發投手逐場 game log（依比賽日切，天然 point-in-time、零洩漏）
    season_year = int(start[:4])
    sp_ids = {g["home_sp"] for g in games if g.get("home_sp")} | \
             {g["away_sp"] for g in games if g.get("away_sp")}
    logs = {}
    for pid in sp_ids:
        try:
            logs[int(pid)] = mlb.fetch_pitcher_gamelog(int(pid), season_year)
        except Exception:  # noqa: BLE001
            continue
    # pid→隊（取自 boxscore 實際先發所屬隊）：供隊基準，避免與 NB 隊防守重複計算
    pid_team = {}
    for g in games:
        if g.get("home_sp"):
            pid_team[int(g["home_sp"])] = g["home"]
        if g.get("away_sp"):
            pid_team[int(g["away_sp"])] = g["away"]
    form = mlb.PitcherFormBook(logs, pid_team=pid_team)
    click.echo(f"[bt] 先發 game log：{len(logs)}/{len(sp_ids)} 位投手（隊基準 {len(form.team_base)} 隊）")

    # 3c) 牛棚近況簿：補抓「全聯盟先發（gs>0）」gamelog 提升涵蓋，
    #     由「全隊該日失分 − 先發失分」推算牛棚逐日 RA/9（point-in-time）
    bp = None
    try:
        season_rows = mlb.fetch_pitchers(season_year)
        pid_team_all = {int(r["id"]): r["team"] for r in season_rows if r.get("team")}
        pid_team_all.update(pid_team)         # boxscore 實際先發隊優先
        gs_ids = {int(r["id"]) for r in season_rows if int(r.get("gs") or 0) > 0}
        extra = 0
        for pid in gs_ids - set(logs):
            try:
                logs[int(pid)] = mlb.fetch_pitcher_gamelog(int(pid), season_year)
                extra += 1
            except Exception:  # noqa: BLE001
                continue
        game_rows = df[["date", "home", "away", "home_goals", "away_goals"]] \
            .to_dict("records")
        bp = mlb.BullpenBook(game_rows, logs, pid_team_all)
        cov = sum(len(v) for v in bp.team_games.values())
        click.echo(f"[bt] 牛棚簿：{len(bp.team_games)} 隊、{cov} 個隊日樣本"
                   f"（補抓 {extra} 位先發 gamelog）")
    except Exception as e:  # noqa: BLE001
        click.echo(f"[bt] 牛棚簿建立失敗，略過：{e}")

    # 4) 比對：NB / NB+季投手 / NB+近況投手（主角），另列 NB+打線；選配 event-sim
    book = mlb_sim.LineupBook(bat_lines, league)
    preds = {
        "負二項(NB)": mlb_sim.nb_predictor(model, dispersion=disp),
        "NB+季投手": mlb_sim.nb_pitcher_predictor(model, form, halflife=1e9, dispersion=disp),
        "NB+近況投手": mlb_sim.nb_pitcher_predictor(model, form, halflife=4.0, dispersion=disp),
        "NB+打線": mlb_sim.nb_lineup_predictor(model, book, dispersion=disp),
        "NB+天氣": mlb_sim.nb_weather_predictor(model, dispersion=disp),
    }
    if bp is not None:
        preds["NB+牛棚"] = mlb_sim.nb_bullpen_predictor(model, bp, dispersion=disp)
        preds["NB+近況投手+牛棚"] = mlb_sim.nb_pitcher_bullpen_predictor(
            model, form, bp, halflife=4.0, dispersion=disp)
    if with_sim:
        preds["event-sim"] = mlb_sim.sim_predictor(n_sims=n_sims, seed=0)
    res = mlb_sim.compare_backtest(games, preds)
    click.echo("\n" + mlb_sim.format_backtest(res))
    nb = res["負二項(NB)"]["ml"]["logloss"]
    sp_s = res["NB+季投手"]["ml"]["logloss"]
    sp_r = res["NB+近況投手"]["ml"]["logloss"]
    if None not in (nb, sp_s, sp_r):
        click.echo(f"\n錢線 log-loss：NB {nb:.4f}｜NB+季投手 {sp_s:.4f}｜NB+近況投手 {sp_r:.4f}")
        click.echo(f"  季投手是否幫 NB：{'✅ 有' if sp_s < nb else '❌ 無'}"
                   f"（{sp_s - nb:+.4f}）")
        click.echo(f"  近況是否勝季投手：{'✅ 有' if sp_r < sp_s else '❌ 無'}"
                   f"（{sp_r - sp_s:+.4f}）")
    # 天氣是大小盤訊號：比 OU log-loss
    nb_ou = res["負二項(NB)"]["ou"]["logloss"]
    wx_ou = res["NB+天氣"]["ou"]["logloss"]
    if None not in (nb_ou, wx_ou):
        click.echo(f"大小 log-loss：NB {nb_ou:.4f}｜NB+天氣 {wx_ou:.4f} → "
                   f"{'✅ 天氣有幫助' if wx_ou < nb_ou else '❌ 無'}（{wx_ou - nb_ou:+.4f}）")
    # 牛棚是大小盤訊號：決策比較 = NB+近況投手 vs +牛棚 的 OU log-loss/Brier
    if bp is not None and "NB+近況投手+牛棚" in res:
        base_ou = res["NB+近況投手"]["ou"]
        bp_ou = res["NB+近況投手+牛棚"]["ou"]
        if None not in (base_ou["logloss"], bp_ou["logloss"]):
            d_ll = bp_ou["logloss"] - base_ou["logloss"]
            d_br = bp_ou["brier"] - base_ou["brier"]
            click.echo(f"大小 log-loss：NB+近況投手 {base_ou['logloss']:.4f}｜"
                       f"+牛棚 {bp_ou['logloss']:.4f} → "
                       f"{'✅ 牛棚有幫助' if d_ll < 0 else '❌ 無'}（{d_ll:+.4f}）")
            click.echo(f"大小 Brier   ：NB+近況投手 {base_ou['brier']:.4f}｜"
                       f"+牛棚 {bp_ou['brier']:.4f} → "
                       f"{'✅' if d_br < 0 else '❌'}（{d_br:+.4f}）")


@mlb_group.command("weather-probe")
@click.option("--date", default=None, help="日期 YYYY-MM-DD（預設今天美東）")
def mlb_weather_probe(date):
    """診斷：對今日各場探測 openweather 天氣抓取結果（需 OPENWEATHER_KEY，跑在部署環境）。"""
    from . import mlb, mlb_weather
    date = date or mlb.us_today()
    try:
        games = mlb.fetch_today(date)
    except Exception as e:  # noqa: BLE001
        raise click.ClickException(f"抓賽程失敗（沙箱擋 statsapi？）：{e}")
    if not games:
        click.echo(f"{date} 無比賽。")
        return
    for g in games:
        r = mlb_weather.probe(g["home"], g.get("game_date_iso"))
        click.echo(f"[wx-probe] {mlb.zh_mlb(g['away'])} @ {mlb.zh_mlb(g['home'])}｜{r}")


@cli.command("odds-check")
def odds_check():
    """診斷 the-odds-api 金鑰：打免費的 /sports（0 額度），印狀態+剩餘/已用用量。

    200 = 金鑰有效（若 /odds 仍 401 → 額度用完）；401 = 金鑰無效/貼錯。
    x-requests-remaining=0 → 額度用盡（額度綁帳號，換金鑰不重置，需等月重置或換帳號）。
    """
    import os

    import requests
    key = os.environ.get("ODDS_API_KEY")
    if not key:
        click.echo("[odds-check] 未設 ODDS_API_KEY")
        return
    try:
        r = requests.get("https://api.the-odds-api.com/v4/sports",
                         params={"apiKey": key}, timeout=20)
    except Exception as e:  # noqa: BLE001
        click.echo(f"[odds-check] 連線錯誤：{e}")
        return
    rem = r.headers.get("x-requests-remaining")
    used = r.headers.get("x-requests-used")
    verdict = ("金鑰有效" if r.status_code == 200 else
               "金鑰無效/貼錯" if r.status_code == 401 else f"HTTP {r.status_code}")
    click.echo(f"[odds-check] HTTP {r.status_code}（{verdict}）｜剩餘 {rem}｜已用 {used}"
               f"｜body：{r.text[:160]}")


@cli.command("serve")
@click.option("--model", "model_path", default="models/intl.pkl")
@click.option("--history", "history_path", default="data/intl.csv", help="國際賽歷史（狀態/H2H）")
@click.option("--schedule", default="data/wc2026.json", help="世界盃賽程 JSON（用其 48 隊當選單）")
@click.option("--port", default=8000, type=int, envvar="PORT", help="埠（雲端讀 $PORT）")
@click.option("--host", default="0.0.0.0")
@click.option("--auto-prepare", is_flag=True, default=False,
              help="缺模型/資料時自動下載國際賽+訓練+抓賽程（雲端部署用）")
def serve(model_path, history_path, schedule, port, host, auto_prepare):
    """啟動互動分析網頁：下拉選隊伍/陣型/缺陣、輸入或自動抓盤口讓球線。"""
    import os
    from . import webapp

    def _model_ok(p):
        if not os.path.exists(p):
            return False
        try:
            dc.DixonColesModel.load(p)
            return True
        except Exception:  # noqa: BLE001 - 版本不相容等，改為重建
            return False

    if auto_prepare and not _model_ok(model_path):
        from pathlib import Path
        if not os.path.exists(history_path):
            click.echo("[serve] 下載國際賽資料…")
            from .intl import data as intl
            df = intl.fetch_international(out_path=None, since="2010-01-01")
            df, _ = intl.compute_elo(df)
            Path(history_path).parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(history_path, index=False)
        click.echo("[serve] 訓練模型（首次啟動，約 30–60 秒）…")
        df2 = loader.load_csv(history_path)
        model = dc.fit(df2, half_life_days=540, use_elo=True, reg=0.5)
        Path(model_path).parent.mkdir(parents=True, exist_ok=True)
        model.save(model_path)
    if schedule and not os.path.exists(schedule):
        schedule = None
    webapp.serve(model_path, history_path=history_path, schedule_path=schedule,
                 host=host, port=port)


@cli.command("fetch-wc")
@click.option("--out", default="data/wc2026.json", help="賽程 JSON 輸出路徑")
def fetch_wc(out):
    """下載最新 2026 世界盃賽程與已踢比分（openfootball）。"""
    from . import worldcup as wc
    p = wc.fetch_schedule(out)
    _, matches, _ = wc.parse_wc_json(p)
    played = sum(1 for m in matches if m.played)
    click.echo(f"[ok] 已存賽程到 {p}（{len(matches)} 場，已踢 {played} 場）")


@cli.command("worldcup")
@click.option("--model", "model_path", required=True)
@click.option("--schedule", required=True, help="世界盃賽程 JSON（openfootball 格式）")
@click.option("--n-sims", default=10000, type=int)
@click.option("--html", "html_out", default="out/worldcup.html", help="輸出網站首頁 HTML")
@click.option("--title", default="2026 世界盃預測", help="頁面標題")
@click.pass_context
def worldcup(ctx, model_path, schedule, n_sims, html_out, title):
    """整屆世界盃模擬 + 單頁首頁（小組賽程預測 + 晉級/奪冠機率）。"""
    from . import report, worldcup as wc
    from .i18n import zh
    model = dc.DixonColesModel.load(model_path)
    click.echo(f"[wc] 模擬整屆 {n_sims:,} 次…")
    result = wc.simulate_worldcup(model, schedule, n_sims=n_sims)
    _, matches, _ = wc.parse_wc_json(schedule)
    champ = sorted(result.champion.items(), key=lambda x: x[1], reverse=True)[:8]
    click.echo("奪冠機率前八：" + "  ".join(f"{zh(t)} {p:.1%}" for t, p in champ))
    report.write_worldcup_html(result, model, matches, html_out, title)
    click.echo(f"[ok] 已輸出網站首頁：{html_out}")


@cli.command("wc-site")
@click.option("--model", "model_path", required=True)
@click.option("--schedule", required=True, help="世界盃賽程 JSON")
@click.option("--history", "history_path", default=None, help="國際賽歷史（狀態/H2H）")
@click.option("--outdir", default="out/wc", help="網站輸出目錄")
@click.option("--n-sims", default=20000, type=int, help="整屆模擬次數")
@click.option("--match-sims", default=20000, type=int, help="每場分析模擬次數")
@click.option("--title", default="2026 世界盃預測", help="標題")
@click.option("--use-injuries", is_flag=True, default=False,
              help="抓 api-football 傷停納入（需環境變數 API_FOOTBALL_KEY；失敗則略過）")
@click.option("--wc-season", default=2026, type=int, help="api-football 賽季")
@click.option("--wc-league", default=1, type=int, help="api-football 賽事 id（世界盃=1）")
@click.option("--ledger", default=None, help="戰績帳本 CSV（記錄推薦+結算+首頁顯示）")
@click.pass_context
def wc_site(ctx, model_path, schedule, history_path, outdir, n_sims, match_sims, title,
            use_injuries, wc_season, wc_league, ledger):
    """產生整屆世界盃多頁網站：首頁 + 每場可點進的單場分析頁。"""
    from . import report, worldcup as wc
    from .i18n import zh
    model = dc.DixonColesModel.load(model_path)
    hist = loader.load_csv(history_path) if history_path else None
    click.echo(f"[wc] 模擬整屆 {n_sims:,} 次…")
    result = wc.simulate_worldcup(model, schedule, n_sims=n_sims)
    _, matches, _ = wc.parse_wc_json(schedule)

    track_text = None
    odds_index = None
    if ledger:
        from . import tracker
        try:  # 有 ODDS_API_KEY 才抓真實盤口 → 盤口讓球線 + +EV 推薦 + ROI/CLV
            from .live.providers import fetch_wc_odds
            odds_index = fetch_wc_odds(matches)
            click.echo(f"[wc-site] 已抓盤口：{len(odds_index)} 場有 +EV 候選")
        except Exception as e:  # noqa: BLE001
            click.echo(f"[wc-site] 無真實盤口（改記勝率自我校驗）：{e}")
        tracker.backfill_played(matches, model, ledger)  # 整屆已踢補記（勝率回顧）
        track_text = tracker.prepare(matches, model, ledger, odds_index=odds_index).text()

    injury_counts = None
    if use_injuries:
        # 失敗安全：抓不到傷停就略過，網站照常產生
        try:
            from . import context
            raw = context.fetch_league_injuries(wc_league, wc_season)
            injury_counts = context.map_injury_counts(raw, list(model.teams))
            click.echo(f"[wc] 傷停：{sum(injury_counts.values())} 人次，"
                       f"涵蓋 {len(injury_counts)} 隊")
        except Exception as e:  # noqa: BLE001
            click.echo(f"[warn] 取傷停失敗，略過（網站照常產生）：{e}")
            injury_counts = None

    # MLB 分頁（有 models/mlb.pkl 才會有內容；失敗寫引導頁）
    mlb_html = None
    try:
        from . import mlb as mlbmod
        mlb_html = mlbmod.build_site_page()
        click.echo("[wc-site] MLB 分頁已產生")
    except Exception as e:  # noqa: BLE001
        click.echo(f"[wc-site] MLB 分頁略過：{e}")

    # NBA 分頁（有 models/nba.pkl 才會有內容；失敗/休賽季寫引導頁）
    nba_html = None
    try:
        from . import nba as nbamod
        nba_html = nbamod.build_site_page()
        click.echo("[wc-site] NBA 分頁已產生")
    except Exception as e:  # noqa: BLE001
        click.echo(f"[wc-site] NBA 分頁略過：{e}")

    click.echo("[wc] 產生首頁與各場分析頁…")
    out, n = report.write_worldcup_site(result, model, matches, outdir,
                                        history=hist, title=title, n_sims=match_sims,
                                        injury_counts=injury_counts, track_text=track_text,
                                        ledger_path=ledger, odds_index=odds_index,
                                        mlb_html=mlb_html, nba_html=nba_html)
    champ = sorted(result.champion.items(), key=lambda x: x[1], reverse=True)[:5]
    click.echo("奪冠機率前五：" + "  ".join(f"{zh(t)} {p:.1%}" for t, p in champ))
    click.echo(f"[ok] 網站已輸出到 {out}/（首頁 index.html，{n} 場分析頁）")


@cli.command("simulate-season")
@click.option("--model", "model_path", required=True)
@click.option("--teams", default=None,
              help="參賽隊伍 CSV（含 team 欄）；省略則用 standings/fixtures 推斷或全模型隊伍")
@click.option("--fixtures", default=None,
              help="剩餘賽程 CSV（home, away）；省略則生成雙循環")
@click.option("--standings", "standings_path", default=None,
              help="目前積分榜 CSV（team, points, gf, ga）；或用 --played 從賽果推算")
@click.option("--played", "played_path", default=None,
              help="本季已踢賽果 CSV（內部格式），自動推算目前積分榜與剩餘賽程")
@click.option("--n-sims", default=10000, type=int)
@click.option("--relegation", default=3, type=int, help="降級名額")
@click.option("--html", "html_out", default=None, help="輸出 HTML 表路徑")
@click.option("--title", default="整季模擬", help="頁面標題")
@click.pass_context
def simulate_season(ctx, model_path, teams, fixtures, standings_path, played_path,
                    n_sims, relegation, html_out, title):
    """整季蒙地卡羅模擬：奪冠/前四/降級機率與預期積分。"""
    from . import report, season
    model = dc.DixonColesModel.load(model_path)

    start_standings = None
    remaining = None
    team_list = None

    if played_path:
        played = loader.load_csv(played_path)
        start_standings = season.standings_from_matches(played)
        team_list = sorted(start_standings.keys())
        # 剩餘賽程 = 雙循環中尚未踢的對戰
        played_pairs = {(str(r["home"]), str(r["away"])) for _, r in played.iterrows()}
        remaining = [(h, a) for (h, a) in season.round_robin_fixtures(team_list)
                     if (h, a) not in played_pairs]
        click.echo(f"[season] 由 {len(played)} 場已踢推算積分榜，剩餘 {len(remaining)} 場")

    if teams:
        team_list = pd.read_csv(teams)["team"].astype(str).tolist()
    if standings_path:
        st = pd.read_csv(standings_path)
        start_standings = {}
        for _, r in st.iterrows():
            start_standings[str(r["team"])] = season.TeamStanding(
                team=str(r["team"]), points=int(r.get("points", 0)),
                gf=int(r.get("gf", 0)), ga=int(r.get("ga", 0)))
        team_list = team_list or list(start_standings.keys())
    if fixtures:
        fx = pd.read_csv(fixtures)
        remaining = [(str(r["home"]), str(r["away"])) for _, r in fx.iterrows()]
        team_list = team_list or sorted({t for pair in remaining for t in pair})

    if not team_list:
        team_list = list(model.teams)

    click.echo(f"[season] {len(team_list)} 隊 × {n_sims:,} 次模擬…")
    sim = season.simulate_season(
        model, team_list, fixtures=remaining, start_standings=start_standings,
        n_sims=n_sims, relegation_spots=relegation)
    click.echo(report.render_season_console(sim))
    if html_out:
        report.write_season_html(sim, html_out, title)
        click.echo(f"[ok] 已輸出 HTML：{html_out}")


@cli.command("backtest")
@click.option("--data", "data_path", required=True)
@click.option("--half-life", default=None, type=float)
@click.option("--edge", default=None, type=float, help="最低 edge 門檻")
@click.option("--kelly", default=None, type=float, help="分數凱利係數")
@click.option("--use-elo", is_flag=True, default=False, help="把賽前 Elo 當特徵")
@click.option("--refit-every", default=20, type=int)
@click.option("--export", default=None, help="把每筆下注匯出成 CSV")
@click.pass_context
def backtest(ctx, data_path, half_life, edge, kelly, use_elo, refit_every, export):
    from .backtest import engine
    cfg: Config = ctx.obj["cfg"]
    if half_life is not None:
        cfg.model.half_life_days = half_life
    if edge is not None:
        cfg.value.min_edge = edge
    if kelly is not None:
        cfg.staking.kelly_fraction = kelly
    if use_elo:
        cfg.model.use_elo = True

    df = loader.load_csv(data_path)
    click.echo(f"[backtest] {len(df)} 場，half_life={cfg.model.half_life_days}天 "
               f"edge>={cfg.value.min_edge} kelly={cfg.staking.kelly_fraction}…")
    res = engine.run(df, cfg, refit_every=refit_every)
    click.echo(res.summary())
    if export and res.bets:
        import pandas as pd
        pd.DataFrame([b.__dict__ for b in res.bets]).to_csv(export, index=False)
        click.echo(f"[ok] 已匯出 {len(res.bets)} 筆下注到 {export}")


@cli.command("tune")
@click.option("--data", "data_path", required=True)
@click.option("--refit-every", default=40, type=int)
@click.option("--min-train", default=300, type=int)
@click.option("--save-config", default=None, help="把最佳組合寫成 YAML 設定檔")
@click.pass_context
def tune(ctx, data_path, refit_every, min_train, save_config):
    """自動調超參數：用樣本外 log-loss 選最佳 half_life/reg/elo/xg。"""
    from . import tuning
    import yaml
    cfg: Config = ctx.obj["cfg"]
    df = loader.load_csv(data_path)
    click.echo(f"[tune] {len(df)} 場，網格搜尋中（每組做一次 walk-forward）…")
    result = tuning.tune(df, base_cfg=cfg, refit_every=refit_every,
                         min_train_matches=min_train)
    click.echo(result.summary())
    if save_config:
        tuning.apply_best(cfg, result)
        with open(save_config, "w", encoding="utf-8") as f:
            yaml.safe_dump(cfg.to_dict(), f, allow_unicode=True, sort_keys=False)
        click.echo(f"[ok] 最佳設定已存：{save_config}")


@cli.command("check-injuries")
@click.option("--league", default=1, type=int, help="賽事 id（世界盃通常是 1）")
@click.option("--season", default=2026, type=int)
def check_injuries(league, season):
    """診斷 api-football：印出方案狀態與某賽事的傷停筆數（需 API_FOOTBALL_KEY）。"""
    import os
    import requests
    key = os.environ.get("API_FOOTBALL_KEY")
    if not key:
        raise click.ClickException("缺少環境變數 API_FOOTBALL_KEY")
    base = "https://v3.football.api-sports.io"
    h = {"x-apisports-key": key}
    # 方案/額度狀態
    try:
        st = requests.get(f"{base}/status", headers=h, timeout=20).json()
        resp = st.get("response", {})
        sub = resp.get("subscription", {})
        click.echo(f"方案：{sub.get('plan')}　到期：{sub.get('end')}　"
                   f"今日用量：{(resp.get('requests') or {}).get('current')}/"
                   f"{(resp.get('requests') or {}).get('limit_day')}")
    except Exception as e:  # noqa: BLE001
        click.echo(f"[warn] 取 status 失敗：{e}")
    # 傷停筆數
    try:
        r = requests.get(f"{base}/injuries", params={"league": league, "season": season},
                         headers=h, timeout=20).json()
        click.echo(f"injuries(league={league}, season={season})："
                   f"results={r.get('results')}　errors={r.get('errors')}")
        for item in (r.get("response") or [])[:3]:
            click.echo(f"  範例：{(item.get('team') or {}).get('name')} - "
                       f"{(item.get('player') or {}).get('name')}")
    except Exception as e:  # noqa: BLE001
        click.echo(f"[warn] 取 injuries 失敗：{e}")


@cli.command("check-lineups")
@click.option("--date", default=None, help="日期 YYYY-MM-DD（預設今天 UTC）")
@click.option("--league", default=1, type=int, help="賽事 id（世界盃通常是 1）")
@click.option("--season", default=2026, type=int)
def check_lineups(date, league, season):
    """診斷 api-football 是否涵蓋世界盃先發：印出某日各場 formation 與先發人數。"""
    import datetime as _dt
    from . import context
    date = date or _dt.datetime.utcnow().strftime("%Y-%m-%d")
    try:
        fixtures = context.fetch_fixtures_by_date(date, league=league, season=season)
    except Exception as e:  # noqa: BLE001
        raise click.ClickException(str(e))
    click.echo(f"{date}　league={league} season={season}：{len(fixtures)} 場")
    if not fixtures:
        click.echo("（當日無比賽，或方案不涵蓋此賽事）")
        return
    got = 0
    for (h, a), fid in fixtures.items():
        try:
            lu = context.fetch_lineups(fid)
        except Exception as e:  # noqa: BLE001
            click.echo(f"  {h} vs {a}（fixture {fid}）：取先發失敗 {e}")
            continue
        if lu:
            got += 1
            desc = "　".join(f"{t}:{d.get('formation') or '?'}"
                             f"({len(d.get('starters') or [])}人)" for t, d in lu.items())
            click.echo(f"  ✅ {h} vs {a}：{desc}")
        else:
            click.echo(f"  ⏳ {h} vs {a}：先發未公布（通常開賽前 ~40 分才有）")
    click.echo(f"\n結論：{got}/{len(fixtures)} 場已取得先發。"
               f"{'先發資料可用 → 可建自動管線。' if got else '尚無先發（可能還太早、或方案不涵蓋世界盃）。'}")


@cli.command("eval-intl")
@click.option("--data", "data_path", default="data/intl.csv", help="國際賽 CSV（含 Elo/neutral）")
@click.option("--since", default="2018-01-01", help="只對此日期後的比賽計分")
@click.option("--half-life", default=540.0, type=float)
@click.option("--reg", default=2.0, type=float)
@click.option("--use-elo/--no-elo", default=True)
@click.option("--model-kind", default="dc", type=click.Choice(["dc", "elo_poisson"]),
              help="dc=每隊攻防；elo_poisson=純 Elo 驅動(公開方法論變體)")
@click.option("--refit-every", default=200, type=int)
@click.pass_context
def eval_intl(ctx, data_path, since, half_life, reg, use_elo, model_kind, refit_every):
    """國際賽（世界盃）模型校準：無賠率，基準為均勻(1/3)。"""
    from . import evaluation
    cfg: Config = ctx.obj["cfg"]
    cfg.model.half_life_days = half_life
    cfg.model.reg = reg
    cfg.model.use_elo = use_elo
    df = loader.load_csv(data_path)
    click.echo(f"[eval-intl] {len(df)} 場，計分自 {since}（model={model_kind}, "
               f"half_life={half_life}, reg={reg}, elo={use_elo}）…")
    res = evaluation.run_intl(df, cfg, test_since=since, refit_every=refit_every,
                              model_kind=model_kind)
    click.echo(res.summary(cfg))


@cli.command("evaluate")
@click.option("--data", "data_path", required=True)
@click.option("--half-life", default=None, type=float)
@click.option("--xg-weight", default=None, type=float)
@click.option("--use-elo", is_flag=True, default=False, help="把賽前 Elo 當特徵")
@click.option("--reg", default=None, type=float, help="L2 正則化強度")
@click.option("--refit-every", default=20, type=int)
@click.pass_context
def evaluate(ctx, data_path, half_life, xg_weight, use_elo, reg, refit_every):
    """模型校準（Brier/LogLoss/可靠度）與 CLV 分析。"""
    from . import evaluation
    cfg: Config = ctx.obj["cfg"]
    if half_life is not None:
        cfg.model.half_life_days = half_life
    if xg_weight is not None:
        cfg.model.xg_weight = xg_weight
    if use_elo:
        cfg.model.use_elo = True
    if reg is not None:
        cfg.model.reg = reg
    df = loader.load_csv(data_path)
    click.echo(f"[evaluate] {len(df)} 場 walk-forward 校準中…")
    res = evaluation.run(df, cfg, refit_every=refit_every)
    click.echo(res.summary(cfg))


@cli.command("live")
@click.option("--model", "model_path", required=True)
@click.option("--feed", default="simulated",
              help="盤口來源：simulated（內建模擬）/ theoddsapi（真實，需 ODDS_API_KEY）")
@click.option("--sport", default="soccer_epl", help="theoddsapi 聯賽鍵，如 soccer_epl")
@click.option("--bookmaker", default=None, help="theoddsapi 指定博彩商，如 pinnacle")
@click.option("--home", default="HomeTeam", help="模擬用主隊名（需在模型中）")
@click.option("--away", default="AwayTeam", help="模擬用客隊名（需在模型中）")
@click.option("--max-polls", default=None, type=int)
@click.option("--sleep", "sleep_s", default=None, type=float, help="輪詢間隔秒（模擬可設 0）")
@click.pass_context
def live(ctx, model_path, feed, sport, bookmaker, home, away, max_polls, sleep_s):
    from .live.feed import SimulatedFeed
    from .live.monitor import LiveMonitor
    cfg: Config = ctx.obj["cfg"]
    model = dc.DixonColesModel.load(model_path)

    if feed == "simulated":
        # 用模型對這兩隊的預期進球當作模擬「真實」進球率，較貼近實際。
        if home in model.attack and away in model.attack:
            lam, mu = model.expected_goals(home, away)
        else:
            lam, mu = 1.5, 1.1
            click.echo(f"[warn] {home}/{away} 不在模型中，用預設進球率模擬。")
        src = SimulatedFeed(home, away, true_lambda=lam, true_mu=mu)
        sleep_default = 0.0
    elif feed in ("theoddsapi", "the-odds-api"):
        from .live.providers import TheOddsApiFeed
        src = TheOddsApiFeed(sport=sport, bookmaker=bookmaker, in_play=True)
        sleep_default = cfg.live.poll_interval_s
        click.echo(f"[live] 已連 The Odds API（{sport}），開始走地盯盤。")
    else:
        raise click.ClickException(
            f"未知 feed '{feed}'。支援 simulated / theoddsapi。")

    monitor = LiveMonitor(model, cfg)
    monitor.run(src, max_polls=max_polls,
                sleep_s=sleep_s if sleep_s is not None else sleep_default)


def _stem(path: str) -> str:
    from pathlib import Path
    return Path(path).stem


if __name__ == "__main__":
    cli()
