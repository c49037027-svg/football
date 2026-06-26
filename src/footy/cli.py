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
    """AI agents（賽前分析/辯論/風控/賽後檢討/新聞抽取）。需設 GEMINI_API_KEY。"""


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
    """印出目前 LLM 設定；--live 會真的呼叫一次驗證連線。"""
    from .agents import llm
    c = llm.config()
    click.echo(f"base_url：{c['base']}　model：{c['model']}　"
               f"金鑰：{'已設定' if c['key'] else '未設定（agent 會略過）'}")
    if not live:
        return
    if not c["key"]:
        click.echo("[ai-check] 未設金鑰，略過實測。")
        return
    try:
        out = llm.complete("用繁體中文回覆「可用」兩個字。", max_tokens=80, timeout=30)
        status = "PASS" if out else "PASS(但回覆為空，可調高 max_tokens 或換非思考型模型)"
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

    click.echo("[wc] 產生首頁與各場分析頁…")
    out, n = report.write_worldcup_site(result, model, matches, outdir,
                                        history=hist, title=title, n_sims=match_sims,
                                        injury_counts=injury_counts, track_text=track_text,
                                        ledger_path=ledger, odds_index=odds_index)
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
