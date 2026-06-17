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
@click.pass_context
def train(ctx, data_path, out, half_life, xg_weight, use_elo):
    cfg: Config = ctx.obj["cfg"]
    if half_life is not None:
        cfg.model.half_life_days = half_life
    if xg_weight is not None:
        cfg.model.xg_weight = xg_weight
    if use_elo:
        cfg.model.use_elo = True
    df = loader.load_csv(data_path)
    click.echo(f"[train] 載入 {len(df)} 場比賽，開始擬合"
               f"（half_life={cfg.model.half_life_days}天, xg_weight={cfg.model.xg_weight}）…")
    model = dc.fit(df, half_life_days=cfg.model.half_life_days,
                   max_goals=cfg.model.max_goals, rho_init=cfg.model.rho_init,
                   xg_weight=cfg.model.xg_weight, use_elo=cfg.model.use_elo, verbose=True)
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


@cli.command("evaluate")
@click.option("--data", "data_path", required=True)
@click.option("--half-life", default=None, type=float)
@click.option("--xg-weight", default=None, type=float)
@click.option("--use-elo", is_flag=True, default=False, help="把賽前 Elo 當特徵")
@click.option("--refit-every", default=20, type=int)
@click.pass_context
def evaluate(ctx, data_path, half_life, xg_weight, use_elo, refit_every):
    """模型校準（Brier/LogLoss/可靠度）與 CLV 分析。"""
    from . import evaluation
    cfg: Config = ctx.obj["cfg"]
    if half_life is not None:
        cfg.model.half_life_days = half_life
    if xg_weight is not None:
        cfg.model.xg_weight = xg_weight
    if use_elo:
        cfg.model.use_elo = True
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
