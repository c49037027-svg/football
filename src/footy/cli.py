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


@cli.command("train")
@click.option("--data", "data_path", required=True, help="歷史資料 CSV")
@click.option("--out", default=None, help="模型輸出路徑（預設 models/<name>.pkl）")
@click.option("--half-life", default=None, type=float, help="時間衰減半衰期（天）")
@click.pass_context
def train(ctx, data_path, out, half_life):
    cfg: Config = ctx.obj["cfg"]
    if half_life is not None:
        cfg.model.half_life_days = half_life
    df = loader.load_csv(data_path)
    click.echo(f"[train] 載入 {len(df)} 場比賽，開始擬合（half_life={cfg.model.half_life_days}天）…")
    model = dc.fit(df, half_life_days=cfg.model.half_life_days,
                   max_goals=cfg.model.max_goals, rho_init=cfg.model.rho_init,
                   verbose=True)
    out = out or f"models/{_stem(data_path)}.pkl"
    model.save(out)
    click.echo(f"[ok] 模型已存：{out}（{len(model.teams)} 隊，主場優勢={model.home_adv:.3f}，rho={model.rho:.3f}）")


@cli.command("scan-prematch")
@click.option("--model", "model_path", required=True)
@click.option("--fixtures", required=True, help="即將開賽盤口 CSV")
@click.pass_context
def scan_prematch(ctx, model_path, fixtures):
    from . import prematch
    cfg: Config = ctx.obj["cfg"]
    model = dc.DixonColesModel.load(model_path)
    fx = pd.read_csv(fixtures)
    rows = prematch.scan(model, fx, cfg)
    if not rows:
        click.echo("沒有找到符合門檻的 value 下注。")
        return
    click.echo(f"\n找到 {len(rows)} 個 value 下注建議：\n")
    for r in rows:
        click.echo(
            f"  {r['match']:<28} {r['market']:>8} {r['selection']:<6} @ {r['odds']:>5} "
            f"| edge={r['edge']:+.1%} EV={r['ev']:+.3f} 下注={r['suggested_stake']:.2f} ({r['risk_note']})"
        )


@cli.command("backtest")
@click.option("--data", "data_path", required=True)
@click.option("--half-life", default=None, type=float)
@click.option("--edge", default=None, type=float, help="最低 edge 門檻")
@click.option("--kelly", default=None, type=float, help="分數凱利係數")
@click.option("--refit-every", default=20, type=int)
@click.option("--export", default=None, help="把每筆下注匯出成 CSV")
@click.pass_context
def backtest(ctx, data_path, half_life, edge, kelly, refit_every, export):
    from .backtest import engine
    cfg: Config = ctx.obj["cfg"]
    if half_life is not None:
        cfg.model.half_life_days = half_life
    if edge is not None:
        cfg.value.min_edge = edge
    if kelly is not None:
        cfg.staking.kelly_fraction = kelly

    df = loader.load_csv(data_path)
    click.echo(f"[backtest] {len(df)} 場，half_life={cfg.model.half_life_days}天 "
               f"edge>={cfg.value.min_edge} kelly={cfg.staking.kelly_fraction}…")
    res = engine.run(df, cfg, refit_every=refit_every)
    click.echo(res.summary())
    if export and res.bets:
        import pandas as pd
        pd.DataFrame([b.__dict__ for b in res.bets]).to_csv(export, index=False)
        click.echo(f"[ok] 已匯出 {len(res.bets)} 筆下注到 {export}")


@cli.command("live")
@click.option("--model", "model_path", required=True)
@click.option("--feed", default="simulated", help="盤口來源：simulated（內建模擬）")
@click.option("--home", default="HomeTeam", help="模擬用主隊名（需在模型中）")
@click.option("--away", default="AwayTeam", help="模擬用客隊名（需在模型中）")
@click.option("--max-polls", default=None, type=int)
@click.option("--sleep", "sleep_s", default=None, type=float, help="輪詢間隔秒（模擬可設 0）")
@click.pass_context
def live(ctx, model_path, feed, home, away, max_polls, sleep_s):
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
    else:
        raise click.ClickException(
            f"未知 feed '{feed}'。內建只有 simulated；接真實數據請實作 OddsFeed 子類別。")

    monitor = LiveMonitor(model, cfg)
    monitor.run(src, max_polls=max_polls, sleep_s=sleep_s if sleep_s is not None else 0.0)


def _stem(path: str) -> str:
    from pathlib import Path
    return Path(path).stem


if __name__ == "__main__":
    cli()
