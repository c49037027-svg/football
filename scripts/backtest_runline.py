"""讓分盤決策門檻回測：現行「選機率過半」vs「門檻 T 才推 +1.5」。

問題：MLB 讓分固定 ±1.5，而「+1.5 過盤」機率結構性地 55~65%（弱隊贏 或
只輸 1 分），所以「選機率過半那邊」幾乎永遠答 +1.5（帳本 97:3）——
預測沒有場次資訊量，且 58.8% 勝率低於該線兩平（~60.8%）。

本回測比較：
  A（現行）：選 max(p) 的一邊
  B(T)     ：p(+1.5) >= T 才推 +1.5，否則推 −1.5（T 為固定常數，非逐場看賠率）

方法：walk-forward，每 refit_days 用「該日之前」的資料重訓 Dixon-Coles，
對測試期每場算 p_cover。不用任何未來資料。無逐場歷史賠率，故損益用
固定價假設（見 PRICE_*，取自帳本實際中位數與無 vig 反推），並報敏感度。

用法：python scripts/backtest_runline.py [--days 120] [--refit 7]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from footy import mlb  # noqa: E402
from footy.models import dixon_coles as dc  # noqa: E402

# 固定價假設（帳本實際：+1.5 中位 1.58；−1.5 樣本少，用無 vig 反推 ~2.35）
PRICE_PLUS, PRICE_MINUS = 1.58, 2.35


def calibration(r: pd.DataFrame) -> None:
    """分桶校準：模型說機率越高，實際過盤率是否真的越高（有無鑑別力）。"""
    r = r.copy()
    r["bucket"] = pd.cut(r.p_plus, [0, .55, .58, .60, .62, .65, 1.0])
    print("\n=== 鑑別力檢查：模型 p(+1.5) 分桶 vs 實際過盤率 ===")
    for b, g in r.groupby("bucket", observed=True):
        if len(g) < 20:
            continue
        print(f"  模型 {str(b):<14} n={len(g):4d}　模型均值 {g.p_plus.mean():.1%}"
              f"　實際過盤 {g.plus_cover.mean():.1%}")
    lo, hi = r[r.p_plus < r.p_plus.median()], r[r.p_plus >= r.p_plus.median()]
    print(f"  低半 vs 高半：{lo.plus_cover.mean():.1%} vs {hi.plus_cover.mean():.1%}"
          f"（差 {hi.plus_cover.mean() - lo.plus_cover.mean():+.1%}）")
    from scipy import stats
    auc = stats.mannwhitneyu(r[r.plus_cover].p_plus, r[~r.plus_cover].p_plus,
                             alternative="greater")
    n1, n0 = r.plus_cover.sum(), (~r.plus_cover).sum()
    print(f"  AUC={auc.statistic / (n1 * n0):.3f}（0.5=無鑑別力）　p={auc.pvalue:.3f}")


def run(days: int, refit_days: int, half_life: float) -> None:
    df = pd.read_csv("data/mlb.csv")
    df["date"] = pd.to_datetime(df["date"])
    df = df.dropna(subset=["home_goals", "away_goals"]).sort_values("date")
    end = df["date"].max()
    start = end - pd.Timedelta(days=days)
    test = df[df["date"] > start]
    print(f"測試期 {start.date()}→{end.date()}　{len(test)} 場　"
          f"（每 {refit_days} 天 refit，half_life={half_life:g}）")

    pf = mlb.park_factors_from_csv("data/mlb.csv")
    disp = mlb.dispersion_from_csv("data/mlb.csv")
    recs, model, fitted_until = [], None, None
    for d, day_games in test.groupby(test["date"].dt.date):
        d = pd.Timestamp(d)
        if model is None or (d - fitted_until).days >= refit_days:
            train = df[df["date"] < d]           # 只用該日之前 → 無前視
            model = dc.fit(train, half_life_days=half_life, max_goals=20,
                           rho_init=0.0, reg=0.3)
            fitted_until = d
        for _, g in day_games.iterrows():
            h, a = g["home"], g["away"]
            if h not in model.attack or a not in model.attack:
                continue
            # 與生產一致：熱門方讓 1.5（先用暫定線取得預期分）
            prov = mlb.analyze_game(model, h, a, total_line=8.5, run_line=-1.5,
                                    park_factor=pf.get(h, 1.0), dispersion=disp)
            rl = -1.5 if prov.exp_home >= prov.exp_away else 1.5
            m = mlb.analyze_game(model, h, a, total_line=8.5, run_line=rl,
                                 park_factor=pf.get(h, 1.0), dispersion=disp)
            # p_plus = 「受讓 +1.5 那一側」過盤機率
            p_plus = m.p_cover_home if rl > 0 else 1.0 - m.p_cover_home
            margin = int(g["home_goals"]) - int(g["away_goals"])
            # +1.5 側是否過盤：主隊受讓→ margin+1.5>0；客隊受讓→ -margin+1.5>0
            plus_cover = (margin + 1.5 > 0) if rl > 0 else (-margin + 1.5 > 0)
            recs.append({"p_plus": p_plus, "plus_cover": plus_cover})
    r = pd.DataFrame(recs)
    if r.empty:
        print("無可回測樣本")
        return
    print(f"可評估 {len(r)} 場　p(+1.5) 分布："
          f"中位 {r.p_plus.median():.1%}　"
          f"範圍 {r.p_plus.min():.1%}~{r.p_plus.max():.1%}　"
          f"實際 +1.5 過盤率 {r.plus_cover.mean():.1%}")

    def evaluate(pick_plus: pd.Series, label: str) -> None:
        """pick_plus=每場是否推 +1.5（False 即推 −1.5）。"""
        win = np.where(pick_plus, r.plus_cover, ~r.plus_cover)
        price = np.where(pick_plus, PRICE_PLUS, PRICE_MINUS)
        pl = np.where(win, price - 1.0, -1.0)
        n_plus = int(pick_plus.sum())
        print(f"{label:<22} n={len(r)}　推+1.5 {n_plus:4d}／推-1.5 {len(r) - n_plus:4d}"
              f"　勝率 {win.mean():.1%}　每注 {pl.mean():+.3f}u　總 {pl.sum():+7.1f}u")

    print("\n=== 策略比較 ===")
    evaluate(r.p_plus >= 0.5, "A 現行(門檻 50%)")
    for t in (0.55, 0.58, 0.60, 0.62, 0.65, 0.70):
        evaluate(r.p_plus >= t, f"B 門檻 {t:.0%}")
    evaluate(pd.Series(False, index=r.index), "全押 -1.5（對照）")
    calibration(r)
    print("\n註：損益用固定價假設 +1.5@%.2f／-1.5@%.2f（帳本中位數與無 vig 反推）；"
          "勝率不受價格假設影響。" % (PRICE_PLUS, PRICE_MINUS))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=120, help="測試期天數")
    ap.add_argument("--refit", type=int, default=7, help="每幾天重訓一次")
    ap.add_argument("--half-life", type=float, default=365.0)
    a = ap.parse_args()
    run(a.days, a.refit, a.half_life)
