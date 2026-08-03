"""三盤口鑑別力回測：模型的機率到底有沒有預測個別比賽的能力？

除了讓分門檻比較，另對 錢線/大小/讓分 各算 AUC（機率排序能力）與分桶校準。
AUC≈0.5 代表模型只是在複述基礎比率、對個別比賽沒有洞見。

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


def discrimination(p: pd.Series, y: pd.Series, name: str, bins) -> dict:
    """鑑別力：分桶（模型機率 vs 實際發生率）＋ AUC。回摘要 dict。"""
    from scipy import stats
    d = pd.DataFrame({"p": p.values, "y": y.values}).dropna()
    d["bucket"] = pd.cut(d.p, bins)
    print(f"\n=== {name}：模型機率 vs 實際 ===")
    for b, g in d.groupby("bucket", observed=True):
        if len(g) < 25:
            continue
        print(f"  {str(b):<14} n={len(g):4d}　模型 {g.p.mean():.1%}　實際 {g.y.mean():.1%}")
    lo, hi = d[d.p < d.p.median()], d[d.p >= d.p.median()]
    gap = hi.y.mean() - lo.y.mean()
    pos, neg = d[d.y].p, d[~d.y].p
    if len(pos) and len(neg):
        u = stats.mannwhitneyu(pos, neg, alternative="greater")
        auc = u.statistic / (len(pos) * len(neg))
        pv = u.pvalue
    else:
        auc, pv = float("nan"), float("nan")
    verdict = ("✅ 有鑑別力" if auc >= 0.56 else
               "🟡 微弱" if auc >= 0.53 else "❌ 幾乎沒有")
    print(f"  低半 {lo.y.mean():.1%} vs 高半 {hi.y.mean():.1%}（差 {gap:+.1%}）"
          f"　AUC={auc:.3f}　p={pv:.4f}　{verdict}")
    return {"name": name, "n": len(d), "auc": auc, "p": pv, "gap": gap}


def run(days: int, refit_days: int, half_life: float,
        sp_csv: str | None = None, pitchers: str = 'data/mlb_pitchers.csv') -> None:
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
    # 先發投手層（可選）：逐場先發對照表 + 投手評分
    sp_map, book = {}, None
    if sp_csv and Path(sp_csv).exists() and Path(pitchers).exists():
        sp = pd.read_csv(sp_csv)
        sp["date"] = pd.to_datetime(sp["date"]).dt.date
        for _, x in sp.iterrows():
            sp_map[(x["date"], x["home"], x["away"])] = (
                x.get("home_pitcher_id"), x.get("away_pitcher_id"))
        book = mlb.PitcherBook.load_csv(pitchers)
        print(f"啟用先發投手層：{len(sp_map)} 場對照、投手檔 {pitchers}")
    else:
        print("未啟用先發投手層（隊級模型）")
    recs, model, fitted_until = [], None, None
    n_sp_used = [0]
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
            hf = af = 1.0
            if book is not None:
                sp_ids = sp_map.get((d.date(), h, a))
                if sp_ids:
                    hid, aid = sp_ids
                    if pd.notna(hid):
                        hf, _ = book.factor(int(hid))
                    if pd.notna(aid):
                        af, _ = book.factor(int(aid))
                    n_sp_used[0] += 1
            prov = mlb.analyze_game(model, h, a, total_line=8.5, run_line=-1.5,
                                    home_pitcher_factor=hf, away_pitcher_factor=af,
                                    park_factor=pf.get(h, 1.0), dispersion=disp)
            rl = -1.5 if prov.exp_home >= prov.exp_away else 1.5
            m = mlb.analyze_game(model, h, a, total_line=8.5, run_line=rl,
                                 home_pitcher_factor=hf, away_pitcher_factor=af,
                                 park_factor=pf.get(h, 1.0), dispersion=disp)
            # p_plus = 「受讓 +1.5 那一側」過盤機率
            p_plus = m.p_cover_home if rl > 0 else 1.0 - m.p_cover_home
            hg, ag = int(g["home_goals"]), int(g["away_goals"])
            margin, total = hg - ag, hg + ag
            plus_cover = (margin + 1.5 > 0) if rl > 0 else (-margin + 1.5 > 0)
            # 錢線：模型看好側是否獲勝（去掉延長平手不可能，MLB 無和局）
            p_ml = max(m.p_home, m.p_away)
            ml_win = (margin > 0) if m.p_home >= m.p_away else (margin < 0)
            # 大小：用模型自取線（floor+0.5），看模型看好側是否命中
            tl = mlb._half_line(m.exp_home + m.exp_away)
            m2 = mlb.analyze_game(model, h, a, total_line=tl, run_line=rl,
                                  home_pitcher_factor=hf, away_pitcher_factor=af,
                                  park_factor=pf.get(h, 1.0), dispersion=disp)
            p_ou = max(m2.p_over, m2.p_under)
            ou_win = (total > tl) if m2.p_over >= 0.5 else (total < tl)
            recs.append({"p_plus": p_plus, "plus_cover": plus_cover,
                         "p_ml": p_ml, "ml_win": ml_win,
                         "p_ou": p_ou, "ou_win": ou_win})
    r = pd.DataFrame(recs)
    if r.empty:
        print("無可回測樣本")
        return
    if book is not None:
        print(f"實際套用先發係數：{n_sp_used[0]} 場")
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
    print("\n" + "=" * 60)
    print("鑑別力總表（AUC：模型能否把「會發生」的場次排在前面）")
    summ = [
        discrimination(r.p_plus, r.plus_cover, "讓分（+1.5 側過盤）",
                       [0, .55, .58, .60, .62, .65, 1.0]),
        discrimination(r.p_ml, r.ml_win, "錢線（模型看好側獲勝）",
                       [0, .52, .55, .58, .62, .70, 1.0]),
        discrimination(r.p_ou, r.ou_win, "大小（模型看好側命中）",
                       [0, .52, .54, .56, .58, .62, 1.0]),
    ]
    print("\n=== 三盤口對比 ===")
    for x in summ:
        print(f"  {x['name']:<22} n={x['n']:4d}　AUC={x['auc']:.3f}　"
              f"高低半差 {x['gap']:+.1%}　p={x['p']:.4f}")
    print("\n註：損益用固定價假設 +1.5@%.2f／-1.5@%.2f（帳本中位數與無 vig 反推）；"
          "勝率不受價格假設影響。" % (PRICE_PLUS, PRICE_MINUS))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=120, help="測試期天數")
    ap.add_argument("--refit", type=int, default=7, help="每幾天重訓一次")
    ap.add_argument("--half-life", type=float, default=365.0)
    ap.add_argument("--sp-csv", default=None,
                    help="含逐場先發的歷史檔（footy mlb fetch-history-sp 產生）;"
                         "給了就啟用投手層,可比較有無投手的鑑別力差異")
    ap.add_argument("--pitchers", default="data/mlb_pitchers.csv",
                    help="投手評分檔(PitcherBook)")
    a = ap.parse_args()
    run(a.days, a.refit, a.half_life, a.sp_csv, a.pitchers)
