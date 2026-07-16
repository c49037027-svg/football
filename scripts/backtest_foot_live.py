"""足球走地驗證重現腳本：英超半場邊界 walk-forward（docs/FINDINGS.md「足球走地」節）。

重現階梯表：訓練 <2022-07 的 E0 比賽擬合 Dixon-Coles（凍結），在 2022-07~2025-06
每場的 45' 半場比分狀態上算終場 1X2 三向 log-loss，逐層加上走地修正：
  純時間衰減 → +下半場升溫 ×1.11 → +比分狀態效應 → +溫度縮放 τ=0.95

資料：xgabora 彙整檔（含 HTHome/HTAway 半場比分），執行期下載、不入庫。
用法：python scripts/backtest_foot_live.py [--csv 本地Matches.csv] [--half-life 180]

已知限制（誠實揭露，詳見 FINDINGS）：
  - 測試期中訓練期未出現過的球隊（升級新軍，如 Luton/Nott'm Forest）因模型
    無評分而被剔除——本腳本會明確列出剔除場數與隊名，不再無聲吞掉。
  - 單一 train/test 切分；三層修正的採納依據是測試期 LL，屬 test-set reuse，
    數字應視為樂觀估計。τ 層增益（~0.001）在 n=990 下無法與噪音區分。
"""
from __future__ import annotations

import argparse
import sys
from io import StringIO
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from footy.data.loader import GITHUB_CONSOLIDATED_URL, normalize_consolidated  # noqa: E402
from footy.foot_live import LEAD_DAMP, TRAIL_BOOST, _rate_scale  # noqa: E402
from footy.live import inplay  # noqa: E402
from footy.models import dixon_coles as dc  # noqa: E402

TRAIN_END = "2022-07-01"
TEST_END = "2025-06-30"


def load_raw(csv: str | None) -> pd.DataFrame:
    if csv:
        raw = pd.read_csv(csv, low_memory=False)
    else:
        import requests
        r = requests.get(GITHUB_CONSOLIDATED_URL,
                         headers={"User-Agent": "Mozilla/5.0"}, timeout=120)
        r.raise_for_status()
        raw = pd.read_csv(StringIO(r.text), low_memory=False)
    raw = raw[raw["Division"] == "E0"].copy()
    raw["MatchDate"] = pd.to_datetime(raw["MatchDate"], errors="coerce")
    return raw.dropna(subset=["MatchDate", "FTHome", "FTAway"])


def one_x_two(mat: np.ndarray, shift_h: int = 0, shift_a: int = 0) -> np.ndarray:
    """比分矩陣 →（主/和/客）機率；shift = 已進球數（矩陣是剩餘進球）。"""
    n = mat.shape[0]
    i, j = np.indices((n, n))
    diff = (shift_h + i) - (shift_a + j)
    p = np.array([mat[diff > 0].sum(), mat[diff == 0].sum(), mat[diff < 0].sum()])
    return p / p.sum()


def live_1x2(model, home, away, hg, ag, scale, use_state, tau) -> np.ndarray:
    hf = af = 1.0
    if use_state:
        if hg > ag:
            af, hf = TRAIL_BOOST, LEAD_DAMP
        elif ag > hg:
            hf, af = TRAIL_BOOST, LEAD_DAMP
    rem = inplay.remaining_score_matrix(model, home, away, 45, scale,
                                        home_factor=hf, away_factor=af)
    p = one_x_two(rem, hg, ag)
    p = np.maximum(p, 1e-12) ** tau
    return p / p.sum()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=None, help="本地 Matches.csv（省下載）")
    ap.add_argument("--half-life", type=float, default=180.0)
    args = ap.parse_args()

    raw = load_raw(args.csv)
    train_raw = raw[raw["MatchDate"] < TRAIN_END]
    test_raw = raw[(raw["MatchDate"] >= TRAIN_END) & (raw["MatchDate"] <= TEST_END)]
    test_raw = test_raw.dropna(subset=["HTHome", "HTAway"])

    train = normalize_consolidated(train_raw)
    model = dc.fit(train, half_life_days=args.half_life)
    print(f"訓練 {len(train)} 場（<{TRAIN_END}，hl={args.half_life:g} 天）；"
          f"測試期原始 {len(test_raw)} 場")

    in_model = test_raw["HomeTeam"].isin(model.attack) & test_raw["AwayTeam"].isin(model.attack)
    dropped = test_raw[~in_model]
    if len(dropped):
        teams = sorted((set(dropped["HomeTeam"]) | set(dropped["AwayTeam"]))
                       - set(model.attack))
        print(f"⚠️ 剔除 {len(dropped)} 場（{len(dropped) / len(test_raw):.0%}）："
              f"訓練期無評分的球隊 {teams}——這些正是賽前評分最不可靠的比賽，"
              f"結論不能外推到含升級隊的真實比賽池")
    test = test_raw[in_model]
    print(f"評估 {len(test)} 場半場邊界（45'）\n")

    heat = _rate_scale(45)
    configs = [
        ("賽前機率凍結（基準）", None),
        ("走地・純時間衰減", dict(scale=1.0, use_state=False, tau=1.0)),
        (f"＋下半場升溫 ×{heat:g}", dict(scale=heat, use_state=False, tau=1.0)),
        ("＋比分狀態效應", dict(scale=heat, use_state=True, tau=1.0)),
        ("＋溫度縮放 τ=0.95（上線配置）", dict(scale=heat, use_state=True, tau=0.95)),
    ]
    y = np.sign(test["FTHome"].values - test["FTAway"].values)  # 1/0/-1
    idx = np.where(y > 0, 0, np.where(y == 0, 1, 2))
    for name, kw in configs:
        lls = []
        for k, (_, g) in enumerate(test.iterrows()):
            if kw is None:
                p = one_x_two(model.score_matrix(g["HomeTeam"], g["AwayTeam"]))
            else:
                p = live_1x2(model, g["HomeTeam"], g["AwayTeam"],
                             int(g["HTHome"]), int(g["HTAway"]), **kw)
            lls.append(-np.log(max(p[idx[k]], 1e-12)))
        print(f"{name:<28s} LL={np.mean(lls):.4f}")


if __name__ == "__main__":
    main()
