"""超參數自動調校：用 walk-forward 樣本外 log-loss 選最佳模型設定。

把原本手設的 half_life / reg / use_elo / xg_weight 改成「跑網格、選讓樣本外
log-loss 最低」的組合——把「猜的」變成「證明過最好的」，且可量化。

評估指標用 evaluation.run 的 model_logloss（越低越準），並一併回報市場 log-loss
當基準（看模型離市場還有多遠）。
"""
from __future__ import annotations

import copy
import itertools
from dataclasses import dataclass, field

import pandas as pd

from .config import Config
from . import evaluation


@dataclass
class TuneResult:
    rows: list[dict] = field(default_factory=list)   # 每個組合的成績
    market_logloss: float = 0.0

    @property
    def best(self) -> dict:
        return min(self.rows, key=lambda r: r["logloss"]) if self.rows else {}

    def table(self) -> pd.DataFrame:
        df = pd.DataFrame(self.rows).sort_values("logloss").reset_index(drop=True)
        return df

    def summary(self) -> str:
        df = self.table()
        b = self.best
        lines = ["============ 超參數調校（樣本外 log-loss，越低越好）============",
                 df.to_string(index=False),
                 "-------------------------------------------------------------",
                 f"市場 log-loss 基準：{self.market_logloss:.4f}",
                 f"最佳組合：{ {k: b[k] for k in b if k not in ('logloss','brier','n')} }",
                 f"最佳 log-loss：{b.get('logloss', float('nan')):.4f}"
                 + ("（已贏市場 ✅）" if b.get('logloss', 9) < self.market_logloss
                    else "（仍輸市場 ⚠️）"),
                 "============================================================="]
        return "\n".join(lines)


# 預設搜尋網格
DEFAULT_GRID = {
    "half_life_days": [120.0, 240.0, 480.0],
    "reg": [0.0, 0.5, 2.0],
    "use_elo": [False, True],
    "xg_weight": [0.0],
}


def tune(df: pd.DataFrame, base_cfg: Config | None = None,
         grid: dict | None = None, refit_every: int = 40,
         min_train_matches: int = 300, verbose: bool = True) -> TuneResult:
    base_cfg = base_cfg or Config()
    grid = grid or DEFAULT_GRID
    keys = list(grid.keys())
    res = TuneResult()

    combos = list(itertools.product(*[grid[k] for k in keys]))
    for ci, combo in enumerate(combos, 1):
        cfg = copy.deepcopy(base_cfg)
        params = dict(zip(keys, combo))
        for k, v in params.items():
            setattr(cfg.model, k, v)
        ev = evaluation.run(df, cfg, refit_every=refit_every,
                            min_train_matches=min_train_matches)
        if not ev.actuals:
            continue
        row = dict(params)
        row["logloss"] = round(ev.model_logloss(), 4)
        row["brier"] = round(ev.model_brier(), 4)
        row["n"] = len(ev.actuals)
        res.rows.append(row)
        res.market_logloss = ev.market_logloss()  # 各組合的市場基準相同
        if verbose:
            print(f"[tune {ci}/{len(combos)}] {params} → logloss={row['logloss']}")
    return res


def apply_best(cfg: Config, result: TuneResult) -> Config:
    """把最佳組合寫回 cfg（回傳同一個 cfg）。"""
    b = result.best
    for k in ("half_life_days", "reg", "use_elo", "xg_weight"):
        if k in b:
            setattr(cfg.model, k, b[k])
    return cfg
