"""Walk-forward 回測引擎：驗證初盤 1X2 策略是否長期 +EV。

方法（避免前視偏誤 look-ahead bias，這是回測最常見的致命錯誤）：
  1. 依時間排序所有比賽。
  2. 從某個起始點開始，逐場（或逐輪）往後走：
       - 只用「該場之前」的資料重新擬合 Dixon–Coles 模型。
       - 對該場用收盤 1X2 賠率掃描 value，套凱利 + 風控決定下注。
       - 用真實賽果結算，更新資金與風控狀態。
  3. 統計總報酬、ROI、命中率、最大回撤、下注數等。

為效率，模型不必每場重擬合，可每隔 refit_every 場重擬一次（預設 20）。
只回測 1X2，因為 football-data.co.uk 的歷史檔有穩定的收盤 1X2 賠率。
亞盤/大小球的歷史賠率欄位較不一致，實戰時由真實 feed 提供即可。
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ..config import Config
from ..data import schema as S
from ..models import dixon_coles as dc
from ..models import markets
from ..risk.manager import RiskManager
from ..value import staking
from ..value.odds import remove_vig_proportional


@dataclass
class BetRecord:
    date: pd.Timestamp
    match: str
    selection: str
    odds: float
    stake: float
    model_prob: float
    fair_prob: float
    edge: float
    ev: float
    won: bool
    pnl: float


@dataclass
class BacktestResult:
    bets: list[BetRecord] = field(default_factory=list)
    starting_bankroll: float = 0.0
    ending_bankroll: float = 0.0
    peak_bankroll: float = 0.0
    max_drawdown: float = 0.0
    equity_curve: list[float] = field(default_factory=list)

    @property
    def n_bets(self) -> int:
        return len(self.bets)

    @property
    def total_staked(self) -> float:
        return sum(b.stake for b in self.bets)

    @property
    def total_pnl(self) -> float:
        return sum(b.pnl for b in self.bets)

    @property
    def roi(self) -> float:
        return self.total_pnl / self.total_staked if self.total_staked > 0 else 0.0

    @property
    def hit_rate(self) -> float:
        if not self.bets:
            return 0.0
        return sum(1 for b in self.bets if b.won) / len(self.bets)

    @property
    def avg_edge(self) -> float:
        if not self.bets:
            return 0.0
        return float(np.mean([b.edge for b in self.bets]))

    def summary(self) -> str:
        growth = (self.ending_bankroll / self.starting_bankroll - 1.0) if self.starting_bankroll else 0.0
        lines = [
            "================ 回測結果 ================",
            f"下注數          : {self.n_bets}",
            f"命中率          : {self.hit_rate:.1%}",
            f"平均 edge       : {self.avg_edge:+.2%}",
            f"總下注額        : {self.total_staked:,.2f}",
            f"總損益          : {self.total_pnl:+,.2f}",
            f"ROI (損益/下注) : {self.roi:+.2%}",
            f"起始資金        : {self.starting_bankroll:,.2f}",
            f"結束資金        : {self.ending_bankroll:,.2f}",
            f"資金成長        : {growth:+.2%}",
            f"最大回撤        : {self.max_drawdown:.2%}",
            "==========================================",
        ]
        if self.roi > 0:
            lines.append("✅ 樣本內呈現正 ROI。注意：這不保證未來，仍需 out-of-sample 與紙上交易驗證。")
        else:
            lines.append("⚠️ 樣本內 ROI 非正。請調整 edge/half-life 等參數，或承認此市場過於有效。")
        return "\n".join(lines)


def run(df: pd.DataFrame, cfg: Config, refit_every: int = 20,
        min_train_matches: int = 200, verbose: bool = True) -> BacktestResult:
    """執行 walk-forward 回測（只測初盤 1X2）。"""
    df = df.dropna(subset=[S.ODDS_HOME, S.ODDS_DRAW, S.ODDS_AWAY]).reset_index(drop=True)
    df = df.sort_values(S.DATE).reset_index(drop=True)

    risk = RiskManager(cfg.risk)
    result = BacktestResult(starting_bankroll=risk.bankroll)
    result.equity_curve.append(risk.bankroll)

    model = None
    since_refit = 0

    for i in range(len(df)):
        if i < min_train_matches:
            continue
        row = df.iloc[i]

        # 視需要重擬模型（只用 i 之前的資料，避免前視）
        if model is None or since_refit >= refit_every:
            train = df.iloc[:i]
            try:
                model = dc.fit(train, half_life_days=cfg.model.half_life_days,
                               max_goals=cfg.model.max_goals,
                               rho_init=cfg.model.rho_init,
                               xg_weight=cfg.model.xg_weight,
                               use_elo=cfg.model.use_elo,
                               reference_date=row[S.DATE])
            except Exception as e:  # noqa: BLE001
                if verbose:
                    print(f"[warn] 第 {i} 場擬合失敗：{e}")
                continue
            since_refit = 0
        since_refit += 1

        home, away = row[S.HOME], row[S.AWAY]
        if home not in model.attack or away not in model.attack:
            continue  # 新升班隊等，模型沒有參數

        mat = model.score_matrix(home, away)
        probs = markets.outcome_1x2(mat)

        odds = {
            "home": float(row[S.ODDS_HOME]),
            "draw": float(row[S.ODDS_DRAW]),
            "away": float(row[S.ODDS_AWAY]),
        }
        fair = dict(zip(["home", "draw", "away"],
                        remove_vig_proportional([odds["home"], odds["draw"], odds["away"]])))

        actual = ("home" if row[S.HOME_GOALS] > row[S.AWAY_GOALS]
                  else "away" if row[S.HOME_GOALS] < row[S.AWAY_GOALS] else "draw")

        day = row[S.DATE].date()
        risk.roll_day(day)

        for sel in ("home", "draw", "away"):
            o = odds[sel]
            if not (cfg.value.min_odds <= o <= cfg.value.max_odds):
                continue
            p = probs[sel]
            ev = p * (o - 1.0) - (1.0 - p)
            edge = (p / fair[sel] - 1.0) if fair[sel] > 0 else 0.0
            if ev < cfg.value.min_ev or edge < cfg.value.min_edge:
                continue

            req = staking.stake(risk.bankroll, ev, o, cfg.staking)
            decision = risk.approve(f"M{i}", sel, req, o, day=day)
            if not decision.approved or decision.stake <= 0:
                continue

            stake_amt = decision.stake
            risk.place(f"M{i}", sel, stake_amt, o)
            won = (sel == actual)
            pnl = stake_amt * (o - 1.0) if won else -stake_amt
            risk.settle(f"M{i}", sel, pnl)

            result.bets.append(BetRecord(
                date=row[S.DATE], match=f"{home} v {away}", selection=sel,
                odds=o, stake=stake_amt, model_prob=p, fair_prob=fair[sel],
                edge=edge, ev=ev, won=won, pnl=pnl,
            ))
            result.equity_curve.append(risk.bankroll)

    result.ending_bankroll = risk.bankroll
    result.peak_bankroll = risk.peak_bankroll
    # 由 equity curve 算最大回撤
    eq = np.array(result.equity_curve)
    if len(eq) > 1:
        peaks = np.maximum.accumulate(eq)
        dd = 1.0 - eq / peaks
        result.max_drawdown = float(dd.max())
    return result
