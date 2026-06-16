"""走地實時盯盤迴圈：串起 feed → 模型 → 價值掃描 → 凱利 → 風控 → 提示。

定位是「實時提示」，不自動下注：偵測到 value 時印出/回呼一則 Alert，
由使用者自行決定是否下單。下單與否、結算結果可回灌給 RiskManager 維持風控狀態。
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from datetime import datetime
from typing import Callable

from ..config import Config
from ..models.dixon_coles import DixonColesModel
from ..risk.manager import RiskManager
from ..value import scanner, staking
from ..value.edge import ValueBet
from .feed import OddsFeed
from .inplay import final_score_matrix


@dataclass
class Alert:
    timestamp: str
    match_id: str
    home: str
    away: str
    minute: int
    score: str
    bet: ValueBet
    suggested_stake: float
    risk_note: str

    def format(self) -> str:
        b = self.bet
        return (
            f"[{self.timestamp}] ⚽ {self.home} {self.score} {self.away} "
            f"({self.minute}') | {b.market} {b.selection} @ {b.odds:.2f}\n"
            f"    模型勝率≈{b.model_prob:.1%}  公平機率≈{b.fair_prob:.1%}  "
            f"edge={b.edge:+.1%}  EV={b.ev:+.3f}/單位\n"
            f"    建議下注：{self.suggested_stake:.2f}（{self.risk_note}）"
        )


def default_alert_sink(alert: Alert) -> None:
    print(alert.format())


class LiveMonitor:
    def __init__(self, model: DixonColesModel, cfg: Config,
                 risk: RiskManager | None = None,
                 alert_sink: Callable[[Alert], None] = default_alert_sink):
        self.model = model
        self.cfg = cfg
        self.risk = risk or RiskManager(cfg.risk)
        self.alert_sink = alert_sink
        # 同一場同一盤口避免短時間重複提示
        self._seen: set[tuple] = set()

    def _evaluate_state(self, state) -> list[Alert]:
        alerts: list[Alert] = []
        try:
            mat = final_score_matrix(
                self.model, state.home, state.away, state.minute,
                state.home_goals, state.away_goals,
                rate_scale=self.cfg.live.inplay_rate_scale,
                home_red=state.home_red, away_red=state.away_red,
                red_attack_boost=self.cfg.live.red_card_attack_boost,
                red_defense_penalty=self.cfg.live.red_card_defense_penalty,
            )
        except KeyError:
            # 模型不認得這兩隊（不在訓練聯賽），略過
            return alerts

        vbs = scanner.value_bets(mat, state.quotes, self.cfg.value,
                                 extra_edge=self.cfg.live.extra_edge)
        for vb in vbs:
            dedup_key = (state.match_id, vb.market, vb.selection,
                         round(vb.odds, 1), state.home_goals, state.away_goals)
            if dedup_key in self._seen:
                continue
            self._seen.add(dedup_key)

            req = staking.stake(self.risk.bankroll, vb.ev, vb.odds, self.cfg.staking)
            decision = self.risk.approve(
                state.match_id, f"{vb.market}:{vb.selection}", req, vb.odds,
                day=datetime.utcnow().date(),
            )
            stake_amt = decision.stake if decision.approved else 0.0
            note = decision.reason if decision.approved else f"風控否決：{decision.reason}"

            alerts.append(Alert(
                timestamp=datetime.utcnow().strftime("%H:%M:%S"),
                match_id=state.match_id, home=state.home, away=state.away,
                minute=state.minute,
                score=f"{state.home_goals}-{state.away_goals}",
                bet=vb, suggested_stake=stake_amt, risk_note=note,
            ))
        return alerts

    def poll_once(self, feed: OddsFeed) -> list[Alert]:
        """拉一次盤、評估、發提示，回傳本次所有 Alert。"""
        all_alerts: list[Alert] = []
        for state in feed.poll():
            for alert in self._evaluate_state(state):
                self.alert_sink(alert)
                all_alerts.append(alert)
        return all_alerts

    def run(self, feed: OddsFeed, max_polls: int | None = None,
            sleep_s: float | None = None) -> None:
        """持續盯盤。max_polls=None 代表直到 feed 沒有進行中比賽為止。"""
        sleep_s = self.cfg.live.poll_interval_s if sleep_s is None else sleep_s
        polls = 0
        print(f"[live] 開始盯盤，輪詢間隔 {sleep_s:.0f}s（Ctrl-C 結束）")
        try:
            while True:
                states_alerts = self.poll_once(feed)
                polls += 1
                if max_polls is not None and polls >= max_polls:
                    break
                # 若 feed 回報已無進行中比賽，poll() 會回空；用一次空輪詢偵測收尾
                if not states_alerts and self._feed_exhausted(feed):
                    print("[live] 已無進行中比賽，結束。")
                    break
                if sleep_s > 0:
                    time.sleep(sleep_s)
        except KeyboardInterrupt:
            print("\n[live] 使用者中止盯盤。")
        finally:
            feed.close()

    @staticmethod
    def _feed_exhausted(feed: OddsFeed) -> bool:
        # SimulatedFeed 在比賽結束後 poll() 回空；真實 feed 可覆寫此邏輯。
        from .feed import SimulatedFeed
        if isinstance(feed, SimulatedFeed):
            return feed.minute >= 90
        return False
