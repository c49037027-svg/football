"""盤口 feed 抽象介面 + 內建模擬 feed。

設計重點：把「資料來源」與「決策邏輯」解耦。
監控迴圈只依賴 `OddsFeed` 介面，因此要接真實數據（The Odds API / Betfair /
api-football…）時，只需實作一個回傳相同 `MatchState` 與 `MarketQuote` 的子類別，
模型、價值、風控、提示等全部複用。

無需任何 API key 即可用 `SimulatedFeed` 跑通整條走地流程做驗證。
"""
from __future__ import annotations

import random
from abc import ABC, abstractmethod
from dataclasses import dataclass, field


@dataclass
class MarketQuote:
    """單一可下注選項的即時報價。

    market/selection 命名需與 markets 模組一致，例如：
      market="1X2", selection="home"/"draw"/"away"
      market="OU", line=2.5, selection="over"/"under"
      market="AH", line=-0.5（home 視角）, selection="home"/"away"
    """

    market: str
    selection: str
    odds: float
    line: float | None = None


@dataclass
class MatchState:
    """某場比賽在某時刻的狀態與當前盤口。"""

    match_id: str
    home: str
    away: str
    minute: int          # 已進行分鐘（0=未開賽，>=90 接近完場）
    home_goals: int
    away_goals: int
    quotes: list[MarketQuote] = field(default_factory=list)
    # 紅牌等情境（可選，進階模型可用；基礎版先不納入）
    home_red: int = 0
    away_red: int = 0


class OddsFeed(ABC):
    """盤口資料來源抽象介面。"""

    @abstractmethod
    def poll(self) -> list[MatchState]:
        """回傳目前所有進行中比賽的最新狀態與盤口。"""
        ...

    def close(self) -> None:  # noqa: B027 - 預設無動作
        """釋放資源（如關閉連線）。"""


class SimulatedFeed(OddsFeed):
    """模擬一場走地比賽：時間推進、隨機進球、盤口隨情勢變動。

    用來在沒有真實數據時驗證整條流程，也方便寫測試。
    為了示範 value，模擬盤口會故意在某些時刻略偏離真實機率。
    """

    def __init__(self, home: str, away: str, true_lambda: float = 1.5,
                 true_mu: float = 1.1, minutes_per_poll: int = 5,
                 seed: int | None = 42):
        self.home = home
        self.away = away
        self.true_lambda = true_lambda
        self.true_mu = true_mu
        self.minutes_per_poll = minutes_per_poll
        self.minute = 0
        self.home_goals = 0
        self.away_goals = 0
        self.rng = random.Random(seed)
        self.match_id = f"SIM-{home}-{away}"

    def _maybe_score(self) -> None:
        """依每分鐘進球率，在這段時間內隨機決定是否進球。"""
        # 每分鐘進球率（pre-match 預期 / 90）
        ph = self.true_lambda / 90.0
        pa = self.true_mu / 90.0
        for _ in range(self.minutes_per_poll):
            if self.rng.random() < ph:
                self.home_goals += 1
            if self.rng.random() < pa:
                self.away_goals += 1

    def _make_quotes(self) -> list[MarketQuote]:
        """產生帶有少量噪音/偏差的盤口，模擬市場非完全效率。"""
        # 這裡用一個粗略的內部估計加 overround 與偏差，產生可下注盤口。
        # 真實情況由真實 feed 提供，這只是讓流程跑得起來。
        bias = self.rng.uniform(-0.06, 0.10)  # 偏向略微高估 over 之類
        q = []
        # 大小球 2.5
        base_over = max(0.05, min(0.95, 0.5 + bias))
        q.append(MarketQuote("OU", "over", round(1.0 / base_over * 1.05, 2), line=2.5))
        q.append(MarketQuote("OU", "under", round(1.0 / (1 - base_over) * 1.05, 2), line=2.5))
        # 1X2（粗略）
        for sel, p in (("home", 0.45), ("draw", 0.28), ("away", 0.27)):
            jitter = self.rng.uniform(-0.03, 0.03)
            q.append(MarketQuote("1X2", sel, round(1.0 / max(0.05, p + jitter) * 1.06, 2)))
        return q

    def poll(self) -> list[MatchState]:
        if self.minute >= 90:
            return []  # 比賽結束
        self.minute = min(90, self.minute + self.minutes_per_poll)
        self._maybe_score()
        state = MatchState(
            match_id=self.match_id, home=self.home, away=self.away,
            minute=self.minute, home_goals=self.home_goals,
            away_goals=self.away_goals, quotes=self._make_quotes(),
        )
        return [state]
