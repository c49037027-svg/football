"""風控管理器：在「凱利算出該下多少」之後，做最後一道把關。

凱利管的是「成長最佳化」，風控管的是「活下來」。即使某注 +EV、凱利建議下注，
仍可能因為以下理由被風控否決或縮減：
  - 單場/單日曝險已達上限
  - 未結算注單過多
  - 當日已觸發停損
  - 帳戶自最高點回撤過大（熔斷）
  - 連敗達冷靜門檻

這個類別維護「目前資金狀態」，並提供 approve() 對每一筆候選下注裁決。
回測引擎與走地盯盤共用同一套風控邏輯，確保「測得到的＝實戰跑的」。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date

from ..config import RiskConfig


@dataclass
class Decision:
    approved: bool
    stake: float          # 核准後的實際下注額（可能被縮減）
    reason: str = ""      # 被拒或被縮減的原因


@dataclass
class OpenBet:
    match_id: str
    selection: str
    stake: float
    odds: float


@dataclass
class RiskManager:
    cfg: RiskConfig
    bankroll: float = 0.0
    peak_bankroll: float = 0.0
    today: "date | None" = None
    daily_pnl: float = 0.0
    daily_exposure: float = 0.0
    match_exposure: dict[str, float] = field(default_factory=dict)
    open_bets: list[OpenBet] = field(default_factory=list)
    losing_streak: int = 0
    halted: bool = False  # 熔斷後手動才能解除

    def __post_init__(self):
        if self.bankroll <= 0:
            self.bankroll = self.cfg.bankroll
        self.peak_bankroll = max(self.peak_bankroll, self.bankroll)

    # ---------- 每日重置 ----------
    def roll_day(self, day: date) -> None:
        if self.today != day:
            self.today = day
            self.daily_pnl = 0.0
            self.daily_exposure = 0.0

    # ---------- 裁決一筆候選下注 ----------
    def approve(self, match_id: str, selection: str, requested_stake: float,
                odds: float, day: "date | None" = None) -> Decision:
        if day is not None:
            self.roll_day(day)

        if requested_stake <= 0:
            return Decision(False, 0.0, "下注額為 0")

        if self.halted:
            return Decision(False, 0.0, "已熔斷（回撤過大），需人工解除")

        # 回撤熔斷
        drawdown = 1.0 - (self.bankroll / self.peak_bankroll) if self.peak_bankroll > 0 else 0.0
        if drawdown >= self.cfg.max_drawdown:
            self.halted = True
            return Decision(False, 0.0,
                            f"回撤 {drawdown:.1%} 達熔斷線 {self.cfg.max_drawdown:.0%}")

        # 連敗冷靜
        if self.losing_streak >= self.cfg.losing_streak_cooldown:
            return Decision(False, 0.0,
                            f"連敗 {self.losing_streak} 場達冷靜門檻")

        # 當日停損
        if self.daily_pnl <= -self.cfg.daily_stop_loss * self.bankroll:
            return Decision(False, 0.0, "已觸發當日停損")

        # 未結算注單數
        if len(self.open_bets) >= self.cfg.max_open_bets:
            return Decision(False, 0.0, "未結算注單數已達上限")

        stake = requested_stake

        # 單場曝險上限（縮減而非直接拒絕）
        cap_match = self.cfg.max_match_exposure * self.bankroll
        used_match = self.match_exposure.get(match_id, 0.0)
        room_match = cap_match - used_match
        if room_match <= 0:
            return Decision(False, 0.0, "單場曝險已達上限")
        stake = min(stake, room_match)

        # 單日曝險上限
        cap_day = self.cfg.max_daily_exposure * self.bankroll
        room_day = cap_day - self.daily_exposure
        if room_day <= 0:
            return Decision(False, 0.0, "單日曝險已達上限")
        stake = min(stake, room_day)

        if stake <= 0:
            return Decision(False, 0.0, "曝險上限後可下注額為 0")

        reason = "ok" if stake >= requested_stake else "因曝險上限縮減下注額"
        return Decision(True, round(stake, 2), reason)

    # ---------- 實際記帳 ----------
    def place(self, match_id: str, selection: str, stake: float, odds: float) -> None:
        """登記一筆已下注（佔用曝險）。"""
        self.daily_exposure += stake
        self.match_exposure[match_id] = self.match_exposure.get(match_id, 0.0) + stake
        self.open_bets.append(OpenBet(match_id, selection, stake, odds))

    def settle(self, match_id: str, selection: str, pnl: float) -> None:
        """結算一筆注單，更新資金、回撤、連敗與當日損益。

        pnl 為淨損益（不含本金時的盈虧；輸了是 -stake，贏了是 +stake*(odds-1)）。
        """
        self.bankroll += pnl
        self.daily_pnl += pnl
        self.peak_bankroll = max(self.peak_bankroll, self.bankroll)
        if pnl < 0:
            self.losing_streak += 1
        elif pnl > 0:
            self.losing_streak = 0
        # 從未結算清單移除（移除第一筆匹配的）
        for i, b in enumerate(self.open_bets):
            if b.match_id == match_id and b.selection == selection:
                self.open_bets.pop(i)
                break

    @property
    def drawdown(self) -> float:
        if self.peak_bankroll <= 0:
            return 0.0
        return 1.0 - (self.bankroll / self.peak_bankroll)
