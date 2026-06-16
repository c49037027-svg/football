"""走地十六法則 規則引擎。

說明（請務必讀）：
「滾球/走地十六法則」是華人足球圈流傳的走地心法，**並無單一權威版本**。
這裡是我綜合常見共識（走地以大小球為主、讓球只在強對弱且弱隊吃牌後才玩、
小球有底線、大球有頂線、紅牌是關鍵信號、賠率跳動快要保守…）整理、編碼成
16 條**透明且可調**的規則。它們不是「真理」，而是疊在 +EV 引擎之上的
**過濾器與注碼調節器**：先由模型找出正期望值的盤，再用這些法則否決不該碰的、
並對高風險情境縮減注碼。所有門檻都在 LiveConfig 可調，請用你自己的資料校準。

每條規則對「某個候選下注」回傳 RuleSignal：
  - allow=False 直接否決（veto）。
  - stake_factor < 1 縮減注碼（風險調節）。
  - note 說明觸發原因，會顯示在提示中。

引擎把所有規則的 allow 做 AND、stake_factor 連乘，得到最終裁決。
"""
from __future__ import annotations

from dataclasses import dataclass

from ..config import LiveConfig


@dataclass
class RuleSignal:
    allow: bool = True
    stake_factor: float = 1.0
    note: str = ""


@dataclass
class RuleContext:
    """規則判斷所需的場況。"""

    minute: int
    home_goals: int
    away_goals: int
    home_red: int
    away_red: int
    market: str            # "1X2" / "OU" / "AH"
    selection: str         # home/away/draw/over/under
    line: float | None
    exp_total_goals: float  # 模型對整場的預期總進球（環境鬆緊）
    # 模型方向與盤口跳動方向是否一致（None=未知）
    odds_moving_with_model: "bool | None" = None

    @property
    def lead(self) -> int:
        """主隊淨領先球數（負為落後）。"""
        return self.home_goals - self.away_goals

    @property
    def man_adv(self) -> int:
        """主隊淨人數優勢（對方紅牌多則為正）。"""
        return self.away_red - self.home_red

    @property
    def total_goals(self) -> int:
        return self.home_goals + self.away_goals


# ---------------- 十六法則 ----------------
# 每條規則 = 一個函式 (ctx, cfg) -> RuleSignal。順序即法則編號。

def r01_live_focus_goals_markets(ctx: RuleContext, cfg: LiveConfig) -> RuleSignal:
    """法則1：走地以大小球為主、讓球為輔。讓球需另由法則5把關。"""
    return RuleSignal(note="法則1：大小球為主")


def r02_small_ball_floor(ctx: RuleContext, cfg: LiveConfig) -> RuleSignal:
    """法則2：小球底線——不碰過低的小球線（如小1.5）。"""
    if ctx.market == "OU" and ctx.selection == "under" and ctx.line is not None:
        if ctx.line < cfg.small_ball_min_line:
            return RuleSignal(False, note=f"法則2否決：小球線 {ctx.line} 低於底線 {cfg.small_ball_min_line}")
    return RuleSignal()


def r03_big_ball_cap(ctx: RuleContext, cfg: LiveConfig) -> RuleSignal:
    """法則3：大球頂線——一般聯賽不追過高大球；高進球環境才放寬。"""
    if ctx.market == "OU" and ctx.selection == "over" and ctx.line is not None:
        cap = cfg.big_ball_max_line
        if ctx.exp_total_goals >= cfg.high_goal_env_total:
            cap += 1.0  # 高進球環境放寬一個整數
        if ctx.line > cap:
            return RuleSignal(False, note=f"法則3否決：大球線 {ctx.line} 超過頂線 {cap}")
    return RuleSignal()


def r04_small_ball_needs_defensive_env(ctx: RuleContext, cfg: LiveConfig) -> RuleSignal:
    """法則4：小球僅在低進球（防守型）環境才考慮。"""
    if ctx.market == "OU" and ctx.selection == "under":
        if ctx.exp_total_goals >= cfg.high_goal_env_total:
            return RuleSignal(False, note=f"法則4否決：高進球環境(預期{ctx.exp_total_goals:.1f})不玩小球")
    return RuleSignal()


def r05_handicap_only_strong_vs_weak_or_red(ctx: RuleContext, cfg: LiveConfig) -> RuleSignal:
    """法則5：讓球只在一方少打多（紅牌）時才考慮，否則否決。"""
    if ctx.market == "AH" and cfg.handicap_requires_man_adv:
        if ctx.man_adv == 0:
            return RuleSignal(False, note="法則5否決：無紅牌人數差，不玩走地讓球")
    return RuleSignal()


def r06_red_card_signal(ctx: RuleContext, cfg: LiveConfig) -> RuleSignal:
    """法則6：紅牌是最強信號——人數佔優方讓球 / 偏大球（比賽更開放）。"""
    if ctx.man_adv != 0:
        # 偏大球與「佔優方讓球」加分（注碼略增，封頂 1.0 由引擎處理）。
        favorable = (
            (ctx.market == "OU" and ctx.selection == "over")
            or (ctx.market == "AH" and (
                (ctx.man_adv > 0 and ctx.selection == "home")
                or (ctx.man_adv < 0 and ctx.selection == "away")))
        )
        if favorable:
            return RuleSignal(note="法則6：紅牌信號順勢")
    return RuleSignal()


def r07_no_bet_early(ctx: RuleContext, cfg: LiveConfig) -> RuleSignal:
    """法則7：開賽前 N 分鐘不下注（盤未穩、樣本太少）。"""
    if ctx.minute < cfg.no_bet_before_minute:
        return RuleSignal(False, note=f"法則7否決：開賽未滿 {cfg.no_bet_before_minute} 分鐘")
    return RuleSignal()


def r08_stalemate_converges_small(ctx: RuleContext, cfg: LiveConfig) -> RuleSignal:
    """法則8：0-0 且過半時，總分傾向收斂——偏大球謹慎、偏小球/平局合理。"""
    if ctx.minute >= 45 and ctx.total_goals == 0:
        if ctx.market == "OU" and ctx.selection == "over":
            return RuleSignal(stake_factor=0.7, note="法則8：半場 0-0，大球收斂風險，注碼折減")
    return RuleSignal()


def r09_late_game_high_variance(ctx: RuleContext, cfg: LiveConfig) -> RuleSignal:
    """法則9：終場前高波動，落後方壓上——大球有機會但風險高，縮減注碼。"""
    if ctx.minute >= cfg.late_game_minute:
        return RuleSignal(stake_factor=cfg.late_game_stake_factor,
                          note=f"法則9：{cfg.late_game_minute}'後高波動，注碼折減")
    return RuleSignal()


def r10_big_lead_caution(ctx: RuleContext, cfg: LiveConfig) -> RuleSignal:
    """法則10：一方大幅領先時比賽可能鬆懈，但盤口多已反映——不追已反映的盤。"""
    if abs(ctx.lead) >= cfg.big_lead_goals:
        # 對「順著大比分繼續加注領先方深盤讓球」保守。
        if ctx.market == "AH":
            leader_side = "home" if ctx.lead > 0 else "away"
            if ctx.selection == leader_side:
                return RuleSignal(stake_factor=0.6, note="法則10：大幅領先方讓球，注碼折減")
    return RuleSignal()


def r11_odds_volatility_guard(ctx: RuleContext, cfg: LiveConfig) -> RuleSignal:
    """法則11：盤口跳動與模型方向不一致時不追（程式看不到球，保守處理）。"""
    if ctx.odds_moving_with_model is False:
        return RuleSignal(False, note="法則11否決：盤口跳動與模型方向相悖")
    return RuleSignal()


def r12_requires_positive_ev(ctx: RuleContext, cfg: LiveConfig) -> RuleSignal:
    """法則12：只下模型 +EV 的盤（已由 value 引擎保證，此處為宣告式佔位）。"""
    return RuleSignal(note="法則12：僅 +EV")


def r13_conservative_live_staking(ctx: RuleContext, cfg: LiveConfig) -> RuleSignal:
    """法則13：走地注碼比初盤更保守（額外乘上保守係數）。"""
    return RuleSignal(stake_factor=0.8, note="法則13：走地保守注碼")


def r14_cooldown_after_losses(ctx: RuleContext, cfg: LiveConfig) -> RuleSignal:
    """法則14：連敗冷靜（由 RiskManager 處理，此處宣告式佔位）。"""
    return RuleSignal(note="法則14：連敗冷靜由風控把關")


def r15_info_consistency(ctx: RuleContext, cfg: LiveConfig) -> RuleSignal:
    """法則15：邊看直播邊確認——程式無法看球，故僅在資訊一致時放行（佔位）。"""
    return RuleSignal(note="法則15：建議人工核對直播場面")


def r16_limit_overtrading(ctx: RuleContext, cfg: LiveConfig) -> RuleSignal:
    """法則16：單場走地下注次數上限（由引擎計數，此處宣告式佔位）。"""
    return RuleSignal(note="法則16：限制過度交易")


ALL_RULES = [
    r01_live_focus_goals_markets, r02_small_ball_floor, r03_big_ball_cap,
    r04_small_ball_needs_defensive_env, r05_handicap_only_strong_vs_weak_or_red,
    r06_red_card_signal, r07_no_bet_early, r08_stalemate_converges_small,
    r09_late_game_high_variance, r10_big_lead_caution, r11_odds_volatility_guard,
    r12_requires_positive_ev, r13_conservative_live_staking, r14_cooldown_after_losses,
    r15_info_consistency, r16_limit_overtrading,
]


@dataclass
class RuleVerdict:
    allow: bool
    stake_factor: float
    notes: list[str]
    vetoed_by: "str | None" = None

    def summary(self) -> str:
        if not self.allow:
            return f"走地法則否決（{self.vetoed_by}）"
        active = [n for n in self.notes if "否決" not in n]
        return "；".join(active[:3]) if active else "通過走地法則"


class RuleEngine:
    """套用十六法則，回傳對某候選下注的最終裁決。"""

    def __init__(self, cfg: LiveConfig):
        self.cfg = cfg
        self._match_bet_count: dict[str, int] = {}

    def evaluate(self, ctx: RuleContext, match_id: str | None = None) -> RuleVerdict:
        if not self.cfg.rules_enabled:
            return RuleVerdict(True, 1.0, ["規則引擎未啟用"])

        notes: list[str] = []
        stake_factor = 1.0
        for rule in ALL_RULES:
            sig = rule(ctx, self.cfg)
            if sig.note:
                notes.append(sig.note)
            if not sig.allow:
                return RuleVerdict(False, 0.0, notes, vetoed_by=sig.note)
            stake_factor *= sig.stake_factor

        # 法則16：單場走地下注次數上限
        if match_id is not None:
            cnt = self._match_bet_count.get(match_id, 0)
            if cnt >= self.cfg.max_live_bets_per_match:
                return RuleVerdict(False, 0.0, notes,
                                   vetoed_by=f"法則16否決：單場已達 {cnt} 注上限")

        return RuleVerdict(True, max(0.0, min(1.0, stake_factor)), notes)

    def record_bet(self, match_id: str) -> None:
        """登記一筆已通過並下注的走地注（供法則16計數）。"""
        self._match_bet_count[match_id] = self._match_bet_count.get(match_id, 0) + 1
