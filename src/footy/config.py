"""集中管理設定。

所有可調參數都在這裡有預設值，並可由 YAML 檔覆寫。
這樣回測、初盤掃描、走地盯盤共用同一套參數，避免不一致。
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

import yaml


@dataclass
class ModelConfig:
    """Dixon–Coles 模型參數。"""

    # 時間衰減半衰期（天）。越小代表越看重近期賽果。
    half_life_days: float = 180.0
    # 比分矩陣最大進球數（單隊）。10 對足球幾乎涵蓋所有情況。
    max_goals: int = 10
    # 低比分相關性參數 rho 的初始值（由 MLE 估計，這只是起點）。
    rho_init: float = -0.05


@dataclass
class ValueConfig:
    """價值偵測參數。"""

    # 最低邊際：模型機率必須比盤口隱含（去 vig）機率高出此比例才算 value。
    # 例如 0.03 表示 model_prob >= fair_prob * 1.03。
    min_edge: float = 0.03
    # 同時要求最低期望值（EV），避免在低賠率區即使有 edge 也下注。
    min_ev: float = 0.02
    # 只考慮賠率落在此區間內的盤口（過低沒空間，過高雜訊大/不可靠）。
    min_odds: float = 1.30
    max_odds: float = 10.0


@dataclass
class StakingConfig:
    """凱利下注參數。"""

    # 分數凱利係數（0~1）。1=全凱利（波動極大），0.25=四分之一凱利（常用）。
    kelly_fraction: float = 0.25
    # 單注佔資金的硬上限（即使凱利算出更多也封頂）。
    max_stake_fraction: float = 0.02
    # 單注最小下注（低於此就不值得下，避免手續費/精力成本）。
    min_stake: float = 1.0


@dataclass
class RiskConfig:
    """風控參數。"""

    # 起始資金。
    bankroll: float = 1000.0
    # 單場比賽總曝險上限（佔資金比例）。
    max_match_exposure: float = 0.05
    # 單日總曝險上限（佔資金比例）。
    max_daily_exposure: float = 0.15
    # 同時持有未結算注單數上限。
    max_open_bets: int = 20
    # 日停損：當日虧損達資金此比例則停止下注。
    daily_stop_loss: float = 0.10
    # 回撤熔斷：自最高點回撤達此比例則暫停所有下注。
    max_drawdown: float = 0.25
    # 連敗冷靜：連續虧損達此次數後暫停（進入冷靜期）。
    losing_streak_cooldown: int = 6


@dataclass
class LiveConfig:
    """走地實時盯盤參數。"""

    # 盯盤輪詢間隔（秒）。
    poll_interval_s: float = 10.0
    # 走地的進球率（goals per minute）相對 pre-match 的縮放，
    # 用來反映「比賽中段往往較收斂、尾段可能開放」等動態。1.0 表示均勻。
    inplay_rate_scale: float = 1.0
    # 走地額外的安全邊際（比 pre-match 更嚴格，因為盤變快、流動性風險高）。
    extra_edge: float = 0.02


@dataclass
class Config:
    model: ModelConfig = field(default_factory=ModelConfig)
    value: ValueConfig = field(default_factory=ValueConfig)
    staking: StakingConfig = field(default_factory=StakingConfig)
    risk: RiskConfig = field(default_factory=RiskConfig)
    live: LiveConfig = field(default_factory=LiveConfig)

    @classmethod
    def load(cls, path: str | Path | None = None) -> "Config":
        cfg = cls()
        if path is None:
            return cfg
        path = Path(path)
        if not path.exists():
            return cfg
        with open(path, "r", encoding="utf-8") as f:
            raw: dict[str, Any] = yaml.safe_load(f) or {}
        for section, klass in (
            ("model", ModelConfig),
            ("value", ValueConfig),
            ("staking", StakingConfig),
            ("risk", RiskConfig),
            ("live", LiveConfig),
        ):
            if section in raw and isinstance(raw[section], dict):
                current = asdict(getattr(cfg, section))
                current.update(raw[section])
                setattr(cfg, section, klass(**current))
        return cfg

    def to_dict(self) -> dict[str, Any]:
        return {
            "model": asdict(self.model),
            "value": asdict(self.value),
            "staking": asdict(self.staking),
            "risk": asdict(self.risk),
            "live": asdict(self.live),
        }
