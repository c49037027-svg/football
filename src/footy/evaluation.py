"""模型校準（calibration）與收盤線價值（CLV）分析。

這是「是否值得用真錢」的關鍵驗證，比回測的 ROI 更基本：
  - **校準**：模型說「30% 機率」的那些事，長期真的有約 30% 發生嗎？
    用 Brier score、log loss 與可靠度表（reliability table）衡量。
  - **贏過市場？**：把模型的 log loss 與「市場收盤盤去 vig 後的隱含機率」比較。
    若模型的 log loss 沒有比市場低，代表模型沒有資訊優勢，長期難以 +EV。
  - **CLV（closing line value）**：你下注的賠率 vs 收盤賠率。
    長期能持續打贏收盤線（拿到比收盤更好的價）是專業盤手最可靠的 +EV 指標，
    因為收盤線通常是最有效率的價格。

全部用 walk-forward（只用過去資料預測未來）以避免前視偏誤。
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .config import Config
from .data import schema as S
from .models import dixon_coles as dc
from .models import markets
from .value.odds import remove_vig_proportional

_OUTCOMES = ("home", "draw", "away")


@dataclass
class EvalResult:
    # 逐場：模型機率、市場機率、實際結果 index(0/1/2)
    model_probs: list[list[float]] = field(default_factory=list)
    market_probs: list[list[float]] = field(default_factory=list)
    actuals: list[int] = field(default_factory=list)
    # CLV：每場每結果 (bet_odds, close_odds) 在我們「會下注」的選項上
    clv_samples: list[float] = field(default_factory=list)
    # 基準名稱（有賠率時為「市場」；國際賽無賠率時為「均勻基準」等）
    baseline_name: str = "市場"

    # ---- 指標 ----
    @staticmethod
    def _brier(probs: np.ndarray, actuals: np.ndarray) -> float:
        onehot = np.zeros_like(probs)
        onehot[np.arange(len(actuals)), actuals] = 1.0
        return float(np.mean(np.sum((probs - onehot) ** 2, axis=1)))

    @staticmethod
    def _logloss(probs: np.ndarray, actuals: np.ndarray) -> float:
        p = np.clip(probs[np.arange(len(actuals)), actuals], 1e-12, 1.0)
        return float(-np.mean(np.log(p)))

    def model_brier(self) -> float:
        return self._brier(np.array(self.model_probs), np.array(self.actuals))

    def model_logloss(self) -> float:
        return self._logloss(np.array(self.model_probs), np.array(self.actuals))

    def market_brier(self) -> float:
        return self._brier(np.array(self.market_probs), np.array(self.actuals))

    def market_logloss(self) -> float:
        return self._logloss(np.array(self.market_probs), np.array(self.actuals))

    # ---- 市場融合（market blending）----
    def blended_logloss(self, w: float) -> float:
        """融合機率 = w*模型 + (1-w)*市場 的 log-loss（w=1 純模型、w=0 純市場）。"""
        mp = np.array(self.model_probs)
        kp = np.array(self.market_probs)
        blended = w * mp + (1.0 - w) * kp
        blended = blended / blended.sum(axis=1, keepdims=True)
        return self._logloss(blended, np.array(self.actuals))

    def best_blend(self, steps: int = 101) -> dict:
        """掃 w∈[0,1] 找讓 log-loss 最低的融合權重（不需重訓，直接用已存機率）。

        注意：w 只是單一純量超參，在數千場上挑選過擬合可忽略；要更嚴謹可用 holdout。
        """
        if not self.actuals:
            return {}
        ws = np.linspace(0.0, 1.0, steps)
        lls = [self.blended_logloss(float(w)) for w in ws]
        i = int(np.argmin(lls))
        return {
            "w": float(ws[i]),
            "blended_logloss": float(lls[i]),
            "model_logloss": self.model_logloss(),
            "market_logloss": self.market_logloss(),
            "curve": list(zip([round(float(x), 2) for x in ws], lls)),
        }

    def reliability_table(self, n_bins: int = 10) -> pd.DataFrame:
        """把所有 (預測機率, 是否發生) 攤平分箱，比較預測 vs 實際頻率。"""
        probs = np.array(self.model_probs).ravel()
        onehot = np.zeros((len(self.actuals), 3))
        onehot[np.arange(len(self.actuals)), self.actuals] = 1.0
        outcomes = onehot.ravel()
        bins = np.linspace(0, 1, n_bins + 1)
        idx = np.clip(np.digitize(probs, bins) - 1, 0, n_bins - 1)
        rows = []
        for b in range(n_bins):
            mask = idx == b
            if mask.sum() == 0:
                continue
            rows.append({
                "bin": f"{bins[b]:.1f}-{bins[b+1]:.1f}",
                "n": int(mask.sum()),
                "pred_mean": round(float(probs[mask].mean()), 4),
                "obs_freq": round(float(outcomes[mask].mean()), 4),
            })
        return pd.DataFrame(rows)

    def mean_clv(self) -> float:
        return float(np.mean(self.clv_samples)) if self.clv_samples else 0.0

    def clv_beat_rate(self) -> float:
        """拿到比收盤更好價（正 CLV）的比例。"""
        if not self.clv_samples:
            return 0.0
        return float(np.mean([1.0 if c > 0 else 0.0 for c in self.clv_samples]))

    def summary(self, cfg: Config | None = None) -> str:
        lines = [
            "============== 校準 / CLV 報告 ==============",
            f"樣本場數          : {len(self.actuals)}",
            f"模型 Brier        : {self.model_brier():.4f}（越低越好）",
            f"{self.baseline_name} Brier  : {self.market_brier():.4f}",
            f"模型 LogLoss      : {self.model_logloss():.4f}（越低越好）",
            f"{self.baseline_name} LogLoss: {self.market_logloss():.4f}",
        ]
        beat = self.model_logloss() < self.market_logloss()
        lines.append(
            (f"✅ 模型 LogLoss 低於{self.baseline_name}：有資訊優勢的跡象。"
             if beat else
             f"⚠️ 模型 LogLoss 未贏{self.baseline_name}。"))

        # 市場融合（僅在有真實市場賠率時才有意義）
        bb = self.best_blend() if self.baseline_name == "市場" else None
        if bb:
            lines += [
                "------------- 市場融合（blending）-------------",
                f"最佳融合權重 w    : {bb['w']:.2f}（w·模型 + (1-w)·市場）",
                f"融合後 LogLoss    : {bb['blended_logloss']:.4f}",
                f"  vs 純模型 {bb['model_logloss']:.4f} / 純市場 {bb['market_logloss']:.4f}",
            ]
            if bb["blended_logloss"] < bb["market_logloss"] - 1e-6:
                lines.append("✅ 融合後贏過市場：模型帶來增量資訊（值得用融合機率）。")
            elif bb["w"] <= 1e-6:
                lines.append("⚠️ 最佳 w≈0：模型對市場沒有增量，純用市場最好。")
            else:
                lines.append("➖ 融合僅與市場持平：模型增量有限。")

        if self.clv_samples:
            lines.append(f"平均 CLV          : {self.mean_clv():+.2%}（下注價相對收盤價的優勢）")
            lines.append(f"打贏收盤線比例    : {self.clv_beat_rate():.1%}")
            if self.mean_clv() > 0:
                lines.append("✅ 平均正 CLV：長期最可靠的 +EV 指標。")
            else:
                lines.append("⚠️ 平均負 CLV：選到的價普遍比收盤差，警訊。")
        lines.append("============================================")
        lines.append("\n可靠度表（pred_mean 應接近 obs_freq）：")
        lines.append(self.reliability_table().to_string(index=False))
        return "\n".join(lines)


def run_intl(df: pd.DataFrame, cfg: Config, test_since: str | None = "2018-01-01",
             refit_every: int = 200, min_train_matches: int = 1000,
             neutral_default: bool = False, model_kind: str = "dc",
             verbose: bool = False) -> EvalResult:
    """國際賽校準（無賠率）：walk-forward 評估模型 1X2 校準，基準為均勻(1/3)。

    model_kind："dc"=Dixon–Coles(每隊攻防)；"elo_poisson"=純 Elo 驅動(無隊參數)。
    df 需含內部欄位與（選用）home_elo/away_elo、neutral。只對 test_since 之後的
    比賽計分（但仍用其之前全部資料擬合，避免洩漏）。
    """
    df = df.sort_values(S.DATE).reset_index(drop=True)
    res = EvalResult(baseline_name="均勻(1/3)基準")
    test_ts = pd.Timestamp(test_since) if test_since else None
    model = None
    since_refit = 0
    uniform = [1 / 3, 1 / 3, 1 / 3]

    for i in range(len(df)):
        if i < min_train_matches:
            continue
        row = df.iloc[i]
        if model is None or since_refit >= refit_every:
            try:
                if model_kind == 'elo_poisson':
                    from .models.elo_poisson import fit_elo_poisson
                    model = fit_elo_poisson(df.iloc[:i], half_life_days=cfg.model.half_life_days,
                                            max_goals=cfg.model.max_goals, reference_date=row[S.DATE])
                else:
                    model = dc.fit(df.iloc[:i], half_life_days=cfg.model.half_life_days,
                                   max_goals=cfg.model.max_goals, rho_init=cfg.model.rho_init,
                                   xg_weight=cfg.model.xg_weight, use_elo=cfg.model.use_elo,
                                   reg=cfg.model.reg, reference_date=row[S.DATE])
            except Exception as e:  # noqa: BLE001
                if verbose:
                    print(f"[warn] 第 {i} 場擬合失敗：{e}")
                continue
            since_refit = 0
        since_refit += 1

        if test_ts is not None and row[S.DATE] < test_ts:
            continue
        home, away = row[S.HOME], row[S.AWAY]
        if home not in model.attack or away not in model.attack:
            continue
        neutral = bool(row[S.NEUTRAL]) if S.NEUTRAL in df.columns else neutral_default
        mat = model.score_matrix(home, away, neutral=neutral)
        mp = markets.outcome_1x2(mat)
        actual = (0 if row[S.HOME_GOALS] > row[S.AWAY_GOALS]
                  else 2 if row[S.HOME_GOALS] < row[S.AWAY_GOALS] else 1)
        res.model_probs.append([mp["home"], mp["draw"], mp["away"]])
        res.market_probs.append(uniform)
        res.actuals.append(actual)
    return res


def run(df: pd.DataFrame, cfg: Config, refit_every: int = 20,
        min_train_matches: int = 200, verbose: bool = False) -> EvalResult:
    """walk-forward 收集模型/市場預測並計算校準與 CLV。"""
    df = df.dropna(subset=[S.ODDS_HOME, S.ODDS_DRAW, S.ODDS_AWAY]).reset_index(drop=True)
    df = df.sort_values(S.DATE).reset_index(drop=True)
    has_open = all(c in df.columns for c in (S.ODDS_HOME_OPEN, S.ODDS_DRAW_OPEN, S.ODDS_AWAY_OPEN))

    res = EvalResult()
    model = None
    since_refit = 0

    for i in range(len(df)):
        if i < min_train_matches:
            continue
        row = df.iloc[i]
        if model is None or since_refit >= refit_every:
            try:
                model = dc.fit(df.iloc[:i], half_life_days=cfg.model.half_life_days,
                               max_goals=cfg.model.max_goals, rho_init=cfg.model.rho_init,
                               xg_weight=cfg.model.xg_weight, use_elo=cfg.model.use_elo,
                               reg=cfg.model.reg, reference_date=row[S.DATE])
            except Exception as e:  # noqa: BLE001
                if verbose:
                    print(f"[warn] 第 {i} 場擬合失敗：{e}")
                continue
            since_refit = 0
        since_refit += 1

        home, away = row[S.HOME], row[S.AWAY]
        if home not in model.attack or away not in model.attack:
            continue

        mat = model.score_matrix(home, away)
        mp = markets.outcome_1x2(mat)
        model_p = [mp["home"], mp["draw"], mp["away"]]

        close_odds = [float(row[S.ODDS_HOME]), float(row[S.ODDS_DRAW]), float(row[S.ODDS_AWAY])]
        market_p = remove_vig_proportional(close_odds)

        actual = (0 if row[S.HOME_GOALS] > row[S.AWAY_GOALS]
                  else 2 if row[S.HOME_GOALS] < row[S.AWAY_GOALS] else 1)

        res.model_probs.append(model_p)
        res.market_probs.append(market_p)
        res.actuals.append(actual)

        # CLV：對「模型認為有 value」的選項，用開盤價當作我們的下注價，比收盤價。
        if has_open:
            open_odds = [row.get(S.ODDS_HOME_OPEN), row.get(S.ODDS_DRAW_OPEN), row.get(S.ODDS_AWAY_OPEN)]
            for k in range(3):
                oo, co = open_odds[k], close_odds[k]
                if pd.isna(oo) or model_p[k] <= 1.0 / oo:
                    continue  # 開盤沒 value 就不會下注，不計入 CLV
                # CLV：下注賠率高於收盤賠率即為正（拿到比收盤更好的價）。
                # 以收盤隱含機率為基準的相對增益 = oo/co - 1。
                clv = (oo / co) - 1.0
                res.clv_samples.append(clv)

    return res
