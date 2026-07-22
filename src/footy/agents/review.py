"""定期模型審查 agents：帳本法醫、校準哨兵、權重調參。

三個 agent 每週由 GitHub Actions 排程執行（footy agent review）：
  1. 帳本法醫（ledger_forensics）：對每個 運動×盤口 做 edge 分桶勝率診斷——
     這是抓出「MLB 大小盤 edge 反向」那類錯誤的同一把刀，自動化執行。
     發現 edge>0 子集顯著差於損益兩平（n≥50、z<−2）→ **自動關閘**
     （寫入 data/market_gates.json，bet_signals 不再出「買」）。
     恢復條件達成時只「建議」重開，不自動開（保守不對稱）。
  2. 校準哨兵（calibration_watch）：足球 odds_log 的模型機率 vs 賽果
     Brier/對數損失（近 30 天 vs 之前），漂移早期警報。
  3. 權重調參（weight_tuner）：tracker.tune_weight 掃融合權重，報告最佳值
     ——只建議、不自動改（BLEND_WEIGHT 是全域行為，改動需人工確認）。

數字判斷全部是確定性 Python；LLM（設 ANTHROPIC_API_KEY 時）只負責把
統計結果寫成人話報告與風險排序，**沒有金鑰照樣運作**（報告為純統計版）。
報告輸出 docs/reports/model-review-<date>.md，由 workflow 提交。
"""
from __future__ import annotations

import json
import math
from pathlib import Path

GATES_PATH = "data/market_gates.json"
# 預設閘門（無檔案時）：MLB/NBA 大小盤停買（FINDINGS 2026-07-18 帳本實證）
DEFAULT_GATES = {
    "MLB:OU": {"disabled": True, "since": "2026-07-18",
               "reason": "edge 反向（153 注診斷，FINDINGS）"},
    "NBA:OU": {"disabled": True, "since": "2026-07-18",
               "reason": "與 MLB 同引擎且無資料，先關（開季後由法醫覆核）"},
}


def load_gates(path: str | Path = GATES_PATH) -> dict:
    p = Path(path)
    if p.exists():
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            return dict(DEFAULT_GATES)
    return dict(DEFAULT_GATES)


def save_gates(gates: dict, path: str | Path = GATES_PATH) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(gates, ensure_ascii=False, indent=2), encoding="utf-8")


# ---------------- 1) 帳本法醫 ----------------

def _z_vs_breakeven(wins: int, n: int, avg_odds: float) -> float:
    """edge>0 子集勝率 vs 該賠率損益兩平勝率的 z 值（<0 = 差於兩平）。"""
    if n == 0 or not avg_odds or avg_odds <= 1.0:
        return 0.0
    be = 1.0 / avg_odds
    p = wins / n
    se = math.sqrt(max(be * (1 - be) / n, 1e-12))
    return (p - be) / se


def ledger_forensics(ledger_paths: dict[str, str]) -> dict:
    """對每本帳（{"MLB": path, ...}）做 盤口×edge 分桶診斷。

    回 {"stats": [...], "gate_actions": [...]}；gate_actions 為建議/執行的
    關閘動作（呼叫端決定是否寫檔）。
    """
    import pandas as pd
    stats, actions = [], []
    for sport, path in ledger_paths.items():
        if not Path(path).exists():
            continue
        df = pd.read_csv(path)
        st = df[df["result"].isin(["win", "loss"])].copy()
        if st.empty:
            continue
        st["edge"] = pd.to_numeric(st["edge"], errors="coerce")
        st["odds"] = pd.to_numeric(st["odds"], errors="coerce")
        st["pl"] = pd.to_numeric(st["pl"], errors="coerce")
        for mk, m in st.groupby("market"):
            pos = m[(m["edge"] > 0)].dropna(subset=["odds"])
            wins = int((pos["result"] == "win").sum())
            z = _z_vs_breakeven(wins, len(pos), float(pos["odds"].mean())
                                if len(pos) else 0.0)
            row = {
                "sport": sport, "market": mk, "n": int(len(m)),
                "win_rate": round(float((m["result"] == "win").mean()), 4),
                "avg_pl": round(float(m["pl"].mean()), 4) if m["pl"].notna().any() else None,
                "pos_edge_n": int(len(pos)), "pos_edge_wins": wins,
                "pos_edge_win_rate": round(wins / len(pos), 4) if len(pos) else None,
                "pos_edge_z": round(z, 2),
            }
            stats.append(row)
            key = f"{sport}:{mk}"
            if len(pos) >= 50 and z < -2.0:
                actions.append({"action": "disable", "key": key,
                                "evidence": f"edge>0 共 {len(pos)} 注勝率 "
                                            f"{wins / len(pos):.1%}，z={z:.2f}"})
            elif len(pos) >= 100 and z > 1.0:
                actions.append({"action": "suggest_enable", "key": key,
                                "evidence": f"edge>0 共 {len(pos)} 注勝率 "
                                            f"{wins / len(pos):.1%}，z={z:.2f}"})
    return {"stats": stats, "gate_actions": actions}


def apply_gate_actions(actions: list[dict], gates_path: str | Path = GATES_PATH,
                       today: str = "") -> list[str]:
    """執行關閘（disable 自動生效；suggest_enable 只記報告，不自動開）。"""
    gates = load_gates(gates_path)
    applied = []
    for a in actions:
        if a["action"] == "disable" and not gates.get(a["key"], {}).get("disabled"):
            gates[a["key"]] = {"disabled": True, "since": today,
                               "reason": f"法醫自動關閘：{a['evidence']}"}
            applied.append(f"🔒 關閘 {a['key']}（{a['evidence']}）")
    if applied:
        save_gates(gates, gates_path)
    return applied


# ---------------- 2) 校準哨兵 ----------------

def calibration_watch(odds_log_path: str = "data/odds_log.csv",
                      recent_days: int = 30) -> dict:
    """足球 odds_log：模型機率 vs 賽果的 Brier（近窗 vs 之前），漂移警報。"""
    import numpy as np
    import pandas as pd
    p = Path(odds_log_path)
    if not p.exists():
        return {"available": False}
    df = pd.read_csv(p)
    df = df[(df.get("played") == 1) & (df["market"] == "1X2")].copy()
    if df.empty:
        return {"available": False}
    df["model_p"] = pd.to_numeric(df["model_p"], errors="coerce")
    df = df.dropna(subset=["model_p", "hg", "ag"])
    res = np.where(df["hg"] > df["ag"], "home",
                   np.where(df["ag"] > df["hg"], "away", "draw"))
    df["y"] = (df["selection"] == res).astype(float)
    df["brier"] = (df["model_p"] - df["y"]) ** 2
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    cut = df["date"].max() - pd.Timedelta(days=recent_days)
    recent, prior = df[df["date"] > cut], df[df["date"] <= cut]
    out = {"available": True, "n_recent": int(len(recent)), "n_prior": int(len(prior)),
           "brier_recent": round(float(recent["brier"].mean()), 4) if len(recent) else None,
           "brier_prior": round(float(prior["brier"].mean()), 4) if len(prior) else None}
    out["drift_alert"] = bool(
        out["brier_recent"] is not None and out["brier_prior"] is not None
        and len(recent) >= 90
        and out["brier_recent"] > out["brier_prior"] * 1.15)
    return out


# ---------------- 3) 權重調參 ----------------

def weight_tuner(odds_log_path: str = "data/odds_log.csv") -> dict:
    """tracker.tune_weight 掃融合權重；只建議不自動改。"""
    from .. import tracker
    try:
        res = tracker.tune_weight(odds_log_path)
    except Exception as e:  # noqa: BLE001
        return {"available": False, "error": str(e)[:200]}
    if not res.get("rows"):
        return {"available": False}
    return {"available": True, "n_matches": res.get("n_matches"),
            "best_roi": res.get("best_roi"), "rows": res.get("rows")}


# ---------------- 組裝：跑全部 + 報告 ----------------

def run_review(ledger_paths: dict[str, str] | None = None,
               odds_log_path: str = "data/odds_log.csv",
               gates_path: str | Path = GATES_PATH,
               today: str = "") -> dict:
    """跑三個 agent，執行自動關閘，回傳整包結果（供報告/測試）。"""
    ledger_paths = ledger_paths or {
        "足球": "data/bets.csv", "MLB": "data/mlb_bets.csv", "NBA": "data/nba_bets.csv",
        "MLB純模型": "data/mlb_model_bets.csv", "NBA純模型": "data/nba_model_bets.csv",
        "走地": "data/live_bets.csv",     # 走地推薦帳本（法醫每週覆核走地 edge）
    }
    fx = ledger_forensics(ledger_paths)
    applied = apply_gate_actions(fx["gate_actions"], gates_path, today=today)
    return {"forensics": fx, "gates_applied": applied,
            "gates": load_gates(gates_path),
            "calibration": calibration_watch(odds_log_path),
            "weight": weight_tuner(odds_log_path)}


def compose_report(result: dict, today: str) -> str:
    """統計轉 Markdown 報告；有 LLM 金鑰時加一段人話解讀與風險排序。"""
    from . import llm
    lines = [f"# 模型週審報告 {today}", ""]
    lines.append("## 帳本法醫（盤口 × edge 診斷）")
    lines.append("| 帳本 | 盤口 | 已結算 | 勝率 | 每注損益 | edge>0 注數 | edge>0 勝率 | z |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for s in result["forensics"]["stats"]:
        pl = f"{s['avg_pl']:+.3f}" if s["avg_pl"] is not None else "—"
        pw = f"{s['pos_edge_win_rate']:.1%}" if s["pos_edge_win_rate"] is not None else "—"
        lines.append(f"| {s['sport']} | {s['market']} | {s['n']} | {s['win_rate']:.1%} "
                     f"| {pl} | {s['pos_edge_n']} | {pw} | {s['pos_edge_z']} |")
    if result["gates_applied"]:
        lines.append("")
        lines.append("### 本週自動關閘")
        lines += [f"- {a}" for a in result["gates_applied"]]
    sugg = [a for a in result["forensics"]["gate_actions"] if a["action"] == "suggest_enable"]
    if sugg:
        lines.append("")
        lines.append("### 建議重開（需人工確認）")
        lines += [f"- {a['key']}：{a['evidence']}" for a in sugg]
    lines.append("")
    lines.append("### 目前閘門")
    for k, v in result["gates"].items():
        if v.get("disabled"):
            lines.append(f"- 🔒 {k}（{v.get('since', '')}：{v.get('reason', '')}）")
    cal = result["calibration"]
    lines.append("")
    lines.append("## 校準哨兵（足球 1X2 Brier）")
    if cal.get("available"):
        lines.append(f"- 近 30 天 {cal['brier_recent']}（n={cal['n_recent']}）｜"
                     f"之前 {cal['brier_prior']}（n={cal['n_prior']}）"
                     + ("　⚠️ **漂移警報**（惡化 >15%）" if cal.get("drift_alert") else "　✅"))
    else:
        lines.append("- 無足夠資料")
    wt = result["weight"]
    lines.append("")
    lines.append("## 權重調參（BLEND_WEIGHT 建議，只報告不自動改）")
    if wt.get("available"):
        b = wt["best_roi"] or {}
        lines.append(f"- 掃描 {wt['n_matches']} 場快照，最佳 ROI 權重 "
                     f"w={float(b.get('weight', 0)):.2f}"
                     f"（{int(b.get('n_bets', 0))} 注，ROI {float(b.get('roi', 0)):+.1%}，"
                     f"CLV {float(b.get('clv', 0)):+.3f}）；現行 BLEND_WEIGHT=0.5，"
                     "如連續數週指向同方向再人工調整")
    else:
        lines.append("- 無足夠資料")
    report = "\n".join(lines)
    if llm.available():
        try:
            narrative = llm.complete(
                "你是體育博弈模型的風控審查員。以下是本週自動診斷的統計報告，"
                "請用繁體中文寫 150 字內的「本週結論」：最重要的 1-3 個風險/行動，"
                "語氣直接、不客套、不要複述表格。\n\n" + report,
                max_tokens=400, timeout=60)
            if narrative:
                report += "\n\n## AI 審查員結論\n\n" + narrative.strip() + "\n"
        except Exception:  # noqa: BLE001
            pass
    return report + "\n"
