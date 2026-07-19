"""模型改進 AI 團隊：守門端（review.py 三員）＋進攻端（本檔四員）＋總指揮。

團隊編制（footy agent team，每週一自動執行）：
  守門端（已有，agents/review.py）——防止模型變壞：
    1. 帳本法醫：edge 反向偵測 → 自動關閘
    2. 校準哨兵：Brier 漂移警報
    3. 權重調參：融合權重建議
  進攻端（本檔）——讓模型變好：
    4. 資料稽核員（data_auditor）：資料新鮮度/覆蓋率的確定性檢查——
       資料悄悄爛掉是模型變爛最常見的原因（例：先發投手覆蓋率跌回 0
       就是當初大小盤慘案的根源）
    5. 賽後檢討員（postmortem_agent）：近 7 天模型高信心卻輸的注，
       重用 roles.postmortem 的 LLM 檢討（無金鑰時列清單）
    6. 改進提案人（proposer）：讀本週診斷＋FINDINGS 延後項 backlog，
       提出前三優先改進（LLM；無金鑰時輸出 backlog 原文）
  總指揮（run_team）：串全部 → 單一週報 docs/reports/team-<date>.md

紀律與 review.py 相同：**能傷害生產行為的判斷全是確定性 Python**
（關閘、警報、稽核門檻）；LLM 只產生敘事與建議，建議需人工採納。
另有每月回測回歸（workflow model-backtest-monthly）：重跑足球走地
回測腳本，階梯 LL 對基準漂移 >0.02 即示警——防「程式改壞了驗證結果」。
"""
from __future__ import annotations

from pathlib import Path

# ---------------- 4) 資料稽核員 ----------------

def data_auditor(today: str | None = None) -> dict:
    """資料新鮮度/覆蓋率檢查。回 {"checks": [{name, status, detail}...]}。

    status ∈ PASS/WARN。全部確定性；門檻寫死並註明理由。
    """
    import datetime as _dt

    import pandas as pd
    today_d = _dt.date.fromisoformat(today) if today else _dt.date.today()
    checks = []

    def add(name, ok, detail):
        checks.append({"name": name, "status": "PASS" if ok else "WARN",
                       "detail": detail})

    def _last_date(path, col="date"):
        if not Path(path).exists():
            return None
        try:
            s = pd.to_datetime(pd.read_csv(path)[col], errors="coerce")
            return s.max().date()
        except Exception:  # noqa: BLE001
            return None

    # MLB 資料新鮮度：球季中（4-10 月）超過 3 天沒新比賽 = 抓取管線壞了
    d = _last_date("data/mlb.csv")
    in_season = 4 <= today_d.month <= 10
    if d is None:
        add("MLB 比賽資料", not in_season, "data/mlb.csv 缺失或不可讀")
    else:
        lag = (today_d - d).days
        add("MLB 比賽資料", (not in_season) or lag <= 3,
            f"最後比賽日 {d}（落後 {lag} 天）")
    # 先發投手覆蓋：大小盤慘案根源——投手檔沒更新，總分就會系統性歪
    p = Path("data/mlb_pitchers.csv")
    if p.exists():
        try:
            n = len(pd.read_csv(p))
            add("先發投手檔", n >= 300, f"{n} 位投手（門檻 300）")
        except Exception as e:  # noqa: BLE001
            add("先發投手檔", False, f"不可讀：{e}")
    else:
        add("先發投手檔", not in_season, "缺失")
    # 盤口覆蓋率：近 7 天帳本有賠率的比例——太低代表建站時機/金鑰壞了
    lp = Path("data/mlb_bets.csv")
    if lp.exists() and in_season:
        try:
            df = pd.read_csv(lp)
            df["date"] = pd.to_datetime(df["date"], errors="coerce")
            recent = df[df["date"] >= pd.Timestamp(today_d) - pd.Timedelta(days=7)]
            if len(recent):
                cov = pd.to_numeric(recent["odds"], errors="coerce").notna().mean()
                add("MLB 盤口覆蓋（近7天）", cov >= 0.5,
                    f"{cov:.0%}（{len(recent)} 筆；門檻 50%——低於此代表建站"
                    "時間點盤未開或 the-odds-api 金鑰/額度出問題）")
        except Exception as e:  # noqa: BLE001
            add("MLB 盤口覆蓋（近7天）", False, f"不可讀：{e}")
    # 閘門檔可讀性
    from .review import load_gates
    try:
        g = load_gates()
        add("市場閘門檔", True, f"{sum(1 for v in g.values() if v.get('disabled'))} 個關閉中")
    except Exception as e:  # noqa: BLE001
        add("市場閘門檔", False, str(e))
    return {"checks": checks,
            "n_warn": sum(1 for c in checks if c["status"] == "WARN")}


# ---------------- 5) 賽後檢討員 ----------------

def postmortem_agent(days: int = 7, today: str | None = None) -> dict:
    """近 N 天「模型信心高（隱含機率 ≥60%）卻輸」的注 → LLM 檢討。"""
    import datetime as _dt

    import pandas as pd
    today_d = _dt.date.fromisoformat(today) if today else _dt.date.today()
    upsets = []
    for sport, path in [("足球", "data/bets.csv"), ("MLB", "data/mlb_bets.csv"),
                        ("NBA", "data/nba_bets.csv")]:
        if not Path(path).exists():
            continue
        df = pd.read_csv(path)
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df["odds"] = pd.to_numeric(df["odds"], errors="coerce")
        m = df[(df["result"] == "loss") & df["odds"].notna()
               & (df["odds"] <= 1.67)      # 隱含 ≥60% 的高信心注
               & (df["date"] >= pd.Timestamp(today_d) - pd.Timedelta(days=days))]
        for _, r in m.iterrows():
            upsets.append({"sport": sport, "home": r["home"], "away": r["away"],
                           "market": r["market"], "selection": str(r["selection"]),
                           "odds": float(r["odds"]), "result": "loss",
                           "pl": r.get("pl")})
    out = {"upsets": upsets[:12], "n": len(upsets)}
    if upsets:
        from . import llm, roles
        if llm.available():
            try:
                out["narrative"] = roles.postmortem(upsets[:12])
            except Exception:  # noqa: BLE001
                pass
    return out


# ---------------- 6) 改進提案人 ----------------

def _read_backlog() -> str:
    """從 FINDINGS 抽「延後項」backlog 段落（找不到回空字串）。"""
    p = Path("docs/FINDINGS.md")
    if not p.exists():
        return ""
    txt = p.read_text(encoding="utf-8")
    key = "**延後項"
    i = txt.find(key)
    if i < 0:
        return ""
    seg = txt[i:i + 1200]
    j = seg.find("\n## ")
    return seg[:j] if j > 0 else seg


def proposer(weekly_report: str) -> str:
    """讀本週診斷＋backlog → 前三優先改進提案（LLM；無金鑰回 backlog 原文）。"""
    backlog = _read_backlog()
    from . import llm
    if not llm.available():
        return ("（未設 ANTHROPIC_API_KEY，僅列 backlog）\n\n" + backlog) if backlog else ""
    try:
        out = llm.complete(
            "你是體育博弈模型團隊的技術負責人。根據本週診斷報告與既有 backlog，"
            "提出「下一步最值得做的 3 個改進」，每個含：一句話內容、預期效益、"
            "工作量估計（小/中/大）、驗收方式。用繁體中文、直接列點、不客套。"
            "只提有數據支持或 backlog 已列的項目，不要發明新需求。\n\n"
            f"## 本週診斷\n{weekly_report[:4000]}\n\n## Backlog\n{backlog}",
            max_tokens=600, timeout=90)
        return out.strip() if out else ""
    except Exception:  # noqa: BLE001
        return ""


# ---------------- 總指揮 ----------------

def run_team(today: str = "") -> tuple[dict, str]:
    """跑守門端＋進攻端，回 (結果 dict, Markdown 週報)。"""
    import datetime as _dt

    from . import review as rv
    today = today or _dt.date.today().isoformat()
    res = rv.run_review(today=today)
    base_report = rv.compose_report(res, today)

    audit = data_auditor(today)
    lines = ["", "## 資料稽核員"]
    for c in audit["checks"]:
        icon = "✅" if c["status"] == "PASS" else "⚠️"
        lines.append(f"- {icon} {c['name']}：{c['detail']}")

    pm = postmortem_agent(today=today)
    lines += ["", f"## 賽後檢討員（近 7 天高信心失手 {pm['n']} 注）"]
    for u in pm["upsets"]:
        lines.append(f"- {u['sport']}｜{u['away']} @ {u['home']}｜{u['market']} "
                     f"{u['selection']} @{u['odds']:.2f} → 輸")
    if pm.get("narrative"):
        lines += ["", pm["narrative"].strip()]
    if not pm["upsets"]:
        lines.append("- 無（高信心注全過或無結算）")

    report = base_report + "\n".join(lines) + "\n"
    prop = proposer(report)
    if prop:
        report += "\n## 改進提案人（需人工採納）\n\n" + prop + "\n"
    return ({"review": res, "audit": audit, "postmortem": pm}, report)
