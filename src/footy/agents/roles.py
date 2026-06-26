"""四個 AI agent 角色：賽前分析、新聞/傷停抽取、多代理辯論、風控+賽後檢討。

設計原則：
- agent 只依「傳入的數字/事實」發言，不得編造未提供的傷停或球員資訊（降低幻覺）。
- 一律繁體中文、精簡、中立，研究與教育用途，不慫恿投注。
- 任何 agent 失敗或未設金鑰 → 回 None / 空，呼叫端安全略過。
"""
from __future__ import annotations

from . import llm

SYSTEM = (
    "你是嚴謹的足球數據分析助手。只能根據使用者提供的數字與事實發言，"
    "嚴禁編造未提供的傷停、轉會、球員或新聞資訊。世界盃為中立場。"
    "輸出用繁體中文、精簡務實、語氣中立；屬研究與教育用途，不得慫恿或保證投注獲利。")


def _g(o, name, default=None):
    return getattr(o, name, default)


def match_facts(a) -> str:
    """把 MatchAnalysis 壓成給 LLM 的事實清單（純文字）。"""
    ou = _g(a, "over_under", {}) or {}
    ou25 = ou.get(2.5, {}) if isinstance(ou, dict) else {}
    tops = _g(a, "top_scores", None) or []
    tops_s = "、".join(f"{h}-{aw} {p:.0%}" for (h, aw), p in tops[:4]) if tops else "—"
    lines = [
        f"對戰：{_g(a,'home')} vs {_g(a,'away')}（中立場）",
        f"1X2 機率：主勝 {_g(a,'odds_home',0):.0%} / 和 {_g(a,'odds_draw',0):.0%} / 客勝 {_g(a,'odds_away',0):.0%}",
        f"亞盤讓球線（主隊視角）：{_g(a,'ah_line',0):+g}（{'真實盤口' if _g(a,'ah_from_market') else '模型開盤'}），supremacy {_g(a,'ah_supremacy',0):+.2f}",
        f"大小 2.5：大 {ou25.get('over',0):.0%} / 小 {ou25.get('under',0):.0%}；兩隊進球(是) {_g(a,'btts_yes',0):.0%}",
        f"最可能比分：{tops_s}",
        f"Elo：主 {_g(a,'elo_home',0):.0f} / 客 {_g(a,'elo_away',0):.0f}",
        f"近況：主 {_g(a,'home_form','') or '—'}；客 {_g(a,'away_form','') or '—'}",
        f"交手：{_g(a,'h2h','') or '—'}",
        f"風格：主 {_g(a,'home_style','') or '—'}；客 {_g(a,'away_style','') or '—'}",
        f"陣型：主 {_g(a,'home_formation','') or '未填'}／客 {_g(a,'away_formation','') or '未填'}",
        f"球員狀態：主『{_g(a,'player_note_home','') or '—'}』／客『{_g(a,'player_note_away','') or '—'}』",
        f"資料充足度：{_g(a,'data_support','') or '—'}",
    ]
    return "\n".join(lines)


# ---------------- 1) 賽前分析撰寫 ----------------
def preview(a, complete=llm.complete) -> str | None:
    """把模型數字寫成白話賽前分析（120~200 字）。未設金鑰回 None。"""
    if not llm.available():
        return None
    home, away = _g(a, "home", ""), _g(a, "away", "")
    prompt = (
        f"這場比賽是「{home}」對「{away}」。務必只用這兩個隊名，"
        "嚴禁提到任何其他球隊（如出現別隊名即為錯誤）。\n"
        "根據以下模型輸出，寫一段 120~200 字的繁體中文賽前分析，"
        "點出：誰較被看好與幅度、預期比分樣貌、大小球與讓盤傾向、以及一個值得注意的變數。"
        "只用下列數字，不要編造傷停或新聞。不要逐項複述機率，要有解讀。\n\n"
        f"{match_facts(a)}")
    try:
        return complete(prompt, system=SYSTEM, max_tokens=1200)
    except Exception:  # noqa: BLE001
        return None


# ---------------- 2) 新聞/傷停/先發抽取 ----------------
def extract_news(home: str, away: str, news_text: str,
                 complete_json=llm.complete_json):
    """從『使用者提供的新聞文字』抽出可餵進模型的調整。不上網、不臆測。

    回傳 {"home":{"missing":int,"formation":str,"note":str}, "away":{...}} 或 None。
    """
    if not llm.available() or not (news_text or "").strip():
        return None
    prompt = (
        f"以下是關於 {home}(主) vs {away}(客) 的新聞文字。只根據文字內容，"
        "抽出每隊『缺陣主力人數(missing, 整數)』、『先發陣型(formation, 如 4-3-3，沒提到留空字串)』、"
        "與一句重點(note)。沒有資訊就 missing=0、formation=\"\"。只輸出 JSON：\n"
        '{"home":{"missing":0,"formation":"","note":""},'
        '"away":{"missing":0,"formation":"","note":""}}\n\n'
        f"新聞：\n{news_text[:6000]}")
    try:
        return complete_json(prompt, system=SYSTEM, max_tokens=700, temperature=0.1)
    except Exception:  # noqa: BLE001
        return None


# ---------------- 3) 多代理預測辯論/集成 ----------------
_LENSES = [
    ("數據派", "只看 Elo、機率、supremacy 等量化指標"),
    ("狀態派", "著重近況 form 與交手 h2h"),
    ("對位派", "著重風格、陣型、攻守傾向的相剋"),
]


def debate(a, complete=llm.complete, complete_json=llm.complete_json,
           lenses=_LENSES):
    """多位分析師各持觀點，裁判綜合。回傳 dict（含 analysts 與 verdict）或 None。"""
    if not llm.available():
        return None
    facts = match_facts(a)
    home, away = _g(a, "home", ""), _g(a, "away", "")
    rule = (f"這場是「{home}」對「{away}」，只能用這兩個隊名，不得提到其他球隊。")
    analysts = []
    for role, lens in lenses:
        p = (f"你是「{role}」分析師，{lens}。{rule}根據以下數字，用 1~2 句給出你的傾向"
             f"（主勝/和/客勝其一）與理由，不得編造資訊：\n\n{facts}")
        try:
            txt = complete(p, system=SYSTEM, max_tokens=600, temperature=0.5)
        except Exception:  # noqa: BLE001
            return None
        analysts.append({"role": role, "view": txt})
    panel = "\n".join(f"- {x['role']}：{x['view']}" for x in analysts)
    judge_prompt = (
        "你是裁判。綜合以下分析師意見與原始數字，輸出 JSON："
        '{"lean":"主勝|和|客勝","confidence":"低|中|高","summary":"40字內結論"}。'
        f"{rule}若分析師與數據矛盾，以數據為準。\n\n"
        f"數字：\n{facts}\n\n分析師：\n{panel}")
    try:
        verdict = complete_json(judge_prompt, system=SYSTEM, max_tokens=600,
                                temperature=0.2)
    except Exception:  # noqa: BLE001
        verdict = None
    return {"analysts": analysts, "verdict": verdict}


# ---------------- 4) 風控 + 賽後檢討 ----------------
def risk_review(picks: list[dict], complete=llm.complete) -> str | None:
    """檢視當日推薦（每筆 dict 含 home/away/market/selection/odds/edge），給風控提醒。"""
    if not llm.available() or not picks:
        return None
    lines = [f"- {p.get('home')} vs {p.get('away')}｜{p.get('market')} "
             f"{p.get('selection')}{(' '+str(p.get('line'))) if p.get('line') not in (None,'','nan') else ''}"
             f"｜賠率 {p.get('odds','—')}" for p in picks[:40]]
    prompt = (
        "以下是今日模型推薦清單（均注 1 單位）。從風控角度用繁體中文條列 3~5 點提醒："
        "是否過度集中同隊/同時段、是否多筆高賠率冷門、是否有相關性高的重複曝險、"
        "整體建議曝險。務實中立，不保證獲利。\n\n" + "\n".join(lines))
    try:
        return complete(prompt, system=SYSTEM, max_tokens=900)
    except Exception:  # noqa: BLE001
        return None


def postmortem(rows: list[dict], complete=llm.complete) -> str | None:
    """賽後檢討：每筆 dict 含 home/away/market/selection/odds/result/pl。"""
    if not llm.available() or not rows:
        return None
    lines = [f"- {r.get('home')} vs {r.get('away')}｜{r.get('market')} {r.get('selection')}"
             f"｜賠率 {r.get('odds','—')}｜{r.get('result')}｜損益 {r.get('pl','—')}"
             for r in rows[:40]]
    prompt = (
        "以下是近期已結算的模型推薦與結果。用繁體中文做簡短賽後檢討："
        "命中與失誤的型態（哪類盤口/哪種對戰較準或較差）、可能原因、"
        "對模型或選注的 2~3 點具體建議。中立務實。\n\n" + "\n".join(lines))
    try:
        return complete(prompt, system=SYSTEM, max_tokens=1000)
    except Exception:  # noqa: BLE001
        return None
