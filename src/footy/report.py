"""把 MatchPrediction 渲染成 console / Markdown / HTML（預測站風格）。"""
from __future__ import annotations

import datetime as _dt
import html
from pathlib import Path

from .predict import MatchPrediction
from .i18n import zh, group_zh


def _tpe_label(m, with_date: bool = True) -> str:
    """場次開球時間轉台北時間（UTC+8）短字串。無時間資料則退回場地日期 MM-DD。"""
    from . import worldcup as wc
    dt = wc.kickoff_taipei(getattr(m, "date", ""), getattr(m, "time", ""))
    if not dt:
        return m.date[5:] if getattr(m, "date", "") else ""
    return f"{dt.month}/{dt.day} {dt:%H:%M}" if with_date else f"{dt:%H:%M}"


def _slot_zh(token: str) -> str:
    """淘汰賽槽位碼轉中文：1A→A組#1、3A/B/C→A/B/C組第三、W74→#74勝者。"""
    import re
    m = re.match(r"^([123])([A-L])$", token)
    if m:
        rank = {"1": "首名", "2": "次名", "3": "第三"}[m.group(1)]
        return f"{m.group(2)}組{rank}"
    m = re.match(r"^3([A-L/]+)$", token)
    if m:
        return f"{m.group(1)}組第三"
    m = re.match(r"^W(\d+)$", token)
    if m:
        return f"#{m.group(1)}勝者"
    return zh(token)


def _score_str(s: tuple[int, int]) -> str:
    return f"{s[0]}-{s[1]}"


def _modal_score_str(p: MatchPrediction) -> str:
    """最可能比分 + 其機率，例如 1-1（13%）。"""
    prob = p.correct_scores[0][1] if p.correct_scores else 0.0
    return f"{_score_str(p.predicted_score)}（{prob:.0%}）"


# ---------------- Console ----------------
def render_console(p: MatchPrediction) -> str:
    cs = "  ".join(f"{_score_str(s)} {prob:.0%}" for s, prob in p.correct_scores[:3])
    ou25 = p.over_under.get(2.5, {})
    return (
        f"⚽ {p.home} vs {p.away}\n"
        f"   1X2     : 主勝 {p.p_home:.0%} | 和 {p.p_draw:.0%} | 客勝 {p.p_away:.0%}\n"
        f"   預期進球: {p.exp_home_goals:.2f} - {p.exp_away_goals:.2f}"
        f"   預測比分: {_modal_score_str(p)}\n"
        f"   大小2.5 : 大 {ou25.get('over',0):.0%} / 小 {ou25.get('under',0):.0%}"
        f"   BTTS    : 是 {p.btts_yes:.0%} / 否 {p.btts_no:.0%}\n"
        f"   正確比分: {cs}\n"
        f"   狀態    : {p.home}={p.home_form or '-'}  {p.away}={p.away_form or '-'}  {p.h2h}\n"
        f"   💡 Tip  : {p.tip}（信心：{p.confidence}）"
    )


# ---------------- Markdown ----------------
def render_markdown(preds: list[MatchPrediction], title: str = "足球預測") -> str:
    today = _dt.date.today().isoformat()
    lines = [f"# {title}", f"_產生於 {today}_", "",
             "> 純機率預測，非投注建議；模型未必勝過市場，詳見 FINDINGS.md。", ""]
    for p in preds:
        ou25 = p.over_under.get(2.5, {})
        cs = ", ".join(f"{_score_str(s)} ({prob:.0%})" for s, prob in p.correct_scores)
        lines += [
            f"## {p.home} vs {p.away}",
            "",
            f"| 市場 | 預測 |",
            f"|---|---|",
            f"| 1X2 | 主勝 **{p.p_home:.0%}** / 和 **{p.p_draw:.0%}** / 客勝 **{p.p_away:.0%}** |",
            f"| 預期進球 | {p.exp_home_goals:.2f} – {p.exp_away_goals:.2f} |",
            f"| 預測比分 | **{_modal_score_str(p)}** |",
            f"| 大小球 2.5 | 大 {ou25.get('over',0):.0%} / 小 {ou25.get('under',0):.0%} |",
            f"| BTTS | 是 {p.btts_yes:.0%} / 否 {p.btts_no:.0%} |",
            f"| 正確比分 Top5 | {cs} |",
            f"| 近期狀態 | {p.home}: {p.home_form or '-'} · {p.away}: {p.away_form or '-'} |",
            f"| H2H | {p.h2h or '-'} |",
            f"| 💡 Tip | {p.tip}（信心：{p.confidence}） |",
            "",
        ]
    return "\n".join(lines)


# ---------------- HTML（預測站風格） ----------------
_CSS = """
:root{--bg:#0f1419;--card:#1a2129;--accent:#21c07a;--muted:#8a97a6;--line:#2a333d;--warn:#e0b341}
*{box-sizing:border-box}body{margin:0;font-family:system-ui,-apple-system,"Noto Sans TC",sans-serif;
background:var(--bg);color:#e6edf3}
.wrap{max-width:980px;margin:0 auto;padding:24px}
h1{font-size:24px;margin:0 0 4px}.sub{color:var(--muted);font-size:13px;margin-bottom:18px}
.disc{background:#2a2410;border:1px solid var(--warn);color:#f0d98c;padding:10px 14px;border-radius:8px;
font-size:13px;margin-bottom:20px}
.card{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:16px 18px;margin-bottom:16px}
.head{display:flex;justify-content:space-between;align-items:center;margin-bottom:12px}
.teams{font-size:18px;font-weight:700}.tip{background:var(--accent);color:#04130c;font-weight:700;
padding:4px 10px;border-radius:20px;font-size:12px}
.bar{display:flex;height:26px;border-radius:6px;overflow:hidden;margin:8px 0;font-size:12px;font-weight:600}
.bar>div{display:flex;align-items:center;justify-content:center;color:#04130c}
.h{background:#21c07a}.d{background:#5b6a7a}.a{background:#e07a5f}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:10px;margin-top:10px}
.box{background:#11161c;border:1px solid var(--line);border-radius:8px;padding:10px}
.box .k{color:var(--muted);font-size:11px;text-transform:uppercase;letter-spacing:.04em}
.box .v{font-size:15px;font-weight:700;margin-top:3px}
.cs span{display:inline-block;background:#11161c;border:1px solid var(--line);border-radius:6px;
padding:3px 8px;margin:2px 4px 2px 0;font-size:12px}
.form{font-family:ui-monospace,monospace;letter-spacing:2px}.W{color:#21c07a}.D{color:#c9b458}.L{color:#e07a5f}
.foot{color:var(--muted);font-size:12px;text-align:center;margin-top:24px}
.nav{position:sticky;top:0;z-index:9;background:rgba(15,20,25,.92);backdrop-filter:blur(8px);
border-bottom:1px solid var(--line);display:flex;gap:6px;padding:10px 14px;margin:-24px -24px 16px}
.nav a{color:var(--muted);text-decoration:none;font-size:14px;font-weight:700;padding:6px 12px;border-radius:8px}
.nav a.on{background:var(--accent);color:#04130c}.nav a:not(.on):active{background:#1c242d}
/* 淘汰賽對陣圖 */
.bracket{display:flex;gap:14px;overflow-x:auto;padding:6px 2px 14px}
.bcol{flex:0 0 auto;min-width:150px;display:flex;flex-direction:column;justify-content:space-around;gap:8px}
.bcol h4{font-size:12px;color:var(--muted);text-align:center;margin:0 0 2px}
.bm{background:#11161c;border:1px solid var(--line);border-radius:8px;padding:7px 9px;font-size:12px}
.bm .t{display:flex;justify-content:space-between;gap:6px;padding:1px 0}
.bm .t.win{color:#7be0b0;font-weight:700}.bm .res{color:var(--warn);font-weight:700}
.fctrl{display:flex;gap:8px;margin:8px 0;font-size:13px}
.fctrl button{background:#11161c;color:#cdd9e5;border:1px solid var(--line);border-radius:8px;
padding:6px 12px;font-weight:700;cursor:pointer}.fctrl button.on{background:var(--accent);color:#04130c}
"""


def _form_html(form: str) -> str:
    return "".join(f'<span class="{c}">{c}</span>' for c in form) or "-"


def _match_card(p: MatchPrediction) -> str:
    ou25 = p.over_under.get(2.5, {})
    cs = "".join(f"<span>{_score_str(s)} · {prob:.0%}</span>" for s, prob in p.correct_scores)
    h, d, a = p.p_home, p.p_draw, p.p_away
    return f"""
    <div class="card">
      <div class="head">
        <div class="teams">{html.escape(p.home)} <span style="color:#8a97a6">vs</span> {html.escape(p.away)}</div>
        <div class="tip">💡 {html.escape(p.tip)}</div>
      </div>
      <div class="bar">
        <div class="h" style="width:{h*100:.1f}%">{h:.0%}</div>
        <div class="d" style="width:{d*100:.1f}%">{d:.0%}</div>
        <div class="a" style="width:{a*100:.1f}%">{a:.0%}</div>
      </div>
      <div class="grid">
        <div class="box"><div class="k">預測比分</div><div class="v">{_modal_score_str(p)}</div></div>
        <div class="box"><div class="k">預期進球</div><div class="v">{p.exp_home_goals:.2f} – {p.exp_away_goals:.2f}</div></div>
        <div class="box"><div class="k">大/小 2.5</div><div class="v">{ou25.get('over',0):.0%} / {ou25.get('under',0):.0%}</div></div>
        <div class="box"><div class="k">BTTS 是/否</div><div class="v">{p.btts_yes:.0%} / {p.btts_no:.0%}</div></div>
        <div class="box"><div class="k">{html.escape(p.home)} 狀態</div><div class="v form">{_form_html(p.home_form)}</div></div>
        <div class="box"><div class="k">{html.escape(p.away)} 狀態</div><div class="v form">{_form_html(p.away_form)}</div></div>
      </div>
      <div class="cs" style="margin-top:10px">正確比分 Top5：{cs}</div>
      <div style="color:#8a97a6;font-size:12px;margin-top:8px">{html.escape(p.h2h or '')}</div>
    </div>"""


def render_html(preds: list[MatchPrediction], title: str = "足球預測") -> str:
    today = _dt.date.today().isoformat()
    cards = "\n".join(_match_card(p) for p in preds)
    return f"""<!doctype html><html lang="zh-Hant"><head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>{html.escape(title)}</title><style>{_CSS}</style></head>
<body><div class="wrap">
  <h1>⚽ {html.escape(title)}</h1>
  <div class="sub">產生於 {today} · 由 Dixon–Coles 機率模型計算</div>
  <div class="disc">⚠️ 純機率預測，非投注建議。模型未必勝過市場盤口；投注有風險，請量力而為並遵守當地法律。</div>
  {cards}
  <div class="foot">Generated by footy · 研究與教育用途</div>
</div></body></html>"""


def write_html(preds: list[MatchPrediction], out_path: str | Path,
               title: str = "足球預測") -> Path:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(render_html(preds, title), encoding="utf-8")
    return out_path


# ---------------- 整季模擬渲染 ----------------
def render_season_console(sim) -> str:
    df = sim.table()
    lines = [f"整季蒙地卡羅模擬（{sim.n_sims:,} 次）",
             f"{'隊伍':<16}{'冠軍%':>7}{'前四%':>7}{'前六%':>7}{'降級%':>7}{'預期分':>7}{'名次':>6}"]
    for _, r in df.iterrows():
        lines.append(
            f"{r['team']:<16}{r['冠軍%']:>7}{r['前四%']:>7}{r['前六%']:>7}"
            f"{r['降級%']:>7}{r['預期積分']:>7}{r['預期名次']:>6}")
    return "\n".join(lines)


def _heat(pct: float, hue: str = "120") -> str:
    """機率→背景色（0%淡、100%深）。hue:120綠/40黃/0紅。"""
    a = min(1.0, pct / 100.0)
    return f"background:hsla({hue},70%,45%,{0.12 + 0.7 * a:.2f})"


def render_season_html(sim, title: str = "整季模擬") -> str:
    import datetime as _dt
    df = sim.table()
    today = _dt.date.today().isoformat()
    rows = []
    for rank, (_, r) in enumerate(df.iterrows(), 1):
        rows.append(
            f"<tr><td class='rk'>{rank}</td><td class='tm'>{html.escape(str(r['team']))}</td>"
            f"<td style=\"{_heat(r['冠軍%'],'48')}\">{r['冠軍%']:.1f}%</td>"
            f"<td style=\"{_heat(r['前四%'],'140')}\">{r['前四%']:.1f}%</td>"
            f"<td style=\"{_heat(r['前六%'],'140')}\">{r['前六%']:.1f}%</td>"
            f"<td style=\"{_heat(r['降級%'],'0')}\">{r['降級%']:.1f}%</td>"
            f"<td class='num'>{r['預期積分']:.1f}</td>"
            f"<td class='num'>{r['預期名次']:.1f}</td></tr>")
    body = "\n".join(rows)
    return f"""<!doctype html><html lang="zh-Hant"><head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>{html.escape(title)}</title><style>{_CSS}
table{{width:100%;border-collapse:collapse;font-size:14px;background:var(--card);border-radius:12px;overflow:hidden}}
th,td{{padding:9px 10px;text-align:center;border-bottom:1px solid var(--line)}}
th{{background:#11161c;color:var(--muted);font-size:12px;text-transform:uppercase;letter-spacing:.03em}}
td.rk{{color:var(--muted)}}td.tm{{text-align:left;font-weight:700}}td.num{{color:#cdd9e5}}
</style></head>
<body><div class="wrap">
  <h1>🏆 {html.escape(title)}</h1>
  <div class="sub">產生於 {today} · {sim.n_sims:,} 次蒙地卡羅模擬 · Dixon–Coles 抽樣</div>
  <div class="disc">⚠️ 純機率模擬，依目前模型強度推估，未含轉會/傷停等未來變動；非投注建議。</div>
  <table>
    <thead><tr><th>#</th><th>隊伍</th><th>冠軍</th><th>前四</th><th>前六</th><th>降級</th><th>預期積分</th><th>預期名次</th></tr></thead>
    <tbody>{body}</tbody>
  </table>
  <div class="foot">Generated by footy · 研究與教育用途</div>
</div></body></html>"""


def write_season_html(sim, out_path: str | Path, title: str = "整季模擬") -> Path:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(render_season_html(sim, title), encoding="utf-8")
    return out_path


# ---------------- 單場深度分析（世界盃 UI） ----------------
def render_analysis_console(a) -> str:
    hz, az = zh(a.home), zh(a.away)
    L = []
    L.append(f"⚽ {hz} vs {az}" + ("（中立場）" if a.neutral else ""))
    if a.data_support < 0.6:
        L.append("  ⚠️ 兩隊近年交手樣本偏少，預測可靠度較低，僅供參考")
    L.append(f"  AI 預測比分 {a.predicted_score[0]}-{a.predicted_score[1]}"
             f"（機率 {a.predicted_score_prob:.0%}；總進球 {a.total_goals}）· xG {a.xg_low}-{a.xg_high}")
    L.append(f"  1X2     主 @{a.odds_home:.2f} / 和 @{a.odds_draw:.2f} / 客 @{a.odds_away:.2f}"
             f"  主勝 {a.p_home:.0%} / 和 {a.p_draw:.0%} / 客勝 {a.p_away:.0%}")
    ouo = a.ou_odds or {}
    for ln in (1.5, 2.5, 3.5):
        d = a.over_under[ln]
        oo = ouo.get(ln, (0, 0))
        L.append(f"  大小{ln}  大 @{oo[0]:.2f} / 小 @{oo[1]:.2f}  "
                 f"大 {d['over']:.0%} → {'買大' if d['over']>=0.5 else '買小'}")
    L.append(f"  BTTS    是 {a.btts_yes:.0%} → {'買是' if a.btts_yes>=0.5 else '買否'}"
             f"（{hz} 進球 {a.p_home_scores:.0%} × {az} 進球 {a.p_away_scores:.0%}）")
    L.append(f"  亞盤     {hz} {a.ah_line:+g} @{a.ah_home_odds:.2f} / "
             f"{az} {-a.ah_line:+g} @{a.ah_away_odds:.2f} → {a.ah_reco}")
    c = a.corners
    L.append(f"  角球     合計 {c.total}（{hz} {c.home} / {az} {c.away}）"
             f" 估線 {c.line} {c.recommend}")
    k = a.cards
    L.append(f"  黃牌     合計 {k.total} 估線 {k.line} {k.recommend}")
    L.append(f"  上半場   主 {a.fh_home:.0%} / 平 {a.fh_draw:.0%} / 客 {a.fh_away:.0%}"
             f"  大0.5 {a.fh_over[0.5]:.0%} / 大1.5 {a.fh_over[1.5]:.0%}")
    L.append(f"  因子     Elo {a.elo_home:.0f} vs {a.elo_away:.0f}｜陣型 "
             f"{a.home_formation or '—'} vs {a.away_formation or '—'}｜"
             f"狀態 {a.home_form or '-'}/{a.away_form or '-'}")
    return "\n".join(L)


def _reco_color(reco: str) -> str:
    if "大" in reco or "是" in reco:
        return "#21c07a"
    if "小" in reco or "否" in reco:
        return "#e07a5f"
    return "#e0b341"


def _conf_tier(p: float, support: float = 1.0) -> tuple[int, str]:
    """可信度 = 決斷度 × 資料量支撐（不是機率本身）。

    決斷度 = 推薦方機率離 50/50 多遠（2c-1）；資料少的對戰會被扣分。
    回傳 星等(1-5) 與分級標籤。
    """
    c = max(p, 1 - p)
    decisiveness = max(0.0, 2 * c - 1.0)          # 0（擲硬幣）~1（壓倒性）
    score = decisiveness * max(0.0, min(1.0, support))
    if score >= 0.55:
        return 5, "極高"
    if score >= 0.40:
        return 4, "高"
    if score >= 0.25:
        return 3, "中"
    if score >= 0.12:
        return 2, "偏低"
    return 1, "低"


def _conf_text(p: float, support: float = 1.0) -> str:
    """純文字可信度（console 用）：★星 + 分級。"""
    stars, tier = _conf_tier(p, support)
    return f"{'★' * stars}{'☆' * (5 - stars)} {tier}"


def _conf_label(p: float = 0.0, support: float = 1.0) -> str:
    """（已停用）為了直觀，每個欄位不再各自顯示可信度星等——只保留機率/賠率。
    改在資料太少時，整場顯示一次提醒（見 render_analysis_html / console）。"""
    return ""


_FORMS = ["", "4-3-3", "4-2-3-1", "4-4-2", "4-1-4-1", "4-5-1", "4-1-2-1-2",
          "4-3-1-2", "3-5-2", "3-4-3", "3-4-1-2", "3-1-4-2", "5-3-2", "5-4-1", "5-2-3"]


def _reanalyze_bar(a) -> str:
    """分析頁上的「調整後重新分析」互動列：陣型下拉、缺陣、盤口線 → 送 /analyze。"""
    opts = "".join(f"<option value='{f}'>" for f in _FORMS if f)
    hf, af = a.home_formation or "", a.away_formation or ""
    return f"""
    <form action="/analyze" method="get" class="rebar">
      <input type="hidden" name="home" value="{html.escape(a.home)}">
      <input type="hidden" name="away" value="{html.escape(a.away)}">
      <input type="hidden" name="neutral" value="{1 if a.neutral else 0}">
      <datalist id="rf">{opts}</datalist>
      <span>陣型 {html.escape(zh(a.home))}<input name="hf" list="rf" value="{html.escape(hf)}" placeholder="如 4-3-3"></span>
      <span>陣型 {html.escape(zh(a.away))}<input name="af" list="rf" value="{html.escape(af)}" placeholder="如 5-4-1"></span>
      <span>盤口讓球線<input name="ah" placeholder="如 -1.5（留空=自動）"></span>
      <button type="submit">重新分析</button>
    </form>"""


def _low_data_note_html(support: float) -> str:
    if support < 0.6:
        return ('<div class="disc" style="background:#2a1410;border-color:#e07a5f;'
                'color:#f0b8a0">⚠️ 兩隊近年交手樣本偏少，預測可靠度較低，僅供參考。</div>')
    return ""


def _odds(o: float) -> str:
    return f"{o:.2f}" if o and o < 50 else "—"


def _ou_reco(over_p: float) -> tuple[str, str]:
    """大小球建議＋顏色：接近 50% 時用「接近五五/略偏」避免與低比分眾數矛盾的觀感。"""
    if over_p >= 0.62:
        return "買大", "#21c07a"
    if over_p >= 0.55:
        return "略偏大", "#7fae6a"
    if over_p > 0.45:
        return "接近五五", "#e0b341"
    if over_p > 0.38:
        return "略偏小", "#cf8a6a"
    return "買小", "#e07a5f"


def _ou_box(line, d, odds=None, support=1.0):
    reco, col = _ou_reco(d["over"])
    price = ""
    if odds:
        price = (f"<div class='price'>大 {_odds(odds[0])} ／ 小 {_odds(odds[1])}</div>")
    return (f"<div class='box'><div class='k'>線 {line}</div>"
            f"<div class='v'>大 {d['over']:.0%}</div>"
            f"{price}"
            f"<div style='color:{col};font-weight:700;margin-top:4px'>{reco}</div></div>")


def _fh_ou_box(line, p, support=1.0):
    reco, col = _ou_reco(p)
    return (f"<div class='box'><div class='k'>上半場線 {line}</div>"
            f"<div class='v'>大 {p:.0%}</div>"
            f"<div style='color:{col};font-weight:700;margin-top:4px'>{reco}</div></div>")


def render_ai_blocks(a, include_debate: bool = True) -> str:
    """賽前分析 + 多方辯論卡（隨選；未設金鑰回空字串）。每張卡各 1~數次 LLM 呼叫。"""
    try:
        from .agents import llm, roles
    except Exception:  # noqa: BLE001
        return ""
    if not llm.available():
        return ""
    out = []
    try:
        txt = roles.preview(a)
        if txt:
            out.append(f"<div class='card'><div class='sec'>🤖 AI 賽前分析</div>"
                       f"<div class='small' style='color:#cdd9e5;line-height:1.7;"
                       f"white-space:pre-line'>{html.escape(txt)}</div></div>")
    except Exception:  # noqa: BLE001
        pass
    if include_debate:
        try:
            d = roles.debate(a)
            if d:
                rows = "".join(
                    f"<div class='small' style='margin:4px 0'><b>{html.escape(x['role'])}</b>："
                    f"{html.escape(x['view'])}</div>" for x in d.get("analysts", []))
                v = d.get("verdict") or {}
                verdict = ""
                if v:
                    verdict = (f"<div class='reco' style='font-size:15px;margin-top:8px'>"
                               f"裁判：{html.escape(str(v.get('lean','?')))}"
                               f"（信心 {html.escape(str(v.get('confidence','?')))}）— "
                               f"{html.escape(str(v.get('summary','')))}</div>")
                out.append(f"<div class='card'><div class='sec'>🧠 AI 多方辯論</div>"
                           f"{rows}{verdict}"
                           f"<div class='small' style='color:var(--muted);margin-top:6px'>"
                           f"多角度檢視，與數據矛盾時以模型數據為準；非投注建議。</div></div>")
        except Exception:  # noqa: BLE001
            pass
    return "".join(out)


def render_analysis_html(a, title: str = "單場分析", back_href: str | None = None,
                         interactive: bool = False, ai_html: str = "") -> str:
    import datetime as _dt
    today = _dt.date.today().isoformat()
    venue = "中立場" if a.neutral else "主客場"
    rebar = _reanalyze_bar(a) if interactive else ""
    sup = a.data_support
    ou_odds = a.ou_odds or {}
    ou_boxes = "".join(_ou_box(ln, a.over_under[ln], ou_odds.get(ln), sup)
                       for ln in (1.5, 2.5, 3.5))
    fh_ou = "".join(_fh_ou_box(ln, p, sup) for ln, p in a.fh_over.items())
    top4 = "".join(
        f"<span class='ts'>{s[0]}-{s[1]} <b>{p*100:.0f}%</b></span>"
        for s, p in (a.top_scores or [])[:4])
    c, k = a.corners, a.cards
    ah_col = _reco_color("大")
    h, d, aw = a.p_home, a.p_draw, a.p_away
    conf_1x2 = max(h, d, aw)
    conf_btts = max(a.btts_yes, 1 - a.btts_yes)
    conf_ah = max(a.ah_cover_prob, 1 - a.ah_cover_prob)
    conf_fh = max(a.fh_home, a.fh_draw, a.fh_away)
    hf = a.home_formation or "—（賽前公布）"
    af = a.away_formation or "—（賽前公布）"
    nav = (f'<a class="back" href="{back_href}">← 返回首頁</a>' if back_href else "")
    return f"""<!doctype html><html lang="zh-Hant"><head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>{html.escape(title)}</title><style>{_CSS}
.sec{{font-weight:700;margin:18px 0 8px;font-size:15px}}
.reco{{font-size:20px;font-weight:800;margin-top:6px}}
.small{{color:var(--muted);font-size:12px}}
.split{{display:grid;grid-template-columns:1fr 1fr;gap:14px}}
.back{{display:inline-block;color:var(--accent);text-decoration:none;font-size:14px;margin-bottom:10px}}
.cf{{margin-top:5px;font-size:11px;color:var(--muted)}}
.cf .stars{{color:#f0c050;letter-spacing:1px}}.cf .dim{{color:#3a4350}}
.cf .pct{{color:#7be0b0;font-weight:700}}
.price{{margin-top:3px;font-size:13px;color:#cdd9e5;font-weight:700;font-variant-numeric:tabular-nums}}
.line-row{{display:flex;justify-content:space-between;gap:10px;margin:6px 0;
font-size:15px;font-weight:700;font-variant-numeric:tabular-nums}}
.line-row .o{{color:#7be0b0}}
.topscores{{margin-top:8px;font-size:13px;color:var(--muted)}}
.ts{{display:inline-block;background:#11161c;border:1px solid var(--line);border-radius:6px;
padding:3px 9px;margin:2px 6px 2px 0;color:#e6edf3;font-variant-numeric:tabular-nums}}
.ts b{{color:#7be0b0}}
.rebar{{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:10px 12px;
margin-bottom:14px;display:flex;flex-wrap:wrap;gap:10px;align-items:flex-end}}
.rebar span{{display:flex;flex-direction:column;font-size:11px;color:var(--muted);gap:3px}}
.rebar input{{background:#11161c;color:#e6edf3;border:1px solid var(--line);border-radius:6px;padding:6px;width:130px}}
.rebar button{{background:var(--accent);color:#04130c;border:0;border-radius:8px;padding:8px 16px;font-weight:800}}
@media(max-width:560px){{.split{{grid-template-columns:1fr}}}}
</style></head>
<body><div class="wrap">
  {nav}
  <h1>⚽ {html.escape(zh(a.home))} <span style="color:#8a97a6">vs</span> {html.escape(zh(a.away))}</h1>
  <div class="sub">{today} · {venue} · 蒙地卡羅 {a.n_sims:,} 次 + Poisson · Dixon–Coles</div>
  <div class="disc">⚠️ 純機率分析，非投注建議。角球/黃牌為先驗近似。投注有風險。</div>
  {_low_data_note_html(a.data_support)}
  {rebar}
  {ai_html}

  <div class="card">
    <div class="head"><div class="teams">AI 最可能比分 {a.predicted_score[0]}-{a.predicted_score[1]}
      <span class="small">（僅 {a.predicted_score_prob:.0%}，眾數非平均）</span></div>
      <div class="small">期望總進球 <b style="color:#cdd9e5">{a.total_goals}</b> · xG {a.xg_low}–{a.xg_high}</div></div>
    <div class="topscores">最可能比分 Top4：{top4}</div>
    <div class="small" style="margin:4px 0 8px;color:var(--muted)">
      註：單一比分機率都偏低，大小球看的是<b>期望總進球 {a.total_goals}</b> 與整體分布，故方向可能不同。</div>
    <div class="bar">
      <div class="h" style="width:{h*100:.1f}%">{h:.0%}</div>
      <div class="d" style="width:{d*100:.1f}%">{d:.0%}</div>
      <div class="a" style="width:{aw*100:.1f}%">{aw:.0%}</div>
    </div>
    <div class="line-row"><span>主勝 <span class="o">{_odds(a.odds_home)}</span></span>
      <span>和 <span class="o">{_odds(a.odds_draw)}</span></span>
      <span>客勝 <span class="o">{_odds(a.odds_away)}</span></span></div>
    <div class="small">模型公平賠率（無水位）</div>
    {_conf_label(conf_1x2, sup)}
  </div>

  <div class="sec">⚽ 大小球</div>
  <div class="grid">{ou_boxes}</div>

  <div class="split">
    <div class="card">
      <div class="sec">🥅 兩隊都進球 BTTS</div>
      <div class="reco" style="color:{_reco_color('買是' if a.btts_yes>=0.5 else '買否')}">
        {'買是' if a.btts_yes>=0.5 else '買否'}（{a.btts_yes:.0%}）</div>
      <div class="small">{html.escape(zh(a.home))} 進球 {a.p_home_scores:.0%} × {html.escape(zh(a.away))} 進球 {a.p_away_scores:.0%}</div>
      {_conf_label(conf_btts, sup)}
    </div>
    <div class="card">
      <div class="sec">⚖️ 亞盤讓球（{'真實盤口' if a.ah_from_market else '模型開盤'}）</div>
      <div class="line-row">
        <span>{html.escape(zh(a.home))} {a.ah_line:+g} <span class="o">{_odds(a.ah_home_odds)}</span></span>
        <span>{html.escape(zh(a.away))} {-a.ah_line:+g} <span class="o">{_odds(a.ah_away_odds)}</span></span>
      </div>
      <div class="reco" style="font-size:16px;color:{_reco_color('讓')}">{html.escape(a.ah_reco)}</div>
      <div class="small">supremacy(xG差) {a.ah_supremacy:+.2f}　公平賠率（無水位）</div>
      {_conf_label(conf_ah, sup)}
    </div>
  </div>

  <div class="split">
    <div class="card">
      <div class="sec">🚩 角球預測</div>
      <div class="head"><div>{html.escape(zh(a.home))} <b>{c.home}</b></div>
        <div style="font-size:22px;font-weight:800;color:var(--warn)">{c.total}</div>
        <div><b>{c.away}</b> {html.escape(zh(a.away))}</div></div>
      <div class="reco" style="color:{_reco_color(c.recommend)}">{c.recommend} {c.line}</div>
      <div class="small">模型估線 {c.line}（點估計 {c.edge_vs_line:+.1f}）· 信心 {c.confidence:.0%}</div>
    </div>
    <div class="card">
      <div class="sec">🟨 黃牌預測</div>
      <div class="head"><div>{html.escape(zh(a.home))} <b>{k.home}</b></div>
        <div style="font-size:22px;font-weight:800;color:var(--warn)">{k.total}</div>
        <div><b>{k.away}</b> {html.escape(zh(a.away))}</div></div>
      <div class="reco" style="color:{_reco_color(k.recommend)}">{k.recommend} {k.line}</div>
      <div class="small">模型估線 {k.line} · 信心 {k.confidence:.0%}</div>
    </div>
  </div>

  <div class="sec">⏱️ 上半場走向</div>
  <div class="card">
    <div class="head">
      <div>主 <b style="color:#21c07a">{a.fh_home:.0%}</b></div>
      <div>平 <b>{a.fh_draw:.0%}</b></div>
      <div>客 <b style="color:#e07a5f">{a.fh_away:.0%}</b></div>
    </div>
    {_conf_label(conf_fh, sup)}
    <div class="grid" style="margin-top:10px">{fh_ou}</div>
  </div>

  <div class="sec">📊 影響因子</div>
  <div class="card">
    <div class="grid">
      <div class="box"><div class="k">Elo 評分</div><div class="v">{a.elo_home:.0f} vs {a.elo_away:.0f}</div></div>
      <div class="box"><div class="k">資料樣本充足度</div><div class="v">{a.data_support*100:.0f}%</div></div>
      <div class="box"><div class="k">陣型 {html.escape(zh(a.home))}</div><div class="v">{html.escape(hf)}</div></div>
      <div class="box"><div class="k">陣型 {html.escape(zh(a.away))}</div><div class="v">{html.escape(af)}</div></div>
      <div class="box"><div class="k">先發/缺陣 {html.escape(zh(a.home))}</div><div class="v" style="font-size:12px">{html.escape(a.player_note_home)}</div></div>
      <div class="box"><div class="k">先發/缺陣 {html.escape(zh(a.away))}</div><div class="v" style="font-size:12px">{html.escape(a.player_note_away)}</div></div>
      <div class="box"><div class="k">近5場 {html.escape(zh(a.home))}</div><div class="v form">{_form_html(a.home_form)}</div></div>
      <div class="box"><div class="k">近5場 {html.escape(zh(a.away))}</div><div class="v form">{_form_html(a.away_form)}</div></div>
      <div class="box"><div class="k" title="有填陣型→依陣型；否則由模型擬合的攻防強度推得">攻守傾向 ⓘ</div><div class="v" style="font-size:13px">{a.home_style} / {a.away_style}</div></div>
    </div>
    <div class="small" style="margin-top:8px">歷史交手：{html.escape(a.h2h or '無紀錄')}</div>
  </div>

  <div class="foot">Generated by footy · 研究與教育用途</div>
</div></body></html>"""


def write_analysis_html(a, out_path: str | Path, title: str = "單場分析") -> Path:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(render_analysis_html(a, title), encoding="utf-8")
    return out_path


# ---------------- 世界盃整屆網站首頁 ----------------
def _match_pred_row(model, m, linked: set | None = None):
    """單場小組賽一列：已踢顯示真實比分，未踢顯示預測比分與 1X2。

    若該場有分析頁（m.num in linked），整列可點進去。
    """
    from .models import markets  # noqa: F401 (保留供未來用)
    t1, t2 = m.team1, m.team2
    if m.played:
        tag = f"<span class='res'>{m.hg}-{m.ag}</span>"
    elif linked and m.num in linked:
        tag = "<span class='small'>預測 ›</span>"
    else:
        tag = ""
    inner = (f"<span class='fxd'>{_tpe_label(m)}</span>"
             f"<span class='fxt'>{html.escape(zh(t1))}</span>"
             f"<span class='fxm'>{tag}</span>"
             f"<span class='fxt r'>{html.escape(zh(t2))}</span>")
    played = "1" if m.played else "0"
    if linked and m.num in linked:
        body = (f"<a class='fx fxlink' href='match_{m.num}.html'>{inner}"
                f"<span class='arow'>›</span></a>")
    else:
        body = f"<div class='fx'>{inner}</div>"
    return f"<div class='fxrow' data-played='{played}'>{body}</div>"


def _group_points(model_matches, g):
    """由該組已踢比賽計算目前積分（勝3平1）。回傳 {team: points}。"""
    pts: dict[str, int] = {}
    for m in model_matches:
        if m.group != g or not m.played:
            continue
        pts.setdefault(m.team1, 0)
        pts.setdefault(m.team2, 0)
        if m.hg > m.ag:
            pts[m.team1] += 3
        elif m.ag > m.hg:
            pts[m.team2] += 3
        else:
            pts[m.team1] += 1
            pts[m.team2] += 1
    return pts


def _group_standings(matches, g, teams):
    """完整目前積分榜：每隊 賽/勝/平/負/進/失/淨/分。回傳依 分→淨→進 排序的列表。"""
    st = {t: dict(P=0, W=0, D=0, L=0, GF=0, GA=0) for t in teams}
    for m in matches:
        if m.group != g or not m.played:
            continue
        for t, gf, ga, res in ((m.team1, m.hg, m.ag, m.hg - m.ag),
                               (m.team2, m.ag, m.hg, m.ag - m.hg)):
            if t not in st:
                st[t] = dict(P=0, W=0, D=0, L=0, GF=0, GA=0)
            s = st[t]
            s["P"] += 1
            s["GF"] += gf
            s["GA"] += ga
            s["W"] += res > 0
            s["D"] += res == 0
            s["L"] += res < 0
    rows = []
    for t, s in st.items():
        gd = s["GF"] - s["GA"]
        pts = s["W"] * 3 + s["D"]
        rows.append((t, s, gd, pts))
    rows.sort(key=lambda x: (x[3], x[2], x[1]["GF"]), reverse=True)
    return rows


def _standings_table_html(matches, g, teams):
    """目前積分榜 HTML（只有已踢過才顯示；橫向可捲避免破版）。"""
    if not any(m.group == g and m.played for m in matches):
        return ""
    rows = _group_standings(matches, g, teams)
    trs = []
    for i, (t, s, gd, pts) in enumerate(rows, 1):
        trs.append(
            f"<tr><td class='rk'>{i}</td><td class='tm'>{html.escape(zh(t))}</td>"
            f"<td>{s['P']}</td><td>{s['W']}</td><td>{s['D']}</td><td>{s['L']}</td>"
            f"<td>{s['GF']}-{s['GA']}</td><td>{gd:+d}</td>"
            f"<td class='num'><b>{pts}</b></td></tr>")
    return (f"<details class='stbl'><summary>目前積分榜（點開）</summary>"
            f"<table><thead><tr><th>#</th><th>隊伍</th>"
            f"<th>賽</th><th>勝</th><th>平</th><th>負</th><th>進失</th><th>淨</th>"
            f"<th>分</th></tr></thead><tbody>{''.join(trs)}</tbody></table></details>")


def _group_card(g, teams, result, model, matches, linked=None):
    rows = sorted(teams, key=lambda t: result.qualify.get(t, 0), reverse=True)
    trs = []
    for rank, t in enumerate(rows, 1):
        q = result.qualify.get(t, 0)
        trs.append(
            f"<tr><td class='rk'>{rank}</td><td class='tm'>{html.escape(zh(t))}</td>"
            f"<td style=\"{_heat(result.group_first.get(t,0)*100,'48')}\">{result.group_first.get(t,0):.0%}</td>"
            f"<td style=\"{_heat(result.group_top2.get(t,0)*100,'140')}\">{result.group_top2.get(t,0):.0%}</td>"
            f"<td style=\"{_heat(q*100,'140')}\">{q:.0%}</td>"
            f"<td class='num'>{result.exp_points.get(t,0):.1f}</td></tr>")
    standings = _standings_table_html(matches, g, teams)
    from . import worldcup as wc
    fixtures = "".join(_match_pred_row(model, m, linked)
                       for m in sorted([mm for mm in matches if mm.group == g],
                                       key=lambda mm: wc.kickoff_utc(mm.date, getattr(mm, "time", ""))
                                       or _dt.datetime.max))
    return f"""
    <div class="card grp">
      <div class="sec">{html.escape(group_zh(g))}</div>
      {standings}
      <table class="gt"><thead><tr><th>#</th><th>隊伍</th><th>首名</th><th>前二</th><th>晉級</th><th>預期分</th></tr></thead>
      <tbody>{''.join(trs)}</tbody></table>
      <div class="fxs">{fixtures}</div>
    </div>"""


def _navbar(active: str) -> str:
    """頂部導覽列。active: 'home' / 'knockout' / 'custom'。"""
    def cls(k):
        return " class='on'" if k == active else ""
    return (f"<div class='nav'><a href='index.html'{cls('home')}>🏆 首頁</a>"
            f"<a href='knockout.html'{cls('knockout')}>🏟️ 晉級&對陣</a>"
            f"<a href='mlb.html'{cls('mlb')}>⚾ MLB</a>"
            f"<a href='performance.html'{cls('perf')}>📈 績效</a>"
            f"<a href='/custom'{cls('custom')}>🔧 自訂分析</a></div>")


def _bracket_order(matches):
    """依晉級樹排序各輪賽事：讓 R16/8強… 的位置貼著其 R32 來源，畫起來像對陣樹。"""
    import re as _re
    cols = [("Round of 32", "32 強"), ("Round of 16", "16 強"),
            ("Quarter-final", "8 強"), ("Semi-final", "4 強"), ("Final", "決賽")]
    ordered_cols = []
    prev_pos = {}
    for rnd, label in cols:
        rms = [m for m in matches if m.round == rnd]
        if not rms:
            continue
        if not ordered_cols:
            rms = sorted(rms, key=lambda m: m.num)
        else:
            def keyf(m):
                feed = [int(x[1:]) for x in (m.team1, m.team2)
                        if _re.match(r"^[WL]\d+$", x)]
                return min((prev_pos.get(f, 999) for f in feed), default=m.num)
            rms = sorted(rms, key=keyf)
        prev_pos = {m.num: i for i, m in enumerate(rms)}
        ordered_cols.append((label, rms))
    return ordered_cols


def _render_bracket(matches, linked=None) -> str:
    """淘汰賽對陣圖：各輪一欄（依晉級樹排序），每場一個對陣框，已賽標出晉級方。"""
    linked = linked or set()
    out = []
    for label, rms in _bracket_order(matches):
        boxes = []
        for m in rms:
            t1, t2 = _slot_zh(m.team1), _slot_zh(m.team2)
            c1 = c2 = ""
            mid = ""
            if m.played:
                mid = f"<span class='res'>{m.hg}-{m.ag}</span>"
                if m.hg > m.ag:
                    c1 = " win"
                elif m.ag > m.hg:
                    c2 = " win"
            inner = (f"<div class='t{c1}'><span>{html.escape(t1)}</span>{mid}</div>"
                     f"<div class='t{c2}'><span>{html.escape(t2)}</span></div>")
            if m.num in linked:
                boxes.append(f"<a class='bm' style='text-decoration:none;color:inherit;display:block' "
                             f"href='match_{m.num}.html'>{inner}</a>")
            else:
                boxes.append(f"<div class='bm'>{inner}</div>")
        out.append(f"<div class='bcol'><h4>{label}</h4>{''.join(boxes)}</div>")
    return f"<div class='bracket'>{''.join(out)}</div>"


def _today_section(matches, model, linked=None):
    """今日比賽；若今天沒有，顯示接下來最近一天的賽事。"""
    import datetime as _dt
    linked = linked or set()
    today = _dt.date.today().isoformat()
    dates = sorted({m.date for m in matches if m.date})
    day = today if today in dates else next((d for d in dates if d >= today), None)
    if not day:
        return ""  # 賽事已全部結束
    label = "今日比賽" if day == today else f"接下來（{day[5:]}）"
    from . import worldcup as wc
    rms = sorted([m for m in matches if m.date == day],
                 key=lambda m: (wc.kickoff_utc(m.date, getattr(m, "time", ""))
                                or _dt.datetime.max, m.num))
    rows = []
    for m in rms:
        from .models import markets
        t1, t2 = _slot_zh(m.team1), _slot_zh(m.team2)
        if m.played:
            mid = f"<span class='res'>{m.hg}-{m.ag}</span>"
        elif m.team1 in model.attack and m.team2 in model.attack:
            o = markets.outcome_1x2(model.score_matrix(m.team1, m.team2, neutral=True))
            mid = f"<span class='small'>{o['home']:.0%}/{o['draw']:.0%}/{o['away']:.0%}</span>"
        else:
            mid = "<span class='small'>—</span>"
        ko = _tpe_label(m, with_date=False)
        inner = (f"<span class='fxk'>{ko}</span>"
                 f"<span class='fxt'>{html.escape(t1)}</span>"
                 f"<span class='fxm'>{mid}</span>"
                 f"<span class='fxt r'>{html.escape(t2)}</span>")
        if m.num in linked:
            rows.append(f"<a class='fx fxlink' style='grid-template-columns:auto 1fr auto 1fr 14px' "
                        f"href='match_{m.num}.html'>{inner}<span class='arow'>›</span></a>")
        else:
            rows.append(f"<div class='fx' style='grid-template-columns:auto 1fr auto 1fr'>{inner}</div>")
    return (f"<div class='card'><div class='sec'>📅 {label}（台北時間）</div>"
            f"<div class='fxs'>{''.join(rows)}</div></div>")


def render_worldcup_html(result, model, matches, title: str = "2026 世界盃預測",
                         linked: set | None = None, track_text: str | None = None) -> str:
    import datetime as _dt
    today = _dt.date.today().isoformat()
    # 冠軍機率前 16
    champ = sorted(result.champion.items(), key=lambda x: x[1], reverse=True)
    top = [(t, p) for t, p in champ if p > 0][:16]
    maxp = top[0][1] if top else 1.0
    champ_bars = "".join(
        f"<div class='crow'><span class='cn'>{html.escape(zh(t))}</span>"
        f"<span class='cbar'><span style='width:{p/maxp*100:.1f}%'></span></span>"
        f"<span class='cp'>{p:.1%}</span></div>" for t, p in top)

    groups_html = "".join(
        _group_card(g, result.groups[g], result, model, matches, linked)
        for g in sorted(result.groups))

    today_html = _today_section(matches, model, linked)
    # 推薦戰績/ROI/CLV 已移至「績效」頁；首頁僅給一個入口連結
    track_html = ("<div class='card'><div class='sec'>📒 模型推薦戰績 & 下注績效</div>"
                  "<div class='small' style='color:var(--muted)'>推薦勝率、真實盤口 ROI/CLV "
                  "與權重校準都在 <a href='performance.html' style='color:var(--accent)'>"
                  "→ 績效頁</a>。</div></div>")

    return f"""<!doctype html><html lang="zh-Hant"><head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>{html.escape(title)}</title><style>{_CSS}
.two{{display:grid;grid-template-columns:1fr 1fr;gap:16px;align-items:start}}
@media(max-width:760px){{.two{{grid-template-columns:1fr}}}}
.sec{{font-weight:800;margin:6px 0 10px;font-size:16px}}
.crow{{display:flex;align-items:center;gap:8px;margin:5px 0;font-size:13px}}
.cn{{width:120px;flex:none;font-weight:600}}.cp{{width:48px;text-align:right;color:#cdd9e5}}
.cbar{{flex:1;background:#11161c;border-radius:5px;height:14px;overflow:hidden}}
.cbar>span{{display:block;height:100%;background:linear-gradient(90deg,#21c07a,#7be0b0)}}
table{{width:100%;border-collapse:collapse;font-size:13px}}
th,td{{padding:6px 7px;text-align:center;border-bottom:1px solid var(--line)}}
th{{color:var(--muted);font-size:11px}}td.tm{{text-align:left;font-weight:700}}td.rk{{color:var(--muted)}}
.grids{{display:grid;grid-template-columns:repeat(auto-fill,minmax(310px,1fr));gap:14px}}
.grp .fxs{{margin-top:8px}}
.stbl{{margin:4px 0 8px}}.stbl>summary{{cursor:pointer;font-size:11px;color:var(--muted);
font-weight:700;list-style:none}}.stbl>summary::-webkit-details-marker{{display:none}}
.stbl>summary::before{{content:'▸ '}}.stbl[open]>summary::before{{content:'▾ '}}
.stbl table{{font-size:11px;min-width:280px;display:block;overflow-x:auto;margin-top:4px}}
.stbl th,.stbl td{{padding:4px 5px}}
.fx{{display:grid;grid-template-columns:64px 1fr auto 1fr;gap:6px;align-items:center;
font-size:12px;padding:3px 0;border-top:1px solid #1c242d}}
.fxt{{font-weight:600}}.fxt.r{{text-align:right}}.fxd{{color:var(--muted)}}.fxm{{text-align:center}}
.fxk{{color:var(--muted);font-variant-numeric:tabular-nums;white-space:nowrap}}
.pred{{color:#7be0b0;font-weight:700}}.res{{color:#e0b341;font-weight:700}}
.cs5{{font-size:11px;color:var(--muted);padding:0 0 5px 64px;letter-spacing:.02em}}
a.fxlink{{text-decoration:none;color:inherit;grid-template-columns:64px 1fr auto 1fr 14px}}
a.fxlink:active{{background:#1c242d}}.arow{{color:var(--accent);text-align:right}}
</style></head>
<body><div class="wrap">
  {_navbar('home')}
  <h1>🏆 {html.escape(title)}</h1>
  <div class="sub">{today} · 蒙地卡羅 {result.n_sims:,} 次 · Dixon–Coles + Elo · 已踢比分納入</div>
  <div class="disc">⚠️ 純機率預測，非投注建議。最佳第三名→R32 槽位為近似指派；晉級機率為主要可信輸出。</div>

  {today_html}
  {track_html}

  <div class="card">
    <div class="sec">🥇 奪冠機率</div>
    {champ_bars}
    <div class="small" style="margin-top:8px"><a href="knockout.html" style="color:var(--accent)">→ 看完整晉級展望與淘汰賽對陣圖</a></div>
  </div>

  <div class="sec" style="margin-top:22px">📋 小組賽程與預測（點擊看單場分析）</div>
  <div class="fctrl">
    <button class="on" onclick="ff(this,'all')">全部</button>
    <button onclick="ff(this,'todo')">只看未踢</button>
  </div>
  <div class="grids">{groups_html}</div>

  <div class="foot">Generated by footy · 研究與教育用途 · 資料：openfootball + martj42</div>
  <script>
  function ff(btn,mode){{
    document.querySelectorAll('.fctrl button').forEach(b=>b.classList.remove('on'));
    btn.classList.add('on');
    document.querySelectorAll('.fxrow').forEach(r=>{{
      r.style.display=(mode==='todo'&&r.dataset.played==='1')?'none':'';
    }});
  }}
  </script>
</div></body></html>"""


def write_worldcup_html(result, model, matches, out_path, title="2026 世界盃預測",
                        linked=None, track_text=None):
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        render_worldcup_html(result, model, matches, title, linked, track_text),
        encoding="utf-8")
    return out_path


def render_knockout_page(result, model, matches, linked=None,
                         title="晉級展望 & 淘汰賽對陣") -> str:
    """獨立頁面：完整晉級展望表 + 淘汰賽對陣圖。"""
    import datetime as _dt
    today = _dt.date.today().isoformat()
    champ = sorted(result.champion.items(), key=lambda x: x[1], reverse=True)
    top = [(t, p) for t, p in champ if p > 0][:24]
    ko_rows = "".join(
        f"<tr><td class='rk'>{i}</td><td class='tm'>{html.escape(zh(t))}</td>"
        f"<td>{result.qualify.get(t,0):.0%}</td><td>{result.r16.get(t,0):.0%}</td>"
        f"<td>{result.quarter.get(t,0):.0%}</td><td>{result.semi.get(t,0):.0%}</td>"
        f"<td>{result.final.get(t,0):.0%}</td>"
        f"<td style=\"{_heat(p*100,'48')}\"><b>{p:.1%}</b></td></tr>"
        for i, (t, p) in enumerate(top, 1))
    bracket = _render_bracket(matches, linked or set())
    return f"""<!doctype html><html lang="zh-Hant"><head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>{html.escape(title)}</title><style>{_CSS}
table{{width:100%;border-collapse:collapse;font-size:13px;background:var(--card);border-radius:12px;overflow:hidden}}
th,td{{padding:7px 8px;text-align:center;border-bottom:1px solid var(--line)}}
th{{color:var(--muted);font-size:11px}}td.tm{{text-align:left;font-weight:700}}td.rk{{color:var(--muted)}}
</style></head>
<body><div class="wrap">
  {_navbar('knockout')}
  <h1>🏟️ {html.escape(title)}</h1>
  <div class="sub">{today} · 蒙地卡羅 {result.n_sims:,} 次 · 晉級機率為主要可信輸出</div>
  <div class="sec">📈 晉級展望（依奪冠機率）</div>
  <table><thead><tr><th>#</th><th>隊伍</th><th>晉級</th><th>16強</th><th>8強</th>
  <th>4強</th><th>決賽</th><th>奪冠</th></tr></thead><tbody>{ko_rows}</tbody></table>
  <div class="sec" style="margin-top:22px">🏟️ 淘汰賽對陣圖</div>
  {bracket}
  <div class="foot">Generated by footy · 研究與教育用途</div>
</div></body></html>"""


def _line_svg(points, w=320, h=90, color="#7be0b0", zero=True):
    """把數列畫成折線 SVG（零依賴）。points 為 y 值序列。"""
    if not points:
        return ""
    n = len(points)
    lo, hi = min(points), max(points)
    if zero:
        lo, hi = min(lo, 0.0), max(hi, 0.0)
    span = (hi - lo) or 1.0
    pad = 6

    def xy(i, v):
        x = pad + (w - 2 * pad) * (i / max(n - 1, 1))
        y = pad + (h - 2 * pad) * (1 - (v - lo) / span)
        return x, y
    pts = " ".join(f"{x:.1f},{y:.1f}" for i, v in enumerate(points)
                   for x, y in [xy(i, v)])
    zero_line = ""
    if zero and lo <= 0 <= hi:
        _, zy = xy(0, 0.0)
        zero_line = (f"<line x1='{pad}' y1='{zy:.1f}' x2='{w - pad}' y2='{zy:.1f}' "
                     f"stroke='#3a4654' stroke-dasharray='3 3'/>")
    last_x, last_y = xy(n - 1, points[-1])
    return (f"<svg viewBox='0 0 {w} {h}' width='100%' height='{h}' "
            f"preserveAspectRatio='none' style='display:block'>{zero_line}"
            f"<polyline fill='none' stroke='{color}' stroke-width='2' points='{pts}'/>"
            f"<circle cx='{last_x:.1f}' cy='{last_y:.1f}' r='3' fill='{color}'/></svg>")


def _pending_body(pending):
    """已連結真實盤口、但下注尚未結算時的頁面內容（列出待結算 +EV 推薦）。"""
    rows = sorted(pending, key=lambda r: -(r.get("edge") or 0))
    trs = ""
    for r in rows[:60]:
        line = f" {r['line']}" if str(r.get("line", "")) not in ("", "nan") else ""
        edge = r.get("edge")
        edge_s = f"{edge:+.1%}" if isinstance(edge, (int, float)) else "—"
        odds = r.get("odds")
        odds_s = f"{odds:.2f}" if isinstance(odds, (int, float)) else "—"
        trs += (f"<tr><td>{str(r.get('date',''))[5:]}</td>"
                f"<td class='tm'>{html.escape(zh(str(r.get('home',''))))} v "
                f"{html.escape(zh(str(r.get('away',''))))}</td>"
                f"<td>{_MK_ZH.get(r.get('market'), r.get('market'))} "
                f"{html.escape(str(r.get('selection','')))}{line}</td>"
                f"<td>{odds_s}</td>"
                f"<td style='color:#7be0b0'>{edge_s}</td></tr>")
    return (
        "<div class='card'><div class='sec'>✅ 已連結真實盤口</div>"
        f"<div class='small' style='color:#cdd9e5;line-height:1.7'>"
        f"系統已抓盤口並記錄 <b>{len(pending)}</b> 注模型 +EV 推薦（均注 1 單位），"
        "但這些比賽<b>尚未開賽</b>，所以還沒有損益。賽後每日建站會自動用"
        "下注/收盤賠率結算，這裡就會出現累積損益曲線、CLV 走勢、勝過收盤比例，"
        "以及回測的最佳融合權重建議。</div></div>"
        "<div class='card'><div class='sec'>🎯 待結算的 +EV 推薦（依 edge 排序）</div>"
        "<table><thead><tr><th>日期</th><th>對戰</th><th>下注</th><th>賠率</th>"
        f"<th>edge</th></tr></thead><tbody>{trs}</tbody></table>"
        "<div class='small' style='color:var(--muted);margin-top:6px'>"
        "edge = 模型(融合市場後)機率 × 賠率 − 1，&gt;0 才下注。edge 高不代表穩贏，"
        "務必等 CLV 累積出來驗證是否真有 value。</div></div>")


def _track_card(track_text):
    """模型推薦勝率卡（次要：收合，自我校驗、無賠率→只計勝率，不代表收益）。"""
    if not track_text:
        return ""
    return (f"<details class='card sec-collapse'><summary>📒 模型推薦勝率（參考用，非收益）</summary>"
            f"<div class='small' style='white-space:pre-line;color:#cdd9e5;margin-top:8px'>"
            f"{html.escape(track_text)}</div>"
            f"<div class='small' style='color:var(--muted);margin-top:6px'>"
            f"涵蓋整屆（含已踢回填）的模型推薦過/沒過。<b>只計勝率、不含賠率，不代表收益</b>"
            f"（勝率高 ≠ 賺錢）；回填場次模型可能已學進賽果，偏樂觀。收益看上方真實盤口 ROI/CLV。"
            f"</div></details>")


def _pl_hero(last):
    """收益主視覺：大字累積損益 + ROI。"""
    c = "#7be0b0" if last["cum_pl"] >= 0 else "#e06a6a"
    return (f"<div class='hero'>"
            f"<div class='hero-main'><span>累積收益（{last['n']} 注 · 均注 1 單位）</span>"
            f"<b style='color:{c}'>{last['cum_pl']:+.2f}u</b></div>"
            f"<div class='hero-side'><span>ROI</span><b style='color:{c}'>{last['cum_roi']:+.1%}</b></div>"
            f"</div>")


def _ai_card(title, text):
    if not text:
        return ""
    return (f"<div class='card'><div class='sec'>{title}</div>"
            f"<div class='small' style='color:#cdd9e5;line-height:1.7;white-space:pre-line'>"
            f"{html.escape(text)}</div></div>")


def render_performance_page(hist, summary_obj=None, tune=None, pending=None,
                            track_text=None, ai_risk=None, ai_review=None,
                            title="下注績效 & CLV") -> str:
    """獨立績效頁：模型勝率卡 + 真實盤口累積損益/CLV 折線、權重校準、逐注紀錄。

    三種狀態：已有結算→完整圖表；已連結但全待結算→列出待結算 +EV 推薦；
    完全沒有真實盤口下注→引導設定 ODDS_API_KEY。模型勝率卡三態皆顯示。
    ai_risk/ai_review：AI 風控/賽後檢討文字（未設金鑰時為 None，不顯示）。
    """
    today = _dt.date.today().isoformat()
    track_card = _track_card(track_text)  # 次要，放最下面
    risk_card = _ai_card("🛡️ AI 風控提醒", ai_risk)
    review_card = _ai_card("🔎 AI 賽後檢討", ai_review)
    if not hist:
        if pending:
            return _perf_doc(title, today, risk_card + _pending_body(pending) + track_card)
        body = ("<div class='card'><div class='sec'>📈 尚無收益紀錄（無已結算的真實盤口下注）</div>"
                "<div class='small' style='color:var(--muted);line-height:1.7'>"
                "需要部署環境設定 <code>ODDS_API_KEY</code>，系統才會抓真實盤口、"
                "只記模型相對市場有正期望值(+EV)的推薦，並在賽後用下注/收盤賠率"
                "算實際 ROI 與 CLV。<br>累積足夠注數後，這裡會出現："
                "<b>累積收益曲線、ROI、CLV 走勢</b>、勝過收盤比例，以及用歷史快照回測的"
                "最佳融合權重建議。</div></div>")
        return _perf_doc(title, today, body + track_card)

    last = hist[-1]
    pl_pts = [r["cum_pl"] for r in hist]
    clv_pts = [r["cum_clv"] * 100 for r in hist]
    pl_color = "#7be0b0" if last["cum_pl"] >= 0 else "#e06a6a"
    hero = _pl_hero(last)
    kpis = (
        f"<div class='kpis'>"
        f"<div class='kpi'><span>注數</span><b>{last['n']}</b></div>"
        f"<div class='kpi'><span>平均 CLV</span><b>{last['cum_clv']:+.1%}</b></div>"
        f"<div class='kpi'><span>勝過收盤</span><b>{last['beat_rate']:.0%}</b></div>"
        f"</div>")
    charts = (
        f"<div class='card'><div class='sec'>💰 累積收益（單位）</div>"
        f"{_line_svg(pl_pts, color=pl_color)}</div>"
        f"<div class='card'><div class='sec'>🎯 累積平均 CLV（%，&gt;0 長期領先指標）</div>"
        f"{_line_svg(clv_pts, color='#6ea8fe')}</div>")
    # 權重校準表
    tune_html = ""
    if tune and tune.get("rows"):
        best = tune.get("best_roi")
        trs = ""
        for r in tune["rows"]:
            if r["n_bets"] == 0:
                continue
            mark = " ★" if best and abs(r["weight"] - best["weight"]) < 1e-9 else ""
            hot = _heat(max(min(r["roi"] * 100 + 20, 100), 0), '140')
            trs += (f"<tr><td>{r['weight']:.1f}{mark}</td><td>{r['n_bets']}</td>"
                    f"<td>{r['pl']:+.2f}</td><td style=\"{hot}\">{r['roi']:+.1%}</td>"
                    f"<td>{r['clv']:+.1%}</td></tr>")
        rec = (f"<div class='small' style='margin-top:6px'>回測 {tune['n_matches']} 個"
               f"決策單位：ROI 最佳權重 <b>BLEND_WEIGHT={best['weight']:.1f}</b>"
               f"（ROI {best['roi']:+.1%}）。樣本少時僅供參考，CLV 比 ROI 更穩。</div>"
               ) if best else ""
        tune_html = (f"<div class='card'><div class='sec'>⚖️ 融合權重校準（回測）</div>"
                     f"<table><thead><tr><th>權重</th><th>注數</th><th>損益</th>"
                     f"<th>ROI</th><th>CLV</th></tr></thead><tbody>{trs}</tbody></table>"
                     f"{rec}</div>")
    # 逐注紀錄（最近 40 筆）
    log_trs = ""
    for r in reversed(hist[-40:]):
        rc = {"win": "#7be0b0", "loss": "#e06a6a", "push": "var(--muted)"}.get(r["result"], "")
        clv = f"{r['clv']:+.1%}" if r["clv"] is not None else "—"
        line = f" {r['line']}" if str(r["line"]) not in ("", "nan") else ""
        log_trs += (
            f"<tr><td>{str(r['date'])[5:]}</td>"
            f"<td class='tm'>{html.escape(zh(r['home']))} v {html.escape(zh(r['away']))}</td>"
            f"<td>{_MK_ZH.get(r['market'], r['market'])} {html.escape(str(r['selection']))}{line}</td>"
            f"<td>{r['odds']:.2f}</td><td>{clv}</td>"
            f"<td style='color:{rc};font-weight:700'>{_RES_ZH.get(r['result'], r['result'])}</td>"
            f"<td style='color:{rc}'>{r['pl']:+.2f}</td></tr>")
    log_html = (f"<div class='card'><div class='sec'>📜 逐注紀錄（近 {min(len(hist),40)} 筆）</div>"
                f"<table><thead><tr><th>日期</th><th>對戰</th><th>下注</th><th>賠率</th>"
                f"<th>CLV</th><th>結果</th><th>損益</th></tr></thead>"
                f"<tbody>{log_trs}</tbody></table></div>")
    sub = (f"{today} · 均注 1 單位 · 只計有真實盤口、模型 +EV 的下注")
    note = ("<div class='small' style='color:var(--muted);margin-top:4px'>"
            "CLV（closing line value）= 你拿到的賠率 vs 收盤賠率；長期 CLV&gt;0 是"
            "判斷模型能否真正贏過市場最可靠的領先指標，比短期勝率/ROI 抗雜訊。</div>")
    return _perf_doc(title, sub, hero + kpis + risk_card + charts + note
                     + tune_html + log_html + review_card + track_card)


_MK_ZH = {"1X2": "勝平負", "OU": "大小", "BTTS": "兩隊進球", "AH": "亞盤"}
_RES_ZH = {"win": "過", "loss": "沒過", "push": "走盤"}


def _perf_doc(title, sub, body):
    return f"""<!doctype html><html lang="zh-Hant"><head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>{html.escape(title)}</title><style>{_CSS}
table{{width:100%;border-collapse:collapse;font-size:12px;background:var(--card);border-radius:12px;overflow:hidden}}
th,td{{padding:6px 7px;text-align:center;border-bottom:1px solid var(--line)}}
th{{color:var(--muted);font-size:11px}}td.tm{{text-align:left;font-weight:600}}
.kpis{{display:grid;grid-template-columns:repeat(3,1fr);gap:8px;margin:4px 0}}
.kpi{{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:8px 6px;text-align:center}}
.kpi span{{display:block;color:var(--muted);font-size:10px;margin-bottom:3px}}
.kpi b{{font-size:15px}}
.hero{{display:flex;align-items:center;justify-content:space-between;gap:12px;
background:linear-gradient(135deg,#11161c,#161d26);border:1px solid var(--line);
border-radius:14px;padding:16px 18px;margin:4px 0 10px}}
.hero-main span,.hero-side span{{display:block;color:var(--muted);font-size:12px;margin-bottom:4px}}
.hero-main b{{font-size:34px;font-weight:800;line-height:1}}
.hero-side{{text-align:right}}.hero-side b{{font-size:22px;font-weight:800}}
.sec-collapse{{margin-top:14px}}
.sec-collapse>summary{{cursor:pointer;font-weight:700;color:var(--muted);list-style:none}}
.sec-collapse>summary::-webkit-details-marker{{display:none}}
.sec-collapse>summary::before{{content:'▸ '}}.sec-collapse[open]>summary::before{{content:'▾ '}}
@media(max-width:480px){{.kpis{{grid-template-columns:repeat(3,1fr)}}.hero-main b{{font-size:28px}}}}
</style></head>
<body><div class="wrap">
  {_navbar('perf')}
  <h1>📈 {html.escape(title)}</h1>
  <div class="sub">{sub}</div>
  {body}
  <div class="foot">Generated by footy · 研究與教育用途，非投注建議</div>
</div></body></html>"""


def _bs_badge(verdict) -> str:
    """買/觀望 徽章。"""
    if verdict == "買":
        return " <span class='bs buy'>買</span>"
    if verdict == "觀望":
        return " <span class='bs wait'>觀望</span>"
    return ""


def _mlb_mrow(market: str, pick: str, prob: float, sig=None) -> str:
    """一列盤口：盤口 | 推薦(+買/觀望) | 機率 | 賠率·edge（只列模型推薦那側）。

    sig 為 mlb.bet_signals 的單一盤口訊號（含融合後 edge / verdict / odds）。
    """
    sig = sig or {}
    odds = sig.get("odds")
    if odds:
        e = sig.get("edge")
        e = e if isinstance(e, (int, float)) else (prob * float(odds) - 1.0)
        c = "#7be0b0" if e > 0 else "#e0b341" if e > -0.05 else "var(--muted)"
        right = f"<span style='color:{c}'>@{float(odds):g} · {e:+.1%}</span>"
    else:
        right = "<span class='dim'>—</span>"
    return (f"<div class='mrow'><span class='mk'>{market}</span>"
            f"<span class='pk'>{pick}{_bs_badge(sig.get('verdict'))}</span>"
            f"<span class='pb'>{prob:.0%}</span>"
            f"<span class='od'>{right}</span></div>")


_MLB_SEL_ZH = {"home": "主", "away": "客", "over": "大", "under": "小"}


def _mlb_card(r, zh_t) -> str:
    """單場預測卡：開賽時間 + 三盤口（含買/觀望）+ 投手 + 可能比分。"""
    g, m = r["game"], r["m"]
    sig = r.get("signals") or {}
    hz, az = zh_t(g["home"]), zh_t(g["away"])
    if m.p_home >= m.p_away:
        ml_pick, ml_prob = hz, m.p_home
    else:
        ml_pick, ml_prob = az, m.p_away
    if m.p_over >= m.p_under:
        ou_pick, ou_prob = f"大 {m.total_line:g}", m.p_over
    else:
        ou_pick, ou_prob = f"小 {m.total_line:g}", m.p_under
    if m.p_cover_home >= 0.5:
        rl_pick, rl_prob = f"{hz} {m.run_line:+g}", m.p_cover_home
    else:
        rl_pick, rl_prob = f"{az} {-m.run_line:+g}", 1 - m.p_cover_home
    mtable = (_mlb_mrow("錢線", ml_pick, ml_prob, sig.get("1X2"))
              + _mlb_mrow("大小", ou_pick, ou_prob, sig.get("OU"))
              + _mlb_mrow("讓分", rl_pick, rl_prob, sig.get("AH")))
    pit = ""
    if g.get("home_pitcher") or g.get("away_pitcher"):
        def _pbadge(name, note):
            nm = html.escape(name or "未定")
            if note:
                tip = html.escape(note)
                return f"<span title='{tip}'>{nm} <span class='pdot'>ⓘ</span></span>"
            return nm
        pit = (f"<div class='pit'>先發 "
               f"{_pbadge(g.get('away_pitcher'), r.get('ap_note'))}"
               f"<span class='vs'>vs</span>"
               f"{_pbadge(g.get('home_pitcher'), r.get('hp_note'))}</div>")
    pf = r.get("pf", 1.0)
    pf_note = (f"<span class='pf'>球場 {pf:.2f}</span>"
               if abs(pf - 1.0) > 0.005 else "")
    # 天氣徽章：有明顯影響大小盤才顯示（風出/進 + 溫度）
    wx, wf = r.get("wx"), r.get("wf", 1.0)
    wx_note = ""
    if wx and abs((wf or 1.0) - 1.0) > 0.005:
        arrow = "風出↑" if wx.get("wind_sign") == 1 else "風進↓" if wx.get("wind_sign") == -1 else ""
        parts = []
        if wx.get("temp") is not None:
            parts.append(f"{wx['temp']:.0f}°F")
        if arrow and wx.get("wind_speed"):
            parts.append(f"{arrow}{wx['wind_speed']:.0f}mph")
        tip = f"天氣調整總分 ×{wf:.2f}"
        wx_note = f"<span class='pf' style='color:#6ea8fe' title='{tip}'>🌤 {' '.join(parts)}</span>"
    t = r.get("time")
    time_note = f"<div class='mtime'>🕒 {html.escape(t)}</div>" if t else ""
    tops = "、".join(f"{h}-{a}" for (h, a), p in m.top_scores[:3])
    return (
        f"<div class='card mgame'>"
        f"<div class='mhd'><b>{az}</b> <span class='at'>@</span> <b>{hz}</b>"
        f"<span class='xr'>{m.exp_away:.1f}–{m.exp_home:.1f}</span>{pf_note}{wx_note}</div>"
        f"{time_note}{pit}<div class='mtab'>{mtable}</div>"
        f"<div class='mtop'>可能比分 {tops}</div></div>")


def _mlb_top5(rows, zh_t) -> str:
    """最推薦 TOP 5：跨場依最強「買」edge 排序，各列出該場最強一注。"""
    picks = []
    for r in rows:
        be = r.get("best_edge")
        if be is None:
            continue
        sig = r.get("signals") or {}
        mk, best = None, None
        for k, s in sig.items():
            if s.get("verdict") == "買" and s.get("edge") is not None:
                if best is None or s["edge"] > best["edge"]:
                    mk, best = k, s
        if best is None:
            continue
        g, m = r["game"], r["m"]
        mk_zh = {"1X2": "錢線", "OU": "大小", "AH": "讓分"}[mk]
        side = best["side"]
        if mk == "1X2":
            sel = zh_t(g["home"]) if side == "home" else zh_t(g["away"])
        elif mk == "OU":
            sel = f"{'大' if side == 'over' else '小'} {m.total_line:g}"
        else:
            sel = (f"{zh_t(g['home'])} {m.run_line:+g}" if side == "home"
                   else f"{zh_t(g['away'])} {-m.run_line:+g}")
        picks.append({"edge": best["edge"], "odds": best["odds"], "mk": mk_zh,
                      "sel": sel, "time": r.get("time", ""),
                      "home": zh_t(g["home"]), "away": zh_t(g["away"])})
    if not picks:
        return ""
    picks.sort(key=lambda x: -x["edge"])
    trs = ""
    for i, p in enumerate(picks[:5], 1):
        trs += (f"<tr><td class='rk'>{i}</td>"
                f"<td class='tm'>{html.escape(p['away'])} @ {html.escape(p['home'])}</td>"
                f"<td class='small' style='color:var(--muted)'>{html.escape(p['time'])}</td>"
                f"<td>{p['mk']} <b>{html.escape(p['sel'])}</b></td>"
                f"<td>@{p['odds']:g}</td>"
                f"<td style='color:#7be0b0;font-weight:700'>{p['edge']:+.1%}</td></tr>")
    return (
        "<div class='card top5'><div class='sec'>🔥 今日最推薦 TOP 5"
        "<span class='small' style='color:var(--muted);font-weight:400'>"
        "（模型融合市場後 +EV，依 edge 排序）</span></div>"
        "<table style='margin-top:6px'><thead><tr><th>#</th><th>對戰</th><th>時間</th>"
        f"<th>推薦</th><th>賠率</th><th>edge</th></tr></thead><tbody>{trs}</tbody></table>"
        "<div class='small' style='color:var(--muted);margin-top:6px'>"
        "「買」= 融合後機率×賠率−1&gt;0 且通過風控；edge 高不代表穩贏，"
        "長期看 <a href='mlb_perf.html' style='color:var(--accent)'>MLB 績效頁</a> 的 CLV。</div></div>")


def render_mlb_page(rows, date: str, power=None, track_text=None,
                    note: str = "", title: str = "MLB 今日預測") -> str:
    """MLB 分頁：TOP5 最推薦 + 今日各場預測卡（含買/觀望、開賽時間）+ 戰力表 + 戰績。"""
    from . import mlb as _mlb
    zh_t = _mlb.zh_mlb
    top5 = _mlb_top5(rows, zh_t)
    cards = [_mlb_card(r, zh_t) for r in rows]
    body = ("<div class='mgrid'>" + "".join(cards) + "</div>") if cards else (
        f"<div class='card'><div class='sec'>今日無可預測比賽</div>"
        f"<div class='small' style='color:var(--muted)'>{html.escape(note) or '賽程空檔，或模型未涵蓋參賽隊。'}</div></div>")
    body = top5 + body
    track_html = ""
    if track_text:
        track_html = (f"<div class='card'><div class='sec'>📒 MLB 推薦戰績</div>"
                      f"<div class='small' style='white-space:pre-line;color:#cdd9e5'>"
                      f"{html.escape(track_text)}</div></div>")
    power_html = ""
    if power:
        trs = "".join(
            f"<tr><td class='rk'>{i}</td><td class='tm'>{html.escape(zh_t(p['team']))}</td>"
            f"<td>{p['rf']:.2f}</td><td>{p['ra']:.2f}</td>"
            f"<td class='num'><b>{p['diff']:+.2f}</b></td></tr>"
            for i, p in enumerate(power, 1))
        power_html = (f"<details class='card sec-collapse'><summary>💪 球隊戰力表（模型評分）</summary>"
                      f"<table style='margin-top:8px'><thead><tr><th>#</th><th>球隊</th>"
                      f"<th>場均得分</th><th>場均失分</th><th>淨</th></tr></thead>"
                      f"<tbody>{trs}</tbody></table></details>")
    return f"""<!doctype html><html lang="zh-Hant"><head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>{html.escape(title)}</title><style>{_CSS}
table{{width:100%;border-collapse:collapse;font-size:12px}}
th,td{{padding:5px 7px;text-align:center;border-bottom:1px solid var(--line)}}
th{{color:var(--muted);font-size:11px}}td.tm{{text-align:left;font-weight:700}}td.rk{{color:var(--muted)}}
.mgrid{{display:grid;grid-template-columns:repeat(auto-fill,minmax(300px,1fr));gap:12px}}
.mgame{{padding:12px 14px}}
.mhd{{display:flex;align-items:baseline;gap:6px;font-size:15px;margin-bottom:2px}}
.mhd .at{{color:var(--muted);font-weight:400}}
.mhd .xr{{margin-left:auto;font-size:12px;color:var(--muted);font-variant-numeric:tabular-nums}}
.mhd .pf{{font-size:11px;color:#e0b341;margin-left:8px}}
.pit{{font-size:11px;color:var(--muted);margin:2px 0 8px}}
.pit .vs{{margin:0 6px;opacity:.6}}
.pit .pdot{{color:var(--accent);cursor:help}}
.mtab{{border-top:1px solid #1c242d}}
.mrow{{display:grid;grid-template-columns:44px 1fr auto auto;gap:10px;align-items:baseline;
font-size:13px;padding:5px 0;border-bottom:1px solid #171e26}}
.mrow:last-child{{border-bottom:0}}
.mrow .mk{{color:var(--muted);font-size:11px}}
.mrow .pk{{font-weight:700;color:#e6edf3}}
.mrow .pb{{font-variant-numeric:tabular-nums;color:#cdd9e5}}
.mrow .od{{font-variant-numeric:tabular-nums;font-size:12px;text-align:right;min-width:96px}}
.mrow .dim{{color:#3a4350}}
.mtop{{margin-top:7px;font-size:11px;color:var(--muted)}}
.mtime{{font-size:11px;color:#8aa0b4;margin:1px 0 6px}}
.bs{{font-size:10px;font-weight:700;padding:1px 6px;border-radius:6px;vertical-align:middle}}
.bs.buy{{background:#1c3a2a;color:#7be0b0;border:1px solid #2e5c42}}
.bs.wait{{background:#2a2a1c;color:#e0b341;border:1px solid #4d4a2a}}
.top5{{border:1px solid #2e5c42}}
.top5 td.tm{{font-weight:600}}
.sec-collapse>summary{{cursor:pointer;font-weight:700;list-style:none}}
.sec-collapse>summary::-webkit-details-marker{{display:none}}
.sec-collapse>summary::before{{content:'▸ '}}.sec-collapse[open]>summary::before{{content:'▾ '}}
</style></head>
<body><div class="wrap">
  {_navbar('mlb')}
  <h1>⚾ {html.escape(title)}</h1>
  <div class="sub">{html.escape(date)}（美東賽程日）· 負二項得分模型 + 先發投手(RA/9+FIP) + 球場因子 + 天氣 · <a href="mlb_perf.html" style="color:var(--accent)">📈 MLB 績效</a></div>
  <div class="disc">⚠️ 純機率預測，非投注建議。只列模型較看好的一側；<b class="bs buy" style="font-size:10px">買</b>=融合市場後 +EV 且過風控、<b class="bs wait" style="font-size:10px">觀望</b>=無正期望值；時間為台北時區。</div>
  {('<div class="small" style="color:var(--warn);margin-bottom:8px">' + html.escape(note) + '</div>') if note else ''}
  {track_html}
  {body}
  {power_html}
  <div class="foot">Generated by footy · 研究與教育用途</div>
</div></body></html>"""


_MLB_MK_ZH = {"1X2": "錢線", "OU": "大小", "AH": "讓分"}


def _mlb_perf_doc(title, sub, body):
    """MLB 績效頁殼（導覽列 MLB 高亮 + 返回 MLB 預測連結）。"""
    return f"""<!doctype html><html lang="zh-Hant"><head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>{html.escape(title)}</title><style>{_CSS}
table{{width:100%;border-collapse:collapse;font-size:12px;background:var(--card);border-radius:12px;overflow:hidden}}
th,td{{padding:6px 7px;text-align:center;border-bottom:1px solid var(--line)}}
th{{color:var(--muted);font-size:11px}}td.tm{{text-align:left;font-weight:600}}
.kpis{{display:grid;grid-template-columns:repeat(3,1fr);gap:8px;margin:4px 0}}
.kpi{{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:8px 6px;text-align:center}}
.kpi span{{display:block;color:var(--muted);font-size:10px;margin-bottom:3px}}
.kpi b{{font-size:15px}}
.hero{{display:flex;align-items:center;justify-content:space-between;gap:12px;
background:linear-gradient(135deg,#11161c,#161d26);border:1px solid var(--line);
border-radius:14px;padding:16px 18px;margin:4px 0 10px}}
.hero-main span,.hero-side span{{display:block;color:var(--muted);font-size:12px;margin-bottom:4px}}
.hero-main b{{font-size:34px;font-weight:800;line-height:1}}
.hero-side{{text-align:right}}.hero-side b{{font-size:22px;font-weight:800}}
@media(max-width:480px){{.hero-main b{{font-size:28px}}}}
</style></head>
<body><div class="wrap">
  {_navbar('mlb')}
  <h1>⚾ {html.escape(title)}</h1>
  <div class="sub">{sub} · <a href="mlb.html" style="color:var(--accent)">← 回今日預測</a></div>
  {body}
  <div class="foot">Generated by footy · 研究與教育用途，非投注建議</div>
</div></body></html>"""


def _mlb_pending_body(pending):
    """MLB 已連結真實盤口、尚未結算時：列出待結算 +EV 推薦（依 edge 排序）。"""
    from . import mlb as _mlb
    zt = _mlb.zh_mlb
    rows = sorted(pending, key=lambda r: -(r.get("edge") or 0))
    trs = ""
    for r in rows[:60]:
        line = f" {r['line']}" if str(r.get("line", "")) not in ("", "nan") else ""
        edge = r.get("edge")
        edge_s = f"{edge:+.1%}" if isinstance(edge, (int, float)) else "—"
        odds = r.get("odds")
        odds_s = f"{odds:.2f}" if isinstance(odds, (int, float)) else "—"
        trs += (f"<tr><td>{str(r.get('date',''))[5:]}</td>"
                f"<td class='tm'>{html.escape(zt(str(r.get('away',''))))} @ "
                f"{html.escape(zt(str(r.get('home',''))))}</td>"
                f"<td>{_MLB_MK_ZH.get(r.get('market'), r.get('market'))} "
                f"{html.escape(str(r.get('selection','')))}{line}</td>"
                f"<td>{odds_s}</td>"
                f"<td style='color:#7be0b0'>{edge_s}</td></tr>")
    return (
        "<div class='card'><div class='sec'>✅ 已連結真實盤口</div>"
        f"<div class='small' style='color:#cdd9e5;line-height:1.7'>"
        f"系統已抓盤口並記錄 <b>{len(pending)}</b> 注模型 +EV 推薦（均注 1 單位），"
        "但這些比賽<b>尚未結束</b>，還沒有損益。賽後每日建站會自動用下注/收盤賠率"
        "結算，這裡就會出現累積損益曲線、CLV 走勢與勝過收盤比例。</div></div>"
        "<div class='card'><div class='sec'>🎯 待結算的 +EV 推薦（依 edge 排序）</div>"
        "<table><thead><tr><th>日期</th><th>對戰</th><th>下注</th><th>賠率</th>"
        f"<th>edge</th></tr></thead><tbody>{trs}</tbody></table></div>")


def render_mlb_perf_page(hist, track_text=None, pending=None,
                         title="MLB 下注績效 & CLV") -> str:
    """MLB 獨立績效頁：累積損益/ROI 主視覺 + CLV 折線 + 逐注過/沒過紀錄。

    與足球績效頁同結構，改用棒球盤口名（錢線/大小/讓分）與台灣譯名。
    三態：已結算→完整圖表；已連結全待結算→列出 +EV 推薦；皆無→引導設定盤口金鑰。
    """
    from . import mlb as _mlb
    zt = _mlb.zh_mlb
    today = _dt.date.today().isoformat()
    track_card = _track_card(track_text)
    if not hist:
        if pending:
            return _mlb_perf_doc(title, today, _mlb_pending_body(pending) + track_card)
        body = ("<div class='card'><div class='sec'>📈 尚無收益紀錄（無已結算的真實盤口下注）</div>"
                "<div class='small' style='color:var(--muted);line-height:1.7'>"
                "需要部署環境設定 <code>ODDS_API_KEY</code>，系統才會抓 MLB 真實盤口、"
                "只記模型相對市場有正期望值(+EV)的推薦，並在賽後用下注/收盤賠率"
                "算實際 ROI 與 CLV。<br>累積足夠注數後，這裡會出現："
                "<b>累積收益曲線、ROI、CLV 走勢</b>與勝過收盤比例。</div></div>")
        return _mlb_perf_doc(title, today, body + track_card)

    last = hist[-1]
    pl_pts = [r["cum_pl"] for r in hist]
    clv_pts = [r["cum_clv"] * 100 for r in hist]
    pl_color = "#7be0b0" if last["cum_pl"] >= 0 else "#e06a6a"
    hero = _pl_hero(last)
    kpis = (
        f"<div class='kpis'>"
        f"<div class='kpi'><span>注數</span><b>{last['n']}</b></div>"
        f"<div class='kpi'><span>平均 CLV</span><b>{last['cum_clv']:+.1%}</b></div>"
        f"<div class='kpi'><span>勝過收盤</span><b>{last['beat_rate']:.0%}</b></div>"
        f"</div>")
    charts = (
        f"<div class='card'><div class='sec'>💰 累積收益（單位）</div>"
        f"{_line_svg(pl_pts, color=pl_color)}</div>"
        f"<div class='card'><div class='sec'>🎯 累積平均 CLV（%，&gt;0 長期領先指標）</div>"
        f"{_line_svg(clv_pts, color='#6ea8fe')}</div>")
    log_trs = ""
    for r in reversed(hist[-40:]):
        rc = {"win": "#7be0b0", "loss": "#e06a6a", "push": "var(--muted)"}.get(r["result"], "")
        clv = f"{r['clv']:+.1%}" if r["clv"] is not None else "—"
        line = f" {r['line']}" if str(r["line"]) not in ("", "nan") else ""
        log_trs += (
            f"<tr><td>{str(r['date'])[5:]}</td>"
            f"<td class='tm'>{html.escape(zt(r['away']))} @ {html.escape(zt(r['home']))}</td>"
            f"<td>{_MLB_MK_ZH.get(r['market'], r['market'])} {html.escape(str(r['selection']))}{line}</td>"
            f"<td>{r['odds']:.2f}</td><td>{clv}</td>"
            f"<td style='color:{rc};font-weight:700'>{_RES_ZH.get(r['result'], r['result'])}</td>"
            f"<td style='color:{rc}'>{r['pl']:+.2f}</td></tr>")
    log_html = (f"<div class='card'><div class='sec'>📜 逐注紀錄（近 {min(len(hist),40)} 筆，過/沒過）</div>"
                f"<table><thead><tr><th>日期</th><th>對戰</th><th>下注</th><th>賠率</th>"
                f"<th>CLV</th><th>結果</th><th>損益</th></tr></thead>"
                f"<tbody>{log_trs}</tbody></table></div>")
    sub = f"{today} · 均注 1 單位 · 只計有真實盤口、模型 +EV 的下注"
    note = ("<div class='small' style='color:var(--muted);margin-top:4px'>"
            "CLV（closing line value）= 你拿到的賠率 vs 收盤賠率；長期 CLV&gt;0 是"
            "判斷模型能否真正贏過市場最可靠的領先指標，比短期勝率/ROI 抗雜訊。</div>")
    return _mlb_perf_doc(title, sub, hero + kpis + charts + note + log_html + track_card)


def write_worldcup_site(result, model, matches, outdir, history=None,
                        title="2026 世界盃預測", n_sims=20000, injury_counts=None,
                        track_text=None, ledger_path=None, odds_index=None,
                        mlb_html=None, mlb_perf_html=None, interactive=False):
    """產生多頁網站：index.html + 每場可分析的 match_<num>.html。

    可分析 = 雙方皆為模型已知球隊（小組賽全部；淘汰賽待隊伍確定後）。
    injury_counts：{隊名: 缺陣人數}（選用，來自 api-football），會套用到該場預期進球
        並在「球員狀態」欄顯示缺陣人數。
    """
    from . import analysis
    from .context import injuries_to_adjustment
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    injury_counts = injury_counts or {}

    linked = set()
    ko_rounds = {"Round of 32", "Round of 16", "Quarter-final", "Semi-final",
                 "Match for third place", "Final"}
    for m in matches:
        t1, t2 = m.team1, m.team2
        if t1 in model.attack and t2 in model.attack:
            knockout = m.round in ko_rounds
            ch, ca = injury_counts.get(t1, 0), injury_counts.get(t2, 0)
            adj = injuries_to_adjustment(ch, ca) if (ch or ca) else None
            # 有真實盤口就用盤口的主讓球線開盤（否則模型自行開盤）
            ah_override = None
            if odds_index and m.num in odds_index:
                try:
                    from . import tracker
                    ah_override = tracker.main_ah_line(odds_index[m.num])
                except Exception:  # noqa: BLE001
                    ah_override = None
            a = analysis.analyze(model, t1, t2, history=history, neutral=True,
                                 knockout=knockout, n_sims=n_sims, adjustment=adj,
                                 ah_line_override=ah_override)
            if injury_counts:
                a.player_note_home = f"缺陣 {ch} 人" if ch else "無重大缺陣"
                a.player_note_away = f"缺陣 {ca} 人" if ca else "無重大缺陣"
            rnd_label = m.group or m.round
            page_title = f"{zh(t1)} vs {zh(t2)}｜{group_zh(rnd_label)}"
            html_doc = render_analysis_html(a, page_title, back_href="index.html",
                                            interactive=interactive)
            (outdir / f"match_{m.num}.html").write_text(html_doc, encoding="utf-8")
            linked.add(m.num)

    write_worldcup_html(result, model, matches, outdir / "index.html", title, linked,
                        track_text=track_text)
    (outdir / "knockout.html").write_text(
        render_knockout_page(result, model, matches, linked), encoding="utf-8")
    # 績效頁：已結算→圖表；已連結待結算→列出 +EV 推薦；皆無→引導訊息
    hist, tune_res, pending = [], None, []
    if ledger_path:
        from . import tracker
        if track_text is None:  # 勝率卡：呼叫端沒給就從帳本自算（獨立 try，不被其他步驟拖累）
            try:
                track_text = tracker.summary(ledger_path).text()
            except Exception:  # noqa: BLE001
                track_text = None
        try:
            hist = tracker.history(ledger_path)
            df = tracker.load_ledger(ledger_path)
            pend = df[(df["source"] == "market") & (df["result"] == "pending")]
            for _, r in pend.iterrows():
                pending.append({"date": r["date"], "home": r["home"], "away": r["away"],
                                "market": r["market"], "selection": r["selection"],
                                "line": r["line"], "odds": tracker._to_float(r["odds"]),
                                "edge": tracker._to_float(r["edge"])})
        except Exception:  # noqa: BLE001
            hist, pending = [], []
        snap = str(Path(ledger_path).with_name("odds_log.csv"))
        if Path(snap).exists():
            try:
                tune_res = tracker.tune_weight(snap)
            except Exception:  # noqa: BLE001
                tune_res = None
    # AI 風控 / 賽後檢討（每日 1~2 次呼叫，僅設 ANTHROPIC_API_KEY 才跑）
    ai_risk = ai_review = None
    if ledger_path:
        try:
            from .agents import llm, roles
            if llm.available():
                ai_risk = roles.risk_review(pending) if pending else None
                settled = [h for h in hist][-40:]
                ai_review = roles.postmortem(settled) if settled else None
        except Exception:  # noqa: BLE001
            ai_risk = ai_review = None
    (outdir / "performance.html").write_text(
        render_performance_page(hist, tune=tune_res, pending=pending,
                                track_text=track_text, ai_risk=ai_risk,
                                ai_review=ai_review), encoding="utf-8")
    # MLB 分頁（呼叫端可傳現成 HTML；沒有就寫引導頁，導覽列連結不 404）
    if mlb_html is None:
        import datetime as _dt2
        mlb_html = render_mlb_page(
            [], date=_dt2.date.today().isoformat(), power=None, track_text=None,
            note="MLB 預測需在部署環境啟用：footy mlb fetch → fetch-pitchers → train，"
                 "每日建站會自動更新本頁。")
    (outdir / "mlb.html").write_text(mlb_html, encoding="utf-8")
    # MLB 績效頁（呼叫端可傳現成 HTML；沒有就從 MLB 帳本自算，連結不 404）
    if mlb_perf_html is None:
        try:
            from . import mlb as _mlbmod, tracker as _tr
            mlb_hist = _tr.history(_mlbmod.MLB_LEDGER)
            mlb_pending = []
            if Path(_mlbmod.MLB_LEDGER).exists():
                mdf = _tr.load_ledger(_mlbmod.MLB_LEDGER)
                mp = mdf[(mdf["source"] == "market") & (mdf["result"] == "pending")]
                for _, r in mp.iterrows():
                    mlb_pending.append({"date": r["date"], "home": r["home"],
                                        "away": r["away"], "market": r["market"],
                                        "selection": r["selection"], "line": r["line"],
                                        "odds": _tr._to_float(r["odds"]),
                                        "edge": _tr._to_float(r["edge"])})
            mlb_perf_html = render_mlb_perf_page(
                mlb_hist, track_text=_mlbmod.summary_text(_mlbmod.MLB_LEDGER),
                pending=mlb_pending)
        except Exception:  # noqa: BLE001
            mlb_perf_html = render_mlb_perf_page([], pending=[])
    (outdir / "mlb_perf.html").write_text(mlb_perf_html, encoding="utf-8")
    return outdir, len(linked)
