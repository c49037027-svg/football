"""把 MatchPrediction 渲染成 console / Markdown / HTML（預測站風格）。"""
from __future__ import annotations

import datetime as _dt
import html
from pathlib import Path

from .predict import MatchPrediction
from .i18n import zh, group_zh


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


def render_analysis_html(a, title: str = "單場分析", back_href: str | None = None,
                         interactive: bool = False) -> str:
    import datetime as _dt
    today = _dt.date.today().isoformat()
    venue = "中立場" if a.neutral else "主客場"
    rebar = _reanalyze_bar(a) if interactive else ""
    sup = a.data_support
    ou_odds = a.ou_odds or {}
    ou_boxes = "".join(_ou_box(ln, a.over_under[ln], ou_odds.get(ln), sup)
                       for ln in (1.5, 2.5, 3.5))
    fh_ou = "".join(_fh_ou_box(ln, p, sup) for ln, p in a.fh_over.items())
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

  <div class="card">
    <div class="head"><div class="teams">AI 最可能比分 {a.predicted_score[0]}-{a.predicted_score[1]}
      <span class="small">（僅 {a.predicted_score_prob:.0%}，眾數非平均）</span></div>
      <div class="small">期望總進球 <b style="color:#cdd9e5">{a.total_goals}</b> · xG {a.xg_low}–{a.xg_high}</div></div>
    <div class="small" style="margin:-4px 0 8px;color:var(--muted)">
      註：最可能比分是「機率最高的單一比分」（偏低），但大小球看的是<b>期望總進球 {a.total_goals}</b> 與整體分布，故兩者可能方向不同。</div>
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
      <div class="sec">⚖️ 亞盤讓球（模型開盤）</div>
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
    from .models import markets
    t1, t2 = m.team1, m.team2
    cs5 = ""
    if m.played:
        tag = f"<span class='res'>{m.hg}-{m.ag}</span> <span class='small'>(已賽)</span>"
    elif t1 in model.attack and t2 in model.attack:
        mat = model.score_matrix(t1, t2, neutral=True)
        s = markets.most_likely_score(mat)
        sp = float(mat[s[0], s[1]])
        o = markets.outcome_1x2(mat)
        tag = (f"<span class='pred'>{s[0]}-{s[1]}</span><span class='small'>（{sp:.0%}）</span> "
               f"<span class='small'>{o['home']:.0%}/{o['draw']:.0%}/{o['away']:.0%}</span>")
        top = markets.correct_score(mat, top_n=5)
        cs5 = ("<div class='cs5'>正確比分：" +
               " · ".join(f"{a}-{b} {p*100:.0f}%" for (a, b), p in top) + "</div>")
    else:
        tag = "<span class='small'>—</span>"
    inner = (f"<span class='fxd'>{m.date[5:]}</span>"
             f"<span class='fxt'>{html.escape(zh(t1))}</span>"
             f"<span class='fxm'>{tag}</span>"
             f"<span class='fxt r'>{html.escape(zh(t2))}</span>")
    played = "1" if m.played else "0"
    if linked and m.num in linked:
        body = (f"<a class='fx fxlink' href='match_{m.num}.html'>{inner}"
                f"<span class='arow'>›</span></a>{cs5}")
    else:
        body = f"<div class='fx'>{inner}</div>{cs5}"
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
    return (f"<div class='stbl'><table><thead><tr><th>#</th><th>隊伍</th>"
            f"<th>賽</th><th>勝</th><th>平</th><th>負</th><th>進失</th><th>淨</th>"
            f"<th>分</th></tr></thead><tbody>{''.join(trs)}</tbody></table></div>")


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
    if standings:
        standings = "<div class='sublbl'>目前積分榜</div>" + standings + "<div class='sublbl'>晉級預測</div>"
    fixtures = "".join(_match_pred_row(model, m, linked)
                       for m in sorted([mm for mm in matches if mm.group == g],
                                       key=lambda mm: mm.date))
    return f"""
    <div class="card grp">
      <div class="sec">{html.escape(group_zh(g))}</div>
      {standings}
      <table class="gt"><thead><tr><th>#</th><th>隊伍</th><th>首名</th><th>前二</th><th>晉級</th><th>預期分</th></tr></thead>
      <tbody>{''.join(trs)}</tbody></table>
      <div class="fxs">{fixtures}</div>
    </div>"""


def _navbar(active: str) -> str:
    """頂部導覽列。active: 'home' 或 'custom'。"""
    def cls(k):
        return " class='on'" if k == active else ""
    return (f"<div class='nav'><a href='/'{cls('home')}>🏆 世界盃預測</a>"
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
    rms = sorted([m for m in matches if m.date == day], key=lambda m: m.num)
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
        inner = (f"<span class='fxt'>{html.escape(t1)}</span>"
                 f"<span class='fxm'>{mid}</span>"
                 f"<span class='fxt r'>{html.escape(t2)}</span>")
        if m.num in linked:
            rows.append(f"<a class='fx fxlink' style='grid-template-columns:1fr auto 1fr 14px' "
                        f"href='match_{m.num}.html'>{inner}<span class='arow'>›</span></a>")
        else:
            rows.append(f"<div class='fx' style='grid-template-columns:1fr auto 1fr'>{inner}</div>")
    return (f"<div class='card'><div class='sec'>📅 {label}</div>"
            f"<div class='fxs'>{''.join(rows)}</div></div>")


def render_worldcup_html(result, model, matches, title: str = "2026 世界盃預測",
                         linked: set | None = None) -> str:
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

    # 晉級展望表（前 16 依冠軍機率）
    ko_rows = "".join(
        f"<tr><td class='tm'>{html.escape(zh(t))}</td>"
        f"<td>{result.qualify.get(t,0):.0%}</td><td>{result.r16.get(t,0):.0%}</td>"
        f"<td>{result.quarter.get(t,0):.0%}</td><td>{result.semi.get(t,0):.0%}</td>"
        f"<td>{result.final.get(t,0):.0%}</td>"
        f"<td style=\"{_heat(p*100,'48')}\"><b>{p:.1%}</b></td></tr>"
        for t, p in top)

    groups_html = "".join(
        _group_card(g, result.groups[g], result, model, matches, linked)
        for g in sorted(result.groups))

    ko_html = _render_bracket(matches, linked)
    today_html = _today_section(matches, model, linked)

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
.sublbl{{font-size:11px;color:var(--muted);margin:8px 0 3px;font-weight:700}}
.stbl{{overflow-x:auto}}.stbl table{{font-size:11px;min-width:280px}}
.stbl th,.stbl td{{padding:4px 5px}}
.fx{{display:grid;grid-template-columns:42px 1fr auto 1fr;gap:6px;align-items:center;
font-size:12px;padding:3px 0;border-top:1px solid #1c242d}}
.fxt{{font-weight:600}}.fxt.r{{text-align:right}}.fxd{{color:var(--muted)}}.fxm{{text-align:center}}
.pred{{color:#7be0b0;font-weight:700}}.res{{color:#e0b341;font-weight:700}}
.cs5{{font-size:11px;color:var(--muted);padding:0 0 5px 42px;letter-spacing:.02em}}
a.fxlink{{text-decoration:none;color:inherit;grid-template-columns:42px 1fr auto 1fr 14px}}
a.fxlink:active{{background:#1c242d}}.arow{{color:var(--accent);text-align:right}}
</style></head>
<body><div class="wrap">
  {_navbar('home')}
  <h1>🏆 {html.escape(title)}</h1>
  <div class="sub">{today} · 蒙地卡羅 {result.n_sims:,} 次 · Dixon–Coles + Elo · 已踢比分納入</div>
  <div class="disc">⚠️ 純機率預測，非投注建議。最佳第三名→R32 槽位為近似指派；晉級機率為主要可信輸出。</div>

  {today_html}

  <div class="two">
    <div class="card">
      <div class="sec">🥇 奪冠機率</div>
      {champ_bars}
    </div>
    <div class="card">
      <div class="sec">📈 晉級展望</div>
      <table><thead><tr><th>隊伍</th><th>晉級</th><th>16強</th><th>8強</th><th>4強</th><th>決賽</th><th>奪冠</th></tr></thead>
      <tbody>{ko_rows}</tbody></table>
    </div>
  </div>

  <div class="sec" style="margin-top:22px">🏟️ 淘汰賽對陣圖</div>
  {ko_html}

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
                        linked=None):
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(render_worldcup_html(result, model, matches, title, linked),
                        encoding="utf-8")
    return out_path


def write_worldcup_site(result, model, matches, outdir, history=None,
                        title="2026 世界盃預測", n_sims=20000, injury_counts=None,
                        interactive=False):
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
            a = analysis.analyze(model, t1, t2, history=history, neutral=True,
                                 knockout=knockout, n_sims=n_sims, adjustment=adj)
            if injury_counts:
                a.player_note_home = f"缺陣 {ch} 人" if ch else "無重大缺陣"
                a.player_note_away = f"缺陣 {ca} 人" if ca else "無重大缺陣"
            rnd_label = m.group or m.round
            page_title = f"{zh(t1)} vs {zh(t2)}｜{group_zh(rnd_label)}"
            html_doc = render_analysis_html(a, page_title, back_href="index.html",
                                            interactive=interactive)
            (outdir / f"match_{m.num}.html").write_text(html_doc, encoding="utf-8")
            linked.add(m.num)

    write_worldcup_html(result, model, matches, outdir / "index.html", title, linked)
    return outdir, len(linked)
