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
    L.append(f"  AI 預測比分 {a.predicted_score[0]}-{a.predicted_score[1]}"
             f"（機率 {a.predicted_score_prob:.0%}；總進球 {a.total_goals}）· xG {a.xg_low}-{a.xg_high}")
    L.append(f"  1X2     主勝 {a.p_home:.0%} | 和 {a.p_draw:.0%} | 客勝 {a.p_away:.0%}")
    for ln in (1.5, 2.5, 3.5):
        d = a.over_under[ln]
        L.append(f"  大小{ln}  大 {d['over']:.0%} / 小 {d['under']:.0%}  "
                 f"建議 {'買大' if d['over']>=0.5 else '買小'}")
    L.append(f"  BTTS    是 {a.btts_yes:.0%}（{hz} 進球 {a.p_home_scores:.0%}"
             f" × {az} 進球 {a.p_away_scores:.0%}）")
    L.append(f"  亞盤     supremacy(xG差) {a.ah_supremacy:+.2f} → {a.ah_reco}"
             f"（covers {a.ah_cover_prob:.0%}）")
    c = a.corners
    L.append(f"  角球     合計 {c.total}（{hz} {c.home} / {az} {c.away}）"
             f" 估線 {c.line} {c.recommend}（信心 {c.confidence:.0%}）")
    k = a.cards
    L.append(f"  黃牌     合計 {k.total} 估線 {k.line} {k.recommend}（信心 {k.confidence:.0%}）")
    L.append(f"  上半場   主 {a.fh_home:.0%} / 平 {a.fh_draw:.0%} / 客 {a.fh_away:.0%}"
             f"  大0.5 {a.fh_over[0.5]:.0%} / 大1.5 {a.fh_over[1.5]:.0%}")
    L.append(f"  因子     Elo {a.elo_home:.0f} vs {a.elo_away:.0f}｜狀態 {a.home_form or '-'}"
             f" / {a.away_form or '-'}｜{a.h2h}｜戰術 {a.home_style} vs {a.away_style}")
    return "\n".join(L)


def _reco_color(reco: str) -> str:
    if "大" in reco or "是" in reco:
        return "#21c07a"
    if "小" in reco or "否" in reco:
        return "#e07a5f"
    return "#e0b341"


def _ou_box(line, d):
    reco = "買大" if d["over"] >= 0.5 else "買小"
    col = _reco_color(reco)
    return (f"<div class='box'><div class='k'>線 {line}</div>"
            f"<div class='v'>大 {d['over']:.0%}</div>"
            f"<div style='color:{col};font-weight:700;margin-top:4px'>{reco}</div></div>")


def _fh_ou_box(line, p):
    reco = "買大" if p >= 0.5 else "買小"
    col = _reco_color(reco)
    return (f"<div class='box'><div class='k'>上半場線 {line}</div>"
            f"<div class='v'>大 {p:.0%}</div>"
            f"<div style='color:{col};font-weight:700;margin-top:4px'>{reco}</div></div>")


def render_analysis_html(a, title: str = "單場分析", back_href: str | None = None) -> str:
    import datetime as _dt
    today = _dt.date.today().isoformat()
    venue = "中立場" if a.neutral else "主客場"
    ou_boxes = "".join(_ou_box(ln, a.over_under[ln]) for ln in (1.5, 2.5, 3.5))
    fh_ou = "".join(_fh_ou_box(ln, p) for ln, p in a.fh_over.items())
    c, k = a.corners, a.cards
    ah_col = _reco_color("大")
    h, d, aw = a.p_home, a.p_draw, a.p_away
    nav = (f'<a class="back" href="{back_href}">← 返回首頁</a>' if back_href else "")
    return f"""<!doctype html><html lang="zh-Hant"><head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>{html.escape(title)}</title><style>{_CSS}
.sec{{font-weight:700;margin:18px 0 8px;font-size:15px}}
.reco{{font-size:20px;font-weight:800;margin-top:6px}}
.small{{color:var(--muted);font-size:12px}}
.split{{display:grid;grid-template-columns:1fr 1fr;gap:14px}}
.back{{display:inline-block;color:var(--accent);text-decoration:none;font-size:14px;margin-bottom:10px}}
@media(max-width:560px){{.split{{grid-template-columns:1fr}}}}
</style></head>
<body><div class="wrap">
  {nav}
  <h1>⚽ {html.escape(zh(a.home))} <span style="color:#8a97a6">vs</span> {html.escape(zh(a.away))}</h1>
  <div class="sub">{today} · {venue} · 蒙地卡羅 {a.n_sims:,} 次 + Poisson · Dixon–Coles</div>
  <div class="disc">⚠️ 純機率分析，非投注建議。角球/黃牌為先驗近似（國際賽無公開統計）。投注有風險。</div>

  <div class="card">
    <div class="head"><div class="teams">AI 預測比分 {a.predicted_score[0]}-{a.predicted_score[1]}
      <span class="small">（機率 {a.predicted_score_prob:.0%}）</span></div>
      <div class="small">總進球 {a.total_goals} · xG {a.xg_low}–{a.xg_high}</div></div>
    <div class="bar">
      <div class="h" style="width:{h*100:.1f}%">{h:.0%}</div>
      <div class="d" style="width:{d*100:.1f}%">{d:.0%}</div>
      <div class="a" style="width:{aw*100:.1f}%">{aw:.0%}</div>
    </div>
    <div class="small">主勝 / 和 / 客勝</div>
  </div>

  <div class="sec">⚽ 大小球</div>
  <div class="grid">{ou_boxes}</div>

  <div class="split">
    <div class="card">
      <div class="sec">🥅 兩隊都進球 BTTS</div>
      <div class="reco" style="color:{_reco_color('買是' if a.btts_yes>=0.5 else '買否')}">
        {'買是' if a.btts_yes>=0.5 else '買否'}（{a.btts_yes:.0%}）</div>
      <div class="small">{html.escape(zh(a.home))} 進球 {a.p_home_scores:.0%} × {html.escape(zh(a.away))} 進球 {a.p_away_scores:.0%}</div>
    </div>
    <div class="card">
      <div class="sec">⚖️ 亞盤讓球</div>
      <div class="v" style="font-size:18px">{html.escape(a.ah_reco)}</div>
      <div class="small">模型 supremacy(xG差)：{a.ah_supremacy:+.2f}　covers {a.ah_cover_prob:.0%}</div>
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
    <div class="grid" style="margin-top:10px">{fh_ou}</div>
  </div>

  <div class="sec">📊 影響因子</div>
  <div class="card">
    <div class="grid">
      <div class="box"><div class="k">Elo 評分</div><div class="v">{a.elo_home:.0f} vs {a.elo_away:.0f}</div></div>
      <div class="box"><div class="k">近5場 {html.escape(zh(a.home))}</div><div class="v form">{_form_html(a.home_form)}</div></div>
      <div class="box"><div class="k">近5場 {html.escape(zh(a.away))}</div><div class="v form">{_form_html(a.away_form)}</div></div>
      <div class="box"><div class="k">戰術對比</div><div class="v" style="font-size:13px">{a.home_style} / {a.away_style}</div></div>
      <div class="box"><div class="k">球員狀態 {html.escape(zh(a.home))}</div><div class="v" style="font-size:12px">{html.escape(a.player_note_home)}</div></div>
      <div class="box"><div class="k">球員狀態 {html.escape(zh(a.away))}</div><div class="v" style="font-size:12px">{html.escape(a.player_note_away)}</div></div>
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
    if linked and m.num in linked:
        return (f"<a class='fx fxlink' href='match_{m.num}.html'>{inner}"
                f"<span class='arow'>›</span></a>{cs5}")
    return f"<div class='fx'>{inner}</div>{cs5}"


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
    fixtures = "".join(_match_pred_row(model, m, linked)
                       for m in sorted([mm for mm in matches if mm.group == g],
                                       key=lambda mm: mm.date))
    return f"""
    <div class="card grp">
      <div class="sec">{html.escape(group_zh(g))}</div>
      <table class="gt"><thead><tr><th>#</th><th>隊伍</th><th>首名</th><th>前二</th><th>晉級</th><th>預期分</th></tr></thead>
      <tbody>{''.join(trs)}</tbody></table>
      <div class="fxs">{fixtures}</div>
    </div>"""


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

    # 淘汰賽賽程（槽位碼；隊伍已定且有分析頁則可點）
    ko_order = ["Round of 32", "Round of 16", "Quarter-final", "Semi-final",
                "Match for third place", "Final"]
    ko_names = {"Round of 32": "32 強", "Round of 16": "16 強", "Quarter-final": "8 強",
                "Semi-final": "4 強", "Match for third place": "季軍戰", "Final": "決賽"}
    ko_sections = []
    for rnd in ko_order:
        rms = sorted([m for m in matches if m.round == rnd], key=lambda m: (m.date, m.num))
        if not rms:
            continue
        rows = []
        for m in rms:
            if m.played:
                mid = f"<span class='res'>{m.hg}-{m.ag}</span>"
            else:
                mid = "<span class='small'>待定</span>"
            t1d = _slot_zh(m.team1)
            t2d = _slot_zh(m.team2)
            inner = (f"<span class='fxd'>{m.date[5:]}</span>"
                     f"<span class='fxt'>{html.escape(t1d)}</span>"
                     f"<span class='fxm'>{mid}</span>"
                     f"<span class='fxt r'>{html.escape(t2d)}</span>")
            if linked and m.num in linked:
                rows.append(f"<a class='fx fxlink' href='match_{m.num}.html'>{inner}<span class='arow'>›</span></a>")
            else:
                rows.append(f"<div class='fx'>{inner}</div>")
        ko_sections.append(
            f"<div class='card grp'><div class='sec'>{ko_names[rnd]}</div>"
            f"<div class='fxs'>{''.join(rows)}</div></div>")
    ko_html = "".join(ko_sections)

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
.fx{{display:grid;grid-template-columns:42px 1fr auto 1fr;gap:6px;align-items:center;
font-size:12px;padding:3px 0;border-top:1px solid #1c242d}}
.fxt{{font-weight:600}}.fxt.r{{text-align:right}}.fxd{{color:var(--muted)}}.fxm{{text-align:center}}
.pred{{color:#7be0b0;font-weight:700}}.res{{color:#e0b341;font-weight:700}}
.cs5{{font-size:11px;color:var(--muted);padding:0 0 5px 42px;letter-spacing:.02em}}
a.fxlink{{text-decoration:none;color:inherit;grid-template-columns:42px 1fr auto 1fr 14px}}
a.fxlink:active{{background:#1c242d}}.arow{{color:var(--accent);text-align:right}}
</style></head>
<body><div class="wrap">
  <h1>🏆 {html.escape(title)}</h1>
  <div class="sub">{today} · 蒙地卡羅 {result.n_sims:,} 次 · Dixon–Coles + Elo · 已踢比分納入</div>
  <div class="disc">⚠️ 純機率預測，非投注建議。最佳第三名→R32 槽位為近似指派；晉級機率為主要可信輸出。</div>

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

  <div class="sec" style="margin-top:22px">📋 小組賽程與預測（點擊看單場分析）</div>
  <div class="grids">{groups_html}</div>

  <div class="sec" style="margin-top:22px">🏟️ 淘汰賽賽程</div>
  <div class="grids">{ko_html}</div>

  <div class="foot">Generated by footy · 研究與教育用途 · 資料：openfootball + martj42</div>
</div></body></html>"""


def write_worldcup_html(result, model, matches, out_path, title="2026 世界盃預測",
                        linked=None):
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(render_worldcup_html(result, model, matches, title, linked),
                        encoding="utf-8")
    return out_path


def write_worldcup_site(result, model, matches, outdir, history=None,
                        title="2026 世界盃預測", n_sims=20000):
    """產生多頁網站：index.html + 每場可分析的 match_<num>.html。

    可分析 = 雙方皆為模型已知球隊（小組賽全部；淘汰賽待隊伍確定後）。
    """
    from . import analysis
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    linked = set()
    ko_rounds = {"Round of 32", "Round of 16", "Quarter-final", "Semi-final",
                 "Match for third place", "Final"}
    for m in matches:
        t1, t2 = m.team1, m.team2
        if t1 in model.attack and t2 in model.attack:
            knockout = m.round in ko_rounds
            a = analysis.analyze(model, t1, t2, history=history, neutral=True,
                                 knockout=knockout, n_sims=n_sims)
            rnd_label = m.group or m.round
            page_title = f"{zh(t1)} vs {zh(t2)}｜{group_zh(rnd_label)}"
            html_doc = render_analysis_html(a, page_title, back_href="index.html")
            (outdir / f"match_{m.num}.html").write_text(html_doc, encoding="utf-8")
            linked.add(m.num)

    write_worldcup_html(result, model, matches, outdir / "index.html", title, linked)
    return outdir, len(linked)
