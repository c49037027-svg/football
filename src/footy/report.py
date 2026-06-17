"""把 MatchPrediction 渲染成 console / Markdown / HTML（預測站風格）。"""
from __future__ import annotations

import datetime as _dt
import html
from pathlib import Path

from .predict import MatchPrediction


def _score_str(s: tuple[int, int]) -> str:
    return f"{s[0]}-{s[1]}"


# ---------------- Console ----------------
def render_console(p: MatchPrediction) -> str:
    cs = "  ".join(f"{_score_str(s)} {prob:.0%}" for s, prob in p.correct_scores[:3])
    ou25 = p.over_under.get(2.5, {})
    return (
        f"⚽ {p.home} vs {p.away}\n"
        f"   1X2     : 主勝 {p.p_home:.0%} | 和 {p.p_draw:.0%} | 客勝 {p.p_away:.0%}\n"
        f"   預期進球: {p.exp_home_goals:.2f} - {p.exp_away_goals:.2f}"
        f"   預測比分: {_score_str(p.predicted_score)}\n"
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
            f"| 預測比分 | **{_score_str(p.predicted_score)}** |",
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
        <div class="box"><div class="k">預測比分</div><div class="v">{_score_str(p.predicted_score)}</div></div>
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
    L = []
    L.append(f"⚽ {a.home} vs {a.away}" + ("（中立場）" if a.neutral else ""))
    L.append(f"  AI 預測比分 {a.predicted_score[0]}-{a.predicted_score[1]}"
             f"（總進球 {a.total_goals}）· xG {a.xg_low}-{a.xg_high}")
    L.append(f"  1X2     主勝 {a.p_home:.0%} | 和 {a.p_draw:.0%} | 客勝 {a.p_away:.0%}")
    for ln in (1.5, 2.5, 3.5):
        d = a.over_under[ln]
        L.append(f"  大小{ln}  大 {d['over']:.0%} / 小 {d['under']:.0%}  "
                 f"建議 {'買大' if d['over']>=0.5 else '買小'}")
    L.append(f"  BTTS    是 {a.btts_yes:.0%}（{a.home} 進球 {a.p_home_scores:.0%}"
             f" × {a.away} 進球 {a.p_away_scores:.0%}）")
    L.append(f"  亞盤     supremacy(xG差) {a.ah_supremacy:+.2f} → {a.ah_reco}"
             f"（covers {a.ah_cover_prob:.0%}）")
    c = a.corners
    L.append(f"  角球     合計 {c.total}（{a.home} {c.home} / {a.away} {c.away}）"
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


def render_analysis_html(a, title: str = "單場分析") -> str:
    import datetime as _dt
    today = _dt.date.today().isoformat()
    venue = "中立場" if a.neutral else "主客場"
    ou_boxes = "".join(_ou_box(ln, a.over_under[ln]) for ln in (1.5, 2.5, 3.5))
    fh_ou = "".join(_fh_ou_box(ln, p) for ln, p in a.fh_over.items())
    c, k = a.corners, a.cards
    ah_col = _reco_color("大")
    h, d, aw = a.p_home, a.p_draw, a.p_away
    return f"""<!doctype html><html lang="zh-Hant"><head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>{html.escape(title)}</title><style>{_CSS}
.sec{{font-weight:700;margin:18px 0 8px;font-size:15px}}
.reco{{font-size:20px;font-weight:800;margin-top:6px}}
.small{{color:var(--muted);font-size:12px}}
.split{{display:grid;grid-template-columns:1fr 1fr;gap:14px}}
@media(max-width:560px){{.split{{grid-template-columns:1fr}}}}
</style></head>
<body><div class="wrap">
  <h1>⚽ {html.escape(a.home)} <span style="color:#8a97a6">vs</span> {html.escape(a.away)}</h1>
  <div class="sub">{today} · {venue} · 蒙地卡羅 {a.n_sims:,} 次 + Poisson · Dixon–Coles</div>
  <div class="disc">⚠️ 純機率分析，非投注建議。角球/黃牌為先驗近似（國際賽無公開統計）。投注有風險。</div>

  <div class="card">
    <div class="head"><div class="teams">AI 預測比分 {a.predicted_score[0]}-{a.predicted_score[1]}</div>
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
      <div class="small">{html.escape(a.home)} 進球 {a.p_home_scores:.0%} × {html.escape(a.away)} 進球 {a.p_away_scores:.0%}</div>
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
      <div class="head"><div>{html.escape(a.home)} <b>{c.home}</b></div>
        <div style="font-size:22px;font-weight:800;color:var(--warn)">{c.total}</div>
        <div><b>{c.away}</b> {html.escape(a.away)}</div></div>
      <div class="reco" style="color:{_reco_color(c.recommend)}">{c.recommend} {c.line}</div>
      <div class="small">模型估線 {c.line}（點估計 {c.edge_vs_line:+.1f}）· 信心 {c.confidence:.0%}</div>
    </div>
    <div class="card">
      <div class="sec">🟨 黃牌預測</div>
      <div class="head"><div>{html.escape(a.home)} <b>{k.home}</b></div>
        <div style="font-size:22px;font-weight:800;color:var(--warn)">{k.total}</div>
        <div><b>{k.away}</b> {html.escape(a.away)}</div></div>
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
      <div class="box"><div class="k">近5場 {html.escape(a.home)}</div><div class="v form">{_form_html(a.home_form)}</div></div>
      <div class="box"><div class="k">近5場 {html.escape(a.away)}</div><div class="v form">{_form_html(a.away_form)}</div></div>
      <div class="box"><div class="k">戰術對比</div><div class="v" style="font-size:13px">{a.home_style} / {a.away_style}</div></div>
      <div class="box"><div class="k">球員狀態 {html.escape(a.home)}</div><div class="v" style="font-size:12px">{html.escape(a.player_note_home)}</div></div>
      <div class="box"><div class="k">球員狀態 {html.escape(a.away)}</div><div class="v" style="font-size:12px">{html.escape(a.player_note_away)}</div></div>
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
