"""本機互動網頁：用下拉選單選隊伍/陣型/缺陣，輸入或自動抓盤口讓球線，即時重算分析。

靜態網站（GitHub Pages）無法即時重算（模型在 Python 跑），所以這支用 Python 內建
http.server 起一個本機網站：表單送出 → 跑 analysis.analyze → 回傳分析頁。零額外套件。

用法：footy serve --model models/intl.pkl --history data/intl.csv [--schedule data/wc2026.json]
然後瀏覽器開 http://localhost:8000
"""
from __future__ import annotations

import html as _html
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

from . import analysis, context, report
from .i18n import zh
from .models.dixon_coles import DixonColesModel

_FORMATIONS = ["", "4-4-2", "4-3-3", "4-2-3-1", "4-1-4-1", "4-5-1", "4-4-1-1",
               "4-1-2-1-2", "4-3-1-2", "4-2-2-2", "4-1-3-2", "4-3-2-1",
               "3-5-2", "3-4-3", "3-4-2-1", "3-4-1-2", "3-1-4-2", "3-5-1-1",
               "3-2-4-1", "5-3-2", "5-4-1", "5-2-3", "5-2-2-1", "4-2-4", "3-3-4"]


def _team_options(teams: list[str], selected: str = "") -> str:
    opts = []
    for t in teams:
        sel = " selected" if t == selected else ""
        opts.append(f'<option value="{_html.escape(t)}"{sel}>{_html.escape(zh(t))}</option>')
    return "".join(opts)


def _form_datalist() -> str:
    opts = "".join(f'<option value="{f}">' for f in _FORMATIONS if f)
    return f'<datalist id="forms">{opts}</datalist>'


def render_form(teams: list[str], q: dict | None = None) -> str:
    q = q or {}
    home = q.get("home", teams[0] if teams else "")
    away = q.get("away", teams[1] if len(teams) > 1 else "")
    css = report._CSS
    return f"""<!doctype html><html lang="zh-Hant"><head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>單場分析（互動）</title><style>{css}
form{{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:16px 18px}}
label{{display:block;font-size:12px;color:var(--muted);margin:10px 0 4px}}
select,input{{width:100%;padding:9px;background:#11161c;color:#e6edf3;
border:1px solid var(--line);border-radius:8px;font-size:15px}}
.row{{display:grid;grid-template-columns:1fr 1fr;gap:12px}}
button{{margin-top:16px;width:100%;padding:12px;background:var(--accent);color:#04130c;
border:0;border-radius:8px;font-size:16px;font-weight:800}}
</style></head><body><div class="wrap">
  <h1>⚽ 單場深度分析（互動）</h1>
  <div class="sub">選隊伍／陣型／缺陣，或自行輸入盤口讓球線，按「分析」即時計算</div>
  <form action="/analyze" method="get">
    <div class="row">
      <div><label>主隊</label><select name="home">{_team_options(teams, home)}</select></div>
      <div><label>客隊</label><select name="away">{_team_options(teams, away)}</select></div>
    </div>
    {_form_datalist()}
    <div class="row">
      <div><label>主隊陣型（可選清單或自行輸入）</label>
        <input name="hf" list="forms" value="{_html.escape(q.get('hf',''))}" placeholder="例：4-3-3 / 3-4-1-2"></div>
      <div><label>客隊陣型</label>
        <input name="af" list="forms" value="{_html.escape(q.get('af',''))}" placeholder="例：5-3-2"></div>
    </div>
    <div class="row">
      <div><label>主隊缺陣主力（選填，不知道就留 0）</label><input type="number" name="hm" value="{q.get('hm','0')}" min="0" max="11"></div>
      <div><label>客隊缺陣主力（選填，不知道就留 0）</label><input type="number" name="am" value="{q.get('am','0')}" min="0" max="11"></div>
    </div>
    <div class="small" style="color:var(--muted);font-size:11px;margin-top:2px">
      不確定誰是主力？留 0 即可——模型的球隊強度(Elo+進球史)已假設正常陣容。只有你看到傷停新聞時才填。</div>
    <div class="row">
      <div><label>盤口讓球線（主隊視角，如 -1.5；留空=模型自動開盤）</label>
        <input type="text" name="ah" value="{q.get('ah','')}" placeholder="例：-1.5"></div>
      <div><label>場地</label><select name="neutral">
        <option value="1" selected>中立場（世界盃）</option>
        <option value="0">主客場</option></select></div>
    </div>
    <label><input type="checkbox" name="ko" value="1" style="width:auto"> 淘汰賽（黃牌加成）</label>
    <label><input type="checkbox" name="fetch" value="1" style="width:auto"> 自動抓 The Odds API 盤口（需設 ODDS_API_KEY）</label>
    <button type="submit">分析 →</button>
  </form>
  <div class="foot">本機互動工具 · footy</div>
</div></body></html>"""


class _Handler(BaseHTTPRequestHandler):
    model: DixonColesModel = None
    history = None
    teams: list[str] = []

    def _send(self, body: str, code: int = 200):
        data = body.encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, *args):  # 靜音
        pass

    def do_GET(self):
        parsed = urlparse(self.path)
        if parsed.path in ("/", "/index.html"):
            self._send(render_form(self.teams))
            return
        if parsed.path != "/analyze":
            self._send("<p>Not found</p>", 404)
            return
        q = {k: v[0] for k, v in parse_qs(parsed.query).items()}
        try:
            self._send(self._analyze(q))
        except Exception as e:  # noqa: BLE001
            self._send(f"<div class='wrap'><p>分析失敗：{_html.escape(str(e))}</p>"
                       f"<a href='/'>← 返回</a></div>", 500)

    def _analyze(self, q: dict) -> str:
        home, away = q.get("home"), q.get("away")
        if home not in self.model.attack or away not in self.model.attack:
            raise ValueError("球隊不在模型中")
        hf, af = q.get("hf", ""), q.get("af", "")
        hm, am = int(q.get("hm", 0) or 0), int(q.get("am", 0) or 0)
        neutral = q.get("neutral", "1") == "1"
        knockout = q.get("ko") == "1"

        ah_line = None
        note = ""
        if q.get("ah", "").strip():
            try:
                ah_line = float(q["ah"])
            except ValueError:
                ah_line = None
        elif q.get("fetch") == "1":
            try:
                from .live.providers import fetch_ah_line
                res = fetch_ah_line(home, away)
                if res:
                    ah_line = res["line"]
                    note = f"（已抓盤口 {ah_line:+g}）"
            except Exception as e:  # noqa: BLE001
                note = f"（抓盤口失敗：{e}）"

        adj = context.combine_adjustments(
            context.formation_adjustment(hf or None, af or None),
            context.injuries_to_adjustment(hm, am) if (hm or am) else None,
        )
        a = analysis.analyze(self.model, home, away, history=self.history,
                             neutral=neutral, knockout=knockout, n_sims=20000,
                             adjustment=adj, home_formation=hf, away_formation=af,
                             ah_line_override=ah_line)
        if hm or am:
            a.player_note_home = f"缺主力 {hm} 人" if hm else "主力盡出"
            a.player_note_away = f"缺主力 {am} 人" if am else "主力盡出"
        page = report.render_analysis_html(a, f"{zh(home)} vs {zh(away)}{note}",
                                           back_href="/")
        return page


def serve(model_path: str, history_path: str | None = None,
          schedule_path: str | None = None, host: str = "0.0.0.0",
          port: int = 8000):
    model = DixonColesModel.load(model_path)
    _Handler.model = model
    _Handler.history = _load_history(history_path)
    teams = _server_teams(model, schedule_path)
    _Handler.teams = teams
    srv = ThreadingHTTPServer((host, port), _Handler)
    print(f"[serve] 互動分析網頁：http://localhost:{port} （Ctrl-C 結束）")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\n[serve] 已停止。")
    finally:
        srv.server_close()


def _load_history(path):
    if not path:
        return None
    from .data import loader
    return loader.load_csv(path)


def _server_teams(model, schedule_path):
    """優先用世界盃 48 隊（中文好選），否則用全模型隊伍。"""
    if schedule_path:
        try:
            from .worldcup import parse_wc_json
            groups, _, _ = parse_wc_json(schedule_path)
            ts = sorted({t for v in groups.values() for t in v if t in model.attack})
            if ts:
                return ts
        except Exception:  # noqa: BLE001
            pass
    return sorted(model.teams)
