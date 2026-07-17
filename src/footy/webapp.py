"""本機互動網頁：用下拉選單選隊伍/陣型/缺陣，輸入或自動抓盤口讓球線，即時重算分析。

靜態網站（GitHub Pages）無法即時重算（模型在 Python 跑），所以這支用 Python 內建
http.server 起一個本機網站：表單送出 → 跑 analysis.analyze → 回傳分析頁。零額外套件。

用法：footy serve --model models/intl.pkl --history data/intl.csv [--schedule data/wc2026.json]
然後瀏覽器開 http://localhost:8000
"""
from __future__ import annotations

import html as _html
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

from . import analysis, context, report
from .i18n import zh
from .models.dixon_coles import DixonColesModel

# 走地頁 TTL 快取：命中直接回 HTML，上游請求上限 = 每 TTL 週期一輪，
# 與同時瀏覽人數脫鉤。TTL 20 秒 < 頁面 45 秒 refresh。
# _live_build_lock = single-flight：快取過期瞬間的併發請求只讓一個重建、
# 其餘等它寫回後讀快取，避免一起打上游（stampede）。
_LIVE_TTL = 20.0
_live_cache: dict[str, tuple[float, str]] = {}
_live_lock = threading.Lock()
_live_build_lock = threading.Lock()


def _live_cache_get(key: str) -> str | None:
    import time
    with _live_lock:
        hit = _live_cache.get(key)
    if hit and time.monotonic() - hit[0] < _LIVE_TTL:
        return hit[1]
    return None


def _live_cache_put(key: str, html: str) -> None:
    import time
    with _live_lock:
        _live_cache[key] = (time.monotonic(), html)

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
    site_dir: str | None = None  # 啟動時產生的世界盃靜態網站目錄

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
        path = parsed.path
        # 自訂分析表單
        if path == "/custom":
            self._send(render_form(self.teams))
            return
        # 輕量健康檢查（保溫 ping 用；不觸發上游 API 與模擬）
        if path == "/healthz":
            self._send("ok")
            return
        # 走地（即時勝率；Render 可達 statsapi/ESPN）
        # /live = 足球+MLB 綜合；/mlb-live = 只看 MLB
        # 20 秒 TTL 快取：上游請求量與同時瀏覽人數脫鉤（meta refresh 45s × N 人）
        if path in ("/live", "/mlb-live"):
            cached = _live_cache_get(path)
            if cached is not None:
                self._send(cached)
                return
            # single-flight：過期瞬間的併發請求只讓一個重建，其餘等快取
            with _live_build_lock:
                cached = _live_cache_get(path)   # 排隊期間可能已被前一位建好
                if cached is not None:
                    self._send(cached)
                    return
                self._build_live(path)
            return
        # 手動走地：看直播自行輸入比分 → 即時分析（繞過比分源 30-90 秒延遲）
        if path == "/manual":
            q = {k: v[0] for k, v in parse_qs(parsed.query).items()}
            try:
                from . import manual_live
                self._send(manual_live.render_manual_page(self.model, q))
            except Exception as e:  # noqa: BLE001
                self._send(f"<p>手動走地載入失敗：{_html.escape(str(e))}</p>", 500)
            return
        # 互動分析結果
        if path == "/analyze":
            q = {k: v[0] for k, v in parse_qs(parsed.query).items()}
            try:
                self._send(self._analyze(q))
            except Exception as e:  # noqa: BLE001
                self._send(f"<div class='wrap'><p>分析失敗：{_html.escape(str(e))}</p>"
                           f"<a href='/custom'>← 返回</a></div>", 500)
            return
        # 世界盃首頁 + 各場靜態分析頁（啟動時已產生）
        if self.site_dir:
            from pathlib import Path
            name = "index.html" if path in ("/", "/index.html") else path.lstrip("/")
            f = Path(self.site_dir) / name
            if name.endswith(".html") and f.is_file() and self.site_dir in str(f.resolve()):
                self._send(f.read_text(encoding="utf-8"))
                return
        # 沒有賽程網站時，首頁退回自訂表單
        if path in ("/", "/index.html"):
            self._send(render_form(self.teams))
            return
        self._send("<p>Not found</p>", 404)

    def _build_live(self, path: str) -> None:
        """現算走地頁、寫入快取並回應（呼叫端已持 _live_build_lock）。"""
        try:
            from . import mlb_live
            # MLB 與足球各自隔離：statsapi 掛掉不能拖垮足球區塊（反之亦然）
            try:
                snap = mlb_live.live_snapshot()
                mlb_err = None
            except Exception as me:  # noqa: BLE001
                snap = {"date": "-", "rows": [], "others": []}
                mlb_err = (f"<div class='card'><div class='small'>⚾ MLB 走地載入失敗"
                           f"（statsapi）：{_html.escape(str(me))}</div></div>")
            extra = [mlb_err] if mlb_err else None
            title = "MLB 走地"
            if path == "/live":
                title = "走地"
                extra = [mlb_err] if mlb_err else []
                try:
                    from . import foot_live
                    fsnap = foot_live.live_snapshot()
                    extra.append(foot_live.render_live_section(fsnap))
                except Exception as fe:  # noqa: BLE001
                    extra.append(f"<div class='card'><div class='small'>足球走地載入失敗："
                                 f"{_html.escape(str(fe))}</div></div>")
            html_page = mlb_live.render_live_page(snap, extra_sections=extra,
                                                  title=title)
            _live_cache_put(path, html_page)
            self._send(html_page)
        except Exception as e:  # noqa: BLE001
            self._send(f"<div class='wrap'><p>走地頁載入失敗：{_html.escape(str(e))}"
                       f"</p><p>需要 models/*.pkl 與可達 statsapi/ESPN 的環境。</p></div>", 500)

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

        # 陣型由 analyze 內部依 hf/af 套用；這裡只給缺陣調整
        adj = context.injuries_to_adjustment(hm, am) if (hm or am) else None
        a = analysis.analyze(self.model, home, away, history=self.history,
                             neutral=neutral, knockout=knockout, n_sims=20000,
                             adjustment=adj, home_formation=hf, away_formation=af,
                             ah_line_override=ah_line)
        if hm or am:
            a.player_note_home = f"缺主力 {hm} 人" if hm else "主力盡出"
            a.player_note_away = f"缺主力 {am} 人" if am else "主力盡出"
        # AI 賽前分析 + 多方辯論（隨選，僅設 ANTHROPIC_API_KEY 才呼叫；失敗略過）
        ai_html = report.render_ai_blocks(a, include_debate=True)
        page = report.render_analysis_html(a, f"{zh(home)} vs {zh(away)}{note}",
                                           back_href="/", interactive=True, ai_html=ai_html)
        return page


def _build_site(model, history, schedule_path, n_sims=12000, match_sims=8000):
    """啟動時產生世界盃靜態網站（首頁含奪冠/晉級/小組排名 + 各場分析頁），
    並在首頁注入「自訂分析」入口。回傳目錄路徑；失敗回 None。"""
    import tempfile
    from pathlib import Path
    from . import report, worldcup as wc
    try:
        result = wc.simulate_worldcup(model, schedule_path, n_sims=n_sims)
        _, matches, _ = wc.parse_wc_json(schedule_path)
        # 戰績：記錄推薦 + 用賽果結算（帳本放 data/bets.csv，部署重啟仍累積）
        track_text = None
        odds_index = None
        try:  # 有 ODDS_API_KEY 才抓真實盤口 → 盤口讓球線 + +EV 推薦 + ROI/CLV
            from .live.providers import fetch_wc_odds
            odds_index = fetch_wc_odds(matches)
            print(f"[serve] 已抓盤口：{len(odds_index)} 場有 +EV 候選", flush=True)
        except Exception as e:  # noqa: BLE001
            print(f"[serve] 無真實盤口（改記勝率自我校驗）：{e}", flush=True)
        try:
            from . import tracker
            tracker.backfill_played(matches, model, "data/bets.csv")
            track_text = tracker.prepare(matches, model, "data/bets.csv",
                                         odds_index=odds_index).text()
        except Exception:  # noqa: BLE001
            track_text = None
        # MLB 分頁（Render 可連 statsapi；失敗寫引導頁）
        mlb_html = None
        try:
            from . import mlb as mlbmod
            mlb_html = mlbmod.build_site_page()
            print("[serve] MLB 分頁已產生", flush=True)
        except Exception as e:  # noqa: BLE001
            print(f"[serve] MLB 分頁略過：{e}", flush=True)
        outdir = tempfile.mkdtemp(prefix="footy_wc_")
        report.write_worldcup_site(result, model, matches, outdir, history=history,
                                   n_sims=match_sims, interactive=True,
                                   track_text=track_text, ledger_path="data/bets.csv",
                                   odds_index=odds_index, mlb_html=mlb_html)
        return outdir
    except Exception as e:  # noqa: BLE001
        print(f"[serve] 產生世界盃首頁失敗（改用自訂表單為首頁）：{e}")
        return None


def serve(model_path: str, history_path: str | None = None,
          schedule_path: str | None = None, host: str = "0.0.0.0",
          port: int = 8000):
    model = DixonColesModel.load(model_path)
    _Handler.model = model
    _Handler.history = _load_history(history_path)
    if schedule_path:
        # 啟動時抓最新賽程與比分（覆寫，失敗則沿用既有檔）
        try:
            from . import worldcup as wc
            wc.fetch_schedule(schedule_path)
            print(f"[serve] 已抓最新賽程：{schedule_path}", flush=True)
        except Exception as e:  # noqa: BLE001
            print(f"[serve] 抓最新賽程失敗，沿用既有檔：{e}", flush=True)
    _Handler.teams = _server_teams(model, schedule_path)
    if schedule_path:
        print("[serve] 產生世界盃首頁（奪冠/晉級/小組排名 + 各場分析）…", flush=True)
        _Handler.site_dir = _build_site(model, _Handler.history, schedule_path)
    srv = ThreadingHTTPServer((host, port), _Handler)
    print(f"[serve] 網站：http://localhost:{port} （首頁=世界盃預測，/custom=自訂分析）", flush=True)
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
