"""五大聯賽（俱樂部）：訓練俱樂部 Dixon-Coles 模型 + 賽程 1X2 預測 + 聯賽頁。

世界盃（國家隊 intl.pkl）結束後的足球主體。俱樂部與國家隊是兩批球隊，
故各聯賽獨立訓練模型（club_{code}.pkl）。資料源沿用 data.loader.fetch_github
（xgabora 彙整檔，含 1X2 收盤賠率）。賽程用 ESPN scoreboard（best-effort）。

空窗期（球季未開打、ESPN 無排定賽程）→ 各聯賽顯示「近期無排定賽程」，
球季開打後 fetch_upcoming 抓到賽程即自動顯示預測。與 MLB/NBA 休賽季同模式。
"""
from __future__ import annotations

# football-data 代碼 → (ESPN 聯賽 code, 中文名, the-odds-api sport code)
LEAGUES = {
    "E0": ("eng.1", "英超", "soccer_epl"),
    "SP1": ("esp.1", "西甲", "soccer_spain_la_liga"),
    "I1": ("ita.1", "義甲", "soccer_italy_serie_a"),
    "D1": ("ger.1", "德甲", "soccer_germany_bundesliga"),
    "F1": ("fra.1", "法甲", "soccer_france_ligue_one"),
}


def train_club_models(data_dir: str = "data", out_dir: str = "models",
                      half_life_days: float = 180.0) -> list[str]:
    """對每個有 data/club_{code}.csv 的聯賽擬合並存 models/club_{code}.pkl。

    資料由 `footy fetch-github --league E0 --out data/club_E0.csv` 事先下載。
    回成功訓練的聯賽代碼清單（缺資料的聯賽略過、不報錯）。
    """
    from pathlib import Path

    from .data import loader
    from .models import dixon_coles as dc
    done = []
    for code in LEAGUES:
        src = Path(data_dir) / f"club_{code}.csv"
        if not src.exists():
            continue
        try:
            df = loader.load_csv(src)
            model = dc.fit(df, half_life_days=half_life_days, use_elo=False, reg=0.3)
            Path(out_dir).mkdir(parents=True, exist_ok=True)
            model.save(Path(out_dir) / f"club_{code}.pkl")
            done.append(code)
        except Exception:  # noqa: BLE001
            continue
    return done


def predict_fixtures(model, fixtures: list[dict], ou_line: float = 2.5) -> list[dict]:
    """對賽程算 1X2 + 大小盤（含主場優勢）。fixtures=[{home, away, time?, ou_line?}]。

    每場的大小線**優先用該場 fixture 的 ou_line（莊家主大小線）**，缺則退回
    參數 ou_line（預設 2.5）。回 [{...p_home/draw/away, ou_line, p_over, p_under}]；
    不在模型的球隊跳過。
    """
    from .models import markets
    out = []
    for f in fixtures:
        h, a = f.get("home"), f.get("away")
        if not h or not a or h not in model.attack or a not in model.attack:
            continue
        line = f.get("ou_line") or ou_line              # 有莊家線就用莊家線
        mat = model.score_matrix(h, a)                  # neutral=False→主場優勢
        p = markets.outcome_1x2(mat)
        ou = markets.over_under(mat, float(line))
        out.append({"home": h, "away": a, "time": f.get("time", ""),
                    "p_home": p["home"], "p_draw": p["draw"], "p_away": p["away"],
                    "ou_line": float(line),
                    "p_over": ou["over_win"], "p_under": ou["under_win"]})
    return out


def fetch_upcoming(espn_code: str, date_range: str | None = None,
                   timeout: float = 8.0) -> list[dict]:
    """ESPN scoreboard 抓排定賽程（best-effort；空窗/失敗回 []）。

    date_range 如 '20260815-20260822'（ESPN dates 參數）；None→今日。
    只取尚未開打（state=='pre'）的場次。
    """
    import requests

    from .foot_live import ESPN_SOCCER
    url = ESPN_SOCCER.format(code=espn_code)
    params = {"dates": date_range} if date_range else {}
    try:
        r = requests.get(url, params=params, headers={"User-Agent": "Mozilla/5.0"},
                         timeout=timeout)
        r.raise_for_status()
        return parse_fixtures(r.json())
    except Exception:  # noqa: BLE001
        return []


def parse_fixtures(payload: dict) -> list[dict]:
    """解析 ESPN scoreboard 成排定賽程（純函式）：只取未開打（state=='pre'）。"""
    from .foot_live import ESPN_RENAMES
    out = []
    for ev in payload.get("events") or []:
        st = ((ev.get("status") or {}).get("type") or {})
        if st.get("state") != "pre":       # 只要未開打的排定賽程
            continue
        comp = (ev.get("competitions") or [{}])[0]
        home = away = None
        for c in comp.get("competitors") or []:
            name = ((c.get("team") or {}).get("displayName") or "").strip()
            name = ESPN_RENAMES.get(name, name)
            if c.get("homeAway") == "home":
                home = name
            else:
                away = name
        if home and away:
            out.append({"home": home, "away": away, "time": ev.get("date", "")})
    return out


def render_leagues_page(preds_by_league: dict, title: str = "足球五大聯賽") -> str:
    """五大聯賽預測頁。preds_by_league={中文名: [predict_fixtures 產物]}。

    某聯賽無賽程 → 顯示「近期無排定賽程」（空窗期/球季未開打）。
    """
    import html as _h

    from .i18n import zh
    from .report import _CSS, _navbar, odds_quota_banner
    quota = odds_quota_banner()
    sections = []
    for code, (_, zh_name, _sport) in LEAGUES.items():
        preds = preds_by_league.get(zh_name) or []
        if not preds:
            body = ("<div class='small' style='color:var(--muted)'>"
                    "近期無排定賽程（球季未開打或空窗期）。開賽後自動顯示。</div>")
        else:
            def _ou(p):
                ln = p.get("ou_line", 2.5)
                po, pu = p.get("p_over"), p.get("p_under")
                if po is None or pu is None:
                    return "<span class='dim'>—</span>"
                return (f"大{ln:g} {po:.0%}·{1 / max(po, 0.005):.2f} ／ "
                        f"小 {pu:.0%}·{1 / max(pu, 0.005):.2f}")
            trs = "".join(
                f"<tr><td class='tm'>{_h.escape(zh(p['home']))}</td>"
                f"<td>{p['p_home']:.0%}·{1 / max(p['p_home'], 0.005):.2f}</td>"
                f"<td>{p['p_draw']:.0%}·{1 / max(p['p_draw'], 0.005):.2f}</td>"
                f"<td>{p['p_away']:.0%}·{1 / max(p['p_away'], 0.005):.2f}</td>"
                f"<td class='tm'>{_h.escape(zh(p['away']))}</td>"
                f"<td class='small'>{_ou(p)}</td></tr>"
                for p in preds)
            body = ("<table><thead><tr><th>主隊</th><th>主勝</th><th>和</th>"
                    "<th>客勝</th><th>客隊</th><th>大小盤</th></tr></thead>"
                    f"<tbody>{trs}</tbody></table>")
        sections.append(f"<div class='card'><div class='sec'>⚽ {zh_name}</div>{body}</div>")
    return f"""<!doctype html><html lang="zh-Hant"><head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>{_h.escape(title)}</title><style>{_CSS}</style></head><body><div class="wrap">
  {_navbar('leagues')}
  <h1>⚽ 足球五大聯賽</h1>
  {quota}
  <div class="small" style="color:var(--muted);margin:8px 0">
    每格：機率·公平賠率。1X2 與大小盤（總進球 2.5）純模型（含主場優勢），非投注建議。</div>
  {''.join(sections)}
  <div class="foot">Generated by footy · 研究與教育用途</div>
</div></body></html>"""


def _attach_market_ou(fixtures: list[dict], sport: str) -> None:
    """就地把莊家主大小線寫進每場 fixture 的 ou_line（抓不到盤口則不動，退回預設 2.5）。"""
    if not fixtures:
        return
    try:
        from . import tracker
        from .live.providers import fetch_wc_odds

        class _G:
            def __init__(self, num, t1, t2):
                self.num, self.team1, self.team2, self.played = num, t1, t2, False
        games = [_G(i + 1, f["home"], f["away"]) for i, f in enumerate(fixtures)]
        # 聯賽頁只用 1X2＋大小 → 只抓 h2h,totals（省 the-odds-api 額度：不抓讓分 spreads）
        idx = fetch_wc_odds(games, sport=sport, markets="h2h,totals")
        for i, f in enumerate(fixtures):
            q = idx.get(i + 1)
            if q:
                ln = tracker.main_ou_line(q)
                if ln:
                    f["ou_line"] = ln
    except Exception:  # noqa: BLE001（盤口抓取失敗 → 全退回預設線，頁面照常）
        pass


def build_site_page(models_dir: str = "models", date_range: str | None = None,
                    fixtures_by_code: dict | None = None, with_odds: bool = True) -> str:
    """建五大聯賽頁（wc-site 呼叫）。載各聯賽 club_{code}.pkl + 抓賽程 → 預測頁。

    大小盤機率**依莊家主大小線**計算（抓不到盤口則退回標準 2.5 線）。
    fixtures_by_code 供測試注入（{football代碼: [{home,away,time,ou_line?}]}）。
    """
    from pathlib import Path

    from .models.dixon_coles import DixonColesModel
    preds_by_league: dict = {}
    for code, (espn_code, zh_name, sport) in LEAGUES.items():
        mp = Path(models_dir) / f"club_{code}.pkl"
        if not mp.exists():
            continue
        try:
            model = DixonColesModel.load(mp)
        except Exception:  # noqa: BLE001
            continue
        fixtures = (fixtures_by_code or {}).get(code)
        if fixtures is None:
            fixtures = fetch_upcoming(espn_code, date_range)
        # 季外（6-7 月）只有熱身賽 → 不抓盤口（省額度），大小盤退回 2.5 預設線；
        # 8 月開季後自動改用莊家主大小線。可用 FOOTY_LEAGUES_ODDS=1 強制開。
        import datetime as _d
        in_season = _d.date.today().month not in (6, 7)
        force = __import__("os").environ.get("FOOTY_LEAGUES_ODDS") == "1"
        if with_odds and fixtures_by_code is None and (in_season or force):
            _attach_market_ou(fixtures, sport)      # 用莊家主大小線
        preds_by_league[zh_name] = predict_fixtures(model, fixtures)
    return render_leagues_page(preds_by_league)
