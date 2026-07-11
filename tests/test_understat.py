"""understat 真實 xG 解析測試（純函式；抓取在部署環境）。"""
from footy.data import understat as u
from footy.data import schema as S

# 模擬 understat 聯賽頁的 datesData（含一場已完賽、一場未賽 + 一個重音隊名轉義）
FIXTURE = (
    "<script>\n"
    "var datesData = JSON.parse('"
    "[{\\x22isResult\\x22:true,\\x22h\\x22:{\\x22title\\x22:\\x22Arsenal\\x22},"
    "\\x22a\\x22:{\\x22title\\x22:\\x22Atl\\xc3\\xa9tico Madrid\\x22},"
    "\\x22goals\\x22:{\\x22h\\x22:\\x222\\x22,\\x22a\\x22:\\x221\\x22},"
    "\\x22xG\\x22:{\\x22h\\x22:\\x221.85\\x22,\\x22a\\x22:\\x220.72\\x22},"
    "\\x22datetime\\x22:\\x222023-08-11 20:00:00\\x22},"
    "{\\x22isResult\\x22:false,\\x22h\\x22:{\\x22title\\x22:\\x22Chelsea\\x22},"
    "\\x22a\\x22:{\\x22title\\x22:\\x22Liverpool\\x22},"
    "\\x22goals\\x22:{\\x22h\\x22:null,\\x22a\\x22:null},"
    "\\x22xG\\x22:{\\x22h\\x22:null,\\x22a\\x22:null},"
    "\\x22datetime\\x22:\\x222099-01-01 12:00:00\\x22}]"
    "');\n</script>"
)


def test_parse_understat_page():
    rows = u.parse_understat_page(FIXTURE)
    assert len(rows) == 1                    # 只回已完賽
    r = rows[0]
    assert r[S.HOME] == "Arsenal"
    assert r[S.AWAY] == "Atlético Madrid"    # 重音正確還原
    assert r[S.HOME_GOALS] == 2 and r[S.AWAY_GOALS] == 1
    assert abs(r[S.HOME_XG] - 1.85) < 1e-9 and abs(r[S.AWAY_XG] - 0.72) < 1e-9
    assert r[S.DATE] == "2023-08-11"


def test_parse_empty():
    assert u.parse_understat_page("<html>no data</html>") == []
    assert "E0" in u.LEAGUE_SLUG and u.LEAGUE_SLUG["SP1"] == "La_liga"
