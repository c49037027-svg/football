"""football-data.co.uk CSV 的欄位定義與標準化。

該站提供各大聯賽免費歷史 CSV，欄位含賽果與多家博彩公司賠率。
我們只取建模與回測必要的欄位，並標準化成內部格式。
參考：https://www.football-data.co.uk/notes.txt
"""
from __future__ import annotations

# 內部標準欄位
DATE = "date"
HOME = "home"
AWAY = "away"
HOME_GOALS = "home_goals"
AWAY_GOALS = "away_goals"

# 選用：每場 xG（expected goals）。多數 football-data.co.uk 主檔沒有，
# 但若你從別的來源（understat / fbref / api-football）併入，放這兩欄即可啟用 xG 建模。
HOME_XG = "home_xg"
AWAY_XG = "away_xg"

# 選用：賽前 Elo 評分（球隊長期實力，含比分差幅資訊，常比近期進球更穩定）。
HOME_ELO = "home_elo"
AWAY_ELO = "away_elo"

# 收盤 1X2 賠率（用 Bet365 收盤盤，缺則用 market average）
ODDS_HOME = "odds_home"
ODDS_DRAW = "odds_draw"
ODDS_AWAY = "odds_away"

# football-data.co.uk 原始欄位 -> 內部欄位
RAW_RESULT_MAP = {
    "Date": DATE,
    "HomeTeam": HOME,
    "AwayTeam": AWAY,
    "FTHG": HOME_GOALS,  # Full Time Home Goals
    "FTAG": AWAY_GOALS,  # Full Time Away Goals
}

# 收盤 1X2 賠率候選欄位（依優先序，取第一個存在的）。C 前綴 = closing（收盤）。
ODDS_HOME_CANDIDATES = ["B365CH", "PSCH", "AvgCH", "B365H", "AvgH", "BbAvH"]
ODDS_DRAW_CANDIDATES = ["B365CD", "PSCD", "AvgCD", "B365D", "AvgD", "BbAvD"]
ODDS_AWAY_CANDIDATES = ["B365CA", "PSCA", "AvgCA", "B365A", "AvgA", "BbAvA"]

# 開盤 1X2 賠率（無 C 前綴 = 早盤/開盤），用於計算 CLV（closing line value）。
ODDS_HOME_OPEN = "odds_home_open"
ODDS_DRAW_OPEN = "odds_draw_open"
ODDS_AWAY_OPEN = "odds_away_open"
ODDS_HOME_OPEN_CANDIDATES = ["B365H", "PSH", "AvgH"]
ODDS_DRAW_OPEN_CANDIDATES = ["B365D", "PSD", "AvgD"]
ODDS_AWAY_OPEN_CANDIDATES = ["B365A", "PSA", "AvgA"]

REQUIRED_INTERNAL = [DATE, HOME, AWAY, HOME_GOALS, AWAY_GOALS]
