# Footy — 足球 +EV 預測與實時盯盤系統

一套以**統計建模 + 價值投注（value betting）+ 凱利資金管理 + 嚴格風控**為核心的足球投注決策輔助系統。

> ⚠️ **重要聲明**：沒有任何模型能保證獲利。盤口已內含莊家利潤（vig/overround），市場也相當有效。
> 本系統的目標是**在長期上把正期望值（+EV）的機會最大化**並控制破產風險，而不是「必勝」。
> 是否真能正收益，取決於數據品質、模型校準與市場效率。請務必先用歷史數據回測、再用紙上交易（paper trading）驗證，最後才考慮投入真實資金。本工具**不自動下注**，只提供提示。
> 投注有風險，請遵守當地法律並量力而為。

## 核心思路

1. **建模真實機率**：用 Dixon–Coles（帶時間衰減的雙變量 Poisson）模型，由歷史比分估計每隊的攻防強度與主場優勢，推導出每場比賽的完整比分機率矩陣。
2. **推導各市場機率**：由比分矩陣計算 1X2（勝平負）、大小球（Over/Under）、亞洲讓盤（Asian Handicap）的真實機率。
3. **找價值（value）**：去除盤口的莊家抽水後得到「市場隱含機率」，與模型機率比較，只在**模型機率 > 盤口隱含機率**（即正 EV）且超過安全邊際時才提示。
4. **凱利下注**：用分數凱利（fractional Kelly）決定下注比例，兼顧成長與風險。
5. **實時盯盤（走地）**：用 in-play 模型，依「當前比分 + 剩餘時間」即時重算各市場機率，掃描走地盤口找價值並發出提示。
6. **風控**：資金管理、單注上限、單場/單日曝險上限、停損、回撤熔斷、連敗冷靜期。

## 安裝

```bash
pip install -r requirements.txt
pip install -e .
```

## 快速開始

```bash
# 1) 下載歷史數據（football-data.co.uk，免費公開）
footy fetch-data --league E0 --seasons 2021 2022 2023 2024

# 2) 訓練 Dixon–Coles 模型
footy train --data data/E0.csv --half-life 180

# 3) 對即將開賽的盤口做初盤（pre-match）價值掃描
footy scan-prematch --model models/E0.pkl --fixtures examples/fixtures.csv

# 4) 用歷史數據回測策略（驗證是否 +EV）
footy backtest --data data/E0.csv --half-life 180 --edge 0.03 --kelly 0.25

# 5) 走地實時盯盤（預設用內建模擬盤口，無需 API key；可接真實 feed）
footy live --model models/E0.pkl --feed simulated
```

## 專案結構

```
src/footy/
├── config.py          # 設定（YAML + 預設值）
├── data/
│   ├── loader.py      # football-data.co.uk 下載與載入
│   └── schema.py      # 欄位定義
├── models/
│   ├── dixon_coles.py # Dixon–Coles 帶時間衰減的雙變量 Poisson
│   └── markets.py     # 由比分矩陣推導 1X2 / O-U / AH 機率
├── value/
│   ├── odds.py        # 賠率↔機率、去 vig
│   ├── edge.py        # 價值偵測（+EV）
│   └── staking.py     # 凱利下注
├── live/
│   ├── inplay.py      # 走地模型（依比分+剩餘時間重算）
│   ├── feed.py        # 盤口 feed 抽象介面 + 模擬 feed
│   └── monitor.py     # 實時盯盤迴圈 + 提示
├── risk/
│   └── manager.py     # 資金管理 / 曝險上限 / 停損 / 熔斷
├── backtest/
│   └── engine.py      # 回測引擎
└── cli.py             # 命令列入口
```

## 接真實實時數據

`live/feed.py` 定義了 `OddsFeed` 抽象介面。內建 `SimulatedFeed` 讓你無需任何 API key 就能跑通整條流程。
要接真實數據，實作一個子類別（例如 `TheOddsApiFeed`、`BetfairFeed`），回傳相同格式的 `MatchState` 與盤口即可，其餘邏輯（模型、價值、風控、提示）完全複用。

## 免責

本專案僅供研究與教育用途。作者不對任何投注盈虧負責。
