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
│   ├── inplay.py      # 走地模型（依比分+剩餘時間+紅牌人數差重算）
│   ├── feed.py        # 盤口 feed 抽象介面 + 模擬 feed
│   ├── providers.py   # 真實賠率來源（The Odds API：初盤+走地+比分）
│   ├── rules.py       # 走地十六法則 規則引擎（過濾 + 注碼調節）
│   └── monitor.py     # 實時盯盤迴圈 + 提示
├── risk/
│   └── manager.py     # 資金管理 / 曝險上限 / 停損 / 熔斷
├── context.py         # 傷停/輪休情境調整（手動 CSV + api-football）
├── evaluation.py      # 模型校準（Brier/LogLoss/可靠度）+ CLV 分析
├── prematch.py        # 初盤價值掃描
├── backtest/
│   └── engine.py      # walk-forward 回測引擎
└── cli.py             # 命令列入口
```

## 進階功能

### 1) 接真實實時賠率（The Odds API）
`live/providers.py` 內建 `TheOddsApiFeed`，接 [the-odds-api.com](https://the-odds-api.com)
（免費方案即可），自動把 h2h / totals / spreads 對應成 1X2 / 大小球 / 亞洲讓盤，
並用 scores 端點取即時比分。
```bash
export ODDS_API_KEY=你的key
footy live --model models/E0.pkl --feed theoddsapi --sport soccer_epl --bookmaker pinnacle
```
解析邏輯（`parse_odds` / `parse_scores`）與網路請求分離、可單元測試。
要接別家（Betfair、api-football…）只要照樣寫一個 `OddsFeed` 子類別即可，其餘邏輯全部複用。

### 2) xG 建模
若歷史資料含 `home_xg` / `away_xg` 欄位（可從 understat / fbref / api-football 併入），
可用 xG 取代或混合實際進球來估強度（xG 雜訊更低、更能反映實力）：
```bash
footy train --data data/E0.csv --xg-weight 0.6   # 0=純進球, 1=純xG
```

### 2b) Elo 特徵
若資料含賽前 `home_elo` / `away_elo`，可把 Elo 當額外特徵：預期進球加入
`elo_coef * (Elo_home − Elo_away)/400` 項，`elo_coef` 由 MLE 學習。Elo 含長期實力與
比分差幅資訊，可補足「只看近期進球」的不足：
```bash
footy train   --data data/E0.csv --use-elo
footy evaluate --data data/E0.csv --use-elo   # 比較有無 Elo 對校準的影響
```

### 走地十六法則（滾球心法規則引擎）
> ⚠️ 「滾球/走地十六法則」是華人足球圈流傳的心法，**並無單一權威版本**。
> `live/rules.py` 是我綜合常見共識（走地以大小球為主、讓球只在少打多時才玩、
> 小球有底線、大球有頂線、紅牌是關鍵信號、賠率跳動快要保守、終場前減碼、
> 限制過度交易…）整理、編碼成 **16 條透明且可調**的規則。

它們**不是真理**，而是疊在 +EV 引擎之上的**過濾器與注碼調節器**：先由模型找出正期望值
的盤，再用這些法則否決不該碰的、並對高風險情境縮減注碼。所有門檻都在 `config.yaml`
的 `live:` 區可調，請用你自己的資料校準。走地盯盤時自動生效，提示中會顯示每筆觸發的法則。

### 3) 走地紅牌 / 少打多即時調整
走地模型會依紅牌造成的人數差，即時調整雙方剩餘時間的進球率
（`live/inplay.py` 的 `man_advantage_factors`，係數可在 config 調整與校準）。
`MatchState.home_red/away_red` 由 feed 提供即可自動生效。

### 4) 傷停 / 輪休情境調整
`context.py` 支援把賽前情境（主力傷缺、輪休）轉成對攻防的乘數修正：
- 手動 CSV：`footy scan-prematch ... --adjustments adjustments.csv`
  （欄位：`home,away,home_attack_mult,away_attack_mult`）
- 自動：api-football 傷停 API（`fetch_injuries`，需 `API_FOOTBALL_KEY`）。

### 5) 模型校準與 CLV 分析（最重要的驗證）
比回測 ROI 更基本的「值不值得用真錢」檢驗：
```bash
footy evaluate --data data/E0.csv --refit-every 20
```
輸出：
- **Brier / LogLoss**，並與市場收盤盤比較 —— 模型若沒贏過市場，代表沒有資訊優勢。
- **可靠度表**（reliability）—— 模型說「30%」的事是否真的約 30% 發生。
- **CLV（closing line value）**—— 你的下注價 vs 收盤價；長期正 CLV 是最可靠的 +EV 指標。

## 免責

本專案僅供研究與教育用途。作者不對任何投注盈虧負責。
