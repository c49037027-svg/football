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
├── predict.py         # 預測站風格內容（1X2/比分/大小球/BTTS/狀態/H2H/Tip）
├── worldcup.py        # 整屆世界盃模擬（小組+淘汰賽，晉級/奪冠機率）
├── analysis.py        # 世界盃單場深度分析（Poisson+蒙地卡羅，全面板）
├── counts.py          # 角球/黃牌 Poisson 計數模型（先驗近似）
├── intl/              # 國際賽資料、Elo 評分、進球分鐘分布
├── season.py          # 整季蒙地卡羅模擬（奪冠/前四/降級機率）
├── report.py          # 渲染預測/分析/季模擬為 console / Markdown / HTML
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

### 模型精度強化：防洩漏 / 正則化 / 自動調參
- **防洩漏**：`dc.fit` 給 `reference_date` 時自動剔除晚於該日的比賽，杜絕用到未來資料
  （回測/評估本就逐場往前，這也修掉「未來比賽因 age<0 被當成全權重」的舊 bug）。
- **L2 正則化** `--reg`：對攻防參數加 `reg·Σ(attack²+defence²)` 懲罰，把弱資料的隊往
  聯盟平均收縮，降低過擬合、穩定擬合（多隊國際賽尤其有用）。
- **自動調參** `footy tune`：用 walk-forward 樣本外 log-loss 網格搜尋最佳
  `half_life/reg/use_elo/xg_weight`，把「手設」變成「證明過最低 log-loss」：
```bash
footy tune     --data data/E0.csv --save-config config.best.yaml
footy --config config.best.yaml evaluate --data data/E0.csv   # 用最佳設定驗證
footy --config config.best.yaml train    --data data/E0.csv   # 用最佳設定訓練
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

## 預測站內容（像 soccermaddy / Forebet）

把模型輸出成「預測站」風格的每場預測——1X2 機率、預測比分、正確比分 Top5、
大小球（1.5/2.5/3.5）、BTTS、雙方預期進球、近期狀態（最近 5 場 W/D/L）、H2H 與推薦 Tip。
可輸出 console、Markdown 或一頁式 HTML（手機友善、深色卡片風格）：
```bash
footy predict --model models/E0.pkl --fixtures examples/epl_fixtures.csv \
              --history data/E0.csv --html out/predictions.html --md out/predictions.md \
              --title "今日英超預測"
```
> 這是**純機率預測內容**（給讀者看的），與 `scan-prematch`／`live`（找 +EV 下注）不同；
> 預測站好看，但別忘了 `docs/FINDINGS.md`：模型機率未必勝過市場盤口。

## 整屆世界盃預測網站（多場一次看）

讀 openfootball 官方 2026 世界盃賽程（含分組、已踢比分），蒙地卡羅模擬整屆，
產出一頁式網站首頁：
```bash
footy worldcup --model models/intl.pkl --schedule examples/wc2026.json \
               --n-sims 20000 --html out/worldcup.html
```
首頁包含：
- **奪冠機率排行**（橫條圖）
- **晉級展望表**：每隊 晉級 / 16強 / 8強 / 4強 / 決賽 / 奪冠 機率
- **12 個小組**：各組出線機率表（首名 / 前二 / 晉級含最佳第三）＋ 預期積分
- **每場小組賽預測**：未踢顯示預測比分與 1X2，已踢顯示真實比分

賽制：12 組各 4 隊，前 2 + 8 個最佳第三名 → 32 強。已踢比分自動納入（隨賽程更新會更準）。
> ⚠️ 「最佳第三名 → R32 槽位」用符合官方允許組別的二分匹配近似；晉級/奪冠機率為主要可信輸出。

### 多頁網站（每場可點進單場分析）＋ 每日自動更新

```bash
# 產生多頁網站：首頁 index.html + 每場 match_<num>.html（手機友善、全中文）
footy wc-site --model models/intl.pkl --schedule examples/wc2026.json \
              --history data/intl.csv --outdir out/wc

# 每日自動更新（抓最新比分→重建網站）；加 --full 連模型一起更新
bash scripts/update_worldcup.sh
# 排程：crontab 加 → 0 8 * * * cd /path/to/football && bash scripts/update_worldcup.sh
```
首頁的每場小組賽與（隊伍確定後的）淘汰賽都可點進**單場深度分析頁**，頁內有「返回首頁」。
手機開啟：把 `out/wc/` 整個資料夾傳到手機，用瀏覽器開 `index.html` 即可（連結為相對路徑）。

### 部署到 GitHub Pages（手機用固定網址看、每日自動更新）

已內建 GitHub Actions：`.github/workflows/deploy-worldcup.yml`
- 每天 UTC 06:00（台灣 14:00）自動：下載最新比分 → 重算 Elo/模型 → 重建網站 → 部署
- 也可在 GitHub 的 Actions 分頁手動觸發（workflow_dispatch）

**一次性啟用**（GitHub 網站操作）：
1. 進 repo **Settings → Pages**
2. **Source** 選 **GitHub Actions**
3. 到 **Actions** 分頁手動跑一次「部署世界盃預測網站」
4. 完成後網址為 `https://<帳號>.github.io/football/`，手機開這個網址即可，每天自動更新。

## 互動分析網頁（用下拉選單／自己輸入盤口）

靜態網站無法即時重算（模型在 Python 跑）。要「**自己選陣型/陣容、輸入盤口讓球線**」就用
本機互動網頁（零額外套件）：
```bash
footy serve --model models/intl.pkl --history data/intl.csv --schedule examples/wc2026.json
# 瀏覽器開 http://localhost:8000
```
表單可：選主/客隊（中文下拉）、選**陣型**、填**缺陣主力人數**、**自行輸入盤口讓球線**
（如 -1.5），或勾選「自動抓 The Odds API 盤口」（需 `ODDS_API_KEY`），按「分析」即時重算。

### 部署到網路（手機隨時開、不用開電腦）

互動網頁要有後端（Python 即時跑模型），不能放 GitHub Pages。已附好一鍵部署設定：
- **Render（最簡單）**：到 [render.com](https://render.com) → New → **Blueprint** → 連這個 repo →
  Apply。`render.yaml` 會自動建好免費 web 服務，幾分鐘後給你固定網址。
- **Docker（Railway / Fly.io / 自架）**：repo 內含 `Dockerfile`，`docker build` 後即可跑；
  或 Railway 連 repo 自動偵測 `Procfile`。

已附帶預訓練模型（`models/intl.pkl`）與歷史資料（`data/intl.csv`），**啟動即用**；
若被刪除，`--auto-prepare` 會自動下載並訓練。要自動抓盤口/傷停，在平台後台設
`ODDS_API_KEY` / `API_FOOTBALL_KEY`（選用）。

## 世界盃單場深度分析（像 soccermaddy 那種 UI）

針對國際賽（世界盃）的單場全面分析，整合 **Poisson（解析）+ 蒙地卡羅**，
模型輸入包含 **Elo 評分（≈FIFA 排名）、近5場表現、歷史交手、戰術對比**
（球員狀態因無公開資料，預設中性、可手動覆寫）：
```bash
footy fetch-intl --out data/intl.csv --since 2010-01-01      # 下載國際賽+算 Elo
footy train --data data/intl.csv --half-life 540 --use-elo --out models/intl.pkl
footy analyze --model models/intl.pkl --home Iraq --away Norway \
              --history data/intl.csv --neutral --knockout \
              --html out/analysis.html
```
產出面板（HTML 卡片 UI）：
- **AI 預測比分 + 總進球 + xG 區間**
- **大小球 1.5 / 2.5 / 3.5**（買大/買小）
- **兩隊都進球 BTTS**（含各自進球機率）
- **亞盤讓球**（模型估線、supremacy/xG 差、建議）
- **角球 / 黃牌**（合計、估線、建議、信心）— ⚠️ 國際賽無公開角球/黃牌統計，屬 Poisson 先驗近似
- **上半場走向**（1X2 + 上半場大小 0.5 / 1.5）— 用蒙地卡羅把進球分配到上下半場
- **影響因子**：Elo、近5場、H2H、戰術風格

> Elo 由國際賽結果自算（前十：阿根廷/西班牙/法國/英格蘭/巴西…，與 FIFA 排名高度吻合），
> 可用 `intl.data.load_fifa_ranking` 匯入官方 FIFA 排名替代/補充。中立場（世界盃）自動取消主場優勢。

## 整季蒙地卡羅模擬（像 FiveThirtyEight）

對整季所有比賽反覆抽樣（從 Dixon–Coles 比分分布，含 DC 低分修正），統計每隊的
**奪冠 / 前四 / 前六 / 降級機率與預期積分、預期名次**。可從零（季前預測）或帶入
目前積分榜＋剩餘賽程（季中更新）：
```bash
# 季前：用一份隊伍清單跑全季雙循環
footy simulate-season --model models/E0.pkl --teams examples/epl_2425_teams.csv \
                      --n-sims 20000 --relegation 3 --html out/season.html

# 季中：由本季已踢賽果自動推算積分榜與剩餘賽程
footy simulate-season --model models/E0.pkl --played data/this_season.csv --n-sims 20000
```
輸出 console 表或熱力圖 HTML。實測：用真實英超模型，三強（Arsenal/Man City/Liverpool）
爭冠、升班三隊降級機率最高，與 2024/25 真實結果相符。

> 這是**解析模型 + 蒙地卡羅抽樣**：單場市場用解析矩陣（精確），整季名次因涉及
> 380 場的聯合分布與排序，才用 MC 抽樣最自然。

## 真實數據實證結論

我們已用真實英超（E0）與英冠（E1）近 9 季資料做了 walk-forward 校準。
**重點：基礎 Dixon–Coles（即使加 Elo）在 1X2 主流市場上贏不過收盤盤**——
詳見 [`docs/FINDINGS.md`](docs/FINDINGS.md)。這代表沒有資訊優勢、難以長期 +EV。
這不是程式失敗，而是系統**誠實地告訴你真相**：在投真錢前，先用 `evaluate` 確認有沒有 edge。

可能有 edge 的方向（未在本 repo 驗證）：打贏開盤線/正 CLV、次級市場（角球/罰牌）、
走地反應速度、更獨家且更快的資料。

## 免責

本專案僅供研究與教育用途。作者不對任何投注盈虧負責。
