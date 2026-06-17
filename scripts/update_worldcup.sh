#!/usr/bin/env bash
# 世界盃網站每日更新腳本
#
# 用法：
#   bash scripts/update_worldcup.sh            # 抓最新比分 → 重建網站（快）
#   bash scripts/update_worldcup.sh --full     # 連同國際賽資料+Elo+模型一起更新（慢）
#
# 排程（每天早上 8 點自動更新）可加到 crontab：
#   0 8 * * * cd /path/to/football && bash scripts/update_worldcup.sh >> logs/wc.log 2>&1
set -euo pipefail
cd "$(dirname "$0")/.."

MODEL="models/intl.pkl"
SCHEDULE="data/wc2026.json"
HISTORY="data/intl.csv"
OUTDIR="out/wc"

if [[ "${1:-}" == "--full" ]]; then
  echo "[update] 重新下載國際賽資料並重算 Elo / 訓練模型…"
  footy fetch-intl --out "$HISTORY" --since 2010-01-01
  footy train --data "$HISTORY" --half-life 540 --use-elo --reg 0.75 --out "$MODEL"
fi

if [[ ! -f "$MODEL" ]]; then
  echo "[update] 找不到模型，先做完整訓練…"
  footy fetch-intl --out "$HISTORY" --since 2010-01-01
  footy train --data "$HISTORY" --half-life 540 --use-elo --reg 0.75 --out "$MODEL"
fi

echo "[update] 下載最新賽程與比分…"
footy fetch-wc --out "$SCHEDULE"

echo "[update] 重建世界盃網站…"
footy wc-site --model "$MODEL" --schedule "$SCHEDULE" --history "$HISTORY" \
              --outdir "$OUTDIR" --n-sims 20000 --match-sims 20000 \
              --use-injuries   # 需 export API_FOOTBALL_KEY；抓不到會自動略過

echo "[update] 完成：$OUTDIR/index.html （更新於 $(date '+%Y-%m-%d %H:%M')）"
