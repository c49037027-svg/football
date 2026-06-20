# 互動分析網頁容器（可部署到 Fly.io / Railway / Render / 任何支援 Docker 的平台）
FROM python:3.11-slim

WORKDIR /app
COPY . .
RUN pip install --no-cache-dir -e .

ENV PORT=8000
# 即時輸出日誌（不緩衝），啟動訊息（[serve] 已抓盤口…）才即時可見
ENV PYTHONUNBUFFERED=1
EXPOSE 8000

# 已附帶預訓練模型(models/intl.pkl)與歷史(data/intl.csv)，啟動即用；
# 若被刪除，--auto-prepare 會自動下載並訓練。
CMD ["sh", "-c", "footy serve --schedule examples/wc2026.json --auto-prepare --port ${PORT:-8000} --host 0.0.0.0"]
