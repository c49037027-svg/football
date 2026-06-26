"""AI agents（解讀/包裝/把關層，供應商無關）。

重要定位：這些 LLM agent 不改變統計模型的預測，只負責「把模型數字寫成白話、
多角度檢視、風控與賽後檢討」。模型才是真相來源；agent 一律只依傳入的數字發言。

供應商無關：透過 OpenAI 相容 API（Gemini/OpenAI/OpenRouter/DeepSeek/本地皆可），
用環境變數設定：
  LLM_API_KEY 或 GEMINI_API_KEY  ── 金鑰
  LLM_BASE_URL  ── 預設 Gemini 的 OpenAI 相容端點
  LLM_MODEL     ── 預設 gemini-2.5-flash（如 404 可改 gemini-2.5-flash-lite 等）
沒設金鑰時，所有 agent 安全地回 None / 略過，不影響主流程。
"""
from . import llm, roles  # noqa: F401
