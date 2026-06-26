"""AI agents（解讀/包裝/把關層，供應商無關）。

重要定位：這些 LLM agent 不改變統計模型的預測，只負責「把模型數字寫成白話、
多角度檢視、風控與賽後檢討」。模型才是真相來源；agent 一律只依傳入的數字發言。

用官方 Anthropic Claude SDK。環境變數設定：
  ANTHROPIC_API_KEY  ── 金鑰
  LLM_MODEL          ── 模型，預設 claude-opus-4-8（要省錢可設 claude-haiku-4-5）
沒設金鑰時，所有 agent 安全地回 None / 略過，不影響主流程。
"""
from . import llm, roles  # noqa: F401
