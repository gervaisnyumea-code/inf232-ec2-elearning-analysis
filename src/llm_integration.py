"""Minimal LLM integration placeholders.

LLMClient is a thin wrapper that selects provider based on available env vars.
It provides a safe interface to integrate real LLM API calls later.
"""
import os
from typing import Optional


class LLMClient:
    def __init__(self, provider: Optional[str] = None):
        self.provider = provider or self._select_provider()

    def _select_provider(self) -> Optional[str]:
        # choose provider in order of preference if API key present
        if os.getenv('GEMINI_API_KEY_1') or os.getenv('GEMINI_API_KEY_2'):
            return 'gemini'
        if os.getenv('MISTRAL_API_KEY_1') or os.getenv('MISTRAL_API_KEY_2'):
            return 'mistral'
        if os.getenv('GROQ_API_KEY_1') or os.getenv('GROQ_API_KEY_2'):
            return 'groq'
        return None

    def summarize(self, text: str, max_tokens: int = 256) -> str:
        """Placeholder summarization. Replace with real API call when keys are provided."""
        provider = self.provider or 'none'
        s = text.strip().replace('\n', ' ')
        # Return deterministic short summary for now
        return f"[LLM summary by {provider}] {s[:200]}..."

    def generate_report_text(self, df_summary: dict) -> str:
        provider = self.provider or 'none'
        lines = [f"{k}: {v}" for k, v in df_summary.items()]
        return f"[Report by {provider}] " + "; ".join(lines)
