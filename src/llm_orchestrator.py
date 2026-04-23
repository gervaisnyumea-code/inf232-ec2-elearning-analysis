"""LLM orchestrator: route tasks to available LLM providers and run high-level pipelines.

Design / Strategy
- Providers detected via environment keys (GEMINI_*, MISTRAL_*, GROQ_*).
- Role assignment (where they'll intervene):
    * GEMINI: fast summarization & anomaly extraction from periodic reports
    * MISTRAL: narrative report generation, recommendations and user-facing text
    * GROQ: reasoning, reconciliation across model outputs and meta-analysis
- The orchestrator exposes a small API for: summarize_report, generate_narrative, reconcile_ensemble, run_full_pipeline

Notes
- This module uses src.llm_integration.LLMClient which currently contains safe placeholders.
- When API keys are provided (.env), LLMClient will pick provider automatically; replace LLMClient methods with real API calls as needed.
"""

import os
from typing import Optional, Dict
from src.llm_integration import LLMClient
import logging

logger = logging.getLogger(__name__)

# mapping of provider -> role for documentation and routing
PROVIDER_ROLES = {
    'gemini': 'summarization',
    'mistral': 'narrative',
    'groq': 'reasoning',
}


class LLMOrchestrator:
    """High-level orchestrator that routes tasks to LLM providers based on role.

    Usage:
        orch = LLMOrchestrator()
        res = orch.run_full_pipeline(df_live)

    The implementation is intentionally lightweight and provider-agnostic; replace
    LLMClient.summarize / generate_report_text with real API calls when keys are configured.
    """

    def __init__(self):
        self.clients: Dict[str, LLMClient] = {}
        self._discover_providers()

    def _discover_providers(self):
        # instantiate clients for available providers (order of preference)
        # GEMINI
        if os.getenv('GEMINI_API_KEY_1') or os.getenv('GEMINI_API_KEY_2'):
            self.clients['gemini'] = LLMClient('gemini')
            logger.info('Gemini client initialized')
        # MISTRAL
        if os.getenv('MISTRAL_API_KEY_1') or os.getenv('MISTRAL_API_KEY_2'):
            self.clients['mistral'] = LLMClient('mistral')
            logger.info('Mistral client initialized')
        # GROQ
        if os.getenv('GROQ_API_KEY_1') or os.getenv('GROQ_API_KEY_2'):
            self.clients['groq'] = LLMClient('groq')
            logger.info('GROQ client initialized')

    def available_providers(self):
        return list(self.clients.keys())

    def summarize_report(self, text: str, max_tokens: int = 256) -> Dict[str, str]:
        """Produce summaries using GEMINI (preferred) and fallback providers.

        Returns a dict mapping provider->summary.
        """
        out = {}
        # try gemini first
        if 'gemini' in self.clients:
            out['gemini'] = self.clients['gemini'].summarize(text, max_tokens=max_tokens)
        # fallback to mistral
        if 'mistral' in self.clients:
            out['mistral'] = self.clients['mistral'].summarize(text, max_tokens=max_tokens)
        # minimal groq usage for short summaries
        if 'groq' in self.clients:
            out['groq'] = self.clients['groq'].summarize(text, max_tokens=max_tokens)
        return out

    def generate_narrative(self, summary_text: str) -> Dict[str, str]:
        """Generate a narrative / recommendations using Mistral (preferred).

        Returns provider->text mapping.
        """
        out = {}
        if 'mistral' in self.clients:
            out['mistral'] = self.clients['mistral'].generate_report_text({'summary': summary_text})
        # fallback to gemini if mistral not available
        if 'gemini' in self.clients and 'mistral' not in out:
            out['gemini'] = self.clients['gemini'].generate_report_text({'summary': summary_text})
        # groq can be asked to reconcile recommendations
        if 'groq' in self.clients:
            out['groq'] = self.clients['groq'].generate_report_text({'summary': summary_text})
        return out

    def reconcile_ensemble(self, ensemble_meta: dict) -> Dict[str, str]:
        """Ask GROQ (preferred) to reconcile model outputs and produce a reasoning summary.

        ensemble_meta: dict with keys like {'models': [...], 'preds': ..., 'probas': ...}
        """
        out = {}
        if 'groq' in self.clients:
            out['groq'] = self.clients['groq'].generate_report_text(ensemble_meta)
        # if groq not available, ask gemini for a lightweight summary
        if 'gemini' in self.clients and 'groq' not in out:
            out['gemini'] = self.clients['gemini'].generate_report_text(ensemble_meta)
        return out

    def run_full_pipeline(self, df_live) -> Dict:
        """Run full LLM pipeline on a live-data snapshot.

        Steps:
        1. Create a short textual summary of the snapshot
        2. Generate narrative & recommendations
        3. Reconcile ensemble outputs (if present in df_live metadata)

        Returns a dict with keys: 'summaries', 'narratives', 'reconciliation'
        """
        # build a tiny text summary
        text = f"rows={len(df_live)}; mean_v1={df_live['value1'].mean():.4f}; mean_v2={df_live['value2'].mean():.4f}"
        summaries = self.summarize_report(text)
        # pick best summary (prefer gemini)
        preferred = summaries.get('gemini') or next(iter(summaries.values()), '')
        narratives = self.generate_narrative(preferred)

        # try to extract ensemble meta if present in df_live attrs
        ensemble_meta = getattr(df_live, '_ensemble_meta', None) or {}
        reconciliation = self.reconcile_ensemble(ensemble_meta)

        return {
            'summaries': summaries,
            'narratives': narratives,
            'reconciliation': reconciliation,
        }


# Utility helper to create orchestrator and run pipeline quickly
def run_on_snapshot(df_live):
    orch = LLMOrchestrator()
    return orch.run_full_pipeline(df_live)
