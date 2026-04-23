"""LLM orchestrator: route tasks to available LLM providers and run high-level pipelines.

Design / Strategy
- Providers detected via environment keys (GEMINI_*, MISTRAL_*, GROQ_*).
- Role assignment (where they'll intervene):
    * MISTRAL: summarization & anomaly extraction from periodic reports
    * GEMINI: narrative report generation, recommendations and user-facing text
    * GROQ: reasoning, reconciliation across model outputs and meta-analysis
- The orchestrator exposes a small API for: summarize_report, generate_narrative, reconcile_ensemble, run_full_pipeline

Notes
- This module uses src.llm_integration.LLMClient which will call real APIs if enabled in .env.
- Data is loaded via src.data_streaming.read_live_data() which handles timestamp adjustments.
"""

import os
import json
from pathlib import Path
from typing import Optional, Dict
from src.llm_integration import LLMClient
import logging

logger = logging.getLogger(__name__)

# mapping of provider -> role for documentation and routing
PROVIDER_ROLES = {
    'mistral': 'summarization',
    'gemini': 'narrative',
    'groq': 'reasoning',
}


class LLMOrchestrator:
    """High-level orchestrator that routes tasks to LLM providers based on role.

    Usage:
        orch = LLMOrchestrator()
        res = orch.run_full_pipeline(df_live)
    """

    def __init__(self):
        self.clients: Dict[str, LLMClient] = {}
        self._discover_providers()

    def _discover_providers(self):
        # instantiate clients for available providers (prefer Mistral for summarization)
        # If environment keys are not present in the current process, try loading .env
        try:
            if not any(os.getenv(k) for k in ['MISTRAL_API_KEY_1','MISTRAL_API_KEY_2','GEMINI_API_KEY_1','GEMINI_API_KEY_2','GROQ_API_KEY_1','GROQ_API_KEY_2']):
                try:
                    from src.env_loader import load_dotenv
                    load_dotenv()
                except Exception:
                    pass
        except Exception:
            pass

        # MISTRAL
        if os.getenv('MISTRAL_API_KEY_1') or os.getenv('MISTRAL_API_KEY_2'):
            self.clients['mistral'] = LLMClient('mistral')
            logger.info('Mistral client initialized')
        # GEMINI
        if os.getenv('GEMINI_API_KEY_1') or os.getenv('GEMINI_API_KEY_2'):
            self.clients['gemini'] = LLMClient('gemini')
            logger.info('Gemini client initialized')
        # GROQ
        if os.getenv('GROQ_API_KEY_1') or os.getenv('GROQ_API_KEY_2'):
            self.clients['groq'] = LLMClient('groq')
            logger.info('GROQ client initialized')

    def available_providers(self):
        return list(self.clients.keys())

    def summarize_report(self, text: str, max_tokens: int = 256, force_real: Optional[bool] = None) -> Dict[str, str]:
        """Produce summaries using MISTRAL (preferred) and fallback providers.

        Returns a dict mapping provider->summary.
        """
        out = {}
        # try mistral first
        if 'mistral' in self.clients:
            out['mistral'] = self.clients['mistral'].summarize(text, max_tokens=max_tokens, force_real=force_real)
        # fallback to gemini
        if 'gemini' in self.clients:
            out['gemini'] = self.clients['gemini'].summarize(text, max_tokens=max_tokens, force_real=force_real)
        # minimal groq usage for short summaries
        if 'groq' in self.clients:
            out['groq'] = self.clients['groq'].summarize(text, max_tokens=max_tokens, force_real=force_real)
        return out

    def generate_narrative(self, summary_text: str, force_real: Optional[bool] = None) -> Dict[str, str]:
        """Generate a narrative / recommendations using GEMINI (preferred).

        Returns provider->text mapping.
        """
        out = {}
        if 'gemini' in self.clients:
            out['gemini'] = self.clients['gemini'].generate_report_text({'summary': summary_text}, force_real=force_real)
        # fallback to mistral if gemini not available
        if 'mistral' in self.clients and 'gemini' not in out:
            out['mistral'] = self.clients['mistral'].generate_report_text({'summary': summary_text}, force_real=force_real)
        # groq can be asked to reconcile recommendations
        if 'groq' in self.clients:
            out['groq'] = self.clients['groq'].generate_report_text({'summary': summary_text}, force_real=force_real)
        return out

    def reconcile_ensemble(self, ensemble_meta: dict, force_real: Optional[bool] = None) -> Dict[str, str]:
        """Ask GROQ (preferred) to reconcile model outputs and produce a reasoning summary.

        ensemble_meta: dict with keys like {'models': [...], 'preds': ..., 'probas': ...}
        """
        out = {}
        if 'groq' in self.clients:
            out['groq'] = self.clients['groq'].generate_report_text(ensemble_meta, force_real=force_real)
        # if groq not available, ask gemini for a lightweight summary
        if 'gemini' in self.clients and 'groq' not in out:
            out['gemini'] = self.clients['gemini'].generate_report_text(ensemble_meta, force_real=force_real)
        return out

    def _format_dataframe_for_llm(self, df) -> str:
        """Format a DataFrame into a text summary suitable for LLM consumption."""
        if df is None or df.empty:
            return "No data available for analysis."
        
        lines = []
        lines.append(f"Data Overview: {len(df)} rows")
        
        # Column information
        lines.append(f"Columns: {', '.join(df.columns.tolist())}")
        
        # Numeric columns statistics
        num_cols = df.select_dtypes(include=['number']).columns.tolist()
        if num_cols:
            lines.append("\nNumeric Statistics:")
            for c in num_cols:
                try:
                    s = df[c].describe()
                    lines.append(f"  {c}: mean={s.get('mean',0):.4f}, std={s.get('std',0):.4f}, min={s.get('min',0):.4f}, 25%={s.get('25%',0):.4f}, 50%={s.get('50%',0):.4f}, 75%={s.get('75%',0):.4f}, max={s.get('max',0):.4f}")
                except Exception as e:
                    lines.append(f"  {c}: Error computing stats: {e}")
        
        # Categorical/other columns
        cat_cols = df.select_dtypes(exclude=['number']).columns.tolist()
        if cat_cols:
            lines.append("\nCategorical/Other Columns:")
            for c in cat_cols[:5]:  # Limit to first 5 to avoid too much text
                try:
                    unique_count = df[c].nunique()
                    lines.append(f"  {c}: {unique_count} unique values")
                    if unique_count <= 10:
                        lines.append(f"    Values: {', '.join(str(v) for v in df[c].dropna().unique()[:10])}")
                except Exception:
                    pass
        
        # Time information
        if 'timestamp' in df.columns:
            try:
                lines.append("\nTime Range:")
                lines.append(f"  Earliest: {df['timestamp'].min()}")
                lines.append(f"  Latest: {df['timestamp'].max()}")
                lines.append(f"  Duration: {(df['timestamp'].max() - df['timestamp'].min())}")
            except Exception:
                pass
        
        # Sample rows
        try:
            sample_size = min(3, len(df))
            if sample_size > 0:
                lines.append(f"\nSample ({sample_size} rows):")
                sample = df.head(sample_size)
                for _, row in sample.iterrows():
                    row_str = ", ".join(f"{col}={val}" for col, val in row.items())
                    lines.append(f"  {row_str}")
        except Exception:
            pass
        
        return "\n".join(lines)

    def concert_and_merge(self, question: str, rounds: int = 1, include_data: bool = True, data_window_sec: int = 3600, max_tokens: int = 256, force_real: Optional[bool] = None) -> Dict:
        """Run a multi-round concertation among available providers and produce a merged answer.

        Returns a dict with keys: question, data_summary, rounds (list of contributions), aggregator, merged.
        """
        data_summary = ''
        df_ctx = None
        if include_data:
            try:
                from src.data_streaming import read_live_data
                # Use absolute path to ensure file is found
                # __file__ is in src/llm_orchestrator.py, so parents[1] is project root
                project_root = Path(__file__).resolve().parents[1]
                data_path = str(project_root / 'data' / 'stream' / 'live_data.csv')
                df_ctx = read_live_data(last_seconds=data_window_sec, limit=1000, path=data_path)
                
                if df_ctx is not None and not df_ctx.empty:
                    # Format data for LLM consumption
                    data_summary = self._format_dataframe_for_llm(df_ctx)
                    logger.info(f"Loaded {len(df_ctx)} rows of live data for LLM analysis")
                else:
                    data_summary = "No live data available for analysis. Check: 1) live_data.csv exists, 2) timestamps are valid, 3) data_streaming module can access the file."
                    logger.warning("No live data available - check file path and timestamps")
            except Exception as e:
                data_summary = f"Error reading live data: {e}. Please check data/stream/live_data.csv exists and is readable."
                logger.error(f"Error loading live data: {e}", exc_info=True)

        # ensure providers discovered
        if not self.clients:
            self._discover_providers()
        providers = list(self.clients.keys())
        # Note: force_real is now passed to individual calls, not set on client.enabled

        conversation = []
        # initial round
        for p in providers:
            client = self.clients[p]
            prompt = question
            if include_data and data_summary:
                prompt = f"{prompt}\n\nData context:\n{data_summary}"
            try:
                ans = client.summarize(prompt, max_tokens=max_tokens, force_real=force_real)
            except Exception as e:
                ans = f"[Error calling {p}: {e}]"
            conversation.append({'round': 0, 'provider': p, 'text': ans})

        # debate rounds
        for r in range(1, rounds + 1):
            prev_texts = {m['provider']: m['text'] for m in conversation if m['round'] == r-1}
            for p in providers:
                client = self.clients[p]
                others = "\n".join([f"{other}: {txt}" for other, txt in prev_texts.items() if other != p])
                prompt = f"Question: {question}\nData context:\n{data_summary}\nContributions:\n{others}\nPlease respond succinctly with your view and a recommendation."
                try:
                    ans = client.summarize(prompt, max_tokens=max_tokens, force_real=force_real)
                except Exception as e:
                    ans = f"[Error calling {p}: {e}]"
                conversation.append({'round': r, 'provider': p, 'text': ans})

        # aggregator selection and merge
        aggregator = None
        for ag in ['groq', 'gemini', 'mistral']:
            if ag in self.clients:
                aggregator = ag
                break
        merged = ''
        if aggregator:
            ag_client = self.clients[aggregator]
            all_texts = "\n".join([f"({c['round']}) {c['provider']}: {c['text']}" for c in conversation])
            agg_prompt = f"You are a conciliator. Question: {question}\nData context:\n{data_summary}\nSynthesize the contributions below, list agreements and disagreements, and produce a final consolidated recommendation.\n\nContributions:\n{all_texts}\n\nProvide a concise merged answer."
            try:
                merged = ag_client.summarize(agg_prompt, max_tokens=512, force_real=force_real)
            except Exception as e:
                merged = f"[Error in aggregation by {aggregator}: {e}]"
        else:
            merged = "[No aggregator configured.]"

        return {'question': question, 'data_summary': data_summary, 'rounds': conversation, 'aggregator': aggregator, 'merged': merged}

    def run_full_pipeline(self, df_live, force_real: Optional[bool] = None) -> Dict:
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
        # pick best summary (prefer mistral)
        preferred = summaries.get('mistral') or next(iter(summaries.values()), '')
        narratives = self.generate_narrative(preferred, force_real=force_real)

        # try to extract ensemble meta if present in df_live attrs
        ensemble_meta = getattr(df_live, '_ensemble_meta', None) or {}
        reconciliation = self.reconcile_ensemble(ensemble_meta, force_real=force_real)

        return {
            'summaries': summaries,
            'narratives': narratives,
            'reconciliation': reconciliation,
        }


# Utility helper to create orchestrator and run pipeline quickly
def run_on_snapshot(df_live):
    orch = LLMOrchestrator()
    return orch.run_full_pipeline(df_live)
