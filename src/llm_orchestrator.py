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

IMPROVEMENTS:
- Context caching: data is cached and reused across calls
- Conversation memory: history is stored and passed to LLMs
- Extended tokens: max_tokens increased for complete responses
"""

import os
import json
import logging
import time
from pathlib import Path
from typing import Optional, Dict, List
from collections import defaultdict

# CRITICAL: Import llm_init FIRST to ensure .env is loaded before any LLMClient
from src.llm_init import ensure_env_loaded
# Force load .env NOW before creating any clients
ensure_env_loaded(override=True)

from src.llm_integration import LLMClient

logger = logging.getLogger(__name__)

# mapping of provider -> role for documentation and routing
PROVIDER_ROLES = {
    'mistral': 'summarization',
    'gemini': 'narrative',
    'groq': 'reasoning',
}


class ConversationMemory:
    """Persistent conversation memory for LLM context."""
    
    def __init__(self, max_history: int = 20):
        self.max_history = max_history
        self.history: List[Dict] = []
        self._cache: Dict = {}
        self._last_data_load = 0
        self._data_cache_ttl = 300  # 5 minutes cache
    
    def add(self, role: str, content: str, metadata: Dict = None):
        """Add a message to history."""
        self.history.append({
            'role': role,
            'content': content,
            'timestamp': time.time(),
            'metadata': metadata or {}
        })
        # Keep only last max_history messages
        if len(self.history) > self.max_history:
            self.history = self.history[-self.max_history:]
    
    def get_context(self, include_history: bool = True) -> str:
        """Get formatted context from history."""
        if not include_history or not self.history:
            return ""
        
        lines = ["=== HISTORY (recent conversations) ==="]
        for msg in self.history[-5:]:  # Last 5 messages
            lines.append(f"[{msg['role']}]: {msg['content'][:200]}...")
        return "\n".join(lines)
    
    def cache_data(self, key: str, data: str):
        """Cache data summary."""
        self._cache[key] = data
        self._last_data_load = time.time()
    
    def get_cached_data(self, key: str) -> Optional[str]:
        """Get cached data if not expired."""
        if key in self._cache and (time.time() - self._last_data_load) < self._data_cache_ttl:
            return self._cache[key]
        return None
    
    def clear(self):
        """Clear all memory."""
        self.history = []
        self._cache = {}


# Global conversation memory
_conversation_memory = ConversationMemory(max_history=20)


def get_conversation_memory() -> ConversationMemory:
    """Get the global conversation memory instance."""
    global _conversation_memory
    return _conversation_memory


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
        # NOTE: .env is now loaded by llm_init at import time (see top of file)
        # This method only discovers and instantiates clients
        
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
        
        # Log discovery results
        if self.clients:
            logger.info(f"Discovered LLM providers: {list(self.clients.keys())}")
        else:
            logger.warning("No LLM providers discovered - check .env file")

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

    def _safe_summarize(self, client, text: str, max_tokens: int = 1024, force_real=None) -> str:
        """Call client.summarize with force_real, falling back without it if TypeError."""
        try:
            return client.summarize(text, max_tokens=max_tokens, force_real=force_real)
        except TypeError:
            # Fallback: stale LLMClient without force_real parameter
            return client.summarize(text, max_tokens=max_tokens)

    def concert_and_merge(
        self, 
        question: str, 
        rounds: int = 2,
        include_data: bool = True, 
        data_window_sec: int = 40000,
        max_tokens: int = 1024,
        force_real: Optional[bool] = None,
        use_memory: bool = True,
        cache_data: bool = True
    ) -> Dict:
        """Run a multi-round concertation among available providers and produce a merged answer.

        Args:
            question: The question/prompt to ask
            rounds: Number of debate rounds (default 2)
            include_data: Include live data context
            data_window_sec: Time window for data (default 40000 = ~11 hours)
            max_tokens: Max tokens for response (default 1024 for complete answers)
            force_real: Force real API calls
            use_memory: Include conversation history
            cache_data: Cache data for reuse

        Returns:
            dict with keys: question, data_summary, rounds, aggregator, merged, memory
        """
        global _conversation_memory
        
        data_summary = ''
        df_ctx = None
        cache_key = f'data_{data_window_sec}'
        
        # USE CACHE if available
        if cache_data:
            cached = _conversation_memory.get_cached_data(cache_key)
            if cached:
                data_summary = cached
                logger.info("Using cached data summary")
        
        if include_data and not data_summary:
            try:
                from src.data_streaming import read_live_data
                project_root = Path(__file__).resolve().parents[1]
                data_path = str(project_root / 'data' / 'stream' / 'live_data.csv')
                df_ctx = read_live_data(last_seconds=data_window_sec, limit=2000, path=data_path)
                
                if df_ctx is not None and not df_ctx.empty:
                    # ENHANCED data formatting with MORE details
                    data_summary = self._format_dataframe_extended(df_ctx)
                    # CACHE it
                    if cache_data:
                        _conversation_memory.cache_data(cache_key, data_summary)
                    logger.info(f"Loaded {len(df_ctx)} rows for LLM analysis")
                else:
                    data_summary = "No live data available."
            except Exception as e:
                data_summary = f"Error: {e}"

        # Get providers
        if not self.clients:
            self._discover_providers()
        providers = list(self.clients.keys())

        # Get conversation history for context
        history_context = ""
        if use_memory:
            history_context = _conversation_memory.get_context()

        conversation = []
        
        # IMPROVED system prompt for better analysis
        system_prompt = """Tu es un analyste de données pédagogiques expert.
Ton rôle est d'analyser les données d'étudiants e-learning et fournir des recommandations ACTIONNABLES.
Réponds en français, de manière DÉTAILLÉE et STRUCTURÉE.
Utilise des listes numérotées et des sections claires."""

        # Initial round
        for p in providers:
            client = self.clients[p]
            
            # Build ENHANCED prompt with history + data + question
            prompt_parts = [system_prompt]
            if history_context:
                prompt_parts.append(f"\n{history_context}")
            prompt_parts.append(f"\n=== DONNÉES ===\n{data_summary}" if data_summary else "\nAucune donnée disponible.")
            prompt_parts.append(f"\n=== QUESTION ===\n{question}")
            prompt_parts.append("\n\nRéponds de façon COMPLETE avec与分析 détaillée.")
            
            full_prompt = "\n".join(prompt_parts)
            
            try:
                ans = self._safe_summarize(client, full_prompt, max_tokens=max_tokens, force_real=force_real)
            except Exception as e:
                ans = f"[Erreur {p}: {e}]"
            
            conversation.append({'round': 0, 'provider': p, 'text': ans})
            # ADD to memory
            if use_memory:
                _conversation_memory.add(p, ans[:500], {'round': 0, 'question': question})

        # Debate rounds
        for r in range(1, rounds + 1):
            prev_texts = {m['provider']: m['text'] for m in conversation if m['round'] == r - 1}
            
            for p in providers:
                client = self.clients[p]
                
                others = "\n".join([f"=== {other.upper()} ===\n{txt[:300]}" for other, txt in prev_texts.items() if other != p])
                
                prompt = f"""{system_prompt}

{history_context}

=== DONNÉES ===
{data_summary}

=== QUESTION ===
{question}

=== CONTRIBUTIONS PRÉCÉDENTES ===
{others}

Ta tâche:
1. Synthétise les insights des autres providers
2. Apporte une ANALYSE COMPLÉMENTAIRE
3. Propose des RECOMMANDATIONS SPÉCIFIQUES

Réponds de façon COMPLETE et détaillée."""

                try:
                    ans = self._safe_summarize(client, prompt, max_tokens=max_tokens, force_real=force_real)
                except Exception as e:
                    ans = f"[Erreur {p}: {e}]"
                
                conversation.append({'round': r, 'provider': p, 'text': ans})
                if use_memory:
                    _conversation_memory.add(p, ans[:500], {'round': r})

        # Aggregator selection - use GROQ for best reasoning
        aggregator = None
        for ag in ['groq', 'gemini', 'mistral']:
            if ag in self.clients:
                aggregator = ag
                break
        
        merged = ''
        if aggregator:
            ag_client = self.clients[aggregator]
            
            all_texts = "\n=== SYNTHÈSE DES CONTRIBUTIONS ===\n" + "\n\n".join([
                f"[{c['provider']} Tour {c['round']}]:\n{c['text']}"
                for c in conversation
            ])
            
            agg_prompt = f"""Tu es un CONCILIATEUR expert. Ta misión:
1. Synthétiser toutes les contributions ci-dessous
2. Identifier les points CONSENSUS
3. Produire une RÉPONSE FINALE COMPLÈTE et structurée

CONTEXT: Données e-learning étudiants
QUESTION: {question}

{all_texts}

=== RÉPONSE FINALE ===
Produis une réponse complète en français avec:
- Résumé Executive
- Analyse détaillée (par section)
- Recommandations numérotées
- Conclusion

Rends la réponse LA PLUS COMPLÈTE possible."""

            try:
                merged = self._safe_summarize(ag_client, agg_prompt, max_tokens=max_tokens * 2, force_real=force_real)
            except Exception as e:
                merged = f"[Erreur aggregation: {e}]"

        # ADD final to memory
        if use_memory:
            _conversation_memory.add('merged', merged[:500], {'type': 'final'})

        return {
            'question': question,
            'data_summary': data_summary,
            'rounds': conversation,
            'aggregator': aggregator,
            'merged': merged,
            'memory': _conversation_memory.history if use_memory else []
        }

    def _format_dataframe_extended(self, df) -> str:
        """Enhanced dataframe formatting with MORE details for better LLM context."""
        if df is None or df.empty:
            return "No data available."
        
        lines = []
        lines.append(f"=== DATASET: {len(df)} lignes, {len(df.columns)} colonnes ===")
        lines.append(f"Colonnes: {', '.join(df.columns.tolist())}")
        
        # Numeric statistics - MORE DETAILED
        num_cols = df.select_dtypes(include=['number']).columns.tolist()
        if num_cols:
            lines.append("\n📊 STATISTIQUES DÉTAILLÉES:")
            for c in num_cols:
                try:
                    s = df[c].describe()
                    lines.append(f"\n--- {c} ---")
                    lines.append(f"  count: {s.get('count', 0)}")
                    lines.append(f"  mean: {s.get('mean', 0):.4f}")
                    lines.append(f"  std: {s.get('std', 0):.4f}")
                    lines.append(f"  min: {s.get('min', 0):.4f}")
                    lines.append(f"  25%: {s.get('25%', 0):.4f}")
                    lines.append(f"  50%: {s.get('50%', 0):.4f}")
                    lines.append(f"  75%: {s.get('75%', 0):.4f}")
                    lines.append(f"  max: {s.get('max', 0):.4f}")
                    
                    # Add correlation with other numeric cols
                    if len(num_cols) > 1:
                        corr_with = df[num_cols].corr()[c].sort_values(ascending=False)
                        top_corr = corr_with[abs(corr_with) > 0.3].index.tolist()
                        if c in top_corr: top_corr.remove(c)
                        if top_corr:
                            lines.append(f"  corrélations fortes: {top_corr[:3]}")
                except Exception as e:
                    lines.append(f"  Erreur: {e}")
        
        # Time info
        if 'timestamp' in df.columns:
            try:
                lines.append("\n⏰ TEMPS:")
                lines.append(f"  Début: {df['timestamp'].min()}")
                lines.append(f"  Fin: {df['timestamp'].max()}")
                dur = df['timestamp'].max() - df['timestamp'].min()
                lines.append(f"  Durée: {dur}")
            except Exception:
                pass
        
        # Sample
        lines.append("\n📝 EXEMPLE (5 premières lignes):")
        for i, row in df.head(5).iterrows():
            vals = ", ".join(f"{c}={row[c]:.3f}" if isinstance(row[c], float) else f"{c}={row[c]}" for c in df.columns[:4])
            lines.append(f"  Row {i}: {vals}")
        
        return "\n".join(lines)

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
