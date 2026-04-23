"""LLM integration with optional real API calls and simple quota limiting.

This module prefers provider endpoints defined by environment variables:
- MISTRAL_API_URL and MISTRAL_API_KEY_*
- GEMINI_API_URL and GEMINI_API_KEY_*
- GROQ_API_URL and GROQ_API_KEY_*

To enable actual network calls set LLM_CALLS_ENABLED=true in .env.
Quota is enforced by LLM_MAX_CALLS_PER_HOUR (default 60).
If network calls are not available or quota is exhausted, the client returns deterministic placeholders.
"""
import json
import os
import time
from pathlib import Path
from typing import Optional
from urllib.error import URLError, HTTPError
from urllib.request import Request, urlopen
from src.llm_usage import ensure_usage_file, estimate_tokens, get_cost_per_1k, log_usage


class LLMClient:
    def __init__(self, provider: Optional[str] = None):
        self.provider = provider or self._select_provider()
        # initial values (will be refreshed at call time)
        self.enabled = os.getenv('LLM_CALLS_ENABLED', 'false').lower() in ('1', 'true', 'yes')
        self.max_calls_per_hour = int(os.getenv('LLM_MAX_CALLS_PER_HOUR', '60'))
        self.quota_file = Path('logs/llm_quota.json')
        self._ensure_quota_file()
        try:
            ensure_usage_file()
        except Exception:
            pass

    def _ensure_quota_file(self):
        if not self.quota_file.parent.exists():
            self.quota_file.parent.mkdir(parents=True, exist_ok=True)
        if not self.quota_file.exists():
            self.quota_file.write_text(json.dumps([]))

    def _select_provider(self) -> Optional[str]:
        # choose provider in order of preference: prefer Mistral for summarization by default
        if os.getenv('MISTRAL_API_KEY_1') or os.getenv('MISTRAL_API_KEY_2'):
            return 'mistral'
        if os.getenv('GEMINI_API_KEY_1') or os.getenv('GEMINI_API_KEY_2'):
            return 'gemini'
        if os.getenv('GROQ_API_KEY_1') or os.getenv('GROQ_API_KEY_2'):
            return 'groq'
        return None

    def available_providers(self):
        providers = []
        if os.getenv('MISTRAL_API_KEY_1') or os.getenv('MISTRAL_API_KEY_2'):
            providers.append('mistral')
        if os.getenv('GEMINI_API_KEY_1') or os.getenv('GEMINI_API_KEY_2'):
            providers.append('gemini')
        if os.getenv('GROQ_API_KEY_1') or os.getenv('GROQ_API_KEY_2'):
            providers.append('groq')
        return providers

    def _acquire_quota(self) -> bool:
        """Prune old entries and attempt to record a new call. Return True if allowed."""
        try:
            now = time.time()
            raw = self.quota_file.read_text()
            arr = json.loads(raw) if raw.strip() else []
            # keep only last hour
            arr = [t for t in arr if now - t < 3600]
            if len(arr) >= self.max_calls_per_hour:
                return False
            arr.append(now)
            self.quota_file.write_text(json.dumps(arr))
            return True
        except Exception:
            # if something goes wrong, be conservative and disallow network call
            return False

    def _http_post(self, url: str, headers: dict, payload: dict, timeout: int = 30, retries: int = 2, backoff: float = 1.0):
        """POST helper with simple retry/backoff for transient network errors."""
        body = json.dumps(payload).encode('utf-8')
        req = Request(url, data=body, headers=headers, method='POST')
        attempt = 0
        while True:
            try:
                with urlopen(req, timeout=timeout) as resp:
                    raw = resp.read().decode('utf-8')
                    try:
                        return json.loads(raw)
                    except Exception:
                        return raw
            except HTTPError as e:
                # Retry on server errors (5xx)
                if 500 <= getattr(e, 'code', 0) < 600 and attempt < retries:
                    attempt += 1
                    time.sleep(backoff * (2 ** (attempt - 1)))
                    continue
                try:
                    body = e.read().decode('utf-8')
                    return {'error': f'HTTPError {e.code}: {e.reason}', 'body': body}
                except Exception:
                    return {'error': f'HTTPError {e.code}: {e.reason}'}
            except URLError as e:
                if attempt < retries:
                    attempt += 1
                    time.sleep(backoff * (2 ** (attempt - 1)))
                    continue
                return {'error': f'URLError: {e}'}
            except Exception as e:
                if attempt < retries:
                    attempt += 1
                    time.sleep(backoff * (2 ** (attempt - 1)))
                    continue
                return {'error': f'Network error: {e}'}

    def summarize(self, text: str, max_tokens: int = 256) -> str:
        """Summarize text using the preferred provider if enabled and quota allows.
        Falls back to a deterministic placeholder.
        This implementation records estimated token usage and approximate cost.
        """
        # refresh dynamic flags from env to allow runtime changes via UI
        self.enabled = os.getenv('LLM_CALLS_ENABLED', 'false').lower() in ('1', 'true', 'yes')
        self.max_calls_per_hour = int(os.getenv('LLM_MAX_CALLS_PER_HOUR', '60'))

        provider = self.provider or 'none'
        s = text.strip().replace('\n', ' ')
        prompt_tokens = estimate_tokens(text)

        if not self.enabled:
            try:
                log_usage(provider, 'summarize', prompt_tokens, 0, 0.0, {'skipped': 'disabled'})
            except Exception:
                pass
            return f"[LLM summary by {provider}] {s[:200]}..."

        if not self._acquire_quota():
            try:
                log_usage(provider, 'summarize', prompt_tokens, 0, 0.0, {'skipped': 'quota'})
            except Exception:
                pass
            return "[LLM skipped due to quota exhaustion]"

        def _process_resp(resp):
            out_text = None
            api_usage = None
            if isinstance(resp, dict):
                out_text = resp.get('output') or resp.get('text')
                if not out_text and 'choices' in resp and isinstance(resp['choices'], list) and resp['choices']:
                    first = resp['choices'][0]
                    if isinstance(first, dict):
                        out_text = first.get('text') or first.get('message') or None
                    else:
                        out_text = str(first)
                if not out_text and 'body' in resp and isinstance(resp['body'], str):
                    out_text = resp['body']

                # extract usage if provider returns it
                usage = resp.get('usage') or resp.get('token_usage') or (resp.get('meta', {}) or {}).get('usage')
                if isinstance(usage, dict):
                    api_prompt = usage.get('prompt_tokens') or usage.get('input_tokens') or usage.get('prompt')
                    api_completion = usage.get('completion_tokens') or usage.get('output_tokens') or usage.get('completion')
                    api_usage = {'prompt_tokens': api_prompt, 'completion_tokens': api_completion, 'raw': usage}

            else:
                out_text = str(resp)

            prompt_tokens_to_log = int(api_usage['prompt_tokens']) if api_usage and api_usage.get('prompt_tokens') is not None else prompt_tokens
            completion_tokens = int(api_usage['completion_tokens']) if api_usage and api_usage.get('completion_tokens') is not None else (estimate_tokens(out_text) if out_text else 0)
            cost = (prompt_tokens_to_log + completion_tokens) / 1000.0 * get_cost_per_1k(provider)
            success = not (isinstance(resp, dict) and 'error' in resp)
            extra = {'success': success}
            if isinstance(resp, dict) and 'error' in resp:
                extra['error'] = resp.get('error')
            if api_usage is not None:
                extra['api_usage'] = api_usage
            try:
                log_usage(provider, 'summarize', prompt_tokens_to_log, completion_tokens, cost, extra)
            except Exception:
                pass

            if isinstance(resp, dict):
                return out_text or (resp.get('output') or resp.get('text') or str(resp))
            return str(resp)

        # Prefer Mistral for summarization (if configured)
        if provider == 'mistral' or (provider is None and 'mistral' in self.available_providers()):
            url = os.getenv('MISTRAL_API_URL')
            key = os.getenv('MISTRAL_API_KEY_1') or os.getenv('MISTRAL_API_KEY_2')
            if url and key:
                headers = {'Authorization': f'Bearer {key}', 'Content-Type': 'application/json'}
                payload = {'input': text, 'max_tokens': max_tokens}
                resp = self._http_post(url, headers, payload)
                return _process_resp(resp)

        # Fallback: Gemini if configured
        if provider == 'gemini' or (provider is None and 'gemini' in self.available_providers()):
            url = os.getenv('GEMINI_API_URL')
            key = os.getenv('GEMINI_API_KEY_1') or os.getenv('GEMINI_API_KEY_2')
            if url and key:
                headers = {'Authorization': f'Bearer {key}', 'Content-Type': 'application/json'}
                payload = {'prompt': text, 'max_tokens': max_tokens}
                resp = self._http_post(url, headers, payload)
                return _process_resp(resp)

        # Groq or other providers - generic attempt
        if provider == 'groq' or (provider is None and 'groq' in self.available_providers()):
            url = os.getenv('GROQ_API_URL')
            key = os.getenv('GROQ_API_KEY_1') or os.getenv('GROQ_API_KEY_2')
            if url and key:
                headers = {'Authorization': f'Bearer {key}', 'Content-Type': 'application/json'}
                payload = {'input': text, 'max_tokens': max_tokens}
                resp = self._http_post(url, headers, payload)
                return _process_resp(resp)

        # Otherwise fallback to placeholder
        try:
            log_usage(provider, 'summarize', prompt_tokens, 0, 0.0, {'skipped': 'no_provider'})
        except Exception:
            pass
        return f"[LLM summary by {provider}] {s[:200]}..."

    def generate_report_text(self, df_summary: dict) -> str:
        """Ask an LLM to craft a narrative report from a dict of summary statistics.
        Prefer Gemini for narrative generation (per routing decision).
        Records estimated token usage and estimated cost.
        """
        # refresh dynamic flags from env to allow runtime changes via UI
        self.enabled = os.getenv('LLM_CALLS_ENABLED', 'false').lower() in ('1', 'true', 'yes')
        self.max_calls_per_hour = int(os.getenv('LLM_MAX_CALLS_PER_HOUR', '60'))

        text = "; ".join(f"{k}: {v}" for k, v in df_summary.items())
        prompt_tokens = estimate_tokens(text)

        if not self.enabled:
            try:
                log_usage(self.provider or 'none', 'generate_report_text', prompt_tokens, 0, 0.0, {'skipped': 'disabled'})
            except Exception:
                pass
            return f"[Report by {self.provider or 'none'}] {text}"

        if not self._acquire_quota():
            try:
                log_usage(self.provider or 'none', 'generate_report_text', prompt_tokens, 0, 0.0, {'skipped': 'quota'})
            except Exception:
                pass
            return "[LLM skipped due to quota exhaustion]"

        def _process_resp(resp):
            out_text = None
            api_usage = None
            if isinstance(resp, dict):
                out_text = resp.get('output') or resp.get('text')
                if not out_text and 'choices' in resp and isinstance(resp['choices'], list) and resp['choices']:
                    first = resp['choices'][0]
                    if isinstance(first, dict):
                        out_text = first.get('text') or first.get('message') or None
                    else:
                        out_text = str(first)
                if not out_text and 'body' in resp and isinstance(resp['body'], str):
                    out_text = resp['body']

                # extract usage if provider returns it
                usage = resp.get('usage') or resp.get('token_usage') or (resp.get('meta', {}) or {}).get('usage')
                if isinstance(usage, dict):
                    api_prompt = usage.get('prompt_tokens') or usage.get('input_tokens') or usage.get('prompt')
                    api_completion = usage.get('completion_tokens') or usage.get('output_tokens') or usage.get('completion')
                    api_usage = {'prompt_tokens': api_prompt, 'completion_tokens': api_completion, 'raw': usage}

            else:
                out_text = str(resp)

            prompt_tokens_to_log = int(api_usage['prompt_tokens']) if api_usage and api_usage.get('prompt_tokens') is not None else prompt_tokens
            completion_tokens = int(api_usage['completion_tokens']) if api_usage and api_usage.get('completion_tokens') is not None else (estimate_tokens(out_text) if out_text else 0)
            cost = (prompt_tokens_to_log + completion_tokens) / 1000.0 * get_cost_per_1k(self.provider or 'none')
            success = not (isinstance(resp, dict) and 'error' in resp)
            extra = {'success': success}
            if isinstance(resp, dict) and 'error' in resp:
                extra['error'] = resp.get('error')
            if api_usage is not None:
                extra['api_usage'] = api_usage
            try:
                log_usage(self.provider or 'none', 'generate_report_text', prompt_tokens_to_log, completion_tokens, cost, extra)
            except Exception:
                pass

            if isinstance(resp, dict):
                return out_text or (resp.get('output') or resp.get('text') or str(resp))
            return str(resp)

        # Prefer Gemini for narrative generation
        if self.provider == 'gemini' or (self.provider is None and 'gemini' in self.available_providers()):
            url = os.getenv('GEMINI_API_URL')
            key = os.getenv('GEMINI_API_KEY_1') or os.getenv('GEMINI_API_KEY_2')
            if url and key:
                headers = {'Authorization': f'Bearer {key}', 'Content-Type': 'application/json'}
                payload = {'prompt': text, 'max_tokens': 512}
                resp = self._http_post(url, headers, payload)
                return _process_resp(resp)

        # Fallback to Mistral or Groq
        if 'mistral' in self.available_providers():
            url = os.getenv('MISTRAL_API_URL')
            key = os.getenv('MISTRAL_API_KEY_1') or os.getenv('MISTRAL_API_KEY_2')
            if url and key:
                headers = {'Authorization': f'Bearer {key}', 'Content-Type': 'application/json'}
                payload = {'input': text, 'max_tokens': 512}
                resp = self._http_post(url, headers, payload)
                return _process_resp(resp)

        try:
            log_usage(self.provider or 'none', 'generate_report_text', prompt_tokens, 0, 0.0, {'skipped': 'no_provider'})
        except Exception:
            pass
        return f"[Report by {self.provider or 'none'}] {text}"
