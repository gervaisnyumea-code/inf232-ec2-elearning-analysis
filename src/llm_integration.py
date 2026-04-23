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


class LLMClient:
    def __init__(self, provider: Optional[str] = None):
        self.provider = provider or self._select_provider()
        # initial values (will be refreshed at call time)
        self.enabled = os.getenv('LLM_CALLS_ENABLED', 'false').lower() in ('1', 'true', 'yes')
        self.max_calls_per_hour = int(os.getenv('LLM_MAX_CALLS_PER_HOUR', '60'))
        self.quota_file = Path('logs/llm_quota.json')
        self._ensure_quota_file()

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
        """
        # refresh dynamic flags from env to allow runtime changes via UI
        self.enabled = os.getenv('LLM_CALLS_ENABLED', 'false').lower() in ('1', 'true', 'yes')
        self.max_calls_per_hour = int(os.getenv('LLM_MAX_CALLS_PER_HOUR', '60'))

        provider = self.provider or 'none'
        s = text.strip().replace('\n', ' ')

        if not self.enabled:
            return f"[LLM summary by {provider}] {s[:200]}..."

        if not self._acquire_quota():
            return "[LLM skipped due to quota exhaustion]"

        # Prefer Mistral for summarization (if configured)
        if provider == 'mistral' or (provider is None and 'mistral' in self.available_providers()):
            url = os.getenv('MISTRAL_API_URL')
            key = os.getenv('MISTRAL_API_KEY_1') or os.getenv('MISTRAL_API_KEY_2')
            if url and key:
                headers = {'Authorization': f'Bearer {key}', 'Content-Type': 'application/json'}
                payload = {'input': text, 'max_tokens': max_tokens}
                resp = self._http_post(url, headers, payload)
                if isinstance(resp, dict):
                    return resp.get('output') or resp.get('text') or str(resp)
                return str(resp)

        # Fallback: Gemini if configured
        if provider == 'gemini' or (provider is None and 'gemini' in self.available_providers()):
            url = os.getenv('GEMINI_API_URL')
            key = os.getenv('GEMINI_API_KEY_1') or os.getenv('GEMINI_API_KEY_2')
            if url and key:
                headers = {'Authorization': f'Bearer {key}', 'Content-Type': 'application/json'}
                payload = {'prompt': text, 'max_tokens': max_tokens}
                resp = self._http_post(url, headers, payload)
                if isinstance(resp, dict):
                    return resp.get('output') or resp.get('text') or str(resp)
                return str(resp)

        # Groq or other providers - generic attempt
        if provider == 'groq' or (provider is None and 'groq' in self.available_providers()):
            url = os.getenv('GROQ_API_URL')
            key = os.getenv('GROQ_API_KEY_1') or os.getenv('GROQ_API_KEY_2')
            if url and key:
                headers = {'Authorization': f'Bearer {key}', 'Content-Type': 'application/json'}
                payload = {'input': text, 'max_tokens': max_tokens}
                resp = self._http_post(url, headers, payload)
                if isinstance(resp, dict):
                    return resp.get('output') or resp.get('text') or str(resp)
                return str(resp)

        # Otherwise fallback to placeholder
        return f"[LLM summary by {provider}] {s[:200]}..."

    def generate_report_text(self, df_summary: dict) -> str:
        """Ask an LLM to craft a narrative report from a dict of summary statistics.
        Prefer Gemini for narrative generation (per routing decision).
        """
        # refresh dynamic flags from env to allow runtime changes via UI
        self.enabled = os.getenv('LLM_CALLS_ENABLED', 'false').lower() in ('1', 'true', 'yes')
        self.max_calls_per_hour = int(os.getenv('LLM_MAX_CALLS_PER_HOUR', '60'))

        text = "; ".join(f"{k}: {v}" for k, v in df_summary.items())

        if not self.enabled:
            return f"[Report by {self.provider or 'none'}] {text}"

        if not self._acquire_quota():
            return "[LLM skipped due to quota exhaustion]"

        # Prefer Gemini for narrative generation
        if self.provider == 'gemini' or (self.provider is None and 'gemini' in self.available_providers()):
            url = os.getenv('GEMINI_API_URL')
            key = os.getenv('GEMINI_API_KEY_1') or os.getenv('GEMINI_API_KEY_2')
            if url and key:
                headers = {'Authorization': f'Bearer {key}', 'Content-Type': 'application/json'}
                payload = {'prompt': text, 'max_tokens': 512}
                resp = self._http_post(url, headers, payload)
                if isinstance(resp, dict):
                    return resp.get('output') or resp.get('text') or str(resp)
                return str(resp)

        # Fallback to Mistral or Groq
        if 'mistral' in self.available_providers():
            url = os.getenv('MISTRAL_API_URL')
            key = os.getenv('MISTRAL_API_KEY_1') or os.getenv('MISTRAL_API_KEY_2')
            if url and key:
                headers = {'Authorization': f'Bearer {key}', 'Content-Type': 'application/json'}
                payload = {'input': text, 'max_tokens': 512}
                resp = self._http_post(url, headers, payload)
                if isinstance(resp, dict):
                    return resp.get('output') or resp.get('text') or str(resp)
                return str(resp)

        return f"[Report by {self.provider or 'none'}] {text}"
