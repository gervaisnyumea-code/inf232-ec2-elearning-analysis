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
import logging
from pathlib import Path
from typing import Optional, Dict, Any
from urllib.error import URLError, HTTPError
from urllib.request import Request, urlopen
from src.llm_usage import ensure_usage_file, estimate_tokens, get_cost_per_1k, log_usage

logger = logging.getLogger(__name__)


class LLMClient:
    def __init__(self, provider: Optional[str] = None):
        self.provider = provider or self._select_provider()
        # initial values (will be refreshed at call time)
        self.enabled = os.getenv('LLM_CALLS_ENABLED', 'false').lower() in ('1', 'true', 'yes')
        self.max_calls_per_hour = int(os.getenv('LLM_MAX_CALLS_PER_HOUR', '60'))
        self.quota_file = Path('logs/llm_quota.json')
        self._ensure_quota_file()
        self._clean_old_quota_entries()  # Clean up old entries on init
        try:
            ensure_usage_file()
        except Exception:
            pass

    def _ensure_quota_file(self):
        if not self.quota_file.parent.exists():
            self.quota_file.parent.mkdir(parents=True, exist_ok=True)
        if not self.quota_file.exists():
            self.quota_file.write_text(json.dumps([]))

    def _clean_old_quota_entries(self):
        """Remove quota entries older than 1 hour."""
        try:
            now = time.time()
            raw = self.quota_file.read_text()
            arr = json.loads(raw) if raw.strip() else []
            # keep only last hour
            arr = [t for t in arr if now - t < 3600]
            self.quota_file.write_text(json.dumps(arr))
        except Exception as e:
            logger.warning(f"Failed to clean quota file: {e}")

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
        self._clean_old_quota_entries()  # Ensure old entries are cleaned first
        try:
            now = time.time()
            raw = self.quota_file.read_text()
            arr = json.loads(raw) if raw.strip() else []
            if len(arr) >= self.max_calls_per_hour:
                logger.warning(f"Quota exhausted for {self.provider}: {len(arr)} calls in last hour, max={self.max_calls_per_hour}")
                return False
            arr.append(now)
            self.quota_file.write_text(json.dumps(arr))
            return True
        except Exception as e:
            logger.error(f"Failed to acquire quota: {e}")
            return False

    def _get_api_config(self, provider: str) -> Dict[str, Optional[str]]:
        """Get URL, API key, and default model for a provider with sensible defaults."""
        # Default API URLs for common providers
        defaults = {
            'mistral': {
                'url': 'https://api.mistral.ai/v1/chat/completions',
                'model': os.getenv('MISTRAL_MODEL', 'mistral-tiny-latest')
            },
            'gemini': {
                'url': 'https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash-latest:generateContent',
                'model': os.getenv('GEMINI_MODEL', 'gemini-1.5-flash-latest')
            },
            'groq': {
                'url': 'https://api.groq.ai/v1/chat/completions',
                'model': os.getenv('GROQ_MODEL', 'llama3-8b-8192')
            }
        }
        
        # Get custom URL from env or use default
        custom_url = os.getenv(f'{provider.upper()}_API_URL')
        custom_model = os.getenv(f'{provider.upper()}_MODEL')
        
        config = defaults.get(provider, {'url': None, 'model': None})
        if config:
            config = {
                'url': custom_url or config['url'],
                'model': custom_model or config['model']
            }
        
        # Get keys and strip any whitespace
        mistral_key = (os.getenv('MISTRAL_API_KEY_1') or '').strip() or (os.getenv('MISTRAL_API_KEY_2') or '').strip()
        gemini_key = (os.getenv('GEMINI_API_KEY_1') or '').strip() or (os.getenv('GEMINI_API_KEY_2') or '').strip()
        groq_key = (os.getenv('GROQ_API_KEY_1') or '').strip() or (os.getenv('GROQ_API_KEY_2') or '').strip()
        
        configs = {
            'mistral': {
                'url': config['url'] if config else None,
                'key': mistral_key if mistral_key else None,
                'model': config['model'] if config else None
            },
            'gemini': {
                'url': config['url'] if config else None,
                'key': gemini_key if gemini_key else None,
                'model': config['model'] if config else None
            },
            'groq': {
                'url': config['url'] if config else None,
                'key': groq_key if groq_key else None,
                'model': config['model'] if config else None
            }
        }
        return configs.get(provider, {'url': None, 'key': None, 'model': None})

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

    def _process_response(self, resp, provider: str, prompt_tokens: int, method: str) -> tuple:
        """Process API response and extract text, usage, and log metrics.
        Handles response formats for Mistral, Gemini, and Groq.
        
        Returns tuple of (output_text, api_usage_dict).
        """
        out_text = None
        api_usage = None
        
        if isinstance(resp, dict):
            # Try to extract error first
            if 'error' in resp:
                # This will be caught later for logging
                pass
            
            # Mistral/Groq (OpenAI-compatible) response format
            if 'choices' in resp and isinstance(resp['choices'], list) and resp['choices']:
                first_choice = resp['choices'][0]
                if isinstance(first_choice, dict):
                    message = first_choice.get('message', {})
                    if isinstance(message, dict):
                        out_text = message.get('content') or message.get('text')
                    elif isinstance(message, str):
                        out_text = message
            
            # Gemini response format (generateContent)
            if not out_text and 'candidates' in resp and isinstance(resp.get('candidates'), list):
                candidates = resp['candidates']
                if candidates and isinstance(candidates[0], dict):
                    content = candidates[0].get('content', {})
                    if isinstance(content, dict):
                        parts = content.get('parts', [])
                        if parts and isinstance(parts[0], dict):
                            out_text = parts[0].get('text')
                    elif isinstance(content, str):
                        out_text = content
            
            # Fallback: try standard fields
            if not out_text:
                out_text = resp.get('output') or resp.get('text') or resp.get('message')
            
            if not out_text and 'body' in resp and isinstance(resp['body'], str):
                out_text = resp['body']

            # extract usage if provider returns it
            usage = resp.get('usage') or resp.get('token_usage') or (resp.get('meta', {}) or {}).get('usage')
            if isinstance(usage, dict):
                api_prompt = usage.get('prompt_tokens') or usage.get('input_tokens') or usage.get('prompt')
                api_completion = usage.get('completion_tokens') or usage.get('output_tokens') or usage.get('completion')
                api_usage = {'prompt_tokens': api_prompt, 'completion_tokens': api_completion, 'raw': usage}
            
            # Also check for usage in response (some APIs put it at root level)
            if not api_usage:
                if 'promptTokenCount' in resp or 'inputTokenCount' in resp:
                    api_usage = {
                        'prompt_tokens': resp.get('promptTokenCount') or resp.get('inputTokenCount'),
                        'completion_tokens': resp.get('candidatesTokenCount') or resp.get('outputTokenCount'),
                        'raw': {'source': 'root_level'}
                    }

        else:
            out_text = str(resp)

        prompt_tokens_to_log = int(api_usage['prompt_tokens']) if api_usage and api_usage.get('prompt_tokens') is not None else prompt_tokens
        completion_tokens = int(api_usage['completion_tokens']) if api_usage and api_usage.get('completion_tokens') is not None else (estimate_tokens(out_text) if out_text else 0)
        cost = (prompt_tokens_to_log + completion_tokens) / 1000.0 * get_cost_per_1k(provider)
        success = not (isinstance(resp, dict) and 'error' in resp)
        extra = {'success': success}
        if isinstance(resp, dict) and 'error' in resp:
            extra['error'] = resp.get('error')
            extra['error_type'] = resp.get('error_type') or resp.get('code')
        if api_usage is not None:
            extra['api_usage'] = api_usage
        try:
            log_usage(provider, method, prompt_tokens_to_log, completion_tokens, cost, extra)
        except Exception:
            pass
        
        return out_text or str(resp), extra

    def _get_placeholder(self, provider: str, text: str, method: str, reason: str) -> str:
        """Generate descriptive placeholder when API call is not possible."""
        provider_name = provider or 'unknown'
        text_preview = text[:100].replace('\n', ' ') if text else ''
        
        if reason == 'disabled':
            return f"[LLM {method} by {provider_name} - DISABLED: Real API calls not enabled. Set LLM_CALLS_ENABLED=true. Input preview: {text_preview}...]"
        elif reason == 'quota':
            return f"[LLM {method} by {provider_name} - QUOTA EXHAUSTED: Max {self.max_calls_per_hour} calls/hour reached. Input preview: {text_preview}...]"
        elif reason == 'no_url':
            return f"[LLM {method} by {provider_name} - CONFIG ERROR: API URL not configured. Set {provider_name.upper()}_API_URL. Input preview: {text_preview}...]"
        elif reason == 'no_key':
            return f"[LLM {method} by {provider_name} - CONFIG ERROR: API key not configured. Set {provider_name.upper()}_API_KEY_1. Input preview: {text_preview}...]"
        elif reason == 'no_provider':
            return f"[LLM {method} - NO PROVIDER: No LLM provider configured. Input preview: {text_preview}...]"
        else:
            return f"[LLM {method} by {provider_name} - ERROR: {reason}. Input preview: {text_preview}...]"

    def _call_provider_api(self, provider: str, text: str, method: str, max_tokens: int = 256, temperature: float = 0.7) -> str:
        """Call a specific provider's API with proper payload formatting."""
        config = self._get_api_config(provider)
        url = config.get('url')
        key = config.get('key')
        model = config.get('model')
        
        if not url:
            logger.warning(f"No API URL configured for {provider}")
            return self._get_placeholder(provider, text, method, 'no_url')
        
        if not key:
            logger.warning(f"No API key configured for {provider}")
            return self._get_placeholder(provider, text, method, 'no_key')
        
        headers = {'Authorization': f'Bearer {key}', 'Content-Type': 'application/json'}
        
        # Build provider-specific payload
        if provider == 'gemini':
            # Gemini uses generateContent API (v1beta)
            # Note: For generateMessage (older), use different structure
            if 'generateContent' in url:
                payload = {
                    'contents': [{'parts': [{'text': text}]}],
                    'generationConfig': {
                        'maxOutputTokens': max_tokens,
                        'temperature': temperature
                    }
                }
            else:
                # Fallback for generateMessage
                payload = {
                    'prompt': text,
                    'max_tokens': max_tokens
                }
                headers['x-goog-api-key'] = key
                headers.pop('Authorization', None)  # Gemini uses different auth
        elif provider in ['mistral', 'groq']:
            # Mistral and Groq use OpenAI-compatible chat completions API
            payload = {
                'model': model or 'default',
                'messages': [
                    {'role': 'user', 'content': text}
                ],
                'max_tokens': max_tokens,
                'temperature': temperature,
                'top_p': 0.9
            }
        else:
            # Default fallback
            payload = {'input': text, 'max_tokens': max_tokens}
        
        try:
            resp = self._http_post(url, headers, payload)
            out_text, extra = self._process_response(resp, provider, estimate_tokens(text), method)
            if 'error' in extra:
                logger.error(f"API call to {provider} failed: {extra.get('error')}")
                return self._get_placeholder(provider, text, method, str(extra.get('error', 'api_error')))
            return out_text
        except Exception as e:
            logger.error(f"Exception calling {provider} API: {e}")
            return self._get_placeholder(provider, text, method, str(e))

    def summarize(self, text: str, max_tokens: int = 256, force_real: Optional[bool] = None) -> str:
        """Summarize text using the preferred provider if enabled and quota allows.
        Falls back to a deterministic placeholder with clear error message.
        This implementation records estimated token usage and approximate cost.
        
        Args:
            text: Text to summarize
            max_tokens: Maximum tokens for response
            force_real: Override enabled state. If None, uses self.enabled. If False, forces disabled.
        """
        # Only refresh from env if force_real is not explicitly set
        if force_real is None:
            self.enabled = os.getenv('LLM_CALLS_ENABLED', 'false').lower() in ('1', 'true', 'yes')
            self.max_calls_per_hour = int(os.getenv('LLM_MAX_CALLS_PER_HOUR', '60'))
        else:
            # force_real overrides the enabled state for this call
            self.enabled = bool(force_real)

        provider = self.provider or 'none'
        s = text.strip().replace('\n', ' ')
        prompt_tokens = estimate_tokens(text)
        
        method = 'summarize'

        if not self.enabled:
            try:
                log_usage(provider, method, prompt_tokens, 0, 0.0, {'skipped': 'disabled'})
            except Exception:
                pass
            return self._get_placeholder(provider, s, method, 'disabled')

        if not self._acquire_quota():
            try:
                log_usage(provider, method, prompt_tokens, 0, 0.0, {'skipped': 'quota'})
            except Exception:
                pass
            return self._get_placeholder(provider, s, method, 'quota')

        # Try providers in order of preference for summarization: mistral, gemini, groq
        preferred_providers = ['mistral', 'gemini', 'groq']
        available = self.available_providers()
        
        for prov in preferred_providers:
            if prov in available:
                result = self._call_provider_api(prov, s, method, max_tokens)
                if not result.startswith('['):  # If we got a real response (not a placeholder)
                    return result
        
        # If we tried all providers and got placeholders, return the first one
        if preferred_providers:
            return self._call_provider_api(preferred_providers[0], s, method, max_tokens)
        
        try:
            log_usage(provider, method, prompt_tokens, 0, 0.0, {'skipped': 'no_provider'})
        except Exception:
            pass
        return self._get_placeholder(provider, s, method, 'no_provider')

    def generate_report_text(self, df_summary: dict, max_tokens: int = 512, force_real: Optional[bool] = None) -> str:
        """Ask an LLM to craft a narrative report from a dict of summary statistics.
        Prefer Gemini for narrative generation (per routing decision).
        Records estimated token usage and estimated cost.
        
        Args:
            df_summary: Dict of summary statistics
            max_tokens: Maximum tokens for response
            force_real: Override enabled state. If None, uses self.enabled. If False, forces disabled.
        """
        # Only refresh from env if force_real is not explicitly set
        if force_real is None:
            self.enabled = os.getenv('LLM_CALLS_ENABLED', 'false').lower() in ('1', 'true', 'yes')
            self.max_calls_per_hour = int(os.getenv('LLM_MAX_CALLS_PER_HOUR', '60'))
        else:
            # force_real overrides the enabled state for this call
            self.enabled = bool(force_real)

        text = "; ".join(f"{k}: {v}" for k, v in df_summary.items()) if isinstance(df_summary, dict) else str(df_summary)
        prompt_tokens = estimate_tokens(text)
        
        method = 'generate_report_text'
        provider = self.provider or 'none'

        if not self.enabled:
            try:
                log_usage(provider, method, prompt_tokens, 0, 0.0, {'skipped': 'disabled'})
            except Exception:
                pass
            return self._get_placeholder(provider, text, method, 'disabled')

        if not self._acquire_quota():
            try:
                log_usage(provider, method, prompt_tokens, 0, 0.0, {'skipped': 'quota'})
            except Exception:
                pass
            return self._get_placeholder(provider, text, method, 'quota')

        # Try providers in order of preference for narrative: gemini, mistral, groq
        preferred_providers = ['gemini', 'mistral', 'groq']
        available = self.available_providers()
        
        for prov in preferred_providers:
            if prov in available:
                result = self._call_provider_api(prov, text, method, max_tokens)
                if not result.startswith('['):  # If we got a real response
                    return result
        
        # If we tried all providers and got placeholders, return the first one
        if preferred_providers:
            return self._call_provider_api(preferred_providers[0], text, method, max_tokens)
        
        try:
            log_usage(provider, method, prompt_tokens, 0, 0.0, {'skipped': 'no_provider'})
        except Exception:
            pass
        return self._get_placeholder(provider, text, method, 'no_provider')
