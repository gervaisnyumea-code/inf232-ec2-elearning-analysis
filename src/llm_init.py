"""LLM initialization - MUST be imported FIRST to load .env before any LLM module.

This module ensures environment variables are loaded BEFORE any LLMClient
is instantiated. Import this at the very start of any script that uses LLMs.

Usage:
    from src.llm_init import ensure_env_loaded
    # OR simply: import src.llm_init (it auto-loads on import)
    
IMPORTANT: This module auto-loads .env on import. Just adding 'import src.llm_init'
at the top of your script is sufficient.
"""
import os
import sys
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

# Flag to prevent double-loading
_ENV_LOADED = False
_ENV_LOADER_CALLED = False


def ensure_env_loaded(env_path: str = None, override: bool = True) -> bool:
    """Load .env file into os.environ BEFORE any LLMClient is instantiated.
    
    This MUST be called before any LLM module is imported, or as early
    as possible in the application lifecycle.
    
    Args:
        env_path: Path to .env file (default: project root .env)
        override: Whether to override existing os.environ values
        
    Returns:
        bool: True if .env was loaded successfully
    """
    global _ENV_LOADED, _ENV_LOADER_CALLED
    
    if _ENV_LOADER_CALLED:
        return _ENV_LOADED
    
    _ENV_LOADER_CALLED = True
    
    if _ENV_LOADED:
        logger.debug(".env already loaded, skipping...")
        return True
        
    # Find project root
    if env_path is None:
        # Try multiple locations: cwd, script location, parent of script location
        candidates = [
            Path.cwd() / '.env',
            Path(__file__).parent.parent / '.env',
        ]
        # Add parent of current script's parent
        script_parent = Path(__file__).resolve().parent
        if str(script_parent) != str(script_parent.parent):
            candidates.append(script_parent.parent / '.env')
        
        for candidate in candidates:
            if candidate.exists():
                env_path = str(candidate)
                break
    
    if env_path is None or not Path(env_path).exists():
        logger.warning(f".env file not found at {env_path}")
        return False
    
    try:
        # Use env_loader if available
        try:
            from src.env_loader import load_dotenv
            loaded = load_dotenv(env_path, override=override)
        except ImportError:
            # Fallback: manual parse
            loaded = _manual_load_dotenv(env_path, override)
        
        if loaded:
            _ENV_LOADED = True
            # Log what was loaded
            providers = []
            for key in ['MISTRAL_API_KEY_1', 'GEMINI_API_KEY_1', 'GROQ_API_KEY_1']:
                val = os.getenv(key, '')
                if val:
                    providers.append(key.replace('_API_KEY_1', '').lower())
                    # Log key (first 4 chars for debugging)
                    logger.debug(f"Loaded {key}: {val[:4]}...")
            
            logger.info(f"✅ .env loaded. Providers: {providers}")
            return True
    except Exception as e:
        logger.error(f"Failed to load .env: {e}")
        return False
    
    return False


def _manual_load_dotenv(env_path: str, override: bool = True) -> dict:
    """Manual .env parser as fallback."""
    loaded = {}
    path = Path(env_path)
    
    if not path.exists():
        return loaded
        
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith('#') or '=' not in line:
            continue
        key, val = line.split('=', 1)
        key = key.strip()
        val = val.strip().strip('"').strip("'")
        if override or key not in os.environ:
            os.environ[key] = val
            loaded[key] = val
    return loaded


# Auto-load on import (THIS IS THE KEY FIX)
# This runs immediately when the module is imported
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_ENV_FILE = _PROJECT_ROOT / '.env'

def _auto_load():
    """Internal auto-load function called on module import."""
    global _ENV_LOADED
    if _ENV_LOADED:
        return True
    
    if _ENV_FILE.exists():
        try:
            ensure_env_loaded(str(_ENV_FILE), override=True)
            return _ENV_LOADED
        except Exception as e:
            print(f"Warning: failed to auto-load .env: {e}")
    return _ENV_LOADED

# AUTO-LOAD NOW when module is imported
_auto_load()


# Export a convenient function for direct use
def get_providers():
    """Return list of available LLM providers from loaded .env."""
    providers = []
    for key in ['MISTRAL_API_KEY_1', 'GEMINI_API_KEY_1', 'GROQ_API_KEY_1']:
        if os.getenv(key):
            providers.append(key.replace('_API_KEY_1', '').lower())
    return providers


# For convenience: allow explicit loading
def force_reload(env_path: str = None, override: bool = True) -> bool:
    """Force reload .env from specified path."""
    global _ENV_LOADED, _ENV_LOADER_CALLED
    _ENV_LOADED = False
    _ENV_LOADER_CALLED = False
    return ensure_env_loaded(env_path, override=override)