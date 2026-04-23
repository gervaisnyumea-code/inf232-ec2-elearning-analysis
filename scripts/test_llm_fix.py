#!/usr/bin/env python3
"""Test script to verify LLM integration works correctly.

Run this to verify the .env loading fix:
    python scripts/test_llm_fix.py
"""
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# CRITICAL: Import llm_init FIRST - this is the fix!
print("=" * 60)
print("TESTING LLM INTEGRATION FIX")
print("=" * 60)

# Step 1: Verify .env is loaded BEFORE any LLMClient
print("\n[1] Importing llm_init (loads .env before LLMClient)...")
from src.llm_init import ensure_env_loaded, _ENV_LOADED

print(f"    _ENV_LOADED flag: {_ENV_LOADED}")
print(f"    MISTRAL_API_KEY_1 set: {bool(os.getenv('MISTRAL_API_KEY_1'))}")
print(f"    GEMINI_API_KEY_1 set: {bool(os.getenv('GEMINI_API_KEY_1'))}")
print(f"    GROQ_API_KEY_1 set: {bool(os.getenv('GROQ_API_KEY_1'))}")
print(f"    LLM_CALLS_ENABLED: {os.getenv('LLM_CALLS_ENABLED')}")

# Step 2: Test LLMClient discovery
print("\n[2] Testing LLMClient provider discovery...")
from src.llm_integration import LLMClient

client = LLMClient()
available = client.available_providers()
print(f"    Available providers: {available}")

# Step 3: Test orchestrator
print("\n[3] Testing LLMOrchestrator...")
from src.llm_orchestrator import LLMOrchestrator

orch = LLMOrchestrator()
providers = orch.available_providers()
print(f"    Orchestrator providers: {providers}")

# Step 4: Test actual LLM call (small test)
print("\n[4] Testing actual LLM call (summarize)...")
if providers and 'mistral' in providers:
    result = orch.summarize_report("Test: Can you hear me? Reply with YES.", max_tokens=50)
    print(f"    Result: {result.get('mistral', 'NO RESPONSE')[:200]}...")
else:
    print("    SKIPPED: No providers available")

# Step 5: Summary
print("\n" + "=" * 60)
print("TEST RESULTS")
print("=" * 60)

errors = []
warnings = []

if not os.getenv('MISTRAL_API_KEY_1'):
    warnings.append("MISTRAL_API_KEY_1 not set in .env")
if not os.getenv('GEMINI_API_KEY_1'):
    warnings.append("GEMINI_API_KEY_1 not set in .env") 
if not os.getenv('GROQ_API_KEY_1'):
    warnings.append("GROQ_API_KEY_1 not set in .env")

if not available:
    errors.append("No LLM providers discovered")
    
if 'mistral' not in available and 'gemini' not in available and 'groq' not in available:
    errors.append("All LLM calls will return placeholders")

if errors:
    print("\n❌ ERRORS:")
    for e in errors:
        print(f"   - {e}")
        
if warnings:
    print("\n⚠️  WARNINGS:")
    for w in warnings:
        print(f"   - {w}")

if not errors and not warnings:
    print("\n✅ ALL TESTS PASSED!")
    print("   .env loaded correctly")
    print("   Providers discovered")
    print("   Ready for LLM calls")
else:
    print("\n" + "=" * 60)
    print("FIX INSTRUCTIONS:")
    print("=" * 60)
    if warnings:
        print("\n1. Get valid API keys from:")
        print("   - Mistral: https://console.mistral.ai/")
        print("   - Gemini: https://aistudio.google.com/app/apikey")
        print("   - Groq: https://console.groq.com/")
        print("\n2. Update .env with valid keys")
        print("3. Run this test again")