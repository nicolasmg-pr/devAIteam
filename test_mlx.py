"""Manual connectivity probe for the configured LLM endpoint.

This is an integration smoke check, NOT a unit test — run it directly
(`python test_mlx.py`) when you want to confirm the local MLX/Ollama server (or
cloud OpenAI) is reachable. It is intentionally skipped by pytest (the real unit
tests live under ``tests/``) and degrades to a clean message if no server is up.
"""

from __future__ import annotations

import json
import sys


def main() -> int:
    from langchain_core.messages import HumanMessage, SystemMessage

    from config.llm_config import get_llm
    from tools.llm_helpers import clean_llm_response

    print("🔌 Test 1: Basic connectivity")
    try:
        llm = get_llm(temperature=0.7, thinking=False)
        r = llm.invoke([HumanMessage(content="Reply exactly: MLX OK")])
    except Exception as e:  # noqa: BLE001 - this is a connectivity probe
        print(f"   ⚠️  No LLM endpoint reachable — skipping probe ({e})")
        return 0
    print(f"   ✅ Response: {r.content.strip()}")

    print("\n🔌 Test 2: Clean JSON response")
    r = llm.invoke([
        SystemMessage(content="Reply ONLY with valid JSON, no extra text."),
        HumanMessage(content='Return: {"status": "ok", "model": "qwen3.6"}'),
    ])
    parsed = json.loads(clean_llm_response(r.content))
    print(f"   ✅ Parsed JSON: {parsed}")

    print("\n🔌 Test 3: Thinking mode")
    llm_think = get_llm(temperature=0.3, thinking=True)
    r = llm_think.invoke([HumanMessage(content="Why is the sky blue? Answer in 2 sentences.")])
    has_think = "<think>" in (r.content or "")
    print(f"   {'✅' if has_think else 'ℹ️ '} <think> blocks: {'present' if has_think else 'not detected (may be normal)'}")
    print(f"   Response (first 200 chars): {r.content[:200]}")

    print("\n✅ Probe complete — endpoint ready for the pipeline")
    return 0


if __name__ == "__main__":
    sys.exit(main())
