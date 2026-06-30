"""Helpers for turning raw LLM output into validated JSON.

The local Qwen model emits ``<think>`` reasoning blocks and frequently wraps its
JSON in prose or markdown fences. These helpers extract the first complete JSON
value with a quote/escape-aware brace scan (so braces inside string values no
longer corrupt the boundaries), repair near-miss JSON via ``json-repair``, and
provide a single retry-with-correction loop shared by the simple agents.
"""

from __future__ import annotations

import json
import re
from typing import Any, Callable

try:  # json-repair is a declared dependency; degrade gracefully if absent.
    from json_repair import repair_json
except Exception:  # pragma: no cover - defensive
    repair_json = None  # type: ignore[assignment]


_THINK_BLOCK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)


def _first_json_delim(s: str) -> int:
    """Index of the first ``{`` or ``[`` in ``s``, or -1 if neither is present."""
    candidates = [i for i in (s.find("{"), s.find("[")) if i != -1]
    return min(candidates) if candidates else -1


def _strip_think(raw: str) -> str:
    """Remove ``<think>...</think>`` blocks, only outside the JSON payload.

    Closed blocks are removed everywhere. An *unclosed* ``<think>`` is only
    stripped from the preamble before the first JSON delimiter, so a literal
    ``<think>`` appearing inside a JSON string value is preserved.
    """
    cleaned = _THINK_BLOCK_RE.sub("", raw)
    if "<think>" in cleaned:
        delim = _first_json_delim(cleaned)
        if delim == -1:
            cleaned = re.sub(r"<think>.*$", "", cleaned, flags=re.DOTALL)
        else:
            head = re.sub(r"<think>.*$", "", cleaned[:delim], flags=re.DOTALL)
            cleaned = head + cleaned[delim:]
    return cleaned


def _extract_balanced(s: str) -> str | None:
    """Return the first complete JSON value using a quote/escape-aware scan.

    Tracks string state so braces/brackets inside string literals do not affect
    nesting depth. Returns ``None`` if no balanced value is found.
    """
    start = _first_json_delim(s)
    if start == -1:
        return None
    open_ch = s[start]
    close_ch = "}" if open_ch == "{" else "]"
    depth = 0
    in_str = False
    escaped = False
    for i in range(start, len(s)):
        c = s[i]
        if in_str:
            if escaped:
                escaped = False
            elif c == "\\":
                escaped = True
            elif c == '"':
                in_str = False
            continue
        if c == '"':
            in_str = True
        elif c == open_ch:
            depth += 1
        elif c == close_ch:
            depth -= 1
            if depth == 0:
                return s[start : i + 1]
    return None  # unbalanced / truncated


def clean_llm_response(raw: str) -> str:
    """Clean an LLM response down to (ideally) a single JSON value as text.

    Strips ``<think>`` blocks, then extracts the first balanced JSON object/array.
    Falls back to stripping markdown fences when no balanced value is found.
    """
    cleaned = _strip_think(raw).strip()

    extracted = _extract_balanced(cleaned)
    if extracted is not None:
        return extracted.strip()

    # Fallback: strip ``` fences and return whatever remains.
    if cleaned.startswith("```"):
        lines = cleaned.split("\n")
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        cleaned = "\n".join(lines)
    return cleaned.strip()


def parse_llm_json(raw: str) -> Any:
    """Clean and parse an LLM response into a Python object.

    Attempts a strict parse first, then falls back to ``json-repair`` when
    available. Raises ``json.JSONDecodeError`` (or ``ValueError``) if the text
    cannot be parsed at all.
    """
    text = clean_llm_response(raw)
    try:
        return json.loads(text, strict=False)
    except json.JSONDecodeError:
        if repair_json is not None:
            repaired = repair_json(text)
            return json.loads(repaired, strict=False)
        raise


RETRY_PROMPT = (
    "Your previous response could not be parsed as valid JSON. "
    "Respond ONLY with a single valid JSON value matching the requested schema. "
    "Do NOT include any prose, markdown code fences, or <think> blocks."
)


def invoke_with_retry(
    llm,
    messages: list,
    parse_fn: Callable[[str], Any],
    *,
    max_attempts: int = 2,
):
    """Invoke ``llm`` and parse the response with ``parse_fn``, retrying on failure.

    ``parse_fn`` receives the raw string content and returns the parsed result; it
    may raise on bad output, which triggers a corrective retry (up to
    ``max_attempts`` total). Re-raises the last exception if every attempt fails so
    the caller can surface a structured failure instead of crashing mid-parse.
    """
    from langchain_core.messages import HumanMessage

    last_exc: Exception | None = None
    for attempt in range(max_attempts):
        convo = list(messages)
        if attempt > 0 and last_exc is not None:
            convo.append(HumanMessage(content=f"{RETRY_PROMPT}\n\nParser error: {last_exc}"))
        response = llm.invoke(convo)
        raw = response.content if isinstance(response.content, str) else str(response.content)
        try:
            return parse_fn(raw)
        except Exception as exc:  # noqa: BLE001 - parse/validation failures are expected
            last_exc = exc
    assert last_exc is not None
    raise last_exc
