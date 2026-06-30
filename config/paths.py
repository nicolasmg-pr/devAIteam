"""Centralized filesystem path resolution and sandbox validation for devAIteam.

This module replaces the hardcoded absolute paths that were scattered across the
codebase (e.g. ``/Users/nikomendez/Documents/SWdevAIgency_project``). The project
root is derived from this file's location so the platform is portable across
machines and CI, and can still be overridden via the ``DEVAITEAM_ROOT`` env var.

It also provides the path-safety primitives used everywhere untrusted (LLM- or
user-supplied) path components are turned into real filesystem paths.
"""

from __future__ import annotations

import os
import re
from pathlib import Path


class UnsafePathError(ValueError):
    """Raised when a name or path component would escape its intended sandbox."""


def _resolve_root() -> Path:
    env = os.getenv("DEVAITEAM_ROOT")
    if env:
        return Path(env).expanduser().resolve()
    # This file lives at <root>/config/paths.py
    return Path(__file__).resolve().parents[1]


# devAIteam repository root.
PROJECT_ROOT: Path = _resolve_root()

# Directory under which all generated projects are written.
OUTPUT_DIR: Path = Path(
    os.getenv("DEVAITEAM_OUTPUT_DIR", str(PROJECT_ROOT / "output"))
).expanduser().resolve()

# A project name must be a simple slug: starts alphanumeric, then [A-Za-z0-9_-].
_SLUG_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]*$")


def validate_project_name(name: str) -> str:
    """Return a validated, stripped project name or raise :class:`UnsafePathError`.

    Rejects empties, path separators, ``..``, leading dashes (which deploy CLIs
    would parse as flags) and anything that is not a simple slug. Call this before
    interpolating a project name into any filesystem path or subprocess argument.
    """
    if not isinstance(name, str) or not name.strip():
        raise UnsafePathError(f"Invalid project name: {name!r}")
    candidate = name.strip()
    if "/" in candidate or "\\" in candidate or os.sep in candidate or (
        os.altsep and os.altsep in candidate
    ):
        raise UnsafePathError(f"Project name must not contain path separators: {name!r}")
    if ".." in candidate:
        raise UnsafePathError(f"Project name must not contain '..': {name!r}")
    if not _SLUG_RE.match(candidate):
        raise UnsafePathError(
            f"Project name must match [A-Za-z0-9_-] and not start with '-': {name!r}"
        )
    return candidate


def ensure_output_dir() -> Path:
    """Create and return the output directory."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    return OUTPUT_DIR


def project_dir(project_name: str) -> Path:
    """Absolute, validated path to a generated project's directory under output/."""
    name = validate_project_name(project_name)
    return OUTPUT_DIR / name


def safe_join(base: Path | str, *parts: str) -> Path:
    """Join untrusted ``parts`` onto ``base`` and assert the result stays inside it.

    Raises :class:`UnsafePathError` if any component is absolute or if the resolved
    path escapes ``base`` (e.g. via ``..``). Use this for every path built from an
    LLM- or user-supplied component before reading or writing.
    """
    base_path = Path(base).resolve()
    candidate = base_path
    for part in parts:
        if part is None or part == "":
            continue
        p = Path(part)
        if p.is_absolute():
            raise UnsafePathError(f"Absolute path component not allowed: {part!r}")
        candidate = candidate / p
    resolved = candidate.resolve()
    if resolved != base_path and base_path not in resolved.parents:
        raise UnsafePathError(f"Path escapes sandbox {base_path}: {resolved}")
    return resolved
