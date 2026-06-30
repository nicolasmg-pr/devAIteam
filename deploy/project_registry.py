"""Central registry of generated projects, persisted to ``output/.registry.json``.

All writes go through a single atomic helper (temp file + ``os.replace``) guarded
by a cross-process file lock, so an interruption or a concurrent invocation can no
longer truncate or clobber the index. A registry file that fails to parse is
backed up (rather than silently overwritten) before we fall back to an empty list.
"""

from __future__ import annotations

import contextlib
import json
import os
import tempfile
from typing import List, Optional

from pydantic import BaseModel, field_validator

from config.paths import OUTPUT_DIR, ensure_output_dir, validate_project_name

try:  # POSIX file locking; degrade to a no-op lock elsewhere.
    import fcntl
except Exception:  # pragma: no cover - non-POSIX
    fcntl = None  # type: ignore[assignment]


class ProjectMeta(BaseModel):
    project_name: str
    created_at: str
    requirement: str
    tech_stack: str
    files_count: int
    quality_score: Optional[int] = None
    deploy_status: str = "not_deployed"
    deploy_url: Optional[str] = None
    deploy_platform: Optional[str] = None
    local_preview_available: bool = False
    github_pr_url: Optional[str] = None
    output_size_mb: float = 0.0

    @field_validator("project_name")
    @classmethod
    def _validate_project_name(cls, v: str) -> str:
        return validate_project_name(v)


REGISTRY_PATH = str(OUTPUT_DIR / ".registry.json")
_LOCK_PATH = str(OUTPUT_DIR / ".registry.lock")


def _ensure_output_dir() -> None:
    ensure_output_dir()


@contextlib.contextmanager
def _registry_lock():
    """Best-effort exclusive lock around a read-modify-write of the registry."""
    _ensure_output_dir()
    if fcntl is None:
        yield
        return
    lock_file = open(_LOCK_PATH, "w")
    try:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        yield
    finally:
        with contextlib.suppress(Exception):
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
        lock_file.close()


def _atomic_write_registry(registry: List[ProjectMeta]) -> None:
    """Serialize and atomically replace the registry file."""
    _ensure_output_dir()
    payload = json.dumps(
        [p.model_dump() for p in registry], indent=2, ensure_ascii=False
    )
    directory = os.path.dirname(REGISTRY_PATH) or "."
    fd, tmp_path = tempfile.mkstemp(dir=directory, prefix=".registry.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(payload)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, REGISTRY_PATH)
    except Exception:
        with contextlib.suppress(FileNotFoundError):
            os.remove(tmp_path)
        raise


def write_registry(registry: List[ProjectMeta]) -> None:
    """Public atomic registry writer used by all mutators (rm, prune, etc.)."""
    with _registry_lock():
        _atomic_write_registry(registry)


def load_registry() -> List[ProjectMeta]:
    _ensure_output_dir()
    if not os.path.exists(REGISTRY_PATH):
        return []
    try:
        with open(REGISTRY_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        # Preserve the corrupt file instead of letting the next save destroy it.
        backup = REGISTRY_PATH + ".corrupt"
        with contextlib.suppress(Exception):
            os.replace(REGISTRY_PATH, backup)
        print(f"⚠️ Corrupt project registry backed up to {backup}: {e}")
        return []
    except Exception as e:  # noqa: BLE001
        print(f"⚠️ Error loading project registry: {e}")
        return []

    if not isinstance(data, list):
        print("⚠️ Project registry is not a list; ignoring its contents.")
        return []
    result: List[ProjectMeta] = []
    for p in data:
        try:
            result.append(ProjectMeta.model_validate(p))
        except Exception as e:  # noqa: BLE001 - skip individual malformed entries
            print(f"⚠️ Skipping malformed registry entry: {e}")
    return result


def save_project(meta: ProjectMeta) -> None:
    with _registry_lock():
        registry = load_registry()
        updated = False
        for i, p in enumerate(registry):
            if p.project_name == meta.project_name:
                registry[i] = meta
                updated = True
                break
        if not updated:
            registry.append(meta)
        try:
            _atomic_write_registry(registry)
        except Exception as e:  # noqa: BLE001
            print(f"❌ Error saving project registry: {e}")


def remove_project(project_name: str) -> bool:
    """Remove a project from the registry atomically. Returns True if removed."""
    with _registry_lock():
        registry = load_registry()
        new_registry = [p for p in registry if p.project_name != project_name]
        removed = len(new_registry) != len(registry)
        if removed:
            _atomic_write_registry(new_registry)
        return removed


def get_project(project_name: str) -> Optional[ProjectMeta]:
    registry = load_registry()
    for p in registry:
        if p.project_name == project_name:
            return p
    return None


def update_deploy_status(project_name: str, status: str, url: str, platform: str) -> None:
    meta = get_project(project_name)
    if meta:
        meta.deploy_status = status
        meta.deploy_url = url
        meta.deploy_platform = platform
        save_project(meta)
    else:
        print(f"⚠️ Project '{project_name}' not found in registry to update deploy.")
