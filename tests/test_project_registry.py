"""Unit tests for the project registry: atomicity, validation, corruption handling."""

import importlib

import pytest


@pytest.fixture()
def registry(tmp_path, monkeypatch):
    """Reload project_registry with OUTPUT_DIR pointed at a temp dir."""
    monkeypatch.setenv("DEVAITEAM_OUTPUT_DIR", str(tmp_path))
    import config.paths as paths

    importlib.reload(paths)
    import deploy.project_registry as reg

    importlib.reload(reg)
    yield reg
    # Restore module state for other tests.
    monkeypatch.delenv("DEVAITEAM_OUTPUT_DIR", raising=False)
    importlib.reload(paths)
    importlib.reload(reg)


def _meta(reg, name="myapp"):
    return reg.ProjectMeta(
        project_name=name,
        created_at="2026-01-01T00:00:00",
        requirement="build a thing",
        tech_stack="NestJS + Flutter + Postgres",
        files_count=3,
    )


def test_save_and_get(registry):
    registry.save_project(_meta(registry))
    got = registry.get_project("myapp")
    assert got is not None and got.files_count == 3


def test_save_is_idempotent_update(registry):
    registry.save_project(_meta(registry))
    m = _meta(registry)
    m.files_count = 99
    registry.save_project(m)
    assert len(registry.load_registry()) == 1
    assert registry.get_project("myapp").files_count == 99


def test_remove_project(registry):
    registry.save_project(_meta(registry))
    assert registry.remove_project("myapp") is True
    assert registry.get_project("myapp") is None
    assert registry.remove_project("myapp") is False


def test_invalid_project_name_rejected(registry):
    from config.paths import UnsafePathError

    with pytest.raises((UnsafePathError, ValueError)):
        _meta(registry, name="../evil")


def test_corrupt_registry_is_backed_up_not_destroyed(registry, tmp_path):
    path = tmp_path / ".registry.json"
    path.write_text("{ this is : not json", encoding="utf-8")
    assert registry.load_registry() == []
    # The corrupt file must be preserved, not silently dropped.
    assert (tmp_path / ".registry.json.corrupt").exists()


def test_non_list_registry_ignored(registry, tmp_path):
    (tmp_path / ".registry.json").write_text('{"not": "a list"}', encoding="utf-8")
    assert registry.load_registry() == []


def test_atomic_write_leaves_no_tmp(registry, tmp_path):
    registry.save_project(_meta(registry))
    leftovers = list(tmp_path.glob(".registry.*.tmp"))
    assert leftovers == []
