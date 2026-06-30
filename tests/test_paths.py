"""Unit tests for path validation and sandboxing."""

import pytest

from config.paths import UnsafePathError, safe_join, validate_project_name


@pytest.mark.parametrize("name", ["myapp", "my-app", "app_2", "Recipe123"])
def test_valid_names(name):
    assert validate_project_name(name) == name


@pytest.mark.parametrize(
    "name",
    ["../etc", "a/b", "a\\b", "..", "-flag", "", "   ", "a/../b"],
)
def test_invalid_names_rejected(name):
    with pytest.raises(UnsafePathError):
        validate_project_name(name)


def test_safe_join_ok(tmp_path):
    result = safe_join(tmp_path, "sub", "file.txt")
    assert str(result).startswith(str(tmp_path.resolve()))
    assert result.name == "file.txt"


def test_safe_join_rejects_traversal(tmp_path):
    with pytest.raises(UnsafePathError):
        safe_join(tmp_path, "..", "..", "etc", "passwd")


def test_safe_join_rejects_absolute(tmp_path):
    with pytest.raises(UnsafePathError):
        safe_join(tmp_path, "/etc/passwd")


def test_safe_join_skips_empty_parts(tmp_path):
    result = safe_join(tmp_path, "", "a")
    assert result.name == "a"
