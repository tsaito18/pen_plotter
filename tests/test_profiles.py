from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.collector.profiles import (
    list_profiles,
    migrate_legacy_root,
    resolve_training_dirs,
    validate_profile_id,
)


def _write_sample(root: Path, char: str = "あ") -> None:
    char_dir = root / char
    char_dir.mkdir(parents=True, exist_ok=True)
    data = {
        "character": char,
        "strokes": [[{"x": 0.0, "y": 0.0}, {"x": 1.0, "y": 1.0}]],
        "metadata": {},
    }
    (char_dir / f"{char}_1.json").write_text(json.dumps(data), encoding="utf-8")


def test_list_profiles_ignores_default(tmp_path: Path) -> None:
    _write_sample(tmp_path / "taiga", "あ")
    _write_sample(tmp_path / "default", "い")
    (tmp_path / "empty").mkdir()

    profiles = list_profiles(tmp_path)

    assert [p.id for p in profiles] == ["empty", "taiga"]
    taiga = profiles[1]
    assert taiga.sample_count == 1
    assert taiga.character_count == 1


def test_migrate_legacy_root_moves_character_dirs_to_taiga(tmp_path: Path) -> None:
    _write_sample(tmp_path, "あ")
    _write_sample(tmp_path, "い")

    moved = migrate_legacy_root(tmp_path, "taiga")

    assert sorted(p.name for p in moved) == ["あ", "い"]
    assert not (tmp_path / "あ").exists()
    assert (tmp_path / "taiga" / "あ" / "あ_1.json").exists()


def test_resolve_training_dirs_supports_current_profiles_and_all(tmp_path: Path) -> None:
    _write_sample(tmp_path / "taiga", "あ")
    _write_sample(tmp_path / "sato", "い")

    current = resolve_training_dirs(tmp_path, {"mode": "current", "profile": "taiga"})
    all_dirs = resolve_training_dirs(tmp_path, {"mode": "all"})
    selected = resolve_training_dirs(tmp_path, {"mode": "profiles", "profiles": ["sato"]})

    assert current == [tmp_path / "taiga"]
    assert all_dirs == [tmp_path / "sato", tmp_path / "taiga"]
    assert selected == [tmp_path / "sato"]


def test_validate_profile_id_rejects_default() -> None:
    with pytest.raises(ValueError):
        validate_profile_id("default")
