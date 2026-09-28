"""ユーザー筆跡の人物プロファイル（``<root>/<profile>/<文字>/*.json``）。"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

_PROFILE_ID_RE = re.compile(r"^[A-Za-z0-9_-]+$")


@dataclass(frozen=True)
class StrokeProfile:
    id: str
    path: Path
    character_count: int
    sample_count: int


def validate_profile_id(profile_id: str) -> str:
    profile_id = profile_id.strip()
    if not profile_id:
        raise ValueError("profile id is required")
    if profile_id == "default":
        raise ValueError("'default' profile is not used in this project")
    if not _PROFILE_ID_RE.match(profile_id):
        raise ValueError("profile id must contain only letters, numbers, '_' or '-'")
    return profile_id


def list_profiles(root_dir: Path) -> list[StrokeProfile]:
    root_dir = Path(root_dir)
    if not root_dir.exists():
        return []

    profiles: list[StrokeProfile] = []
    for profile_dir in sorted(root_dir.iterdir()):
        if not profile_dir.is_dir() or profile_dir.name == "default":
            continue
        if list(profile_dir.glob("*.json")):
            continue
        char_dirs = [d for d in profile_dir.iterdir() if d.is_dir() and list(d.glob("*.json"))]
        sample_count = sum(len(list(d.glob("*.json"))) for d in char_dirs)
        profiles.append(
            StrokeProfile(
                id=profile_dir.name,
                path=profile_dir,
                character_count=len(char_dirs),
                sample_count=sample_count,
            )
        )
    return profiles


def profile_to_dict(profile: StrokeProfile) -> dict:
    return {
        "id": profile.id,
        "path": str(profile.path),
        "character_count": profile.character_count,
        "sample_count": profile.sample_count,
    }


def ensure_profile(root_dir: Path, profile_id: str) -> Path:
    profile_id = validate_profile_id(profile_id)
    path = Path(root_dir) / profile_id
    path.mkdir(parents=True, exist_ok=True)
    return path


def resolve_character_root(root: Path | str | None, profile_id: str | None = None) -> Path | None:
    """ユーザー筆跡ディレクトリを「文字ディレクトリ群の親」へ解決する。

    ``root`` がプロファイルのルート（``<root>/<profile>/<文字>/*.json``）なら
    ``profile_id`` のプロファイル（未指定・不明なら先頭）を返す。``root`` 自体が
    文字ディレクトリ群の親ならそのまま返す。存在しなければ None。
    """
    if root is None or not Path(root).is_dir():
        return None
    profiles = list_profiles(Path(root))
    if not profiles:
        return Path(root)
    for p in profiles:
        if p.id == profile_id:
            return p.path
    return profiles[0].path


def resolve_training_dirs(root_dir: Path, dataset: dict | None) -> list[Path]:
    dataset = dataset or {"mode": "current"}
    mode = dataset.get("mode", "current")
    root_dir = Path(root_dir)
    profiles = {p.id: p.path for p in list_profiles(root_dir)}

    if mode == "all":
        return list(profiles.values())

    if mode == "profiles":
        ids = dataset.get("profiles") or []
        result = []
        for profile_id in ids:
            profile_id = validate_profile_id(str(profile_id))
            if profile_id not in profiles:
                raise ValueError(f"profile not found: {profile_id}")
            result.append(profiles[profile_id])
        return result

    profile_id = validate_profile_id(str(dataset.get("profile") or "taiga"))
    if profile_id not in profiles:
        raise ValueError(f"profile not found: {profile_id}")
    return [profiles[profile_id]]
