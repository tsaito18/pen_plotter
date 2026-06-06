from __future__ import annotations

import re
import shutil
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


def is_character_data_dir(path: Path) -> bool:
    if not path.is_dir():
        return False
    for char_dir in path.iterdir():
        if char_dir.is_dir() and list(char_dir.glob("*.json")):
            return True
    return False


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


def resolve_profile_dir(root_dir: Path, profile_id: str) -> Path:
    profile_id = validate_profile_id(profile_id)
    path = Path(root_dir) / profile_id
    if not path.is_dir():
        raise ValueError(f"profile not found: {profile_id}")
    return path


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


def migrate_legacy_root(root_dir: Path, profile_id: str = "taiga") -> list[Path]:
    """Move legacy root-level character dirs into a named profile.

    This is intentionally conservative: only directories that directly contain JSON files
    are moved. Existing profile directories are left untouched.
    """
    root_dir = Path(root_dir)
    target = ensure_profile(root_dir, profile_id)
    moved: list[Path] = []
    for child in sorted(root_dir.iterdir()):
        if child == target or not child.is_dir():
            continue
        if not list(child.glob("*.json")):
            continue
        dest = target / child.name
        if dest.exists():
            raise FileExistsError(f"migration target already exists: {dest}")
        shutil.move(str(child), str(dest))
        moved.append(dest)
    return moved
