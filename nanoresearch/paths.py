"""Centralized filesystem paths for NanoResearch runtime state."""

from __future__ import annotations

import os
from pathlib import Path


def normalize_runtime_path(path: str | os.PathLike[str] | Path) -> Path:
    """Return a stable absolute path for runtime workspace comparisons."""

    return Path(path).expanduser().resolve(strict=False)


def _env_path(name: str) -> Path | None:
    value = os.environ.get(name, "").strip()
    return normalize_runtime_path(value) if value else None


def get_nanoresearch_home() -> Path:
    """Return the root directory for NanoResearch runtime state."""

    return _env_path("NANORESEARCH_HOME") or normalize_runtime_path(Path.home() / ".nanoresearch")


def ensure_nanoresearch_home(subdirs: list[str] | tuple[str, ...] = ()) -> Path:
    """Create NanoResearch home and optional subdirectories.

    This handles ``~/.nanoresearch`` being a symlink, including a symlink whose
    target directory has not been created yet.
    """

    home = get_nanoresearch_home()
    if home.is_symlink():
        target = home.resolve(strict=False)
        target.mkdir(parents=True, exist_ok=True)
        base = target
    elif home.exists():
        if not home.is_dir():
            raise RuntimeError(f"{home} exists but is not a directory")
        base = home
    else:
        home.mkdir(parents=True, exist_ok=True)
        base = home

    for subdir in subdirs:
        (base / subdir).mkdir(parents=True, exist_ok=True)
    return base


def get_workspace_root() -> Path:
    root = (
        _env_path("NANORESEARCH_WORKSPACE_ROOT")
        or (get_nanoresearch_home() / "workspace" / "research")
    )
    return normalize_runtime_path(root)


def get_chat_memory_dir() -> Path:
    return get_nanoresearch_home() / "chat_memory"


def get_cache_dir() -> Path:
    return get_nanoresearch_home() / "cache"


def get_config_path() -> Path:
    return _env_path("NANORESEARCH_CONFIG") or (get_nanoresearch_home() / "config.json")


def get_private_endpoints_path() -> Path:
    return get_nanoresearch_home() / "private_endpoints.json"


def get_memory_dir() -> Path:
    return get_nanoresearch_home() / "memory"


def get_skills_dir() -> Path:
    return get_nanoresearch_home() / "skills"


def get_ram_data_dir() -> Path:
    return get_nanoresearch_home() / "ram_data"
