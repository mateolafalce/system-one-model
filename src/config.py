"""YAML config helpers and a writable Hugging Face cache."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]


def _dir_is_writable(path: Path) -> bool:
    try:
        path.mkdir(parents=True, exist_ok=True)
        probe = path / ".write_test"
        probe.write_text("ok")
        probe.unlink()
        return True
    except OSError:
        return False


def ensure_hf_home() -> Path:
    """Use ~/.cache/huggingface when writable; otherwise repo-local .hf.

    This machine's default hub is root-owned, so new models (ModernBERT)
    cannot be downloaded there. Existing snapshots are symlinked in.
    """
    if os.environ.get("HF_HOME"):
        home = Path(os.environ["HF_HOME"])
        home.mkdir(parents=True, exist_ok=True)
        return home
    default = Path.home() / ".cache" / "huggingface"
    if _dir_is_writable(default / "hub"):
        return default
    local = ROOT / ".hf"
    hub = local / "hub"
    hub.mkdir(parents=True, exist_ok=True)
    (local / "datasets").mkdir(exist_ok=True)
    src_hub = default / "hub"
    if src_hub.is_dir():
        for item in src_hub.iterdir():
            if not item.name.startswith("models--"):
                continue
            dest = hub / item.name
            if not dest.exists():
                try:
                    dest.symlink_to(item)
                except OSError:
                    pass
    os.environ["HF_HOME"] = str(local)
    os.environ.setdefault("HF_HUB_CACHE", str(hub))
    os.environ.setdefault("HF_DATASETS_CACHE", str(local / "datasets"))
    os.environ.setdefault("TRANSFORMERS_CACHE", str(hub))
    return local


ensure_hf_home()

import yaml  # noqa: E402


def load_yaml(path: str | Path) -> dict[str, Any]:
    p = Path(path)
    if not p.is_absolute():
        p = ROOT / p
    with p.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"config {p} is not a mapping")
    return data


def resolve_path(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else ROOT / p
