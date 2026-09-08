"""장면 저장/불러오기. DESIGN.md §3.5."""

from __future__ import annotations

import json
from pathlib import Path

from .model import Scene


def save_scene(scene: Scene, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(scene.model_dump_json(indent=2), encoding="utf-8")
    return path


def load_scene(path: str | Path) -> Scene:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return Scene.model_validate(data)
