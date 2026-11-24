"""Tiny progress tracker: each passed check is recorded in <workdir>/progress.json."""
from __future__ import annotations

import json
import time
from pathlib import Path

_WORKDIR: Path | None = None


def bind(workdir: Path) -> None:
    global _WORKDIR
    _WORKDIR = Path(workdir)


def _file() -> Path | None:
    return (_WORKDIR / "progress.json") if _WORKDIR else None


def load() -> dict:
    f = _file()
    if f and f.exists():
        return json.loads(f.read_text())
    return {}


def mark(exercise_id: str, passed: bool, detail: str = "") -> None:
    f = _file()
    if not f:
        return
    data = load()
    prev = data.get(exercise_id, {})
    data[exercise_id] = {
        "passed": passed or prev.get("passed", False),
        "attempts": prev.get("attempts", 0) + 1,
        "last": time.strftime("%Y-%m-%d %H:%M"),
        "detail": detail,
    }
    f.write_text(json.dumps(data, indent=2))


def summary(prefix: str = "") -> None:
    data = {k: v for k, v in load().items() if k.startswith(prefix)}
    if not data:
        print("No exercises recorded yet.")
        return
    for k, v in sorted(data.items()):
        print(f"  {'PASS' if v['passed'] else 'todo'}  {k:24s} attempts={v['attempts']}")
    done = sum(v["passed"] for v in data.values())
    print(f"\n{done}/{len(data)} exercises passed")
