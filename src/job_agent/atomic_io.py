from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any, Mapping


def atomic_write_bytes(path: Path | str, payload: bytes) -> Path:
    """Write one complete file or leave the previous version untouched."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=target.parent,
            prefix=f".{target.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary_path = Path(handle.name)
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, target)
        temporary_path = None
    finally:
        if temporary_path is not None:
            try:
                temporary_path.unlink()
            except OSError:
                pass
    return target


def atomic_write_text(
    path: Path | str,
    text: str,
    *,
    encoding: str = "utf-8",
) -> Path:
    return atomic_write_bytes(Path(path), text.encode(encoding))


def atomic_write_json(path: Path | str, payload: Any) -> Path:
    rendered = json.dumps(payload, ensure_ascii=False, indent=2)
    return atomic_write_text(path, rendered, encoding="utf-8")


def atomic_write_text_batch(
    writes: Mapping[Path | str, str],
    *,
    encoding: str = "utf-8",
) -> tuple[Path, ...]:
    """Best-effort all-or-old commit for a small related set of text files."""
    normalized = {Path(path): text for path, text in writes.items()}
    previous = {
        path: path.read_bytes() if path.is_file() else None
        for path in normalized
    }
    staged: dict[Path, Path] = {}
    committed: list[Path] = []
    try:
        for target, text in normalized.items():
            target.parent.mkdir(parents=True, exist_ok=True)
            with tempfile.NamedTemporaryFile(
                mode="wb",
                dir=target.parent,
                prefix=f".{target.name}.",
                suffix=".tmp",
                delete=False,
            ) as handle:
                staged[target] = Path(handle.name)
                handle.write(text.encode(encoding))
                handle.flush()
                os.fsync(handle.fileno())
        for target, temporary in list(staged.items()):
            os.replace(temporary, target)
            committed.append(target)
            staged.pop(target)
    except Exception:
        for target in reversed(committed):
            original = previous[target]
            if original is None:
                try:
                    target.unlink()
                except FileNotFoundError:
                    pass
            else:
                atomic_write_bytes(target, original)
        raise
    finally:
        for temporary in staged.values():
            try:
                temporary.unlink()
            except OSError:
                pass
    return tuple(normalized)
