from __future__ import annotations

import json
from pathlib import Path

import pytest

from job_agent.atomic_io import (
    atomic_write_json,
    atomic_write_text,
    atomic_write_text_batch,
)


def test_atomic_write_text_replaces_complete_unicode_content(tmp_path: Path) -> None:
    path = tmp_path / "artifact.md"
    path.write_text("上一成功版本", encoding="utf-8")

    atomic_write_text(path, "新的中文内容\n\\textbf{LaTeX}")

    assert path.read_text(encoding="utf-8") == "新的中文内容\n\\textbf{LaTeX}"
    assert list(tmp_path.glob(f".{path.name}.*.tmp")) == []


def test_atomic_write_json_is_immediately_readable(tmp_path: Path) -> None:
    path = tmp_path / "state.json"

    atomic_write_json(path, {"状态": "完成", "count": 2})

    assert json.loads(path.read_text(encoding="utf-8")) == {
        "状态": "完成",
        "count": 2,
    }


def test_atomic_write_failure_preserves_previous_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "artifact.md"
    path.write_text("上一成功版本", encoding="utf-8")

    def fail_replace(source, target) -> None:
        raise OSError("simulated replace failure")

    monkeypatch.setattr("job_agent.atomic_io.os.replace", fail_replace)

    with pytest.raises(OSError, match="simulated replace failure"):
        atomic_write_text(path, "不应提交的新版本")

    assert path.read_text(encoding="utf-8") == "上一成功版本"
    assert list(tmp_path.glob(f".{path.name}.*.tmp")) == []


def test_atomic_batch_restores_all_previous_files_on_commit_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first = tmp_path / "first.md"
    second = tmp_path / "second.md"
    first.write_text("旧一", encoding="utf-8")
    second.write_text("旧二", encoding="utf-8")
    import job_agent.atomic_io as atomic_io

    real_replace = atomic_io.os.replace
    calls = 0

    def fail_second_replace(source, target) -> None:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("second commit failed")
        real_replace(source, target)

    monkeypatch.setattr(atomic_io.os, "replace", fail_second_replace)

    with pytest.raises(OSError, match="second commit failed"):
        atomic_write_text_batch({first: "新一", second: "新二"})

    assert first.read_text(encoding="utf-8") == "旧一"
    assert second.read_text(encoding="utf-8") == "旧二"
    assert list(tmp_path.glob(".*.tmp")) == []
