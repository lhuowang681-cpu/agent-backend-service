from __future__ import annotations

from collections.abc import Collection

from pydantic import ValidationError


def safe_validation_error_detail(
    error: ValidationError,
    *,
    allowed_fields: Collection[str],
    max_chars: int = 400,
) -> str:
    """Return schema paths and runtime types without rejected values."""
    allowed = set(allowed_fields)
    items: list[str] = []
    for item in error.errors():
        parts: list[str] = []
        for part in item.get("loc", ()):
            text = str(part)
            parts.append(text if text.isdigit() or text in allowed else "<field>")
        location = ".".join(parts) or "root"
        input_type = type(item.get("input")).__name__ if "input" in item else "unknown"
        items.append(f"{location}:{item.get('type', 'validation_error')}:{input_type}")
    return ";".join(items)[:max_chars] or "schema_validation_failed"
