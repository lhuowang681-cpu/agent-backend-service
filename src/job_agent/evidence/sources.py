from __future__ import annotations

import hashlib
import unicodedata
from pathlib import Path, PurePosixPath

from job_agent.evidence.contracts import (
    EvidenceV2Error,
    SourceCatalog,
    SourceDocumentRef,
    SourceInput,
    SourceQuoteLocator,
    SourceSpan,
)
from job_agent.evidence.paths import EvidenceArtifactPaths


_DENIED_ARTIFACT_TYPES = {
    "target_resume",
    "interview_transcript",
    "interview_debrief",
    "answer_card",
    "interview_score",
}
_DENIED_NAMES = {"glm.txt", ".env", ".git"}


def normalize_artifact_type(value: str | None, *, kind: str) -> str | None:
    if value:
        normalized = "_".join(value.strip().casefold().replace("-", " ").split())
        return normalized or None
    if kind == "original_resume":
        return "original_resume"
    return None


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def normalize_text(value: str) -> str:
    return unicodedata.normalize("NFC", value.replace("\r\n", "\n").replace("\r", "\n"))


class EvidenceSourceStore:
    def __init__(self, session_dir: Path, *, allowed_roots: tuple[Path, ...] = ()) -> None:
        self.session_dir = Path(session_dir).resolve()
        self.allowed_roots = tuple(Path(item).resolve() for item in allowed_roots)

    def snapshot(self, run_id: str, inputs: list[SourceInput]) -> SourceCatalog:
        if not inputs:
            raise EvidenceV2Error("empty_source_inputs")
        paths = EvidenceArtifactPaths(self.session_dir, run_id)
        paths.sources_dir.mkdir(parents=True, exist_ok=True)
        documents: list[SourceDocumentRef] = []
        for index, item in enumerate(inputs, start=1):
            self._validate_input_policy(item)
            raw = self._read_input(item)
            try:
                decoded = raw.decode("utf-8")
            except UnicodeDecodeError as exc:
                raise EvidenceV2Error("unsupported_source_format") from exc
            normalized = normalize_text(decoded)
            if not normalized.strip():
                raise EvidenceV2Error("empty_source")
            encoded = normalized.encode("utf-8")
            source_id = f"src_{index:03d}_{item.kind}"
            target = paths.source_path(source_id)
            if target.exists():
                raise EvidenceV2Error("immutable_source_exists")
            target.write_bytes(encoded)
            relative = PurePosixPath(target.relative_to(self.session_dir)).as_posix()
            documents.append(
                SourceDocumentRef(
                    source_id=source_id,
                    kind=item.kind,
                    raw_hash=sha256_bytes(raw),
                    content_hash=sha256_bytes(encoded),
                    snapshot_path=relative,
                    display_label=item.display_label,
                    artifact_type=normalize_artifact_type(item.artifact_type, kind=item.kind),
                )
            )
        return SourceCatalog(documents=documents)

    def read(self, catalog: SourceCatalog, source_id: str) -> str:
        document = self._document(catalog, source_id)
        path = self._resolve_snapshot(document.snapshot_path)
        raw = path.read_bytes()
        if sha256_bytes(raw) != document.content_hash:
            raise EvidenceV2Error("stale_source_snapshot")
        try:
            return raw.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise EvidenceV2Error("unsupported_source_format") from exc

    def validate_span(self, catalog: SourceCatalog, span: SourceSpan) -> None:
        document = self._document(catalog, span.source_id)
        if document.content_hash != span.content_hash:
            raise EvidenceV2Error("source_hash_mismatch")
        text = self.read(catalog, span.source_id)
        if span.end_offset > len(text) or text[span.start_offset : span.end_offset] != span.exact_quote:
            raise EvidenceV2Error("invalid_source_span")

    def canonicalize_span(self, catalog: SourceCatalog, span: SourceSpan) -> tuple[SourceSpan, bool]:
        """Correct offsets only when the exact quote has one immutable-source location."""
        document = self._document(catalog, span.source_id)
        if document.content_hash != span.content_hash:
            raise EvidenceV2Error("source_hash_mismatch")
        text = self.read(catalog, span.source_id)
        if (
            span.end_offset <= len(text)
            and text[span.start_offset : span.end_offset] == span.exact_quote
        ):
            return span, False
        start = text.find(span.exact_quote)
        if start < 0 or text.find(span.exact_quote, start + 1) >= 0:
            raise EvidenceV2Error("invalid_source_span")
        return (
            span.model_copy(
                update={
                    "start_offset": start,
                    "end_offset": start + len(span.exact_quote),
                }
            ),
            True,
        )

    def resolve_locator(self, catalog: SourceCatalog, locator: SourceQuoteLocator) -> SourceSpan:
        return SourceQuoteResolver(self).resolve(catalog, locator)

    def make_span(self, catalog: SourceCatalog, source_id: str, start: int, end: int) -> SourceSpan:
        document = self._document(catalog, source_id)
        text = self.read(catalog, source_id)
        if start < 0 or end <= start or end > len(text):
            raise EvidenceV2Error("invalid_source_span")
        return SourceSpan(
            source_id=source_id,
            content_hash=document.content_hash,
            start_offset=start,
            end_offset=end,
            exact_quote=text[start:end],
        )

    @staticmethod
    def independence_group(span: SourceSpan) -> str:
        raw = f"{span.source_id}:{span.content_hash}:{span.start_offset}:{span.end_offset}"
        return sha256_bytes(raw.encode("utf-8"))

    def _read_input(self, item: SourceInput) -> bytes:
        if item.text is not None:
            return item.text.encode("utf-8")
        assert item.path is not None
        path = Path(item.path)
        if path.is_symlink():
            raise EvidenceV2Error("forbidden_source")
        try:
            resolved = path.resolve(strict=True)
        except OSError as exc:
            raise EvidenceV2Error("source_not_found") from exc
        if not self.allowed_roots or not any(resolved.is_relative_to(root) for root in self.allowed_roots):
            raise EvidenceV2Error("forbidden_source")
        if not resolved.is_file() or resolved.stat().st_size > 2 * 1024 * 1024:
            raise EvidenceV2Error("unsupported_source_format")
        return resolved.read_bytes()

    @staticmethod
    def _validate_input_policy(item: SourceInput) -> None:
        if item.artifact_type and item.artifact_type.casefold() in _DENIED_ARTIFACT_TYPES:
            raise EvidenceV2Error("generated_source_forbidden")
        if item.path:
            parts = [part.casefold() for part in PurePosixPath(item.path.replace("\\", "/")).parts]
            if (
                any(part in _DENIED_NAMES or "secret" in part or "credential" in part for part in parts)
                or "learning" in parts and "docs" in parts
            ):
                raise EvidenceV2Error("forbidden_source")

    @staticmethod
    def _document(catalog: SourceCatalog, source_id: str) -> SourceDocumentRef:
        for item in catalog.documents:
            if item.source_id == source_id:
                return item
        raise EvidenceV2Error("unknown_source")

    def _resolve_snapshot(self, relative_path: str) -> Path:
        path = (self.session_dir / Path(*PurePosixPath(relative_path).parts)).resolve()
        if not path.is_relative_to(self.session_dir) or not path.is_file():
            raise EvidenceV2Error("forbidden_source")
        return path


class SourceQuoteResolver:
    """Resolve quote-only model output to a unique audited persisted span."""

    def __init__(self, source_store: EvidenceSourceStore) -> None:
        self.source_store = source_store

    def resolve(self, catalog: SourceCatalog, locator: SourceQuoteLocator) -> SourceSpan:
        document = self.source_store._document(catalog, locator.source_id)
        text = self.source_store.read(catalog, locator.source_id)
        starts: list[int] = []
        cursor = 0
        while True:
            start = text.find(locator.exact_quote, cursor)
            if start < 0:
                break
            end = start + len(locator.exact_quote)
            prefix_ok = locator.prefix_anchor is None or text[:start].endswith(locator.prefix_anchor)
            suffix_ok = locator.suffix_anchor is None or text[end:].startswith(locator.suffix_anchor)
            if prefix_ok and suffix_ok:
                starts.append(start)
            cursor = start + 1
        if not starts:
            raise EvidenceV2Error("source_quote_not_found")
        if len(starts) != 1:
            raise EvidenceV2Error("ambiguous_source_quote")
        start = starts[0]
        return SourceSpan(
            source_id=document.source_id,
            content_hash=document.content_hash,
            start_offset=start,
            end_offset=start + len(locator.exact_quote),
            exact_quote=locator.exact_quote,
        )
