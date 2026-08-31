from pathlib import Path

import pytest

from job_agent.evidence.contracts import EvidenceV2Error, SourceInput, SourceQuoteLocator, SourceSpan
from job_agent.evidence.sources import EvidenceSourceStore


def test_source_snapshot_normalizes_and_validates_exact_span(tmp_path: Path):
    store = EvidenceSourceStore(tmp_path)
    catalog = store.snapshot(
        "run-1",
        [SourceInput(kind="raw_jd", display_label="JD", text="能力A\r\n能力B")],
    )
    document = catalog.documents[0]
    assert store.read(catalog, document.source_id) == "能力A\n能力B"
    span = store.make_span(catalog, document.source_id, 0, 3)
    store.validate_span(catalog, span)
    with pytest.raises(EvidenceV2Error, match="invalid_source_span"):
        store.validate_span(catalog, span.model_copy(update={"exact_quote": "伪造"}))


def test_source_store_rejects_generated_and_sensitive_sources(tmp_path: Path):
    store = EvidenceSourceStore(tmp_path)
    with pytest.raises(EvidenceV2Error, match="generated_source_forbidden"):
        store.snapshot(
            "run-1",
            [
                SourceInput(
                    kind="original_resume",
                    display_label="target",
                    text="generated",
                    artifact_type="target_resume",
                )
            ],
        )
    secret = tmp_path / "glm.txt"
    secret.write_text("key", encoding="utf-8")
    with pytest.raises(EvidenceV2Error, match="forbidden_source"):
        store.snapshot(
            "run-2",
            [SourceInput(kind="user_artifact", display_label="secret", path=str(secret))],
        )


@pytest.mark.parametrize(
    "artifact_type",
    [
        "target_resume",
        "interview_transcript",
        "interview_debrief",
        "answer_card",
        "interview_score",
    ],
)
def test_generated_consumer_artifacts_cannot_become_evidence(
    tmp_path: Path, artifact_type: str
):
    with pytest.raises(EvidenceV2Error, match="generated_source_forbidden"):
        EvidenceSourceStore(tmp_path).snapshot(
            f"run-{artifact_type}",
            [
                SourceInput(
                    kind="user_artifact",
                    display_label=artifact_type,
                    text="generated content",
                    artifact_type=artifact_type,
                )
            ],
        )


def test_learning_document_path_is_forbidden(tmp_path: Path):
    learning = tmp_path / "docs" / "learning" / "notes.md"
    learning.parent.mkdir(parents=True)
    learning.write_text("private study notes", encoding="utf-8")
    store = EvidenceSourceStore(tmp_path / "session", allowed_roots=(tmp_path,))
    with pytest.raises(EvidenceV2Error, match="forbidden_source"):
        store.snapshot(
            "run-learning",
            [SourceInput(kind="user_artifact", display_label="notes", path=str(learning))],
        )


def test_source_store_detects_stale_snapshot(tmp_path: Path):
    store = EvidenceSourceStore(tmp_path)
    catalog = store.snapshot(
        "run-1",
        [SourceInput(kind="original_resume", display_label="resume", text="Python project")],
    )
    snapshot = tmp_path / catalog.documents[0].snapshot_path
    snapshot.write_text("tampered", encoding="utf-8")
    with pytest.raises(EvidenceV2Error, match="stale_source_snapshot"):
        store.read(catalog, catalog.documents[0].source_id)


def test_unique_exact_quote_can_canonicalize_wrong_offsets(tmp_path: Path):
    store = EvidenceSourceStore(tmp_path)
    catalog = store.snapshot(
        "run-1",
        [SourceInput(kind="raw_jd", display_label="JD", text="前缀 独立评测 后缀")],
    )
    document = catalog.documents[0]
    wrong = SourceSpan(
        source_id=document.source_id,
        content_hash=document.content_hash,
        start_offset=0,
        end_offset=4,
        exact_quote="独立评测",
    )
    canonical, changed = store.canonicalize_span(catalog, wrong)
    assert changed is True
    assert (canonical.start_offset, canonical.end_offset) == (3, 7)
    store.validate_span(catalog, canonical)


def test_duplicate_quote_cannot_be_canonicalized(tmp_path: Path):
    store = EvidenceSourceStore(tmp_path)
    catalog = store.snapshot(
        "run-1",
        [SourceInput(kind="raw_jd", display_label="JD", text="评测和评测")],
    )
    document = catalog.documents[0]
    wrong = SourceSpan(
        source_id=document.source_id,
        content_hash=document.content_hash,
        start_offset=1,
        end_offset=3,
        exact_quote="评测",
    )
    with pytest.raises(EvidenceV2Error, match="invalid_source_span"):
        store.canonicalize_span(catalog, wrong)


def test_quote_locator_resolves_unicode_and_persists_source_metadata(tmp_path: Path):
    store = EvidenceSourceStore(tmp_path)
    catalog = store.snapshot(
        "run-1",
        [
            SourceInput(
                kind="project_source",
                display_label="实验记录",
                text="前缀 🧪准确率 91.2% 后缀",
                artifact_type="Experiment-Record",
            )
        ],
    )
    span = store.resolve_locator(
        catalog,
        SourceQuoteLocator(source_id=catalog.documents[0].source_id, exact_quote="🧪准确率 91.2%"),
    )
    assert span.exact_quote == "🧪准确率 91.2%"
    assert span.start_offset == 3
    assert catalog.documents[0].artifact_type == "experiment_record"
    store.validate_span(catalog, span)


def test_quote_locator_requires_unique_quote_or_disambiguating_anchor(tmp_path: Path):
    store = EvidenceSourceStore(tmp_path)
    catalog = store.snapshot(
        "run-1",
        [SourceInput(kind="original_resume", display_label="resume", text="项目A：提效。项目B：提效。")],
    )
    source_id = catalog.documents[0].source_id
    with pytest.raises(EvidenceV2Error, match="ambiguous_source_quote"):
        store.resolve_locator(catalog, SourceQuoteLocator(source_id=source_id, exact_quote="提效"))
    span = store.resolve_locator(
        catalog,
        SourceQuoteLocator(source_id=source_id, exact_quote="提效", prefix_anchor="项目B："),
    )
    assert span.start_offset == len("项目A：提效。项目B：")


def test_quote_locator_fails_closed_for_missing_quote_and_stale_snapshot(tmp_path: Path):
    store = EvidenceSourceStore(tmp_path)
    catalog = store.snapshot(
        "run-1", [SourceInput(kind="original_resume", display_label="resume", text="唯一事实")]
    )
    source = catalog.documents[0]
    with pytest.raises(EvidenceV2Error, match="source_quote_not_found"):
        store.resolve_locator(catalog, SourceQuoteLocator(source_id=source.source_id, exact_quote="不存在"))
    (tmp_path / source.snapshot_path).write_text("被篡改", encoding="utf-8")
    with pytest.raises(EvidenceV2Error, match="stale_source_snapshot"):
        store.resolve_locator(catalog, SourceQuoteLocator(source_id=source.source_id, exact_quote="唯一事实"))
