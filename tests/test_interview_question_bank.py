from job_agent.interview.question_bank import QuestionAnchorLibrary


def test_question_bank_filters_before_seeded_ordering() -> None:
    library = QuestionAnchorLibrary()
    first = library.search(
        topics={"idempotency"}, roles={"agent-engineer"}, difficulty="medium", seed="run-1"
    )
    assert [anchor.id for anchor in first] == ["backend-idempotency"]
    assert library.search(
        topics={"database-index"}, roles={"agent-engineer"}, difficulty="medium", seed="run-1"
    ) == []


def test_question_bank_public_view_hides_reference_points() -> None:
    anchor = QuestionAnchorLibrary().search(
        topics={"agent-loop"}, roles={"agent-engineer"}, difficulty="medium", seed="run-1"
    )[0]
    assert "reference_points" not in anchor.public_view()
    assert anchor.reference_points


def test_same_seed_is_reproducible() -> None:
    library = QuestionAnchorLibrary()
    kwargs = dict(topics=set(), roles={"agent-engineer"}, difficulty=None, limit=6)
    assert [item.id for item in library.search(seed="same", **kwargs)] == [
        item.id for item in library.search(seed="same", **kwargs)
    ]
