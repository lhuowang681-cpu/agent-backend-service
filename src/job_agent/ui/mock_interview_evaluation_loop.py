from __future__ import annotations

import json
import re
from pathlib import Path

from job_agent.agents.mock_interview_evaluation import (
    MockInterviewEvaluationAgent,
)
from job_agent.atomic_io import atomic_write_json, atomic_write_text_batch
from job_agent.llm.harness import NodePolicy
from job_agent.nodes.mock_interview_evaluation import (
    render_mock_interview_evaluation,
    render_rule_evidence_audit,
)
from job_agent.schemas import (
    AnswerCardDeck,
    EvidenceItem,
    MockInterviewEvaluationResult,
    MockInterviewPlan,
    MockInterviewTranscriptItem,
    RuleEvidenceAudit,
)
from job_agent.session_orchestrator import ARTIFACT_FILES


MOCK_INTERVIEW_EVALUATION_MAX_OUTPUT_TOKENS = 8192
_RUN_FILE = re.compile(r"^10_mock_interview_run_(.+)\.json$")


def evaluation_paths(run_path: Path) -> tuple[Path, Path]:
    run_path = Path(run_path)
    match = _RUN_FILE.fullmatch(run_path.name)
    if match is None:
        raise ValueError("invalid mock interview run path")
    run_id = match.group(1)
    base = f"11_mock_interview_evaluation_{run_id}"
    return run_path.parent / f"{base}.json", run_path.parent / f"{base}.md"


def load_mock_interview_evaluation(
    run_path: Path,
) -> MockInterviewEvaluationResult | None:
    json_path, _ = evaluation_paths(run_path)
    try:
        return MockInterviewEvaluationResult.model_validate_json(
            json_path.read_text(encoding="utf-8")
        )
    except (OSError, ValueError):
        return None


def update_run_evaluation_status(
    run_path: Path,
    *,
    status: str,
    error_code: str | None = None,
) -> None:
    payload = json.loads(Path(run_path).read_text(encoding="utf-8"))
    payload["evaluation_status"] = status
    if error_code is None:
        payload.pop("evaluation_error_code", None)
    else:
        payload["evaluation_error_code"] = error_code
    atomic_write_json(Path(run_path), payload)


def _run_id(run_path: Path) -> str:
    match = _RUN_FILE.fullmatch(Path(run_path).name)
    if match is None:
        raise ValueError("invalid mock interview run path")
    return match.group(1)


def generate_mock_interview_evaluation_live(
    session_dir: Path,
    run_path: Path,
    *,
    plan: MockInterviewPlan,
    history: list[dict],
    answer_cards: AnswerCardDeck,
    runtime,
) -> MockInterviewEvaluationResult:
    session_dir = Path(session_dir)
    evidence_payload = json.loads(
        (session_dir / ARTIFACT_FILES["evidence_mapping"]).read_text(
            encoding="utf-8"
        )
    )
    evidence = [EvidenceItem.model_validate(item) for item in evidence_payload]
    transcript = [
        MockInterviewTranscriptItem(
            question_id=str(item["question_id"]),
            requirement_id=str(item["requirement_id"]),
            question=str(item["prompt"]),
            answer=str(item.get("answer", "")),
            elapsed_seconds=int(item.get("elapsed_seconds", 0)),
        )
        for item in history
    ]
    effective_plan = plan.model_copy(
        update={
            "questions": [
                plan.questions[0].model_copy(
                    update={
                        "question_id": item.question_id,
                        "requirement_id": item.requirement_id,
                        "prompt": item.question,
                    }
                )
                for item in transcript
            ]
        }
    )
    run_id = _run_id(run_path)
    result = MockInterviewEvaluationAgent(
        registry=runtime.registry,
        harness=runtime.harness,
        policy=NodePolicy(
            allow_rule_fallback=False,
            max_output_tokens=MOCK_INTERVIEW_EVALUATION_MAX_OUTPUT_TOKENS,
        ),
    ).run(
        effective_plan,
        transcript=transcript,
        answer_cards=answer_cards,
        evidence=evidence,
        session_id=session_dir.name,
        run_id=run_id,
    )

    payload = json.loads(run_path.read_text(encoding="utf-8"))
    audits = [
        RuleEvidenceAudit.model_validate(item)
        for item in payload.get("rule_evidence_audit", [])
    ]
    evaluation_markdown = render_mock_interview_evaluation(
        result.value,
        transcript,
    )
    combined_markdown = evaluation_markdown
    if audits:
        combined_markdown += "\n" + render_rule_evidence_audit(audits)
    json_path, markdown_path = evaluation_paths(run_path)
    atomic_write_text_batch(
        {
            json_path: json.dumps(
                result.value.model_dump(mode="json"),
                ensure_ascii=False,
                indent=2,
            ),
            markdown_path: evaluation_markdown,
            session_dir / "15_mock_interview_debrief.md": combined_markdown,
        }
    )

    if result.value.next_mock_topics:
        from job_agent.ui.post_interview_review_loop import (
            write_next_interview_focus,
        )

        write_next_interview_focus(
            session_dir,
            round_id=f"mock:{run_id}",
            topics=result.value.next_mock_topics,
        )
    update_run_evaluation_status(
        run_path,
        status="ai_complete",
    )
    return result.value
