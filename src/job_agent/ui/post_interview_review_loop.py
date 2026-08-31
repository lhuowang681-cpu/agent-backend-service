from __future__ import annotations

import json
from pathlib import Path

from job_agent.agents.post_interview_review import PostInterviewReviewAgent
from job_agent.atomic_io import atomic_write_text_batch
from job_agent.atomic_io import atomic_write_json
from job_agent.llm.harness import NodePolicy
from job_agent.nodes.post_interview_review import render_post_interview_review
from job_agent.schemas import (
    EvidenceItem,
    FitVerdictResult,
    PostInterviewQuestionInput,
    PostInterviewReviewResult,
    PostInterviewRoundInput,
    StructuredJD,
)
from job_agent.session_orchestrator import (
    ARTIFACT_FILES,
    _parse_targeted_resume_markdown,
    write_session_state,
)
from job_agent.ui.real_interview_rounds import (
    RealInterviewRound,
    list_real_interview_rounds,
)


POST_INTERVIEW_REVIEW_MAX_OUTPUT_TOKENS = 8192
NEXT_INTERVIEW_FOCUS_FILE = "16_next_interview_focus.json"


def review_paths(session_dir: Path, round_id: str) -> tuple[Path, Path]:
    safe_round_id = Path(round_id).name
    if safe_round_id != round_id or not safe_round_id.startswith(
        "13_real_interview_round_"
    ):
        raise ValueError("invalid real interview round id")
    base = f"14_real_interview_review_{safe_round_id}"
    return Path(session_dir) / f"{base}.json", Path(session_dir) / f"{base}.md"


def load_post_interview_review_result(
    session_dir: Path,
    round_id: str,
) -> PostInterviewReviewResult | None:
    json_path, _ = review_paths(session_dir, round_id)
    try:
        return PostInterviewReviewResult.model_validate_json(
            json_path.read_text(encoding="utf-8")
        )
    except (OSError, ValueError):
        return None


def write_next_interview_focus(
    session_dir: Path,
    *,
    round_id: str,
    topics: list[str],
) -> Path:
    cleaned = list(dict.fromkeys(item.strip() for item in topics if item.strip()))
    if not cleaned:
        raise ValueError("AI 复盘没有可加入下一轮的练习主题")
    return atomic_write_json(
        Path(session_dir) / NEXT_INTERVIEW_FOCUS_FILE,
        {
            "source_round_id": round_id,
            "topics": cleaned,
        },
    )


def load_next_interview_focus(session_dir: Path) -> list[str]:
    try:
        payload = json.loads(
            (Path(session_dir) / NEXT_INTERVIEW_FOCUS_FILE).read_text(
                encoding="utf-8"
            )
        )
        topics = payload.get("topics", [])
    except (OSError, json.JSONDecodeError, AttributeError):
        return []
    if not isinstance(topics, list):
        return []
    return [
        item.strip()
        for item in topics
        if isinstance(item, str) and item.strip()
    ]


def _round_input(round_item: RealInterviewRound) -> PostInterviewRoundInput:
    questions = [
        PostInterviewQuestionInput(
            question_id=item.question_id,
            question=item.question,
            answer=item.answer,
            interviewer_follow_up=item.interviewer_follow_up,
            interviewer_feedback=item.interviewer_feedback,
            self_assessment=item.self_assessment,
        )
        for item in round_item.questions
    ]
    if not questions and round_item.notes.strip():
        questions = [
            PostInterviewQuestionInput(
                question_id="legacy_notes",
                question="根据本轮自由文本记录进行复盘",
                answer=round_item.notes,
                self_assessment="未判断",
            )
        ]
    return PostInterviewRoundInput(
        round_id=round_item.round_id,
        stage=round_item.stage,
        notes=round_item.notes,
        questions=questions,
    )


def generate_post_interview_review_live(
    session_dir: Path,
    round_id: str,
    runtime,
) -> tuple[Path, Path, PostInterviewReviewResult]:
    session_dir = Path(session_dir)
    rounds = list_real_interview_rounds(session_dir)
    round_item = next(
        (item for item in rounds if item.round_id == round_id),
        None,
    )
    if round_item is None:
        raise ValueError("real interview round not found")

    structured_jd = StructuredJD.model_validate(
        json.loads(
            (session_dir / ARTIFACT_FILES["structured_jd"]).read_text(
                encoding="utf-8"
            )
        )
    )
    evidence = [
        EvidenceItem.model_validate(item)
        for item in json.loads(
            (session_dir / ARTIFACT_FILES["evidence_mapping"]).read_text(
                encoding="utf-8"
            )
        )
    ]
    fit_result = FitVerdictResult.model_validate(
        json.loads(
            (session_dir / ARTIFACT_FILES["fit_verdict"]).read_text(
                encoding="utf-8"
            )
        )
    )
    del fit_result  # 读取用于确认完整 session 合同；复盘输出不消费 Fit 分数。
    targeted_resume = _parse_targeted_resume_markdown(
        (session_dir / ARTIFACT_FILES["targeted_resume"]).read_text(
            encoding="utf-8"
        ),
        structured_jd,
    )
    round_input = _round_input(round_item)
    result = PostInterviewReviewAgent(
        registry=runtime.registry,
        harness=runtime.harness,
        policy=NodePolicy(
            allow_rule_fallback=False,
            max_output_tokens=POST_INTERVIEW_REVIEW_MAX_OUTPUT_TOKENS,
        ),
    ).run(
        structured_jd,
        evidence=evidence,
        targeted_resume=targeted_resume,
        interview_round=round_input,
        session_id=session_dir.name,
        run_id=f"ui-post-interview-review-{round_id}",
    )

    json_path, markdown_path = review_paths(session_dir, round_id)
    rendered_json = json.dumps(
        result.value.model_dump(mode="json"),
        ensure_ascii=False,
        indent=2,
    )
    rendered_markdown = render_post_interview_review(
        result.value,
        round_input,
    )
    writes: dict[Path, str] = {
        json_path: rendered_json,
        markdown_path: rendered_markdown,
    }
    latest_is_selected = bool(rounds) and rounds[0].round_id == round_id
    if latest_is_selected:
        writes[
            session_dir / ARTIFACT_FILES["post_interview_review"]
        ] = rendered_markdown
    atomic_write_text_batch(writes)
    write_session_state(
        session_dir,
        fresh_artifacts=(
            {
                "post_interview_review": (
                    "Latest real interview round analyzed by the live review agent."
                )
            }
            if latest_is_selected
            else None
        ),
        updated_by="ui_live_post_interview_review",
    )
    return json_path, markdown_path, result.value
