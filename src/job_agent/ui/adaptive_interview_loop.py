from __future__ import annotations

import uuid
from pathlib import Path

from job_agent.career import CareerStore
from job_agent.interview.context import InterviewContextBuilder
from job_agent.interview.contracts import AdaptiveInterviewRun, InterviewContextSnapshot, ProjectDossier
from job_agent.interview.controller import AdaptiveInterviewController
from job_agent.interview.paths import InterviewArtifactPaths
from job_agent.interview.run_store import AdaptiveInterviewRunStore


def start_project_grill(
    *,
    output_root: Path,
    repo_root: Path,
    dossier: ProjectDossier,
    duration_minutes: int,
    pressure: str,
    focus: str | None,
    runtime,
    run_id: str | None = None,
    controller: AdaptiveInterviewController | None = None,
) -> tuple[Path, AdaptiveInterviewRun]:
    run_id = run_id or f"project-grill-{uuid.uuid4().hex[:12]}"
    snapshot = InterviewContextBuilder.for_project_grill(dossier=dossier, run_id=run_id)
    root = Path(output_root).resolve()
    controller = controller or AdaptiveInterviewController(root, repo_root=repo_root)
    blueprint = controller.build_blueprint(snapshot, duration_minutes=duration_minutes, pressure=pressure, focus=focus, runtime=runtime)
    store = AdaptiveInterviewRunStore(root)
    run = store.create(snapshot, blueprint)
    decision = controller.next_turn(run, answer=None, runtime=runtime)
    return root, store.commit_turn(run, decision, answer=None, elapsed_seconds=0)


def start_full_mock(
    *,
    career_store: CareerStore,
    repo_root: Path,
    job_id: str,
    dossier: ProjectDossier | None,
    duration_minutes: int,
    pressure: str,
    runtime,
    run_id: str | None = None,
    controller: AdaptiveInterviewController | None = None,
) -> tuple[Path, AdaptiveInterviewRun]:
    job = career_store.get_job(job_id)
    if not job.session_dir:
        raise ValueError("岗位尚未关联 JD/简历 session")
    run_id = run_id or f"full-mock-{uuid.uuid4().hex[:12]}"
    snapshot = InterviewContextBuilder(career_store).for_full_mock(job_id=job_id, run_id=run_id, dossier=dossier)
    root = Path(job.session_dir).resolve()
    controller = controller or AdaptiveInterviewController(root, repo_root=repo_root)
    blueprint = controller.build_blueprint(snapshot, duration_minutes=duration_minutes, pressure=pressure, runtime=runtime)
    store = AdaptiveInterviewRunStore(root)
    run = store.create(snapshot, blueprint)
    decision = controller.next_turn(run, answer=None, runtime=runtime)
    return root, store.commit_turn(run, decision, answer=None, elapsed_seconds=0)


def submit_adaptive_answer(
    *,
    root: Path,
    repo_root: Path,
    run_id: str,
    answer: str,
    elapsed_seconds: int,
    runtime,
    controller: AdaptiveInterviewController | None = None,
) -> AdaptiveInterviewRun:
    store = AdaptiveInterviewRunStore(root)
    run = store.load(run_id)
    if run.status != "waiting" or run.pending_turn is None:
        raise ValueError("面试当前没有待回答问题")
    controller = controller or AdaptiveInterviewController(root, repo_root=repo_root)
    projected = run.model_copy(
        update={
            "active_seconds": run.active_seconds + elapsed_seconds,
            "turn_count": run.turn_count + 1,
        }
    )
    decision = controller.next_turn(projected, answer=answer, runtime=runtime)
    return store.commit_turn(run, decision, answer=answer, elapsed_seconds=elapsed_seconds)


def index_finished_interview(
    *,
    career_store: CareerStore,
    root: Path,
    run: AdaptiveInterviewRun,
) -> None:
    if run.status not in {"completed", "ended_by_user"}:
        raise ValueError("only a finished interview can be indexed")
    paths = InterviewArtifactPaths.for_run(root, run.run_id)
    if not paths.transcript.exists():
        raise ValueError("finished interview transcript is missing")
    snapshot = InterviewContextSnapshot.model_validate_json(paths.context.read_text(encoding="utf-8"))
    transcript_path = paths.transcript.relative_to(paths.root).as_posix()
    review_path = paths.debrief_json.relative_to(paths.root).as_posix() if paths.debrief_json.exists() else None
    career_store.add_interview(
        job_id=snapshot.job_id,
        company_id=snapshot.company_id,
        kind=run.kind,
        round_label="项目拷打" if run.kind == "project_grill" else "完整模拟面试",
        occurred_at=run.updated_at,
        session_dir=paths.root,
        artifact_path=transcript_path,
        ai_review_path=review_path,
    )
