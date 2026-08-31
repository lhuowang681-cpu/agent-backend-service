from __future__ import annotations

import copy
import hashlib
import json
from typing import Any, Callable, TypedDict

from job_agent.agent_runtime.failures import CheckpointError
from job_agent.agents.answer_cards import (
    PROMPT_VERSION as ANSWER_CARDS_PROMPT_VERSION,
    AnswerCardsAgent,
)
from job_agent.agents.evidence_mapping import (
    PROMPT_VERSION as EVIDENCE_PROMPT_VERSION,
    EvidenceMappingAgent,
)
from job_agent.agents.interview_prep import (
    PROMPT_VERSION as INTERVIEW_PROMPT_VERSION,
    InterviewPrepAgent,
)
from job_agent.agents.jd_structurer import (
    PROMPT_VERSION as JD_PROMPT_VERSION,
    JDStructurerAgent,
)
from job_agent.agents.resume_tailoring import (
    PROMPT_VERSION as RESUME_PROMPT_VERSION,
    ResumeTailoringAgent,
)
from job_agent.graph import JobAgentState
from job_agent.llm.harness import LLMHarness, NodePolicy
from job_agent.llm.provider import ProviderTrace
from job_agent.nodes.action_suggestion import suggest_action
from job_agent.nodes.fit_verdict import FitVerdictBackend, RuleFitVerdictBackend
from job_agent.nodes.mock_interview import build_mock_interview_plan
from job_agent.nodes.verdict_route import build_verdict_route
from job_agent.runtime.concurrency import ExecutionCoordinator
from job_agent.runtime.modes import RuntimeMode
from job_agent.runtime.semantic_checkpoint import (
    SEMANTIC_NODE_ORDER,
    SemanticCheckpointStore,
    SemanticGraphCheckpoint,
)
from job_agent.schemas import (
    ActionSuggestion,
    AnswerCardDeck,
    EvidenceItem,
    FitInput,
    FitVerdictResult,
    InterviewPrep,
    MockInterviewPlan,
    RawJob,
    StructuredJD,
    TargetedResume,
    VerdictRouteDecision,
)


DEFAULT_MAX_OUTPUT_TOKENS = 1024
EVIDENCE_MAX_OUTPUT_TOKENS = 4096
RESUME_MAX_OUTPUT_TOKENS = 4096
INTERVIEW_MAX_OUTPUT_TOKENS = 4096
ANSWER_CARDS_MAX_OUTPUT_TOKENS = 8192
SEMANTIC_SKILL_IDS = (
    "jd-analysis",
    "evidence-contract",
    "resume-tailoring",
    "interview-grilling",
    "answer-cards",
)
SEMANTIC_PROMPT_VERSIONS = {
    "jd": JD_PROMPT_VERSION,
    "evidence": EVIDENCE_PROMPT_VERSION,
    "resume": RESUME_PROMPT_VERSION,
    "interview": INTERVIEW_PROMPT_VERSION,
    "answer_cards": ANSWER_CARDS_PROMPT_VERSION,
}


class LangGraphExecutionState(TypedDict, total=False):
    """State channels owned by the native LangGraph execution engine."""

    selected_job: RawJob
    resume_text: str
    session_id: str
    run_id: str
    structured_jd: StructuredJD
    evidence: list[EvidenceItem]
    fit_input: FitInput
    fit_result: FitVerdictResult
    action: ActionSuggestion
    verdict_route: VerdictRouteDecision
    resume_tailoring: TargetedResume
    interview_prep: InterviewPrep
    answer_cards: AnswerCardDeck
    mock_interview_plan: MockInterviewPlan
    provider_traces: list[ProviderTrace]
    runtime_mode: str
    llm_call_count: int
    queue_wait_ms: int
    cache_hit: bool
    resumed_from_checkpoint: bool


class AgentGraphState(LangGraphExecutionState, total=False):
    """Application state, including checkpoint metadata outside LangGraph."""

    checkpoint_id: str
    _checkpoint_completed_node: str


class AgentGraphRuntime:
    topology = (
        "jd_structurer",
        "evidence_mapping",
        "fit_verdict",
        "verdict_route",
        "resume_tailoring",
        "interview_prep",
        "answer_cards",
        "mock_interview",
    )

    def __init__(
        self,
        *,
        registry,
        harness: LLMHarness,
        fit_backend: FitVerdictBackend | None = None,
        policy: NodePolicy | None = None,
        coordinator: ExecutionCoordinator | None = None,
        runtime_mode: RuntimeMode = RuntimeMode.AGENT_API,
        checkpoint_store: SemanticCheckpointStore | None = None,
        forbidden_output_strings: tuple[str, ...] = (),
        progress_callback: Callable[[str], None] | None = None,
    ) -> None:
        if runtime_mode == RuntimeMode.OFFLINE_RULE:
            raise ValueError("AgentGraphRuntime requires an agent runtime mode")
        self.registry = registry
        self.harness = harness
        self.fit_backend = fit_backend or RuleFitVerdictBackend()
        self.policy = policy or NodePolicy(allow_rule_fallback=False)
        self.coordinator = coordinator or ExecutionCoordinator()
        self.runtime_mode = runtime_mode
        self.checkpoint_store = checkpoint_store
        self.progress_callback = progress_callback
        self.forbidden_output_strings = tuple(
            value for value in forbidden_output_strings if value
        )
        self.jd_agent = JDStructurerAgent(registry=registry, harness=harness, policy=self.policy)
        self.evidence_agent = EvidenceMappingAgent(
            registry=registry,
            harness=harness,
            policy=self._node_output_policy(self.policy, EVIDENCE_MAX_OUTPUT_TOKENS),
        )
        self.resume_agent = ResumeTailoringAgent(
            registry=registry,
            harness=harness,
            policy=self._node_output_policy(self.policy, RESUME_MAX_OUTPUT_TOKENS),
        )
        self.interview_agent = InterviewPrepAgent(
            registry=registry,
            harness=harness,
            policy=self._node_output_policy(self.policy, INTERVIEW_MAX_OUTPUT_TOKENS),
        )
        self.answer_cards_agent = AnswerCardsAgent(
            registry=registry,
            harness=harness,
            policy=self._node_output_policy(
                self.policy, ANSWER_CARDS_MAX_OUTPUT_TOKENS
            ),
        )
        self._compiled_graph = self._build_langgraph_if_available()
        self.engine_name = (
            "equivalent_state_graph_checkpointed"
            if checkpoint_store is not None
            else "langgraph" if self._compiled_graph is not None else "equivalent_state_graph"
        )

    @staticmethod
    def _node_output_policy(policy: NodePolicy, max_output_tokens: int) -> NodePolicy:
        if policy.max_output_tokens != DEFAULT_MAX_OUTPUT_TOKENS:
            return policy
        return policy.model_copy(update={"max_output_tokens": max_output_tokens})

    def _notify_progress(self, stage: str) -> None:
        if self.progress_callback is not None:
            self.progress_callback(stage)

    def run_selected_job(
        self,
        *,
        selected_job: RawJob,
        resume_text: str,
        session_id: str,
        run_id: str,
        resume: bool = False,
    ) -> JobAgentState:
        key = self._idempotency_key(
            selected_job=selected_job,
            resume_text=resume_text,
            session_id=session_id,
        )
        state, queue_wait_ms, cache_hit = self.coordinator.execute(
            session_id=session_id,
            idempotency_key=key,
            operation=lambda: self._execute_graph(
                selected_job=selected_job,
                resume_text=resume_text,
                session_id=session_id,
                run_id=run_id,
                resume=resume,
            ),
        )
        result = copy.deepcopy(state)
        result["queue_wait_ms"] = queue_wait_ms
        result["cache_hit"] = cache_hit
        return result

    def _execute_graph(
        self,
        *,
        selected_job: RawJob,
        resume_text: str,
        session_id: str,
        run_id: str,
        resume: bool,
    ) -> JobAgentState:
        audit_start = len(self.harness.audit_events)
        initial: AgentGraphState = {
            "selected_job": selected_job,
            "resume_text": resume_text,
            "session_id": session_id,
            "run_id": run_id,
            "provider_traces": [],
            "runtime_mode": self.runtime_mode.value,
        }
        if self.checkpoint_store is not None:
            state = self._checkpoint_initial_state(
                initial,
                selected_job=selected_job,
                resume_text=resume_text,
                session_id=session_id,
                run_id=run_id,
                resume=resume,
            )
            state = self._invoke_checkpointed(state)
            self.checkpoint_store.delete()
        elif resume:
            raise ValueError("resume requires a semantic checkpoint store")
        elif self._compiled_graph is not None:
            state = self._compiled_graph.invoke(initial)
        else:
            state = self._invoke_equivalent(initial)
        state["llm_call_count"] = len(state.get("provider_traces", []))
        state["agent_audit_events"] = list(self.harness.audit_events[audit_start:])
        return state  # type: ignore[return-value]

    def _checkpoint_initial_state(
        self,
        initial: AgentGraphState,
        *,
        selected_job: RawJob,
        resume_text: str,
        session_id: str,
        run_id: str,
        resume: bool,
    ) -> AgentGraphState:
        if not resume:
            return initial
        assert self.checkpoint_store is not None
        checkpoint = self.checkpoint_store.load()
        input_hash, skill_versions, prompt_versions = self._checkpoint_identity(
            selected_job=selected_job,
            resume_text=resume_text,
            session_id=session_id,
        )
        if checkpoint.session_id != session_id or checkpoint.run_id != run_id:
            raise CheckpointError("semantic_checkpoint_run_identity_mismatch")
        if checkpoint.input_hash != input_hash:
            raise CheckpointError("semantic_checkpoint_input_mismatch")
        if checkpoint.skill_versions != skill_versions:
            raise CheckpointError("semantic_checkpoint_skill_version_mismatch")
        if checkpoint.prompt_versions != prompt_versions:
            raise CheckpointError("semantic_checkpoint_prompt_version_mismatch")
        restored = self._restore_checkpoint_state(initial, checkpoint)
        restored["resumed_from_checkpoint"] = True
        restored["checkpoint_id"] = checkpoint.checkpoint_id
        return restored

    def _invoke_checkpointed(self, state: AgentGraphState) -> AgentGraphState:
        nodes = {
            "jd_structurer": self._jd_node,
            "evidence_mapping": self._evidence_node,
            "fit_verdict": self._fit_node,
            "verdict_route": self._verdict_route_node,
            "resume_tailoring": self._resume_node,
            "interview_prep": self._interview_node,
            "answer_cards": self._answer_cards_node,
            "mock_interview": self._mock_interview_node,
        }
        completed_node = self._completed_checkpoint_node(state)
        start_index = 0 if completed_node is None else SEMANTIC_NODE_ORDER.index(completed_node) + 1
        if state.get("verdict_route") is not None and state["verdict_route"].gate == "stop_and_reselect":
            return state
        for node_name in SEMANTIC_NODE_ORDER[start_index:]:
            state.update(nodes[node_name](state))
            self._save_semantic_checkpoint(state, completed_node=node_name)
            if node_name == "verdict_route" and state["verdict_route"].gate == "stop_and_reselect":
                break
        return state

    @staticmethod
    def _completed_checkpoint_node(state: AgentGraphState) -> str | None:
        checkpoint_id = state.get("checkpoint_id")
        if checkpoint_id is None:
            return None
        completed = state.get("_checkpoint_completed_node")  # type: ignore[typeddict-item]
        return str(completed) if completed is not None else None

    def _save_semantic_checkpoint(self, state: AgentGraphState, *, completed_node: str) -> None:
        assert self.checkpoint_store is not None
        input_hash, skill_versions, prompt_versions = self._checkpoint_identity(
            selected_job=state["selected_job"],
            resume_text=state["resume_text"],
            session_id=state["session_id"],
        )
        checkpoint = SemanticGraphCheckpoint.create(
            session_id=state["session_id"],
            run_id=state["run_id"],
            input_hash=input_hash,
            completed_node=completed_node,
            skill_versions=skill_versions,
            prompt_versions=prompt_versions,
            state=self._checkpoint_state_payload(state),
            provider_traces=list(state.get("provider_traces", [])),
        )
        self.checkpoint_store.save(checkpoint)

    @staticmethod
    def _checkpoint_state_payload(state: AgentGraphState) -> dict[str, Any]:
        excluded = {
            "selected_job",
            "resume_text",
            "session_id",
            "run_id",
            "provider_traces",
            "agent_audit_events",
            "queue_wait_ms",
            "cache_hit",
            "resumed_from_checkpoint",
            "checkpoint_id",
            "_checkpoint_completed_node",
        }
        return {
            key: value.model_dump(mode="json") if hasattr(value, "model_dump") else [
                item.model_dump(mode="json") if hasattr(item, "model_dump") else item
                for item in value
            ] if isinstance(value, list) else value
            for key, value in state.items()
            if key not in excluded
        }

    @staticmethod
    def _restore_checkpoint_state(
        initial: AgentGraphState,
        checkpoint: SemanticGraphCheckpoint,
    ) -> AgentGraphState:
        state: AgentGraphState = dict(initial)  # type: ignore[assignment]
        parsers = {
            "structured_jd": StructuredJD,
            "fit_input": FitInput,
            "fit_result": FitVerdictResult,
            "action": ActionSuggestion,
            "verdict_route": VerdictRouteDecision,
            "resume_tailoring": TargetedResume,
            "interview_prep": InterviewPrep,
            "answer_cards": AnswerCardDeck,
            "mock_interview_plan": MockInterviewPlan,
        }
        for key, value in checkpoint.state.items():
            if key == "evidence":
                state[key] = [EvidenceItem.model_validate(item) for item in value]  # type: ignore[literal-required]
            elif key in parsers:
                state[key] = parsers[key].model_validate(value)  # type: ignore[literal-required]
            else:
                state[key] = value  # type: ignore[literal-required]
        state["provider_traces"] = list(checkpoint.provider_traces)
        state["_checkpoint_completed_node"] = checkpoint.completed_node  # type: ignore[typeddict-unknown-key]
        return state

    def _invoke_equivalent(self, state: AgentGraphState) -> AgentGraphState:
        for node in (
            self._jd_node,
            self._evidence_node,
            self._fit_node,
            self._verdict_route_node,
        ):
            state.update(node(state))
        if state["verdict_route"].gate == "stop_and_reselect":
            return state
        for node in (
            self._resume_node,
            self._interview_node,
            self._answer_cards_node,
            self._mock_interview_node,
        ):
            state.update(node(state))
        return state

    def _build_langgraph_if_available(self):
        try:
            from langgraph.graph import END, START, StateGraph
        except ImportError:
            return None
        # checkpoint_id is public application metadata, but recent LangGraph
        # versions reserve that channel name. Keep it in AgentGraphState for
        # persisted-session compatibility while excluding it from the native
        # graph schema.
        builder = StateGraph(LangGraphExecutionState)
        builder.add_node("jd_structurer", self._jd_node)
        builder.add_node("evidence_mapping", self._evidence_node)
        builder.add_node("fit_verdict", self._fit_node)
        builder.add_node("verdict_route", self._verdict_route_node)
        builder.add_node("resume_tailoring", self._resume_node)
        builder.add_node("interview_prep", self._interview_node)
        builder.add_node("answer_cards", self._answer_cards_node)
        builder.add_node("mock_interview", self._mock_interview_node)
        builder.add_edge(START, "jd_structurer")
        builder.add_edge("jd_structurer", "evidence_mapping")
        builder.add_edge("evidence_mapping", "fit_verdict")
        builder.add_edge("fit_verdict", "verdict_route")
        builder.add_conditional_edges(
            "verdict_route",
            lambda state: "stop" if state["verdict_route"].gate == "stop_and_reselect" else "continue",
            {"stop": END, "continue": "resume_tailoring"},
        )
        builder.add_edge("resume_tailoring", "interview_prep")
        builder.add_edge("interview_prep", "answer_cards")
        builder.add_edge("answer_cards", "mock_interview")
        builder.add_edge("mock_interview", END)
        return builder.compile()

    def _jd_node(self, state: LangGraphExecutionState) -> dict[str, Any]:
        self._notify_progress("jd_structurer")
        trace_start = len(self.harness.traces)
        result = self.jd_agent.run(
            state["selected_job"],
            session_id=state["session_id"],
            run_id=state["run_id"],
            forbidden_output_strings=self.forbidden_output_strings,
        )
        return {
            "structured_jd": result.value,
            "provider_traces": [
                *state.get("provider_traces", []),
                *(self.harness.traces[trace_start:] or [result.trace]),
            ],
        }

    def _evidence_node(self, state: LangGraphExecutionState) -> dict[str, Any]:
        self._notify_progress("evidence_mapping")
        trace_start = len(self.harness.traces)
        result = self.evidence_agent.run(
            state["structured_jd"],
            resume_text=state["resume_text"],
            session_id=state["session_id"],
            run_id=state["run_id"],
            forbidden_output_strings=self.forbidden_output_strings,
        )
        return {
            "evidence": result.value.items,
            "provider_traces": [
                *state.get("provider_traces", []),
                *(self.harness.traces[trace_start:] or [result.trace]),
            ],
        }

    def _fit_node(self, state: LangGraphExecutionState) -> dict[str, Any]:
        fit_input = FitInput(
            role_type=state["structured_jd"].role_type,
            requirements=state["structured_jd"].must_have,
            evidence=state["evidence"],
            toy_signals=[],
        )
        return {"fit_input": fit_input, "fit_result": self.fit_backend.evaluate(fit_input)}

    @staticmethod
    def _verdict_route_node(state: LangGraphExecutionState) -> dict[str, Any]:
        return {
            "action": suggest_action(state["fit_result"]),
            "verdict_route": build_verdict_route(state["fit_result"]),
        }

    def _resume_node(self, state: LangGraphExecutionState) -> dict[str, Any]:
        self._notify_progress("resume_tailoring")
        trace_start = len(self.harness.traces)
        result = self.resume_agent.run(
            state["structured_jd"],
            evidence=state["evidence"],
            resume_text=state["resume_text"],
            session_id=state["session_id"],
            run_id=state["run_id"],
            forbidden_output_strings=self.forbidden_output_strings,
        )
        return {
            "resume_tailoring": result.value,
            "provider_traces": [
                *state.get("provider_traces", []),
                *(self.harness.traces[trace_start:] or [result.trace]),
            ],
        }

    def _interview_node(self, state: LangGraphExecutionState) -> dict[str, Any]:
        self._notify_progress("interview_prep")
        trace_start = len(self.harness.traces)
        result = self.interview_agent.run(
            state["structured_jd"],
            evidence=state["evidence"],
            fit_result=state["fit_result"],
            targeted_resume=state["resume_tailoring"],
            session_id=state["session_id"],
            run_id=state["run_id"],
            forbidden_output_strings=self.forbidden_output_strings,
        )
        return {
            "interview_prep": result.value,
            "provider_traces": [
                *state.get("provider_traces", []),
                *(self.harness.traces[trace_start:] or [result.trace]),
            ],
        }

    def _answer_cards_node(self, state: LangGraphExecutionState) -> dict[str, Any]:
        self._notify_progress("answer_cards")
        trace_start = len(self.harness.traces)
        result = self.answer_cards_agent.run(
            state["interview_prep"],
            evidence=state["evidence"],
            targeted_resume=state["resume_tailoring"],
            session_id=state["session_id"],
            run_id=state["run_id"],
            forbidden_output_strings=self.forbidden_output_strings,
        )
        return {
            "answer_cards": result.value,
            "provider_traces": [
                *state.get("provider_traces", []),
                *(self.harness.traces[trace_start:] or [result.trace]),
            ],
        }

    @staticmethod
    def _mock_interview_node(state: LangGraphExecutionState) -> dict[str, Any]:
        return {
            "mock_interview_plan": build_mock_interview_plan(
                interview_prep=state["interview_prep"],
                answer_cards=state["answer_cards"],
            )
        }

    def _idempotency_key(self, *, selected_job: RawJob, resume_text: str, session_id: str) -> str:
        input_hash, _, _ = self._checkpoint_identity(
            selected_job=selected_job,
            resume_text=resume_text,
            session_id=session_id,
        )
        return input_hash

    def _checkpoint_identity(
        self,
        *,
        selected_job: RawJob,
        resume_text: str,
        session_id: str,
    ) -> tuple[str, dict[str, str], dict[str, str]]:
        skill_versions = {
            skill_id: self.registry.get(skill_id).version
            for skill_id in SEMANTIC_SKILL_IDS
        }
        prompt_versions = dict(SEMANTIC_PROMPT_VERSIONS)
        payload = {
            "session_id": session_id,
            "job": selected_job.model_dump(mode="json"),
            "resume_sha256": hashlib.sha256(resume_text.encode("utf-8")).hexdigest(),
            "forbidden_output_sha256": [
                hashlib.sha256(value.encode("utf-8")).hexdigest()
                for value in self.forbidden_output_strings
            ],
            "skill_versions": skill_versions,
            "prompt_versions": prompt_versions,
        }
        encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest(), skill_versions, prompt_versions
