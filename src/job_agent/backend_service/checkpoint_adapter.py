from __future__ import annotations

from datetime import UTC, datetime

from job_agent.agent_runtime.checkpoint import AgentCheckpoint
from job_agent.agent_runtime.failures import CheckpointError
from job_agent.backend_service.contracts import ClaimedRun
from job_agent.backend_service.postgres_repository import PostgresRunRepository
from job_agent.runtime.semantic_checkpoint import SemanticGraphCheckpoint


class PostgresAgentCheckpointStore:
    """Existing domain Agent CheckpointStore protocol backed by a user-owned Run."""

    kind = "domain_agent_v1"

    def __init__(
        self,
        *,
        repository: PostgresRunRepository,
        claim: ClaimedRun | None,
    ) -> None:
        if claim is None:
            raise CheckpointError("checkpoint_attempt_fence_missing")
        self.repository = repository
        self.claim = claim
        self.user_id = claim.run.user_id
        self.run_id = claim.run.run_id

    def save(self, checkpoint: AgentCheckpoint) -> None:
        if checkpoint.state.run_id != self.run_id:
            raise CheckpointError("checkpoint_identity_mismatch")
        run = self.repository.get_for_user(user_id=self.user_id, run_id=self.run_id)
        if checkpoint.state.session_id != run.session_id:
            raise CheckpointError("checkpoint_identity_mismatch")
        self.repository.save_private_checkpoint(
            claim=self.claim,
            checkpoint_kind=self.kind,
            checkpoint_id=checkpoint.checkpoint_id,
            payload=checkpoint.model_dump(mode="json"),
        )

    def exists(self) -> bool:
        return self.repository.has_private_checkpoint(
            user_id=self.user_id,
            run_id=self.run_id,
            checkpoint_kind=self.kind,
        )

    def load(self) -> AgentCheckpoint:
        payload = self.repository.load_private_checkpoint(
            user_id=self.user_id,
            run_id=self.run_id,
            checkpoint_kind=self.kind,
        )
        try:
            checkpoint = AgentCheckpoint.model_validate(payload)
        except ValueError as exc:
            raise CheckpointError("checkpoint_invalid") from exc
        if checkpoint.state.run_id != self.run_id:
            raise CheckpointError("checkpoint_identity_mismatch")
        return checkpoint

    def delete(self) -> None:
        self.repository.delete_private_checkpoint(
            claim=self.claim,
            checkpoint_kind=self.kind,
        )


class PostgresSemanticCheckpointStore:
    """Existing semantic graph checkpoint contract backed by PostgreSQL JSONB."""

    kind = "semantic_graph_v1"

    def __init__(
        self,
        *,
        repository: PostgresRunRepository,
        claim: ClaimedRun | None,
    ) -> None:
        if claim is None:
            raise CheckpointError("checkpoint_attempt_fence_missing")
        self.repository = repository
        self.claim = claim
        self.user_id = claim.run.user_id
        self.run_id = claim.run.run_id

    def exists(self) -> bool:
        return self.repository.has_private_checkpoint(
            user_id=self.user_id,
            run_id=self.run_id,
            checkpoint_kind=self.kind,
        )

    def save(self, checkpoint: SemanticGraphCheckpoint) -> None:
        if checkpoint.run_id != self.run_id:
            raise CheckpointError("semantic_checkpoint_run_identity_mismatch")
        run = self.repository.get_for_user(user_id=self.user_id, run_id=self.run_id)
        if checkpoint.session_id != run.session_id:
            raise CheckpointError("semantic_checkpoint_run_identity_mismatch")
        self.repository.save_private_checkpoint(
            claim=self.claim,
            checkpoint_kind=self.kind,
            checkpoint_id=checkpoint.checkpoint_id,
            payload=checkpoint.model_dump(mode="json"),
            expires_at=datetime.fromisoformat(checkpoint.expires_at),
        )

    def load(self, *, now: datetime | None = None) -> SemanticGraphCheckpoint:
        payload = self.repository.load_private_checkpoint(
            user_id=self.user_id,
            run_id=self.run_id,
            checkpoint_kind=self.kind,
        )
        try:
            checkpoint = SemanticGraphCheckpoint.model_validate(payload)
        except ValueError as exc:
            raise CheckpointError("semantic_checkpoint_invalid") from exc
        if checkpoint.run_id != self.run_id:
            raise CheckpointError("semantic_checkpoint_run_identity_mismatch")
        current = now or datetime.now(UTC)
        if current.tzinfo is None:
            current = current.replace(tzinfo=UTC)
        if datetime.fromisoformat(checkpoint.expires_at) <= current.astimezone(UTC):
            raise CheckpointError("semantic_checkpoint_expired")
        return checkpoint

    def delete(self) -> None:
        self.repository.delete_private_checkpoint(
            claim=self.claim,
            checkpoint_kind=self.kind,
        )
