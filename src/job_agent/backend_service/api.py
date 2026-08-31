from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Collection
from contextlib import asynccontextmanager
from functools import partial
from typing import Protocol
from uuid import uuid4

import psycopg
from fastapi import FastAPI, Header, HTTPException, Query, Request, status
from fastapi.responses import JSONResponse, PlainTextResponse
from psycopg_pool import PoolTimeout

from job_agent.backend_service.contracts import (
    ApprovalDecisionRequest,
    CreateRunRequest,
    CreateRunResponse,
    EventPage,
    RequestContext,
    ResumeRunRequest,
    RunMutationResponse,
    RunDebugView,
    RunView,
)
from job_agent.backend_service.repository import (
    IdempotencyConflictError,
    QueueCapacityExceededError,
    RunNotFoundError,
)
from job_agent.backend_service.postgres_repository import (
    ApprovalPersistenceError,
    RunStateConflictError,
)
from job_agent.backend_service.service import IdempotencyKeyRequiredError, RunService
from job_agent.backend_service.metrics import (
    MetricsRecorder,
    NullMetricsRecorder,
    PrometheusMetricsService,
    Timer,
)


class ServiceLifecycle(Protocol):
    def start(self) -> None: ...

    def stop(self) -> None: ...


class ConfiguredHeaderIdentityAdapter:
    """Controlled dev/test identity seam; never accepts arbitrary users."""

    def __init__(self, allowed_user_ids: Collection[str], *, production: bool = False) -> None:
        if production:
            raise ValueError("unsigned_header_identity_is_forbidden_in_production")
        self._allowed = frozenset(allowed_user_ids)
        if not self._allowed:
            raise ValueError("at least one configured user is required")

    def resolve(self, *, user_id: str | None, request_id: str | None) -> RequestContext:
        if user_id is None or user_id not in self._allowed:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail={"code": "identity_required"},
            )
        return RequestContext(user_id=user_id, request_id=request_id or str(uuid4()))


def create_app(
    *,
    service: RunService,
    worker: ServiceLifecycle | None,
    identity_adapter: ConfiguredHeaderIdentityAdapter,
    metrics_recorder: MetricsRecorder | None = None,
    metrics_service: PrometheusMetricsService | None = None,
) -> FastAPI:
    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        if worker is not None:
            worker.start()
        try:
            yield
        finally:
            if worker is not None:
                worker.stop()

    app = FastAPI(title="Job Agent Backend Service", version="0.1.0", lifespan=lifespan)
    recorder = metrics_recorder or NullMetricsRecorder()

    @app.exception_handler(psycopg.Error)
    @app.exception_handler(PoolTimeout)
    async def database_unavailable(_request: Request, _exc: Exception) -> JSONResponse:
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content={"detail": {"code": "database_unavailable"}},
            headers={"Retry-After": "1"},
        )

    @app.middleware("http")
    async def record_request_metrics(request: Request, call_next):
        timer = Timer()
        status_code = 500
        try:
            response = await call_next(request)
            status_code = response.status_code
            return response
        finally:
            route = request.scope.get("route")
            route_path = getattr(route, "path", request.url.path)
            asyncio.get_running_loop().run_in_executor(
                None,
                partial(
                    recorder.record_api_request,
                    method=request.method,
                    route=route_path,
                    status_code=status_code,
                    duration_seconds=timer.elapsed(),
                ),
            )

    @app.get("/health/live", include_in_schema=False)
    def health_live() -> dict[str, str]:
        return {"status": "ok"}

    if metrics_service is not None:

        @app.get("/metrics", include_in_schema=False, response_class=PlainTextResponse)
        def metrics() -> str:
            return metrics_service.render()

    def context_from_request(request: Request) -> RequestContext:
        return identity_adapter.resolve(
            user_id=request.headers.get("X-User-ID"),
            request_id=request.headers.get("X-Request-ID"),
        )

    @app.post(
        "/api/v1/runs",
        response_model=CreateRunResponse,
        status_code=status.HTTP_202_ACCEPTED,
    )
    def create_run(
        request_body: CreateRunRequest,
        request: Request,
        idempotency_key: str | None = Header(default=None, alias="Idempotency-Key"),
    ) -> CreateRunResponse:
        context = context_from_request(request)
        try:
            return service.create_run(
                context,
                request_body,
                idempotency_key=idempotency_key,
            )
        except IdempotencyKeyRequiredError as exc:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail={"code": str(exc)},
            ) from exc
        except IdempotencyConflictError as exc:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail={"code": str(exc)},
            ) from exc
        except QueueCapacityExceededError as exc:
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail={
                    "code": "queued_run_capacity_exceeded",
                    "scope": exc.scope,
                },
                headers={
                    "Retry-After": str(max(1, int(exc.retry_after_seconds))),
                },
            ) from exc

    @app.get("/api/v1/runs/{run_id}", response_model=RunView)
    def get_run(run_id: str, request: Request) -> RunView:
        context = context_from_request(request)
        try:
            return service.get_run(context, run_id)
        except RunNotFoundError as exc:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail={"code": "resource_not_found"},
            ) from exc

    @app.get("/api/v1/runs/{run_id}/events", response_model=EventPage)
    def list_events(
        run_id: str,
        request: Request,
        after: int = Query(default=0, ge=0),
        limit: int = Query(default=100, ge=1, le=500),
    ) -> EventPage:
        context = context_from_request(request)
        try:
            return service.list_events(
                context,
                run_id,
                after=after,
                limit=limit,
            )
        except RunNotFoundError as exc:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail={"code": "resource_not_found"},
            ) from exc

    @app.get("/api/v1/runs/{run_id}/debug", response_model=RunDebugView)
    def get_debug_view(run_id: str, request: Request) -> RunDebugView:
        context = context_from_request(request)
        try:
            return service.get_debug_view(context, run_id)
        except RunNotFoundError as exc:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail={"code": "resource_not_found"},
            ) from exc

    def mutate_approval(
        *,
        run_id: str,
        request: Request,
        body: ApprovalDecisionRequest,
        approved: bool,
    ) -> RunMutationResponse:
        context = context_from_request(request)
        try:
            return service.decide_approval(context, run_id, body, approved=approved)
        except RunNotFoundError as exc:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail={"code": "resource_not_found"},
            ) from exc
        except (ApprovalPersistenceError, RunStateConflictError) as exc:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail={"code": str(exc)},
            ) from exc

    @app.post(
        "/api/v1/runs/{run_id}/approve",
        response_model=RunMutationResponse,
        status_code=status.HTTP_202_ACCEPTED,
    )
    def approve_run(
        run_id: str,
        request_body: ApprovalDecisionRequest,
        request: Request,
    ) -> RunMutationResponse:
        return mutate_approval(
            run_id=run_id,
            request=request,
            body=request_body,
            approved=True,
        )

    @app.post(
        "/api/v1/runs/{run_id}/reject",
        response_model=RunMutationResponse,
        status_code=status.HTTP_202_ACCEPTED,
    )
    def reject_run(
        run_id: str,
        request_body: ApprovalDecisionRequest,
        request: Request,
    ) -> RunMutationResponse:
        return mutate_approval(
            run_id=run_id,
            request=request,
            body=request_body,
            approved=False,
        )

    @app.post(
        "/api/v1/runs/{run_id}/resume",
        response_model=RunMutationResponse,
        status_code=status.HTTP_202_ACCEPTED,
    )
    def resume_run(
        run_id: str,
        request_body: ResumeRunRequest,
        request: Request,
    ) -> RunMutationResponse:
        context = context_from_request(request)
        try:
            return service.resume_run(
                context,
                run_id,
                expected_status_version=request_body.expected_status_version,
            )
        except RunNotFoundError as exc:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail={"code": "resource_not_found"},
            ) from exc
        except RunStateConflictError as exc:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail={"code": str(exc)},
            ) from exc

    return app
