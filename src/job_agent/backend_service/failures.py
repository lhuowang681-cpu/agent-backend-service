from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from random import random
from typing import Callable


class FailureDisposition(str, Enum):
    RETRY = "retry"
    FAIL = "fail"
    UNCERTAIN = "uncertain"


@dataclass(frozen=True)
class FailureDecision:
    disposition: FailureDisposition
    error_code: str
    retry_delay_seconds: float | None = None


class FailureClassifier:
    """Explicit bounded service retry policy; unknown errors fail closed."""

    RETRYABLE = frozenset(
        {
            "rate_limit",
            "timeout",
            "transport_error",
            "provider_rate_limit",
            "provider_timeout",
            "provider_transport_error",
        }
    )
    UNCERTAIN = frozenset(
        {
            "non_idempotent_execution_uncertain",
            "external_side_effect_outcome_unknown",
        }
    )
    EXPLICITLY_RESUMABLE = frozenset(
        {
            "worker_recovery_exhausted",
            "provider_retry_exhausted",
            "agent_io_failed",
        }
    )

    def __init__(
        self,
        *,
        max_attempts: int = 3,
        base_delay_seconds: float = 0.5,
        cap_delay_seconds: float = 30.0,
        jitter: Callable[[], float] = random,
    ) -> None:
        if max_attempts < 1:
            raise ValueError("max_attempts_must_be_positive")
        self.max_attempts = max_attempts
        self.base_delay_seconds = base_delay_seconds
        self.cap_delay_seconds = cap_delay_seconds
        self.jitter = jitter

    def classify(self, *, error_code: str, attempt_no: int) -> FailureDecision:
        if error_code in self.UNCERTAIN:
            return FailureDecision(FailureDisposition.UNCERTAIN, error_code)
        if error_code not in self.RETRYABLE:
            return FailureDecision(FailureDisposition.FAIL, error_code)
        if attempt_no >= self.max_attempts:
            return FailureDecision(
                FailureDisposition.FAIL,
                "provider_retry_exhausted",
            )
        exponential = min(
            self.cap_delay_seconds,
            self.base_delay_seconds * (2 ** max(attempt_no - 1, 0)),
        )
        factor = 0.5 + min(max(self.jitter(), 0.0), 1.0)
        return FailureDecision(
            FailureDisposition.RETRY,
            error_code,
            exponential * factor,
        )

    @classmethod
    def can_explicitly_resume(cls, error_code: str | None) -> bool:
        return error_code in cls.EXPLICITLY_RESUMABLE
