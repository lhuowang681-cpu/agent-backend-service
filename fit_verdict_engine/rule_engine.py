from __future__ import annotations

from job_agent.nodes.fit_verdict import evaluate_fit
from job_agent.schemas import FitInput, Verdict


def label_fit_input(fit_input: FitInput) -> Verdict:
    return evaluate_fit(fit_input).verdict
