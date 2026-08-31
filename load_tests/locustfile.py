from __future__ import annotations

from uuid import uuid4

from locust import HttpUser, between, task


class ControlPlaneUser(HttpUser):
    """Measures HTTP Run creation only; it does not claim LLM throughput."""

    wait_time = between(0.01, 0.05)

    @task
    def create_run(self) -> None:
        unique = uuid4().hex
        payload = {
            "task_type": "semantic_job_flow",
            "session_id": f"load-session-{unique}",
            "input": {
                "selected_job": {
                    "job_id": f"load-job-{unique}",
                    "company": "Load Fixture",
                    "title": "Backend Agent Engineer",
                    "desc": "Controlled Mock Provider load scenario",
                    "url": f"https://example.invalid/load/{unique}",
                    "location": "Beijing",
                },
                "resume_ref": "artifact://demo/resume",
            },
            "provider_profile": "mock",
            "budget_profile": "quick",
        }
        with self.client.post(
            "/api/v1/runs",
            headers={
                "X-User-ID": "demo-user-a",
                "X-Request-ID": f"load-request-{unique}",
                "Idempotency-Key": f"load-idempotency-{unique}",
            },
            json=payload,
            name="POST /api/v1/runs",
            catch_response=True,
        ) as response:
            if response.status_code != 202:
                response.failure(f"expected 202, received {response.status_code}")
