"""Deployable Mock-Provider tracer entrypoint.

Run with:
    uvicorn job_agent.backend_service.main:app
"""

from job_agent.backend_service.bootstrap import create_tracer_stack


_stack = create_tracer_stack(allowed_user_ids=("demo-user-a", "demo-user-b"))
_stack.snapshots.put(
    user_id="demo-user-a",
    resume_ref="artifact://demo/resume",
    resume_text="LoRA loss checkpoint evaluation exposure",
)
_stack.snapshots.put(
    user_id="demo-user-b",
    resume_ref="artifact://demo/resume",
    resume_text="Backend API testing and queue reliability experience",
)

app = _stack.app
