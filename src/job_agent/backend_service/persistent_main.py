"""PostgreSQL-backed API entrypoint; the Worker runs as a separate process."""

from job_agent.backend_service.persistent_bootstrap import create_persistent_api_stack
from job_agent.backend_service.settings import BackendSettings


_stack = create_persistent_api_stack(BackendSettings.from_env())
app = _stack.app
