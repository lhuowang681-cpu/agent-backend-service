from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


DEFAULT_API_KEY_ENV = "JOB_AGENT_LIVE_API_KEY"
DEFAULT_SKILL_ROOT_ENV = "LLM_INTERN_SKILL_ROOT"


@dataclass(frozen=True)
class RuntimeDefaults:
    api_key: str | None = field(default=None, repr=False)
    api_key_source: str | None = None
    base_url: str = "https://open.bigmodel.cn/api/anthropic"
    model: str = "glm-4.7"
    skill_root: str | None = None
    skill_root_source: str | None = None


LocalRuntimeDefaults = RuntimeDefaults


def workspace_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _valid_skill_root(path: Path) -> bool:
    return (
        path.is_dir()
        and (path / "SKILL.md").is_file()
        and (path / "release-manifest.txt").is_file()
    )


def load_local_runtime_defaults(
    *,
    root: Path | None = None,
    skill_root_override: str | None = None,
) -> RuntimeDefaults:
    """Resolve local-only runtime values without logging or persisting secrets."""

    resolved_root = Path(root) if root is not None else workspace_root()
    api_key = os.environ.get(DEFAULT_API_KEY_ENV, "").strip() or None
    api_key_source = DEFAULT_API_KEY_ENV if api_key else None

    explicit_skill_root = (skill_root_override or "").strip()
    if explicit_skill_root and _valid_skill_root(Path(explicit_skill_root)):
        skill_root = str(Path(explicit_skill_root))
        skill_root_source = "页面设置"
    else:
        env_skill_root = os.environ.get(DEFAULT_SKILL_ROOT_ENV, "").strip()
        env_path = Path(env_skill_root) if env_skill_root else None
        default_path = resolved_root / "skill-references" / "llm-intern-skill"
        if env_path is not None and _valid_skill_root(env_path):
            skill_root = str(env_path)
            skill_root_source = f"环境变量 {DEFAULT_SKILL_ROOT_ENV}"
        elif _valid_skill_root(default_path):
            skill_root = str(default_path)
            skill_root_source = "本机技能包"
        else:
            skill_root = explicit_skill_root or env_skill_root or None
            skill_root_source = None

    return RuntimeDefaults(
        api_key=api_key,
        api_key_source=api_key_source,
        base_url=os.environ.get(
            "JOB_AGENT_LIVE_BASE_URL",
            "https://open.bigmodel.cn/api/anthropic",
        ),
        model=os.environ.get("JOB_AGENT_LIVE_MODEL", "glm-4.7"),
        skill_root=skill_root,
        skill_root_source=skill_root_source,
    )
