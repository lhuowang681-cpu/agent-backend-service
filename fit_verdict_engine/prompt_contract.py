from __future__ import annotations

import json

from job_agent.schemas import FitInput


FIT_VERDICT_SYSTEM_PROMPT = "You are a fit verdict classifier. Return only compact JSON matching the requested schema."


def build_fit_verdict_user_prompt(fit_input: FitInput) -> str:
    return "\n".join(
        [
            "Return only JSON matching this schema:",
            '{"verdict":"strong fit|weak fit|risky fit|not recommended","score":0.0,',
            '"coverage":0.0,"risk_level":"low|medium|high","need_human_review":true,',
            '"reason_codes":["short_code"],"explanation":"brief explanation"}',
            "Do not output thinking process, analysis, markdown, or commentary.",
            "Return only the final JSON object.",
            "Input:",
            json.dumps(fit_input.model_dump(mode="json"), ensure_ascii=False),
        ]
    )


def build_fit_verdict_messages(fit_input: FitInput) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": FIT_VERDICT_SYSTEM_PROMPT},
        {"role": "user", "content": build_fit_verdict_user_prompt(fit_input)},
    ]
