from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path


_QUESTION_HEADING = re.compile(r"^###\s+\d+\.\s+(?P<question>.+?)\s*$")
_FIELD = re.compile(r"^-\s+(?P<name>[^:]+):\s*(?P<value>.*)$")
_PRACTICE_TRANSLATIONS = {
    "Answer in 60 seconds without adding unsupported metrics.": (
        "控制在 60 秒内，不添加没有依据的指标或成果。"
    ),
    "Name the exact artifact that proves the claim.": (
        "说出能够证明这段经历的具体代码、日志、文档或实验结果。"
    ),
    "Keep the answer aligned with the 6 resume-safe claims.": (
        "回答只围绕简历中可以安全使用的事实，不扩大个人贡献。"
    ),
}
_RESUME_SAFE_PRACTICE = re.compile(
    r"^Keep the answer aligned with the \d+ resume-safe claims\.$",
    re.IGNORECASE,
)
_EVIDENCE_LABELS = {
    "C3": "匹配充分",
    "C2": "有直接材料支持",
    "C1": "只有部分材料支持",
    "C0": "暂未找到材料支持",
}


@dataclass
class InterviewPrepQuestion:
    question: str
    intent: str = ""
    risk: str = ""
    follow_ups: list[str] = field(default_factory=list)
    answer_lead: str = ""
    supporting_evidence: str = ""
    truth_boundary: str = ""
    practice_actions: list[str] = field(default_factory=list)


def _strip_answer_prefix(value: str) -> str:
    prefixes = (
        "I can discuss the related module cautiously:",
        "I can walk through the evidence-backed work:",
    )
    for prefix in prefixes:
        if value.startswith(prefix):
            value = value[len(prefix) :].strip()
            return _clean_internal_labels(
                f"建议围绕已有事实回答：{value.rstrip('.')}。"
            )
    return _clean_internal_labels(value)


def _strip_boundary_prefix(value: str) -> str:
    prefixes = (
        "Do not overclaim depth. Mention the risk explicitly:",
        "Do not overclaim beyond the available proof. Be ready to explain:",
    )
    for prefix in prefixes:
        if value.startswith(prefix):
            return _clean_internal_labels(value[len(prefix) :].strip())
    return _clean_internal_labels(value)


def _clean_internal_labels(value: str) -> str:
    for internal, user_label in _EVIDENCE_LABELS.items():
        value = value.replace(internal, user_label)
    value = value.replace("证据为只有部分材料支持", "当前只有部分材料支持")
    value = value.replace("证据为有直接材料支持", "当前有直接材料支持")
    replacements = (
        ("tool call schema", "工具调用参数结构"),
        ("function calling", "工具调用"),
        ("decision accuracy", "决策准确率"),
        ("schema invalid", "结构校验失败"),
        ("JSON/Schema valid", "JSON/结构校验通过率"),
        ("training pipeline", "训练流程"),
        ("reward model", "奖励模型"),
        ("reward design", "奖励设计"),
        ("post-training", "后训练"),
        ("skill exposure", "skill 暴露度"),
        ("prompt builder", "提示词构建器"),
        ("evaluator", "评估器"),
        ("100-step controlled run", "100 步受控训练"),
        ("skill executor", "skill 执行器"),
        ("Agent runtime", "Agent 运行时"),
        ("tool schema", "工具参数结构"),
    )
    for english, chinese in replacements:
        value = re.sub(re.escape(english), chinese, value, flags=re.IGNORECASE)
    word_replacements = {
        "pipeline": "流程",
        "runtime": "运行时",
        "trace": "运行记录",
        "prompt": "提示词",
        "case": "案例",
        "reward": "奖励",
        "schema": "参数结构",
    }
    for english, chinese in word_replacements.items():
        value = re.sub(
            rf"(?i)(?<![A-Za-z_]){english}(?![A-Za-z_])",
            chinese,
            value,
        )
    value = value.replace("运行时运行记录", "运行记录")
    return value


def _translate_practice_action(value: str) -> str:
    translated = _PRACTICE_TRANSLATIONS.get(value)
    if translated is not None:
        return translated
    if _RESUME_SAFE_PRACTICE.fullmatch(value.strip()):
        return "回答只围绕简历中可以安全使用的事实，不扩大个人贡献。"
    return _clean_internal_labels(value)


def _parse_sections(path: Path) -> list[tuple[str, dict[str, str], dict[str, list[str]]]]:
    if not path.is_file():
        return []
    sections: list[tuple[str, dict[str, str], dict[str, list[str]]]] = []
    question = ""
    fields: dict[str, str] = {}
    lists: dict[str, list[str]] = {}
    active_list = ""
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        heading = _QUESTION_HEADING.match(raw_line)
        if heading:
            if question:
                sections.append((question, fields, lists))
            question = heading.group("question").strip()
            fields = {}
            lists = {}
            active_list = ""
            continue
        if not question:
            continue
        field_match = _FIELD.match(raw_line)
        if field_match:
            name = field_match.group("name").strip()
            value = field_match.group("value").strip()
            if value:
                fields[name] = value
                active_list = ""
            else:
                active_list = name
                lists.setdefault(name, [])
            continue
        if active_list and raw_line.startswith("  - "):
            lists[active_list].append(raw_line[4:].strip())
    if question:
        sections.append((question, fields, lists))
    return sections


def load_interview_prep_questions(session_dir: Path) -> list[InterviewPrepQuestion]:
    session_dir = Path(session_dir)
    questions: dict[str, InterviewPrepQuestion] = {}
    order: list[str] = []

    for question, fields, lists in _parse_sections(
        session_dir / "07_interview_grilling.md"
    ):
        order.append(question)
        questions[question] = InterviewPrepQuestion(
            question=_clean_internal_labels(question),
            intent=_clean_internal_labels(fields.get("Intent", "")),
            risk=_clean_internal_labels(fields.get("Risk", "")),
            follow_ups=[
                _clean_internal_labels(item)
                for item in lists.get("Follow-ups", [])
            ],
        )

    for question, fields, lists in _parse_sections(session_dir / "08_answer_cards.md"):
        item = questions.setdefault(question, InterviewPrepQuestion(question=question))
        if question not in order:
            order.append(question)
        item.answer_lead = _strip_answer_prefix(fields.get("Short answer", ""))
        item.supporting_evidence = _clean_internal_labels(
            fields.get("Supporting evidence", "")
        )
        item.truth_boundary = _strip_boundary_prefix(
            fields.get("Truth boundary", "")
        )
        item.practice_actions = [
            _translate_practice_action(action)
            for action in lists.get("Practice prompts", [])
        ]
    return [questions[question] for question in order]
