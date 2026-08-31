from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

from job_agent.ui.real_interview_rounds import list_real_interview_rounds


@dataclass(frozen=True)
class RunRound:
    """单轮模拟面试的摘要条目。"""
    timestamp: str       # "2026-07-24T143000"
    path: Path
    average_score: float
    round_number: int
    has_ai_evaluation: bool = False


def list_run_rounds(session_dir: Path) -> list[RunRound]:
    """列出 session_dir 下所有 10_mock_interview_run_*.json，按时间戳倒序。

    损坏文件 / 非标准时间戳文件名被跳过（不抛异常）。
    """
    ts_pattern = re.compile(
        r"^10_mock_interview_run_(\d{4}-\d{2}-\d{2}T\d{6})"
        r"(?:_(\d+))?\.json$"
    )
    parsed: list[tuple[str, Path, float, bool]] = []
    for path in session_dir.glob("10_mock_interview_run_*.json"):
        m = ts_pattern.match(path.name)
        if not m:
            continue
        timestamp = m.group(1)
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            avg = payload.get("result", {}).get("average_score", 0.0)
            from job_agent.ui.mock_interview_evaluation_loop import (
                load_mock_interview_evaluation,
            )

            evaluation = load_mock_interview_evaluation(path)
            has_ai_evaluation = evaluation is not None
            if evaluation is not None:
                avg = evaluation.overall_score
        except (json.JSONDecodeError, OSError):
            continue
        parsed.append((timestamp, path, float(avg), has_ai_evaluation))
    chronological = sorted(parsed, key=lambda item: (item[0], item[1].name))
    round_number = {
        path: index
        for index, (_, path, _, _) in enumerate(chronological, start=1)
    }
    return [
        RunRound(
            timestamp=timestamp,
            path=path,
            average_score=average_score,
            round_number=round_number[path],
            has_ai_evaluation=has_ai_evaluation,
        )
        for timestamp, path, average_score, has_ai_evaluation in reversed(
            chronological
        )
    ]


def load_run_payload(path: Path) -> dict | None:
    """加载单个 run JSON → {"result": ..., "history": [...]}。损坏或缺失返回 None。"""
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None


def build_model_answer_map(session_dir: Path) -> dict[str, str]:
    """从 08_answer_cards.md 解析 → {requirement_id: short_answer}。

    复用 session_orchestrator._parse_answer_cards 的解析逻辑。
    解析失败返回空 dict（不抛异常）。
    """
    cards_path = session_dir / "08_answer_cards.md"
    if not cards_path.exists():
        return {}
    try:
        from job_agent.session_orchestrator import _parse_answer_cards
        deck = _parse_answer_cards(cards_path.read_text(encoding="utf-8"))
        return {card.requirement_id: card.short_answer for card in deck.cards}
    except Exception:
        return {}


def build_question_model_answer_map(session_dir: Path) -> dict[str, str]:
    """从 09 plan + 08 answer_cards 解析 → {question_id: short_answer}。

    用于复盘 section 的"我的答案 vs model answer"对比渲染。
    需要 09 和 08 同时存在；任一缺失或解析失败返回空 dict。
    """
    plan_path = session_dir / "09_mock_interview_plan.md"
    cards_path = session_dir / "08_answer_cards.md"
    if not plan_path.exists() or not cards_path.exists():
        return {}
    try:
        from job_agent.session_orchestrator import _parse_answer_cards, _parse_mock_plan
        plan = _parse_mock_plan(plan_path.read_text(encoding="utf-8"))
        deck = _parse_answer_cards(cards_path.read_text(encoding="utf-8"))
        rid_to_answer = {card.requirement_id: card.short_answer for card in deck.cards}
        return {
            q.question_id: rid_to_answer[q.requirement_id]
            for q in plan.questions
            if q.requirement_id in rid_to_answer
        }
    except Exception:
        return {}


@dataclass(frozen=True)
class GlobalInterviewEntry:
    """跨 session 面试历史汇总条目。"""
    session_dir: Path
    company: str
    title: str
    date: str               # ISO date "YYYY-MM-DD"（最新一轮的时间戳日期）
    type: str                # "模拟面试" / "真实面经" / "模拟面试 + 真实面经"
    round_count: int         # 模拟面试轮数
    latest_avg_score: float | None


def collect_global_history(output_dir: Path) -> list[GlobalInterviewEntry]:
    """扫描 output_dir/sessions/*/ 聚合所有面试记录（模拟 + 真实面经）。

    每个 session 目录读取 session_state.json 获取 company/title，
    扫描 10_mock_interview_run_*.json 获取模拟记录，
    检查 13_post_interview_review.md 判断是否有真实面经。

    无任何面试记录的 session 不会出现在结果中。
    损坏的 session_state / run 文件被跳过（不崩全扫描）。
    """
    sessions_dir = output_dir / "sessions"
    if not sessions_dir.exists():
        return []

    entries: list[GlobalInterviewEntry] = []
    for session_dir in sorted(sessions_dir.iterdir(), reverse=True):
        if not session_dir.is_dir():
            continue

        # 读 session_state.json 取元数据
        company = "unknown"
        title = "unknown"
        state_path = session_dir / "session_state.json"
        if state_path.exists():
            try:
                state = json.loads(state_path.read_text(encoding="utf-8"))
                company = state.get("company", "unknown") or "unknown"
                title = state.get("title", "unknown") or "unknown"
            except (json.JSONDecodeError, OSError):
                pass

        real_rounds = list_real_interview_rounds(session_dir)
        has_review = bool(real_rounds) or (
            session_dir / "13_post_interview_review.md"
        ).exists()

        # 扫描模拟面试轮次
        rounds = list_run_rounds(session_dir)

        if not rounds and not has_review:
            continue  # 无任何面试记录，跳过

        # date：取最新 run 的时间戳日期，fallback 为空（无日期）
        if rounds:
            date = rounds[0].timestamp[:10]  # "2026-07-24"
            latest_avg = rounds[0].average_score
        else:
            date = real_rounds[0].created_at[:10] if real_rounds else ""
            latest_avg = None

        # type 标签
        if rounds and has_review:
            type_label = "模拟面试 + 真实面经"
        elif has_review:
            type_label = "真实面经"
        else:
            type_label = "模拟面试"

        entries.append(GlobalInterviewEntry(
            session_dir=session_dir,
            company=company,
            title=title,
            date=date,
            type=type_label,
            round_count=len(rounds),
            latest_avg_score=latest_avg,
        ))

    return entries


@dataclass(frozen=True)
class TimelineItem:
    """面试日志时间线中的单条记录（一周一轮/面试 = 一条）。"""
    session_dir: Path
    company: str
    title: str
    date: str               # "YYYY-MM-DD"
    item_type: str           # "mock" | "real"
    round_label: str         # "模拟第N轮"（mock）/ ""（real）
    run_path: Path | None    # 10_mock_interview_run_*.json 路径（mock）
    average_score: float | None  # 均分（mock）
    review_path: Path | None     # 13_post_interview_review.md 路径（real）
    ai_review_path: Path | None = None


def build_timeline(output_dir: Path) -> list[TimelineItem]:
    """从所有 session 构建面试日志时间线，按日期倒序排列。

    每个 session 的每轮模拟面试展开为独立 TimelineItem；
    真实面经（13_post_interview_review.md）也展开为独立 TimelineItem。
    """
    entries = collect_global_history(output_dir)
    items: list[TimelineItem] = []

    for entry in entries:
        # 模拟轮次：每个 run 文件一条
        rounds = list_run_rounds(entry.session_dir)
        for r in rounds:
            items.append(TimelineItem(
                session_dir=entry.session_dir,
                company=entry.company,
                title=entry.title,
                date=r.timestamp[:10],
                item_type="mock",
                round_label=f"模拟第{r.round_number}轮",
                run_path=r.path,
                average_score=r.average_score,
                review_path=None,
                ai_review_path=(
                    (
                        entry.session_dir
                        / (
                            "11_mock_interview_evaluation_"
                            f"{r.path.stem.removeprefix('10_mock_interview_run_')}.json"
                        )
                    )
                    if r.has_ai_evaluation
                    else None
                ),
            ))

        real_rounds = list_real_interview_rounds(entry.session_dir)
        for real_round in real_rounds:
            items.append(TimelineItem(
                session_dir=entry.session_dir,
                company=entry.company,
                title=entry.title,
                date=real_round.created_at[:10],
                item_type="real",
                round_label=real_round.stage,
                run_path=None,
                average_score=None,
                review_path=real_round.path,
                ai_review_path=(
                    entry.session_dir
                    / (
                        "14_real_interview_review_"
                        f"{real_round.round_id}.json"
                    )
                ),
            ))
        if not real_rounds:
            review_path = entry.session_dir / "13_post_interview_review.md"
            if review_path.exists():
                items.append(TimelineItem(
                    session_dir=entry.session_dir,
                    company=entry.company,
                    title=entry.title,
                    date=entry.date if entry.date else "",
                    item_type="real",
                    round_label="历史记录",
                    run_path=None,
                    average_score=None,
                    review_path=review_path,
                    ai_review_path=None,
                ))

    # 按日期倒序
    items.sort(key=lambda i: i.date, reverse=True)
    return items
