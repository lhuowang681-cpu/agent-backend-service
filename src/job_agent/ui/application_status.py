from __future__ import annotations

from collections.abc import Iterable

from job_agent.schemas import ApplicationStatus


STATUS_LABELS = {
    ApplicationStatus.TO_APPLY: "待投递",
    ApplicationStatus.APPLIED: "已投递",
    ApplicationStatus.RESUME_SCREEN: "简历筛选",
    ApplicationStatus.FIRST_INTERVIEW: "一面",
    ApplicationStatus.SECOND_INTERVIEW: "二面",
    ApplicationStatus.OTHER_INTERVIEW: "其他面试（三面及以上）",
    ApplicationStatus.HR_INTERVIEW: "HR 面",
    ApplicationStatus.OFFER: "Offer",
    ApplicationStatus.ACCEPTED: "已接受 Offer",
    ApplicationStatus.REJECTED: "被拒绝",
    ApplicationStatus.ABANDONED: "已放弃",
    ApplicationStatus.CLOSED: "已结束",
}

_STATUS_ORDER = {
    status: index
    for index, status in enumerate(
        (
            ApplicationStatus.APPLIED,
            ApplicationStatus.RESUME_SCREEN,
            ApplicationStatus.FIRST_INTERVIEW,
            ApplicationStatus.SECOND_INTERVIEW,
            ApplicationStatus.OTHER_INTERVIEW,
            ApplicationStatus.HR_INTERVIEW,
            ApplicationStatus.OFFER,
            ApplicationStatus.ACCEPTED,
            ApplicationStatus.REJECTED,
            ApplicationStatus.ABANDONED,
            ApplicationStatus.CLOSED,
        )
    )
}

WORKBENCH_TRANSITIONS = {
    ApplicationStatus.TO_APPLY: {
        ApplicationStatus.APPLIED,
        ApplicationStatus.CLOSED,
    },
    ApplicationStatus.APPLIED: {
        ApplicationStatus.RESUME_SCREEN,
        ApplicationStatus.CLOSED,
    },
    ApplicationStatus.RESUME_SCREEN: {
        ApplicationStatus.FIRST_INTERVIEW,
        ApplicationStatus.CLOSED,
    },
    ApplicationStatus.FIRST_INTERVIEW: {
        ApplicationStatus.SECOND_INTERVIEW,
        ApplicationStatus.OFFER,
        ApplicationStatus.CLOSED,
    },
    ApplicationStatus.SECOND_INTERVIEW: {
        ApplicationStatus.OTHER_INTERVIEW,
        ApplicationStatus.OFFER,
        ApplicationStatus.CLOSED,
    },
    ApplicationStatus.OTHER_INTERVIEW: {
        ApplicationStatus.OFFER,
        ApplicationStatus.CLOSED,
    },
    ApplicationStatus.HR_INTERVIEW: {
        ApplicationStatus.OFFER,
        ApplicationStatus.CLOSED,
    },
    ApplicationStatus.OFFER: {
        ApplicationStatus.ACCEPTED,
        ApplicationStatus.CLOSED,
    },
    ApplicationStatus.ACCEPTED: set(),
    ApplicationStatus.REJECTED: set(),
    ApplicationStatus.ABANDONED: set(),
    ApplicationStatus.CLOSED: set(),
}


def status_label(status: ApplicationStatus | str) -> str:
    value = status if isinstance(status, ApplicationStatus) else ApplicationStatus(status)
    return STATUS_LABELS[value]


def ordered_statuses(
    statuses: Iterable[ApplicationStatus],
) -> list[ApplicationStatus]:
    return sorted(statuses, key=lambda item: _STATUS_ORDER.get(item, 999))


def workbench_next_statuses(
    status: ApplicationStatus,
) -> list[ApplicationStatus]:
    """Return the intentionally small set of choices shown in the desktop UI."""
    return ordered_statuses(WORKBENCH_TRANSITIONS.get(status, set()))
