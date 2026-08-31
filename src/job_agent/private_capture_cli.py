from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
from typing import Sequence

from job_agent.agent_runtime.failures import TrajectoryError
from job_agent.agent_runtime.private_capture import PrivateRunStore


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="job-agent private-capture-maintenance",
        description=(
            "List expired repository-external private captures. Deletion requires an "
            "explicit --delete-expired acknowledgement."
        ),
    )
    parser.add_argument("--private-root", required=True)
    parser.add_argument("--workspace-root", default=".")
    parser.add_argument(
        "--delete-expired",
        action="store_true",
        help="Permanently delete only runs whose retention metadata is already expired.",
    )
    return parser


def main(
    argv: Sequence[str] | None = None,
    *,
    now: datetime | None = None,
) -> int:
    args = build_parser().parse_args(argv)
    try:
        store = PrivateRunStore(
            Path(args.private_root),
            workspace_root=Path(args.workspace_root),
        )
        expired = store.list_expired(now=now)
    except (OSError, ValueError, TrajectoryError) as exc:
        print(f"Private capture maintenance did not start: {exc}")
        return 2

    print(f"Expired private capture count: {len(expired)}")
    if not args.delete_expired:
        print("Dry run only; no private capture was deleted.")
        return 0

    removed = store.cleanup_expired(now=now)
    print(f"Deleted expired private capture count: {len(removed)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
