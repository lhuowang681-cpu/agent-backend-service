from __future__ import annotations

from pathlib import Path

import psycopg


MIGRATION_DIR = Path(__file__).with_name("migrations")


def apply_migrations(database_url: str) -> list[str]:
    """Apply immutable SQL migrations once, under a transaction-scoped advisory lock."""

    applied: list[str] = []
    with psycopg.connect(database_url) as connection:
        with connection.transaction():
            connection.execute("SELECT pg_advisory_xact_lock(%s)", (814_202_603,))
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS schema_migrations (
                    version TEXT PRIMARY KEY,
                    applied_at TIMESTAMPTZ NOT NULL DEFAULT now()
                )
                """
            )
            existing = {
                row[0]
                for row in connection.execute("SELECT version FROM schema_migrations")
            }
            for path in sorted(MIGRATION_DIR.glob("*.sql")):
                if path.name in existing:
                    continue
                connection.execute(path.read_text(encoding="utf-8"))
                connection.execute(
                    "INSERT INTO schema_migrations(version) VALUES (%s)",
                    (path.name,),
                )
                applied.append(path.name)
    return applied
