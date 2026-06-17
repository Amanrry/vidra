"""Small SQLite boundary for schema initialization and simple queries."""

from __future__ import annotations

import sqlite3
import threading
from contextlib import contextmanager
from collections.abc import Iterable
from pathlib import Path
from types import TracebackType
from typing import Any, Iterator


SCHEMA_PATH = Path(__file__).with_name("schema.sql")


class Database:
    """Thin SQLite wrapper.

    This layer deliberately stays infrastructure-focused. Higher-level services
    own business operations such as storing tool batches or building context.
    """

    def __init__(self, database_path: str | Path) -> None:
        self.database_path = Path(database_path)
        self._connection: sqlite3.Connection | None = None
        self._lock = threading.RLock()

    def connect(self) -> sqlite3.Connection:
        with self._lock:
            if self._connection is None:
                self.database_path.parent.mkdir(parents=True, exist_ok=True)
                connection = sqlite3.connect(self.database_path, check_same_thread=False)
                connection.row_factory = sqlite3.Row
                connection.execute("PRAGMA foreign_keys = ON")
                self._connection = connection
            return self._connection

    def initialize(self) -> None:
        with self._lock:
            schema = SCHEMA_PATH.read_text(encoding="utf-8")
            self.connect().executescript(schema)
            self.connect().commit()

    def execute(
        self,
        sql: str,
        params: Iterable[Any] | None = None,
    ) -> sqlite3.Cursor:
        with self._lock:
            cursor = self.connect().execute(sql, tuple(params or ()))
            self.connect().commit()
            return cursor

    def executemany(
        self,
        sql: str,
        params: Iterable[Iterable[Any]],
    ) -> sqlite3.Cursor:
        with self._lock:
            cursor = self.connect().executemany(sql, [tuple(item) for item in params])
            self.connect().commit()
            return cursor

    def query(
        self,
        sql: str,
        params: Iterable[Any] | None = None,
    ) -> list[sqlite3.Row]:
        with self._lock:
            cursor = self.connect().execute(sql, tuple(params or ()))
            return list(cursor.fetchall())

    @contextmanager
    def transaction(self) -> Iterator[sqlite3.Connection]:
        """Run several statements atomically on the shared connection."""

        with self._lock:
            connection = self.connect()
            try:
                connection.execute("BEGIN IMMEDIATE")
                yield connection
                connection.commit()
            except Exception:
                connection.rollback()
                raise

    def table_names(self) -> set[str]:
        rows = self.query(
            "SELECT name FROM sqlite_master WHERE type = 'table' AND name NOT LIKE 'sqlite_%'"
        )
        return {str(row["name"]) for row in rows}

    def close(self) -> None:
        with self._lock:
            if self._connection is not None:
                self._connection.close()
                self._connection = None

    def __enter__(self) -> "Database":
        self.connect()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()


def initialize_database(database_path: str | Path) -> Database:
    database = Database(database_path)
    database.initialize()
    return database
