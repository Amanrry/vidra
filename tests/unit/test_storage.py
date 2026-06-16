from __future__ import annotations

import json
from pathlib import Path

import pytest

from video_qa.storage import Database, RunLayout, initialize_database


def test_database_initializes_expected_tables(tmp_path: Path) -> None:
    database_path = tmp_path / "vidra.sqlite3"
    database = initialize_database(database_path)

    assert {
        "videos",
        "memory",
        "video_context",
        "lineage",
        "schema_version",
    }.issubset(database.table_names())

    version_rows = database.query("SELECT version FROM schema_version")
    assert version_rows[0]["version"] == 1
    database.close()


def test_database_enforces_foreign_keys(tmp_path: Path) -> None:
    database = initialize_database(tmp_path / "vidra.sqlite3")

    with pytest.raises(Exception):
        database.execute(
            """
            INSERT INTO video_context
            (context_id, video_id, context_type, timestamp_sec, data, tool_name)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            ("ctx-1", "missing-video", "caption", 0.0, "{}", "caption_frames"),
        )

    database.close()


def test_database_accepts_video_context_after_video_insert(tmp_path: Path) -> None:
    database = initialize_database(tmp_path / "vidra.sqlite3")
    database.execute(
        """
        INSERT INTO videos (video_id, filename, file_path, duration_sec)
        VALUES (?, ?, ?, ?)
        """,
        ("video-1", "demo.mp4", "data/input/demo.mp4", 3.5),
    )
    database.execute(
        """
        INSERT INTO video_context
        (context_id, video_id, context_type, timestamp_sec, data, tool_name, model_name)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        (
            "ctx-1",
            "video-1",
            "caption",
            1.2,
            json.dumps({"text": "a person near a car"}),
            "caption_frames",
            "blip",
        ),
    )

    rows = database.query(
        "SELECT context_type, json_extract(data, '$.text') AS text FROM video_context"
    )
    assert rows[0]["context_type"] == "caption"
    assert rows[0]["text"] == "a person near a car"
    database.close()


def test_run_layout_is_deterministic_and_creates_directories(tmp_path: Path) -> None:
    layout = RunLayout(tmp_path / "runs")

    first = layout.for_run("run-123", create=True)
    second = layout.for_run("run-123", create=False)

    assert first == second
    assert first.root == tmp_path / "runs" / "run-123"
    assert first.source_dir.is_dir()
    assert first.frames_dir.is_dir()
    assert first.annotated_frames_dir.is_dir()
    assert first.crops_dir.is_dir()
    assert first.reports_dir.is_dir()
    assert first.report_json == first.reports_dir / "report.json"
    assert first.detections_csv == first.reports_dir / "detections.csv"


def test_run_layout_rejects_empty_or_relative_run_ids(tmp_path: Path) -> None:
    layout = RunLayout(tmp_path / "runs")

    with pytest.raises(ValueError, match="run_id"):
        layout.for_run("")
    with pytest.raises(ValueError, match="relative"):
        layout.for_run("../escape")
