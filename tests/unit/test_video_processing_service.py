from __future__ import annotations

from pathlib import Path

from video_qa.models.tools import ContextType, VideoContextRecord
from video_qa.services.video_processing_service import VideoProcessingService
from video_qa.storage import Database


def insert_video(database: Database, video_id: str = "video-1") -> None:
    database.execute(
        """
        INSERT INTO videos (video_id, filename, file_path, duration_sec)
        VALUES (?, ?, ?, ?)
        """,
        (video_id, "demo.mp4", "data/input/demo.mp4", 0.0),
    )


def context(context_id: str, context_type: ContextType, data: dict, timestamp: float | None = 1.0):
    return VideoContextRecord(
        context_id=context_id,
        video_id="video-1",
        context_type=context_type,
        timestamp_sec=timestamp,
        data=data,
        tool_name="test_tool",
        model_name="fake",
    )


def test_processing_service_persists_context_lineage_and_idempotency(tmp_path: Path) -> None:
    database = Database(tmp_path / "vidra.sqlite3")
    database.initialize()
    insert_video(database)
    service = VideoProcessingService(database)
    records = [
        context("frame-1", ContextType.frame, {"frame_id": "frame-1", "image_path": "frame.jpg"}),
        context("caption-1", ContextType.caption, {"text": "a person walking"}),
    ]

    first = service.store_tool_results(
        video_id="video-1",
        tool_name="extract_frames",
        records=records,
        idempotency_key="run-1:extract",
        run_id="run-1",
        parameters={"interval": 2.0},
    )
    second = service.store_tool_results(
        video_id="video-1",
        tool_name="extract_frames",
        records=[],
        idempotency_key="run-1:extract",
        run_id="run-1",
    )

    assert first.frames_extracted == 1
    assert first.captions_generated == 1
    assert second == first
    assert database.query("SELECT COUNT(*) AS count FROM video_context")[0]["count"] == 2
    assert database.query("SELECT COUNT(*) AS count FROM lineage")[0]["count"] == 2
    assert database.query("SELECT COUNT(*) AS count FROM processing_idempotency")[0]["count"] == 1


def test_processing_service_completeness_counts_reports(tmp_path: Path) -> None:
    database = Database(tmp_path / "vidra.sqlite3")
    database.initialize()
    insert_video(database)
    service = VideoProcessingService(database)
    service.store_tool_results(
        video_id="video-1",
        tool_name="caption_frames",
        records=[
            context(
                "frame-1",
                ContextType.frame,
                {"frame_id": "frame-1", "image_path": "frame.jpg"},
            ),
            context("caption-1", ContextType.caption, {"text": "a scene"}),
        ],
        idempotency_key="run-1:caption",
        run_id="run-1",
    )
    reports = tmp_path / "reports"
    reports.mkdir()
    for name in ["report.json", "detections.csv", "summary.md"]:
        (reports / name).write_text("ok", encoding="utf-8")

    completeness = service.verify_video_data_completeness("video-1", run_reports_dir=reports)

    assert completeness["complete"]
    assert completeness["counts"]["frame"] == 1
    assert completeness["counts"]["caption"] == 1
    assert completeness["reports"]["report_json"]
