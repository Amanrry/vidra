from __future__ import annotations

from pathlib import Path

from video_qa.models.processing import (
    JobPriority,
    JobStatus,
    ProcessingCounts,
    ProcessingStage,
)
from video_qa.services.processing_progress import ProcessingProgressRepository
from video_qa.storage import Database


def insert_video(database: Database, video_id: str = "video-1") -> None:
    database.execute(
        """
        INSERT INTO videos (video_id, filename, file_path, duration_sec)
        VALUES (?, ?, ?, ?)
        """,
        (video_id, f"{video_id}.mp4", f"data/input/{video_id}.mp4", 0.0),
    )


def test_processing_progress_persists_and_survives_repository_recreation(
    tmp_path: Path,
) -> None:
    database_path = tmp_path / "vidra.sqlite3"
    database = Database(database_path)
    database.initialize()
    insert_video(database)
    repo = ProcessingProgressRepository(database)

    job = repo.create_job(
        job_id="job-1",
        video_id="video-1",
        run_id="video-1",
        source_path=tmp_path / "demo.mp4",
        priority=JobPriority.high,
    )
    repo.set_queue_position(job.job_id, 1)
    repo.mark_processing(
        job.job_id,
        stage=ProcessingStage.extracting,
        message="Extracting frames.",
    )
    updated = repo.update_job(
        job.job_id,
        stage=ProcessingStage.captioning,
        progress_percent=42.5,
        message="Captioning frames.",
        counts=ProcessingCounts(frames_extracted=3, captions_generated=2),
    )

    assert updated.stage == ProcessingStage.captioning
    assert updated.counts.frames_extracted == 3

    database.close()
    recreated = Database(database_path)
    recreated_repo = ProcessingProgressRepository(recreated)

    persisted = recreated_repo.get_job("job-1")
    assert persisted is not None
    assert persisted.priority == JobPriority.high
    assert persisted.status == JobStatus.processing
    assert persisted.stage == ProcessingStage.captioning
    assert persisted.progress_percent == 42.5
    assert persisted.counts.captions_generated == 2
    assert persisted.started_at is not None

    progress = recreated_repo.progress_for_video("video-1")
    assert progress is not None
    assert progress.message == "Captioning frames."
    assert progress.counts.frames_extracted == 3

    rows = recreated.query("SELECT status FROM videos WHERE video_id = ?", ("video-1",))
    assert rows[0]["status"] == "processing"
    recreated.close()


def test_processing_progress_records_terminal_failure_and_video_status(
    tmp_path: Path,
) -> None:
    database = Database(tmp_path / "vidra.sqlite3")
    database.initialize()
    insert_video(database)
    repo = ProcessingProgressRepository(database)
    repo.create_job(
        job_id="job-1",
        video_id="video-1",
        run_id="video-1",
        source_path=tmp_path / "demo.mp4",
    )

    failed = repo.mark_complete(
        "job-1",
        status=JobStatus.failed,
        message="Pipeline failed.",
        error="boom",
    )

    assert failed.status == JobStatus.failed
    assert failed.stage == ProcessingStage.failed
    assert failed.progress_percent == 100.0
    assert failed.completed_at is not None

    rows = database.query(
        "SELECT status, error FROM videos WHERE video_id = ?",
        ("video-1",),
    )
    assert rows[0]["status"] == "failed"
    assert rows[0]["error"] == "boom"
    database.close()
