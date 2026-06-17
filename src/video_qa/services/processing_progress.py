"""SQLite-backed processing progress repository."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from sqlite3 import Row

from video_qa.models.media import ProcessingStatus
from video_qa.models.processing import (
    JobPriority,
    JobStatus,
    ProcessingCounts,
    ProcessingJobRecord,
    ProcessingProgress,
    ProcessingStage,
)
from video_qa.storage import Database


def _utc_now() -> datetime:
    return datetime.now(UTC)


def _to_db_datetime(value: datetime | None) -> str | None:
    return value.isoformat() if value is not None else None


def _from_db_datetime(value: str | None) -> datetime | None:
    if value is None:
        return None
    parsed = datetime.fromisoformat(value)
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=UTC)
    return parsed


def _video_status_for_job(status: JobStatus) -> ProcessingStatus:
    if status == JobStatus.queued:
        return ProcessingStatus.queued
    if status == JobStatus.processing:
        return ProcessingStatus.processing
    if status == JobStatus.complete:
        return ProcessingStatus.complete
    if status == JobStatus.partial:
        return ProcessingStatus.partial
    if status == JobStatus.cancelled:
        return ProcessingStatus.cancelled
    return ProcessingStatus.failed


class ProcessingProgressRepository:
    """Persists queue jobs and user-visible progress snapshots."""

    def __init__(self, database: Database) -> None:
        self.database = database
        self.database.initialize()

    def create_job(
        self,
        *,
        job_id: str,
        video_id: str,
        run_id: str,
        source_path: str | Path,
        priority: JobPriority = JobPriority.normal,
        message: str = "Queued for processing.",
    ) -> ProcessingJobRecord:
        now = _utc_now()
        job = ProcessingJobRecord(
            job_id=job_id,
            video_id=video_id,
            run_id=run_id,
            source_path=Path(source_path),
            priority=priority,
            status=JobStatus.queued,
            stage=ProcessingStage.pending,
            progress_percent=0.0,
            message=message,
            created_at=now,
            updated_at=now,
        )
        self.database.execute(
            """
            INSERT INTO processing_jobs
            (job_id, video_id, run_id, source_path, priority, status, stage,
             progress_percent, message, queue_position, frames_extracted,
             captions_generated, transcript_segments, detections_created,
             crops_created, text_vectors_indexed, image_vectors_indexed, error,
             created_at, started_at, completed_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            self._job_params(job),
        )
        self._update_video_status(video_id, job.status, None)
        return job

    def get_job(self, job_id: str) -> ProcessingJobRecord | None:
        rows = self.database.query(
            "SELECT * FROM processing_jobs WHERE job_id = ?",
            (job_id,),
        )
        return self._row_to_job(rows[0]) if rows else None

    def get_latest_for_video(self, video_id: str) -> ProcessingJobRecord | None:
        rows = self.database.query(
            """
            SELECT *
            FROM processing_jobs
            WHERE video_id = ?
            ORDER BY created_at DESC, job_id DESC
            LIMIT 1
            """,
            (video_id,),
        )
        return self._row_to_job(rows[0]) if rows else None

    def progress_for_video(self, video_id: str) -> ProcessingProgress | None:
        job = self.get_latest_for_video(video_id)
        if job is None:
            return None
        return self.to_progress(job)

    def mark_processing(
        self,
        job_id: str,
        *,
        stage: ProcessingStage = ProcessingStage.extracting,
        message: str = "Processing started.",
    ) -> ProcessingJobRecord:
        started_at = _utc_now()
        return self.update_job(
            job_id,
            status=JobStatus.processing,
            stage=stage,
            progress_percent=0.0,
            message=message,
            queue_position=None,
            started_at=started_at,
        )

    def update_job(
        self,
        job_id: str,
        *,
        status: JobStatus | None = None,
        stage: ProcessingStage | None = None,
        progress_percent: float | None = None,
        message: str | None = None,
        queue_position: int | None | object = ...,
        counts: ProcessingCounts | None = None,
        error: str | None | object = ...,
        started_at: datetime | None | object = ...,
        completed_at: datetime | None | object = ...,
    ) -> ProcessingJobRecord:
        current = self.get_job(job_id)
        if current is None:
            raise ValueError(f"Processing job not found: {job_id}")

        updated = current.model_copy(
            update={
                "status": status or current.status,
                "stage": stage or current.stage,
                "progress_percent": (
                    current.progress_percent if progress_percent is None else progress_percent
                ),
                "message": current.message if message is None else message,
                "queue_position": (
                    current.queue_position if queue_position is ... else queue_position
                ),
                "counts": counts or current.counts,
                "error": current.error if error is ... else error,
                "started_at": current.started_at if started_at is ... else started_at,
                "completed_at": current.completed_at if completed_at is ... else completed_at,
                "updated_at": _utc_now(),
            }
        )
        self.database.execute(
            """
            UPDATE processing_jobs
            SET status = ?, stage = ?, progress_percent = ?, message = ?,
                queue_position = ?, frames_extracted = ?, captions_generated = ?,
                transcript_segments = ?, detections_created = ?, crops_created = ?,
                text_vectors_indexed = ?, image_vectors_indexed = ?, error = ?,
                started_at = ?, completed_at = ?, updated_at = ?
            WHERE job_id = ?
            """,
            (
                updated.status.value,
                updated.stage.value,
                updated.progress_percent,
                updated.message,
                updated.queue_position,
                updated.counts.frames_extracted,
                updated.counts.captions_generated,
                updated.counts.transcript_segments,
                updated.counts.detections_created,
                updated.counts.crops_created,
                updated.counts.text_vectors_indexed,
                updated.counts.image_vectors_indexed,
                updated.error,
                _to_db_datetime(updated.started_at),
                _to_db_datetime(updated.completed_at),
                _to_db_datetime(updated.updated_at),
                updated.job_id,
            ),
        )
        self._update_video_status(updated.video_id, updated.status, updated.error)
        return updated

    def mark_complete(
        self,
        job_id: str,
        *,
        status: JobStatus = JobStatus.complete,
        message: str = "Processing complete.",
        counts: ProcessingCounts | None = None,
        error: str | None = None,
    ) -> ProcessingJobRecord:
        if status not in {JobStatus.complete, JobStatus.partial, JobStatus.failed, JobStatus.cancelled}:
            raise ValueError("terminal status is required")
        stage = ProcessingStage.complete if status in {JobStatus.complete, JobStatus.partial} else ProcessingStage.failed
        return self.update_job(
            job_id,
            status=status,
            stage=stage,
            progress_percent=100.0,
            message=message,
            queue_position=None,
            counts=counts,
            error=error,
            completed_at=_utc_now(),
        )

    def set_queue_position(self, job_id: str, position: int | None) -> ProcessingJobRecord:
        return self.update_job(job_id, queue_position=position)

    def to_progress(self, job: ProcessingJobRecord) -> ProcessingProgress:
        return ProcessingProgress(
            video_id=job.video_id,
            run_id=job.run_id,
            status=job.status,
            stage=job.stage,
            progress_percent=job.progress_percent,
            message=job.message,
            queue_position=job.queue_position,
            counts=job.counts,
            error=job.error,
        )

    def _job_params(self, job: ProcessingJobRecord) -> tuple[object, ...]:
        return (
            job.job_id,
            job.video_id,
            job.run_id,
            str(job.source_path),
            job.priority.name,
            job.status.value,
            job.stage.value,
            job.progress_percent,
            job.message,
            job.queue_position,
            job.counts.frames_extracted,
            job.counts.captions_generated,
            job.counts.transcript_segments,
            job.counts.detections_created,
            job.counts.crops_created,
            job.counts.text_vectors_indexed,
            job.counts.image_vectors_indexed,
            job.error,
            _to_db_datetime(job.created_at),
            _to_db_datetime(job.started_at),
            _to_db_datetime(job.completed_at),
            _to_db_datetime(job.updated_at),
        )

    def _row_to_job(self, row: Row) -> ProcessingJobRecord:
        return ProcessingJobRecord(
            job_id=str(row["job_id"]),
            video_id=str(row["video_id"]),
            run_id=str(row["run_id"]),
            source_path=Path(str(row["source_path"])),
            priority=JobPriority[str(row["priority"])],
            status=JobStatus(str(row["status"])),
            stage=ProcessingStage(str(row["stage"])),
            progress_percent=float(row["progress_percent"]),
            message=str(row["message"] or ""),
            queue_position=row["queue_position"],
            counts=ProcessingCounts(
                frames_extracted=int(row["frames_extracted"]),
                captions_generated=int(row["captions_generated"]),
                transcript_segments=int(row["transcript_segments"]),
                detections_created=int(row["detections_created"]),
                crops_created=int(row["crops_created"]),
                text_vectors_indexed=int(row["text_vectors_indexed"]),
                image_vectors_indexed=int(row["image_vectors_indexed"]),
            ),
            error=row["error"],
            created_at=_from_db_datetime(row["created_at"]) or _utc_now(),
            started_at=_from_db_datetime(row["started_at"]),
            completed_at=_from_db_datetime(row["completed_at"]),
            updated_at=_from_db_datetime(row["updated_at"]),
        )

    def _update_video_status(
        self,
        video_id: str,
        status: JobStatus,
        error: str | None,
    ) -> None:
        self.database.execute(
            """
            UPDATE videos
            SET status = ?, error = ?, updated_at = CURRENT_TIMESTAMP
            WHERE video_id = ?
            """,
            (_video_status_for_job(status).value, error, video_id),
        )
