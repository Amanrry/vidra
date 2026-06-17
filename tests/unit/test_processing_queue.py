from __future__ import annotations

import asyncio
from pathlib import Path

from video_qa.models.processing import JobPriority, JobStatus
from video_qa.services.processing_progress import ProcessingProgressRepository
from video_qa.services.processing_queue import ProcessingJob, ProcessingQueue
from video_qa.storage import Database


def insert_video(database: Database, video_id: str) -> None:
    database.execute(
        """
        INSERT INTO videos (video_id, filename, file_path, duration_sec)
        VALUES (?, ?, ?, ?)
        """,
        (video_id, f"{video_id}.mp4", f"data/input/{video_id}.mp4", 0.0),
    )


def test_queue_runs_jobs_in_priority_order(tmp_path: Path) -> None:
    async def scenario() -> None:
        processed: list[str] = []

        async def runner(job: ProcessingJob) -> None:
            processed.append(job.video_id)

        queue = ProcessingQueue(runner=runner, max_workers=1)
        await queue.add_job(
            video_id="low",
            run_id="low",
            source_path=tmp_path / "low.mp4",
            priority=JobPriority.low,
            job_id="job-low",
        )
        await queue.add_job(
            video_id="high",
            run_id="high",
            source_path=tmp_path / "high.mp4",
            priority=JobPriority.high,
            job_id="job-high",
        )
        await queue.add_job(
            video_id="normal",
            run_id="normal",
            source_path=tmp_path / "normal.mp4",
            priority=JobPriority.normal,
            job_id="job-normal",
        )

        await queue.run_until_idle()

        assert processed == ["high", "normal", "low"]
        assert queue.get_status().completed_jobs == 3
        status = queue.get_job_status("high")
        assert status is not None
        assert status.status == JobStatus.complete

    asyncio.run(scenario())


def test_duplicate_queued_and_active_jobs_are_reused(tmp_path: Path) -> None:
    async def scenario() -> None:
        release = asyncio.Event()

        async def runner(job: ProcessingJob) -> None:
            await release.wait()

        queue = ProcessingQueue(runner=runner, max_workers=1, sleep_seconds=0.01)

        first = await queue.add_job(
            video_id="video-1",
            run_id="video-1",
            source_path=tmp_path / "demo.mp4",
            job_id="job-1",
        )
        duplicate_queued = await queue.add_job(
            video_id="video-1",
            run_id="video-1",
            source_path=tmp_path / "demo.mp4",
            job_id="job-duplicate",
        )

        assert duplicate_queued is first
        assert queue.get_status().queued_jobs == 1

        await queue.start_workers()
        await asyncio.sleep(0.05)
        duplicate_active = await queue.add_job(
            video_id="video-1",
            run_id="video-1",
            source_path=tmp_path / "demo.mp4",
            job_id="job-duplicate-active",
        )

        assert duplicate_active.job_id == first.job_id
        assert duplicate_active.status == JobStatus.processing
        assert queue.get_status().active_jobs == 1

        release.set()
        await asyncio.sleep(0.05)
        await queue.shutdown(timeout_seconds=1.0)
        assert queue.get_status().shutdown_requested

    asyncio.run(scenario())


def test_worker_records_failed_jobs(tmp_path: Path) -> None:
    async def scenario() -> None:
        async def runner(job: ProcessingJob) -> None:
            raise RuntimeError("model exploded")

        queue = ProcessingQueue(runner=runner, max_workers=1)
        await queue.add_job(
            video_id="video-1",
            run_id="video-1",
            source_path=tmp_path / "demo.mp4",
            job_id="job-1",
        )

        await queue.run_until_idle()

        status = queue.get_job_status("video-1")
        assert status is not None
        assert status.status == JobStatus.failed
        assert status.error == "model exploded"

    asyncio.run(scenario())


def test_queue_persists_job_progress_and_positions(tmp_path: Path) -> None:
    async def scenario() -> None:
        database = Database(tmp_path / "vidra.sqlite3")
        database.initialize()
        insert_video(database, "video-low")
        insert_video(database, "video-high")
        repository = ProcessingProgressRepository(database)

        async def runner(job: ProcessingJob) -> None:
            return None

        queue = ProcessingQueue(
            runner=runner,
            max_workers=1,
            progress_repository=repository,
        )
        low = await queue.add_job(
            video_id="video-low",
            run_id="video-low",
            source_path=tmp_path / "low.mp4",
            priority=JobPriority.low,
            job_id="job-low",
        )
        high = await queue.add_job(
            video_id="video-high",
            run_id="video-high",
            source_path=tmp_path / "high.mp4",
            priority=JobPriority.high,
            job_id="job-high",
        )

        persisted_high = repository.get_job(high.job_id)
        persisted_low = repository.get_job(low.job_id)
        assert persisted_high is not None
        assert persisted_low is not None
        assert persisted_high.queue_position == 1
        assert persisted_low.queue_position == 2

        await queue.run_until_idle()

        completed_high = repository.get_job(high.job_id)
        completed_low = repository.get_job(low.job_id)
        progress = repository.progress_for_video("video-high")
        assert completed_high is not None
        assert completed_low is not None
        assert progress is not None
        assert completed_high.status == JobStatus.complete
        assert completed_low.status == JobStatus.complete
        assert progress.progress_percent == 100.0
        database.close()

    asyncio.run(scenario())


def test_shutdown_cancels_idle_workers_gracefully(tmp_path: Path) -> None:
    async def scenario() -> None:
        async def runner(job: ProcessingJob) -> None:
            return None

        queue = ProcessingQueue(runner=runner, max_workers=2, sleep_seconds=0.01)
        await queue.start_workers()

        assert queue.get_status().workers == 2

        await queue.shutdown(timeout_seconds=1.0)

        status = queue.get_status()
        assert status.workers == 0
        assert status.shutdown_requested

    asyncio.run(scenario())
