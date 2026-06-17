"""Bri-style in-process processing queue for local Vidra runs."""

from __future__ import annotations

import asyncio
import inspect
import threading
import time
import uuid
from collections import deque
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from video_qa.models.processing import JobPriority, JobStatus, ProcessingJobRecord
from video_qa.services.processing_progress import ProcessingProgressRepository


@dataclass(order=True)
class ProcessingJob:
    """In-memory schedulable job snapshot."""

    priority: int = field(compare=True)
    sequence: int = field(compare=True)
    job_id: str = field(compare=False)
    video_id: str = field(compare=False)
    run_id: str = field(compare=False)
    source_path: Path = field(compare=False)
    created_at: float = field(default_factory=time.time, compare=False)
    started_at: float | None = field(default=None, compare=False)
    completed_at: float | None = field(default=None, compare=False)
    status: JobStatus = field(default=JobStatus.queued, compare=False)
    error: str | None = field(default=None, compare=False)


@dataclass(frozen=True)
class QueueStatus:
    queued_jobs: int
    active_jobs: int
    completed_jobs: int
    workers: int
    shutdown_requested: bool


@dataclass(frozen=True)
class JobStatusSnapshot:
    job_id: str
    video_id: str
    status: JobStatus
    priority: JobPriority
    queue_position: int | None = None
    started_at: float | None = None
    completed_at: float | None = None
    duration_sec: float | None = None
    error: str | None = None


@runtime_checkable
class ProcessingJobRunner(Protocol):
    def __call__(self, job: ProcessingJob) -> Awaitable[Any] | Any:
        """Run one processing job."""


class DuplicateJobPolicy:
    """Duplicate handling kept as a named policy for future queue adapters."""

    def should_reuse(self, existing_status: JobStatus) -> bool:
        return existing_status in {JobStatus.queued, JobStatus.processing}


class ProcessingQueue:
    """Local async priority queue with bounded workers and status snapshots."""

    def __init__(
        self,
        *,
        runner: ProcessingJobRunner,
        max_workers: int = 1,
        progress_repository: ProcessingProgressRepository | None = None,
        duplicate_policy: DuplicateJobPolicy | None = None,
        sleep_seconds: float = 0.05,
        history_limit: int = 100,
    ) -> None:
        if max_workers < 1:
            raise ValueError("max_workers must be positive")
        self.runner = runner
        self.max_workers = max_workers
        self.progress_repository = progress_repository
        self.duplicate_policy = duplicate_policy or DuplicateJobPolicy()
        self.sleep_seconds = sleep_seconds
        self.queue: list[ProcessingJob] = []
        self.active_jobs: dict[str, ProcessingJob] = {}
        self.completed_jobs: deque[ProcessingJob] = deque(maxlen=history_limit)
        self.workers: list[asyncio.Task[None]] = []
        self.shutdown_requested = False
        self._sequence = 0
        self._state_lock = threading.RLock()

    async def add_job(
        self,
        *,
        video_id: str,
        run_id: str,
        source_path: str | Path,
        priority: JobPriority = JobPriority.normal,
        job_id: str | None = None,
    ) -> ProcessingJob:
        """Add a job, reusing any queued/active job for the same video."""

        clean_video_id = video_id.strip()
        clean_run_id = run_id.strip()
        if not clean_video_id:
            raise ValueError("video_id is required")
        if not clean_run_id:
            raise ValueError("run_id is required")

        with self._state_lock:
            existing = self._find_queued_or_active(clean_video_id)
            if existing is not None and self.duplicate_policy.should_reuse(existing.status):
                return existing
            self._sequence += 1
            job = ProcessingJob(
                priority=int(priority),
                sequence=self._sequence,
                job_id=job_id or str(uuid.uuid4()),
                video_id=clean_video_id,
                run_id=clean_run_id,
                source_path=Path(source_path),
            )
            self.queue.append(job)
            self.queue.sort()
            self._refresh_queue_positions_locked()

        if self.progress_repository is not None:
            self.progress_repository.create_job(
                job_id=job.job_id,
                video_id=job.video_id,
                run_id=job.run_id,
                source_path=job.source_path,
                priority=priority,
            )
            self._persist_queue_positions()

        return job

    async def get_next_job(self) -> ProcessingJob | None:
        with self._state_lock:
            if not self.queue:
                return None
            job = self.queue.pop(0)
            job.status = JobStatus.processing
            job.started_at = time.time()
            self.active_jobs[job.video_id] = job
            self._refresh_queue_positions_locked()

        if self.progress_repository is not None:
            self.progress_repository.mark_processing(job.job_id)
            self._persist_queue_positions()
        return job

    async def complete_job(
        self,
        job: ProcessingJob,
        *,
        status: JobStatus = JobStatus.complete,
        error: str | None = None,
    ) -> None:
        if status not in {JobStatus.complete, JobStatus.partial, JobStatus.failed, JobStatus.cancelled}:
            raise ValueError("terminal job status is required")

        with self._state_lock:
            active = self.active_jobs.pop(job.video_id, job)
            active.status = status
            active.error = error
            active.completed_at = time.time()
            self.completed_jobs.append(active)
            self._refresh_queue_positions_locked()

        if self.progress_repository is not None:
            self.progress_repository.mark_complete(
                active.job_id,
                status=status,
                message=self._terminal_message(status),
                error=error,
            )
            self._persist_queue_positions()

    async def start_workers(self) -> None:
        if self.shutdown_requested:
            self.shutdown_requested = False
        live_workers = [worker for worker in self.workers if not worker.done()]
        missing = self.max_workers - len(live_workers)
        self.workers = live_workers
        for worker_id in range(len(live_workers), len(live_workers) + missing):
            self.workers.append(asyncio.create_task(self._worker(worker_id)))

    async def run_until_idle(self) -> None:
        """Drain all currently queued work without starting persistent workers."""

        while True:
            job = await self.get_next_job()
            if job is None:
                return
            await self._run_one(job)

    async def shutdown(self, *, timeout_seconds: float = 30.0) -> None:
        self.shutdown_requested = True
        live_workers = [worker for worker in self.workers if not worker.done()]
        if not live_workers:
            self.workers = []
            return
        try:
            await asyncio.wait_for(
                asyncio.gather(*live_workers, return_exceptions=True),
                timeout=timeout_seconds,
            )
        except TimeoutError:
            for worker in live_workers:
                worker.cancel()
            await asyncio.gather(*live_workers, return_exceptions=True)
        finally:
            self.workers = []

    def get_status(self) -> QueueStatus:
        with self._state_lock:
            return QueueStatus(
                queued_jobs=len(self.queue),
                active_jobs=len(self.active_jobs),
                completed_jobs=len(self.completed_jobs),
                workers=len([worker for worker in self.workers if not worker.done()]),
                shutdown_requested=self.shutdown_requested,
            )

    def get_job_status(self, video_id: str) -> JobStatusSnapshot | None:
        with self._state_lock:
            for index, job in enumerate(self.queue, start=1):
                if job.video_id == video_id:
                    return self._snapshot(job, queue_position=index)

            active = self.active_jobs.get(video_id)
            if active is not None:
                return self._snapshot(active)

            for job in self.completed_jobs:
                if job.video_id == video_id:
                    return self._snapshot(job)

        if self.progress_repository is not None:
            persisted = self.progress_repository.get_latest_for_video(video_id)
            if persisted is not None:
                return self._snapshot_from_record(persisted)
        return None

    async def _worker(self, worker_id: int) -> None:
        while not self.shutdown_requested:
            job = await self.get_next_job()
            if job is None:
                await asyncio.sleep(self.sleep_seconds)
                continue
            await self._run_one(job)

    async def _run_one(self, job: ProcessingJob) -> None:
        try:
            result = self.runner(job)
            if inspect.isawaitable(result):
                result = await result
            await self.complete_job(job, status=self._terminal_status_from_result(result))
        except Exception as exc:
            await self.complete_job(job, status=JobStatus.failed, error=str(exc))

    def _terminal_status_from_result(self, result: Any) -> JobStatus:
        if result is None:
            return JobStatus.complete
        if isinstance(result, JobStatus):
            return result
        status = getattr(result, "status", None)
        if isinstance(status, JobStatus):
            return status
        if isinstance(status, str):
            return JobStatus(status)
        return JobStatus.complete

    def _find_queued_or_active(self, video_id: str) -> ProcessingJob | None:
        with self._state_lock:
            if video_id in self.active_jobs:
                return self.active_jobs[video_id]
            for job in self.queue:
                if job.video_id == video_id:
                    return job
        return None

    def _refresh_queue_positions_locked(self) -> None:
        self.queue.sort()

    def _persist_queue_positions(self) -> None:
        if self.progress_repository is None:
            return
        with self._state_lock:
            for index, job in enumerate(self.queue, start=1):
                self.progress_repository.set_queue_position(job.job_id, index)

    def _snapshot(
        self,
        job: ProcessingJob,
        *,
        queue_position: int | None = None,
    ) -> JobStatusSnapshot:
        completed_at = job.completed_at
        started_at = job.started_at
        duration = None
        if started_at is not None:
            duration = (completed_at or time.time()) - started_at
        return JobStatusSnapshot(
            job_id=job.job_id,
            video_id=job.video_id,
            status=job.status,
            priority=JobPriority(job.priority),
            queue_position=queue_position,
            started_at=started_at,
            completed_at=completed_at,
            duration_sec=duration,
            error=job.error,
        )

    def _snapshot_from_record(self, job: ProcessingJobRecord) -> JobStatusSnapshot:
        duration = None
        if job.started_at is not None:
            end = job.completed_at or time.time()
            if not isinstance(end, float):
                duration = (end - job.started_at).total_seconds()
        return JobStatusSnapshot(
            job_id=job.job_id,
            video_id=job.video_id,
            status=job.status,
            priority=job.priority,
            queue_position=job.queue_position,
            started_at=job.started_at.timestamp() if job.started_at else None,
            completed_at=job.completed_at.timestamp() if job.completed_at else None,
            duration_sec=duration,
            error=job.error,
        )

    def _terminal_message(self, status: JobStatus) -> str:
        messages = {
            JobStatus.complete: "Processing complete.",
            JobStatus.partial: "Processing partially complete.",
            JobStatus.failed: "Processing failed.",
            JobStatus.cancelled: "Processing cancelled.",
        }
        return messages[status]


def create_noop_queue(
    *,
    progress_repository: ProcessingProgressRepository | None = None,
    max_workers: int = 1,
) -> ProcessingQueue:
    async def noop_runner(job: ProcessingJob) -> None:
        return None

    return ProcessingQueue(
        runner=noop_runner,
        max_workers=max_workers,
        progress_repository=progress_repository,
    )
