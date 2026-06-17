"""Progressive video processing orchestration.

The processor mirrors Bri's local progressive pipeline while keeping Vidra's
tooling behind small ports. Heavy model adapters can be attached later without
changing queue or UI request handling.
"""

from __future__ import annotations

import asyncio
import inspect
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol, runtime_checkable

from video_qa.models.processing import (
    JobStatus,
    ProcessingCounts,
    ProcessingStage,
)
from video_qa.services.processing_progress import ProcessingProgressRepository
from video_qa.services.processing_queue import ProcessingJob


class DuplicateProcessingJobError(RuntimeError):
    """Raised when a second worker tries to run the same video."""


@dataclass(frozen=True)
class StageResult:
    """Normalized output from one processing stage."""

    counts: ProcessingCounts = field(default_factory=ProcessingCounts)
    message: str | None = None
    warnings: tuple[str, ...] = ()


@dataclass(frozen=True)
class ProcessorResult:
    """Terminal result returned to the processing queue."""

    job_id: str
    video_id: str
    status: JobStatus
    counts: ProcessingCounts
    error: str | None = None


@dataclass(frozen=True)
class VideoProcessingContext:
    """Stable per-job context passed to stage adapters."""

    job_id: str
    video_id: str
    run_id: str
    source_path: Path


@runtime_checkable
class ProcessingStagePort(Protocol):
    def run(
        self,
        context: VideoProcessingContext,
        counts: ProcessingCounts,
    ) -> StageResult | Awaitable[StageResult]:
        """Run one stage and return cumulative or incremental counts."""


ProgressCallback = Callable[[str], None]


class NoopStage:
    """Default stage adapter used before real ML tools are wired."""

    def __init__(self, message: str) -> None:
        self.message = message

    def run(
        self,
        context: VideoProcessingContext,
        counts: ProcessingCounts,
    ) -> StageResult:
        return StageResult(counts=counts, message=self.message)


class ProgressiveVideoProcessor:
    """Coordinate staged video intelligence and durable progress snapshots."""

    def __init__(
        self,
        *,
        progress_repository: ProcessingProgressRepository,
        frame_extractor: ProcessingStagePort | None = None,
        captioner: ProcessingStagePort | None = None,
        enricher: ProcessingStagePort | None = None,
        indexer_reporter: ProcessingStagePort | None = None,
    ) -> None:
        self.progress_repository = progress_repository
        self.frame_extractor = frame_extractor or NoopStage("Frame extraction skipped.")
        self.captioner = captioner or NoopStage("Frame captioning skipped.")
        self.enricher = enricher or NoopStage("Video enrichment skipped.")
        self.indexer_reporter = indexer_reporter or NoopStage("Indexing and reporting skipped.")
        self._job_locks_guard = asyncio.Lock()
        self._job_locks: dict[str, asyncio.Lock] = {}

    async def process_job(self, job: ProcessingJob) -> ProcessorResult:
        context = VideoProcessingContext(
            job_id=job.job_id,
            video_id=job.video_id,
            run_id=job.run_id,
            source_path=job.source_path,
        )
        return await self.process(context)

    async def process(self, context: VideoProcessingContext) -> ProcessorResult:
        job_lock = await self._reserve_video_job(context.video_id)
        counts = ProcessingCounts()
        optional_errors: list[str] = []
        try:
            counts = await self._run_required_stage(
                context,
                stage=ProcessingStage.extracting,
                progress_percent=15.0,
                message="Extracting representative frames.",
                adapter=self.frame_extractor,
                counts=counts,
                complete_progress=30.0,
            )
            counts = await self._run_required_stage(
                context,
                stage=ProcessingStage.captioning,
                progress_percent=35.0,
                message="Captioning sampled frames.",
                adapter=self.captioner,
                counts=counts,
                complete_progress=55.0,
            )
            counts = await self._run_optional_stage(
                context,
                stage=ProcessingStage.enriching,
                progress_percent=60.0,
                message="Enriching video with optional signals.",
                adapter=self.enricher,
                counts=counts,
                complete_progress=78.0,
                errors=optional_errors,
            )
            counts = await self._run_required_stage(
                context,
                stage=ProcessingStage.indexing,
                progress_percent=82.0,
                message="Indexing context and preparing reports.",
                adapter=self.indexer_reporter,
                counts=counts,
                complete_progress=95.0,
            )

            status = JobStatus.partial if optional_errors else JobStatus.complete
            message = (
                "Processing partially complete."
                if status == JobStatus.partial
                else "Processing complete."
            )
            error = "; ".join(optional_errors) if optional_errors else None
            self.progress_repository.mark_complete(
                context.job_id,
                status=status,
                message=message,
                counts=counts,
                error=error,
            )
            return ProcessorResult(
                job_id=context.job_id,
                video_id=context.video_id,
                status=status,
                counts=counts,
                error=error,
            )
        except Exception as exc:
            self.progress_repository.mark_complete(
                context.job_id,
                status=JobStatus.failed,
                message="Processing failed.",
                counts=counts,
                error=str(exc),
            )
            raise
        finally:
            await self._release_video_job(context.video_id, job_lock)

    async def _reserve_video_job(self, video_id: str) -> asyncio.Lock:
        async with self._job_locks_guard:
            job_lock = self._job_locks.setdefault(video_id, asyncio.Lock())
            if job_lock.locked():
                raise DuplicateProcessingJobError(f"Video {video_id} is already processing")
            await job_lock.acquire()
            return job_lock

    async def _release_video_job(self, video_id: str, job_lock: asyncio.Lock) -> None:
        async with self._job_locks_guard:
            job_lock.release()
            if not job_lock.locked():
                self._job_locks.pop(video_id, None)

    async def _run_required_stage(
        self,
        context: VideoProcessingContext,
        *,
        stage: ProcessingStage,
        progress_percent: float,
        message: str,
        adapter: ProcessingStagePort,
        counts: ProcessingCounts,
        complete_progress: float,
    ) -> ProcessingCounts:
        self.progress_repository.update_job(
            context.job_id,
            status=JobStatus.processing,
            stage=stage,
            progress_percent=progress_percent,
            message=message,
            counts=counts,
        )
        result = await self._run_adapter(adapter, context, counts)
        next_counts = result.counts
        self.progress_repository.update_job(
            context.job_id,
            status=JobStatus.processing,
            stage=stage,
            progress_percent=complete_progress,
            message=result.message or f"{stage.value.title()} complete.",
            counts=next_counts,
        )
        return next_counts

    async def _run_optional_stage(
        self,
        context: VideoProcessingContext,
        *,
        stage: ProcessingStage,
        progress_percent: float,
        message: str,
        adapter: ProcessingStagePort,
        counts: ProcessingCounts,
        complete_progress: float,
        errors: list[str],
    ) -> ProcessingCounts:
        self.progress_repository.update_job(
            context.job_id,
            status=JobStatus.processing,
            stage=stage,
            progress_percent=progress_percent,
            message=message,
            counts=counts,
        )
        try:
            result = await self._run_adapter(adapter, context, counts)
        except Exception as exc:
            errors.append(str(exc))
            self.progress_repository.update_job(
                context.job_id,
                status=JobStatus.processing,
                stage=stage,
                progress_percent=complete_progress,
                message=f"Optional enrichment failed: {exc}",
                counts=counts,
                error=str(exc),
            )
            return counts

        next_counts = result.counts
        errors.extend(result.warnings)
        self.progress_repository.update_job(
            context.job_id,
            status=JobStatus.processing,
            stage=stage,
            progress_percent=complete_progress,
            message=result.message or "Optional enrichment complete.",
            counts=next_counts,
            error=None,
        )
        return next_counts

    async def _run_adapter(
        self,
        adapter: ProcessingStagePort,
        context: VideoProcessingContext,
        counts: ProcessingCounts,
    ) -> StageResult:
        result = adapter.run(context, counts)
        if inspect.isawaitable(result):
            result = await result
        if not isinstance(result, StageResult):
            raise TypeError("processing stage adapters must return StageResult")
        return result
