from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from video_qa.models.processing import (
    JobPriority,
    JobStatus,
    ProcessingCounts,
    ProcessingStage,
)
from video_qa.services.processing_progress import ProcessingProgressRepository
from video_qa.services.video_processor import (
    DuplicateProcessingJobError,
    ProgressiveVideoProcessor,
    StageResult,
    VideoProcessingContext,
)
from video_qa.storage import Database


def insert_video(database: Database, video_id: str = "video-1") -> None:
    database.execute(
        """
        INSERT INTO videos (video_id, filename, file_path, duration_sec)
        VALUES (?, ?, ?, ?)
        """,
        (video_id, f"{video_id}.mp4", f"data/input/{video_id}.mp4", 0.0),
    )


class RecordingStage:
    def __init__(
        self,
        name: str,
        calls: list[str],
        *,
        fail: bool = False,
        warnings: tuple[str, ...] = (),
        block: asyncio.Event | None = None,
    ) -> None:
        self.name = name
        self.calls = calls
        self.fail = fail
        self.warnings = warnings
        self.block = block

    async def run(
        self,
        context: VideoProcessingContext,
        counts: ProcessingCounts,
    ) -> StageResult:
        self.calls.append(self.name)
        if self.block is not None:
            await self.block.wait()
        if self.fail:
            raise RuntimeError(f"{self.name} failed")

        updates = {
            "extract": counts.model_copy(update={"frames_extracted": 3}),
            "caption": counts.model_copy(update={"captions_generated": 3}),
            "enrich": counts.model_copy(update={"detections_created": 2, "crops_created": 1}),
            "index": counts.model_copy(
                update={"text_vectors_indexed": 3, "image_vectors_indexed": 3}
            ),
        }
        return StageResult(
            counts=updates[self.name],
            message=f"{self.name} done",
            warnings=self.warnings,
        )


def make_repo(tmp_path: Path, video_id: str = "video-1") -> ProcessingProgressRepository:
    database = Database(tmp_path / "vidra.sqlite3")
    database.initialize()
    insert_video(database, video_id)
    repo = ProcessingProgressRepository(database)
    repo.create_job(
        job_id="job-1",
        video_id=video_id,
        run_id=video_id,
        source_path=tmp_path / "demo.mp4",
        priority=JobPriority.normal,
    )
    repo.mark_processing("job-1", stage=ProcessingStage.extracting)
    return repo


def test_progressive_processor_runs_stages_in_order_and_persists_progress(
    tmp_path: Path,
) -> None:
    async def scenario() -> None:
        repo = make_repo(tmp_path)
        calls: list[str] = []
        processor = ProgressiveVideoProcessor(
            progress_repository=repo,
            frame_extractor=RecordingStage("extract", calls),
            captioner=RecordingStage("caption", calls),
            enricher=RecordingStage("enrich", calls),
            indexer_reporter=RecordingStage("index", calls),
        )

        result = await processor.process(
            VideoProcessingContext(
                job_id="job-1",
                video_id="video-1",
                run_id="video-1",
                source_path=tmp_path / "demo.mp4",
            )
        )

        assert calls == ["extract", "caption", "enrich", "index"]
        assert result.status == JobStatus.complete
        assert result.counts.frames_extracted == 3
        assert result.counts.image_vectors_indexed == 3

        progress = repo.progress_for_video("video-1")
        assert progress is not None
        assert progress.status == JobStatus.complete
        assert progress.stage == ProcessingStage.complete
        assert progress.progress_percent == 100.0

    asyncio.run(scenario())


def test_optional_enrichment_failure_produces_partial_without_losing_context(
    tmp_path: Path,
) -> None:
    async def scenario() -> None:
        repo = make_repo(tmp_path)
        calls: list[str] = []
        processor = ProgressiveVideoProcessor(
            progress_repository=repo,
            frame_extractor=RecordingStage("extract", calls),
            captioner=RecordingStage("caption", calls),
            enricher=RecordingStage("enrich", calls, fail=True),
            indexer_reporter=RecordingStage("index", calls),
        )

        result = await processor.process(
            VideoProcessingContext(
                job_id="job-1",
                video_id="video-1",
                run_id="video-1",
                source_path=tmp_path / "demo.mp4",
            )
        )

        assert calls == ["extract", "caption", "enrich", "index"]
        assert result.status == JobStatus.partial
        assert result.counts.frames_extracted == 3
        assert result.counts.captions_generated == 3
        assert result.counts.detections_created == 0

        progress = repo.progress_for_video("video-1")
        assert progress is not None
        assert progress.status == JobStatus.partial
        assert progress.error == "enrich failed"

    asyncio.run(scenario())


def test_optional_enrichment_warning_produces_partial_and_keeps_counts(tmp_path: Path) -> None:
    async def scenario() -> None:
        repo = make_repo(tmp_path)
        calls: list[str] = []
        processor = ProgressiveVideoProcessor(
            progress_repository=repo,
            frame_extractor=RecordingStage("extract", calls),
            captioner=RecordingStage("caption", calls),
            enricher=RecordingStage("enrich", calls, warnings=("audio missing",)),
            indexer_reporter=RecordingStage("index", calls),
        )

        result = await processor.process(
            VideoProcessingContext(
                job_id="job-1",
                video_id="video-1",
                run_id="video-1",
                source_path=tmp_path / "demo.mp4",
            )
        )

        assert calls == ["extract", "caption", "enrich", "index"]
        assert result.status == JobStatus.partial
        assert result.counts.detections_created == 2
        assert result.error == "audio missing"

    asyncio.run(scenario())


def test_duplicate_per_video_processing_is_rejected(tmp_path: Path) -> None:
    async def scenario() -> None:
        repo = make_repo(tmp_path)
        release = asyncio.Event()
        calls: list[str] = []
        processor = ProgressiveVideoProcessor(
            progress_repository=repo,
            frame_extractor=RecordingStage("extract", calls, block=release),
            captioner=RecordingStage("caption", calls),
            enricher=RecordingStage("enrich", calls),
            indexer_reporter=RecordingStage("index", calls),
        )
        context = VideoProcessingContext(
            job_id="job-1",
            video_id="video-1",
            run_id="video-1",
            source_path=tmp_path / "demo.mp4",
        )

        first = asyncio.create_task(processor.process(context))
        await asyncio.sleep(0.05)
        with pytest.raises(DuplicateProcessingJobError):
            await processor.process(context)

        release.set()
        await first

    asyncio.run(scenario())
