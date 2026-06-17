from __future__ import annotations

from argparse import Namespace
from pathlib import Path

from video_qa.config import Settings
from video_qa.cli import (
    ask,
    create_processor,
    enqueue_process,
    process_direct,
    run_command,
    search,
    status,
)
from video_qa.models.media import ProcessingStatus
from video_qa.models.processing import JobStatus, ProcessingCounts
from video_qa.models.qa import EvidenceSource, RetrievalHit
from video_qa.services.application import ProcessingResult, UploadResult
from video_qa.services.processing_progress import ProcessingProgressRepository
from video_qa.services.video_context import VideoContextRepository
from video_qa.services.video_processing_service import VideoProcessingService
from video_qa.storage import Database, RunLayout


class FakeVideo:
    def __init__(self, video_id: str, file_path: Path) -> None:
        self.video_id = video_id
        self.file_path = file_path


class FakeApplicationService:
    def __init__(self) -> None:
        self.uploads: list[Path] = []
        self.started: list[str] = []

    def upload_video(self, video_path: Path, video_id: str | None = None):
        self.uploads.append(video_path)
        resolved = video_id or "video-1"
        return UploadResult(
            ok=True,
            message="Video uploaded.",
            video=FakeVideo(resolved, video_path),  # type: ignore[arg-type]
        )

    def start_processing(self, video_id: str):
        self.started.append(video_id)
        return ProcessingResult(
            ok=True,
            message="Processing queued.",
            video_id=video_id,
            status=ProcessingStatus.queued,
            job_id="job-1",
        )

    def get_processing_status(self, video_id: str):
        return ProcessingResult(
            ok=True,
            message="Captioning frames.",
            video_id=video_id,
            status=ProcessingStatus.processing,
            progress_percent=45.0,
        )


class FakeProgressRepository:
    def create_job(self, **kwargs):
        return type(
            "JobRecord",
            (),
            {
                "job_id": kwargs["job_id"],
                "video_id": kwargs["video_id"],
                "run_id": kwargs["run_id"],
                "source_path": Path(kwargs["source_path"]),
                "priority": kwargs["priority"],
            },
        )()

    def mark_processing(self, job_id: str):
        return None


class FakeProcessor:
    async def process_job(self, job):
        return type(
            "ProcessorResult",
            (),
            {
                "job_id": job.job_id,
                "video_id": job.video_id,
                "status": JobStatus.complete,
                "counts": ProcessingCounts(frames_extracted=1),
                "error": None,
            },
        )()


class FakeVectorIndex:
    def query(self, query: str, *, video_id: str | None = None, top_k: int = 5):
        return [
            RetrievalHit(
                id="hit-1",
                modality="text",
                score=0.9,
                source=EvidenceSource(
                    video_id=video_id or "video-1",
                    context_id="caption-1",
                    context_type="caption",
                    timestamp_sec=1.0,
                ),
                text=f"result for {query}",
            )
        ]


class FakePipelineVectorIndex(FakeVectorIndex):
    def upsert_contexts(self, contexts):
        return len(contexts)


class FakeQAAgent:
    def answer(self, *, video_id: str, question: str):
        return type(
            "QAResponse",
            (),
            {
                "video_id": video_id,
                "question": question,
                "answer": "grounded answer",
                "evidence_ids": ["caption-1"],
            },
        )()


def test_cli_enqueue_status_search_and_ask(tmp_path: Path) -> None:
    service = FakeApplicationService()
    video_path = tmp_path / "demo.mp4"
    video_path.write_bytes(b"video")

    enqueued = enqueue_process(service, video_path, video_id="video-1")
    inspected = status(service, "video-1")
    searched = search(FakeVectorIndex(), "person", video_id="video-1", top_k=3)
    answered = ask(FakeQAAgent(), "video-1", "What is visible?")

    assert enqueued["status"] == "queued"
    assert inspected["progress_percent"] == 45.0
    assert searched["hits"][0]["text"] == "result for person"
    assert answered["answer"] == "grounded answer"


def test_cli_direct_process_runs_processor_synchronously(tmp_path: Path) -> None:
    service = FakeApplicationService()
    video_path = tmp_path / "demo.mp4"
    video_path.write_bytes(b"video")

    result = process_direct(
        service,  # type: ignore[arg-type]
        FakeProgressRepository(),  # type: ignore[arg-type]
        FakeProcessor(),  # type: ignore[arg-type]
        video_path,
        video_id="video-direct",
    )

    assert result["ok"]
    assert result["status"] == "complete"
    assert result["counts"]["frames_extracted"] == 1


def test_run_command_dispatches_to_runtime(tmp_path: Path) -> None:
    service = FakeApplicationService()
    video_path = tmp_path / "demo.mp4"
    video_path.write_bytes(b"video")
    runtime = {"app_service": service}

    result = run_command(
        Namespace(command="enqueue", video=str(video_path), video_id="video-1"),
        runtime,
    )

    assert result["video_id"] == "video-1"


def test_create_runtime_for_enqueue_does_not_initialize_vector_dependencies(
    tmp_path: Path,
) -> None:
    from video_qa.cli import create_runtime

    settings = Settings(
        paths={
            "data_dir": tmp_path / "data",
            "input_dir": tmp_path / "data" / "input",
            "runs_dir": tmp_path / "data" / "runs",
            "chroma_dir": tmp_path / "data" / "chroma",
            "database_path": tmp_path / "data" / "vidra.sqlite3",
        }
    )

    runtime = create_runtime(settings)

    assert "app_service" in runtime
    assert "vector_index" not in runtime
    assert "qa_agent" not in runtime


def test_real_processor_uses_configured_model_names(tmp_path: Path) -> None:
    settings = Settings(
        paths={
            "data_dir": tmp_path / "data",
            "input_dir": tmp_path / "data" / "input",
            "runs_dir": tmp_path / "data" / "runs",
            "chroma_dir": tmp_path / "data" / "chroma",
            "database_path": tmp_path / "data" / "vidra.sqlite3",
        },
        models={
            "yolo_model": "custom-yolo.pt",
            "caption_model": "custom-blip",
            "siglip_model": "custom-siglip",
            "whisper_model": "tiny",
            "device": "auto",
        },
    )
    database = Database(settings.paths.database_path)
    database.initialize()
    processor = create_processor(
        settings=settings,
        layout=RunLayout(settings.paths.runs_dir),
        progress_repository=ProcessingProgressRepository(database),
        context_repository=VideoContextRepository(database),
        processing_service=VideoProcessingService(database),
        vector_index=FakePipelineVectorIndex(),  # type: ignore[arg-type]
        real_pipeline=True,
    )

    assert processor.captioner.captioner.backend.model_name == "custom-blip"
    assert processor.enricher.transcriber.backend.model_name == "tiny"
    assert processor.enricher.detector.backend.model_name == "custom-yolo.pt"
