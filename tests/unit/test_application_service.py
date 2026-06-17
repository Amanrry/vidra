from __future__ import annotations

from pathlib import Path

from video_qa.config import Settings
from video_qa.models.media import ProcessingStatus, RunPaths, VideoRecord
from video_qa.models.processing import JobStatus, ProcessingStage
from video_qa.services.application import (
    ApplicationService,
    ChatPort,
    ProcessingPort,
    ProcessingResult,
)
from video_qa.services.processing_progress import ProcessingProgressRepository
from video_qa.services.processing_queue import ProcessingJob, ProcessingQueue
from video_qa.storage import Database


class RecordingProcessor(ProcessingPort):
    def __init__(self, status: ProcessingStatus = ProcessingStatus.processing) -> None:
        self.calls: list[tuple[VideoRecord, RunPaths]] = []
        self.status = status

    def start(self, video: VideoRecord, run_paths: RunPaths) -> ProcessingResult:
        self.calls.append((video, run_paths))
        return ProcessingResult(
            ok=True,
            message="queued",
            video_id=video.video_id,
            status=self.status,
        )


class RecordingChatAgent(ChatPort):
    def __init__(self) -> None:
        self.calls: list[tuple[VideoRecord, str]] = []

    def answer(self, video: VideoRecord, message: str) -> str:
        self.calls.append((video, message))
        return f"answer for {video.video_id}: {message}"


def make_service(tmp_path: Path) -> ApplicationService:
    settings = Settings(
        paths={
            "data_dir": tmp_path / "data",
            "input_dir": tmp_path / "data" / "input",
            "runs_dir": tmp_path / "data" / "runs",
            "chroma_dir": tmp_path / "data" / "chroma",
            "database_path": tmp_path / "data" / "vidra.sqlite3",
        },
        video={"max_upload_mb": 1},
    )
    database = Database(settings.paths.database_path)
    return ApplicationService(settings, database)


def write_video(path: Path, size: int = 16) -> Path:
    path.write_bytes(b"0" * size)
    return path


def test_upload_video_persists_file_and_video_row(tmp_path: Path) -> None:
    service = make_service(tmp_path)
    source = write_video(tmp_path / "demo.mp4")

    result = service.upload_video(source, video_id="video-1")

    assert result.ok
    assert result.video is not None
    assert result.video.video_id == "video-1"
    assert result.run_paths is not None
    assert result.run_paths.source_dir.is_dir()
    assert (result.run_paths.source_dir / "demo.mp4").read_bytes() == source.read_bytes()
    assert service.get_processing_status("video-1").status == ProcessingStatus.pending


def test_upload_rejects_unsupported_extension(tmp_path: Path) -> None:
    service = make_service(tmp_path)
    source = write_video(tmp_path / "not-video.txt")

    result = service.upload_video(source, video_id="video-1")

    assert not result.ok
    assert "Unsupported video type" in result.message


def test_start_processing_uses_injected_processor_and_updates_status(tmp_path: Path) -> None:
    settings = Settings(
        paths={
            "data_dir": tmp_path / "data",
            "input_dir": tmp_path / "data" / "input",
            "runs_dir": tmp_path / "data" / "runs",
            "chroma_dir": tmp_path / "data" / "chroma",
            "database_path": tmp_path / "data" / "vidra.sqlite3",
        }
    )
    processor = RecordingProcessor(status=ProcessingStatus.complete)
    service = ApplicationService(settings, Database(settings.paths.database_path), processor=processor)
    source = write_video(tmp_path / "demo.mp4")
    upload = service.upload_video(source, video_id="video-2")

    result = service.start_processing(upload.video.video_id)  # type: ignore[union-attr]

    assert result.ok
    assert result.status == ProcessingStatus.complete
    assert len(processor.calls) == 1
    video, run_paths = processor.calls[0]
    assert video.video_id == "video-2"
    assert run_paths.root.name == "video-2"
    assert service.get_processing_status("video-2").status == ProcessingStatus.complete


def test_upload_can_start_processing_immediately(tmp_path: Path) -> None:
    settings = Settings(
        paths={
            "data_dir": tmp_path / "data",
            "input_dir": tmp_path / "data" / "input",
            "runs_dir": tmp_path / "data" / "runs",
            "chroma_dir": tmp_path / "data" / "chroma",
            "database_path": tmp_path / "data" / "vidra.sqlite3",
        }
    )
    processor = RecordingProcessor()
    service = ApplicationService(settings, Database(settings.paths.database_path), processor=processor)
    source = write_video(tmp_path / "demo.mp4")

    result = service.upload_video(source, video_id="video-3", start_processing=True)

    assert result.ok
    assert result.message == "queued"
    assert len(processor.calls) == 1
    assert service.get_processing_status("video-3").status == ProcessingStatus.processing


def test_start_processing_enqueues_without_running_pipeline_in_request_path(
    tmp_path: Path,
) -> None:
    settings = Settings(
        paths={
            "data_dir": tmp_path / "data",
            "input_dir": tmp_path / "data" / "input",
            "runs_dir": tmp_path / "data" / "runs",
            "chroma_dir": tmp_path / "data" / "chroma",
            "database_path": tmp_path / "data" / "vidra.sqlite3",
        }
    )
    database = Database(settings.paths.database_path)
    progress_repository = ProcessingProgressRepository(database)
    runner_calls: list[str] = []

    async def runner(job: ProcessingJob) -> None:
        runner_calls.append(job.video_id)

    queue = ProcessingQueue(
        runner=runner,
        max_workers=1,
        progress_repository=progress_repository,
    )
    service = ApplicationService(
        settings,
        database,
        processing_queue=queue,
        progress_repository=progress_repository,
    )
    source = write_video(tmp_path / "demo.mp4")
    upload = service.upload_video(source, video_id="video-queued")

    result = service.start_processing(upload.video.video_id)  # type: ignore[union-attr]

    assert result.ok
    assert result.status == ProcessingStatus.queued
    assert result.job_id is not None
    assert runner_calls == []

    queued = service.get_processing_status("video-queued")
    assert queued.status == ProcessingStatus.queued
    assert queued.progress_percent == 0.0

    progress_repository.update_job(
        result.job_id,
        status=JobStatus.processing,
        stage=ProcessingStage.captioning,
        progress_percent=45.0,
        message="Captioning frames.",
    )

    status = service.get_processing_status("video-queued")
    assert status.status == ProcessingStatus.processing
    assert status.message == "Captioning frames."
    assert status.progress_percent == 45.0


def test_chat_uses_injected_agent_without_touching_vector_db(tmp_path: Path) -> None:
    settings = Settings(
        paths={
            "data_dir": tmp_path / "data",
            "input_dir": tmp_path / "data" / "input",
            "runs_dir": tmp_path / "data" / "runs",
            "chroma_dir": tmp_path / "data" / "chroma",
            "database_path": tmp_path / "data" / "vidra.sqlite3",
        }
    )
    chat_agent = RecordingChatAgent()
    service = ApplicationService(settings, Database(settings.paths.database_path), chat_agent=chat_agent)
    source = write_video(tmp_path / "demo.mp4")
    service.upload_video(source, video_id="video-4")

    result = service.chat("video-4", " What is visible? ")

    assert result.ok
    assert result.message == "answer for video-4: What is visible?"
    assert len(chat_agent.calls) == 1


def test_chat_validates_empty_message_and_missing_video(tmp_path: Path) -> None:
    service = make_service(tmp_path)

    empty = service.chat("video-404", " ")
    missing = service.chat("video-404", "question")

    assert not empty.ok
    assert "enter a question" in empty.message
    assert not missing.ok
    assert "not found" in missing.message
