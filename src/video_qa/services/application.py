"""Gradio-facing application facade.

The UI should call this service instead of coordinating files, SQLite rows,
processing workers, retrieval, vector stores, and chat agents directly. This is
an application-service/facade boundary inspired by Bri's middle layer.
"""

from __future__ import annotations

import shutil
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO, Protocol, runtime_checkable

from video_qa.config import Settings
from video_qa.models.media import ProcessingStatus, RunPaths, VideoRecord
from video_qa.storage import Database, RunLayout


SUPPORTED_VIDEO_EXTENSIONS = {".mp4", ".mov", ".mkv", ".avi", ".webm", ".m4v"}


@dataclass(frozen=True)
class UploadResult:
    ok: bool
    message: str
    video: VideoRecord | None = None
    run_paths: RunPaths | None = None


@dataclass(frozen=True)
class ProcessingResult:
    ok: bool
    message: str
    video_id: str
    status: ProcessingStatus


@dataclass(frozen=True)
class ChatResult:
    ok: bool
    message: str
    video_id: str | None = None


@runtime_checkable
class ProcessingPort(Protocol):
    def start(self, video: VideoRecord, run_paths: RunPaths) -> ProcessingResult:
        """Start or enqueue processing for one video."""


@runtime_checkable
class ChatPort(Protocol):
    def answer(self, video: VideoRecord, message: str) -> str:
        """Answer a user question using stored video context."""


class NoopProcessingPort:
    """Safe placeholder until the real VideoProcessor task is implemented."""

    def start(self, video: VideoRecord, run_paths: RunPaths) -> ProcessingResult:
        return ProcessingResult(
            ok=True,
            message="Processing accepted.",
            video_id=video.video_id,
            status=ProcessingStatus.processing,
        )


class NoopChatPort:
    """Safe placeholder until ContextBuilder and QAAgent are implemented."""

    def answer(self, video: VideoRecord, message: str) -> str:
        return (
            "Video context is not indexed yet. Processing and QA services will "
            "provide grounded answers once implemented."
        )


class VideoRepository:
    """Persistence adapter for the videos table."""

    def __init__(self, database: Database) -> None:
        self.database = database

    def add(self, video: VideoRecord) -> None:
        self.database.execute(
            """
            INSERT INTO videos
            (video_id, filename, file_path, duration_sec, status, error)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                video.video_id,
                video.filename,
                str(video.file_path),
                video.duration_sec,
                video.status.value,
                video.error,
            ),
        )

    def get(self, video_id: str) -> VideoRecord | None:
        rows = self.database.query(
            """
            SELECT video_id, filename, file_path, duration_sec, status, error, created_at, updated_at
            FROM videos
            WHERE video_id = ?
            """,
            (video_id,),
        )
        if not rows:
            return None
        row = rows[0]
        return VideoRecord(
            video_id=str(row["video_id"]),
            filename=str(row["filename"]),
            file_path=Path(str(row["file_path"])),
            duration_sec=float(row["duration_sec"]),
            status=ProcessingStatus(str(row["status"])),
            error=row["error"],
        )

    def update_status(
        self,
        video_id: str,
        status: ProcessingStatus,
        error: str | None = None,
    ) -> None:
        self.database.execute(
            """
            UPDATE videos
            SET status = ?, error = ?, updated_at = CURRENT_TIMESTAMP
            WHERE video_id = ?
            """,
            (status.value, error, video_id),
        )


class FileStorage:
    """Filesystem adapter for uploaded source videos."""

    def __init__(self, layout: RunLayout, max_upload_mb: int) -> None:
        self.layout = layout
        self.max_upload_bytes = max_upload_mb * 1024 * 1024

    def save_upload(
        self,
        uploaded_file: str | Path | BinaryIO,
        run_id: str,
        filename: str | None = None,
    ) -> tuple[Path, RunPaths]:
        run_paths = self.layout.for_run(run_id, create=True)
        source_name = self._resolve_filename(uploaded_file, filename)
        self._validate_extension(source_name)
        destination = run_paths.source_dir / source_name

        if isinstance(uploaded_file, str | Path):
            source = Path(uploaded_file)
            if not source.exists():
                raise FileNotFoundError(f"Uploaded file not found: {source}")
            self._validate_size(source.stat().st_size)
            shutil.copy2(source, destination)
        else:
            current = uploaded_file.tell() if uploaded_file.seekable() else None
            uploaded_file.seek(0, 2)
            size = uploaded_file.tell()
            self._validate_size(size)
            uploaded_file.seek(0)
            with destination.open("wb") as output:
                shutil.copyfileobj(uploaded_file, output)
            if current is not None:
                uploaded_file.seek(current)

        return destination, run_paths

    def _resolve_filename(
        self,
        uploaded_file: str | Path | BinaryIO,
        filename: str | None,
    ) -> str:
        raw_name = filename or getattr(uploaded_file, "name", None) or "uploaded.mp4"
        clean_name = Path(str(raw_name)).name
        if not clean_name:
            raise ValueError("filename is required")
        return clean_name

    def _validate_extension(self, filename: str) -> None:
        suffix = Path(filename).suffix.lower()
        if suffix not in SUPPORTED_VIDEO_EXTENSIONS:
            allowed = ", ".join(sorted(SUPPORTED_VIDEO_EXTENSIONS))
            raise ValueError(f"Unsupported video type '{suffix}'. Allowed: {allowed}")

    def _validate_size(self, size_bytes: int) -> None:
        if size_bytes <= 0:
            raise ValueError("Uploaded file is empty")
        if size_bytes > self.max_upload_bytes:
            raise ValueError("Uploaded file exceeds configured size limit")


class ApplicationService:
    """Facade for upload, processing, progress, and chat workflows."""

    def __init__(
        self,
        settings: Settings,
        database: Database,
        *,
        processor: ProcessingPort | None = None,
        chat_agent: ChatPort | None = None,
        file_storage: FileStorage | None = None,
        video_repository: VideoRepository | None = None,
    ) -> None:
        self.settings = settings
        self.database = database
        self.database.initialize()
        self.layout = RunLayout(settings.paths.runs_dir)
        self.file_storage = file_storage or FileStorage(
            self.layout,
            settings.video.max_upload_mb,
        )
        self.videos = video_repository or VideoRepository(database)
        self.processor = processor or NoopProcessingPort()
        self.chat_agent = chat_agent or NoopChatPort()

    def upload_video(
        self,
        uploaded_file: str | Path | BinaryIO | None,
        *,
        start_processing: bool = False,
        filename: str | None = None,
        video_id: str | None = None,
    ) -> UploadResult:
        if uploaded_file is None:
            return UploadResult(ok=False, message="No video file was provided.")

        resolved_video_id = video_id or str(uuid.uuid4())
        try:
            saved_path, run_paths = self.file_storage.save_upload(
                uploaded_file,
                resolved_video_id,
                filename,
            )
            video = VideoRecord(
                video_id=resolved_video_id,
                filename=saved_path.name,
                file_path=saved_path,
                status=ProcessingStatus.pending,
            )
            self.videos.add(video)
            if start_processing:
                process_result = self.start_processing(video.video_id)
                video = self.videos.get(video.video_id) or video
                return UploadResult(
                    ok=process_result.ok,
                    message=process_result.message,
                    video=video,
                    run_paths=run_paths,
                )
            return UploadResult(
                ok=True,
                message="Video uploaded.",
                video=video,
                run_paths=run_paths,
            )
        except Exception as exc:
            return UploadResult(ok=False, message=f"Upload failed: {exc}")

    def start_processing(self, video_id: str) -> ProcessingResult:
        video = self.videos.get(video_id)
        if video is None:
            return ProcessingResult(
                ok=False,
                message="Video not found.",
                video_id=video_id,
                status=ProcessingStatus.error,
            )

        run_paths = self.layout.for_run(video.video_id, create=True)
        self.videos.update_status(video.video_id, ProcessingStatus.processing)
        current_video = self.videos.get(video.video_id) or video
        try:
            result = self.processor.start(current_video, run_paths)
            self.videos.update_status(video.video_id, result.status)
            return result
        except Exception as exc:
            self.videos.update_status(video.video_id, ProcessingStatus.error, str(exc))
            return ProcessingResult(
                ok=False,
                message=f"Processing failed: {exc}",
                video_id=video.video_id,
                status=ProcessingStatus.error,
            )

    def get_processing_status(self, video_id: str) -> ProcessingResult:
        video = self.videos.get(video_id)
        if video is None:
            return ProcessingResult(
                ok=False,
                message="Video not found.",
                video_id=video_id,
                status=ProcessingStatus.error,
            )
        return ProcessingResult(
            ok=True,
            message=f"Video is {video.status.value}.",
            video_id=video.video_id,
            status=video.status,
        )

    def chat(self, video_id: str, message: str) -> ChatResult:
        clean_message = (message or "").strip()
        if not clean_message:
            return ChatResult(ok=False, message="Please enter a question.", video_id=video_id)

        video = self.videos.get(video_id)
        if video is None:
            return ChatResult(ok=False, message="Video not found.", video_id=video_id)

        try:
            answer = self.chat_agent.answer(video, clean_message)
            return ChatResult(ok=True, message=answer, video_id=video.video_id)
        except Exception as exc:
            return ChatResult(ok=False, message=f"Chat failed: {exc}", video_id=video.video_id)
