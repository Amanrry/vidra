"""Gradio-facing application facade.

The UI should call this service instead of coordinating files, SQLite rows,
processing workers, retrieval, vector stores, and chat agents directly. This is
an application-service/facade boundary inspired by Bri's middle layer.
"""

from __future__ import annotations

import asyncio
import shutil
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO, Protocol, runtime_checkable

from video_qa.config import Settings
from video_qa.models.media import ProcessingStatus, RunPaths, VideoRecord
from video_qa.models.processing import JobPriority, JobStatus
from video_qa.models.tools import ContextType, VideoContextRecord
from video_qa.services.processing_progress import ProcessingProgressRepository
from video_qa.services.processing_queue import ProcessingQueue, create_noop_queue
from video_qa.services.video_context import VideoContextRepository
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
    job_id: str | None = None
    progress_percent: float | None = None


@dataclass(frozen=True)
class ChatResult:
    ok: bool
    message: str
    video_id: str | None = None


@dataclass(frozen=True)
class VideoDashboardResult:
    """Read-only UI projection for video evidence and visual processing output."""

    ok: bool
    message: str
    video_id: str | None = None
    source_video_path: Path | None = None
    preview_image_path: Path | None = None
    counts: dict[str, int] | None = None
    objects: list[dict[str, object]] | None = None
    evidence: list[dict[str, object]] | None = None


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
            SELECT
                video_id, filename, file_path, duration_sec,
                status, error, created_at, updated_at
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
        processing_queue: ProcessingQueue | None = None,
        progress_repository: ProcessingProgressRepository | None = None,
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
        self.progress_repository = progress_repository or ProcessingProgressRepository(database)
        self.video_contexts = VideoContextRepository(database)
        self.processing_queue = processing_queue
        if self.processing_queue is None and processor is None:
            self.processing_queue = create_noop_queue(
                progress_repository=self.progress_repository,
            )
        self.processor = processor
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
        try:
            if self.processing_queue is not None:
                job = self._run_async(
                    self.processing_queue.add_job(
                        video_id=video.video_id,
                        run_id=run_paths.run_id,
                        source_path=video.file_path,
                        priority=JobPriority.normal,
                    )
                )
                return ProcessingResult(
                    ok=True,
                    message="Processing queued.",
                    video_id=video.video_id,
                    status=ProcessingStatus.queued,
                    job_id=job.job_id,
                    progress_percent=0.0,
                )

            if self.processor is None:
                self.processor = NoopProcessingPort()
            self.videos.update_status(video.video_id, ProcessingStatus.processing)
            current_video = self.videos.get(video.video_id) or video
            result = self.processor.start(current_video, run_paths)
            self.videos.update_status(video.video_id, result.status)
            return result
        except Exception as exc:
            self.videos.update_status(video.video_id, ProcessingStatus.error, str(exc))
            return ProcessingResult(
                ok=False,
                message=f"Processing could not be queued: {exc}",
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
        progress = self.progress_repository.progress_for_video(video_id)
        if progress is not None:
            status = self._processing_status_from_job(progress.status)
            return ProcessingResult(
                ok=True,
                message=progress.message or f"Video is {status.value}.",
                video_id=video.video_id,
                status=status,
                progress_percent=progress.progress_percent,
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

    def get_video_dashboard(self, video_id: str) -> VideoDashboardResult:
        """Return a product-facing projection for Gradio without exposing storage details."""

        video = self.videos.get(video_id)
        if video is None:
            return VideoDashboardResult(ok=False, message="Video not found.", video_id=video_id)

        contexts = self.video_contexts.list_by_video(video_id)
        counts = {
            context_type.value: self.video_contexts.count_by_video(
                video_id,
                context_type=context_type,
            )
            for context_type in ContextType
        }
        preview_path = self._select_preview_image(contexts)
        objects = self._build_object_rows(contexts)
        evidence = self._build_evidence_rows(contexts)
        return VideoDashboardResult(
            ok=True,
            message="Video dashboard loaded.",
            video_id=video.video_id,
            source_video_path=video.file_path,
            preview_image_path=preview_path,
            counts=counts,
            objects=objects,
            evidence=evidence,
        )

    def _run_async(self, awaitable):
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(awaitable)
        raise RuntimeError("start_processing must be called from a synchronous handler")

    def _processing_status_from_job(self, status: JobStatus) -> ProcessingStatus:
        mapping = {
            JobStatus.queued: ProcessingStatus.queued,
            JobStatus.processing: ProcessingStatus.processing,
            JobStatus.complete: ProcessingStatus.complete,
            JobStatus.partial: ProcessingStatus.partial,
            JobStatus.failed: ProcessingStatus.failed,
            JobStatus.cancelled: ProcessingStatus.cancelled,
        }
        return mapping[status]

    def _select_preview_image(self, contexts: list[VideoContextRecord]) -> Path | None:
        objects = [
            context
            for context in contexts
            if context.context_type == ContextType.object
            and context.data.get("annotated_frame_path")
        ]
        for context in reversed(objects):
            path = Path(str(context.data["annotated_frame_path"]))
            if path.exists():
                return path

        if contexts:
            video_id = contexts[0].video_id
            annotated_dir = self.layout.for_run(video_id).annotated_frames_dir
            if annotated_dir.exists():
                annotated_images = sorted(
                    [
                        path
                        for path in annotated_dir.iterdir()
                        if path.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}
                    ],
                    key=lambda path: path.stat().st_mtime,
                )
                if annotated_images:
                    return annotated_images[-1]

        frames = [
            context
            for context in contexts
            if context.context_type == ContextType.frame and context.data.get("image_path")
        ]
        for context in reversed(frames):
            path = Path(str(context.data["image_path"]))
            if path.exists():
                return path
        return None

    def _build_object_rows(
        self,
        contexts: list[VideoContextRecord],
        *,
        limit: int = 80,
    ) -> list[dict[str, object]]:
        rows: list[dict[str, object]] = []
        for context in contexts:
            if context.context_type != ContextType.object:
                continue
            rows.append(
                {
                    "time": context.timestamp_sec,
                    "label": context.data.get("label", ""),
                    "confidence": context.data.get("confidence", ""),
                    "frame": context.data.get("frame_id", ""),
                    "crop": context.data.get("crop_path") or "",
                }
            )
        return rows[-limit:]

    def _build_evidence_rows(
        self,
        contexts: list[VideoContextRecord],
        *,
        limit: int = 80,
    ) -> list[dict[str, object]]:
        evidence_types = {
            ContextType.caption,
            ContextType.transcript,
            ContextType.object,
            ContextType.crop,
        }
        rows: list[dict[str, object]] = []
        for context in contexts:
            if context.context_type not in evidence_types:
                continue
            text = (
                context.data.get("text")
                or context.data.get("label")
                or context.data.get("crop_id")
                or ""
            )
            rows.append(
                {
                    "type": context.context_type.value,
                    "time": context.timestamp_sec,
                    "text": text,
                    "media": (
                        context.data.get("crop_path")
                        or context.data.get("annotated_frame_path")
                        or context.data.get("image_path")
                        or ""
                    ),
                }
            )
        return rows[-limit:]
