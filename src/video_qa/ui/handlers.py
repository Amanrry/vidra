"""Directly testable Gradio event handlers."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any, TypedDict

from video_qa.models.media import ProcessingStatus
from video_qa.services.application import ApplicationService
from video_qa.services.queue_runtime import QueueWorkerRuntime

TERMINAL_STATUSES = {
    ProcessingStatus.complete,
    ProcessingStatus.partial,
    ProcessingStatus.failed,
    ProcessingStatus.cancelled,
    ProcessingStatus.error,
}


class RefreshReason(StrEnum):
    none = "none"
    changed = "changed"
    terminal = "terminal"


class ChatMessage(TypedDict):
    role: str
    content: str


@dataclass(frozen=True)
class ProgressSnapshot:
    text: str
    signature: str
    should_refresh_dashboard: bool
    reason: RefreshReason
    should_continue_polling: bool


@dataclass
class UIHandlers:
    """Presentation boundary for upload, progress polling, search, and chat."""

    application_service: ApplicationService
    retriever: Any | None = None
    queue_runtime: QueueWorkerRuntime | None = None

    def upload_and_enqueue(self, file_value: Any) -> tuple[str | None, str]:
        path = self._file_to_path(file_value)
        if path is None:
            return None, "No video file selected."
        upload = self.application_service.upload_video(path)
        if not upload.ok or upload.video is None:
            return None, upload.message
        process = self.application_service.start_processing(upload.video.video_id)
        return upload.video.video_id, (
            f"{upload.message} {process.message} "
            f"Status: {process.status.value}."
        ).strip()

    def poll_status(self, video_id: str | None) -> tuple[str, bool]:
        if not video_id:
            return "No active video.", False
        result = self.application_service.get_processing_status(video_id)
        return format_progress(
            result.status,
            result.message,
            result.progress_percent,
        ), result.status not in TERMINAL_STATUSES

    def poll_status_snapshot(
        self,
        video_id: str | None,
        previous_signature: str | None = None,
    ) -> tuple[str, str]:
        snapshot = self.progress_snapshot(video_id, previous_signature)
        return snapshot.text, snapshot.signature

    def progress_snapshot(
        self,
        video_id: str | None,
        previous_signature: str | None = None,
    ) -> ProgressSnapshot:
        if not video_id:
            signature = "no-video"
            changed = signature != previous_signature
            return ProgressSnapshot(
                text="No active video.",
                signature=signature,
                should_refresh_dashboard=changed,
                reason=RefreshReason.changed if changed else RefreshReason.none,
                should_continue_polling=False,
            )
        result = self.application_service.get_processing_status(video_id)
        text = format_progress(result.status, result.message, result.progress_percent)
        signature = progress_signature(
            video_id,
            result.status,
            result.progress_percent,
            result.message,
        )
        is_terminal = result.status in TERMINAL_STATUSES
        changed = signature != previous_signature
        return ProgressSnapshot(
            text=text,
            signature=signature,
            should_refresh_dashboard=changed,
            reason=(
                RefreshReason.terminal
                if is_terminal and changed
                else RefreshReason.changed
                if changed
                else RefreshReason.none
            ),
            should_continue_polling=not is_terminal,
        )

    def dashboard(
        self,
        video_id: str | None,
    ) -> tuple[str | None, str | None, str, list[list[Any]], list[list[Any]]]:
        if not video_id:
            return None, None, "Upload a video to see processing results.", [], []
        result = self.application_service.get_video_dashboard(video_id)
        if not result.ok:
            return None, None, result.message, [], []

        object_rows = [
            [
                format_timestamp(row.get("time")),
                row.get("label", ""),
                format_confidence(row.get("confidence")),
                row.get("frame", ""),
                row.get("crop", ""),
            ]
            for row in result.objects or []
        ]
        evidence_rows = [
            [
                row.get("type", ""),
                format_timestamp(row.get("time")),
                row.get("text", ""),
                row.get("media", ""),
            ]
            for row in result.evidence or []
        ]
        return (
            str(result.source_video_path) if result.source_video_path else None,
            str(result.preview_image_path) if result.preview_image_path else None,
            format_dashboard_summary(result.counts or {}),
            object_rows,
            evidence_rows,
        )

    def dashboard_if_changed(
        self,
        video_id: str | None,
        current_signature: str | None,
        rendered_signature: str | None,
    ) -> tuple[Any, Any, Any, Any, Any, str | None]:
        if not video_id:
            return (*self.dashboard(video_id), current_signature)
        if current_signature and current_signature == rendered_signature:
            return (
                no_update(),
                no_update(),
                no_update(),
                no_update(),
                no_update(),
                rendered_signature,
            )
        return (*self.dashboard(video_id), current_signature)

    def chat(self, video_id: str | None, question: str) -> str:
        if not video_id:
            return "Upload and process a video before asking questions."
        result = self.application_service.chat(video_id, question)
        return result.message

    def add_chat_turn(
        self,
        video_id: str | None,
        message: str,
        history: list[ChatMessage] | list[tuple[str, str]] | None,
    ) -> tuple[list[ChatMessage], str]:
        clean_message = (message or "").strip()
        current_history = normalize_chat_history(history)
        if not clean_message:
            return current_history, ""
        answer = self.chat(video_id, clean_message)
        current_history.extend(
            [
                {"role": "user", "content": clean_message},
                {"role": "assistant", "content": answer},
            ]
        )
        return current_history, ""

    def search(self, video_id: str | None, query: str, top_k: int = 5) -> list[list[Any]]:
        if self.retriever is None or not query.strip():
            return []
        hits = self.retriever.query(query, video_id=video_id, top_k=top_k)
        return [
            [
                hit.source.context_type,
                hit.source.timestamp_sec,
                hit.source.label or "",
                round(hit.score, 4),
                hit.text or "",
                str(hit.source.crop_path or hit.source.frame_path or ""),
            ]
            for hit in hits
        ]

    def start_workers(self) -> None:
        if self.queue_runtime is not None:
            self.queue_runtime.start()

    def stop_workers(self) -> None:
        if self.queue_runtime is not None:
            self.queue_runtime.stop()

    def _file_to_path(self, file_value: Any) -> Path | None:
        if file_value is None:
            return None
        if isinstance(file_value, str | Path):
            return Path(file_value)
        path = getattr(file_value, "name", None) or getattr(file_value, "path", None)
        return Path(path) if path else None


def status_badge(status: ProcessingStatus) -> str:
    labels = {
        ProcessingStatus.pending: "Pending",
        ProcessingStatus.queued: "Queued",
        ProcessingStatus.processing: "Processing",
        ProcessingStatus.complete: "Complete",
        ProcessingStatus.partial: "Partial",
        ProcessingStatus.failed: "Failed",
        ProcessingStatus.cancelled: "Cancelled",
        ProcessingStatus.error: "Error",
    }
    return labels[status]


def format_progress(
    status: ProcessingStatus,
    message: str,
    progress_percent: float | None,
) -> str:
    progress = 0.0 if progress_percent is None else progress_percent
    return f"{status_badge(status)} | {progress:.1f}% | {message}"


def progress_signature(
    video_id: str,
    status: ProcessingStatus,
    progress_percent: float | None,
    message: str,
) -> str:
    progress = 0.0 if progress_percent is None else round(progress_percent, 1)
    return f"{video_id}:{status.value}:{progress}:{message}"


def no_update() -> Any:
    try:
        import gradio as gr  # type: ignore[import-not-found]
    except Exception:  # pragma: no cover - only used outside Gradio in unusual tests
        return None
    return gr.update()


def normalize_chat_history(
    history: list[ChatMessage] | list[tuple[str, str]] | None,
) -> list[ChatMessage]:
    messages: list[ChatMessage] = []
    for item in history or []:
        if isinstance(item, dict):
            role = str(item.get("role") or "")
            content = str(item.get("content") or "")
            if role and content:
                messages.append({"role": role, "content": content})
            continue
        if isinstance(item, tuple | list) and len(item) == 2:
            user_message, assistant_message = item
            if user_message:
                messages.append({"role": "user", "content": str(user_message)})
            if assistant_message:
                messages.append({"role": "assistant", "content": str(assistant_message)})
    return messages


def format_dashboard_summary(counts: dict[str, int]) -> str:
    frames = counts.get("frame", 0)
    captions = counts.get("caption", 0)
    transcripts = counts.get("transcript", 0)
    objects = counts.get("object", 0)
    crops = counts.get("crop", 0)
    return (
        f"Frames: {frames} | Captions: {captions} | "
        f"Transcripts: {transcripts} | Objects: {objects} | Crops: {crops}"
    )


def format_timestamp(value: object) -> str:
    if value is None or value == "":
        return ""
    try:
        seconds = float(value)
    except (TypeError, ValueError):
        return str(value)
    minutes, sec = divmod(seconds, 60)
    return f"{int(minutes):02d}:{sec:05.2f}"


def format_confidence(value: object) -> str:
    if value is None or value == "":
        return ""
    try:
        return f"{float(value):.2f}"
    except (TypeError, ValueError):
        return str(value)
