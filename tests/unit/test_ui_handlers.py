from __future__ import annotations

from pathlib import Path

from video_qa.models.media import ProcessingStatus
from video_qa.models.qa import EvidenceSource, RetrievalHit
from video_qa.services.application import ProcessingResult, UploadResult
from video_qa.ui.handlers import RefreshReason, UIHandlers, format_progress


class FakeVideo:
    def __init__(self, video_id: str) -> None:
        self.video_id = video_id


class FakeApplicationService:
    def __init__(self, status: ProcessingStatus = ProcessingStatus.queued) -> None:
        self.status = status
        self.uploaded: list[Path] = []
        self.started: list[str] = []
        self.dashboard_calls = 0

    def upload_video(self, path: Path):
        self.uploaded.append(path)
        return UploadResult(
            ok=True,
            message="Video uploaded.",
            video=FakeVideo("video-1"),  # type: ignore[arg-type]
        )

    def start_processing(self, video_id: str):
        self.started.append(video_id)
        return ProcessingResult(
            ok=True,
            message="Processing queued.",
            video_id=video_id,
            status=ProcessingStatus.queued,
            progress_percent=0.0,
        )

    def get_processing_status(self, video_id: str):
        return ProcessingResult(
            ok=True,
            message=f"Video is {self.status.value}.",
            video_id=video_id,
            status=self.status,
            progress_percent=100.0 if self.status == ProcessingStatus.complete else 25.0,
        )

    def chat(self, video_id: str, question: str):
        return type("ChatResult", (), {"ok": True, "message": f"answer: {question}"})()

    def get_video_dashboard(self, video_id: str):
        self.dashboard_calls += 1
        return type(
            "DashboardResult",
            (),
            {
                "ok": True,
                "message": "loaded",
                "source_video_path": Path("demo.mp4"),
                "preview_image_path": None,
                "counts": {"frame": 1, "caption": 1, "transcript": 0, "object": 0, "crop": 0},
                "objects": [],
                "evidence": [],
            },
        )()


class FakeRetriever:
    def query(self, query: str, *, video_id: str | None = None, top_k: int = 5):
        return [
            RetrievalHit(
                id="hit-1",
                modality="text",
                score=0.8,
                source=EvidenceSource(
                    video_id=video_id or "video-1",
                    context_id="caption-1",
                    context_type="caption",
                    timestamp_sec=1.5,
                    label="person",
                ),
                text=f"match {query}",
            )
        ]


def test_upload_and_enqueue_returns_quick_status(tmp_path: Path) -> None:
    video_path = tmp_path / "demo.mp4"
    video_path.write_bytes(b"video")
    service = FakeApplicationService()
    handlers = UIHandlers(application_service=service)  # type: ignore[arg-type]

    video_id, message = handlers.upload_and_enqueue(str(video_path))

    assert video_id == "video-1"
    assert "Video uploaded." in message
    assert "Processing queued." in message
    assert service.started == ["video-1"]


def test_poll_status_renders_active_and_terminal_states() -> None:
    active = UIHandlers(
        application_service=FakeApplicationService(ProcessingStatus.processing)  # type: ignore[arg-type]
    )
    complete = UIHandlers(
        application_service=FakeApplicationService(ProcessingStatus.complete)  # type: ignore[arg-type]
    )

    active_text, keep_polling = active.poll_status("video-1")
    complete_text, complete_polling = complete.poll_status("video-1")

    assert "Processing" in active_text
    assert keep_polling
    assert "Complete" in complete_text
    assert not complete_polling


def test_dashboard_refresh_is_driven_by_progress_signature() -> None:
    service = FakeApplicationService(ProcessingStatus.processing)
    handlers = UIHandlers(application_service=service)  # type: ignore[arg-type]

    snapshot = handlers.progress_snapshot("video-1")
    rendered = handlers.dashboard_if_changed("video-1", snapshot.signature, None)
    skipped = handlers.dashboard_if_changed("video-1", snapshot.signature, snapshot.signature)
    same_snapshot = handlers.progress_snapshot("video-1", snapshot.signature)

    assert snapshot.reason == RefreshReason.changed
    assert same_snapshot.reason == RefreshReason.none
    assert rendered[-1] == snapshot.signature
    assert skipped[-1] == snapshot.signature
    assert service.dashboard_calls == 1


def test_progress_snapshot_marks_terminal_statuses_as_not_pollable() -> None:
    active = UIHandlers(
        application_service=FakeApplicationService(ProcessingStatus.processing)  # type: ignore[arg-type]
    )
    complete = UIHandlers(
        application_service=FakeApplicationService(ProcessingStatus.complete)  # type: ignore[arg-type]
    )
    no_video = UIHandlers(application_service=FakeApplicationService())  # type: ignore[arg-type]

    assert active.progress_snapshot("video-1").should_continue_polling
    assert not complete.progress_snapshot("video-1").should_continue_polling
    assert not no_video.progress_snapshot(None).should_continue_polling


def test_chat_and_search_handlers_are_directly_callable() -> None:
    handlers = UIHandlers(
        application_service=FakeApplicationService(),  # type: ignore[arg-type]
        retriever=FakeRetriever(),
    )

    assert handlers.chat("video-1", "What happens?") == "answer: What happens?"
    rows = handlers.search("video-1", "person", top_k=2)

    assert rows == [["caption", 1.5, "person", 0.8, "match person", ""]]
    assert format_progress(ProcessingStatus.queued, "Queued.", 0.0) == "Queued | 0.0% | Queued."


def test_add_chat_turn_returns_gradio_messages_format() -> None:
    handlers = UIHandlers(application_service=FakeApplicationService())  # type: ignore[arg-type]

    history, textbox = handlers.add_chat_turn("video-1", "What is visible?", None)

    assert textbox == ""
    assert history == [
        {"role": "user", "content": "What is visible?"},
        {"role": "assistant", "content": "answer: What is visible?"},
    ]
