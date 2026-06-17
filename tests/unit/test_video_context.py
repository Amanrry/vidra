from __future__ import annotations

from pathlib import Path

from video_qa.models.tools import (
    BoundingBox,
    CaptionRecord,
    ContextType,
    CropRecord,
    DetectionRecord,
    FrameRecord,
    TranscriptSegmentRecord,
    VideoContextRecord,
)
from video_qa.services.video_context import VideoContextRepository
from video_qa.storage import Database


def insert_video(database: Database, video_id: str = "video-1") -> None:
    database.execute(
        """
        INSERT INTO videos (video_id, filename, file_path, duration_sec)
        VALUES (?, ?, ?, ?)
        """,
        (video_id, f"{video_id}.mp4", f"data/input/{video_id}.mp4", 0.0),
    )


def make_repository(tmp_path: Path) -> VideoContextRepository:
    database = Database(tmp_path / "vidra.sqlite3")
    database.initialize()
    insert_video(database)
    return VideoContextRepository(database)


def test_video_context_repository_stores_all_context_types_idempotently(
    tmp_path: Path,
) -> None:
    repository = make_repository(tmp_path)
    frame = FrameRecord(
        video_id="video-1",
        frame_id="frame-1",
        timestamp_sec=1.0,
        frame_number=30,
        image_path=tmp_path / "frame.jpg",
    )
    caption = CaptionRecord(
        video_id="video-1",
        frame_id="frame-1",
        timestamp_sec=1.0,
        text="a person enters",
        confidence=0.8,
        model_name="fake-captioner",
    )
    transcript = TranscriptSegmentRecord(
        video_id="video-1",
        start_sec=0.5,
        end_sec=2.0,
        text="hello",
        confidence=0.9,
    )
    detection = DetectionRecord(
        video_id="video-1",
        object_id="object-1",
        frame_id="frame-1",
        timestamp_sec=1.0,
        frame_index=30,
        label="person",
        confidence=0.95,
        bbox=BoundingBox(x1=1, y1=2, x2=20, y2=30),
        frame_path=tmp_path / "frame.jpg",
        annotated_frame_path=tmp_path / "annotated.jpg",
        crop_path=tmp_path / "crop.jpg",
    )
    crop = CropRecord(
        video_id="video-1",
        crop_id="crop-1",
        object_id="object-1",
        frame_id="frame-1",
        label="person",
        timestamp_sec=1.0,
        crop_path=tmp_path / "crop.jpg",
    )

    repository.save_frames([frame])
    repository.save_captions([caption])
    repository.save_transcripts([transcript])
    repository.save_objects([detection])
    repository.save_crops([crop])

    assert repository.count_by_video("video-1") == 5
    assert repository.count_by_video("video-1", context_type=ContextType.frame) == 1
    assert repository.count_by_video("video-1", context_type=ContextType.caption) == 1
    assert repository.count_by_video("video-1", context_type=ContextType.transcript) == 1
    assert repository.count_by_video("video-1", context_type=ContextType.object) == 1
    assert repository.count_by_video("video-1", context_type=ContextType.crop) == 1

    caption_update = caption.model_copy(update={"text": "a person enters the room"})
    repository.save_captions([caption_update])

    assert repository.count_by_video("video-1") == 5
    captions = repository.list_by_video("video-1", context_type=ContextType.caption)
    assert captions[0].data["text"] == "a person enters the room"


def test_video_context_repository_upserts_explicit_context_records(
    tmp_path: Path,
) -> None:
    repository = make_repository(tmp_path)
    context = VideoContextRecord(
        context_id="ctx-1",
        video_id="video-1",
        context_type=ContextType.metadata,
        timestamp_sec=None,
        data={"status": "ok"},
        tool_name="test_tool",
        model_name=None,
    )

    assert repository.upsert_contexts([context]) == 1
    assert repository.upsert_contexts([context.model_copy(update={"data": {"status": "updated"}})]) == 1

    records = repository.list_by_video("video-1")
    assert len(records) == 1
    assert records[0].context_id == "ctx-1"
    assert records[0].data == {"status": "updated"}
