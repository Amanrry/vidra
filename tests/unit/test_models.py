from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from video_qa.models import (
    BoundingBox,
    CaptionRecord,
    DetectionRecord,
    FrameRecord,
    RetrievalHit,
    TranscriptSegmentRecord,
    VideoContextRecord,
)
from video_qa.models.qa import EvidenceSource
from video_qa.models.tools import ContextType


def test_frame_caption_and_transcript_records_validate() -> None:
    frame = FrameRecord(
        video_id="video-1",
        frame_id="frame-1",
        timestamp_sec=2.5,
        frame_number=75,
        image_path=Path("frames/frame-1.jpg"),
    )
    caption = CaptionRecord(
        video_id=frame.video_id,
        frame_id=frame.frame_id,
        timestamp_sec=frame.timestamp_sec,
        text="a person walks near a car",
        confidence=0.83,
        model_name="blip",
    )
    transcript = TranscriptSegmentRecord(
        video_id=frame.video_id,
        start_sec=1.0,
        end_sec=3.0,
        text="hello world",
        confidence=0.9,
    )

    assert caption.frame_id == "frame-1"
    assert transcript.end_sec >= transcript.start_sec


def test_detection_record_normalizes_label_and_validates_bbox() -> None:
    detection = DetectionRecord(
        video_id="video-1",
        object_id="object-1",
        frame_id="frame-1",
        timestamp_sec=1.0,
        frame_index=1,
        label=" Person ",
        confidence=0.91,
        bbox=BoundingBox(x1=1, y1=2, x2=20, y2=30),
        frame_path=Path("frames/frame.jpg"),
    )

    assert detection.label == "person"


def test_rejects_invalid_time_range_and_bbox() -> None:
    with pytest.raises(ValidationError, match="end_sec"):
        TranscriptSegmentRecord(
            video_id="video-1",
            start_sec=5.0,
            end_sec=4.0,
            text="bad",
        )

    with pytest.raises(ValidationError, match="x2"):
        BoundingBox(x1=10, y1=0, x2=5, y2=10)


def test_video_context_and_retrieval_contracts() -> None:
    context = VideoContextRecord(
        context_id="ctx-1",
        video_id="video-1",
        context_type=ContextType.caption,
        timestamp_sec=1.0,
        data={"text": "a car enters the frame"},
        tool_name="caption_frames",
        model_name="blip",
    )
    source = EvidenceSource(
        video_id=context.video_id,
        context_id=context.context_id,
        context_type=context.context_type.value,
        timestamp_sec=context.timestamp_sec,
        label="car",
    )
    hit = RetrievalHit(
        id="hit-1",
        modality="text",
        score=0.77,
        source=source,
        text="a car enters the frame",
    )

    assert hit.source.context_id == "ctx-1"
    assert hit.metadata == {}
