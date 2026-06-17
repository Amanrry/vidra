"""Persistence service for normalized video context records."""

from __future__ import annotations

import json
from sqlite3 import Row
from typing import Iterable

from video_qa.models.tools import (
    CaptionRecord,
    ContextType,
    CropRecord,
    DetectionRecord,
    FrameRecord,
    TranscriptSegmentRecord,
    VideoContextRecord,
)
from video_qa.storage import Database


class VideoContextRepository:
    """Idempotent SQLite repository for video-derived multimodal context."""

    def __init__(self, database: Database) -> None:
        self.database = database
        self.database.initialize()

    def upsert_contexts(
        self,
        contexts: Iterable[VideoContextRecord],
    ) -> int:
        records = list(contexts)
        for context in records:
            self.database.execute(
                """
                INSERT INTO video_context
                (context_id, video_id, context_type, timestamp_sec, data, tool_name, model_name)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(context_id) DO UPDATE SET
                    video_id = excluded.video_id,
                    context_type = excluded.context_type,
                    timestamp_sec = excluded.timestamp_sec,
                    data = excluded.data,
                    tool_name = excluded.tool_name,
                    model_name = excluded.model_name
                """,
                (
                    context.context_id,
                    context.video_id,
                    context.context_type.value,
                    context.timestamp_sec,
                    json.dumps(context.data, ensure_ascii=False, sort_keys=True),
                    context.tool_name,
                    context.model_name,
                ),
            )
        return len(records)

    def save_frames(self, frames: Iterable[FrameRecord]) -> int:
        return self.upsert_contexts(
            VideoContextRecord(
                context_id=frame.frame_id,
                video_id=frame.video_id,
                context_type=ContextType.frame,
                timestamp_sec=frame.timestamp_sec,
                data={
                    "frame_id": frame.frame_id,
                    "frame_number": frame.frame_number,
                    "image_path": str(frame.image_path),
                },
                tool_name="extract_frames",
            )
            for frame in frames
        )

    def save_captions(self, captions: Iterable[CaptionRecord]) -> int:
        return self.upsert_contexts(
            VideoContextRecord(
                context_id=f"{caption.video_id}-caption-{caption.frame_id}",
                video_id=caption.video_id,
                context_type=ContextType.caption,
                timestamp_sec=caption.timestamp_sec,
                data={
                    "frame_id": caption.frame_id,
                    "text": caption.text,
                    "confidence": caption.confidence,
                },
                tool_name="caption_frames",
                model_name=caption.model_name,
            )
            for caption in captions
        )

    def save_transcripts(self, segments: Iterable[TranscriptSegmentRecord]) -> int:
        return self.upsert_contexts(
            VideoContextRecord(
                context_id=f"{segment.video_id}-transcript-{index:06d}",
                video_id=segment.video_id,
                context_type=ContextType.transcript,
                timestamp_sec=segment.start_sec,
                data={
                    "start_sec": segment.start_sec,
                    "end_sec": segment.end_sec,
                    "text": segment.text,
                    "confidence": segment.confidence,
                },
                tool_name="transcribe_audio",
            )
            for index, segment in enumerate(segments)
        )

    def save_objects(self, detections: Iterable[DetectionRecord]) -> int:
        return self.upsert_contexts(
            VideoContextRecord(
                context_id=detection.object_id,
                video_id=detection.video_id,
                context_type=ContextType.object,
                timestamp_sec=detection.timestamp_sec,
                data={
                    "object_id": detection.object_id,
                    "frame_id": detection.frame_id,
                    "frame_index": detection.frame_index,
                    "label": detection.label,
                    "confidence": detection.confidence,
                    "bbox": detection.bbox.model_dump(),
                    "frame_path": str(detection.frame_path),
                    "annotated_frame_path": (
                        str(detection.annotated_frame_path)
                        if detection.annotated_frame_path is not None
                        else None
                    ),
                    "crop_path": str(detection.crop_path) if detection.crop_path else None,
                },
                tool_name="detect_objects",
            )
            for detection in detections
        )

    def save_crops(self, crops: Iterable[CropRecord]) -> int:
        return self.upsert_contexts(
            VideoContextRecord(
                context_id=crop.crop_id,
                video_id=crop.video_id,
                context_type=ContextType.crop,
                timestamp_sec=crop.timestamp_sec,
                data={
                    "crop_id": crop.crop_id,
                    "object_id": crop.object_id,
                    "frame_id": crop.frame_id,
                    "label": crop.label,
                    "crop_path": str(crop.crop_path),
                    "embedding_id": crop.embedding_id,
                },
                tool_name="extract_crops",
            )
            for crop in crops
        )

    def count_by_video(
        self,
        video_id: str,
        *,
        context_type: ContextType | None = None,
    ) -> int:
        if context_type is None:
            rows = self.database.query(
                "SELECT COUNT(*) AS count FROM video_context WHERE video_id = ?",
                (video_id,),
            )
        else:
            rows = self.database.query(
                """
                SELECT COUNT(*) AS count
                FROM video_context
                WHERE video_id = ? AND context_type = ?
                """,
                (video_id, context_type.value),
            )
        return int(rows[0]["count"])

    def list_by_video(
        self,
        video_id: str,
        *,
        context_type: ContextType | None = None,
    ) -> list[VideoContextRecord]:
        if context_type is None:
            rows = self.database.query(
                """
                SELECT *
                FROM video_context
                WHERE video_id = ?
                ORDER BY timestamp_sec IS NULL, timestamp_sec, context_id
                """,
                (video_id,),
            )
        else:
            rows = self.database.query(
                """
                SELECT *
                FROM video_context
                WHERE video_id = ? AND context_type = ?
                ORDER BY timestamp_sec IS NULL, timestamp_sec, context_id
                """,
                (video_id, context_type.value),
            )
        return [self._row_to_context(row) for row in rows]

    def _row_to_context(self, row: Row) -> VideoContextRecord:
        return VideoContextRecord(
            context_id=str(row["context_id"]),
            video_id=str(row["video_id"]),
            context_type=ContextType(str(row["context_type"])),
            timestamp_sec=row["timestamp_sec"],
            data=json.loads(str(row["data"])),
            tool_name=str(row["tool_name"]),
            model_name=row["model_name"],
        )
