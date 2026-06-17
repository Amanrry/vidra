"""Concrete processing stages that wire Vidra tools into the progressive pipeline."""

from __future__ import annotations

from pathlib import Path
from typing import Protocol, runtime_checkable

from video_qa.config import Settings
from video_qa.models.processing import ProcessingCounts
from video_qa.models.tools import ContextType, FrameRecord, VideoContextRecord
from video_qa.services.reporting import ReportWriter
from video_qa.services.vector_index import VideoVectorIndex
from video_qa.services.video_context import VideoContextRepository
from video_qa.services.video_processing_service import VideoProcessingService
from video_qa.services.video_processor import StageResult, VideoProcessingContext
from video_qa.storage import RunLayout
from video_qa.tools import AudioTranscriber, FrameExtractor, ImageCaptioner, ObjectDetector


@runtime_checkable
class FrameExtractorPort(Protocol):
    def get_video_metadata(self, video_path: str | Path): ...

    def extract_frames(
        self,
        video_path: str | Path,
        *,
        video_id: str,
        output_dir: str | Path,
        interval_seconds: float,
        max_frames: int,
    ) -> list[FrameRecord]: ...


class FrameExtractionStage:
    """Extract sampled frames and persist frame/metadata context."""

    def __init__(
        self,
        *,
        settings: Settings,
        layout: RunLayout,
        processing_service: VideoProcessingService,
        extractor: FrameExtractorPort | None = None,
    ) -> None:
        self.settings = settings
        self.layout = layout
        self.processing_service = processing_service
        self.extractor = extractor or FrameExtractor()

    def run(self, context: VideoProcessingContext, counts: ProcessingCounts) -> StageResult:
        run_paths = self.layout.for_run(context.run_id, create=True)
        metadata = self.extractor.get_video_metadata(context.source_path)
        frames = self.extractor.extract_frames(
            context.source_path,
            video_id=context.video_id,
            output_dir=run_paths.frames_dir,
            interval_seconds=self.settings.video.frame_interval_seconds,
            max_frames=self.settings.video.max_frames_per_video,
        )
        records = [
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
        ]
        records.append(
            VideoContextRecord(
                context_id=f"{context.video_id}-metadata",
                video_id=context.video_id,
                context_type=ContextType.metadata,
                timestamp_sec=None,
                data=metadata.model_dump(mode="json"),
                tool_name="extract_frames",
            )
        )
        stored = self.processing_service.store_tool_results(
            video_id=context.video_id,
            tool_name="extract_frames",
            records=records,
            idempotency_key=f"{context.run_id}:extract_frames",
            run_id=context.run_id,
            parameters={
                "interval_seconds": self.settings.video.frame_interval_seconds,
                "max_frames": self.settings.video.max_frames_per_video,
            },
        )
        next_counts = counts.model_copy(
            update={"frames_extracted": stored.frames_extracted or len(frames)}
        )
        return StageResult(
            counts=next_counts,
            message=f"Extracted {next_counts.frames_extracted} frames.",
        )


class CaptionStage:
    """Caption persisted frames."""

    def __init__(
        self,
        *,
        context_repository: VideoContextRepository,
        processing_service: VideoProcessingService,
        captioner: ImageCaptioner | None = None,
    ) -> None:
        self.context_repository = context_repository
        self.processing_service = processing_service
        self.captioner = captioner or ImageCaptioner()

    def run(self, context: VideoProcessingContext, counts: ProcessingCounts) -> StageResult:
        frames = load_frame_records(self.context_repository, context.video_id)
        captions = self.captioner.caption_frames(frames)
        stored = self.processing_service.store_tool_results(
            video_id=context.video_id,
            tool_name="caption_frames",
            records=self.captioner.to_context_records(captions),
            idempotency_key=f"{context.run_id}:caption_frames",
            run_id=context.run_id,
            model_name=self.captioner.backend.model_name,
            parameters={"frames": len(frames)},
        )
        next_counts = counts.model_copy(
            update={"captions_generated": stored.captions_generated or len(captions)}
        )
        return StageResult(
            counts=next_counts,
            message=f"Generated {next_counts.captions_generated} captions.",
        )


class EnrichmentStage:
    """Run optional transcription, object detection, and crop extraction."""

    def __init__(
        self,
        *,
        layout: RunLayout,
        context_repository: VideoContextRepository,
        processing_service: VideoProcessingService,
        transcriber: AudioTranscriber | None = None,
        detector: ObjectDetector | None = None,
    ) -> None:
        self.layout = layout
        self.context_repository = context_repository
        self.processing_service = processing_service
        self.transcriber = transcriber or AudioTranscriber()
        self.detector = detector or ObjectDetector()

    def run(self, context: VideoProcessingContext, counts: ProcessingCounts) -> StageResult:
        frames = load_frame_records(self.context_repository, context.video_id)
        run_paths = self.layout.for_run(context.run_id, create=True)
        warnings: list[str] = []
        transcript_counts = ProcessingCounts()
        detection_counts = ProcessingCounts()

        try:
            transcript = self.transcriber.transcribe_video(
                context.source_path,
                video_id=context.video_id,
            )
            transcript_counts = self.processing_service.store_tool_results(
                video_id=context.video_id,
                tool_name="transcribe_audio",
                records=self.transcriber.to_context_records(transcript),
                idempotency_key=f"{context.run_id}:transcribe_audio",
                run_id=context.run_id,
                model_name=transcript.model_name,
                parameters={"language": transcript.language},
            )
            if transcript.error:
                warnings.append(f"audio transcription unavailable: {transcript.error}")
        except Exception as exc:
            warnings.append(f"audio transcription failed: {exc}")

        try:
            detection_batch = self.detector.detect_frames(
                frames,
                annotated_dir=run_paths.annotated_frames_dir,
                crops_dir=run_paths.crops_dir,
            )
            detection_counts = self.processing_service.store_tool_results(
                video_id=context.video_id,
                tool_name="detect_objects",
                records=self.detector.to_context_records(detection_batch),
                idempotency_key=f"{context.run_id}:detect_objects",
                run_id=context.run_id,
                model_name=self.detector.backend.model_name,
                parameters={"frames": len(frames)},
            )
        except Exception as exc:
            warnings.append(f"object detection failed: {exc}")

        if transcript_counts == ProcessingCounts() and detection_counts == ProcessingCounts():
            raise RuntimeError("; ".join(warnings) or "Video enrichment produced no results.")

        next_counts = counts.model_copy(
            update={
                "transcript_segments": transcript_counts.transcript_segments,
                "detections_created": detection_counts.detections_created,
                "crops_created": detection_counts.crops_created,
            }
        )
        return StageResult(
            counts=next_counts,
            message=(
                f"Enriched with {next_counts.transcript_segments} transcript segments, "
                f"{next_counts.detections_created} detections, "
                f"and {next_counts.crops_created} crops."
            ),
            warnings=tuple(warnings),
        )


class IndexAndReportStage:
    """Index stored context and write final reports."""

    def __init__(
        self,
        *,
        layout: RunLayout,
        context_repository: VideoContextRepository,
        vector_index: VideoVectorIndex,
        report_writer: ReportWriter,
    ) -> None:
        self.layout = layout
        self.context_repository = context_repository
        self.vector_index = vector_index
        self.report_writer = report_writer

    def run(self, context: VideoProcessingContext, counts: ProcessingCounts) -> StageResult:
        contexts = self.context_repository.list_by_video(context.video_id)
        indexed = self.vector_index.upsert_contexts(contexts)
        text_indexed = sum(
            1
            for item in contexts
            if item.context_type
            in {ContextType.caption, ContextType.transcript, ContextType.object}
        )
        image_indexed = max(0, indexed - text_indexed)
        run_paths = self.layout.for_run(context.run_id, create=True)
        self.report_writer.write(video_id=context.video_id, run_paths=run_paths)
        next_counts = counts.model_copy(
            update={
                "text_vectors_indexed": min(text_indexed, indexed),
                "image_vectors_indexed": image_indexed,
            }
        )
        return StageResult(
            counts=next_counts,
            message=f"Indexed {indexed} context records and wrote final reports.",
        )


def load_frame_records(
    context_repository: VideoContextRepository,
    video_id: str,
) -> list[FrameRecord]:
    frames = []
    for context in context_repository.list_by_video(video_id, context_type=ContextType.frame):
        frames.append(
            FrameRecord(
                video_id=context.video_id,
                frame_id=str(context.data["frame_id"]),
                timestamp_sec=float(context.timestamp_sec or 0.0),
                frame_number=int(context.data["frame_number"]),
                image_path=Path(str(context.data["image_path"])),
            )
        )
    return frames
