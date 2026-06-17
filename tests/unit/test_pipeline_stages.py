from __future__ import annotations

from pathlib import Path

from video_qa.config import Settings
from video_qa.models.processing import ProcessingCounts
from video_qa.models.tools import (
    BoundingBox,
    CaptionRecord,
    ContextType,
    DetectionRecord,
    FrameRecord,
    TranscriptSegmentRecord,
    VideoContextRecord,
)
from video_qa.services.embeddings import EmbeddingService, FakeEmbeddingBackend
from video_qa.services.pipeline_stages import (
    CaptionStage,
    EnrichmentStage,
    FrameExtractionStage,
    IndexAndReportStage,
)
from video_qa.services.reporting import ReportWriter
from video_qa.services.vector_index import VideoVectorIndex
from video_qa.services.video_context import VideoContextRepository
from video_qa.services.video_processing_service import VideoProcessingService
from video_qa.services.video_processor import VideoProcessingContext
from video_qa.storage import Database, RunLayout
from video_qa.tools.audio_transcriber import TranscriptResult
from video_qa.tools.object_detector import ObjectDetectionBatch


class FakeExtractor:
    def __init__(self, frame_path: Path) -> None:
        self.frame_path = frame_path

    def get_video_metadata(self, video_path: str | Path):
        return type(
            "Metadata",
            (),
            {
                "model_dump": lambda self, mode="json": {
                    "duration_sec": 2.0,
                    "fps": 1.0,
                    "frame_count": 2,
                    "width": 10,
                    "height": 10,
                    "codec": "fake",
                    "file_size_bytes": 5,
                }
            },
        )()

    def extract_frames(self, video_path, *, video_id, output_dir, interval_seconds, max_frames):
        self.frame_path.write_bytes(b"fake-frame")
        return [
            FrameRecord(
                video_id=video_id,
                frame_id=f"{video_id}-frame-000000",
                timestamp_sec=0.0,
                frame_number=0,
                image_path=self.frame_path,
            )
        ]


class FakeCaptioner:
    class Backend:
        model_name = "fake-captioner"

    backend = Backend()

    def caption_frames(self, frames):
        return [
            CaptionRecord(
                video_id=frame.video_id,
                frame_id=frame.frame_id,
                timestamp_sec=frame.timestamp_sec,
                text="a person near a car",
                model_name="fake-captioner",
            )
            for frame in frames
        ]

    def to_context_records(self, captions):
        from video_qa.tools.image_captioner import ImageCaptioner

        return ImageCaptioner(backend=self.backend).to_context_records(captions)


class FakeTranscriber:
    def transcribe_video(self, media_path, *, video_id, language=None):
        return TranscriptResult(
            video_id=video_id,
            segments=[
                TranscriptSegmentRecord(
                    video_id=video_id,
                    start_sec=0.0,
                    end_sec=1.0,
                    text="hello",
                )
            ],
            language="en",
            model_name="fake-whisper",
        )

    def to_context_records(self, transcript):
        from video_qa.tools.audio_transcriber import AudioTranscriber

        fake_backend = type("Backend", (), {"model_name": "fake-whisper"})()
        return AudioTranscriber(backend=fake_backend).to_context_records(transcript)


class FailingTranscriber(FakeTranscriber):
    def transcribe_video(self, media_path, *, video_id, language=None):
        raise RuntimeError("ffmpeg could not read audio")


class FakeDetector:
    class Backend:
        model_name = "fake-yolo"

    backend = Backend()

    def detect_frames(self, frames, *, annotated_dir, crops_dir):
        crop = Path(crops_dir) / "crop.jpg"
        crop.write_bytes(b"crop")
        annotated = Path(annotated_dir) / "annotated.jpg"
        annotated.write_bytes(b"annotated")
        detection = DetectionRecord(
            video_id=frames[0].video_id,
            object_id="object-1",
            frame_id=frames[0].frame_id,
            timestamp_sec=0.0,
            frame_index=0,
            label="person",
            confidence=0.9,
            bbox=BoundingBox(x1=0, y1=0, x2=1, y2=1),
            frame_path=frames[0].image_path,
            annotated_frame_path=annotated,
            crop_path=crop,
        )
        from video_qa.models.tools import CropRecord

        return ObjectDetectionBatch(
            detections=[detection],
            crops=[
                CropRecord(
                    video_id=frames[0].video_id,
                    crop_id="crop-1",
                    object_id="object-1",
                    frame_id=frames[0].frame_id,
                    label="person",
                    timestamp_sec=0.0,
                    crop_path=crop,
                )
            ],
        )

    def to_context_records(self, batch):
        from video_qa.tools.object_detector import ObjectDetector

        return ObjectDetector(backend=self.backend).to_context_records(batch)


class MemoryCollection:
    def __init__(self) -> None:
        self.items = {}

    def upsert(self, *, ids, embeddings, documents, metadatas):
        for index, item_id in enumerate(ids):
            self.items[item_id] = (embeddings[index], documents[index], metadatas[index])

    def query(self, *, query_embeddings, n_results, where=None):
        return {"ids": [[]], "documents": [[]], "metadatas": [[]], "distances": [[]]}

    def count(self):
        return len(self.items)


def test_real_pipeline_stage_adapters_write_context_index_and_reports(
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
    settings.ensure_directories()
    database = Database(settings.paths.database_path)
    database.initialize()
    source = tmp_path / "demo.mp4"
    source.write_bytes(b"video")
    database.execute(
        "INSERT INTO videos (video_id, filename, file_path, duration_sec) VALUES (?, ?, ?, ?)",
        ("video-1", "demo.mp4", str(source), 0.0),
    )
    layout = RunLayout(settings.paths.runs_dir)
    context_repo = VideoContextRepository(database)
    processing = VideoProcessingService(database)
    run_paths = layout.for_run("video-1", create=True)
    context = VideoProcessingContext(
        job_id="job-1",
        video_id="video-1",
        run_id="video-1",
        source_path=source,
    )

    counts = ProcessingCounts()
    counts = FrameExtractionStage(
        settings=settings,
        layout=layout,
        processing_service=processing,
        extractor=FakeExtractor(run_paths.frames_dir / "frame.jpg"),
    ).run(context, counts).counts
    counts = CaptionStage(
        context_repository=context_repo,
        processing_service=processing,
        captioner=FakeCaptioner(),
    ).run(context, counts).counts
    counts = EnrichmentStage(
        layout=layout,
        context_repository=context_repo,
        processing_service=processing,
        transcriber=FakeTranscriber(),
        detector=FakeDetector(),
    ).run(context, counts).counts
    vector = VideoVectorIndex(
        collection=MemoryCollection(),
        embedder=EmbeddingService(FakeEmbeddingBackend()),
    )
    counts = IndexAndReportStage(
        layout=layout,
        context_repository=context_repo,
        vector_index=vector,
        report_writer=ReportWriter(
            context_repository=context_repo,
            processing_service=processing,
        ),
    ).run(context, counts).counts

    assert counts.frames_extracted == 1
    assert counts.captions_generated == 1
    assert counts.transcript_segments == 1
    assert counts.detections_created == 1
    assert counts.crops_created == 1
    assert counts.text_vectors_indexed >= 3
    assert run_paths.report_json.is_file()
    assert run_paths.detections_csv.is_file()
    assert run_paths.summary_markdown.is_file()


def test_enrichment_detection_runs_when_transcription_fails(tmp_path: Path) -> None:
    settings = Settings(
        paths={
            "data_dir": tmp_path / "data",
            "input_dir": tmp_path / "data" / "input",
            "runs_dir": tmp_path / "data" / "runs",
            "chroma_dir": tmp_path / "data" / "chroma",
            "database_path": tmp_path / "data" / "vidra.sqlite3",
        }
    )
    settings.ensure_directories()
    database = Database(settings.paths.database_path)
    database.initialize()
    source = tmp_path / "demo.mp4"
    source.write_bytes(b"video")
    database.execute(
        "INSERT INTO videos (video_id, filename, file_path, duration_sec) VALUES (?, ?, ?, ?)",
        ("video-1", "demo.mp4", str(source), 0.0),
    )
    layout = RunLayout(settings.paths.runs_dir)
    context_repo = VideoContextRepository(database)
    processing = VideoProcessingService(database)
    run_paths = layout.for_run("video-1", create=True)
    frame = FrameRecord(
        video_id="video-1",
        frame_id="video-1-frame-000000",
        timestamp_sec=0.0,
        frame_number=0,
        image_path=run_paths.frames_dir / "frame.jpg",
    )
    frame.image_path.write_bytes(b"fake-frame")
    processing.store_tool_results(
        video_id="video-1",
        tool_name="extract_frames",
        records=[
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
        ],
        idempotency_key="video-1:extract_frames",
        run_id="video-1",
    )

    result = EnrichmentStage(
        layout=layout,
        context_repository=context_repo,
        processing_service=processing,
        transcriber=FailingTranscriber(),
        detector=FakeDetector(),
    ).run(
        VideoProcessingContext(
            job_id="job-1",
            video_id="video-1",
            run_id="video-1",
            source_path=source,
        ),
        ProcessingCounts(frames_extracted=1, captions_generated=1),
    )

    assert result.counts.detections_created == 1
    assert result.counts.crops_created == 1
    assert result.warnings == ("audio transcription failed: ffmpeg could not read audio",)
