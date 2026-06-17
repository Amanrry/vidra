from __future__ import annotations

import shutil
from collections.abc import Sequence
from pathlib import Path

from video_qa.models.tools import BoundingBox, ContextType, FrameRecord
from video_qa.tools.object_detector import ObjectDetector, RawDetection


class FakeDetectionBackend:
    model_name = "fake-yolo"

    def __init__(self) -> None:
        self.calls: list[Path] = []

    def detect(self, frame_path: Path, *, confidence_threshold: float) -> list[RawDetection]:
        self.calls.append(frame_path)
        return [
            RawDetection(
                label=" Person ",
                confidence=0.91,
                bbox=BoundingBox(x1=1, y1=2, x2=10, y2=12),
            )
        ]


class FakeArtifactBackend:
    def annotate_frame(
        self,
        frame_path: Path,
        detections: Sequence[RawDetection],
        output_path: Path,
    ) -> None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(b"annotated:" + frame_path.read_bytes())

    def crop_object(
        self,
        frame_path: Path,
        bbox: BoundingBox,
        output_path: Path,
    ) -> None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(
            f"crop:{bbox.x1},{bbox.y1},{bbox.x2},{bbox.y2}".encode("ascii")
        )


def make_frame(tmp_path: Path) -> FrameRecord:
    image_path = tmp_path / "frame.jpg"
    image_path.write_bytes(b"image")
    return FrameRecord(
        video_id="video-1",
        frame_id="frame-000001",
        timestamp_sec=2.5,
        frame_number=75,
        image_path=image_path,
    )


def test_mocked_yolo_detections_create_artifacts_and_context_records(
    tmp_path: Path,
) -> None:
    frame = make_frame(tmp_path)
    backend = FakeDetectionBackend()
    detector = ObjectDetector(
        backend=backend,
        artifact_backend=FakeArtifactBackend(),
        confidence_threshold=0.5,
    )

    batch = detector.detect_frames(
        [frame],
        annotated_dir=tmp_path / "annotated",
        crops_dir=tmp_path / "crops",
    )
    contexts = detector.to_context_records(batch)

    assert backend.calls == [frame.image_path]
    assert len(batch.detections) == 1
    assert len(batch.crops) == 1

    detection = batch.detections[0]
    crop = batch.crops[0]
    assert detection.object_id == "frame-000001-object-0000"
    assert detection.label == "person"
    assert detection.frame_index == 75
    assert detection.annotated_frame_path is not None
    assert detection.annotated_frame_path.exists()
    assert detection.crop_path is not None
    assert detection.crop_path.exists()
    assert crop.crop_path == detection.crop_path

    assert [context.context_type for context in contexts] == [
        ContextType.object,
        ContextType.crop,
    ]
    assert contexts[0].context_id == detection.object_id
    assert contexts[0].timestamp_sec == 2.5
    assert contexts[0].data["label"] == "person"
    assert contexts[0].data["bbox"] == {"x1": 1.0, "y1": 2.0, "x2": 10.0, "y2": 12.0}
    assert contexts[1].context_id == crop.crop_id
    assert Path(contexts[1].data["crop_path"]).exists()


def test_empty_detections_still_create_annotated_frame(tmp_path: Path) -> None:
    class EmptyBackend(FakeDetectionBackend):
        def detect(self, frame_path: Path, *, confidence_threshold: float) -> list[RawDetection]:
            return []

    class CopyArtifactBackend(FakeArtifactBackend):
        def annotate_frame(
            self,
            frame_path: Path,
            detections: Sequence[RawDetection],
            output_path: Path,
        ) -> None:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(frame_path, output_path)

    frame = make_frame(tmp_path)
    detector = ObjectDetector(
        backend=EmptyBackend(),
        artifact_backend=CopyArtifactBackend(),
    )

    batch = detector.detect_frames(
        [frame],
        annotated_dir=tmp_path / "annotated",
        crops_dir=tmp_path / "crops",
    )

    assert batch.detections == []
    assert batch.crops == []
    assert (tmp_path / "annotated" / "frame-000001_annotated.jpg").read_bytes() == b"image"
