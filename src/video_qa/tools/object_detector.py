"""YOLO-style object detection, annotation, and crop extraction."""

from __future__ import annotations

import shutil
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, runtime_checkable

from video_qa.models.tools import (
    BoundingBox,
    ContextType,
    CropRecord,
    DetectionRecord,
    FrameRecord,
    VideoContextRecord,
)


class ObjectDetectionError(RuntimeError):
    """Raised when object detection or artifact generation fails."""


@dataclass(frozen=True)
class RawDetection:
    """Backend-neutral detection payload in xyxy pixel coordinates."""

    label: str
    confidence: float
    bbox: BoundingBox


@runtime_checkable
class DetectionBackend(Protocol):
    model_name: str

    def detect(self, frame_path: Path, *, confidence_threshold: float) -> list[RawDetection]:
        """Detect objects in one frame."""


@runtime_checkable
class DetectionArtifactBackend(Protocol):
    def annotate_frame(
        self,
        frame_path: Path,
        detections: Sequence[RawDetection],
        output_path: Path,
    ) -> None:
        """Write an annotated frame image."""

    def crop_object(
        self,
        frame_path: Path,
        bbox: BoundingBox,
        output_path: Path,
    ) -> None:
        """Write one object crop image."""


class YoloDetectionBackend:
    """Lazy Ultralytics YOLO adapter."""

    def __init__(self, model_name: str = "yolov8n.pt") -> None:
        self.model_name = model_name
        self._model = None

    def detect(self, frame_path: Path, *, confidence_threshold: float) -> list[RawDetection]:
        if not frame_path.exists() or not frame_path.is_file():
            raise FileNotFoundError(f"Frame image not found: {frame_path}")
        model = self._load_model()
        results = model(str(frame_path), conf=confidence_threshold, verbose=False)
        if not results:
            return []

        result = results[0]
        names = getattr(result, "names", {}) or {}
        boxes = getattr(result, "boxes", None)
        if boxes is None:
            return []

        detections: list[RawDetection] = []
        for box in boxes:
            confidence = float(box.conf[0])
            class_id = int(box.cls[0])
            label = str(names.get(class_id, class_id)).strip().lower()
            xyxy = [float(value) for value in box.xyxy[0].tolist()]
            detections.append(
                RawDetection(
                    label=label,
                    confidence=confidence,
                    bbox=BoundingBox(x1=xyxy[0], y1=xyxy[1], x2=xyxy[2], y2=xyxy[3]),
                )
            )
        return detections

    def _load_model(self):
        if self._model is not None:
            return self._model
        try:
            from ultralytics import YOLO  # type: ignore[import-not-found]
        except Exception as exc:  # pragma: no cover - optional dependency
            raise ObjectDetectionError(
                "YOLO detection requires ultralytics. "
                "Install Vidra with the 'ai' optional dependencies."
            ) from exc
        self._model = YOLO(self.model_name)
        return self._model


class PillowDetectionArtifactBackend:
    """Pillow-based annotation and crop writer."""

    def annotate_frame(
        self,
        frame_path: Path,
        detections: Sequence[RawDetection],
        output_path: Path,
    ) -> None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        if not detections:
            shutil.copy2(frame_path, output_path)
            return
        try:
            from PIL import Image, ImageDraw  # type: ignore[import-not-found]
        except Exception as exc:  # pragma: no cover - optional dependency
            raise ObjectDetectionError(
                "Annotation requires pillow. Install Vidra with the 'ai' optional dependencies."
            ) from exc

        with Image.open(frame_path).convert("RGB") as image:
            draw = ImageDraw.Draw(image)
            for detection in detections:
                bbox = detection.bbox
                draw.rectangle((bbox.x1, bbox.y1, bbox.x2, bbox.y2), outline=(0, 134, 195), width=2)
                draw.text(
                    (bbox.x1, max(0.0, bbox.y1 - 12.0)),
                    f"{detection.label} {detection.confidence:.2f}",
                    fill=(0, 134, 195),
                )
            image.save(output_path)

    def crop_object(
        self,
        frame_path: Path,
        bbox: BoundingBox,
        output_path: Path,
    ) -> None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            from PIL import Image  # type: ignore[import-not-found]
        except Exception as exc:  # pragma: no cover - optional dependency
            raise ObjectDetectionError(
                "Crop extraction requires pillow. Install Vidra with the 'ai' optional dependencies."
            ) from exc

        with Image.open(frame_path).convert("RGB") as image:
            width, height = image.size
            crop_box = (
                max(0, int(round(bbox.x1))),
                max(0, int(round(bbox.y1))),
                min(width, int(round(bbox.x2))),
                min(height, int(round(bbox.y2))),
            )
            if crop_box[2] <= crop_box[0] or crop_box[3] <= crop_box[1]:
                raise ObjectDetectionError(f"Invalid crop bounds for {frame_path}: {crop_box}")
            image.crop(crop_box).save(output_path)


@dataclass(frozen=True)
class ObjectDetectionBatch:
    detections: list[DetectionRecord]
    crops: list[CropRecord]


class ObjectDetector:
    """Detect objects in sampled frames and materialize annotation/crop artifacts."""

    tool_name = "detect_objects"

    def __init__(
        self,
        backend: DetectionBackend | None = None,
        artifact_backend: DetectionArtifactBackend | None = None,
        *,
        confidence_threshold: float = 0.25,
    ) -> None:
        self.backend = backend or YoloDetectionBackend()
        self.artifact_backend = artifact_backend or PillowDetectionArtifactBackend()
        self.confidence_threshold = confidence_threshold

    def detect_frames(
        self,
        frames: Sequence[FrameRecord],
        *,
        annotated_dir: str | Path,
        crops_dir: str | Path,
    ) -> ObjectDetectionBatch:
        annotated_path = Path(annotated_dir)
        crops_path = Path(crops_dir)
        annotated_path.mkdir(parents=True, exist_ok=True)
        crops_path.mkdir(parents=True, exist_ok=True)

        detections: list[DetectionRecord] = []
        crops: list[CropRecord] = []
        for frame in frames:
            raw_detections = self.backend.detect(
                frame.image_path,
                confidence_threshold=self.confidence_threshold,
            )
            annotated_frame_path = annotated_path / f"{frame.frame_id}_annotated.jpg"
            self.artifact_backend.annotate_frame(
                frame.image_path,
                raw_detections,
                annotated_frame_path,
            )

            for index, raw in enumerate(raw_detections):
                object_id = f"{frame.frame_id}-object-{index:04d}"
                crop_id = f"{object_id}-crop"
                crop_path = crops_path / f"{crop_id}.jpg"
                self.artifact_backend.crop_object(frame.image_path, raw.bbox, crop_path)
                detection = DetectionRecord(
                    video_id=frame.video_id,
                    object_id=object_id,
                    frame_id=frame.frame_id,
                    timestamp_sec=frame.timestamp_sec,
                    frame_index=frame.frame_number,
                    label=raw.label,
                    confidence=raw.confidence,
                    bbox=raw.bbox,
                    frame_path=frame.image_path,
                    annotated_frame_path=annotated_frame_path,
                    crop_path=crop_path,
                )
                crop = CropRecord(
                    video_id=frame.video_id,
                    crop_id=crop_id,
                    object_id=object_id,
                    frame_id=frame.frame_id,
                    label=detection.label,
                    timestamp_sec=frame.timestamp_sec,
                    crop_path=crop_path,
                )
                detections.append(detection)
                crops.append(crop)

        return ObjectDetectionBatch(detections=detections, crops=crops)

    def detections_to_context_records(
        self,
        detections: Sequence[DetectionRecord],
    ) -> list[VideoContextRecord]:
        return [
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
                tool_name=self.tool_name,
                model_name=self.backend.model_name,
            )
            for detection in detections
        ]

    def crops_to_context_records(self, crops: Sequence[CropRecord]) -> list[VideoContextRecord]:
        return [
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
                model_name=self.backend.model_name,
            )
            for crop in crops
        ]

    def to_context_records(self, batch: ObjectDetectionBatch) -> list[VideoContextRecord]:
        return self.detections_to_context_records(batch.detections) + self.crops_to_context_records(
            batch.crops
        )
