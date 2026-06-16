"""Deterministic video frame sampling.

The extractor follows Bri's adaptive frame limit and timestamp-first contracts,
while keeping Thales' direct seek-by-frame-number approach. OpenCV is isolated
behind a tiny backend boundary so tests and future ffmpeg/PyAV adapters do not
change the application contract.
"""

from __future__ import annotations

import base64
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from pydantic import BaseModel, Field

from video_qa.models.tools import FrameRecord


class FrameExtractionError(RuntimeError):
    """Raised when a video cannot be sampled into usable image frames."""


class VideoMetadata(BaseModel):
    """Stable video properties needed by downstream pipeline stages."""

    duration_sec: float = Field(..., ge=0.0)
    fps: float = Field(..., ge=0.0)
    frame_count: int = Field(..., ge=0)
    width: int = Field(..., ge=0)
    height: int = Field(..., ge=0)
    codec: str | None = None
    file_size_bytes: int = Field(..., ge=0)


@dataclass(frozen=True)
class SamplePoint:
    """One planned frame read."""

    sequence_index: int
    frame_number: int
    timestamp_sec: float


@runtime_checkable
class VideoCaptureLike(Protocol):
    def isOpened(self) -> bool: ...

    def get(self, prop_id: int) -> float: ...

    def set(self, prop_id: int, value: float) -> bool: ...

    def read(self) -> tuple[bool, Any]: ...

    def release(self) -> None: ...


@runtime_checkable
class FrameBackend(Protocol):
    prop_fps: int
    prop_frame_count: int
    prop_frame_width: int
    prop_frame_height: int
    prop_fourcc: int
    prop_pos_frames: int

    def open(self, video_path: Path) -> VideoCaptureLike: ...

    def write_image(self, image_path: Path, frame: Any) -> bool: ...


class OpenCVFrameBackend:
    """OpenCV adapter loaded lazily to keep the core package lightweight."""

    def __init__(self) -> None:
        try:
            import cv2  # type: ignore[import-not-found]
        except Exception as exc:  # pragma: no cover - depends on optional extra
            raise FrameExtractionError(
                "opencv-python is required for frame extraction. "
                "Install Vidra with the 'ai' optional dependencies."
            ) from exc

        self._cv2 = cv2
        self.prop_fps = cv2.CAP_PROP_FPS
        self.prop_frame_count = cv2.CAP_PROP_FRAME_COUNT
        self.prop_frame_width = cv2.CAP_PROP_FRAME_WIDTH
        self.prop_frame_height = cv2.CAP_PROP_FRAME_HEIGHT
        self.prop_fourcc = cv2.CAP_PROP_FOURCC
        self.prop_pos_frames = cv2.CAP_PROP_POS_FRAMES

    def open(self, video_path: Path) -> VideoCaptureLike:
        return self._cv2.VideoCapture(str(video_path))

    def write_image(self, image_path: Path, frame: Any) -> bool:
        return bool(self._cv2.imwrite(str(image_path), frame))


class FrameExtractor:
    """Sample frames from arbitrary uploaded videos into deterministic paths."""

    tool_name = "extract_frames"

    def __init__(self, backend: FrameBackend | None = None) -> None:
        self._backend = backend

    @property
    def backend(self) -> FrameBackend:
        if self._backend is None:
            self._backend = OpenCVFrameBackend()
        return self._backend

    def get_video_metadata(self, video_path: str | Path) -> VideoMetadata:
        """Read video metadata without extracting frames."""

        path = Path(video_path)
        self._validate_video_path(path)
        capture = self.backend.open(path)
        if not capture.isOpened():
            capture.release()
            raise FrameExtractionError(f"Unable to open video: {path}")

        try:
            fps = float(capture.get(self.backend.prop_fps) or 0.0)
            frame_count = max(0, int(capture.get(self.backend.prop_frame_count) or 0))
            width = max(0, int(capture.get(self.backend.prop_frame_width) or 0))
            height = max(0, int(capture.get(self.backend.prop_frame_height) or 0))
            codec = self._decode_fourcc(int(capture.get(self.backend.prop_fourcc) or 0))
            duration = frame_count / fps if fps > 0 and frame_count > 0 else 0.0
            return VideoMetadata(
                duration_sec=duration,
                fps=fps,
                frame_count=frame_count,
                width=width,
                height=height,
                codec=codec,
                file_size_bytes=path.stat().st_size,
            )
        finally:
            capture.release()

    def build_sample_plan(
        self,
        metadata: VideoMetadata,
        *,
        interval_seconds: float,
        max_frames: int,
    ) -> list[SamplePoint]:
        """Create a deterministic seek plan from metadata and sampling limits."""

        if interval_seconds <= 0:
            raise ValueError("interval_seconds must be positive")
        if max_frames < 1:
            raise ValueError("max_frames must be positive")
        if metadata.frame_count <= 0:
            return []

        fps = metadata.fps if metadata.fps > 0 else 1.0
        duration = metadata.duration_sec
        effective_interval = self._adaptive_interval(
            duration_sec=duration,
            requested_interval_sec=interval_seconds,
            max_frames=max_frames,
        )

        points: list[SamplePoint] = []
        seen_frame_numbers: set[int] = set()
        index = 0
        sample_cursor = 0

        while len(points) < max_frames:
            timestamp = sample_cursor * effective_interval
            sample_cursor += 1
            if duration > 0 and timestamp > duration:
                break

            frame_number = int(round(timestamp * fps))
            frame_number = min(frame_number, metadata.frame_count - 1)
            if frame_number not in seen_frame_numbers:
                points.append(
                    SamplePoint(
                        sequence_index=index,
                        frame_number=frame_number,
                        timestamp_sec=frame_number / fps,
                    )
                )
                seen_frame_numbers.add(frame_number)
                index += 1

            if frame_number >= metadata.frame_count - 1:
                break

        return points

    def extract_frames(
        self,
        video_path: str | Path,
        *,
        video_id: str,
        output_dir: str | Path,
        interval_seconds: float,
        max_frames: int,
    ) -> list[FrameRecord]:
        """Extract sampled frames and return pipeline-ready frame contracts."""

        path = Path(video_path)
        self._validate_video_path(path)
        clean_video_id = video_id.strip()
        if not clean_video_id:
            raise ValueError("video_id is required")

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        metadata = self.get_video_metadata(path)
        sample_plan = self.build_sample_plan(
            metadata,
            interval_seconds=interval_seconds,
            max_frames=max_frames,
        )
        if not sample_plan:
            return []

        capture = self.backend.open(path)
        if not capture.isOpened():
            capture.release()
            raise FrameExtractionError(f"Unable to open video for frame extraction: {path}")

        records: list[FrameRecord] = []
        try:
            for point in sample_plan:
                frame_path = output_path / self._frame_filename(point)
                ok, frame = self._read_frame(capture, point.frame_number)
                if not ok:
                    continue
                if not self.backend.write_image(frame_path, frame):
                    raise FrameExtractionError(f"Failed to write sampled frame: {frame_path}")

                frame_id = self._frame_id(clean_video_id, point.sequence_index)
                records.append(
                    FrameRecord(
                        video_id=clean_video_id,
                        frame_id=frame_id,
                        timestamp_sec=point.timestamp_sec,
                        frame_number=point.frame_number,
                        image_path=frame_path,
                    )
                )
        finally:
            capture.release()

        return records

    def extract_frame_at_timestamp(
        self,
        video_path: str | Path,
        *,
        video_id: str,
        output_dir: str | Path,
        timestamp_sec: float,
    ) -> FrameRecord:
        """Extract one frame for seek/timeline workflows."""

        if timestamp_sec < 0:
            raise ValueError("timestamp_sec must be non-negative")
        metadata = self.get_video_metadata(video_path)
        fps = metadata.fps if metadata.fps > 0 else 1.0
        if metadata.duration_sec > 0 and timestamp_sec > metadata.duration_sec:
            raise ValueError("timestamp_sec exceeds video duration")
        frame_number = min(int(round(timestamp_sec * fps)), max(0, metadata.frame_count - 1))
        point = SamplePoint(
            sequence_index=0,
            frame_number=frame_number,
            timestamp_sec=frame_number / fps,
        )
        records = self._extract_specific_points(
            Path(video_path),
            video_id=video_id,
            output_dir=Path(output_dir),
            sample_plan=[point],
            id_prefix="seek",
        )
        if not records:
            raise FrameExtractionError(f"Failed to extract frame at {timestamp_sec:.3f}s")
        return records[0]

    def encode_frame_to_base64(self, frame_path: str | Path) -> str:
        """Encode a stored frame for VLM calls."""

        path = Path(frame_path)
        if not path.exists() or not path.is_file():
            raise FileNotFoundError(f"Frame file not found: {path}")
        return base64.b64encode(path.read_bytes()).decode("utf-8")

    def _extract_specific_points(
        self,
        video_path: Path,
        *,
        video_id: str,
        output_dir: Path,
        sample_plan: list[SamplePoint],
        id_prefix: str,
    ) -> list[FrameRecord]:
        output_dir.mkdir(parents=True, exist_ok=True)
        capture = self.backend.open(video_path)
        if not capture.isOpened():
            capture.release()
            raise FrameExtractionError(f"Unable to open video: {video_path}")

        clean_video_id = video_id.strip()
        records: list[FrameRecord] = []
        try:
            for point in sample_plan:
                frame_path = output_dir / self._frame_filename(point, prefix=id_prefix)
                ok, frame = self._read_frame(capture, point.frame_number)
                if not ok:
                    continue
                if not self.backend.write_image(frame_path, frame):
                    raise FrameExtractionError(f"Failed to write sampled frame: {frame_path}")
                records.append(
                    FrameRecord(
                        video_id=clean_video_id,
                        frame_id=f"{clean_video_id}-{id_prefix}-{point.frame_number:08d}",
                        timestamp_sec=point.timestamp_sec,
                        frame_number=point.frame_number,
                        image_path=frame_path,
                    )
                )
        finally:
            capture.release()
        return records

    def _read_frame(self, capture: VideoCaptureLike, frame_number: int) -> tuple[bool, Any]:
        capture.set(self.backend.prop_pos_frames, float(frame_number))
        return capture.read()

    def _adaptive_interval(
        self,
        *,
        duration_sec: float,
        requested_interval_sec: float,
        max_frames: int,
    ) -> float:
        if duration_sec <= 0:
            return requested_interval_sec
        estimated_frames = math.floor(duration_sec / requested_interval_sec) + 1
        if estimated_frames <= max_frames:
            return requested_interval_sec
        return max(requested_interval_sec, duration_sec / max(max_frames - 1, 1))

    def _frame_filename(self, point: SamplePoint, prefix: str = "frame") -> str:
        millis = int(round(point.timestamp_sec * 1000))
        return f"{prefix}_{point.sequence_index:06d}_t{millis:010d}_f{point.frame_number:08d}.jpg"

    def _frame_id(self, video_id: str, sequence_index: int) -> str:
        return f"{video_id}-frame-{sequence_index:06d}"

    def _validate_video_path(self, video_path: Path) -> None:
        if not video_path.exists():
            raise FileNotFoundError(f"Video file not found: {video_path}")
        if not video_path.is_file():
            raise FrameExtractionError(f"Video path is not a file: {video_path}")

    def _decode_fourcc(self, fourcc: int) -> str | None:
        if fourcc <= 0:
            return None
        chars = [chr((fourcc >> (8 * index)) & 0xFF) for index in range(4)]
        codec = "".join(chars).strip("\x00").strip()
        return codec or None
