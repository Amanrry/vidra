from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from video_qa.tools.frame_extractor import (
    FrameExtractionError,
    FrameExtractor,
    VideoMetadata,
)


class FakeCapture:
    def __init__(
        self,
        *,
        fps: float = 10.0,
        frame_count: int = 50,
        width: int = 640,
        height: int = 360,
        opened: bool = True,
        unreadable_frames: set[int] | None = None,
    ) -> None:
        self.fps = fps
        self.frame_count = frame_count
        self.width = width
        self.height = height
        self.opened = opened
        self.unreadable_frames = unreadable_frames or set()
        self.position = 0
        self.released = False
        self.seeked_frames: list[int] = []

    def isOpened(self) -> bool:
        return self.opened

    def get(self, prop_id: int) -> float:
        values = {
            FakeBackend.PROP_FPS: self.fps,
            FakeBackend.PROP_FRAME_COUNT: float(self.frame_count),
            FakeBackend.PROP_FRAME_WIDTH: float(self.width),
            FakeBackend.PROP_FRAME_HEIGHT: float(self.height),
            FakeBackend.PROP_FOURCC: float(FakeBackend.fourcc("mp4v")),
        }
        return values.get(prop_id, 0.0)

    def set(self, prop_id: int, value: float) -> bool:
        if prop_id == FakeBackend.PROP_POS_FRAMES:
            self.position = int(value)
            self.seeked_frames.append(self.position)
        return True

    def read(self) -> tuple[bool, Any]:
        if self.position in self.unreadable_frames:
            return False, None
        if self.position >= self.frame_count:
            return False, None
        return True, {"frame_number": self.position}

    def release(self) -> None:
        self.released = True


class FakeBackend:
    PROP_FPS = 1
    PROP_FRAME_COUNT = 2
    PROP_FRAME_WIDTH = 3
    PROP_FRAME_HEIGHT = 4
    PROP_FOURCC = 5
    PROP_POS_FRAMES = 6

    prop_fps = PROP_FPS
    prop_frame_count = PROP_FRAME_COUNT
    prop_frame_width = PROP_FRAME_WIDTH
    prop_frame_height = PROP_FRAME_HEIGHT
    prop_fourcc = PROP_FOURCC
    prop_pos_frames = PROP_POS_FRAMES

    def __init__(self, captures: list[FakeCapture] | None = None, write_ok: bool = True) -> None:
        self.captures = captures or [FakeCapture()]
        self.write_ok = write_ok
        self.writes: list[tuple[Path, Any]] = []

    def open(self, video_path: Path) -> FakeCapture:
        if len(self.captures) == 1:
            return self.captures[0]
        return self.captures.pop(0)

    def write_image(self, image_path: Path, frame: Any) -> bool:
        self.writes.append((image_path, frame))
        if self.write_ok:
            image_path.write_bytes(f"fake-{frame['frame_number']}".encode("ascii"))
        return self.write_ok

    @staticmethod
    def fourcc(value: str) -> int:
        return sum(ord(char) << (8 * index) for index, char in enumerate(value))


def write_video(path: Path) -> Path:
    path.write_bytes(b"video")
    return path


def test_get_video_metadata_reads_properties_and_releases_capture(tmp_path: Path) -> None:
    capture = FakeCapture(fps=25.0, frame_count=250, width=1920, height=1080)
    extractor = FrameExtractor(backend=FakeBackend([capture]))
    video_path = write_video(tmp_path / "demo.mp4")

    metadata = extractor.get_video_metadata(video_path)

    assert metadata.duration_sec == 10.0
    assert metadata.fps == 25.0
    assert metadata.frame_count == 250
    assert metadata.width == 1920
    assert metadata.height == 1080
    assert metadata.codec == "mp4v"
    assert metadata.file_size_bytes == 5
    assert capture.released


def test_sample_plan_uses_interval_and_includes_frame_metadata() -> None:
    extractor = FrameExtractor(backend=FakeBackend())
    metadata = VideoMetadata(
        duration_sec=5.0,
        fps=10.0,
        frame_count=50,
        width=100,
        height=100,
        file_size_bytes=5,
    )

    plan = extractor.build_sample_plan(metadata, interval_seconds=2.0, max_frames=10)

    assert [(point.frame_number, point.timestamp_sec) for point in plan] == [
        (0, 0.0),
        (20, 2.0),
        (40, 4.0),
    ]


def test_sample_plan_adapts_interval_to_max_frame_limit() -> None:
    extractor = FrameExtractor(backend=FakeBackend())
    metadata = VideoMetadata(
        duration_sec=10.0,
        fps=10.0,
        frame_count=100,
        width=100,
        height=100,
        file_size_bytes=5,
    )

    plan = extractor.build_sample_plan(metadata, interval_seconds=1.0, max_frames=3)

    assert len(plan) == 3
    assert [point.frame_number for point in plan] == [0, 50, 99]
    assert plan[-1].timestamp_sec == pytest.approx(9.9)


def test_extract_frames_writes_deterministic_files_and_returns_records(tmp_path: Path) -> None:
    metadata_capture = FakeCapture(fps=10.0, frame_count=50)
    extraction_capture = FakeCapture(fps=10.0, frame_count=50)
    backend = FakeBackend([metadata_capture, extraction_capture])
    extractor = FrameExtractor(backend=backend)
    video_path = write_video(tmp_path / "demo.mp4")
    output_dir = tmp_path / "frames"

    frames = extractor.extract_frames(
        video_path,
        video_id="video-1",
        output_dir=output_dir,
        interval_seconds=2.0,
        max_frames=10,
    )

    assert [frame.frame_id for frame in frames] == [
        "video-1-frame-000000",
        "video-1-frame-000001",
        "video-1-frame-000002",
    ]
    assert [frame.frame_number for frame in frames] == [0, 20, 40]
    assert [frame.timestamp_sec for frame in frames] == [0.0, 2.0, 4.0]
    assert [frame.image_path.name for frame in frames] == [
        "frame_000000_t0000000000_f00000000.jpg",
        "frame_000001_t0000002000_f00000020.jpg",
        "frame_000002_t0000004000_f00000040.jpg",
    ]
    assert all(frame.image_path.exists() for frame in frames)
    assert extraction_capture.seeked_frames == [0, 20, 40]
    assert metadata_capture.released
    assert extraction_capture.released


def test_extract_frames_skips_unreadable_sample_without_breaking_sequence(
    tmp_path: Path,
) -> None:
    backend = FakeBackend(
        [
            FakeCapture(fps=10.0, frame_count=50),
            FakeCapture(fps=10.0, frame_count=50, unreadable_frames={20}),
        ]
    )
    extractor = FrameExtractor(backend=backend)
    video_path = write_video(tmp_path / "demo.mp4")

    frames = extractor.extract_frames(
        video_path,
        video_id="video-1",
        output_dir=tmp_path / "frames",
        interval_seconds=2.0,
        max_frames=10,
    )

    assert [frame.frame_number for frame in frames] == [0, 40]
    assert [frame.frame_id for frame in frames] == [
        "video-1-frame-000000",
        "video-1-frame-000002",
    ]


def test_extract_frame_at_timestamp_returns_seek_record(tmp_path: Path) -> None:
    backend = FakeBackend([FakeCapture(fps=10.0, frame_count=50), FakeCapture()])
    extractor = FrameExtractor(backend=backend)
    video_path = write_video(tmp_path / "demo.mp4")

    frame = extractor.extract_frame_at_timestamp(
        video_path,
        video_id="video-1",
        output_dir=tmp_path / "frames",
        timestamp_sec=2.4,
    )

    assert frame.frame_id == "video-1-seek-00000024"
    assert frame.frame_number == 24
    assert frame.timestamp_sec == 2.4
    assert frame.image_path.name == "seek_000000_t0000002400_f00000024.jpg"


def test_extract_frames_raises_when_video_cannot_open(tmp_path: Path) -> None:
    extractor = FrameExtractor(backend=FakeBackend([FakeCapture(opened=False)]))
    video_path = write_video(tmp_path / "demo.mp4")

    with pytest.raises(FrameExtractionError, match="Unable to open video"):
        extractor.get_video_metadata(video_path)


def test_extract_frames_raises_when_image_write_fails(tmp_path: Path) -> None:
    backend = FakeBackend(
        [FakeCapture(fps=10.0, frame_count=10), FakeCapture(fps=10.0, frame_count=10)],
        write_ok=False,
    )
    extractor = FrameExtractor(backend=backend)
    video_path = write_video(tmp_path / "demo.mp4")

    with pytest.raises(FrameExtractionError, match="Failed to write"):
        extractor.extract_frames(
            video_path,
            video_id="video-1",
            output_dir=tmp_path / "frames",
            interval_seconds=1.0,
            max_frames=2,
        )


def test_encode_frame_to_base64(tmp_path: Path) -> None:
    frame_path = tmp_path / "frame.jpg"
    frame_path.write_bytes(b"abc")
    extractor = FrameExtractor(backend=FakeBackend())

    assert extractor.encode_frame_to_base64(frame_path) == "YWJj"
