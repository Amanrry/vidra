from __future__ import annotations

from pathlib import Path
from typing import Any

from video_qa.tools.frame_extractor import FrameExtractor


class SmokeCapture:
    def __init__(self) -> None:
        self.position = 0

    def isOpened(self) -> bool:
        return True

    def get(self, prop_id: int) -> float:
        return {
            SmokeBackend.prop_fps: 5.0,
            SmokeBackend.prop_frame_count: 15.0,
            SmokeBackend.prop_frame_width: 320.0,
            SmokeBackend.prop_frame_height: 180.0,
        }.get(prop_id, 0.0)

    def set(self, prop_id: int, value: float) -> bool:
        if prop_id == SmokeBackend.prop_pos_frames:
            self.position = int(value)
        return True

    def read(self) -> tuple[bool, Any]:
        return True, {"frame_number": self.position}

    def release(self) -> None:
        return None


class SmokeBackend:
    prop_fps = 1
    prop_frame_count = 2
    prop_frame_width = 3
    prop_frame_height = 4
    prop_fourcc = 5
    prop_pos_frames = 6

    def open(self, video_path: Path) -> SmokeCapture:
        return SmokeCapture()

    def write_image(self, image_path: Path, frame: Any) -> bool:
        image_path.write_bytes(f"frame-{frame['frame_number']}".encode("ascii"))
        return True


def test_video_io_frame_sampling_contract(tmp_path: Path) -> None:
    extractor = FrameExtractor(backend=SmokeBackend())
    video_path = tmp_path / "demo.mp4"
    video_path.write_bytes(b"video")

    frames = extractor.extract_frames(
        video_path,
        video_id="video-io",
        output_dir=tmp_path / "frames",
        interval_seconds=1.0,
        max_frames=4,
    )

    assert [(frame.frame_number, frame.timestamp_sec) for frame in frames] == [
        (0, 0.0),
        (5, 1.0),
        (10, 2.0),
        (14, 2.8),
    ]
