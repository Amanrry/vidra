"""Media processing tools used by the Vidra pipeline."""

from video_qa.tools.frame_extractor import (
    FrameExtractionError,
    FrameExtractor,
    SamplePoint,
    VideoMetadata,
)

__all__ = [
    "FrameExtractionError",
    "FrameExtractor",
    "SamplePoint",
    "VideoMetadata",
]
