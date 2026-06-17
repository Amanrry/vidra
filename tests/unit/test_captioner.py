from __future__ import annotations

from pathlib import Path

from video_qa.models.tools import ContextType, FrameRecord
from video_qa.tools.image_captioner import ImageCaptioner, resolve_transformers_model_path


class FakeCaptionBackend:
    model_name = "fake-captioner"

    def __init__(self, *, fail_on: set[str] | None = None) -> None:
        self.fail_on = fail_on or set()
        self.calls: list[Path] = []

    def caption_image(self, image_path: Path) -> tuple[str, float]:
        self.calls.append(image_path)
        if image_path.name in self.fail_on:
            raise RuntimeError("caption model unavailable")
        return f"caption for {image_path.stem}", 0.74


def make_frame(tmp_path: Path, index: int, timestamp: float) -> FrameRecord:
    image_path = tmp_path / f"frame-{index}.jpg"
    image_path.write_bytes(b"image")
    return FrameRecord(
        video_id="video-1",
        frame_id=f"frame-{index}",
        timestamp_sec=timestamp,
        frame_number=index * 10,
        image_path=image_path,
    )


def test_captioner_produces_timestamped_caption_and_context_records(
    tmp_path: Path,
) -> None:
    frames = [make_frame(tmp_path, 1, 1.5), make_frame(tmp_path, 2, 3.0)]
    backend = FakeCaptionBackend()
    captioner = ImageCaptioner(backend=backend)

    captions = captioner.caption_frames(frames)
    contexts = captioner.to_context_records(captions)

    assert [caption.timestamp_sec for caption in captions] == [1.5, 3.0]
    assert [caption.frame_id for caption in captions] == ["frame-1", "frame-2"]
    assert captions[0].text == "caption for frame-1"
    assert captions[0].confidence == 0.74
    assert captions[0].model_name == "fake-captioner"

    assert [context.context_type for context in contexts] == [
        ContextType.caption,
        ContextType.caption,
    ]
    assert contexts[0].context_id == "video-1-caption-frame-1"
    assert contexts[0].timestamp_sec == 1.5
    assert contexts[0].data == {
        "frame_id": "frame-1",
        "text": "caption for frame-1",
        "confidence": 0.74,
    }
    assert backend.calls == [frames[0].image_path, frames[1].image_path]


def test_captioner_tolerates_per_frame_backend_failure(tmp_path: Path) -> None:
    frames = [make_frame(tmp_path, 1, 1.5)]
    captioner = ImageCaptioner(
        backend=FakeCaptionBackend(fail_on={"frame-1.jpg"}),
        unavailable_caption="[Caption unavailable]",
    )

    captions = captioner.caption_frames(frames)
    contexts = captioner.caption_frames_as_context(frames)

    assert captions[0].text == "[Caption unavailable]"
    assert captions[0].confidence == 0.0
    assert contexts[0].data["text"] == "[Caption unavailable]"
    assert contexts[0].timestamp_sec == 1.5


def test_repo_id_prefers_pre_downloaded_local_model_directory(tmp_path: Path) -> None:
    local_dir = tmp_path / ".models" / "Salesforce__blip-image-captioning-base"
    local_dir.mkdir(parents=True)

    resolved = resolve_transformers_model_path(
        "Salesforce/blip-image-captioning-base",
        tmp_path / ".models",
    )

    assert resolved == str(local_dir)
