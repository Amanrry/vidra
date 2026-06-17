"""Frame captioning tool with injectable model backends."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Protocol, runtime_checkable

from video_qa.models.tools import (
    CaptionRecord,
    ContextType,
    FrameRecord,
    VideoContextRecord,
)


class CaptioningError(RuntimeError):
    """Raised when the captioning backend cannot produce usable output."""


@runtime_checkable
class CaptionBackend(Protocol):
    model_name: str

    def caption_image(self, image_path: Path) -> tuple[str, float]:
        """Return caption text and confidence for one image."""


class BlipCaptionBackend:
    """Lazy BLIP adapter kept out of tests and request-path contracts."""

    def __init__(
        self,
        model_name: str = "Salesforce/blip-image-captioning-large",
        *,
        max_length: int = 50,
        num_beams: int = 5,
        local_models_dir: str | Path = ".models",
    ) -> None:
        self.model_name = model_name
        self.model_path = resolve_transformers_model_path(model_name, local_models_dir)
        self.max_length = max_length
        self.num_beams = num_beams
        self._loaded = False

    def caption_image(self, image_path: Path) -> tuple[str, float]:
        if not image_path.exists() or not image_path.is_file():
            raise FileNotFoundError(f"Frame image not found: {image_path}")
        self._load()
        image = self._image_open(image_path).convert("RGB")
        inputs = self._processor(image, return_tensors="pt").to(self._device)
        with self._torch.no_grad():
            outputs = self._model.generate(
                **inputs,
                max_length=self.max_length,
                num_beams=self.num_beams,
                early_stopping=True,
            )
        text = self._processor.decode(outputs[0], skip_special_tokens=True).strip()
        if not text:
            raise CaptioningError(f"Caption backend returned empty text for {image_path}")
        return text, 0.85

    def _load(self) -> None:
        if self._loaded:
            return
        try:
            import torch  # type: ignore[import-not-found]
            from PIL import Image  # type: ignore[import-not-found]
            from transformers import (  # type: ignore[import-not-found]
                BlipForConditionalGeneration,
                BlipProcessor,
            )
        except Exception as exc:  # pragma: no cover - optional dependencies
            raise CaptioningError(
                "BLIP captioning requires pillow, torch, and transformers. "
                "Install Vidra with the 'ai' optional dependencies."
            ) from exc

        self._torch = torch
        self._image_open = Image.open
        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        local_only = Path(self.model_path).exists()
        self._processor = BlipProcessor.from_pretrained(
            self.model_path,
            local_files_only=local_only,
        )
        self._model = BlipForConditionalGeneration.from_pretrained(
            self.model_path,
            local_files_only=local_only,
        ).to(
            self._device
        )
        self._loaded = True


class ImageCaptioner:
    """Generate timestamped frame captions and context records."""

    tool_name = "caption_frames"

    def __init__(
        self,
        backend: CaptionBackend | None = None,
        *,
        unavailable_caption: str = "[Caption unavailable]",
    ) -> None:
        self.backend = backend or BlipCaptionBackend()
        self.unavailable_caption = unavailable_caption

    def caption_frame(self, frame: FrameRecord) -> CaptionRecord:
        """Caption one sampled frame while preserving frame timestamp metadata."""

        try:
            text, confidence = self.backend.caption_image(frame.image_path)
        except Exception as exc:
            text = self.unavailable_caption
            confidence = 0.0
            _ = exc

        return CaptionRecord(
            video_id=frame.video_id,
            frame_id=frame.frame_id,
            timestamp_sec=frame.timestamp_sec,
            text=text,
            confidence=confidence,
            model_name=self.backend.model_name,
        )

    def caption_frames(self, frames: Sequence[FrameRecord]) -> list[CaptionRecord]:
        """Caption frames in order with per-frame fallback behavior."""

        return [self.caption_frame(frame) for frame in frames]

    def to_context_records(
        self,
        captions: Sequence[CaptionRecord],
    ) -> list[VideoContextRecord]:
        return [
            VideoContextRecord(
                context_id=self._context_id(caption),
                video_id=caption.video_id,
                context_type=ContextType.caption,
                timestamp_sec=caption.timestamp_sec,
                data={
                    "frame_id": caption.frame_id,
                    "text": caption.text,
                    "confidence": caption.confidence,
                },
                tool_name=self.tool_name,
                model_name=caption.model_name,
            )
            for caption in captions
        ]

    def caption_frames_as_context(
        self,
        frames: Sequence[FrameRecord],
    ) -> list[VideoContextRecord]:
        return self.to_context_records(self.caption_frames(frames))

    def _context_id(self, caption: CaptionRecord) -> str:
        return f"{caption.video_id}-caption-{caption.frame_id}"


def resolve_transformers_model_path(
    model_name: str,
    local_models_dir: str | Path = ".models",
) -> str:
    """Prefer a pre-downloaded local model when config still uses a HF repo id."""

    configured = Path(model_name)
    if configured.exists():
        return str(configured)

    model_dir_name = model_name.replace("/", "__")
    local_candidate = Path(local_models_dir) / model_dir_name
    if local_candidate.exists():
        return str(local_candidate)

    return model_name
