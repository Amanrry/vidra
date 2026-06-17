"""Audio transcription tool with resilient missing-audio behavior."""

from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from video_qa.models.tools import (
    ContextType,
    TranscriptSegmentRecord,
    VideoContextRecord,
)
from video_qa.runtime import ensure_ffmpeg_on_path


class TranscriptionError(RuntimeError):
    """Raised for unrecoverable transcription failures."""


class MissingAudioError(TranscriptionError):
    """Raised by backends when a video has no usable audio track."""


@dataclass(frozen=True)
class TranscriptResult:
    video_id: str
    segments: list[TranscriptSegmentRecord]
    language: str = "unknown"
    full_text: str = ""
    model_name: str | None = None
    error: str | None = None


@runtime_checkable
class TranscriptionBackend(Protocol):
    model_name: str

    def transcribe(
        self,
        media_path: Path,
        *,
        language: str | None = None,
    ) -> dict[str, Any]:
        """Return a Whisper-like transcription payload."""


class WhisperTranscriptionBackend:
    """Lazy OpenAI Whisper adapter."""

    def __init__(self, model_name: str = "base") -> None:
        self.model_name = model_name
        self._model = None

    def transcribe(
        self,
        media_path: Path,
        *,
        language: str | None = None,
    ) -> dict[str, Any]:
        if not media_path.exists() or not media_path.is_file():
            raise FileNotFoundError(f"Media file not found: {media_path}")
        self._ensure_ffmpeg()
        model = self._load_model()
        try:
            return model.transcribe(
                str(media_path),
                language=language,
                verbose=False,
                word_timestamps=False,
            )
        except Exception as exc:  # pragma: no cover - library-specific edge cases
            message = str(exc)
            if "no audio" in message.lower() or "empty sequence" in message.lower():
                raise MissingAudioError("No usable audio track found.") from exc
            raise TranscriptionError(f"Transcription failed: {message}") from exc

    def _load_model(self):
        if self._model is not None:
            return self._model
        try:
            import whisper  # type: ignore[import-not-found]
        except Exception as exc:  # pragma: no cover - optional dependency
            raise TranscriptionError(
                "Whisper transcription requires openai-whisper. "
                "Install Vidra with the 'ai' optional dependencies."
            ) from exc
        self._model = whisper.load_model(self.model_name)
        return self._model

    def _ensure_ffmpeg(self) -> None:
        ensure_ffmpeg_on_path()
        if shutil.which("ffmpeg"):
            return
        raise TranscriptionError(
            "Whisper requires ffmpeg. Install ffmpeg or install Vidra with "
            "the 'ai' optional dependencies."
        )


class AudioTranscriber:
    """Produce timestamped transcript records and context records."""

    tool_name = "transcribe_audio"

    def __init__(self, backend: TranscriptionBackend | None = None) -> None:
        self.backend = backend or WhisperTranscriptionBackend()

    def transcribe_video(
        self,
        media_path: str | Path,
        *,
        video_id: str,
        language: str | None = None,
    ) -> TranscriptResult:
        clean_video_id = video_id.strip()
        if not clean_video_id:
            raise ValueError("video_id is required")
        path = Path(media_path)
        if not path.exists() or not path.is_file():
            raise FileNotFoundError(f"Media file not found: {path}")

        try:
            payload = self.backend.transcribe(path, language=language)
        except MissingAudioError as exc:
            return TranscriptResult(
                video_id=clean_video_id,
                segments=[],
                language="unknown",
                full_text="",
                model_name=self.backend.model_name,
                error=str(exc),
            )

        segments: list[TranscriptSegmentRecord] = []
        text_parts: list[str] = []
        for raw_segment in payload.get("segments", []) or []:
            text = str(raw_segment.get("text", "")).strip()
            if not text:
                continue
            segment = TranscriptSegmentRecord(
                video_id=clean_video_id,
                start_sec=round(float(raw_segment.get("start", 0.0)), 3),
                end_sec=round(float(raw_segment.get("end", 0.0)), 3),
                text=text,
                confidence=self._confidence_from_segment(raw_segment),
            )
            segments.append(segment)
            text_parts.append(text)

        full_text = str(payload.get("text", "")).strip()
        if not full_text:
            full_text = " ".join(text_parts)

        return TranscriptResult(
            video_id=clean_video_id,
            segments=segments,
            language=str(payload.get("language", "unknown") or "unknown"),
            full_text=full_text,
            model_name=self.backend.model_name,
        )

    def to_context_records(
        self,
        transcript: TranscriptResult,
    ) -> list[VideoContextRecord]:
        if transcript.error is not None:
            return [
                VideoContextRecord(
                    context_id=f"{transcript.video_id}-transcript-metadata",
                    video_id=transcript.video_id,
                    context_type=ContextType.metadata,
                    timestamp_sec=None,
                    data={
                        "language": transcript.language,
                        "segments": 0,
                        "text": "",
                        "error": transcript.error,
                    },
                    tool_name=self.tool_name,
                    model_name=transcript.model_name,
                )
            ]

        return [
            VideoContextRecord(
                context_id=self._context_id(transcript.video_id, index),
                video_id=segment.video_id,
                context_type=ContextType.transcript,
                timestamp_sec=segment.start_sec,
                data={
                    "start_sec": segment.start_sec,
                    "end_sec": segment.end_sec,
                    "text": segment.text,
                    "confidence": segment.confidence,
                    "language": transcript.language,
                },
                tool_name=self.tool_name,
                model_name=transcript.model_name,
            )
            for index, segment in enumerate(transcript.segments)
        ]

    def transcribe_video_as_context(
        self,
        media_path: str | Path,
        *,
        video_id: str,
        language: str | None = None,
    ) -> list[VideoContextRecord]:
        return self.to_context_records(
            self.transcribe_video(media_path, video_id=video_id, language=language)
        )

    def _confidence_from_segment(self, segment: dict[str, Any]) -> float:
        if "confidence" in segment:
            return self._clamp_confidence(float(segment["confidence"]))
        if "avg_logprob" in segment:
            return self._clamp_confidence(1.0 + float(segment["avg_logprob"]))
        if "no_speech_prob" in segment:
            return self._clamp_confidence(1.0 - float(segment["no_speech_prob"]))
        return 1.0

    def _clamp_confidence(self, value: float) -> float:
        return min(1.0, max(0.0, value))

    def _context_id(self, video_id: str, index: int) -> str:
        return f"{video_id}-transcript-{index:06d}"
