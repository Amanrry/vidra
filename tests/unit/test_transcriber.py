from __future__ import annotations

from pathlib import Path

import pytest

from video_qa.models.tools import ContextType
from video_qa.tools.audio_transcriber import AudioTranscriber, MissingAudioError


class FakeTranscriptionBackend:
    model_name = "fake-whisper"

    def __init__(self, payload: dict | None = None, *, missing_audio: bool = False) -> None:
        self.payload = payload or {}
        self.missing_audio = missing_audio
        self.calls: list[tuple[Path, str | None]] = []

    def transcribe(
        self,
        media_path: Path,
        *,
        language: str | None = None,
    ) -> dict:
        self.calls.append((media_path, language))
        if self.missing_audio:
            raise MissingAudioError("No usable audio track found.")
        return self.payload


def write_video(path: Path) -> Path:
    path.write_bytes(b"video")
    return path


def test_transcriber_produces_timestamped_segments_and_context_records(
    tmp_path: Path,
) -> None:
    video_path = write_video(tmp_path / "demo.mp4")
    backend = FakeTranscriptionBackend(
        {
            "language": "en",
            "text": "hello world next line",
            "segments": [
                {"start": 0.0, "end": 1.25, "text": " hello world ", "confidence": 0.8},
                {"start": 1.25, "end": 2.0, "text": "next line", "no_speech_prob": 0.2},
            ],
        }
    )
    transcriber = AudioTranscriber(backend=backend)

    result = transcriber.transcribe_video(video_path, video_id="video-1", language="en")
    contexts = transcriber.to_context_records(result)

    assert result.language == "en"
    assert result.full_text == "hello world next line"
    assert [segment.start_sec for segment in result.segments] == [0.0, 1.25]
    assert [segment.end_sec for segment in result.segments] == [1.25, 2.0]
    assert [segment.text for segment in result.segments] == ["hello world", "next line"]
    assert result.segments[1].confidence == pytest.approx(0.8)

    assert [context.context_type for context in contexts] == [
        ContextType.transcript,
        ContextType.transcript,
    ]
    assert contexts[0].context_id == "video-1-transcript-000000"
    assert contexts[0].timestamp_sec == 0.0
    assert contexts[0].data["start_sec"] == 0.0
    assert contexts[0].data["end_sec"] == 1.25
    assert contexts[0].data["language"] == "en"
    assert backend.calls == [(video_path, "en")]


def test_transcriber_tolerates_missing_audio_with_metadata_context(
    tmp_path: Path,
) -> None:
    video_path = write_video(tmp_path / "silent.mp4")
    transcriber = AudioTranscriber(
        backend=FakeTranscriptionBackend(missing_audio=True),
    )

    result = transcriber.transcribe_video(video_path, video_id="video-1")
    contexts = transcriber.to_context_records(result)

    assert result.segments == []
    assert result.error == "No usable audio track found."
    assert len(contexts) == 1
    assert contexts[0].context_type == ContextType.metadata
    assert contexts[0].timestamp_sec is None
    assert contexts[0].data["segments"] == 0
    assert contexts[0].data["error"] == "No usable audio track found."


def test_transcriber_validates_input_path_and_video_id(tmp_path: Path) -> None:
    transcriber = AudioTranscriber(backend=FakeTranscriptionBackend())
    video_path = write_video(tmp_path / "demo.mp4")

    with pytest.raises(ValueError, match="video_id"):
        transcriber.transcribe_video(video_path, video_id=" ")

    with pytest.raises(FileNotFoundError):
        transcriber.transcribe_video(tmp_path / "missing.mp4", video_id="video-1")
