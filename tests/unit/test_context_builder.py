from __future__ import annotations

from pathlib import Path

from video_qa.models.qa import EvidenceSource, RetrievalHit
from video_qa.models.tools import ContextType, VideoContextRecord
from video_qa.services.context_builder import ContextBuilder
from video_qa.services.video_context import VideoContextRepository
from video_qa.storage import Database


class FakeRetriever:
    def __init__(self, hits: list[RetrievalHit]) -> None:
        self.hits = hits
        self.calls: list[tuple[str, str | None, int]] = []

    def query(
        self,
        query_text: str,
        *,
        video_id: str | None = None,
        top_k: int = 5,
    ) -> list[RetrievalHit]:
        self.calls.append((query_text, video_id, top_k))
        return self.hits


def insert_video(database: Database, video_id: str = "video-1") -> None:
    database.execute(
        """
        INSERT INTO videos (video_id, filename, file_path, duration_sec)
        VALUES (?, ?, ?, ?)
        """,
        (video_id, f"{video_id}.mp4", f"data/input/{video_id}.mp4", 0.0),
    )


def make_repository(tmp_path: Path) -> VideoContextRepository:
    database = Database(tmp_path / "vidra.sqlite3")
    database.initialize()
    insert_video(database)
    return VideoContextRepository(database)


def make_context(
    context_id: str,
    context_type: ContextType,
    data: dict,
    timestamp_sec: float,
) -> VideoContextRecord:
    return VideoContextRecord(
        context_id=context_id,
        video_id="video-1",
        context_type=context_type,
        timestamp_sec=timestamp_sec,
        data=data,
        tool_name=f"{context_type.value}_tool",
        model_name="fake",
    )


def test_context_builder_prioritizes_captions_transcripts_objects_crops(
    tmp_path: Path,
) -> None:
    repository = make_repository(tmp_path)
    repository.upsert_contexts(
        [
            make_context(
                "crop-1",
                ContextType.crop,
                {"label": "person", "crop_path": "crop.jpg"},
                4.0,
            ),
            make_context("object-1", ContextType.object, {"label": "car", "confidence": 0.8}, 3.0),
            make_context("transcript-1", ContextType.transcript, {"text": "hello there"}, 2.0),
            make_context("caption-1", ContextType.caption, {"text": "a person near a car"}, 1.0),
        ]
    )
    builder = ContextBuilder(context_repository=repository, top_k=8)

    context = builder.build(video_id="video-1", question="What is happening?")

    assert [item.context_type for item in context.evidence] == [
        ContextType.caption,
        ContextType.transcript,
        ContextType.object,
        ContextType.crop,
    ]
    prompt = builder.build_prompt(context)
    assert "Use only the evidence below" in prompt
    assert "Do not invent people, objects, speech, actions, locations, or timestamps" in prompt
    assert prompt.index("[caption]") < prompt.index("[transcript]")
    assert prompt.index("[transcript]") < prompt.index("[object]")
    assert prompt.index("[object]") < prompt.index("[crop]")


def test_context_builder_merges_retrieval_hits_with_stored_fallback(
    tmp_path: Path,
) -> None:
    repository = make_repository(tmp_path)
    repository.upsert_contexts(
        [
            make_context("caption-1", ContextType.caption, {"text": "stored caption"}, 1.0),
            make_context(
                "transcript-1",
                ContextType.transcript,
                {"text": "stored transcript"},
                2.0,
            ),
        ]
    )
    hit = RetrievalHit(
        id="caption-1",
        modality="text",
        score=0.91,
        source=EvidenceSource(
            video_id="video-1",
            context_id="caption-1",
            context_type="caption",
            timestamp_sec=1.0,
        ),
        text="retrieved caption",
    )
    retriever = FakeRetriever([hit])
    builder = ContextBuilder(context_repository=repository, retriever=retriever, top_k=4)

    context = builder.build(video_id="video-1", question="Find the caption")

    assert retriever.calls == [("Find the caption", "video-1", 4)]
    assert [item.context_id for item in context.evidence] == ["caption-1", "transcript-1"]
    assert context.evidence[0].text == "retrieved caption"
    assert context.evidence[0].score == 0.91


def test_context_builder_keeps_object_evidence_when_prompt_is_limited(
    tmp_path: Path,
) -> None:
    repository = make_repository(tmp_path)
    repository.upsert_contexts(
        [
            make_context(
                f"caption-{index}",
                ContextType.caption,
                {"text": f"caption {index}"},
                index,
            )
            for index in range(4)
        ]
        + [
            make_context(
                f"object-{index}",
                ContextType.object,
                {"label": label, "confidence": 0.8},
                10.0,
            )
            for index, label in enumerate(["person", "bicycle", "motorcycle", "person"])
        ]
    )
    builder = ContextBuilder(context_repository=repository, top_k=6, fallback_per_type=4)

    context = builder.build(video_id="video-1", question="What objects are detected?")

    object_labels = [
        item.label
        for item in context.evidence
        if item.context_type == ContextType.object
    ]
    assert object_labels == ["person", "bicycle", "motorcycle", "person"]


def test_context_builder_handles_missing_evidence(tmp_path: Path) -> None:
    repository = make_repository(tmp_path)
    builder = ContextBuilder(context_repository=repository)

    context = builder.build(video_id="video-1", question="Anything?")
    prompt = builder.build_prompt(context)

    assert not context.has_evidence
    assert "No processed evidence is available yet." in prompt


def test_context_builder_keyword_timestamp_object_and_rich_description(tmp_path: Path) -> None:
    repository = make_repository(tmp_path)
    repository.upsert_contexts(
        [
            make_context(
                "metadata-1",
                ContextType.metadata,
                {"duration_sec": 12.0, "width": 640, "height": 360},
                0.0,
            ),
            make_context(
                "frame-1",
                ContextType.frame,
                {"frame_id": "frame-1", "image_path": "frame.jpg"},
                1.0,
            ),
            make_context(
                "caption-1",
                ContextType.caption,
                {"text": "A man is walking beside a vehicle"},
                1.0,
            ),
            make_context(
                "transcript-1",
                ContextType.transcript,
                {"text": "The speaker says hello"},
                2.0,
            ),
            make_context(
                "object-1",
                ContextType.object,
                {
                    "label": "person",
                    "confidence": 0.9,
                    "frame_id": "frame-1",
                    "frame_path": "frame.jpg",
                },
                1.0,
            ),
        ]
    )
    builder = ContextBuilder(context_repository=repository)

    captions = builder.search_captions_keyword("video-1", "person car", top_k=2)
    transcripts = builder.search_transcripts_keyword("video-1", "speaker hello", top_k=2)
    timestamp = builder.get_context_at_timestamp("video-1", 1.2, window_sec=1.0)
    objects = builder.get_objects("video-1", "person")
    frames = builder.get_frames_with_object("video-1", "person")
    description = builder.build_rich_context_description("video-1")

    assert captions[0].context_id == "caption-1"
    assert transcripts[0].context_id == "transcript-1"
    assert {item.context_id for item in timestamp} >= {"caption-1", "object-1"}
    assert objects[0].label == "person"
    assert frames[0].context_id == "frame-1"
    assert "Visual Analysis" in description
    assert "Detected Objects" in description
