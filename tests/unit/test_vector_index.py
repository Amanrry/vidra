from __future__ import annotations

import math
from pathlib import Path
from typing import Any

from video_qa.models.tools import ContextType, VideoContextRecord
from video_qa.services.embeddings import EmbeddingService, FakeEmbeddingBackend
from video_qa.services.vector_index import ChromaVectorCollection, VideoVectorIndex


class MemoryVectorCollection:
    def __init__(self) -> None:
        self.items: dict[str, dict[str, Any]] = {}

    def upsert(
        self,
        *,
        ids: list[str],
        embeddings: list[list[float]],
        documents: list[str],
        metadatas: list[dict[str, Any]],
    ) -> None:
        for item_id, embedding, document, metadata in zip(
            ids,
            embeddings,
            documents,
            metadatas,
            strict=True,
        ):
            self.items[item_id] = {
                "embedding": embedding,
                "document": document,
                "metadata": metadata,
            }

    def query(
        self,
        *,
        query_embeddings: list[list[float]],
        n_results: int,
        where: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        query = query_embeddings[0]
        scored = []
        for item_id, item in self.items.items():
            metadata = item["metadata"]
            if where and any(metadata.get(key) != value for key, value in where.items()):
                continue
            distance = math.sqrt(
                sum(
                    (left - right) ** 2
                    for left, right in zip(query, item["embedding"], strict=True)
                )
            )
            scored.append((distance, item_id, item))
        scored.sort(key=lambda row: row[0])
        selected = scored[:n_results]
        return {
            "ids": [[item_id for _, item_id, _ in selected]],
            "documents": [[item["document"] for _, _, item in selected]],
            "metadatas": [[item["metadata"] for _, _, item in selected]],
            "distances": [[distance for distance, _, _ in selected]],
        }

    def count(self) -> int:
        return len(self.items)


def context(
    context_id: str,
    context_type: ContextType,
    data: dict[str, Any],
    *,
    timestamp_sec: float | None = 1.0,
) -> VideoContextRecord:
    return VideoContextRecord(
        context_id=context_id,
        video_id="video-1",
        context_type=context_type,
        timestamp_sec=timestamp_sec,
        data=data,
        tool_name=f"{context_type.value}_tool",
        model_name="fake-model",
    )


def test_vector_index_upserts_and_queries_multimodal_context_records(
    tmp_path: Path,
) -> None:
    frame_path = tmp_path / "frame.jpg"
    crop_path = tmp_path / "crop.jpg"
    frame_path.write_bytes(b"frame")
    crop_path.write_bytes(b"crop")
    collection = MemoryVectorCollection()
    index = VideoVectorIndex(
        collection=collection,
        embedder=EmbeddingService(FakeEmbeddingBackend(dimension=8)),
    )

    records = [
        context("caption-1", ContextType.caption, {"text": "a red car enters"}),
        context("transcript-1", ContextType.transcript, {"text": "someone says hello"}),
        context("object-1", ContextType.object, {"label": "person", "frame_path": str(frame_path)}),
        context("crop-1", ContextType.crop, {"label": "person", "crop_path": str(crop_path)}),
        context(
            "frame-1",
            ContextType.frame,
            {"frame_id": "frame-1", "image_path": str(frame_path)},
        ),
    ]

    indexed = index.upsert_contexts(records)

    assert indexed == 5
    assert index.count() == 5
    assert set(collection.items) == {
        "caption-1",
        "transcript-1",
        "object-1",
        "crop-1",
        "frame-1",
    }
    assert collection.items["caption-1"]["metadata"]["modality"] == "text"
    assert collection.items["transcript-1"]["metadata"]["context_type"] == "transcript"
    assert collection.items["object-1"]["document"] == "person"
    assert collection.items["crop-1"]["metadata"]["modality"] == "image"
    assert collection.items["frame-1"]["metadata"]["modality"] == "image"

    hits = index.query("person", video_id="video-1", top_k=3)

    assert len(hits) == 3
    assert all(hit.source.video_id == "video-1" for hit in hits)
    assert hits[0].score > 0.0
    assert hits[0].source.context_id in collection.items


def test_vector_index_upsert_is_idempotent(tmp_path: Path) -> None:
    frame_path = tmp_path / "frame.jpg"
    frame_path.write_bytes(b"frame")
    collection = MemoryVectorCollection()
    index = VideoVectorIndex(
        collection=collection,
        embedder=EmbeddingService(FakeEmbeddingBackend()),
    )
    record = context("caption-1", ContextType.caption, {"text": "first"})

    index.upsert_contexts([record])
    index.upsert_contexts([record.model_copy(update={"data": {"text": "updated"}})])

    assert index.count() == 1
    assert collection.items["caption-1"]["document"] == "updated"


def test_vector_index_skips_non_indexable_metadata() -> None:
    collection = MemoryVectorCollection()
    index = VideoVectorIndex(
        collection=collection,
        embedder=EmbeddingService(FakeEmbeddingBackend()),
    )
    metadata = context(
        "meta-1",
        ContextType.metadata,
        {"status": "no audio"},
        timestamp_sec=None,
    )

    assert index.upsert_contexts([metadata]) == 0
    assert index.count() == 0


def test_chroma_collection_persists_across_instances(tmp_path: Path) -> None:
    collection_name = "vidra_test_context"
    first = ChromaVectorCollection(
        persist_directory=tmp_path / "chroma",
        collection_name=collection_name,
    )
    first.upsert(
        ids=["item-1"],
        embeddings=[[1.0, 0.0, 0.0]],
        documents=["person"],
        metadatas=[{"video_id": "video-1", "context_id": "item-1", "context_type": "object"}],
    )

    second = ChromaVectorCollection(
        persist_directory=tmp_path / "chroma",
        collection_name=collection_name,
    )

    assert second.count() == 1
    result = second.query(query_embeddings=[[1.0, 0.0, 0.0]], n_results=1)
    assert result["ids"][0] == ["item-1"]
