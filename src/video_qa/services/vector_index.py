"""Chroma-compatible indexing and retrieval for Vidra context records."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from video_qa.models.qa import EvidenceSource, RetrievalHit
from video_qa.models.tools import ContextType, VideoContextRecord
from video_qa.services.embeddings import EmbeddingService


class VectorIndexError(RuntimeError):
    """Raised when vector indexing or retrieval fails."""


@dataclass(frozen=True)
class IndexableContext:
    id: str
    document: str
    modality: str
    context: VideoContextRecord
    image_path: Path | None = None


@runtime_checkable
class VectorCollection(Protocol):
    def upsert(
        self,
        *,
        ids: list[str],
        embeddings: list[list[float]],
        documents: list[str],
        metadatas: list[dict[str, Any]],
    ) -> None:
        """Upsert vectors into the collection."""

    def query(
        self,
        *,
        query_embeddings: list[list[float]],
        n_results: int,
        where: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Query vectors from the collection."""

    def count(self) -> int:
        """Return collection item count."""


class ChromaVectorCollection:
    """Thin ChromaDB adapter hidden behind Vidra's collection protocol."""

    def __init__(
        self,
        *,
        persist_directory: str | Path,
        collection_name: str = "vidra_context",
    ) -> None:
        try:
            import chromadb  # type: ignore[import-not-found]
        except Exception as exc:  # pragma: no cover - optional dependency
            raise VectorIndexError(
                "ChromaDB indexing requires chromadb. Install Vidra with the 'ai' extras."
            ) from exc

        persist_path = Path(persist_directory)
        persist_path.mkdir(parents=True, exist_ok=True)
        if hasattr(chromadb, "PersistentClient"):
            self.client = chromadb.PersistentClient(path=str(persist_path))
        else:  # pragma: no cover - old chromadb compatibility
            from chromadb.config import Settings  # type: ignore[import-not-found]

            self.client = chromadb.Client(
                Settings(
                    persist_directory=str(persist_path),
                    anonymized_telemetry=False,
                    is_persistent=True,
                )
            )
        self.collection = self.client.get_or_create_collection(name=collection_name)

    def upsert(
        self,
        *,
        ids: list[str],
        embeddings: list[list[float]],
        documents: list[str],
        metadatas: list[dict[str, Any]],
    ) -> None:
        self.collection.upsert(
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas,
        )

    def query(
        self,
        *,
        query_embeddings: list[list[float]],
        n_results: int,
        where: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        return self.collection.query(
            query_embeddings=query_embeddings,
            n_results=n_results,
            where=where,
        )

    def count(self) -> int:
        return int(self.collection.count())


class VideoVectorIndex:
    """Indexes caption, transcript, label, crop, and frame context records."""

    def __init__(
        self,
        *,
        collection: VectorCollection,
        embedder: EmbeddingService,
    ) -> None:
        self.collection = collection
        self.embedder = embedder

    def upsert_contexts(self, contexts: list[VideoContextRecord]) -> int:
        indexables = [item for context in contexts if (item := self._to_indexable(context))]
        if not indexables:
            return 0

        text_items = [item for item in indexables if item.modality == "text"]
        image_items = [item for item in indexables if item.modality == "image"]

        all_items: list[IndexableContext] = []
        all_embeddings: list[list[float]] = []
        if text_items:
            all_items.extend(text_items)
            all_embeddings.extend(self.embedder.embed_texts([item.document for item in text_items]))
        if image_items:
            image_paths = [item.image_path for item in image_items]
            if any(path is None for path in image_paths):
                raise VectorIndexError("image indexables require image_path")
            all_items.extend(image_items)
            all_embeddings.extend(
                self.embedder.embed_images([path for path in image_paths if path])
            )

        self.collection.upsert(
            ids=[item.id for item in all_items],
            embeddings=all_embeddings,
            documents=[item.document for item in all_items],
            metadatas=[self._metadata(item) for item in all_items],
        )
        return len(all_items)

    def query(
        self,
        query_text: str,
        *,
        video_id: str | None = None,
        top_k: int = 5,
    ) -> list[RetrievalHit]:
        clean_query = query_text.strip()
        if not clean_query:
            raise ValueError("query_text is required")
        if top_k < 1:
            raise ValueError("top_k must be positive")

        where = {"video_id": video_id} if video_id else None
        vectors = self.embedder.embed_texts([clean_query])
        result = self.collection.query(
            query_embeddings=vectors,
            n_results=top_k,
            where=where,
        )
        return self._to_hits(result)

    def count(self) -> int:
        return self.collection.count()

    def _to_indexable(self, context: VideoContextRecord) -> IndexableContext | None:
        if context.context_type == ContextType.caption:
            text = str(context.data.get("text", "")).strip()
            return self._text_indexable(context, text)
        if context.context_type == ContextType.transcript:
            text = str(context.data.get("text", "")).strip()
            return self._text_indexable(context, text)
        if context.context_type == ContextType.object:
            label = str(context.data.get("label", "")).strip()
            return self._text_indexable(context, label)
        if context.context_type == ContextType.crop:
            label = str(context.data.get("label", "")).strip()
            path = self._path_from_context(context, "crop_path")
            return self._image_indexable(context, label, path)
        if context.context_type == ContextType.frame:
            document = str(context.data.get("frame_id") or context.context_id)
            path = self._path_from_context(context, "image_path")
            return self._image_indexable(context, document, path)
        return None

    def _text_indexable(self, context: VideoContextRecord, text: str) -> IndexableContext | None:
        if not text:
            return None
        return IndexableContext(
            id=context.context_id,
            document=text,
            modality="text",
            context=context,
        )

    def _image_indexable(
        self,
        context: VideoContextRecord,
        document: str,
        image_path: Path | None,
    ) -> IndexableContext | None:
        if image_path is None:
            return None
        return IndexableContext(
            id=context.context_id,
            document=document or context.context_id,
            modality="image",
            context=context,
            image_path=image_path,
        )

    def _path_from_context(self, context: VideoContextRecord, key: str) -> Path | None:
        raw = context.data.get(key)
        if not raw:
            return None
        return Path(str(raw))

    def _metadata(self, item: IndexableContext) -> dict[str, Any]:
        data = item.context.data
        metadata: dict[str, Any] = {
            "video_id": item.context.video_id,
            "context_id": item.context.context_id,
            "context_type": item.context.context_type.value,
            "timestamp_sec": item.context.timestamp_sec,
            "modality": item.modality,
            "tool_name": item.context.tool_name,
            "model_name": item.context.model_name,
            "label": data.get("label"),
            "frame_path": data.get("frame_path") or data.get("image_path"),
            "crop_path": data.get("crop_path"),
        }
        return {key: value for key, value in metadata.items() if value is not None}

    def _to_hits(self, result: dict[str, Any]) -> list[RetrievalHit]:
        ids = self._first(result.get("ids"))
        documents = self._first(result.get("documents"))
        metadatas = self._first(result.get("metadatas"))
        distances = self._first(result.get("distances"))

        hits: list[RetrievalHit] = []
        for index, item_id in enumerate(ids):
            metadata = dict(metadatas[index] or {})
            distance = float(distances[index]) if index < len(distances) else 0.0
            score = 1.0 / (1.0 + max(0.0, distance))
            source = EvidenceSource(
                video_id=str(metadata.get("video_id", "")),
                context_id=str(metadata.get("context_id", item_id)),
                context_type=str(metadata.get("context_type", "")),
                timestamp_sec=metadata.get("timestamp_sec"),
                label=metadata.get("label"),
                frame_path=(
                    Path(str(metadata["frame_path"]))
                    if metadata.get("frame_path")
                    else None
                ),
                crop_path=Path(str(metadata["crop_path"])) if metadata.get("crop_path") else None,
            )
            hits.append(
                RetrievalHit(
                    id=str(item_id),
                    modality=str(metadata.get("modality", "text")),  # type: ignore[arg-type]
                    score=score,
                    source=source,
                    text=documents[index] if index < len(documents) else None,
                    metadata={
                        key: value
                        for key, value in metadata.items()
                        if key
                        not in {
                            "video_id",
                            "context_id",
                            "context_type",
                            "timestamp_sec",
                            "label",
                            "frame_path",
                            "crop_path",
                        }
                    },
                )
            )
        return hits

    def _first(self, value: Any) -> list[Any]:
        if not value:
            return []
        if isinstance(value, list) and value and isinstance(value[0], list):
            return value[0]
        if isinstance(value, list):
            return value
        return []
