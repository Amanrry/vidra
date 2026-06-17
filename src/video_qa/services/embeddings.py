"""SigLIP embedding service with normalized vector contracts."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Protocol, runtime_checkable

from video_qa.tools.image_captioner import resolve_transformers_model_path


class EmbeddingError(RuntimeError):
    """Raised when an embedding backend cannot produce usable vectors."""


@runtime_checkable
class EmbeddingBackend(Protocol):
    model_name: str
    dimension: int | None

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of text strings."""

    def embed_images(self, image_paths: list[Path]) -> list[list[float]]:
        """Embed a batch of image files."""


class FakeEmbeddingBackend:
    """Deterministic test backend that still exercises normalization."""

    def __init__(self, *, dimension: int = 8, model_name: str = "fake-siglip") -> None:
        self.dimension = dimension
        self.model_name = model_name

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        return [self._vector_from_seed(text) for text in texts]

    def embed_images(self, image_paths: list[Path]) -> list[list[float]]:
        return [self._vector_from_seed(str(path)) for path in image_paths]

    def _vector_from_seed(self, seed: str) -> list[float]:
        values = []
        encoded = seed.encode("utf-8") or b"0"
        for index in range(self.dimension):
            byte = encoded[index % len(encoded)]
            values.append(float(((byte + index * 31) % 127) + 1))
        return values


class SiglipEmbeddingBackend:
    """Lazy SigLIP text/image embedding adapter."""

    def __init__(
        self,
        model_name: str = "google/siglip-base-patch16-224",
        *,
        local_models_dir: str | Path = ".models",
    ) -> None:
        self.model_name = model_name
        self.model_path = resolve_transformers_model_path(model_name, local_models_dir)
        self.dimension: int | None = None
        self._loaded = False

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        self._load()
        inputs = self._processor(text=texts, padding=True, truncation=True, return_tensors="pt").to(
            self._device
        )
        with self._torch.no_grad():
            features = self._model.get_text_features(**inputs)
        return self._tensor_to_vectors(features)

    def embed_images(self, image_paths: list[Path]) -> list[list[float]]:
        self._load()
        images = []
        for path in image_paths:
            if not path.exists() or not path.is_file():
                raise FileNotFoundError(f"Image file not found: {path}")
            images.append(self._image_open(path).convert("RGB"))
        inputs = self._processor(images=images, return_tensors="pt").to(self._device)
        with self._torch.no_grad():
            features = self._model.get_image_features(**inputs)
        return self._tensor_to_vectors(features)

    def _load(self) -> None:
        if self._loaded:
            return
        try:
            import torch  # type: ignore[import-not-found]
            from PIL import Image  # type: ignore[import-not-found]
            from transformers import AutoModel, AutoProcessor  # type: ignore[import-not-found]
        except Exception as exc:  # pragma: no cover - optional dependencies
            raise EmbeddingError(
                "SigLIP embeddings require pillow, torch, and transformers. "
                "Install Vidra with the 'ai' optional dependencies."
            ) from exc

        self._torch = torch
        self._image_open = Image.open
        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        local_only = Path(self.model_path).exists()
        self._processor = AutoProcessor.from_pretrained(
            self.model_path,
            local_files_only=local_only,
        )
        self._model = AutoModel.from_pretrained(
            self.model_path,
            local_files_only=local_only,
        ).to(self._device)
        self._model.eval()
        self._loaded = True

    def _tensor_to_vectors(self, output) -> list[list[float]]:
        tensor = self._extract_feature_tensor(output)
        normalized = self._torch.nn.functional.normalize(tensor, p=2, dim=1)
        vectors = normalized.detach().cpu().tolist()
        if vectors:
            self.dimension = len(vectors[0])
        return vectors

    def _extract_feature_tensor(self, output):
        if hasattr(output, "norm"):
            return output
        for attribute in ("pooler_output", "image_embeds", "text_embeds", "last_hidden_state"):
            value = getattr(output, attribute, None)
            if value is None:
                continue
            if attribute == "last_hidden_state":
                return value[:, 0, :]
            return value
        if isinstance(output, (tuple, list)) and output:
            return output[0]
        raise EmbeddingError(
            f"SigLIP backend returned unsupported output type: {type(output).__name__}"
        )


class EmbeddingService:
    """Application-facing embedding service that enforces L2 normalization."""

    def __init__(self, backend: EmbeddingBackend | None = None) -> None:
        self.backend = backend or SiglipEmbeddingBackend()

    @property
    def model_name(self) -> str:
        return self.backend.model_name

    @property
    def dimension(self) -> int | None:
        return self.backend.dimension

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        clean_texts = [text.strip() for text in texts]
        if any(not text for text in clean_texts):
            raise ValueError("texts must not contain empty strings")
        return self._normalize_batch(self.backend.embed_texts(clean_texts), len(clean_texts))

    def embed_images(self, image_paths: list[str | Path]) -> list[list[float]]:
        paths = [Path(path) for path in image_paths]
        return self._normalize_batch(self.backend.embed_images(paths), len(paths))

    def _normalize_batch(
        self,
        vectors: list[list[float]],
        expected_count: int,
    ) -> list[list[float]]:
        if len(vectors) != expected_count:
            raise EmbeddingError(
                f"backend returned {len(vectors)} vectors for {expected_count} inputs"
            )
        return [self._normalize_vector(vector) for vector in vectors]

    def _normalize_vector(self, vector: list[float]) -> list[float]:
        if not vector:
            raise EmbeddingError("embedding vector must not be empty")
        norm = math.sqrt(sum(float(value) * float(value) for value in vector))
        if norm <= 0:
            raise EmbeddingError("embedding vector norm must be positive")
        return [float(value) / norm for value in vector]
