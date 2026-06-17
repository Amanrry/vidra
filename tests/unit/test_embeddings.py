from __future__ import annotations

import math
from pathlib import Path

import pytest

from video_qa.services.embeddings import (
    EmbeddingError,
    EmbeddingService,
    FakeEmbeddingBackend,
    SiglipEmbeddingBackend,
)


class RealInterfaceBackend:
    model_name = "google/siglip-test-double"
    dimension = 3

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        return [[3.0, 4.0, 0.0] for _ in texts]

    def embed_images(self, image_paths: list[Path]) -> list[list[float]]:
        return [[0.0, 5.0, 12.0] for _ in image_paths]


def vector_norm(vector: list[float]) -> float:
    return math.sqrt(sum(value * value for value in vector))


def test_fake_embedder_returns_normalized_text_and_image_vectors(tmp_path: Path) -> None:
    image_path = tmp_path / "frame.jpg"
    image_path.write_bytes(b"image")
    service = EmbeddingService(FakeEmbeddingBackend(dimension=6))

    text_vectors = service.embed_texts(["a person walking", "red car"])
    image_vectors = service.embed_images([image_path])

    assert len(text_vectors) == 2
    assert len(text_vectors[0]) == 6
    assert vector_norm(text_vectors[0]) == pytest.approx(1.0)
    assert vector_norm(text_vectors[1]) == pytest.approx(1.0)
    assert vector_norm(image_vectors[0]) == pytest.approx(1.0)


def test_real_service_interface_vectors_are_normalized(tmp_path: Path) -> None:
    image_path = tmp_path / "crop.jpg"
    image_path.write_bytes(b"image")
    service = EmbeddingService(RealInterfaceBackend())

    assert service.model_name == "google/siglip-test-double"
    assert service.embed_texts(["hello"])[0] == pytest.approx([0.6, 0.8, 0.0])
    assert service.embed_images([image_path])[0] == pytest.approx([0.0, 5.0 / 13.0, 12.0 / 13.0])


def test_embedding_service_rejects_empty_or_bad_vectors() -> None:
    service = EmbeddingService(FakeEmbeddingBackend())

    with pytest.raises(ValueError, match="empty"):
        service.embed_texts([" "])

    class BadBackend(RealInterfaceBackend):
        def embed_texts(self, texts: list[str]) -> list[list[float]]:
            return [[0.0, 0.0, 0.0]]

    with pytest.raises(EmbeddingError, match="norm"):
        EmbeddingService(BadBackend()).embed_texts(["hello"])


def test_siglip_backend_accepts_pooling_model_outputs() -> None:
    backend = object.__new__(SiglipEmbeddingBackend)

    class TorchStub:
        class nn:
            class functional:
                @staticmethod
                def normalize(tensor, p, dim):
                    _ = (p, dim)
                    return tensor

    class TensorStub:
        def __init__(self, rows):
            self.rows = rows

        def detach(self):
            return self

        def cpu(self):
            return self

        def tolist(self):
            return self.rows

    class Output:
        pooler_output = TensorStub([[1.0, 0.0, 0.0]])

    backend._torch = TorchStub()
    backend.dimension = None

    vectors = backend._tensor_to_vectors(Output())

    assert vectors == [[1.0, 0.0, 0.0]]
    assert backend.dimension == 3
