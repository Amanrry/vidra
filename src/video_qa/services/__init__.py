"""Application services and orchestration boundaries."""

from video_qa.services.application import (
    ApplicationService,
    ChatResult,
    ProcessingResult,
    UploadResult,
)
from video_qa.services.context_builder import BuiltContext, ContextBuilder, PromptEvidence
from video_qa.services.embeddings import (
    EmbeddingService,
    FakeEmbeddingBackend,
    SiglipEmbeddingBackend,
)
from video_qa.services.qa_agent import OpenAICompatibleClient, QAAgent, QAResponse
from video_qa.services.vector_index import ChromaVectorCollection, VideoVectorIndex
from video_qa.services.video_context import VideoContextRepository

__all__ = [
    "ApplicationService",
    "ChatResult",
    "ChromaVectorCollection",
    "BuiltContext",
    "ContextBuilder",
    "EmbeddingService",
    "FakeEmbeddingBackend",
    "OpenAICompatibleClient",
    "ProcessingResult",
    "PromptEvidence",
    "QAAgent",
    "QAResponse",
    "SiglipEmbeddingBackend",
    "UploadResult",
    "VideoVectorIndex",
    "VideoContextRepository",
]
