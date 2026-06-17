"""Command line entry points for Vidra workflows."""

from __future__ import annotations

import argparse
import asyncio
import json
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

from video_qa.config import Settings, load_settings
from video_qa.models.processing import JobPriority, JobStatus
from video_qa.runtime import ensure_media_binaries_on_path
from video_qa.services.application import ApplicationService
from video_qa.services.context_builder import ContextBuilder
from video_qa.services.embeddings import EmbeddingService, SiglipEmbeddingBackend
from video_qa.services.pipeline_stages import (
    CaptionStage,
    EnrichmentStage,
    FrameExtractionStage,
    IndexAndReportStage,
)
from video_qa.services.processing_progress import ProcessingProgressRepository
from video_qa.services.processing_queue import ProcessingQueue
from video_qa.services.qa_agent import OpenAICompatibleClient, QAAgent
from video_qa.services.reporting import ReportWriter
from video_qa.services.vector_index import ChromaVectorCollection, VideoVectorIndex
from video_qa.services.video_context import VideoContextRepository
from video_qa.services.video_processing_service import VideoProcessingService
from video_qa.services.video_processor import ProgressiveVideoProcessor
from video_qa.storage import Database, RunLayout
from video_qa.tools import (
    AudioTranscriber,
    BlipCaptionBackend,
    ImageCaptioner,
    ObjectDetector,
    WhisperTranscriptionBackend,
    YoloDetectionBackend,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="video_qa", description="Vidra CLI")
    parser.add_argument("--config", default=None, help="Path to YAML config file.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    enqueue = subparsers.add_parser("enqueue", help="Upload a video and enqueue processing.")
    enqueue.add_argument("--video", required=True)
    enqueue.add_argument("--video-id", default=None)

    process = subparsers.add_parser("process", help="Run direct synchronous processing.")
    process.add_argument("--video", required=True)
    process.add_argument("--run-id", default=None)
    process.add_argument("--video-id", default=None)

    status = subparsers.add_parser("status", help="Inspect processing status.")
    status.add_argument("--video-id", required=True)

    worker = subparsers.add_parser("worker", help="Run in-process queue workers.")
    worker.add_argument("--drain", action="store_true", help="Drain queued jobs and exit.")

    search = subparsers.add_parser("search", help="Search indexed video context.")
    search.add_argument("--query", required=True)
    search.add_argument("--video-id", default=None)
    search.add_argument("--top-k", type=int, default=5)

    ask = subparsers.add_parser("ask", help="Ask a grounded QA question.")
    ask.add_argument("--video-id", required=True)
    ask.add_argument("--question", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    ensure_media_binaries_on_path()
    parser = build_parser()
    args = parser.parse_args(argv)
    settings = load_settings(args.config)
    runtime = create_runtime(
        settings,
        enable_vector=args.command in {"search", "ask"},
        enable_qa=args.command == "ask",
        real_pipeline=args.command in {"enqueue", "process", "worker"},
    )
    result = run_command(args, runtime)
    print(json.dumps(_jsonable(result), ensure_ascii=False, indent=2))
    return 0 if result.get("ok", True) else 1


def create_runtime(
    settings: Settings,
    *,
    enable_vector: bool = False,
    enable_qa: bool = False,
    real_pipeline: bool = False,
) -> dict[str, Any]:
    settings.ensure_directories()
    database = Database(settings.paths.database_path)
    progress_repository = ProcessingProgressRepository(database)
    context_repository = VideoContextRepository(database)
    layout = RunLayout(settings.paths.runs_dir)
    processing_service = VideoProcessingService(database)
    vector_index = None
    if enable_vector or enable_qa or real_pipeline:
        vector_index = VideoVectorIndex(
            collection=ChromaVectorCollection(persist_directory=settings.paths.chroma_dir),
            embedder=EmbeddingService(
                SiglipEmbeddingBackend(settings.models.siglip_model)
            ),
        )
    processor = create_processor(
        settings=settings,
        layout=layout,
        progress_repository=progress_repository,
        context_repository=context_repository,
        processing_service=processing_service,
        vector_index=vector_index,
        real_pipeline=real_pipeline,
    )
    queue = ProcessingQueue(
        runner=processor.process_job,
        max_workers=1,
        progress_repository=progress_repository,
    )
    app_service = ApplicationService(
        settings,
        database,
        processing_queue=queue,
        progress_repository=progress_repository,
    )
    runtime = {
        "settings": settings,
        "database": database,
        "progress_repository": progress_repository,
        "processing_service": processing_service,
        "processor": processor,
        "queue": queue,
        "app_service": app_service,
        "context_repository": context_repository,
    }
    if vector_index is not None and (enable_vector or enable_qa or real_pipeline):
        runtime["vector_index"] = vector_index
    if enable_qa:
        if vector_index is None:
            raise RuntimeError("QA requires vector indexing to be enabled")
        runtime["qa_agent"] = QAAgent(
            context_builder=ContextBuilder(
                context_repository=context_repository,
                retriever=vector_index,
                top_k=settings.retrieval.top_k_text,
            ),
            chat_client=OpenAICompatibleClient(settings.llm),
        )
        app_service.chat_agent = QAChatAdapter(runtime["qa_agent"])
    return runtime


def create_processor(
    *,
    settings: Settings,
    layout: RunLayout,
    progress_repository: ProcessingProgressRepository,
    context_repository: VideoContextRepository,
    processing_service: VideoProcessingService,
    vector_index: VideoVectorIndex | None,
    real_pipeline: bool,
) -> ProgressiveVideoProcessor:
    if not real_pipeline:
        return ProgressiveVideoProcessor(progress_repository=progress_repository)
    if vector_index is None:
        raise RuntimeError("real pipeline requires a vector index")
    report_writer = ReportWriter(
        context_repository=context_repository,
        processing_service=processing_service,
    )
    return ProgressiveVideoProcessor(
        progress_repository=progress_repository,
        frame_extractor=FrameExtractionStage(
            settings=settings,
            layout=layout,
            processing_service=processing_service,
        ),
        captioner=CaptionStage(
            context_repository=context_repository,
            processing_service=processing_service,
            captioner=ImageCaptioner(
                backend=BlipCaptionBackend(settings.models.caption_model)
            ),
        ),
        enricher=EnrichmentStage(
            layout=layout,
            context_repository=context_repository,
            processing_service=processing_service,
            transcriber=AudioTranscriber(
                backend=WhisperTranscriptionBackend(settings.models.whisper_model)
            ),
            detector=ObjectDetector(
                backend=YoloDetectionBackend(settings.models.yolo_model)
            ),
        ),
        indexer_reporter=IndexAndReportStage(
            layout=layout,
            context_repository=context_repository,
            vector_index=vector_index,
            report_writer=report_writer,
        ),
    )


def run_command(args: argparse.Namespace, runtime: dict[str, Any]) -> dict[str, Any]:
    command = args.command
    if command == "enqueue":
        return enqueue_process(runtime["app_service"], Path(args.video), video_id=args.video_id)
    if command == "process":
        return process_direct(
            runtime["app_service"],
            runtime["progress_repository"],
            runtime["processor"],
            Path(args.video),
            video_id=args.video_id or args.run_id,
        )
    if command == "status":
        return status(runtime["app_service"], args.video_id)
    if command == "worker":
        return worker(runtime["queue"], drain=args.drain)
    if command == "search":
        return search(runtime["vector_index"], args.query, video_id=args.video_id, top_k=args.top_k)
    if command == "ask":
        return ask(runtime["qa_agent"], args.video_id, args.question)
    raise ValueError(f"Unsupported command: {command}")


class QAChatAdapter:
    def __init__(self, qa_agent: QAAgent) -> None:
        self.qa_agent = qa_agent

    def answer(self, video, message: str) -> str:
        return self.qa_agent.answer(video_id=video.video_id, question=message).answer


def enqueue_process(
    service: ApplicationService,
    video_path: Path,
    *,
    video_id: str | None = None,
) -> dict[str, Any]:
    upload = service.upload_video(video_path, video_id=video_id)
    if not upload.ok or upload.video is None:
        return {"ok": False, "message": upload.message}
    process = service.start_processing(upload.video.video_id)
    return {
        "ok": process.ok,
        "message": process.message,
        "video_id": process.video_id,
        "status": process.status.value,
        "job_id": process.job_id,
    }


def process_direct(
    service: ApplicationService,
    progress_repository: ProcessingProgressRepository,
    processor: ProgressiveVideoProcessor,
    video_path: Path,
    *,
    video_id: str | None = None,
) -> dict[str, Any]:
    upload = service.upload_video(video_path, video_id=video_id)
    if not upload.ok or upload.video is None:
        return {"ok": False, "message": upload.message}
    job = progress_repository.create_job(
        job_id=f"{upload.video.video_id}-direct",
        video_id=upload.video.video_id,
        run_id=upload.video.video_id,
        source_path=upload.video.file_path,
        priority=JobPriority.high,
        message="Direct processing started.",
    )
    progress_repository.mark_processing(job.job_id)
    processing_job = _processing_job_from_record(job)
    result = asyncio.run(processor.process_job(processing_job))
    return {
        "ok": result.status in {JobStatus.complete, JobStatus.partial},
        "message": "Direct processing finished.",
        "video_id": result.video_id,
        "job_id": result.job_id,
        "status": result.status.value,
        "counts": result.counts.model_dump(),
        "error": result.error,
    }


def status(service: ApplicationService, video_id: str) -> dict[str, Any]:
    result = service.get_processing_status(video_id)
    return {
        "ok": result.ok,
        "message": result.message,
        "video_id": result.video_id,
        "status": result.status.value,
        "progress_percent": result.progress_percent,
        "job_id": result.job_id,
    }


def worker(queue: ProcessingQueue, *, drain: bool = False) -> dict[str, Any]:
    async def run() -> dict[str, Any]:
        if drain:
            await queue.run_until_idle()
            status = queue.get_status()
            return {
                "ok": True,
                "message": "Queue drained.",
                "queued_jobs": status.queued_jobs,
                "completed_jobs": status.completed_jobs,
            }
        await queue.start_workers()
        try:
            while True:
                await asyncio.sleep(3600)
        except (KeyboardInterrupt, asyncio.CancelledError):
            await queue.shutdown(timeout_seconds=10.0)
            status = queue.get_status()
            return {
                "ok": True,
                "message": "Queue workers stopped.",
                "workers": status.workers,
                "queued_jobs": status.queued_jobs,
            }

    return asyncio.run(run())


def search(
    vector_index: VideoVectorIndex,
    query: str,
    *,
    video_id: str | None = None,
    top_k: int = 5,
) -> dict[str, Any]:
    hits = vector_index.query(query, video_id=video_id, top_k=top_k)
    return {
        "ok": True,
        "query": query,
        "hits": [_hit_to_dict(hit) for hit in hits],
    }


def ask(qa_agent: QAAgent, video_id: str, question: str) -> dict[str, Any]:
    response = qa_agent.answer(video_id=video_id, question=question)
    return {
        "ok": True,
        "video_id": response.video_id,
        "question": response.question,
        "answer": response.answer,
        "evidence_ids": response.evidence_ids,
    }


def _processing_job_from_record(record):
    from video_qa.services.processing_queue import ProcessingJob

    return ProcessingJob(
        priority=int(record.priority),
        sequence=0,
        job_id=record.job_id,
        video_id=record.video_id,
        run_id=record.run_id,
        source_path=record.source_path,
    )


def _hit_to_dict(hit) -> dict[str, Any]:
    return {
        "id": hit.id,
        "modality": hit.modality,
        "score": hit.score,
        "text": hit.text,
        "source": hit.source.model_dump(mode="json"),
        "metadata": hit.metadata,
    }


def _jsonable(value: Any) -> Any:
    if is_dataclass(value):
        return _jsonable(asdict(value))
    if isinstance(value, dict):
        return {key: _jsonable(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_jsonable(item) for item in value]
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    if isinstance(value, Path):
        return str(value)
    return value


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
