"""Transactional persistence service for video processing tool results."""

from __future__ import annotations

import json
import time
import uuid
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from video_qa.models.processing import ProcessingCounts
from video_qa.models.tools import ContextType, VideoContextRecord
from video_qa.storage import Database


class VideoProcessingServiceError(RuntimeError):
    """Raised when processing results cannot be durably stored."""


class VideoProcessingService:
    """Bri-style persistence boundary for generated video intelligence."""

    def __init__(
        self,
        database: Database,
        *,
        max_retries: int = 3,
        retry_seconds: float = 0.1,
    ) -> None:
        if max_retries < 1:
            raise ValueError("max_retries must be positive")
        self.database = database
        self.database.initialize()
        self.max_retries = max_retries
        self.retry_seconds = retry_seconds

    def store_tool_results(
        self,
        *,
        video_id: str,
        tool_name: str,
        records: Iterable[VideoContextRecord],
        idempotency_key: str,
        run_id: str,
        model_name: str | None = None,
        parameters: Mapping[str, Any] | None = None,
    ) -> ProcessingCounts:
        """Validate and persist context records atomically with idempotency."""

        clean_video_id = video_id.strip()
        clean_tool = tool_name.strip()
        clean_key = idempotency_key.strip()
        clean_run_id = run_id.strip()
        if not clean_video_id or not clean_tool or not clean_key or not clean_run_id:
            raise ValueError("video_id, tool_name, idempotency_key, and run_id are required")

        existing = self._get_idempotency_counts(clean_video_id, clean_tool, clean_key)
        if existing is not None:
            return existing

        context_records = self._validate_records(clean_video_id, records)
        counts = self._counts_for_records(context_records)
        params_json = json.dumps(dict(parameters or {}), ensure_ascii=False, sort_keys=True)
        counts_json = counts.model_dump_json()

        for attempt in range(1, self.max_retries + 1):
            try:
                with self.database.transaction() as connection:
                    connection.executemany(
                        """
                        INSERT INTO video_context
                        (context_id, video_id, context_type, timestamp_sec,
                         data, tool_name, model_name)
                        VALUES (?, ?, ?, ?, ?, ?, ?)
                        ON CONFLICT(context_id) DO UPDATE SET
                            video_id = excluded.video_id,
                            context_type = excluded.context_type,
                            timestamp_sec = excluded.timestamp_sec,
                            data = excluded.data,
                            tool_name = excluded.tool_name,
                            model_name = excluded.model_name
                        """,
                        [
                            (
                                context.context_id,
                                context.video_id,
                                context.context_type.value,
                                context.timestamp_sec,
                                json.dumps(
                                    context.data,
                                    ensure_ascii=False,
                                    sort_keys=True,
                                ),
                                context.tool_name,
                                context.model_name,
                            )
                            for context in context_records
                        ],
                    )
                    connection.executemany(
                        """
                        INSERT INTO lineage
                        (lineage_id, video_id, context_id, operation,
                         tool_name, model_name, parameters)
                        VALUES (?, ?, ?, 'create', ?, ?, ?)
                        """,
                        [
                            (
                                str(uuid.uuid4()),
                                clean_video_id,
                                context.context_id,
                                clean_tool,
                                context.model_name or model_name,
                                params_json,
                            )
                            for context in context_records
                        ],
                    )
                    connection.execute(
                        """
                        INSERT INTO processing_idempotency
                        (video_id, tool_name, idempotency_key, run_id, counts)
                        VALUES (?, ?, ?, ?, ?)
                        """,
                        (clean_video_id, clean_tool, clean_key, clean_run_id, counts_json),
                    )
                return counts
            except Exception as exc:
                if attempt >= self.max_retries:
                    raise VideoProcessingServiceError(
                        f"Failed to store {clean_tool} results for {clean_video_id}: {exc}"
                    ) from exc
                time.sleep(self.retry_seconds * attempt)

        return counts

    def verify_video_data_completeness(
        self,
        video_id: str,
        *,
        run_reports_dir: str | Path | None = None,
    ) -> dict[str, Any]:
        """Return counts and missing context/report categories for one video."""

        clean_video_id = video_id.strip()
        if not clean_video_id:
            raise ValueError("video_id is required")

        counts = {
            context_type.value: self._count_context(clean_video_id, context_type)
            for context_type in [
                ContextType.frame,
                ContextType.caption,
                ContextType.transcript,
                ContextType.object,
                ContextType.crop,
                ContextType.metadata,
            ]
        }
        reports = self._report_status(run_reports_dir)
        missing = [
            name
            for name in ["frame", "caption"]
            if counts[name] == 0
        ]
        if run_reports_dir is not None:
            missing.extend(name for name, present in reports.items() if not present)

        return {
            "video_id": clean_video_id,
            "complete": not missing,
            "counts": counts,
            "reports": reports,
            "missing": missing,
        }

    def _validate_records(
        self,
        video_id: str,
        records: Iterable[VideoContextRecord],
    ) -> list[VideoContextRecord]:
        validated: list[VideoContextRecord] = []
        for record in records:
            try:
                context = VideoContextRecord.model_validate(record)
            except ValidationError as exc:
                raise VideoProcessingServiceError(f"Invalid context record: {exc}") from exc
            if context.video_id != video_id:
                raise VideoProcessingServiceError(
                    f"Context {context.context_id} belongs to {context.video_id}, not {video_id}"
                )
            validated.append(context)
        return validated

    def _get_idempotency_counts(
        self,
        video_id: str,
        tool_name: str,
        idempotency_key: str,
    ) -> ProcessingCounts | None:
        rows = self.database.query(
            """
            SELECT counts
            FROM processing_idempotency
            WHERE video_id = ? AND tool_name = ? AND idempotency_key = ?
            """,
            (video_id, tool_name, idempotency_key),
        )
        if not rows:
            return None
        return ProcessingCounts.model_validate_json(str(rows[0]["counts"]))

    def _counts_for_records(self, records: list[VideoContextRecord]) -> ProcessingCounts:
        counts = ProcessingCounts()
        for record in records:
            if record.context_type == ContextType.frame:
                counts.frames_extracted += 1
            elif record.context_type == ContextType.caption:
                counts.captions_generated += 1
            elif record.context_type == ContextType.transcript:
                counts.transcript_segments += 1
            elif record.context_type == ContextType.object:
                counts.detections_created += 1
            elif record.context_type == ContextType.crop:
                counts.crops_created += 1
        return counts

    def _count_context(self, video_id: str, context_type: ContextType) -> int:
        rows = self.database.query(
            """
            SELECT COUNT(*) AS count
            FROM video_context
            WHERE video_id = ? AND context_type = ?
            """,
            (video_id, context_type.value),
        )
        return int(rows[0]["count"])

    def _report_status(self, reports_dir: str | Path | None) -> dict[str, bool]:
        if reports_dir is None:
            return {}
        root = Path(reports_dir)
        return {
            "report_json": (root / "report.json").is_file(),
            "detections_csv": (root / "detections.csv").is_file(),
            "summary_markdown": (root / "summary.md").is_file(),
        }
