"""Run report generation for processed videos."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from video_qa.models.media import RunPaths
from video_qa.models.tools import ContextType, VideoContextRecord
from video_qa.services.video_context import VideoContextRepository
from video_qa.services.video_processing_service import VideoProcessingService


@dataclass(frozen=True)
class ReportResult:
    report_json: Path
    detections_csv: Path
    summary_markdown: Path
    completeness: dict[str, Any]


class ReportWriter:
    """Write JSON, CSV, and Markdown outputs from persisted context."""

    def __init__(
        self,
        *,
        context_repository: VideoContextRepository,
        processing_service: VideoProcessingService,
    ) -> None:
        self.context_repository = context_repository
        self.processing_service = processing_service

    def write(self, *, video_id: str, run_paths: RunPaths) -> ReportResult:
        run_paths.reports_dir.mkdir(parents=True, exist_ok=True)
        contexts = self.context_repository.list_by_video(video_id)
        summary = self._summary(contexts)

        self._write_detections_csv(run_paths.detections_csv, contexts)
        run_paths.summary_markdown.write_text("", encoding="utf-8")
        run_paths.report_json.write_text("{}", encoding="utf-8")
        completeness = self.processing_service.verify_video_data_completeness(
            video_id,
            run_reports_dir=run_paths.reports_dir,
        )
        report = {
            "video_id": video_id,
            "run_id": run_paths.run_id,
            "counts": completeness["counts"],
            "missing": completeness["missing"],
            "summary": summary,
            "contexts": [
                context.model_dump(mode="json")
                for context in contexts
            ],
        }
        run_paths.report_json.write_text(
            json.dumps(report, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        run_paths.summary_markdown.write_text(
            self._summary_markdown(video_id, run_paths.run_id, summary, completeness),
            encoding="utf-8",
        )
        return ReportResult(
            report_json=run_paths.report_json,
            detections_csv=run_paths.detections_csv,
            summary_markdown=run_paths.summary_markdown,
            completeness=completeness,
        )

    def _summary(self, contexts: list[VideoContextRecord]) -> dict[str, Any]:
        counts: dict[str, int] = {context_type.value: 0 for context_type in ContextType}
        labels: dict[str, int] = {}
        caption_samples: list[str] = []
        transcript_samples: list[str] = []
        for context in contexts:
            counts[context.context_type.value] += 1
            if context.context_type in {ContextType.object, ContextType.crop}:
                label = str(context.data.get("label", "")).strip()
                if label:
                    labels[label] = labels.get(label, 0) + 1
            if context.context_type == ContextType.caption and len(caption_samples) < 5:
                caption_samples.append(str(context.data.get("text", "")))
            if context.context_type == ContextType.transcript and len(transcript_samples) < 5:
                transcript_samples.append(str(context.data.get("text", "")))
        return {
            "counts": counts,
            "top_labels": sorted(labels.items(), key=lambda item: (-item[1], item[0]))[:10],
            "caption_samples": [item for item in caption_samples if item],
            "transcript_samples": [item for item in transcript_samples if item],
        }

    def _write_detections_csv(self, path: Path, contexts: list[VideoContextRecord]) -> None:
        with path.open("w", newline="", encoding="utf-8") as output:
            writer = csv.DictWriter(
                output,
                fieldnames=[
                    "context_id",
                    "timestamp_sec",
                    "label",
                    "confidence",
                    "frame_id",
                    "bbox",
                    "crop_path",
                    "annotated_frame_path",
                ],
            )
            writer.writeheader()
            for context in contexts:
                if context.context_type != ContextType.object:
                    continue
                writer.writerow(
                    {
                        "context_id": context.context_id,
                        "timestamp_sec": context.timestamp_sec,
                        "label": context.data.get("label"),
                        "confidence": context.data.get("confidence"),
                        "frame_id": context.data.get("frame_id"),
                        "bbox": json.dumps(context.data.get("bbox", {}), sort_keys=True),
                        "crop_path": context.data.get("crop_path"),
                        "annotated_frame_path": context.data.get("annotated_frame_path"),
                    }
                )

    def _summary_markdown(
        self,
        video_id: str,
        run_id: str,
        summary: dict[str, Any],
        completeness: dict[str, Any],
    ) -> str:
        lines = [
            f"# Vidra Report: {video_id}",
            "",
            f"- Run: `{run_id}`",
            f"- Complete: `{completeness['complete']}`",
            f"- Missing: {', '.join(completeness['missing']) or 'none'}",
            "",
            "## Counts",
        ]
        for key, value in summary["counts"].items():
            lines.append(f"- {key}: {value}")
        if summary["top_labels"]:
            lines.extend(["", "## Top Labels"])
            for label, count in summary["top_labels"]:
                lines.append(f"- {label}: {count}")
        if summary["caption_samples"]:
            lines.extend(["", "## Caption Samples"])
            lines.extend(f"- {caption}" for caption in summary["caption_samples"])
        if summary["transcript_samples"]:
            lines.extend(["", "## Transcript Samples"])
            lines.extend(f"- {text}" for text in summary["transcript_samples"])
        return "\n".join(lines) + "\n"
