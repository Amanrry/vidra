"""Build source-grounded context for video QA prompts."""

from __future__ import annotations

import string
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

from video_qa.models.qa import RetrievalHit
from video_qa.models.tools import ContextType, VideoContextRecord
from video_qa.services.video_context import VideoContextRepository


@runtime_checkable
class RetrievalPort(Protocol):
    def query(
        self,
        query_text: str,
        *,
        video_id: str | None = None,
        top_k: int = 5,
    ) -> list[RetrievalHit]:
        """Return ranked retrieval hits for a question."""


@dataclass(frozen=True)
class PromptEvidence:
    context_id: str
    context_type: ContextType
    timestamp_sec: float | None
    text: str
    score: float | None = None
    label: str | None = None
    frame_path: str | None = None
    crop_path: str | None = None
    source: str = "context"


@dataclass(frozen=True)
class BuiltContext:
    video_id: str
    question: str
    evidence: list[PromptEvidence] = field(default_factory=list)

    @property
    def has_evidence(self) -> bool:
        return bool(self.evidence)


class ContextBuilder:
    """Prioritize captions, transcripts, objects, and crops for QA."""

    priority = {
        ContextType.caption: 0,
        ContextType.transcript: 1,
        ContextType.object: 2,
        ContextType.crop: 3,
        ContextType.frame: 4,
        ContextType.metadata: 5,
    }

    def __init__(
        self,
        *,
        context_repository: VideoContextRepository,
        retriever: RetrievalPort | None = None,
        top_k: int = 8,
        fallback_per_type: int = 4,
    ) -> None:
        self.context_repository = context_repository
        self.retriever = retriever
        self.top_k = top_k
        self.fallback_per_type = fallback_per_type

    def build_rich_context_description(self, video_id: str, max_items: int = 10) -> str:
        """Build a product-facing summary from whatever processed context exists."""

        contexts = self.context_repository.list_by_video(video_id)
        if not contexts:
            return "No processed data available for this video yet."

        captions = [item for item in contexts if item.context_type == ContextType.caption]
        transcripts = [item for item in contexts if item.context_type == ContextType.transcript]
        objects = [item for item in contexts if item.context_type == ContextType.object]
        frames = [item for item in contexts if item.context_type == ContextType.frame]
        metadata = [item for item in contexts if item.context_type == ContextType.metadata]

        parts: list[str] = []
        if metadata:
            data = metadata[-1].data
            duration = data.get("duration_sec")
            size = ""
            if data.get("width") and data.get("height"):
                size = f" ({data['width']}x{data['height']})"
            if duration is not None:
                parts.append(f"Video duration: {float(duration):.1f}s{size}")
        if captions:
            parts.append(f"\nVisual Analysis ({len(captions)} scenes):")
            for context in captions[:max_items]:
                timestamp = self._format_timestamp(context.timestamp_sec)
                parts.append(f"  [{timestamp}] {context.data.get('text', '')}")
        if transcripts:
            parts.append(f"\nAudio Transcript ({len(transcripts)} segments):")
            for context in transcripts[:max_items]:
                timestamp = self._format_timestamp(context.timestamp_sec)
                parts.append(f"  [{timestamp}] {context.data.get('text', '')}")
        if objects:
            parts.append(f"\nDetected Objects ({len(objects)} detections):")
            for label, timestamps in list(self._object_summary(objects).items())[:max_items]:
                sample = ", ".join(
                    self._format_timestamp(timestamp)
                    for timestamp in timestamps[:3]
                )
                suffix = f" (+{len(timestamps) - 3} more)" if len(timestamps) > 3 else ""
                parts.append(f"  {label}: {sample}{suffix}")
        if frames and not captions:
            timestamps = ", ".join(
                self._format_timestamp(item.timestamp_sec)
                for item in frames[:max_items]
            )
            parts.append(f"\nExtracted Frames ({len(frames)} frames): {timestamps}")
        return "\n".join(parts).strip() or "No processed data available for this video yet."

    def search_captions_keyword(
        self,
        video_id: str,
        query: str,
        *,
        top_k: int = 5,
        use_semantic: bool = True,
    ) -> list[PromptEvidence]:
        return self._keyword_search(
            video_id,
            query,
            context_type=ContextType.caption,
            top_k=top_k,
            use_semantic=use_semantic,
        )

    def search_transcripts_keyword(
        self,
        video_id: str,
        query: str,
        *,
        top_k: int = 5,
        use_semantic: bool = True,
    ) -> list[PromptEvidence]:
        return self._keyword_search(
            video_id,
            query,
            context_type=ContextType.transcript,
            top_k=top_k,
            use_semantic=use_semantic,
        )

    def get_context_at_timestamp(
        self,
        video_id: str,
        timestamp_sec: float,
        *,
        window_sec: float = 5.0,
    ) -> list[PromptEvidence]:
        if timestamp_sec < 0:
            raise ValueError("timestamp_sec must be non-negative")
        if window_sec < 0:
            raise ValueError("window_sec must be non-negative")
        start = max(0.0, timestamp_sec - window_sec)
        end = timestamp_sec + window_sec
        evidence = []
        for context in self.context_repository.list_by_video(video_id):
            if context.timestamp_sec is None:
                continue
            if start <= context.timestamp_sec <= end:
                item = self._evidence_from_context(context)
                if item is not None:
                    evidence.append(item)
        return sorted(
            evidence,
            key=lambda item: (
                abs((item.timestamp_sec or 0.0) - timestamp_sec),
                self.priority.get(item.context_type, 99),
            ),
        )

    def get_objects(self, video_id: str, label_query: str) -> list[PromptEvidence]:
        clean = label_query.strip().lower()
        if not clean:
            raise ValueError("label_query is required")
        matches = []
        for context in self.context_repository.list_by_video(
            video_id,
            context_type=ContextType.object,
        ):
            label = str(context.data.get("label", "")).lower()
            if clean in label:
                item = self._evidence_from_context(context)
                if item is not None:
                    matches.append(item)
        return matches

    def get_frames_with_object(self, video_id: str, label_query: str) -> list[PromptEvidence]:
        object_hits = self.get_objects(video_id, label_query)
        frame_ids = {
            str(context.data.get("frame_id"))
            for context in self.context_repository.list_by_video(
                video_id,
                context_type=ContextType.object,
            )
            if any(hit.context_id == context.context_id for hit in object_hits)
        }
        frames = []
        for context in self.context_repository.list_by_video(
            video_id,
            context_type=ContextType.frame,
        ):
            if str(context.data.get("frame_id")) in frame_ids:
                item = self._evidence_from_context(context)
                if item is not None:
                    frames.append(item)
        return frames

    def build(self, *, video_id: str, question: str) -> BuiltContext:
        clean_question = question.strip()
        if not clean_question:
            raise ValueError("question is required")

        evidence: list[PromptEvidence] = []
        seen: set[str] = set()

        if self.retriever is not None:
            for hit in self.retriever.query(clean_question, video_id=video_id, top_k=self.top_k):
                item = self._evidence_from_hit(hit)
                if item is not None and item.context_id not in seen:
                    evidence.append(item)
                    seen.add(item.context_id)

        for context_type in [
            ContextType.caption,
            ContextType.transcript,
            ContextType.object,
            ContextType.crop,
        ]:
            contexts = self.context_repository.list_by_video(video_id, context_type=context_type)
            for context in contexts[: self.fallback_per_type]:
                item = self._evidence_from_context(context)
                if item is not None and item.context_id not in seen:
                    evidence.append(item)
                    seen.add(item.context_id)

        evidence = self._select_balanced_evidence(evidence)
        return BuiltContext(
            video_id=video_id,
            question=clean_question,
            evidence=evidence,
        )

    def _select_balanced_evidence(self, evidence: list[PromptEvidence]) -> list[PromptEvidence]:
        evidence.sort(key=self._evidence_sort_key)
        if len(evidence) <= self.top_k:
            return evidence

        minimum_by_type = {
            ContextType.caption: 2,
            ContextType.transcript: 1,
            ContextType.object: min(4, self.fallback_per_type),
            ContextType.crop: 1,
        }
        selected: list[PromptEvidence] = []
        seen: set[str] = set()
        for context_type, minimum in minimum_by_type.items():
            typed_candidates = [
                candidate
                for candidate in evidence
                if candidate.context_type == context_type
            ]
            for item in typed_candidates[:minimum]:
                if item.context_id not in seen and len(selected) < self.top_k:
                    selected.append(item)
                    seen.add(item.context_id)

        remaining = sorted(
            (item for item in evidence if item.context_id not in seen),
            key=lambda item: (-(item.score or 0.0), self._evidence_sort_key(item)),
        )
        for item in remaining:
            if len(selected) >= self.top_k:
                break
            selected.append(item)
            seen.add(item.context_id)

        selected.sort(key=self._evidence_sort_key)
        return selected

    def _evidence_sort_key(self, item: PromptEvidence) -> tuple[int, bool, float, float]:
        return (
            self.priority.get(item.context_type, 99),
            item.timestamp_sec is None,
            item.timestamp_sec or 0.0,
            -(item.score or 0.0),
        )

    def build_prompt(self, context: BuiltContext) -> str:
        lines = [
            "You answer questions about one uploaded video.",
            "Use only the evidence below. Do not invent people, objects, speech, "
            "actions, locations, or timestamps that are not present in the evidence.",
            "If the evidence is insufficient, say that the video context does not "
            "show enough information.",
            "Cite timestamps or labels when they are available.",
            "",
            f"Video ID: {context.video_id}",
            "Evidence, ordered by reliability for open-domain QA:",
        ]
        if not context.evidence:
            lines.append("- No processed evidence is available yet.")
        else:
            for index, item in enumerate(context.evidence, start=1):
                lines.append(f"{index}. {self._format_evidence(item)}")

        lines.extend(["", f"Question: {context.question}", "Answer:"])
        return "\n".join(lines)

    def _evidence_from_hit(self, hit: RetrievalHit) -> PromptEvidence | None:
        try:
            context_type = ContextType(hit.source.context_type)
        except ValueError:
            return None
        text = hit.text or hit.source.label or ""
        if not text:
            return None
        return PromptEvidence(
            context_id=hit.source.context_id,
            context_type=context_type,
            timestamp_sec=hit.source.timestamp_sec,
            text=text,
            score=hit.score,
            label=hit.source.label,
            frame_path=str(hit.source.frame_path) if hit.source.frame_path else None,
            crop_path=str(hit.source.crop_path) if hit.source.crop_path else None,
            source="retrieval",
        )

    def _evidence_from_context(self, context: VideoContextRecord) -> PromptEvidence | None:
        data = context.data
        text = ""
        label = data.get("label")
        if context.context_type in {ContextType.caption, ContextType.transcript}:
            text = str(data.get("text", "")).strip()
        elif context.context_type == ContextType.object:
            text = (
                f"Detected {data.get('label', 'object')} "
                f"with confidence {data.get('confidence', 'unknown')}"
            )
        elif context.context_type == ContextType.crop:
            text = f"Crop showing {data.get('label', 'object')}"
        elif context.context_type == ContextType.frame:
            text = f"Sampled frame {data.get('frame_id', context.context_id)}"
        if not text:
            return None
        return PromptEvidence(
            context_id=context.context_id,
            context_type=context.context_type,
            timestamp_sec=context.timestamp_sec,
            text=text,
            label=str(label) if label else None,
            frame_path=data.get("frame_path") or data.get("image_path"),
            crop_path=data.get("crop_path"),
            source="stored_context",
        )

    def _format_evidence(self, item: PromptEvidence) -> str:
        parts = [f"[{item.context_type.value}]"]
        if item.timestamp_sec is not None:
            parts.append(f"t={item.timestamp_sec:.2f}s")
        if item.label:
            parts.append(f"label={item.label}")
        if item.score is not None:
            parts.append(f"score={item.score:.3f}")
        parts.append(item.text)
        if item.crop_path:
            parts.append(f"crop={item.crop_path}")
        if item.frame_path:
            parts.append(f"frame={item.frame_path}")
        return " | ".join(parts)

    def _keyword_search(
        self,
        video_id: str,
        query: str,
        *,
        context_type: ContextType,
        top_k: int,
        use_semantic: bool,
    ) -> list[PromptEvidence]:
        clean = query.strip()
        if not clean:
            raise ValueError("query is required")
        scored: list[tuple[float, PromptEvidence]] = []
        for context in self.context_repository.list_by_video(video_id, context_type=context_type):
            item = self._evidence_from_context(context)
            if item is None:
                continue
            score = self._keyword_score(clean, item.text)
            if score > 0:
                scored.append((score, item))

        scored.sort(key=lambda pair: (-pair[0], pair[1].timestamp_sec or 0.0))
        results = [
            PromptEvidence(
                context_id=item.context_id,
                context_type=item.context_type,
                timestamp_sec=item.timestamp_sec,
                text=item.text,
                score=score / 100.0,
                label=item.label,
                frame_path=item.frame_path,
                crop_path=item.crop_path,
                source="keyword",
            )
            for score, item in scored[:top_k]
        ]
        if use_semantic and self.retriever is not None:
            seen = {item.context_id for item in results}
            for hit in self.retriever.query(clean, video_id=video_id, top_k=top_k):
                item = self._evidence_from_hit(hit)
                if item is None or item.context_type != context_type or item.context_id in seen:
                    continue
                results.append(item)
                seen.add(item.context_id)
                if len(results) >= top_k:
                    break
        return results[:top_k]

    def _keyword_score(self, query: str, text: str) -> float:
        query_lower = query.lower()
        text_lower = text.lower()
        score = 0.0
        if query_lower in text_lower:
            score += 100.0
        query_terms = self._tokenize_and_stem(query_lower)
        text_terms = self._tokenize_and_stem(text_lower)
        if query_terms:
            score += (len(query_terms & text_terms) / len(query_terms)) * 50.0
        score += self._synonym_score(query_lower, text_lower)
        return score

    def _tokenize_and_stem(self, text: str) -> set[str]:
        cleaned = text.translate(str.maketrans("", "", string.punctuation))
        terms = set()
        for word in cleaned.lower().split():
            if len(word) <= 2:
                terms.add(word)
            elif word.endswith("ing"):
                terms.add(word[:-3])
            elif word.endswith("ed"):
                terms.add(word[:-2])
            elif word.endswith("s") and not word.endswith("ss"):
                terms.add(word[:-1])
            else:
                terms.add(word)
        return terms

    def _synonym_score(self, query: str, text: str) -> float:
        synonyms = {
            "person": ["man", "woman", "people", "human"],
            "car": ["vehicle", "automobile", "truck", "van"],
            "walk": ["walking", "stroll", "move"],
            "talk": ["speaking", "speech", "conversation"],
        }
        score = 0.0
        for term, related in synonyms.items():
            if term in query and any(word in text for word in related):
                score += 10.0
            if any(word in query for word in related) and term in text:
                score += 10.0
        return score

    def _object_summary(self, objects: list[VideoContextRecord]) -> dict[str, list[float]]:
        summary: dict[str, list[float]] = {}
        for context in objects:
            label = str(context.data.get("label", "object"))
            summary.setdefault(label, []).append(float(context.timestamp_sec or 0.0))
        return summary

    def _format_timestamp(self, seconds: float | None) -> str:
        if seconds is None:
            return "--:--"
        total = int(seconds)
        hours = total // 3600
        minutes = (total % 3600) // 60
        secs = total % 60
        if hours:
            return f"{hours}:{minutes:02d}:{secs:02d}"
        return f"{minutes}:{secs:02d}"
