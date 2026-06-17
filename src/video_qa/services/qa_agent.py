"""Source-grounded QA agent over retrieved video evidence."""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from video_qa.config import LLMSettings
from video_qa.services.context_builder import BuiltContext, ContextBuilder


class QAError(RuntimeError):
    """Raised when the QA agent cannot produce an answer."""


@dataclass(frozen=True)
class QAResponse:
    video_id: str
    question: str
    answer: str
    prompt: str
    evidence_ids: list[str]


@runtime_checkable
class ChatCompletionClient(Protocol):
    def complete(self, *, messages: list[dict[str, str]]) -> str:
        """Return assistant text for OpenAI-style chat messages."""


class OpenAICompatibleClient:
    """Minimal OpenAI-compatible chat completions HTTP client."""

    def __init__(self, settings: LLMSettings) -> None:
        self.settings = settings

    def complete(self, *, messages: list[dict[str, str]]) -> str:
        payload = {
            "model": self.settings.model,
            "messages": messages,
            "temperature": self.settings.temperature,
            "max_tokens": self.settings.max_tokens,
        }
        request = urllib.request.Request(
            f"{self.settings.base_url}/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers=self._headers(),
            method="POST",
        )
        try:
            with urllib.request.urlopen(
                request,
                timeout=self.settings.timeout_seconds,
            ) as response:
                data = json.loads(response.read().decode("utf-8"))
        except urllib.error.URLError as exc:  # pragma: no cover - network boundary
            raise QAError(f"LLM request failed: {exc}") from exc
        try:
            return str(data["choices"][0]["message"]["content"]).strip()
        except (KeyError, IndexError, TypeError) as exc:
            raise QAError("LLM response did not match OpenAI chat completion format") from exc

    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.settings.api_key:
            headers["Authorization"] = f"Bearer {self.settings.api_key}"
        return headers


class QAAgent:
    """Build a grounded prompt and ask an OpenAI-compatible LLM."""

    system_prompt = (
        "You are Vidra's video QA assistant. Answer only from provided video evidence. "
        "When evidence is weak or missing, say so plainly."
    )

    def __init__(
        self,
        *,
        context_builder: ContextBuilder,
        chat_client: ChatCompletionClient,
    ) -> None:
        self.context_builder = context_builder
        self.chat_client = chat_client

    def answer(self, *, video_id: str, question: str) -> QAResponse:
        built_context = self.context_builder.build(video_id=video_id, question=question)
        prompt = self.context_builder.build_prompt(built_context)
        answer = self.chat_client.complete(
            messages=[
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": prompt},
            ]
        )
        if not answer:
            raise QAError("LLM returned an empty answer")
        return QAResponse(
            video_id=video_id,
            question=built_context.question,
            answer=answer,
            prompt=prompt,
            evidence_ids=[evidence.context_id for evidence in built_context.evidence],
        )
