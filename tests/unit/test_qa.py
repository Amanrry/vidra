from __future__ import annotations

from pathlib import Path

import pytest

from video_qa.models.tools import ContextType, VideoContextRecord
from video_qa.services.context_builder import ContextBuilder
from video_qa.services.qa_agent import QAAgent, QAError
from video_qa.services.video_context import VideoContextRepository
from video_qa.storage import Database


class FakeChatClient:
    def __init__(self, answer: str = "At 1.00s, a person is near a car.") -> None:
        self.answer = answer
        self.messages: list[dict[str, str]] | None = None

    def complete(self, *, messages: list[dict[str, str]]) -> str:
        self.messages = messages
        return self.answer


def insert_video(database: Database, video_id: str = "video-1") -> None:
    database.execute(
        """
        INSERT INTO videos (video_id, filename, file_path, duration_sec)
        VALUES (?, ?, ?, ?)
        """,
        (video_id, f"{video_id}.mp4", f"data/input/{video_id}.mp4", 0.0),
    )


def make_agent(tmp_path: Path, client: FakeChatClient) -> QAAgent:
    database = Database(tmp_path / "vidra.sqlite3")
    database.initialize()
    insert_video(database)
    repository = VideoContextRepository(database)
    repository.upsert_contexts(
        [
            VideoContextRecord(
                context_id="caption-1",
                video_id="video-1",
                context_type=ContextType.caption,
                timestamp_sec=1.0,
                data={"text": "a person is near a car"},
                tool_name="caption_frames",
                model_name="fake-captioner",
            )
        ]
    )
    return QAAgent(
        context_builder=ContextBuilder(context_repository=repository),
        chat_client=client,
    )


def test_qa_agent_builds_grounded_prompt_and_calls_openai_compatible_client(
    tmp_path: Path,
) -> None:
    client = FakeChatClient()
    agent = make_agent(tmp_path, client)

    response = agent.answer(video_id="video-1", question="What is visible?")

    assert response.answer == "At 1.00s, a person is near a car."
    assert response.evidence_ids == ["caption-1"]
    assert "Use only the evidence below" in response.prompt
    assert "Do not invent" in response.prompt
    assert client.messages is not None
    assert client.messages[0]["role"] == "system"
    assert client.messages[1]["role"] == "user"
    assert "Question: What is visible?" in client.messages[1]["content"]


def test_qa_agent_rejects_empty_llm_answer(tmp_path: Path) -> None:
    agent = make_agent(tmp_path, FakeChatClient(answer=""))

    with pytest.raises(QAError, match="empty"):
        agent.answer(video_id="video-1", question="What is visible?")
