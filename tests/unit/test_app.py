from __future__ import annotations

from video_qa.app import create_default_handlers


class FakeQueue:
    pass


class FakeQueueRuntime:
    def __init__(self, queue) -> None:
        self.queue = queue
        self.started = False
        self.stopped = False

    def start(self) -> None:
        self.started = True

    def stop(self) -> None:
        self.stopped = True


class FakeApplicationService:
    pass


class FakeRetriever:
    pass


def test_create_default_handlers_starts_queue_runtime_from_runtime() -> None:
    runtime = {
        "queue": FakeQueue(),
        "app_service": FakeApplicationService(),
        "vector_index": FakeRetriever(),
    }

    handlers = create_default_handlers(
        runtime=runtime,
        queue_runtime_factory=FakeQueueRuntime,
    )

    assert handlers.application_service is runtime["app_service"]
    assert handlers.retriever is runtime["vector_index"]
    assert isinstance(handlers.queue_runtime, FakeQueueRuntime)
    assert handlers.queue_runtime.started
