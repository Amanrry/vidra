"""Background event-loop runner for in-process processing queues."""

from __future__ import annotations

import asyncio
import threading
from dataclasses import dataclass

from video_qa.services.processing_queue import ProcessingQueue


@dataclass
class QueueWorkerRuntime:
    """Run queue workers on a dedicated loop for synchronous UIs."""

    queue: ProcessingQueue
    loop: asyncio.AbstractEventLoop | None = None
    thread: threading.Thread | None = None

    def start(self) -> None:
        if self.thread is not None and self.thread.is_alive():
            return
        ready = threading.Event()

        def run_loop() -> None:
            self.loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self.loop)
            self.loop.run_until_complete(self.queue.start_workers())
            ready.set()
            self.loop.run_forever()
            self.loop.run_until_complete(self.queue.shutdown(timeout_seconds=5.0))
            self.loop.close()

        self.thread = threading.Thread(target=run_loop, name="vidra-queue-workers", daemon=True)
        self.thread.start()
        ready.wait(timeout=5.0)

    def stop(self) -> None:
        if self.loop is None:
            return

        self.loop.call_soon_threadsafe(self.loop.stop)
        if self.thread is not None:
            self.thread.join(timeout=5.0)
