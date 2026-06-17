"""Gradio application composition for Vidra."""

from __future__ import annotations

import argparse
import atexit
import inspect
from collections.abc import Callable
from typing import Any

from video_qa.cli import create_runtime
from video_qa.config import load_settings
from video_qa.runtime import ensure_media_binaries_on_path, has_ffprobe
from video_qa.services.queue_runtime import QueueWorkerRuntime
from video_qa.ui.handlers import UIHandlers

APP_CSS = """
:root {
  --vidra-bg: #f6f7f9;
  --vidra-panel: #ffffff;
  --vidra-border: #dfe3ea;
  --vidra-text: #17181c;
  --vidra-muted: #626b79;
  --vidra-accent: #145c4a;
}
.gradio-container {
  background: var(--vidra-bg);
  color: var(--vidra-text);
}
#vidra-shell {
  max-width: 1440px;
  margin: 0 auto;
}
#vidra-header {
  padding: 12px 4px 4px;
}
#vidra-header h1 {
  font-size: 24px;
  line-height: 1.2;
  margin: 0;
  letter-spacing: 0;
}
#vidra-header p {
  color: var(--vidra-muted);
  margin: 4px 0 0;
}
#results-panel,
#chat-panel {
  background: var(--vidra-panel);
  border: 1px solid var(--vidra-border);
  border-radius: 8px;
  padding: 12px;
}
#progress-line,
#upload-line,
#summary-line {
  font-family: ui-monospace, SFMono-Regular, Consolas, "Liberation Mono", monospace;
}
#chatbot {
  min-height: 560px;
}
#chat-panel .form {
  border: 0;
}
button.primary {
  background: var(--vidra-accent);
}
"""


def create_app(handlers: UIHandlers | None = None):
    ensure_media_binaries_on_path()
    can_render_video = has_ffprobe()
    try:
        import gradio as gr  # type: ignore[import-not-found]
    except Exception as exc:  # pragma: no cover - optional dependency
        raise RuntimeError(
            "Gradio is required to launch the UI. Install Vidra with .[ai]."
        ) from exc

    handlers = handlers or create_default_handlers()

    def inactive_timer_update() -> Any:
        return gr.update(active=False)

    def active_timer_update() -> Any:
        return gr.update(active=True)

    def upload_and_start(file_value: Any) -> tuple[str | None, str, None, None, Any]:
        video_id, message = handlers.upload_and_enqueue(file_value)
        timer_update = active_timer_update() if video_id else inactive_timer_update()
        return video_id, message, None, None, timer_update

    def poll_status_with_timer(
        video_id: str | None,
        previous_signature: str | None,
    ) -> tuple[str, str, Any]:
        snapshot = handlers.progress_snapshot(video_id, previous_signature)
        timer_update = (
            active_timer_update()
            if snapshot.should_continue_polling
            else inactive_timer_update()
        )
        return snapshot.text, snapshot.signature, timer_update

    with gr.Blocks(title="Vidra") as demo:
        gr.Markdown(
            "# Vidra\nOpen-domain video detection, retrieval, and grounded QA.",
            elem_id="vidra-header",
        )
        active_video_id = gr.State(value=None)
        progress_signature = gr.State(value=None)
        rendered_dashboard_signature = gr.State(value=None)
        poll_timer = gr.Timer(1.0, active=False)

        with gr.Row(elem_id="vidra-shell", equal_height=True):
            with gr.Column(scale=7, elem_id="results-panel"):
                with gr.Row():
                    video_file = gr.File(
                        label="Upload video",
                        file_types=["video"],
                        scale=4,
                    )
                    upload_button = gr.Button(
                        "Start processing",
                        variant="primary",
                        scale=1,
                    )

                upload_message = gr.Markdown(
                    "No video uploaded.",
                    elem_id="upload-line",
                )
                progress_text = gr.Markdown(
                    "No active video.",
                    elem_id="progress-line",
                )
                summary_text = gr.Markdown(
                    "Upload a video to see processing results.",
                    elem_id="summary-line",
                )

                with gr.Row(equal_height=True):
                    if can_render_video:
                        source_video = gr.Video(label="Source video", interactive=False)
                    else:
                        source_video = gr.File(
                            label="Source video",
                            interactive=False,
                            file_types=["video"],
                        )
                    detection_preview = gr.Image(
                        label="Latest YOLO visualization",
                        type="filepath",
                        interactive=False,
                    )

                with gr.Tab("Detections"):
                    object_results = gr.Dataframe(
                        headers=["time", "label", "confidence", "frame", "crop"],
                        datatype=["str", "str", "str", "str", "str"],
                        interactive=False,
                        wrap=True,
                    )
                with gr.Tab("Evidence"):
                    evidence_results = gr.Dataframe(
                        headers=["type", "time", "text", "media"],
                        datatype=["str", "str", "str", "str"],
                        interactive=False,
                        wrap=True,
                    )
                with gr.Tab("Search"):
                    with gr.Row():
                        search_query = gr.Textbox(
                            label="Search processed context",
                            placeholder="Search captions, transcripts, objects, or crops",
                            scale=5,
                        )
                        search_button = gr.Button("Search", scale=1)
                    search_results = gr.Dataframe(
                        headers=["type", "timestamp", "label", "score", "text", "media"],
                        interactive=False,
                        wrap=True,
                    )

            with gr.Column(scale=5, elem_id="chat-panel"):
                chatbot = gr.Chatbot(
                    label="AI chat",
                    elem_id="chatbot",
                    height=620,
                    layout="bubble",
                    buttons=["copy"],
                )
                question = gr.Textbox(
                    label="Message",
                    placeholder="Ask what happened in the video...",
                    lines=2,
                    max_lines=5,
                    show_label=False,
                )
                with gr.Row():
                    ask_button = gr.Button("Send", variant="primary")
                    clear_chat_button = gr.Button("Clear")

        upload_button.click(
            fn=upload_and_start,
            inputs=[video_file],
            outputs=[
                active_video_id,
                upload_message,
                progress_signature,
                rendered_dashboard_signature,
                poll_timer,
            ],
            show_progress="minimal",
        ).then(
            fn=poll_status_with_timer,
            inputs=[active_video_id, progress_signature],
            outputs=[progress_text, progress_signature, poll_timer],
            show_progress="hidden",
        ).then(
            fn=handlers.dashboard_if_changed,
            inputs=[active_video_id, progress_signature, rendered_dashboard_signature],
            outputs=[
                source_video,
                detection_preview,
                summary_text,
                object_results,
                evidence_results,
                rendered_dashboard_signature,
            ],
            show_progress="hidden",
        )
        poll_timer.tick(
            fn=poll_status_with_timer,
            inputs=[active_video_id, progress_signature],
            outputs=[progress_text, progress_signature, poll_timer],
            show_progress="hidden",
            queue=False,
        ).then(
            fn=handlers.dashboard_if_changed,
            inputs=[active_video_id, progress_signature, rendered_dashboard_signature],
            outputs=[
                source_video,
                detection_preview,
                summary_text,
                object_results,
                evidence_results,
                rendered_dashboard_signature,
            ],
            show_progress="hidden",
        )
        ask_button.click(
            fn=handlers.add_chat_turn,
            inputs=[active_video_id, question, chatbot],
            outputs=[chatbot, question],
            show_progress="minimal",
        )
        question.submit(
            fn=handlers.add_chat_turn,
            inputs=[active_video_id, question, chatbot],
            outputs=[chatbot, question],
            show_progress="minimal",
        )
        clear_chat_button.click(
            fn=lambda: [],
            inputs=None,
            outputs=[chatbot],
            show_progress="hidden",
        )
        search_button.click(
            fn=handlers.search,
            inputs=[active_video_id, search_query],
            outputs=[search_results],
            show_progress="minimal",
        )
    return demo


def launch_app(app: Any, *, server_name: str | None, server_port: int | None) -> None:
    launch_kwargs: dict[str, Any] = {
        "server_name": server_name,
        "server_port": server_port,
    }
    if "css" in inspect.signature(app.launch).parameters:
        launch_kwargs["css"] = APP_CSS
    app.launch(**launch_kwargs)


def create_default_handlers(
    config_path: str | None = None,
    *,
    runtime: dict[str, Any] | None = None,
    queue_runtime_factory: Callable[[Any], QueueWorkerRuntime] | None = None,
) -> UIHandlers:
    settings = load_settings(config_path) if runtime is None else None
    runtime = runtime or create_runtime(
        settings,
        enable_vector=True,
        enable_qa=True,
        real_pipeline=True,
    )
    factory = queue_runtime_factory or QueueWorkerRuntime
    worker_runtime = factory(runtime["queue"])
    worker_runtime.start()
    atexit.register(worker_runtime.stop)
    return UIHandlers(
        application_service=runtime["app_service"],
        retriever=runtime["vector_index"],
        queue_runtime=worker_runtime,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="video_qa.app")
    parser.add_argument("--config", default=None)
    parser.add_argument("--server-name", default=None)
    parser.add_argument("--server-port", type=int, default=None)
    args = parser.parse_args(argv)
    app = create_app(create_default_handlers(args.config))
    app.queue()
    launch_app(app, server_name=args.server_name, server_port=args.server_port)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
