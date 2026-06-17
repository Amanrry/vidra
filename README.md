# Vidra

Vidra is an open-domain multimodal video QA application. It uploads videos,
extracts frames, captions scenes, transcribes audio when present, detects
objects, indexes video context with ChromaDB, and answers questions through an
OpenAI-compatible chat-completions server.

## Quick Start

Run these commands from the project root:

```powershell
cd D:\Dev_Project\Sotfware_management\vidra
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
uv venv
.\.venv\Scripts\Activate.ps1
uv pip install -e ".[ai,dev]"
```

If `uv` is already installed, skip the install script.

Start an OpenAI-compatible LLM server before asking questions. The default
endpoint is:

```yaml
http://localhost:8000/v1/chat/completions
```

You can change it in `configs/default.yaml`:

```yaml
llm:
  base_url: http://localhost:8000/v1
  model: qwen2.5
  api_key: null
```

Launch the Gradio app:

```powershell
python -m video_qa.app --config configs/default.yaml --server-port 7860
```

Open:

```text
http://127.0.0.1:7860
```

Upload a video, wait for progress to reach complete or partial, then use Ask or
Search.

## First Run Notes

The real pipeline is enabled by default for the app and CLI processing commands.
The first run may download model files for:

- YOLO: object detection
- BLIP: frame captioning
- Whisper: audio transcription
- SigLIP: text/image embeddings
- ChromaDB: local vector index

Generated runtime data is written under `data/`, which is ignored by git.

## Model Downloads and Caches

Most models are downloaded lazily the first time their stage runs. If you want
to control where model files are stored, set cache directories before launching
Vidra:

```powershell
$env:HF_HOME = "D:\Dev_Project\Sotfware_management\model_cache\huggingface"
$env:TRANSFORMERS_CACHE = "D:\Dev_Project\Sotfware_management\model_cache\huggingface\transformers"
$env:TORCH_HOME = "D:\Dev_Project\Sotfware_management\model_cache\torch"
$env:WHISPER_CACHE_DIR = "D:\Dev_Project\Sotfware_management\model_cache\whisper"
```

Create the cache root if needed:

```powershell
New-Item -ItemType Directory -Force D:\Dev_Project\Sotfware_management\model_cache
```

You can either let Vidra download models automatically during the first
processing run, or pre-download them with the commands below. If you change a
model name in `configs/default.yaml`, use the same model name in the
pre-download command.

The model names used by the pipeline are:

```yaml
models:
  yolo_model: yolov8n.pt
  caption_model: Salesforce/blip-image-captioning-base
  siglip_model: google/siglip-base-patch16-224
  whisper_model: base
  device: auto
```

### PyTorch

Install a PyTorch build that matches your machine. CPU works but will be slow
for real video processing.

CPU-only:

```powershell
uv pip install torch torchvision torchaudio
```

For NVIDIA CUDA, install the matching command from the PyTorch website, then
check CUDA availability:

```powershell
@'
import torch
print(torch.__version__)
print("cuda:", torch.cuda.is_available())
'@ | python -
```

### YOLO

Vidra defaults to `yolov8n.pt`. Ultralytics downloads it automatically when
object detection first runs.

To pre-download it:

```powershell
@'
from ultralytics import YOLO
YOLO("yolov8n.pt")
print("YOLO ready")
'@ | python -
```

To use another YOLO model, edit `configs/default.yaml`:

```yaml
models:
  yolo_model: yolov8s.pt
```

### BLIP Caption Model

Vidra defaults to `Salesforce/blip-image-captioning-base`.

To pre-download it:

```powershell
@'
from transformers import BlipForConditionalGeneration, BlipProcessor
name = "Salesforce/blip-image-captioning-base"
BlipProcessor.from_pretrained(name)
BlipForConditionalGeneration.from_pretrained(name)
print("BLIP ready")
'@ | python -
```

To change it:

```yaml
models:
  caption_model: Salesforce/blip-image-captioning-base
```

### SigLIP Embedding Model

Vidra defaults to `google/siglip-base-patch16-224`.

To pre-download it:

```powershell
@'
from transformers import AutoModel, AutoProcessor
name = "google/siglip-base-patch16-224"
AutoProcessor.from_pretrained(name)
AutoModel.from_pretrained(name)
print("SigLIP ready")
'@ | python -
```

To change it:

```yaml
models:
  siglip_model: google/siglip-base-patch16-224
```

### Whisper

Vidra defaults to Whisper `base`. Whisper downloads the model automatically.

To pre-download it:

```powershell
@'
import whisper
whisper.load_model("base")
print("Whisper ready")
'@ | python -
```

To use a faster or stronger model:

```yaml
models:
  whisper_model: base
```

Common options are `tiny`, `base`, `small`, `medium`, and `large`. Larger models
are slower and need more memory.

### ChromaDB

ChromaDB stores the local vector index under:

```text
data/chroma/
```

You usually do not need to configure it. To reset the vector index during
development, stop Vidra and delete:

```powershell
Remove-Item -Recurse -Force data\chroma
```

The next processing run will rebuild the collection.

## CLI Usage

Run a full synchronous processing job:

```powershell
python -m video_qa.cli --config configs/default.yaml process --video D:\path\to\demo.mp4
```

The command writes outputs under:

```text
data/runs/<video_id>/
```

Important files:

```text
data/runs/<video_id>/frames/
data/runs/<video_id>/annotated_frames/
data/runs/<video_id>/crops/
data/runs/<video_id>/reports/report.json
data/runs/<video_id>/reports/detections.csv
data/runs/<video_id>/reports/summary.md
```

Ask a question after processing:

```powershell
python -m video_qa.cli --config configs/default.yaml ask --video-id <video_id> --question "What happens in this video?"
```

Search indexed context:

```powershell
python -m video_qa.cli --config configs/default.yaml search --video-id <video_id> --query "person walking"
```

Queue-backed processing is available too:

```powershell
python -m video_qa.cli --config configs/default.yaml enqueue --video D:\path\to\demo.mp4
python -m video_qa.cli --config configs/default.yaml worker --drain
python -m video_qa.cli --config configs/default.yaml status --video-id <video_id>
```

For the Gradio app, queue workers are started automatically in the background.

## Configuration

Default settings live in `configs/default.yaml`.

Useful knobs:

```yaml
video:
  frame_interval_seconds: 2.0
  max_frames_per_video: 180
  max_upload_mb: 512

models:
  yolo_model: yolov8n.pt
  caption_model: Salesforce/blip-image-captioning-base
  siglip_model: google/siglip-base-patch16-224
  whisper_model: base

retrieval:
  top_k_text: 6
  top_k_images: 6
```

Environment variables can override nested settings with the `VIDRA_` prefix and
double underscores. Example:

```powershell
$env:VIDRA_LLM__BASE_URL = "http://localhost:8000/v1"
$env:VIDRA_LLM__MODEL = "qwen2.5"
```

Vidra intentionally does not require domain or label YAML files. It uses
general-purpose frame captions, transcript segments, object labels, crops, and
semantic retrieval for arbitrary videos.

## Development

Install development dependencies:

```powershell
uv pip install -e ".[dev]"
```

Run fast unit tests:

```powershell
python -m pytest tests/unit -q
```

Run lint checks:

```powershell
python -m ruff check .
```

The fast unit tests use fakes and mocks. Real-model integration tests should be
kept separate and marked as slow because they may download large model weights.

## Troubleshooting

If `python -m video_qa...` cannot find the package, either install the project in
editable mode:

```powershell
uv pip install -e ".[ai,dev]"
```

or run with `PYTHONPATH`:

```powershell
$env:PYTHONPATH = "src"
python -m video_qa.app --help
```

If Gradio is missing, install the AI extras:

```powershell
uv pip install -e ".[ai]"
```

If the app processes video but Ask fails, check that your OpenAI-compatible LLM
server is running and that `llm.base_url` and `llm.model` are correct.
