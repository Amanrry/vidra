"""Application settings for Vidra.

Settings are loaded from three sources, in increasing precedence:

1. Built-in defaults in the Pydantic models.
2. A YAML file such as ``configs/default.yaml``.
3. Environment variables prefixed with ``VIDRA_``.

The app intentionally has no domain or label configuration requirement. It can
start with only runtime/model settings and discovers video context from frames,
captions, transcripts, and detections.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import yaml
from pydantic import BaseModel, Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class AppSettings(BaseModel):
    name: str = "Vidra"
    environment: str = "development"


class PathSettings(BaseModel):
    data_dir: Path = Path("data")
    input_dir: Path = Path("data/input")
    runs_dir: Path = Path("data/runs")
    chroma_dir: Path = Path("data/chroma")
    database_path: Path = Path("data/vidra.sqlite3")


class VideoSettings(BaseModel):
    frame_interval_seconds: float = Field(default=2.0, gt=0)
    max_frames_per_video: int = Field(default=180, ge=1)
    max_upload_mb: int = Field(default=512, ge=1)


class ModelSettings(BaseModel):
    yolo_model: str = "yolov8n.pt"
    caption_model: str = "Salesforce/blip-image-captioning-base"
    siglip_model: str = "google/siglip-base-patch16-224"
    whisper_model: str = "base"
    device: str = "auto"


class RetrievalSettings(BaseModel):
    top_k_text: int = Field(default=6, ge=1)
    top_k_images: int = Field(default=6, ge=1)
    min_similarity: float = Field(default=0.0, ge=0.0, le=1.0)


class AlignmentSettings(BaseModel):
    same_class_threshold: float = Field(default=0.70, ge=0.0, le=1.0)
    different_class_threshold: float = Field(default=0.30, ge=0.0, le=1.0)

    @field_validator("different_class_threshold")
    @classmethod
    def different_threshold_not_above_same(
        cls, value: float, info: Any
    ) -> float:
        same = info.data.get("same_class_threshold")
        if same is not None and value > same:
            raise ValueError("different_class_threshold must be <= same_class_threshold")
        return value


class LLMSettings(BaseModel):
    base_url: str = "http://localhost:8000/v1"
    model: str = "qwen2.5"
    api_key: str | None = None
    timeout_seconds: float = Field(default=120.0, gt=0)
    max_tokens: int = Field(default=1024, ge=1)
    temperature: float = Field(default=0.1, ge=0.0, le=2.0)

    @field_validator("base_url")
    @classmethod
    def validate_openai_compatible_base_url(cls, value: str) -> str:
        clean = value.strip().rstrip("/")
        if not clean:
            raise ValueError("llm.base_url is required")
        if not clean.startswith(("http://", "https://")):
            raise ValueError("llm.base_url must be an HTTP(S) OpenAI-compatible endpoint")
        return clean


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="VIDRA_",
        env_nested_delimiter="__",
        extra="ignore",
    )

    app: AppSettings = Field(default_factory=AppSettings)
    paths: PathSettings = Field(default_factory=PathSettings)
    video: VideoSettings = Field(default_factory=VideoSettings)
    models: ModelSettings = Field(default_factory=ModelSettings)
    retrieval: RetrievalSettings = Field(default_factory=RetrievalSettings)
    alignment: AlignmentSettings = Field(default_factory=AlignmentSettings)
    llm: LLMSettings = Field(default_factory=LLMSettings)

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: Any,
        env_settings: Any,
        dotenv_settings: Any,
        file_secret_settings: Any,
    ) -> tuple[Any, ...]:
        """Let environment variables override YAML/init values."""
        return env_settings, init_settings, dotenv_settings, file_secret_settings

    def ensure_directories(self) -> None:
        """Create runtime directories needed before processing uploads."""
        for directory in [
            self.paths.data_dir,
            self.paths.input_dir,
            self.paths.runs_dir,
            self.paths.chroma_dir,
            self.paths.database_path.parent,
        ]:
            directory.mkdir(parents=True, exist_ok=True)


def _read_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Settings file not found: {path}")
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Settings file must contain a YAML mapping: {path}")
    return data


def load_settings(config_path: str | Path | None = None) -> Settings:
    """Load settings from optional YAML and environment variables.

    Passing ``None`` intentionally works: the app can start with built-in
    defaults and environment overrides only.
    """
    yaml_data: Mapping[str, Any] = {}
    if config_path is not None:
        yaml_data = _read_yaml(Path(config_path))
    return Settings(**yaml_data)
