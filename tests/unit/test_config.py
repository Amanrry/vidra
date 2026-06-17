from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from video_qa import Settings, load_settings


def test_package_imports() -> None:
    assert Settings().app.name == "Vidra"


def test_loads_default_yaml_without_label_config() -> None:
    settings = load_settings(Path("configs/default.yaml"))

    assert settings.app.name == "Vidra"
    assert settings.models.yolo_model == "yolov8n.pt"
    assert settings.llm.base_url == "http://localhost:8000/v1"
    assert not hasattr(settings, "labels")
    assert not hasattr(settings, "domain")


def test_environment_overrides_yaml(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("VIDRA_LLM__BASE_URL", "http://localhost:9000/v1")
    monkeypatch.setenv("VIDRA_LLM__MODEL", "demo-model")
    monkeypatch.setenv("VIDRA_VIDEO__MAX_FRAMES_PER_VIDEO", "12")

    settings = load_settings(Path("configs/default.yaml"))

    assert settings.llm.base_url == "http://localhost:9000/v1"
    assert settings.llm.model == "demo-model"
    assert settings.video.max_frames_per_video == 12


def test_can_load_from_builtin_defaults_only() -> None:
    settings = load_settings()

    assert settings.app.name == "Vidra"
    assert settings.models.siglip_model == "google/siglip-base-patch16-224"
    assert settings.paths.runs_dir == Path("data/runs")


def test_rejects_non_http_llm_base_url(tmp_path: Path) -> None:
    config = tmp_path / "bad.yaml"
    config.write_text(
        """
llm:
  base_url: localhost:8000/v1
""".strip(),
        encoding="utf-8",
    )

    with pytest.raises(ValidationError, match="HTTP"):
        load_settings(config)

