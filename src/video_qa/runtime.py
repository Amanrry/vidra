"""Runtime bootstrap helpers for external media executables."""

from __future__ import annotations

import os
import shutil
from pathlib import Path


def ensure_media_binaries_on_path() -> None:
    """Expose bundled ffmpeg/ffprobe binaries when the host does not provide them."""

    if shutil.which("ffmpeg") and shutil.which("ffprobe"):
        return

    try:
        import static_ffmpeg  # type: ignore[import-not-found]
    except Exception:
        return

    try:
        static_ffmpeg.add_paths(weak=True)
    except TypeError:
        static_ffmpeg.add_paths()


def has_ffprobe() -> bool:
    """Return whether Gradio can probe video files in the current process."""

    ensure_media_binaries_on_path()
    return shutil.which("ffprobe") is not None


def ensure_ffmpeg_on_path() -> None:
    """Expose ffmpeg for Whisper, with imageio fallback for older installs."""

    ensure_media_binaries_on_path()
    if shutil.which("ffmpeg"):
        return

    try:
        import imageio_ffmpeg  # type: ignore[import-not-found]
    except Exception:
        return

    ffmpeg_path = Path(imageio_ffmpeg.get_ffmpeg_exe())
    if not ffmpeg_path.exists():
        return

    shim_dir = Path.cwd() / "data" / "runtime-bin"
    shim_dir.mkdir(parents=True, exist_ok=True)
    if os.name == "nt":
        shim_path = shim_dir / "ffmpeg.exe"
        if not shim_path.exists() or shim_path.stat().st_size != ffmpeg_path.stat().st_size:
            try:
                if shim_path.exists():
                    shim_path.unlink()
                os.link(ffmpeg_path, shim_path)
            except OSError:
                shutil.copy2(ffmpeg_path, shim_path)
    else:
        shim_path = shim_dir / "ffmpeg"
        shim_path.write_text(f'#!/bin/sh\nexec "{ffmpeg_path}" "$@"\n', encoding="utf-8")
        shim_path.chmod(0o755)

    os.environ["PATH"] = os.pathsep.join(
        [str(shim_dir), str(ffmpeg_path.parent), os.environ.get("PATH", "")]
    )
