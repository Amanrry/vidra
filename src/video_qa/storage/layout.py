"""Deterministic runtime file layout."""

from __future__ import annotations

from pathlib import Path

from video_qa.models.media import RunPaths


class RunLayout:
    """Factory for per-run directories.

    Keeping layout generation in one place prevents paths from being scattered
    across tools and services.
    """

    def __init__(self, runs_dir: str | Path) -> None:
        self.runs_dir = Path(runs_dir)

    def for_run(self, run_id: str, create: bool = False) -> RunPaths:
        clean_run_id = run_id.strip()
        if not clean_run_id:
            raise ValueError("run_id is required")
        if any(part in {".", ".."} for part in Path(clean_run_id).parts):
            raise ValueError("run_id must not contain relative path segments")

        root = self.runs_dir / clean_run_id
        paths = RunPaths(
            run_id=clean_run_id,
            root=root,
            source_dir=root / "source",
            frames_dir=root / "frames",
            annotated_frames_dir=root / "frames_annotated",
            crops_dir=root / "crops",
            reports_dir=root / "reports",
        )
        if create:
            for directory in [
                paths.root,
                paths.source_dir,
                paths.frames_dir,
                paths.annotated_frames_dir,
                paths.crops_dir,
                paths.reports_dir,
            ]:
                directory.mkdir(parents=True, exist_ok=True)
        return paths
