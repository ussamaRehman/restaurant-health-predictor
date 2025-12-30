from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Paths:
    repo_root: Path
    data_raw: Path
    data_processed: Path
    models: Path
    reports: Path


def paths(repo_root: Path | None = None) -> Paths:
    root = (repo_root or Path(__file__).resolve().parents[2]).resolve()
    return Paths(
        repo_root=root,
        data_raw=root / "data" / "raw",
        data_processed=root / "data" / "processed",
        models=root / "models",
        reports=root / "reports",
    )


FAIL_GRADES = {"B", "C"}

