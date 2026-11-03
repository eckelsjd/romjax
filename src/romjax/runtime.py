"""Explicit runtime context shared by recipe materialization and workers."""

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class RuntimeContext:
    """Concrete paths available while one routine is materialized.

    :param root: absolute shared workflow directory, if one was selected
    :param artifact_root: absolute artifact directory owned by the current routine
    """

    root: Path | None = None
    artifact_root: Path | None = None

    def __post_init__(self) -> None:
        for name in ("root", "artifact_root"):
            value = getattr(self, name)
            if value is None:
                continue
            path = Path(value).expanduser().absolute()
            object.__setattr__(self, name, path)
