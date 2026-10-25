"""Fixture builders for human-readable logical seed/sample labels.

These labels are only fixture coordinates; no legacy directories/files are created.
"""

from pathlib import Path

from romjax.storage import H5StorageConfig, H5Store, Record
from romjax.utils import save_h5 as save_artifact


def write_component(data, label, mode="w"):
    """Write a fixture component identified by its seed/sample coordinate label."""
    label = Path(label)
    names = {
        "input.h5": "inputs",
        "solution.h5": "solutions",
        "solution_residual.h5": "solution_residuals",
        "conditions.h5": "conditions",
        "output.h5": "values",
        "residual.h5": "residuals",
        "source.h5": "values",
    }
    if label.name not in names:
        return save_artifact(data, label, mode=mode)
    parts = label.parts
    first = next(i for i, part in enumerate(parts) if part.startswith("seed_"))
    root = Path(*parts[:first])
    seed, index = int(parts[first][5:]), int(parts[first + 1][7:])
    output = len(parts) - first == 5
    kind = "output" if output else "source" if label.name == "source.h5" else "input"
    store = H5Store(root, "source" if kind == "source" else "implicit", H5StorageConfig(layout="grouped"))
    store.write(
        [
            Record(
                seed,
                index,
                kind,
                {names[label.name]: data},
                int(parts[first + 2][5:]) if output else None,
                int(parts[first + 3][7:]) if output else None,
            )
        ]
    )
