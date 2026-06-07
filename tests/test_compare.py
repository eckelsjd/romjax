from __future__ import annotations

import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import yaml
from orbax.checkpoint import v1 as ocp

from romjax.compare import CompareTable
from romjax.data_gen import DataLoader
from romjax.train import GraphLoss
from tests.test_train import ToyLinearReconstructionEdge, _write_graph_dataset


def squared_error(params: dict[str, jax.Array], single_data: dict[str, jax.Array]) -> jax.Array:
    return jnp.square(params["w"] - single_data["x"])


def absolute_error(params: dict[str, jax.Array], single_data: dict[str, jax.Array]) -> jax.Array:
    return jnp.abs(params["w"] - single_data["x"])


def graph_single_squared_error(params: dict, single_data: dict, graph: object) -> jax.Array:
    del graph
    return jnp.square(params["toy"]["weight"] - single_data["toy"]["inputs"]["x"])


class FiniteLoader:
    def __init__(self, batches):
        self.batches = batches

    def __iter__(self):
        return iter(self.batches)


def test_compare_table_format_print_and_write(capsys: pytest.CaptureFixture[str], tmp_path: Path) -> None:
    table = CompareTable(
        cases={},
        dataloaders={"train": FiniteLoader([]), "test": FiniteLoader([])},
        metrics={"sq": squared_error, "abs": absolute_error},
        stats=["mean", "std"],
        col_format="{mean:4.2f} ({std:4.2f})",
    )
    data = {
        "case_a": {
            "sq": {"train": {"mean": 1.0, "std": 0.25}, "test": {"mean": 2.0, "std": 0.5}},
            "abs": {"train": {"mean": 3.0, "std": 0.75}, "test": {"mean": 4.0, "std": 1.0}},
        },
    }

    assert table._format_table(data) == [["case_a", "1.00 (0.50)", "2.00 (1.00)", "3.00 (1.50)", "4.00 (2.00)"]]

    table.print_table(data)
    printed = capsys.readouterr().out
    assert "sq" in printed
    assert "abs" in printed
    assert "train" in printed
    assert "case_a" in printed
    assert "1.00 (0.50)" in printed

    tex_path = tmp_path / "table.tex"
    table.write_table(data, tex_path)
    tex = tex_path.read_text(encoding="utf-8")
    assert r"\begin{tabular}{lcccc}" in tex
    assert "Case & sq (train) & sq (test) & abs (train) & abs (test)" in tex
    assert r"case_a & 1.00 (0.50) & 2.00 (1.00) & 3.00 (1.50) & 4.00 (2.00) \\" in tex


@pytest.mark.skipif(sys.platform == "win32", reason="Orbax checkpointer issues on Windows")
def test_compare_basic_callables_params_templates_and_multiple_dataloaders(tmp_path: Path) -> None:
    checkpoint_root = tmp_path / "checkpoint"
    with ocp.training.Checkpointer(checkpoint_root) as ckptr:
        ckptr.save_checkpointables(
            step=0,
            checkpointables={"params": {"w": jnp.array(2.0), "label": "static"}},
            force=True,
        )

    compare = CompareTable(
        root=tmp_path / "compare",
        show_table=False,
        show_progress=False,
        cases={
            "direct": {"w": jnp.array(0.0), "label": "direct"},
            "orbax": checkpoint_root,
        },
        params_template={"w": jnp.array(0.0), "label": "static"},
        dataloaders={
            "train": FiniteLoader([
                {"x": jnp.array(0.0)},
                {"x": jnp.array(1.0)},
                {"x": jnp.array(2.0)},
            ]),
            "test": FiniteLoader([
                {"x": jnp.array(1.0)},
                {"x": jnp.array(3.0)},
            ]),
        },
        metrics={"sq": squared_error, "abs": absolute_error},
        stats={"mean": "mean", "max": "max"},
        col_format="{mean:.2f}/{max:.2f}",
    )

    assert compare.run() == 0

    result_path = tmp_path / "compare" / "compare_table.yml"
    tex_path = tmp_path / "compare" / "compare_table.tex"
    assert result_path.exists()
    assert tex_path.exists()

    results = yaml.safe_load(result_path.read_text(encoding="utf-8"))
    assert results["direct"]["sq"]["train"]["mean"] == pytest.approx(np.mean([0.0, 1.0, 4.0]))
    assert results["direct"]["sq"]["train"]["max"] == pytest.approx(4.0)
    assert results["direct"]["abs"]["test"]["mean"] == pytest.approx(np.mean([1.0, 3.0]))
    assert results["orbax"]["sq"]["train"]["mean"] == pytest.approx(np.mean([4.0, 1.0, 0.0]))
    assert results["orbax"]["abs"]["test"]["max"] == pytest.approx(1.0)

    tex = tex_path.read_text(encoding="utf-8")
    assert "direct & 1.67/4.00 & 5.00/9.00 & 1.00/2.00 & 2.00/3.00" in tex
    assert "orbax & 1.67/4.00 & 1.00/1.00 & 1.00/2.00 & 1.00/1.00" in tex


def test_compare_graph_loss_with_file_backed_dataloader(tmp_path: Path) -> None:
    data_root = tmp_path / "data"
    _write_graph_dataset(data_root, "toy", n_inputs=2, n_outputs=1)

    graph = {"edges": {"toy": ToyLinearReconstructionEdge()}}
    compare = CompareTable(
        root=tmp_path / "compare",
        show_table=False,
        show_progress=False,
        graph=graph,
        cases={
            "zero": {"toy": {"weight": jnp.array(0.0)}},
            "one": {"toy": {"weight": jnp.array(1.0)}},
        },
        dataloaders={
            "validation": DataLoader(
                root=data_root,
                datasets={"toy": {"kind": "implicit", "batch_size": 1, "shuffle_seed": 0}},
            ),
        },
        metrics={"loss": GraphLoss(terms=[{"function": graph_single_squared_error, "batch_reduce": None}])},
        stats=["mean", "max"],
        col_format="{mean:.1f}/{max:.1f}",
    )

    assert compare.run() == 0

    results = yaml.safe_load((tmp_path / "compare" / "compare_table.yml").read_text(encoding="utf-8"))
    assert results["zero"]["loss"]["validation"]["mean"] == pytest.approx(np.mean([1.0, 1.0, 121.0, 121.0]))
    assert results["zero"]["loss"]["validation"]["max"] == pytest.approx(121.0)
    assert results["one"]["loss"]["validation"]["mean"] == pytest.approx(np.mean([0.0, 0.0, 100.0, 100.0]))
    assert results["one"]["loss"]["validation"]["max"] == pytest.approx(100.0)

    tex = (tmp_path / "compare" / "compare_table.tex").read_text(encoding="utf-8")
    assert "zero & 61.0/121.0" in tex
    assert "one & 50.0/100.0" in tex
