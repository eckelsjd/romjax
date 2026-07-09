from __future__ import annotations

import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pytest
import yaml
from orbax.checkpoint import v1 as ocp

import romjax.compare as compare_module
from romjax.compare import CompareTable
from romjax.data_gen import DataLoader
from romjax.train import GraphLoss
from romjax.utils import load_h5, save_h5
from tests.test_train import ToyLinearReconstructionEdge, _write_graph_dataset


def squared_error(params: dict[str, jax.Array], single_data: dict[str, jax.Array]) -> jax.Array:
    return jnp.square(params["w"] - single_data["x"])


def absolute_error(params: dict[str, jax.Array], single_data: dict[str, jax.Array]) -> jax.Array:
    return jnp.abs(params["w"] - single_data["x"])


def graph_single_squared_error(params: dict, single_data: dict, graph: object) -> jax.Array:
    del graph
    sample = single_data["toy"] if "toy" in single_data else single_data
    return jnp.square(params["toy"]["weight"] - sample["inputs"]["x"])


def mapped_batch_metric(params: dict[str, jax.Array], batch: dict[str, list[dict[str, jax.Array]]]) -> jax.Array:
    del params
    return sum(item["x"] for item in batch["train"]) + sum(item["x"] for item in batch["test"])


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
    dist_path = tmp_path / "compare" / "compare_table.h5"
    tex_path = tmp_path / "compare" / "compare_table.tex"
    assert result_path.exists()
    assert dist_path.exists()
    assert tex_path.exists()

    results = yaml.safe_load(result_path.read_text(encoding="utf-8"))
    assert results["direct"]["sq"]["train"]["mean"] == pytest.approx(np.mean([0.0, 1.0, 4.0]))
    assert results["direct"]["sq"]["train"]["max"] == pytest.approx(4.0)
    assert results["direct"]["abs"]["test"]["mean"] == pytest.approx(np.mean([1.0, 3.0]))
    assert results["orbax"]["sq"]["train"]["mean"] == pytest.approx(np.mean([4.0, 1.0, 0.0]))
    assert results["orbax"]["abs"]["test"]["max"] == pytest.approx(1.0)

    distributions = load_h5({}, dist_path, jax=False)
    np.testing.assert_allclose(distributions["direct"]["sq"]["train"], np.array([0.0, 1.0, 4.0]))
    np.testing.assert_allclose(distributions["direct"]["abs"]["test"], np.array([1.0, 3.0]))
    np.testing.assert_allclose(distributions["orbax"]["sq"]["test"], np.array([1.0, 1.0]))

    tex = tex_path.read_text(encoding="utf-8")
    assert "direct & 1.67/4.00 & 5.00/9.00 & 1.00/2.00 & 2.00/3.00" in tex
    assert "orbax & 1.67/4.00 & 1.00/1.00 & 1.00/2.00 & 1.00/1.00" in tex


def test_compare_table_reuses_h5_distribution_for_missing_yaml_stats(tmp_path: Path) -> None:
    root = tmp_path / "compare"
    root.mkdir()
    save_h5({"case": {"sq": {"train": np.array([1.0, 2.0, 3.0])}}}, root / "compare_table.h5", mode="w")

    def fail_metric(params: dict[str, jax.Array], single_data: dict[str, jax.Array]) -> jax.Array:
        del params, single_data
        raise AssertionError("metric should not be evaluated when the H5 distribution can be reused")

    compare = CompareTable(
        root=root,
        show_table=False,
        show_progress=False,
        cases={"case": {"w": jnp.array(0.0)}},
        params_template={"w": jnp.array(0.0)},
        dataloaders={"train": FiniteLoader([{"x": jnp.array(0.0)}])},
        metrics={"sq": fail_metric},
        stats={"mean": "mean", "max": "max"},
        col_format="{mean:.1f}/{max:.1f}",
    )

    assert compare.run() == 0

    results = yaml.safe_load((root / "compare_table.yml").read_text(encoding="utf-8"))
    assert results["case"]["sq"]["train"]["mean"] == pytest.approx(2.0)
    assert results["case"]["sq"]["train"]["max"] == pytest.approx(3.0)


def test_compare_table_write_policy_overwrite_and_error_consider_h5(tmp_path: Path) -> None:
    root = tmp_path / "compare"

    compare = CompareTable(
        root=root,
        show_table=False,
        show_progress=False,
        cases={"case": {"w": jnp.array(0.0)}},
        params_template={"w": jnp.array(0.0)},
        dataloaders={"train": FiniteLoader([{"x": jnp.array(1.0)}, {"x": jnp.array(2.0)}])},
        metrics={"sq": squared_error},
        stats=["mean"],
        col_format="{mean:.1f}",
    )
    assert compare.run() == 0

    save_h5({"case": {"sq": {"train": np.array([99.0])}}}, root / "compare_table.h5", mode="w")
    overwrite = compare.model_copy(update={"write_policy": "overwrite"})
    assert overwrite.run() == 0

    distributions = load_h5({}, root / "compare_table.h5", jax=False)
    np.testing.assert_allclose(distributions["case"]["sq"]["train"], np.array([1.0, 4.0]))

    error = compare.model_copy(update={"write_policy": "error"})
    with pytest.raises(ValueError, match="Results already computed"):
        error.run()


def test_compare_table_histogram_plot_uses_gridplot(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    calls = {}

    def fake_gridplot(plots, **cfg):
        calls["plots"] = plots
        calls["cfg"] = cfg
        fig, axs = plt.subplots(*cfg["shape"], squeeze=False)
        for spec in plots[0][0]:
            axs[0, 0].hist([0.0, 1.0], label=spec.opts.leg_label)
        axs[0, 0].legend()
        cfg["adjust"](fig, axs, [], [])
        calls["legend_ncols"] = fig.legends[0]._ncols
        calls["y_ticks"] = [ax.get_yticks().tolist() for ax in axs.flat]
        calls["y_tick_label_visible"] = [
            any(label.get_visible() for label in ax.get_yticklabels()) for ax in axs.flat
        ]
        Path(cfg["save"]).touch()
        plt.close(fig)
        return fig, axs

    monkeypatch.setattr("romjax.compare.gridplot", fake_gridplot)

    compare = CompareTable(
        root=tmp_path / "compare",
        show_table=False,
        show_progress=False,
        cases={"first": {"w": jnp.array(0.0)}, "second": {"w": jnp.array(1.0)}},
        params_template={"w": jnp.array(0.0)},
        dataloaders={
            "train": FiniteLoader([{"x": jnp.array(0.0)}, {"x": jnp.array(1.0)}]),
            "test": FiniteLoader([{"x": jnp.array(2.0)}]),
        },
        metrics={"sq": squared_error, "abs": absolute_error},
        stats=["mean"],
        col_format="{mean:.1f}",
        hist={
            "case_labels": {"first": "Case 1", "second": "Case 2"},
            "dataset_labels": {"train": "Training", "test": "Testing"},
            "metric_labels": {"sq": "Squared", "abs": "Absolute"},
            "bins": 5,
            "density": True,
        },
    )

    assert compare.run() == 0

    assert calls["cfg"]["shape"] == (2, 2)
    assert calls["cfg"]["save"] == tmp_path / "compare" / "compare_table.pdf"
    assert calls["legend_ncols"] == 2
    assert calls["y_ticks"] == [[], [], [], []]
    assert calls["y_tick_label_visible"] == [False, False, False, False]
    assert len(calls["plots"]) == 2
    assert len(calls["plots"][0]) == 2
    assert len(calls["plots"][0][0]) == 2
    assert calls["plots"][0][0][0].opts.ylabel == "Training"
    assert calls["plots"][0][0][0].opts.xlabel is None
    assert calls["plots"][1][0][0].opts.ylabel == "Testing"
    assert calls["plots"][1][0][0].opts.xlabel == "Squared"
    assert calls["plots"][1][1][0].opts.xlabel == "Absolute"
    assert calls["plots"][0][0][0].opts.leg_label == "Case 1"
    assert calls["plots"][0][0][0].kwargs["bins"] == 5
    assert calls["plots"][0][0][0].kwargs["density"] is True
    assert (tmp_path / "compare" / "compare_table.pdf").exists()


def test_compare_table_histogram_legend_colors_match_plotted_histograms(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    real_gridplot = compare_module.gridplot
    calls = {}

    def capture_gridplot(plots, **cfg):
        fig, axs = real_gridplot(plots, **cfg)
        legend = fig.legends[0]
        calls["legend_colors"] = [
            mcolors.to_rgba(handle.get_facecolor()) for handle in legend.legend_handles
        ]
        calls["patch_colors"] = [
            mcolors.to_rgba(patch.get_facecolor()) for patch in axs[0, 0].patches
        ]
        plt.close(fig)
        return fig, axs

    monkeypatch.setattr("romjax.compare.gridplot", capture_gridplot)

    compare = CompareTable(
        root=tmp_path / "compare",
        show_table=False,
        show_progress=False,
        cases={"first": {"w": jnp.array(0.0)}, "second": {"w": jnp.array(1.0)}},
        params_template={"w": jnp.array(0.0)},
        dataloaders={"train": FiniteLoader([{"x": jnp.array(0.0)}, {"x": jnp.array(1.0)}])},
        metrics={"sq": squared_error},
        stats=["mean"],
        col_format="{mean:.1f}",
        hist={"histtype": "stepfilled", "bins": 2},
    )

    assert compare.run() == 0

    np.testing.assert_allclose(calls["legend_colors"], calls["patch_colors"])


def test_compare_table_iterates_dataloader_style_batch_mappings(tmp_path: Path) -> None:
    compare = CompareTable(
        root=tmp_path / "compare",
        show_table=False,
        show_progress=False,
        cases={"case": {"w": jnp.array(0.0)}},
        params_template={"w": jnp.array(0.0)},
        dataloaders={
            "validation": FiniteLoader(
                [
                    {
                        "train": [{"x": jnp.array(1.0)}, {"x": jnp.array(2.0)}],
                        "test": [{"x": jnp.array(10.0)}, {"x": jnp.array(20.0)}],
                    }
                ]
            ),
        },
        metrics={"sum": mapped_batch_metric},
        stats=["mean"],
        col_format="{mean:.1f}",
    )

    assert compare.run() == 0

    results = yaml.safe_load((tmp_path / "compare" / "compare_table.yml").read_text(encoding="utf-8"))
    assert results["case"]["sum"]["validation"]["mean"] == pytest.approx(33.0)


def test_compare_graph_loss_with_file_backed_dataloader(tmp_path: Path) -> None:
    data_root = tmp_path / "data"
    _write_graph_dataset(data_root, "toy", n_inputs=4, n_outputs=1)

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
                datasets={"toy": {"kind": "implicit", "batch_size": 2, "max_epochs": 1, "shuffle_seed": 0}},
            ),
        },
        metrics={"loss": GraphLoss(terms=[{"callable": graph_single_squared_error}])},
        stats=["mean", "max"],
        col_format="{mean:.1f}/{max:.1f}",
    )

    assert compare.run() == 0

    results = yaml.safe_load((tmp_path / "compare" / "compare_table.yml").read_text(encoding="utf-8"))
    assert results["zero"]["loss"]["validation"]["mean"] == pytest.approx(np.mean([221.0, 541.0]))
    assert results["zero"]["loss"]["validation"]["max"] == pytest.approx(701.0)
    assert results["one"]["loss"]["validation"]["mean"] == pytest.approx(np.mean([200.0, 500.0]))
    assert results["one"]["loss"]["validation"]["max"] == pytest.approx(650.0)

    tex = (tmp_path / "compare" / "compare_table.tex").read_text(encoding="utf-8")
    assert "zero & 381.0/701.0" in tex
    assert "one & 350.0/650.0" in tex
