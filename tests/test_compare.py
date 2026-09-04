from __future__ import annotations

from pathlib import Path

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import pytest

import romjax.compare as compare_module
from romjax.compare import CompareMetric, CompareOrbax
from romjax.utils import load_h5


def metric(params: dict[str, jnp.ndarray]) -> dict[str, jnp.ndarray]:
    """A small pytree-valued metric used by comparison tests."""
    return {"error": jnp.asarray([params["value"], params["value"] + 1.0])}


def test_compare_orbax_resolves_case_bases_and_templates() -> None:
    compare = CompareOrbax(
        bases=[{"name": "shared", "params": {"value": jnp.array(1.0)}, "template": {"unused": 0}}],
        cases=[
            {"name": "first", "base": "shared"},
            {"name": "second", "base": "shared", "params": {"value": jnp.array(3.0)}},
        ],
    )

    assert float(compare.resolved_cases["first"].params["value"]) == 1.0
    assert float(compare.resolved_cases["second"].params["value"]) == 3.0
    assert compare.resolved_cases["first"].template == {"unused": 0}


def test_compare_orbax_rejects_duplicate_or_unknown_bases() -> None:
    with pytest.raises(ValueError, match="unique names"):
        CompareOrbax(cases=[{"name": "same"}, {"name": "same"}])
    with pytest.raises(ValueError, match="unknown base"):
        CompareOrbax(cases=[{"name": "case", "base": "missing"}])


def test_compare_orbax_replaces_top_level_fields_and_deep_merges_specs() -> None:
    def base_metric(params: dict[str, jnp.ndarray]) -> jnp.ndarray:
        return params["value"]

    def case_metric(params: dict[str, jnp.ndarray]) -> jnp.ndarray:
        return params["value"] + 1

    compare = CompareOrbax(
        bases=[
            {
                "name": "base",
                "metric": base_metric,
                "hist": {"opts": {"xlabel": "Error"}, "kwargs": {"bins": 10, "color": "blue"}},
                "table": {"stats": ["mean", "std"], "format": "{mean:.2f}"},
            }
        ],
        cases=[
            {
                "name": "case",
                "base": "base",
                "metric": case_metric,
                "hist": {"opts": {"ylabel": "Validation"}, "kwargs": {"color": "red"}},
                "table": {"format": "{mean:.1f} ({std:.1f})"},
            }
        ],
    )
    resolved = compare.resolved_cases["case"]

    assert resolved.metric is case_metric
    assert resolved.hist.opts.xlabel == "Error"
    assert resolved.hist.opts.ylabel == "Validation"
    assert resolved.hist.kwargs == {"bins": 10, "color": "red"}
    assert list(resolved.table.stats) == ["mean", "std"]
    assert resolved.table.format == "{mean:.1f} ({std:.1f})"


def test_compare_metric_normalizes_compact_layouts() -> None:
    compare = CompareMetric(
        hist={"layout": ["first", "second"]},
        table={"layout": [["first"], ["second"]]},
        cases=[{"name": "first"}, {"name": "second"}],
    )

    assert compare.hist.layout == [[("first",), ("second",)]]
    assert compare.table.layout == [["first"], ["second"]]
    with pytest.raises(ValueError, match="unknown cases"):
        CompareMetric(hist={"layout": "missing"}, cases=[{"name": "first"}])


def test_compare_metric_writes_flat_case_h5_and_reuses_it(tmp_path: Path) -> None:
    compare = CompareMetric(
        root=tmp_path,
        show_progress=False,
        bases=[{"name": "base", "metric": metric}],
        cases=[
            {"name": "first", "base": "base", "params": {"value": jnp.array(1.0)}},
            {"name": "second", "base": "base", "params": {"value": jnp.array(3.0)}},
        ],
    )
    assert compare.run() == 0
    saved = load_h5({}, tmp_path / "compare_metric.h5", jax=False)
    np.testing.assert_allclose(saved["first"]["error"], [1.0, 2.0])
    np.testing.assert_allclose(saved["second"]["error"], [3.0, 4.0])

    def fail_metric(params: dict[str, jnp.ndarray]) -> jnp.ndarray:
        del params
        raise AssertionError("reused metrics must not be called")

    reused = compare.model_copy(deep=True)
    reused.resolved_cases["first"].metric = fail_metric
    reused.resolved_cases["second"].metric = fail_metric
    assert reused.run() == 0
    with pytest.raises(ValueError, match="already computed"):
        reused.model_copy(update={"write_policy": "error"}).run()


def test_compare_metric_overwrites_per_case_results(tmp_path: Path) -> None:
    compare = CompareMetric(
        root=tmp_path,
        show_progress=False,
        cases=[{"name": "case", "params": {"value": jnp.array(1.0)}, "metric": metric}],
    )
    compare.run()
    overwritten = compare.model_copy(update={"write_policy": "overwrite"}, deep=True)
    overwritten.resolved_cases["case"].params = {"value": jnp.array(10.0)}
    overwritten.run()
    saved = load_h5({}, tmp_path / "compare_metric.h5", jax=False)
    np.testing.assert_allclose(saved["case"]["error"], [10.0, 11.0])


def test_compare_metric_renders_aligned_table(capsys: pytest.CaptureFixture[str], tmp_path: Path) -> None:
    compare = CompareMetric(
        root=tmp_path,
        show_progress=False,
        show_table=True,
        table={
            "fname": "table.tex",
            "row_labels": ["validation"],
            "col_labels": ["first", "second"],
            "template": "Results:\n{{ table }}",
            "layout": [["first", "second"]],
        },
        cases=[
            {
                "name": "first",
                "params": {"value": jnp.array(1.0)},
                "metric": metric,
                "table": {"stats": ["mean", "std"], "format": "{mean:.1f} ({std:.1f})"},
            },
            {
                "name": "second",
                "params": {"value": jnp.array(3.0)},
                "metric": metric,
                "table": {"stats": ["mean", "std"], "format": "{mean:.1f} ({std:.1f})"},
            },
        ],
    )
    compare.run()
    output = capsys.readouterr().out
    assert "1.5 (0.5)" in output
    assert "validation  1.5 (0.5)  3.5 (0.5)" in output
    text = (tmp_path / "table.tex").read_text(encoding="utf-8")
    assert text.startswith("Results:")
    assert "validation & 1.5 (0.5) & 3.5 (0.5)" in text


def test_compare_metric_renders_histograms_with_case_overrides(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    calls = {}

    def fake_gridplot(plots, **kwargs):
        calls["plots"] = plots
        calls["kwargs"] = kwargs
        fig, axes = plt.subplots(len(plots), len(plots[0]), squeeze=False)
        kwargs["adjust"](fig, axes, [], [])
        return fig, axes

    monkeypatch.setattr(compare_module, "gridplot", fake_gridplot)
    compare = CompareMetric(
        root=tmp_path,
        show_progress=False,
        show_histogram=True,
        hist={
            "layout": [[ ["first", "second"] ]],
        },
        cases=[
            {
                "name": "first",
                "params": {"value": jnp.array(1.0)},
                "metric": metric,
                "hist": {"opts": {"xlabel": "Error"}, "kwargs": {"bins": 4, "alpha": 0.5}},
            },
            {
                "name": "second",
                "params": {"value": jnp.array(3.0)},
                "metric": metric,
                "hist": {"opts": {"leg_label": "Second"}, "kwargs": {"color": "crimson"}},
            },
        ],
    )
    compare.run()
    first, second = calls["plots"][0][0]
    assert first.kwargs["bins"] == 4
    assert first.kwargs["alpha"] == 0.5
    assert second.kwargs["color"] == "crimson"
    assert second.opts.leg_label == "Second"


def test_compare_metric_without_root_prints_and_shows(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    shown = []
    monkeypatch.setattr(compare_module.plt, "show", lambda: shown.append(True))
    monkeypatch.setattr(compare_module, "gridplot", lambda plots, **kwargs: (plt.figure(), np.asarray([[plt.gca()]])))
    compare = CompareMetric(
        show_progress=False,
        show_histogram=True,
        show_table=True,
        hist={"layout": "case"},
        table={"layout": "case"},
        cases=[
            {
                "name": "case",
                "params": {"value": jnp.array(1.0)},
                "metric": metric,
                "hist": {"kwargs": {"bins": 2}},
                "table": {"stats": ["mean"], "format": "{mean:.1f}"},
            }
        ],
    )
    assert compare.run() == 0
    assert shown == [True]
    assert "1.5" in capsys.readouterr().out
