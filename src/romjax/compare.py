"""Checkpoint-backed metric comparison routines."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Annotated, Any, Callable, Literal, Mapping, Sequence

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from alive_progress import alive_bar
from jaxtyping import PyTree
from pydantic import BeforeValidator, Field, PrivateAttr, SkipValidation, field_validator, model_validator

from romjax.graph import FunctionGraph
from romjax.loss import GraphTest
from romjax.operators import UnaryOp
from romjax.plotting import AxisOptions, PlotSpec, gridplot
from romjax.routine import Routine
from romjax.train import resolve_orbax_params
from romjax.tree import pytree_resolve_refs
from romjax.typing import DictModel, from_yaml
from romjax.utils import _NullProgress, load_h5, save_h5

type SUPPORTED_POLICIES = Literal["reuse", "overwrite", "error"]

__all__ = ["CompareCase", "CompareOrbax", "CompareMetric"]


class HistogramSpec(DictModel):
    """Per-case histogram configuration."""

    opts: AxisOptions = Field(default_factory=AxisOptions)
    kwargs: dict[str, Any] = Field(default_factory=dict)


class TableSpec(DictModel):
    """Per-case table configuration."""

    stats: Mapping[str, UnaryOp] | None = None
    format: str | None = None

    @field_validator("stats", mode="before")
    @classmethod
    def _normalize_stats(cls, value: Any) -> Mapping[str, UnaryOp] | None:
        return _normalize_stats(value)


class HistogramConfig(DictModel):
    """Global histogram figure configuration for :class:`CompareMetric`."""

    layout: list[list[tuple[str, ...]]] | None = None
    legend: Mapping[str, Any] | None = None  # cases and opts for figure legend

    @field_validator("layout", mode="before")
    @classmethod
    def _normalize_layout(cls, value: Any) -> list[list[tuple[str, ...]]] | None:
        """Normalize compact histogram layouts to a rectangular 2D plot grid."""
        if value is None:
            return None
        if isinstance(value, str):
            return [[(value,)]]
        if not isinstance(value, Sequence) or isinstance(value, bytes):
            raise ValueError("Histogram layout must contain case names")
        if all(isinstance(item, str) for item in value):
            return [[(item,) for item in value]]
    
        layout: list[list[tuple[str, ...]]] = []
        for row in value:
            if isinstance(row, str):
                layout.append([(row,)])
                continue
            if not isinstance(row, Sequence) or isinstance(row, bytes):
                raise ValueError("Histogram layout rows must contain case names or case-name tuples")
            if all(isinstance(item, str) for item in row):
                layout.append([(item,) for item in row])
                continue
            entries: list[tuple[str, ...]] = []
            for names in row:
                if isinstance(names, str):
                    entries.append((names,))
                elif (
                    isinstance(names, Sequence)
                    and not isinstance(names, bytes)
                    and all(isinstance(name, str) for name in names)
                ):
                    entries.append(tuple(names))
                else:
                    raise ValueError("Histogram subplot entries must be case-name tuples")
            layout.append(entries)
        return _pad_layout(layout, ())


class TableConfig(DictModel):
    """Global table rendering configuration for :class:`CompareMetric`."""

    fname: Path | None = None
    layout: list[list[str]] | None = None
    row_labels: Sequence[str] | None = None
    col_labels: Sequence[str] | None = None
    template: str | None = None

    @field_validator("layout", mode="before")
    @classmethod
    def _normalize_layout(cls, value: Any) -> list[list[str]] | None:
        """Normalize compact table layouts to a rectangular 2D case-name grid."""
        if value is None:
            return None
        if isinstance(value, str):
            return [[value]]
        if not isinstance(value, Sequence) or isinstance(value, bytes):
            raise ValueError("Table layout must contain case names")
        if all(isinstance(item, str) for item in value):
            return [list(value)]
        layout: list[list[str]] = []
        for row in value:
            if isinstance(row, str):
                layout.append([row])
            elif (
                isinstance(row, Sequence)
                and not isinstance(row, bytes)
                and all(isinstance(name, str) for name in row)
            ):
                layout.append(list(row))
            else:
                raise ValueError("Table layout rows must contain case names")
        return _pad_layout(layout, "")


class CompareCase(DictModel):
    """One named parameter case used by :class:`CompareOrbax`.

    :param name: unique name used for cache and figure/table references
    :param params: direct parameters or an Orbax checkpoint reference
    :param template: optional template used to load an Orbax checkpoint
    :param graph: optional graph used to resolve the template and GraphTest metric
    :param base: optional named base specification
    :param metric: callable metric to compare cases
    :param hist: configs for plotting as a histogram
    :param table: configs for printing as a table
    """

    name: str
    params: Any = None
    template: Any = None
    graph: Annotated[FunctionGraph | None, BeforeValidator(from_yaml)] = None
    base: str | None = None
    metric: SkipValidation[Callable[[PyTree], PyTree] | None] = None
    hist: HistogramSpec | None = None
    table: TableSpec | None = None


class CompareOrbax(Routine):
    """Resolve named direct or Orbax-backed parameter cases.

    Subclasses may override ``cases`` and ``bases`` with a ``CompareCase``
    subclass to add their own YAML fields.
    """

    cases: Sequence[CompareCase]
    bases: Sequence[CompareCase] = Field(default_factory=list)
    _resolved_cases: dict[str, CompareCase] = PrivateAttr(default_factory=dict)

    @model_validator(mode="after")
    def _resolve_cases(self) -> "CompareOrbax":
        bases = _cases_by_name(self.bases, "bases")
        cases = _cases_by_name(self.cases, "cases")
        resolved: dict[str, CompareCase] = {}
        for name, case in cases.items():
            merged = case
            if case.base is not None:
                if case.base not in bases:
                    raise ValueError(f"Case {name!r} references unknown base {case.base!r}")
                merged = _merge_case(bases[case.base], case)
            resolved[name] = self._resolve_case(merged)
        self._resolved_cases = resolved
        return self

    @staticmethod
    def _resolve_case(case: CompareCase) -> CompareCase:
        """Resolve one case template, graph references, and checkpoint payload."""
        case = copy.deepcopy(case)
        template = from_yaml(case.template)

        if case.graph is not None:
            case.graph.resolve_norms()

            if isinstance(case.metric, GraphTest):
                case.metric.bind_graph(case.graph, default_datasets=True)

            template = pytree_resolve_refs(template, case.graph, raise_on_missing=False)

        sample = getattr(template, "sample", None)
        if callable(sample):
            template = sample(jax.random.key(0))
        case.template = template
        case.params = resolve_orbax_params(case.params, template)
        return case

    @property
    def resolved_cases(self) -> Mapping[str, CompareCase]:
        """Return resolved cases keyed by their configured names."""
        return self._resolved_cases

    def run(self) -> int:
        """Resolve cases during validation; subclasses provide additional work."""
        return 0


class CompareMetric(CompareOrbax):
    """Evaluate and compare one metric callable for each resolved case.

    :param root: directory containing the HDF5 metric cache and optional outputs
    :param write_policy: cache reuse policy, evaluated independently per case
    :param hist: optional global histogram configuration
    :param table: optional global table configuration
    """

    root: Path | None = None
    write_policy: SUPPORTED_POLICIES = "reuse"
    hist: HistogramConfig | None = None
    table: TableConfig | None = None
    show_histogram: bool = False
    show_table: bool = False
    show_progress: bool = True
    fname: str = "compare_metric.h5"

    @model_validator(mode="after")
    def _validate_layouts(self) -> "CompareMetric":
        if self.hist is not None and self.hist.layout is not None:
            names = (name for row in self.hist.layout for entries in row for name in entries)
            _validate_layout_names(names, self.resolved_cases)
        if self.table is not None and self.table.layout is not None:
            _validate_layout_names((name for row in self.table.layout for name in row), self.resolved_cases)
        return self

    def run(self) -> int:
        """Evaluate missing metrics, write the HDF5 cache, and render enabled outputs."""
        path = None
        cached: dict[str, PyTree] = {}
        if self.root is not None:
            self.root.mkdir(parents=True, exist_ok=True)
            path = self.root / self.fname
            cached = load_h5({}, path, jax=False) if path.exists() else {}
        results: dict[str, PyTree] = dict(cached)
        context = alive_bar(len(self._resolved_cases)) if self.show_progress else _NullProgress()
        with context as bar:
            for name, case in self._resolved_cases.items():
                bar.text(f"Comparing case: {name}")
                exists = name in cached
                if exists and self.write_policy == "error":
                    raise ValueError(f"Metric already computed for {name!r} and write_policy='error'")
                if exists and self.write_policy == "reuse":
                    bar()
                    continue
                if case.metric is None:
                    raise ValueError(f"Case {name!r} must define a metric, directly or through its base")
                results[name] = jax.device_get(case.metric(case.params))
                bar()
        if path is not None:
            save_h5(results, path, mode="w")
        if self.show_histogram:
            self._plot_histograms(results)
        if self.show_table:
            self._render_table(results)
        if self.root is None and self.show_histogram:
            plt.show()
        return 0

    def _plot_histograms(self, results: Mapping[str, PyTree]) -> None:
        """Render configured per-case histogram overlays."""
        if self.hist is None or self.hist.layout is None:
            raise ValueError("show_histogram=True requires a histogram layout")
        layout = self.hist.layout
        rows = len(layout)
        cols = max(len(row) for row in layout)
        plots: list[list[list[PlotSpec]]] = [[[] for _ in range(cols)] for _ in range(rows)]
        legend_labels: dict[str, str] = {}
        for row, entries in enumerate(layout):
            for col, names in enumerate(entries):
                for name in names:
                    case = self._resolved_cases[name]
                    if case.hist is None:
                        raise ValueError(f"Histogram layout references case {name!r} without a hist specification")
                    legend_labels[name] = case.hist.opts.leg_label or name
                    plots[row][col].append(
                        PlotSpec(kind="hist", name=name, data=_metric_values(results[name]), **case.hist.model_dump())
                    )

        def adjust(fig: Any, axes: Any, artists: Any, cbars: Any) -> None:
            del artists, cbars
            axes = np.asarray(axes).reshape(rows, cols)
            for row in range(rows):
                for col in range(cols):
                    axis = axes[row, col]
                    axis.set_yticks([])
                    axis.tick_params(axis="y", left=False, labelleft=False)
            if self.hist.legend is None:
                return
            names = self.hist.legend.get("cases", [])
            opts = dict(self.hist.legend.get("opts", {}))
            handles: list[Any] = []
            labels: list[str] = []
            for name in names:
                label = legend_labels.get(name, name)
                for axis in axes.flat:
                    axis_handles, axis_labels = axis.get_legend_handles_labels()
                    if label in axis_labels:
                        handles.append(axis_handles[axis_labels.index(label)])
                        labels.append(label)
                        break
            for axis in axes.flat:
                legend = axis.get_legend()
                if legend is not None:
                    legend.remove()
            fig.legend(handles, labels, **opts)

        return gridplot(
            [[tuple(cell) for cell in row] for row in plots],
            adjust=adjust,
        )

    def _render_table(self, results: Mapping[str, PyTree]) -> None:
        """Print and optionally save the configured LaTeX comparison table."""
        if self.table is None or self.table.layout is None:
            raise ValueError("show_table=True requires a table layout")
        rows = len(self.table.layout)
        cols = max(len(row) for row in self.table.layout)
        values = [["" for _ in range(cols)] for _ in range(rows)]
        for row, names in enumerate(self.table.layout):
            for col, name in enumerate(names):
                if not name:
                    continue
                case = self._resolved_cases[name]
                if case.table is None or case.table.stats is None or case.table.format is None:
                    raise ValueError(f"Table layout references case {name!r} without stats and format")
                metric_values = jnp.asarray(_metric_values(results[name]))
                stats = {stat: float(operation(metric_values)) for stat, operation in case.table.stats.items()}
                values[row][col] = case.table.format.format(**stats)
        body_rows = values
        if self.table.row_labels is not None:
            if len(self.table.row_labels) != rows:
                raise ValueError("table.row_labels length must match the table row count")
            body_rows = [[label, *row] for label, row in zip(self.table.row_labels, values, strict=True)]
        header = list(self.table.col_labels or [])
        if header:
            if len(header) != cols:
                raise ValueError("table.col_labels length must match the table column count")
            body_rows = [["", *header] if self.table.row_labels is not None else header, *body_rows]
        body = "\n".join(" & ".join(row) + r" \\" for row in body_rows)
        column_count = cols + int(self.table.row_labels is not None)
        latex = "\n".join([rf"\begin{{tabular}}{{{'l' * column_count}}}", body, r"\end{tabular}", ""])
        text = self.table.template.replace("{{ table }}", latex) if self.table.template is not None else latex
        print(_format_aligned_table(body_rows))
        if self.root is not None and self.table.fname is not None:
            output = self.table.fname if self.table.fname.is_absolute() else self.root / self.table.fname
            output.write_text(text, encoding="utf-8")


def _normalize_stats(value: Any) -> Mapping[str, UnaryOp] | None:
    """Normalize YAML-friendly statistic declarations."""
    if value is None:
        return None
    if not isinstance(value, Mapping):
        value = [value] if isinstance(value, str) or not isinstance(value, Sequence) else value
        value = {str(item): item for item in value}
    return {name: UnaryOp(operation) for name, operation in value.items()}


def _merge_case(base: CompareCase, case: CompareCase) -> CompareCase:
    """Merge a case over its base without serializing runtime objects.

    Ordinary case fields replace their base values as a single unit. Histogram
    and table specifications are the only nested configuration objects, so
    they merge recursively to preserve shared plot and statistic options.
    """
    merged = copy.deepcopy(base)
    for field_name in type(case).model_fields:
        case_value = getattr(case, field_name)
        if case_value is None:
            continue
        if field_name in {"hist", "table"} and getattr(merged, field_name) is not None:
            setattr(merged, field_name, _deep_merge_model(getattr(merged, field_name), case_value))
        else:
            setattr(merged, field_name, copy.deepcopy(case_value))
    return merged


def _deep_merge_model[T: DictModel](base: T, override: T) -> T:
    """Recursively merge non-null Pydantic model fields and mapping values."""
    merged = copy.deepcopy(base)
    for field_name in type(override).model_fields:
        value = getattr(override, field_name)
        if value is None:
            continue
        base_value = getattr(merged, field_name)
        if isinstance(base_value, DictModel) and isinstance(value, DictModel):
            setattr(merged, field_name, _deep_merge_model(base_value, value))
        elif isinstance(base_value, Mapping) and isinstance(value, Mapping):
            setattr(merged, field_name, _deep_merge_mapping(base_value, value))
        else:
            setattr(merged, field_name, copy.deepcopy(value))
    return merged


def _deep_merge_mapping(base: Mapping[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    """Merge nested mappings while retaining non-overridden base values."""
    merged = copy.deepcopy(dict(base))
    for key, value in override.items():
        if key in merged and isinstance(merged[key], Mapping) and isinstance(value, Mapping):
            merged[key] = _deep_merge_mapping(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def _pad_layout[T](layout: list[list[T]], filler: T) -> list[list[T]]:
    """Pad a layout to a rectangular grid without changing configured positions."""
    if not layout or not any(layout):
        raise ValueError("Layout must contain at least one case name")
    width = max(len(row) for row in layout)
    return [row + [filler] * (width - len(row)) for row in layout]


def _validate_layout_names(names: Any, cases: Mapping[str, CompareCase]) -> None:
    """Ensure all non-padding layout names refer to configured cases."""
    unknown = sorted({name for name in names if name and name not in cases})
    if unknown:
        raise ValueError(f"Layout references unknown cases: {unknown}")


def _format_aligned_table(rows: Sequence[Sequence[str]]) -> str:
    """Format a left-aligned, evenly spaced text table for stdout."""
    widths = [max(len(row[index]) for row in rows) for index in range(len(rows[0]))]
    return "\n".join("  ".join(value.ljust(widths[index]) for index, value in enumerate(row)) for row in rows)


def _cases_by_name(cases: Sequence[CompareCase], label: str) -> dict[str, CompareCase]:
    """Validate and index an ordered sequence of case specifications."""
    indexed = {case.name: case for case in cases}
    if len(indexed) != len(cases):
        raise ValueError(f"{label} must have unique names")
    return indexed


def _metric_values(metric: PyTree) -> np.ndarray:
    """Concatenate numeric leaves of a metric pytree for plotting or statistics."""
    arrays = []
    for leaf in jax.tree.leaves(metric):
        array = np.asarray(leaf)
        if np.issubdtype(array.dtype, np.number):
            arrays.append(np.ravel(array))
    if not arrays:
        raise ValueError("Metric result must contain at least one numeric array leaf")
    return np.concatenate(arrays)
