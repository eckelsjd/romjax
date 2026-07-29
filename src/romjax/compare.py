import copy
from pathlib import Path
from typing import Annotated, Any, Callable, Iterable, Literal, Mapping, Sequence

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import yaml
from alive_progress import alive_bar
from jaxtyping import PyTree
from matplotlib import rcParams
from pydantic import (
    BeforeValidator,
    Field,
    SkipValidation,
    field_validator,
    model_validator,
)

from romjax.data_gen import DataLoader, LoadDataConfig
from romjax.graph import FunctionGraph
from romjax.operators import UnaryOp
from romjax.plotting import PlotSpec, gridplot
from romjax.routine import Routine
from romjax.train import resolve_orbax_params
from romjax.tree import pytree_path_iter, pytree_resolve_refs
from romjax.typing import DictModel, from_yaml
from romjax.utils import _NullProgress, load_h5, save_h5

type SUPPORTED_POLICIES = Literal["reuse", "overwrite", "error"]

__all__ = ["CompareOrbax", "CompareTable"]


class CompareOrbax(Routine):
    """
    Routine for comparing models via orbax checkpoints from `Train`.
    """
    cases: Mapping[str, PyTree]

    root: Path | None = None
    write_policy: SUPPORTED_POLICIES = "reuse"
    params_template: PyTree | Mapping[str, PyTree] | None = None
    graph: Annotated[FunctionGraph | None, BeforeValidator(from_yaml)] = None

    @model_validator(mode="after")
    def _setup_graph_configs(self):
        if ((isinstance(self.params_template, Mapping) and not all(case in self.params_template for case in self.cases))
            or not isinstance(self.params_template, Mapping)):
            # Use same params for all cases
            self.params_template = {case: copy.deepcopy(self.params_template) for case in self.cases}
        
        for case in self.cases:
            self.params_template[case] = from_yaml(self.params_template[case])  # validate from yaml (optional)

        if self.graph is not None:
            self.graph.resolve_norms()
            for case in self.cases:
                self.params_template[case] = pytree_resolve_refs(
                    self.params_template[case], self.graph, raise_on_missing=False
                )
        
        # Initialize param templates just like in train
        for case in self.cases:
            sample_fn = getattr(self.params_template[case], "sample", None)
            if callable(sample_fn):
                self.params_template[case] = sample_fn(jax.random.key(0))
        
        return self


class CompareHistogram(DictModel):
    """
    Histogram plotting options for :class:`CompareTable` error distributions.

    :param case_labels: optional display labels keyed by case name
    :param dataset_labels: optional y-axis labels keyed by dataset name
    :param metric_labels: optional x-axis labels keyed by metric name
    :param bins: histogram bin specification passed to matplotlib
    :param density: whether to normalize histograms to probability densities
    :param alpha: histogram transparency, either a default float or per-case values;
        cases not included in a mapping use ``0.45``
    :param colors: optional matplotlib colors keyed by case name
    :param histtype: histogram rendering style
    :param scale: the scale for the x-axis (linear or log)
    :param leg_anchor: bbox anchor for legend
    """

    case_labels: Mapping[str, str] | None = None
    dataset_labels: Mapping[str, str] | None = None
    metric_labels: Mapping[str, str] | None = None
    bins: int | str | Sequence[float] = "auto"
    density: bool = False
    alpha: float | Mapping[str, float] = 0.45
    colors: Mapping[str, str] | None = None
    histtype: str = "stepfilled"
    scale: str | None = None
    leg_anchor: tuple[float, float] = (0.5, 1.1)
    leg_ncols: int | None = None


class CompareTable(CompareOrbax):
    """
    Construct a table of the form: case (row) -> metric (col) -> dataset -> stat.

    !!! Example
        ```yaml
        case_one: 
          reconstruction_error:
            train: { mean: 0.1, std: 0.02 }
            test: { mean: 0.2, std: 0.01 }
          residual_error:
            train: { ... }
            test: { ... }
        
        case_two:
          ...
        ```
    
    Will write/load table results from a yaml file of this format. Cases are loaded from orbax checkpoints. 
    Metrics are callable of the form `f(params, data) -> float`. 
    Stats are computed over all data loaded from each dataloader.
    """

    dataloaders: SkipValidation[Mapping[str, Iterable[Any]]]
    metrics: Mapping[str, Callable[[PyTree, Any], float]]

    stats: Mapping[str, UnaryOp] = Field(default_factory=lambda: ["mean", "std"])
    latex_template: str | None = None
    col_format: Mapping[str, str] = r"{mean:5.3f} ({std:5.3f})"
    filename: str = "compare_table.yml"
    hist: CompareHistogram | None = None
    show_table: bool = True
    show_progress: bool = True

    @field_validator("stats", mode="before")
    @classmethod
    def _from_plain_list(cls, value):
        if not isinstance(value, Mapping):
            if not isinstance(value, tuple | list):
                value = [value]
            if isinstance(value, tuple | list):
                value = {str(v): v for v in value}
        
        if isinstance(value, Mapping):
            value = {k: UnaryOp(v) for k, v in value.items()}

        return value
    
    @field_validator("col_format", mode="before")
    @classmethod
    def _single_col_format(cls, value, info):
        if not isinstance(value, Mapping):
            value = {metric_name: value for metric_name in info.data["metrics"]}

        return value

    @field_validator("hist", mode="before")
    @classmethod
    def _from_bool_or_mapping(cls, value):
        """Allow simple YAML-friendly histogram plot configuration."""
        if value is None or value is False:
            return None
        if value is True:
            return CompareHistogram()
        if isinstance(value, CompareHistogram):
            return value
        if isinstance(value, Mapping):
            return CompareHistogram(**value)
        raise ValueError("histogram_plot must be a bool, mapping, CompareHistogramConfig, or None")

    @model_validator(mode="after")
    def _bind_graph_and_jit_metrics(self):
        """Bind a graph to any metrics that need it and jit them."""
        for name, metric_fn in list(self.metrics.items()):
            if hasattr(metric_fn, "bind_graph") and hasattr(metric_fn, "graph"):
                if metric_fn.graph is None:
                    metric_fn.bind_graph(self.graph)
                self.metrics[name] = eqx.filter_jit(metric_fn.__call__)
            else:
                self.metrics[name] = eqx.filter_jit(metric_fn)
        
        # Enforce max_epochs=1
        for _, dl in self.dataloaders.items():
            if isinstance(dl, DataLoader):
                for _, ds_cfg in pytree_path_iter(dl.datasets, is_leaf=lambda leaf: isinstance(leaf, LoadDataConfig)):
                    ds_cfg.max_epochs = 1

        return self

    def _format_table(self, data: Mapping) -> list[list[str]]:
        """Format computed comparison values as case rows with metric/dataset columns."""
        rows = []
        for case_name, case_data in data.items():
            row = [str(case_name)]
            for metric_name in self.metrics:
                metric_data = case_data.get(metric_name, {})
                for dataset_name in self.dataloaders:
                    dataset = dict(metric_data.get(dataset_name, {}))
                    if "std" in dataset:
                        dataset["std"] = 2.0 * dataset["std"]
                    try:
                        row.append(self.col_format[metric_name].format(**dataset))
                    except KeyError as exc:
                        raise ValueError(
                            f"Missing stat {exc.args[0]!r} for {case_name}->{metric_name}->{dataset_name}"
                        ) from exc
            rows.append(row)
        return rows

    def _table_columns(self) -> list[tuple[str, str]]:
        """Return metric/dataset column labels in table order."""
        return [(metric_name, ds_name) for metric_name in self.metrics for ds_name in self.dataloaders]

    def _plot_histograms(self, distributions: Mapping[str, Mapping], path: str | Path) -> None:
        """Save a histogram grid of all cached error distributions."""
        cfg = self.hist
        dataset_labels = cfg.dataset_labels or {}
        metric_labels = cfg.metric_labels or {}
        case_labels = cfg.case_labels or {}
        dataset_names = list(self.dataloaders)
        metric_names = list(self.metrics)
        case_names = list(self.cases)
        prop_cycle = rcParams["axes.prop_cycle"].by_key()
        default_colors = prop_cycle.get("color", [])
        case_colors = cfg.colors or {}
        case_alphas = cfg.alpha if isinstance(cfg.alpha, Mapping) else {}
        default_alpha = cfg.alpha if isinstance(cfg.alpha, float) else 0.45

        plots = []
        for row_idx, dataset_name in enumerate(dataset_names):
            row = []
            for col_idx, metric_name in enumerate(metric_names):
                specs = []
                for case_idx, case_name in enumerate(case_names):
                    try:
                        values = distributions[case_name][metric_name][dataset_name]
                    except KeyError as exc:
                        raise ValueError(
                            f"Missing distribution for {case_name}->{metric_name}->{dataset_name}"
                        ) from exc

                    specs.append(
                        PlotSpec(
                            kind="hist",
                            name=f"{dataset_name}_{metric_name}",
                            data=np.ravel(np.asarray(values)),
                            opts={
                                "ylabel": dataset_labels.get(dataset_name, dataset_name) if col_idx == 0 else None,
                                "xlabel": metric_labels.get(metric_name, metric_name)
                                if row_idx == len(dataset_names) - 1 else None,
                                "leg_label": case_labels.get(case_name, case_name),
                                "xscale": cfg.scale
                            },
                            kwargs={
                                "bins": cfg.bins,
                                "density": cfg.density,
                                "alpha": case_alphas.get(case_name, default_alpha),
                                "histtype": cfg.histtype,
                                **(
                                    {"color": case_colors[case_name]}
                                    if case_name in case_colors
                                    else (
                                        {"color": default_colors[case_idx % len(default_colors)]}
                                        if default_colors
                                        else {}
                                    )
                                ),
                            },
                        )
                    )
                row.append(tuple(specs))
            plots.append(row)

        def adjust(fig, axs, artists, cbars):
            """Replace per-axis legends with one compact figure legend."""
            for ax in axs.flat:
                legend = ax.get_legend()
                if legend is not None:
                    legend.remove()
                ax.set_yticks([])
                ax.tick_params(axis="y", left=False, labelleft=False)

            handles, labels = axs[0, 0].get_legend_handles_labels()
            fig.legend(
                handles,
                labels,
                loc="upper center",
                bbox_to_anchor=cfg.leg_anchor,
                ncols=cfg.leg_ncols or len(metric_names),
                frameon=True,
            )

        gridplot(plots, shape=(len(dataset_names), len(metric_names)), save=path, adjust=adjust)

    def print_table(self, data: Mapping):
        """Print a fixed-width comparison table to stdout."""
        rows = self._format_table(data)
        columns = self._table_columns()
        if not rows:
            print("No comparison results.")
            return

        case_width = max(len("Case"), *(len(row[0]) for row in rows))
        col_widths = [
            max(len(metric), len(dataset), *(len(row[index + 1]) for row in rows))
            for index, (metric, dataset) in enumerate(columns)
        ]

        metric_line = " " * (case_width + 2)
        metric_spans = []
        cursor = 0
        for metric_name in self.metrics:
            count = len(self.dataloaders)
            span = sum(col_widths[cursor:cursor + count]) + 2 * (count - 1)
            metric_spans.append(span)
            metric_line += f"{metric_name:^{span}}  "
            cursor += count

        rule_line = " " * (case_width + 2) + "  ".join("-" * span for span in metric_spans)
        dataset_line = f"{'Case':<{case_width}}  " + "  ".join(
            f"{dataset:>{width}}" for width, (_, dataset) in zip(col_widths, columns)
        )
        data_lines = [
            f"{row[0]:<{case_width}}  " + "  ".join(
                f"{value:>{width}}" for width, value in zip(col_widths, row[1:])
            )
            for row in rows
        ]

        print(metric_line.rstrip())
        print(rule_line.rstrip())
        print(dataset_line.rstrip())
        for line in data_lines:
            print(line.rstrip())

    def write_table(self, data: Mapping, path: str | Path):
        """Write formatted comparison results as a LaTeX tabular file."""
        path = Path(path)
        rows = self._format_table(data)
        body = "\n".join(" & ".join(row) + r" \\" for row in rows)

        if self.latex_template is not None:
            text = self.latex_template.replace("{compare}", body)
        else:
            columns = self._table_columns()
            header = "Case & " + " & ".join(f"{metric} ({dataset})" for metric, dataset in columns) + r" \\"
            col_spec = "l" + "c" * len(columns)
            text = "\n".join([
                rf"\begin{{tabular}}{{{col_spec}}}",
                header,
                r"\hline",
                body,
                r"\end{tabular}",
                "",
            ])

        path.write_text(text, encoding="utf-8")

    def run(self):
        if self.root is not None:
            self.root = Path(self.root)
            self.root.mkdir(exist_ok=True, parents=True)

        # Construct a table with cases in rows and metrics in columns
        tab_results = {}
        distributions = {}
        if self.root is not None:
            tab_file = self.root / self.filename
            if tab_file.exists():
                with tab_file.open("r", encoding="utf-8") as fh:
                    tab_results = yaml.safe_load(fh) or {}
            dist_file = tab_file.with_suffix(".h5")
            if dist_file.exists():
                distributions = load_h5({}, dist_file, jax=False)
        
        num_items = len(self.cases) * len(self.metrics)
        ctxt = alive_bar(num_items) if self.show_progress else _NullProgress()

        with ctxt as bar:
            for case_name, case in self.cases.items():
                bar.text(f"Comparing case: {case_name}")
                case_results = tab_results.setdefault(case_name, {})
                case_distributions = distributions.setdefault(case_name, {})

                params = resolve_orbax_params(case, self.params_template[case_name])

                for metric_name, metric_fn in self.metrics.items():
                    metric_results = case_results.setdefault(metric_name, {})
                    metric_distributions = case_distributions.setdefault(metric_name, {})
                
                    for ds_name, loader in self.dataloaders.items():
                        ds_results = metric_results.setdefault(ds_name, {})

                        has_stats = all(stat_name in ds_results for stat_name in self.stats)
                        has_distribution = ds_name in metric_distributions

                        if (has_stats or has_distribution) and self.write_policy == "error":
                            raise ValueError(
                                f"Results already computed for {case_name}->{metric_name}->{ds_name} "
                                f"and write_policy='error'"
                            )
                        if has_stats and has_distribution and self.write_policy == "reuse":
                            continue

                        if has_distribution and self.write_policy == "reuse":
                            results = jnp.ravel(jnp.asarray(metric_distributions[ds_name]))
                        else:
                            results = [metric_fn(params, data) for data in loader]
                            if not results:
                                raise ValueError(f"No data loaded for dataset '{ds_name}'")
                            results = jnp.ravel(jnp.asarray(results))
                            metric_distributions[ds_name] = np.asarray(jax.device_get(results))

                        for stat_name, stat_fn in self.stats.items():
                            ds_results[stat_name] = float(stat_fn(results))
                        
                    bar()
        
        if self.root is not None:
            tab_file = self.root / self.filename
            with tab_file.open("w", encoding="utf-8") as fh:
                yaml.dump(tab_results, fh, sort_keys=False)
            
            save_h5(distributions, tab_file.with_suffix(".h5"), mode="w")
            self.write_table(tab_results, tab_file.with_suffix(".tex"))

            if self.hist:
                self._plot_histograms(distributions, tab_file.with_suffix(".pdf"))
        
        if self.show_table:
            self.print_table(tab_results)

        return 0
