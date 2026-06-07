import copy
from pathlib import Path
from typing import Annotated, Any, Callable, Iterable, Iterator, Literal, Mapping

import equinox as eqx
import jax
import jax.numpy as jnp
import yaml
from alive_progress import alive_bar
from jaxtyping import PyTree
from pydantic import (
    BaseModel,
    BeforeValidator,
    ConfigDict,
    Field,
    SkipValidation,
    field_validator,
    model_validator,
)

from romjax.data_gen import DataLoader
from romjax.graph import FunctionGraph
from romjax.routine import Routine
from romjax.train import GraphLoss
from romjax.tree import UnaryOperator, get_unary_operator
from romjax.typing import OrbaxParams, from_yaml, resolve_graph_refs
from romjax.utils import _NullProgress

type SUPPORTED_POLICIES = Literal["reuse", "overwrite", "error"]

__all__ = ["Compare"]


def _iter_loader_once(loader: Iterable[Any]) -> Iterator[Any]:
    """Iterate a loader once, using one epoch for file-backed ``DataLoader`` instances."""
    if isinstance(loader, DataLoader):
        return loader._iter_datasets(max_epochs=1)
    return iter(loader)
        

class CompareTableConfig(BaseModel):
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

    model_config = ConfigDict(arbitrary_types_allowed=True, validate_default=True)

    dataloaders: SkipValidation[Mapping[str, Iterable[Any]]]
    metrics: Mapping[str, Callable[[PyTree, Any], float]]

    stats: Mapping[str, UnaryOperator] = Field(default_factory=lambda: ["mean", "std"])
    latex_template: str | None = None
    col_format: Mapping[str, str] = r"{mean:5.3f} ({std:5.3f})"
    filename: str = "compare_table.yml"

    @field_validator("stats", mode="before")
    @classmethod
    def _from_plain_list(cls, value):
        if not isinstance(value, Mapping):
            if not isinstance(value, tuple | list):
                value = [value]
            if isinstance(value, tuple | list):
                value = {str(v): v for v in value}
        
        if isinstance(value, Mapping):
            value = {k: get_unary_operator(v) for k, v in value.items()}

        return value
    
    @field_validator("col_format", mode="before")
    @classmethod
    def _single_col_format(cls, value, info):
        if not isinstance(value, Mapping):
            value = {metric_name: value for metric_name in info.data["metrics"]}

        return value
    
    @model_validator(mode="after")
    def _jit_metrics(self):
        for name in list(self.metrics.keys()):
            metric = self.metrics[name]
            if isinstance(metric, GraphLoss):
                if metric.graph is not None:
                    self.metrics[name] = eqx.filter_jit(metric.__call__)
            else:
                self.metrics[name] = eqx.filter_jit(metric)
        
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
    
    def _bind_graph(self, graph: FunctionGraph):
        """Bind a graph to any metrics that need it."""
        if graph is not None:
            for name, metric_fn in list(self.metrics.items()):
                if isinstance(metric_fn, GraphLoss):
                    if metric_fn.graph is None:
                        metric_fn.graph = graph
                        metric_fn._set_default_datasets()
                    self.metrics[name] = eqx.filter_jit(metric_fn.__call__)
                elif hasattr(metric_fn, "graph"):
                    if metric_fn.graph is None:
                        metric_fn.graph = graph


class Compare(Routine):
    """
    Routine for comparing models via orbax checkpoints from Train.
    """
    cases: Mapping[str, OrbaxParams]

    table: CompareTableConfig | None = None
    print_table: bool = True
    show_progress: bool = True

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
            for case in self.cases:
                self.params_template[case] = resolve_graph_refs(self.params_template[case], self.graph)
            
            if self.table is not None:
                self.table._bind_graph(self.graph)
        
        # Initialize param templates just like in train
        for case in self.cases:
            sample_fn = getattr(self.params_template[case], "sample", None)
            if callable(sample_fn):
                self.params_template[case] = sample_fn(jax.random.key(0))
        
        return self
    
    def run(self):
        if self.root is not None:
            self.root = Path(self.root)
            self.root.mkdir(exist_ok=True, parents=True)

        # Construct a table with cases in rows and metrics in columns
        if self.table is not None:
            tab_results = {}
            if self.root is not None:
                tab_file = self.root / self.table.filename
                if tab_file.exists():
                    with tab_file.open("r", encoding="utf-8") as fh:
                        tab_results = yaml.safe_load(fh)
            
            num_items = len(self.cases) * len(self.table.metrics)
            ctxt = alive_bar(num_items) if self.show_progress else _NullProgress()

            with ctxt as bar:
                for case_name, case in self.cases.items():
                    bar.text(f"Comparing case: {case_name}")
                    case_results = tab_results.setdefault(case_name, {})

                    params = case.resolve_params(self.params_template[case_name])

                    for metric_name, metric_fn in self.table.metrics.items():
                        metric_results = case_results.setdefault(metric_name, {})
                    
                        for ds_name, loader in self.table.dataloaders.items():
                            ds_results = metric_results.setdefault(ds_name, {})

                            has_stats = all(stat_name in ds_results for stat_name in self.table.stats)

                            if has_stats and self.write_policy == "error":
                                raise ValueError(f"Stats already computed for {case_name}->{metric_name}->{ds_name} "
                                                f"and write_policy='error'")
                            if has_stats and self.write_policy == "reuse":
                                continue

                            values = [metric_fn(params, single_data) for single_data in _iter_loader_once(loader)]
                            if not values:
                                raise ValueError(f"No data loaded for dataset '{ds_name}'")
                            results = jnp.asarray(values)

                            for stat_name, stat_fn in self.table.stats.items():
                                ds_results[stat_name] = float(stat_fn(results))
                            
                        bar()
            
            if self.root is not None:
                tab_file = self.root / self.table.filename
                with tab_file.open("w", encoding="utf-8") as fh:
                    yaml.dump(tab_results, fh, sort_keys=False)
                
                self.table.write_table(tab_results, tab_file.with_suffix(".tex"))
            
            if self.print_table:
                self.table.print_table(tab_results)

        return 0
    
