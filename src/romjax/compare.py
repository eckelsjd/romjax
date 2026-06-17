import copy
from pathlib import Path
from typing import Annotated, Any, Callable, Iterable, Iterator, Literal, Mapping

import equinox as eqx
import jax
import jax.numpy as jnp
import yaml
from alive_progress import alive_bar
from jaxtyping import PyTree
from orbax.checkpoint import v1 as ocp
from pydantic import (
    BaseModel,
    BeforeValidator,
    ConfigDict,
    Field,
    PrivateAttr,
    SkipValidation,
    field_validator,
    model_validator,
)

from romjax.data_gen import DataLoader
from romjax.graph import FunctionGraph
from romjax.routine import Routine
from romjax.train import GraphLoss
from romjax.tree import UnaryOperator, get_unary_operator
from romjax.typing import from_yaml, resolve_graph_refs
from romjax.utils import _NullProgress

type SUPPORTED_POLICIES = Literal["reuse", "overwrite", "error"]

__all__ = ["CompareOrbax", "CompareTable", "OrbaxParams"]


class OrbaxParams(BaseModel):
    """Utility for loading params from orbax checkpoints."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    params: PyTree | str | Path
    _resolved_params: PyTree | None = PrivateAttr(default=None)

    @model_validator(mode="before")
    @classmethod
    def _from_plain_params(cls, value):
        if not isinstance(value, OrbaxParams):
            if isinstance(value, Mapping) and "params" in value:
                return value
            return {"params": value}
        return value

    def resolve_params(self, template: PyTree | None = None) -> PyTree | None:
        """Load parameters from orbax using a template."""
        if self._resolved_params is not None:
            return self._resolved_params
        
        if isinstance(self.params, str | Path):
            with ocp.training.Checkpointer(Path(self.params).absolute()) as ckptr:
                if ckptr.latest is not None:
                    if template is not None:
                        dynamic_params, static_params = eqx.partition(template, eqx.is_array)
                        _loaded = ckptr.load_checkpointables(abstract_checkpointables={"params": dynamic_params})
                        params = eqx.combine(_loaded["params"], static_params)
                    else:
                        params = ckptr.load_checkpointables()["params"]

                    self._resolved_params = params
                    return params
            
            return None

        else:
            return self.params


class CompareOrbax(Routine):
    """
    Routine for comparing models via orbax checkpoints from `Train`.
    """
    cases: Mapping[str, OrbaxParams]

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
        
        # Initialize param templates just like in train
        for case in self.cases:
            sample_fn = getattr(self.params_template[case], "sample", None)
            if callable(sample_fn):
                self.params_template[case] = sample_fn(jax.random.key(0))
        
        return self


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

    stats: Mapping[str, UnaryOperator] = Field(default_factory=lambda: ["mean", "std"])
    latex_template: str | None = None
    col_format: Mapping[str, str] = r"{mean:5.3f} ({std:5.3f})"
    filename: str = "compare_table.yml"
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
            value = {k: get_unary_operator(v) for k, v in value.items()}

        return value
    
    @field_validator("col_format", mode="before")
    @classmethod
    def _single_col_format(cls, value, info):
        if not isinstance(value, Mapping):
            value = {metric_name: value for metric_name in info.data["metrics"]}

        return value

    @model_validator(mode="after")
    def _bind_graph_and_jit_metrics(self):
        """Bind a graph to any metrics that need it and jit them."""
        for name, metric_fn in list(self.metrics.items()):
            if isinstance(metric_fn, GraphLoss):
                if metric_fn.graph is None:
                    metric_fn.graph = self.graph
                    metric_fn._set_default_datasets()
                self.metrics[name] = eqx.filter_jit(metric_fn.__call__)
            else:
                self.metrics[name] = eqx.filter_jit(metric_fn)
        
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

    @staticmethod
    def _iter_sample_payloads(batch: Any) -> Iterator[Any]:
        """Yield sample-level payloads from a loader batch when possible.

        ``DataLoader`` yields per-dataset batches, but file-backed datasets commonly return a list of individual
        sample pytrees. In that case, CompareTable should score each sample independently rather than handing the
        whole list to a metric that expects a single sample.
        """
        if isinstance(batch, Mapping) and batch:
            values = list(batch.values())
            if all(isinstance(value, (list, tuple)) for value in values):
                lengths = {len(value) for value in values}
                if len(lengths) == 1:
                    batch_size = lengths.pop()
                    for index in range(batch_size):
                        yield {key: value[index] for key, value in batch.items()}
                    return

        if isinstance(batch, (list, tuple)):
            yield from batch
            return

        yield batch

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
        def _iter_loader_once(loader: Iterable[Any]) -> Iterator[Any]:
            """Iterate a loader once, using one epoch for file-backed ``DataLoader`` instances."""
            if isinstance(loader, DataLoader):
                return loader._iter_datasets(max_epochs=1)
            return iter(loader)
        
        if self.root is not None:
            self.root = Path(self.root)
            self.root.mkdir(exist_ok=True, parents=True)

        # Construct a table with cases in rows and metrics in columns
        tab_results = {}
        if self.root is not None:
            tab_file = self.root / self.filename
            if tab_file.exists():
                with tab_file.open("r", encoding="utf-8") as fh:
                    tab_results = yaml.safe_load(fh)
        
        num_items = len(self.cases) * len(self.metrics)
        ctxt = alive_bar(num_items) if self.show_progress else _NullProgress()

        with ctxt as bar:
            for case_name, case in self.cases.items():
                bar.text(f"Comparing case: {case_name}")
                case_results = tab_results.setdefault(case_name, {})

                params = case.resolve_params(self.params_template[case_name])

                for metric_name, metric_fn in self.metrics.items():
                    metric_results = case_results.setdefault(metric_name, {})
                
                    for ds_name, loader in self.dataloaders.items():
                        ds_results = metric_results.setdefault(ds_name, {})

                        has_stats = all(stat_name in ds_results for stat_name in self.stats)

                        if has_stats and self.write_policy == "error":
                            raise ValueError(f"Stats already computed for {case_name}->{metric_name}->{ds_name} "
                                            f"and write_policy='error'")
                        if has_stats and self.write_policy == "reuse":
                            continue

                        values = [
                            metric_fn(params, single_data)
                            for batch in _iter_loader_once(loader)
                            for single_data in self._iter_sample_payloads(batch)
                        ]
                        if not values:
                            raise ValueError(f"No data loaded for dataset '{ds_name}'")
                        results = jnp.asarray(values)

                        for stat_name, stat_fn in self.stats.items():
                            ds_results[stat_name] = float(stat_fn(results))
                        
                    bar()
        
        if self.root is not None:
            tab_file = self.root / self.filename
            with tab_file.open("w", encoding="utf-8") as fh:
                yaml.dump(tab_results, fh, sort_keys=False)
            
            self.write_table(tab_results, tab_file.with_suffix(".tex"))
        
        if self.show_table:
            self.print_table(tab_results)

        return 0
