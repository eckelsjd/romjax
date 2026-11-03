from pathlib import Path
from typing import Annotated

import pytest
import yaml
from pydantic import BeforeValidator, PrivateAttr

import romjax
from romjax.configuration import YamlTemplate
from romjax.runtime import RuntimeContext
from romjax.typing import from_yaml


class RecipeRoutine(romjax.Routine):
    root: Path
    value: str
    constructed: list[Path] = []
    _loaded: str = PrivateAttr(default="")

    def __init__(self, **data):
        type(self).constructed.append(Path(data["root"]))
        super().__init__(**data)

    def run(self) -> int:
        return 0


class ProducerRoutine(romjax.Routine):
    root: Path

    def run(self) -> int:
        self.root.mkdir(parents=True, exist_ok=True)
        (self.root / "data.txt").write_text("fresh", encoding="utf-8")
        return 0


class ConsumerRoutine(romjax.Routine):
    root: Path
    data: str

    def __init__(self, **values):
        values["data"] = Path(values["data"]).read_text(encoding="utf-8")
        super().__init__(**values)

    def run(self) -> int:
        assert self.data == "fresh"
        return 0


class InlineRoutine(romjax.Routine):
    root: Path
    value: str

    def run(self) -> int:
        assert self.value == "parent-value"
        return 0


class FailingConstructorRoutine(romjax.Routine):
    def __init__(self, **values):
        raise RuntimeError("constructor failed")

    def run(self) -> int:
        return 0


class GraphRoutine(romjax.Routine):
    graph: Annotated[dict, BeforeValidator(from_yaml)]
    observed: list[dict] = []

    def run(self) -> int:
        type(self).observed.append(self.graph)
        return 0


class LateSourceRoutine(romjax.Routine):
    root: Path
    source: object
    observed: list[object] = []

    def run(self) -> int:
        type(self).observed.append(romjax.load(self.source))
        return 0


def test_runtime_template_renders_shared_and_artifact_paths(tmp_path: Path) -> None:
    template = YamlTemplate.from_source(
        romjax.YamlSource(
            yaml.compose(
                "shared: \"{{runtime.root}}/data\"\n"
                "artifact: \"{{runtime.artifact_root}}/loss.pdf\"\n"
                "whole: \"{{runtime.root}}\"\n"
            ),
            None,
        )
    )

    source = template.render_runtime(RuntimeContext(tmp_path / "run", tmp_path / "run" / "train"))
    assert romjax.load(source.source) == {
        "shared": str(tmp_path / "run" / "data"),
        "artifact": str(tmp_path / "run" / "train" / "loss.pdf"),
        "whole": str(tmp_path / "run"),
    }


def test_runtime_template_requires_context_for_runtime_reference() -> None:
    template = YamlTemplate.from_source(
        romjax.YamlSource(yaml.compose("path: '{{runtime.root}}/data'"), None)
    )
    with pytest.raises(ValueError, match="runtime.root"):
        template.render_runtime(RuntimeContext(None, None))


def test_runtime_template_rejects_malformed_and_ambiguous_references() -> None:
    context = RuntimeContext(Path("/tmp/run"), Path("/tmp/run/train"))
    malformed = YamlTemplate.from_source(romjax.YamlSource(yaml.compose("path: '{{runtime.root}'"), None))
    with pytest.raises(ValueError, match="Malformed runtime"):
        malformed.render_runtime(context)
    ambiguous = YamlTemplate.from_source(
        romjax.YamlSource(yaml.compose("runtime: {root: user}\npath: '{{runtime.root}}'"), None)
    )
    with pytest.raises(ValueError, match="conflicts"):
        ambiguous.render_runtime(context)


def test_eager_loading_keeps_runtime_mapping_and_static_templates() -> None:
    assert romjax.load("root: literal\nruntime: {root: literal}\nvalue: '{{root}}'") == {
        "root": "literal",
        "runtime": {"root": "literal"},
        "value": "literal",
    }


def test_routine_spec_defers_construction_and_rebuilds_each_run(tmp_path: Path) -> None:
    RecipeRoutine.constructed = []
    spec = romjax.RoutineSpec.from_yaml(
        f"!pd:tests.test_runtime.RecipeRoutine\nroot: child\nvalue: '{{{{runtime.root}}}}'\n"
        f"runtime_root: {tmp_path.as_posix()}/run_{{{{timestamp}}}}\n"
    )
    assert RecipeRoutine.constructed == []
    assert spec.run() == 0
    assert spec.run() == 0
    assert len(RecipeRoutine.constructed) == 2
    assert RecipeRoutine.constructed[0] != RecipeRoutine.constructed[1]


def test_serial_yaml_children_materialize_after_prior_sibling(tmp_path: Path) -> None:
    producer = tmp_path / "producer.yml"
    producer.write_text(
        "!pd:tests.test_runtime.ProducerRoutine\nroot: '{{runtime.root}}/data'\n",
        encoding="utf-8",
    )
    consumer = tmp_path / "consumer.yml"
    consumer.write_text(
        "!pd:tests.test_runtime.ConsumerRoutine\n"
        "root: consumer\n"
        "data: '{{runtime.root}}/data/data.txt'\n",
        encoding="utf-8",
    )
    recipe = tmp_path / "workflow.yml"
    recipe.write_text(
        "!romx:CompositeRoutine\n"
        "executor: {show_progress: false}\n"
        f"runtime_root: {tmp_path.as_posix()}/run_{{{{timestamp}}}}\n"
        f"routines: [{producer.as_posix()}, {consumer.as_posix()}]\n",
        encoding="utf-8",
    )
    spec = romjax.RoutineSpec.from_yaml(recipe)
    assert spec.run() == 0
    assert (spec.resolved_runtime_root / "data" / "data.txt").read_text(encoding="utf-8") == "fresh"


def test_inline_child_keeps_parent_ordinary_references_until_execution(tmp_path: Path) -> None:
    spec = romjax.RoutineSpec.from_yaml(
        f"""!romx:CompositeRoutine
runtime_root: {tmp_path.as_posix()}/run_{{{{timestamp}}}}
executor: {{show_progress: false}}
routine_config: {{extra: {{parent_value: parent-value}}}}
routines:
  - !pd:tests.test_runtime.InlineRoutine
    root: child
    value: "{{{{routine_config.extra.parent_value}}}}"
"""
    )
    assert spec.run() == 0


def test_recipe_removes_timestamp_root_when_construction_fails(tmp_path: Path) -> None:
    spec = romjax.RoutineSpec.from_yaml(
        f"!pd:tests.test_runtime.FailingConstructorRoutine\n"
        f"runtime_root: {tmp_path.as_posix()}/run_{{{{timestamp}}}}\n"
    )
    with pytest.raises(RuntimeError, match="constructor failed"):
        spec.run()
    assert not list(tmp_path.glob("run_*"))


def test_runtime_recipe_preserves_nested_override_template_source(tmp_path: Path, monkeypatch) -> None:
    GraphRoutine.observed = []
    (tmp_path / "graph.yml").write_text(
        "edges:\n  - grid: {shape: [2, 3]}\n"
        "shape: '{{edges.0.grid.shape}}'\n"
        "artifact: '{{runtime.root}}/data'\n",
        encoding="utf-8",
    )
    config = tmp_path / "gen_data.yml"
    config.write_text(
        "!pd:tests.test_runtime.GraphRoutine\n"
        "runtime_root: run_{{timestamp}}\n"
        "graph: !overrides:graph.yml\n"
        "  label: demo\n",
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)
    spec = romjax.RoutineSpec.from_yaml(config)
    assert spec.run() == 0
    assert GraphRoutine.observed[0]["shape"] == [2, 3]
    assert GraphRoutine.observed[0]["artifact"] == str(spec.resolved_runtime_root / "data")

    direct = tmp_path / "direct.yml"
    direct.write_text(
        "!pd:tests.test_runtime.GraphRoutine\n"
        "runtime_root: run_{{timestamp}}\n"
        "graph: graph.yml\n",
        encoding="utf-8",
    )
    direct_spec = romjax.RoutineSpec.from_yaml(direct)
    assert direct_spec.run() == 0
    assert GraphRoutine.observed[1]["artifact"] == str(direct_spec.resolved_runtime_root / "data")


def test_composite_inline_override_child_is_not_double_deferred(tmp_path: Path) -> None:
    child = tmp_path / "child.yml"
    child.write_text(
        "!pd:tests.test_runtime.InlineRoutine\nroot: child\nvalue: parent-value\n",
        encoding="utf-8",
    )
    spec = romjax.RoutineSpec.from_yaml(
        f"""!romx:CompositeRoutine
runtime_root: {tmp_path.as_posix()}/run_{{{{timestamp}}}}
executor: {{show_progress: false}}
routines:
  - !overrides:{child.as_posix()}
    value: parent-value
"""
    )
    assert spec.run() == 0


def test_runtime_context_is_available_to_nested_loads_during_run(tmp_path: Path) -> None:
    LateSourceRoutine.observed = []
    graph = tmp_path / "late.yml"
    graph.write_text("artifact: '{{runtime.root}}/data'\n", encoding="utf-8")
    spec = romjax.RoutineSpec.from_yaml(
        f"""!pd:tests.test_runtime.LateSourceRoutine
runtime_root: {tmp_path.as_posix()}/run_{{{{timestamp}}}}
root: output
source: {graph.as_posix()}
"""
    )
    assert spec.run() == 0
    assert LateSourceRoutine.observed == [{"artifact": str(spec.resolved_runtime_root / "data")}]
