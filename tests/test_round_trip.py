"""Configuration and worker round trips for the built-in Poisson graph."""

from pathlib import Path
from typing import Annotated

import jax
import numpy as np
import pytest
from pydantic import BaseModel, ConfigDict, Field, PlainSerializer, RootModel, field_serializer, model_serializer

import romjax
from romjax.data_gen import DataGeneration, GenImplicitModel, _GenerationWorker
from romjax.graph import Node, NodeList
from romjax.pde import ConstantForcing, ForcingCallable, ImplicitAffine
from romjax.storage import H5Store
from romjax.typing import to_module_spec

FIXTURE = Path(__file__).with_name("fixtures_poisson_graph.yml")


@pytest.fixture(autouse=True)
def cpu_device():
    """Keep reference solves on the same backend as spawned process/MPI workers."""
    with jax.default_device(jax.devices("cpu")[0]):
        yield


class Child(BaseModel):
    value: int = 1


type SerializedChild = Annotated[Child, PlainSerializer(lambda child: {"value": child.value + 10})]


class SerializedFields(BaseModel):
    decorated: Child = Field(default_factory=Child)
    annotated: SerializedChild = Field(default_factory=Child)
    excluded: Child = Field(default_factory=Child, exclude=True)

    @field_serializer("decorated")
    def serialize_decorated(self, child: Child) -> dict[str, int]:
        return {"value": child.value + 20}


class InheritedSerializedFields(SerializedFields):
    pass


class SerializedModel(BaseModel):
    child: Child = Field(default_factory=Child)

    @model_serializer
    def serialize_model(self) -> dict[str, dict[str, int]]:
        return {"child": {"value": self.child.value + 30}}


class RootChildren(RootModel[dict[str, Child]]):
    pass


class UnionFields(BaseModel):
    value: SerializedChild | int
    sequence: list[SerializedChild] | str


class DerivedChild(Child):
    extra: int = 2


class AliasedFields(BaseModel):
    model_config = ConfigDict(serialize_by_alias=True, validate_by_name=True)
    child: Child = Field(alias="renamed")


class SerializedNodeList(NodeList):
    @model_serializer
    def serialize_nodes(self) -> dict[str, Node]:
        return {"renamed": self[0]}


@pytest.mark.parametrize("preserve", [False, True])
def test_dump_respects_explicit_serializers(preserve):
    options = {"_preserve_model_types": preserve}
    dumped = romjax.YamlLoader.dump(InheritedSerializedFields(), **options)
    restored = romjax.load(dumped)
    assert restored.decorated.value == 21
    assert restored.annotated.value == 11
    assert "excluded:" not in dumped
    restored_model = romjax.load(romjax.YamlLoader.dump(SerializedModel(), **options))
    assert restored_model.child.value == 31


def test_dump_preserves_root_model_children():
    original = RootChildren({"a": DerivedChild()})
    restored = romjax.load(romjax.YamlLoader.dump(original))
    assert type(restored.root["a"]) is DerivedChild
    assert restored.root["a"].extra == 2


def test_dump_respects_union_branch_serializers():
    original = UnionFields(value=Child(), sequence=[Child()])
    restored = romjax.load(romjax.YamlLoader.dump(original))
    assert restored.value.value == 11
    assert restored.sequence[0].value == 11


def test_dump_respects_list_model_serializer_override():
    original = SerializedNodeList({"old": Node(name="sample")})
    restored = romjax.load(romjax.YamlLoader.dump(original))
    assert list(restored) == ["renamed"]
    assert restored[0].name == "sample"


def test_dump_preserves_aliased_field_model_type():
    original = AliasedFields(child=DerivedChild())
    restored = romjax.load(romjax.YamlLoader.dump(original))
    assert type(restored.child) is DerivedChild
    assert restored.child.extra == 2


@pytest.mark.parametrize("schema", [ForcingCallable.Inputs, ForcingCallable.Outputs, ConstantForcing.Inputs])
def test_nested_model_qualified_tags(schema):
    original = schema()
    text = romjax.YamlLoader.dump(original)
    assert f"{schema.__module__}.{schema.__qualname__}" in text
    assert type(romjax.load(text)) is schema


@pytest.mark.parametrize("value", [None, (), [], ("extra",), [("inputs", "extra")]])
def test_implicit_affine_additional_inputs_round_trip(value):
    original = ImplicitAffine(additional_inputs=value)
    restored = ImplicitAffine.model_validate(original.model_dump())
    assert restored.additional_inputs == original.additional_inputs
    assert romjax.load(romjax.YamlLoader.dump(original)).additional_inputs == original.additional_inputs


@pytest.mark.parametrize("preserve", [False, True])
def test_list_model_preserves_storage_keys_and_order(preserve):
    original = NodeList({"z": Node(name="first"), "a": Node(name="second")})
    restored = romjax.load(romjax.YamlLoader.dump(original, _preserve_model_types=preserve))
    assert list(restored) == ["z", "a"]
    assert [node.name for node in restored.values()] == ["first", "second"]
    assert restored[0].name == "first"


def poisson_graph(size=8):
    """Load the standalone fixture, resolving templates at the requested grid size."""
    text = FIXTURE.read_text().replace("[64, 64]", f"[{size}, {size}]")
    return romjax.load(text)


@pytest.mark.parametrize("size", [8, 64])
def test_poisson_graph_round_trip(size):
    graph = poisson_graph(size)
    before = graph.edges["Fh"]
    if size == 8:
        key = jax.random.key(4)
        inputs = before.sample_inputs(key)
        outputs = before.solve(inputs)
        residuals = before.evaluate(inputs, outputs)

    for preserve in (False, True):
        text = romjax.YamlLoader.dump(graph, _preserve_model_types=preserve)
        restored = romjax.load(text)
        assert list(restored.nodes) == list(graph.nodes)
        assert list(restored.edges) == list(graph.edges)
        for name, edge in graph.edges.items():
            other = restored.edges[name]
            assert type(other) is type(edge)
            assert other.model_dump() == edge.model_dump()
            assert other.source is restored.nodes[edge.source.name]
            assert other.target is restored.nodes[edge.target.name]
        after = restored.edges["Fh"]
        assert to_module_spec(after.solver.solver) == to_module_spec(before.solver.solver)
        assert type(after.solver.preconditioner) is type(before.solver.preconditioner)
        # Defaults use their declared mapping serializers; grid coordinates are derived.
        assert "!pd:romjax.pde.Inputs" not in text
        assert "coords:" not in text
        if size == 8:
            other_inputs = after.sample_inputs(key)
            for left, right in zip(jax.tree.leaves(inputs), jax.tree.leaves(other_inputs), strict=True):
                np.testing.assert_array_equal(left, right)
            other_outputs = after.solve(other_inputs)
            for left, right in zip(jax.tree.leaves(outputs), jax.tree.leaves(other_outputs), strict=True):
                np.testing.assert_allclose(left, right, rtol=1e-5, atol=1e-6)
            other_residuals = after.evaluate(other_inputs, other_outputs)
            for left, right in zip(jax.tree.leaves(residuals), jax.tree.leaves(other_residuals), strict=True):
                np.testing.assert_allclose(left, right, rtol=1e-5, atol=1e-6)


def test_poisson_remote_preparation(tmp_path):
    generator = GenImplicitModel(
        graph=poisson_graph(), input_samples=1, outputs_per_input=0, input_seed=1, output_seed=2
    )
    prepared = _GenerationWorker(generator, tmp_path / "Fh").prepare(remote=True)
    assert prepared.generator is None
    restored = romjax.load(prepared.configuration)
    assert list(restored.graph.edges) == ["Fh", "Tz", "Tr", "Fl"]
    assert restored.graph.edges["Fl"].additional_inputs is None


def test_poisson_process_generation_matches_serial(tmp_path, monkeypatch):
    monkeypatch.setenv("JAX_PLATFORMS", "cpu")
    roots = [tmp_path / "serial", tmp_path / "process"]
    for root, backend in zip(roots, ["serial", "process"], strict=True):
        executor = {"kind": backend, "show_progress": False}
        if backend == "process":
            executor["max_workers"] = 1
        generation = DataGeneration(
            root=root,
            graph=poisson_graph(),
            executor=executor,
            datasets={"Fh": GenImplicitModel(input_samples=2, outputs_per_input=0, input_seed=1, output_seed=2)},
        )
        assert generation.run() == 0
    stores = [H5Store(root / "Fh") for root in roots]
    refs = [store.discover() for store in stores]
    assert [ref.identity for ref in refs[0]] == [ref.identity for ref in refs[1]]
    for before, after in zip(stores[0].read_batch(refs[0]), stores[1].read_batch(refs[1]), strict=True):
        for left, right in zip(jax.tree.leaves(before), jax.tree.leaves(after), strict=True):
            np.testing.assert_allclose(left, right, rtol=1e-5, atol=1e-6)
