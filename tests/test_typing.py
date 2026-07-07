from pathlib import Path

import lineax as lx
import optax
import optimistix as optx
import pytest
from orbax.checkpoint import v1 as ocp
from pydantic import BaseModel, TypeAdapter, ValidationError, field_validator, model_validator

from romjax import DictModel, YamlLoader
from romjax.typing import ListModel, ThirdPartyType, require_type

GradientTransformation = ThirdPartyType(default_modules="optax")


class GradientConfig(BaseModel):
    transform: GradientTransformation

    @field_validator("transform")
    @classmethod
    def _require_gradient_transformation(cls, value):
        return require_type(optax.GradientTransformation, value)


def test_third_party_type_constructs_and_round_trips_nested_optax_specs():
    data = {
        "name": "optax.chain",
        "args": [
            {"name": "optax.clip_by_global_norm", "args": [1.0]},
            {"name": "optax.scale_by_adam"},
            {
                "name": "optax.scale_by_schedule",
                "args": [
                    {
                        "name": "optax.exponential_decay",
                        "kwargs": {
                            "init_value": 0.01,
                            "transition_steps": 1000,
                            "decay_rate": 0.99,
                        },
                    }
                ],
            },
            {"name": "optax.scale", "args": [-1.0]},
        ],
    }

    adapter = TypeAdapter(ThirdPartyType(default_modules="optax"))
    transform = adapter.validate_python(data)
    dumped = adapter.dump_python(transform)

    assert isinstance(transform, optax.GradientTransformation)
    assert dumped == data


def test_third_party_type_preserves_default_modules_for_nested_specs():
    data = {
        "name": "chain",
        "args": [
            {"name": "adam", "args": [0.1]},
        ],
    }

    adapter = TypeAdapter(ThirdPartyType(default_modules="optax"))
    transform = adapter.validate_python(data)
    dumped = adapter.dump_python(transform)

    assert isinstance(transform, optax.GradientTransformation)
    assert dumped == data


def test_third_party_type_accepts_partially_validated_nested_objects():
    adapter = TypeAdapter(ThirdPartyType(default_modules="optax"))
    clip = adapter.validate_python({"name": "optax.clip_by_global_norm", "args": [1.0]})
    schedule = adapter.validate_python(
        {
            "name": "optax.exponential_decay",
            "kwargs": {
                "init_value": 0.01,
                "transition_steps": 1000,
                "decay_rate": 0.99,
            },
        }
    )

    transform = adapter.validate_python(
        {
            "name": "optax.chain",
            "args": [
                clip,
                {"name": "optax.scale_by_schedule", "args": [schedule]},
                {"name": "optax.scale", "args": [-1.0]},
            ],
        }
    )
    dumped = adapter.dump_python(transform)

    assert isinstance(transform, optax.GradientTransformation)
    assert dumped["args"][0] == {"name": "optax.clip_by_global_norm", "args": [1.0]}
    assert dumped["args"][1]["args"][0] == {
        "name": "optax.exponential_decay",
        "kwargs": {
            "init_value": 0.01,
            "transition_steps": 1000,
            "decay_rate": 0.99,
        },
    }


def test_third_party_type_supports_custom_default_modules():
    adapter = TypeAdapter(ThirdPartyType(default_modules=ocp.training.save_decision_policies.__name__))
    policy = adapter.validate_python(
        {
            "name": "AnySavePolicy",
            "args": [
                [
                    {"name": "FixedIntervalPolicy", "args": [3]},
                    {"name": "SpecificStepsPolicy", "args": [[5]]},
                ]
            ],
        }
    )
    dumped = adapter.dump_python(policy)

    assert isinstance(policy, ocp.training.save_decision_policies.SaveDecisionPolicy)
    assert dumped["name"] == "AnySavePolicy"
    assert dumped["args"][0][0] == {"name": "FixedIntervalPolicy", "args": [3]}


def test_gradient_transformation_type_rejects_wrong_object_type():
    with pytest.raises(ValidationError):
        GradientConfig.model_validate({"transform": {"name": "optax.exponential_decay", "args": [0.1, 10, 0.9]}})


def test_gradient_transformation_model_round_trip():
    data = {
        "transform": {
            "name": "optax.chain",
            "args": [
                {"name": "optax.clip_by_global_norm", "args": [1.0]},
                {"name": "optax.scale_by_adam"},
                {"name": "optax.scale", "args": [-1.0]},
            ],
        }
    }

    config = GradientConfig.model_validate(data)
    dumped = config.model_dump()

    assert isinstance(config.transform, optax.GradientTransformation)
    assert dumped == data


class CustomSettings(DictModel, extra='forbid'):
    one: str
    two: str
    four: float = 1.1


def test_custom_settings():
    """Make sure subclasses of DictModel work"""
    settings = CustomSettings(one='1', two='2')

    # get, set, del, iter, len
    assert settings['four'] == 1.1
    settings['four'] = '3'
    assert settings['four'] == 3
    assert isinstance(settings['four'], float)
    del settings['one']
    
    with pytest.raises(KeyError):
        _ = settings['one']
    
    settings['one'] = '1'

    keys = []
    values = []
    for k, v in settings.items():
        keys.append(k)
        values.append(v)
    
    assert all([k in settings.keys() for k in keys])
    assert all([settings[k] == v for k,v in zip(keys, values)])
    
    assert len(keys) == len(settings) == 3
    assert 'one' in settings
    assert settings.get('five') is None

    with pytest.raises(ValidationError):
        settings['three'] = 'forbidden'


def test_plain_dict():
    """Make sure DictModel works without predefined schema."""
    d = {'a': 1, 'b': 2, 'c': 3}
    model = DictModel(**d)
    model.e = 'e'
    model['f'] = 42
    del model['a']
    model.update({'g': 'helo', 'h': 242.3})

    del d['a']
    d.update({'g': 'helo', 'h': 242.3, 'e': 'e', 'f': 42})

    assert len(d) == len(model)
    assert d == dict(model)


def test_from_yaml():
    """Make sure validation for DictModel works from file."""
    fixture_path = Path("tests/fixtures_typing.yml")
    data = YamlLoader.load(fixture_path)

    assert isinstance(data['settings'], CustomSettings)
    assert isinstance(data['configs'], DictModel)

    assert isinstance(data['settings']['four'], float)

    data['configs'].update({'1': 1, '2': 2})
    assert all(k in data['configs'].keys() for k in ['1', '2', 'one', 'two'])

    with pytest.raises(ValidationError):
        data['settings']['3'] = 'three'  # extras are forbidden


class SolverConfig(BaseModel):
    solver: ThirdPartyType


class LinearSolverConfig(BaseModel):
    solver: ThirdPartyType


class ListNode(BaseModel):
    name: str

    @model_validator(mode="before")
    @classmethod
    def _from_str(cls, value):
        if isinstance(value, str):
            return {"name": value}
        return value


class ListEdge(BaseModel):
    src: str
    dst: str

    @model_validator(mode="before")
    @classmethod
    def _from_tuple(cls, value):
        if isinstance(value, tuple) and len(value) == 2:
            return {"src": value[0], "dst": value[1]}
        return value


class NodeListModel(ListModel[ListNode]):
    pass


class EdgeListModel(ListModel[ListEdge]):
    pass


def test_module_object_nested_validation():
    data = {
        "name": "optimistix.Newton",
        "kwargs": {
            "rtol": 1e-3,
            "atol": 1e-6,
            "linear_solver": {
                "name": "lineax.CG",
                "kwargs": {
                    "rtol": 1e-2,
                    "atol": 1e4,
                },
            },
        },
    }

    cfg = SolverConfig.model_validate({"solver": data})

    assert cfg.solver.__class__.__name__ == "Newton"
    assert cfg.solver.linear_solver.__class__.__name__ == "CG"

    dumped = cfg.model_dump()
    assert dumped["solver"]["name"] == "optimistix.Newton"
    assert dumped["solver"]["kwargs"]["linear_solver"]["name"] == "lineax.CG"
    assert dumped["solver"]["kwargs"]["linear_solver"]["kwargs"]["rtol"] == 1e-2


def test_module_object_external_serialization():
    solver = optx.Newton(
        rtol=1e-3,
        atol=1e-6,
        linear_solver=lx.CG(rtol=1e-2, atol=1e4),
    )
    cfg = SolverConfig(solver=solver)
    dumped = cfg.model_dump()

    assert dumped["solver"]["name"] == "optimistix.Newton"
    assert isinstance(dumped["solver"]["kwargs"], dict)
    if "linear_solver" in dumped["solver"]["kwargs"]:
        assert dumped["solver"]["kwargs"]["linear_solver"]["name"] == "lineax.CG"
    
    solver = lx.CG(rtol=1e-2, atol=1e4)
    cfg = LinearSolverConfig(solver=solver)
    dumped = cfg.model_dump()

    assert dumped["solver"]["name"] == "lineax.CG"
    assert isinstance(dumped["solver"]["kwargs"], dict)
    assert "rtol" in dumped["solver"]["kwargs"]
    assert "atol" in dumped["solver"]["kwargs"]
    

def test_list_model():
    nodes = NodeListModel(["a", "b", "c"])
    edges = EdgeListModel([("a", "b"), ("b", "c")])

    assert list(nodes.keys()) == ["a", "b", "c"]
    assert nodes[0].name == "a"
    assert [n.name for n in nodes[1:]] == ["b", "c"]
    assert [n.name for n in nodes[[2, 0]]] == ["c", "a"]
    assert nodes["b"].name == "b"
    assert isinstance(edges[0], ListEdge)
    assert edges[0].src == "a"
    assert edges[0].dst == "b"

    nodes[1] = "beta"
    nodes["d"] = {"name": "d"}
    nodes.append("e")
    assert list(nodes.keys()) == ["a", "b", "c", "d", "e"]
    assert nodes[1].name == "beta"
    assert all(isinstance(node, ListNode) for node in nodes.values())

    with pytest.raises(ValidationError):
        nodes["bad"] = {"not_name": "x"}
    with pytest.raises(ValidationError):
        edges["bad"] = {"src": "x"}

    del nodes[0]
    assert list(nodes.keys()) == ["b", "c", "d", "e"]
    del nodes[[1, 2]]
    assert list(nodes.keys()) == ["b", "e"]

    assert "_adapter" not in nodes.model_dump()
    assert "_adapter" not in repr(nodes)

    yaml_text = (
        "nodes: !romx:tests.test_typing.NodeListModel\n"
        "  a: a\n"
        "  b: {name: b}\n"
        "edges: !romx:tests.test_typing.EdgeListModel\n"
        "  ab: {src: a, dst: b}\n"
        "  bc: {src: b, dst: c}\n"
    )
    data = YamlLoader.load(yaml_text)
    assert isinstance(data["nodes"], NodeListModel)
    assert isinstance(data["edges"], EdgeListModel)
    assert [n.name for n in data["nodes"].values()] == ["a", "b"]
    assert [(e.src, e.dst) for e in data["edges"].values()] == [("a", "b"), ("b", "c")]

    dumped = YamlLoader.dump({"nodes": nodes, "edges": edges})
    reloaded = YamlLoader.load(dumped)
    assert isinstance(reloaded["nodes"], NodeListModel)
    assert isinstance(reloaded["edges"], EdgeListModel)
    assert [n.name for n in reloaded["nodes"].values()] == ["beta", "e"]
    assert [(e.src, e.dst) for e in reloaded["edges"].values()] == [("a", "b"), ("b", "c")]


def test_list_model_serialization_round_trip():
    nodes = NodeListModel(["alpha", {"name": "beta"}])

    dumped = nodes.model_dump()
    assert dumped == [{"name": "alpha"}, {"name": "beta"}]

    reloaded = NodeListModel(dumped)
    assert isinstance(reloaded, NodeListModel)
    assert [node.name for node in reloaded.values()] == ["alpha", "beta"]
