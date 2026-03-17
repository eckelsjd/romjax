from pathlib import Path

import lineax as lx
import optimistix as optx
import pytest
from pydantic import ValidationError, BaseModel

from romtools import YamlLoader
from romtools.typing import DictModel, LxObject, OptxObject


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
    solver: OptxObject


class LinearSolverConfig(BaseModel):
    solver: LxObject


def test_module_object_nested_validation():
    data = {
        "name": "Newton",
        "opts": {
            "rtol": 1e-3,
            "atol": 1e-6,
            "linear_solver": {
                "name": "CG",
                "opts": {
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
    assert dumped["solver"]["name"] == "Newton"
    assert dumped["solver"]["opts"]["linear_solver"]["name"] == "CG"
    assert dumped["solver"]["opts"]["linear_solver"]["opts"]["rtol"] == 1e-2


def test_module_object_external_serialization():
    solver = optx.Newton(
        rtol=1e-3,
        atol=1e-6,
        linear_solver=lx.CG(rtol=1e-2, atol=1e4),
    )
    cfg = SolverConfig(solver=solver)
    dumped = cfg.model_dump()

    assert dumped["solver"]["name"] == "Newton"
    assert isinstance(dumped["solver"]["opts"], dict)
    if "linear_solver" in dumped["solver"]["opts"]:
        assert dumped["solver"]["opts"]["linear_solver"]["name"] == "CG"


def test_module_object_external_serialization_lx():
    solver = lx.CG(rtol=1e-2, atol=1e4)
    cfg = LinearSolverConfig(solver=solver)
    dumped = cfg.model_dump()

    assert dumped["solver"]["name"] == "CG"
    assert isinstance(dumped["solver"]["opts"], dict)
    assert "rtol" in dumped["solver"]["opts"]
    assert "atol" in dumped["solver"]["opts"]
