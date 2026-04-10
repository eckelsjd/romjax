from __future__ import annotations

from collections.abc import Callable
from io import StringIO
from pathlib import Path

import pytest

from romjax import YamlLoader
from romjax.graph import ImplicitModel
from romjax.poisson import Poisson2D


def example_forcing(inputs: dict, outputs: dict):
    return int(inputs.get("value", 0))


class CustomModel(ImplicitModel):
    opts: dict[str, int]
    detail: str

    def evaluate(self, *args, **kwargs) -> int:
        return 0

    def solve(self, *args, **kwargs) -> int:
        return 0
    
    def sample_inputs(self, *args, **kwargs):
        return 0
    
    def sample_outputs(self, *args, **kwargs):
        return 0


def _basic_yaml() -> str:
    return (
        "solver: !rox:tests.test_loader.CustomModel\n"
        "  opts: {a: 1, b: 2}\n"
        "  detail: hello\n"
    )

def _assert_round_trip(data: dict) -> None:
    dumped = YamlLoader.dump(data)
    assert dumped is not None
    reloaded = YamlLoader.load(dumped)
    assert type(reloaded["solver"]) is type(data["solver"])

    orig_solver = data["solver"]
    new_solver = reloaded["solver"]
    assert orig_solver.model_dump() == new_solver.model_dump()


def test_basic_model_load_and_dump(tmp_path: Path) -> None:
    yaml_text = _basic_yaml()

    data = YamlLoader.load(yaml_text)
    assert isinstance(data["solver"], CustomModel)
    assert data["solver"].opts["a"] == 1

    data_bytes = YamlLoader.load(yaml_text.encode("utf-8"))
    assert isinstance(data_bytes["solver"], CustomModel)

    fixture_path = Path("tests/fixtures_basic.yml")
    data_file = YamlLoader.load(fixture_path)
    assert isinstance(data_file["solver"], CustomModel)

    with fixture_path.open("r", encoding="utf-8") as fh:
        data_stream = YamlLoader.load(fh)
    assert isinstance(data_stream["solver"], CustomModel)

    dumped = YamlLoader.dump(data)
    assert dumped is not None
    assert "!rox:tests.test_loader.CustomModel" in dumped
    _assert_round_trip(data)

    sio = StringIO()
    assert YamlLoader.dump(data, sio) is None
    assert "!rox:tests.test_loader.CustomModel" in sio.getvalue()

    out_path = tmp_path / "out.yml"
    assert YamlLoader.dump(data, out_path) is None
    assert out_path.exists()


def test_poisson_model_load_and_dump() -> None:
    fixture_path = Path("tests/fixtures_poisson.yml")
    data = YamlLoader.load(fixture_path)
    solver = data["solver"]
    assert isinstance(solver, Poisson2D)
    assert isinstance(solver.forcing, Callable)
    assert solver.config.grid.shape == (50, 50)

    dumped = YamlLoader.dump(data)
    assert dumped is not None
    assert "!rox:romjax.poisson.Poisson2D" in dumped
    assert "!!python/name" in dumped
    _assert_round_trip(data)


def test_custom_model_load_and_dump() -> None:
    yaml_text_colon = (
        "solver: !rox:romjax.poisson.Poisson2D\n"
        "  forcing: !!python/name:tests.test_loader.example_forcing\n"
        "  config:\n"
        "    grid:\n"
        "      shape: [2, 4]\n"
        "      bounds: [[0, 1], [0, 1]]\n"
    )
    data_colon = YamlLoader.load(yaml_text_colon)
    assert isinstance(data_colon["solver"], Poisson2D)
    assert data_colon["solver"].forcing is example_forcing
    _assert_round_trip(data_colon)

    yaml_text_space = (
        "solver: !rox:romjax.poisson.Poisson2D\n"
        "  forcing: !!python/name tests.test_loader.example_forcing\n"
        "  config:\n"
        "    grid:\n"
        "      shape: [2, 4]\n"
        "      bounds: [[0, 1], [0, 1]]\n"
    )
    data_space = YamlLoader.load(yaml_text_space)
    assert isinstance(data_space["solver"], Poisson2D)
    assert data_space["solver"].forcing is example_forcing
    _assert_round_trip(data_space)


def test_yaml_invalid_cases() -> None:
    with pytest.raises(TypeError):
        YamlLoader.load(123)

    with pytest.raises(TypeError):
        YamlLoader.dump({}, 123)

    with pytest.raises(FileNotFoundError):
        YamlLoader.load("tests/does_not_exist.yml")

    with pytest.raises(ValueError):
        YamlLoader.load("solver: !rox:badpath\n  opts: {a: 1}\n  detail: x\n")

    with pytest.raises(TypeError):
        YamlLoader.load("solver: !rox:builtins.dict\n  a: 1\n")
