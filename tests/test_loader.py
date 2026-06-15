from __future__ import annotations

from io import StringIO
from pathlib import Path

import pytest

from romjax import YamlLoader
from romjax.model import ImplicitModel
from romjax.poisson import GaussianForcing, Poisson2D
from romjax.rng import Distribution, NearSolutionSampler


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
        "solver: !romx:tests.test_loader.CustomModel\n"
        "  opts: {a: 1, b: 2}\n"
        "  detail: hello\n"
    )

def _assert_round_trip(data: dict) -> None:
    dumped = YamlLoader.dump(data)
    assert dumped is not None
    reloaded = YamlLoader.load(dumped)
    assert type(reloaded["solver"]) is type(data["solver"])
    redumped = YamlLoader.dump(reloaded)
    assert redumped == dumped


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
    assert "!pd:tests.test_loader.CustomModel" in dumped
    _assert_round_trip(data)

    sio = StringIO()
    assert YamlLoader.dump(data, sio) is None
    assert "!pd:tests.test_loader.CustomModel" in sio.getvalue()

    out_path = tmp_path / "out.yml"
    assert YamlLoader.dump(data, out_path) is None
    assert out_path.exists()


def test_poisson_model_load_and_dump() -> None:
    fixture_path = Path("tests/fixtures_poisson.yml")
    data = YamlLoader.load(fixture_path)
    solver = data["solver"]
    assert isinstance(solver, Poisson2D)
    assert isinstance(solver.forcing, GaussianForcing)
    assert solver.grid.shape == (8, 8)
    assert solver.forcing.inputs_default["A0"] == 0.5

    dumped = YamlLoader.dump(data)
    assert dumped is not None
    assert "!romx:romjax.poisson.Poisson2D" in dumped
    _assert_round_trip(data)


def test_custom_model_load_and_dump() -> None:
    yaml_text_colon = (
        "solver: !romx:romjax.poisson.Poisson2D\n"
        "  forcing:\n"
        "    callable: !!python/name:tests.test_loader.example_forcing\n"
        "    inputs_default:\n"
        "      value: 4\n"
        "  grid:\n"
        "    shape: [2, 4]\n"
        "    bounds: [[0, 1], [0, 1]]\n"
    )
    data_colon = YamlLoader.load(yaml_text_colon)
    assert isinstance(data_colon["solver"], Poisson2D)
    assert data_colon["solver"].forcing.callable is example_forcing
    assert data_colon["solver"].forcing({"value": 5}, {"phi": 0}) == 5
    _assert_round_trip(data_colon)

    yaml_text_space = (
        "solver: !romx:romjax.poisson.Poisson2D\n"
        "  forcing:\n"
        "    callable: !!python/name tests.test_loader.example_forcing\n"
        "    inputs_default:\n"
        "      value: 2\n"
        "  grid:\n"
        "    shape: [2, 4]\n"
        "    bounds: [[0, 1], [0, 1]]\n"
    )
    data_space = YamlLoader.load(yaml_text_space)
    assert isinstance(data_space["solver"], Poisson2D)
    assert data_space["solver"].forcing.callable is example_forcing
    assert data_space["solver"].forcing({}, {"phi": 0}) == 2
    _assert_round_trip(data_space)


def test_poisson_registered_callable_inline_load_and_dump() -> None:
    yaml_text = (
        "solver: !romx:romjax.poisson.Poisson2D\n"
        "  forcing:\n"
        "    callable: gaussian\n"
        "    inputs_default:\n"
        "      A0: 0.75\n"
        "      sigma: 0.2\n"
        "  grid:\n"
        "    shape: [2, 4]\n"
        "    bounds: [[0, 1], [0, 1]]\n"
    )
    data = YamlLoader.load(yaml_text)
    solver = data["solver"]

    assert isinstance(solver, Poisson2D)
    assert isinstance(solver.forcing, GaussianForcing)
    assert solver.forcing.inputs_default["A0"] == 0.75
    assert solver.forcing.inputs_default["sigma"] == 0.2
    _assert_round_trip(data)


def test_poisson_builtin_outputs_sampler_load_and_dump() -> None:
    yaml_text = (
        "solver: !romx:romjax.poisson.Poisson2D\n"
        "  outputs_sampler: !romx:NearSolutionSampler\n"
        "    phi:\n"
        "      callable: normal\n"
        "      std: 0.1\n"
        "      shape: [4, 4]\n"
        "  grid:\n"
        "    shape: [4, 4]\n"
        "    bounds: [[0, 1], [0, 1]]\n"
    )
    data = YamlLoader.load(yaml_text)
    solver = data["solver"]

    assert isinstance(solver, Poisson2D)
    assert isinstance(solver.outputs_sampler, NearSolutionSampler)
    assert isinstance(solver.outputs_sampler.template["phi"], Distribution)
    _assert_round_trip(data)


def test_yaml_overrides_merge_mappings_sequences_and_missing_branches(tmp_path: Path) -> None:
    base_path = tmp_path / "base.yml"
    override_path = tmp_path / "override.yml"
    base_path.write_text(
        """
settings:
  nested: {a: 1, b: 2}
  items:
    - one
    - two
    - {x: 1, y: 2}
  scalar: old
""",
        encoding="utf-8",
    )
    override_path.write_text(
        """
!overrides:__parent__/base.yml
settings:
  nested: {b: 3, c: 4}
  items:
    - null
    - value
    - {y: 9, z: 10}
    - extra
  scalar: new
  created:
    branch: true
""",
        encoding="utf-8",
    )

    data = YamlLoader.load(override_path)

    assert data == {
        "settings": {
            "nested": {"a": 1, "b": 3, "c": 4},
            "items": ["one", "value", {"x": 1, "y": 9, "z": 10}, "extra"],
            "scalar": "new",
            "created": {"branch": True},
        }
    }


def test_yaml_overrides_preserve_base_file_contents(tmp_path: Path) -> None:
    base_path = tmp_path / "base.yml"
    override_path = tmp_path / "override.yml"
    base_text = "settings: {one: 1, two: 2}\n"
    base_path.write_text(base_text, encoding="utf-8")
    override_path.write_text("!overrides:__parent__/base.yml\nsettings: {two: 3}\n", encoding="utf-8")

    assert YamlLoader.load(override_path) == {"settings": {"one": 1, "two": 3}}
    assert base_path.read_text(encoding="utf-8") == base_text


def test_yaml_overrides_construct_after_merging_root_tagged_base(tmp_path: Path) -> None:
    base_path = tmp_path / "base.yml"
    override_path = tmp_path / "override.yml"
    base_path.write_text(
        """
!romx:tests.test_loader.CustomModel
opts: {a: 1, b: 2}
detail: base
""",
        encoding="utf-8",
    )
    override_path.write_text("!overrides:__parent__/base.yml\nopts: {b: 3}\n", encoding="utf-8")

    model = YamlLoader.load(override_path)

    assert isinstance(model, CustomModel)
    assert model.opts == {"a": 1, "b": 3}
    assert model.detail == "base"


def test_yaml_overrides_tagged_override_subtree_replaces_base(tmp_path: Path) -> None:
    base_path = tmp_path / "base.yml"
    override_path = tmp_path / "override.yml"
    base_path.write_text(
        """
solver: !romx:tests.test_loader.CustomModel
  opts: {a: 1, b: 2}
  detail: base
""",
        encoding="utf-8",
    )
    override_path.write_text(
        """
!overrides:__parent__/base.yml
solver: !romx:tests.test_loader.CustomModel
  opts: {c: 3}
  detail: override
""",
        encoding="utf-8",
    )

    data = YamlLoader.load(override_path)

    assert isinstance(data["solver"], CustomModel)
    assert data["solver"].opts == {"c": 3}
    assert data["solver"].detail == "override"


def test_yaml_overrides_resolve_recursive_chains(tmp_path: Path) -> None:
    base_path = tmp_path / "base.yml"
    middle_path = tmp_path / "middle.yml"
    top_path = tmp_path / "top.yml"
    base_path.write_text("settings: {one: base, two: base, three: base}\n", encoding="utf-8")
    middle_path.write_text("!overrides:__parent__/base.yml\nsettings: {two: middle, three: middle}\n", encoding="utf-8")
    top_path.write_text("!overrides:__parent__/middle.yml\nsettings: {three: top}\n", encoding="utf-8")

    assert YamlLoader.load(top_path) == {"settings": {"one": "base", "two": "middle", "three": "top"}}


def test_yaml_overrides_parent_path_resolves_from_declaring_file(tmp_path: Path) -> None:
    parent_base = tmp_path / "base.yml"
    child_dir = tmp_path / "child"
    child_dir.mkdir()
    override_path = child_dir / "override.yml"
    parent_base.write_text("settings: {one: base, two: base}\n", encoding="utf-8")
    override_path.write_text("!overrides:__parent__/../base.yml\nsettings: {two: override}\n", encoding="utf-8")

    assert YamlLoader.load(override_path) == {"settings": {"one": "base", "two": "override"}}




def test_yaml_invalid_cases() -> None:
    with pytest.raises(TypeError):
        YamlLoader.load(123)

    with pytest.raises(TypeError):
        YamlLoader.dump({}, 123)

    with pytest.raises(FileNotFoundError):
        YamlLoader.load("tests/does_not_exist.yml")

    with pytest.raises(ValueError):
        YamlLoader.load("solver: !romx:badpath\n  opts: {a: 1}\n  detail: x\n")

    with pytest.raises(TypeError):
        YamlLoader.load("solver: !romx:builtins.dict\n  a: 1\n")
