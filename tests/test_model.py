import jax.numpy as jnp

from romjax import YamlLoader
from romjax.model import FilterModel


def _empty_forward_tree() -> dict:
    return {
        "outputs": {"a_plus": None, "b_times": None},
        "meta": {"shift": None, "gain": None, "tag": None},
    }


def _empty_backward_tree() -> dict:
    return {
        "inputs": {"a": None, "b": [None, {"gain": None}]},
        "meta": {"shift": None, "tag": None},
    }


def forward_a(x: dict) -> dict:
    y = _empty_forward_tree()
    y["outputs"]["a_plus"] = x["inputs"]["a"] + x["meta"]["shift"]
    y["meta"]["shift"] = x["meta"]["shift"]
    return y


def backward_a(y: dict) -> dict:
    x = _empty_backward_tree()
    shift = y["meta"]["shift"]
    x["inputs"]["a"] = y["outputs"]["a_plus"] - shift
    x["meta"]["shift"] = shift
    return x


def forward_b(x: dict, multiplier: float = 1.0) -> dict:
    y = _empty_forward_tree()
    b0 = x["inputs"]["b"][0]
    gain = x["inputs"]["b"][1]["gain"]
    y["outputs"]["b_times"] = b0 * gain * multiplier
    y["meta"]["gain"] = gain
    y["meta"]["tag"] = x["meta"]["tag"]
    return y


def backward_b(y: dict, multiplier: float = 1.0) -> dict:
    x = _empty_backward_tree()
    gain = y["meta"]["gain"]
    x["inputs"]["b"] = [y["outputs"]["b_times"] / (gain * multiplier), {"gain": gain}]
    x["meta"]["tag"] = y["meta"]["tag"]
    return x


def _model_yaml() -> str:
    return (
        "model: !rox:romjax.model.FilterModel\n"
        "  source: latent\n"
        "  target: observed\n"
        "  filters:\n"
        "    - in_paths:\n"
        "        - path: [inputs, a]\n"
        "        - path: [meta, shift]\n"
        "      out_paths:\n"
        "        - path: [outputs, a_plus]\n"
        "        - path: [meta, shift]\n"
        "      forward: !!python/name:tests.test_model.forward_a\n"
        "      backward: !!python/name:tests.test_model.backward_a\n"
        "    - in_spec:\n"
        "        inputs:\n"
        "          a: false\n"
        "          b: [true, {gain: true}]\n"
        "        meta:\n"
        "          shift: false\n"
        "          tag: true\n"
        "      out_spec:\n"
        "        outputs:\n"
        "          a_plus: false\n"
        "          b_times: true\n"
        "        meta:\n"
        "          shift: false\n"
        "          gain: true\n"
        "          tag: true\n"
        "      forward: !!python/name:tests.test_model.forward_b\n"
        "      forward_opts: {multiplier: 2.0}\n"
        "      backward: !!python/name:tests.test_model.backward_b\n"
        "      backward_opts: {multiplier: 2.0}\n"
    )


def _input_tree() -> dict:
    return {
        "inputs": {
            "a": jnp.array([1.0, 3.0, 5.0]),
            "b": [jnp.array([[2.0, 4.0], [6.0, 8.0]]), {"gain": jnp.array(3.0)}],
        },
        "meta": {"shift": jnp.array(0.25), "tag": "case-a"},
    }


def test_filter_model_load_from_yaml() -> None:
    data = YamlLoader.load(_model_yaml())
    model = data["model"]
    assert isinstance(model, FilterModel)
    assert model.source.name == "latent"
    assert model.target.name == "observed"
    assert len(model.filters) == 2


def test_filter_model_round_trip_nontrivial_pytree() -> None:
    model = YamlLoader.load(_model_yaml())["model"]
    x = _input_tree()

    y = model.forward(x)
    assert jnp.allclose(y["outputs"]["a_plus"], x["inputs"]["a"] + x["meta"]["shift"])
    assert jnp.allclose(y["outputs"]["b_times"], x["inputs"]["b"][0] * x["inputs"]["b"][1]["gain"] * 2.0)
    assert jnp.allclose(y["meta"]["shift"], x["meta"]["shift"])
    assert jnp.allclose(y["meta"]["gain"], x["inputs"]["b"][1]["gain"])
    assert y["meta"]["tag"] == x["meta"]["tag"]

    x_hat = model.backward(y)
    assert jnp.allclose(x_hat["inputs"]["a"], x["inputs"]["a"])
    assert jnp.allclose(x_hat["inputs"]["b"][0], x["inputs"]["b"][0])
    assert jnp.allclose(x_hat["inputs"]["b"][1]["gain"], x["inputs"]["b"][1]["gain"])
    assert jnp.allclose(x_hat["meta"]["shift"], x["meta"]["shift"])
    assert x_hat["meta"]["tag"] == x["meta"]["tag"]
