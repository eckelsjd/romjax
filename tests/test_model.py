from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import optax
import pytest

from romjax import YamlLoader
from romjax.graph import FunctionGraph, Node
from romjax.model import (
    ExplicitModel,
    FilterModel,
    ImplicitModel,
    eqx_evaluate,
)
from romjax.nn import LinearProjection


def _tree_input() -> dict:
    return {
        "pde1": {
            "inputs": {
                "forcing": jnp.array([1.0, 2.0, 3.0]),
                "alpha": jnp.array(0.25),
            },
            "outputs": {
                "phi": jnp.array([2.0, 4.0, 6.0]),
            },
        },
        "meta": {
            "tag": "case-a",
        },
    }


def forward_pack_fields(x: dict, _: object | None = None) -> dict:
    forcing = x["pde1"]["inputs"]["forcing"]
    phi = x["pde1"]["outputs"]["phi"]
    return {
        "features": {
            "pair": jnp.stack([forcing, phi], axis=0),
        },
        "carry": {
            "alpha": x["pde1"]["inputs"]["alpha"],
        },
        "meta": {
            "tag": x["meta"]["tag"],
        },
    }


def backward_unpack_fields(y: dict, _: object | None = None) -> dict:
    pair = y["latent"]["pair"]
    return {
        "recovered": {
            "forcing": pair[0],
            "phi": pair[1],
            "alpha": y["latent"]["alpha"],
            "tag": y["meta"]["tag"],
        }
    }


def forward_expect_pruned_view(x: dict, _: object | None = None) -> dict:
    assert "meta" not in x
    assert "outputs" not in x["pde1"]
    assert set(x["pde1"]["inputs"]) == {"forcing"}
    return {"latent": {"z": x["pde1"]["inputs"]["forcing"]}}


def backward_expect_pruned_view(x: dict, _: object | None = None) -> dict:
    assert set(x) == {"latent"}
    assert set(x["latent"]) == {"z"}
    return {"pde1": {"inputs": {"forcing": x["latent"]["z"]}}}


def identity_callable(x):
    return x


def scale_value(x: jax.Array, scale: float | jax.Array | None = None) -> jax.Array:
    if scale is None:
        raise ValueError("scale_value requires a runtime scale.")
    return jnp.asarray(scale) * jnp.asarray(x)


def combine_latents(x: dict, bias: float | jax.Array | None = None) -> dict:
    if bias is None:
        raise ValueError("combine_latents requires a runtime bias.")
    return {"y": x["z1"] + x["z2"] + jnp.asarray(bias)}


def _model_yaml() -> str:
    return (
        "model: !romx:romjax.model.FilterModel\n"
        "  source: pde1_state\n"
        "  target: pde2_state\n"
        "  filters:\n"
        "    - forward:\n"
        "        callable: !!python/name:tests.test_model.forward_pack_fields\n"
        "        input_routes:\n"
        "          - [pde1, inputs, forcing]\n"
        "          - [pde1, outputs, phi]\n"
        "          - [pde1, inputs, alpha]\n"
        "          - [meta, tag]\n"
        "        output_routes:\n"
        "          - inner: [features, pair]\n"
        "            outer: [latent, pair]\n"
        "          - inner: [carry, alpha]\n"
        "            outer: [latent, alpha]\n"
        "          - inner: [meta, tag]\n"
        "            outer: [meta, tag]\n"
        "      backward:\n"
        "        callable: !!python/name:tests.test_model.backward_unpack_fields\n"
        "        input_routes:\n"
        "          - [latent, pair]\n"
        "          - [latent, alpha]\n"
        "          - [meta, tag]\n"
        "        output_routes:\n"
        "          - inner: [recovered, forcing]\n"
        "            outer: [pde1, inputs, forcing]\n"
        "          - inner: [recovered, phi]\n"
        "            outer: [pde1, outputs, phi]\n"
        "          - inner: [recovered, alpha]\n"
        "            outer: [pde1, inputs, alpha]\n"
        "          - inner: [recovered, tag]\n"
        "            outer: [meta, tag]\n"
    )


class ToyHighPDE(ImplicitModel):
    source: Node = Node(name="high_inputs")
    target: Node = Node(name="high_outputs")

    def evaluate(self, inputs: dict, outputs: dict) -> dict:
        pred_phi = inputs["forcing"] + inputs["alpha"]
        return {"phi_residual": outputs["phi"] - pred_phi}

    def solve(self, inputs: dict, residuals: dict) -> dict:
        pred_phi = inputs["forcing"] + inputs["alpha"]
        return {"phi": residuals["phi_residual"] + pred_phi}


class ToyLowPDE(ImplicitModel):
    source: Node = Node(name="low_inputs")
    target: Node = Node(name="low_outputs")

    def evaluate(self, inputs: dict, outputs: dict) -> dict:
        pred_phi = inputs["x_inputs"] + inputs["x_outputs"] + inputs["x_shared"]
        return {"phi_red_residual": outputs["phi_red"] - pred_phi}

    def solve(self, inputs: dict, residuals: dict) -> dict:
        pred_phi = inputs["x_inputs"] + inputs["x_outputs"] + inputs["x_shared"]
        return {"phi_red": residuals["phi_red_residual"] + pred_phi}


class LatentGainEdge(ExplicitModel):
    source: Node = Node(name="latent")
    target: Node = Node(name="latent_proc")

    def pushforward(self, inputs: dict) -> dict:
        return {"z": inputs["gain"] * inputs["z"]}


def high_to_low(x: dict, _: object | None = None) -> dict:
    forcing = x["high"]["inputs"]["forcing"]
    phi = x["high"]["outputs"]["phi"]
    return {
        "features": {
            "x_inputs": forcing,
            "x_outputs": phi,
            "x_shared": forcing + phi,
        }
    }


def low_to_high(y: dict, _: object | None = None) -> dict:
    forcing = y["low"]["inputs"]["x_inputs"]
    phi = y["low"]["inputs"]["x_outputs"]
    return {
        "recovered": {
            "forcing": forcing,
            "phi": phi,
            "alpha": phi - forcing,
        }
    }


def test_eqx_evaluate():
    """Tests that we can gather/scatter arraylike pytrees through an equinox module evaluation."""
    class Affine(eqx.Module):
        scale: jax.Array
        bias: jax.Array

        def reduce(self, x: jax.Array) -> jax.Array:
            return self.scale * x + self.bias

        def reconstruct(self, z: jax.Array) -> jax.Array:
            return (z - self.bias) / self.scale

    module = Affine(scale=jnp.array(2.0), bias=jnp.array(-1.5))

    flat_tree = {
        "left": jnp.array([1.0, 2.0]),
        "right": {"v": jnp.array([-3.0, 4.0, 5.0])},
    }
    encoded_flat, aux_flat = eqx_evaluate(flat_tree, module, gather="flat", method="reduce", return_aux=True)
    decoded_flat = eqx_evaluate(encoded_flat, module, method="reconstruct", scatter="flat", aux=aux_flat)
    assert encoded_flat.ndim == 1
    assert jnp.allclose(decoded_flat["left"], flat_tree["left"])
    assert jnp.allclose(decoded_flat["right"]["v"], flat_tree["right"]["v"])

    stack_tree = {
        "a": jnp.array([0.5, -1.0, 2.5]),
        "b": jnp.array([3.0, 4.0, -2.0]),
    }
    encoded_stack, aux_stack = eqx_evaluate(stack_tree, module, gather="stack", method="reduce", return_aux=True)
    decoded_stack = eqx_evaluate(encoded_stack, module, method="reconstruct", scatter="stack", aux=aux_stack)
    assert encoded_stack.shape == (2, 3)
    assert jnp.allclose(decoded_stack["a"], stack_tree["a"])
    assert jnp.allclose(decoded_stack["b"], stack_tree["b"])

    tree = {
        "nested": {
            "scalar": jnp.array(2.5),
            "vector": jnp.array([1.0, -2.0, 3.0]),
        },
        "tail": [jnp.array([-1.0, 4.0, 2.0])],
    }

    encoded, aux = eqx_evaluate(tree, lambda x: x, gather="flat", return_aux=True)
    decoded = eqx_evaluate(encoded, lambda x: x, scatter="flat", aux=aux)

    assert jnp.allclose(decoded["nested"]["scalar"], tree["nested"]["scalar"])
    assert jnp.allclose(decoded["nested"]["vector"], tree["nested"]["vector"])
    assert jnp.allclose(decoded["tail"][0], tree["tail"][0])


def test_filter_model_yaml_round_trip_and_routing() -> None:
    data = YamlLoader.load(_model_yaml())
    model = data["model"]
    assert isinstance(model, FilterModel)
    assert len(model.filters) == 1

    x = _tree_input()
    y = model.forward(x)

    assert jnp.allclose(y["latent"]["pair"][0], x["pde1"]["inputs"]["forcing"])
    assert jnp.allclose(y["latent"]["pair"][1], x["pde1"]["outputs"]["phi"])
    assert jnp.allclose(y["latent"]["alpha"], x["pde1"]["inputs"]["alpha"])
    assert y["meta"]["tag"] == x["meta"]["tag"]

    x_hat = model.backward(y)
    assert jnp.allclose(x_hat["pde1"]["inputs"]["forcing"], x["pde1"]["inputs"]["forcing"])
    assert jnp.allclose(x_hat["pde1"]["outputs"]["phi"], x["pde1"]["outputs"]["phi"])
    assert jnp.allclose(x_hat["pde1"]["inputs"]["alpha"], x["pde1"]["inputs"]["alpha"])
    assert x_hat["meta"]["tag"] == x["meta"]["tag"]

    dumped = YamlLoader.dump(data)
    assert dumped is not None
    reloaded = YamlLoader.load(dumped)
    assert isinstance(reloaded["model"], FilterModel)
    assert reloaded["model"].model_dump() == model.model_dump()


def test_filter_model_integration_with_toy_pde_graph() -> None:
    high = ToyHighPDE(source="high_inputs", target="high_outputs")
    low = ToyLowPDE(source="low_inputs", target="low_outputs")

    filter_edge = FilterModel(
        source="high_state",
        target="low_state",
        filters=[
            {
                "forward": {
                    "callable": high_to_low,
                    "input_routes": [
                        ["high", "inputs", "forcing"],
                        ["high", "outputs", "phi"],
                    ],
                    "output_routes": [
                        {"inner": ["features", "x_inputs"], "outer": ["low", "inputs", "x_inputs"]},
                        {"inner": ["features", "x_outputs"], "outer": ["low", "inputs", "x_outputs"]},
                        {"inner": ["features", "x_shared"], "outer": ["low", "inputs", "x_shared"]},
                    ],
                },
                "backward": {
                    "callable": low_to_high,
                    "input_routes": [
                        ["low", "inputs", "x_inputs"],
                        ["low", "inputs", "x_outputs"],
                    ],
                    "output_routes": [
                        {"inner": ["recovered", "forcing"], "outer": ["high", "inputs", "forcing"]},
                        {"inner": ["recovered", "phi"], "outer": ["high", "outputs", "phi"]},
                        {"inner": ["recovered", "alpha"], "outer": ["high", "inputs", "alpha"]},
                    ],
                },
            }
        ],
    )

    graph = FunctionGraph(edges={"high": high, "filter": filter_edge, "low": low})
    assert "high_state" in [node.name for node in graph.nodes.values()]
    assert "low_state" in [node.name for node in graph.nodes.values()]

    high_inputs = {"forcing": jnp.array(2.0), "alpha": jnp.array(1.0)}
    high_outputs = high.solve(high_inputs, {"phi_residual": jnp.array(0.0)})

    low_state = filter_edge.forward({"high": {"inputs": high_inputs, "outputs": high_outputs}})
    low_outputs = low.solve(low_state["low"]["inputs"], {"phi_red_residual": jnp.array(0.0)})

    high_state_hat = filter_edge.backward({"low": {"inputs": low_state["low"]["inputs"], "outputs": low_outputs}})
    high_res = high.evaluate(high_state_hat["high"]["inputs"], high_state_hat["high"]["outputs"])

    assert jnp.allclose(high_res["phi_residual"], jnp.array(0.0))


def test_linear_projection() -> None:
    """Make sure filter model with linear projection matches POD."""
    key = jax.random.PRNGKey(7)
    k_basis, k_coef, k_noise, k_init = jax.random.split(key, 4)

    n_samples = 32
    n_full = 8
    n_rank = 3
    n_latent = 2

    basis_raw = jax.random.normal(k_basis, (n_rank, n_full))
    basis, _ = jnp.linalg.qr(basis_raw.T)
    basis = basis.T

    coeffs = jax.random.normal(k_coef, (n_samples, n_rank))
    data = coeffs @ basis + 0.02 * jax.random.normal(k_noise, (n_samples, n_full))

    _, _, vh = jnp.linalg.svd(data, full_matrices=False)
    pod_basis = vh[:n_latent, :]
    pod_recon = data @ pod_basis.T @ pod_basis
    pod_loss = jnp.mean((pod_recon - data) ** 2)

    model = FilterModel(
        source="full",
        target="latent",
        filters=[
            {
                "forward": {
                    "callable": eqx_evaluate,
                    "input_routes": [{"outer": ["full", "x"], "inner": []}],
                    "output_routes": [{"outer": ["latent", "z"]}],
                    "opts": {"method": "reduce"},
                },
                "backward": {
                    "callable": eqx_evaluate,
                    "input_routes": [{"outer": ["latent", "z"], "inner": []}],
                    "output_routes": [{"outer": ["full", "x"]}],
                    "opts": {"method": "reconstruct"},
                },
            }
        ],
    )

    projection = LinearProjection(n_latent=n_latent, n_full=n_full, key=k_init)
    opt = optax.adam(1e-1)
    opt_state = opt.init(eqx.filter(projection, eqx.is_array))

    @eqx.filter_jit
    def loss_fn(module: LinearProjection) -> jax.Array:
        z_tree, aux = model.forward_aux({"full": {"x": data}, "call_args": module})
        x_hat, _ = model.backward_aux({"latent": {"z": z_tree["latent"]["z"]}, "call_args": module}, aux=aux)
        return jnp.mean((x_hat["full"]["x"] - data) ** 2)

    @eqx.filter_jit
    def step(module: LinearProjection, state: optax.OptState) -> tuple[LinearProjection, optax.OptState, jax.Array]:
        loss, grads = eqx.filter_value_and_grad(loss_fn)(module)
        updates, state = opt.update(grads, state, eqx.filter(module, eqx.is_array))
        module = eqx.apply_updates(module, updates)
        return module, state, loss

    init_loss = float(loss_fn(projection))
    for _ in range(80):
        projection, opt_state, _ = step(projection, opt_state)
    final_loss = float(loss_fn(projection))

    assert final_loss < init_loss
    assert final_loss <= float(pod_loss) * 1.35


def test_filter_model_in_graph() -> None:
    """Test linear projection with a filter model in a FunctionGraph"""
    key = jax.random.PRNGKey(24)
    k_basis1, k_basis2, k_coef, k_noise1, k_noise2, k_init = jax.random.split(key, 6)

    n_samples = 24
    n_full = 12
    n_shared = 3
    n_latent = 3

    basis1_raw = jax.random.normal(k_basis1, (n_shared, n_full))
    basis2_raw = jax.random.normal(k_basis2, (n_shared, n_full))
    basis1 = basis1_raw / jnp.linalg.norm(basis1_raw, axis=1, keepdims=True)
    basis2 = basis2_raw / jnp.linalg.norm(basis2_raw, axis=1, keepdims=True)
    coeffs = jax.random.normal(k_coef, (n_samples, n_shared))

    field1 = coeffs @ basis1 + 0.03 * jax.random.normal(k_noise1, (n_samples, n_full))
    field2 = 0.8 * coeffs @ basis2 + 0.2 * field1 + 0.03 * jax.random.normal(k_noise2, (n_samples, n_full))

    filter_edge = FilterModel(
        source="full",
        target="latent",
        filters=[
            {
                "forward": {
                    "callable": eqx_evaluate,
                    "input_routes": [
                        {"outer": ["full", "field1"], "inner": ["field1"]},
                        {"outer": ["full", "field2"], "inner": ["field2"]},
                    ],
                    "output_routes": [
                        {"inner": [0], "outer": ["latent", "z1"]},
                        {"inner": [1], "outer": ["latent", "z2"]},
                    ],
                    "opts": {"gather": "stack", "method": "reduce"},
                },
                "backward": {
                    "callable": eqx_evaluate,
                    "input_routes": [
                        {"outer": ["latent", "z1"], "inner": ["z1"]},
                        {"outer": ["latent", "z2"], "inner": ["z2"]},
                    ],
                    "output_routes": [
                        {"inner": ["field1"], "outer": ["full", "field1"]},
                        {"inner": ["field2"], "outer": ["full", "field2"]},
                    ],
                    "opts": {"gather": "stack", "method": "reconstruct", "scatter": "stack"},
                },
            }
        ],
    )
    graph = FunctionGraph(edges={"filter": filter_edge})

    module = LinearProjection(n_latent=n_latent, n_full=n_full, key=k_init)
    opt = optax.adam(1e-1)
    opt_state = opt.init(eqx.filter(module, eqx.is_array))

    @eqx.filter_jit
    def loss_fn(curr_module: LinearProjection) -> jax.Array:
        decoded = graph.push_path(
            {"full": {"field1": field1, "field2": field2}, "call_args": curr_module},
            path=["filter", "filter"],
            start="full",
        )
        field1_mse = jnp.mean((decoded["full"]["field1"] - field1) ** 2)
        field2_mse = jnp.mean((decoded["full"]["field2"] - field2) ** 2)
        return 0.5 * (field1_mse + field2_mse)

    @eqx.filter_jit
    def step(
        curr_module: LinearProjection,
        state: optax.OptState,
    ) -> tuple[LinearProjection, optax.OptState, jax.Array]:
        loss, grads = eqx.filter_value_and_grad(loss_fn)(curr_module)
        updates, state = opt.update(grads, state, eqx.filter(curr_module, eqx.is_array))
        curr_module = eqx.apply_updates(curr_module, updates)
        return curr_module, state, loss

    init_loss = float(loss_fn(module))
    for _ in range(80):
        module, opt_state, _ = step(module, opt_state)
    final_loss = float(loss_fn(module))

    assert final_loss < init_loss * 0.6

    encoded, aux_cache = graph.push_path(
        {"full": {"field1": field1, "field2": field2}, "call_args": module},
        path=["filter"],
        start="full",
        return_aux=True,
    )
    assert aux_cache["full->latent"]["backward"]["cached_states"][0]["template"]["field1"].shape == field1.shape
    decoded = graph.push_path(
        {"latent": {"z1": encoded["latent"]["z1"], "z2": encoded["latent"]["z2"]}, "call_args": module},
        path=["filter"],
        start="latent",
        aux=aux_cache,
    )
    assert decoded["full"]["field1"].shape == field1.shape
    assert decoded["full"]["field2"].shape == field2.shape

    with pytest.raises(ValueError):
        graph.push_path(
            {"latent": {"z1": encoded["latent"]["z1"], "z2": encoded["latent"]["z2"]}, "call_args": module},
            path=["filter"],
            start="latent",
        )


def test_filter_model_prunes_none_leaves_from_views() -> None:
    model = FilterModel(
        source="pde1",
        target="latent",
        filters=[
            {
                "forward": {
                    "callable": forward_expect_pruned_view,
                    "input_routes": [["pde1", "inputs", "forcing"]],
                },
                "backward": {
                    "callable": backward_expect_pruned_view,
                    "input_routes": [["latent", "z"]],
                },
            }
        ],
    )

    encoded = model.forward(_tree_input())
    assert jnp.array_equal(encoded["latent"]["z"], _tree_input()["pde1"]["inputs"]["forcing"])

    decoded = model.backward({"latent": {"z": encoded["latent"]["z"]}})
    assert jnp.array_equal(decoded["pde1"]["inputs"]["forcing"], encoded["latent"]["z"])


def test_filter_model_runtime_input_normalization_and_errors() -> None:
    shared_model = FilterModel(
        source="full",
        target="latent",
        filters=[
            {
                "forward": {
                    "callable": eqx_evaluate,
                    "input_routes": [{"outer": ["full", "x"], "inner": []}],
                    "output_routes": [["latent", "z"]],
                    "opts": {"method": "reduce"},
                },
                "backward": {
                    "callable": eqx_evaluate,
                    "input_routes": [{"outer": ["latent", "z"], "inner": []}],
                    "output_routes": [["full", "x"]],
                    "opts": {"method": "reconstruct"},
                },
            }
        ],
    )
    multi_model = FilterModel(
        source="full",
        target="latent",
        filters=[
            {"forward": {"input_routes": [{"outer": ["full", "x"], "inner": []}], "output_routes": [["latent", "z1"]]}},
            {"forward": {"input_routes": [{"outer": ["full", "x"], "inner": []}], "output_routes": [["latent", "z2"]]}},
        ],
    )
    module = LinearProjection(n_latent=2, n_full=3, key=jax.random.PRNGKey(0))
    data = jnp.ones((4, 3))

    encoded = shared_model.forward({"full": {"x": data}, "call_args": module})
    assert encoded["latent"]["z"].shape == (4, 2)

    encoded_shared = multi_model.forward({"full": {"x": jnp.array([1.0, 2.0])}, "call_args": {"shared": 3.0}})
    assert jnp.array_equal(encoded_shared["latent"]["z1"], jnp.array([1.0, 2.0]))
    assert jnp.array_equal(encoded_shared["latent"]["z2"], jnp.array([1.0, 2.0]))

    with pytest.raises(ValueError):
        multi_model.forward({"full": {"x": jnp.array([1.0, 2.0])}, "call_args": [1.0]})

    with pytest.raises(ValueError):
        multi_model.forward(
            {"full": {"x": jnp.array([1.0, 2.0])}, "call_args": {"shared": 1.0, "per_spec": [1.0, 2.0]}}
        )


def test_graph_edge_payload_patches_route_distinct_filter_model_call_args() -> None:
    encode = FilterModel(
        source="full",
        target="latent",
        name="encode",
        filters=[
            {
                "forward": {
                    "callable": scale_value,
                    "input_routes": [{"outer": ["full", "x"], "inner": []}],
                    "output_routes": [["latent", "z1"]],
                }
            },
            {
                "forward": {
                    "callable": scale_value,
                    "input_routes": [{"outer": ["full", "x"], "inner": []}],
                    "output_routes": [["latent", "z2"]],
                }
            },
        ],
    )
    decode = FilterModel(
        source="latent",
        target="summary",
        name="decode",
        filters=[
            {
                "forward": {
                    "callable": combine_latents,
                    "input_routes": [
                        {"outer": ["latent", "z1"], "inner": ["z1"]},
                        {"outer": ["latent", "z2"], "inner": ["z2"]},
                    ],
                    "output_routes": [{"inner": ["y"], "outer": ["summary", "y"]}],
                }
            }
        ],
    )
    graph = FunctionGraph(edges={"encode": encode, "decode": decode})
    patches = {
        "encode": {"call_args": {"per_spec": [jnp.array(2.0), jnp.array(3.0)]}},
        "decode": {"call_args": {"shared": jnp.array(7.0)}},
    }

    out = graph.push_path(
        {"full": {"x": jnp.array([1.0, -2.0])}},
        path=["encode", "decode"],
        edge_payload_patches=patches,
    )

    assert jnp.array_equal(out["summary"]["y"], jnp.array([12.0, -3.0]))


def test_filter_model_missing_cached_state_and_aux_shape_errors() -> None:
    model = FilterModel(
        source="inputs",
        target="latent",
        filters=[
            {
                "forward": {
                    "callable": eqx_evaluate,
                    "input_routes": [["left"], ["right"]],
                    "output_routes": [["latent", "z"]],
                    "opts": {"gather": "stack"},
                },
                "backward": {
                    "callable": eqx_evaluate,
                    "input_routes": [{"outer": ["latent", "z"], "inner": []}],
                    "opts": {"scatter": "stack"},
                },
            }
        ],
    )
    data = {"left": jnp.array([1.0, 2.0]), "right": jnp.array([3.0, 4.0])}
    encoded, aux = model.forward_aux({"left": data["left"], "right": data["right"], "call_args": identity_callable})

    with pytest.raises(ValueError):
        model.backward({"latent": {"z": encoded["latent"]["z"]}, "call_args": identity_callable})

    with pytest.raises(ValueError):
        model.backward_aux({"latent": {"z": encoded["latent"]["z"]}, "call_args": identity_callable}, aux=[aux])


def test_filter_model_invalid_routes_raise() -> None:
    bad_input_model = FilterModel(
        source="full",
        target="latent",
        filters=[{"forward": {"input_routes": [["full", "missing"]], "output_routes": [["latent", "z"]]}}],
    )
    with pytest.raises(KeyError):
        bad_input_model.forward({"full": {"x": jnp.array([1.0, 2.0])}})

    bad_output_model = FilterModel(
        source="full",
        target="latent",
        filters=[
            {
                "forward": {
                    "callable": lambda x, _: {"present": x["full"]["x"]},
                    "input_routes": [["full", "x"]],
                    "output_routes": [{"inner": ["missing"], "outer": ["latent", "z"]}],
                }
            }
        ],
    )
    with pytest.raises(KeyError):
        bad_output_model.forward({"full": {"x": jnp.array([1.0, 2.0])}})


# def test_conv_autoencoder_filter_model_with_joint_graph_optimization() -> None:
#     key = jax.random.PRNGKey(11)
#     k_phase, k_amp, k_model = jax.random.split(key, 3)

#     n_samples = 24
#     h, w = 8, 8
#     latent_dim = 6

#     xx, yy = jnp.meshgrid(jnp.linspace(0.0, 1.0, h), jnp.linspace(0.0, 1.0, w), indexing="ij")
#     phase = jax.random.uniform(k_phase, (n_samples, 1, 1, 1), minval=0.0, maxval=2.0 * jnp.pi)
#     amp = 0.8 + 0.4 * jax.random.uniform(k_amp, (n_samples, 1, 1, 1))
#     fields = amp * jnp.sin(2.0 * jnp.pi * (xx[None, None, :, :] + yy[None, None, :, :]) + phase)

#     autoencoder = ConvAutoencoder2D(input_shape=(h, w), latent_dim=latent_dim, key=k_model, in_channels=1)
#     gain_edge = LatentGainEdge(source="latent", target="latent_proc")

#     filter_edge = FilterModel(
#         source="full",
#         target="latent",
#         filters=[
#             {
#                 "in_paths": [{"path": ["full", "phi"]}],
#                 "out_paths": [{"path": ["latent", "z"]}],
#                 "forward": eqx_evaluate,
#                 "backward": eqx_evaluate,
#                 "forward_opts": {"collect": collect_full_phi, "method": "encode"},
#                 "backward_opts": {"collect": collect_latent_z, "method": "decode"},
#                 "forward_routes": [{"source": [], "target": ["latent", "z"]}],
#                 "backward_routes": [{"source": [], "target": ["full", "phi"]}],
#             }
#         ],
#     )

#     graph = FunctionGraph(edges={"filter": filter_edge, "gain": gain_edge})
#     assert graph.edges["filter"].name == "full->latent"
#     assert graph.edges["gain"].name == "latent->latent_proc"

#     params = {
#         "autoencoder": autoencoder,
#         "gain": jnp.array(0.65),
#     }

#     opt = optax.adam(5e-2)
#     opt_state = opt.init(eqx.filter(params, eqx.is_array))

#     @eqx.filter_jit
#     def loss_fn(curr_params: dict) -> jax.Array:
#         encoded = graph.edges["filter"].forward(
#             {"full": {"phi": fields}, "filters": [curr_params["autoencoder"]]}
#         )
#         latent_proc = graph.edges["gain"].pushforward(
#             {"z": encoded["latent"]["z"], "gain": curr_params["gain"]}
#         )
#         decoded = graph.edges["filter"].backward(
#             {"latent": {"z": latent_proc["z"]}, "filters": [curr_params["autoencoder"]]}
#         )
#         return jnp.mean((decoded["full"]["phi"] - fields) ** 2)

#     @eqx.filter_jit
#     def step(curr_params: dict, state: optax.OptState) -> tuple[dict, optax.OptState, jax.Array]:
#         loss, grads = eqx.filter_value_and_grad(loss_fn)(curr_params)
#         updates, state = opt.update(grads, state, eqx.filter(curr_params, eqx.is_array))
#         curr_params = eqx.apply_updates(curr_params, updates)
#         return curr_params, state, loss

#     init_loss = float(loss_fn(params))
#     for _ in range(220):
#         params, opt_state, _ = step(params, opt_state)
#     final_loss = float(loss_fn(params))

#     assert final_loss < init_loss * 0.75
#     assert float(jnp.abs(params["gain"] - 0.65)) > 1e-3
