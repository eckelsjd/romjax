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


def _model_yaml() -> str:
    return (
        "model: !rox:romjax.model.FilterModel\n"
        "  source: pde1_state\n"
        "  target: pde2_state\n"
        "  filters:\n"
        "    - in_paths:\n"
        "        - path: [pde1, inputs, forcing]\n"
        "        - path: [pde1, outputs, phi]\n"
        "        - path: [pde1, inputs, alpha]\n"
        "        - path: [meta, tag]\n"
        "      out_paths:\n"
        "        - path: [latent, pair]\n"
        "        - path: [latent, alpha]\n"
        "        - path: [meta, tag]\n"
        "      forward: !!python/name:tests.test_model.forward_pack_fields\n"
        "      backward: !!python/name:tests.test_model.backward_unpack_fields\n"
        "      forward_routes:\n"
        "        - source: [features, pair]\n"
        "          target: [latent, pair]\n"
        "        - source: [carry, alpha]\n"
        "          target: [latent, alpha]\n"
        "        - source: [meta, tag]\n"
        "          target: [meta, tag]\n"
        "      backward_routes:\n"
        "        - source: [recovered, forcing]\n"
        "          target: [pde1, inputs, forcing]\n"
        "        - source: [recovered, phi]\n"
        "          target: [pde1, outputs, phi]\n"
        "        - source: [recovered, alpha]\n"
        "          target: [pde1, inputs, alpha]\n"
        "        - source: [recovered, tag]\n"
        "          target: [meta, tag]\n"
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
                "in_paths": [
                    {"path": ["high", "inputs", "forcing"]},
                    {"path": ["high", "outputs", "phi"]},
                ],
                "out_paths": [
                    {"path": ["low", "inputs", "x_inputs"]},
                    {"path": ["low", "inputs", "x_outputs"]},
                    {"path": ["low", "inputs", "x_shared"]},
                ],
                "forward": high_to_low,
                "backward": low_to_high,
                "forward_routes": [
                    {"source": ["features", "x_inputs"], "target": ["low", "inputs", "x_inputs"]},
                    {"source": ["features", "x_outputs"], "target": ["low", "inputs", "x_outputs"]},
                    {"source": ["features", "x_shared"], "target": ["low", "inputs", "x_shared"]},
                ],
                "backward_routes": [
                    {"source": ["recovered", "forcing"], "target": ["high", "inputs", "forcing"]},
                    {"source": ["recovered", "phi"], "target": ["high", "outputs", "phi"]},
                    {"source": ["recovered", "alpha"], "target": ["high", "inputs", "alpha"]},
                ],
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

    n_samples = 96
    n_full = 12
    n_rank = 5
    n_latent = 3

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
                "forward": eqx_evaluate,
                "backward": eqx_evaluate,
                "in_paths": [{"path": ["full", "x"]}],
                "out_paths": [{"path": ["latent", "z"]}],
                "forward_opts": {"gather": "flat", "method": "reduce"},
                "backward_opts": {"gather": "flat", "method": "reconstruct", "scatter": "flat"},
                "forward_routes": [{"source": [], "target": ["latent", "z"]}],
                "backward_routes": [{"source": ["full", "x"], "target": ["full", "x"]}],
            }
        ],
    )

    projection = LinearProjection(n_latent=n_latent, n_full=n_full, key=k_init)
    opt = optax.adam(1e-1)
    opt_state = opt.init(eqx.filter(projection, eqx.is_array))

    @eqx.filter_jit
    def loss_fn(module: LinearProjection) -> jax.Array:
        z_tree, aux = model.forward_aux({"full": {"x": data}, "filters": [module]})
        x_hat, _ = model.backward_aux({"latent": {"z": z_tree["latent"]["z"]}, "filters": [module]}, aux=aux)
        return jnp.mean((x_hat["full"]["x"] - data) ** 2)

    @eqx.filter_jit
    def step(module: LinearProjection, state: optax.OptState) -> tuple[LinearProjection, optax.OptState, jax.Array]:
        loss, grads = eqx.filter_value_and_grad(loss_fn)(module)
        updates, state = opt.update(grads, state, eqx.filter(module, eqx.is_array))
        module = eqx.apply_updates(module, updates)
        return module, state, loss

    init_loss = float(loss_fn(projection))
    for _ in range(450):
        projection, opt_state, _ = step(projection, opt_state)
    final_loss = float(loss_fn(projection))

    assert final_loss < init_loss
    assert final_loss <= float(pod_loss) * 1.20


def test_filter_model_in_graph() -> None:
    """Test linear projection with a filter model in a FunctionGraph"""
    key = jax.random.PRNGKey(24)
    k_basis1, k_basis2, k_coef, k_noise1, k_noise2, k_init = jax.random.split(key, 6)

    n_samples = 72
    n_full = 32
    n_shared = 6
    n_latent = 4

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
                "forward": eqx_evaluate,
                "backward": eqx_evaluate,
                "in_paths": [["full", "field1"], ["full", "field2"]],
                "out_paths": [["latent", "z1"], ["latent", "z2"]],
                "forward_opts": {"gather": "stack", "method": "reduce"},
                "backward_opts": {"gather": "stack", "method": "reconstruct", "scatter": "stack"},
                "forward_routes": [
                    {"source": [0], "target": ["latent", "z1"]},
                    {"source": [1], "target": ["latent", "z2"]},
                ],
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
            {"full": {"field1": field1, "field2": field2}, "filters": [curr_module]},
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
    for _ in range(300):
        module, opt_state, _ = step(module, opt_state)
    final_loss = float(loss_fn(module))

    assert final_loss < init_loss * 0.45

    encoded, aux_cache = graph.push_path(
        {"full": {"field1": field1, "field2": field2}, "filters": [module]},
        path=["filter"],
        start="full",
        return_aux=True,
    )
    decoded = graph.push_path(
        {"latent": {"z1": encoded["latent"]["z1"], "z2": encoded["latent"]["z2"]}, "filters": [module]},
        path=["filter"],
        start="latent",
        aux=aux_cache,
    )
    assert decoded["full"]["field1"].shape == field1.shape
    assert decoded["full"]["field2"].shape == field2.shape

    with pytest.raises(ValueError):
        graph.push_path(
            {"latent": {"z1": encoded["latent"]["z1"], "z2": encoded["latent"]["z2"]}, "filters": [module]},
            path=["filter"],
            start="latent",
        )


def test_filter_model_prunes_none_leaves_from_views() -> None:
    model = FilterModel(
        source="pde1",
        target="latent",
        filters=[
            {
                "forward": forward_expect_pruned_view,
                "backward": backward_expect_pruned_view,
                "in_paths": [["pde1", "inputs", "forcing"]],
                "out_paths": [["latent", "z"]],
            }
        ],
    )

    encoded = model.forward(_tree_input())
    assert jnp.array_equal(encoded["latent"]["z"], _tree_input()["pde1"]["inputs"]["forcing"])

    decoded = model.backward({"latent": {"z": encoded["latent"]["z"]}})
    assert jnp.array_equal(decoded["pde1"]["inputs"]["forcing"], encoded["latent"]["z"])


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
