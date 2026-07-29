import jax
import jax.numpy as jnp
import numpy as np
import pytest
from pydantic import TypeAdapter

from romjax import YamlLoader
from romjax.operators import BinaryOp, UnaryOp
from romjax.tree import (
    ShapeDtypePyTree,
    TreeRef,
    as_shape_dtype_pytree,
    pytree_at,
    pytree_iter,
    pytree_mean,
    pytree_merge,
    pytree_norm,
    pytree_path_iter,
    pytree_reduce,
    pytree_resolve_refs,
    pytree_size,
    to_pytree,
)
from romjax.typing import DictModel


def unary_shift_sum(x):
    return jnp.sum(x) + 1.0


def error_l1(x, xhat):
    return jnp.sum(jnp.abs(x - xhat))


def tree_l1(tree, tree_hat):
    return jnp.sum(jnp.abs(tree["a"] - tree_hat["a"]))


def test_shape_dtype_template_pytree_validator() -> None:
    static_fn = lambda value: value
    template = {
        "array": jnp.ones((2, 3), dtype=jnp.float32),
        "yaml": {"shape": [4], "dtype": "int32"},
        "default_dtype": {"shape": [1, 2]},
        "nested": ["static", static_fn, {"array": np.zeros(2)}],
        "not_a_template": {"shape": [3], "label": "static metadata"},
    }

    validated = TypeAdapter(ShapeDtypePyTree).validate_python(template)

    assert validated == as_shape_dtype_pytree(template)
    assert isinstance(validated["array"], jax.ShapeDtypeStruct)
    assert validated["array"].shape == (2, 3)
    assert validated["array"].dtype == jnp.float32
    assert validated["yaml"] == jax.ShapeDtypeStruct((4,), jnp.int32)
    assert validated["default_dtype"].dtype == jnp.asarray(0.0).dtype
    assert isinstance(validated["nested"][2]["array"], jax.ShapeDtypeStruct)
    assert validated["nested"][0] == "static"
    assert validated["nested"][1] is static_fn
    assert validated["not_a_template"] == template["not_a_template"]


def test_tree_ref_yaml_and_resolution() -> None:
    tree = YamlLoader.load("target: 3\nreference: !romx:TreeRef [target]\n")

    assert isinstance(tree["reference"], TreeRef)
    assert pytree_resolve_refs(tree)["reference"] == 3


def test_to_pytree():
    class Inner(DictModel):
        value: jnp.ndarray

    class Outer(DictModel):
        inner: Inner
        entries: list
        pair: tuple

    arr = jnp.array([1.0, 2.0])
    model = Outer(
        inner=Inner(value=arr),
        entries=[{"a": jnp.array(3.0)}, (jnp.array(4.0),)],
        pair=(jnp.array(5.0), {"b": jnp.array(6.0)}),
    )

    tree = to_pytree(model)

    assert isinstance(tree, dict)
    assert isinstance(tree["entries"], list)
    assert isinstance(tree["pair"], tuple)
    assert tree["inner"]["value"] is arr


def test_merge_pytrees():
    class Data(DictModel):
        data: jax.Array

    data = Data(data=jnp.linspace(0, 1, 10))
    more_data = Data(data=jnp.array([1, 2, 3]))

    shared = jnp.array([1.0, 2.0])
    defaults = {
        "a": {"x": shared, "y": jnp.array(2.0)},
        "b": (jnp.array(3.0), {"z": jnp.array(4.0)}),
        "d": data,
    }
    overrides = {
        "a": {"y": jnp.array(5.0), "new": jnp.array(6.0)},
        "b": (jnp.array(7.0),),
        "c": jnp.array(8.0),
        "e": more_data,
    }

    merged = pytree_merge(to_pytree(defaults), to_pytree(overrides))

    assert merged is not defaults
    assert merged["a"]["x"] is shared
    assert merged["d"]["data"] is data.data
    assert merged["e"]["data"] is more_data.data
    assert float(merged["a"]["y"]) == 5.0
    assert float(merged["a"]["new"]) == 6.0
    assert float(merged["b"][0]) == 7.0
    assert float(merged["b"][1]["z"]) == 4.0
    assert float(merged["c"]) == 8.0


def test_to_pytree_merge_and_jit_grad():
    """We can still do jit/grad/vmap through all of the pytree merging shenanigans."""

    class Inner(DictModel):
        value: jnp.ndarray

    class Outer(DictModel):
        inner: Inner
        extra: dict

    defaults = Outer(inner=Inner(value=jnp.array(1.0)), extra={"scale": jnp.array(2.0)})
    overrides = Outer(inner=Inner(value=jnp.array(3.0)), extra={"bias": jnp.array(4.0)})

    def merged_sum(scale: jnp.ndarray) -> jnp.ndarray:
        override = {"inner": {"value": scale}}
        tree = pytree_merge(to_pytree(defaults), pytree_merge(to_pytree(overrides), override))
        return 2 * tree["inner"]["value"] + tree["extra"]["scale"] + tree["extra"]["bias"]

    true_val = merged_sum(jnp.array(5.0))
    true_vmap_val = jnp.array([merged_sum(1.0), merged_sum(2.0)])
    jit_val = jax.jit(merged_sum)(jnp.array(5.0))
    grad_val = jax.grad(merged_sum)(jnp.array(5.0))
    vmap_val = jax.vmap(merged_sum)(jnp.array([1.0, 2.0]))

    assert jnp.allclose(true_val, jit_val)
    assert jnp.allclose(true_vmap_val, vmap_val)
    assert jnp.allclose(grad_val, 2.0)


def test_unary_op_array_and_tree_inputs():
    operator = UnaryOp("max-abs")
    grad_operator = UnaryOp("mean-square")
    tree = {"a": jnp.array([-1.0, 2.0]), "b": ("ignore", jnp.array([-4.0]))}

    assert jnp.allclose(operator(jnp.array([-1.0, 2.0, -4.0])), 4.0)
    assert jnp.allclose(operator(tree), 4.0)
    assert jnp.allclose(UnaryOp("mean")(tree, ignore=[("b",)]), 0.5)

    jit_value = jax.jit(operator)(jnp.array([-1.0, 2.0, -4.0]))
    grad_value = jax.grad(grad_operator)(jnp.array([1.0, 2.0]))
    vmap_value = jax.vmap(operator)(jnp.array([[-1.0, 2.0], [3.0, -5.0]]))

    assert jnp.allclose(jit_value, 4.0)
    assert jnp.allclose(grad_value, jnp.array([1.0, 2.0]))
    assert jnp.allclose(vmap_value, jnp.array([2.0, 5.0]))


def test_unary_op_leaf_reduce_config():
    operator = UnaryOp({"leaf_op": "square", "reduce_op": "mean"})
    tree = {"a": jnp.array([1.0, 2.0]), "b": jnp.array([3.0])}

    assert jnp.allclose(operator(tree), jnp.mean(jnp.array([1.0, 4.0, 9.0])))
    assert jnp.allclose(operator(jnp.array([1.0, 2.0])), jnp.mean(jnp.array([1.0, 4.0])))


def test_binary_op_array_inputs():
    custom = BinaryOp(lambda x, xhat: jnp.sum(x - xhat))
    relative = BinaryOp(("abs", "norm"))

    x = jnp.array([3.0, 4.0])
    xhat = jnp.array([1.0, 1.0])

    assert jnp.allclose(custom(x, xhat), 5.0)
    assert jnp.allclose(relative(x, xhat), jnp.abs(x - xhat) / jnp.linalg.norm(x))

    jit_value = jax.jit(relative)(x, xhat)
    grad_value = jax.grad(lambda arr: jnp.sum(relative(arr, xhat)))(x)

    assert jnp.allclose(jit_value, relative(x, xhat))
    assert grad_value.shape == x.shape


def test_binary_op_tree_inputs_and_aliases():
    tree = {"a": jnp.array([1.0, 3.0]), "b": jnp.array([-2.0])}
    tree_hat = {"a": jnp.array([0.0, 1.0]), "b": jnp.array([1.0])}

    mean_error = BinaryOp("mean")
    mse = BinaryOp("mse")
    relative = BinaryOp("relative")

    expected_mean = jnp.mean(jnp.concatenate([jnp.array([1.0, 2.0]), jnp.array([-3.0])]))
    expected_mse = jnp.mean(jnp.square(jnp.concatenate([jnp.array([1.0, 2.0]), jnp.array([-3.0])])))
    expected_relative = jnp.linalg.norm(jnp.array([1.0, 2.0, -3.0])) / jnp.linalg.norm(jnp.array([1.0, 3.0, -2.0]))

    assert jnp.ndim(mean_error(tree, tree_hat)) == 0
    assert jnp.allclose(mean_error(tree, tree_hat), expected_mean)
    assert jnp.allclose(mse(tree, tree_hat), expected_mse)
    assert jnp.allclose(relative(tree, tree_hat), expected_relative)

    jit_value = jax.jit(relative)(tree, tree_hat)
    assert jnp.allclose(jit_value, expected_relative)


def test_binary_op_common_overlap_ignore_and_strict_errors():
    tree = {
        "keep": {"x": jnp.array(3.0), "ignore": {"y": jnp.array(100.0)}, "left_only": jnp.array(9.0)},
        "shared": jnp.array(1.0),
        "seq": [jnp.array(10.0), jnp.array(20.0), jnp.array(30.0)],
    }
    tree_hat = {
        "keep": {"x": jnp.array(1.0), "ignore": {"y": jnp.array(-100.0)}, "right_only": jnp.array(4.0)},
        "shared": jnp.array(4.0),
        "seq": [jnp.array(13.0), jnp.array(21.0)],
    }

    operator = BinaryOp("mae", ignore=[("keep", "ignore")])

    assert jnp.allclose(operator(tree, tree_hat), jnp.array(2.25))
    assert jnp.allclose(BinaryOp("mae")(tree, tree_hat, ignore=[("keep",)]), jnp.array(7.0 / 3.0))
    with pytest.raises(ValueError, match="keys differ"):
        BinaryOp({"op": "mae", "overlap": "strict"})(tree, tree_hat)
    with pytest.raises(ValueError, match="no selected overlapping"):
        BinaryOp("mae")(tree, tree_hat, ignore=[("keep",), ("shared",), ("seq",)])


def test_binary_op_leaf_reduce_config():
    tree = {"a": jnp.array([3.0, 4.0]), "b": jnp.array([1.0, 1.0])}
    tree_hat = {"a": jnp.array([1.0, 1.0]), "b": jnp.array([0.0, 1.0])}
    operator = BinaryOp({"reduce_op": "mean", "leaf_op": ("norm", "norm")})

    expected = jnp.mean(
        jnp.array(
            [
                jnp.linalg.norm(tree["a"] - tree_hat["a"]) / jnp.linalg.norm(tree["a"]),
                jnp.linalg.norm(tree["b"] - tree_hat["b"]) / jnp.linalg.norm(tree["b"]),
            ]
        )
    )

    assert jnp.allclose(operator(tree, tree_hat), expected)


def test_pytree_reduce():
    tree = {"a": jnp.array([1.0, -2.0]), "b": ("ignore", jnp.array([3.0]))}
    flat = jnp.array([1.0, -2.0, 3.0])

    mean_value = pytree_mean(tree)
    norm_value = pytree_norm(tree)
    generic_mean = pytree_reduce("mean", tree)
    generic_norm = pytree_reduce(UnaryOp("norm"), tree)
    generic_max = pytree_reduce("max_abs", tree)

    assert jnp.ndim(mean_value) == 0
    assert jnp.ndim(norm_value) == 0
    assert jnp.allclose(mean_value, jnp.mean(flat))
    assert jnp.allclose(norm_value, jnp.linalg.norm(flat))
    assert jnp.allclose(generic_mean, mean_value)
    assert jnp.allclose(generic_norm, norm_value)
    assert jnp.allclose(generic_max, 3.0)


def test_unary_op_yaml_round_trip(tmp_path):
    path = tmp_path / "unary.yml"
    payload = {
        "string_op": UnaryOp("max_abs"),
        "callable_op": UnaryOp(unary_shift_sum),
        "structured_op": UnaryOp({"leaf_op": "square", "reduce_op": "mean", "ignore": ["skip"]}),
    }

    YamlLoader.dump(payload, path)
    dumped = path.read_text()
    reloaded = YamlLoader.load(path)

    assert "tests.test_tree.unary_shift_sum" in dumped
    assert jnp.allclose(reloaded["string_op"](jnp.array([-1.0, 3.0])), 3.0)
    assert jnp.allclose(reloaded["callable_op"](jnp.array([1.0, 2.0])), 4.0)
    assert jnp.allclose(reloaded["structured_op"]({"x": jnp.array([2.0]), "skip": jnp.array([100.0])}), 4.0)


def test_binary_op_yaml_round_trip(tmp_path):
    path = tmp_path / "error.yml"
    payload = {
        "op_only": BinaryOp("abs"),
        "with_norm": BinaryOp(("abs", "norm")),
        "structured": BinaryOp({"reduce_op": "norm", "leaf_op": ("abs", "norm"), "norm": "norm"}),
        "callable": BinaryOp(error_l1),
        "tree_callable": BinaryOp(tree_l1),
    }

    YamlLoader.dump(payload, path)
    dumped = path.read_text()
    reloaded = YamlLoader.load(path)
    x = jnp.array([3.0, 4.0])
    xhat = jnp.array([1.0, 1.0])
    tree = {"a": jnp.array([3.0, 4.0])}
    tree_hat = {"a": jnp.array([1.0, 1.0])}

    assert "tests.test_tree.error_l1" in dumped
    expected = jnp.linalg.norm(jnp.abs(tree["a"] - tree_hat["a"]) / jnp.linalg.norm(tree["a"])) / jnp.linalg.norm(
        tree["a"]
    )
    assert jnp.allclose(reloaded["op_only"](x, xhat), jnp.abs(x - xhat))
    assert jnp.allclose(reloaded["with_norm"](x, xhat), jnp.abs(x - xhat) / jnp.linalg.norm(x))
    assert jnp.allclose(reloaded["structured"](tree, tree_hat), expected)
    assert jnp.allclose(reloaded["callable"](x, xhat), 5.0)
    assert jnp.allclose(reloaded["tree_callable"](tree, tree_hat), 5.0)


def test_pytree_iter():
    tree = {
        "a": jnp.array([[1.0, 2.0], [3.0, 4.0]]),
        "b": {"c": jnp.array([5.0, 6.0])},
    }

    items = list(pytree_iter(tree))

    assert len(items) == 2
    assert np.allclose(np.asarray(items[0]["a"]), np.asarray(jnp.array([1.0, 2.0])))
    assert np.isclose(float(items[1]["b"]["c"]), 6.0)
    assert np.allclose(np.asarray(pytree_at(tree, 1)["a"]), np.asarray(jnp.array([3.0, 4.0])))
    assert pytree_size(tree) == 2


def test_pytree_path_iter_preserves_container_order():
    tree = {
        "b": jnp.array(1.0),
        "a": {"d": jnp.array(2.0), "c": jnp.array(3.0)},
        "c": [jnp.array(4.0), jnp.array(5.0)],
        "d": (jnp.array(6.0), jnp.array(7.0)),
    }

    items = list(pytree_path_iter(tree))
    paths = [path for path, _ in items]

    assert paths == [("b",), ("a", "d"), ("a", "c"), ("c", 0), ("c", 1), ("d", 0), ("d", 1)]
