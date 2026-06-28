from __future__ import annotations

from pathlib import Path

import h5py
import jax.numpy as jnp
import pytest
from pydantic import ValidationError

from romjax import YamlLoader
from romjax.graph import Edge, FunctionGraph, IdentityEdge, Node
from romjax.norm import NormOperator, NormTree, minmax


def plus_one(x):
    return x + 1.0


class ScaleEdge(Edge):
    source: Node = Node(name="a")
    target: Node = Node(name="b")

    def forward(self, x):
        return 2.0 * x

    def backward(self, x):
        return x / 2.0


class AuxCallsForwardEdge(Edge):
    source: Node = Node(name="a")
    target: Node = Node(name="b")

    def forward(self, x):
        return x + 1.0

    def backward(self, x):
        return x - 1.0

    def forward_aux(self, x, aux=None):
        del aux
        return self.forward(x), {"seen": jnp.asarray(1.0)}


class OverridePreNormEdge(IdentityEdge):
    def _forward_pre_norm(self, x):
        return x + 5.0


def test_default_norm_preserves_existing_edge_api() -> None:
    edge = ScaleEdge(source="a", target="b")

    assert jnp.allclose(edge.forward(jnp.asarray(3.0)), 6.0)
    assert jnp.allclose(edge.backward(jnp.asarray(6.0)), 3.0)

    graph = FunctionGraph(edges={"ab": edge})
    out = graph.push_path(jnp.asarray(3.0), path=["a->b"])
    assert jnp.allclose(out, 6.0)


def test_forward_pre_infers_backward_post_inverse() -> None:
    edge = IdentityEdge(
        source="a",
        target="b",
        norm={"forward": {"pre": {"callable": "zscore", "mean": 2.0, "std": 4.0}}},
    )

    y = edge.forward(jnp.asarray(10.0))
    x = edge.backward(y)

    assert jnp.allclose(y, 2.0)
    assert jnp.allclose(x, 10.0)


def test_norm_tree_supports_broadcast_and_per_leaf_specs() -> None:
    x = {
        "a": jnp.asarray([2.0, 4.0]),
        "b": {"c": jnp.asarray(5.0), "untouched": "meta"},
    }

    broadcast = NormTree(root={"callable": "zscore", "mean": 1.0, "std": 2.0})
    assert jnp.allclose(broadcast(x)["a"], jnp.asarray([0.5, 1.5]))
    assert broadcast(x)["b"]["untouched"] == "meta"

    per_leaf = NormTree(root={"b": {"c": {"callable": "minmax", "xmin": 1.0, "xmax": 5.0}}})
    out = per_leaf(x)
    assert jnp.allclose(out["a"], x["a"])
    assert jnp.allclose(out["b"]["c"], 1.0)


def test_norm_tree_skips_paths_missing_from_payload() -> None:
    norm = NormTree(
        root={
            "inputs": {"k0": {"callable": "zscore", "mean": 1.0, "std": 2.0}},
            "outputs": {"phi": {"callable": "zscore", "mean": 2.0, "std": 4.0}},
        }
    )
    payload = {"outputs": {"phi": jnp.asarray([10.0]), "carry": jnp.asarray([3.0])}}

    out = norm(payload)

    assert "inputs" not in out
    assert jnp.allclose(out["outputs"]["phi"], jnp.asarray([2.0]))
    assert jnp.allclose(out["outputs"]["carry"], payload["outputs"]["carry"])


def test_yaml_configures_edge_normalization() -> None:
    data = YamlLoader.load(
        """
edge: !romx:romjax.graph.IdentityEdge
  source: a
  target: b
  norm:
    forward:
      pre:
        callable: zscore
        mean: 1.0
        std: 3.0
"""
    )

    edge = data["edge"]
    assert jnp.allclose(edge.forward(jnp.asarray(10.0)), 3.0)
    assert jnp.allclose(edge.backward(jnp.asarray(3.0)), 10.0)


def test_h5_norm_artifact_loads_constants(tmp_path: Path) -> None:
    artifact = tmp_path / "norm.h5"
    with h5py.File(artifact, "w") as h5:
        h5.attrs["callable"] = "zscore"
        h5.create_dataset("mean", data=jnp.asarray([1.0, 2.0]))
        h5.create_dataset("std", data=jnp.asarray([2.0, 4.0]))

    norm = NormTree(root=str(artifact))
    out = norm(jnp.asarray([3.0, 10.0]))

    assert jnp.allclose(out, jnp.asarray([1.0, 2.0]))


def test_h5_norm_tree_artifact_loads_and_applies(tmp_path: Path) -> None:
    artifact = tmp_path / "tree_norm.h5"
    with h5py.File(artifact, "w") as h5:
        h5.attrs["romjax_type"] = "norm_tree"
        h5.attrs["version"] = 1
        leaf = h5.create_group("tree/source/state/x")
        leaf.attrs["callable"] = "zscore"
        leaf.create_dataset("mean", data=jnp.asarray([1.0]))
        leaf.create_dataset("std", data=jnp.asarray([2.0]))

    norm = NormTree(root=str(artifact))
    out = norm({"source": {"state": {"x": jnp.asarray([5.0])}}})

    assert jnp.allclose(out["source"]["state"]["x"], jnp.asarray([2.0]))


def test_h5_norm_tree_artifact_overrides_merge_into_leaf(tmp_path: Path) -> None:
    artifact = tmp_path / "tree_norm.h5"
    with h5py.File(artifact, "w") as h5:
        h5.attrs["romjax_type"] = "norm_tree"
        leaf = h5.create_group("tree/source/state/x")
        leaf.attrs["callable"] = "minmax"
        leaf.create_dataset("xmin", data=jnp.asarray([0.0]))
        leaf.create_dataset("xmax", data=jnp.asarray([10.0]))
        leaf.create_dataset("ymin", data=jnp.asarray([0.0]))
        leaf.create_dataset("ymax", data=jnp.asarray([1.0]))

    norm = NormTree(root={"artifact": str(artifact), "overrides": {"source": {"state": {"x": {"ymax": 2.0}}}}})
    out = norm({"source": {"state": {"x": jnp.asarray([10.0])}}})

    assert jnp.allclose(out["source"]["state"]["x"], jnp.asarray([2.0]))


def test_h5_norm_tree_artifact_overrides_can_create_paths(tmp_path: Path) -> None:
    artifact = tmp_path / "tree_norm.h5"
    with h5py.File(artifact, "w") as h5:
        h5.attrs["romjax_type"] = "norm_tree"
        leaf = h5.create_group("tree/inputs/x")
        leaf.attrs["callable"] = "zscore"
        leaf.create_dataset("mean", data=jnp.asarray([1.0]))
        leaf.create_dataset("std", data=jnp.asarray([2.0]))

    norm = NormTree(
        root={
            "artifact": str(artifact),
            "overrides": {
                "outputs": {
                    "y": {
                        "callable": "minmax",
                        "xmin": 0.0,
                        "xmax": 10.0,
                        "ymin": -1.0,
                        "ymax": 1.0,
                    }
                }
            },
        }
    )
    out = norm({"inputs": {"x": jnp.asarray([5.0])}, "outputs": {"y": jnp.asarray([10.0])}})

    assert "inputs" in norm.resolve_root()
    assert "outputs" in norm.resolve_root()
    assert jnp.allclose(out["inputs"]["x"], jnp.asarray([2.0]))
    assert jnp.allclose(out["outputs"]["y"], jnp.asarray([1.0]))


def test_edge_norm_artifact_loads_lazily_and_caches(tmp_path: Path) -> None:
    artifact = tmp_path / "lazy_norm.h5"
    edge = IdentityEdge(
        source="a",
        target="b",
        norm={"forward": {"pre": str(artifact)}},
    )

    assert edge.norm.forward.pre.artifact == artifact
    assert edge.norm.forward.pre._resolved_root is None
    assert edge.norm.backward.post.artifact == artifact
    assert edge.norm.backward.post.inverse_artifact

    with h5py.File(artifact, "w") as h5:
        h5.attrs["callable"] = "zscore"
        h5.create_dataset("mean", data=jnp.asarray([1.0]))
        h5.create_dataset("std", data=jnp.asarray([2.0]))

    edge.resolve_norms()
    y = edge.forward(jnp.asarray([5.0]))
    x = edge.backward(y)

    assert edge.norm.forward.pre._resolved_root is not None
    assert jnp.allclose(y, jnp.asarray([2.0]))
    assert jnp.allclose(x, jnp.asarray([5.0]))


def test_registered_minmax_and_composite_inverse() -> None:
    x = jnp.asarray([1.0, 5.0])
    scaled = minmax(x, xmin=1.0, xmax=5.0, ymin=1.0, ymax=jnp.e)
    assert jnp.allclose(scaled, jnp.asarray([1.0, jnp.e]))

    op = NormOperator(callable="log-minmax", xmin=1.0, xmax=5.0, ymin=1.0, ymax=jnp.e)
    inverse = op.inverse()

    y = op(x)
    x_hat = inverse(y)

    assert jnp.allclose(y, jnp.asarray([0.0, 1.0]))
    assert jnp.allclose(x_hat, x)


def test_unregistered_callable_requires_explicit_inverse() -> None:
    with pytest.raises(ValidationError):
        IdentityEdge(
            source="a",
            target="b",
            norm={"forward": {"pre": {"callable": plus_one}}},
        )

    edge = IdentityEdge(
        source="a",
        target="b",
        norm={
            "forward": {"pre": {"callable": plus_one}},
            "backward": {"post": {"callable": lambda x: x - 1.0}},
        },
    )
    assert jnp.allclose(edge.forward(jnp.asarray(1.0)), 2.0)
    assert jnp.allclose(edge.backward(jnp.asarray(2.0)), 1.0)


def test_aux_can_override_norm_constants_online() -> None:
    edge = IdentityEdge(
        source="a",
        target="b",
        norm={"forward": {"pre": {"callable": "zscore", "mean": 0.0, "std": 1.0}}},
    )
    aux = {"norm": {"forward": {"pre": {"mean": 2.0, "std": 4.0}}}}

    y, edge_aux = edge.forward_aux(jnp.asarray(10.0), aux=aux)

    assert edge_aux is None
    assert jnp.allclose(y, 2.0)


def test_graph_aux_can_override_norm_constants_online() -> None:
    edge = IdentityEdge(
        source="a",
        target="b",
        norm={"forward": {"pre": {"callable": "zscore", "mean": 0.0, "std": 1.0}}},
    )
    graph = FunctionGraph(edges={"ab": edge})
    aux = {"a->b": {"forward": {"norm": {"forward": {"pre": {"mean": 2.0, "std": 4.0}}}}}}

    y = graph.push_path(jnp.asarray(10.0), path=["a->b"], aux=aux)

    assert jnp.allclose(y, 2.0)


def test_child_norm_hook_override_replaces_default_stage() -> None:
    edge = OverridePreNormEdge(
        source="a",
        target="b",
        norm={"forward": {"pre": {"callable": "zscore", "mean": 100.0, "std": 1.0}}},
    )

    assert jnp.allclose(edge.forward(jnp.asarray(1.0)), 6.0)


def test_aux_wrapper_does_not_apply_norm_twice() -> None:
    edge = AuxCallsForwardEdge(
        source="a",
        target="b",
        norm={"forward": {"pre": {"callable": "zscore", "mean": 2.0, "std": 4.0}}},
    )

    y, aux = edge.forward_aux(jnp.asarray(10.0))

    assert jnp.allclose(y, 3.0)
    assert jnp.allclose(aux["seen"], 1.0)
