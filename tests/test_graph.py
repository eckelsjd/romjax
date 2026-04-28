import jax
import jax.numpy as jnp
import pytest
from pydantic import ValidationError

from romjax import YamlLoader
from romjax.graph import CompositeEdge, Edge, EdgeList, FunctionGraph, IdentityEdge, Node, NodeList


def test_node_list():
    nodes = NodeList(["a", "b", "c"])

    assert list(nodes.keys()) == ["a", "b", "c"]
    assert [str(n) for n in nodes.values()] == ["a", "b", "c"]

    assert nodes[0].name == "a"
    assert [n.name for n in nodes[1:]] == ["b", "c"]
    assert [n.name for n in nodes[[2, 0]]] == ["c", "a"]
    assert nodes["b"].name == "b"

    nodes[1] = "beta"
    nodes["d"] = {"name": "d"}
    nodes.append(Node(name="e"))
    assert list(nodes.keys()) == ["a", "b", "c", "d", "e"]
    assert nodes[1].name == "beta"
    assert all(isinstance(n, Node) for n in nodes.values())

    with pytest.raises(ValidationError):
        nodes["bad"] = {"not_name": "x"}

    del nodes[0]
    assert list(nodes.keys()) == ["b", "c", "d", "e"]
    del nodes[[1, 2]]
    assert list(nodes.keys()) == ["b", "e"]

    node = Node(name="omega")
    assert node == "omega"
    assert node == Node(name="omega")
    assert node != Node(name="other")
    assert str(node) == "omega"
    assert hash(node) == hash("omega")
    assert {node: 1}["omega"] == 1

    yaml_text = (
        "nodes: !rox:romjax.graph.NodeList\n"
        "  a: a\n"
        "  b: {name: b}\n"
        "  c:\n"
        "    name: c\n"
    )
    data = YamlLoader.load(yaml_text)
    assert isinstance(data["nodes"], NodeList)
    assert list(data["nodes"].keys()) == ["a", "b", "c"]
    assert [n.name for n in data["nodes"].values()] == ["a", "b", "c"]


def test_edge_list():
    edges = EdgeList(
        [
            IdentityEdge(source="a", target="b"),
            IdentityEdge(source="b", target="c", name="b->c"),
            IdentityEdge(source="c", target="d", name=""),
        ]
    )

    assert list(edges.keys()) == ["a->b", "b->c", "c->d"]
    assert [str(e) for e in edges.values()] == ["a->b", "b->c", "c->d"]

    assert edges[0].name == "a->b"
    assert [e.name for e in edges[1:]] == ["b->c", "c->d"]
    assert [e.name for e in edges[[2, 0]]] == ["c->d", "a->b"]
    assert edges["b->c"].name == "b->c"

    edges[1] = {"source": "b", "target": "beta", "name": "b->beta"}
    edges["d->e"] = {"source": "d", "target": "e", "name": "d->e"}
    edges.append(IdentityEdge(source="e", target="f", name="e->f"))
    assert list(edges.keys()) == ["a->b", "b->c", "c->d", "d->e", "e->f"]
    assert edges[1].name == "b->beta"
    assert all(isinstance(e, Edge) for e in edges.values())

    with pytest.raises(ValidationError):
        edges["bad"] = {"source": "x"}

    del edges[0]
    assert list(edges.keys()) == ["b->c", "c->d", "d->e", "e->f"]
    del edges[[1, 2]]
    assert list(edges.keys()) == ["b->c", "e->f"]

    edge = IdentityEdge(source="omega", target="psi", name="omega->psi")
    assert edge == "omega->psi"
    assert edge != "omega->other"
    assert edge != Node(name="omega")
    assert str(edge) == "omega->psi"
    assert hash(edge) == hash("omega->psi")
    assert {edge: 1}["omega->psi"] == 1

    yaml_text = (
        "edges: !rox:romjax.graph.EdgeList\n"
        "  a->b: {source: a, target: b, name: a->b}\n"
        "  b->c: {source: b, target: c, name: b->c}\n"
        "  c->d:\n"
        "    source: c\n"
        "    target: d\n"
        "    name: c->d\n"
    )
    data = YamlLoader.load(yaml_text)
    assert isinstance(data["edges"], EdgeList)
    assert list(data["edges"].keys()) == ["a->b", "b->c", "c->d"]
    assert [e.name for e in data["edges"].values()] == ["a->b", "b->c", "c->d"]


class AuxShiftEdge(Edge):
    source: Node = Node(name="a")
    target: Node = Node(name="b")

    def forward(self, x):
        return x + 2.0

    def backward(self, x):
        return x - 2.0

    def forward_aux(self, x, aux=None):
        del aux
        return self.forward(x), {"offset": jnp.array(2.0)}

    def backward_aux(self, x, aux=None):
        if aux is None or "offset" not in aux:
            raise ValueError("Missing auxiliary offset for backward pass.")
        return x - aux["offset"], None


def test_function_graph_push_path_aux_round_trip() -> None:
    graph = FunctionGraph(
        edges={
            "ab": AuxShiftEdge(source="a", target="b"),
            "bc": IdentityEdge(source="b", target="c"),
        }
    )

    value = jnp.array(3.0)
    forward_out, aux_cache = graph.push_path(
        value,
        path=["ab", "bc"],
        return_aux=True,
    )
    assert jnp.allclose(forward_out, jnp.array(5.0))
    assert "a->b" in aux_cache
    assert "backward" in aux_cache["a->b"]

    backward_out = graph.push_path(
        forward_out,
        path=["bc", "ab"],
        start="c",
        aux=aux_cache,
    )
    assert jnp.allclose(backward_out, value)


def test_function_graph_push_path_missing_or_precomputed_aux() -> None:
    graph = FunctionGraph(edges={"ab": AuxShiftEdge(source="a", target="b")})

    with pytest.raises(ValueError):
        graph.push_path(jnp.array(5.0), path=["ab"], start="b")

    precomputed_aux = {"a->b": {"backward": {"offset": jnp.array(2.0)}}}
    out = graph.push_path(jnp.array(5.0), path=["ab"], start="b", aux=precomputed_aux)
    assert jnp.allclose(out, jnp.array(3.0))


def test_jit_grad_vmap_graph_push_path():
    class AuxAffineEdge(Edge):
        source: Node = Node(name="a")
        target: Node = Node(name="b")

        def forward(self, x):
            return 3.0 * x - 1.0

        def backward(self, x):
            return (x + 1.0) / 3.0

        def forward_aux(self, x, aux=None):
            del aux
            return self.forward(x), {"scale": jnp.array(3.0), "shift": jnp.array(-1.0)}

        def backward_aux(self, x, aux=None):
            if aux is None:
                raise ValueError("Missing auxiliary data for backward pass.")
            return (x - aux["shift"]) / aux["scale"], None

    graph = FunctionGraph(edges={"ab": AuxAffineEdge(source="a", target="b")})

    def push_forward(x):
        y, aux = graph.push_path(x, path=["ab"], return_aux=True)
        return y, aux

    def round_trip(x):
        y, aux = push_forward(x)
        return graph.push_path(y, path=["ab"], start="b", aux=aux)

    x0 = jnp.array(2.0)
    expected = 3.0 * x0 - 1.0

    jit_forward, jit_aux = jax.jit(push_forward)(x0)
    jit_round_trip = jax.jit(round_trip)(x0)
    grad_out = jax.grad(round_trip)(x0)
    vmap_in = jnp.array([-1.0, 0.0, 2.0])
    vmap_out = jax.vmap(round_trip)(vmap_in)

    assert jnp.allclose(jit_forward, expected)
    assert jnp.allclose(jit_aux["a->b"]["backward"]["scale"], jnp.array(3.0))
    assert jnp.allclose(jit_aux["a->b"]["backward"]["shift"], jnp.array(-1.0))
    assert jnp.allclose(jit_round_trip, x0)
    assert jnp.allclose(grad_out, jnp.array(1.0))
    assert jnp.allclose(vmap_out, vmap_in)


class AffineIntEdge(Edge):
    scale: int = 1
    shift: int = 0

    def forward(self, x):
        return self.scale * x + self.shift

    def backward(self, x):
        return (x - self.shift) // self.scale


class RuntimeDeltaEdge(Edge):
    source: Node = Node(name="a")
    target: Node = Node(name="b")

    def forward(self, x):
        return {"value": x["value"] + x["call_args"]["delta"]}

    def backward(self, x):
        return {"value": x["value"] - x["call_args"]["delta"]}


class RuntimeScaleEdge(Edge):
    source: Node = Node(name="b")
    target: Node = Node(name="c")

    def forward(self, x):
        return {"value": x["value"] * x["call_args"]["scale"]}

    def backward(self, x):
        return {"value": x["value"] / x["call_args"]["scale"]}


def test_composite_edge_matches_explicit_path_forward_and_backward() -> None:
    graph = FunctionGraph(
        edges={
            "ab": AffineIntEdge(source="a", target="b", name="ab", scale=2, shift=3),
            "bc": AffineIntEdge(source="b", target="c", name="bc", scale=5, shift=7),
            "ac": CompositeEdge(source="a", target="c", name="ac", path=["ab", "bc"]),
        }
    )

    composite = graph.edges["ac"]
    x = jnp.array(4, dtype=jnp.int32)

    explicit_forward = graph.push_path(x, path=["ab", "bc"], start="a")
    explicit_backward = graph.push_path(explicit_forward, path=["bc", "ab"], start="c")

    assert composite.forward(x) == explicit_forward
    assert composite.backward(explicit_forward) == explicit_backward
    assert explicit_backward == x


def test_function_graph_push_path_edge_payload_patches() -> None:
    graph = FunctionGraph(
        edges={
            "ab": RuntimeDeltaEdge(source="a", target="b", name="ab"),
            "bc": RuntimeScaleEdge(source="b", target="c", name="bc"),
        }
    )
    patches = {
        "ab": {"call_args": {"delta": jnp.array(3.0)}},
        "bc": {"call_args": {"scale": jnp.array(5.0)}},
        "unused": {"call_args": {"delta": jnp.array(100.0)}},
    }

    forward = graph.push_path({"value": jnp.array(2.0)}, path=["ab", "bc"], start="a", edge_payload_patches=patches)
    backward = graph.push_path(forward, path=["bc", "ab"], start="c", edge_payload_patches=patches)

    assert jnp.allclose(forward["value"], jnp.array(25.0))
    assert jnp.allclose(backward["value"], jnp.array(2.0))


def test_composite_edge_propagates_edge_payload_patches() -> None:
    graph = FunctionGraph(
        edges={
            "ab": RuntimeDeltaEdge(source="a", target="b", name="ab"),
            "bc": RuntimeScaleEdge(source="b", target="c", name="bc"),
            "ac": CompositeEdge(source="a", target="c", name="ac", path=["ab", "bc"]),
        }
    )
    patches = {
        "ab": {"call_args": {"delta": jnp.array(4.0)}},
        "bc": {"call_args": {"scale": jnp.array(6.0)}},
    }

    explicit = graph.push_path({"value": jnp.array(1.0)}, path=["ab", "bc"], start="a", edge_payload_patches=patches)
    composite = graph.push_path({"value": jnp.array(1.0)}, path=["ac"], start="a", edge_payload_patches=patches)
    round_trip = graph.push_path(composite, path=["ac"], start="c", edge_payload_patches=patches)

    assert jnp.allclose(composite["value"], explicit["value"])
    assert jnp.allclose(composite["value"], jnp.array(30.0))
    assert jnp.allclose(round_trip["value"], jnp.array(1.0))


def test_composite_edge_preserves_graph_aux_behavior() -> None:
    graph = FunctionGraph(
        edges={
            "ab": AuxShiftEdge(source="a", target="b", name="ab"),
            "bc": IdentityEdge(source="b", target="c", name="bc"),
            "ac": CompositeEdge(source="a", target="c", name="ac", path=["ab", "bc"]),
        }
    )

    composite = graph.edges["ac"]
    value = jnp.array(3.0)

    forward_out, composite_aux = composite.forward_aux(value)
    explicit_out, explicit_aux = graph.push_path(value, path=["ab", "bc"], start="a", return_aux=True)

    assert jnp.allclose(forward_out, explicit_out)
    assert jnp.allclose(composite_aux["ab"]["backward"]["offset"], explicit_aux["ab"]["backward"]["offset"])

    backward_out, backward_aux = composite.backward_aux(forward_out, composite_aux)
    explicit_back, explicit_back_aux = graph.push_path(
        explicit_out,
        path=["bc", "ab"],
        start="c",
        aux=explicit_aux,
        return_aux=True,
    )

    assert jnp.allclose(backward_out, value)
    assert jnp.allclose(backward_out, explicit_back)
    assert backward_aux == explicit_back_aux


def test_composite_edge_yaml_load_and_validation() -> None:
    yaml_text = """
!rox:romjax.graph.FunctionGraph
edges:
  - !rox:tests.test_graph.AffineIntEdge
    source: a
    target: b
    name: ab
    scale: 2
    shift: 3
  - !rox:tests.test_graph.AffineIntEdge
    source: b
    target: c
    name: bc
    scale: 5
    shift: 7
  - !rox:romjax.graph.CompositeEdge
    source: a
    target: c
    name: ac
    path: [ab, bc]
"""
    graph = YamlLoader.load(yaml_text)
    composite = graph.edges["ac"]

    assert isinstance(composite, CompositeEdge)
    assert composite.path == ["ab", "bc"]
    assert composite.forward(jnp.array(1, dtype=jnp.int32)) == jnp.array(32, dtype=jnp.int32)


def test_composite_edge_rejects_invalid_paths() -> None:
    with pytest.raises(ValidationError, match="unknown edge"):
        FunctionGraph(
            edges={
                "ab": IdentityEdge(source="a", target="b", name="ab"),
                "ac": CompositeEdge(source="a", target="c", name="ac", path=["ab", "missing"]),
            }
        )

    with pytest.raises(ValidationError, match="cannot include itself"):
        FunctionGraph(edges={"ac": CompositeEdge(source="a", target="a", name="ac", path=["ac"])})

    with pytest.raises(ValidationError, match="Path discontinuity"):
        FunctionGraph(
            edges={
                "ab": IdentityEdge(source="a", target="b", name="ab"),
                "cd": IdentityEdge(source="c", target="d", name="cd"),
                "ad": CompositeEdge(source="a", target="d", name="ad", path=["ab", "cd"]),
            }
        )

    with pytest.raises(ValidationError, match="recursion cycle"):
        FunctionGraph(
            edges={
                "ab": IdentityEdge(source="a", target="b", name="ab"),
                "ba": IdentityEdge(source="b", target="a", name="ba"),
                "ac": CompositeEdge(source="a", target="c", name="ac", path=["ab", "ba", "ca"]),
                "ca": CompositeEdge(source="c", target="a", name="ca", path=["ac"]),
            }
        )
