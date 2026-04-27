import jax
import jax.numpy as jnp
import pytest
from pydantic import ValidationError

from romjax import YamlLoader
from romjax.graph import Edge, EdgeList, FunctionGraph, IdentityEdge, Node, NodeList


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
