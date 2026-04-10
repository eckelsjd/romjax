import pytest
from pydantic import ValidationError

from romjax.graph import Node, Edge, IdentityEdge, NodeList, EdgeList
from romjax import YamlLoader


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
