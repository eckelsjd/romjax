
import pytest
from pydantic import ValidationError

from romjax.graph import Node, NodeList
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
