import jax
import jax.numpy as jnp
import pytest

from romjax import YamlLoader
from romjax.graph import Edge, FunctionGraph, Node
from romjax.loss import CyclicPathError, GraphLoss, GraphLossTerm, GraphLossTermGenerator, cyclic_path_error_terms
from romjax.model import SourceSampleable
from romjax.train import Train


class OffsetEdge(Edge):
    """Test edge that applies independent offsets in each direction."""

    forward_offset: float
    backward_offset: float

    def forward(self, x):
        return x + self.forward_offset

    def backward(self, x):
        return x + self.backward_offset


class SampleableOffsetEdge(OffsetEdge, SourceSampleable):
    """Offset edge marked as a dataset source for default dataset inference tests."""

    def sample_source(self, key):
        del key
        return jnp.asarray(0.0)


class TreeOffsetEdge(Edge):
    """Test edge that adds separate offsets to two dictionary leaves."""

    forward_keep: float
    forward_skip: float
    backward_keep: float
    backward_skip: float

    def forward(self, x):
        return {"keep": x["keep"] + self.forward_keep, "skip": x["skip"] + self.forward_skip}

    def backward(self, x):
        return {"keep": x["keep"] + self.backward_keep, "skip": x["skip"] + self.backward_skip}


def four_node_cycle(sampleable_dataset: bool = False) -> FunctionGraph:
    dataset_edge_type = SampleableOffsetEdge if sampleable_dataset else OffsetEdge
    return FunctionGraph(
        nodes={
            "i": Node(name="i", error_op="mae"),
            "j": Node(name="j", error_op="mse"),
            "k": Node(name="k", error_op="mae"),
            "l": Node(name="l", error_op="mae"),
        },
        edges={
            "ij": OffsetEdge(source="i", target="j", name="ij", forward_offset=1.0, backward_offset=2.0),
            "jk": OffsetEdge(source="j", target="k", name="jk", forward_offset=10.0, backward_offset=20.0),
            "kl": OffsetEdge(source="k", target="l", name="kl", forward_offset=100.0, backward_offset=200.0),
            "li": dataset_edge_type(
                source="l", target="i", name="li", forward_offset=1000.0, backward_offset=2000.0
            ),
        },
    )


def generated_cycle_loss(**kwargs) -> GraphLoss:
    return GraphLoss(
        graph=four_node_cycle(),
        terms=[
            {
                "callable": "commutativity",
                "nodes": ["i", "j", "k", "l"],
                "dataset": "li",
                **kwargs,
            }
        ],
    )


def term_by_name(loss: GraphLoss, name: str) -> GraphLossTerm:
    for term in loss.terms:
        if term.name == name:
            assert isinstance(term, GraphLossTerm)
            return term
    raise AssertionError(f"Missing term {name!r}")


def test_cyclic_path_error_generator_expands_all_terms_and_uses_destination_error() -> None:
    loss = generated_cycle_loss(weight=2.0, batch_reduce=None)

    assert len(loss.terms) == 16
    assert set(loss.term_names) == {f"{source}->{dest}" for source in "ijkl" for dest in "ijkl"}
    assert all(isinstance(term, GraphLossTerm) for term in loss.terms)
    assert all(term.dataset == "li" for term in loss.terms)
    assert all(term.weight == pytest.approx(2.0) for term in loss.terms)

    params = {}
    single_data = jnp.array(0.0)

    # i->j compares +1 against skipped li plus kl/jk backward offsets: 1 - (200 + 20).
    raw_j = term_by_name(loss, "i->j").raw_value(params, single_data, loss.graph)
    assert raw_j == pytest.approx((1.0 - 220.0) ** 2)

    # i->l compares ij+jk+kl against skipped direct li data. Destination l uses MAE.
    raw_l = term_by_name(loss, "i->l").raw_value(params, single_data, loss.graph)
    assert raw_l == pytest.approx(abs(111.0 - 0.0))

    # i->i evaluates the final li edge on the clockwise full-cycle path.
    raw_i = term_by_name(loss, "i->i").raw_value(params, single_data, loss.graph)
    assert raw_i == pytest.approx(abs(1111.0 - 222.0))


def test_cyclic_path_error_generator_defaults_to_graph_node_order_and_sampleable_dataset() -> None:
    graph = four_node_cycle(sampleable_dataset=True)
    terms = cyclic_path_error_terms(graph, batch_reduce=None, cache_policy="none")

    assert len(terms) == 16
    assert [term.name for term in terms[:4]] == ["i->i", "i->j", "i->k", "i->l"]
    assert all(term.dataset == "li" for term in terms)
    assert terms[1].term.callable.spec.path_a == ("ij",)
    assert terms[1].term.callable.spec.path_b == ("li", "kl", "jk")


def test_cyclic_path_error_generator_seeds_outside_nodes_from_nearest_dataset_endpoint() -> None:
    loss = generated_cycle_loss(batch_reduce=None)
    j_term = term_by_name(loss, "j->i")
    k_term = term_by_name(loss, "k->i")

    assert j_term.term.callable.spec.logical_start == "i"
    assert j_term.term.callable.spec.path_a[0] == "ij"
    assert j_term.term.callable.spec.path_b[0] == "ij"

    assert k_term.term.callable.spec.logical_start == "l"
    assert k_term.term.callable.spec.path_a[0] == "kl"
    assert k_term.term.callable.spec.path_b[0] == "kl"


def test_cyclic_path_error_generator_yaml_and_train_deferred_binding() -> None:
    graph = four_node_cycle(sampleable_dataset=True)
    yaml_text = """
!romx:GraphLoss
terms:
  - callable: commutativity
    batch_reduce: null
  - {callable: tikhonov, batch_reduce: null}
"""
    loss = YamlLoader.load(yaml_text)

    assert isinstance(loss.terms[0], GraphLossTermGenerator)
    loss.bind_graph(graph)
    assert len(loss.terms) == 17
    assert isinstance(loss.terms[0], GraphLossTerm)
    assert loss.terms[-1].name == "term_16"

    train_yaml = """
!romx:Train
graph: !romx:FunctionGraph
  nodes:
    i: {name: i, error_op: mae}
    j: {name: j, error_op: mse}
    k: {name: k, error_op: mae}
    l: {name: l, error_op: mae}
  edges:
    ij: !romx:tests.test_loss.OffsetEdge
      source: i
      target: j
      name: ij
      forward_offset: 1.0
      backward_offset: 2.0
    jk: !romx:tests.test_loss.OffsetEdge
      source: j
      target: k
      name: jk
      forward_offset: 10.0
      backward_offset: 20.0
    kl: !romx:tests.test_loss.OffsetEdge
      source: k
      target: l
      name: kl
      forward_offset: 100.0
      backward_offset: 200.0
    li: !romx:tests.test_loss.SampleableOffsetEdge
      source: l
      target: i
      name: li
      forward_offset: 1000.0
      backward_offset: 2000.0
loss: !romx:GraphLoss
  terms:
    - callable: commutativity
      batch_reduce: null
init_params: {}
optimizer:
  name: sgd
  args: [0.1]
"""
    train = YamlLoader.load(train_yaml)

    assert isinstance(train, Train)
    assert isinstance(train.loss, GraphLoss)
    assert len(train.loss.terms) == 16
    assert train.loss.graph is train.graph


def test_cyclic_path_payload_cache_reuses_and_evicts_prefixes() -> None:
    graph = four_node_cycle()
    terms = cyclic_path_error_terms(
        graph,
        nodes=["i", "j", "k", "l"],
        dataset="li",
        batch_reduce=None,
        cache_payloads=True,
        cache_policy="none",
    )
    first = terms[0]
    second = terms[1]

    value, aux = first.raw_value({}, jnp.array(0.0), graph, return_aux=True)
    assert jnp.isfinite(value)
    assert "graph_path_payloads" in aux
    assert len(aux["graph_path_payloads"]) > 0

    reused_key = ("li", "i", ("ij",))
    assert reused_key in aux["graph_path_payloads"]
    value, aux = second.raw_value({}, jnp.array(0.0), graph, aux=aux, return_aux=True)
    assert jnp.isfinite(value)
    assert reused_key in aux["graph_path_payloads"]

    for term in terms[2:]:
        value, aux = term.raw_value({}, jnp.array(0.0), graph, aux=aux, return_aux=True)
        assert jnp.isfinite(value)

    assert aux["graph_path_payloads"] == {}


def test_cyclic_path_payload_cache_batches_between_generated_terms() -> None:
    loss = generated_cycle_loss(batch_reduce="mean", cache_payloads=True, cache_policy="none")
    params = {}
    batch = {"li": jnp.array([0.0, 1.0, 2.0])}

    value = loss(params, batch)
    assert jnp.isfinite(value)
    assert jnp.isfinite(jax.grad(lambda scale: loss(params, {"li": scale * batch["li"]}))(jnp.array(1.0)))


def test_cyclic_path_error_cache_ordering_limits_peak_cache_size() -> None:
    graph = four_node_cycle()
    naive = cyclic_path_error_terms(
        graph,
        nodes=["i", "j", "k", "l"],
        dataset="li",
        batch_reduce=None,
        cache_payloads=True,
        cache_policy="none",
    )
    ordered = cyclic_path_error_terms(
        graph,
        nodes=["i", "j", "k", "l"],
        dataset="li",
        batch_reduce=None,
        cache_payloads=True,
        cache_policy="last_use",
    )

    def peak_live_count(terms: list[GraphLossTerm]) -> int:
        live: set[tuple[str, str, tuple[str, ...]]] = set()
        peak = 0
        for idx, term in enumerate(terms):
            spec = term.term.callable.spec
            live.update(spec.cache_keys)
            peak = max(peak, len(live))
            for key, last_idx in spec.last_use.items():
                if last_idx <= idx:
                    live.discard(key)
        return peak

    assert peak_live_count(ordered) <= peak_live_count(naive)


def test_cyclic_path_error_returns_node_ordered_matrix_or_norm() -> None:
    graph = four_node_cycle()
    cyclic_error = CyclicPathError(
        graph=graph,
        nodes=["i", "j", "k", "l"],
        dataset="li",
        cache_policy="none",
    )

    matrix = cyclic_error({}, {"li": jnp.asarray([0.0])})

    assert matrix.shape == (4, 4)
    assert matrix[0, 1] == pytest.approx((1.0 - 220.0) ** 2)
    assert matrix[0, 3] == pytest.approx(abs(111.0))

    normed_error = CyclicPathError(
        graph=graph,
        nodes=["i", "j", "k", "l"],
        dataset="li",
        cache_policy="none",
        norm="sum",
    )
    assert normed_error({}, {"li": jnp.asarray([0.0])}) == pytest.approx(jnp.sum(matrix))


def test_cyclic_path_error_shared_overrides_replace_destination_defaults() -> None:
    graph = FunctionGraph(
        nodes={
            "i": Node(name="i", error_op="mae"),
            "j": Node(name="j", error_op="mae", ignore=["skip"]),
            "k": Node(name="k", error_op="mse"),
            "l": Node(name="l", error_op="mse"),
        },
        edges={
            "ij": TreeOffsetEdge(
                source="i", target="j", name="ij", forward_keep=1.0, forward_skip=10.0,
                backward_keep=2.0, backward_skip=20.0,
            ),
            "jk": TreeOffsetEdge(
                source="j", target="k", name="jk", forward_keep=3.0, forward_skip=30.0,
                backward_keep=4.0, backward_skip=40.0,
            ),
            "kl": TreeOffsetEdge(
                source="k", target="l", name="kl", forward_keep=7.0, forward_skip=70.0,
                backward_keep=8.0, backward_skip=80.0,
            ),
            "li": TreeOffsetEdge(
                source="l", target="i", name="li", forward_keep=9.0, forward_skip=90.0,
                backward_keep=10.0, backward_skip=100.0,
            ),
        },
    )
    batch = {"li": {"keep": jnp.asarray([0.0]), "skip": jnp.asarray([0.0])}}
    default_error = CyclicPathError(graph=graph, nodes=["i", "j", "k", "l"], dataset="li")
    overridden_error = CyclicPathError(
        graph=graph,
        nodes=["i", "j", "k", "l"],
        dataset="li",
        error_op="mae",
        ignore=[],
    )

    default_matrix = default_error({}, batch)
    overridden_matrix = overridden_error({}, batch)

    # The i->j destination normally ignores ``skip``; an explicit empty override includes it.
    assert default_matrix[0, 1] != overridden_matrix[0, 1]
    # The shared error operator also replaces the MSE configured on node k.
    assert default_matrix[0, 2] != overridden_matrix[0, 2]


def test_cyclic_path_error_yaml_cache_and_grad() -> None:
    yaml_text = """
!romx:CyclicPathError
graph: !romx:FunctionGraph
  nodes:
    i: {name: i, error_op: mae}
    j: {name: j, error_op: mse}
    k: {name: k, error_op: mae}
    l: {name: l, error_op: mae}
  edges:
    ij: !romx:tests.test_loss.OffsetEdge
      {source: i, target: j, name: ij, forward_offset: 1.0, backward_offset: 2.0}
    jk: !romx:tests.test_loss.OffsetEdge
      {source: j, target: k, name: jk, forward_offset: 10.0, backward_offset: 20.0}
    kl: !romx:tests.test_loss.OffsetEdge
      {source: k, target: l, name: kl, forward_offset: 100.0, backward_offset: 200.0}
    li: !romx:tests.test_loss.OffsetEdge
      {source: l, target: i, name: li, forward_offset: 1000.0, backward_offset: 2000.0}
nodes: [i, j, k, l]
dataset: li
cache_payloads: true
cache_policy: none
norm: sum
"""
    cyclic_error = YamlLoader.load(yaml_text)
    assert isinstance(cyclic_error, CyclicPathError)

    batch = {"li": jnp.asarray([0.0, 1.0, 2.0])}
    assert jnp.isfinite(cyclic_error({}, batch))
    assert jnp.isfinite(jax.grad(lambda scale: cyclic_error({}, {"li": scale * batch["li"]}))(jnp.asarray(1.0)))
