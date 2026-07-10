from __future__ import annotations

import sys
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax
import optimistix as optx
import pytest
from orbax.checkpoint import v1 as ocp
from pydantic import ValidationError

from romjax import YamlLoader
from romjax.compression import SVD
from romjax.data_gen import DataLoader, LoadImplicitModel, LoadSource
from romjax.graph import Edge, FunctionGraph, IdentityEdge, Node
from romjax.loss import GraphLoss, GraphTest, similarity_loss
from romjax.model import ImplicitSampleable
from romjax.nn import LinearProjection
from romjax.pde import ImplicitIterativeGalerkin
from romjax.rng import PyTreeSampler
from romjax.routine import RoutineError
from romjax.train import BatchLoader, CheckpointerConfig, DiagnosticsConfig, OrbaxParams, TerminationConfig, Train
from romjax.tree import is_shape_dtype, pytree_norm, pytree_resolve_refs
from romjax.typing import GraphRef
from romjax.utils import save_h5


def scalar_quadratic_loss(params: dict[str, jax.Array], batch: object) -> jax.Array:
    del batch
    return jnp.square(params["w"] - 1.0)


def scalar_zero_loss(params: dict[str, jax.Array], batch: object) -> jax.Array:
    del batch
    return jnp.square(params["w"])


def scalar_quadratic_test(params: dict[str, jax.Array]) -> jax.Array:
    return jnp.abs(params["w"] - 1.0)


def mlp_quadratic_loss(params: eqx.nn.MLP, batch: object) -> jax.Array:
    del batch
    x = jnp.array([1.0, -1.0])
    y = params(x)
    return jnp.sum(jnp.square(y - 0.5))


def graph_batch_squared_error(params: dict, single_data: dict, graph: FunctionGraph) -> jax.Array:
    del graph
    x = single_data["inputs"]["x"] if "inputs" in single_data else single_data["x"]
    return jnp.square(params["toy"]["weight"] - x)


def graph_batch_absolute_error(params: dict, single_data: dict, graph: FunctionGraph) -> jax.Array:
    del graph
    x = single_data["inputs"]["x"] if "inputs" in single_data else single_data["x"]
    return jnp.abs(params["toy"]["weight"] - x)


def graph_batch_large_squared_error(params: dict, single_data: dict, graph: FunctionGraph) -> jax.Array:
    return 100.0 * graph_batch_squared_error(params, single_data, graph)


def graph_reference_loss(params: dict, single_data: dict, graph: FunctionGraph) -> jax.Array:
    del graph
    weight = params["toy"]["weight"]
    alias = params["toy"]["alias"]
    return jnp.square(weight - single_data["x"]) + jnp.square(alias - weight)


def graph_aux_producer(
    params: dict,
    batch_data: dict,
    graph: FunctionGraph,
    aux: dict | None = None,
) -> tuple[jax.Array, dict]:
    del batch_data, graph, aux
    weight = params["toy"]["weight"]
    return jnp.square(weight), {"shared": weight + 1.0}


def graph_aux_consumer(params: dict, batch_data: dict, graph: FunctionGraph, aux: dict | None = None) -> jax.Array:
    del params, batch_data, graph
    return aux["shared"]


def graph_aux_kwargs_consumer(params: dict, batch_data: dict, graph: FunctionGraph, **kwargs: dict) -> jax.Array:
    del params, batch_data, graph
    return 2.0 * kwargs["aux"]["shared"]


def graph_batched_aux_producer(
    params: dict,
    single_data: dict,
    graph: FunctionGraph,
    aux: dict | None = None,
) -> tuple[jax.Array, dict]:
    del params, graph, aux
    return single_data["x"], {"double": 2.0 * single_data["x"]}


def graph_stacked_aux_consumer(
    params: dict,
    batch_data: dict,
    graph: FunctionGraph,
    aux: dict | None = None,
) -> jax.Array:
    del params, batch_data, graph
    return jnp.sum(aux["double"])


class ToyLinearReconstructionEdge(Edge, ImplicitSampleable):
    source: Node = Node(name="state", error_op="mse")
    target: Node = Node(name="latent")
    name: str = "toy"

    def forward(self, payload: dict[str, jax.Array]) -> dict[str, jax.Array]:
        return {"x": payload["x"]}

    def backward(self, payload: dict[str, jax.Array]) -> dict[str, jax.Array]:
        bias = payload.get("bias", 0.0)
        return {"x": payload["weight"] * payload["x"] + bias}

    def sample_inputs(self, key: jax.Array) -> dict[str, jax.Array]:
        return {"x": jax.random.normal(key, ())}

    def sample_outputs(
        self,
        key: jax.Array,
        inputs: dict[str, jax.Array] | None = None,
        solution: dict[str, jax.Array] | None = None,
    ) -> dict[str, jax.Array]:
        del key, solution
        assert inputs is not None
        return self.forward(inputs)


class TemplateAuxEdge(Edge):
    source: Node = Node(name="template_source")
    target: Node = Node(name="template_target")
    name: str = "template"

    def forward(self, x):
        return x

    def backward(self, x):
        return x

    def forward_aux(self, x, aux=None):
        del x
        if aux is None:
            raise ValueError("TemplateAuxEdge requires forward aux data.")
        return jax.tree.map(self._zeros_from_template, aux), None

    @staticmethod
    def _zeros_from_template(leaf):
        if is_shape_dtype(leaf):
            return jnp.zeros(leaf.shape, leaf.dtype)
        if eqx.is_array(leaf):
            raise AssertionError("Template aux should carry shape/dtype metadata, not arrays.")
        return leaf


class OptimistixRootFindEdge(Edge):
    source: Node = Node(name="root_source")
    target: Node = Node(name="root_target")
    name: str = "root"

    def forward(self, x):
        def residual(y, target):
            return y - target

        x_value = jnp.asarray(x["x"])
        target = jnp.asarray(x["weight"]) * x_value
        solution = optx.root_find(
            residual,
            solver=optx.Newton(rtol=1e-6, atol=1e-6),
            y0=jnp.zeros_like(target),
            args=target,
            max_steps=8,
            adjoint=optx.ImplicitAdjoint(),
            throw=False,
        )
        return {"x": x_value, "u": solution.value}

    def backward(self, x):
        return x


class RepeatBatchLoader:
    def __init__(self, batch: object):
        self.batch = batch

    def __iter__(self):
        return self

    def __next__(self):
        return self.batch


def _history_csv(path: Path) -> tuple[np.ndarray, np.ndarray]:
    arr = np.atleast_2d(np.loadtxt(path, delimiter=",", skiprows=1))
    return arr[:, 0].astype(int), arr[:, 1]


def _table_csv(path: Path) -> tuple[list[str], np.ndarray]:
    header = path.read_text(encoding="utf-8").splitlines()[0].split(",")
    return header, np.atleast_2d(np.loadtxt(path, delimiter=",", skiprows=1))


def _tree_allclose(lhs, rhs) -> bool:
    return bool(
        jax.tree.reduce(
            lambda acc, leaf: acc and bool(np.all(np.asarray(leaf))),
            jax.tree.map(lambda a, b: jnp.isclose(a, b), lhs, rhs),
            True,
        )
    )


def _write_graph_dataset(root: Path, dataset_name: str, *, n_inputs: int = 2, n_outputs: int = 2) -> None:
    for input_idx in range(n_inputs):
        input_dir = root / dataset_name / "seed_0" / f"sample_{input_idx}"
        input_dir.mkdir(parents=True, exist_ok=True)
        save_h5({"x": np.asarray(10 * input_idx + 1)}, input_dir / "input.h5", mode="w")
        save_h5({"y": np.asarray(10 * input_idx + 2)}, input_dir / "solution.h5", mode="w")
        save_h5({"r": np.asarray(-(10 * input_idx + 2))}, input_dir / "solution_residual.h5", mode="w")

        for output_idx in range(n_outputs):
            output_dir = input_dir / "seed_0" / f"sample_{output_idx}"
            output_dir.mkdir(parents=True, exist_ok=True)
            save_h5({"y": np.asarray(100 * input_idx + output_idx)}, output_dir / "output.h5", mode="w")
            save_h5({"r": np.asarray(-100 * input_idx - output_idx)}, output_dir / "residual.h5", mode="w")


def _write_source_dataset(root: Path, dataset_name: str, *, n_samples: int = 2) -> None:
    for sample_idx in range(n_samples):
        sample_dir = root / dataset_name / "seed_0" / f"sample_{sample_idx}"
        sample_dir.mkdir(parents=True, exist_ok=True)
        save_h5(
            {"x": np.asarray(1000 + sample_idx), "meta": {"i": np.asarray(sample_idx)}},
            sample_dir / "source.h5",
            mode="w",
        )


@pytest.fixture
def toy_graph() -> FunctionGraph:
    return FunctionGraph(edges={"toy": ToyLinearReconstructionEdge()})


@pytest.mark.skipif(sys.platform == "win32", reason="Orbax checkpointer issues on Windows")
def test_diagnostics(tmp_path: Path) -> None:
    callback_calls: list[tuple[float, Path | None]] = []
    checkpoint_history_steps: list[int] = []

    def progress_callback(params, graph, root):
        del graph
        callback_calls.append((float(params["w"]), root))

    def checkpoint_history_callback(params, graph, root):
        del params, graph
        assert root is not None
        loss_csv = root / "loss.csv"
        assert loss_csv.exists()
        checkpoint_history_steps.append(int(_history_csv(loss_csv)[0][-1]))

    root = tmp_path.resolve() / "diagnostics"
    train = Train(
        routine_config=dict(progress_bar={"disable": True}),
        loss=scalar_quadratic_loss,
        init_params={"w": jnp.array(0.0)},
        optimizer=optax.sgd(0.2),
        test=scalar_quadratic_test,
        termination=3,
        diagnostics=DiagnosticsConfig(
            log_interval=1,
            plot_interval=1,
            test_interval=1,
            callback_interval=1,
            progress_callback=progress_callback,
            save_plot="history.pdf",
        ),
        root=root,
        checkpointer=CheckpointerConfig(save_decision_policy=1, preservation_policy=1, step_name_format="step"),
    )

    assert train.run() == 0

    loss_steps, loss_values = _history_csv(root / "loss.csv")
    test_steps, test_values = _history_csv(root / "test.csv")
    assert root.joinpath("history.pdf").exists()
    assert loss_steps.tolist() == [0, 1, 2, 3]
    assert test_steps.tolist() == [0, 1, 2, 3]
    assert loss_values[0] > loss_values[-1]
    assert test_values[0] > test_values[-1]
    assert len(callback_calls) == 4
    assert all(call_root == root for _, call_root in callback_calls)

    final_test_root = tmp_path.resolve() / "final_test"
    final_test_train = Train(
        routine_config=dict(progress_bar={"disable": True}),
        loss=scalar_quadratic_loss,
        init_params={"w": jnp.array(0.0)},
        optimizer=optax.sgd(0.2),
        test=scalar_quadratic_test,
        termination=3,
        diagnostics=DiagnosticsConfig(test_interval=10),
        root=final_test_root,
    )
    assert final_test_train.run() == 0
    assert _history_csv(final_test_root / "loss.csv")[0].tolist() == [0, 1, 2, 3]
    assert _history_csv(final_test_root / "test.csv")[0].tolist() == [0, 3]

    checkpoint_history_root = tmp_path.resolve() / "checkpoint_history"
    checkpoint_history_train = Train(
        routine_config=dict(progress_bar={"disable": True}),
        loss=scalar_quadratic_loss,
        init_params={"w": jnp.array(0.0)},
        optimizer=optax.sgd(0.2),
        termination=2,
        diagnostics=DiagnosticsConfig(
            plot_interval=None,
            callback_interval=1,
            progress_callback=checkpoint_history_callback,
        ),
        root=checkpoint_history_root,
        checkpointer=CheckpointerConfig(save_decision_policy=2, preservation_policy=1, step_name_format="step"),
    )
    assert checkpoint_history_train.run() == 0
    assert checkpoint_history_steps == [0, 0, 2]


@pytest.mark.skipif(sys.platform == "win32", reason="Orbax checkpointer issues on Windows")
def test_train_host_interval_gates_host_diagnostics(tmp_path: Path) -> None:
    callback_values: list[float] = []
    root = tmp_path.resolve() / "host_interval"

    def progress_callback(params, graph, root):
        del graph
        assert root is not None
        callback_values.append(float(params["w"]))

    train = Train(
        routine_config=dict(progress_bar={"disable": True}),
        loss=scalar_quadratic_loss,
        init_params={"w": jnp.array(0.0)},
        optimizer=optax.sgd(0.2),
        test=scalar_quadratic_test,
        termination=5,
        diagnostics=DiagnosticsConfig(
            test_interval=1,
            callback_interval=1,
            progress_callback=progress_callback,
        ),
        host_interval=2,
        root=root,
    )

    assert train.run() == 0
    assert _history_csv(root / "loss.csv")[0].tolist() == [0, 2, 4, 5]
    assert _history_csv(root / "test.csv")[0].tolist() == [0, 2, 4, 5]
    assert len(callback_values) == 4


@pytest.mark.skipif(sys.platform == "win32", reason="Orbax checkpointer issues on Windows")
def test_train_host_interval_none_syncs_only_at_final_step(tmp_path: Path) -> None:
    root = tmp_path.resolve() / "host_interval_none"
    train = Train(
        routine_config=dict(progress_bar={"disable": True}),
        loss=scalar_quadratic_loss,
        init_params={"w": jnp.array(0.0)},
        optimizer=optax.sgd(0.2),
        test=scalar_quadratic_test,
        termination=4,
        diagnostics=DiagnosticsConfig(test_interval=1),
        host_interval=None,
        root=root,
    )

    assert train.run() == 0
    assert _history_csv(root / "loss.csv")[0].tolist() == [4]
    assert _history_csv(root / "test.csv")[0].tolist() == [4]


@pytest.mark.skipif(sys.platform == "win32", reason="Orbax checkpointer issues on Windows")
def test_train_checkpoint_interval_gates_orbax_save_checks(tmp_path: Path) -> None:
    root = tmp_path.resolve() / "checkpoint_interval"
    checkpointer = CheckpointerConfig(save_decision_policy=1, preservation_policy=1, step_name_format="step")
    train = Train(
        routine_config=dict(progress_bar={"disable": True}),
        loss=scalar_quadratic_loss,
        init_params={"w": jnp.array(0.0)},
        optimizer=optax.sgd(0.2),
        termination=5,
        checkpoint_interval=2,
        root=root,
        checkpointer=checkpointer,
    )

    assert train.run() == 0
    checkpoint_names = sorted(path.name for path in root.iterdir() if path.is_dir())
    assert checkpoint_names == ["step_0", "step_2", "step_4", "step_5"]


def test_train_checkpoint_interval_infers_fixed_orbax_save_policy() -> None:
    train = Train(
        loss=scalar_quadratic_loss,
        init_params={"w": jnp.array(0.0)},
        optimizer=optax.sgd(0.2),
        checkpointer=CheckpointerConfig(save_decision_policy=3),
    )

    explicit = Train(
        loss=scalar_quadratic_loss,
        init_params={"w": jnp.array(0.0)},
        optimizer=optax.sgd(0.2),
        checkpoint_interval=2,
        checkpointer=CheckpointerConfig(save_decision_policy=3),
    )

    assert train.checkpoint_interval == 3
    assert explicit.checkpoint_interval == 2


@pytest.mark.skipif(sys.platform == "win32", reason="Orbax checkpointer issues on Windows")
def test_termination(tmp_path: Path) -> None:
    loss_tol_root = tmp_path.resolve() / "loss_tol"
    loss_tol_train = Train(
        loss=scalar_zero_loss,
        init_params={"w": jnp.array(0.0)},
        optimizer=optax.sgd(0.1),
        termination=TerminationConfig(max_steps=5, loss_tol=1e-8),
        root=loss_tol_root,
    )
    assert loss_tol_train.run() == 0
    assert _history_csv(loss_tol_root / "loss.csv")[0].tolist() == [0]

    test_tol_root = tmp_path.resolve() / "test_tol"
    test_tol_train = Train(
        loss=scalar_zero_loss,
        init_params={"w": jnp.array(0.0)},
        optimizer=optax.sgd(0.1),
        test=scalar_quadratic_test,
        termination=TerminationConfig(max_steps=5, test_tol=1.1),
        diagnostics=DiagnosticsConfig(test_interval=1),
        root=test_tol_root,
    )
    assert test_tol_train.run() == 0
    assert _history_csv(test_tol_root / "test.csv")[0].tolist() == [0]

    grad_tol_root = tmp_path.resolve() / "grad_tol"
    grad_tol_train = Train(
        loss=scalar_zero_loss,
        init_params={"w": jnp.array(0.0)},
        optimizer=optax.sgd(0.1),
        termination=TerminationConfig(max_steps=5, grad_tol=1e-8),
        root=grad_tol_root,
    )
    assert grad_tol_train.run() == 0
    assert _history_csv(grad_tol_root / "loss.csv")[0].tolist() == [0]

    runtime = TerminationConfig(max_steps=50, max_runtime="PT0.001S")
    runtime_root = tmp_path.resolve() / "runtime_tol"
    runtime_train = Train(
        loss=scalar_quadratic_loss,
        init_params={"w": jnp.array(0.0)},
        optimizer=optax.sgd(0.1),
        termination=runtime,
        root=runtime_root,
    )
    assert runtime.max_runtime.total_seconds() == pytest.approx(0.001)
    assert runtime_train.run() == 0
    assert _history_csv(runtime_root / "loss.csv")[0].tolist() == [0]


@pytest.mark.skipif(sys.platform == "win32", reason="Orbax checkpointer issues on Windows")
def test_checkpointer_and_restart(tmp_path: Path) -> None:
    checkpointer = CheckpointerConfig(
        save_decision_policy=1,
        preservation_policy=1,
        step_name_format={"step_prefix": "train", "step_format_fixed_length": 3},
    )
    root = tmp_path.resolve() / "reuse"

    first = Train(
        loss=scalar_quadratic_loss,
        init_params={"w": jnp.array(0.0)},
        optimizer=optax.sgd(0.2),
        termination=TerminationConfig(max_steps=2),
        root=root,
        checkpointer=checkpointer,
    )
    assert first.run() == 0

    checkpoint_names = sorted(path.name for path in root.iterdir() if path.is_dir())
    assert checkpoint_names[:2] == ["train_000", "train_001"]
    with ocp.training.Checkpointer(root, **dict(checkpointer)) as ckptr:
        assert ckptr.latest is not None
        assert ckptr.latest.step == 2

    resumed = Train(
        loss=scalar_quadratic_loss,
        init_params={"w": jnp.array(5.0)},
        optimizer=optax.sgd(0.2),
        termination=TerminationConfig(max_steps=4),
        root=root,
        write_policy="reuse",
        checkpointer=checkpointer,
    )
    assert resumed.run() == 0
    assert _history_csv(root / "loss.csv")[0].tolist() == [0, 1, 2, 3, 4]
    with ocp.training.Checkpointer(root, **dict(checkpointer)) as ckptr:
        assert ckptr.latest is not None
        assert ckptr.latest.step == 4

    overwrite_root = tmp_path.resolve() / "overwrite"
    overwrite_root.mkdir(parents=True, exist_ok=True)
    overwrite_root.joinpath("sentinel.txt").write_text("keep-me")
    overwrite_root.joinpath("loss.csv").write_text("Iteration,Value\n0,1.0\n")
    overwrite_root.joinpath("history.pdf").write_text("old-figure")
    overwrite_root.joinpath("notes").mkdir()
    overwrite_root.joinpath("train_999").mkdir()
    overwrite_root.joinpath("train_999", "_CHECKPOINT_METADATA").write_text("")
    overwrite = Train(
        loss=scalar_quadratic_loss,
        init_params={"w": jnp.array(0.0)},
        optimizer=optax.sgd(0.2),
        termination=TerminationConfig(max_steps=2),
        root=overwrite_root,
        write_policy="overwrite",
        checkpointer=checkpointer,
    )
    assert overwrite_root.joinpath("sentinel.txt").exists()
    assert overwrite_root.joinpath("notes").exists()
    assert not overwrite_root.joinpath("loss.csv").exists()
    assert not overwrite_root.joinpath("history.pdf").exists()
    assert not overwrite_root.joinpath("train_999").exists()
    assert overwrite.run() == 0

    error_root = tmp_path.resolve() / "error"
    error_root.mkdir(parents=True, exist_ok=True)
    error_root.joinpath("sentinel.txt").write_text("keep-me")
    with pytest.raises(RoutineError):
        Train(
            loss=scalar_quadratic_loss,
            init_params={"w": jnp.array(0.0)},
            optimizer=optax.sgd(0.2),
            termination=TerminationConfig(max_steps=2),
            root=error_root,
            write_policy="error",
            checkpointer=checkpointer,
        )


@pytest.mark.skipif(sys.platform == "win32", reason="Orbax checkpointer issues on Windows")
def test_checkpointer_restores_optax_namedtuple_state(tmp_path: Path) -> None:
    checkpointer = CheckpointerConfig(
        save_decision_policy=1,
        preservation_policy=1,
        step_name_format={"step_prefix": "train", "step_format_fixed_length": 3},
    )
    root = tmp_path.resolve() / "adam_reuse"

    first = Train(
        loss=scalar_quadratic_loss,
        init_params={"w": jnp.array(0.0)},
        optimizer=optax.adam(0.2),
        termination=TerminationConfig(max_steps=2),
        root=root,
        checkpointer=checkpointer,
    )
    assert first.run() == 0

    resumed = Train(
        loss=scalar_quadratic_loss,
        init_params={"w": jnp.array(5.0)},
        optimizer=optax.adam(0.2),
        termination=TerminationConfig(max_steps=4),
        root=root,
        write_policy="reuse",
        checkpointer=checkpointer,
    )
    assert resumed.run() == 0
    assert _history_csv(root / "loss.csv")[0].tolist() == [0, 1, 2, 3, 4]

    with ocp.training.Checkpointer(root, **dict(checkpointer)) as ckptr:
        assert ckptr.latest is not None
        assert ckptr.latest.step == 4
        loaded = ckptr.load_checkpointables(
            abstract_checkpointables={
                "params": {"w": jnp.array(0.0)},
                "opt_state": optax.adam(0.2).init({"w": jnp.array(0.0)}),
            }
        )

    assert hasattr(loaded["opt_state"][0], "mu")


@pytest.mark.skipif(sys.platform == "win32", reason="Orbax checkpointer issues on Windows")
def test_checkpointer_reuses_eqx_module_params(tmp_path: Path) -> None:
    checkpointer = CheckpointerConfig(
        save_decision_policy=1,
        preservation_policy=1,
        step_name_format={"step_prefix": "train", "step_format_fixed_length": 3},
    )
    root = tmp_path.resolve() / "mlp_reuse"
    init_params = eqx.nn.MLP(in_size=2, out_size=1, width_size=4, depth=2, key=jax.random.key(0))

    first = Train(
        loss=mlp_quadratic_loss,
        init_params=init_params,
        optimizer=optax.adam(0.1),
        termination=TerminationConfig(max_steps=2),
        root=root,
        checkpointer=checkpointer,
    )
    assert first.run() == 0

    resumed = Train(
        loss=mlp_quadratic_loss,
        init_params=eqx.nn.MLP(in_size=2, out_size=1, width_size=4, depth=2, key=jax.random.key(999)),
        optimizer=optax.adam(0.1),
        termination=TerminationConfig(max_steps=4),
        root=root,
        write_policy="reuse",
        checkpointer=checkpointer,
    )
    assert resumed.run() == 0
    assert _history_csv(root / "loss.csv")[0].tolist() == [0, 1, 2, 3, 4]

    with ocp.training.Checkpointer(root, **dict(checkpointer)) as ckptr:
        assert ckptr.latest is not None
        loaded = ckptr.load_checkpointables(
            abstract_checkpointables={
                "params": eqx.filter(init_params, eqx.is_array),
                "opt_state": optax.adam(0.1).init(eqx.filter(init_params, eqx.is_array)),
            }
        )

    assert isinstance(resumed.init_params, eqx.nn.MLP)
    assert isinstance(loaded["params"], eqx.nn.MLP)
    assert hasattr(loaded["opt_state"][0], "mu")


@pytest.mark.filterwarnings("ignore:Sharding info:UserWarning")
@pytest.mark.skipif(sys.platform == "win32", reason="Orbax checkpointer issues on Windows")
def test_train_load_orbax_warm_starts_fresh_run(tmp_path: Path) -> None:
    checkpoint_root = tmp_path.resolve() / "source"
    with ocp.training.Checkpointer(checkpoint_root) as ckptr:
        ckptr.save_checkpointables(
            step=0,
            checkpointables={"params": {"w": jnp.array(3.0)}},
            force=True,
        )

    root = tmp_path.resolve() / "warm_start"
    train = Train(
        loss=scalar_quadratic_loss,
        init_params={"w": jnp.array(0.0)},
        optimizer=optax.sgd(0.1),
        termination=TerminationConfig(max_steps=1),
        root=root,
        load_orbax=checkpoint_root,
    )

    assert isinstance(train.load_orbax, OrbaxParams)
    assert train.run() == 0
    assert _history_csv(root / "loss.csv")[1].tolist() == pytest.approx([4.0, 2.56])


@pytest.mark.skipif(sys.platform == "win32", reason="Orbax checkpointer issues on Windows")
def test_train_load_orbax_uses_eqx_static_template(tmp_path: Path) -> None:
    checkpoint_root = tmp_path.resolve() / "mlp_source"
    source_params = eqx.nn.MLP(in_size=2, out_size=1, width_size=4, depth=2, key=jax.random.key(0))
    init_params = eqx.nn.MLP(in_size=2, out_size=1, width_size=4, depth=2, key=jax.random.key(999))

    with ocp.training.Checkpointer(checkpoint_root) as ckptr:
        ckptr.save_checkpointables(
            step=0,
            checkpointables={"params": eqx.filter(source_params, eqx.is_array)},
            force=True,
        )

    train = Train(
        loss=mlp_quadratic_loss,
        init_params=init_params,
        optimizer=optax.sgd(0.0),
        termination=TerminationConfig(max_steps=1),
        load_orbax={"params": checkpoint_root},
    )
    loaded_params = train()

    assert isinstance(loaded_params, eqx.nn.MLP)
    np.testing.assert_allclose(loaded_params.layers[0].weight, source_params.layers[0].weight)
    np.testing.assert_allclose(loaded_params.layers[0].bias, source_params.layers[0].bias)


def test_orbax_params_compare_import_compatibility() -> None:
    from romjax.compare import OrbaxParams as CompareOrbaxParams

    assert CompareOrbaxParams is OrbaxParams


@pytest.mark.filterwarnings("ignore:Sharding info:UserWarning")
@pytest.mark.skipif(sys.platform == "win32", reason="Orbax checkpointer issues on Windows")
def test_orbax_params_resolve_params_accepts_shape_dtype_template(tmp_path: Path) -> None:
    checkpoint_root = tmp_path.resolve() / "shape_dtype_template"
    source_params = {"w": jnp.array([1.0, 2.0], dtype=jnp.float32), "b": jnp.array(3.0, dtype=jnp.float32)}
    template = {
        "w": jax.ShapeDtypeStruct(source_params["w"].shape, source_params["w"].dtype),
        "b": jax.ShapeDtypeStruct(source_params["b"].shape, source_params["b"].dtype),
    }

    with ocp.training.Checkpointer(checkpoint_root) as ckptr:
        ckptr.save_checkpointables(step=0, checkpointables={"params": source_params}, force=True)

    loaded_params = OrbaxParams(params=checkpoint_root).resolve_params(template)

    assert loaded_params is not None
    np.testing.assert_allclose(loaded_params["w"], source_params["w"])
    np.testing.assert_allclose(loaded_params["b"], source_params["b"])


@pytest.mark.filterwarnings("ignore:Sharding info:UserWarning")
@pytest.mark.skipif(sys.platform == "win32", reason="Orbax checkpointer issues on Windows")
def test_orbax_params_resolve_params_accepts_yaml_shape_dtype_template(tmp_path: Path) -> None:
    checkpoint_root = tmp_path.resolve() / "yaml_shape_dtype_template"
    source_params = {"w": jnp.array([1.0, 2.0], dtype=jnp.float32), "b": jnp.array(3.0, dtype=jnp.float32)}
    template = {"w": {"shape": [2], "dtype": "float32"}, "b": {"shape": []}}

    with ocp.training.Checkpointer(checkpoint_root) as ckptr:
        ckptr.save_checkpointables(step=0, checkpointables={"params": source_params}, force=True)

    loaded_params = OrbaxParams(params=checkpoint_root).resolve_params(template)

    assert loaded_params is not None
    np.testing.assert_allclose(loaded_params["w"], source_params["w"])
    np.testing.assert_allclose(loaded_params["b"], source_params["b"])


def test_basic_batch_loader():
    empty_loader = BatchLoader()
    assert next(empty_loader) == ()

    data = np.arange(5)
    loader = BatchLoader(data=data, batch_size=2, shuffle_seed=0, max_epochs=1)
    assert [batch.tolist() for batch in loader] == [[2, 4], [3, 0], [1]]

    deterministic_a = BatchLoader(data=data, batch_size=2, shuffle_seed=7, max_epochs=2)
    deterministic_b = BatchLoader(data=data, batch_size=2, shuffle_seed=7, max_epochs=2)
    assert [batch.tolist() for batch in deterministic_a] == [batch.tolist() for batch in deterministic_b]

    paired_data = ([1, 2, 3], [4, 5, 6])
    paired_loader = BatchLoader(data=paired_data, batch_size=2, max_epochs=1)
    assert list(paired_loader) == [([1, 2], [4, 5]), ([3], [6])]

    with pytest.raises(ValueError, match="equal length"):
        BatchLoader(data=([1, 2], [3]), batch_size=1)


def test_train_convergence():
    def rosenbrock_loss(params: dict[str, jax.Array], batch: object) -> jax.Array:
        del batch
        x, y = params["x"]
        return jnp.square(1.0 - x) + 100.0 * jnp.square(y - jnp.square(x))

    train = Train(
        loss=rosenbrock_loss,
        init_params={"x": jnp.array([-1.2, 1.0])},
        optimizer=optax.adam(0.01),
        termination=5_000,
    )

    params = train()
    expected = {"x": jnp.array([1.0, 1.0])}
    assert jnp.allclose(params["x"], expected["x"], atol=1e-3)
    assert float(rosenbrock_loss(params, None)) == pytest.approx(0.0, abs=1e-6)



def test_graph_init_params(toy_graph: FunctionGraph) -> None:
    train = Train(
        loss=GraphLoss(terms=["reconstruction"]),
        init_params=PyTreeSampler(
            template={
                "toy": {
                    "weight": {"callable": "normal", "mean": 0.0, "std": 0.1, "shape": ()},
                    "bias": 0.25,
                    "alias": "toy,weight",
                    "module": {"name": "LinearProjection", "kwargs": {"latent": 1, "dof": 1}},
                }
            }
        ),
        optimizer=optax.sgd(0.1),
        graph=toy_graph,
        init_seed=7,
    )

    assert train.loss.graph is toy_graph
    assert train.loss.terms[0].dataset == "toy"
    assert isinstance(train.init_params["toy"]["module"], LinearProjection)
    assert jnp.shape(train.init_params["toy"]["weight"]) == ()
    assert float(train.init_params["toy"]["bias"]) == pytest.approx(0.25)
    assert train.init_params["toy"]["alias"] == "toy,weight"


def test_data_loader(tmp_path: Path) -> None:
    root = tmp_path.resolve() / "graph_data"
    _write_graph_dataset(root, "beta")
    _write_graph_dataset(root, "alpha")

    loader = DataLoader(root=root)
    assert sorted(loader.datasets) == ["alpha", "beta"]
    assert isinstance(loader.datasets["alpha"], LoadImplicitModel)
    assert loader.datasets["alpha"].stack_batch is False

    first_batch = next(loader)
    assert set(first_batch) == {"alpha", "beta"}
    assert len(first_batch["alpha"]) == 6
    assert len(first_batch["beta"]) == 6
    assert first_batch["alpha"][0]["inputs"]["x"].shape == ()
    assert first_batch["alpha"][0]["outputs"]["y"].shape == ()
    assert first_batch["alpha"][0]["residuals"]["r"].shape == ()

    skipped = DataLoader(
        root=root,
        datasets={
            "alpha": {
                "kind": "implicit",
                "batch_size": 4,
                "skip_input": lambda path: path.name == "sample_0",
                "skip_output": lambda path: path.name == "sample_0",
                "load_solution": False,
                "max_epochs": 1,
            }
        },
    )
    skipped_batch = next(skipped)
    assert len(skipped_batch["alpha"]) == 1
    assert skipped_batch["alpha"][0]["inputs"]["x"].tolist() == 11
    assert skipped_batch["alpha"][0]["outputs"]["y"].tolist() == 101

    limited = DataLoader(
        root=root,
        datasets={
            "alpha": {
                "kind": "implicit",
                "batch_size": 4,
                "max_samples": 1,
                "max_input_samples": 1,
                "max_outputs_per_input": 1,
                "load_solution": False,
                "max_epochs": 1,
            }
        },
    )
    limited_batch = next(limited)
    assert len(limited_batch["alpha"]) == 1
    assert limited_batch["alpha"][0]["inputs"]["x"].shape == ()
    assert limited_batch["alpha"][0]["outputs"]["y"].shape == ()

    iterator = iter(DataLoader(root=root))
    assert set(next(iterator)) == {"alpha", "beta"}
    assert set(next(iterator)) == {"alpha", "beta"}

    resumed_source = DataLoader(root=root)
    _ = next(resumed_source)
    second_batch = next(resumed_source)

    resumed = DataLoader(root=root)
    resumed.set_iterator(train_step=1)
    assert _tree_allclose(second_batch, next(resumed))


def test_data_loader_config_types(tmp_path: Path) -> None:
    root = tmp_path.resolve() / "graph_data_types"
    _write_graph_dataset(root, "alpha")
    _write_source_dataset(root, "source")

    inferred = DataLoader(root=root)
    assert isinstance(inferred.datasets["alpha"], LoadImplicitModel)
    assert isinstance(inferred.datasets["source"], LoadSource)

    explicit = DataLoader(
        root=root,
        datasets={
            "alpha": {"batch_size": 1, "max_epochs": 1},
            "source": {"kind": "source", "batch_size": 1, "max_epochs": 1},
        },
    )
    assert isinstance(explicit.datasets["alpha"], LoadImplicitModel)
    assert isinstance(explicit.datasets["source"], LoadSource)


def test_graph_loss(toy_graph: FunctionGraph) -> None:
    batch = {"toy": [{"x": jnp.array([1.0])}, {"x": jnp.array([2.0])}, {"x": jnp.array([3.0])}]}

    squared = GraphLoss(terms=[{"term": graph_batch_squared_error, "dataset": "toy"}], graph=toy_graph)
    params = {"toy": {"weight": jnp.array(0.5)}}
    assert squared(params, batch) == pytest.approx(np.mean((0.5 - np.array([1.0, 2.0, 3.0])) ** 2))
    assert squared.term_names == ("term_0",)
    loss_value, (raw_terms, scaled_terms) = squared(params, batch, return_aux=True)
    assert loss_value == pytest.approx(squared(params, batch))
    assert raw_terms["term_0"] == pytest.approx(np.mean((0.5 - np.array([1.0, 2.0, 3.0])) ** 2))
    assert scaled_terms["term_0"] == pytest.approx(loss_value)

    reference = GraphLoss(terms=[{"term": graph_reference_loss, "dataset": "toy"}], graph=toy_graph)
    ref_params = {"toy": {"weight": jnp.array(0.5), "alias": "toy,weight"}}
    resolved_params = pytree_resolve_refs(ref_params)
    assert reference(ref_params, batch) == pytest.approx(squared(params, batch))
    assert eqx.filter_jit(reference)(ref_params, batch) == pytest.approx(squared(params, batch))
    assert jax.jit(lambda data: reference(resolved_params, data))(batch) == pytest.approx(squared(params, batch))
    grad = jax.grad(lambda weight: reference({"toy": {"weight": weight, "alias": weight}}, batch))(jnp.array(0.5))
    assert jnp.isfinite(grad)
    assert grad == pytest.approx(-3)
    stacked_x = jnp.array([[1.0, 2.0, 3.0], [2.0, 3.0, 4.0]])
    vmapped = jax.vmap(lambda x: reference(resolved_params, {"toy": {"x": x}}))(stacked_x)
    assert vmapped.shape == (2,)
    assert np.all(np.isfinite(np.asarray(vmapped)))

    weighted = GraphLoss(
        terms=[
            {"term": graph_batch_squared_error, "dataset": "toy", "weight": 2.0},
            {"term": graph_batch_absolute_error, "dataset": "toy", "weight": 0.5},
        ],
        graph=toy_graph,
    )
    expected_weighted = (
        2.0 * np.mean((0.5 - np.array([1.0, 2.0, 3.0])) ** 2)
        + 0.5 * np.mean(np.abs(0.5 - np.array([1.0, 2.0, 3.0])))
    )
    assert weighted(params, batch) == pytest.approx(expected_weighted)
    scaled_loss, (raw_terms, scaled_terms) = weighted(
        params,
        batch,
        scales={"term_0": jnp.asarray(0.25), "term_1": jnp.asarray(4.0)},
        return_aux=True,
    )
    assert scaled_terms["term_0"] == pytest.approx(2.0 * 0.25 * raw_terms["term_0"])
    assert scaled_terms["term_1"] == pytest.approx(0.5 * 4.0 * raw_terms["term_1"])
    assert scaled_loss == pytest.approx(scaled_terms["term_0"] + scaled_terms["term_1"])

    reconstructed = GraphLoss(
        terms=[
            {"callable": "reconstruction", "path": ["toy"]},
            {"term": "tikhonov", "weight": 0.1, "batch_reduce": None},
        ],
        graph=toy_graph,
    )
    recon_params = {"toy": {"weight": jnp.array(0.5), "bias": jnp.array(0.2)}}
    expected_reconstruction = np.mean((np.array([1.0, 2.0, 3.0]) - (0.5 * np.array([1.0, 2.0, 3.0]) + 0.2)) ** 2)
    expected_regularization = 0.1 * float(pytree_norm(recon_params)**2)
    assert reconstructed(recon_params, batch) == pytest.approx(expected_reconstruction + expected_regularization)


def test_graph_loss_passes_aux_between_terms(toy_graph: FunctionGraph) -> None:
    batch = {"toy": [{"x": jnp.array([1.0])}]}
    params = {"toy": {"weight": jnp.array(2.0)}}
    loss = GraphLoss(
        terms=[
            {"term": graph_aux_producer, "batch_reduce": None},
            {"term": graph_aux_consumer, "batch_reduce": None},
            {"term": graph_aux_kwargs_consumer, "batch_reduce": None},
        ],
        graph=toy_graph,
    )

    value, (raw_terms, scaled_terms) = loss(params, batch, return_aux=True)

    assert value == pytest.approx(13.0)
    assert raw_terms["term_0"] == pytest.approx(4.0)
    assert raw_terms["term_1"] == pytest.approx(3.0)
    assert raw_terms["term_2"] == pytest.approx(6.0)
    assert scaled_terms["term_0"] == pytest.approx(4.0)
    assert eqx.filter_jit(loss)(params, batch) == pytest.approx(13.0)

    grad = jax.grad(lambda weight: loss({"toy": {"weight": weight}}, batch))(jnp.array(2.0))
    assert grad == pytest.approx(7.0)


def test_graph_loss_stacks_batched_aux_for_downstream_terms(toy_graph: FunctionGraph) -> None:
    batch = {"toy": [{"x": jnp.array(1.0)}, {"x": jnp.array(2.0)}, {"x": jnp.array(3.0)}]}
    params = {"toy": {"weight": jnp.array(0.0)}}
    loss = GraphLoss(
        terms=[
            {"term": graph_batched_aux_producer, "dataset": "toy"},
            {"term": graph_stacked_aux_consumer, "batch_reduce": None},
        ],
        graph=toy_graph,
    )

    value, (raw_terms, scaled_terms) = loss(params, batch, return_aux=True)

    assert raw_terms["term_0"] == pytest.approx(2.0)
    assert raw_terms["term_1"] == pytest.approx(12.0)
    assert scaled_terms["term_1"] == pytest.approx(12.0)
    assert value == pytest.approx(14.0)


def test_graph_loss_balancing_config_and_term_names(toy_graph: FunctionGraph) -> None:
    loss = GraphLoss(
        terms=[
            {"name": "small", "term": graph_batch_squared_error, "dataset": "toy"},
            {"name": "large", "term": graph_batch_large_squared_error, "dataset": "toy"},
        ],
        balancing={"kind": "ema", "decay": 0.5, "min_scale": 1e-4},
        graph=toy_graph,
    )

    assert loss.balancing.kind == "ema"
    assert loss.balancing.decay == pytest.approx(0.5)
    assert loss.balancing.bootstrap is True
    assert loss.term_names == ("small", "large")

    with pytest.raises(ValidationError):
        GraphLoss(
            terms=[
                {"name": "duplicate", "term": graph_batch_squared_error, "dataset": "toy"},
                {"name": "duplicate", "term": graph_batch_absolute_error, "dataset": "toy"},
            ],
            graph=toy_graph,
        )

    with pytest.raises(ValidationError):
        GraphLoss(
            terms=[{"term": graph_batch_squared_error, "dataset": "toy"}],
            balancing={"kind": "ema", "decay": 1.0},
            graph=toy_graph,
        )

    with pytest.raises(ValidationError):
        GraphLoss(
            terms=[{"term": graph_batch_squared_error, "dataset": "toy"}],
            balancing={"kind": "ema", "normalize": "median"},
            graph=toy_graph,
        )

    log_loss = GraphLoss(
        terms=[{"term": graph_batch_squared_error, "dataset": "toy"}],
        balancing={"kind": "ema_log", "normalize": True},
        graph=toy_graph,
    )
    assert log_loss.balancing.kind == "ema_log"
    assert log_loss.balancing.normalize


def test_graph_loss_scale_arrays_do_not_retrace_on_value_changes(toy_graph: FunctionGraph) -> None:
    batch = {"toy": [{"x": jnp.asarray([1.0], dtype=jnp.float32)}]}
    params = {"toy": {"weight": jnp.asarray(0.0, dtype=jnp.float32)}}
    loss = GraphLoss(
        terms=[
            {"name": "small", "term": graph_batch_squared_error, "dataset": "toy"},
            {"name": "large", "term": graph_batch_large_squared_error, "dataset": "toy"},
        ],
        graph=toy_graph,
    )
    trace_count = 0

    @eqx.filter_jit
    def loss_and_grad(params, batch, scales):
        nonlocal trace_count
        trace_count += 1

        def _loss(params, batch):
            return loss(params, batch, scales=scales)

        return eqx.filter_value_and_grad(_loss)(params, batch)

    scales_a = {"small": jnp.asarray(1.0, dtype=jnp.float32), "large": jnp.asarray(0.01, dtype=jnp.float32)}
    scales_b = {"small": jnp.asarray(0.5, dtype=jnp.float32), "large": jnp.asarray(0.02, dtype=jnp.float32)}

    value_a, _ = loss_and_grad(params, batch, scales_a)
    value_b, _ = loss_and_grad(params, batch, scales_b)
    jax.block_until_ready((value_a, value_b))

    assert trace_count == 1
    assert value_a != value_b


def test_graph_validation(tmp_path: Path, toy_graph: FunctionGraph) -> None:
    data_root = tmp_path.resolve() / "graph_validation"
    _write_graph_dataset(data_root, "toy", n_inputs=1, n_outputs=1)
    dataloader = DataLoader(
        root=data_root,
        datasets={"toy": {"kind": "implicit", "batch_size": 1, "max_epochs": 2}},
    )
    test = GraphTest(
        terms=[{"term": graph_batch_squared_error, "dataset": "toy"}],
        graph=toy_graph,
        loader=dataloader,
        reduce="mean",
    )

    value = test({"toy": {"weight": jnp.array(0.0)}})
    expected = np.mean([np.mean((0.0 - np.array([1])) ** 2), np.mean((0.0 - np.array([1])) ** 2)])
    assert value == pytest.approx(expected)


def test_load_train_from_yaml(tmp_path: Path) -> None:
    data_root = tmp_path.resolve() / "yaml_data"
    run_root = tmp_path.resolve() / "yaml_run"
    _write_graph_dataset(data_root, "toyset", n_inputs=1, n_outputs=1)

    yaml_text = f"""
train: !romx:Train
  graph: !romx:FunctionGraph
    edges:
      toy: !pd:tests.test_train.ToyLinearReconstructionEdge
        source: state
        target: latent
        name: toy
  loss: !romx:GraphLoss
    terms:
      - reconstruction
      - term: !!python/name:tests.test_train.graph_batch_squared_error
        weight: 0.5
        dataset: toy
  test: !romx:GraphTest
    terms:
      - reconstruction
    loader: !romx:DataLoader
      root: {data_root}
      datasets:
        toyset:
          kind: implicit
          batch_size: 1
  dataloader: !romx:DataLoader
    root: {data_root}
    datasets:
      toyset:
        kind: implicit
        batch_size: 1
  init_params: !romx:PyTreeSampler
    template:
      toy:
        weight:
          callable: normal
          mean: 0.0
          std: 0.1
          shape: []
        bias: 0.25
        alias: toy,weight
        module:
          name: LinearProjection
          kwargs:
            latent: 1
            dof: 1
  optimizer:
    name: sgd
    args: [0.1]
  root: {run_root}
"""
    train = YamlLoader.load(yaml_text)["train"]

    assert isinstance(train, Train)
    assert isinstance(train.loss, GraphLoss)
    assert isinstance(train.test, GraphTest)
    assert isinstance(train.dataloader, DataLoader)
    assert isinstance(train.dataloader.datasets["toyset"], LoadImplicitModel)
    assert isinstance(train.init_params["toy"]["module"], LinearProjection)
    assert train.root == run_root.resolve()
    assert train.loss.graph is train.graph
    assert train.test.graph is train.graph
    assert train.loss.terms[0].dataset == "toy"
    assert train.test.terms[0].dataset == "toy"
    assert train.init_params["toy"]["alias"] == "toy,weight"


def test_train_initialization_resolves_graph_latent_dim_from_source_sampler(tmp_path: Path) -> None:
    compression = SVD(
        energy_tol=0.9,
        center=False,
        rank=3,
        mean=np.asarray([0.0, 0.0, 0.0, 0.0]),
        basis=np.asarray(
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
            ]
        ),
        singular_values=np.asarray([3.0, 2.0, 1.0]),
        minval=np.asarray([-1.0, -1.0, -1.0]),
        maxval=np.asarray([1.0, 1.0, 1.0]),
    )

    graph = FunctionGraph(
        edges={
            "ab": IdentityEdge(source="a", target="b", name="ab"),
            "bc": IdentityEdge(source="b", target="c", name="bc"),
            "galerkin": ImplicitIterativeGalerkin(
                source="a",
                target="c",
                name="galerkin",
                path=["ab", "bc"],
                compression=compression,
            ),
        }
    )
    train = Train(
        loss=scalar_zero_loss,
        init_params=PyTreeSampler(
            toy={
                "module": {
                    "name": "LinearProjection",
                    "kwargs": {
                        "latent": GraphRef(path=("edges", "galerkin", "compression", "rank")),
                        "dof": 4,
                    },
                }
            }
        ),
        optimizer=optax.sgd(0.1),
        graph=graph,
    )

    assert train.init_params["toy"]["module"].matrix.shape == (3, 4)
    assert train.graph.edges["galerkin"].resolve_latent_dim() == 3


@pytest.mark.skipif(sys.platform == "win32", reason="Orbax checkpointer issues on Windows")
def test_run_graph_train(tmp_path: Path, toy_graph: FunctionGraph) -> None:
    train_batch = {"toy": {"x": jnp.array([1.0, 2.0, 3.0])}}
    validation_root = tmp_path.resolve() / "validation_graph_train"
    _write_graph_dataset(validation_root, "toy", n_inputs=1, n_outputs=1)
    validation_loader = DataLoader(
        root=validation_root,
        datasets={"toy": {"kind": "implicit", "batch_size": 1, "max_epochs": 2}},
    )
    loss = GraphLoss(terms=[{"callable": "reconstruction", "path": "toy"}])

    def validation_error(params: dict, single_data: dict, graph: FunctionGraph) -> jax.Array:
        del graph
        x = single_data["inputs"]["x"] if "inputs" in single_data else single_data["x"]
        return jnp.square(x - (params["toy"]["weight"] * x + params["toy"]["bias"]))

    test = GraphTest(
        terms=[{"callable": validation_error}],
        loader=validation_loader,
    )

    max_steps = 20
    root = tmp_path.resolve() / "graph_train"
    train = Train(
        loss=loss,
        init_params={"toy": {"weight": jnp.array(0.0), "bias": jnp.array(0.0)}},
        optimizer=optax.sgd(0.15),
        test=test,
        dataloader=RepeatBatchLoader(train_batch),
        termination=TerminationConfig(max_steps=max_steps),
        diagnostics=DiagnosticsConfig(test_interval=1),
        root=root,
        graph=toy_graph,
    )

    assert train.loss.graph is toy_graph
    assert train.test.graph is toy_graph
    assert train.run() == 0

    loss_steps, loss_values = _history_csv(root / "loss.csv")
    test_steps, test_values = _history_csv(root / "test.csv")
    assert loss_steps.tolist() == list(range(max_steps + 1))
    assert test_steps.tolist() == list(range(max_steps + 1))
    assert loss_values[-1] < 0.1 * loss_values[0]
    assert test_values[-1] < test_values[0]

    with ocp.training.Checkpointer(root) as ckptr:
        assert ckptr.latest is not None
        assert ckptr.latest.step == max_steps
        params = ckptr.load_checkpointables(
            abstract_checkpointables={
                "params": {"toy": {"weight": jnp.array(0.0), "bias": jnp.array(0.0)}}
            }
        )["params"]
    assert float(params["toy"]["weight"]) == pytest.approx(1.0, abs=0.2)
    assert abs(float(params["toy"]["bias"])) < 0.2


def test_similarity_loss_template_paths_adds_edge_aux_template() -> None:
    graph = FunctionGraph(
        edges={
            "pass": IdentityEdge(source="state", target="template_source", name="pass"),
            "template": TemplateAuxEdge(),
        }
    )
    single_data = {"x": jnp.array([2.0, -3.0]), "nested": {"y": jnp.array(4.0)}}

    loss = similarity_loss(
        {},
        single_data,
        graph,
        path=["pass", "template"],
        template_paths=[("nested", "y")],
        aux_paths=[("template", "forward")],
    )

    assert float(loss) == pytest.approx(4.0**2)


def test_similarity_loss_trains_through_optimistix_root_find_edge() -> None:
    graph = FunctionGraph(edges={"root": OptimistixRootFindEdge()})
    batch = {"root": [{"x": jnp.array(2.0), "u": jnp.array(6.0)}]}
    train = Train(
        loss=GraphLoss(
            terms=[{"term": {"callable": "similarity", "path": "root"}, "dataset": "root"}],
            graph=graph,
        ),
        init_params={"root": {"weight": jnp.array(0.0)}},
        optimizer=optax.sgd(0.05),
        dataloader=RepeatBatchLoader(batch),
        termination=TerminationConfig(max_steps=10),
        diagnostics=DiagnosticsConfig(show_progress=False),
    )

    params = train()

    assert float(params["root"]["weight"]) == pytest.approx(3.0, abs=0.03)


@pytest.mark.skipif(sys.platform == "win32", reason="Orbax checkpointer issues on Windows")
def test_graph_loss_ema_balancing_logs_and_plots_terms(tmp_path: Path, toy_graph: FunctionGraph) -> None:
    batch = {"toy": [{"x": jnp.array([1.0])}, {"x": jnp.array([2.0])}, {"x": jnp.array([3.0])}]}
    root = tmp_path.resolve() / "graph_loss_ema"
    train = Train(
        loss=GraphLoss(
            terms=[
                {"name": "small", "term": graph_batch_squared_error, "dataset": "toy"},
                {"name": "large", "term": graph_batch_large_squared_error, "dataset": "toy"},
            ],
            balancing={
                "kind": "ema",
                "decay": 0.0,
                "target": 1.0,
                "min_scale": 1e-6,
                "max_scale": 1e6,
                "normalize": True
            },
            graph=toy_graph,
        ),
        init_params={"toy": {"weight": jnp.array(0.0)}},
        optimizer=optax.sgd(0.0),
        dataloader=RepeatBatchLoader(batch),
        termination=TerminationConfig(max_steps=1),
        diagnostics=DiagnosticsConfig(
            plot_interval=1,
            raw_terms_plot={"enabled": True, "include": ["small"]},
            scaled_terms_plot={"enabled": True, "exclude": ["small"]},
        ),
        root=root,
    )

    assert train.run() == 0

    raw_header, raw_values = _table_csv(root / "loss_terms_raw.csv")
    scaled_header, scaled_values = _table_csv(root / "loss_terms_scaled.csv")
    scale_header, scale_values = _table_csv(root / "loss_term_scales.csv")

    assert raw_header == ["Iteration", "small", "large"]
    assert scaled_header == ["Iteration", "small", "large"]
    assert scale_header == ["Iteration", "small", "large"]
    assert raw_values[:, 0].tolist() == [0.0, 1.0]
    assert raw_values[0, 1] == pytest.approx(np.mean(np.array([1.0, 2.0, 3.0]) ** 2))
    assert raw_values[0, 2] == pytest.approx(100.0 * raw_values[0, 1])
    inverse_scales = np.array([1.0 / raw_values[0, 1], 1.0 / raw_values[0, 2]])
    expected_scales = inverse_scales / np.mean(inverse_scales)
    assert scale_values[0, 1] == pytest.approx(expected_scales[0])
    assert scale_values[0, 2] == pytest.approx(expected_scales[1])
    assert scale_values[1, 1] == pytest.approx(expected_scales[0])
    assert scale_values[1, 2] == pytest.approx(expected_scales[1])
    assert scaled_values[0, 1] == pytest.approx(expected_scales[0] * raw_values[0, 1])
    assert scaled_values[0, 2] == pytest.approx(expected_scales[1] * raw_values[0, 2])
    assert (root / "loss.pdf").exists()

    with ocp.training.Checkpointer(root) as ckptr:
        assert ckptr.latest is not None
        loaded = ckptr.load_checkpointables(
            abstract_checkpointables={
                "loss_state": {
                    "step": jnp.asarray(0, dtype=jnp.int32),
                    "ema": {"small": jnp.asarray(0.0), "large": jnp.asarray(0.0)},
                    "scales": {"small": jnp.asarray(1.0), "large": jnp.asarray(1.0)},
                    "initialized": {"small": jnp.asarray(False), "large": jnp.asarray(False)},
                }
            }
        )
    assert int(np.asarray(loaded["loss_state"]["step"])) >= 2


@pytest.mark.skipif(sys.platform == "win32", reason="Orbax checkpointer issues on Windows")
def test_graph_loss_ema_log_balancing_uses_geometric_normalization(tmp_path: Path, toy_graph: FunctionGraph) -> None:
    batch = {"toy": [{"x": jnp.array([1.0])}, {"x": jnp.array([2.0])}, {"x": jnp.array([3.0])}]}
    root = tmp_path.resolve() / "graph_loss_ema_log"
    train = Train(
        loss=GraphLoss(
            terms=[
                {"name": "small", "term": graph_batch_squared_error, "dataset": "toy"},
                {"name": "large", "term": graph_batch_large_squared_error, "dataset": "toy"},
            ],
            balancing={
                "kind": "ema_log",
                "decay": 0.0,
                "target": 1.0,
                "min_scale": 1e-6,
                "max_scale": 1e6,
                "normalize": True,
            },
            graph=toy_graph,
        ),
        init_params={"toy": {"weight": jnp.array(0.0)}},
        optimizer=optax.sgd(0.0),
        dataloader=RepeatBatchLoader(batch),
        termination=TerminationConfig(max_steps=1),
        diagnostics=DiagnosticsConfig(show_progress=False),
        root=root,
    )

    assert train.run() == 0

    scale_header, scale_values = _table_csv(root / "loss_term_scales.csv")
    assert scale_header == ["Iteration", "small", "large"]
    raw_small = np.mean(np.array([1.0, 2.0, 3.0]) ** 2)
    raw_large = 100.0 * raw_small
    expected_scales = np.array([np.sqrt(raw_large / raw_small), np.sqrt(raw_small / raw_large)])
    assert scale_values[0, 1] == pytest.approx(expected_scales[0])
    assert scale_values[0, 2] == pytest.approx(expected_scales[1])
    assert np.sqrt(scale_values[0, 1] * scale_values[0, 2]) == pytest.approx(1.0)


def test_graph_loss_ema_bootstrap_balances_first_optimizer_step(toy_graph: FunctionGraph) -> None:
    batch = {"toy": [{"x": jnp.array([1.0])}, {"x": jnp.array([2.0])}, {"x": jnp.array([3.0])}]}
    train = Train(
        loss=GraphLoss(
            terms=[
                {"name": "small", "term": graph_batch_squared_error, "dataset": "toy"},
                {"name": "large", "term": graph_batch_large_squared_error, "dataset": "toy"},
            ],
            balancing={"kind": "ema", "decay": 0.0, "eps": 1e-12, "normalize": True},
            graph=toy_graph,
        ),
        init_params={"toy": {"weight": jnp.array(0.0)}},
        optimizer=optax.sgd(0.001),
        dataloader=RepeatBatchLoader(batch),
        termination=TerminationConfig(max_steps=1),
        diagnostics=DiagnosticsConfig(show_progress=False),
    )

    params = train()

    raw_small = np.mean(np.array([1.0, 2.0, 3.0]) ** 2)
    raw_large = 100.0 * raw_small
    inverse_scales = np.array([1.0 / raw_small, 1.0 / raw_large])
    scales = inverse_scales / np.mean(inverse_scales)
    expected_grad = scales[0] * -4.0 + scales[1] * -400.0
    assert float(params["toy"]["weight"]) == pytest.approx(-0.001 * expected_grad, rel=1e-5)
