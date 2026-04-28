from __future__ import annotations

import argparse
import logging
import random
import shutil
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Literal, NamedTuple

import equinox as eqx
import jax
import jax.numpy as jnp
import optax
import yaml

import romjax as rox
from romjax.graph import FunctionGraph
from romjax.nn import LinearProjection
from romjax.poisson import Poisson2D
from romjax.typing import DictModel
from romjax.utils import get_logger, load_h5, save_h5


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = Path(__file__).with_name("poisson_graph_demo.yml")
TRAINING_PATH = ["coordinate transform", "galerkin rom", "residual transform"]


class SplitConfig(DictModel):
    """Persistent sampling configuration for one dataset split."""

    input_samples: int
    outputs_per_input: int
    input_seed: int
    output_seed: int


class PathsConfig(DictModel):
    """Filesystem layout for demo artifacts."""

    root: str = "demo/artifacts/poisson_linear"
    dataset_dir: str = "dataset"
    training_dir: str = "training"


class OptimizerConfig(DictModel):
    """Optimizer settings."""

    name: Literal["adam", "adamw"] = "adam"
    learning_rate: float = 1.0e-2
    weight_decay: float = 0.0


class TrainingConfig(DictModel):
    """Training hyperparameters."""

    coordinate_latent_dim: int = 20
    residual_latent_dim: int = 20
    batch_size: int = 8
    steps: int = 200
    log_interval: int = 10
    validation_interval: int = 20
    init_seed: int = 0
    shuffle_seed: int = 1
    live_plot: bool = False


class LossConfig(DictModel):
    """Reconstruction loss weights."""

    coordinate_weight: float = 1.0
    residual_weight: float = 1.0
    graph_weight: float = 1.0


class DemoConfig(DictModel):
    """Top-level demo configuration."""

    graph_path: str = "demo/poisson_graph.yml"
    dataset_policy: Literal["reuse", "overwrite", "error"] = "reuse"
    overwrite_training: bool = True
    paths: PathsConfig = PathsConfig()
    train: SplitConfig
    validation: SplitConfig
    optimizer: OptimizerConfig = OptimizerConfig()
    training: TrainingConfig = TrainingConfig()
    loss: LossConfig = LossConfig()


class SampleDescriptor(NamedTuple):
    """Disk locations for one persisted training sample."""

    input_file: Path
    output_file: Path


class ProjectionModules(eqx.Module):
    """Learned modules routed into the two FilterModel edges."""

    coordinate: LinearProjection
    residual: LinearProjection


def resolve_repo_path(path: str | Path) -> Path:
    """Resolve a path relative to the repository root."""
    candidate = Path(path)
    return candidate if candidate.is_absolute() else REPO_ROOT / candidate


def load_config(path: Path) -> DemoConfig:
    """Load and validate the demo configuration."""
    data = rox.YamlLoader.load(path)
    return DemoConfig.model_validate(data)


def dump_yaml(path: Path, data: dict) -> None:
    """Write a small YAML document to disk."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(data, handle, sort_keys=False)


def load_yaml(path: Path) -> dict | None:
    """Read a small YAML document from disk."""
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle)
    return {} if loaded is None else loaded


def dataset_root(config: DemoConfig) -> Path:
    """Dataset root directory."""
    return resolve_repo_path(config.paths.root) / config.paths.dataset_dir


def training_root(config: DemoConfig) -> Path:
    """Training artifact directory."""
    return resolve_repo_path(config.paths.root) / config.paths.training_dir


def dataset_manifest(config: DemoConfig) -> dict:
    """Serializable manifest used for compatibility checks."""
    return {
        "graph_path": config.graph_path,
        "train": config.train.model_dump(mode="python"),
        "validation": config.validation.model_dump(mode="python"),
    }


def split_config(config: DemoConfig, split: str) -> SplitConfig:
    """Return one split configuration by name."""
    if split == "train":
        return config.train
    if split == "validation":
        return config.validation
    raise KeyError(split)


def expected_output_count(split_cfg: SplitConfig) -> int:
    """Total number of persisted output samples for one split."""
    return split_cfg.input_samples * split_cfg.outputs_per_input


def input_directories(split_root: Path) -> list[Path]:
    """List persisted top-level input sample directories in deterministic order."""
    return sorted(
        sample_dir
        for seed_dir in sorted(split_root.glob("seed_*"))
        for sample_dir in sorted(seed_dir.glob("sample_*"))
        if sample_dir.is_dir()
    )


def output_sample_files(input_dir: Path) -> list[Path]:
    """List persisted nested output sample files for one fixed input."""
    return sorted(input_dir.glob("outputs/seed_*/sample_*/sample.h5"))


def dataset_is_complete(root: Path, config: DemoConfig) -> bool:
    """Check that the on-disk dataset matches the configured sample counts."""
    for split_name in ("train", "validation"):
        split_root = root / split_name
        split_cfg = split_config(config, split_name)
        inputs = input_directories(split_root)
        if len(inputs) != split_cfg.input_samples:
            return False
        if sum(len(output_sample_files(input_dir)) for input_dir in inputs) != expected_output_count(split_cfg):
            return False
        for input_dir in inputs:
            if not (input_dir / "input.h5").exists() or not (input_dir / "solution.h5").exists():
                return False
    return True


def generate_split(
    poisson: Poisson2D,
    split_root: Path,
    split_cfg: SplitConfig,
    *,
    logger: logging.Logger,
) -> None:
    """Generate one persisted split incrementally without retaining the full dataset in memory."""
    split_root.mkdir(parents=True, exist_ok=True)
    sample_inputs = jax.jit(poisson.sample_inputs)
    sample_outputs = jax.jit(poisson.sample_outputs)
    solve = jax.jit(poisson.solve)
    evaluate = jax.jit(poisson.evaluate)

    for input_index, (input_key, input_dir) in enumerate(
        rox.gen_keys(split_cfg.input_samples, seed=split_cfg.input_seed, path=split_root)
    ):
        logger.info("Generating %s input sample %d/%d", split_root.name, input_index + 1, split_cfg.input_samples)
        inputs = sample_inputs(input_key)
        solution = solve(inputs)
        save_h5({"inputs": inputs}, input_dir / "input.h5", mode="w")
        save_h5({"solution": solution}, input_dir / "solution.h5", mode="w")

        nested_seed = split_cfg.output_seed + input_index
        output_root = input_dir / "outputs"
        for output_index, (output_key, output_dir) in enumerate(
            rox.gen_keys(split_cfg.outputs_per_input, seed=nested_seed, path=output_root)
        ):
            logger.info(
                "Generating %s output sample %d/%d for %s",
                split_root.name,
                output_index + 1,
                split_cfg.outputs_per_input,
                input_dir.name,
            )
            outputs = sample_outputs(output_key, inputs=inputs, solution=solution)
            residuals = evaluate(inputs, outputs)
            save_h5({"outputs": outputs, "residuals": residuals}, output_dir / "sample.h5", mode="w")


def ensure_dataset(graph: FunctionGraph, config: DemoConfig, logger: logging.Logger) -> Path:
    """Create, reuse, or reject the persisted dataset according to the configured policy."""
    root = dataset_root(config)
    manifest_path = root / "manifest.yml"
    expected_manifest = dataset_manifest(config)
    existing_manifest = load_yaml(manifest_path)

    if root.exists():
        compatible = existing_manifest == expected_manifest and dataset_is_complete(root, config)
        if compatible and config.dataset_policy == "reuse":
            logger.info("Reusing compatible dataset at %s", root)
            return root
        if config.dataset_policy == "error":
            raise ValueError(f"Dataset already exists at {root} and policy='error'.")
        if config.dataset_policy == "reuse":
            raise ValueError(f"Dataset at {root} is incompatible or incomplete. Use policy='overwrite'.")
        shutil.rmtree(root)

    root.mkdir(parents=True, exist_ok=True)
    dump_yaml(manifest_path, expected_manifest)
    dump_yaml(root / "config.yml", config.model_dump(mode="python"))

    poisson = graph.edges["poisson"]
    if not isinstance(poisson, Poisson2D):
        raise TypeError("Expected graph edge 'poisson' to be a Poisson2D instance.")

    for split_name in ("train", "validation"):
        generate_split(poisson, root / split_name, split_config(config, split_name), logger=logger)

    return root


def collect_split_samples(root: Path, split: str) -> list[SampleDescriptor]:
    """Collect deterministic sample descriptors for one split."""
    split_root = root / split
    descriptors: list[SampleDescriptor] = []
    for input_dir in input_directories(split_root):
        input_file = input_dir / "input.h5"
        for output_file in output_sample_files(input_dir):
            descriptors.append(SampleDescriptor(input_file=input_file, output_file=output_file))
    return descriptors


def load_sample(descriptor: SampleDescriptor) -> dict:
    """Load one sample from disk using the repository HDF5 helpers."""
    input_tree: dict = {}
    load_h5(input_tree, descriptor.input_file, jax=True)
    output_tree: dict = {}
    load_h5(output_tree, descriptor.output_file, jax=True)
    return {
        "inputs": input_tree["inputs"],
        "outputs": output_tree["outputs"],
        "residuals": output_tree["residuals"],
    }


def stack_samples(samples: Sequence[dict]) -> dict:
    """Stack a list of per-sample pytrees into one minibatch pytree."""
    if not samples:
        raise ValueError("Cannot stack an empty sample list.")
    return jax.tree.map(lambda *xs: jnp.stack(xs), *samples)


class DiskBatchLoader:
    """Infinite deterministic minibatch loader backed by persisted sample files."""

    def __init__(self, descriptors: Sequence[SampleDescriptor], batch_size: int, seed: int):
        if len(descriptors) == 0:
            raise ValueError("Training loader requires at least one sample descriptor.")
        self.descriptors = list(descriptors)
        self.batch_size = batch_size
        self.rng = random.Random(seed)
        self.order: list[int] = []
        self.cursor = 0
        self._reshuffle()

    def _reshuffle(self) -> None:
        self.order = list(range(len(self.descriptors)))
        self.rng.shuffle(self.order)
        self.cursor = 0

    def __iter__(self) -> "DiskBatchLoader":
        return self

    def __next__(self) -> dict:
        indices: list[int] = []
        while len(indices) < self.batch_size:
            if self.cursor >= len(self.order):
                self._reshuffle()
            remaining = self.batch_size - len(indices)
            take = min(remaining, len(self.order) - self.cursor)
            indices.extend(self.order[self.cursor : self.cursor + take])
            self.cursor += take
        return stack_samples([load_sample(self.descriptors[index]) for index in indices])


def iter_batches(descriptors: Sequence[SampleDescriptor], batch_size: int) -> Iterator[dict]:
    """Yield deterministic finite batches for validation."""
    for start in range(0, len(descriptors), batch_size):
        yield stack_samples([load_sample(desc) for desc in descriptors[start : start + batch_size]])


def edge_payload_patches(modules: ProjectionModules) -> dict[str, dict]:
    """Build graph-level payload patches for the two learned edges."""
    return {
        "coordinate transform": {"call_args": modules.coordinate},
        "residual transform": {"call_args": modules.residual},
    }


def pytree_mse(left: dict, right: dict) -> jax.Array:
    """Compute a size-weighted mean-squared error between two pytrees."""
    leaves = jax.tree.leaves(jax.tree.map(lambda x, y: jnp.ravel(x - y), left, right))
    total_sq = sum(jnp.sum(jnp.square(leaf)) for leaf in leaves)
    total_size = sum(leaf.size for leaf in leaves)
    return total_sq / total_size


def sample_losses(graph: FunctionGraph, modules: ProjectionModules, sample: dict) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Compute the three reconstruction losses for one sample."""
    patches = edge_payload_patches(modules)

    coord_payload = {"inputs": sample["inputs"], "outputs": sample["outputs"]}
    coord_latent, coord_aux = graph.push_path(
        coord_payload,
        path=["coordinate transform"],
        start="hf_coord",
        edge_payload_patches=patches,
        return_aux=True,
    )
    coord_recon = graph.push_path(
        coord_latent,
        path=["coordinate transform"],
        start="lf_coord",
        aux=coord_aux,
        edge_payload_patches=patches,
    )

    residual_payload = {"inputs": sample["inputs"], "residuals": sample["residuals"]}
    residual_latent, residual_aux = graph.push_path(
        residual_payload,
        path=["residual transform"],
        start="hf_res",
        edge_payload_patches=patches,
        return_aux=True,
    )
    residual_recon = graph.push_path(
        residual_latent,
        path=["residual transform"],
        start="lf_res",
        aux=residual_aux,
        edge_payload_patches=patches,
    )

    # graph_recon = graph.push_path(
    #     coord_payload,
    #     path=TRAINING_PATH,
    #     start="hf_coord",
    #     edge_payload_patches=patches,
    # )

    return (
        pytree_mse(coord_recon["outputs"], sample["outputs"]),
        pytree_mse(residual_recon["residuals"], sample["residuals"]),
        # pytree_mse(graph_recon["residuals"], sample["residuals"]),
    )


def batch_metrics(graph: FunctionGraph, modules: ProjectionModules, batch: dict) -> dict[str, jax.Array]:
    """Compute mean losses across one minibatch."""
    coord_loss, residual_loss = jax.vmap(lambda sample: sample_losses(graph, modules, sample))(batch)
    return {
        "coordinate": jnp.mean(coord_loss),
        "residual": jnp.mean(residual_loss),
        # "graph": jnp.mean(graph_loss),
    }


def total_loss(graph: FunctionGraph, config: DemoConfig, modules: ProjectionModules, batch: dict) -> jax.Array:
    """Weighted training objective."""
    metrics = batch_metrics(graph, modules, batch)
    return (
        config.loss.coordinate_weight * metrics["coordinate"]
        + config.loss.residual_weight * metrics["residual"]
        # + config.loss.graph_weight * metrics["graph"]
    )


def build_optimizer(config: DemoConfig) -> optax.GradientTransformation:
    """Construct the requested Optax optimizer."""
    if config.optimizer.name == "adam":
        return optax.adam(config.optimizer.learning_rate)
    if config.optimizer.name == "adamw":
        return optax.adamw(config.optimizer.learning_rate, weight_decay=config.optimizer.weight_decay)
    raise ValueError(f"Unsupported optimizer {config.optimizer.name!r}")


def infer_state_sizes(descriptors: Sequence[SampleDescriptor]) -> tuple[int, int]:
    """Infer flattened full-space dimensions for the two learned projections."""
    sample = load_sample(descriptors[0])
    output_size = sum(leaf.size for leaf in jax.tree.leaves(sample["outputs"]))
    residual_size = sum(leaf.size for leaf in jax.tree.leaves(sample["residuals"]))
    return output_size, residual_size


def initialize_modules(config: DemoConfig, train_descriptors: Sequence[SampleDescriptor]) -> ProjectionModules:
    """Initialize the coordinate and residual linear projections."""
    output_size, residual_size = infer_state_sizes(train_descriptors)
    key = jax.random.key(config.training.init_seed)
    coord_key, residual_key = jax.random.split(key)
    return ProjectionModules(
        coordinate=LinearProjection(
            n_latent=config.training.coordinate_latent_dim,
            n_full=output_size,
            key=coord_key,
        ),
        residual=LinearProjection(
            n_latent=config.training.residual_latent_dim,
            n_full=residual_size,
            key=residual_key,
        ),
    )


def evaluate_descriptors(
    graph: FunctionGraph,
    config: DemoConfig,
    modules: ProjectionModules,
    descriptors: Sequence[SampleDescriptor],
) -> dict[str, float]:
    """Evaluate mean losses over a finite set of descriptors."""
    accum = {"coordinate": 0.0, "residual": 0.0, "graph": 0.0, "total": 0.0}
    count = 0
    for batch in iter_batches(descriptors, config.training.batch_size):
        metrics = batch_metrics(graph, modules, batch)
        total = total_loss(graph, config, modules, batch)
        batch_size = next(iter(jax.tree.leaves(batch))).shape[0]
        accum["coordinate"] += metrics["coordinate"] * batch_size
        accum["residual"] += metrics["residual"] * batch_size
        # accum["graph"] += float(metrics["graph"]) * batch_size
        accum["total"] += total * batch_size
        count += batch_size
    return {key: value / count for key, value in accum.items()}


def save_modules(path: Path, modules: ProjectionModules) -> None:
    """Persist the learned projection matrices with the repo HDF5 helper."""
    save_h5(
        {
            "coordinate": {"matrix": modules.coordinate.matrix},
            "residual": {"matrix": modules.residual.matrix},
        },
        path,
        mode="w",
    )


def train_demo(graph: FunctionGraph, config: DemoConfig, root: Path, logger: logging.Logger) -> dict[str, float]:
    """Train both projections from the persisted dataset."""
    train_descriptors = collect_split_samples(root, "train")
    validation_descriptors = collect_split_samples(root, "validation")
    if not train_descriptors:
        raise ValueError("Training split is empty.")
    if not validation_descriptors:
        raise ValueError("Validation split is empty.")

    output_dir = training_root(config)
    if output_dir.exists() and config.overwrite_training:
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    dump_yaml(output_dir / "config.yml", config.model_dump(mode="python"))

    modules0 = initialize_modules(config, train_descriptors)
    optimizer = build_optimizer(config)
    train_loader = DiskBatchLoader(train_descriptors, config.training.batch_size, seed=config.training.shuffle_seed)

    loss_fn = lambda params, batch: total_loss(graph, config, params, batch)
    test_fn = lambda params: evaluate_descriptors(graph, config, params, validation_descriptors)["total"]

    modules = rox.train(
        loss_fn=jax.jit(loss_fn),
        params0=modules0,
        optimizer=optimizer,
        dataloader=iter(train_loader),
        max_steps=config.training.steps,
        log_interval=config.training.log_interval,
        hist_interval=config.training.validation_interval,
        plot_interval=0,
        live_plot=config.training.live_plot,
        test_fn=jax.jit(test_fn),
        save=output_dir,
        logger=logger,
    )

    train_metrics = evaluate_descriptors(graph, config, modules, train_descriptors)
    validation_metrics = evaluate_descriptors(graph, config, modules, validation_descriptors)
    metrics = {
        **{f"train_{key}": float(value) for key, value in train_metrics.items()},
        **{f"validation_{key}": float(value) for key, value in validation_metrics.items()},
    }
    dump_yaml(output_dir / "metrics.yml", metrics)
    save_modules(output_dir / "modules.h5", modules)
    return metrics


def parse_args() -> argparse.Namespace:
    """Parse the thin demo CLI."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH, help="Path to the demo YAML config.")
    parser.add_argument(
        "--stage",
        choices=("all", "data", "train"),
        default="all",
        help="Run data generation, training, or both.",
    )
    return parser.parse_args()


def main() -> None:
    """Run the requested demo stages."""
    args = parse_args()
    config = load_config(resolve_repo_path(args.config))
    logger = get_logger("poisson-demo")
    graph = rox.YamlLoader.load(resolve_repo_path(config.graph_path))
    if not isinstance(graph, FunctionGraph):
        raise TypeError("Expected the graph YAML to load a FunctionGraph.")

    root = dataset_root(config)
    if args.stage in {"all", "data"}:
        root = ensure_dataset(graph, config, logger)
    elif not root.exists():
        raise FileNotFoundError(f"Dataset not found at {root}. Run with --stage data or --stage all first.")

    if args.stage in {"all", "train"}:
        metrics = train_demo(graph, config, root, logger)
        print(
            "Poisson demo complete: "
            f"train_total={metrics['train_total']:.6e}, "
            f"validation_total={metrics['validation_total']:.6e}, "
            f"validation_coordinate={metrics['validation_coordinate']:.6e}, "
            f"validation_residual={metrics['validation_residual']:.6e}, "
            f"validation_graph={metrics['validation_graph']:.6e}"
        )


if __name__ == "__main__":
    main()
