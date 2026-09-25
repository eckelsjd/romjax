from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Annotated, Any, Mapping, Sequence

import equinox as eqx
import h5py
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import PyTree
from orbax.checkpoint import v1 as ocp
from pydantic import BaseModel, BeforeValidator, ConfigDict, Field, NonNegativeFloat, PositiveInt, model_validator
from pydantic_core import core_schema

from romjax.nn import LinearProjection, SplitLinearProjection
from romjax.tree import ShapeDtypePyTree, is_shape_dtype
from romjax.typing import from_yaml

__all__ = ["Compression", "SVD", "SplitLinearCompression"]

_ARTIFACT_VERSION = 2
_CLASS_ATTR = "compression_class"
_KIND_ATTR = "romjax_kind"
_VALUE_ATTR = "value"
_TEMPLATE_KIND = "shape_dtype"


def _split_linear_mse(params: SplitLinearProjection, batch: jax.Array) -> jax.Array:
    """Return the mean squared reconstruction error for a split projection.

    :param params: split projection being optimized.
    :param batch: batch of flattened full-state vectors.
    :return: mean squared reconstruction error.
    """
    reconstructed = jax.vmap(params.reconstruct)(jax.vmap(params.reduce)(batch))
    return jnp.mean(jnp.square(reconstructed - batch))


def _coerce_split_linear_train(value: Any) -> Any:
    """Validate mapping-based split-compression training configuration.

    The reconstruction loss and loader are supplied here so a compact YAML
    configuration only needs to describe the optimizer, parameters, and stopping
    behavior.  ``fit`` replaces the loader with the vectors it gathers.
    """
    value = from_yaml(value)
    if not isinstance(value, Mapping):
        return value

    from romjax.train import BatchLoader, Train

    config = dict(value)
    config.setdefault("loss", _split_linear_mse)
    config.setdefault("dataloader", BatchLoader())
    return Train.model_validate(config)


def _decode_h5_attr(value: Any) -> Any:
    """Convert HDF5 scalar attributes to their Python representation."""
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if isinstance(value, np.generic):
        return value.item()
    return value


def _write_tree_node(parent: h5py.Group, name: str, value: Any) -> None:
    """Write one typed PyTree node without relying on pickle serialization."""
    if is_shape_dtype(value):
        node = parent.create_dataset(name, data=np.zeros(value.shape, dtype=value.dtype))
        node.attrs[_KIND_ATTR] = _TEMPLATE_KIND
        return
    if isinstance(value, (jax.Array, np.ndarray)):
        node = parent.create_dataset(name, data=np.asarray(value))
        node.attrs[_KIND_ATTR] = "array"
        return

    node = parent.create_group(name, track_order=True)
    if value is None:
        node.attrs[_KIND_ATTR] = "none"
    elif isinstance(value, Mapping):
        node.attrs[_KIND_ATTR] = "mapping"
        for index, (key, item) in enumerate(value.items()):
            entry = node.create_group(str(index), track_order=True)
            _write_tree_node(entry, "key", key)
            _write_tree_node(entry, "value", item)
    elif isinstance(value, list):
        node.attrs[_KIND_ATTR] = "list"
        for index, item in enumerate(value):
            _write_tree_node(node, str(index), item)
    elif isinstance(value, tuple):
        node.attrs[_KIND_ATTR] = "tuple"
        for index, item in enumerate(value):
            _write_tree_node(node, str(index), item)
    elif isinstance(value, str):
        node.attrs[_KIND_ATTR] = "str"
        node.attrs[_VALUE_ATTR] = value
    elif isinstance(value, bool):
        node.attrs[_KIND_ATTR] = "bool"
        node.attrs[_VALUE_ATTR] = value
    elif isinstance(value, int):
        node.attrs[_KIND_ATTR] = "int"
        node.attrs[_VALUE_ATTR] = value
    elif isinstance(value, float):
        node.attrs[_KIND_ATTR] = "float"
        node.attrs[_VALUE_ATTR] = value
    elif isinstance(value, np.generic):
        node.attrs[_KIND_ATTR] = "numpy_scalar"
        node.attrs["dtype"] = value.dtype.str
        node.attrs[_VALUE_ATTR] = value
    else:
        raise TypeError(f"Unsupported compression artifact value: {type(value).__name__}.")


def _read_tree_node(node: h5py.Group | h5py.Dataset) -> Any:
    """Read one typed PyTree node written by :func:`_write_tree_node`."""
    kind = _decode_h5_attr(node.attrs.get(_KIND_ATTR))
    if isinstance(node, h5py.Dataset):
        if kind == _TEMPLATE_KIND:
            return jax.ShapeDtypeStruct(node.shape, node.dtype)
        if kind == "array":
            return node[()]
        raise ValueError(f"Unsupported compression dataset kind: {kind!r}.")

    if kind == "none":
        return None
    if kind == "mapping":
        return {
            _read_tree_node(node[str(index)]["key"]): _read_tree_node(node[str(index)]["value"])
            for index in range(len(node))
        }
    if kind == "list":
        return [_read_tree_node(node[str(index)]) for index in range(len(node))]
    if kind == "tuple":
        return tuple(_read_tree_node(node[str(index)]) for index in range(len(node)))
    if kind == "str":
        return str(_decode_h5_attr(node.attrs[_VALUE_ATTR]))
    if kind == "bool":
        return bool(_decode_h5_attr(node.attrs[_VALUE_ATTR]))
    if kind == "int":
        return int(_decode_h5_attr(node.attrs[_VALUE_ATTR]))
    if kind == "float":
        return float(_decode_h5_attr(node.attrs[_VALUE_ATTR]))
    if kind == "numpy_scalar":
        return np.asarray(_decode_h5_attr(node.attrs[_VALUE_ATTR]), dtype=node.attrs["dtype"])[()]
    raise ValueError(f"Unsupported compression group kind: {kind!r}.")


class Compression(BaseModel, ABC):
    """Persisted latent-space compressor."""

    model_config = ConfigDict(arbitrary_types_allowed=True, validate_default=True)

    @classmethod
    def _from_registry(cls, value):
        if isinstance(value, Compression):
            return value

        if isinstance(value, str):
            name = value
            opts: dict[str, Any] = {}
        elif isinstance(value, Mapping):
            opts = dict(value)
            name = opts.pop("kind", "svd")
        else:
            raise TypeError(
                "Compression config must be a string, mapping, or Compression instance; "
                f"got {type(value).__name__}."
            )

        if name is None:
            raise ValueError("Must specify compression 'kind'")
        if name == "svd":
            return SVD(**opts)
        if name == "split_linear":
            return SplitLinearCompression(**opts)
        raise ValueError(f"Compression '{name}' not recognized.")

    @classmethod
    def __get_pydantic_core_schema__(cls, source_type: Any, handler: Any) -> Any:
        """Accept compression configs as either an instance or a registry-backed mapping."""
        if cls.__name__ != "Compression":
            return handler(source_type)

        def _validate(value: Any) -> "Compression":
            if isinstance(value, Compression):
                return value
            return cls._from_registry(value)

        return core_schema.no_info_plain_validator_function(
            _validate,
            json_schema_input_schema=core_schema.any_schema(),
        )

    @abstractmethod
    def compress(self, sample: PyTree) -> PyTree:
        """Project a sample into latent coordinates."""
        raise NotImplementedError

    @abstractmethod
    def reconstruct(self, latent: PyTree) -> PyTree:
        """Map latent coordinates back to the original feature space."""
        raise NotImplementedError

    @abstractmethod
    def latent_size(self) -> int:
        """Return the latent dimension."""
        raise NotImplementedError

    @abstractmethod
    def latent_bounds(self) -> tuple[jax.Array, jax.Array] | None:
        """Return latent-space min/max bounds if available."""
        raise NotImplementedError
    
    @abstractmethod
    def latent_normal(self) -> tuple[jax.Array, jax.Array] | None:
        """Return latent-space mean/std if available."""
        raise NotImplementedError

    def sample(self, key: jax.Array) -> PyTree:
        """Sample latent coordinates using an artifact-defined distribution.

        :param key: JAX random key.
        :return: artifact-defined latent sample.
        :raises NotImplementedError: if the concrete artifact defines no sampler.
        """
        del key
        raise NotImplementedError(f"{type(self).__name__} does not define artifact sampling.")

    @staticmethod
    def _empirical_covariance(samples: jax.Array) -> jax.Array:
        """Return the unbiased empirical covariance of row-wise samples.

        A single sample has zero covariance, which is preferable to propagating
        undefined values into a persisted artifact.

        :param samples: array with shape ``(n_samples, n_features)``.
        :return: covariance array with shape ``(n_features, n_features)``.
        """
        centered = samples - jnp.mean(samples, axis=0, keepdims=True)
        denominator = max(samples.shape[0] - 1, 1)
        return centered.T @ centered / denominator

    @abstractmethod
    def fit(self, samples: Sequence[PyTree]) -> "Compression":
        """Fit the compressor to a sequence of single-sample pytrees."""
        raise NotImplementedError

    @staticmethod
    def _class_spec(compression_cls: type["Compression"]) -> str:
        """Return a serialized class spec for a concrete compression type."""
        return f"{compression_cls.__module__}:{compression_cls.__qualname__}"

    @staticmethod
    def _resolve_class(spec: str) -> type["Compression"]:
        """Resolve a serialized compression class spec."""
        module_name, _, qualname = spec.partition(":")
        if not module_name or not qualname:
            raise ValueError(f"Invalid compression class spec: {spec!r}")
        module = __import__(module_name, fromlist=["*"])
        resolved: Any = module
        for attr in qualname.split("."):
            resolved = getattr(resolved, attr)
        if not isinstance(resolved, type) or not issubclass(resolved, Compression):
            raise ValueError(f"Compression class spec {spec!r} does not resolve to a Compression subclass.")
        return resolved

    @classmethod
    def load(cls, path: str | Path) -> "Compression":
        """Load a persisted HDF5 compression artifact.

        :param path: artifact file or directory containing ``compression.h5``.
        :return: restored concrete compression instance.
        """
        artifact_path = Path(path)
        if artifact_path.is_dir():
            artifact_path = artifact_path / "compression.h5"
        if artifact_path.suffix != ".h5":
            raise ValueError(f"Unsupported compression artifact path: {artifact_path}")

        with h5py.File(artifact_path, "r") as artifact:
            if artifact.attrs.get("romjax_type") != "compression":
                raise ValueError(f"Unsupported compression artifact: {artifact_path}")
            version = int(artifact.attrs.get("version", -1))
            if version not in {1, _ARTIFACT_VERSION}:
                raise ValueError(f"Unsupported compression artifact version: {artifact_path}")
            class_spec = _decode_h5_attr(artifact.attrs.get(_CLASS_ATTR))
            if version == 1:
                if "payload" not in artifact:
                    raise ValueError(f"Compression artifact {artifact_path} is missing its payload.")
                payload = _read_tree_node(artifact["payload"])
            else:
                payload = {
                    name: _read_tree_node(node)
                    for name, node in artifact.items()
                }

        if not isinstance(payload, dict):
            raise ValueError(f"Compression artifact {artifact_path} payload must be a mapping.")

        target_cls: type[Compression]
        if cls is Compression:
            if class_spec is None:
                raise ValueError(f"Compression artifact {artifact_path} is missing '__compression_class__'.")
            target_cls = cls._resolve_class(str(class_spec))
        else:
            target_cls = cls
            if class_spec is not None:
                resolved_cls = cls._resolve_class(str(class_spec))
                if resolved_cls is not cls:
                    raise ValueError(
                        f"Compression artifact {artifact_path} contains {resolved_cls.__name__}, not {cls.__name__}."
                    )

        return target_cls.model_validate(payload)

    def dump(self, path: str | Path) -> Path:
        """Persist the compressor as an HDF5 artifact.

        :param path: target ``.h5`` artifact or directory.
        :return: saved artifact path.
        """
        artifact_path = Path(path)
        if artifact_path.is_dir() or artifact_path.suffix == "":
            artifact_path.mkdir(parents=True, exist_ok=True)
            artifact_path = artifact_path / "compression.h5"
        else:
            artifact_path.parent.mkdir(parents=True, exist_ok=True)
        if artifact_path.suffix != ".h5":
            raise ValueError(f"Unsupported compression artifact path: {artifact_path}")

        payload = self.model_dump()
        with h5py.File(artifact_path, "w", track_order=True) as artifact:
            artifact.attrs["romjax_type"] = "compression"
            artifact.attrs["version"] = _ARTIFACT_VERSION
            artifact.attrs[_CLASS_ATTR] = self._class_spec(type(self))
            for name, value in payload.items():
                _write_tree_node(artifact, name, value)
        return artifact_path


class SVD(Compression):
    """Persisted SVD/POD compression model and configuration."""

    energy_tol: float | None = None
    center: bool = True
    rank: PositiveInt | None = None
    mean: np.ndarray | None = None
    basis: np.ndarray | None = None
    singular_values: np.ndarray | None = None
    minval: np.ndarray | None = None
    maxval: np.ndarray | None = None
    latent_mean: np.ndarray | None = None
    latent_std: np.ndarray | None = None
    latent_covariance: np.ndarray | None = None
    template: ShapeDtypePyTree | None = None  # for samples
    orbax_template: PyTree | None = None

    @model_validator(mode="after")
    def _validate_rank_policy(self):
        if self.energy_tol is None and self.rank is None:
            raise ValueError("Must specify rank or energy_tol for SVD compression.")
        if self.energy_tol is not None and not (0.0 < self.energy_tol <= 1.0):
            raise ValueError("energy_tol must be in the interval (0, 1].")
        return self

    def _flatten_sample(self, sample: PyTree) -> jax.Array:
        """Flatten one sample pytree into a single feature vector."""
        if self.template is None:
            leaves = [jnp.ravel(jnp.asarray(leaf)) for leaf in jax.tree.leaves(sample) if eqx.is_array_like(leaf)]
        else:
            sample_leaves, sample_treedef = jax.tree.flatten(sample)
            template_leaves, template_treedef = jax.tree.flatten(self.template)
            if sample_treedef != template_treedef:
                raise ValueError("Sample pytree structure does not match the SVD template.")

            leaves = []
            for leaf, template_leaf in zip(sample_leaves, template_leaves):
                if not is_shape_dtype(template_leaf):
                    continue
                if is_shape_dtype(leaf):
                    # Shape/dtype-only templates carry no sample values. Zeros let
                    # the template remain usable for shape-only reconstruction.
                    array = jnp.zeros(leaf.shape, dtype=leaf.dtype)
                else:
                    array = jnp.asarray(leaf)
                if array.shape != template_leaf.shape or array.dtype != template_leaf.dtype:
                    raise ValueError(
                        "Sample array does not match the SVD template: "
                        f"expected shape/dtype {template_leaf.shape}/{template_leaf.dtype}, "
                        f"got {array.shape}/{array.dtype}."
                    )
                leaves.append(jnp.ravel(array))
        if not leaves:
            return jnp.asarray([], dtype=jnp.float32)
        return jnp.concatenate(leaves, axis=0)

    def _resolve_rank(self, singular_values: jax.Array) -> int:
        if self.rank is not None:
            return int(self.rank)
        if singular_values.size == 0:
            raise ValueError("Cannot infer latent rank from an empty singular spectrum.")
        if self.energy_tol is None:
            raise ValueError("Must specify rank or energy_tol for SVD compression.")

        energy = jnp.square(singular_values)
        total_energy = jnp.sum(energy)
        if float(total_energy) <= 0.0:
            return 1
        cumulative = jnp.cumsum(energy) / total_energy
        return int(jnp.searchsorted(cumulative, self.energy_tol, side="left") + 1)

    def fit(self, samples: Sequence[PyTree]) -> "SVD":
        """Fit an SVD/POD latent-space compression model."""
        sample_vectors = [self._flatten_sample(sample) for sample in samples]
        if len(sample_vectors) == 0:
            raise ValueError("No samples were loaded for latent-space fitting.")

        matrix = jnp.stack(sample_vectors, axis=0)
        mean = jnp.mean(matrix, axis=0) if self.center else jnp.zeros(matrix.shape[1], dtype=matrix.dtype)
        centered = matrix - mean
        _, singular_values, vt = jnp.linalg.svd(centered, full_matrices=False)
        rank = min(self._resolve_rank(singular_values), vt.shape[0])
        basis = vt[:rank]
        latent = centered @ basis.T
        minval = jnp.min(latent, axis=0)
        maxval = jnp.max(latent, axis=0)
        latent_mean = jnp.mean(latent, axis=0)
        latent_std = jnp.std(latent, axis=0)
        latent_covariance = self._empirical_covariance(latent)

        return type(self)(
            energy_tol=self.energy_tol,
            center=self.center,
            rank=rank,
            mean=np.asarray(mean),
            basis=np.asarray(basis),
            singular_values=np.asarray(singular_values),
            minval=np.asarray(minval),
            maxval=np.asarray(maxval),
            latent_mean=np.asarray(latent_mean),
            latent_std=np.asarray(latent_std),
            latent_covariance=np.asarray(latent_covariance),
            template=self.template if self.template is not None else samples[0],
            orbax_template=self.orbax_template,
        )

    def save_orbax(self, path: str | Path):
        """Save POD basis to an orbax checkpoint using LinearProjection."""
        if self.basis is None:
            raise ValueError("Cannot save orbax with no basis. Must call fit() first.")

        params = LinearProjection(matrix=self.basis, bias=self.mean)

        if self.orbax_template is not None:
            # Replaces all "Nones" in the template with the projection basis object
            chkptable = jax.tree.map(
                lambda value: params if value is None else value,
                self.orbax_template,
                is_leaf=lambda value: value is None,
            )
        else:
            chkptable = params

        with ocp.training.Checkpointer(Path(path).absolute()) as ckptr:
            ckptr.save_checkpointables(
                step=0,
                checkpointables={"params": eqx.filter(chkptable, eqx.is_array)},
                force=True,
                overwrite=True,
            )

    def compress(self, sample: PyTree) -> jax.Array:
        """Project one sample into latent coordinates."""
        if self.mean is None or self.basis is None:
            raise ValueError("SVD must be fitted before compressing samples.")
        sample_vector = self._flatten_sample(sample)
        centered = sample_vector - jnp.asarray(self.mean)
        return centered @ jnp.asarray(self.basis).T

    def reconstruct(self, latent: PyTree) -> PyTree:
        """Map latent coordinates back to the original pytree shape when available."""
        if self.mean is None or self.basis is None:
            raise ValueError("SVD must be fitted before reconstructing samples.")
        vector = jnp.asarray(latent) @ jnp.asarray(self.basis) + jnp.asarray(self.mean)
        if self.template is None:
            return vector

        template_leaves, treedef = jax.tree.flatten(self.template)
        offset = 0
        reconstructed_leaves = []
        for leaf in template_leaves:
            if not is_shape_dtype(leaf):
                reconstructed_leaves.append(leaf)
                continue
            size = int(np.prod(leaf.shape, dtype=int))
            chunk = vector[offset : offset + size]
            reconstructed_leaves.append(jnp.asarray(chunk, dtype=leaf.dtype).reshape(leaf.shape))
            offset += size

        if offset != vector.shape[-1]:
            raise ValueError("Latent reconstruction produced a vector with an incompatible template size.")
        return jax.tree.unflatten(treedef, reconstructed_leaves)

    def latent_size(self) -> int:
        if self.rank is not None:
            return int(self.rank)
        if self.basis is not None:
            return int(np.asarray(self.basis).shape[0])
        raise ValueError("SVD does not define a latent size.")

    def latent_bounds(self) -> tuple[jax.Array, jax.Array] | None:
        if self.minval is None or self.maxval is None:
            return None
        return jnp.asarray(self.minval), jnp.asarray(self.maxval)
    
    def latent_normal(self) -> tuple[jax.Array, jax.Array] | None:
        if self.latent_mean is None or self.latent_std is None:
            return None
        return jnp.asarray(self.latent_mean), jnp.asarray(self.latent_std)


class SplitLinearCompression(Compression):
    """Persisted trainable compression backed by :class:`SplitLinearProjection`.

    The saved template determines how the concatenated decoder result is unpacked.
    Its array leaves must occupy the decoder's ``b_output`` segment followed by its
    ``u_output`` segment.
    """

    train: Annotated[Any | None, BeforeValidator(_coerce_split_linear_train)] = Field(default=None, exclude=True)
    orthogonal_reg: NonNegativeFloat | None = Field(default=None, exclude=True)
    test: bool = Field(default=False, exclude=True)
    encoder_b: np.ndarray | None = None
    encoder_u: np.ndarray | None = None
    decoder_b: np.ndarray | None = None
    decoder_u: np.ndarray | None = None
    encoder_b_bias: np.ndarray | None = None
    encoder_u_bias: np.ndarray | None = None
    decoder_b_bias: np.ndarray | None = None
    decoder_u_bias: np.ndarray | None = None
    input_size: PositiveInt | None = None
    b_latent: PositiveInt | None = None
    u_latent: PositiveInt | None = None
    b_output: PositiveInt | None = None
    u_output: PositiveInt | None = None
    minval: np.ndarray | None = None
    maxval: np.ndarray | None = None
    latent_mean: np.ndarray | None = None
    latent_std: np.ndarray | None = None
    latent_covariance: np.ndarray | None = None
    template: ShapeDtypePyTree | None = None
    orbax_template: PyTree | None = None
    show_progress: bool = Field(default=True, exclude=True)

    @model_validator(mode="before")
    @classmethod
    def _extract_train_options(cls, value: Any) -> Any:
        """Lift split-compression-only options out of a mapping train config.

        ``orthogonal_reg`` and ``test`` are artifacts options rather than
        :class:`Train` fields, but placing them under ``train`` keeps YAML
        configurations cohesive. Top-level values take precedence when both are
        supplied.
        """
        if not isinstance(value, Mapping):
            return value
        config = dict(value)
        train_config = config.get("train")
        if isinstance(train_config, Mapping):
            train_config = dict(train_config)
            for name in ("orthogonal_reg", "test"):
                if name in train_config:
                    config.setdefault(name, train_config.pop(name))
            config["train"] = train_config
        return config

    def _projection(self) -> SplitLinearProjection:
        """Rebuild the projection module from persisted matrices."""
        matrices = (self.encoder_b, self.encoder_u, self.decoder_b, self.decoder_u)
        if any(matrix is None for matrix in matrices):
            raise ValueError("SplitLinearCompression must be fitted before use.")
        return SplitLinearProjection(
            encoder_b=self.encoder_b,
            encoder_u=self.encoder_u,
            decoder_b=self.decoder_b,
            decoder_u=self.decoder_u,
            encoder_b_bias=self.encoder_b_bias,
            encoder_u_bias=self.encoder_u_bias,
            decoder_b_bias=self.decoder_b_bias,
            decoder_u_bias=self.decoder_u_bias,
        )

    def _flatten_sample(self, sample: PyTree) -> jax.Array:
        """Flatten one sample using the fitted template's canonical leaf order."""
        if self.template is None:
            leaves = [jnp.ravel(jnp.asarray(leaf)) for leaf in jax.tree.leaves(sample) if eqx.is_array_like(leaf)]
            return jnp.concatenate(leaves) if leaves else jnp.asarray([], dtype=jnp.float32)

        template = self.template
        sample_leaves, sample_treedef = jax.tree.flatten(sample)
        template_leaves, template_treedef = jax.tree.flatten(template)
        if sample_treedef != template_treedef:
            raise ValueError("Sample pytree structure does not match the SplitLinearCompression template.")
        leaves: list[jax.Array] = []
        for leaf, shape in zip(sample_leaves, template_leaves):
            if not is_shape_dtype(shape):
                continue
            value = jnp.zeros(shape.shape, shape.dtype) if is_shape_dtype(leaf) else jnp.asarray(leaf)
            if value.shape != shape.shape or value.dtype != shape.dtype:
                raise ValueError("Sample array does not match the SplitLinearCompression template.")
            leaves.append(jnp.ravel(value))
        return jnp.concatenate(leaves) if leaves else jnp.asarray([], dtype=jnp.float32)

    @staticmethod
    def _unflatten(vector: jax.Array, template: ShapeDtypePyTree | None) -> PyTree:
        """Restore one vector into array leaves of a shape/dtype template."""
        if template is None:
            return vector
        leaves, treedef = jax.tree.flatten(template)
        offset = 0
        restored: list[Any] = []
        for leaf in leaves:
            if not is_shape_dtype(leaf):
                restored.append(leaf)
                continue
            size = int(np.prod(leaf.shape, dtype=int))
            restored.append(jnp.asarray(vector[offset : offset + size], dtype=leaf.dtype).reshape(leaf.shape))
            offset += size
        if offset != vector.shape[-1]:
            raise ValueError("SplitLinear reconstruction produced a vector with an incompatible template size.")
        return jax.tree.unflatten(treedef, restored)

    def fit(self, samples: Sequence[PyTree]) -> "SplitLinearCompression":
        """Train a split linear projection against mean reconstruction error."""
        if not samples:
            raise ValueError("No samples were loaded for latent-space fitting.")
        from romjax.train import BatchLoader, Train

        if not isinstance(self.train, Train):
            raise TypeError("SplitLinearCompression.train must be a configured Train instance.")
        template = self.template if self.template is not None else jax.tree.map(
            lambda leaf: jax.ShapeDtypeStruct(leaf.shape, leaf.dtype) if eqx.is_array_like(leaf) else leaf,
            samples[0],
        )
        vectors = jnp.stack([self._flatten_sample(sample) for sample in samples])
        initial = self.train.init_params
        if not isinstance(initial, SplitLinearProjection):
            raise TypeError("SplitLinearCompression.train.init_params must be a SplitLinearProjection.")
        if initial.input_size != vectors.shape[-1]:
            raise ValueError("SplitLinearProjection input_size must match the gathered sample size.")
        if initial.b_output + initial.u_output != vectors.shape[-1]:
            raise ValueError("SplitLinearProjection decoder output sizes must match the gathered sample size.")

        routine = self.train.model_copy(deep=True)
        if self.orthogonal_reg is None:
            loss = _split_linear_mse
        else:
            from romjax.loss import orthogonal_regularization

            weight = jnp.asarray(self.orthogonal_reg)

            def loss(params: SplitLinearProjection, batch: jax.Array) -> jax.Array:
                penalty = sum(
                    orthogonal_regularization(params, batch, None, [matrix_name])
                    for matrix_name in ("encoder_b", "encoder_u")
                    # for matrix_name in ("encoder_b", "encoder_u", "decoder_b", "decoder_u")
                )
                return _split_linear_mse(params, batch) + weight * penalty

        object.__setattr__(routine, "loss", loss)
        object.__setattr__(routine, "dataloader", BatchLoader(data=vectors, max_epochs=None))
        object.__setattr__(routine.diagnostics, "show_progress", self.show_progress)
        if self.test:
            def reconstruction_test(params: SplitLinearProjection) -> jax.Array:
                reconstructed = jax.vmap(params.reconstruct)(jax.vmap(params.reduce)(vectors))
                denominator = jnp.linalg.norm(vectors)
                return jnp.where(
                    denominator > 0,
                    jnp.linalg.norm(reconstructed - vectors) / denominator,
                    jnp.asarray(0.0, dtype=vectors.dtype),
                )

            object.__setattr__(routine, "test", reconstruction_test)
            if routine.diagnostics.test_interval is None:
                object.__setattr__(routine.diagnostics, "test_interval", 1)
        trained = routine()
        if not isinstance(trained, SplitLinearProjection):
            raise TypeError("SplitLinearCompression training did not return a SplitLinearProjection.")
        latent = jax.vmap(trained.reduce)(vectors)
        return type(self)(
            encoder_b=np.asarray(trained.encoder_b), encoder_u=np.asarray(trained.encoder_u),
            decoder_b=np.asarray(trained.decoder_b), decoder_u=np.asarray(trained.decoder_u),
            encoder_b_bias=None if trained.encoder_b_bias is None else np.asarray(trained.encoder_b_bias),
            encoder_u_bias=None if trained.encoder_u_bias is None else np.asarray(trained.encoder_u_bias),
            decoder_b_bias=None if trained.decoder_b_bias is None else np.asarray(trained.decoder_b_bias),
            decoder_u_bias=None if trained.decoder_u_bias is None else np.asarray(trained.decoder_u_bias),
            input_size=trained.input_size, b_latent=trained.b_latent, u_latent=trained.u_latent,
            b_output=trained.b_output, u_output=trained.u_output,
            minval=np.asarray(jnp.min(latent, axis=0)), maxval=np.asarray(jnp.max(latent, axis=0)),
            latent_mean=np.asarray(jnp.mean(latent, axis=0)), latent_std=np.asarray(jnp.std(latent, axis=0)),
            latent_covariance=np.asarray(self._empirical_covariance(latent)),
            template=template, orbax_template=self.orbax_template,
        )

    def compress(self, sample: PyTree) -> jax.Array:
        """Encode one sample pytree into concatenated split latent coordinates."""
        return self._projection().reduce(self._flatten_sample(sample))

    def reconstruct(self, latent: PyTree) -> PyTree:
        """Decode latent coordinates and restore the saved sample pytree."""
        return self._unflatten(self._projection().reconstruct(jnp.asarray(latent)), self.template)

    def latent_size(self) -> int:
        """Return the total split latent dimension."""
        if self.b_latent is None or self.u_latent is None:
            raise ValueError("SplitLinearCompression does not define a latent size.")
        return int(self.b_latent + self.u_latent)

    def latent_bounds(self) -> tuple[jax.Array, jax.Array] | None:
        """Return fitted latent bounds."""
        if self.minval is None or self.maxval is None:
            return None
        return jnp.asarray(self.minval), jnp.asarray(self.maxval)

    def latent_normal(self) -> tuple[jax.Array, jax.Array] | None:
        """Return fitted latent normal statistics."""
        if self.latent_mean is None or self.latent_std is None:
            return None
        return jnp.asarray(self.latent_mean), jnp.asarray(self.latent_std)

    def save_orbax(self, path: str | Path) -> None:
        """Save the trained split projection through Orbax."""
        projection = self._projection()
        template = projection if self.orbax_template is None else jax.tree.map(
            lambda value: projection if value is None else value,
            self.orbax_template,
            is_leaf=lambda value: value is None,
        )
        with ocp.training.Checkpointer(Path(path).absolute()) as ckptr:
            ckptr.save_checkpointables(
                step=0,
                checkpointables={"params": eqx.filter(template, eqx.is_array)},
                force=True,
                overwrite=True,
            )
