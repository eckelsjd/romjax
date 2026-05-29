from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Mapping, Sequence

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import PyTree
from pydantic import BaseModel, ConfigDict, PositiveInt, model_validator
from pydantic_core import core_schema

__all__ = ["Compression"]


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
    def load(cls, path: Path) -> "Compression":
        """Load a persisted compression artifact."""
        artifact_path = Path(path)
        if artifact_path.is_dir():
            artifact_path = artifact_path / "compression.npz"
        if artifact_path.suffix != ".npz":
            raise ValueError(f"Unsupported compression artifact path: {artifact_path}")

        with np.load(artifact_path, allow_pickle=True) as data:
            payload = {key: data[key] for key in data.files}

        class_spec = payload.pop("__compression_class__", None)
        if isinstance(class_spec, np.ndarray) and class_spec.shape == ():
            class_spec = class_spec.item()
        for key, value in list(payload.items()):
            if isinstance(value, np.ndarray) and value.shape == ():
                payload[key] = value.item()

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

    def dump(self, path: Path) -> Path:
        """Persist the compressor artifact."""
        artifact_path = Path(path)
        if artifact_path.is_dir() or artifact_path.suffix == "":
            artifact_path.mkdir(parents=True, exist_ok=True)
            artifact_path = artifact_path / "compression.npz"
        else:
            artifact_path.parent.mkdir(parents=True, exist_ok=True)

        payload = self.model_dump()
        payload["__compression_class__"] = self._class_spec(type(self))
        arrays: dict[str, np.ndarray] = {}
        for key, value in payload.items():
            if value is None:
                continue
            arrays[key] = np.asarray(value)
        save_path = artifact_path if artifact_path.suffix == ".npz" else artifact_path.with_suffix(".npz")
        np.savez_compressed(save_path, **arrays)
        return save_path


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
    template: PyTree | None = None

    @model_validator(mode="after")
    def _validate_rank_policy(self):
        if self.energy_tol is None and self.rank is None:
            raise ValueError("Must specify rank or energy_tol for SVD compression.")
        if self.energy_tol is not None and not (0.0 < self.energy_tol <= 1.0):
            raise ValueError("energy_tol must be in the interval (0, 1].")
        return self

    @staticmethod
    def _flatten_sample(sample: PyTree) -> jax.Array:
        """Flatten one sample pytree into a single feature vector."""
        leaves = [jnp.ravel(jnp.asarray(leaf)) for leaf in jax.tree.leaves(sample) if eqx.is_array_like(leaf)]
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
            template=samples[0],
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

        array_template, static_template = eqx.partition(self.template, eqx.is_array_like)
        leaves, treedef = jax.tree.flatten(array_template)
        sizes = [int(jnp.size(leaf)) for leaf in leaves]

        offset = 0
        reconstructed_leaves = []
        for leaf, size in zip(leaves, sizes):
            chunk = vector[offset : offset + size]
            reconstructed_leaves.append(jnp.asarray(chunk).reshape(jnp.asarray(leaf).shape))
            offset += size

        reconstructed_arrays = jax.tree.unflatten(treedef, reconstructed_leaves)
        return eqx.combine(reconstructed_arrays, static_template)

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
