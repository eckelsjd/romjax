from collections.abc import Mapping
from typing import Callable, Literal

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import ArrayLike, Key, PyTree

__all__ = ["Affine", "LinearProjection"]


class Affine(eqx.Module):
    """Input-conditioned affine residual operator.

    The residual is parameterized as

    .. math:: f(b, u) = H(b, u) (u - g(b)).

    ``solution`` is the MLP for :math:`g`.  ``lower``, ``upper``, and ``diagonal``
    produce the factors of :math:`H = LDU`, with unit diagonals in ``L`` and
    ``U``.  The Jacobian MLPs can depend on inputs, outputs, or their
    concatenation.
    """

    solution: eqx.nn.MLP | None
    lower: eqx.nn.MLP | None
    upper: eqx.nn.MLP | None
    diagonal: eqx.nn.MLP | None
    inputs_rank: int = eqx.field(static=True)
    outputs_rank: int = eqx.field(static=True)
    jacobian_inputs: Literal["inputs", "outputs", "both"] = eqx.field(static=True)
    identity_jac: bool = eqx.field(static=True)
    eps: float = eqx.field(static=True)

    def __init__(
        self,
        inputs_rank: int | None = None,
        outputs_rank: int | None = None,
        key: Key | None = None,
        identity_jac: bool = False,
        jacobian_inputs: Literal["inputs", "outputs", "both"] = "inputs",
        matrix_width_size: int | None = None,
        vector_width_size: int | None = None,
        matrix_depth: int = 2,
        vector_depth: int = 2,
        activation: Callable = jax.nn.swish,
        eps: float = 0.0,
    ) -> None:
        """
        Initialize the affine residual MLPs.

        :param inputs_rank: input vector dimension
        :param outputs_rank: output vector dimension
        :param key: JAX random key for the solution and optional Jacobian MLPs
        :param identity_jac: use a fixed identity Jacobian with no Jacobian MLPs
        :param jacobian_inputs: values supplied to the Jacobian MLPs
        :param matrix_width_size: hidden width for the lower and upper MLPs
        :param vector_width_size: hidden width for the solution and diagonal MLPs
        :param matrix_depth: depth of the lower and upper MLPs
        :param vector_depth: depth of the solution and diagonal MLPs
        :param activation: activation shared by all MLPs
        :param eps: optional nugget added to the diagonal of ``H``
        """
        if inputs_rank is None or outputs_rank is None:
            raise ValueError("Affine requires inputs_rank and outputs_rank.")
        if key is None:
            raise ValueError("Affine requires inputs_rank, outputs_rank, and key.")
        if inputs_rank < 1 or outputs_rank < 1:
            raise ValueError("inputs_rank and outputs_rank must be positive.")
        if jacobian_inputs not in ("inputs", "outputs", "both"):
            raise ValueError("jacobian_inputs must be 'inputs', 'outputs', or 'both'.")
        if matrix_depth < 0 or vector_depth < 0:
            raise ValueError("MLP depths must be nonnegative.")
        if eps < 0.0:
            raise ValueError("eps must be nonnegative.")

        self.inputs_rank = inputs_rank
        self.outputs_rank = outputs_rank
        self.jacobian_inputs = jacobian_inputs
        self.identity_jac = identity_jac
        self.eps = eps

        lower_size = outputs_rank * (outputs_rank - 1) // 2
        jacobian_size = {
            "inputs": inputs_rank,
            "outputs": outputs_rank,
            "both": inputs_rank + outputs_rank,
        }[jacobian_inputs]
        vector_width = vector_width_size if vector_width_size is not None else max(1, (inputs_rank + outputs_rank) // 2)
        matrix_width = (
            matrix_width_size
            if matrix_width_size is not None
            else max(1, (jacobian_size + outputs_rank**2) // 2)
        )
        if vector_width < 1 or matrix_width < 1:
            raise ValueError("MLP widths must be positive.")

        def make_mlp(
            *,
            in_size: int,
            out_size: int,
            width_size: int,
            depth: int,
            mlp_key: Key,
        ) -> eqx.nn.MLP:
            module = eqx.nn.MLP(
                in_size=in_size,
                out_size=out_size,
                width_size=width_size,
                depth=depth,
                activation=activation,
                key=mlp_key,
            )
            return module

        solution_key, lower_key, upper_key, diagonal_key = jax.random.split(key, 4)
        self.solution = make_mlp(
            in_size=inputs_rank,
            out_size=outputs_rank,
            width_size=vector_width,
            depth=vector_depth,
            mlp_key=solution_key,
        )
        if identity_jac:
            self.lower = None
            self.upper = None
            self.diagonal = None
            return

        self.lower = None if lower_size == 0 else make_mlp(
            in_size=jacobian_size,
            out_size=lower_size,
            width_size=matrix_width,
            depth=matrix_depth,
            mlp_key=lower_key,
        )
        self.upper = None if lower_size == 0 else make_mlp(
            in_size=jacobian_size,
            out_size=lower_size,
            width_size=matrix_width,
            depth=matrix_depth,
            mlp_key=upper_key,
        )
        self.diagonal = make_mlp(
            in_size=jacobian_size,
            out_size=outputs_rank,
            width_size=vector_width,
            depth=vector_depth,
            mlp_key=diagonal_key,
        )

    def _vector(self, value: ArrayLike, rank: int, name: str) -> ArrayLike:
        values = jnp.asarray(value).reshape(-1)
        if values.shape != (rank,):
            raise ValueError(f"{name} must have shape ({rank},) or be scalar when rank is one; got {values.shape}.")
        return values

    def _jacobian_values(self, inputs: ArrayLike, outputs: ArrayLike | None) -> ArrayLike:
        values = self._vector(inputs, self.inputs_rank, "inputs")
        if self.jacobian_inputs == "inputs":
            return values
        if outputs is None:
            raise ValueError("outputs are required when jacobian_inputs is 'outputs' or 'both'.")
        output_values = self._vector(outputs, self.outputs_rank, "outputs")
        return output_values if self.jacobian_inputs == "outputs" else jnp.concatenate((values, output_values))

    def _triangular(self, values: ArrayLike, lower: bool) -> ArrayLike:
        rows, cols = jnp.tril_indices(self.outputs_rank, -1) if lower else jnp.triu_indices(self.outputs_rank, 1)
        matrix = jnp.zeros((self.outputs_rank, self.outputs_rank), dtype=values.dtype)
        return matrix.at[rows, cols].set(values)

    def materialize(self, inputs: ArrayLike, outputs: ArrayLike | None = None) -> tuple[ArrayLike, ArrayLike]:
        """Materialize ``H`` and ``g`` for one input/output pair.

        :param inputs: input vector, accepting a scalar when ``inputs_rank == 1``
        :param outputs: output vector, required for output-dependent Jacobians
        :return: ``(H, g)`` with shapes ``(outputs_rank, outputs_rank)`` and ``(outputs_rank,)``
        """
        input_values = self._vector(inputs, self.inputs_rank, "inputs")
        if self.identity_jac:
            solution = self.solution(input_values)
            return jnp.eye(self.outputs_rank, dtype=solution.dtype), solution

        jacobian_values = self._jacobian_values(input_values, outputs)
        solution = self.solution(input_values)
        diagonal = self.diagonal(jacobian_values) + jnp.asarray(self.eps, dtype=solution.dtype)
        if self.outputs_rank == 1:
            matrix = diagonal.reshape(1, 1)
        else:
            lower = jnp.eye(self.outputs_rank, dtype=diagonal.dtype) + self._triangular(
                self.lower(jacobian_values), True
            )
            upper = jnp.eye(self.outputs_rank, dtype=diagonal.dtype) + self._triangular(
                self.upper(jacobian_values), False
            )
            matrix = lower @ jnp.diag(diagonal) @ upper
        return matrix, solution

    def log_determinant(self, payload: PyTree, square: bool = True) -> ArrayLike:
        """Return the log absolute determinant from an implicit-model payload. Optionally sum the squared log instead.

        :param payload: mapping with ``inputs`` and ``outputs`` value payloads
        :param square: whether to square the log *before* summing (prevents total volume scaling and spread/condition)
        :return: sum of the log absolute values of the LDU diagonal
        """
        if self.identity_jac:
            return jnp.asarray(0.0)
        if not isinstance(payload, Mapping) or 'inputs' not in payload or 'outputs' not in payload:
            raise TypeError("Affine.log_determinant payload must be a Mapping with 'inputs' and 'outputs'.")

        def payload_value(value: PyTree, name: str) -> ArrayLike:
            if not isinstance(value, Mapping) or 'value' not in value:
                raise TypeError(f"Affine.log_determinant {name} must contain 'value'.")
            return value["value"]

        input_values = self._vector(payload_value(payload["inputs"], "inputs"), self.inputs_rank, "inputs")
        output_values = self._vector(payload_value(payload["outputs"], "outputs"), self.outputs_rank, "outputs")
        jacobian_values = self._jacobian_values(input_values, output_values)
        diagonal = self.diagonal(jacobian_values)
        diagonal = diagonal + jnp.asarray(self.eps, dtype=diagonal.dtype)
        return jnp.sum(jnp.square(jnp.log(jnp.abs(diagonal)))) if square else jnp.sum(jnp.log(jnp.abs(diagonal)))

    def __call__(self, inputs: ArrayLike, outputs: ArrayLike | None = None) -> tuple[ArrayLike, ArrayLike]:
        """Alias for :meth:`materialize`."""
        return self.materialize(inputs, outputs)


class LinearProjection(eqx.Module):
    """Affine projection module with tied transpose reconstruction."""

    matrix: ArrayLike  # (r x N)
    bias: ArrayLike | None  # (N,)

    def __init__(
        self,
        latent: int | None = None,
        dof: int | None = None,
        key: Key | None = None,
        matrix: ArrayLike | None = None,
        bias: ArrayLike | None = None,
        random_bias: bool = False,
        scale: float = 0.25,
    ):
        """
        Initialize projection weights.

        This supports two equivalent styles:

        - explicit matrix: ``LinearProjection(matrix=...)``
        - random init: ``LinearProjection(latent=..., dof=..., key=...)``

        When supplied, ``bias`` is a full-space offset. The projection centers
        inputs with this offset and adds it back during reconstruction.

        :param latent: latent dimension when using random initialization
        :param dof: full-space dimension when using random initialization
        :param key: random key when using random initialization
        :param matrix: explicit projection matrix with shape ``(latent, dof)``
        :param bias: optional full-space offset with shape ``(dof,)``
        :param random_bias: whether to initialize an omitted bias randomly
        :param scale: random init scaling factor
        """
        if matrix is not None:
            self.matrix = jnp.asarray(matrix)
            if self.matrix.ndim != 2:
                raise ValueError(f"matrix must have dim 2, got shape {self.matrix.shape}.")
            if bias is not None:
                self.bias = jnp.asarray(bias)
                if self.bias.shape != (self.matrix.shape[1],):
                    raise ValueError(
                        f"bias must have shape {(self.matrix.shape[1],)}, got shape {self.bias.shape}."
                    )
            else:
                self.bias = None
            return

        if key is None or latent is None or dof is None:
            raise ValueError(
                "LinearProjection requires either `matrix` or all of (`latent`, `dof`, `key`)."
            )
        matrix_key, bias_key = jax.random.split(key)
        self.matrix = scale * jax.random.normal(matrix_key, (latent, dof))
        if bias is None:
            self.bias = scale * jax.random.normal(bias_key, (dof,)) if random_bias else None
        else:
            self.bias = jnp.asarray(bias)
            if self.bias.shape != (dof,):
                raise ValueError(f"bias must have shape {(dof,)}, got shape {self.bias.shape}.")

    def reduce(self, x: ArrayLike) -> ArrayLike:
        """
        Project from full to reduced coordinates using ``z = x W^T``.

        :param x: full-space vector/tensor with last axis ``n_full``
        :return: reduced coordinates with last axis ``n_latent``
        """
        matrix = jnp.asarray(self.matrix)
        values = jnp.asarray(x)
        if self.bias is not None:
            values = values - jnp.asarray(self.bias)
        return jnp.matmul(values, jnp.swapaxes(matrix, -1, -2))

    def reconstruct(self, z: ArrayLike) -> ArrayLike:
        """
        Reconstruct from reduced to full coordinates using ``x_hat = z W``.

        :param z: reduced coordinates with last axis ``n_latent``
        :return: reconstructed full coordinates with last axis ``n_full``
        """
        values = jnp.matmul(jnp.asarray(z), jnp.asarray(self.matrix))
        if self.bias is not None:
            values = values + jnp.asarray(self.bias)
        return values

    def __call__(self, x: ArrayLike) -> ArrayLike:
        """Alias for :meth:`reduce`."""
        return self.reduce(x)


class ConvAutoencoder2D(eqx.Module):
    """
    Small convolutional autoencoder for 2D fields.

    Single-sample tensor convention is ``(channels, height, width)``.
    """

    encoder_conv: eqx.nn.Conv2d
    encoder_linear: eqx.nn.Linear
    decoder_linear: eqx.nn.Linear
    decoder_conv: eqx.nn.ConvTranspose2d

    input_shape: tuple[int, int] = eqx.field(static=True)
    in_channels: int = eqx.field(static=True)
    hidden_channels: int = eqx.field(static=True)
    latent_dim: int = eqx.field(static=True)

    def __init__(
        self,
        input_shape: tuple[int, int],
        latent_dim: int,
        key: Key,
        in_channels: int = 1,
        hidden_channels: int = 4,
    ):
        """
        Build a convolutional autoencoder with stride-2 encoder and transpose-conv decoder.

        :param input_shape: spatial input shape ``(height, width)`` (both must be even)
        :param latent_dim: latent code dimension
        :param key: random key for initialization
        :param in_channels: input channels
        :param hidden_channels: hidden channel count
        """
        h, w = input_shape
        if h % 2 != 0 or w % 2 != 0:
            raise ValueError("ConvAutoencoder2D expects even input_shape in both dimensions.")

        h2, w2 = h // 2, w // 2
        flat_dim = hidden_channels * h2 * w2
        k1, k2, k3, k4 = jax.random.split(key, 4)

        self.encoder_conv = eqx.nn.Conv2d(
            in_channels=in_channels,
            out_channels=hidden_channels,
            kernel_size=3,
            stride=2,
            padding=1,
            key=k1,
        )
        self.encoder_linear = eqx.nn.Linear(flat_dim, latent_dim, key=k2)
        self.decoder_linear = eqx.nn.Linear(latent_dim, flat_dim, key=k3)
        self.decoder_conv = eqx.nn.ConvTranspose2d(
            in_channels=hidden_channels,
            out_channels=in_channels,
            kernel_size=4,
            stride=2,
            padding=1,
            key=k4,
        )

        self.input_shape = input_shape
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.latent_dim = latent_dim

    def encode(self, x: ArrayLike) -> ArrayLike:
        """
        Encode one sample or a batch to latent coordinates.

        :param x: input sample ``(channels, height, width)`` or batch ``(batch, channels, height, width)``
        :return: latent vector ``(latent_dim,)`` or batch ``(batch, latent_dim)``
        """
        x_array = jnp.asarray(x)
        if x_array.ndim == 3:
            return self._encode_one(x_array)
        if x_array.ndim == 4:
            return jax.vmap(self._encode_one)(x_array)
        raise ValueError(f"encode expects rank-3 or rank-4 input but received shape {x_array.shape}.")

    def decode(self, z: ArrayLike) -> ArrayLike:
        """
        Decode one latent vector or a batch to reconstructed samples.

        :param z: latent vector ``(latent_dim,)`` or batch ``(batch, latent_dim)``
        :return: reconstructed sample ``(channels, height, width)`` or batch ``(batch, channels, height, width)``
        """
        z_array = jnp.asarray(z)
        if z_array.ndim == 1:
            return self._decode_one(z_array)
        if z_array.ndim == 2:
            return jax.vmap(self._decode_one)(z_array)
        raise ValueError(f"decode expects rank-1 or rank-2 input but received shape {z_array.shape}.")

    def _encode_one(self, x: ArrayLike) -> ArrayLike:
        """Encode one sample with shape ``(channels, height, width)``."""
        y = self.encoder_conv(jnp.asarray(x))
        y = jax.nn.tanh(y)
        return self.encoder_linear(jnp.ravel(y))

    def _decode_one(self, z: ArrayLike) -> ArrayLike:
        """Decode one latent vector with shape ``(latent_dim,)``."""
        h2, w2 = self.input_shape[0] // 2, self.input_shape[1] // 2
        y = self.decoder_linear(jnp.asarray(z))
        y = jax.nn.tanh(y)
        y = jnp.reshape(y, (self.hidden_channels, h2, w2))
        return self.decoder_conv(y)

    def __call__(self, x: ArrayLike) -> ArrayLike:
        """Autoencode one sample."""
        return self.decode(self.encode(x))
    
