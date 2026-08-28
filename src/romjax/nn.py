from collections.abc import Mapping
from typing import Callable, Literal

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import ArrayLike, Key, PyTree

__all__ = ["Affine", "LinearProjection", "SplitLinearProjection"]


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
    identity_jac: bool | Literal["init"] = eqx.field(static=True)
    eps: float = eqx.field(static=True)

    def __init__(
        self,
        inputs_rank: int | None = None,
        outputs_rank: int | None = None,
        key: Key | None = None,
        identity_jac: bool | Literal["init"] = False,
        jacobian_inputs: Literal["inputs", "outputs", "both"] = "inputs",
        matrix_width_size: int | None = None,
        vector_width_size: int | None = None,
        matrix_depth: int = 2,
        vector_depth: int = 2,
        activation: Callable = jax.nn.swish,
        eps: float = 0.0,
        last_layer_var: float | None = None,
    ) -> None:
        """
        Initialize the affine residual MLPs.

        :param inputs_rank: input vector dimension
        :param outputs_rank: output vector dimension
        :param key: JAX random key for the solution and optional Jacobian MLPs
        :param identity_jac: use a fixed identity Jacobian with ``True``; use ``"init"`` to initialize learned
            Jacobian factors at identity
        :param jacobian_inputs: values supplied to the Jacobian MLPs
        :param matrix_width_size: hidden width for the lower and upper MLPs
        :param vector_width_size: hidden width for the solution and diagonal MLPs
        :param matrix_depth: depth of the lower and upper MLPs
        :param vector_depth: depth of the solution and diagonal MLPs
        :param activation: activation shared by all MLPs
        :param eps: optional nugget added to the diagonal of ``H``
        :param last_layer_var: optional variance for normally initialized final Jacobian MLP layer weights
        """
        if inputs_rank is None or outputs_rank is None:
            raise ValueError("Affine requires inputs_rank and outputs_rank.")
        if key is None:
            raise ValueError("Affine requires inputs_rank, outputs_rank, and key.")
        if inputs_rank < 1 or outputs_rank < 1:
            raise ValueError("inputs_rank and outputs_rank must be positive.")
        if jacobian_inputs not in ("inputs", "outputs", "both"):
            raise ValueError("jacobian_inputs must be 'inputs', 'outputs', or 'both'.")
        if identity_jac is not True and identity_jac is not False and identity_jac != "init":
            raise ValueError("identity_jac must be True, False, or 'init'.")
        if matrix_depth < 0 or vector_depth < 0:
            raise ValueError("MLP depths must be nonnegative.")
        if eps < 0.0:
            raise ValueError("eps must be nonnegative.")
        if last_layer_var is not None and last_layer_var < 0.0:
            raise ValueError("last_layer_var must be nonnegative.")
        if identity_jac == "init" and last_layer_var is None:
            last_layer_var = 1e-3

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
            initialize_last_layer: bool = False,
        ) -> eqx.nn.MLP:
            module = eqx.nn.MLP(
                in_size=in_size,
                out_size=out_size,
                width_size=width_size,
                depth=depth,
                activation=activation,
                key=mlp_key,
            )
            if initialize_last_layer and last_layer_var is not None:
                weight_key = jax.random.split(mlp_key)[1]
                last_weight = jax.random.normal(weight_key, module.layers[-1].weight.shape)
                last_weight = last_weight * jnp.sqrt(jnp.asarray(last_layer_var, dtype=last_weight.dtype))
                module = eqx.tree_at(lambda mlp: mlp.layers[-1].weight, module, last_weight)
            return module

        solution_key, lower_key, upper_key, diagonal_key = jax.random.split(key, 4)
        self.solution = make_mlp(
            in_size=inputs_rank,
            out_size=outputs_rank,
            width_size=vector_width,
            depth=vector_depth,
            mlp_key=solution_key,
        )
        if identity_jac is True:
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
            initialize_last_layer=True,
        )
        self.upper = None if lower_size == 0 else make_mlp(
            in_size=jacobian_size,
            out_size=lower_size,
            width_size=matrix_width,
            depth=matrix_depth,
            mlp_key=upper_key,
            initialize_last_layer=True,
        )
        self.diagonal = make_mlp(
            in_size=jacobian_size,
            out_size=outputs_rank,
            width_size=vector_width,
            depth=vector_depth,
            mlp_key=diagonal_key,
            initialize_last_layer=True,
        )
        if identity_jac == "init":
            if self.lower is not None:
                self.lower = eqx.tree_at(
                    lambda mlp: mlp.layers[-1].bias,
                    self.lower,
                    jnp.zeros_like(self.lower.layers[-1].bias),
                )
                self.upper = eqx.tree_at(
                    lambda mlp: mlp.layers[-1].bias,
                    self.upper,
                    jnp.zeros_like(self.upper.layers[-1].bias),
                )
            self.diagonal = eqx.tree_at(
                lambda mlp: mlp.layers[-1].bias,
                self.diagonal,
                jnp.ones_like(self.diagonal.layers[-1].bias),
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

    def materialize(
        self, 
        inputs: ArrayLike, 
        outputs: ArrayLike | None | Literal["solution"] = None
    ) -> tuple[ArrayLike, ArrayLike]:
        """Materialize ``H`` and ``g`` for one input/output pair.

        :param inputs: input vector, accepting a scalar when ``inputs_rank == 1``
        :param outputs: output vector, required for output-dependent Jacobians; use "solution" to materialize the 
                        Jacobian `H` using the solution operator outputs `g`
        :return: ``(H, g)`` with shapes ``(outputs_rank, outputs_rank)`` and ``(outputs_rank,)``
        """
        input_values = self._vector(inputs, self.inputs_rank, "inputs")
        if self.identity_jac is True:
            solution = self.solution(input_values)
            return jnp.eye(self.outputs_rank, dtype=solution.dtype), solution

        solution = self.solution(input_values)
        if outputs == "solution":
            outputs = solution
        jacobian_values = self._jacobian_values(input_values, outputs)
        
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
        if self.identity_jac is True:
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


class SplitLinearProjection(eqx.Module):
    """Linear encoder-decoder with separate latent partitions for two output fields.

    The encoder maps one flat full-state vector to the concatenated coordinates
    ``(lb, lu)``. The decoder maps that full latent vector to concatenated
    reconstructions ``(bhat, uhat)``. Tree gathering, latent splitting, and
    reconstruction scattering are deliberately delegated to
    :func:`romjax.model.eqx_evaluate`.
    """

    encoder_b: ArrayLike
    encoder_u: ArrayLike
    decoder_b: ArrayLike
    decoder_u: ArrayLike
    encoder_b_bias: ArrayLike | None
    encoder_u_bias: ArrayLike | None
    decoder_b_bias: ArrayLike | None
    decoder_u_bias: ArrayLike | None
    input_size: int = eqx.field(static=True)
    b_latent: int = eqx.field(static=True)
    u_latent: int = eqx.field(static=True)
    b_output: int = eqx.field(static=True)
    u_output: int = eqx.field(static=True)

    def __init__(
        self,
        input_size: int | None = None,
        b_latent: int | None = None,
        u_latent: int | None = None,
        b_output: int | None = None,
        u_output: int | None = None,
        key: Key | None = None,
        encoder_b: ArrayLike | None = None,
        encoder_u: ArrayLike | None = None,
        decoder_b: ArrayLike | None = None,
        decoder_u: ArrayLike | None = None,
        encoder_b_bias: ArrayLike | None = None,
        encoder_u_bias: ArrayLike | None = None,
        decoder_b_bias: ArrayLike | None = None,
        decoder_u_bias: ArrayLike | None = None,
        random_bias: bool = False,
        scale: float = 0.25,
    ) -> None:
        """
        Initialize split encoder and decoder weights.

        Supply either all four matrices explicitly or dimensions and ``key`` for
        random initialization. Biases are optional affine offsets for their
        corresponding maps.

        :param input_size: shared full-state input dimension for both encoders
        :param b_latent: dimension of the first latent partition
        :param u_latent: dimension of the second latent partition
        :param b_output: reconstruction size of the first output field
        :param u_output: reconstruction size of the second output field
        :param key: JAX random key for random initialization
        :param encoder_b: first encoder matrix with shape ``(b_latent, input_size)``
        :param encoder_u: second encoder matrix with shape ``(u_latent, input_size)``
        :param decoder_b: first decoder matrix with shape ``(b_output, b_latent + u_latent)``
        :param decoder_u: second decoder matrix with shape ``(u_output, b_latent + u_latent)``
        :param encoder_b_bias: optional first encoder bias with shape ``(b_latent,)``
        :param encoder_u_bias: optional second encoder bias with shape ``(u_latent,)``
        :param decoder_b_bias: optional first decoder bias with shape ``(b_output,)``
        :param decoder_u_bias: optional second decoder bias with shape ``(u_output,)``
        :param random_bias: randomly initialize omitted biases in random-init mode
        :param scale: random initialization scaling factor
        """
        matrices = (encoder_b, encoder_u, decoder_b, decoder_u)
        bias_keys: tuple[Key | None, Key | None, Key | None, Key | None] = (None, None, None, None)
        if any(matrix is not None for matrix in matrices):
            if not all(matrix is not None for matrix in matrices):
                raise ValueError("SplitLinearProjection requires all four matrices when using explicit weights.")
            self.encoder_b = jnp.asarray(encoder_b)
            self.encoder_u = jnp.asarray(encoder_u)
            self.decoder_b = jnp.asarray(decoder_b)
            self.decoder_u = jnp.asarray(decoder_u)
            self._set_dimensions_from_matrices(input_size, b_latent, u_latent, b_output, u_output)
        else:
            dimensions = (input_size, b_latent, u_latent, b_output, u_output)
            if key is None or any(dimension is None for dimension in dimensions):
                raise ValueError(
                    "SplitLinearProjection requires all dimensions and key when explicit matrices are not supplied."
                )
            if any(dimension < 1 for dimension in dimensions):
                raise ValueError("SplitLinearProjection dimensions must be positive.")
            self.input_size = input_size
            self.b_latent = b_latent
            self.u_latent = u_latent
            self.b_output = b_output
            self.u_output = u_output
            encoder_b_key, encoder_u_key, decoder_b_key, decoder_u_key, *bias_keys = jax.random.split(key, 8)
            latent_size = b_latent + u_latent
            self.encoder_b = scale * jax.random.normal(encoder_b_key, (b_latent, input_size))
            self.encoder_u = scale * jax.random.normal(encoder_u_key, (u_latent, input_size))
            self.decoder_b = scale * jax.random.normal(decoder_b_key, (b_output, latent_size))
            self.decoder_u = scale * jax.random.normal(decoder_u_key, (u_output, latent_size))

        self.encoder_b_bias = self._bias(
            encoder_b_bias, self.b_latent, "encoder_b_bias", random_bias, bias_keys[0], scale
        )
        self.encoder_u_bias = self._bias(
            encoder_u_bias, self.u_latent, "encoder_u_bias", random_bias, bias_keys[1], scale
        )
        self.decoder_b_bias = self._bias(
            decoder_b_bias, self.b_output, "decoder_b_bias", random_bias, bias_keys[2], scale
        )
        self.decoder_u_bias = self._bias(
            decoder_u_bias, self.u_output, "decoder_u_bias", random_bias, bias_keys[3], scale
        )

    def _set_dimensions_from_matrices(
        self,
        input_size: int | None,
        b_latent: int | None,
        u_latent: int | None,
        b_output: int | None,
        u_output: int | None,
    ) -> None:
        """Validate explicit matrices and set their dimensions."""
        matrices = (self.encoder_b, self.encoder_u, self.decoder_b, self.decoder_u)
        if any(matrix.ndim != 2 for matrix in matrices):
            raise ValueError("SplitLinearProjection matrices must all be two-dimensional.")
        inferred = (
            self.encoder_b.shape[1],
            self.encoder_b.shape[0],
            self.encoder_u.shape[0],
            self.decoder_b.shape[0],
            self.decoder_u.shape[0],
        )
        if self.encoder_u.shape[1] != inferred[0]:
            raise ValueError("encoder_b and encoder_u must have the same input size.")
        latent_size = inferred[1] + inferred[2]
        if self.decoder_b.shape[1] != latent_size or self.decoder_u.shape[1] != latent_size:
            raise ValueError("Decoder matrices must accept the concatenated latent size.")
        supplied = (input_size, b_latent, u_latent, b_output, u_output)
        if any(value is not None and value != expected for value, expected in zip(supplied, inferred)):
            raise ValueError("Supplied dimensions must match the explicit matrix shapes.")
        self.input_size, self.b_latent, self.u_latent, self.b_output, self.u_output = inferred

    @staticmethod
    def _bias(
        bias: ArrayLike | None,
        size: int,
        name: str,
        random_bias: bool,
        key: Key | None,
        scale: float,
    ) -> ArrayLike | None:
        """Validate one optional bias or initialize it in random-init mode."""
        if bias is not None:
            values = jnp.asarray(bias)
            if values.shape != (size,):
                raise ValueError(f"{name} must have shape {(size,)}, got shape {values.shape}.")
            return values
        if random_bias:
            if key is None:
                raise ValueError("random_bias requires random initialization with a key.")
            return scale * jax.random.normal(key, (size,))
        return None

    def reduce(self, x: ArrayLike) -> ArrayLike:
        """Encode full-space values as concatenated split latent coordinates.

        :param x: full-space vector or batch with last axis ``input_size``
        :return: concatenated coordinates with last axis ``b_latent + u_latent``
        """
        values = jnp.asarray(x)
        lb = jnp.matmul(values, jnp.swapaxes(self.encoder_b, -1, -2))
        lu = jnp.matmul(values, jnp.swapaxes(self.encoder_u, -1, -2))
        if self.encoder_b_bias is not None:
            lb = lb + self.encoder_b_bias
        if self.encoder_u_bias is not None:
            lu = lu + self.encoder_u_bias
        return jnp.concatenate((lb, lu), axis=-1)

    def reconstruct(self, z: ArrayLike) -> ArrayLike:
        """Decode concatenated split latent coordinates into concatenated fields.

        :param z: latent vector or batch with last axis ``b_latent + u_latent``
        :return: concatenated reconstructions with last axis ``b_output + u_output``
        """
        values = jnp.asarray(z)
        bhat = jnp.matmul(values, jnp.swapaxes(self.decoder_b, -1, -2))
        uhat = jnp.matmul(values, jnp.swapaxes(self.decoder_u, -1, -2))
        if self.decoder_b_bias is not None:
            bhat = bhat + self.decoder_b_bias
        if self.decoder_u_bias is not None:
            uhat = uhat + self.decoder_u_bias
        return jnp.concatenate((bhat, uhat), axis=-1)

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
    
