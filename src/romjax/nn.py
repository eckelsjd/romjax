from collections.abc import Mapping, Sequence
from typing import Callable, Literal

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import ArrayLike, Key, PyTree

__all__ = ["Affine", "BlockLinearAutoencoder", "LinearProjection"]


class _TriangularMLP(eqx.Module):
    """Wrap an MLP that parameterizes one strict triangular matrix factor."""

    mlp: eqx.nn.MLP
    outputs_rank: int = eqx.field(static=True)
    jacobian_inputs: Literal["inputs", "outputs", "both"] = eqx.field(static=True)
    inputs_rank: int = eqx.field(static=True)
    lower: bool = eqx.field(static=True)

    @property
    def layers(self):
        """Expose underlying layers for compatibility with :class:`equinox.nn.MLP`."""
        return self.mlp.layers

    def __call__(self, inputs: ArrayLike) -> ArrayLike:
        """Evaluate the wrapped MLP."""
        return self.mlp(inputs)

    def tikhonov(self, payload: PyTree) -> ArrayLike:
        """Regularize the materialized strict triangular entries for one training payload.

        :param payload: mapping with ``inputs`` and ``outputs`` value payloads
        :return: squared Frobenius norm of the materialized strict triangular matrix
        """
        if not isinstance(payload, Mapping) or "inputs" not in payload or "outputs" not in payload:
            raise TypeError("Triangular Affine MLP tikhonov payload must be a Mapping with 'inputs' and 'outputs'.")

        def payload_value(value: PyTree, name: str) -> ArrayLike:
            if not isinstance(value, Mapping) or "value" not in value:
                raise TypeError(f"Triangular Affine MLP tikhonov {name} must contain 'value'.")
            return jnp.asarray(value["value"]).reshape(-1)

        input_values = payload_value(payload["inputs"], "inputs")
        output_values = payload_value(payload["outputs"], "outputs")
        if input_values.shape != (self.inputs_rank,):
            raise ValueError(f"inputs must have shape ({self.inputs_rank},); got {input_values.shape}.")
        if output_values.shape != (self.outputs_rank,):
            raise ValueError(f"outputs must have shape ({self.outputs_rank},); got {output_values.shape}.")
        values = {
            "inputs": input_values,
            "outputs": output_values,
            "both": jnp.concatenate((input_values, output_values)),
        }[self.jacobian_inputs]
        rows, cols = (
            jnp.tril_indices(self.outputs_rank, -1)
            if self.lower
            else jnp.triu_indices(self.outputs_rank, 1)
        )
        matrix = jnp.zeros((self.outputs_rank, self.outputs_rank), dtype=values.dtype)
        matrix = matrix.at[rows, cols].set(self.mlp(values))
        return jnp.sum(jnp.square(matrix))


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
    lower: _TriangularMLP | None
    upper: _TriangularMLP | None
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

        self.lower = None if lower_size == 0 else _TriangularMLP(
            mlp=make_mlp(
                in_size=jacobian_size,
                out_size=lower_size,
                width_size=matrix_width,
                depth=matrix_depth,
                mlp_key=lower_key,
                initialize_last_layer=True,
            ),
            outputs_rank=outputs_rank,
            jacobian_inputs=jacobian_inputs,
            inputs_rank=inputs_rank,
            lower=True,
        )
        self.upper = None if lower_size == 0 else _TriangularMLP(
            mlp=make_mlp(
                in_size=jacobian_size,
                out_size=lower_size,
                width_size=matrix_width,
                depth=matrix_depth,
                mlp_key=upper_key,
                initialize_last_layer=True,
            ),
            outputs_rank=outputs_rank,
            jacobian_inputs=jacobian_inputs,
            inputs_rank=inputs_rank,
            lower=False,
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
                    lambda module: module.mlp.layers[-1].bias,
                    self.lower,
                    jnp.zeros_like(self.lower.mlp.layers[-1].bias),
                )
                self.upper = eqx.tree_at(
                    lambda module: module.mlp.layers[-1].bias,
                    self.upper,
                    jnp.zeros_like(self.upper.mlp.layers[-1].bias),
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
    skip_bias: bool = eqx.field(static=True)

    def __init__(
        self,
        latent: int | None = None,
        dof: int | None = None,
        key: Key | None = None,
        matrix: ArrayLike | None = None,
        bias: ArrayLike | None = None,
        random_bias: bool = False,
        scale: float = 0.25,
        skip_bias: bool = False,
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
        :param skip_bias: if true, do not use the bias during evaluation (default false)
        """
        self.skip_bias = skip_bias

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
        if self.bias is not None and not self.skip_bias:
            values = values - jnp.asarray(self.bias)
        return jnp.matmul(values, jnp.swapaxes(matrix, -1, -2))

    def reconstruct(self, z: ArrayLike) -> ArrayLike:
        """
        Reconstruct from reduced to full coordinates using ``x_hat = z W``.

        :param z: reduced coordinates with last axis ``n_latent``
        :return: reconstructed full coordinates with last axis ``n_full``
        """
        values = jnp.matmul(jnp.asarray(z), jnp.asarray(self.matrix))
        if self.bias is not None and not self.skip_bias:
            values = values + jnp.asarray(self.bias)
        return values

    def __call__(self, x: ArrayLike) -> ArrayLike:
        """Alias for :meth:`reduce`."""
        return self.reduce(x)


class BlockLinearAutoencoder(eqx.Module):
    """Linear autoencoder represented by corresponding input and latent blocks.

    The encoder block ``(i, j)`` maps input partition ``j`` to latent partition
    ``i``. The decoder block ``(i, j)`` maps latent partition ``j`` back to input
    partition ``i``. In diagonal mode, off-diagonal blocks are stored as ``None``
    and do not participate in evaluation.
    """

    encoder_blocks: tuple[tuple[ArrayLike | None, ...], ...]
    decoder_blocks: tuple[tuple[ArrayLike | None, ...], ...]
    bias: ArrayLike | None
    input_sizes: tuple[int, ...] = eqx.field(static=True)
    latent_sizes: tuple[int, ...] = eqx.field(static=True)
    diagonal: bool = eqx.field(static=True)
    ignore_nan: bool = eqx.field(static=True)

    def __init__(
        self,
        input_sizes: Sequence[int],
        latent_sizes: Sequence[int],
        key: Key | None = None,
        encoder_blocks: Sequence[Sequence[ArrayLike | None]] | None = None,
        decoder_blocks: Sequence[Sequence[ArrayLike | None]] | None = None,
        bias: ArrayLike | None = None,
        random_bias: bool = False,
        diagonal: bool = False,
        ignore_nan: bool = False,
        scale: float = 0.25,
    ) -> None:
        """Initialize block encoder and decoder weights.

        Supply both block grids explicitly, or provide ``key`` to initialize all
        active blocks randomly. ``input_sizes`` and ``latent_sizes`` are segment
        lengths, not cumulative split indices.

        :param input_sizes: sizes of the full-space vector partitions
        :param latent_sizes: sizes of the corresponding latent partitions
        :param key: JAX random key used for random initialization
        :param encoder_blocks: block grid with shapes ``(latent_sizes[i], input_sizes[j])``
        :param decoder_blocks: block grid with shapes ``(input_sizes[i], latent_sizes[j])``
        :param bias: optional full-space centering vector
        :param random_bias: randomly initialize an omitted bias using ``key``
        :param diagonal: omit and skip every off-diagonal block
        :param ignore_nan: omit NaN input components from every encoder contribution
        :param scale: random initialization scaling factor
        """
        self.input_sizes = tuple(int(size) for size in input_sizes)
        self.latent_sizes = tuple(int(size) for size in latent_sizes)
        self.diagonal = diagonal
        self.ignore_nan = ignore_nan
        self._validate_sizes()

        if (encoder_blocks is None) != (decoder_blocks is None):
            raise ValueError("encoder_blocks and decoder_blocks must be supplied together.")
        if encoder_blocks is None:
            if key is None:
                raise ValueError("Random block initialization requires a key.")
            self.encoder_blocks, self.decoder_blocks, bias_key = self._random_blocks(key, scale)
        else:
            self.encoder_blocks = self._validate_blocks(encoder_blocks, encoder=True)
            self.decoder_blocks = self._validate_blocks(decoder_blocks, encoder=False)
            bias_key = key

        if bias is not None:
            self.bias = jnp.asarray(bias)
            if self.bias.shape != (self.input_size,):
                raise ValueError(f"bias must have shape {(self.input_size,)}, got shape {self.bias.shape}.")
        elif random_bias:
            if bias_key is None:
                raise ValueError("random_bias requires a key.")
            self.bias = scale * jax.random.normal(bias_key, (self.input_size,))
        else:
            self.bias = None

    @property
    def input_size(self) -> int:
        """Return the total full-space dimension."""
        return sum(self.input_sizes)

    @property
    def latent_size(self) -> int:
        """Return the total latent dimension."""
        return sum(self.latent_sizes)

    def _validate_sizes(self) -> None:
        """Validate static partition metadata."""
        if not self.input_sizes or not self.latent_sizes:
            raise ValueError("input_sizes and latent_sizes must be non-empty.")
        if len(self.input_sizes) != len(self.latent_sizes):
            raise ValueError("input_sizes and latent_sizes must contain the same number of partitions.")
        if any(size < 1 for size in (*self.input_sizes, *self.latent_sizes)):
            raise ValueError("Block partition sizes must be positive.")

    def _random_blocks(
        self, key: Key, scale: float
    ) -> tuple[
        tuple[tuple[ArrayLike | None, ...], ...],
        tuple[tuple[ArrayLike | None, ...], ...],
        Key,
    ]:
        """Initialize active blocks and return a remaining bias key."""
        count = len(self.input_sizes)
        active_count = count if self.diagonal else count * count
        keys = iter(jax.random.split(key, 2 * active_count + 1))
        encoder = tuple(
            tuple(
                scale * jax.random.normal(next(keys), (self.latent_sizes[i], self.input_sizes[j]))
                if not self.diagonal or i == j else None
                for j in range(count)
            )
            for i in range(count)
        )
        decoder = tuple(
            tuple(
                scale * jax.random.normal(next(keys), (self.input_sizes[i], self.latent_sizes[j]))
                if not self.diagonal or i == j else None
                for j in range(count)
            )
            for i in range(count)
        )
        return encoder, decoder, next(keys)

    def _validate_blocks(
        self,
        blocks: Sequence[Sequence[ArrayLike | None]],
        *,
        encoder: bool,
    ) -> tuple[tuple[ArrayLike | None, ...], ...]:
        """Validate and normalize one square block grid."""
        count = len(self.input_sizes)
        if len(blocks) != count or any(len(row) != count for row in blocks):
            raise ValueError(f"Block grids must have shape {(count, count)}.")
        normalized: list[tuple[ArrayLike | None, ...]] = []
        for i, row in enumerate(blocks):
            normalized_row: list[ArrayLike | None] = []
            for j, block in enumerate(row):
                active = not self.diagonal or i == j
                if not active:
                    if block is not None:
                        raise ValueError("Off-diagonal blocks must be None when diagonal=True.")
                    normalized_row.append(None)
                    continue
                if block is None:
                    raise ValueError("Every active block must be supplied.")
                values = jnp.asarray(block)
                expected = (
                    (self.latent_sizes[i], self.input_sizes[j])
                    if encoder else (self.input_sizes[i], self.latent_sizes[j])
                )
                if values.shape != expected:
                    kind = "encoder" if encoder else "decoder"
                    raise ValueError(f"{kind}_blocks[{i}][{j}] must have shape {expected}, got {values.shape}.")
                normalized_row.append(values)
            normalized.append(tuple(normalized_row))
        return tuple(normalized)

    @staticmethod
    def _apply_blocks(
        blocks: tuple[tuple[ArrayLike | None, ...], ...], values: tuple[jax.Array, ...]
    ) -> jax.Array:
        """Apply one block matrix to partitioned values without materializing it."""
        outputs: list[jax.Array] = []
        for row in blocks:
            result = None
            for block, value in zip(row, values):
                if block is None:
                    continue
                contribution = jnp.matmul(value, jnp.swapaxes(jnp.asarray(block), -1, -2))
                result = contribution if result is None else result + contribution
            if result is None:  # pragma: no cover - constructor validation prevents this
                raise ValueError("Each block row must contain at least one active block.")
            outputs.append(result)
        return jnp.concatenate(outputs, axis=-1)

    def reduce(self, x: ArrayLike) -> ArrayLike:
        """Encode full-space values into concatenated latent coordinates.

        :param x: full-space vector or batch with last axis ``sum(input_sizes)``
        :return: latent coordinates with last axis ``sum(latent_sizes)``
        """
        values = jnp.asarray(x)
        if values.ndim == 0 or values.shape[-1] != self.input_size:
            raise ValueError(f"x must have last-axis size {self.input_size}, got shape {values.shape}.")
        if self.bias is not None:
            values = values - jnp.asarray(self.bias)
        if self.ignore_nan:
            values = jnp.where(jnp.isnan(values), jnp.zeros_like(values), values)
        split_indices = tuple(sum(self.input_sizes[:index]) for index in range(1, len(self.input_sizes)))
        partitions = tuple(jnp.split(values, split_indices, axis=-1))
        return self._apply_blocks(self.encoder_blocks, partitions)

    def reconstruct(self, z: ArrayLike) -> ArrayLike:
        """Decode latent coordinates into the concatenated full-space vector.

        :param z: latent vector or batch with last axis ``sum(latent_sizes)``
        :return: reconstruction with last axis ``sum(input_sizes)``
        """
        values = jnp.asarray(z)
        if values.ndim == 0 or values.shape[-1] != self.latent_size:
            raise ValueError(f"z must have last-axis size {self.latent_size}, got shape {values.shape}.")
        split_indices = tuple(sum(self.latent_sizes[:index]) for index in range(1, len(self.latent_sizes)))
        partitions = tuple(jnp.split(values, split_indices, axis=-1))
        reconstructed = self._apply_blocks(self.decoder_blocks, partitions)
        if self.bias is not None:
            reconstructed = reconstructed + jnp.asarray(self.bias)
        return reconstructed

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
    
