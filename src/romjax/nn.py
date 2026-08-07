from typing import Callable

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import ArrayLike, Key

__all__ = ["Affine", "LinearProjection"]


class Affine(eqx.Module):
    """Input-conditioned invertible affine operator in latent coordinates.

    The matrix generator and offset are affine in ``inputs``: 
        A(inputs) * outputs + c(inputs) 
    
    There are two options for the matrix and offset generators:

    Basis approach:
        A(b) = expm( sum Ai*bi )
        c(b) = sum ci*bi
    MLP approach:
        A(b) = A0*(I + U(b)V(b)^T),  with U(b)~MLP(b) and V(b)~MLP(b)
        c(b) = MLP(b)
    
    The matrix exponential guarantees invertibility. The MLPs are normalized to guarantee invertibility.

    You may either pass in the corresponding arrays/modules directly, or use kwargs to randomly initialize them.
    You are responsible for using appropriate MLP options (e.g. shape consistency).
    """

    matrix_basis: ArrayLike  # (inputs_rank + 1, outputs_rank, outputs_rank)
    offset_basis: ArrayLike  # (inputs_rank + 1, outputs_rank)
    
    base_matrix: ArrayLike   # (outputs_rank, outputs_rank)
    u_mlp: eqx.nn.MLP        # f(inputs_rank) -> (outputs_rank x r)
    v_mlp: eqx.nn.MLP        # f(inputs_rank) -> (outputs_rank x r)
    offset_mlp: eqx.nn.MLP   # f(inputs_rank) -> outputs_rank

    eps: float
    rho: float
    mlp: bool

    def __init__(
        self,
        inputs_rank: int | None = None,
        outputs_rank: int | None = None,
        key: Key | None = None,
        matrix_basis: ArrayLike | None = None,
        offset_basis: ArrayLike | None = None,
        base_matrix: ArrayLike | None = None,
        u_mlp: eqx.nn.MLP | None = None,
        v_mlp: eqx.nn.MLP | None = None,
        offset_mlp: eqx.nn.MLP | None = None,
        mlp: bool = False,
        matrix_width_size: int | None = None,
        offset_width_size: int | None = None,
        matrix_depth: int = 2,
        offset_depth: int = 2,
        mlp_rank: int = 4,
        activation: Callable = jax.nn.swish,
        scale: float = 0.25,
        eps: float = 1e-4,
        rho: float = 0.95,
    ) -> None:
        """
        Initialize affine bases.

        :param inputs_rank: input latent dimension for random initialization
        :param outputs_rank: output latent dimension for random initialization
        :param key: JAX random key for random initialization
        :param matrix_basis: explicit generator bases with shape
            ``(inputs_rank + 1, outputs_rank, outputs_rank)``
        :param offset_basis: explicit offset bases with shape
            ``(inputs_rank + 1, outputs_rank)``
        :param scale: random initialization scale (std dev)
        :param eps: nugget to add to diagonal of matrix operator
        """
        self.eps = eps
        self.rho = rho
        self.mlp = mlp

        ## EXPLICIT INITIALIZATION
        if matrix_basis is not None or offset_basis is not None:
            if matrix_basis is None or offset_basis is None:
                raise ValueError("Affine requires both matrix_basis and offset_basis when explicitly initialized.")
            self.matrix_basis = jnp.asarray(matrix_basis)
            self.offset_basis = jnp.asarray(offset_basis)
            if self.matrix_basis.ndim != 3 or self.offset_basis.ndim != 2:
                raise ValueError("Affine basis arrays must have dimensions 3 and 2 respectively.")
            expected = (self.matrix_basis.shape[0], self.matrix_basis.shape[1])
            if self.matrix_basis.shape[1:] != (self.matrix_basis.shape[2], self.matrix_basis.shape[2]):
                raise ValueError("matrix_basis must have square output matrices.")
            if self.offset_basis.shape != expected:
                raise ValueError(f"offset_basis must have shape {expected}, got {self.offset_basis.shape}.")

            # Not used for matrix/offset basis approach
            if base_matrix is not None or u_mlp is not None or v_mlp is not None or offset_mlp is not None:
                raise ValueError("Affine can only do matrix/offset basis approach or MLP approach, not both.")
            
            self.base_matrix = None
            self.u_mlp = None
            self.v_mlp = None
            self.offset_mlp = None 
            self.mlp = False
            return

        if base_matrix is not None or u_mlp is not None or v_mlp is not None or offset_mlp is not None:
            if base_matrix is None or u_mlp is None or v_mlp is None or offset_mlp is None:
                raise ValueError("Affine requires all base_matrix, u_mlp, v_mlp, offset_mlp for explicit init.")

            # Not used for MLP approach
            if matrix_basis is not None or offset_basis is not None:
                raise ValueError("Affine can only do matrix/offset basis approach or MLP approach, not both.")
            
            self.matrix_basis = None
            self.offset_basis = None
            self.mlp = True
            return

        ## RANDOM INITIALIZATION
        if key is None or inputs_rank is None or outputs_rank is None:
            raise ValueError("Affine requires explicit bases or inputs_rank, outputs_rank, and key.")

        if mlp:
            out_size = outputs_rank*mlp_rank  # low-rank matrix
            default_width = int((inputs_rank + out_size) / 2)

            base_key, u_key, v_key, offset_key = jax.random.split(key, 4)

            self.base_matrix = scale * jax.random.normal(base_key, (outputs_rank, outputs_rank))
            self.u_mlp = eqx.nn.MLP(
                in_size=inputs_rank, 
                out_size=out_size,
                width_size=matrix_width_size or default_width,
                depth=matrix_depth,
                activation=activation,
                key=u_key,
            )
            self.v_mlp = eqx.nn.MLP(
                in_size=inputs_rank,
                out_size=out_size,
                width_size=matrix_width_size or default_width,
                depth=matrix_depth,
                activation=activation,
                key=v_key,
            )
            self.offset_mlp = eqx.nn.MLP(
                in_size=inputs_rank,
                out_size=outputs_rank,
                width_size=offset_width_size or int((inputs_rank+outputs_rank) / 2),
                depth=offset_depth,
                activation=activation,
                key=offset_key,
            )

            self.matrix_basis = None
            self.offset_basis = None

        # Default to matrix/offset basis approach
        else:
            matrix_key, offset_key = jax.random.split(key)
            self.matrix_basis = scale * jax.random.normal(matrix_key, (inputs_rank + 1, outputs_rank, outputs_rank))
            self.offset_basis = scale * jax.random.normal(offset_key, (inputs_rank + 1, outputs_rank))

            self.base_matrix = None
            self.u_mlp = None
            self.v_mlp = None
            self.offset_mlp = None 


    def materialize(self, inputs: ArrayLike) -> tuple[ArrayLike, ArrayLike]:
        """Materialize the invertible matrix and offset for one input latent vector.

        :param inputs: latent input vector with shape ``(inputs_rank,)``
        :return: ``(matrix, offset)`` with shapes ``(outputs_rank, outputs_rank)``
            and ``(outputs_rank,)``
        """
        # MLP approach
        if self.mlp:
            values = jnp.asarray(inputs)
            outputs_rank = self.base_matrix.shape[-1]

            alpha = jnp.linalg.norm(self.base_matrix) + self.eps
            I = jnp.eye(outputs_rank)
            A0 = self.base_matrix + alpha * I    # (outputs_rank, outputs_rank)

            root_rho = jnp.sqrt(self.rho)
            U = jnp.reshape(self.u_mlp(values), (outputs_rank, -1))   # (outputs_rank, r) ~ e.g. (32, 4)
            V = jnp.reshape(self.v_mlp(values), (outputs_rank, -1))
            U = (root_rho / jnp.maximum(jnp.linalg.norm(U), root_rho)) * U  # ||U|| < sqrt(rho)
            V = (root_rho / jnp.maximum(jnp.linalg.norm(V), root_rho)) * V  # ||V|| < sqrt(rho)

            generator = A0 @ (I + U @ jnp.matrix_transpose(V)) # guaranteed invertibility
            offset = self.offset_mlp(values)

            return generator, offset

        # Matrix/offset basis approach
        else:
            values = jnp.asarray(inputs)
            if values.ndim != 1 or values.shape[0] != self.matrix_basis.shape[0] - 1:
                raise ValueError(f"inputs must have shape {(self.matrix_basis.shape[0] - 1,)}, got {values.shape}.")
            coefficients = jnp.concatenate((jnp.ones((1,), dtype=values.dtype), values))
            generator = jnp.einsum("i,ijk->jk", coefficients, self.matrix_basis)
            offset = jnp.einsum("i,ij->j", coefficients, self.offset_basis)
            r = offset.shape[-1]

            return jax.scipy.linalg.expm(generator) + self.eps * jnp.eye(r), offset

    def __call__(self, inputs: ArrayLike) -> tuple[ArrayLike, ArrayLike]:
        """Alias for :meth:`materialize`."""
        return self.materialize(inputs)


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
    
