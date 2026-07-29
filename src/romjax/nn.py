import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import ArrayLike, Key

__all__ = ["Affine", "LinearProjection"]


class Affine(eqx.Module):
    """Input-conditioned invertible affine operator in latent coordinates.

    The matrix generator and offset are affine in ``inputs``. The actual
    residual matrix is the matrix exponential of the generator, which is
    invertible for every finite input.
    """

    matrix_basis: ArrayLike  # (inputs_rank + 1, outputs_rank, outputs_rank)
    offset_basis: ArrayLike  # (inputs_rank + 1, outputs_rank)

    def __init__(
        self,
        inputs_rank: int | None = None,
        outputs_rank: int | None = None,
        key: Key | None = None,
        matrix_basis: ArrayLike | None = None,
        offset_basis: ArrayLike | None = None,
        scale: float = 0.25,
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
        :param scale: random initialization scale
        """
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
            return

        if key is None or inputs_rank is None or outputs_rank is None:
            raise ValueError("Affine requires explicit bases or inputs_rank, outputs_rank, and key.")
        matrix_key, offset_key = jax.random.split(key)
        self.matrix_basis = scale * jax.random.normal(matrix_key, (inputs_rank + 1, outputs_rank, outputs_rank))
        self.offset_basis = scale * jax.random.normal(offset_key, (inputs_rank + 1, outputs_rank))

    def materialize(self, inputs: ArrayLike) -> tuple[ArrayLike, ArrayLike]:
        """Materialize the invertible matrix and offset for one input vector.

        :param inputs: latent input vector with shape ``(inputs_rank,)``
        :return: ``(matrix, offset)`` with shapes ``(outputs_rank, outputs_rank)``
            and ``(outputs_rank,)``
        """
        values = jnp.asarray(inputs)
        if values.ndim != 1 or values.shape[0] != self.matrix_basis.shape[0] - 1:
            raise ValueError(f"inputs must have shape {(self.matrix_basis.shape[0] - 1,)}, got {values.shape}.")
        coefficients = jnp.concatenate((jnp.ones((1,), dtype=values.dtype), values))
        generator = jnp.einsum("i,ijk->jk", coefficients, self.matrix_basis)
        offset = jnp.einsum("i,ij->j", coefficients, self.offset_basis)
        return jax.scipy.linalg.expm(generator), offset

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
    
