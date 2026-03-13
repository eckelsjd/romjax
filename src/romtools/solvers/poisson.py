from typing import Callable, Literal, Mapping, TypedDict, TypeAlias, Union

import jax.numpy as jnp
from jax.typing import ArrayLike
from pydantic import BaseModel, ConfigDict, Field, field_validator

from romtools.model import Model
from romtools.typing import PyTree

ForcingName: TypeAlias = Literal["gaussian"]
ConductivityName: TypeAlias = Literal["nonlinear"]
ForcingCallable: TypeAlias = Callable[[PyTree, PyTree], ArrayLike]


def _get(inputs: object, name: str) -> ArrayLike:
    if hasattr(inputs, name):
        return getattr(inputs, name)
    if isinstance(inputs, Mapping):
        return inputs[name]
    raise TypeError(f"Expected inputs to be a mapping or object with attribute {name!r}.")


class GaussianForcingInputs(BaseModel):
    """Inputs for Gaussian forcing function.

    :ivar A0: amplitude
    :ivar sigma: symmetric width of Gaussian
    :ivar mu_x: center of Gaussian in x-direction
    :ivar mu_y: center of Gaussian in y-direction
    :ivar x: the x coordinates
    :ivar y: the y coordinates
    """
    model_config = ConfigDict(arbitrary_types_allowed=True, validate_assignment=True, validate_default=True)

    A0: ArrayLike = Field(default=1.0, description="Amplitude.")
    sigma: ArrayLike = Field(default=1.0, description="Symmetric width of the Gaussian.")
    mu_x: ArrayLike = Field(default=0.0, description="Center of Gaussian in x-direction.")
    mu_y: ArrayLike = Field(default=0.0, description="Center of Gaussian in y-direction.")
    x: ArrayLike = Field(default=0.0, description="x-coordinates.")
    y: ArrayLike = Field(default=0.0, description="y-coordinates.")


class NonlinearConductivityInputs(BaseModel):
    """Inputs for nonlinear conductivity function.
    
    :ivar k0: the background random field conductivity (2D)
    :ivar alpha: the strength of the nonlinearity
    """
    model_config = ConfigDict(arbitrary_types_allowed=True, validate_assignment=True, validate_default=True)

    k0: ArrayLike = Field(default=1.0, description="Background random field conductivity (2D).")
    alpha: ArrayLike = Field(default=1.0, description="Strength of the nonlinearity.")


class PoissonOutputs(TypedDict):
    """Outputs for the Poisson equation.
    
    :ivar phi: the scalar potential on the grid
    """
    phi: ArrayLike


def gaussian_forcing(inputs: GaussianForcingInputs, outputs: PoissonOutputs) -> ArrayLike:
    """Symmetric Gaussian bump.

        $f(x,y) = A_0 \exp(-1/(2\sigma) ((x-\mu_x)^2 + (y-\mu_y)^2))$
    
    :param inputs: the input parameters
    :param outputs: the scalar potential on the grid (not used)
    :return: the forcing on the grid
    """
    # TODO: should work with numpy, jax, plain floats, broadcasting, etc.
    x = _get(inputs, "x")
    y = _get(inputs, "y")
    mu_x = _get(inputs, "mu_x")
    mu_y = _get(inputs, "mu_y")
    dx = x - mu_x
    dy = y - mu_y
    r2 = dx * dx + dy * dy
    A0 = _get(inputs, "A0")
    sigma = _get(inputs, "sigma")
    return A0 * jnp.exp(-r2 / (2 * sigma))


def nonlinear_conductivity(inputs: NonlinearConductivityInputs, outputs: PoissonOutputs) -> ArrayLike:
    """Nonlinear conductivity.

        $k(x,y) = k_0(1 + \alpha \phi^2)$
    
    :param inputs: the input parameters
    :param outputs: the scalar potential on the grid
    :return: the conductivity on the grid
    """
    # TODO: should work with numpy, jax, floats, broadcasting, xarray, etc.
    phi = _get(outputs, "phi")
    k0 = _get(inputs, "k0")
    alpha = _get(inputs, "alpha")
    return k0 * (1 + alpha * (phi * phi))


class Poisson2D(Model):

    forcing: Union[ForcingCallable, ForcingName] = gaussian_forcing
    forcing_defaults: PyTree = Field(
        default_factory=GaussianForcingInputs,
        description="Default inputs for the forcing function (any PyTree).",
    )
    conductivity: Union[ForcingCallable, ConductivityName] = nonlinear_conductivity
    conductivity_defaults: PyTree = Field(
        default_factory=NonlinearConductivityInputs,
        description="Default inputs for the conductivity function (any PyTree).",
    )

    @field_validator("forcing", mode="before")
    @classmethod
    def _coerce_forcing(cls, value: object) -> ForcingCallable:
        if isinstance(value, str):
            mapping: Mapping[str, ForcingCallable] = {
                "gaussian": gaussian_forcing,
            }
            if value not in mapping:
                raise ValueError(f"Unknown forcing function: {value!r}")
            return mapping[value]
        if callable(value):
            return value
        raise TypeError("forcing must be a callable or a supported string literal.")

    @field_validator("conductivity", mode="before")
    @classmethod
    def _coerce_conductivity(cls, value: object) -> ForcingCallable:
        if isinstance(value, str):
            mapping: Mapping[str, ForcingCallable] = {
                "nonlinear": nonlinear_conductivity,
            }
            if value not in mapping:
                raise ValueError(f"Unknown conductivity function: {value!r}")
            return mapping[value]
        if callable(value):
            return value
        raise TypeError("conductivity must be a callable or a supported string literal.")

    def evaluate(self, inputs: PyTree, outputs: PyTree) -> PyTree:
        """Evalute the Poisson residual on a 2D grid.
        
        :param inputs: boundary conditions, params for forcing and conductivity
        :param outputs: the scalar potential on a 2D grid
        :return: the scalar residual on the 2D grid.
        """
        pass

    def solve(self):
        pass
    
