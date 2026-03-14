from typing import Callable, Literal, Mapping, TypedDict

import jax.numpy as jnp
from jax.typing import ArrayLike
from pydantic import Field, field_validator

from romtools.model import Model
from romtools.typing import PyTree, DictModel

type ForcingName = Literal["gaussian", "nonlinear"]
type ForcingCallable = Callable[[PyTree, PyTree], ArrayLike]


class GaussianForcingInputs(DictModel):
    """Inputs for Gaussian forcing function.

    :ivar A0: amplitude
    :ivar sigma: symmetric width of Gaussian
    :ivar mu_x: center of Gaussian in x-direction
    :ivar mu_y: center of Gaussian in y-direction
    :ivar x: the x coordinates
    :ivar y: the y coordinates
    """
    A0: ArrayLike = Field(default=1.0, description="Amplitude.")
    sigma: ArrayLike = Field(default=1.0, description="Symmetric width of the Gaussian.")
    mu_x: ArrayLike = Field(default=0.0, description="Center of Gaussian in x-direction.")
    mu_y: ArrayLike = Field(default=0.0, description="Center of Gaussian in y-direction.")
    x: ArrayLike = Field(default=0.0, description="x-coordinates.")
    y: ArrayLike = Field(default=0.0, description="y-coordinates.")


class NonlinearConductivityInputs(DictModel):
    """Inputs for nonlinear conductivity function.
    
    :ivar k0: the background random field conductivity (2D)
    :ivar alpha: the strength of the nonlinearity
    """
    k0: ArrayLike = Field(default=1.0, description="Background random field conductivity (2D).")
    alpha: ArrayLike = Field(default=1.0, description="Strength of the nonlinearity.")


class DirichletInputs(DictModel):
    pass # TODO: dirichlet boundary conditions


class PoissonInputs(DictModel):
    forcing: DictModel = Field(default_factory=GaussianForcingInputs)
    conductivity: DictModel = Field(default_factory=NonlinearConductivityInputs)
    boundary: DictModel = Field(default_factory=DirichletInputs)


class PoissonOutputs(TypedDict):
    """Outputs for the Poisson equation.
    
    :ivar phi: the scalar potential on the grid
    """
    phi: ArrayLike


# TODO: coerce inputs to match pydantic schemas in jax grad friendly way
def gaussian_forcing(inputs: GaussianForcingInputs, outputs: PoissonOutputs) -> ArrayLike:
    """Symmetric Gaussian bump.

        $f(x,y) = A_0 \exp(-1/(2\sigma) ((x-\mu_x)^2 + (y-\mu_y)^2))$
    
    :param inputs: the input parameters
    :param outputs: the scalar potential on the grid (not used)
    :return: the forcing on the grid
    """
    dx = inputs['x'] - inputs['mu_x']
    dy = inputs['y'] - inputs['mu_y']
    return inputs['A0'] * jnp.exp(-(dx * dx + dy * dy) / (2 * inputs['sigma']))


# TODO: coerce inputs to match pydantic schemas in jax grad friendly way
def nonlinear_conductivity(inputs: NonlinearConductivityInputs, outputs: PoissonOutputs) -> ArrayLike:
    """Nonlinear conductivity.

        $k(x,y) = k_0(1 + \alpha \phi^2)$
    
    :param inputs: the input parameters
    :param outputs: the scalar potential on the grid
    :return: the conductivity on the grid
    """
    phi = outputs['phi']
    return inputs['k0'] * (1 + inputs['alpha'] * (phi * phi))


def dirichlet():
    pass  # TODO: dirichlet boundary conditions


def neumann():
    pass # TODO: neumann boundary conditions


class Poisson2D(Model):

    forcing: ForcingCallable | ForcingName = gaussian_forcing
    forcing_defaults: DictModel = Field(
        default_factory=GaussianForcingInputs,
        description="Default inputs for the forcing function (any PyTree).",
    )
    conductivity: ForcingCallable | ForcingName = nonlinear_conductivity
    conductivity_defaults: DictModel = Field(
        default_factory=NonlinearConductivityInputs,
        description="Default inputs for the conductivity function (any PyTree).",
    )

    @field_validator("forcing", "conductivity", mode="before")
    @classmethod
    def _coerce_forcing(cls, value: object) -> ForcingCallable:
        if isinstance(value, str):
            mapping: Mapping[str, ForcingCallable] = {
                "gaussian": gaussian_forcing,
                "nonlinear": nonlinear_conductivity,
            }
            if value not in mapping:
                raise ValueError(f"Unknown function: {value!r}")
            return mapping[value]
        if callable(value):
            return value
        raise TypeError("forcing and conductivity must be a callable or a supported string literal.")

    @field_validator("forcing_defaults", "conductivity_defaults", mode="before")
    @classmethod
    def _coerce_defaults(cls, value: object) -> DictModel:
        if isinstance(value, DictModel):
            return value
        if isinstance(value, type) and issubclass(value, DictModel):
            return value()
        if isinstance(value, Mapping):
            return DictModel.model_validate(value)
        raise TypeError("forcing_defaults and conductivity_defaults must be a DictModel or a dict-like mapping.")

    def evaluate(self, inputs: PoissonInputs, outputs: PoissonOutputs) -> ArrayLike:
        """Evalute the Poisson residual on a 2D domain.
        
        :param inputs: params for forcing, conductivity, and boundary conditions
        :param outputs: the scalar potential on a 2D domain
        :return: the scalar residual on the 2D domain
        """
        pass

    def solve(self):
        pass
    
