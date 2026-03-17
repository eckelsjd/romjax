from pydantic import BaseModel, Field, TypeAdapter
from pydantic_core import core_schema

from typing import Any, Protocol
from weakref import WeakKeyDictionary

from jax.typing import ArrayLike
import jax.numpy as jnp
import jax

import optimistix as optx
import lineax as lx

import matplotlib.pyplot as plt

# from romtools.solvers.utils import homogeneous_boundary

class ModuleObjectSpec(BaseModel):
    name: str
    opts: dict[str, Any] = Field(default_factory=dict)

class ModuleObjectBuilder(Protocol):
    def __call__(self, name: str, opts: dict[str, Any]) -> Any: ...

_SPEC_REGISTRY: WeakKeyDictionary[object, dict[str, Any]] = WeakKeyDictionary()

def _build_from_module(module) -> ModuleObjectBuilder:
    def _build(name: str, opts: dict[str, Any]) -> Any:
        if not hasattr(module, name):
            raise ValueError(f"Module '{module.__name__}' does not contain attribute '{name}'")
        return getattr(module, name)(**opts)
    return _build

def _store_spec(obj: object, name: str, opts: dict[str, Any]) -> None:
    spec = {"name": name, "opts": opts}
    _SPEC_REGISTRY[obj] = spec
    try:
        setattr(obj, "__romtools_spec__", spec)
    except Exception:
        pass

def _get_spec(obj: object) -> dict[str, Any] | None:
    if hasattr(obj, "__romtools_spec__"):
        return getattr(obj, "__romtools_spec__")
    return _SPEC_REGISTRY.get(obj)

def _serialize_value(value: Any) -> Any:
    spec = _get_spec(value) if not isinstance(value, (int, float, str, bool, type(None))) else None
    if spec is None:
        return value
    return {
        "name": spec["name"],
        "opts": {k: _serialize_value(v) for k, v in spec["opts"].items()},
    }

def _module_object_schema(builder: ModuleObjectBuilder, opts_adapter: TypeAdapter | None) -> core_schema.CoreSchema:
    def validate(value: Any) -> Any:
        if isinstance(value, dict):
            spec = ModuleObjectSpec.model_validate(value)
            opts = spec.opts
            if opts_adapter is not None:
                opts = opts_adapter.validate_python(opts)
            obj = builder(spec.name, opts)
            _store_spec(obj, spec.name, opts)
            return obj
        return value

    def serialize(value: Any) -> Any:
        spec = _get_spec(value)
        if spec is None:
            return {"name": value.__class__.__name__, "opts": getattr(value, "__dict__", {})}
        return _serialize_value(value)

    return core_schema.no_info_plain_validator_function(
        validate,
        serialization=core_schema.plain_serializer_function_ser_schema(serialize),
    )

def module_object_type(builder: ModuleObjectBuilder, *, opts_adapter: TypeAdapter | None = None) -> type:
    class ModuleObject:
        @classmethod
        def __get_pydantic_core_schema__(cls, _source, _handler):
            return _module_object_schema(builder, opts_adapter)
    return ModuleObject

LxObject = module_object_type(_build_from_module(lx))
LxOptsAdapter = TypeAdapter(dict[str, LxObject | Any])
OptxObject = module_object_type(_build_from_module(optx), opts_adapter=LxOptsAdapter)

d = {
    "name": "Newton",
    "opts": {
        "rtol": 1e-3,
        "atol": 1e-6,
        "linear_solver": {
            "name": "CG",
            "opts": {
                "rtol": 1e-2,
                "atol": 1e4
            }
        }
    }
}
class SolverConfig(BaseModel):
    solver: OptxObject

cfg = SolverConfig.model_validate({"solver": d})
print("Constructed solver:", cfg.solver)
print("Nested linear_solver type:", type(cfg.solver.linear_solver))
print("Serialized back to dict:", cfg.model_dump())

# b = homogeneous_boundary(ndim=2)

# d = b.model_dump()
# print(d)


# class MyModel(DictModel):
#     alpha: ArrayLike = 1.
#     beta: ArrayLike = Field(default_factory=lambda: jnp.linspace(0, 1, 10))

# def func(a, *overrides):
#     for d in overrides:
#         a.update(d)

# a = MyModel()
# b = a.model_dump()
# print(b['beta'] is a['beta'])
# print(b)
# a = {'1': 1}

# print(a)

# func(a, {'2': 2, '3': 3}, {'hello': 'goodbye'}, {'1': 4})

# print(a)

# a = MyModel()
# arr = jnp.linspace(0.5, 1.5, 20)

# b = MyModel(alpha=jnp.linspace(1, 2, 30))
# a.update(b)
# a.update({'beta': arr})

# def f(x):
#     return x ** 3

# df = jax.vmap(jax.grad(f))


# fig, ax = plt.subplots(layout='tight', figsize=(4,3))
# ax.plot(a['beta'], df(a['beta']), label='Beta')
# ax.plot(a['alpha'], df(a['alpha']), label='alpha')
# ax.legend()
# plt.show()
