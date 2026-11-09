"""Type definitions and protocols for Somax."""

from collections.abc import Callable, Mapping
from typing import Any, Protocol, TypeAlias

import jax
from jax.typing import ArrayLike
import optax
from flax.core import FrozenDict

# ----- Core -----
Array: TypeAlias = jax.Array

# ----- PyTree aliases -----
PyTree: TypeAlias = Any
Params: TypeAlias = PyTree
Grads: TypeAlias = PyTree
Updates: TypeAlias = PyTree

# ----- Data I/O -----
Batch: TypeAlias = Mapping[str, ArrayLike | PyTree]
Inputs: TypeAlias = PyTree
Targets: TypeAlias = PyTree
Labels: TypeAlias = Array  # integer array

# ----- Scalars / Keys -----
Scalar: TypeAlias = Array  # usually 0-d Array
PRNGKey: TypeAlias = Array  # usually uint32[2] (opaque)

# ----- Optax -----
ScalarOrSchedule: TypeAlias = float | optax.Schedule
GradientTransformation: TypeAlias = optax.GradientTransformation


# ----- User-facing callables -----
class PredictFn(Protocol):
    def __call__(self, params: Params, x: Inputs) -> Any: ...


class StatefulPredictFn(Protocol):
    """Prediction function with non-parameter state updated by one primal call.

    The primary output is differentiated with respect to ``params``. The
    returned state update is auxiliary data and must be a JAX-compatible
    PyTree, for example Flax Linen's ``batch_stats`` collection.
    """

    def __call__(
            self,
            params: Params,
            x: Inputs,
            model_state: PyTree,
    ) -> tuple[Any, PyTree]: ...


PredictFnLike: TypeAlias = PredictFn | StatefulPredictFn


class LossFn(Protocol):
    def __call__(self, params: Params, batch: Batch) -> Scalar: ...


# ----- Linear-operator types (matrix-free) -----
MatVec = Callable[[Updates], Updates]  # v -> A v, PyTree in/out
PrecondFn = Callable[[Updates], Updates]  # param-space
RowPrecondFn = Callable[[Array], Array]  # row-space

JvpFn = Callable[[Updates], Array]  # parameter tangent -> row tangent
VjpFn = Callable[[Array], Updates]  # row cotangent -> parameter cotangent

# ----- StepInfo: metrics container -----
StepInfo: TypeAlias = FrozenDict[str, jax.Array]
