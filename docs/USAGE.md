# Usage



## A paper in a few lines

Somax builds curvature-aware optimizers by composing curvature operators,
estimators, solvers, damping, preconditioners, and Optax transforms.

Here are four methods expressed using the same building blocks:
| Method | Curvature | Estimator | Update |
| --- | --- | --- | --- |
| AdaHessian | Hessian | Hutchinson diagonal | EMA scaling |
| Sophia-G | GGN | GNB diagonal | EMA scaling and clipping |
| Stochastic Gauss-Newton | GGN | - | Parameter-space CG |
| Fast Gauss-Newton | FGN | - | Row-space CG |

The examples below implement the optimization rules, not the complete experimental protocols of the original papers. Hyperparameters are illustrative.




### Shared setup

All examples use logits for multiclass classification and a batch with `"x"` and `"y"` keys. 
AdaHessian uses the scalar loss; the other methods use the prediction function.

```python
import jax
import jax.numpy as jnp
import optax
import somax

from somax.specs import (
    CurvatureSpec, DampingSpec, EstimatorSpec, PrecondSpec, SolverSpec
)
from somax.optax import sophia_tx


def predict_fn(params, x):
    return x @ params["W"] + params["b"]


def loss_fn(params, batch):
    logits = predict_fn(params, batch["x"])
    return optax.softmax_cross_entropy_with_integer_labels(
        logits, batch["y"]
    ).mean()


key = jax.random.PRNGKey(0)
kx, ky, kw = jax.random.split(key, 3)
params = {
    "W": 0.1 * jax.random.normal(kw, (8, 3)),
    "b": jnp.zeros((3,)),
}
batch = {
    "x": jax.random.normal(kx, (32, 8)),
    "y": jax.random.randint(ky, (32,), 0, 3),
}
```



### AdaHessian

True Hessian, Hutchinson diagonal estimation, exponential moving averages, and diagonal scaling:

```python
method = somax.assemble(
    curvature=CurvatureSpec("hessian", {"loss_fn": loss_fn}),
    estimator=EstimatorSpec("hutchinson", {"n_probes": 1}),
    solver=SolverSpec("identity"),
    precond=PrecondSpec("diag_ema", {
        "beta1": 0.9, "beta2": 0.999,
        "hessian_power": 1.0, "spatial_averaging": True,
    }),
    learning_rate=1e-3,
)
```



### Sophia-G

GGN, Gauss-Newton-Bartlett (GNB) diagonal estimation, EMA scaling, 
coordinate-wise clipping, and weight decay:

```python
method = somax.assemble(
    curvature=CurvatureSpec("ggn_ce", {"predict_fn": predict_fn}),
    estimator=EstimatorSpec("gnb_ce", {"n_samples": 1}),
    solver=SolverSpec("identity"),
    precond=PrecondSpec("diag_ema", {
        "beta1": 0.965, "beta2": 0.99,
        "gamma": 0.05, "eval_every_k": 10,
    }),
    tx=sophia_tx(learning_rate=1e-4, weight_decay=0.2, clip_value=1.0),
)
```




### Stochastic Gauss-Newton (SGN)

Generalized Gauss-Newton curvature with damping and parameter-space conjugate gradients:

```python
method = somax.assemble(
    curvature=CurvatureSpec("ggn_ce", {"predict_fn": predict_fn}),
    solver=SolverSpec("cg", {"maxiter": 20, "tol": 1e-4}),
    damping=DampingSpec("constant", lam0=1e-2),
    learning_rate=0.1,
)
```



### Fast Gauss-Newton (FGN)

FGN approximates the multiclass cross-entropy GGN and solves in row space.
Lifted CG accumulates the parameter direction during iterations, avoiding
a final backprojection.


```python
method = somax.assemble(
    curvature=CurvatureSpec("fgnha", {"predict_fn": predict_fn}),
    solver=SolverSpec("row_cg_lifted", {"maxiter": 20, "tol": 1e-4}),
    damping=DampingSpec("constant", lam0=1e-2),
    learning_rate=0.1,
)
```



## Prefer a preset?

For standard configurations, choose one of the built-in presets:

```python
method = somax.adahessian(loss_fn=loss_fn, learning_rate=1e-3)
method = somax.sophia_g(predict_fn=predict_fn, learning_rate=1e-4)
method = somax.sgn_ce(predict_fn=predict_fn, lam0=1e-2, learning_rate=0.1)
method = somax.fgn_ce(predict_fn=predict_fn, lam0=1e-2, lifted=True, learning_rate=0.1)
```

Presets expose additional method-specific options. To inspect the registry:

```python
somax.list_methods()
somax.describe("sophia_g")
```



## Run any method

Every assembled method and preset uses the same `init`/`step` interface:

```python
state = method.init(params)


@jax.jit
def train_step(params, batch, state, rng):
    return method.step(params, batch, state, rng)


for i in range(100):
    step_key = jax.random.fold_in(key, i)
    params, state, info = train_step(params, batch, state, step_key)
```

In real training, pass a new minibatch to each step. 
Construct the method once, outside `jax.jit`.
