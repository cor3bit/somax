<h1 align="center">Somax</h1>

<p align="center">
  <img src="assets/somax_logo_mini.png" alt="Somax logo" width="250px"/>
</p>

<p align="center">
  Composable Second-Order Optimization for JAX and Optax.
</p>

<p align="center">
  A small research-engineering library for curvature-aware training:
  modular, matrix-free, and explicit about the moving parts.
</p>

---

Somax is a JAX-native library for building and running second-order optimization methods from explicit components.

Rather than treating an optimizer as a monolithic object, Somax factors a step into swappable pieces:
- curvature operator
- solver
- damping policy
- optional preconditioner
- update transform
- optional telemetry and control signals

That decomposition is the point.

Somax is built for users who want a clean second-order stack in JAX without hiding the execution model. 
It aims to make curvature-aware training easier to inspect, compare, and extend.


> The catfish in the logo is a small nod to "som", the Belarusian word for catfish. A quiet bottom-dweller, but not a first-order creature.




## What Somax does

Somax treats curvature-aware optimization as a **planned step pipeline**.

A method is assembled once, its execution path is fixed, and the resulting step can be JIT-compiled. 
In a typical step, Somax:
1. builds the step-local linearization
2. constructs the required curvature actions
3. solves the local second-order subproblem
4. applies the chosen update transform
5. returns updated state and optional step information

This structure makes choices that are usually hidden inside optimizer-specific code explicit:
- which curvature approximation is used
- whether solving happens in diagonal, parameter, or row space
- how damping is controlled
- how diagonal or spectral statistics are refreshed
- which first-order machinery is applied after the direction is computed





## Installation

Install from source:

```bash
git clone https://github.com/cor3bit/somax.git
cd somax
pip install -e .
````

JAX installation is backend-specific. 
Install the appropriate JAX build for your CPU, GPU, or TPU environment before using Somax.






## Minimal example

```python
import jax
import jax.numpy as jnp
import somax


def predict_fn(params, x):
    h = jnp.tanh(x @ params["W1"] + params["b1"])
    return h @ params["W2"] + params["b2"]


key = jax.random.PRNGKey(0)

params = {
    "W1": jax.random.normal(key, (16, 32)),
    "b1": jnp.zeros((32,)),
    "W2": jax.random.normal(key, (32, 10)),
    "b2": jnp.zeros((10,)),
}

batch = {
    "x": jax.random.normal(key, (64, 16)),
    "y": jax.random.randint(key, (64,), 0, 10),
}

method = somax.sgn_ce(
    predict_fn=predict_fn,
    lam0=1e-2,
    tol=1e-4,
    maxiter=20,
    learning_rate=1e-1,
)

state = method.init(params)

@jax.jit
def train_step(params, state, rng):
    params, state, info = method.step(params, batch, state, rng)
    return params, state, info

for step in range(10):
    params, state, info = train_step(params, state, jax.random.fold_in(key, step))
```





## Architecture

A simplified view of the stack:

```text
preset / assemble
    ->
planner
    ->
executor
    ->
{curvature, solver, damping, preconditioner, update transform}
```

Key module families include:

* `somax.curvature`
* `somax.solvers`
* `somax.damping`
* `somax.preconditioners`
* `somax.methods`

The central design choice is to separate **assembly and planning** from **step execution**. 
This keeps the public API compact while preserving explicit control over curvature, solving, damping, and telemetry.






## Related projects

**Optimization in JAX**  
[Optax](https://github.com/google-deepmind/optax): first-order gradient (e.g., SGD, Adam) optimisers.  
[JAXopt](https://github.com/google/jaxopt): deterministic second-order methods (e.g., Gauss-Newton, Levenberg
Marquardt), stochastic first-order methods PolyakSGD, ArmijoSGD.

**Awesome Projects**  
[Awesome JAX](https://github.com/n2cholas/awesome-jax): a longer list of various JAX projects.  
[Awesome SOMs](https://github.com/cor3bit/awesome-soms): a list
of resources for second-order optimization methods in machine learning.




## License

Apache-2.0
