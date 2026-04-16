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


> The catfish in the logo is a small nod to *som*, the Belarusian word for catfish. 
> A quiet bottom-dweller, but not a first-order creature.



## Why Somax

- **Composable**: build methods from curvature, solver, damping, preconditioner, and update components.
- **Optax-native**: computed directions are fed through Optax-style update transforms.
- **Planned execution**: a method is assembled once, planned once, and then executed as a stable step pipeline.
- **JAX-first**: intended for `jit`-compiled training loops and explicit control over execution.
- **Multiple solve lanes**: diagonal, parameter-space, and row-space paths are first-class parts of the stack.
- **Research-friendly**: easy to inspect, compare, ablate, and extend.




## Installation

Install JAX for your backend first:

- JAX installation guide: https://docs.jax.dev/en/latest/installation.html

Then install Somax:

```bash
pip install python-somax
```

For local development:

```bash
git clone https://github.com/cor3bit/somax.git
cd somax
pip install -e ".[dev]"
```

Optional:
- install `lineax` only if you want to use CG backends with `backend="lineax"`.



## Quickstart

```python
import jax
import jax.numpy as jnp
import somax


def predict_fn(params, x):
    h = jnp.tanh(x @ params["W1"] + params["b1"])
    return h @ params["W2"] + params["b2"]


rng = jax.random.PRNGKey(0)
k1, k2, k3, k4 = jax.random.split(rng, 4)

params = {
    "W1": 0.1 * jax.random.normal(k1, (16, 32)),
    "b1": jnp.zeros((32,)),
    "W2": 0.1 * jax.random.normal(k2, (32, 10)),
    "b2": jnp.zeros((10,)),
}

batch = {
    "x": jax.random.normal(k3, (64, 16)),
    "y": jax.random.randint(k4, (64,), 0, 10),
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
    params, state, info = train_step(params, state, jax.random.fold_in(rng, step))
```




## Citation

If Somax is useful in your academic work, please cite:

**Second-Order, First-Class: A Composable Stack for Curvature-Aware Training**  
Mikalai Korbit and Mario Zanon  
https://arxiv.org/abs/2603.25976


```bibtex
@article{korbit2026second,
  title={Second-Order, First-Class: A Composable Stack for Curvature-Aware Training},
  author={Korbit, Mikalai and Zanon, Mario},
  journal={arXiv preprint arXiv:2603.25976},
  year={2026}
}
```


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
