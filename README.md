<h1 align="center">Somax</h1>

<p align="center">
  <img src="assets/somax_logo_mini.png" alt="Somax logo" width="350" />
</p>

<p align="center">
  <strong>Second-Order Methods in JAX</strong>
</p>



Somax is a JAX-native library for composable second-order optimization.
It combines curvature operators, solvers, damping, preconditioners,
and update transforms into flexible optimization methods.


Features:
- **Composable:** combine curvature, solvers, damping, preconditioners, and update rules.
- **Matrix-free:** compute curvature products without forming dense matrices.
- **JAX-first:** JIT-compiled steps with explicit control over execution.
- **Optax integration:** apply standard Optax transformations to computed directions.
- **Research-friendly:** swap components, run ablations, and prototype new methods.




> The catfish in the logo is a small nod to *som*, the Belarusian word for catfish. 
> A quiet bottom-dweller, but not a first-order creature.






## Installation

Install [JAX](https://docs.jax.dev/en/latest/installation.html) for your hardware, then:

```bash
pip install python-somax
```

For development:

```bash
git clone https://github.com/cor3bit/somax.git
cd somax
pip install -e ".[dev]"
```

The `lineax` package is optional and needed only for CG with `backend="lineax"`.






## Quickstart


A minimal stochastic Gauss-Newton training loop for multiclass classification:

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
    learning_rate=0.1,
    lam0=1.0,
    maxiter=10,
    tol=1e-4,
)

state = method.init(params)


@jax.jit
def train_step(params, batch, state, rng):
    return method.step(params, batch, state, rng)


for step in range(10):
    step_rng = jax.random.fold_in(rng, step)
    params, state, info = train_step(params, batch, state, step_rng)
```

For more examples, see [Usage: A paper in a few lines](docs/USAGE.md).




## Citation

Mikalai Korbit and Mario Zanon,
"Second-Order, First-Class: A Composable Stack for Curvature-Aware Training",
ECML PKDD 2026, [arXiv](https://arxiv.org/abs/2603.25976).

```bibtex
@inproceedings{korbit2026second,
  title={Second-Order, First-Class: A Composable Stack for Curvature-Aware Training},
  author={Korbit, Mikalai and Zanon, Mario},
  booktitle={Joint European Conference on Machine Learning and Knowledge Discovery in Databases},
  pages={90--105},
  year={2026},
  organization={Springer}
}
```



## License

Apache-2.0

