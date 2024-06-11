<h1 align='center'>Somax</h1>

Somax is a library of Second-Order Methods for stochastic optimization
written in [JAX](https://github.com/google/jax).
Somax is based on the [JAXopt](https://github.com/google/jaxopt) StochasticSolver API,
and can be used as a drop-in
replacement for JAXopt as well as
[Optax](https://github.com/google-deepmind/optax) solvers.

Currently supported methods:

- Diagonal Scaling:
    - [AdaHessian](https://ojs.aaai.org/index.php/AAAI/article/view/17275);
    - [Sophia](https://arxiv.org/abs/2305.14342);
- Hessian-free Optimization:
    - [Newton-CG](https://epubs.siam.org/doi/10.1137/10079923X);
- Quasi-Newton:
    - [Stochastic quasi-Newton with Line Search (SQN)](https://www.sciencedirect.com/science/article/pii/S0005109821000236);
- Gauss-Newton:
    - [Exact Gauss-Newton (EGN)](https://arxiv.org/abs/2405.14402);
    - [Stochastic Gauss-Newton (SGN)](https://arxiv.org/abs/2006.02409);
- Natural Gradient:
    - [Natural Gradient with Sherman-Morrison-Woodbury formula (SWM-NG)](https://arxiv.org/abs/1906.02353).

Future releases:

- Add support for separate "gradient batches"
  and "curvature batches" for all solvers;
- Add support for Optax rate schedules.

⚠️ Since JAXopt is currently being merged into Optax,
Somax at some point will switch to the Optax API as well.

## Installation

```bash
pip install python-somax
```

Requires [JAXopt](https://github.com/patrick-kidger/equinox) 0.8.2+.

## Quick example

```py
from somax import EGN

# initialize the solver
solver = EGN(
    predict_fun=model.apply,
    loss_type='mse',
    learning_rate=0.1,
    regularizer=1.0,
)

# initialize the solver state
opt_state = solver.init_state(params)

# run the optimization loop
for i in range(10):
    params, opt_state = solver.update(params, opt_state, batch_x, batch_y)
```

See more in the [examples](examples) folder.

## Citation

```bibtex
@misc{korbit2024somax,
  author = {Nick Korbit},
  title = {{SOMAX}: a library of second-order methods for stochastic optimization written in {JAX}},
  year = {2024},
  url = {https://github.com/cor3bit/somax},
}
```

## See also

**Optimization with JAX**  
[Optax](https://github.com/google-deepmind/optax): first-order gradient (SGD, Adam, ...) optimisers.  
[JAXopt](https://github.com/google/jaxopt): deterministic second-order methods (e.g., Gauss-Newton, Levenberg
Marquardt), stochastic first-order methods PolyakSGD, ArmijoSGD.

**Awesome Projects**  
[Awesome JAX](https://github.com/n2cholas/awesome-jax): a longer list of various JAX projects.  
[Awesome SOM for ML](https://github.com/cor3bit/awesome-som4ml): a list
of resources for second-order optimization methods in machine learning.

## Acknowledgements

Some of the implementation ideas are inspired by the following repositories:

- Line Search in JAXopt: https://github.com/google/jaxopt/blob/main/jaxopt/_src/armijo_sgd.py

- AdaHessian (official implementation): https://github.com/amirgholami/adahessian

- AdaHessian (Nestor Demeure's implementation): https://github.com/nestordemeure/AdaHessianJax

- Sophia (official implementation): https://github.com/Liuhong99/Sophia

- Sophia (levanter implementation): https://github.com/stanford-crfm/levanter/blob/main/src/levanter/optim/sophia.py

