<h1 align='center'>Somax</h1>

Somax is a libray of Second-Order Methods for stochastic optimization 
written in [JAX](https://github.com/google/jax).
Somax is based on the [JAXopt](https://github.com/google/jaxopt) API and can be used as a drop-in 
replacement for JAXopt as well as 
[Optax](https://github.com/google-deepmind/optax) solvers.

Currently supported methods:
- Diagonal Scaling:
    - [AdaHessian](https://arxiv.org/abs/2006.00719) [WIP];
    - Aurora [WIP];
    - [Sophia](https://arxiv.org/abs/2305.14342) [WIP];
- Hessian-free Optimisation:
    - [Newton-CG](https://academic.oup.com/imajna/article/39/2/545/4959058);
- Quasi-Newton:
    - Stochastic L-BFGS [WIP];
- Gauss-Newton:
    - EGN;
    - GNB;
    - IGND;
    - [SGN](https://arxiv.org/abs/2006.02409);
- Natural Gradient:
    - K-FAC [WIP];
    - [SWM-NG](https://arxiv.org/abs/1906.02353).


Future releases:

⚠️ Since JAXopt is being merged into Optax, 
Somax at some point will switch to the Optax API.  



## Installation

```bash
pip install pysomax
```

Requires [JAXopt](https://github.com/patrick-kidger/equinox) 0.8.2+.




## Quick example

TODO




## Citation

If you found this library to be useful in academic work, then please cite:
([arXiv link](TODO))

```bibtex
TODO
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

TODO if a given implementation is ported from another library, 
add credits and more details here.


