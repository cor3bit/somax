<h1 align='center'>Somax</h1>

Somax is a [JAX](https://github.com/google/jax) library for 
stochastic second-order optimization 
based on [JAXopt](https://github.com/google/jaxopt) API.

Supported solvers:
- Hessian-free optimisation;
- SGN;
- IGND;
- EGN;
- Natural gradient.

TODO links to papers and descriptions of the algorithms. 


## Installation

```bash
pip install somax
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
[JAXopt](https://github.com/google/jaxopt): deterministic second-order methods (e.g., Gauss-Newton, Levenberg Marquardt), stochastic first-order methods PolyakSGD, ArmijoSGD.

**Awesome Projects**  
[Awesome JAX](https://github.com/n2cholas/awesome-jax): a longer list of various JAX projects.  
[Awesome SOM for ML](https://github.com/cor3bit/awesome-som4ml): a list 
of resources for second-order optimization methods in machine learning.  
