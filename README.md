# SDE parameter estimation

`sdeparams` is a simple python module for parameter estimation in stochastic differential equations with demographic noise of the form

$$
\frac{\rm{d}\mathbf{x}}{\rm{d}t} = \mathbf{A}(\mathbf{x}) + \frac{1}{\sqrt{N}}\boldsymbol \eta(t)
$$

$$
\left\langle \boldsymbol \eta(t)\boldsymbol \eta(t')^\top\right\rangle = \mathbf{B}(\mathbf{x})\delta(t-t')
$$

An objective function is constructed from the linear noise approximation (LNA), using a multiple-shooting approach as described in [Zimmer and Sahle (2014)](http://ieeexplore.ieee.org/abstract/document/7277317/) and [Zimmer (2015)](https://www.sciencedirect.com/science/article/pii/S0025556415001698).

The LNA is determined by the matrices $\bm A$, $\bm B$, and $\bm J$, where $\bm J$ is the Jacobian matrix of $\bm A$, and it is explicitly given by

$$
\frac{\rm{d}\mathbf{x_{\rm det}}}{\rm{d}t} = \mathbf{A}(\mathbf{x}_{\rm det})
$$

$$
\frac{\mathrm{d}\boldsymbol{\Xi}}{\mathrm{d} t} = \mathbf{J}(\mathbf{x}_{\rm det}(t))\, \boldsymbol{\Xi} + \boldsymbol{\Xi}\, \mathbf{J}(\mathbf{x}_{\rm det}(t))^\top + \mathbf{B}(\mathbf{x}_{\rm det}(t))
$$

with  
$$
\boldsymbol{\Xi}(t) = N \left\langle (\mathbf{x}(t) - \mathbf{x}_{\rm det}(t))\, (\mathbf{x}(t) - \mathbf{x}_{\rm det}(t))^\top \right\rangle
$$

## Usage

`Zimmer(A, B, J, data, n_dims, n_obs)` constructs an object with the LNA and the objective function given the corresponding `n_dims x n_dims` matrices and the data with `n_obs` observed variables in the tuple form `(observations, timepoints)`.

The objective function can be then directly evaluated for a set of parameters or passed as an argument to an optimiser (arbitrary choice):

```python
zimmer = Zimmer(A, B, J, data, n_dims, n_obs)
single_evaluation = zimmer.costfn(parameter_1, parameter_2,...)```

or

```python
estimation = optimiser(zimmer.costfn, optimiser_specific_arguments)```

The first part of the Jupyter notebook [`examples.ipynb`](examples.ipynb) reproduces the original results for the cases with and without unobserved states.  
**Note:** the case with external noise has not been implemented yet.

Please, refer to the examples to see how to specify the list of parameters to be estimated.

The optimisation itself is performed here using Differential Evolution ([Storn and Price (1997)](https://link.springer.com/article/10.1023/A:1008202821328)).

#### Note on custom objective functions

The internal methods of the `Zimmer()` object are also present outside of the class, so that special cases can be addressed by manually specifying the objective function.

The second part of [`examples.ipynb`](examples.ipynb) explores this case in a setting for which one of the parameters being estimated appears directly in the observed variable: an SEIR model for norovirus where the observed data corresponds to $S_0-S(t)$, with $S_0=S(t=0)$ unspecified, rather than directly to one of $S$, $E$, $I$, $R$. This reproduces the results from [Buckingham-Jeffery *et al.* (2018)](https://www.sciencedirect.com/science/article/pii/S0025556417303644)

## Acknowledgements

Thanks to Elizabeth Buckingham-Jeffery and Thomas House for helpful discussions.
