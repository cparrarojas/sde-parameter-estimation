# SDE parameter estimation

`sdeparams` is a simple python module for parameter estimation in stochastic differential equations with demographic noise of the form

<p align="center"><img src="https://rawgit.com/cparrarojas/sde-parameter-estimation/svgs/svgs/0b5dcb93ea61d33f4ece87c5e14a9177.svg?invert_in_darkmode" align=middle width=161.98281pt height=37.82361pt/></p>

<p align="center"><img src="https://rawgit.com/cparrarojas/sde-parameter-estimation/svgs/svgs/1f98df381e7e7f1299dd2d58c4da8f6a.svg?invert_in_darkmode" align=middle width=205.0356pt height=20.53161pt/></p>

An objective function is constructed from the linear noise approximation (LNA), using a multiple-shooting approach as described in [Zimmer and Sahle (2014)](http://ieeexplore.ieee.org/abstract/document/7277317/) and [Zimmer (2015)](https://www.sciencedirect.com/science/article/pii/S0025556415001698).

The LNA is determined by the matrices <img src="https://rawgit.com/cparrarojas/sde-parameter-estimation/svgs/svgs/49143f09f4138f20d1ce793010d0e081.svg?invert_in_darkmode" align=middle width=12.328800000000005pt height=22.46574pt/>, <img src="https://rawgit.com/cparrarojas/sde-parameter-estimation/svgs/svgs/6829020def9b68d78be5c1a3f7ecd4cb.svg?invert_in_darkmode" align=middle width=13.293555000000003pt height=22.46574pt/>, and <img src="https://rawgit.com/cparrarojas/sde-parameter-estimation/svgs/svgs/12378a26763bd0bfd07680b17290f1ab.svg?invert_in_darkmode" align=middle width=10.696455000000004pt height=22.46574pt/>, where <img src="https://rawgit.com/cparrarojas/sde-parameter-estimation/svgs/svgs/12378a26763bd0bfd07680b17290f1ab.svg?invert_in_darkmode" align=middle width=10.696455000000004pt height=22.46574pt/> is the Jacobian matrix of <img src="https://rawgit.com/cparrarojas/sde-parameter-estimation/svgs/svgs/49143f09f4138f20d1ce793010d0e081.svg?invert_in_darkmode" align=middle width=12.328800000000005pt height=22.46574pt/>, and it is explicitly given by

<p align="center"><img src="https://rawgit.com/cparrarojas/sde-parameter-estimation/svgs/svgs/6d288631abac61570cfd41d1bd8516aa.svg?invert_in_darkmode" align=middle width=118.18256999999998pt height=33.812129999999996pt/></p>

<p align="center"><img src="https://rawgit.com/cparrarojas/sde-parameter-estimation/svgs/svgs/b441ef7d75dfc7c5abc8ebc04740c5f8.svg?invert_in_darkmode" align=middle width=342.22319999999996pt height=33.812129999999996pt/></p>

with  
<p align="center"><img src="https://rawgit.com/cparrarojas/sde-parameter-estimation/svgs/svgs/0a1438d4801f1ab1f7cd6b7e500d4f40.svg?invert_in_darkmode" align=middle width=319.01759999999996pt height=20.53161pt/></p>

## Usage

`Zimmer(A, B, J, data, n_dims, n_obs)` constructs an object with the LNA and the objective function given the corresponding `n_dims x n_dims` matrices and the data with `n_obs` observed variables in the tuple form `(observations, timepoints)`.

The objective function can be then directly evaluated for a set of parameters or passed as an argument to an optimiser (arbitrary choice):

```python
zimmer = Zimmer(A, B, J, data, n_dims, n_obs)
single_evaluation = zimmer.costfn(parameter_1, parameter_2,...)
```
or

```python
estimation = optimiser(zimmer.costfn, optimiser_specific_arguments)
```

The first part of the Jupyter notebook [`examples.ipynb`](examples.ipynb) reproduces the original results for the cases with and without unobserved states.  
**Note:** the case with external noise has not been implemented yet.

Please, refer to the examples to see how to specify the list of parameters to be estimated.

The optimisation itself is performed here using Differential Evolution ([Storn and Price (1997)](https://link.springer.com/article/10.1023/A:1008202821328)).

#### Note on custom objective functions

The internal methods of the `Zimmer()` object are also present outside of the class, so that special cases can be addressed by manually specifying the objective function.

The second part of [`examples.ipynb`](examples.ipynb) explores this case in a setting for which one of the parameters being estimated appears directly in the observed variable: an SEIR model for norovirus where the observed data corresponds to <img src="https://rawgit.com/cparrarojas/sde-parameter-estimation/svgs/svgs/a066b8904c15e71f16dd08433937d192.svg?invert_in_darkmode" align=middle width=67.29459pt height=24.65759999999998pt/>, with <img src="https://rawgit.com/cparrarojas/sde-parameter-estimation/svgs/svgs/f1e0f1890902ff3674614852310a856e.svg?invert_in_darkmode" align=middle width=99.257895pt height=24.65759999999998pt/> unspecified, rather than directly to one of <img src="https://rawgit.com/cparrarojas/sde-parameter-estimation/svgs/svgs/e257acd1ccbe7fcb654708f1a866bfe9.svg?invert_in_darkmode" align=middle width=11.027445000000004pt height=22.46574pt/>, <img src="https://rawgit.com/cparrarojas/sde-parameter-estimation/svgs/svgs/84df98c65d88c6adf15d4645ffa25e47.svg?invert_in_darkmode" align=middle width=13.082190000000004pt height=22.46574pt/>, <img src="https://rawgit.com/cparrarojas/sde-parameter-estimation/svgs/svgs/21fd4e8eecd6bdf1a4d3d6bd1fb8d733.svg?invert_in_darkmode" align=middle width=8.515980000000004pt height=22.46574pt/>, <img src="https://rawgit.com/cparrarojas/sde-parameter-estimation/svgs/svgs/1e438235ef9ec72fc51ac5025516017c.svg?invert_in_darkmode" align=middle width=12.608475000000004pt height=22.46574pt/>. This reproduces the results from [Buckingham-Jeffery *et al.* (2018)](https://www.sciencedirect.com/science/article/pii/S0025556417303644)

## Acknowledgements

Thanks to Elizabeth Buckingham-Jeffery and Thomas House for helpful discussions.
