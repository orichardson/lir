# Calculating the Gradients of Inconsistency.

As currently implemented

Currently the code in the PDG repository allows for the calculation of inconsistency, but not in a way that allows us to get its gradients easily. One of the first tasks in this project is to extend the implementation to use pytorch gradients.  I suspect we will be able to use both the [`torchopt`](https://github.com/metaopt/torchopt) library and [cvxpy parameters](https://www.cvxpy.org/api_reference/cvxpy.expressions.html#parameter) for this purpose.

Roughly speaking, there are several approaches which can be pursued in parallel:

1. **Gradients from The Convex Solver.** Extend the functions in `code/pdg/alg/interior_pt.py` to use `cvxpy` [parameters](https://www.cvxpy.org/tutorial/dpp/index.html), so we can differentiate the optimum with respect to our parameters of interest.  
  We should start with the method `cvx_opt_joint` specialized to  `idef=False` for simplicity. 
  
2. Extend the functions in `code/pdg/alg/torch_opt.py` to get gradients of the inconsistency, using [torchopt](https://github.com/metaopt/torchopt)

    * Expected Gradient: traces the gradients through the eentire optimization process. This is likely to be most effective in practice. We may be able to alternate a few steps of an optimizer that aims to adjust $\mu$ so as to more closely approximate $\min_\mu F(\mu, \theta)$, and steps of that adjust $\theta$ itself. 
    
    * Implicit Gradient: get the gradient through differentiating the optimality conditions. This would be ideal, but the interface exposed by the `torchopt` library may prove too restrictive for our purposes. 
    
    * Zero Order Differentiation: this may be a necessary fallback to get things working in general in situations where the methods above do not cover LIR in every case. 
    
    
