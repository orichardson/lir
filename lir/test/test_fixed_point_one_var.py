"""
Fixed-point check for the one-variable, two-CPD PDG.

Goal
  Verify that with r = s and γ = 0, repeated LIR steps (each time solving the
  inner μ* exactly) push the two unconditional CPDs p and q to agree, and that
  their common value equals the normalized geometric mean of the initial
  distributions p₀ and q₀.

Rationale
  In the 1-variable setting with two edges, the inner optimum is
  μ*(x) ∝ p(x)^{1/2} q(x)^{1/2} for r = s. Under envelope differentiation, the
  outer gradient update aligns p and q towards μ*. At convergence, p = q = μ*.

Test
  - Build the tiny PDG; snapshot p₀, q₀, and their geometric mean μ_geo.
  - Run a short LIR training loop (γ=0) that re-solves μ* every step.
  - Check that the learned p_T and q_T (i) agree and (ii) match μ_geo within a
    small tolerance.
"""

import torch
import pytest
from pathlib import Path
import sys

from pdg.alg.torch_opt import opt_joint
from lir__simpler import lir_train

# Ensure the repository's code/ and this test dir are on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # .../code
sys.path.insert(0, str(Path(__file__).parent))                # .../code/test
from helpers_one_var import make_one_var_two_cpd_pdg, normalized_geometric_mean  # noqa: E402


@pytest.mark.parametrize("K", [3, 4])
def test_fixed_point_equals_geometric_mean(K: int):
    """p_T ≈ q_T ≈ normalized geometric mean(p₀, q₀) when r = s, γ = 0."""
    pdg, X, key_p, key_q = make_one_var_two_cpd_pdg(K=K, seed=0)

    with torch.no_grad():
        p0 = pdg.edgedata[key_p]["cpd"].probs()[0].view(-1)
        q0 = pdg.edgedata[key_q]["cpd"].probs()[0].view(-1)
        mu_geo = normalized_geometric_mean(p0, q0)

    mu_init = opt_joint(pdg, gamma=0.0, iters=10, verbose=False)

    lir_train(
        M=pdg,
        gamma=0.0,
        T=60,
        outer_iters=1,
        inner_iters=10,
        lr=5e-2,
        optimizer_ctor=torch.optim.Adam,
        verbose=False,
        mu_init=mu_init,
    )

    with torch.no_grad():
        pT = pdg.edgedata[key_p]["cpd"].probs()[0].view(-1)
        qT = pdg.edgedata[key_q]["cpd"].probs()[0].view(-1)

    # Agree with each other and match the geometric mean of the initial distributions
    assert torch.allclose(pT, qT, atol=5e-3, rtol=0)
    assert torch.allclose(pT, mu_geo, atol=2e-2, rtol=0)
