"""
Shared helpers for one-variable, two-CPD PDG tests.

This module centralizes small utilities used by multiple tests to avoid code
duplication and keep the intent clear. It includes:

- PDG construction for a single variable X with two ParamCPDs labeled "p" and
  "q" (both unconditional over X)
- Probability extraction helpers
- Analytic ingredients for the 1-variable case: weighted geometric mean μ*,
  the closed-form inconsistency f(p, q; r, s), and KL(μ||p)
"""

from typing import Tuple
import torch

from pdg.pdg import PDG
from pdg.rv import Variable as Var, Unit
from pdg.dist import CPT, ParamCPD


def make_one_var_two_cpd_pdg(K: int = 3, seed: int = 0) -> Tuple[PDG, Var, tuple, tuple]:
    """Build a 1-variable PDG with two ParamCPDs over the same variable.

    The edges are labeled "p" and "q" and both are unconditional distributions
    over X. We only set β=1.0 in the PDG edge data, matching the simplified
    experiments used throughout.
    """
    torch.manual_seed(seed)
    X = Var.alph("X", K)
    pdg = PDG() + X

    # Use Unit->X CPTs to obtain unconditional table shapes
    P_p = CPT.make_random(Unit, X)
    P_q = CPT.make_random(Unit, X)

    cpd_p = ParamCPD(src_var=X, tgt_var=X, name="p", init="random", mask=None, cpd=P_p)
    cpd_q = ParamCPD(src_var=X, tgt_var=X, name="q", init="random", mask=None, cpd=P_q)

    key_p = (X.name, X.name, "p")
    key_q = (X.name, X.name, "q")

    # Use ASCII field names expected by PDG tooling
    pdg.edgedata[key_p] = {"cpd": cpd_p, "beta": 1.0, "alpha": 1.0}
    pdg.edgedata[key_q] = {"cpd": cpd_q, "beta": 1.0, "alpha": 1.0}

    return pdg, X, key_p, key_q


def get_pq_from_pdg(pdg: PDG, key_p: tuple, key_q: tuple) -> Tuple[torch.Tensor, torch.Tensor]:
    """Extract 1-D probability vectors for p and q (shape [K])."""
    p = pdg.edgedata[key_p]["cpd"].probs()[0].view(-1)
    q = pdg.edgedata[key_q]["cpd"].probs()[0].view(-1)
    return p, q


def normalized_geometric_mean(p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    """Return the normalized geometric mean of two 1-D probability vectors."""
    mu = torch.sqrt(p * q)
    return mu / mu.sum()


def weighted_geometric_mean(p: torch.Tensor, q: torch.Tensor, r: float, s: float, eps: float = 1e-12) -> torch.Tensor:
    """Return μ*(x) ∝ p(x)^{α} q(x)^{1-α} with α = r/(r+s), normalized."""
    α = r / (r + s)
    a = α * torch.log(p + eps) + (1 - α) * torch.log(q + eps)
    logZ = torch.logsumexp(a, dim=-1)
    mu = torch.exp(a - logZ)
    return mu


def f_inconsistency(p: torch.Tensor, q: torch.Tensor, r: float, s: float, eps: float = 1e-12) -> torch.Tensor:
    """Analytic inconsistency f (with the conventional leading negative sign).

    f(p, q; r, s) = - (r+s) * log sum_x (p(x)^r q(x)^s)^{1/(r+s)}
                   = - (r+s) * logsumexp( (r*log p + s*log q)/(r+s) ).
    """
    a = (r * torch.log(p + eps) + s * torch.log(q + eps)) / (r + s)
    logZ = torch.logsumexp(a, dim=-1)
    return - (r + s) * logZ


def kl_mu_to_p(mu: torch.Tensor, p: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """Return KL(μ || p) = Σ_x μ(x) [log μ(x) − log p(x)]."""
    return (mu * (torch.log(mu + eps) - torch.log(p + eps))).sum()
