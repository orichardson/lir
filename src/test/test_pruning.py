import numpy as np

from pdg.pdg import PDG
from pdg.rv import Variable as Var
from pdg.dist import CPT
from lir__simpler import (
    pdg_cleanup,
    pdg_decompose,
    decompose_and_infer,
)
from pdg.alg.torch_opt import opt_joint


def _build_dummy_pdg():
    """
    Build a small PDG with two disconnected components:
      - Component 1: A -> B (deterministic copy) with β=1
      - Component 2: C -> D (deterministic copy) with β=1

    This allows us to test that:
      - cleanup (isolated var pruning) does not change results
      - decomposition and per-component inference equals monolithic inference
    """
    A = Var.binvar("A")
    B = Var.binvar("B")
    C = Var.binvar("C")
    D = Var.binvar("D")

    # deterministic aligned copy CPDs by index:
    # map the i-th value of source to the i-th value of target
    P_BA = CPT.det(A, B, {A.ordered[i]: B.ordered[i] for i in range(len(A))})
    P_DC = CPT.det(C, D, {C.ordered[i]: D.ordered[i] for i in range(len(C))})

    M = PDG()
    M += ("A", A)
    M += ("B", B)
    M += ("C", C)
    M += ("D", D)
    M += ("pAB", P_BA)
    M += ("pCD", P_DC)

    # Explicitly set β=1 to ensure edges contribute
    M.set_beta(("A", "B", "pAB"), 1.0)
    M.set_beta(("C", "D", "pCD"), 1.0)
    return M


def _infer_joint(M, gamma=0.0, iters=200):
    """Convenience wrapper to run the default torch-based joint optimization and return RJD."""
    return opt_joint(M, gamma=gamma, iters=iters, verbose=False)


def test_cleanup_keeps_results_equal():
    """
    Verify that pruning isolated variables (exact cleanup) does not change inference results
    on a simple disconnected PDG.

    Procedure:
      1) Build dummy PDG with two disconnected components.
      2) Run monolithic inference -> μ_full.
      3) Cleanup (no tiny-edge pruning) and run inference -> μ_clean.
      4) Compare marginals on variables present in both to ensure equality.
    """
    M = _build_dummy_pdg()

    mu_full = _infer_joint(M, gamma=0.0, iters=200)
    M_clean = pdg_cleanup(M, drop_zero_weight_edges=False)
    mu_clean = _infer_joint(M_clean, gamma=0.0, iters=200)

    for V in M_clean.varlist:
        mf = mu_full[V].to_numpy()
        mc = mu_clean[V].to_numpy()
        assert np.allclose(mf, mc, atol=1e-8)


def test_decompose_and_infer_matches_monolithic():
    """
    Verify that decomposing into connected components and inferring per component,
    then combining, matches monolithic inference on the original PDG.
    """
    M = _build_dummy_pdg()

    mu_full = _infer_joint(M, gamma=0.0, iters=200)

    def _infer_fn(subM: PDG, gamma=0.0, iters=200):
        return _infer_joint(subM, gamma=gamma, iters=iters)

    mu_combo = decompose_and_infer(
        M,
        _infer_fn,
        decompose=True,
        combine_result=True,
        cleanup=True,                 # exact cleanup only
        drop_zero_weight_edges=False, # keep exact semantics
        inference_kwargs=dict(gamma=0.0, iters=200),
    )

    # Compare marginals for all variables
    for V in mu_full.varlist:
        assert np.allclose(mu_full[V].to_numpy(), mu_combo[V].to_numpy(), atol=1e-8)


