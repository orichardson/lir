from pathlib import Path
import sys

sys.path.append(str(Path(__file__).absolute().parent.parent.parent))

import random
from typing import List
from typing import Tuple

import torch
import numpy as np

from pdg.pdg import PDG
from pdg.rv import Variable as Var
from pdg.dist import RawJointDist as RJD
from pdg.dist import CPT, ParamCPD
from lir__simpler import lir_train
# from pdg.alg.torch_opt_lir import opt_joint
from pdg.alg.torch_opt import opt_joint


def generate_random_pdg(num_vars: int = 3,
                        num_edges: int = 3,
                        val_range=(2, 3),
                        src_range=(1, 2),
                        tgt_range=(1, 1),
                        seed: int = 0) -> PDG:
    """
    Make a random PDG  (starts with fixed CPDs).
    """
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    pdg = PDG()
    varlist: List[Var] = []

    # variables
    for i in range(num_vars):
        domain_size = random.randint(*val_range)
        var = Var.alph(name=chr(65 + i), n=domain_size)
        pdg += var
        varlist.append(var)

    # edges
    for _ in range(num_edges):
        src_size = min(random.randint(*src_range), len(varlist))
        src = random.sample(population=varlist, k=src_size)
        remaining_vars = [v for v in varlist if v not in src]
        if not remaining_vars:
            continue
        tgt_size = min(random.randint(*tgt_range), len(remaining_vars))
        if tgt_size == 0:
            continue
        tgt = random.sample(population=remaining_vars, k=tgt_size)

        pdg += CPT.make_random(vfrom=Var.product(src), vto=Var.product(tgt))

    return pdg


def _mask_from_cpd(P):
    """
    Boolean mask of claimed entries (True where finite), or None for 'all claimed'.
    """
    try:
        arr = P.to_numpy() if hasattr(P, "to_numpy") else np.asarray(P)
        return torch.tensor(np.isfinite(arr), dtype=torch.bool)
    except Exception:
        return None


def make_every_cpd_parametric(pdg, init: str = "from_cpd"):
    """
    Replace each edge's CPD with a learnable ParamCPD.
    updates the existing (X, Y, L) key, not a new key .
    """
    edges_snapshot = list(pdg.edges("l,X,Y,α,β,P"))  # freeze view before edits
    for L, X, Y, α, β, P in edges_snapshot:
        learnable = ParamCPD(
            src_var=X,
            tgt_var=Y,
            name=str(L),
            init=init,
            mask=_mask_from_cpd(P),
            cpd=P
        )

        key = (X.name, Y.name, L)
        if key in pdg.edgedata:  # jump over π edges
            pdg.edgedata[key]['cpd'] = learnable
            if L[0] == "π":
                pdg.edgedata[key]['cpd'].logits.requires_grad_(requires_grad=False)  # π edges are not learnable

    return pdg


def make_every_cpd_parametric_projections_fixed(pdg, init: str = "from_cpd"):
    """
    Replace each edge's CPD with a learnable ParamCPD.
    updates the existing (X, Y, L) key, not a new key .
    """
    edges_snapshot = list(pdg.edges("l,X,Y,α,β,P"))  # freeze view before edits
    print(edges_snapshot)

    for L, X, Y, α, β, P in edges_snapshot:
        if L[0] == "π":
            learnable = ParamCPD(
                src_var=X,
                tgt_var=Y,
                name=str(L),
                init="from_cpd",
                mask=_mask_from_cpd(P),
                cpd=P
            )
        else:
            learnable = ParamCPD(
                src_var=X,
                tgt_var=Y,
                name=str(L),
                init=init,
                mask=_mask_from_cpd(P),
                cpd=P
            )

        key = (X.name, Y.name, L)

        if key in pdg.edgedata:
            pdg.edgedata[key]['cpd'] = learnable
            if L[0] == "π":
                pdg.edgedata[key]['cpd'].logits.requires_grad_(requires_grad=False)  # π edges are not learnable

    return pdg


def _collect_learnables(pdg):
    out = []
    for L, P in pdg.edges("l,P"):
        if hasattr(P, "logits") and not (isinstance(L, str) and L.startswith("π")):
            out.append((L, P))
    return out


def demo_refocus(M: PDG, t: int):
    """Return (attn_alpha, attn_beta, control_mask) for step t.
    - Zero-out β for the first learnable edge (drop its contribution).
    - Double β for the second learnable edge (amplify).
    - Freeze the third learnable edge's parameters via control_mask=0.
    Unspecified edges implicitly use scale 1 and control 1.
    """
    learns = _collect_learnables(pdg=M)
    attn_alpha = {}
    attn_beta = {}
    control = {}
    if len(learns) >= 1:
        attn_beta[learns[0][0]] = 0.0
    if len(learns) >= 2:
        attn_beta[learns[1][0]] = 2.0
    if len(learns) >= 3:
        control[learns[2][0]] = 0.0
    return attn_alpha, attn_beta, control


# -----------------------------
# tests
# -----------------------------

def test_lir_on_random_pdg(num_vars=4, num_edges=4, gamma=1.0, seed=0, init="from_cpd"):
    print("=== Testing simple LIR on a random PDG (Standard/Adam) ===")

    pdg = generate_random_pdg(num_vars=num_vars,
                              num_edges=num_edges,
                              val_range=(2, 4),
                              src_range=(1, 2),
                              tgt_range=(1, 1),
                              seed=seed)
    pdg = make_every_cpd_parametric(pdg=pdg, init=init)

    print("\nEdges (label -> X -> Y):")
    for L, X, Y, α, β, P in pdg.edges("l,X,Y,α,β,P"):
        print(f"  {L}: {X.name} -> {Y.name}  |X|={len(X)}  |Y|={len(Y)}  α={α:.2f} β={β:.2f}  param={hasattr(P, 'probs')}")

    mu0 = opt_joint(pdg=pdg, gamma=gamma, iters=25, verbose=False)

    lir_train(
        M=pdg,
        gamma=gamma,
        T=5,
        outer_iters=5,
        inner_iters=5,
        lr=1e-2,
        verbose=True,
        mu_init=mu0,
        outer_backend="standard",
        standard_type="adam",
    )

    mu_star = opt_joint(pdg=pdg, gamma=gamma, iters=350, verbose=False)

    print("\nLearned CPDs:")
    for L, X, Y, α, β, P in pdg.edges("l,X,Y,α,β,P"):
        if hasattr(P, "logits"):
            print(f"  {L}: P({Y.name}|{X.name}) =\n{P.probs().detach().cpu().numpy()}")
        else:
            print(f"  {L}: P({Y.name}|{X.name}) =\n{P}")

    return mu_star, pdg


def test_lir_on_random_pdg_rk4(num_vars=4, num_edges=4, gamma=1.0, seed=0, init="from_cpd"):
    print("=== Testing simple LIR on a random PDG (ODE/Fixed RK4) ===")

    pdg = generate_random_pdg(num_vars=num_vars,
                              num_edges=num_edges,
                              val_range=(2, 4),
                              src_range=(1, 2),
                              tgt_range=(1, 1),
                              seed=seed)
    pdg = make_every_cpd_parametric(pdg=pdg, init=init)

    print("\nEdges (label -> X -> Y):")
    for L, X, Y, α, β, P in pdg.edges("l,X,Y,α,β,P"):
        print(f"  {L}: {X.name} -> {Y.name}  |X|={len(X)}  |Y|={len(Y)}  α={α:.2f} β={β:.2f}  param={hasattr(P, 'probs')}")

    mu0 = opt_joint(pdg=pdg, gamma=gamma, iters=25, verbose=False)

    # RK4 over [0,1] with n_steps=8; smaller lr recommended for ODE-mode sometimes
    lir_train(
        M=pdg,
        gamma=gamma,
        T=5,
        outer_iters=5,
        inner_iters=5,
        lr=1e-2,
        verbose=True,
        mu_init=mu0,
        outer_backend="ode",
        ode_solve_mode="fixed",
        ode_fixed_method="rk4",
        n_steps=8,
        # You can tune these if needed:
        # grad_clip_norm=5.0,
    )

    mu_star = opt_joint(pdg=pdg, gamma=gamma, iters=350, verbose=False)

    print("\nLearned CPDs (RK4):")
    for L, X, Y, α, β, P in pdg.edges("l,X,Y,α,β,P"):
        if hasattr(P, "logits"):
            print(f"  {L}: P({Y.name}|{X.name}) =\n{P.probs().detach().cpu().numpy()}")
        else:
            print(f"  {L}: P({Y.name}|{X.name}) =\n{P}")

    return mu_star, pdg


def test_refocus_masks(num_vars=4, num_edges=5, gamma=0.2, seed=1, init="from_cpd"):
    print("=== Testing refocus() masks (attention + control) ===")
    pdg = generate_random_pdg(num_vars=num_vars,
                              num_edges=num_edges,
                              val_range=(2, 4),
                              src_range=(1, 2),
                              tgt_range=(1, 1),
                              seed=seed)
    pdg = make_every_cpd_parametric_projections_fixed(pdg=pdg, init=init)

    learns = _collect_learnables(pdg=pdg)
    if len(learns) < 2:
        print("Not enough learnable edges to meaningfully test refocus; skipping.")
        return None, pdg

    before = {L: P.logits.detach().clone() for (L, P) in learns}

    mu0 = opt_joint(pdg=pdg, gamma=gamma, iters=25, verbose=False)

    lir_train(
        M=pdg,
        gamma=gamma,
        T=15,
        outer_iters=8,
        inner_iters=20,
        lr=5e-3,
        verbose=True,
        mu_init=mu0,
        refocus=demo_refocus,
        outer_backend="standard",
        standard_type="adam",
    )

    after = {L: P.logits.detach().clone() for (L, P) in learns}

    attn_alpha, attn_beta, control = demo_refocus(M=pdg, t=0)
    frozen_label = next((L for L, _ in learns[2:3]), None)

    if frozen_label is not None:
        changed = (after[frozen_label] - before[frozen_label]).abs().max().item()
        print(f"Frozen edge {frozen_label}: max |Δ| = {changed:.3e} (expected ~0)")
        assert changed < 1e-8, "Control mask failed: frozen edge parameters changed."

    any_changed = False
    for L, _ in learns:
        if L == frozen_label:
            continue
        delta = (after[L] - before[L]).abs().max().item()
        if delta > 1e-6:
            any_changed = True
            break
    print(f"Any unfrozen edge changed? {any_changed}")
    assert any_changed, "No learnable edges changed; attention/control masks may be misapplied."

    print("Refocus mask test passed.")
    return None, pdg


if __name__ == "__main__":

    _mu, _pdg = test_lir_on_random_pdg(init="random", gamma=0.0)  # "from_cpd" or "uniform" or "random"
    print(_mu)
    print(_pdg)

    print("\n--- Running RK4 ODE test ---")
    _mu_rk4, _pdg_rk4 = test_lir_on_random_pdg_rk4(init="random", gamma=0.0)
    print(_mu_rk4)
    print(_pdg_rk4)

    print("\n--- Running refocus mask test ---")
    test_refocus_masks(init="random", gamma=0.0)
