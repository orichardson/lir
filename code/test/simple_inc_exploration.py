#!/usr/bin/env python3
# test_lir_one_var_simple.py
from pathlib import Path
import sys
import argparse

import torch
import torch.nn.functional as F  # not strictly needed, kept for parity with your imports
from pdg.pdg import PDG
from pdg.rv import Variable as Var, Unit, JointStructure  # JointStructure unused, kept for parity
from pdg.dist import CPT, ParamCPD
from lir__simpler import lir_train_simple
from pdg.alg.torch_opt import opt_joint


def make_one_var_two_cpd_pdg(K: int = 4, seed: int = 0):
    torch.manual_seed(seed)
    X = Var.alph("X", K)
    pdg = PDG() + X

    # Two CPDs over X; use Unit as the "source" just to get an unconditional table shape
    P_p = CPT.make_random(Unit, X)
    P_q = CPT.make_random(Unit, X)

    # Replace with ParamCPD; keep your simple setting: src_var=X, tgt_var=X, labels "p"/"q"
    cpd_p = ParamCPD(src_var=X, tgt_var=X, name="p", init="random", mask=None, cpd=P_p)
    cpd_q = ParamCPD(src_var=X, tgt_var=X, name="q", init="random", mask=None, cpd=P_q)

    key_p = (X.name, X.name, "p")
    key_q = (X.name, X.name, "q")

    # Your code sets only β; keep that
    pdg.edgedata[key_p] = {"cpd": cpd_p, "β": 1.0}
    pdg.edgedata[key_q] = {"cpd": cpd_q, "β": 1.0}

    return pdg, X, key_p, key_q


def normalized_geometric_mean(p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    mu = torch.sqrt(p * q)
    return mu / mu.sum()


def run(K: int, T: int, inner: int, lr: float, gamma: float, seed: int, verbose: bool, anomaly: bool):
    if anomaly:
        torch.autograd.set_detect_anomaly(True)

    pdg, X, key_p, key_q = make_one_var_two_cpd_pdg(K=K, seed=seed)

    # Initial distributions (your code uses probs()[0] then view(-1))
    with torch.no_grad():
        p0 = pdg.edgedata[key_p]["cpd"].probs()[0].detach().view(-1)
        q0 = pdg.edgedata[key_q]["cpd"].probs()[0].detach().view(-1)
        mu0 = normalized_geometric_mean(p0, q0)

    print("\n=== Initial ===")
    print("p0:", p0.tolist())
    print("q0:", q0.tolist())
    print("mu0 (geomean):", mu0.tolist())

    # Warm-start μ and train (same calls you had)
    mu0_joint = opt_joint(pdg, gamma=gamma, iters=10, verbose=False)
    lir_train_simple(
        M=pdg,
        gamma=gamma,
        T=T,
        inner_iters=inner,
        lr=lr,
        optimizer_ctor=torch.optim.Adam,
        verbose=verbose,
        mu_init=mu0_joint,
    )

    with torch.no_grad():
        pT = pdg.edgedata[key_p]["cpd"].probs()[0].view(-1)
        qT = pdg.edgedata[key_q]["cpd"].probs()[0].view(-1)

    print("\n=== Final ===")
    print("pT:", pT.tolist())
    print("qT:", qT.tolist())

    # Same checks as your test, but just print PASS/FAIL instead of pytest asserts
    ok_agree = torch.allclose(pT, qT, atol=3e-3, rtol=0)
    ok_match = torch.allclose(pT, mu0, atol=1e-2, rtol=0)

    print("\n=== Checks ===")
    print(f"(a) pT ≈ qT (atol=3e-3): {'PASS' if ok_agree else 'FAIL'}")
    print(f"(b) pT ≈ mu0 (atol=1e-2): {'PASS' if ok_match else 'FAIL'}")

    if not (ok_agree and ok_match):
        # non-zero exit makes it easy to spot failures in shell
        sys.exit(1)

    print("\nAll checks passed ✅")


def parse_args():
    ap = argparse.ArgumentParser(description="Run simple LIR debug on one-var PDG (your minimal setting).")
    ap.add_argument("--K", type=int, default=3, help="Cardinality of X")
    ap.add_argument("--T", type=int, default=60, help="Outer steps")
    ap.add_argument("--inner", type=int, default=30, help="Inner opt_joint iterations per outer step")
    ap.add_argument("--lr", type=float, default=5e-2, help="Adam learning rate")
    ap.add_argument("--gamma", type=float, default=0.0, help="opt_joint gamma")
    ap.add_argument("--seed", type=int, default=0, help="Random seed")
    ap.add_argument("--verbose", action="store_true", help="Verbose outer-loop logging")
    ap.add_argument("--anomaly", action="store_true", help="Enable torch autograd anomaly detection")
    return ap.parse_args()


def main():
    args = parse_args()
    run(
        K=args.K,
        T=args.T,
        inner=args.inner,
        lr=args.lr,
        gamma=args.gamma,
        seed=args.seed,
        verbose=args.verbose,
        anomaly=args.anomaly,
    )


if __name__ == "__main__":
    main()
