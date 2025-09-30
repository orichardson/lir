#!/usr/bin/env python3
# one-var exploration/demo
from pathlib import Path
import sys
import argparse

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from pdg.alg.torch_opt import opt_joint
from lir__simpler import lir_train
from .helpers_one_var import make_one_var_two_cpd_pdg, normalized_geometric_mean


def run(K: int, T: int, inner: int, lr: float, gamma: float, seed: int, verbose: bool):
    torch.autograd.set_detect_anomaly(True)

    pdg, X, key_p, key_q = make_one_var_two_cpd_pdg(K=K, seed=seed)

    # Initial distributions
    with torch.no_grad():
        p0 = pdg.edgedata[key_p]["cpd"].probs()[0].detach().view(-1)
        q0 = pdg.edgedata[key_q]["cpd"].probs()[0].detach().view(-1)
        mu0 = normalized_geometric_mean(p0, q0)

    print("\n=== Initial ===")
    print("p0:", p0.tolist())
    print("q0:", q0.tolist())
    print("mu0 (geomean):", mu0.tolist())

    # Warm-start μ and train
    mu0_joint = opt_joint(pdg, gamma=gamma, iters=10, verbose=False)
    lir_train(
        M=pdg,
        gamma=gamma,
        T=T,
        outer_iters=1,
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

    # Print PASS/FAIL for quick manual runs
    ok_agree = torch.allclose(pT, qT, atol=3e-3, rtol=0)
    ok_match = torch.allclose(pT, mu0, atol=1e-2, rtol=0)
    print(f"(a) pT ≈ qT (atol=3e-3): {'PASS' if ok_agree else 'FAIL'}")
    print(f"(b) pT ≈ mu0 (atol=1e-2): {'PASS' if ok_match else 'FAIL'}")
    if ok_agree and ok_match:
        print("\nAll checks passed ✅")


def parse_args():
    ap = argparse.ArgumentParser(description="Run one-var LIR demo (de-duplicated)")
    ap.add_argument("--K", type=int, default=3, help="Cardinality of X")
    ap.add_argument("--T", type=int, default=60, help="Outer steps")
    ap.add_argument("--inner", type=int, default=30, help="Inner opt_joint iterations per outer step")
    ap.add_argument("--lr", type=float, default=5e-2, help="Adam learning rate")
    ap.add_argument("--gamma", type=float, default=0.0, help="opt_joint gamma")
    ap.add_argument("--seed", type=int, default=0, help="Random seed")
    ap.add_argument("--verbose", action="store_true", help="Verbose outer-loop logging")
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
    )


if __name__ == "__main__":
    main()
