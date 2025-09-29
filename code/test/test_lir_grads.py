#!/usr/bin/env python3
# run_me_check_f_vs_LIR_grads.py
from pathlib import Path
import sys
import argparse
import torch
import torch.nn.functional as F

# If needed, uncomment this to add your repo root to PYTHONPATH
# sys.path.append(str(Path(__file__).resolve().parents[1]))

from pdg.pdg import PDG
from pdg.rv import Variable as Var, Unit  # JointStructure not needed here
from pdg.dist import CPT, ParamCPD


# ---------- math helpers ----------

def weighted_geometric_mean(p: torch.Tensor, q: torch.Tensor, r: float, s: float, eps: float = 1e-12):
    """
    μ*(x) ∝ p(x)^{α} q(x)^{1-α}, α = r/(r+s).
    Computed stably in log-space and returned normalized.
    """
    α = r / (r + s)
    a = α * torch.log(p + eps) + (1 - α) * torch.log(q + eps)   # log of unnormalized μ
    logZ = torch.logsumexp(a, dim=-1)
    mu = torch.exp(a - logZ)
    return mu  # shape [K]

def f_inconsistency(p: torch.Tensor, q: torch.Tensor, r: float, s: float, eps: float = 1e-12):
    """
    f(p,q,r,s) = (r+s) * log sum_x (p(x)^r q(x)^s)^{1/(r+s)}
               = (r+s) * logsumexp( (r*log p + s*log q)/(r+s) )
    """
    a = (r * torch.log(p + eps) + s * torch.log(q + eps)) / (r + s)
    logZ = torch.logsumexp(a, dim=-1)
    return (r + s) * logZ


def kl_mu_to_p(mu: torch.Tensor, p: torch.Tensor, eps: float = 1e-12):
    """
    KL(mu || p) = sum_x mu(x) * (log mu(x) - log p(x))
    NOTE: mu is treated as a constant during backward when we detach it.
    """
    return (mu * (torch.log(mu + eps) - torch.log(p + eps))).sum()


# ---------- PDG (your simple setting) ----------

def make_one_var_two_cpd_pdg(K: int = 4, seed: int = 0):
    torch.manual_seed(seed)
    X = Var.alph("X", K)
    pdg = PDG() + X

    # make shapes via Unit->X (unconditional table shape)
    P_p = CPT.make_random(Unit, X)
    P_q = CPT.make_random(Unit, X)

    # keep your simple wiring: ParamCPD(src=X, tgt=X) and labels "p"/"q"
    cpd_p = ParamCPD(src_var=X, tgt_var=X, name="p", init="random", mask=None, cpd=P_p)
    cpd_q = ParamCPD(src_var=X, tgt_var=X, name="q", init="random", mask=None, cpd=P_q)

    key_p = (X.name, X.name, "p")
    key_q = (X.name, X.name, "q")
    pdg.edgedata[key_p] = {"cpd": cpd_p, "β": 1.0}
    pdg.edgedata[key_q] = {"cpd": cpd_q, "β": 1.0}
    return pdg, X, key_p, key_q


def get_pq_from_pdg(pdg, key_p, key_q):
    """
    Matches your own access pattern: probs()[0].view(-1)
    """
    p = pdg.edgedata[key_p]["cpd"].probs()[0].view(-1)  # shape [K]
    q = pdg.edgedata[key_q]["cpd"].probs()[0].view(-1)
    return p, q


# ---------- experiment ----------

def run(K=3, r=1.0, s=1.0, seed=0, atol=1e-6):
    pdg, X, key_p, key_q = make_one_var_two_cpd_pdg(K=K, seed=seed)
    cpd_p = pdg.edgedata[key_p]["cpd"]
    cpd_q = pdg.edgedata[key_q]["cpd"]

    # --- grads of f ---
    # zero
    if cpd_p.logits.grad is not None: cpd_p.logits.grad.zero_()
    if cpd_q.logits.grad is not None: cpd_q.logits.grad.zero_()

    p, q = get_pq_from_pdg(pdg, key_p, key_q)
    loss_f = f_inconsistency(p, q, r, s)      # (r+s) * logsumexp(...)
    loss_f.backward()

    grads_f_p = cpd_p.logits.grad.detach().clone()
    grads_f_q = cpd_q.logits.grad.detach().clone()

    # --- grads from LIR outer objective with control = 0 ---
    # (re-zero)
    cpd_p.logits.grad.zero_()
    cpd_q.logits.grad.zero_()

    # recompute p, q for a clean graph (they still depend on logits)
    p, q = get_pq_from_pdg(pdg, key_p, key_q)
    mu = weighted_geometric_mean(p.detach(), q.detach(), r, s)  # detach: envelope theorem (control=0)

    loss_lir = r * kl_mu_to_p(mu, p) + s * kl_mu_to_p(mu, q)
    loss_lir.backward()

    grads_lir_p = cpd_p.logits.grad.detach().clone()
    grads_lir_q = cpd_q.logits.grad.detach().clone()

    # --- compare ---
    diff_p = (grads_f_p - grads_lir_p).abs().max().item()
    diff_q = (grads_f_q - grads_lir_q).abs().max().item()

    print("loss_f:", float(loss_f))
    print("loss_lir:", float(loss_lir))
    print("max |grad_p(f) - grad_p(LIR)|:", diff_p)
    print("max |grad_q(f) - grad_q(LIR)|:", diff_q)

    ok = (diff_p <= atol) and (diff_q <= atol)
    print("GRADIENT MATCH:", "OK" if ok else "FAIL")

    # Optionally also show the analytic form expected sign:
    # For reference: ∇_{θ_p} f = - r * (p - μ), ∇_{θ_q} f = - s * (q - μ)
    # (because f is +(r+s) logZ; if you used - (r+s) logZ, the sign flips)
    with torch.no_grad():
        # map per-row logits -> per-row probs gradient using your layout:
        # We only check norms here since ParamCPD may arrange logits along rows/cols internally.
        gp_expected = -r * (p - mu)   # shape [K]
        gq_expected = -s * (q - mu)   # shape [K]
        print("‖p - μ‖₁:", float((p - mu).abs().sum()))
        print("‖q - μ‖₁:", float((q - mu).abs().sum()))
        print("(Reference sign only)")

    return ok


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Check that ParamCPD grads after LIR equal grads of f.")
    parser.add_argument("--K", type=int, default=3)
    parser.add_argument("--r", type=float, default=1.0)
    parser.add_argument("--s", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--atol", type=float, default=1e-6)
    args = parser.parse_args()

    ok = run(K=args.K, r=args.r, s=args.s, seed=args.seed, atol=args.atol)
    sys.exit(0 if ok else 1)
