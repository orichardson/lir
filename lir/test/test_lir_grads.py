#!/usr/bin/env python3
"""
Analytic-gradient check for the one-variable PDG.

Problem setup (smallest possible case):
  - A single random variable X with |X| = K.
  - Two learnable CPDs over X with labels "p" and "q" (both unconditional; we
    use Unit->X only to get the right array shapes). The graph stores only β=1
    weights; α is irrelevant here.

Analytic inconsistency for this case:
  Let p, q be the probability vectors parameterized by the two CPDs and let
  r, s > 0 be weights. Define

      f(p, q; r, s) = (r + s) * logsumexp( (r log p + s log q) / (r + s) ).

  The inner optimum μ* (the μ that minimizes the inner objective for fixed p, q)
  is the normalized weighted geometric mean

      μ*(x) ∝ p(x)^{r/(r+s)} q(x)^{s/(r+s)}.

Envelope theorem (control = 1, but μ detached):
  When we differentiate the outer objective with μ fixed at μ*, the gradient w.r.t.
  the parameters of p and q equals the gradient of

      r * KL( μ* || p ) + s * KL( μ* || q ),

  where μ* is treated as a constant during backward(). This test verifies that
  autograd through the library implementation produces the same gradients as the
  analytic f(p, q; r, s) above. In other words,

      ∇_θ f(p, q; r, s)  ==  ∇_θ [ r KL(μ*||p) + s KL(μ*||q) ]  (μ* detached)

We compare the actual PyTorch gradients on ParamCPD logits on both sides and
require exact agreement to numerical precision (atol ≈ 1e-6) for several (r, s).
"""
# run_me_check_f_vs_LIR_grads.py
import pytest
import torch
from pathlib import Path
import sys

# Ensure project dir (containing lir__simpler.py) and this test dir are on sys.path BEFORE imports
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # .../lir
sys.path.insert(0, str(Path(__file__).parent))                # .../lir/test
from helpers_one_var import (  # noqa: E402
    make_one_var_two_cpd_pdg,
    get_pq_from_pdg,
    weighted_geometric_mean,
    f_inconsistency,
    kl_mu_to_p,
)


# ---------- experiment ----------
def run(K=3, r=1.0, s=1.0, seed=0, atol=1e-6):
    """Execute the gradient-equivalence experiment for given K, r, s.

    Steps
      1) Build the PDG and pull out `p` and `q`.
      2) Compute analytic `loss_f = f(p, q; r, s)` and backprop to get grads on
         the ParamCPD logits.
      3) Re-zero grads. Compute μ* = weighted_geometric_mean(p, q, r, s) and
         DETACH it. Backprop `r KL(μ*||p) + s KL(μ*||q)` to get envelope grads.
      4) Compare both gradient tensors entry-wise (max abs diff).
    Returns True iff both p- and q-side gradients match within `atol`.
    """
    pdg, X, key_p, key_q = make_one_var_two_cpd_pdg(K=K, seed=seed)
    cpd_p = pdg.edgedata[key_p]["cpd"]
    cpd_q = pdg.edgedata[key_q]["cpd"]

    # --- grads of f ---
    # zero
    if cpd_p.logits.grad is not None:
        cpd_p.logits.grad.zero_()
    if cpd_q.logits.grad is not None:
        cpd_q.logits.grad.zero_()

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
        # Reference norms only (avoid unused variables)
        print("‖p - μ‖₁:", float((p - mu).abs().sum()))
        print("‖q - μ‖₁:", float((q - mu).abs().sum()))
        print("(Reference sign only)")

    return ok


# ----------------------
# PyTest entry points
# ----------------------
@pytest.mark.parametrize("r,s", [
    (1.0, 1.0),
    (2.0, 1.0),
    (0.7, 1.3),
])
def test_gradients_match_one_var_pdg(r, s):
    assert run(K=3, r=r, s=s, seed=0, atol=1e-6)


# No CLI entry point in pytest files
