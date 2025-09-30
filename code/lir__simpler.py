# param_cpd.py
from __future__ import annotations
from typing import Dict
import torch
from pdg.dist import ParamCPD
from pdg.alg.torch_opt import opt_joint, torch_score


def _collect_learnables(pdg) -> Dict[str, ParamCPD]:
    out = {}
    for l, P in pdg.edges("l,P"):
        if isinstance(P, ParamCPD) and P.logits.requires_grad:
            if l in out:
                raise ValueError(f"Duplicate ParamCPD label '{l}' found in PDG.")
            out[l] = P
    return out


@torch.no_grad()
def _detach_mu(mu):
    mu.data = mu.data.detach().clone()
    return mu


def apply_attn_mask(M, attn_mask_beta=None, attn_mask_alpha=None,
                    beta_default=1.0, alpha_default=1.0):
    """
    Return a copy of `M` with per-edge weights scaled.

    Args:
        attn_mask_beta  (dict): edge_spec → β scale
        attn_mask_alpha (dict): edge_spec → α scale
        beta_default    (float): fallback β scale for specs present in a mask but missing in that mask
        alpha_default   (float): fallback α scale (same convention)

    Notes:
        - `edge_spec` may be any format accepted by `PDG._get_edgekey`.
        - If both α and β scale to 0 for an edge, the edge is removed.
    """
    attn_mask_beta  = attn_mask_beta  or {}
    attn_mask_alpha = attn_mask_alpha or {}

    M2 = M.copy()

    # operate on the union of all referenced specs
    all_specs = set(attn_mask_beta) | set(attn_mask_alpha)
    for spec in all_specs:
        edgekey = M2._get_edgekey(spec)            # (src, tgt, label)
        ed = M2.edgedata[edgekey]

        b = attn_mask_beta.get(spec,  beta_default)
        a = attn_mask_alpha.get(spec, alpha_default)

        if b == 0 and a == 0:
            del M2[edgekey]                        # remove edge cleanly
            continue

        # ensure fields exist, then scale
        ed['beta']  = ed.get('beta',  1.0) * b
        ed['alpha'] = ed.get('alpha', 1.0) * a

    return M2


def lir_step(
    M,                             # PDG containing ParamCPDs (learnable θ)
    optimizer,                     # torch optimizer over θ (created once by caller)
    gamma: float = 0.0,            # LIR γ (inner problem)
    outer_iters: int = 200,        # θ-update iterations within this step
    inner_iters: int = 300,        # μ*-solve iterations (opt_joint)
    mu_init = None,                # initial μ for warm-starting inner solve
    attn_mask_alpha = None,        # edge_spec → α scale
    attn_mask_beta  = None,        # edge_spec → β scale
    alpha_default: float = 1.0,    # default α scale for unspecified edges
    beta_default: float = 1.0,     # default β scale for unspecified edges
    control_mask = None,           # edge_spec → gradient scale
    **inner_kwargs                 # extra kwargs for opt_joint
):
    """
    Perform one LIR step:
      repeat `outer_iters` times:
        μ*  = argmin_μ  OInc_γ(M_masked(θ); μ)        # inner solve (opt_joint)
        θ  ← θ - η ∇_θ OInc_γ(M_masked(θ); μ*)        # envelope theorem: treat μ* constant
    Returns:
        (mu_init_next, last_loss)
    """
    control_mask = control_mask or {}
    attn_mask_alpha = attn_mask_alpha or {}
    attn_mask_beta = attn_mask_beta or {}

    learnables = _collect_learnables(M)  # map: label -> ParamCPD
    if not learnables:
        raise ValueError("No ParamCPDs found in PDG M. Nothing to learn.")

    # Apply attention mask once per θ-iteration block (masks may be static within lir_step)
    M2 = apply_attn_mask(
        M=M,
        attn_mask_beta=attn_mask_beta,
        attn_mask_alpha=attn_mask_alpha,
        beta_default=beta_default,
        alpha_default=alpha_default
    )

    last_loss = None
    for _ in range(outer_iters):

        def warm_start_init(shape, dtype=torch.double):
            if mu_init is not None:
                return mu_init.data.clone().to(dtype)
            else:
                return torch.ones(shape, dtype=dtype)

        # inner solve for μ*, given current θ
        μ_star = opt_joint(M2, gamma=gamma, iters=inner_iters, verbose=False, init=warm_start_init, **inner_kwargs)
        μ_star = _detach_mu(μ_star)
        mu_init = μ_star.data.detach().clone()

        # outer gradient on θ only
        optimizer.zero_grad(set_to_none=True)
        loss = torch_score(M2, μ_star, gamma)
        loss.backward()

        # apply control mask to gradients
        for spec, ctrl in control_mask.items():
            l = M._get_edgekey(spec)[2]  # edge label
            P = learnables.get(l)
            if P is not None and P.logits.grad is not None:
                P.logits.grad *= ctrl

        optimizer.step()
        last_loss = loss

    return mu_init, last_loss


def lir_train(
    M,                                    # PDG containing ParamCPDs (learnable θ)
    T: int = 100,                         # number of outer refocus steps
    outer_iters: int = 200,               # θ-iterations per step
    inner_iters: int = 300,               # μ*-solve iterations (opt_joint)
    gamma: float = 0.0,                   # LIR γ (inner problem)
    lr: float = 1e-2,                     # θ learning rate
    optimizer_ctor = torch.optim.Adam,    # optimizer constructor for θ-update
    opt_kwargs = None,                    # kwargs for optimizer_ctor
    verbose: bool = False,                # print progress
    mu_init = None,                       # warm-start μ for first step
    refocus = None,                       # function M,t -> (attn_mask_alpha, attn_mask_beta, control_mask)
    **inner_kwargs                        # forwarded to opt_joint
):
    """
    LIR training loop: calls `refocus` to get (attn_mask_alpha, attn_mask_beta, control_mask),
    then performs a `lir_step`. Parameters θ are updated in-place.
    """
    opt_kwargs = opt_kwargs or {}

    learnables = _collect_learnables(M)
    if not learnables:
        raise ValueError("No ParamCPDs found in PDG M. Nothing to learn.")

    # Create optimizer
    opt = optimizer_ctor([P.logits for P in learnables.values()], lr=lr / outer_iters, **opt_kwargs)

    last = None
    for t in range(T):
        # Obtain masks/control for this step
        if refocus is None:
            attn_a, attn_b, ctrl = {}, {}, {}
        else:
            attn_a, attn_b, ctrl = refocus(M, t)

        mu_init, loss = lir_step(
            M=M,
            optimizer=opt,
            gamma=gamma,
            outer_iters=outer_iters,
            inner_iters=inner_iters,
            mu_init=mu_init,
            attn_mask_alpha=attn_a,
            attn_mask_beta=attn_b,
            control_mask=ctrl,
            **inner_kwargs
        )

        if verbose and (t % max(1, T // 10) == 0):
            val = float(loss.detach().cpu()) if loss is not None else float("nan")
            delta = "" if last is None else f"  Δ={val - last:+.3e}"
            print(f"[LIR {t:4d}/{T}]  γ={gamma:.3g}  loss={val:.6e}{delta}")
            last = val

    return M  # parameters are updated in-place
