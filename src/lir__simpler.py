from __future__ import annotations
from typing import Dict, Optional, Callable, Tuple, Union, List
import torch
import numpy as np
import networkx as nx

from pdg.dist import ParamCPD
from pdg.alg.torch_opt import opt_joint, torch_score
from pdg.pdg import PDG
from pdg.dist import RawJointDist as RJD

# ⬇️ import your new Optimizer (adjust path if needed)
from optimizer import Optimizer


def _collect_learnables(pdg) -> Dict[str, ParamCPD]:
    out: Dict[str, ParamCPD] = {}
    for l, P in pdg.edges("l,P"):
        if isinstance(P, ParamCPD) and P.logits.requires_grad:
            if l in out:
                raise ValueError(f"Duplicate ParamCPD label '{l}' found in PDG.")
            out[l] = P
    return out


@torch.no_grad()
def _detach_mu(mu: torch.Tensor) -> torch.Tensor:
    mu.data = mu.data.detach().clone()
    return mu


def apply_attn_mask(M, attn_mask_beta=None, attn_mask_alpha=None,
                    beta_default=1.0, alpha_default=1.0):
    attn_mask_beta = attn_mask_beta or {}
    attn_mask_alpha = attn_mask_alpha or {}

    M2 = M.copy()
    all_specs = set(attn_mask_beta) | set(attn_mask_alpha)
    for spec in all_specs:
        edgekey = M2._get_edgekey(spec)
        ed = M2.edgedata[edgekey]

        b = attn_mask_beta.get(spec, beta_default)
        a = attn_mask_alpha.get(spec, alpha_default)

        if b == 0 and a == 0:
            del M2[edgekey]
            continue

        ed["beta"] = ed.get("beta", 1.0) * b
        ed["alpha"] = ed.get("alpha", 1.0) * a

    return M2


def pdg_prune_isolated_vars(M: PDG, keep_unit: bool = True) -> PDG:
    M2 = M.copy()
    for vn in list(M2.vars.keys()):
        if keep_unit and vn == "1":
            continue
        if M2.graph.degree(vn) == 0:
            del M2[vn]
    return M2


def pdg_cleanup(M: PDG,
                drop_zero_weight_edges: bool = False,
                zero_tol: float = 0.0,
                keep_unit: bool = True) -> PDG:
    M2 = M.copy()
    if drop_zero_weight_edges:
        for (xn, yn, l), ed in list(M2.edgedata.items()):
            a = ed.get("alpha", 1.0)
            b = ed.get("beta", 1.0)
            if abs(a) <= zero_tol and abs(b) <= zero_tol:
                del M2[(xn, yn, l)]
    M2 = pdg_prune_isolated_vars(M2, keep_unit=keep_unit)
    return M2


def pdg_decompose(M: PDG) -> list[PDG]:
    comps = list(nx.connected_components(M.graph.to_undirected()))
    return [M.subpdg(*C) for C in comps]


def _combine_independent_rdjs(rdjs: list[RJD]) -> RJD:
    seen = set()
    for r in rdjs:
        for v in r.varlist:
            if v.name in seen:
                raise ValueError(f"Variable '{v.name}' appears in multiple components.")
            seen.add(v.name)

    if len(rdjs) == 0:
        raise ValueError("No component distributions to combine.")
    if len(rdjs) == 1:
        return rdjs[0]

    data = rdjs[0].data
    varlist = list(rdjs[0].varlist)
    for r in rdjs[1:]:
        data = np.multiply.outer(data, r.data)
        varlist.extend(r.varlist)

    shape = tuple(len(v) for v in varlist)
    data = data.reshape(shape)
    return RJD(data, varlist)


def decompose_and_infer(M: PDG,
                        inference_fn,
                        *,
                        decompose: bool = True,
                        combine_result: bool = False,
                        cleanup: bool = True,
                        drop_zero_weight_edges: bool = False,
                        zero_tol: float = 0.0,
                        keep_unit: bool = True,
                        inference_kwargs: dict | None = None):
    inference_kwargs = inference_kwargs or {}
    M2 = pdg_cleanup(
        M,
        drop_zero_weight_edges=drop_zero_weight_edges if cleanup else False,
        zero_tol=zero_tol,
        keep_unit=keep_unit,
    ) if cleanup else M

    if not decompose:
        return inference_fn(M2, **inference_kwargs)

    submodels = [S for S in pdg_decompose(M2) if len(S.atomic_vars) > 0]
    results = [inference_fn(S, **inference_kwargs) for S in submodels]

    if combine_result:
        if not all(isinstance(r, RJD) for r in results):
            raise ValueError("combine_result=True requires inference_fn to return RJD per component.")
        return _combine_independent_rdjs(results)

    return results


# ----------------------------------------------------------------------
# Utility: construct per-parameter LR from control_mask
# ----------------------------------------------------------------------
def make_param_lrs(*,
                   M,
                   learnables: Dict[str, ParamCPD],
                   lr: float,
                   control_mask: Optional[Dict] = None,
                   tol: float = 0.0) -> Union[float, List[float]]:
    """
    Build learning rates for each ParamCPD parameter tensor.

    Returns:
        - float if all LR equal (within tol),
        - otherwise list of per-param LRs aligned with learnables.values().
    """
    control_mask = control_mask or {}
    control_by_label: Dict[str, float] = {}
    for spec, ctrl in control_mask.items():
        try:
            label = M._get_edgekey(spec)[2]
        except Exception:
            continue
        control_by_label[label] = float(ctrl)

    labels: List[str] = list(learnables.keys())
    per_param: List[float] = [lr * float(control_by_label.get(lbl, 1.0)) for lbl in labels]

    if len(per_param) == 0:
        return float(lr)

    if tol > 0.0:
        ref = per_param[0]
        if all(abs(v - ref) <= tol for v in per_param[1:]):
            return float(ref)
    else:
        if all(v == per_param[0] for v in per_param[1:]):
            return float(per_param[0])

    return per_param


# ----------------------------------------------------------------------
# LIR core
# ----------------------------------------------------------------------
def lir_step(M,
             *,
             gamma: float = 0.0,
             outer_iters: int = 200,
             inner_iters: int = 300,
             mu_init=None,
             attn_mask_alpha=None,
             attn_mask_beta=None,
             alpha_default: float = 1.0,
             beta_default: float = 1.0,
             control_mask=None,
             # optimizer config
             lr: float = 1e-2,
             outer_backend: str = "standard",
             standard_type: Optional[str] = "adam",
             standard_kwargs: Optional[dict] = None,
             standard_instance: Optional[torch.optim.Optimizer] = None,
             ode_solve_mode: str = "adaptive",
             ode_method: str = "dopri5",
             ode_fixed_method: str = "rk4",
             atol: float = 1e-4,
             rtol: float = 1e-6,
             max_num_steps: Optional[int] = None,
             n_steps: Optional[int] = None,
             grad_clip_norm: Optional[float] = 5.0,
             preconditioner: Optional[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = None,
             **inner_kwargs) -> Tuple[torch.Tensor, torch.Tensor]:

    control_mask = control_mask or {}
    attn_mask_alpha = attn_mask_alpha or {}
    attn_mask_beta = attn_mask_beta or {}

    learnables = _collect_learnables(M)
    if not learnables:
        raise ValueError("No ParamCPDs found in PDG M. Nothing to learn.")

    M2 = apply_attn_mask(M,
                         attn_mask_beta=attn_mask_beta,
                         attn_mask_alpha=attn_mask_alpha,
                         beta_default=beta_default,
                         alpha_default=alpha_default)

    # Build LR list or scalar
    param_lrs = make_param_lrs(M=M, learnables=learnables, lr=lr, control_mask=control_mask)

    params = [P.logits for P in learnables.values()]
    opt = Optimizer(
        params=params,
        backend=outer_backend,
        lr=param_lrs,
        grad_clip_norm=grad_clip_norm,
        preconditioner=preconditioner,
        # ODE
        ode_solve_mode=ode_solve_mode,
        ode_method=ode_method,
        ode_fixed_method=ode_fixed_method,
        atol=atol,
        rtol=rtol,
        max_num_steps=max_num_steps,
        n_steps=n_steps,
        # Standard
        standard_type=standard_type if standard_instance is None else None,
        standard_kwargs=standard_kwargs or {},
        standard_instance=standard_instance,
    )

    last_loss = None
    for _ in range(outer_iters):

        def warm_start_init(shape, dtype=torch.double):
            if mu_init is not None:
                return mu_init.data.clone().to(dtype=dtype)
            else:
                return torch.ones(size=shape, dtype=dtype)

        def loss_closure() -> torch.Tensor:
            nonlocal mu_init
            mu_star = opt_joint(M2, gamma=gamma, iters=inner_iters,
                                verbose=False, init=warm_start_init, **inner_kwargs)
            mu_star = _detach_mu(mu_star)
            mu_init = mu_star.data.detach().clone()
            return torch_score(M2, mu_star, gamma)

        loss_val = opt.step(loss_closure=loss_closure)
        last_loss = torch.as_tensor(loss_val, dtype=torch.float64)

    return mu_init, last_loss


def lir_train(M,
              *,
              T: int = 100,
              outer_iters: int = 200,
              inner_iters: int = 300,
              gamma: float = 0.0,
              lr: float = 1e-2,
              verbose: bool = False,
              mu_init=None,
              refocus=None,
              outer_backend: str = "standard",
              standard_type: Optional[str] = "adam",
              standard_kwargs: Optional[dict] = None,
              standard_instance: Optional[torch.optim.Optimizer] = None,
              ode_solve_mode: str = "adaptive",
              ode_method: str = "dopri5",
              ode_fixed_method: str = "rk4",
              atol: float = 1e-4,
              rtol: float = 1e-6,
              max_num_steps: Optional[int] = None,
              n_steps: Optional[int] = None,
              grad_clip_norm: Optional[float] = 5.0,
              preconditioner: Optional[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = None,
              **inner_kwargs):

    learnables = _collect_learnables(M)
    if not learnables:
        raise ValueError("No ParamCPDs found in PDG M. Nothing to learn.")

    last = None
    for t in range(T):
        if refocus is None:
            attn_a, attn_b, ctrl = {}, {}, {}
        else:
            attn_a, attn_b, ctrl = refocus(M, t)

        step_lr = lr / float(outer_iters)
        mu_init, loss = lir_step(M,
                                 gamma=gamma,
                                 outer_iters=outer_iters,
                                 inner_iters=inner_iters,
                                 mu_init=mu_init,
                                 attn_mask_alpha=attn_a,
                                 attn_mask_beta=attn_b,
                                 control_mask=ctrl,
                                 lr=step_lr,
                                 outer_backend=outer_backend,
                                 standard_type=standard_type,
                                 standard_kwargs=standard_kwargs,
                                 standard_instance=standard_instance,
                                 ode_solve_mode=ode_solve_mode,
                                 ode_method=ode_method,
                                 ode_fixed_method=ode_fixed_method,
                                 atol=atol,
                                 rtol=rtol,
                                 max_num_steps=max_num_steps,
                                 n_steps=n_steps,
                                 grad_clip_norm=grad_clip_norm,
                                 preconditioner=preconditioner,
                                 **inner_kwargs)

        if verbose and (t % max(1, T // 10) == 0):
            val = float(loss.detach().cpu()) if loss is not None else float("nan")
            delta = "" if last is None else f"  Δ={val - last:+.3e}"
            print(f"[LIR {t:4d}/{T}]  γ={gamma:.3g}  loss={val:.6e}{delta}")
            last = val

    return M
