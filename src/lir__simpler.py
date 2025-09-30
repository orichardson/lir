# param_cpd.py
from __future__ import annotations
from typing import Dict
import torch
from pdg.dist import ParamCPD
from pdg.alg.torch_opt import opt_joint, torch_score
import numpy as np
import networkx as nx
from pdg.pdg import PDG
from pdg.dist import RawJointDist as RJD


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


def pdg_prune_isolated_vars(M: PDG, keep_unit: bool = True) -> PDG:
    """
    Return a new PDG with all variables of degree 0 removed.

    Why:
        - After masking/removing edges, some variables may become isolated (no incident edges).
          Those variables needlessly inflate tensor shapes and slow inference.

    Behavior:
        - Only variables with degree 0 in `M.graph` are removed.
        - If `keep_unit=True`, the distinguished unit variable named "1" is preserved.
          This variable has cardinality 1 and typically does not affect atomic varlists.

    Args:
        M (PDG): Input graph.
        keep_unit (bool): Preserve the special unit variable "1".

    Returns:
        PDG: A (shallow) copy of `M` with isolated variables removed.
    """
    M2 = M.copy()
    for vn in list(M2.vars.keys()):
        if keep_unit and vn == "1":
            continue
        if M2.graph.degree(vn) == 0:
            # This removes the node from both `vars` and `graph` via PDG.__delitem__
            del M2[vn]
    return M2


def pdg_cleanup(
    M: PDG,
    drop_zero_weight_edges: bool = False,
    zero_tol: float = 0.0,
    keep_unit: bool = True,
) -> PDG:
    """
    Clean up a PDG after masking/removal of edges.

    Steps:
        1) Optionally drop edges whose α and β are both (near-)zero.
           - Controlled by `drop_zero_weight_edges` (default False = exact behavior preserved).
           - An edge is dropped if abs(alpha) <= zero_tol AND abs(beta) <= zero_tol.
           - Missing α/β are treated as default 1.0 (do NOT drop).
        2) Remove isolated variables (degree-0) using `pdg_prune_isolated_vars`.

    Args:
        M (PDG): Input graph.
        drop_zero_weight_edges (bool): If True, drop (near-)zero-weight edges (approximate).
        zero_tol (float): Tolerance for deciding "near zero".
        keep_unit (bool): Preserve the "1" unit variable.

    Returns:
        PDG: Cleaned PDG.
    """
    M2 = M.copy()

    if drop_zero_weight_edges:
        for (xn, yn, l), ed in list(M2.edgedata.items()):
            a = ed.get('alpha', 1.0)
            b = ed.get('beta', 1.0)
            if abs(a) <= zero_tol and abs(b) <= zero_tol:
                del M2[(xn, yn, l)]

    # Always prune isolated variables (safe, exact)
    M2 = pdg_prune_isolated_vars(M2, keep_unit=keep_unit)
    return M2


def pdg_decompose(M: PDG) -> list[PDG]:
    """
    Split a PDG into PDGs corresponding to connected components.

    Why:
        - Inference cost scales with graph size/treewidth.
          Decomposing lets you solve smaller problems independently.

    Returns:
        list[PDG]: One PDG per connected component in the undirected version of `M.graph`.
    """
    comps = list(nx.connected_components(M.graph.to_undirected()))
    return [M.subpdg(*C) for C in comps]


def _combine_independent_rdjs(rdjs: list[RJD]) -> RJD:
    """
    Combine independent component distributions into a single joint via tensor product.

    Preconditions:
        - Components must have disjoint variable names.
        - Each input is a valid RJD over its own component's varlist.

    Returns:
        RJD: The joint distribution over the concatenated varlist.

    Raises:
        ValueError: If a variable name appears in more than one component.
    """
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
        # Outer product combines independent factors; shape is the concatenation of dimensions.
        data = np.multiply.outer(data, r.data)
        varlist.extend(r.varlist)

    # Ensure data has the correct shape for the concatenated varlist.
    shape = tuple(len(v) for v in varlist)
    data = data.reshape(shape)
    return RJD(data, varlist)


def decompose_and_infer(
    M: PDG,
    inference_fn,
    *,
    decompose: bool = True,
    combine_result: bool = False,
    cleanup: bool = True,
    drop_zero_weight_edges: bool = False,
    zero_tol: float = 0.0,
    keep_unit: bool = True,
    inference_kwargs: dict | None = None,
):
    """
    Optionally clean up and decompose a PDG, run an inference routine per component, and
    optionally combine the independent results.

    Typical usage:
        - For exact semantics:
            result = decompose_and_infer(M, my_infer, decompose=True, combine_result=False)
          returns a list of per-component results.
        - To get a global joint when components are disconnected and your infer returns RJD:
            μ = decompose_and_infer(M, my_infer, decompose=True, combine_result=True)

    Args:
        M (PDG): Model graph.
        inference_fn (callable): A function `fn(sub_M: PDG, **kwargs) -> Any`.
                                 If `combine_result=True`, it must return `RJD`.
        decompose (bool): If True, run per-component; otherwise, run on `M` as-is.
        combine_result (bool): If True, combine per-component `RJD`s into one `RJD`.
        cleanup (bool): If True, run `pdg_cleanup` first (with exact defaults unless you opt-in).
        drop_zero_weight_edges (bool): If True, drop (near-)zero α/β edges (approximate).
        zero_tol (float): Tolerance for near-zero edge dropping.
        keep_unit (bool): Preserve "1" unit variable during cleanup.
        inference_kwargs (dict|None): Extra kwargs passed to `inference_fn`.

    Returns:
        - If decompose=False: returns whatever `inference_fn(M, **kwargs)` returns.
        - If decompose=True and combine_result=False: returns a list of per-component results.
        - If decompose=True and combine_result=True: returns an `RJD` over the union of variables.

    Raises:
        ValueError: If `combine_result=True` but `inference_fn` does not return `RJD`.
    """
    inference_kwargs = inference_kwargs or {}

    M2 = pdg_cleanup(
        M,
        drop_zero_weight_edges=drop_zero_weight_edges if cleanup else False,
        zero_tol=zero_tol,
        keep_unit=keep_unit,
    ) if cleanup else M

    if not decompose:
        return inference_fn(M2, **inference_kwargs)

    # Skip components that contain no atomic variables (i.e., only the unit variable).
    # Such components contribute a multiplicative factor of 1 and do not affect inference.
    submodels = [S for S in pdg_decompose(M2) if len(S.atomic_vars) > 0]
    results = [inference_fn(S, **inference_kwargs) for S in submodels]

    if combine_result:
        # All results must be RJD to combine.
        if not all(isinstance(r, RJD) for r in results):
            raise ValueError("combine_result=True requires inference_fn to return RJD per component.")
        return _combine_independent_rdjs(results)

    return results

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
