# param_cpd.py
from __future__ import annotations
from typing import Tuple
import torch
from typing import List, Dict, Any
from pdg.dist import ParamCPD


# ________________________________________________________________________________________
# ________________________________________________________________________________________

from pdg.alg.torch_opt import opt_joint, torch_score

def _collect_learnables(pdg) -> List[Tuple[str, ParamCPD]]:
    """
    Collects all learnable ParamCPD objects from the edges of a PDG in order to add them to an optimizer.
    """
    out = []
    for l, P in pdg.edges("l,P"):
        if isinstance(P, ParamCPD):
            out.append((l, P))
    return out
    

@torch.no_grad()
def _detach_mu(mu):
    mu.data = mu.data.detach().clone()
    return mu



# def random_refocus(views) :
#     v = random.choice(views)
#     return v.get("phi", {}), v.get("chi", {}), float(v.get("gamma", 0.0))

def refocus_ID(views):
    return views[0].get("phi", {}), views[0].get("chi", {}), float(views[0].get("gamma", 0.0))


def lir_train_simple_Attn_ctrl(
    M,                          # PDG containing ParamCPDs (learnable θ)
    gamma: float = 0.0,
    T: int = 200,               # outer steps
    inner_iters: int = 300,     # μ*-solve iterations (opt_joint)
    lr: float = 1e-2,
    optimizer_ctor = torch.optim.Adam,
    verbose: bool = False,
    mu_init = None,
    views = None,
    REFOCUS = refocus_ID,
    **inner_kwargs              
):
    """
    Simplified LIR:
      repeat T times:
        μ*  = argmin_μ  OInc_γ(M(θ); μ)        # inner solve (uses your opt_joint)
        θ  ← θ - η ∇_θ OInc_γ(M(θ); μ*)        # envelope theorem: μ* is treated constant
    """

    if views is None:
        learned = {L for L, P in _collect_learnables(M)}
        
        # Default view: attend to all arcs, control only learned CPDs.
        all_phi = {L: 1.0 for L, *_ in M.edges("l,X,Y,P")}
        all_chi = {L: 1.0 if L in learned else 0.0 for L in all_phi}
        views = [dict(phi=all_phi, chi=all_chi, gamma=(gamma if gamma is not None else getattr(M, "gamma_default", 0.0)))]

    learnables = _collect_learnables(M)
    if not learnables:
        raise ValueError("No ParamCPDs found in PDG M. Nothing to learn.")

    opt = optimizer_ctor([P.logits for (_, P) in learnables], lr=lr)
    last = None
    
    mu_init = mu_init
    for t in range(T):

        # # (1) refresh context
        # _ = REFRESH(_)
        
        # (2) refocus
        refocused = REFOCUS(views)
        phi, chi, gamma = refocused # refocused["phi"], refocused["chi"], refocused["gamma"]
        
        #If refocus not working yet use this:
        # phi, chi, gamma = all_phi, all_chi, gamma 


        def warm_start_init(shape, dtype=torch.double):
            if mu_init is not None:
                return mu_init.data.clone().to(dtype)
            else:
                return torch.ones(shape, dtype=dtype)
            
        attended = _AttnPDG(M, phi)
        # inner solve for μ*, given current θ
        μ_star = opt_joint(M, gamma=gamma, iters=inner_iters, verbose=False, init=warm_start_init, **inner_kwargs)
        μ_star = _detach_mu(μ_star)
        mu_init = μ_star.data.detach().clone()

        # outer gradient on θ only
        opt.zero_grad(set_to_none=True)
        loss = torch_score(M, μ_star, gamma)
        loss.backward()

       # (6) Scale per-edge gradients by chi 
        for L, P in learnables:
            if P.logits.grad is not None:
                P.logits.grad.mul_(float(chi.get(L, 0.0)))  # zero == freeze

        opt.step()

        if verbose and (t % max(1, T // 10) == 0):
            val = float(loss.detach().cpu())
            delta = "" if last is None else f"  Δ={val - last:+.3e}"
            print(f"[LIR {t:4d}/{T}]  γ={gamma:.3g}  loss={val:.6e}{delta}")
            last = val

    return M  # parameters are updated in-place

class _AttnPDG:
    """
    View over a PDG that scales edge weights by φ[L] (both α and β).
    φ can be float or (φ_β, φ_α) pair. Missing -> 1.0.
    """
    def __init__(self, pdg, phi: Dict[Any, float | Tuple[float,float]]):
        self._pdg = pdg
        self._phi = phi or {}


    # 
    def edges(self, spec: str):
        for e in self._pdg.edges("l,X,Y,α,β,P"):
            L, X, Y, α, β, P = e
            scale = self._phi.get(L, 1.0)
            if isinstance(scale, tuple):
                sβ, sα = scale
            else:
                sβ = sα = float(scale)
            # yield in whatever spec the caller requested
            fmt = spec #.replace(" ", ",")
            parts = {
                "l": L, "X": X, "Y": Y, "α": sα * α, "β": sβ * β, "P": P
            }
            yield tuple(parts[k] for k in fmt)#.split(","))
    
    def __getattr__(self, name):
        return getattr(self._pdg, name)


def lir_train_simple(
    M,                          # PDG containing ParamCPDs (learnable θ)
    gamma: float = 0.0,
    T: int = 200,               # outer steps
    inner_iters: int = 300,     # μ*-solve iterations (opt_joint)
    lr: float = 1e-2,
    optimizer_ctor = torch.optim.Adam,
    verbose: bool = False,
    mu_init = None,
    **inner_kwargs              
):
    """
    Simplified LIR:
      repeat T times:
        μ*  = argmin_μ  OInc_γ(M(θ); μ)        # inner solve (uses your opt_joint)
        θ  ← θ - η ∇_θ OInc_γ(M(θ); μ*)        # envelope theorem: μ* is treated constant
    """
    learnables = _collect_learnables(M)
    if not learnables:
        raise ValueError("No ParamCPDs found in PDG M. Nothing to learn.")

    opt = optimizer_ctor([P.logits for (_, P) in learnables], lr=lr)
    last = None
    
    mu_init = mu_init
    for t in range(T):

        def warm_start_init(shape, dtype=torch.double):
            if mu_init is not None:
                return mu_init.data.clone().to(dtype)
            else:
                return torch.ones(shape, dtype=dtype)
        # inner solve for μ*, given current θ
        μ_star = opt_joint(M, gamma=gamma, iters=inner_iters, verbose=False, init=warm_start_init, **inner_kwargs)
        μ_star = _detach_mu(μ_star)
        mu_init = μ_star.data.detach().clone()

        # outer gradient on θ only
        opt.zero_grad(set_to_none=True)
        loss = torch_score(M, μ_star, gamma)
        loss.backward()
        opt.step()

        if verbose and (t % max(1, T // 10) == 0):
            val = float(loss.detach().cpu())
            delta = "" if last is None else f"  Δ={val - last:+.3e}"
            print(f"[LIR {t:4d}/{T}]  γ={gamma:.3g}  loss={val:.6e}{delta}")
            last = val

    return M  # parameters are updated in-place


