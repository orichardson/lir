"""
Multi-replicate LIR experiments with fixed graph structure and re-randomized CPDs.

Re-runs the style of analysis behind:
  - TV distortion accumulated through the LIR refocus schedule
  - Initial vs. final inconsistency (torch_score at mu*) for uniform / partial / hub refocus

Usage (from repository root):

  PYTHONPATH=src python3 -m experiments.lir_multiseed_refocus --n-replicates 50

Use --n-replicates 100 or 1000 for a heavier run. Outputs go to --out-dir (default results/lir_refocus).
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import torch

# package root = directory containing lir__simpler.py
_SRC = Path(__file__).resolve().parents[1]
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from pdg.dist import CPT, ParamCPD
from pdg.pdg import PDG
from pdg.alg.torch_opt import opt_joint, torch_score

from lir__simpler import lir_step


def generate_random_pdg(
    num_vars: int = 4,
    num_edges: int = 5,
    val_range=(2, 4),
    src_range=(1, 2),
    tgt_range=(1, 1),
    seed: int = 0,
) -> PDG:
    """Random PDG structure and CPDs (same recipe as src/testing_lir_simple.py)."""
    from pdg.rv import Variable as Var

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    pdg = PDG()
    varlist: List = []

    for i in range(num_vars):
        domain_size = random.randint(*val_range)
        var = Var.alph(name=chr(65 + i), n=domain_size)
        pdg += var
        varlist.append(var)

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
    try:
        arr = P.to_numpy() if hasattr(P, "to_numpy") else np.asarray(P)
        return torch.tensor(np.isfinite(arr), dtype=torch.bool)
    except Exception:
        return None


def make_every_cpd_parametric(pdg: PDG, init: str = "from_cpd") -> PDG:
    edges_snapshot = list(pdg.edges("l,X,Y,α,β,P"))
    for L, X, Y, α, β, P in edges_snapshot:
        learnable = ParamCPD(
            src_var=X,
            tgt_var=Y,
            name=str(L),
            init=init,
            mask=_mask_from_cpd(P),
            cpd=P,
        )
        key = (X.name, Y.name, L)
        if key in pdg.edgedata:
            pdg.edgedata[key]["cpd"] = learnable
            if isinstance(L, str) and L.startswith("π"):
                pdg.edgedata[key]["cpd"].logits.requires_grad_(requires_grad=False)
    return pdg


def resample_cpds_same_structure(template: PDG, seed: int) -> PDG:
    """Deep-enough copy with new random CPTs on every edge (structure and labels unchanged)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    M = template.copy()
    for L, X, Y, α, β, P in M.edges("l,X,Y,α,β,P"):
        if isinstance(L, str) and L.startswith("π"):
            continue
        key = (X.name, Y.name, L)
        if key not in M.edgedata:
            continue
        M.edgedata[key]["cpd"] = CPT.make_random(X, Y)
    return M


def learnable_param_cpds(pdg: PDG) -> List[Tuple[object, ParamCPD]]:
    out: List[Tuple[object, ParamCPD]] = []
    for L, P in pdg.edges("l,P"):
        if isinstance(P, ParamCPD) and P.logits.requires_grad:
            if isinstance(L, str) and L.startswith("π"):
                continue
            out.append((L, P))
    return out


def snapshot_probs(pdg: PDG) -> Dict[object, torch.Tensor]:
    return {L: P.probs().detach().clone() for L, P in learnable_param_cpds(pdg)}


def total_tv_from_reference(pdg: PDG, ref: Dict[object, torch.Tensor]) -> float:
    tot = torch.tensor(0.0, dtype=torch.double)
    for L, P in learnable_param_cpds(pdg):
        p = P.probs().detach()
        r = ref[L]
        tot = tot + 0.5 * (p - r).abs().sum()
    return float(tot.item())


def hub_node_name(M: PDG) -> Optional[str]:
    best, deg = None, -1
    for n in M.graph.nodes():
        if n == "1":
            continue
        d = M.graph.degree(n)
        if d > deg:
            deg, best = d, n
    return best


def edge_touches_atomic_name(X, Y, name: str) -> bool:
    return any(a.name == name for a in X.atoms) or any(a.name == name for a in Y.atoms)


def refocus_uniform(_M: PDG, _t: int) -> Tuple[dict, dict, dict]:
    return {}, {}, {}


def refocus_partial_factory(rng: random.Random, frac: float = 0.5) -> Callable[[PDG, int], Tuple[dict, dict, dict]]:
    def _partial(M: PDG, t: int) -> Tuple[dict, dict, dict]:
        learns = [L for L, _ in learnable_param_cpds(M)]
        if not learns:
            return {}, {}, {}
        k = max(1, int(round(frac * len(learns))))
        rng_t = random.Random(rng.randint(0, 2**30) + t * 1_000_003)
        chosen = set(rng_t.sample(learns, k=min(k, len(learns))))
        ctrl = {L: (1.0 if L in chosen else 0.0) for L in learns}
        return {}, {}, ctrl

    return _partial


def refocus_hub(M: PDG, _t: int) -> Tuple[dict, dict, dict]:
    hub = hub_node_name(M)
    learns = learnable_param_cpds(M)
    if hub is None or not learns:
        return {}, {}, {}
    ctrl = {}
    for L, X, Y, α, β, P in M.edges("l,X,Y,α,β,P"):
        if not isinstance(P, ParamCPD) or not P.logits.requires_grad:
            continue
        if isinstance(L, str) and L.startswith("π"):
            continue
        ctrl[L] = 1.0 if edge_touches_atomic_name(X, Y, hub) else 0.0
    return {}, {}, ctrl


STRATEGIES = {
    "uniform": refocus_uniform,
    "partial": None,
    "hub": refocus_hub,
}


def inconsistency_at_mu(M: PDG, gamma: float, mu_init: Optional[torch.Tensor], inner_iters: int) -> Tuple[float, torch.Tensor]:
    def warm_start_init(shape, dtype=torch.double):
        if mu_init is not None:
            return mu_init.clone().to(dtype=dtype)
        return torch.ones(size=shape, dtype=dtype)

    mu = opt_joint(M, gamma=gamma, iters=inner_iters, verbose=False, init=warm_start_init)
    inc = float(torch_score(M, mu, gamma).detach().cpu())
    return inc, mu.data.detach().clone()


def run_lir_with_trace(
    M: PDG,
    *,
    gamma: float,
    T: int,
    outer_iters: int,
    inner_iters: int,
    lr: float,
    refocus: Callable[[PDG, int], Tuple[dict, dict, dict]],
    inner_warmup: int,
    inner_final: int,
) -> Tuple[float, float, List[float], torch.Tensor]:
    """
    Returns:
      inc_init, inc_final, tv_cumulative_trace (length T+1), mu_final_data
    """
    mu_init = None
    inc_init, mu_tensor = inconsistency_at_mu(M, gamma, mu_init, inner_warmup)
    mu_init = mu_tensor

    P0 = snapshot_probs(M)
    tv_trace = [total_tv_from_reference(M, P0)]

    last_mu = mu_init
    for t in range(T):
        attn_a, attn_b, ctrl = refocus(M, t)
        step_lr = lr / float(outer_iters)
        last_mu, _loss = lir_step(
            M,
            gamma=gamma,
            outer_iters=outer_iters,
            inner_iters=inner_iters,
            mu_init=last_mu,
            attn_mask_alpha=attn_a,
            attn_mask_beta=attn_b,
            control_mask=ctrl,
            lr=step_lr,
            outer_backend="standard",
            standard_type="adam",
        )
        tv_trace.append(total_tv_from_reference(M, P0))

    inc_final, mu_f = inconsistency_at_mu(M, gamma, last_mu, inner_final)
    return inc_init, inc_final, tv_trace, mu_f


def build_structure_templates(
    n_pdgs: int,
    *,
    num_vars: int,
    num_edges: int,
    structure_seed_base: int,
) -> List[PDG]:
    out = []
    for i in range(n_pdgs):
        s = structure_seed_base + i * 10_000
        out.append(generate_random_pdg(num_vars=num_vars, num_edges=num_edges, seed=s))
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--n-replicates", type=int, default=100, help="CPD re-randomizations per structure (100–1000 typical).")
    p.add_argument("--n-pdgs", type=int, default=4, help="Number of distinct fixed structures.")
    p.add_argument("--structure-seed-base", type=int, default=0)
    p.add_argument("--num-vars", type=int, default=4)
    p.add_argument("--num-edges", type=int, default=5)
    p.add_argument("--gamma", type=float, default=0.2)
    p.add_argument("--T", type=int, default=15, help="Refocus steps.")
    p.add_argument("--outer-iters", type=int, default=8)
    p.add_argument("--inner-iters", type=int, default=25)
    p.add_argument("--inner-warmup", type=int, default=35)
    p.add_argument("--inner-final", type=int, default=120)
    p.add_argument("--lr", type=float, default=5e-3)
    p.add_argument("--cpd-seed-base", type=int, default=1_000_000, help="Offsets replicate index for CPD RNG.")
    p.add_argument("--out-dir", type=Path, default=Path("results/lir_refocus"))
    p.add_argument("--no-plots", action="store_true")
    p.add_argument(
        "--partial-frac",
        type=float,
        default=0.5,
        help="Fraction of learnable edges with control=1 on each partial-refocus step.",
    )
    args = p.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    templates = build_structure_templates(
        args.n_pdgs,
        num_vars=args.num_vars,
        num_edges=args.num_edges,
        structure_seed_base=args.structure_seed_base,
    )

    traj_rows: List[dict] = []
    summary_rows: List[dict] = []

    for pdg_id, template in enumerate(templates):
        for rep in range(args.n_replicates):
            cpd_seed = args.cpd_seed_base + rep + pdg_id * 1_000_003
            probe = make_every_cpd_parametric(resample_cpds_same_structure(template, cpd_seed), init="from_cpd")
            if not learnable_param_cpds(probe):
                continue

            partial_ref = refocus_partial_factory(random.Random(cpd_seed), frac=args.partial_frac)

            for strat_name in ("uniform", "partial", "hub"):
                M = make_every_cpd_parametric(resample_cpds_same_structure(template, cpd_seed), init="from_cpd")
                if strat_name == "uniform":
                    rf = STRATEGIES["uniform"]
                elif strat_name == "partial":
                    rf = partial_ref
                else:
                    rf = STRATEGIES["hub"]

                inc_i, inc_f, tv_tr, _mu = run_lir_with_trace(
                    M,
                    gamma=args.gamma,
                    T=args.T,
                    outer_iters=args.outer_iters,
                    inner_iters=args.inner_iters,
                    lr=args.lr,
                    refocus=rf,
                    inner_warmup=args.inner_warmup,
                    inner_final=args.inner_final,
                )

                summary_rows.append(
                    {
                        "pdg_id": pdg_id,
                        "replicate": rep,
                        "strategy": strat_name,
                        "inc_init": inc_i,
                        "inc_final": inc_f,
                        "tv_final": tv_tr[-1],
                        "gamma": args.gamma,
                    }
                )
                for step, tv in enumerate(tv_tr):
                    traj_rows.append(
                        {
                            "pdg_id": pdg_id,
                            "replicate": rep,
                            "strategy": strat_name,
                            "step": step,
                            "tv_cumulative": tv,
                        }
                    )

    import pandas as pd

    df_sum = pd.DataFrame(summary_rows)
    df_traj = pd.DataFrame(traj_rows)
    sum_path = args.out_dir / "summary.csv"
    traj_path = args.out_dir / "tv_trajectories.csv"
    df_sum.to_csv(sum_path, index=False)
    df_traj.to_csv(traj_path, index=False)
    print(f"Wrote {sum_path} ({len(df_sum)} rows) and {traj_path} ({len(df_traj)} rows).")

    if args.no_plots or df_sum.empty:
        return

    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed; skip figures.")
        return

    # Figure: TV distortion through LIR (mean cumulative TV vs step).
    fig1, axes1 = plt.subplots(1, args.n_pdgs, figsize=(4 * args.n_pdgs, 3.5), squeeze=False)
    for ax, pid in zip(axes1[0], range(args.n_pdgs)):
        sub = df_traj[df_traj["pdg_id"] == pid]
        if sub.empty:
            continue
        for strat in ("uniform", "partial", "hub"):
            s = sub[sub["strategy"] == strat]
            if s.empty:
                continue
            g = s.groupby("step")["tv_cumulative"].agg(["mean", "std"]).reset_index()
            ax.plot(g["step"], g["mean"], label=strat)
            ax.fill_between(
                g["step"],
                g["mean"] - g["std"],
                g["mean"] + g["std"],
                alpha=0.15,
            )
        ax.set_title(f"PDG {pid}")
        ax.set_xlabel("Refocus step")
        ax.set_ylabel("Mean TV from initial θ (± std)")
        ax.legend(fontsize=8)
    fig1.suptitle("TV distortion through LIR (cumulative from initial CPDs)")
    fig1.tight_layout()
    f1 = args.out_dir / "fig_tv_distortion_lir.png"
    fig1.savefig(f1, dpi=150)
    plt.close(fig1)
    print(f"Wrote {f1}")

    # Figure: initial vs final inconsistency.
    fig2, axes2 = plt.subplots(1, args.n_pdgs, figsize=(4 * args.n_pdgs, 3.8), squeeze=False)
    for ax, pid in zip(axes2[0], range(args.n_pdgs)):
        sub = df_sum[df_sum["pdg_id"] == pid]
        if sub.empty:
            continue
        for strat in ("uniform", "partial", "hub"):
            s = sub[sub["strategy"] == strat]
            if s.empty:
                continue
            ax.scatter(s["inc_init"], s["inc_final"], s=12, alpha=0.35, label=strat)
        lo = min(sub["inc_init"].min(), sub["inc_final"].min())
        hi = max(sub["inc_init"].max(), sub["inc_final"].max())
        if lo < hi:
            ax.plot([lo, hi], [lo, hi], "k--", alpha=0.3, linewidth=1)
        ax.set_title(f"PDG {pid}")
        ax.set_xlabel("Initial inconsistency")
        ax.set_ylabel("Final inconsistency")
        ax.legend(fontsize=8)
    fig2.suptitle("Initial vs final inconsistency (uniform / partial / hub)")
    fig2.tight_layout()
    f2 = args.out_dir / "fig_inconsistency_initial_vs_final.png"
    fig2.savefig(f2, dpi=150)
    plt.close(fig2)
    print(f"Wrote {f2}")


if __name__ == "__main__":
    main()
