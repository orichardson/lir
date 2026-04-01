#!/usr/bin/env python3
"""
Run global, local, and node-based attention strategies on all PDGs and visualize results.

Supports many CPD re-draws for the same graph structure (--n-cpd-replicates): each replicate
resamples CPDs on a fixed template PDG, then runs uniform / partial / hub. 



  - fig_tv_theta_lir_process.png — mean cumulative TV of learned CPDs vs initial θ over refocus steps
  - fig_inconsistency_initial_vs_final.png — initial vs final inconsistency (one point per replicate, by PDG)
  - fig_inconsistency_initial_vs_final_bars.png — mean initial vs final inconsistency (grouped bars by PDG and strategy; legend upper-left)
  - strategy_resolution_visualization.png — full dashboard; also split into strategy_resolution_panel_inconsistency.png, _avg_resolution.png, _heatmap.png, _per_pdg_grid.png.
  
Use --n-pdgs to scale the number of distinct fixed structures (default 12; was 4 in early runs).
Each structure k uses num_vars=4+k, num_edges=3+k, RNG seed 104+k (same family as chain_4v_3e...).
"""

import argparse
import json
import math
import re
import sys
from pathlib import Path
import random
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Tuple, Callable, Optional
import pandas as pd

# Add project paths
sys.path.insert(0, str(Path(__file__).resolve().parents[0]))

from pdg.pdg import PDG
from pdg.rv import Variable as Var
from pdg.dist import CPT, ParamCPD
from pdg.alg.torch_opt import opt_joint, torch_score
from lir__simpler import lir_step

# Match the original lir_train(...) settings in this script
LIR_T = 20
LIR_OUTER_ITERS = 10
LIR_INNER_ITERS = 20
LIR_LR = 0.05
LIR_GAMMA = 0.0
OPT_JOINT_ITERS = 50


def _var_to_json(v) -> Dict:
    return {
        "name": v.name,
        "domain": [str(x) for x in getattr(v, "ordered", list(v))],
    }


def _pdg_structure_to_json(pdg: PDG) -> Dict:
    vars_json = [_var_to_json(v) for v in pdg.vars.values() if v.name != "1"]
    edges_json = []
    for L, X, Y, α, β, _P in pdg.edges("l,X,Y,α,β,P"):
        if X.name == "1" or Y.name == "1":
            continue
        edges_json.append(
            {
                "label": str(L),
                "src": X.name,
                "src_atoms": [a.name for a in X.atoms],
                "tgt": Y.name,
                "tgt_atoms": [a.name for a in Y.atoms],
                "alpha": float(α),
                "beta": float(β),
            }
        )
    return {"variables": vars_json, "edges": edges_json}


def _cpds_to_json(pdg: PDG) -> List[Dict]:
    cpds = []
    for L, X, Y, _α, _β, P in pdg.edges("l,X,Y,α,β,P"):
        if X.name == "1" or Y.name == "1":
            continue
        arr = P.to_numpy() if hasattr(P, "to_numpy") else np.asarray(P)
        cpds.append(
            {
                "label": str(L),
                "src": X.name,
                "tgt": Y.name,
                "shape": list(arr.shape),
                "values": np.asarray(arr).tolist(),
            }
        )
    return cpds


def _write_json_file_atomic(path: Path, payload: Dict) -> None:
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    tmp_path.replace(path)


def _replicate_file_path(out_dir: Path, pdg_name: str, rep: int) -> Path:
    pdg_dir = out_dir / "intermediate" / pdg_name
    pdg_dir.mkdir(parents=True, exist_ok=True)
    return pdg_dir / f"rep_{rep:04d}.json"


def _write_json_checkpoint(
    out_path: Path,
    *,
    args: argparse.Namespace,
    results: List[Dict],
    pdg_structures: Dict[str, Dict],
    intermediate_files: List[str],
    complete: bool,
) -> None:
    """Atomically write a JSON checkpoint with all results collected so far."""
    payload = {
        "complete": complete,
        "n_results": len(results),
        "config": {
            "n_pdgs": args.n_pdgs,
            "pdg_start_index": args.pdg_start_index,
            "n_cpd_replicates": args.n_cpd_replicates,
            "cpd_seed_offset": args.cpd_seed_offset,
            "out_dir": str(args.out_dir),
        },
        "pdg_structures": pdg_structures,
        "intermediate_files": intermediate_files,
        "results": results,
    }
    _write_json_file_atomic(out_path, payload)


def generate_pdg(num_vars: int, num_edges: int, seed: int) -> PDG:
    """Generate a PDG - same as in the experiment."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    pdg = PDG()
    varlist = []
    
    # Create variables
    for i in range(num_vars):
        domain_size = random.randint(2, 3)
        var = Var.alph(chr(65 + i), domain_size)
        pdg += var
        varlist.append(var)
    
    edges_added = 0
    edge_pairs = set()
    target_edges = {}
    
    # First, create a basic chain structure
    base_chain_edges = min(num_vars - 1, max(2, num_edges // 2))
    for i in range(base_chain_edges):
        if i + 1 < len(varlist):
            src = varlist[i]
            tgt = varlist[i + 1]
            
            try:
                pdg += CPT.make_random(Var.product([src]), Var.product([tgt]))
                edges_added += 1
                edge_pairs.add((i, i + 1))
                
                if tgt not in target_edges:
                    target_edges[tgt] = []
                target_edges[tgt].append(src)
            except Exception:
                continue
    
    # Add additional edges to create conflicts
    max_attempts = num_edges * 10
    attempts = 0
    
    while edges_added < num_edges and attempts < max_attempts:
        attempts += 1
        
        if target_edges and random.random() > 0.3:
            tgt = random.choice(list(target_edges.keys()))
            tgt_idx = varlist.index(tgt)
            
            src_idx = random.randint(0, num_vars - 1)
            if src_idx == tgt_idx or (src_idx, tgt_idx) in edge_pairs:
                continue
                
            src = varlist[src_idx]
        else:
            src_idx = random.randint(0, num_vars - 1)
            tgt_idx = random.randint(0, num_vars - 1)
            
            if src_idx == tgt_idx or (src_idx, tgt_idx) in edge_pairs:
                continue
            
            src = varlist[src_idx]
            tgt = varlist[tgt_idx]
        
        try:
            pdg += CPT.make_random(Var.product([src]), Var.product([tgt]))
            edges_added += 1
            edge_pairs.add((src_idx, tgt_idx))
            
            if tgt not in target_edges:
                target_edges[tgt] = []
            target_edges[tgt].append(src)
        except Exception:
            continue
    
    return pdg


def build_pdg_configs(n_pdgs: int, start_index: int = 0) -> List[Tuple[int, int, str, int]]:
    """
    Chain-style family used across this repo: for k = 0..n_pdgs-1,
    num_vars = 4+k, num_edges = 3+k, structure RNG seed = 104+k.
    Reproduces the original four PDGs when n_pdgs == 4.
    """
    if n_pdgs < 1:
        raise ValueError("n_pdgs must be >= 1")
    if start_index < 0:
        raise ValueError("start_index must be >= 0")
    configs: List[Tuple[int, int, str, int]] = []
    for k in range(start_index, start_index + n_pdgs):
        nv, ne = 4 + k, 3 + k
        configs.append((nv, ne, f"chain_{nv}v_{ne}e", 104 + k))
    return configs


def pdg_name_sort_key(name: str):
    m = re.match(r"chain_(\d+)v_(\d+)e", str(name))
    if m:
        return (int(m.group(1)), int(m.group(2)))
    return (0, 0, name)


def _figure_axes_for_pdgs(
    pdgs: List[str], *, max_cols: int = 6, figsize_per_cell: Tuple[float, float] = (3.6, 3.2)
):
    """Grid of axes for one panel per PDG; hide unused cells when N is large."""
    n = len(pdgs)
    if n == 0:
        fig, ax = plt.subplots(1, 1, figsize=(4, 3))
        ax.axis("off")
        return fig, []
    nc = min(max_cols, n)
    nr = int(math.ceil(n / nc))
    fw, fh = figsize_per_cell
    fig, axes = plt.subplots(nr, nc, figsize=(fw * nc, fh * nr), squeeze=False)
    flat = axes.ravel()
    for j in range(n, len(flat)):
        flat[j].axis("off")
    return fig, list(zip(flat[:n], pdgs))


def global_strategy(pdg: PDG, t: int) -> Tuple[Dict, Dict, Dict]:
    """Global: β = 1 for all edges."""
    attn_alpha = {}
    attn_beta = {}
    control = {}
    
    for L, X, Y, α, β, P in pdg.edges("l,X,Y,α,β,P"):
        if X.name != "1" and Y.name != "1":
            attn_alpha[L] = 0.0
            attn_beta[L] = 1.0
            control[L] = 1.0
    
    return attn_alpha, attn_beta, control


def local_strategy(pdg: PDG, t: int) -> Tuple[Dict, Dict, Dict]:
    """Local: β = 1 for half edges, β = 0 for others."""
    attn_alpha = {}
    attn_beta = {}
    control = {}
    
    edges = [(L, X, Y, α, β, P) for L, X, Y, α, β, P in pdg.edges("l,X,Y,α,β,P") 
            if X.name != "1" and Y.name != "1"]
    
    if not edges:
        return attn_alpha, attn_beta, control
    
    num_active = max(1, len(edges) // 2)
    active_edges = random.sample(edges, num_active)
    active_labels = {L for L, X, Y, α, β, P in active_edges}
    
    for L, X, Y, α, β, P in edges:
        attn_alpha[L] = 0.0
        if L in active_labels:
            attn_beta[L] = 1.0
            control[L] = 1.0
        else:
            attn_beta[L] = 0.0
            control[L] = 0.0
    
    return attn_alpha, attn_beta, control


def node_based_strategy(pdg: PDG, t: int) -> Tuple[Dict, Dict, Dict]:
    """Node-based: β = 1 for edges connected to a random node, β = 0.1 for others."""
    attn_alpha = {}
    attn_beta = {}
    control = {}
    
    varlist = [v for v in pdg.vars.values() if v.name != "1"]
    if not varlist:
        return attn_alpha, attn_beta, control
    
    focus_node = random.choice(varlist)
    
    for L, X, Y, α, β, P in pdg.edges("l,X,Y,α,β,P"):
        if X.name != "1" and Y.name != "1":
            attn_alpha[L] = 0.0
            # Use 0.1 for non-focused edges to maintain gradient flow
            attn_beta[L] = 1.0 if (X.name == focus_node.name or Y.name == focus_node.name) else 0.1
            control[L] = 1.0
    
    return attn_alpha, attn_beta, control


def resample_cpds_same_structure(template: PDG, seed: int) -> PDG:
    """Copy template and draw fresh random CPTs on every edge (structure and labels unchanged)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    M = template.copy()
    for L, X, Y, α, β, P in M.edges("l,X,Y,α,β,P"):
        if X.name == "1" or Y.name == "1":
            continue
        if str(L)[0] == "π":
            continue
        key = (X.name, Y.name, L)
        if key in M.edgedata:
            M.edgedata[key]["cpd"] = CPT.make_random(X, Y)
    return M


def _learnable_param_cpds(pdg: PDG) -> List[Tuple[object, ParamCPD]]:
    out: List[Tuple[object, ParamCPD]] = []
    for L, X, Y, α, β, P in pdg.edges("l,X,Y,α,β,P"):
        if X.name == "1" or Y.name == "1":
            continue
        if str(L)[0] == "π":
            continue
        if isinstance(P, ParamCPD) and P.logits.requires_grad:
            out.append((L, P))
    return out


def _snapshot_theta_probs(pdg: PDG) -> Dict[object, torch.Tensor]:
    return {L: P.probs().detach().clone() for L, P in _learnable_param_cpds(pdg)}


def _total_tv_theta_vs_ref(pdg: PDG, ref: Dict[object, torch.Tensor]) -> float:
    tot = 0.0
    for L, P in _learnable_param_cpds(pdg):
        tot += float(0.5 * (P.probs().detach() - ref[L]).abs().sum().item())
    return tot


def _parametrize_edges(pdg_copy: PDG) -> None:
    edges_snapshot = list(pdg_copy.edges("l,X,Y,α,β,P"))
    for L, X, Y, α, β, P in edges_snapshot:
        if L[0] != "π" and X.name != "1" and Y.name != "1":
            learnable = ParamCPD(
                src_var=X,
                tgt_var=Y,
                name=str(L),
                init="from_cpd",
                mask=None,
                cpd=P,
            )
            key = (X.name, Y.name, L)
            if key in pdg_copy.edgedata:
                pdg_copy.edgedata[key]["cpd"] = learnable


def run_strategy(
    pdg: PDG,
    pdg_name: str,
    strategy_name: str,
    strategy_func: Callable,
    *,
    replicate: int = 0,
    rng_seed: Optional[int] = None,
) -> Dict:
    """Run a single strategy; record cumulative θ–TV after each refocus step (LIR process)."""
    if rng_seed is not None:
        random.seed(rng_seed)

    try:
        pdg_copy = PDG.copy(pdg)
        _parametrize_edges(pdg_copy)

        if not _learnable_param_cpds(pdg_copy):
            raise RuntimeError("no learnable ParamCPDs")

        mu_state = opt_joint(pdg_copy, gamma=LIR_GAMMA, iters=OPT_JOINT_ITERS, verbose=False)
        initial_inconsistency = float(torch_score(pdg_copy, mu_state, LIR_GAMMA))

        mu_init_array = (
            mu_state.data.detach().cpu().numpy()
            if hasattr(mu_state.data, "detach")
            else mu_state.data
        )

        theta_ref = _snapshot_theta_probs(pdg_copy)
        tv_theta_trace = [_total_tv_theta_vs_ref(pdg_copy, theta_ref)]

        # lir_step warm-start: first value may be RJD (has .data); later steps use plain tensors
        last_mu = mu_state
        for t in range(LIR_T):
            attn_a, attn_b, ctrl = strategy_func(pdg_copy, t)
            step_lr = LIR_LR / float(LIR_OUTER_ITERS)
            last_mu, _loss = lir_step(
                pdg_copy,
                gamma=LIR_GAMMA,
                outer_iters=LIR_OUTER_ITERS,
                inner_iters=LIR_INNER_ITERS,
                mu_init=last_mu,
                attn_mask_alpha=attn_a,
                attn_mask_beta=attn_b,
                control_mask=ctrl,
                lr=step_lr,
                outer_backend="standard",
                standard_type="adam",
            )
            tv_theta_trace.append(_total_tv_theta_vs_ref(pdg_copy, theta_ref))

        mu_tensor = (
            last_mu.data.detach().clone()
            if hasattr(last_mu, "data")
            else last_mu.detach().clone()
        )

        def _final_warm(shape, dtype=torch.double):
            return mu_tensor.clone().to(dtype=dtype)

        mu_final = opt_joint(
            pdg_copy,
            gamma=LIR_GAMMA,
            iters=OPT_JOINT_ITERS,
            verbose=False,
            init=_final_warm,
        )
        final_inconsistency = float(torch_score(pdg_copy, mu_final, LIR_GAMMA))

        mu_final_array = (
            mu_final.data.detach().cpu().numpy()
            if hasattr(mu_final.data, "detach")
            else mu_final.data
        )

        distortion_tv = float(0.5 * np.sum(np.abs(mu_init_array - mu_final_array)))

        resolution_pct = (
            (initial_inconsistency - final_inconsistency) / initial_inconsistency * 100
            if initial_inconsistency > 0
            else 0
        )

        return {
            "pdg_name": pdg_name,
            "strategy": strategy_name,
            "replicate": replicate,
            "initial": initial_inconsistency,
            "final": final_inconsistency,
            "resolution_pct": resolution_pct,
            "distortion_tv": distortion_tv,
            "tv_theta_trace": tv_theta_trace,
            "success": True,
        }

    except Exception as e:
        print(f"  ⚠️  Error with {strategy_name}: {str(e)}")
        return {
            "pdg_name": pdg_name,
            "strategy": strategy_name,
            "replicate": replicate,
            "initial": 0.0,
            "final": 0.0,
            "resolution_pct": 0.0,
            "distortion_tv": 0.0,
            "tv_theta_trace": [],
            "success": False,
            "error": str(e),
        }


STRATEGY_VIZ_COLORS = {"uniform": "#2E86AB", "partial": "#A23B72", "hub": "#F18F01"}


def _draw_strategy_panel_inconsistency(
    ax,
    df_mean: pd.DataFrame,
    pdgs: List[str],
    strategies,
    colors: Dict[str, str],
) -> None:
    x = np.arange(len(pdgs))
    width = 0.15
    n_pdgs = len(pdgs)
    for i, strategy in enumerate(strategies):
        strategy_data = df_mean[df_mean["strategy"] == strategy]
        initial_vals = [
            strategy_data[strategy_data["pdg_name"] == pdg]["initial"].values[0]
            if len(strategy_data[strategy_data["pdg_name"] == pdg]) > 0
            else 0
            for pdg in pdgs
        ]
        final_vals = [
            strategy_data[strategy_data["pdg_name"] == pdg]["final"].values[0]
            if len(strategy_data[strategy_data["pdg_name"] == pdg]) > 0
            else 0
            for pdg in pdgs
        ]
        offset = (i - len(strategies) / 2 + 0.5) * width
        ax.bar(
            x + offset,
            initial_vals,
            width,
            label=f"{strategy} (initial)",
            color=colors.get(strategy, "gray"),
            alpha=0.3,
            edgecolor="black",
            linewidth=0.5,
        )
        ax.bar(
            x + offset,
            final_vals,
            width,
            label=f"{strategy} (final)",
            color=colors.get(strategy, "gray"),
            alpha=1.0,
            edgecolor="black",
            linewidth=1.5,
        )
    ax.set_xlabel("PDG", fontsize=12, fontweight="normal")
    ax.set_ylabel("Inconsistency", fontsize=12, fontweight="normal")
    ax.set_xticks(x)
    rot = 35 if n_pdgs > 6 else (20 if n_pdgs > 4 else 0)
    ax.set_xticklabels(pdgs, rotation=rot, ha="right" if rot else "center")
    ax.legend(
        ncol=2,
        fontsize=9,
        loc="upper left",
        bbox_to_anchor=(0.02, 0.98),
        bbox_transform=ax.transAxes,
        borderaxespad=0.0,
        framealpha=0.95,
    )
    ax.grid(True, alpha=0.3, axis="y")


def _draw_strategy_panel_avg_resolution(
    ax, df: pd.DataFrame, colors: Dict[str, str]
) -> None:
    strategy_means = df.groupby("strategy")["resolution_pct"].mean().sort_values(ascending=False)
    x_max = float(strategy_means.values.max()) if len(strategy_means) else 100.0
    label_offset = max(2.0, 0.045 * x_max)
    bars = ax.barh(
        strategy_means.index,
        strategy_means.values,
        color=[colors.get(s, "gray") for s in strategy_means.index],
        edgecolor="black",
        linewidth=1.5,
    )
    ax.set_xlabel("Resolution (%)", fontsize=12, fontweight="normal")
    ax.grid(True, alpha=0.3, axis="x")
    for bar in bars:
        w = bar.get_width()
        ax.text(
            w + label_offset,
            bar.get_y() + bar.get_height() / 2,
            f"{w:.1f}%",
            ha="left",
            va="center",
            fontweight="bold",
            fontsize=10,
        )
    ax.set_xlim(0, x_max + 5.5 * label_offset)


def _draw_strategy_panel_heatmap(
    ax,
    df_mean: pd.DataFrame,
    pdgs: List[str],
    n_pdgs: int,
) -> None:
    pivot_data = df_mean.pivot(index="strategy", columns="pdg_name", values="resolution_pct")
    pivot_data = pivot_data.reindex(columns=pdgs)
    ann = n_pdgs <= 12
    akws = {"fontsize": 7} if n_pdgs > 8 else {"fontsize": 9}
    hm_kw = dict(
        cmap="RdYlGn",
        cbar_kws={"label": "Resolution (%)"},
        ax=ax,
        vmin=0,
        vmax=100,
        linewidths=1,
        linecolor="black",
    )
    if ann:
        hm_kw["annot"] = True
        hm_kw["fmt"] = ".1f"
        hm_kw["annot_kws"] = akws
    sns.heatmap(pivot_data, **hm_kw)
    ax.set_xlabel("PDG", fontsize=12, fontweight="normal")
    ax.set_ylabel("Strategy", fontsize=12, fontweight="normal")


def _draw_strategy_panel_single_pdg_resolution(
    ax,
    pdg_name: str,
    df_mean: pd.DataFrame,
    strategies,
    colors: Dict[str, str],
) -> None:
    pdg_data = df_mean[df_mean["pdg_name"] == pdg_name]
    x_pos = np.arange(len(strategies))
    resolutions = [
        pdg_data[pdg_data["strategy"] == s]["resolution_pct"].values[0]
        if len(pdg_data[pdg_data["strategy"] == s]) > 0
        else 0
        for s in strategies
    ]
    bars = ax.bar(
        x_pos,
        resolutions,
        color=[colors.get(s, "gray") for s in strategies],
        edgecolor="black",
        linewidth=1.5,
    )
    ax.set_title(pdg_name, fontsize=10, fontweight="normal")
    ax.set_ylabel("Resolution (%)", fontsize=10)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(strategies, rotation=45, ha="right", fontsize=9)
    ax.set_ylim(0, 110)
    ax.grid(True, alpha=0.3, axis="y")
    for bar in bars:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            height + 0.8,
            f"{height:.1f}%",
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="normal",
        )


def _save_strategy_resolution_panel_pngs(
    output_dir: Path,
    df: pd.DataFrame,
    df_mean: pd.DataFrame,
    pdgs: List[str],
    strategies,
    colors: Dict[str, str],
    dpi: int = 300,
) -> None:
    """Write each dashboard panel from strategy_resolution_visualization as its own PNG."""
    n_pdgs = len(pdgs)
    out = output_dir

    fig1, ax1 = plt.subplots(figsize=(15, 6))
    _draw_strategy_panel_inconsistency(ax1, df_mean, pdgs, strategies, colors)
    fig1.tight_layout()
    p1 = out / "strategy_resolution_panel_inconsistency.png"
    fig1.savefig(p1, dpi=dpi, bbox_inches="tight")
    plt.close(fig1)
    print(f"✓ Saved {p1}")

    fig2, ax2 = plt.subplots(figsize=(5, 5))
    _draw_strategy_panel_avg_resolution(ax2, df, colors)
    fig2.tight_layout()
    p2 = out / "strategy_resolution_panel_avg_resolution.png"
    fig2.savefig(p2, dpi=dpi, bbox_inches="tight")
    plt.close(fig2)
    print(f"✓ Saved {p2}")

    fig3, ax3 = plt.subplots(figsize=(16, 4.5))
    _draw_strategy_panel_heatmap(ax3, df_mean, pdgs, n_pdgs)
    fig3.tight_layout()
    p3 = out / "strategy_resolution_panel_heatmap.png"
    fig3.savefig(p3, dpi=dpi, bbox_inches="tight")
    plt.close(fig3)
    print(f"✓ Saved {p3}")

    bottom_rows = max(1, int(math.ceil(n_pdgs / 4)))
    fig4, axes4 = plt.subplots(bottom_rows, 4, figsize=(16, 3.2 * bottom_rows), squeeze=False)
    for idx, pdg_name in enumerate(pdgs):
        r, c = divmod(idx, 4)
        _draw_strategy_panel_single_pdg_resolution(
            axes4[r, c], pdg_name, df_mean, strategies, colors
        )
    for k in range(len(pdgs), bottom_rows * 4):
        r, c = divmod(k, 4)
        axes4[r, c].axis("off")
    fig4.tight_layout()
    p4 = out / "strategy_resolution_panel_per_pdg_grid.png"
    fig4.savefig(p4, dpi=dpi, bbox_inches="tight")
    plt.close(fig4)
    print(f"✓ Saved {p4}")


def visualize_results(results: List[Dict], output_dir: Optional[Path] = None):
    """Create comprehensive visualizations (bar/heatmap panels aggregate over CPD replicates)."""
    output_dir = output_dir or Path(".")
    output_dir.mkdir(parents=True, exist_ok=True)

    successful = [r for r in results if r["success"]]

    if not successful:
        print("No successful results to visualize!")
        return

    df = pd.DataFrame(successful)
    df_mean = (
        df.groupby(["pdg_name", "strategy"], as_index=False)[
            ["initial", "final", "resolution_pct", "distortion_tv"]
        ]
        .mean()
    )

    pdgs = sorted(df_mean["pdg_name"].unique(), key=pdg_name_sort_key)
    n_pdgs = len(pdgs)
    strategies = df_mean["strategy"].unique()
    colors = STRATEGY_VIZ_COLORS

    bottom_rows = max(1, int(math.ceil(n_pdgs / 4)))
    fig_h = 8 + 2.8 * bottom_rows
    fig = plt.figure(figsize=(20, fig_h))
    gs = fig.add_gridspec(3, 4, hspace=0.35, wspace=0.3, height_ratios=[1.0, 1.1, bottom_rows])


    ax1 = fig.add_subplot(gs[0, :3])
    _draw_strategy_panel_inconsistency(ax1, df_mean, pdgs, strategies, colors)

    ax2 = fig.add_subplot(gs[0, 3])
    _draw_strategy_panel_avg_resolution(ax2, df, colors)

    ax3 = fig.add_subplot(gs[1, :])
    _draw_strategy_panel_heatmap(ax3, df_mean, pdgs, n_pdgs)

    subgs = gs[2, :].subgridspec(bottom_rows, 4, hspace=0.45, wspace=0.35)
    for idx, pdg_name in enumerate(pdgs):
        r, c = divmod(idx, 4)
        ax = fig.add_subplot(subgs[r, c])
        _draw_strategy_panel_single_pdg_resolution(ax, pdg_name, df_mean, strategies, colors)

    for k in range(len(pdgs), bottom_rows * 4):
        r, c = divmod(k, 4)
        ax_empty = fig.add_subplot(subgs[r, c])
        ax_empty.axis("off")

    plt.tight_layout()
    strat_path = output_dir / "strategy_resolution_visualization.png"
    plt.savefig(strat_path, dpi=300, bbox_inches="tight")
    print(f"\n✓ Visualization saved to: {strat_path}")
    plt.close()

    _save_strategy_resolution_panel_pngs(output_dir, df, df_mean, pdgs, strategies, colors)

    visualize_distortion(successful, output_dir=output_dir)
    visualize_theta_tv_through_lir(successful, output_dir=output_dir)
    visualize_initial_vs_final_inconsistency_scatter(successful, output_dir=output_dir)
    visualize_initial_vs_final_inconsistency_bars(successful, output_dir=output_dir)


def visualize_initial_vs_final_inconsistency_bars(
    results: List[Dict], output_dir: Path
) -> None:
    """Grouped bars: mean initial vs final inconsistency by PDG and strategy."""
    successful = [r for r in results if r.get("success")]
    if not successful:
        return
    df = pd.DataFrame(successful)
    df_mean = (
        df.groupby(["pdg_name", "strategy"], as_index=False)[
            ["initial", "final", "resolution_pct", "distortion_tv"]
        ]
        .mean()
    )
    pdgs = sorted(df_mean["pdg_name"].unique(), key=pdg_name_sort_key)
    strategies = df_mean["strategy"].unique()
    n_pdgs = len(pdgs)
    colors = {"uniform": "#2E86AB", "partial": "#A23B72", "hub": "#F18F01"}

    fig_w = max(10.0, 1.15 * n_pdgs + 5)
    fig, ax = plt.subplots(figsize=(fig_w, 6.5))
    x = np.arange(len(pdgs))
    width = 0.15

    for i, strategy in enumerate(strategies):
        strategy_data = df_mean[df_mean["strategy"] == strategy]
        initial_vals = [
            strategy_data[strategy_data["pdg_name"] == pdg]["initial"].values[0]
            if len(strategy_data[strategy_data["pdg_name"] == pdg]) > 0
            else 0
            for pdg in pdgs
        ]
        final_vals = [
            strategy_data[strategy_data["pdg_name"] == pdg]["final"].values[0]
            if len(strategy_data[strategy_data["pdg_name"] == pdg]) > 0
            else 0
            for pdg in pdgs
        ]
        offset = (i - len(strategies) / 2 + 0.5) * width
        ax.bar(
            x + offset,
            initial_vals,
            width,
            label=f"{strategy} (initial)",
            color=colors.get(strategy, "gray"),
            alpha=0.3,
            edgecolor="black",
            linewidth=0.5,
        )
        ax.bar(
            x + offset,
            final_vals,
            width,
            label=f"{strategy} (final)",
            color=colors.get(strategy, "gray"),
            alpha=1.0,
            edgecolor="black",
            linewidth=1.5,
        )

    ax.set_xlabel("PDG", fontsize=12, fontweight="normal")
    ax.set_ylabel("Inconsistency", fontsize=12, fontweight="normal")
    ax.set_xticks(x)
    rot = 35 if n_pdgs > 6 else (20 if n_pdgs > 4 else 0)
    ax.set_xticklabels(pdgs, rotation=rot, ha="right" if rot else "center")
    ax.grid(True, alpha=0.3, axis="y")
    ax.legend(
        ncol=2,
        fontsize=9,
        loc="upper left",
        bbox_to_anchor=(0.02, 0.98),
        bbox_transform=ax.transAxes,
        borderaxespad=0.0,
        framealpha=0.95,
    )

    fig.tight_layout()
    p = output_dir / "fig_inconsistency_initial_vs_final_bars.png"
    fig.savefig(p, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"✓ Saved {p}")


def visualize_theta_tv_through_lir(results: List[Dict], output_dir: Path) -> None:
    """Mean cumulative TV of CPD parameters vs initial θ along refocus steps (LIR trajectory)."""
    rows = []
    for r in results:
        if not r.get("success"):
            continue
        tr = r.get("tv_theta_trace") or []
        if not tr:
            continue
        rep = r.get("replicate", 0)
        for step, tv in enumerate(tr):
            rows.append(
                {
                    "pdg_name": r["pdg_name"],
                    "strategy": r["strategy"],
                    "replicate": rep,
                    "step": step,
                    "tv_theta": tv,
                }
            )
    if not rows:
        print("No θ–TV trajectories to plot.")
        return

    dfl = pd.DataFrame(rows)
    pdgs = sorted(dfl["pdg_name"].unique(), key=pdg_name_sort_key)
    colors = {"uniform": "#2E86AB", "partial": "#A23B72", "hub": "#F18F01"}

    fig, ax_pdgs = _figure_axes_for_pdgs(pdgs, max_cols=6)
    for ax, pdg_name in ax_pdgs:
        sub = dfl[dfl["pdg_name"] == pdg_name]
        for strat in ("uniform", "partial", "hub"):
            s = sub[sub["strategy"] == strat]
            if s.empty:
                continue
            g = s.groupby("step")["tv_theta"].agg(["mean", "std"]).reset_index()
            c = colors.get(strat, "gray")
            ax.plot(g["step"], g["mean"], label=strat, color=c)
            ax.fill_between(
                g["step"],
                g["mean"] - g["std"].fillna(0),
                g["mean"] + g["std"].fillna(0),
                alpha=0.2,
                color=c,
            )
        ax.set_xlabel("Refocus step")
        ax.set_ylabel("Cumulative TV vs initial θ")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    fig.tight_layout()
    p = output_dir / "fig_tv_theta_lir_process.png"
    fig.savefig(p, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"✓ Saved {p}")


def visualize_initial_vs_final_inconsistency_scatter(results: List[Dict], output_dir: Path) -> None:
    """Initial vs final inconsistency: one point per CPD replicate, colored by strategy."""
    rows = []
    for r in results:
        if not r.get("success"):
            continue
        rows.append(
            {
                "pdg_name": r["pdg_name"],
                "strategy": r["strategy"],
                "replicate": r.get("replicate", 0),
                "initial": r["initial"],
                "final": r["final"],
            }
        )
    df = pd.DataFrame(rows)
    if df.empty:
        return

    pdgs = sorted(df["pdg_name"].unique(), key=pdg_name_sort_key)
    colors = {"uniform": "#2E86AB", "partial": "#A23B72", "hub": "#F18F01"}

    fig, ax_pdgs = _figure_axes_for_pdgs(pdgs, max_cols=6)
    pt_size = 10 if len(df) > 2000 else 14
    pt_alpha = 0.35 if len(df) > 2000 else 0.45
    for ax, pdg_name in ax_pdgs:
        sub = df[df["pdg_name"] == pdg_name]
        for strat in ("uniform", "partial", "hub"):
            s = sub[sub["strategy"] == strat]
            if s.empty:
                continue
            ax.scatter(
                s["initial"],
                s["final"],
                s=pt_size,
                alpha=pt_alpha,
                label=strat,
                color=colors.get(strat, "gray"),
                edgecolors="none",
            )
        lo = min(sub["initial"].min(), sub["final"].min())
        hi = max(sub["initial"].max(), sub["final"].max())
        if np.isfinite(lo) and np.isfinite(hi) and lo < hi:
            ax.plot([lo, hi], [lo, hi], "k--", alpha=0.35, linewidth=1)
        ax.set_title(pdg_name, fontsize=10, fontweight="normal")
        ax.set_xlabel("Initial inconsistency")
        ax.set_ylabel("Final inconsistency")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    fig.tight_layout()
    p = output_dir / "fig_inconsistency_initial_vs_final.png"
    fig.savefig(p, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"✓ Saved {p}")


def visualize_distortion(results: List[Dict], output_dir: Optional[Path] = None):
    """Create visualizations for distortion metric (Total Variation of μ*)."""
    output_dir = output_dir or Path(".")
    output_dir.mkdir(parents=True, exist_ok=True)
    print("\nCreating distortion visualization (Total Variation)...")

    df = pd.DataFrame(results)
    
    # Create figure with 3 panels focused on Total Variation
    fig = plt.figure(figsize=(18, 6))
    
    
    strategies = df["strategy"].unique()
    pdgs = sorted(df["pdg_name"].unique(), key=pdg_name_sort_key)
    n_pdgs_d = len(pdgs)
    colors = {"uniform": "#2E86AB", "partial": "#A23B72", "hub": "#F18F01"}
    
    # Plot 1: Average Total Variation Distortion by Strategy
    ax1 = plt.subplot(1, 3, 1)
    means = [df[df['strategy'] == s]['distortion_tv'].mean() for s in strategies]
    bars = ax1.bar(strategies, means, color=[colors.get(s, 'gray') for s in strategies],
                   alpha=0.8, edgecolor='black', linewidth=2)
    
    ax1.set_xlabel('Focus', fontsize=14, fontweight='normal')
    ax1.set_ylabel('Average Total Variation Distance', fontsize=14, fontweight='normal')
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.4f}', ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    # Plot 2: Resolution vs Total Variation Distortion scatter
    # Drop extreme negative-resolution outliers to keep the main trend readable.
    resolution_floor = -50.0
    ax2 = plt.subplot(1, 3, 2)
    for strategy in strategies:
        strategy_data = df[(df['strategy'] == strategy) & (df['resolution_pct'] >= resolution_floor)]
        ax2.scatter(strategy_data['resolution_pct'], strategy_data['distortion_tv'],
                   label=strategy, alpha=0.7, s=180, color=colors.get(strategy, 'gray'),
                   edgecolors='black', linewidth=2)
    
    ax2.set_xlabel('Resolution (%)', fontsize=14, fontweight='normal')
    ax2.set_ylabel('Total Variation Distance', fontsize=14, fontweight='normal')
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Heatmap of Total Variation Distortion
    ax3 = plt.subplot(1, 3, 3)
    pivot_tv = df.pivot_table(
        index="strategy", columns="pdg_name", values="distortion_tv", aggfunc="mean"
    )
    pivot_tv = pivot_tv.reindex(columns=pdgs)
    ann_tv = n_pdgs_d <= 12
    tv_kw = dict(
        cmap="YlOrRd",
        ax=ax3,
        cbar_kws={"label": "Total Variation Distance"},
        linewidths=1,
        linecolor="black",
    )
    if ann_tv:
        tv_kw["annot"] = True
        tv_kw["fmt"] = ".4f"
        tv_kw["annot_kws"] = {"fontsize": 8 if n_pdgs_d > 8 else 11, "fontweight": "bold"}
    sns.heatmap(pivot_tv, **tv_kw)
    ax3.set_xlabel('PDG', fontsize=12, fontweight='normal')
    ax3.set_ylabel('Focus', fontsize=12, fontweight='normal')
    ax3.tick_params(labelsize=10)
    
    plt.tight_layout()
    dist_path = output_dir / "distortion_analysis.png"
    plt.savefig(dist_path, dpi=300, bbox_inches="tight")
    print(f"✓ Distortion visualization saved to: {dist_path}")
    plt.close()


def print_results_table(results: List[Dict]):
    """Print detailed results table (aggregated over replicates when many rows)."""
    successful = [r for r in results if r["success"]]
    failed = [r for r in results if not r["success"]]

    print("\n" + "=" * 120)
    print("FOCUS RESOLUTION RESULTS WITH DISTORTION (TOTAL VARIATION)")
    print("=" * 120)

    print(
        f"\n{'PDG':<15} {'Focus':<15} {'Initial':<12} {'Final':<12} {'Resolution %':<15} {'Distortion (TV)':<15}"
    )
    print("=" * 120)

    if len(successful) > 64:
        print("(Many CPD replicates: showing means over replicates per PDG × strategy)\n")
        sdf = pd.DataFrame(successful)
        agg = (
            sdf.groupby(["pdg_name", "strategy"], as_index=False)
            .agg(
                {
                    "initial": "mean",
                    "final": "mean",
                    "resolution_pct": "mean",
                    "distortion_tv": "mean",
                }
            )
            .sort_values(["pdg_name", "strategy"])
        )
        for _, row in agg.iterrows():
            print(
                f"{row['pdg_name']:<15} {row['strategy']:<15} "
                f"{row['initial']:<12.6f} {row['final']:<12.6f} "
                f"{row['resolution_pct']:<15.2f} {row['distortion_tv']:<15.6f}"
            )
    else:
        for result in successful:
            print(
                f"{result['pdg_name']:<15} {result['strategy']:<15} "
                f"{result['initial']:<12.6f} {result['final']:<12.6f} "
                f"{result['resolution_pct']:<15.2f} {result['distortion_tv']:<15.6f}"
            )
    
    if failed:
        print(f"\n⚠️  FAILED EXPERIMENTS: {len(failed)}")
        for result in failed:
            print(f"  {result['pdg_name']} - {result['strategy']}: {result.get('error', 'Unknown error')}")
    
    # Summary by focus
    print("\n" + "="*120)
    print("SUMMARY BY FOCUS")
    print("="*120)
    
    df = pd.DataFrame(successful)
    strategy_summary = df.groupby('strategy').agg({
        'initial': 'mean',
        'final': 'mean',
        'resolution_pct': 'mean',
        'distortion_tv': 'mean'
    }).round(6)
    
    print(f"\n{'Focus':<15} {'Avg Initial':<12} {'Avg Final':<12} {'Avg Res %':<12} {'Avg Distortion (TV)':<20}")
    print("-"*71)
    for strategy, row in strategy_summary.iterrows():
        print(f"{strategy:<15} {row['initial']:<12.6f} {row['final']:<12.6f} {row['resolution_pct']:<12.2f} "
              f"{row['distortion_tv']:<20.6f}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--n-cpd-replicates",
        type=int,
        default=100,
        help="Number of random CPD draws per fixed structure (use 100–1000 for stable means).",
    )
    parser.add_argument(
        "--cpd-seed-offset",
        type=int,
        default=200_000,
        help="Base offset for CPD RNG seeds (replicate r uses offset + r * 1_000_003 + pdg index).",
    )
    parser.add_argument(
        "--n-pdgs",
        type=int,
        default=12,
        help="Number of fixed PDG structures (family chain_(4+k)v_(3+k)e, seeds 104+k). Use 4 to match the original experiment set.",
    )
    parser.add_argument(
        "--pdg-start-index",
        type=int,
        default=0,
        help="Start index k for chain_(4+k)v_(3+k)e. Use with --n-pdgs to run a shard.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("results") / "attention_strategies",
        help="Directory for PNGs and CSV.",
    )
    parser.add_argument(
        "--live-plot-every",
        type=int,
        default=9,
        help="Refresh figures every N completed strategy results during the run (0 disables live plotting).",
    )
    parser.add_argument(
        "--skip-final-visualizations",
        action="store_true",
        help="Skip end-of-run visualization generation (useful for parallel shard runs).",
    )
    args = parser.parse_args()
    if args.n_pdgs < 1:
        parser.error("--n-pdgs must be >= 1")
    if args.pdg_start_index < 0:
        parser.error("--pdg-start-index must be >= 0")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = args.out_dir / "results_checkpoint.json"

    print("=" * 120)
    print("RUNNING ATTENTION STRATEGIES ON ALL PDGs")
    print("=" * 120)

    pdg_configs = build_pdg_configs(args.n_pdgs, start_index=args.pdg_start_index)

    strategies = {
        "uniform": global_strategy,
        "partial": local_strategy,
        "hub": node_based_strategy,
    }

    templates: List[Tuple[PDG, str]] = []
    pdg_structures: Dict[str, Dict] = {}
    for num_vars, num_edges, pdg_name, seed in pdg_configs:
        template = generate_pdg(num_vars, num_edges, seed=seed)
        templates.append((template, pdg_name))
        pdg_structures[pdg_name] = {
            "num_vars": num_vars,
            "num_edges": num_edges,
            "structure_seed": seed,
            "graph": _pdg_structure_to_json(template),
        }

    all_results: List[Dict] = []
    intermediate_files: List[str] = []

    # Write an initial empty checkpoint so downstream consumers can detect the run.
    _write_json_checkpoint(
        checkpoint_path,
        args=args,
        results=all_results,
        pdg_structures=pdg_structures,
        intermediate_files=intermediate_files,
        complete=False,
    )
    try:
        for pdg_idx, (template, pdg_name) in enumerate(templates):
            print(f"\n{'─' * 120}")
            print(f"Processing {pdg_name} (fixed structure; {args.n_cpd_replicates} CPD replicates)")
            print(f"{'─' * 120}")

            for rep in range(args.n_cpd_replicates):
                cpd_seed = args.cpd_seed_offset + rep * 1_000_003 + pdg_idx * 17_171
                pdg_instance = resample_cpds_same_structure(template, cpd_seed)
                rep_path = _replicate_file_path(args.out_dir, pdg_name, rep)
                rep_payload = {
                    "complete": False,
                    "pdg_name": pdg_name,
                    "pdg_index": pdg_idx,
                    "replicate": rep,
                    "cpd_seed": cpd_seed,
                    "graph": pdg_structures[pdg_name]["graph"],
                    "cpds": _cpds_to_json(pdg_instance),
                    "strategy_results": {},
                }
                _write_json_file_atomic(rep_path, rep_payload)
                rep_rel = str(rep_path.relative_to(args.out_dir))
                if rep_rel not in intermediate_files:
                    intermediate_files.append(rep_rel)
                    _write_json_checkpoint(
                        checkpoint_path,
                        args=args,
                        results=all_results,
                        pdg_structures=pdg_structures,
                        intermediate_files=intermediate_files,
                        complete=False,
                    )

                for si, (strategy_name, strategy_func) in enumerate(strategies.items()):
                    rng_seed = cpd_seed + si * 99991 + 13
                    msg = f"  [{pdg_name}] rep {rep + 1}/{args.n_cpd_replicates} {strategy_name}..."
                    print(msg, end=" ", flush=True)
                    result = run_strategy(
                        pdg_instance,
                        pdg_name,
                        strategy_name,
                        strategy_func,
                        replicate=rep,
                        rng_seed=rng_seed,
                    )
                    all_results.append(result)
                    rep_payload["strategy_results"][strategy_name] = result
                    rep_payload["complete"] = len(rep_payload["strategy_results"]) == len(strategies)
                    _write_json_file_atomic(rep_path, rep_payload)
                    # Persist after every result so partial data survives crashes/preemption.
                    _write_json_checkpoint(
                        checkpoint_path,
                        args=args,
                        results=all_results,
                        pdg_structures=pdg_structures,
                        intermediate_files=intermediate_files,
                        complete=False,
                    )
                    if args.live_plot_every > 0 and (len(all_results) % args.live_plot_every == 0):
                        try:
                            print(" [refreshing figures...]", end="", flush=True)
                            visualize_results(all_results, output_dir=args.out_dir)
                            print(" done", flush=True)
                        except Exception as viz_err:
                            # Never stop experiments because of plotting failures.
                            print(f" [plot refresh failed: {viz_err}]", flush=True)
                    if result["success"]:
                        print(f"✓ Res {result['resolution_pct']:.1f}%")
                    else:
                        print("✗ Failed")
    finally:
        _write_json_checkpoint(
            checkpoint_path,
            args=args,
            results=all_results,
            pdg_structures=pdg_structures,
            intermediate_files=intermediate_files,
            complete=True,
        )

    df_out = pd.DataFrame(all_results)
    csv_path = args.out_dir / "results_all_replicates.csv"
    _export = df_out.drop(columns=["tv_theta_trace"], errors="ignore")
    _export.to_csv(csv_path, index=False)
    print(f"\nWrote {csv_path}")

    traj_rows = []
    for r in all_results:
        if not r.get("success") or not r.get("tv_theta_trace"):
            continue
        for step, tv in enumerate(r["tv_theta_trace"]):
            traj_rows.append(
                {
                    "pdg_name": r["pdg_name"],
                    "strategy": r["strategy"],
                    "replicate": r.get("replicate", 0),
                    "step": step,
                    "tv_theta": tv,
                }
            )
    if traj_rows:
        pd.DataFrame(traj_rows).to_csv(args.out_dir / "tv_theta_trajectories.csv", index=False)

    print_results_table(all_results)

    if not args.skip_final_visualizations:
        print("\n" + "=" * 120)
        print("CREATING VISUALIZATIONS")
        print("=" * 120)
        visualize_results(all_results, output_dir=args.out_dir)
    else:
        print("\nSkipping final visualizations for this shard (--skip-final-visualizations).")

    print("\n" + "=" * 120)
    print("EXPERIMENT COMPLETE")
    print("=" * 120)


if __name__ == "__main__":
    main()

