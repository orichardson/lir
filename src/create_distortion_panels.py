#!/usr/bin/env python3
"""
Create individual panels for distortion analysis (PNG and PDF).
"""

import sys
from pathlib import Path
import random
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Add project paths
sys.path.insert(0, str(Path(__file__).resolve().parents[0]))

from pdg.pdg import PDG
from pdg.rv import Variable as Var
from pdg.dist import CPT, ParamCPD
from pdg.alg.torch_opt import opt_joint, torch_score
from lir__simpler import lir_train


def generate_pdg(num_vars: int, num_edges: int, seed: int) -> PDG:
    """Generate a PDG - same as in the experiment."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    pdg = PDG()
    varlist = []
    
    for i in range(num_vars):
        domain_size = random.randint(2, 3)
        var = Var.alph(chr(65 + i), domain_size)
        pdg += var
        varlist.append(var)
    
    edges_added = 0
    edge_pairs = set()
    target_edges = {}
    
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


def global_strategy(pdg: PDG, t: int):
    attn_alpha = {}
    attn_beta = {}
    control = {}
    
    for L, X, Y, α, β, P in pdg.edges("l,X,Y,α,β,P"):
        if X.name != "1" and Y.name != "1":
            attn_alpha[L] = 0.0
            attn_beta[L] = 1.0
            control[L] = 1.0
    
    return attn_alpha, attn_beta, control


def local_strategy(pdg: PDG, t: int):
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


def node_based_strategy(pdg: PDG, t: int):
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
            attn_beta[L] = 1.0 if (X.name == focus_node.name or Y.name == focus_node.name) else 0.1
            control[L] = 1.0
    
    return attn_alpha, attn_beta, control


def run_strategy(pdg: PDG, pdg_name: str, strategy_name: str, strategy_func):
    try:
        pdg_copy = PDG.copy(pdg)
        
        edges_snapshot = list(pdg_copy.edges("l,X,Y,α,β,P"))
        for L, X, Y, α, β, P in edges_snapshot:
            if L[0] != "π" and X.name != "1" and Y.name != "1":
                learnable = ParamCPD(
                    src_var=X,
                    tgt_var=Y,
                    name=str(L),
                    init="from_cpd",
                    mask=None,
                    cpd=P
                )
                key = (X.name, Y.name, L)
                if key in pdg_copy.edgedata:
                    pdg_copy.edgedata[key]['cpd'] = learnable
        
        mu_init = opt_joint(pdg_copy, gamma=0.0, iters=50, verbose=False)
        initial_inconsistency = float(torch_score(pdg_copy, mu_init, 0.0))
        
        mu_init_array = mu_init.data.detach().cpu().numpy() if hasattr(mu_init.data, 'detach') else mu_init.data
        
        lir_train(
            pdg_copy,
            gamma=0.0,
            T=20,
            outer_iters=10,
            inner_iters=20,
            lr=0.05,
            refocus=strategy_func,
            verbose=False,
            mu_init=mu_init
        )
        
        mu_final = opt_joint(pdg_copy, gamma=0.0, iters=50, verbose=False)
        final_inconsistency = float(torch_score(pdg_copy, mu_final, 0.0))
        
        mu_final_array = mu_final.data.detach().cpu().numpy() if hasattr(mu_final.data, 'detach') else mu_final.data
        
        distortion_tv = float(0.5 * np.sum(np.abs(mu_init_array - mu_final_array)))
        
        resolution_pct = ((initial_inconsistency - final_inconsistency) / 
                         initial_inconsistency * 100) if initial_inconsistency > 0 else 0
        
        return {
            'pdg_name': pdg_name,
            'strategy': strategy_name,
            'initial': initial_inconsistency,
            'final': final_inconsistency,
            'resolution_pct': resolution_pct,
            'distortion_tv': distortion_tv,
            'success': True
        }
    
    except Exception as e:
        return {
            'pdg_name': pdg_name,
            'strategy': strategy_name,
            'initial': 0.0,
            'final': 0.0,
            'resolution_pct': 0.0,
            'distortion_tv': 0.0,
            'success': False,
            'error': str(e)
        }


def create_distortion_panel_1(df, output_dir):
    """Panel: Average Total Variation Distortion by Focus"""
    fig, ax = plt.subplots(figsize=(10, 7))
    
    colors = {'uniform': '#2E86AB', 'partial': '#A23B72', 'hub': '#F18F01'}
    strategies = df['strategy'].unique()
    means = [df[df['strategy'] == s]['distortion_tv'].mean() for s in strategies]
    
    bars = ax.bar(strategies, means, color=[colors.get(s, 'gray') for s in strategies],
                   alpha=0.8, edgecolor='black', linewidth=2.5, width=0.6)
    
    ax.set_xlabel('Focus', fontsize=16, fontweight='bold')
    ax.set_ylabel('Average Total Variation Distance', fontsize=16, fontweight='bold')
    ax.set_title('Average Distortion by Focus', fontsize=18, fontweight='bold', pad=20)
    ax.grid(True, alpha=0.3, axis='y')
    ax.tick_params(axis='both', labelsize=14)
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.005,
                f'{height:.4f}', ha='center', va='bottom', fontsize=13, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/distortion_panel_1_average.png', dpi=300, bbox_inches='tight')
    plt.savefig(f'{output_dir}/distortion_panel_1_average.pdf', bbox_inches='tight')
    plt.close()
    print("✓ Distortion Panel 1: Average by Focus")


def create_distortion_panel_2(df, output_dir):
    """Panel: Resolution vs Distortion Trade-off"""
    fig, ax = plt.subplots(figsize=(10, 7))
    
    colors = {'uniform': '#2E86AB', 'partial': '#A23B72', 'hub': '#F18F01'}
    strategies = df['strategy'].unique()
    
    for strategy in strategies:
        strategy_data = df[df['strategy'] == strategy]
        ax.scatter(strategy_data['resolution_pct'], strategy_data['distortion_tv'],
                   label=strategy, alpha=0.8, s=250, color=colors.get(strategy, 'gray'),
                   edgecolors='black', linewidth=2.5)
    
    ax.set_xlabel('Resolution (%)', fontsize=16, fontweight='bold')
    ax.set_ylabel('Total Variation Distance', fontsize=16, fontweight='bold')
    ax.set_title('Resolution vs Distortion Trade-off', fontsize=18, fontweight='bold', pad=20)
    ax.legend(fontsize=14, loc='upper left')
    ax.grid(True, alpha=0.3)
    ax.tick_params(axis='both', labelsize=14)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/distortion_panel_2_tradeoff.png', dpi=300, bbox_inches='tight')
    plt.savefig(f'{output_dir}/distortion_panel_2_tradeoff.pdf', bbox_inches='tight')
    plt.close()
    print("✓ Distortion Panel 2: Resolution vs Distortion Trade-off")


def create_distortion_panel_3(df, output_dir):
    """Panel: Heatmap of Total Variation Distortion"""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    pivot_tv = df.pivot(index='strategy', columns='pdg_name', values='distortion_tv')
    sns.heatmap(pivot_tv, annot=True, fmt='.4f', cmap='YlOrRd', ax=ax,
               cbar_kws={'label': 'Total Variation Distance'}, linewidths=3, linecolor='black',
               annot_kws={'fontsize': 14, 'fontweight': 'bold'})
    ax.set_title('Distortion Heatmap (Focus × PDG)', fontsize=18, fontweight='bold', pad=20)
    ax.set_xlabel('PDG', fontsize=16, fontweight='bold')
    ax.set_ylabel('Focus', fontsize=16, fontweight='bold')
    ax.tick_params(labelsize=13)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/distortion_panel_3_heatmap.png', dpi=300, bbox_inches='tight')
    plt.savefig(f'{output_dir}/distortion_panel_3_heatmap.pdf', bbox_inches='tight')
    plt.close()
    print("✓ Distortion Panel 3: Heatmap")


def main():
    """Main function."""
    print("="*100)
    print("CREATING INDIVIDUAL DISTORTION PANELS (PNG & PDF)")
    print("="*100)
    
    output_dir = "individual_panels"
    Path(output_dir).mkdir(exist_ok=True)
    
    # Run experiments
    print("\n1. Running experiments...")
    pdg_configs = [
        (4, 3, "chain_4v_3e", 104),
        (5, 4, "chain_5v_4e", 105),
        (6, 5, "chain_6v_5e", 106),
        (7, 6, "chain_7v_6e", 107),
    ]
    
    strategies = {
        'uniform': global_strategy,
        'partial': local_strategy,
        'hub': node_based_strategy,
    }
    
    all_results = []
    
    for num_vars, num_edges, pdg_name, seed in pdg_configs:
        print(f"  Processing {pdg_name}...")
        pdg = generate_pdg(num_vars, num_edges, seed=seed)
        
        for strategy_name, strategy_func in strategies.items():
            result = run_strategy(pdg, pdg_name, strategy_name, strategy_func)
            all_results.append(result)
    
    successful = [r for r in all_results if r['success']]
    df = pd.DataFrame(successful)
    
    # Create individual distortion panels
    print("\n2. Creating individual distortion panels...")
    create_distortion_panel_1(df, output_dir)
    create_distortion_panel_2(df, output_dir)
    create_distortion_panel_3(df, output_dir)
    
    print("\n" + "="*100)
    print(f"✓ ALL DISTORTION PANELS CREATED")
    print(f"  Location: {output_dir}/")
    print(f"  Files created:")
    print(f"    - distortion_panel_1_average.{{png,pdf}}")
    print(f"    - distortion_panel_2_tradeoff.{{png,pdf}}")
    print(f"    - distortion_panel_3_heatmap.{{png,pdf}}")
    print("="*100)


if __name__ == "__main__":
    main()

