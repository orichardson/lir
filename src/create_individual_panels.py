#!/usr/bin/env python3
"""
Create individual panels as separate figures (PNG and PDF).
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
        
        resolution_pct = ((initial_inconsistency - final_inconsistency) / 
                         initial_inconsistency * 100) if initial_inconsistency > 0 else 0
        
        return {
            'pdg_name': pdg_name,
            'strategy': strategy_name,
            'initial': initial_inconsistency,
            'final': final_inconsistency,
            'resolution_pct': resolution_pct,
            'success': True
        }
    
    except Exception as e:
        return {
            'pdg_name': pdg_name,
            'strategy': strategy_name,
            'initial': 0.0,
            'final': 0.0,
            'resolution_pct': 0.0,
            'success': False,
            'error': str(e)
        }


def create_panel_1(df, output_dir):
    """Panel 1: Initial vs Final Inconsistency by PDG and Strategy"""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    pdgs = df['pdg_name'].unique()
    strategies = df['strategy'].unique()
    x = np.arange(len(pdgs))
    width = 0.15
    
    colors = {'global': '#2E86AB', 'local': '#A23B72', 'node_based': '#F18F01'}
    
    for i, strategy in enumerate(strategies):
        strategy_data = df[df['strategy'] == strategy]
        initial_vals = [strategy_data[strategy_data['pdg_name'] == pdg]['initial'].values[0] 
                       if len(strategy_data[strategy_data['pdg_name'] == pdg]) > 0 else 0
                       for pdg in pdgs]
        final_vals = [strategy_data[strategy_data['pdg_name'] == pdg]['final'].values[0]
                     if len(strategy_data[strategy_data['pdg_name'] == pdg]) > 0 else 0
                     for pdg in pdgs]
        
        offset = (i - len(strategies)/2 + 0.5) * width
        ax.bar(x + offset, initial_vals, width, label=f'{strategy} (initial)', 
               color=colors.get(strategy, 'gray'), alpha=0.3, edgecolor='black', linewidth=0.5)
        ax.bar(x + offset, final_vals, width, label=f'{strategy} (final)', 
               color=colors.get(strategy, 'gray'), alpha=1.0, edgecolor='black', linewidth=1.5)
    
    ax.set_xlabel('PDG', fontsize=14, fontweight='bold')
    ax.set_ylabel('Inconsistency', fontsize=14, fontweight='bold')
    ax.set_title('Initial vs Final Inconsistency by PDG and Strategy', fontsize=16, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(pdgs, rotation=0, fontsize=12)
    ax.legend(ncol=2, fontsize=10, loc='upper right')
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/panel_1_initial_vs_final.png', dpi=300, bbox_inches='tight')
    plt.savefig(f'{output_dir}/panel_1_initial_vs_final.pdf', bbox_inches='tight')
    plt.close()
    print("✓ Panel 1: Initial vs Final Inconsistency")


def create_panel_2(df, output_dir):
    """Panel 2: Average Resolution by Strategy"""
    fig, ax = plt.subplots(figsize=(8, 6))
    
    colors = {'global': '#2E86AB', 'local': '#A23B72', 'node_based': '#F18F01'}
    
    strategy_means = df.groupby('strategy')['resolution_pct'].mean().sort_values(ascending=False)
    bars = ax.barh(strategy_means.index, strategy_means.values, 
                   color=[colors.get(s, 'gray') for s in strategy_means.index],
                   edgecolor='black', linewidth=2)
    ax.set_xlabel('Resolution (%)', fontsize=14, fontweight='bold')
    ax.set_title('Average Resolution by Strategy', fontsize=16, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='x')
    ax.tick_params(axis='both', labelsize=12)
    
    # Add value labels
    for bar in bars:
        width = bar.get_width()
        ax.text(width + 1, bar.get_y() + bar.get_height()/2, 
                f'{width:.1f}%', ha='left', va='center', fontweight='bold', fontsize=12)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/panel_2_average_resolution.png', dpi=300, bbox_inches='tight')
    plt.savefig(f'{output_dir}/panel_2_average_resolution.pdf', bbox_inches='tight')
    plt.close()
    print("✓ Panel 2: Average Resolution by Strategy")


def create_panel_3(df, output_dir):
    """Panel 3: Resolution Percentage Heatmap"""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    pivot_data = df.pivot(index='strategy', columns='pdg_name', values='resolution_pct')
    sns.heatmap(pivot_data, annot=True, fmt='.1f', cmap='RdYlGn', 
               cbar_kws={'label': 'Resolution (%)'}, ax=ax,
               vmin=0, vmax=100, linewidths=3, linecolor='black',
               annot_kws={'fontsize': 14, 'fontweight': 'bold'})
    ax.set_title('Resolution Percentage Heatmap (Strategy × PDG)', 
                 fontsize=16, fontweight='bold')
    ax.set_xlabel('PDG', fontsize=14, fontweight='bold')
    ax.set_ylabel('Strategy', fontsize=14, fontweight='bold')
    ax.tick_params(axis='both', labelsize=12)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/panel_3_heatmap.png', dpi=300, bbox_inches='tight')
    plt.savefig(f'{output_dir}/panel_3_heatmap.pdf', bbox_inches='tight')
    plt.close()
    print("✓ Panel 3: Resolution Heatmap")


def create_panel_4(df, output_dir):
    """Panel 4: Individual PDG Comparisons (4 subplots)"""
    pdgs = df['pdg_name'].unique()
    strategies = df['strategy'].unique()
    colors = {'global': '#2E86AB', 'local': '#A23B72', 'node_based': '#F18F01'}
    
    for pdg_name in pdgs:
        fig, ax = plt.subplots(figsize=(8, 6))
        pdg_data = df[df['pdg_name'] == pdg_name]
        
        x_pos = np.arange(len(strategies))
        resolutions = [pdg_data[pdg_data['strategy'] == s]['resolution_pct'].values[0]
                      if len(pdg_data[pdg_data['strategy'] == s]) > 0 else 0
                      for s in strategies]
        
        bars = ax.bar(x_pos, resolutions, 
                     color=[colors.get(s, 'gray') for s in strategies],
                     edgecolor='black', linewidth=2, width=0.6)
        
        ax.set_title(f'{pdg_name}', fontsize=16, fontweight='bold')
        ax.set_ylabel('Resolution (%)', fontsize=14, fontweight='bold')
        ax.set_xlabel('Strategy', fontsize=14, fontweight='bold')
        ax.set_xticks(x_pos)
        ax.set_xticklabels(strategies, rotation=0, fontsize=12)
        ax.set_ylim(0, 105)
        ax.grid(True, alpha=0.3, axis='y')
        ax.tick_params(axis='both', labelsize=12)
        
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                   f'{height:.1f}%', ha='center', va='bottom', fontsize=12, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(f'{output_dir}/panel_4_{pdg_name}.png', dpi=300, bbox_inches='tight')
        plt.savefig(f'{output_dir}/panel_4_{pdg_name}.pdf', bbox_inches='tight')
        plt.close()
        print(f"✓ Panel 4: {pdg_name}")


def main():
    """Main function."""
    print("="*100)
    print("CREATING INDIVIDUAL PANELS (PNG & PDF)")
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
        'global': global_strategy,
        'local': local_strategy,
        'node_based': node_based_strategy,
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
    
    # Create individual panels
    print("\n2. Creating individual panels...")
    create_panel_1(df, output_dir)
    create_panel_2(df, output_dir)
    create_panel_3(df, output_dir)
    create_panel_4(df, output_dir)
    
    print("\n" + "="*100)
    print(f"✓ ALL PANELS CREATED")
    print(f"  Location: {output_dir}/")
    print(f"  Formats: PNG (300 DPI) and PDF")
    print(f"  Total files: {len(list(Path(output_dir).glob('*')))} files")
    print("="*100)


if __name__ == "__main__":
    main()

