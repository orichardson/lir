#!/usr/bin/env python3
"""
Run global, local, and node-based attention strategies on all PDGs and visualize results.
"""

import sys
from pathlib import Path
import random
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Tuple, Callable
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


def run_strategy(pdg: PDG, pdg_name: str, strategy_name: str, 
                 strategy_func: Callable) -> Dict:
    """Run a single strategy and return results."""
    try:
        pdg_copy = PDG.copy(pdg)
        
        # Convert CPTs to ParamCPD for learning
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
        
        # Get initial inconsistency and save initial mu*
        mu_init = opt_joint(pdg_copy, gamma=0.0, iters=50, verbose=False)
        initial_inconsistency = float(torch_score(pdg_copy, mu_init, 0.0))
        
        # Save initial mu* as numpy array for distance calculation
        mu_init_array = mu_init.data.detach().cpu().numpy() if hasattr(mu_init.data, 'detach') else mu_init.data
        
        # Run training
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
        
        # Get final inconsistency and final mu*
        mu_final = opt_joint(pdg_copy, gamma=0.0, iters=50, verbose=False)
        final_inconsistency = float(torch_score(pdg_copy, mu_final, 0.0))
        
        # Get final mu* as numpy array
        mu_final_array = mu_final.data.detach().cpu().numpy() if hasattr(mu_final.data, 'detach') else mu_final.data
        
        # Calculate total variation distance (distortion metric)
        distortion_tv = float(0.5 * np.sum(np.abs(mu_init_array - mu_final_array)))
        
        # Calculate resolution percentage
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
        print(f"  ⚠️  Error with {strategy_name}: {str(e)}")
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


def visualize_results(results: List[Dict]):
    """Create comprehensive visualizations."""
    
    # Filter successful results
    successful = [r for r in results if r['success']]
    
    if not successful:
        print("No successful results to visualize!")
        return
    
    # Create DataFrame
    df = pd.DataFrame(successful)
    
    # Create figure with subplots
    fig = plt.figure(figsize=(20, 14))
    gs = fig.add_gridspec(3, 4, hspace=0.3, wspace=0.3)
    
    fig.suptitle('Attention Strategy Performance Across PDGs', 
                fontsize=18, fontweight='bold', y=0.98)
    
    # 1. Initial vs Final Inconsistency by PDG
    ax1 = fig.add_subplot(gs[0, :3])
    pdgs = df['pdg_name'].unique()
    strategies = df['strategy'].unique()
    x = np.arange(len(pdgs))
    width = 0.15
    
    colors = {'uniform': '#2E86AB', 'partial': '#A23B72', 'hub': '#F18F01'}
    
    for i, strategy in enumerate(strategies):
        strategy_data = df[df['strategy'] == strategy]
        initial_vals = [strategy_data[strategy_data['pdg_name'] == pdg]['initial'].values[0] 
                       if len(strategy_data[strategy_data['pdg_name'] == pdg]) > 0 else 0
                       for pdg in pdgs]
        final_vals = [strategy_data[strategy_data['pdg_name'] == pdg]['final'].values[0]
                     if len(strategy_data[strategy_data['pdg_name'] == pdg]) > 0 else 0
                     for pdg in pdgs]
        
        offset = (i - len(strategies)/2 + 0.5) * width
        ax1.bar(x + offset, initial_vals, width, label=f'{strategy} (initial)', 
               color=colors.get(strategy, 'gray'), alpha=0.3, edgecolor='black', linewidth=0.5)
        ax1.bar(x + offset, final_vals, width, label=f'{strategy} (final)', 
               color=colors.get(strategy, 'gray'), alpha=1.0, edgecolor='black', linewidth=1.5)
    
    ax1.set_xlabel('PDG', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Inconsistency', fontsize=12, fontweight='bold')
    ax1.set_title('Initial vs Final Inconsistency by PDG and Strategy', fontsize=14, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(pdgs, rotation=0)
    ax1.legend(ncol=2, fontsize=9, loc='upper right')
    ax1.grid(True, alpha=0.3, axis='y')
    
    # 2. Resolution Percentage Comparison
    ax2 = fig.add_subplot(gs[0, 3])
    strategy_means = df.groupby('strategy')['resolution_pct'].mean().sort_values(ascending=False)
    bars = ax2.barh(strategy_means.index, strategy_means.values, 
                    color=[colors.get(s, 'gray') for s in strategy_means.index],
                    edgecolor='black', linewidth=1.5)
    ax2.set_xlabel('Resolution (%)', fontsize=12, fontweight='bold')
    ax2.set_title('Average Resolution\nby Strategy', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='x')
    
    # Add value labels
    for bar in bars:
        width = bar.get_width()
        ax2.text(width, bar.get_y() + bar.get_height()/2, 
                f'{width:.1f}%', ha='left', va='center', fontweight='bold', fontsize=10)
    
    # 3. Resolution Percentage Heatmap
    ax3 = fig.add_subplot(gs[1, :])
    pivot_data = df.pivot(index='strategy', columns='pdg_name', values='resolution_pct')
    sns.heatmap(pivot_data, annot=True, fmt='.1f', cmap='RdYlGn', 
               cbar_kws={'label': 'Resolution (%)'}, ax=ax3,
               vmin=0, vmax=100, linewidths=2, linecolor='black')
    ax3.set_title('Resolution Percentage Heatmap (Strategy × PDG)', 
                 fontsize=14, fontweight='bold')
    ax3.set_xlabel('PDG', fontsize=12, fontweight='bold')
    ax3.set_ylabel('Strategy', fontsize=12, fontweight='bold')
    
    # 4. Individual PDG plots
    for idx, pdg_name in enumerate(pdgs):
        ax = fig.add_subplot(gs[2, idx])
        pdg_data = df[df['pdg_name'] == pdg_name]
        
        x_pos = np.arange(len(strategies))
        resolutions = [pdg_data[pdg_data['strategy'] == s]['resolution_pct'].values[0]
                      if len(pdg_data[pdg_data['strategy'] == s]) > 0 else 0
                      for s in strategies]
        
        bars = ax.bar(x_pos, resolutions, 
                     color=[colors.get(s, 'gray') for s in strategies],
                     edgecolor='black', linewidth=1.5)
        
        ax.set_title(f'{pdg_name}', fontsize=12, fontweight='bold')
        ax.set_ylabel('Resolution (%)', fontsize=10)
        ax.set_xticks(x_pos)
        ax.set_xticklabels(strategies, rotation=45, ha='right', fontsize=9)
        ax.set_ylim(0, 105)
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.1f}%', ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('strategy_resolution_visualization.png', dpi=300, bbox_inches='tight')
    print("\n✓ Visualization saved to: strategy_resolution_visualization.png")
    plt.close()
    
    # Create distortion visualization
    visualize_distortion(successful)


def visualize_distortion(results: List[Dict]):
    """Create visualizations for distortion metric (Total Variation)."""
    print("\nCreating distortion visualization (Total Variation)...")
    
    df = pd.DataFrame(results)
    
    # Create figure with 3 panels focused on Total Variation
    fig = plt.figure(figsize=(18, 6))
    
    fig.suptitle('Distortion Analysis: How Much Does μ* Change? (Total Variation Distance)', 
                fontsize=16, fontweight='bold', y=0.98)
    
    strategies = df['strategy'].unique()
    pdgs = df['pdg_name'].unique()
    colors = {'uniform': '#2E86AB', 'partial': '#A23B72', 'hub': '#F18F01'}
    
    # Plot 1: Average Total Variation Distortion by Strategy
    ax1 = plt.subplot(1, 3, 1)
    means = [df[df['strategy'] == s]['distortion_tv'].mean() for s in strategies]
    bars = ax1.bar(strategies, means, color=[colors.get(s, 'gray') for s in strategies],
                   alpha=0.8, edgecolor='black', linewidth=2)
    
    ax1.set_xlabel('Focus', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Average Total Variation Distance', fontsize=14, fontweight='bold')
    ax1.set_title('Average Distortion by Focus', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.4f}', ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    # Plot 2: Resolution vs Total Variation Distortion scatter
    ax2 = plt.subplot(1, 3, 2)
    for strategy in strategies:
        strategy_data = df[df['strategy'] == strategy]
        ax2.scatter(strategy_data['resolution_pct'], strategy_data['distortion_tv'],
                   label=strategy, alpha=0.7, s=180, color=colors.get(strategy, 'gray'),
                   edgecolors='black', linewidth=2)
    
    ax2.set_xlabel('Resolution (%)', fontsize=14, fontweight='bold')
    ax2.set_ylabel('Total Variation Distance', fontsize=14, fontweight='bold')
    ax2.set_title('Resolution vs Distortion Trade-off', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Heatmap of Total Variation Distortion
    ax3 = plt.subplot(1, 3, 3)
    pivot_tv = df.pivot(index='strategy', columns='pdg_name', values='distortion_tv')
    sns.heatmap(pivot_tv, annot=True, fmt='.4f', cmap='YlOrRd', ax=ax3,
               cbar_kws={'label': 'Total Variation Distance'}, linewidths=2, linecolor='black',
               annot_kws={'fontsize': 11, 'fontweight': 'bold'})
    ax3.set_title('Distortion Heatmap\n(Focus × PDG)', fontsize=14, fontweight='bold')
    ax3.set_xlabel('PDG', fontsize=12, fontweight='bold')
    ax3.set_ylabel('Focus', fontsize=12, fontweight='bold')
    ax3.tick_params(labelsize=10)
    
    plt.tight_layout()
    plt.savefig('distortion_analysis.png', dpi=300, bbox_inches='tight')
    print("✓ Distortion visualization saved to: distortion_analysis.png")
    plt.close()


def print_results_table(results: List[Dict]):
    """Print detailed results table."""
    successful = [r for r in results if r['success']]
    failed = [r for r in results if not r['success']]
    
    print("\n" + "="*120)
    print("FOCUS RESOLUTION RESULTS WITH DISTORTION (TOTAL VARIATION)")
    print("="*120)
    
    print(f"\n{'PDG':<15} {'Focus':<15} {'Initial':<12} {'Final':<12} {'Resolution %':<15} {'Distortion (TV)':<15}")
    print("="*120)
    
    for result in successful:
        print(f"{result['pdg_name']:<15} {result['strategy']:<15} "
              f"{result['initial']:<12.6f} {result['final']:<12.6f} "
              f"{result['resolution_pct']:<15.2f} {result['distortion_tv']:<15.6f}")
    
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
    """Main function."""
    print("="*120)
    print("RUNNING ATTENTION STRATEGIES ON ALL PDGs")
    print("="*120)
    
    # Define PDGs
    pdg_configs = [
        (4, 3, "chain_4v_3e", 104),
        (5, 4, "chain_5v_4e", 105),
        (6, 5, "chain_6v_5e", 106),
        (7, 6, "chain_7v_6e", 107),
    ]
    
    # Define strategies
    strategies = {
        'uniform': global_strategy,
        'partial': local_strategy,
        'hub': node_based_strategy,
    }
    
    all_results = []
    
    # Run experiments
    for num_vars, num_edges, pdg_name, seed in pdg_configs:
        print(f"\n{'─'*120}")
        print(f"Processing {pdg_name} ({num_vars} variables, {num_edges} edges)")
        print(f"{'─'*120}")
        
        pdg = generate_pdg(num_vars, num_edges, seed=seed)
        
        for strategy_name, strategy_func in strategies.items():
            print(f"  Running {strategy_name} strategy...", end=" ")
            result = run_strategy(pdg, pdg_name, strategy_name, strategy_func)
            all_results.append(result)
            if result['success']:
                print(f"✓ Resolution: {result['resolution_pct']:.2f}%")
            else:
                print(f"✗ Failed")
    
    # Print results
    print_results_table(all_results)
    
    # Create visualizations
    print("\n" + "="*120)
    print("CREATING VISUALIZATIONS")
    print("="*120)
    visualize_results(all_results)
    
    print("\n" + "="*120)
    print("EXPERIMENT COMPLETE")
    print("="*120)


if __name__ == "__main__":
    main()

