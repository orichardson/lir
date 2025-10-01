#!/usr/bin/env python3
"""
Compare local vs global inconsistency improvement during training.
"""

import sys
from pathlib import Path
import pickle
import json
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import torch
from typing import List, Dict, Any, Tuple

# Add project paths
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).parent))

from pdg.pdg import PDG
from pdg.dist import ParamCPD
from pdg.alg.torch_opt import opt_joint, torch_score
from lir__simpler import lir_train, _collect_learnables, apply_attn_mask


def load_simple_dataset():
    """Load the simple PDG dataset."""
    with open('simple_pdg_dataset/pdgs.pkl', 'rb') as f:
        pdgs = pickle.load(f)
    
    with open('simple_pdg_dataset/specs.json', 'r') as f:
        specs = json.load(f)
    
    return pdgs, specs


def create_optimal_attention_strategy(alpha: float, beta: float):
    """Create an attention strategy with optimal alpha and beta values."""
    def attention_strategy(M: PDG, t: int):
        """Attention strategy with optimal alpha and beta."""
        attn_alpha = {}
        attn_beta = {}
        control = {}
        
        learnables = _collect_learnables(M)
        
        for label in learnables.keys():
            attn_alpha[label] = alpha
            attn_beta[label] = beta
        
        return attn_alpha, attn_beta, control
    
    return attention_strategy


def track_inconsistency_during_training(pdg: PDG, spec: Dict, alpha: float, beta: float, 
                                      num_steps: int = 15) -> Dict[str, Any]:
    """Track inconsistency metrics during training."""
    print(f"  Tracking training for {spec['name']} ({spec['pattern']}) with α={alpha}, β={beta}")
    
    # Make PDG parametric
    pdg_copy = pdg.copy()
    edges_snapshot = list(pdg_copy.edges("l,X,Y,α,β,P"))
    
    for L, X, Y, α, β, P in edges_snapshot:
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
    
    # Compute initial metrics
    mu_init = opt_joint(pdg_copy, gamma=0.1, iters=20, verbose=False)
    initial_global = float(torch_score(pdg_copy, mu_init, 0.01))
    initial_local = float(torch_score(pdg_copy, mu_init, 0.001))
    
    # Create attention strategy
    attention_strategy = create_optimal_attention_strategy(alpha, beta)
    
    # Track metrics during training
    global_inconsistency = [initial_global]
    local_inconsistency = [initial_local]
    training_steps = [0]
    
    # Custom training loop to track metrics
    learnables = _collect_learnables(pdg_copy)
    if not learnables:
        raise ValueError("No ParamCPDs found in PDG")
    
    # Create optimizer
    opt = torch.optim.Adam([P.logits for P in learnables.values()], lr=0.01/5)
    
    mu_current = mu_init
    for t in range(num_steps):
        # Get attention masks
        attn_alpha, attn_beta, control = attention_strategy(pdg_copy, t)
        
        # Apply attention masks
        pdg_with_attention = apply_attn_mask(
            pdg_copy, 
            attn_mask_beta=attn_beta, 
            attn_mask_alpha=attn_alpha
        )
        
        # Inner optimization (μ* solve)
        mu_star = opt_joint(pdg_with_attention, gamma=0.1, iters=5, verbose=False)
        
        # Outer optimization (θ update)
        for _ in range(5):  # outer_iters
            opt.zero_grad()
            loss = torch_score(pdg_with_attention, mu_star, 0.1)
            loss.backward()
            opt.step()
        
        # Track metrics
        current_global = float(torch_score(pdg_copy, mu_star, 0.01))
        current_local = float(torch_score(pdg_copy, mu_star, 0.001))
        
        global_inconsistency.append(current_global)
        local_inconsistency.append(current_local)
        training_steps.append(t + 1)
        
        mu_current = mu_star.data.detach().clone()
    
    return {
        'pattern': spec['pattern'],
        'pdg_name': spec['name'],
        'alpha': alpha,
        'beta': beta,
        'training_steps': training_steps,
        'global_inconsistency': global_inconsistency,
        'local_inconsistency': local_inconsistency,
        'initial_global': initial_global,
        'initial_local': initial_local,
        'final_global': global_inconsistency[-1],
        'final_local': local_inconsistency[-1]
    }


def compare_inconsistency_during_training():
    """Compare local vs global inconsistency during training."""
    print("=== Comparing Local vs Global Inconsistency During Training ===")
    
    # Load dataset
    print("Loading simple dataset...")
    pdgs, specs = load_simple_dataset()
    print(f"Loaded {len(pdgs)} PDGs")
    
    # Best hyperparameters from the search
    best_hyperparams = {
        'chain': {'alpha': 0.5, 'beta': 0.5},
        'star': {'alpha': 2.0, 'beta': 0.5},
        'tree': {'alpha': 0.5, 'beta': 0.5},
        'random': {'alpha': 2.0, 'beta': 0.5},
        'cycle': {'alpha': 0.5, 'beta': 0.5}
    }
    
    # Track training for each PDG
    results = []
    
    for pdg, spec in zip(pdgs, specs):
        pattern = spec['pattern']
        alpha = best_hyperparams[pattern]['alpha']
        beta = best_hyperparams[pattern]['beta']
        
        print(f"\n--- Tracking {spec['name']} ({pattern} pattern) ---")
        print(f"  Description: {spec['description']}")
        print(f"  Using optimal hyperparameters: α={alpha}, β={beta}")
        
        try:
            result = track_inconsistency_during_training(pdg, spec, alpha, beta, num_steps=15)
            results.append(result)
            
            print(f"    Initial: Global={result['initial_global']:.6f}, Local={result['initial_local']:.6f}")
            print(f"    Final:   Global={result['final_global']:.6f}, Local={result['final_local']:.6f}")
            
        except Exception as e:
            print(f"    Error: {e}")
    
    if results:
        # Create visualizations
        create_training_comparison_plots(results)
        
        # Save results
        with open('inconsistency_training_comparison.json', 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\nResults saved to 'inconsistency_training_comparison.json'")
        print("Visualizations saved to 'inconsistency_training_*.png'")
    else:
        print("\nNo successful results to visualize")
    
    return results


def create_training_comparison_plots(results: List[Dict]):
    """Create plots comparing local vs global inconsistency during training."""
    
    # Plot 1: All patterns on one plot
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Local vs Global Inconsistency During Training', fontsize=16, fontweight='bold')
    
    # Plot 1: Global inconsistency over time
    ax1 = axes[0, 0]
    for result in results:
        ax1.plot(result['training_steps'], result['global_inconsistency'], 
                marker='o', label=f"{result['pattern']} (α={result['alpha']}, β={result['beta']})", 
                linewidth=2, markersize=4)
    
    ax1.set_xlabel('Training Step')
    ax1.set_ylabel('Global Inconsistency')
    ax1.set_title('Global Inconsistency Evolution')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Local inconsistency over time
    ax2 = axes[0, 1]
    for result in results:
        ax2.plot(result['training_steps'], result['local_inconsistency'], 
                marker='s', label=f"{result['pattern']} (α={result['alpha']}, β={result['beta']})", 
                linewidth=2, markersize=4)
    
    ax2.set_xlabel('Training Step')
    ax2.set_ylabel('Local Inconsistency')
    ax2.set_title('Local Inconsistency Evolution')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Improvement comparison
    ax3 = axes[1, 0]
    patterns = [r['pattern'] for r in results]
    global_improvements = []
    local_improvements = []
    
    for result in results:
        # Calculate improvement as absolute change
        global_improvement = abs(result['final_global'] - result['initial_global'])
        local_improvement = abs(result['final_local'] - result['initial_local'])
        
        global_improvements.append(global_improvement)
        local_improvements.append(local_improvement)
    
    x = np.arange(len(patterns))
    width = 0.35
    
    bars1 = ax3.bar(x - width/2, global_improvements, width, label='Global', alpha=0.7)
    bars2 = ax3.bar(x + width/2, local_improvements, width, label='Local', alpha=0.7)
    
    ax3.set_xlabel('Pattern')
    ax3.set_ylabel('Absolute Improvement')
    ax3.set_title('Absolute Improvement Comparison')
    ax3.set_xticks(x)
    ax3.set_xticklabels(patterns, rotation=45)
    ax3.legend()
    
    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height + 0.001,
                    f'{height:.4f}', ha='center', va='bottom', fontsize=8)
    
    # Plot 4: Individual pattern details
    ax4 = axes[1, 1]
    ax4.axis('tight')
    ax4.axis('off')
    
    # Create summary table
    table_data = []
    for result in results:
        global_change = result['final_global'] - result['initial_global']
        local_change = result['final_local'] - result['initial_local']
        
        table_data.append([
            result['pattern'],
            f"{result['alpha']}",
            f"{result['beta']}",
            f"{global_change:+.4f}",
            f"{local_change:+.4f}"
        ])
    
    table = ax4.table(cellText=table_data, 
                     colLabels=['Pattern', 'α', 'β', 'Global Δ', 'Local Δ'],
                     cellLoc='center', loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.5)
    ax4.set_title('Change Summary (Final - Initial)')
    
    plt.tight_layout()
    plt.savefig('inconsistency_training_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Create individual plots for each pattern
    create_individual_pattern_plots(results)


def create_individual_pattern_plots(results: List[Dict]):
    """Create individual plots for each pattern."""
    n_patterns = len(results)
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Individual Pattern Training Evolution', fontsize=16, fontweight='bold')
    
    # Flatten axes for easier indexing
    axes_flat = axes.flatten()
    
    for i, result in enumerate(results):
        ax = axes_flat[i]
        
        # Plot both global and local on the same subplot
        ax.plot(result['training_steps'], result['global_inconsistency'], 
               marker='o', label='Global', linewidth=2, markersize=4, color='blue')
        ax.plot(result['training_steps'], result['local_inconsistency'], 
               marker='s', label='Local', linewidth=2, markersize=4, color='red')
        
        ax.set_xlabel('Training Step')
        ax.set_ylabel('Inconsistency')
        ax.set_title(f"{result['pattern'].title()} Pattern\nα={result['alpha']}, β={result['beta']}")
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # Hide unused subplots
    for i in range(n_patterns, len(axes_flat)):
        axes_flat[i].axis('off')
    
    plt.tight_layout()
    plt.savefig('inconsistency_training_individual_patterns.png', dpi=300, bbox_inches='tight')
    plt.show()


def main():
    """Main function."""
    results = compare_inconsistency_during_training()
    return results


if __name__ == "__main__":
    results = main()
