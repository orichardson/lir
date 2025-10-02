#!/usr/bin/env python3
"""
Test attention functionality with the best hyperparameters found in the search.
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


def create_optimal_attention_strategy(alpha: float, beta: float): #  
    """Create an attention strategy with optimal alpha and beta values."""
    def refocus(M: PDG, t: int):
        """Attention strategy with optimal alpha and beta."""
        attn_alpha = {}
        attn_beta = {}
        control = {}
        
        learnables = _collect_learnables(M)
        
        for label in learnables.keys():
            attn_alpha[label] = alpha
            attn_beta[label] = beta
        
        return attn_alpha, attn_beta, control
    
    return refocus 


def test_attention_with_best_hyperparams():
    """Test attention functionality with the best hyperparameters."""
    print("=== Testing Attention with Best Hyperparameters ===")
    
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
    
    print("\nBest hyperparameters by pattern:")
    for pattern, params in best_hyperparams.items():
        print(f"  {pattern}: α={params['alpha']}, β={params['beta']}")
    
    # Test each PDG with its optimal hyperparameters
    results = []
    
    for pdg, spec in zip(pdgs, specs):
        pattern = spec['pattern']
        alpha = best_hyperparams[pattern]['alpha']
        beta = best_hyperparams[pattern]['beta']
        
        print(f"\n--- Testing {spec['name']} ({pattern} pattern) ---")
        print(f"  Description: {spec['description']}")
        print(f"  Variables: {spec['num_vars']}, Edges: {spec['num_edges']}")
        print(f"  Using optimal hyperparameters: α={alpha}, β={beta}")
        
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
        print("  Computing initial metrics...")
        mu_init = opt_joint(pdg_copy, gamma=0.1, iters=20, verbose=False)
        initial_global_inconsistency = float(torch_score(pdg_copy, mu_init, 0.001))
        initial_local_inconsistency = float(torch_score(pdg_copy, mu_init, 0.0001))
        
        print(f"    Initial global inconsistency: {initial_global_inconsistency:.6f}")
        print(f"    Initial local inconsistency: {initial_local_inconsistency:.6f}")
        
        # Create attention strategy with optimal hyperparameters
        attention_strategy = create_optimal_attention_strategy(alpha, beta)
        
        # Test attention mask application
        print("  Testing attention mask application...")
        attn_alpha, attn_beta, control = attention_strategy(pdg_copy, 0)
        
        print(f"    Attention masks created:")
        for label in attn_beta.keys():
            print(f"      {label}: α={attn_alpha.get(label, 1.0):.1f}, β={attn_beta[label]:.1f}")
        
        # Apply attention masks
        pdg_with_attention = apply_attn_mask(
            pdg_copy, 
            attn_mask_beta=attn_beta, 
            attn_mask_alpha=attn_alpha
        )
        
        print("  Attention masks applied successfully")
        
        # Run LIR training with optimal hyperparameters
        print("  Running LIR training with optimal attention...")
        try:
            lir_train(
                pdg_with_attention,
                gamma=0.1,
                T=10,  # More steps for better results
                outer_iters=5,
                inner_iters=5,
                lr=0.01,
                refocus=attention_strategy,
                verbose=True,  # Show progress
                mu_init=mu_init
            )
            
            # Compute final metrics
            print("  Computing final metrics...")
            mu_final = opt_joint(pdg_with_attention, gamma=0.1, iters=20, verbose=False)
            final_global_inconsistency = float(torch_score(pdg_with_attention, mu_final, 0.001))
            final_local_inconsistency = float(torch_score(pdg_with_attention, mu_final, 0.0001))
            
            improvement_global = (initial_global_inconsistency - final_global_inconsistency) / initial_global_inconsistency * 100
            improvement_local = (initial_local_inconsistency - final_local_inconsistency) / initial_local_inconsistency * 100
            
            print(f"    Final global inconsistency: {final_global_inconsistency:.6f}")
            print(f"    Final local inconsistency: {final_local_inconsistency:.6f}")
            print(f"    Global improvement: {improvement_global:.1f}%")
            print(f"    Local improvement: {improvement_local:.1f}%")
            
            results.append({
                'pattern': pattern,
                'pdg_name': spec['name'],
                'alpha': alpha,
                'beta': beta,
                'initial_global': initial_global_inconsistency,
                'final_global': final_global_inconsistency,
                'improvement_global': improvement_global,
                'initial_local': initial_local_inconsistency,
                'final_local': final_local_inconsistency,
                'improvement_local': improvement_local,
                'success': True
            })
            
        except Exception as e:
            print(f"    Error during training: {e}")
            results.append({
                'pattern': pattern,
                'pdg_name': spec['name'],
                'alpha': alpha,
                'beta': beta,
                'success': False,
                'error': str(e)
            })
    
    # Print summary
    print("\n=== Results Summary ===")
    successful_results = [r for r in results if r['success']]
    
    if successful_results:
        print(f"\nSuccessful experiments: {len(successful_results)}/{len(results)}")
        
        for result in successful_results:
            print(f"\n{result['pattern'].upper()} Pattern ({result['pdg_name']}):")
            print(f"  Hyperparameters: α={result['alpha']}, β={result['beta']}")
            print(f"  Global improvement: {result['improvement_global']:.1f}%")
            print(f"  Local improvement: {result['improvement_local']:.1f}%")
        
        # Create visualization
        create_results_visualization(successful_results)
        
        # Save results
        with open('best_hyperparameters_test_results.json', 'w') as f:
            json.dump(successful_results, f, indent=2)
        
        print(f"\nResults saved to 'best_hyperparameters_test_results.json'")
        print("Visualization saved to 'best_hyperparameters_test_plot.png'")
    
    return results


def create_results_visualization(results: List[Dict]):
    """Create visualization of the test results."""
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Attention Functionality Test with Best Hyperparameters', fontsize=16, fontweight='bold')
    
    # Convert to DataFrame for easier plotting
    import pandas as pd
    df = pd.DataFrame(results)
    
    # Plot 1: Global improvement by pattern
    ax1 = axes[0, 0]
    bars1 = ax1.bar(df['pattern'], df['improvement_global'], color='skyblue')
    ax1.set_title('Global Inconsistency Improvement by Pattern')
    ax1.set_ylabel('Improvement (%)')
    ax1.tick_params(axis='x', rotation=45)
    
    # Add value labels on bars
    for bar, value in zip(bars1, df['improvement_global']):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
                f'{value:.1f}%', ha='center', va='bottom')
    
    # Plot 2: Local improvement by pattern
    ax2 = axes[0, 1]
    bars2 = ax2.bar(df['pattern'], df['improvement_local'], color='lightcoral')
    ax2.set_title('Local Inconsistency Improvement by Pattern')
    ax2.set_ylabel('Improvement (%)')
    ax2.tick_params(axis='x', rotation=45)
    
    # Add value labels on bars
    for bar, value in zip(bars2, df['improvement_local']):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
                f'{value:.1f}%', ha='center', va='bottom')
    
    # Plot 3: Hyperparameters used
    ax3 = axes[1, 0]
    ax3.axis('tight')
    ax3.axis('off')
    
    # Create hyperparameters table
    hyperparams_data = []
    for _, row in df.iterrows():
        hyperparams_data.append([
            row['pattern'],
            f"{row['alpha']}",
            f"{row['beta']}",
            f"{row['improvement_global']:.1f}%"
        ])
    
    table = ax3.table(cellText=hyperparams_data, 
                     colLabels=['Pattern', 'Alpha (α)', 'Beta (β)', 'Global %'],
                     cellLoc='center', loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1.2, 1.5)
    ax3.set_title('Optimal Hyperparameters and Results')
    
    # Plot 4: Before vs After comparison
    ax4 = axes[1, 1]
    x = np.arange(len(df))
    width = 0.35
    
    bars1 = ax4.bar(x - width/2, df['initial_global'], width, label='Initial', alpha=0.7)
    bars2 = ax4.bar(x + width/2, df['final_global'], width, label='Final', alpha=0.7)
    
    ax4.set_xlabel('Pattern')
    ax4.set_ylabel('Global Inconsistency')
    ax4.set_title('Before vs After Training')
    ax4.set_xticks(x)
    ax4.set_xticklabels(df['pattern'], rotation=45)
    ax4.legend()
    
    plt.tight_layout()
    plt.savefig('best_hyperparameters_test_plot.png', dpi=300, bbox_inches='tight')
    plt.show()


def main():
    """Main function."""
    results = test_attention_with_best_hyperparams()
    return results


if __name__ == "__main__":
    results = main()
