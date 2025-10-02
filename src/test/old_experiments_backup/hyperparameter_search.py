#!/usr/bin/env python3
"""
Hyperparameter search for attention strategy alpha and beta values.
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
import itertools
from dataclasses import dataclass

# Add project paths
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).parent))

from pdg.pdg import PDG
from pdg.dist import ParamCPD
from pdg.alg.torch_opt import opt_joint, torch_score
from lir__simpler import lir_train, _collect_learnables


@dataclass
class HyperparameterConfig:
    """Configuration for hyperparameter search."""
    alpha_values: List[float]
    beta_values: List[float]
    num_steps: int = 10
    outer_iters: int = 5
    inner_iters: int = 5
    lr: float = 0.01
    gamma: float = 0.1


def load_simple_dataset():
    """Load the simple PDG dataset."""
    with open('simple_pdg_dataset/pdgs.pkl', 'rb') as f:
        pdgs = pickle.load(f)
    
    with open('simple_pdg_dataset/specs.json', 'r') as f:
        specs = json.load(f)
    
    return pdgs, specs


def create_attention_strategy(alpha: float, beta: float):
    """Create an attention strategy with fixed alpha and beta values."""
    def attention_strategy(M: PDG, t: int):
        """Attention strategy with fixed alpha and beta."""
        attn_alpha = {}
        attn_beta = {}
        control = {}
        
        learnables = _collect_learnables(M)
        
        for label in learnables.keys():
            attn_alpha[label] = alpha
            attn_beta[label] = beta
        
        return attn_alpha, attn_beta, control
    
    return attention_strategy


def run_experiment_with_hyperparams(pdg: PDG, spec: Dict, alpha: float, beta: float, 
                                  config: HyperparameterConfig) -> Dict[str, Any]:
    """Run a single experiment with specific alpha and beta values."""
    
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
    mu_init = opt_joint(pdg_copy, gamma=config.gamma, iters=20, verbose=False)
    initial_global_inconsistency = float(torch_score(pdg_copy, mu_init, 0.001))
    initial_local_inconsistency = float(torch_score(pdg_copy, mu_init, 0.0001))
    
    # Create attention strategy with specific alpha and beta
    attention_strategy = create_attention_strategy(alpha, beta)
    
    # Run LIR training
    try:
        lir_train(
            pdg_copy,
            gamma=config.gamma,
            T=config.num_steps,
            outer_iters=config.outer_iters,
            inner_iters=config.inner_iters,
            lr=config.lr,
            refocus=attention_strategy,
            verbose=False,
            mu_init=mu_init
        )
        
        # Compute final metrics
        mu_final = opt_joint(pdg_copy, gamma=config.gamma, iters=20, verbose=False)
        final_global_inconsistency = float(torch_score(pdg_copy, mu_final, 0.001))
        final_local_inconsistency = float(torch_score(pdg_copy, mu_final, 0.0001))
        
        improvement_global = (initial_global_inconsistency - final_global_inconsistency) / initial_global_inconsistency * 100
        improvement_local = (initial_local_inconsistency - final_local_inconsistency) / initial_local_inconsistency * 100
        
        return {
            'alpha': alpha,
            'beta': beta,
            'pdg_name': spec['name'],
            'pattern': spec['pattern'],
            'description': spec['description'],
            'num_vars': spec['num_vars'],
            'num_edges': spec['num_edges'],
            'initial_global': initial_global_inconsistency,
            'final_global': final_global_inconsistency,
            'improvement_global': improvement_global,
            'initial_local': initial_local_inconsistency,
            'final_local': final_local_inconsistency,
            'improvement_local': improvement_local,
            'success': True
        }
        
    except Exception as e:
        return {
            'alpha': alpha,
            'beta': beta,
            'pdg_name': spec['name'],
            'pattern': spec['pattern'],
            'description': spec['description'],
            'num_vars': spec['num_vars'],
            'num_edges': spec['num_edges'],
            'success': False,
            'error': str(e)
        }


def run_hyperparameter_search():
    """Run hyperparameter search on the simple dataset."""
    print("=== Hyperparameter Search for Attention Strategy ===")
    
    # Load dataset
    print("Loading simple dataset...")
    pdgs, specs = load_simple_dataset()
    print(f"Loaded {len(pdgs)} PDGs")
    
    # Define hyperparameter search space
    config = HyperparameterConfig(
        alpha_values=[0.5, 1.0, 1.5, 2.0],
        beta_values=[0.5, 1.0, 1.5, 2.0, 2.5],
        num_steps=8,  # Shorter for faster search
        outer_iters=3,
        inner_iters=3,
        lr=0.01,
        gamma=0.1
    )
    
    print(f"Search space: {len(config.alpha_values)} alpha values × {len(config.beta_values)} beta values = {len(config.alpha_values) * len(config.beta_values)} combinations")
    print(f"Alpha values: {config.alpha_values}")
    print(f"Beta values: {config.beta_values}")
    
    # Run experiments
    results = []
    total_experiments = len(pdgs) * len(config.alpha_values) * len(config.beta_values)
    experiment_count = 0
    
    for pdg, spec in zip(pdgs, specs):
        print(f"\nRunning hyperparameter search on {spec['name']} ({spec['pattern']} pattern)...")
        print(f"  Description: {spec['description']}")
        print(f"  Variables: {spec['num_vars']}, Edges: {spec['num_edges']}")
        
        for alpha, beta in itertools.product(config.alpha_values, config.beta_values):
            experiment_count += 1
            print(f"  [{experiment_count}/{total_experiments}] Testing α={alpha}, β={beta}...")
            
            result = run_experiment_with_hyperparams(pdg, spec, alpha, beta, config)
            results.append(result)
    
    # Filter successful results
    successful_results = [r for r in results if r['success']]
    
    if not successful_results:
        print("No successful experiments!")
        return
    
    print(f"\nCompleted {len(successful_results)}/{len(results)} successful experiments")
    
    # Analyze results
    analyze_hyperparameter_results(successful_results, config)
    
    # Create visualizations
    create_hyperparameter_visualizations(successful_results, config)
    
    # Save results
    with open('hyperparameter_search_results.json', 'w') as f:
        json.dump(successful_results, f, indent=2)
    
    print(f"\nResults saved to 'hyperparameter_search_results.json'")
    print("Visualizations saved to 'hyperparameter_search_*.png'")
    
    return successful_results


def analyze_hyperparameter_results(results: List[Dict], config: HyperparameterConfig):
    """Analyze the hyperparameter search results."""
    print("\n=== Hyperparameter Search Analysis ===")
    
    # Convert to DataFrame for easier analysis
    import pandas as pd
    df = pd.DataFrame(results)
    
    # Find best hyperparameters overall
    best_global = df.loc[df['improvement_global'].idxmax()]
    best_local = df.loc[df['improvement_local'].idxmax()]
    
    print(f"\nBest Global Improvement:")
    print(f"  α={best_global['alpha']}, β={best_global['beta']}")
    print(f"  PDG: {best_global['pdg_name']} ({best_global['pattern']})")
    print(f"  Improvement: {best_global['improvement_global']:.1f}%")
    
    print(f"\nBest Local Improvement:")
    print(f"  α={best_local['alpha']}, β={best_local['beta']}")
    print(f"  PDG: {best_local['pdg_name']} ({best_local['pattern']})")
    print(f"  Improvement: {best_local['improvement_local']:.1f}%")
    
    # Analyze by pattern
    print(f"\nBest hyperparameters by pattern:")
    for pattern in df['pattern'].unique():
        pattern_data = df[df['pattern'] == pattern]
        best_pattern = pattern_data.loc[pattern_data['improvement_global'].idxmax()]
        print(f"  {pattern}: α={best_pattern['alpha']}, β={best_pattern['beta']} "
              f"(Global: {best_pattern['improvement_global']:.1f}%)")
    
    # Average performance by hyperparameters
    print(f"\nAverage performance by hyperparameters:")
    avg_performance = df.groupby(['alpha', 'beta']).agg({
        'improvement_global': 'mean',
        'improvement_local': 'mean'
    }).round(1)
    
    print(avg_performance)
    
    # Find best average hyperparameters
    best_avg_global = avg_performance.loc[avg_performance['improvement_global'].idxmax()]
    best_avg_local = avg_performance.loc[avg_performance['improvement_local'].idxmax()]
    
    print(f"\nBest average global improvement: α={best_avg_global.name[0]}, β={best_avg_global.name[1]} "
          f"({best_avg_global['improvement_global']:.1f}%)")
    print(f"Best average local improvement: α={best_avg_local.name[0]}, β={best_avg_local.name[1]} "
          f"({best_avg_local['improvement_local']:.1f}%)")


def create_hyperparameter_visualizations(results: List[Dict], config: HyperparameterConfig):
    """Create visualizations of hyperparameter search results."""
    import pandas as pd
    df = pd.DataFrame(results)
    
    # Create a comprehensive visualization
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Hyperparameter Search Results for Attention Strategy', fontsize=16, fontweight='bold')
    
    # Plot 1: Heatmap of average global improvement
    ax1 = axes[0, 0]
    pivot_global = df.groupby(['alpha', 'beta'])['improvement_global'].mean().unstack()
    sns.heatmap(pivot_global, annot=True, fmt='.1f', cmap='viridis', ax=ax1)
    ax1.set_title('Average Global Improvement by α and β')
    ax1.set_xlabel('Beta (β)')
    ax1.set_ylabel('Alpha (α)')
    
    # Plot 2: Heatmap of average local improvement
    ax2 = axes[0, 1]
    pivot_local = df.groupby(['alpha', 'beta'])['improvement_local'].mean().unstack()
    sns.heatmap(pivot_local, annot=True, fmt='.1f', cmap='plasma', ax=ax2)
    ax2.set_title('Average Local Improvement by α and β')
    ax2.set_xlabel('Beta (β)')
    ax2.set_ylabel('Alpha (α)')
    
    # Plot 3: Performance by pattern
    ax3 = axes[0, 2]
    pattern_performance = df.groupby('pattern')['improvement_global'].mean()
    bars = ax3.bar(pattern_performance.index, pattern_performance.values, color='lightblue')
    ax3.set_title('Average Global Improvement by Pattern')
    ax3.set_ylabel('Improvement (%)')
    ax3.tick_params(axis='x', rotation=45)
    
    # Add value labels on bars
    for bar, value in zip(bars, pattern_performance.values):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, 
                f'{value:.1f}%', ha='center', va='bottom')
    
    # Plot 4: Alpha vs Beta scatter plot
    ax4 = axes[1, 0]
    scatter = ax4.scatter(df['alpha'], df['beta'], c=df['improvement_global'], 
                         cmap='viridis', alpha=0.7, s=50)
    ax4.set_xlabel('Alpha (α)')
    ax4.set_ylabel('Beta (β)')
    ax4.set_title('Global Improvement by α and β')
    plt.colorbar(scatter, ax=ax4, label='Global Improvement (%)')
    
    # Plot 5: Distribution of improvements
    ax5 = axes[1, 1]
    ax5.hist(df['improvement_global'], bins=20, alpha=0.7, edgecolor='black', color='lightblue')
    ax5.set_xlabel('Global Improvement (%)')
    ax5.set_ylabel('Frequency')
    ax5.set_title('Distribution of Global Improvements')
    
    # Plot 6: Best hyperparameters summary table
    ax6 = axes[1, 2]
    ax6.axis('tight')
    ax6.axis('off')
    
    # Create summary table
    summary_data = []
    for pattern in df['pattern'].unique():
        pattern_data = df[df['pattern'] == pattern]
        best_pattern = pattern_data.loc[pattern_data['improvement_global'].idxmax()]
        summary_data.append([
            pattern,
            f"{best_pattern['alpha']}",
            f"{best_pattern['beta']}",
            f"{best_pattern['improvement_global']:.1f}%"
        ])
    
    table = ax6.table(cellText=summary_data, 
                     colLabels=['Pattern', 'Best α', 'Best β', 'Global %'],
                     cellLoc='center', loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.5)
    ax6.set_title('Best Hyperparameters by Pattern')
    
    plt.tight_layout()
    plt.savefig('hyperparameter_search_results.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Create individual heatmaps for each pattern
    create_pattern_specific_heatmaps(df)


def create_pattern_specific_heatmaps(df):
    """Create individual heatmaps for each pattern."""
    patterns = df['pattern'].unique()
    n_patterns = len(patterns)
    
    fig, axes = plt.subplots(1, n_patterns, figsize=(5 * n_patterns, 4))
    if n_patterns == 1:
        axes = [axes]
    
    fig.suptitle('Global Improvement Heatmaps by Pattern', fontsize=16, fontweight='bold')
    
    for i, pattern in enumerate(patterns):
        pattern_data = df[df['pattern'] == pattern]
        pivot = pattern_data.groupby(['alpha', 'beta'])['improvement_global'].mean().unstack()
        
        sns.heatmap(pivot, annot=True, fmt='.1f', cmap='viridis', ax=axes[i])
        axes[i].set_title(f'{pattern.title()} Pattern')
        axes[i].set_xlabel('Beta (β)')
        if i == 0:
            axes[i].set_ylabel('Alpha (α)')
        else:
            axes[i].set_ylabel('')
    
    plt.tight_layout()
    plt.savefig('hyperparameter_search_by_pattern.png', dpi=300, bbox_inches='tight')
    plt.show()


def main():
    """Main function."""
    results = run_hyperparameter_search()
    return results


if __name__ == "__main__":
    results = main()
