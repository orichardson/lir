#!/usr/bin/env python3
"""
Hyperparameter search for LIR training focusing on learning rate, iterations, 
optimizer/ODE solver, and attention/control mask strategies.

α and β are properties of PDGs, not hyperparameters.
The relevant hyperparameters are:
- Learning rate
- Number of inner and outer iterations  
- Optimizer/ODE solver
- Attention/control masks 
"""

import sys
from pathlib import Path
import pickle
import json
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import torch
from typing import List, Dict, Any, Tuple, Callable
import itertools
from dataclasses import dataclass
import pandas as pd

# Add project paths
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).parent))

from pdg.pdg import PDG
from pdg.dist import ParamCPD
from pdg.alg.torch_opt import opt_joint, torch_score
from lir__simpler import lir_train, _collect_learnables


@dataclass
class HyperparameterConfigV2:
    """Configuration for hyperparameter search focusing on training hyperparameters."""
    # Learning rates to test
    learning_rates: List[float]
    
    # Iteration counts to test
    outer_iterations: List[int]
    inner_iterations: List[int]
    
    # Optimizers to test
    optimizers: List[Tuple[str, Dict[str, Any]]]  # (name, kwargs)
    
    # Attention/control mask strategies to test
    attention_strategies: List[Tuple[str, Callable]]  # (name, strategy_function)
    
    # Fixed parameters (not hyperparameters)
    num_steps: int = 10
    gamma: float = 0.1
    alpha: float = 1.0  # Fixed alpha (property of PDG)
    beta: float = 1.0   # Fixed beta (property of PDG)


def load_simple_dataset():
    """Load the simple PDG dataset."""
    dataset_path = Path(__file__).parent / 'simple_pdg_dataset'
    
    with open(dataset_path / 'pdgs.pkl', 'rb') as f:
        pdgs = pickle.load(f)
    
    with open(dataset_path / 'specs.json', 'r') as f:
        specs = json.load(f)
    
    return pdgs, specs


def create_attention_strategy_fixed_alpha_beta(alpha: float, beta: float):
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


def create_attention_strategy_uniform():
    """Uniform attention strategy - all edges get equal attention."""
    def attention_strategy(M: PDG, t: int):
        attn_alpha = {}
        attn_beta = {}
        control = {}
        
        learnables = _collect_learnables(M)
        
        for label in learnables.keys():
            attn_alpha[label] = 1.0
            attn_beta[label] = 1.0
        
        return attn_alpha, attn_beta, control
    
    return attention_strategy


def create_attention_strategy_random():
    """Random attention strategy - random attention weights."""
    def attention_strategy(M: PDG, t: int):
        attn_alpha = {}
        attn_beta = {}
        control = {}
        
        learnables = _collect_learnables(M)
        
        for label in learnables.keys():
            attn_alpha[label] = np.random.uniform(0.5, 2.0)
            attn_beta[label] = np.random.uniform(0.5, 2.0)
        
        return attn_alpha, attn_beta, control
    
    return attention_strategy


def create_attention_strategy_adaptive():
    """Adaptive attention strategy - changes over time."""
    def attention_strategy(M: PDG, t: int):
        attn_alpha = {}
        attn_beta = {}
        control = {}
        
        learnables = _collect_learnables(M)
        
        # Adaptive scaling based on time step
        time_factor = 1.0 + 0.1 * t
        
        for i, label in enumerate(learnables.keys()):
            # Different edges get different attention based on their index and time
            attn_alpha[label] = 1.0 + 0.2 * np.sin(i + t * 0.1)
            attn_beta[label] = 1.0 + 0.2 * np.cos(i + t * 0.1)
        
        return attn_alpha, attn_beta, control
    
    return attention_strategy


def create_attention_strategy_selective():
    """Selective attention strategy - focus on some edges, ignore others."""
    def attention_strategy(M: PDG, t: int):
        attn_alpha = {}
        attn_beta = {}
        control = {}
        
        learnables = _collect_learnables(M)
        learnable_labels = list(learnables.keys())
        
        # Focus on first half of edges, ignore second half
        focus_threshold = len(learnable_labels) // 2
        
        for i, label in enumerate(learnable_labels):
            if i < focus_threshold:
                attn_alpha[label] = 2.0  # High attention
                attn_beta[label] = 2.0
            else:
                attn_alpha[label] = 0.1  # Low attention
                attn_beta[label] = 0.1
        
        return attn_alpha, attn_beta, control
    
    return attention_strategy


def create_attention_strategy_control_freeze():
    """Control strategy that freezes some parameters."""
    def attention_strategy(M: PDG, t: int):
        attn_alpha = {}
        attn_beta = {}
        control = {}
        
        learnables = _collect_learnables(M)
        learnable_labels = list(learnables.keys())
        
        # Freeze every other parameter
        for i, label in enumerate(learnable_labels):
            attn_alpha[label] = 1.0
            attn_beta[label] = 1.0
            
            if i % 2 == 0:
                control[label] = 0.0  # Freeze
            else:
                control[label] = 1.0  # Allow updates
        
        return attn_alpha, attn_beta, control
    
    return attention_strategy


def get_default_attention_strategies():
    """Get the default set of attention strategies to test."""
    return [
        ("uniform", create_attention_strategy_uniform()),
        ("random", create_attention_strategy_random()),
        ("adaptive", create_attention_strategy_adaptive()),
        ("selective", create_attention_strategy_selective()),
        ("control_freeze", create_attention_strategy_control_freeze()),
    ]


def get_default_optimizers():
    """Get the default set of optimizers to test."""
    return [
        ("Adam", {"lr": 1.0}),  # lr will be overridden by the learning rate parameter
        ("SGD", {"lr": 1.0, "momentum": 0.9}),
        ("AdamW", {"lr": 1.0, "weight_decay": 0.01}),
        ("RMSprop", {"lr": 1.0, "alpha": 0.99}),
    ]


def run_experiment_with_hyperparams_v2(pdg: PDG, spec: Dict, lr: float, outer_iters: int, 
                                     inner_iters: int, optimizer_name: str, optimizer_kwargs: Dict,
                                     attention_strategy_name: str, attention_strategy: Callable,
                                     config: HyperparameterConfigV2) -> Dict[str, Any]:
    """Run a single experiment with specific hyperparameters."""
    
    # Create a copy of the PDG to avoid modifying the original
    pdg_copy = pdg.copy()
    
    # Make PDG parametric by converting CPDs to ParamCPD objects
    edges_snapshot = list(pdg_copy.edges("l,X,Y,α,β,P"))
    
    for L, X, Y, α, β, P in edges_snapshot:
        if L[0] == "π":
            # π edges are not learnable, keep as is
            continue
        else:
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
    
    # Initialize parameters
    mu_init = opt_joint(pdg_copy, gamma=config.gamma, iters=20, verbose=False)
    initial_global_inconsistency = float(torch_score(pdg_copy, mu_init, 0.001))
    initial_local_inconsistency = float(torch_score(pdg_copy, mu_init, 0.0001))
    
    # Create attention strategy with fixed alpha and beta
    attention_strategy_with_alpha_beta = create_attention_strategy_fixed_alpha_beta(
        config.alpha, config.beta
    )
    
    # Combine with the specific attention strategy
    def combined_strategy(M: PDG, t: int):
        # Get base attention from fixed alpha/beta strategy
        attn_alpha_base, attn_beta_base, control_base = attention_strategy_with_alpha_beta(M, t)
        
        # Get specific attention strategy
        attn_alpha_spec, attn_beta_spec, control_spec = attention_strategy(M, t)
        
        # Combine them (multiply the attention values)
        attn_alpha = {}
        attn_beta = {}
        control = {}
        
        for label in attn_alpha_base.keys():
            attn_alpha[label] = attn_alpha_base[label] * attn_alpha_spec.get(label, 1.0)
            attn_beta[label] = attn_beta_base[label] * attn_beta_spec.get(label, 1.0)
            control[label] = control_spec.get(label, 1.0)
        
        return attn_alpha, attn_beta, control
    
    # Get optimizer constructor
    optimizer_ctor = getattr(torch.optim, optimizer_name)
    
    # Remove lr from optimizer_kwargs to avoid duplicate lr parameter
    opt_kwargs_clean = optimizer_kwargs.copy()
    if 'lr' in opt_kwargs_clean:
        del opt_kwargs_clean['lr']
    
    # Run LIR training
    try:
        lir_train(
            pdg_copy,
            gamma=config.gamma,
            T=config.num_steps,
            outer_iters=outer_iters,
            inner_iters=inner_iters,
            lr=lr,
            optimizer_ctor=optimizer_ctor,
            opt_kwargs=opt_kwargs_clean,
            refocus=combined_strategy,
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
            'learning_rate': lr,
            'outer_iters': outer_iters,
            'inner_iters': inner_iters,
            'optimizer': optimizer_name,
            'optimizer_kwargs': optimizer_kwargs,
            'attention_strategy': attention_strategy_name,
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
            'learning_rate': lr,
            'outer_iters': outer_iters,
            'inner_iters': inner_iters,
            'optimizer': optimizer_name,
            'optimizer_kwargs': optimizer_kwargs,
            'attention_strategy': attention_strategy_name,
            'pdg_name': spec['name'],
            'pattern': spec['pattern'],
            'description': spec['description'],
            'num_vars': spec['num_vars'],
            'num_edges': spec['num_edges'],
            'error': str(e),
            'success': False
        }


def run_hyperparameter_search_v2():
    """Run hyperparameter search focusing on training hyperparameters."""
    print("=== Hyperparameter Search V2: Training Hyperparameters ===")
    
    # Load dataset
    print("Loading simple dataset...")
    pdgs, specs = load_simple_dataset()
    print(f"Loaded {len(pdgs)} PDGs")
    
    # Define hyperparameter search space
    config = HyperparameterConfigV2(
        learning_rates=[0.001, 0.01, 0.1],
        outer_iterations=[3, 5, 8],
        inner_iterations=[10, 20, 30],
        optimizers=get_default_optimizers(),
        attention_strategies=get_default_attention_strategies(),
        num_steps=8,  # Shorter for faster search
        gamma=0.1,
        alpha=1.0,  # Fixed alpha
        beta=1.0    # Fixed beta
    )
    
    # Calculate total experiments
    total_combinations = (len(config.learning_rates) * 
                         len(config.outer_iterations) * 
                         len(config.inner_iterations) * 
                         len(config.optimizers) * 
                         len(config.attention_strategies))
    
    total_experiments = len(pdgs) * total_combinations
    
    print(f"Search space:")
    print(f"  Learning rates: {config.learning_rates}")
    print(f"  Outer iterations: {config.outer_iterations}")
    print(f"  Inner iterations: {config.inner_iterations}")
    print(f"  Optimizers: {[opt[0] for opt in config.optimizers]}")
    print(f"  Attention strategies: {[strat[0] for strat in config.attention_strategies]}")
    print(f"  Total combinations per PDG: {total_combinations}")
    print(f"  Total experiments: {total_experiments}")
    
    # Run experiments
    results = []
    experiment_count = 0
    
    for pdg, spec in zip(pdgs, specs):
        print(f"\nRunning hyperparameter search on {spec['name']} ({spec['pattern']} pattern)...")
        print(f"  Description: {spec['description']}")
        print(f"  Variables: {spec['num_vars']}, Edges: {spec['num_edges']}")
        
        for lr, outer_iters, inner_iters, (opt_name, opt_kwargs), (strat_name, strat_func) in itertools.product(
            config.learning_rates, config.outer_iterations, config.inner_iterations,
            config.optimizers, config.attention_strategies
        ):
            experiment_count += 1
            print(f"  [{experiment_count}/{total_experiments}] Testing lr={lr}, outer={outer_iters}, inner={inner_iters}, opt={opt_name}, strat={strat_name}...")
            
            result = run_experiment_with_hyperparams_v2(
                pdg, spec, lr, outer_iters, inner_iters, 
                opt_name, opt_kwargs, strat_name, strat_func, config
            )
            results.append(result)
    
    # Filter successful results
    successful_results = [r for r in results if r['success']]
    
    if not successful_results:
        print("No successful experiments!")
        return
    
    print(f"\nCompleted {len(successful_results)}/{len(results)} successful experiments")
    
    # Analyze results
    analyze_hyperparameter_results_v2(successful_results, config)
    
    # Create visualizations
    create_hyperparameter_visualizations_v2(successful_results, config)
    
    # Save results
    with open('hyperparameter_search_v2_results.json', 'w') as f:
        json.dump(successful_results, f, indent=2)
    
    print(f"\nResults saved to 'hyperparameter_search_v2_results.json'")
    print("Visualizations saved to 'hyperparameter_search_v2_*.png'")
    
    return successful_results


def analyze_hyperparameter_results_v2(results: List[Dict], config: HyperparameterConfigV2):
    """Analyze the hyperparameter search results."""
    print("\n=== Hyperparameter Search V2 Analysis ===")
    
    # Convert to DataFrame for easier analysis
    df = pd.DataFrame(results)
    
    # Find best hyperparameters overall
    best_global = df.loc[df['improvement_global'].idxmax()]
    best_local = df.loc[df['improvement_local'].idxmax()]
    
    print(f"\nBest Global Improvement:")
    print(f"  Learning Rate: {best_global['learning_rate']}")
    print(f"  Outer/Inner Iterations: {best_global['outer_iters']}/{best_global['inner_iters']}")
    print(f"  Optimizer: {best_global['optimizer']}")
    print(f"  Attention Strategy: {best_global['attention_strategy']}")
    print(f"  PDG: {best_global['pdg_name']} ({best_global['pattern']})")
    print(f"  Improvement: {best_global['improvement_global']:.1f}%")
    
    print(f"\nBest Local Improvement:")
    print(f"  Learning Rate: {best_local['learning_rate']}")
    print(f"  Outer/Inner Iterations: {best_local['outer_iters']}/{best_local['inner_iters']}")
    print(f"  Optimizer: {best_local['optimizer']}")
    print(f"  Attention Strategy: {best_local['attention_strategy']}")
    print(f"  PDG: {best_local['pdg_name']} ({best_local['pattern']})")
    print(f"  Improvement: {best_local['improvement_local']:.1f}%")
    
    # Analyze by attention strategy
    print(f"\nAverage performance by attention strategy:")
    strategy_performance = df.groupby('attention_strategy').agg({
        'improvement_global': ['mean', 'std'],
        'improvement_local': ['mean', 'std']
    }).round(1)
    print(strategy_performance)
    
    # Analyze by optimizer
    print(f"\nAverage performance by optimizer:")
    optimizer_performance = df.groupby('optimizer').agg({
        'improvement_global': ['mean', 'std'],
        'improvement_local': ['mean', 'std']
    }).round(1)
    print(optimizer_performance)
    
    # Analyze by learning rate
    print(f"\nAverage performance by learning rate:")
    lr_performance = df.groupby('learning_rate').agg({
        'improvement_global': ['mean', 'std'],
        'improvement_local': ['mean', 'std']
    }).round(1)
    print(lr_performance)
    
    # Find best combinations
    print(f"\nBest attention strategy: {strategy_performance[('improvement_global', 'mean')].idxmax()}")
    print(f"Best optimizer: {optimizer_performance[('improvement_global', 'mean')].idxmax()}")
    print(f"Best learning rate: {lr_performance[('improvement_global', 'mean')].idxmax()}")


def create_hyperparameter_visualizations_v2(results: List[Dict], config: HyperparameterConfigV2):
    """Create visualizations of hyperparameter search results."""
    df = pd.DataFrame(results)
    
    # Create a comprehensive visualization
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    fig.suptitle('Hyperparameter Search V2 Results: Training Hyperparameters', fontsize=16, fontweight='bold')
    
    # Plot 1: Performance by attention strategy
    ax1 = axes[0, 0]
    strategy_performance = df.groupby('attention_strategy')['improvement_global'].mean()
    bars = ax1.bar(strategy_performance.index, strategy_performance.values, color='lightblue')
    ax1.set_title('Average Global Improvement by Attention Strategy')
    ax1.set_ylabel('Improvement (%)')
    ax1.tick_params(axis='x', rotation=45)
    
    # Plot 2: Performance by optimizer
    ax2 = axes[0, 1]
    optimizer_performance = df.groupby('optimizer')['improvement_global'].mean()
    bars = ax2.bar(optimizer_performance.index, optimizer_performance.values, color='lightgreen')
    ax2.set_title('Average Global Improvement by Optimizer')
    ax2.set_ylabel('Improvement (%)')
    
    # Plot 3: Performance by learning rate
    ax3 = axes[0, 2]
    lr_performance = df.groupby('learning_rate')['improvement_global'].mean()
    bars = ax3.bar([str(lr) for lr in lr_performance.index], lr_performance.values, color='lightcoral')
    ax3.set_title('Average Global Improvement by Learning Rate')
    ax3.set_ylabel('Improvement (%)')
    ax3.set_xlabel('Learning Rate')
    
    # Plot 4: Heatmap of learning rate vs attention strategy
    ax4 = axes[1, 0]
    pivot_lr_strat = df.groupby(['learning_rate', 'attention_strategy'])['improvement_global'].mean().unstack()
    sns.heatmap(pivot_lr_strat, annot=True, fmt='.1f', cmap='viridis', ax=ax4)
    ax4.set_title('Learning Rate vs Attention Strategy')
    ax4.set_xlabel('Attention Strategy')
    ax4.set_ylabel('Learning Rate')
    
    # Plot 5: Heatmap of optimizer vs attention strategy
    ax5 = axes[1, 1]
    pivot_opt_strat = df.groupby(['optimizer', 'attention_strategy'])['improvement_global'].mean().unstack()
    sns.heatmap(pivot_opt_strat, annot=True, fmt='.1f', cmap='plasma', ax=ax5)
    ax5.set_title('Optimizer vs Attention Strategy')
    ax5.set_xlabel('Attention Strategy')
    ax5.set_ylabel('Optimizer')
    
    # Plot 6: Performance by pattern
    ax6 = axes[1, 2]
    pattern_performance = df.groupby('pattern')['improvement_global'].mean()
    bars = ax6.bar(pattern_performance.index, pattern_performance.values, color='lightyellow')
    ax6.set_title('Average Global Improvement by Pattern')
    ax6.set_ylabel('Improvement (%)')
    ax6.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig('hyperparameter_search_v2_results.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Create detailed attention strategy comparison
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    fig.suptitle('Detailed Attention Strategy Analysis', fontsize=14, fontweight='bold')
    
    # Box plot of attention strategies
    ax1 = axes[0]
    df.boxplot(column='improvement_global', by='attention_strategy', ax=ax1)
    ax1.set_title('Distribution of Global Improvement by Attention Strategy')
    ax1.set_xlabel('Attention Strategy')
    ax1.set_ylabel('Global Improvement (%)')
    ax1.tick_params(axis='x', rotation=45)
    
    # Scatter plot of attention strategies vs local improvement
    ax2 = axes[1]
    for strategy in df['attention_strategy'].unique():
        strategy_data = df[df['attention_strategy'] == strategy]
        ax2.scatter(strategy_data['improvement_global'], strategy_data['improvement_local'], 
                   label=strategy, alpha=0.6)
    ax2.set_xlabel('Global Improvement (%)')
    ax2.set_ylabel('Local Improvement (%)')
    ax2.set_title('Global vs Local Improvement by Attention Strategy')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('hyperparameter_search_v2_attention_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()


if __name__ == "__main__":
    run_hyperparameter_search_v2()
