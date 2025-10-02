#!/usr/bin/env python3
"""
Streamlined experimental setup for LIR (Local Inconsistency Resolution) research.

This module provides a single, clean experimental framework that:
1. Generates PDG datasets with varying numbers of variables and edges
2. Supports multiple edges between the same nodes with different CPDs
3. Implements four attention strategies for beta values
4. Computes global and local inconsistency measures
"""

import sys
from pathlib import Path
import random
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
from typing import List, Dict, Tuple, Any, Callable
import json
import pickle
from dataclasses import dataclass, asdict
import itertools

# Add project paths
sys.path.insert(0, str(Path(__file__).resolve().parents[0]))

from pdg.pdg import PDG
from pdg.rv import Variable as Var
from pdg.dist import CPT, ParamCPD
from pdg.alg.torch_opt import opt_joint, torch_score
from lir__simpler import lir_train, _collect_learnables, apply_attn_mask


@dataclass
class PDGSpec:
    """Specification for generating a PDG."""
    name: str
    num_vars: int
    num_edges: int
    val_range: Tuple[int, int]
    max_edges_per_pair: int = 3  # Maximum edges between same node pair
    seed: int = 42
    description: str = ""


@dataclass
class ExperimentResult:
    """Result of a single experiment."""
    pdg_name: str
    num_vars: int
    num_edges: int
    attention_strategy: str
    initial_global_inconsistency: float
    final_global_inconsistency: float
    initial_local_inconsistency: float
    final_local_inconsistency: float
    improvement_global: float
    improvement_local: float
    success: bool
    error: str = ""


class PDGGenerator:
    """Generator for creating PDGs with varying structures and multiple edges."""
    
    def __init__(self, seed: int = 42):
        self.seed = seed
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
    
    def generate_pdg(self, spec: PDGSpec) -> PDG:
        """Generate a PDG based on specification with support for multiple edges between nodes."""
        random.seed(spec.seed)
        np.random.seed(spec.seed)
        torch.manual_seed(spec.seed)
        
        pdg = PDG()
        varlist: List[Var] = []
        
        # Create variables
        for i in range(spec.num_vars):
            domain_size = random.randint(*spec.val_range)
            var = Var.alph(chr(65 + i), domain_size)
            pdg += var
            varlist.append(var)
        
        # Generate edges with potential for multiple edges between same nodes
        edges_added = 0
        max_attempts = spec.num_edges * 10  # Prevent infinite loops
        attempts = 0
        
        while edges_added < spec.num_edges and attempts < max_attempts:
            attempts += 1
            
            # Randomly select source and target variables
            src_idx = random.randint(0, spec.num_vars - 1)
            tgt_idx = random.randint(0, spec.num_vars - 1)
            
            # Skip self-loops
            if src_idx == tgt_idx:
                continue
            
            src = varlist[src_idx]
            tgt = varlist[tgt_idx]
            
            # Check how many edges already exist between these nodes
            existing_edges = 0
            for L, X, Y, α, β, P in pdg.edges("l,X,Y,α,β,P"):
                if (X.name == src.name and Y.name == tgt.name) or \
                   (X.name == tgt.name and Y.name == src.name):
                    existing_edges += 1
            
            # Add edge if we haven't exceeded the limit
            if existing_edges < spec.max_edges_per_pair:
                try:
                    pdg += CPT.make_random(Var.product([src]), Var.product([tgt]))
                    edges_added += 1
                except Exception:
                    # Skip if edge creation fails
                    continue
        
        return pdg


class AttentionStrategy:
    """Collection of attention strategies for beta values."""
    
    @staticmethod
    def global_strategy(pdg: PDG, t: int) -> Tuple[Dict, Dict, Dict]:
        """Global strategy: β = 1 for all edges."""
        attn_alpha = {}
        attn_beta = {}
        control = {}
        
        for L, X, Y, α, β, P in pdg.edges("l,X,Y,α,β,P"):
            attn_alpha[L] = 0.0  # α = 0 as specified
            attn_beta[L] = 1.0   # β = 1 for all edges
            control[L] = 1.0
        
        return attn_alpha, attn_beta, control
    
    @staticmethod
    def local_strategy(pdg: PDG, t: int) -> Tuple[Dict, Dict, Dict]:
        """Local strategy: β = 1 for some edges, 0 for others."""
        attn_alpha = {}
        attn_beta = {}
        control = {}
        
        edges = list(pdg.edges("l,X,Y,α,β,P"))
        # Select half of the edges to have β = 1
        selected_edges = random.sample(edges, len(edges) // 2)
        selected_labels = {L for L, X, Y, α, β, P in selected_edges}
        
        for L, X, Y, α, β, P in edges:
            attn_alpha[L] = 0.0  # α = 0 as specified
            attn_beta[L] = 1.0 if L in selected_labels else 0.0
            control[L] = 1.0
        
        return attn_alpha, attn_beta, control
    
    @staticmethod
    def node_based_strategy(pdg: PDG, t: int) -> Tuple[Dict, Dict, Dict]:
        """Node-based strategy: β = 1 for edges connected to current node, 0 for others."""
        attn_alpha = {}
        attn_beta = {}
        control = {}
        
        # Select a random node to focus on
        varlist = list(pdg.vars.values())
        if not varlist:
            return attn_alpha, attn_beta, control
        
        focus_node = random.choice(varlist)
        
        for L, X, Y, α, β, P in pdg.edges("l,X,Y,α,β,P"):
            attn_alpha[L] = 0.0  # α = 0 as specified
            # β = 1 if edge is connected to the focus node
            attn_beta[L] = 1.0 if (X.name == focus_node.name or Y.name == focus_node.name) else 0.0
            control[L] = 1.0
        
        return attn_alpha, attn_beta, control
    
    @staticmethod
    def exponential_strategy(pdg: PDG, t: int) -> Tuple[Dict, Dict, Dict]:
        """Exponential strategy: β drawn from exponential distribution with rate 1/n_edges."""
        attn_alpha = {}
        attn_beta = {}
        control = {}
        
        edges = list(pdg.edges("l,X,Y,α,β,P"))
        n_edges = len(edges)
        rate = 1.0 / n_edges if n_edges > 0 else 1.0
        
        for L, X, Y, α, β, P in edges:
            attn_alpha[L] = 0.0  # α = 0 as specified
            # Sample β from exponential distribution
            attn_beta[L] = np.random.exponential(1.0 / rate)
            control[L] = 1.0
        
        return attn_alpha, attn_beta, control


class ExperimentalSetup:
    """Main experimental setup class."""
    
    def __init__(self, seed: int = 42):
        self.seed = seed
        self.generator = PDGGenerator(seed)
        self.attention_strategies = {
            "global": AttentionStrategy.global_strategy,
            "local": AttentionStrategy.local_strategy,
            "node_based": AttentionStrategy.node_based_strategy,
            "exponential": AttentionStrategy.exponential_strategy
        }
    
    def create_pdg_specs(self) -> List[PDGSpec]:
        """Create specifications for PDGs with varying structures."""
        specs = []
        
        # Varying number of variables and edges
        configs = [
            (3, 2, "small_chain"),
            (4, 3, "small_star"),
            (5, 4, "medium_chain"),
            (6, 5, "medium_tree"),
            (7, 6, "large_chain"),
            (8, 7, "large_complex"),
        ]
        
        for i, (num_vars, num_edges, name) in enumerate(configs):
            spec = PDGSpec(
                name=name,
                num_vars=num_vars,
                num_edges=num_edges,
                val_range=(2, 3),
                max_edges_per_pair=2,  # Allow up to 2 edges between same nodes
                seed=100 + i,
                description=f"PDG with {num_vars} variables and {num_edges} edges"
            )
            specs.append(spec)
        
        return specs
    
    def run_single_experiment(self, pdg: PDG, spec: PDGSpec, 
                            attention_strategy_name: str, 
                            attention_strategy: Callable) -> ExperimentResult:
        """Run a single experiment with specific attention strategy."""
        
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
        
        try:
            # Initialize parameters
            mu_init = opt_joint(pdg_copy, gamma=0.1, iters=20, verbose=False)
            
            # Compute initial inconsistencies
            initial_global = float(torch_score(pdg_copy, mu_init, 0.001))
            initial_local = float(torch_score(pdg_copy, mu_init, 0.0001))
            
            # Run LIR training
            lir_train(
                pdg_copy,
                gamma=0.1,
                T=10,  # Number of training steps
                outer_iters=5,
                inner_iters=10,
                lr=0.01,
                refocus=attention_strategy,
                verbose=False,
                mu_init=mu_init
            )
            
            # Compute final inconsistencies
            mu_final = opt_joint(pdg_copy, gamma=0.1, iters=20, verbose=False)
            final_global = float(torch_score(pdg_copy, mu_final, 0.001))
            final_local = float(torch_score(pdg_copy, mu_final, 0.0001))
            
            # Calculate improvements
            improvement_global = (initial_global - final_global) / initial_global * 100 if initial_global > 0 else 0
            improvement_local = (initial_local - final_local) / initial_local * 100 if initial_local > 0 else 0
            
            return ExperimentResult(
                pdg_name=spec.name,
                num_vars=spec.num_vars,
                num_edges=spec.num_edges,
                attention_strategy=attention_strategy_name,
                initial_global_inconsistency=initial_global,
                final_global_inconsistency=final_global,
                initial_local_inconsistency=initial_local,
                final_local_inconsistency=final_local,
                improvement_global=improvement_global,
                improvement_local=improvement_local,
                success=True
            )
            
        except Exception as e:
            return ExperimentResult(
                pdg_name=spec.name,
                num_vars=spec.num_vars,
                num_edges=spec.num_edges,
                attention_strategy=attention_strategy_name,
                initial_global_inconsistency=0.0,
                final_global_inconsistency=0.0,
                initial_local_inconsistency=0.0,
                final_local_inconsistency=0.0,
                improvement_global=0.0,
                improvement_local=0.0,
                success=False,
                error=str(e)
            )
    
    def run_experiments(self) -> List[ExperimentResult]:
        """Run all experiments across all PDGs and attention strategies."""
        print("=== Running LIR Experiments ===")
        
        # Create PDG specifications
        specs = self.create_pdg_specs()
        print(f"Created {len(specs)} PDG specifications")
        
        # Generate PDGs
        pdgs = []
        for spec in specs:
            print(f"Generating {spec.name}...")
            pdg = self.generator.generate_pdg(spec)
            pdgs.append(pdg)
        
        # Run experiments
        results = []
        total_experiments = len(pdgs) * len(self.attention_strategies)
        experiment_count = 0
        
        for pdg, spec in zip(pdgs, specs):
            print(f"\nRunning experiments on {spec.name} ({spec.num_vars} vars, {spec.num_edges} edges)...")
            
            for strategy_name, strategy_func in self.attention_strategies.items():
                experiment_count += 1
                print(f"  [{experiment_count}/{total_experiments}] Testing {strategy_name} strategy...")
                
                result = self.run_single_experiment(pdg, spec, strategy_name, strategy_func)
                results.append(result)
        
        return results
    
    def analyze_results(self, results: List[ExperimentResult]):
        """Analyze and display experimental results."""
        print("\n=== Experimental Results Analysis ===")
        
        # Filter successful results
        successful_results = [r for r in results if r.success]
        print(f"Successful experiments: {len(successful_results)}/{len(results)}")
        
        if not successful_results:
            print("No successful experiments to analyze!")
            return
        
        # Convert to DataFrame for easier analysis
        import pandas as pd
        df = pd.DataFrame([asdict(r) for r in successful_results])
        
        # Overall performance by strategy
        print("\nAverage performance by attention strategy:")
        strategy_performance = df.groupby('attention_strategy').agg({
            'improvement_global': ['mean', 'std'],
            'improvement_local': ['mean', 'std']
        }).round(2)
        print(strategy_performance)
        
        # Performance by PDG size
        print("\nAverage performance by PDG size:")
        size_performance = df.groupby(['num_vars', 'num_edges']).agg({
            'improvement_global': ['mean', 'std'],
            'improvement_local': ['mean', 'std']
        }).round(2)
        print(size_performance)
        
        # Best performing combinations
        print("\nBest performing combinations:")
        best_global = df.loc[df['improvement_global'].idxmax()]
        best_local = df.loc[df['improvement_local'].idxmax()]
        
        print(f"Best Global Improvement: {best_global['pdg_name']} with {best_global['attention_strategy']} strategy ({best_global['improvement_global']:.1f}%)")
        print(f"Best Local Improvement: {best_local['pdg_name']} with {best_local['attention_strategy']} strategy ({best_local['improvement_local']:.1f}%)")
    
    def create_visualizations(self, results: List[ExperimentResult]):
        """Create visualizations of experimental results."""
        import pandas as pd
        
        # Filter successful results
        successful_results = [r for r in results if r.success]
        if not successful_results:
            print("No successful results to visualize!")
            return
        
        df = pd.DataFrame([asdict(r) for r in successful_results])
        
        # Create comprehensive visualization
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('LIR Experimental Results: Attention Strategy Comparison', fontsize=16, fontweight='bold')
        
        # Plot 1: Performance by attention strategy
        ax1 = axes[0, 0]
        strategy_performance = df.groupby('attention_strategy')['improvement_global'].mean()
        bars = ax1.bar(strategy_performance.index, strategy_performance.values, color='lightblue')
        ax1.set_title('Average Global Improvement by Attention Strategy')
        ax1.set_ylabel('Improvement (%)')
        ax1.tick_params(axis='x', rotation=45)
        
        # Plot 2: Performance by PDG size
        ax2 = axes[0, 1]
        size_performance = df.groupby(['num_vars', 'num_edges'])['improvement_global'].mean()
        size_labels = [f"{v}v{e}e" for v, e in size_performance.index]
        bars = ax2.bar(size_labels, size_performance.values, color='lightgreen')
        ax2.set_title('Average Global Improvement by PDG Size')
        ax2.set_ylabel('Improvement (%)')
        ax2.tick_params(axis='x', rotation=45)
        
        # Plot 3: Global vs Local improvement scatter
        ax3 = axes[1, 0]
        for strategy in df['attention_strategy'].unique():
            strategy_data = df[df['attention_strategy'] == strategy]
            ax3.scatter(strategy_data['improvement_global'], strategy_data['improvement_local'], 
                       label=strategy, alpha=0.6)
        ax3.set_xlabel('Global Improvement (%)')
        ax3.set_ylabel('Local Improvement (%)')
        ax3.set_title('Global vs Local Improvement by Strategy')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: Heatmap of strategy vs PDG size
        ax4 = axes[1, 1]
        pivot_data = df.groupby(['attention_strategy', 'num_vars'])['improvement_global'].mean().unstack()
        sns.heatmap(pivot_data, annot=True, fmt='.1f', cmap='viridis', ax=ax4)
        ax4.set_title('Strategy vs PDG Size Heatmap')
        ax4.set_xlabel('Number of Variables')
        ax4.set_ylabel('Attention Strategy')
        
        plt.tight_layout()
        plt.savefig('lir_experimental_results.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def save_results(self, results: List[ExperimentResult], filename: str = "lir_experimental_results.json"):
        """Save experimental results to JSON file."""
        results_data = [asdict(r) for r in results]
        with open(filename, 'w') as f:
            json.dump(results_data, f, indent=2)
        print(f"Results saved to {filename}")


def main():
    """Main function to run the complete experimental setup."""
    print("=== LIR Experimental Setup ===")
    
    # Create experimental setup
    setup = ExperimentalSetup(seed=42)
    
    # Run experiments
    results = setup.run_experiments()
    
    # Analyze results
    setup.analyze_results(results)
    
    # Create visualizations
    setup.create_visualizations(results)
    
    # Save results
    setup.save_results(results)
    
    print("\n=== Experiment Complete ===")
    return results


if __name__ == "__main__":
    results = main()
