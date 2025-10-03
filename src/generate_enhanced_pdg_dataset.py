#!/usr/bin/env python3
"""
Generate an enhanced PDG dataset with:
- 10 chain PDGs with varying structures
- 10 random structure PDGs
- Multiple edges between same nodes (different CPDs)
- Varying numbers of variables and edges
"""

import sys
from pathlib import Path
import random
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
from typing import List, Dict, Tuple, Any
import json
import pickle
from dataclasses import dataclass, asdict

# Add project paths
sys.path.insert(0, str(Path(__file__).resolve().parents[0]))

from pdg.pdg import PDG
from pdg.rv import Variable as Var
from pdg.dist import CPT
from pdg.alg.torch_opt import opt_joint, torch_score


@dataclass
class EnhancedPDGSpec:
    """Specification for generating an enhanced PDG."""
    name: str
    pattern: str  # "chain" or "random"
    num_vars: int
    num_edges: int
    max_edges_per_pair: int
    val_range: Tuple[int, int]
    seed: int
    description: str


class EnhancedPDGGenerator:
    """Generator for creating enhanced PDGs with multiple edges between nodes."""
    
    def __init__(self, seed: int = 42):
        self.seed = seed
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
    
    def generate_chain_pdg(self, spec: EnhancedPDGSpec) -> PDG:
        """Generate a chain PDG with potential for multiple edges between nodes."""
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
        
        # Create chain structure with potential for multiple edges
        edges_added = 0
        max_attempts = spec.num_edges * 10
        attempts = 0
        
        # First, create the basic chain structure
        basic_chain_edges = []
        for i in range(min(spec.num_vars - 1, spec.num_edges)):
            if i + 1 < len(varlist):
                basic_chain_edges.append((i, i + 1))
        
        # Add basic chain edges
        for src_idx, tgt_idx in basic_chain_edges:
            if edges_added >= spec.num_edges:
                break
            src = varlist[src_idx]
            tgt = varlist[tgt_idx]
            try:
                pdg += CPT.make_random(Var.product([src]), Var.product([tgt]))
                edges_added += 1
            except Exception:
                continue
        
        # Add additional edges between existing chain nodes
        while edges_added < spec.num_edges and attempts < max_attempts:
            attempts += 1
            
            # Select two nodes from the chain
            if len(varlist) >= 2:
                src_idx = random.randint(0, len(varlist) - 1)
                tgt_idx = random.randint(0, len(varlist) - 1)
                
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
                        continue
        
        return pdg
    
    def generate_random_pdg(self, spec: EnhancedPDGSpec) -> PDG:
        """Generate a random structure PDG with multiple edges between nodes."""
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
        
        # Generate random edges with potential for multiple edges between nodes
        edges_added = 0
        max_attempts = spec.num_edges * 20
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
                    continue
        
        return pdg
    
    def generate_pdg(self, spec: EnhancedPDGSpec) -> PDG:
        """Generate a PDG based on specification."""
        if spec.pattern == "chain":
            return self.generate_chain_pdg(spec)
        elif spec.pattern == "random":
            return self.generate_random_pdg(spec)
        else:
            raise ValueError(f"Unknown pattern: {spec.pattern}")


def create_enhanced_pdg_specs() -> List[EnhancedPDGSpec]:
    """Create specifications for the enhanced PDG dataset."""
    specs = []
    
    # Chain PDGs (10 total)
    chain_configs = [
        (4, 3, 2, "Small chain with multiple edges"),
        (5, 4, 2, "Medium chain with multiple edges"),
        (6, 5, 3, "Large chain with multiple edges"),
        (4, 5, 3, "Dense small chain"),
        (5, 6, 3, "Dense medium chain"),
        (6, 7, 3, "Dense large chain"),
        (7, 6, 2, "Long sparse chain"),
        (8, 7, 2, "Very long sparse chain"),
        (5, 8, 4, "Highly connected medium chain"),
        (6, 9, 4, "Highly connected large chain"),
    ]
    
    for i, (num_vars, num_edges, max_edges_per_pair, description) in enumerate(chain_configs):
        spec = EnhancedPDGSpec(
            name=f"chain_{i+1:02d}",
            pattern="chain",
            num_vars=num_vars,
            num_edges=num_edges,
            max_edges_per_pair=max_edges_per_pair,
            val_range=(2, 4),  # Domain sizes 2-4
            seed=200 + i,
            description=description
        )
        specs.append(spec)
    
    # Random PDGs (10 total)
    random_configs = [
        (4, 4, 2, "Small random with multiple edges"),
        (5, 5, 2, "Medium random with multiple edges"),
        (6, 6, 3, "Large random with multiple edges"),
        (4, 6, 3, "Dense small random"),
        (5, 7, 3, "Dense medium random"),
        (6, 8, 3, "Dense large random"),
        (7, 5, 2, "Sparse large random"),
        (8, 6, 2, "Sparse very large random"),
        (5, 9, 4, "Highly connected medium random"),
        (6, 10, 4, "Highly connected large random"),
    ]
    
    for i, (num_vars, num_edges, max_edges_per_pair, description) in enumerate(random_configs):
        spec = EnhancedPDGSpec(
            name=f"random_{i+1:02d}",
            pattern="random",
            num_vars=num_vars,
            num_edges=num_edges,
            max_edges_per_pair=max_edges_per_pair,
            val_range=(2, 4),  # Domain sizes 2-4
            seed=300 + i,
            description=description
        )
        specs.append(spec)
    
    return specs


def analyze_enhanced_pdgs(pdgs: List[PDG], specs: List[EnhancedPDGSpec]):
    """Analyze the enhanced PDGs."""
    print("\n=== Enhanced PDG Dataset Analysis ===")
    
    # Overall statistics
    total_pdgs = len(pdgs)
    chain_pdgs = len([s for s in specs if s.pattern == "chain"])
    random_pdgs = len([s for s in specs if s.pattern == "random"])
    
    print(f"Total PDGs: {total_pdgs}")
    print(f"Chain PDGs: {chain_pdgs}")
    print(f"Random PDGs: {random_pdgs}")
    
    # Edge statistics
    edge_counts = []
    multi_edge_counts = []
    var_counts = []
    
    for pdg, spec in zip(pdgs, specs):
        # Count total edges
        total_edges = len([e for e in pdg.edges("l,X,Y,α,β,P") 
                          if e[1].name != "1" and e[2].name != "1"])
        edge_counts.append(total_edges)
        var_counts.append(spec.num_vars)
        
        # Count edges with multiple connections between same nodes
        edge_pairs = {}
        for L, X, Y, α, β, P in pdg.edges("l,X,Y,α,β,P"):
            if X.name != "1" and Y.name != "1":
                pair = tuple(sorted([X.name, Y.name]))
                edge_pairs[pair] = edge_pairs.get(pair, 0) + 1
        
        multi_edges = sum(1 for count in edge_pairs.values() if count > 1)
        multi_edge_counts.append(multi_edges)
    
    print(f"\nEdge Statistics:")
    print(f"  Average edges per PDG: {np.mean(edge_counts):.1f}")
    print(f"  Edge range: {min(edge_counts)} - {max(edge_counts)}")
    print(f"  Average multi-edge pairs: {np.mean(multi_edge_counts):.1f}")
    print(f"  Multi-edge range: {min(multi_edge_counts)} - {max(multi_edge_counts)}")
    
    # Pattern comparison
    chain_edge_counts = [edge_counts[i] for i, s in enumerate(specs) if s.pattern == "chain"]
    random_edge_counts = [edge_counts[i] for i, s in enumerate(specs) if s.pattern == "random"]
    
    print(f"\nPattern Comparison:")
    print(f"  Chain PDGs - Average edges: {np.mean(chain_edge_counts):.1f}")
    print(f"  Random PDGs - Average edges: {np.mean(random_edge_counts):.1f}")
    
    # Detailed analysis
    print(f"\nDetailed PDG Analysis:")
    print("-" * 100)
    print(f"{'Name':<12} {'Pattern':<8} {'Vars':<5} {'Edges':<6} {'Multi':<6} {'Max/Edge':<8} {'Description'}")
    print("-" * 100)
    
    for pdg, spec in zip(pdgs, specs):
        total_edges = len([e for e in pdg.edges("l,X,Y,α,β,P") 
                          if e[1].name != "1" and e[2].name != "1"])
        
        # Count multi-edge pairs
        edge_pairs = {}
        for L, X, Y, α, β, P in pdg.edges("l,X,Y,α,β,P"):
            if X.name != "1" and Y.name != "1":
                pair = tuple(sorted([X.name, Y.name]))
                edge_pairs[pair] = edge_pairs.get(pair, 0) + 1
        
        multi_edges = sum(1 for count in edge_pairs.values() if count > 1)
        max_edges_per_pair = max(edge_pairs.values()) if edge_pairs else 0
        
        print(f"{spec.name:<12} {spec.pattern:<8} {spec.num_vars:<5} {total_edges:<6} "
              f"{multi_edges:<6} {max_edges_per_pair:<8} {spec.description}")


def visualize_enhanced_pdgs(pdgs: List[PDG], specs: List[EnhancedPDGSpec]):
    """Create visualizations of the enhanced PDGs."""
    print("\n=== Creating Enhanced PDG Visualizations ===")
    
    # Create a large grid visualization
    fig, axes = plt.subplots(4, 5, figsize=(25, 20))
    fig.suptitle('Enhanced PDG Dataset: 10 Chain + 10 Random PDGs', fontsize=16, fontweight='bold')
    
    for idx, (pdg, spec) in enumerate(zip(pdgs, specs)):
        row = idx // 5
        col = idx % 5
        ax = axes[row, col]
        
        # Create NetworkX graph for visualization
        G = nx.DiGraph()
        
        # Add nodes
        for var_name, var in pdg.vars.items():
            if var_name != "1":
                G.add_node(var_name, size=len(var))
        
        # Add edges with labels
        edge_labels = {}
        for L, X, Y, α, β, P in pdg.edges("l,X,Y,α,β,P"):
            if X.name != "1" and Y.name != "1":
                G.add_edge(X.name, Y.name, label=L, alpha=α, beta=β)
                edge_labels[(X.name, Y.name)] = f"{L}\nα={α:.1f}, β={β:.1f}"
        
        # Layout
        pos = nx.spring_layout(G, seed=42, k=1.5, iterations=50)
        
        # Draw nodes
        node_sizes = [G.nodes[node].get('size', 2) * 200 for node in G.nodes()]
        nx.draw_networkx_nodes(G, pos, node_color='lightblue', 
                              node_size=node_sizes, alpha=0.8, ax=ax)
        
        # Draw edges
        nx.draw_networkx_edges(G, pos, edge_color='gray', 
                              arrows=True, arrowsize=15, ax=ax)
        
        # Draw labels
        nx.draw_networkx_labels(G, pos, font_size=8, font_weight='bold', ax=ax)
        
        # Draw edge labels (only for small graphs to avoid clutter)
        if len(edge_labels) <= 6:
            nx.draw_networkx_edge_labels(G, pos, edge_labels, font_size=6, ax=ax)
        
        # Title
        title = f"{spec.name.upper()}\n{spec.pattern} - {spec.num_vars}v, {len(G.edges())}e"
        ax.set_title(title, fontsize=9, pad=10)
        ax.axis('off')
    
    plt.tight_layout()
    plt.savefig('enhanced_pdg_dataset.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Create statistics visualization
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Enhanced PDG Dataset Statistics', fontsize=16, fontweight='bold')
    
    # Extract data
    chain_specs = [s for s in specs if s.pattern == "chain"]
    random_specs = [s for s in specs if s.pattern == "random"]
    
    chain_edges = [len([e for e in pdgs[i].edges("l,X,Y,α,β,P") 
                       if e[1].name != "1" and e[2].name != "1"]) 
                   for i, s in enumerate(specs) if s.pattern == "chain"]
    random_edges = [len([e for e in pdgs[i].edges("l,X,Y,α,β,P") 
                        if e[1].name != "1" and e[2].name != "1"]) 
                    for i, s in enumerate(specs) if s.pattern == "random"]
    
    # Plot 1: Edges by pattern
    ax1 = axes[0, 0]
    ax1.bar(['Chain', 'Random'], [np.mean(chain_edges), np.mean(random_edges)], 
            color=['lightblue', 'lightcoral'])
    ax1.set_ylabel('Average Number of Edges')
    ax1.set_title('Average Edges by Pattern')
    
    # Plot 2: Edges distribution
    ax2 = axes[0, 1]
    ax2.hist(chain_edges, alpha=0.7, label='Chain', color='lightblue', bins=8)
    ax2.hist(random_edges, alpha=0.7, label='Random', color='lightcoral', bins=8)
    ax2.set_xlabel('Number of Edges')
    ax2.set_ylabel('Frequency')
    ax2.set_title('Edge Distribution by Pattern')
    ax2.legend()
    
    # Plot 3: Variables vs Edges
    ax3 = axes[1, 0]
    chain_vars = [s.num_vars for s in chain_specs]
    random_vars = [s.num_vars for s in random_specs]
    ax3.scatter(chain_vars, chain_edges, label='Chain', color='lightblue', s=100)
    ax3.scatter(random_vars, random_edges, label='Random', color='lightcoral', s=100)
    ax3.set_xlabel('Number of Variables')
    ax3.set_ylabel('Number of Edges')
    ax3.set_title('Variables vs Edges')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Multi-edge analysis
    ax4 = axes[1, 1]
    chain_multi = []
    random_multi = []
    
    for i, spec in enumerate(specs):
        pdg = pdgs[i]
        edge_pairs = {}
        for L, X, Y, α, β, P in pdg.edges("l,X,Y,α,β,P"):
            if X.name != "1" and Y.name != "1":
                pair = tuple(sorted([X.name, Y.name]))
                edge_pairs[pair] = edge_pairs.get(pair, 0) + 1
        
        multi_edges = sum(1 for count in edge_pairs.values() if count > 1)
        if spec.pattern == "chain":
            chain_multi.append(multi_edges)
        else:
            random_multi.append(multi_edges)
    
    ax4.bar(['Chain', 'Random'], [np.mean(chain_multi), np.mean(random_multi)], 
            color=['lightblue', 'lightcoral'])
    ax4.set_ylabel('Average Multi-Edge Pairs')
    ax4.set_title('Multi-Edge Pairs by Pattern')
    
    plt.tight_layout()
    plt.savefig('enhanced_pdg_statistics.png', dpi=300, bbox_inches='tight')
    plt.show()


def save_enhanced_dataset(pdgs: List[PDG], specs: List[EnhancedPDGSpec], base_path: str = "enhanced_pdg_dataset"):
    """Save the enhanced dataset."""
    base_path = Path(base_path)
    base_path.mkdir(parents=True, exist_ok=True)
    
    # Save specs as JSON
    specs_data = [asdict(spec) for spec in specs]
    with open(base_path / "specs.json", 'w') as f:
        json.dump(specs_data, f, indent=2)
    
    # Save PDGs as pickle
    with open(base_path / "pdgs.pkl", 'wb') as f:
        pickle.dump(pdgs, f)
    
    print(f"\nEnhanced dataset saved to {base_path}/")
    print("Files created:")
    print(f"  - {base_path}/specs.json (specifications)")
    print(f"  - {base_path}/pdgs.pkl (PDG objects)")


def main():
    """Main function to generate the enhanced PDG dataset."""
    print("=== Generating Enhanced PDG Dataset ===")
    
    # Create specifications
    specs = create_enhanced_pdg_specs()
    print(f"Created {len(specs)} PDG specifications")
    print(f"  - {len([s for s in specs if s.pattern == 'chain'])} chain PDGs")
    print(f"  - {len([s for s in specs if s.pattern == 'random'])} random PDGs")
    
    # Generate PDGs
    generator = EnhancedPDGGenerator()
    pdgs = []
    
    print("\nGenerating PDGs...")
    for spec in specs:
        print(f"  Generating {spec.name} ({spec.pattern} pattern)...")
        pdg = generator.generate_pdg(spec)
        pdgs.append(pdg)
    
    # Analyze PDGs
    analyze_enhanced_pdgs(pdgs, specs)
    
    # Create visualizations
    visualize_enhanced_pdgs(pdgs, specs)
    
    # Save dataset
    save_enhanced_dataset(pdgs, specs)
    
    print(f"\nGenerated {len(pdgs)} enhanced PDGs")
    print("Visualizations saved to 'enhanced_pdg_dataset.png' and 'enhanced_pdg_statistics.png'")
    
    return pdgs, specs


if __name__ == "__main__":
    pdgs, specs = main()
