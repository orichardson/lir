#!/usr/bin/env python3
"""
Generate a simple dataset with 5 PDGs - one for each connectivity pattern type.
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
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).parent))

from pdg.pdg import PDG
from pdg.rv import Variable as Var
from pdg.dist import CPT, ParamCPD
from pdg.alg.torch_opt import opt_joint, torch_score


@dataclass
class SimplePDGSpec:
    """Specification for a simple PDG."""
    pattern: str
    num_vars: int
    num_edges: int
    val_range: Tuple[int, int]
    seed: int
    name: str
    description: str


class SimplePDGGenerator:
    """Generator for creating simple PDGs with one of each connectivity pattern."""
    
    def __init__(self, seed: int = 42):
        self.seed = seed
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
    
    def generate_chain_pdg(self, spec: SimplePDGSpec) -> PDG:
        """Generate a chain PDG: A->B->C->D->E"""
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
        
        # Create chain edges
        for i in range(min(spec.num_edges, spec.num_vars - 1)):
            if i + 1 < len(varlist):
                src = varlist[i]
                tgt = varlist[i + 1]
                pdg += CPT.make_random(Var.product([src]), Var.product([tgt]))
        
        return pdg
    
    def generate_star_pdg(self, spec: SimplePDGSpec) -> PDG:
        """Generate a star PDG: A->B, A->C, A->D, A->E"""
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
        
        # Create star edges (center -> leaves)
        center = varlist[0]
        for i in range(1, min(spec.num_edges + 1, spec.num_vars)):
            leaf = varlist[i]
            pdg += CPT.make_random(Var.product([center]), Var.product([leaf]))
        
        return pdg
    
    def generate_tree_pdg(self, spec: SimplePDGSpec) -> PDG:
        """Generate a tree PDG: balanced binary tree structure"""
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
        
        # Create tree edges
        edges_added = 0
        for i in range(spec.num_vars // 2):
            if edges_added >= spec.num_edges:
                break
            parent = varlist[i]
            left_child = varlist[2 * i + 1] if 2 * i + 1 < spec.num_vars else None
            right_child = varlist[2 * i + 2] if 2 * i + 2 < spec.num_vars else None
            
            if left_child and edges_added < spec.num_edges:
                pdg += CPT.make_random(Var.product([parent]), Var.product([left_child]))
                edges_added += 1
            if right_child and edges_added < spec.num_edges:
                pdg += CPT.make_random(Var.product([parent]), Var.product([right_child]))
                edges_added += 1
        
        return pdg
    
    def generate_random_pdg(self, spec: SimplePDGSpec) -> PDG:
        """Generate a random connected PDG with diverse structure (avoiding chains)"""
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
        
        # Create a more complex random structure that's clearly not a chain
        # Strategy: Create a "diamond" or "clique-like" structure
        
        if spec.num_vars == 5 and spec.num_edges == 4:
            # For 5 vars, 4 edges: create a diamond structure
            # A-B-C-D with A-C and B-D (diamond shape)
            edges = [(0, 1), (1, 2), (2, 3), (0, 2)]  # A-B, B-C, C-D, A-C
        elif spec.num_vars == 5 and spec.num_edges == 5:
            # For 5 vars, 5 edges: create a more complex structure
            # A-B-C-D-E with A-C (creates a triangle and a chain)
            edges = [(0, 1), (1, 2), (2, 3), (3, 4), (0, 2)]  # A-B, B-C, C-D, D-E, A-C
        elif spec.num_vars == 4 and spec.num_edges == 4:
            # For 4 vars, 4 edges: create a complete graph (clique)
            edges = [(0, 1), (0, 2), (0, 3), (1, 2)]  # A-B, A-C, A-D, B-C
        else:
            # Fallback: create a more complex structure
            # Start with a triangle, then add edges that create cycles
            edges = []
            if spec.num_vars >= 3:
                # Create a triangle
                edges = [(0, 1), (1, 2), (2, 0)]
            
            # Add remaining edges to create more complexity
            edges_added = len(edges)
            while edges_added < spec.num_edges and edges_added < spec.num_vars * (spec.num_vars - 1) // 2:
                # Find all possible edges
                possible_edges = []
                for i in range(spec.num_vars):
                    for j in range(i + 1, spec.num_vars):
                        if (i, j) not in edges and (j, i) not in edges:
                            possible_edges.append((i, j))
                
                if possible_edges:
                    # Pick a random edge
                    edge = random.choice(possible_edges)
                    edges.append(edge)
                    edges_added += 1
                else:
                    break
        
        # Add the edges to the PDG
        for src_idx, tgt_idx in edges:
            src = varlist[src_idx]
            tgt = varlist[tgt_idx]
            pdg += CPT.make_random(Var.product([src]), Var.product([tgt]))
        
        return pdg
    
    def generate_cycle_pdg(self, spec: SimplePDGSpec) -> PDG:
        """Generate a cycle PDG: A->B->C->D->A"""
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
        
        # Create cycle edges
        for i in range(min(spec.num_edges, spec.num_vars)):
            src = varlist[i]
            tgt = varlist[(i + 1) % spec.num_vars]
            pdg += CPT.make_random(Var.product([src]), Var.product([tgt]))
        
        return pdg
    
    def generate_pdg(self, spec: SimplePDGSpec) -> PDG:
        """Generate a PDG based on specification."""
        if spec.pattern == "chain":
            return self.generate_chain_pdg(spec)
        elif spec.pattern == "star":
            return self.generate_star_pdg(spec)
        elif spec.pattern == "tree":
            return self.generate_tree_pdg(spec)
        elif spec.pattern == "random":
            return self.generate_random_pdg(spec)
        elif spec.pattern == "cycle":
            return self.generate_cycle_pdg(spec)
        else:
            raise ValueError(f"Unknown pattern: {spec.pattern}")
    
    def compute_inconsistency_score(self, pdg: PDG) -> float:
        """Compute the inconsistency score of a PDG."""
        try:
            # Make PDG parametric for evaluation
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
            
            # Solve for optimal μ
            mu_star = opt_joint(pdg_copy, gamma=0.0, iters=100, verbose=False)
            
            # Compute inconsistency score
            score = float(torch_score(pdg_copy, mu_star, 0.0))
            return abs(score)  # Return absolute value for easier interpretation
            
        except Exception as e:
            print(f"Error computing inconsistency: {e}")
            return 0.0


def create_simple_pdg_specs() -> List[SimplePDGSpec]:
    """Create specifications for 5 simple PDGs - one of each pattern."""
    specs = []
    
    # Define one PDG for each pattern type
    patterns = [
        ("chain", 5, 4, "Linear chain: A->B->C->D->E"),
        ("star", 5, 4, "Star: A->B, A->C, A->D, A->E"),
        ("tree", 7, 6, "Binary tree: A->B, A->C, B->D, B->E, C->F, C->G"),
        ("random", 5, 4, "Random connected graph"),
        ("cycle", 5, 5, "Cycle: A->B->C->D->E->A")
    ]
    
    for i, (pattern, num_vars, num_edges, description) in enumerate(patterns):
        spec = SimplePDGSpec(
            pattern=pattern,
            num_vars=num_vars,
            num_edges=num_edges,
            val_range=(2, 3),  # Small domain sizes for simplicity
            seed=100 + i,  # Different seed for each PDG
            name=f"{pattern}_pdg",
            description=description
        )
        specs.append(spec)
    
    return specs


def visualize_simple_pdgs(pdgs: List[PDG], specs: List[SimplePDGSpec], save_path: str = None):
    """Visualize the 5 simple PDGs in a grid layout."""
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Simple PDG Dataset - One of Each Connectivity Pattern', fontsize=16, fontweight='bold')
    
    for idx, (pdg, spec) in enumerate(zip(pdgs, specs)):
        row = idx // 3
        col = idx % 3
        ax = axes[row, col]
        
        # Draw the graph
        G = pdg.graph.to_undirected()
        pos = nx.spring_layout(G, seed=42)
        
        nx.draw_networkx_nodes(G, pos, node_color='lightblue', 
                              node_size=800, alpha=0.8, ax=ax)
        nx.draw_networkx_edges(G, pos, edge_color='gray', 
                              arrows=True, arrowsize=20, ax=ax)
        nx.draw_networkx_labels(G, pos, font_size=12, font_weight='bold', ax=ax)
        
        ax.set_title(f"{spec.pattern.upper()} Pattern\n{spec.description}\n"
                    f"Vars: {spec.num_vars}, Edges: {spec.num_edges}", 
                    fontsize=11)
        ax.axis('off')
    
    # Hide the last subplot
    axes[1, 2].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def analyze_simple_pdgs(pdgs: List[PDG], specs: List[SimplePDGSpec]):
    """Analyze the simple PDGs."""
    print("\n=== Simple PDG Dataset Analysis ===")
    
    generator = SimplePDGGenerator()
    
    for pdg, spec in zip(pdgs, specs):
        print(f"\n{spec.pattern.upper()} Pattern ({spec.name}):")
        print(f"  Description: {spec.description}")
        print(f"  Variables: {spec.num_vars}")
        print(f"  Edges: {spec.num_edges}")
        print(f"  Domain sizes: {spec.val_range}")
        
        # Compute inconsistency score
        inconsistency = generator.compute_inconsistency_score(pdg)
        print(f"  Inconsistency score: {inconsistency:.4f}")
        
        # Show edge information
        edges = list(pdg.edges("l,X,Y,α,β,P"))
        print(f"  Edge details:")
        for L, X, Y, α, β, P in edges:
            print(f"    {X.name}->{Y.name} (label: {L}, α={α:.2f}, β={β:.2f})")


def save_simple_dataset(pdgs: List[PDG], specs: List[SimplePDGSpec], base_path: str = "simple_pdg_dataset"):
    """Save the simple dataset."""
    base_path = Path(base_path)
    base_path.mkdir(parents=True, exist_ok=True)
    
    # Save specs as JSON
    specs_data = [asdict(spec) for spec in specs]
    with open(base_path / "specs.json", 'w') as f:
        json.dump(specs_data, f, indent=2)
    
    # Save PDGs as pickle
    with open(base_path / "pdgs.pkl", 'wb') as f:
        pickle.dump(pdgs, f)
    
    print(f"\nSimple dataset saved to {base_path}/")
    print("Files created:")
    print(f"  - {base_path}/specs.json (specifications)")
    print(f"  - {base_path}/pdgs.pkl (PDG objects)")


def main():
    """Main function to generate the simple PDG dataset."""
    print("=== Generating Simple PDG Dataset ===")
    
    # Create specifications
    specs = create_simple_pdg_specs()
    print(f"Created {len(specs)} PDG specifications")
    
    # Generate PDGs
    generator = SimplePDGGenerator()
    pdgs = []
    
    print("Generating PDGs...")
    for spec in specs:
        print(f"  Generating {spec.name} ({spec.pattern} pattern)...")
        pdg = generator.generate_pdg(spec)
        pdgs.append(pdg)
    
    # Visualize PDGs
    print("Creating visualization...")
    visualize_simple_pdgs(pdgs, specs, "simple_pdgs_grid.png")
    
    # Analyze PDGs
    analyze_simple_pdgs(pdgs, specs)
    
    # Save dataset
    save_simple_dataset(pdgs, specs)
    
    print(f"\nGenerated {len(pdgs)} simple PDGs")
    print("Visualization saved to 'simple_pdgs_grid.png'")
    
    return pdgs, specs


if __name__ == "__main__":
    pdgs, specs = main()
