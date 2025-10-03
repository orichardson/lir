#!/usr/bin/env python3
"""
Generate a PDG dataset with bidirectional edges:
- A→B AND B→A
- B→C AND C→B
- Multiple bidirectional pairs in each PDG
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
class BidirectionalPDGSpec:
    """Specification for generating a PDG with bidirectional edges."""
    name: str
    pattern: str  # "chain", "random", "grid", "star"
    num_vars: int
    num_bidirectional_pairs: int
    additional_edges: int  # Extra unidirectional edges
    val_range: Tuple[int, int]
    seed: int
    description: str


class BidirectionalPDGGenerator:
    """Generator for creating PDGs with bidirectional edges."""
    
    def __init__(self, seed: int = 42):
        self.seed = seed
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
    
    def generate_chain_bidirectional_pdg(self, spec: BidirectionalPDGSpec) -> PDG:
        """Generate a chain PDG with bidirectional edges."""
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
        
        # Create bidirectional pairs in chain structure
        bidirectional_pairs = []
        edges_added = 0
        
        # First, create bidirectional pairs along the chain
        for i in range(min(spec.num_vars - 1, spec.num_bidirectional_pairs)):
            if i + 1 < len(varlist):
                src = varlist[i]
                tgt = varlist[i + 1]
                
                # Add A→B
                try:
                    pdg += CPT.make_random(Var.product([src]), Var.product([tgt]))
                    edges_added += 1
                except Exception:
                    continue
                
                # Add B→A
                try:
                    pdg += CPT.make_random(Var.product([tgt]), Var.product([src]))
                    edges_added += 1
                    bidirectional_pairs.append((src.name, tgt.name))
                except Exception:
                    continue
        
        # Add additional unidirectional edges
        additional_added = 0
        max_attempts = spec.additional_edges * 10
        attempts = 0
        
        while additional_added < spec.additional_edges and attempts < max_attempts:
            attempts += 1
            
            src_idx = random.randint(0, spec.num_vars - 1)
            tgt_idx = random.randint(0, spec.num_vars - 1)
            
            if src_idx == tgt_idx:
                continue
            
            src = varlist[src_idx]
            tgt = varlist[tgt_idx]
            
            # Check if this pair already has bidirectional edges
            pair_exists = (src.name, tgt.name) in bidirectional_pairs or (tgt.name, src.name) in bidirectional_pairs
            
            # Add unidirectional edge
            try:
                pdg += CPT.make_random(Var.product([src]), Var.product([tgt]))
                additional_added += 1
            except Exception:
                continue
        
        return pdg
    
    def generate_random_bidirectional_pdg(self, spec: BidirectionalPDGSpec) -> PDG:
        """Generate a random PDG with bidirectional edges."""
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
        
        # Create bidirectional pairs randomly
        bidirectional_pairs = []
        edges_added = 0
        
        # Create bidirectional pairs
        for _ in range(spec.num_bidirectional_pairs):
            if len(varlist) < 2:
                break
                
            # Select two different variables
            src_idx = random.randint(0, spec.num_vars - 1)
            tgt_idx = random.randint(0, spec.num_vars - 1)
            
            if src_idx == tgt_idx:
                continue
            
            src = varlist[src_idx]
            tgt = varlist[tgt_idx]
            
            # Check if this pair already exists
            pair_exists = (src.name, tgt.name) in bidirectional_pairs or (tgt.name, src.name) in bidirectional_pairs
            if pair_exists:
                continue
            
            # Add A→B
            try:
                pdg += CPT.make_random(Var.product([src]), Var.product([tgt]))
                edges_added += 1
            except Exception:
                continue
            
            # Add B→A
            try:
                pdg += CPT.make_random(Var.product([tgt]), Var.product([src]))
                edges_added += 1
                bidirectional_pairs.append((src.name, tgt.name))
            except Exception:
                continue
        
        # Add additional unidirectional edges
        additional_added = 0
        max_attempts = spec.additional_edges * 10
        attempts = 0
        
        while additional_added < spec.additional_edges and attempts < max_attempts:
            attempts += 1
            
            src_idx = random.randint(0, spec.num_vars - 1)
            tgt_idx = random.randint(0, spec.num_vars - 1)
            
            if src_idx == tgt_idx:
                continue
            
            src = varlist[src_idx]
            tgt = varlist[tgt_idx]
            
            # Check if this pair already has bidirectional edges
            pair_exists = (src.name, tgt.name) in bidirectional_pairs or (tgt.name, src.name) in bidirectional_pairs
            
            # Add unidirectional edge
            try:
                pdg += CPT.make_random(Var.product([src]), Var.product([tgt]))
                additional_added += 1
            except Exception:
                continue
        
        return pdg
    
    def generate_grid_bidirectional_pdg(self, spec: BidirectionalPDGSpec) -> PDG:
        """Generate a grid PDG with bidirectional edges."""
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
        
        # Create grid structure with bidirectional edges
        bidirectional_pairs = []
        edges_added = 0
        
        # Create a simple grid (2D layout)
        grid_size = int(np.ceil(np.sqrt(spec.num_vars)))
        
        # Add bidirectional edges in grid pattern
        for i in range(min(spec.num_bidirectional_pairs, grid_size - 1)):
            for j in range(min(spec.num_bidirectional_pairs, grid_size - 1)):
                if i * grid_size + j + 1 < len(varlist) and (i + 1) * grid_size + j < len(varlist):
                    # Horizontal bidirectional edge
                    src = varlist[i * grid_size + j]
                    tgt = varlist[i * grid_size + j + 1]
                    
                    try:
                        pdg += CPT.make_random(Var.product([src]), Var.product([tgt]))
                        edges_added += 1
                        pdg += CPT.make_random(Var.product([tgt]), Var.product([src]))
                        edges_added += 1
                        bidirectional_pairs.append((src.name, tgt.name))
                    except Exception:
                        continue
                
                if (i + 1) * grid_size + j < len(varlist) and i * grid_size + j < len(varlist):
                    # Vertical bidirectional edge
                    src = varlist[i * grid_size + j]
                    tgt = varlist[(i + 1) * grid_size + j]
                    
                    try:
                        pdg += CPT.make_random(Var.product([src]), Var.product([tgt]))
                        edges_added += 1
                        pdg += CPT.make_random(Var.product([tgt]), Var.product([src]))
                        edges_added += 1
                        bidirectional_pairs.append((src.name, tgt.name))
                    except Exception:
                        continue
        
        # Add additional unidirectional edges
        additional_added = 0
        max_attempts = spec.additional_edges * 10
        attempts = 0
        
        while additional_added < spec.additional_edges and attempts < max_attempts:
            attempts += 1
            
            src_idx = random.randint(0, spec.num_vars - 1)
            tgt_idx = random.randint(0, spec.num_vars - 1)
            
            if src_idx == tgt_idx:
                continue
            
            src = varlist[src_idx]
            tgt = varlist[tgt_idx]
            
            # Check if this pair already has bidirectional edges
            pair_exists = (src.name, tgt.name) in bidirectional_pairs or (tgt.name, src.name) in bidirectional_pairs
            
            # Add unidirectional edge
            try:
                pdg += CPT.make_random(Var.product([src]), Var.product([tgt]))
                additional_added += 1
            except Exception:
                continue
        
        return pdg
    
    def generate_star_bidirectional_pdg(self, spec: BidirectionalPDGSpec) -> PDG:
        """Generate a star PDG with bidirectional edges."""
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
        
        # Create star structure with bidirectional edges
        bidirectional_pairs = []
        edges_added = 0
        
        # Center node is the first variable
        center = varlist[0]
        
        # Create bidirectional edges from center to other nodes
        for i in range(1, min(spec.num_bidirectional_pairs + 1, len(varlist))):
            tgt = varlist[i]
            
            # Add center→target
            try:
                pdg += CPT.make_random(Var.product([center]), Var.product([tgt]))
                edges_added += 1
            except Exception:
                continue
            
            # Add target→center
            try:
                pdg += CPT.make_random(Var.product([tgt]), Var.product([center]))
                edges_added += 1
                bidirectional_pairs.append((center.name, tgt.name))
            except Exception:
                continue
        
        # Add additional unidirectional edges between non-center nodes
        additional_added = 0
        max_attempts = spec.additional_edges * 10
        attempts = 0
        
        while additional_added < spec.additional_edges and attempts < max_attempts:
            attempts += 1
            
            # Select two non-center nodes
            src_idx = random.randint(1, spec.num_vars - 1)
            tgt_idx = random.randint(1, spec.num_vars - 1)
            
            if src_idx == tgt_idx:
                continue
            
            src = varlist[src_idx]
            tgt = varlist[tgt_idx]
            
            # Add unidirectional edge
            try:
                pdg += CPT.make_random(Var.product([src]), Var.product([tgt]))
                additional_added += 1
            except Exception:
                continue
        
        return pdg
    
    def generate_pdg(self, spec: BidirectionalPDGSpec) -> PDG:
        """Generate a PDG based on specification."""
        if spec.pattern == "chain":
            return self.generate_chain_bidirectional_pdg(spec)
        elif spec.pattern == "random":
            return self.generate_random_bidirectional_pdg(spec)
        elif spec.pattern == "grid":
            return self.generate_grid_bidirectional_pdg(spec)
        elif spec.pattern == "star":
            return self.generate_star_bidirectional_pdg(spec)
        else:
            raise ValueError(f"Unknown pattern: {spec.pattern}")


def create_bidirectional_pdg_specs() -> List[BidirectionalPDGSpec]:
    """Create specifications for the bidirectional PDG dataset."""
    specs = []
    
    # Chain PDGs with bidirectional edges (5 total)
    chain_configs = [
        (4, 2, 1, "Small chain with 2 bidirectional pairs"),
        (5, 2, 2, "Medium chain with 2 bidirectional pairs"),
        (6, 3, 1, "Large chain with 3 bidirectional pairs"),
        (5, 3, 3, "Dense chain with 3 bidirectional pairs"),
        (6, 4, 2, "Very dense chain with 4 bidirectional pairs"),
    ]
    
    for i, (num_vars, num_bidirectional, additional, description) in enumerate(chain_configs):
        spec = BidirectionalPDGSpec(
            name=f"chain_bidir_{i+1:02d}",
            pattern="chain",
            num_vars=num_vars,
            num_bidirectional_pairs=num_bidirectional,
            additional_edges=additional,
            val_range=(2, 4),
            seed=400 + i,
            description=description
        )
        specs.append(spec)
    
    # Random PDGs with bidirectional edges (5 total)
    random_configs = [
        (4, 2, 1, "Small random with 2 bidirectional pairs"),
        (5, 2, 2, "Medium random with 2 bidirectional pairs"),
        (6, 3, 1, "Large random with 3 bidirectional pairs"),
        (5, 3, 3, "Dense random with 3 bidirectional pairs"),
        (6, 4, 2, "Very dense random with 4 bidirectional pairs"),
    ]
    
    for i, (num_vars, num_bidirectional, additional, description) in enumerate(random_configs):
        spec = BidirectionalPDGSpec(
            name=f"random_bidir_{i+1:02d}",
            pattern="random",
            num_vars=num_vars,
            num_bidirectional_pairs=num_bidirectional,
            additional_edges=additional,
            val_range=(2, 4),
            seed=500 + i,
            description=description
        )
        specs.append(spec)
    
    # Grid PDGs with bidirectional edges (5 total)
    grid_configs = [
        (4, 2, 1, "Small grid with 2 bidirectional pairs"),
        (6, 3, 2, "Medium grid with 3 bidirectional pairs"),
        (9, 4, 2, "Large grid with 4 bidirectional pairs"),
        (6, 4, 3, "Dense grid with 4 bidirectional pairs"),
        (9, 6, 3, "Very dense grid with 6 bidirectional pairs"),
    ]
    
    for i, (num_vars, num_bidirectional, additional, description) in enumerate(grid_configs):
        spec = BidirectionalPDGSpec(
            name=f"grid_bidir_{i+1:02d}",
            pattern="grid",
            num_vars=num_vars,
            num_bidirectional_pairs=num_bidirectional,
            additional_edges=additional,
            val_range=(2, 4),
            seed=600 + i,
            description=description
        )
        specs.append(spec)
    
    # Star PDGs with bidirectional edges (5 total)
    star_configs = [
        (4, 2, 1, "Small star with 2 bidirectional pairs"),
        (5, 3, 1, "Medium star with 3 bidirectional pairs"),
        (6, 4, 2, "Large star with 4 bidirectional pairs"),
        (5, 4, 2, "Dense star with 4 bidirectional pairs"),
        (6, 5, 2, "Very dense star with 5 bidirectional pairs"),
    ]
    
    for i, (num_vars, num_bidirectional, additional, description) in enumerate(star_configs):
        spec = BidirectionalPDGSpec(
            name=f"star_bidir_{i+1:02d}",
            pattern="star",
            num_vars=num_vars,
            num_bidirectional_pairs=num_bidirectional,
            additional_edges=additional,
            val_range=(2, 4),
            seed=700 + i,
            description=description
        )
        specs.append(spec)
    
    return specs


def analyze_bidirectional_pdgs(pdgs: List[PDG], specs: List[BidirectionalPDGSpec]):
    """Analyze the bidirectional PDGs."""
    print("\n=== Bidirectional PDG Dataset Analysis ===")
    
    # Overall statistics
    total_pdgs = len(pdgs)
    pattern_counts = {}
    for spec in specs:
        pattern_counts[spec.pattern] = pattern_counts.get(spec.pattern, 0) + 1
    
    print(f"Total PDGs: {total_pdgs}")
    for pattern, count in pattern_counts.items():
        print(f"{pattern.capitalize()} PDGs: {count}")
    
    # Edge statistics
    total_edges = []
    bidirectional_pairs = []
    unidirectional_edges = []
    var_counts = []
    
    for pdg, spec in zip(pdgs, specs):
        # Count total edges
        all_edges = [e for e in pdg.edges("l,X,Y,α,β,P") 
                    if e[1].name != "1" and e[2].name != "1"]
        total_edges.append(len(all_edges))
        var_counts.append(spec.num_vars)
        
        # Count bidirectional pairs
        edge_pairs = {}
        for L, X, Y, α, β, P in all_edges:
            pair = tuple(sorted([X.name, Y.name]))
            edge_pairs[pair] = edge_pairs.get(pair, 0) + 1
        
        bidir_count = sum(1 for count in edge_pairs.values() if count >= 2)
        bidirectional_pairs.append(bidir_count)
        
        # Count unidirectional edges
        unidir_count = sum(1 for count in edge_pairs.values() if count == 1)
        unidirectional_edges.append(unidir_count)
    
    print(f"\nEdge Statistics:")
    print(f"  Average total edges per PDG: {np.mean(total_edges):.1f}")
    print(f"  Total edge range: {min(total_edges)} - {max(total_edges)}")
    print(f"  Average bidirectional pairs: {np.mean(bidirectional_pairs):.1f}")
    print(f"  Bidirectional pairs range: {min(bidirectional_pairs)} - {max(bidirectional_pairs)}")
    print(f"  Average unidirectional edges: {np.mean(unidirectional_edges):.1f}")
    print(f"  Unidirectional edges range: {min(unidirectional_edges)} - {max(unidirectional_edges)}")
    
    # Pattern comparison
    for pattern in pattern_counts.keys():
        pattern_indices = [i for i, s in enumerate(specs) if s.pattern == pattern]
        pattern_total_edges = [total_edges[i] for i in pattern_indices]
        pattern_bidir_pairs = [bidirectional_pairs[i] for i in pattern_indices]
        
        print(f"\n{pattern.capitalize()} Pattern:")
        print(f"  Average total edges: {np.mean(pattern_total_edges):.1f}")
        print(f"  Average bidirectional pairs: {np.mean(pattern_bidir_pairs):.1f}")
    
    # Detailed analysis
    print(f"\nDetailed PDG Analysis:")
    print("-" * 120)
    print(f"{'Name':<15} {'Pattern':<8} {'Vars':<5} {'Total':<6} {'Bidir':<6} {'Unidir':<7} {'Bidir/Total':<12} {'Description'}")
    print("-" * 120)
    
    for pdg, spec in zip(pdgs, specs):
        all_edges = [e for e in pdg.edges("l,X,Y,α,β,P") 
                    if e[1].name != "1" and e[2].name != "1"]
        total_edge_count = len(all_edges)
        
        # Count bidirectional pairs
        edge_pairs = {}
        for L, X, Y, α, β, P in all_edges:
            pair = tuple(sorted([X.name, Y.name]))
            edge_pairs[pair] = edge_pairs.get(pair, 0) + 1
        
        bidir_count = sum(1 for count in edge_pairs.values() if count >= 2)
        unidir_count = sum(1 for count in edge_pairs.values() if count == 1)
        bidir_ratio = (bidir_count * 2) / total_edge_count if total_edge_count > 0 else 0
        
        print(f"{spec.name:<15} {spec.pattern:<8} {spec.num_vars:<5} {total_edge_count:<6} "
              f"{bidir_count:<6} {unidir_count:<7} {bidir_ratio:<12.2f} {spec.description}")


def visualize_bidirectional_pdgs(pdgs: List[PDG], specs: List[BidirectionalPDGSpec]):
    """Create visualizations of the bidirectional PDGs."""
    print("\n=== Creating Bidirectional PDG Visualizations ===")
    
    # Create a large grid visualization
    fig, axes = plt.subplots(4, 5, figsize=(25, 20))
    fig.suptitle('Bidirectional PDG Dataset: 20 PDGs with A→B AND B→A edges', fontsize=16, fontweight='bold')
    
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
        bidirectional_edges = set()
        
        for L, X, Y, α, β, P in pdg.edges("l,X,Y,α,β,P"):
            if X.name != "1" and Y.name != "1":
                G.add_edge(X.name, Y.name, label=L, alpha=α, beta=β)
                edge_labels[(X.name, Y.name)] = f"{L}\nα={α:.1f}, β={β:.1f}"
                
                # Check if this is part of a bidirectional pair
                reverse_edge = (Y.name, X.name)
                if reverse_edge in edge_labels:
                    bidirectional_edges.add((X.name, Y.name))
                    bidirectional_edges.add((Y.name, X.name))
        
        # Layout
        pos = nx.spring_layout(G, seed=42, k=1.5, iterations=50)
        
        # Draw nodes
        node_sizes = [G.nodes[node].get('size', 2) * 200 for node in G.nodes()]
        nx.draw_networkx_nodes(G, pos, node_color='lightblue', 
                              node_size=node_sizes, alpha=0.8, ax=ax)
        
        # Draw edges with different colors for bidirectional
        regular_edges = [(u, v) for u, v in G.edges() if (u, v) not in bidirectional_edges]
        bidir_edges = [(u, v) for u, v in G.edges() if (u, v) in bidirectional_edges]
        
        nx.draw_networkx_edges(G, pos, edgelist=regular_edges, edge_color='gray', 
                              arrows=True, arrowsize=15, ax=ax)
        nx.draw_networkx_edges(G, pos, edgelist=bidir_edges, edge_color='red', 
                              arrows=True, arrowsize=15, ax=ax, width=2)
        
        # Draw labels
        nx.draw_networkx_labels(G, pos, font_size=8, font_weight='bold', ax=ax)
        
        # Draw edge labels (only for small graphs to avoid clutter)
        if len(edge_labels) <= 8:
            nx.draw_networkx_edge_labels(G, pos, edge_labels, font_size=6, ax=ax)
        
        # Title
        bidir_count = len(bidirectional_edges) // 2
        title = f"{spec.name.upper()}\n{spec.pattern} - {spec.num_vars}v, {len(G.edges())}e, {bidir_count}↔"
        ax.set_title(title, fontsize=9, pad=10)
        ax.axis('off')
    
    plt.tight_layout()
    plt.savefig('bidirectional_pdg_dataset.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Create statistics visualization
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Bidirectional PDG Dataset Statistics', fontsize=16, fontweight='bold')
    
    # Extract data
    patterns = list(set(s.pattern for s in specs))
    pattern_data = {pattern: {'total_edges': [], 'bidir_pairs': [], 'unidir_edges': []} 
                   for pattern in patterns}
    
    # Store individual data for scatter plot
    all_vars = []
    all_bidir = []
    all_patterns = []
    
    for pdg, spec in zip(pdgs, specs):
        all_edges = [e for e in pdg.edges("l,X,Y,α,β,P") 
                    if e[1].name != "1" and e[2].name != "1"]
        total_edge_count = len(all_edges)
        
        # Count bidirectional pairs
        edge_pairs = {}
        for L, X, Y, α, β, P in all_edges:
            pair = tuple(sorted([X.name, Y.name]))
            edge_pairs[pair] = edge_pairs.get(pair, 0) + 1
        
        bidir_count = sum(1 for count in edge_pairs.values() if count >= 2)
        unidir_count = sum(1 for count in edge_pairs.values() if count == 1)
        
        pattern_data[spec.pattern]['total_edges'].append(total_edge_count)
        pattern_data[spec.pattern]['bidir_pairs'].append(bidir_count)
        pattern_data[spec.pattern]['unidir_edges'].append(unidir_count)
        
        # Store for scatter plot
        all_vars.append(spec.num_vars)
        all_bidir.append(bidir_count)
        all_patterns.append(spec.pattern)
    
    # Plot 1: Total edges by pattern
    ax1 = axes[0, 0]
    pattern_names = list(pattern_data.keys())
    avg_total_edges = [np.mean(pattern_data[p]['total_edges']) for p in pattern_names]
    ax1.bar(pattern_names, avg_total_edges, color=['lightblue', 'lightcoral', 'lightgreen', 'lightyellow'])
    ax1.set_ylabel('Average Total Edges')
    ax1.set_title('Average Total Edges by Pattern')
    
    # Plot 2: Bidirectional pairs by pattern
    ax2 = axes[0, 1]
    avg_bidir_pairs = [np.mean(pattern_data[p]['bidir_pairs']) for p in pattern_names]
    ax2.bar(pattern_names, avg_bidir_pairs, color=['lightblue', 'lightcoral', 'lightgreen', 'lightyellow'])
    ax2.set_ylabel('Average Bidirectional Pairs')
    ax2.set_title('Average Bidirectional Pairs by Pattern')
    
    # Plot 3: Bidirectional ratio
    ax3 = axes[0, 2]
    bidir_ratios = []
    for pattern in pattern_names:
        ratios = []
        for i, total in enumerate(pattern_data[pattern]['total_edges']):
            bidir_pairs = pattern_data[pattern]['bidir_pairs'][i]
            ratio = (bidir_pairs * 2) / total if total > 0 else 0
            ratios.append(ratio)
        bidir_ratios.append(np.mean(ratios))
    
    ax3.bar(pattern_names, bidir_ratios, color=['lightblue', 'lightcoral', 'lightgreen', 'lightyellow'])
    ax3.set_ylabel('Average Bidirectional Ratio')
    ax3.set_title('Bidirectional Edge Ratio by Pattern')
    
    # Plot 4: Total edges distribution
    ax4 = axes[1, 0]
    all_total_edges = []
    for pattern in pattern_names:
        all_total_edges.extend(pattern_data[pattern]['total_edges'])
    ax4.hist(all_total_edges, bins=10, alpha=0.7, color='lightblue')
    ax4.set_xlabel('Total Edges')
    ax4.set_ylabel('Frequency')
    ax4.set_title('Total Edges Distribution')
    
    # Plot 5: Bidirectional pairs distribution
    ax5 = axes[1, 1]
    all_bidir_pairs = []
    for pattern in pattern_names:
        all_bidir_pairs.extend(pattern_data[pattern]['bidir_pairs'])
    ax5.hist(all_bidir_pairs, bins=8, alpha=0.7, color='lightcoral')
    ax5.set_xlabel('Bidirectional Pairs')
    ax5.set_ylabel('Frequency')
    ax5.set_title('Bidirectional Pairs Distribution')
    
    # Plot 6: Variables vs Bidirectional pairs
    ax6 = axes[1, 2]
    colors = ['lightblue', 'lightcoral', 'lightgreen', 'lightyellow']
    pattern_colors = {pattern: colors[i] for i, pattern in enumerate(pattern_names)}
    
    for pattern in pattern_names:
        pattern_vars = [all_vars[i] for i, p in enumerate(all_patterns) if p == pattern]
        pattern_bidir = [all_bidir[i] for i, p in enumerate(all_patterns) if p == pattern]
        ax6.scatter(pattern_vars, pattern_bidir, label=pattern, 
                   color=pattern_colors[pattern], s=100)
    
    ax6.set_xlabel('Number of Variables')
    ax6.set_ylabel('Number of Bidirectional Pairs')
    ax6.set_title('Variables vs Bidirectional Pairs')
    ax6.legend()
    ax6.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('bidirectional_pdg_statistics.png', dpi=300, bbox_inches='tight')
    plt.show()


def save_bidirectional_dataset(pdgs: List[PDG], specs: List[BidirectionalPDGSpec], base_path: str = "bidirectional_pdg_dataset"):
    """Save the bidirectional dataset."""
    base_path = Path(base_path)
    base_path.mkdir(parents=True, exist_ok=True)
    
    # Save specs as JSON
    specs_data = [asdict(spec) for spec in specs]
    with open(base_path / "specs.json", 'w') as f:
        json.dump(specs_data, f, indent=2)
    
    # Save PDGs as pickle
    with open(base_path / "pdgs.pkl", 'wb') as f:
        pickle.dump(pdgs, f)
    
    print(f"\nBidirectional dataset saved to {base_path}/")
    print("Files created:")
    print(f"  - {base_path}/specs.json (specifications)")
    print(f"  - {base_path}/pdgs.pkl (PDG objects)")


def main():
    """Main function to generate the bidirectional PDG dataset."""
    print("=== Generating Bidirectional PDG Dataset ===")
    
    # Create specifications
    specs = create_bidirectional_pdg_specs()
    print(f"Created {len(specs)} PDG specifications")
    pattern_counts = {}
    for spec in specs:
        pattern_counts[spec.pattern] = pattern_counts.get(spec.pattern, 0) + 1
    for pattern, count in pattern_counts.items():
        print(f"  - {count} {pattern} PDGs")
    
    # Generate PDGs
    generator = BidirectionalPDGGenerator()
    pdgs = []
    
    print("\nGenerating PDGs...")
    for spec in specs:
        print(f"  Generating {spec.name} ({spec.pattern} pattern)...")
        pdg = generator.generate_pdg(spec)
        pdgs.append(pdg)
    
    # Analyze PDGs
    analyze_bidirectional_pdgs(pdgs, specs)
    
    # Create visualizations
    visualize_bidirectional_pdgs(pdgs, specs)
    
    # Save dataset
    save_bidirectional_dataset(pdgs, specs)
    
    print(f"\nGenerated {len(pdgs)} bidirectional PDGs")
    print("Visualizations saved to 'bidirectional_pdg_dataset.png' and 'bidirectional_pdg_statistics.png'")
    
    return pdgs, specs


if __name__ == "__main__":
    pdgs, specs = main()
