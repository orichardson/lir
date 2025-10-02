#!/usr/bin/env python3
"""
PDG Dataset Generator

This module generates random PDGs with different connectivity patterns and creates
a dataset for running attention strategy experiments. It also provides visualization
capabilities for the generated PDGs.
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
from collections import defaultdict

# Add project paths
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).parent))

from pdg.pdg import PDG
from pdg.rv import Variable as Var
from pdg.dist import CPT, ParamCPD


@dataclass
class PDGConfig:
    """Configuration for generating a single PDG."""
    num_vars: int
    num_edges: int
    val_range: Tuple[int, int]
    connectivity_pattern: str
    seed: int
    name: str = ""


@dataclass
class PDGDataset:
    """Container for a collection of PDGs."""
    configs: List[PDGConfig]
    pdgs: List[PDG]
    metadata: Dict[str, Any]


class PDGGenerator:
    """Generator for creating random PDGs with different connectivity patterns."""
    
    def __init__(self, seed: int = 42):
        self.seed = seed
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
    
    def generate_chain_pdg(self, num_vars: int, num_edges: int, val_range: Tuple[int, int]) -> PDG:
        """Generate a linear chain PDG: A->B->C->D->E->F"""
        pdg = PDG()
        varlist: List[Var] = []
        
        # Create variables
        for i in range(num_vars):
            domain_size = random.randint(*val_range)
            var = Var.alph(chr(65 + i), domain_size)
            pdg += var
            varlist.append(var)
        
        # Create chain edges
        edges_added = 0
        for i in range(min(num_edges, num_vars - 1)):
            if i + 1 < len(varlist) and edges_added < num_edges:
                src = varlist[i]
                tgt = varlist[i + 1]
                pdg += CPT.make_random(Var.product([src]), Var.product([tgt]))
                edges_added += 1
        
        return pdg
    
    def generate_star_pdg(self, num_vars: int, num_edges: int, val_range: Tuple[int, int]) -> PDG:
        """Generate a star PDG: A->B, A->C, A->D, etc."""
        pdg = PDG()
        varlist: List[Var] = []
        
        # Create variables
        for i in range(num_vars):
            domain_size = random.randint(*val_range)
            var = Var.alph(chr(65 + i), domain_size)
            pdg += var
            varlist.append(var)
        
        # Create star edges (center -> leaves)
        center = varlist[0]
        edges_added = 0
        for i in range(1, min(num_edges + 1, num_vars)):
            if edges_added < num_edges:
                leaf = varlist[i]
                pdg += CPT.make_random(Var.product([center]), Var.product([leaf]))
                edges_added += 1
        
        return pdg
    
    def generate_tree_pdg(self, num_vars: int, num_edges: int, val_range: Tuple[int, int]) -> PDG:
        """Generate a tree PDG: balanced binary tree structure"""
        pdg = PDG()
        varlist: List[Var] = []
        
        # Create variables
        for i in range(num_vars):
            domain_size = random.randint(*val_range)
            var = Var.alph(chr(65 + i), domain_size)
            pdg += var
            varlist.append(var)
        
        # Create tree edges
        edges_added = 0
        for i in range(num_vars // 2):
            if edges_added >= num_edges:
                break
            parent = varlist[i]
            left_child = varlist[2 * i + 1] if 2 * i + 1 < num_vars else None
            right_child = varlist[2 * i + 2] if 2 * i + 2 < num_vars else None
            
            if left_child and edges_added < num_edges:
                pdg += CPT.make_random(Var.product([parent]), Var.product([left_child]))
                edges_added += 1
            if right_child and edges_added < num_edges:
                pdg += CPT.make_random(Var.product([parent]), Var.product([right_child]))
                edges_added += 1
        
        return pdg
    
    def generate_random_pdg(self, num_vars: int, num_edges: int, val_range: Tuple[int, int]) -> PDG:
        """Generate a random connected PDG"""
        pdg = PDG()
        varlist: List[Var] = []
        
        # Create variables
        for i in range(num_vars):
            domain_size = random.randint(*val_range)
            var = Var.alph(chr(65 + i), domain_size)
            pdg += var
            varlist.append(var)
        
        # Create random edges but ensure connectivity
        G = nx.Graph()
        G.add_nodes_from(range(num_vars))
        
        # Add edges randomly but ensure connectivity
        edges_added = 0
        while edges_added < num_edges and not nx.is_connected(G):
            src_idx = random.randint(0, num_vars - 1)
            tgt_idx = random.randint(0, num_vars - 1)
            if src_idx != tgt_idx and not G.has_edge(src_idx, tgt_idx):
                G.add_edge(src_idx, tgt_idx)
                src = varlist[src_idx]
                tgt = varlist[tgt_idx]
                pdg += CPT.make_random(Var.product([src]), Var.product([tgt]))
                edges_added += 1
        
        return pdg
    
    def generate_cycle_pdg(self, num_vars: int, num_edges: int, val_range: Tuple[int, int]) -> PDG:
        """Generate a cycle PDG: A->B->C->D->A"""
        pdg = PDG()
        varlist: List[Var] = []
        
        # Create variables
        for i in range(num_vars):
            domain_size = random.randint(*val_range)
            var = Var.alph(chr(65 + i), domain_size)
            pdg += var
            varlist.append(var)
        
        # Create cycle edges
        edges_added = 0
        for i in range(min(num_edges, num_vars)):
            if edges_added < num_edges:
                src = varlist[i]
                tgt = varlist[(i + 1) % num_vars]
                pdg += CPT.make_random(Var.product([src]), Var.product([tgt]))
                edges_added += 1
        
        return pdg
    
    def generate_clique_pdg(self, num_vars: int, num_edges: int, val_range: Tuple[int, int]) -> PDG:
        """Generate a clique PDG: every variable connected to every other"""
        pdg = PDG()
        varlist: List[Var] = []
        
        # Create variables
        for i in range(num_vars):
            domain_size = random.randint(*val_range)
            var = Var.alph(chr(65 + i), domain_size)
            pdg += var
            varlist.append(var)
        
        # Create clique edges
        edges_added = 0
        for i in range(num_vars):
            for j in range(i + 1, num_vars):
                if edges_added < num_edges:
                    src = varlist[i]
                    tgt = varlist[j]
                    pdg += CPT.make_random(Var.product([src]), Var.product([tgt]))
                    edges_added += 1
        
        return pdg
    
    def generate_pdg(self, config: PDGConfig) -> PDG:
        """Generate a PDG based on configuration."""
        if config.connectivity_pattern == "chain":
            return self.generate_chain_pdg(config.num_vars, config.num_edges, config.val_range)
        elif config.connectivity_pattern == "star":
            return self.generate_star_pdg(config.num_vars, config.num_edges, config.val_range)
        elif config.connectivity_pattern == "tree":
            return self.generate_tree_pdg(config.num_vars, config.num_edges, config.val_range)
        elif config.connectivity_pattern == "random":
            return self.generate_random_pdg(config.num_vars, config.num_edges, config.val_range)
        elif config.connectivity_pattern == "cycle":
            return self.generate_cycle_pdg(config.num_vars, config.num_edges, config.val_range)
        elif config.connectivity_pattern == "clique":
            return self.generate_clique_pdg(config.num_vars, config.num_edges, config.val_range)
        else:
            raise ValueError(f"Unknown connectivity pattern: {config.connectivity_pattern}")


class PDGVisualizer:
    """Visualizer for PDGs."""
    
    def __init__(self):
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
    
    def visualize_pdg(self, pdg: PDG, title: str = "PDG", save_path: str = None) -> None:
        """Visualize a single PDG."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Plot 1: Graph structure
        G = pdg.graph.to_undirected()
        pos = nx.spring_layout(G, seed=42)
        
        # Draw nodes
        nx.draw_networkx_nodes(G, pos, node_color='lightblue', 
                              node_size=1000, alpha=0.8, ax=ax1)
        
        # Draw edges
        nx.draw_networkx_edges(G, pos, edge_color='gray', 
                              arrows=True, arrowsize=20, ax=ax1)
        
        # Draw labels
        nx.draw_networkx_labels(G, pos, font_size=12, font_weight='bold', ax=ax1)
        
        ax1.set_title(f"{title} - Graph Structure")
        ax1.axis('off')
        
        # Plot 2: Edge information
        edge_info = []
        for L, X, Y, α, β, P in pdg.edges("l,X,Y,α,β,P"):
            edge_info.append({
                'Edge': f"{X.name}→{Y.name}",
                'Label': str(L),
                'Alpha': α,
                'Beta': β,
                'Domain_X': len(X),
                'Domain_Y': len(Y)
            })
        
        if edge_info:
            # Create a simple table-like visualization
            ax2.axis('tight')
            ax2.axis('off')
            
            table_data = []
            headers = ['Edge', 'Label', 'α', 'β', '|X|', '|Y|']
            for info in edge_info:
                table_data.append([
                    info['Edge'],
                    info['Label'],
                    f"{info['Alpha']:.2f}",
                    f"{info['Beta']:.2f}",
                    info['Domain_X'],
                    info['Domain_Y']
                ])
            
            table = ax2.table(cellText=table_data, colLabels=headers,
                            cellLoc='center', loc='center')
            table.auto_set_font_size(False)
            table.set_fontsize(10)
            table.scale(1.2, 1.5)
        
        ax2.set_title(f"{title} - Edge Information")
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def visualize_dataset_overview(self, dataset: PDGDataset, save_path: str = None) -> None:
        """Visualize an overview of the entire dataset."""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('PDG Dataset Overview', fontsize=16, fontweight='bold')
        
        # Group PDGs by connectivity pattern
        pattern_groups = defaultdict(list)
        for i, (config, pdg) in enumerate(zip(dataset.configs, dataset.pdgs)):
            pattern_groups[config.connectivity_pattern].append((config, pdg))
        
        # Plot each pattern type
        patterns = list(pattern_groups.keys())
        for idx, pattern in enumerate(patterns[:6]):  # Limit to 6 patterns
            row = idx // 3
            col = idx % 3
            ax = axes[row, col]
            
            # Get a representative PDG for this pattern
            config, pdg = pattern_groups[pattern][0]
            
            # Draw the graph
            G = pdg.graph.to_undirected()
            pos = nx.spring_layout(G, seed=42)
            
            nx.draw_networkx_nodes(G, pos, node_color='lightblue', 
                                  node_size=500, alpha=0.8, ax=ax)
            nx.draw_networkx_edges(G, pos, edge_color='gray', 
                                  arrows=True, arrowsize=15, ax=ax)
            nx.draw_networkx_labels(G, pos, font_size=10, font_weight='bold', ax=ax)
            
            ax.set_title(f"{pattern.title()} Pattern\n"
                        f"Vars: {config.num_vars}, Edges: {config.num_edges}")
            ax.axis('off')
        
        # Hide unused subplots
        for idx in range(len(patterns), 6):
            row = idx // 3
            col = idx % 3
            axes[row, col].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def plot_dataset_statistics(self, dataset: PDGDataset, save_path: str = None) -> None:
        """Plot statistics about the dataset."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('PDG Dataset Statistics', fontsize=16, fontweight='bold')
        
        # Collect statistics
        patterns = [config.connectivity_pattern for config in dataset.configs]
        num_vars = [config.num_vars for config in dataset.configs]
        num_edges = [config.num_edges for config in dataset.configs]
        val_ranges = [config.val_range for config in dataset.configs]
        
        # Plot 1: Pattern distribution
        pattern_counts = {}
        for pattern in patterns:
            pattern_counts[pattern] = pattern_counts.get(pattern, 0) + 1
        
        ax1 = axes[0, 0]
        ax1.pie(pattern_counts.values(), labels=pattern_counts.keys(), autopct='%1.1f%%')
        ax1.set_title('Distribution of Connectivity Patterns')
        
        # Plot 2: Number of variables distribution
        ax2 = axes[0, 1]
        ax2.hist(num_vars, bins=10, alpha=0.7, edgecolor='black')
        ax2.set_xlabel('Number of Variables')
        ax2.set_ylabel('Frequency')
        ax2.set_title('Distribution of Number of Variables')
        
        # Plot 3: Number of edges distribution
        ax3 = axes[1, 0]
        ax3.hist(num_edges, bins=10, alpha=0.7, edgecolor='black')
        ax3.set_xlabel('Number of Edges')
        ax3.set_ylabel('Frequency')
        ax3.set_title('Distribution of Number of Edges')
        
        # Plot 4: Variables vs Edges scatter plot
        ax4 = axes[1, 1]
        scatter = ax4.scatter(num_vars, num_edges, c=range(len(num_vars)), 
                             cmap='viridis', alpha=0.7)
        ax4.set_xlabel('Number of Variables')
        ax4.set_ylabel('Number of Edges')
        ax4.set_title('Variables vs Edges')
        plt.colorbar(scatter, ax=ax4, label='PDG Index')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()


class PDGDatasetGenerator:
    """Main class for generating PDG datasets."""
    
    def __init__(self, seed: int = 42):
        self.generator = PDGGenerator(seed)
        self.visualizer = PDGVisualizer()
    
    def generate_dataset(self, 
                        num_pdgs: int = 50,
                        var_range: Tuple[int, int] = (3, 8),
                        edge_range: Tuple[int, int] = (2, 10),
                        val_range: Tuple[int, int] = (2, 5),
                        patterns: List[str] = None) -> PDGDataset:
        """Generate a dataset of random PDGs."""
        
        if patterns is None:
            patterns = ["chain", "star", "tree", "random", "cycle", "clique"]
        
        configs = []
        pdgs = []
        
        for i in range(num_pdgs):
            # Random configuration
            num_vars = random.randint(*var_range)
            num_edges = random.randint(*edge_range)
            pattern = random.choice(patterns)
            seed = self.generator.seed + i
            
            config = PDGConfig(
                num_vars=num_vars,
                num_edges=num_edges,
                val_range=val_range,
                connectivity_pattern=pattern,
                seed=seed,
                name=f"pdg_{i:03d}_{pattern}"
            )
            
            # Generate PDG
            pdg = self.generator.generate_pdg(config)
            
            configs.append(config)
            pdgs.append(pdg)
        
        # Create metadata
        metadata = {
            'num_pdgs': num_pdgs,
            'var_range': var_range,
            'edge_range': edge_range,
            'val_range': val_range,
            'patterns': patterns,
            'generation_seed': self.generator.seed
        }
        
        return PDGDataset(configs=configs, pdgs=pdgs, metadata=metadata)
    
    def save_dataset(self, dataset: PDGDataset, base_path: str) -> None:
        """Save dataset to files."""
        base_path = Path(base_path)
        base_path.mkdir(parents=True, exist_ok=True)
        
        # Save configs as JSON
        configs_data = [asdict(config) for config in dataset.configs]
        with open(base_path / "configs.json", 'w') as f:
            json.dump(configs_data, f, indent=2)
        
        # Save metadata
        with open(base_path / "metadata.json", 'w') as f:
            json.dump(dataset.metadata, f, indent=2)
        
        # Save PDGs as pickle (since they contain complex objects)
        with open(base_path / "pdgs.pkl", 'wb') as f:
            pickle.dump(dataset.pdgs, f)
        
        print(f"Dataset saved to {base_path}")
    
    def load_dataset(self, base_path: str) -> PDGDataset:
        """Load dataset from files."""
        base_path = Path(base_path)
        
        # Load configs
        with open(base_path / "configs.json", 'r') as f:
            configs_data = json.load(f)
        configs = [PDGConfig(**config) for config in configs_data]
        
        # Load metadata
        with open(base_path / "metadata.json", 'r') as f:
            metadata = json.load(f)
        
        # Load PDGs
        with open(base_path / "pdgs.pkl", 'rb') as f:
            pdgs = pickle.load(f)
        
        return PDGDataset(configs=configs, pdgs=pdgs, metadata=metadata)


def main():
    """Main function to generate and visualize a PDG dataset."""
    print("=== PDG Dataset Generator ===")
    
    # Create generator
    generator = PDGDatasetGenerator(seed=42)
    
    # Generate dataset
    print("Generating dataset...")
    dataset = generator.generate_dataset(
        num_pdgs=30,
        var_range=(4, 8),
        edge_range=(3, 8),
        val_range=(2, 4),
        patterns=["chain", "star", "tree", "random", "cycle"]
    )
    
    print(f"Generated {len(dataset.pdgs)} PDGs")
    
    # Visualize dataset overview
    print("Creating dataset overview...")
    generator.visualizer.visualize_dataset_overview(dataset, "pdg_dataset_overview.png")
    
    # Plot statistics
    print("Creating statistics plots...")
    generator.visualizer.plot_dataset_statistics(dataset, "pdg_dataset_statistics.png")
    
    # Visualize a few individual PDGs
    print("Visualizing individual PDGs...")
    for i in range(min(3, len(dataset.pdgs))):
        config = dataset.configs[i]
        pdg = dataset.pdgs[i]
        generator.visualizer.visualize_pdg(
            pdg, 
            title=f"PDG {i}: {config.connectivity_pattern}",
            save_path=f"pdg_{i:03d}_{config.connectivity_pattern}.png"
        )
    
    # Save dataset
    print("Saving dataset...")
    generator.save_dataset(dataset, "pdg_dataset")
    
    # Print summary
    print("\n=== Dataset Summary ===")
    print(f"Total PDGs: {len(dataset.pdgs)}")
    print(f"Patterns: {set(config.connectivity_pattern for config in dataset.configs)}")
    print(f"Variable range: {min(config.num_vars for config in dataset.configs)}-{max(config.num_vars for config in dataset.configs)}")
    print(f"Edge range: {min(config.num_edges for config in dataset.configs)}-{max(config.num_edges for config in dataset.configs)}")
    
    return dataset


if __name__ == "__main__":
    dataset = main()
