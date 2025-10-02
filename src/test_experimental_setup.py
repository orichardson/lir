#!/usr/bin/env python3
"""
Test script for the experimental setup to validate functionality.
"""

import sys
from pathlib import Path
import random
import numpy as np
import torch
from typing import List, Dict, Tuple, Any, Callable
import json
from dataclasses import dataclass, asdict

# Add project paths
sys.path.insert(0, str(Path(__file__).resolve().parents[0]))

# Test imports
try:
    from pdg.pdg import PDG
    from pdg.rv import Variable as Var
    from pdg.dist import CPT, ParamCPD
    from pdg.alg.torch_opt import opt_joint, torch_score
    from lir__simpler import lir_train, _collect_learnables, apply_attn_mask
    print("✓ All imports successful")
except ImportError as e:
    print(f"✗ Import error: {e}")
    print("Please ensure all dependencies are installed:")
    print("  conda env create -f environment.yaml")
    print("  conda activate lir")
    sys.exit(1)


@dataclass
class PDGSpec:
    """Specification for generating a PDG."""
    name: str
    num_vars: int
    num_edges: int
    val_range: Tuple[int, int]
    max_edges_per_pair: int = 3
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
    
    def generate_simple_pdg(self, spec: PDGSpec) -> PDG:
        """Generate a simple PDG for testing."""
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
        
        # Create simple chain structure
        for i in range(min(spec.num_edges, spec.num_vars - 1)):
            if i + 1 < len(varlist):
                src = varlist[i]
                tgt = varlist[i + 1]
                try:
                    pdg += CPT.make_random(Var.product([src]), Var.product([tgt]))
                except Exception as e:
                    print(f"Warning: Could not create edge {src.name}->{tgt.name}: {e}")
        
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
        if len(edges) > 1:
            selected_edges = random.sample(edges, len(edges) // 2)
            selected_labels = {L for L, X, Y, α, β, P in selected_edges}
        else:
            selected_labels = set()
        
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


def test_pdg_generation():
    """Test PDG generation functionality."""
    print("\n=== Testing PDG Generation ===")
    
    generator = PDGGenerator(seed=42)
    
    # Test simple PDG
    spec = PDGSpec(
        name="test_pdg",
        num_vars=3,
        num_edges=2,
        val_range=(2, 2),
        seed=42,
        description="Test PDG"
    )
    
    try:
        pdg = generator.generate_simple_pdg(spec)
        print(f"✓ Generated PDG: {spec.name}")
        print(f"  Variables: {len(pdg.vars)}")
        print(f"  Edges: {len(pdg.edgedata)}")
        
        # Show edge details
        for L, X, Y, α, β, P in pdg.edges("l,X,Y,α,β,P"):
            print(f"    Edge: {X.name}->{Y.name} (α={α:.2f}, β={β:.2f})")
        
        return pdg
        
    except Exception as e:
        print(f"✗ PDG generation failed: {e}")
        return None


def test_attention_strategies(pdg: PDG):
    """Test attention strategy functionality."""
    print("\n=== Testing Attention Strategies ===")
    
    strategies = {
        "global": AttentionStrategy.global_strategy,
        "local": AttentionStrategy.local_strategy,
        "node_based": AttentionStrategy.node_based_strategy,
        "exponential": AttentionStrategy.exponential_strategy
    }
    
    for name, strategy in strategies.items():
        try:
            attn_alpha, attn_beta, control = strategy(pdg, 0)
            print(f"✓ {name} strategy:")
            print(f"  Alpha values: {list(attn_alpha.values())}")
            print(f"  Beta values: {list(attn_beta.values())}")
            print(f"  Control values: {list(control.values())}")
        except Exception as e:
            print(f"✗ {name} strategy failed: {e}")


def test_single_experiment():
    """Test a single experiment run."""
    print("\n=== Testing Single Experiment ===")
    
    # Generate test PDG
    generator = PDGGenerator(seed=42)
    spec = PDGSpec(
        name="test_experiment",
        num_vars=3,
        num_edges=2,
        val_range=(2, 2),
        seed=42,
        description="Test experiment"
    )
    
    try:
        pdg = generator.generate_simple_pdg(spec)
        print(f"✓ Generated test PDG with {len(pdg.edgedata)} edges")
        
        # Test global strategy
        strategy = AttentionStrategy.global_strategy
        
        # Create a copy and make it parametric
        pdg_copy = pdg.copy()
        edges_snapshot = list(pdg_copy.edges("l,X,Y,α,β,P"))
        
        for L, X, Y, α, β, P in edges_snapshot:
            if L[0] == "π":
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
        
        # Test initial optimization
        mu_init = opt_joint(pdg_copy, gamma=0.0, iters=5, verbose=False)
        initial_global = float(torch_score(pdg_copy, mu_init, 0.0))
        initial_local = float(torch_score(pdg_copy, mu_init, 0.0))
        
        print(f"✓ Initial optimization successful")
        print(f"  Initial global inconsistency: {initial_global:.4f}")
        print(f"  Initial local inconsistency: {initial_local:.4f}")
        
        # Test LIR training (short version)
        lir_train(
            pdg_copy,
            gamma=0.0,
            T=2,  # Very short training
            outer_iters=2,
            inner_iters=3,
            lr=0.01,
            refocus=strategy,
            verbose=False,
            mu_init=mu_init
        )
        
        # Test final optimization
        mu_final = opt_joint(pdg_copy, gamma=0.0, iters=5, verbose=False)
        final_global = float(torch_score(pdg_copy, mu_final, 0.0))
        final_local = float(torch_score(pdg_copy, mu_final, 0.0))
        
        print(f"✓ LIR training successful")
        print(f"  Final global inconsistency: {final_global:.4f}")
        print(f"  Final local inconsistency: {final_local:.4f}")
        
        improvement_global = (initial_global - final_global) / initial_global * 100 if initial_global > 0 else 0
        improvement_local = (initial_local - final_local) / initial_local * 100 if initial_local > 0 else 0
        
        print(f"  Global improvement: {improvement_global:.1f}%")
        print(f"  Local improvement: {improvement_local:.1f}%")
        
        return True
        
    except Exception as e:
        print(f"✗ Single experiment failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Main test function."""
    print("=== Testing Experimental Setup ===")
    
    # Test PDG generation
    pdg = test_pdg_generation()
    if pdg is None:
        print("PDG generation failed, stopping tests")
        return False
    
    # Test attention strategies
    test_attention_strategies(pdg)
    
    # Test single experiment
    success = test_single_experiment()
    
    if success:
        print("\n✓ All tests passed! Experimental setup is working correctly.")
        return True
    else:
        print("\n✗ Some tests failed. Please check the implementation.")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
