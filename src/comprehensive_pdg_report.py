#!/usr/bin/env python3
"""
Generate comprehensive report of all PDGs with full details including CPDs and inconsistency.
"""

import sys
from pathlib import Path
import random
import numpy as np
import torch
import pandas as pd

# Add project paths
sys.path.insert(0, str(Path(__file__).resolve().parents[0]))

from pdg.pdg import PDG
from pdg.rv import Variable as Var
from pdg.dist import CPT, ParamCPD
from pdg.alg.torch_opt import opt_joint, torch_score


def generate_pdg(num_vars: int, num_edges: int, seed: int) -> PDG:
    """Generate a PDG - same as in the experiment."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    pdg = PDG()
    varlist = []
    
    # Create variables
    for i in range(num_vars):
        domain_size = random.randint(2, 3)
        var = Var.alph(chr(65 + i), domain_size)
        pdg += var
        varlist.append(var)
    
    edges_added = 0
    edge_pairs = set()
    target_edges = {}
    
    # First, create a basic chain structure
    base_chain_edges = min(num_vars - 1, max(2, num_edges // 2))
    for i in range(base_chain_edges):
        if i + 1 < len(varlist):
            src = varlist[i]
            tgt = varlist[i + 1]
            
            try:
                pdg += CPT.make_random(Var.product([src]), Var.product([tgt]))
                edges_added += 1
                edge_pairs.add((i, i + 1))
                
                if tgt not in target_edges:
                    target_edges[tgt] = []
                target_edges[tgt].append(src)
            except Exception:
                continue
    
    # Add additional edges to create conflicts
    max_attempts = num_edges * 10
    attempts = 0
    
    while edges_added < num_edges and attempts < max_attempts:
        attempts += 1
        
        if target_edges and random.random() > 0.3:
            tgt = random.choice(list(target_edges.keys()))
            tgt_idx = varlist.index(tgt)
            
            src_idx = random.randint(0, num_vars - 1)
            if src_idx == tgt_idx or (src_idx, tgt_idx) in edge_pairs:
                continue
                
            src = varlist[src_idx]
        else:
            src_idx = random.randint(0, num_vars - 1)
            tgt_idx = random.randint(0, num_vars - 1)
            
            if src_idx == tgt_idx or (src_idx, tgt_idx) in edge_pairs:
                continue
            
            src = varlist[src_idx]
            tgt = varlist[tgt_idx]
        
        try:
            pdg += CPT.make_random(Var.product([src]), Var.product([tgt]))
            edges_added += 1
            edge_pairs.add((src_idx, tgt_idx))
            
            if tgt not in target_edges:
                target_edges[tgt] = []
            target_edges[tgt].append(src)
        except Exception:
            continue
    
    return pdg


def compute_inconsistency(pdg: PDG) -> float:
    """Compute the initial inconsistency of a PDG."""
    pdg_copy = PDG.copy(pdg)
    
    # Convert CPTs to ParamCPD for evaluation
    edges_snapshot = list(pdg_copy.edges("l,X,Y,α,β,P"))
    for L, X, Y, α, β, P in edges_snapshot:
        if L[0] != "π" and X.name != "1" and Y.name != "1":
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
    
    # Get optimal joint distribution
    mu_init = opt_joint(pdg_copy, gamma=0.0, iters=50, verbose=False)
    
    # Compute inconsistency score
    inconsistency = float(torch_score(pdg_copy, mu_init, 0.0))
    
    return inconsistency


def print_comprehensive_report():
    """Generate comprehensive report for all PDGs."""
    
    print("\n" + "="*120)
    print("COMPREHENSIVE PDG ANALYSIS REPORT")
    print("="*120)
    
    # First, explain how inconsistency is computed
    print("\n" + "="*120)
    print("INCONSISTENCY COMPUTATION EXPLANATION")
    print("="*120)
    print("""
The inconsistency score measures how much the inferred joint distribution μ deviates from
the conditional probability distributions (CPDs) specified by the edges.

MATHEMATICAL FORMULA:
━━━━━━━━━━━━━━━━━━━━

Inconsistency(μ, PDG) = Σ_edges β_edge × D_KL(μ(Y|X) || CPD(Y|X))

Where:
  • μ = Joint probability distribution over all variables (inferred from the PDG)
  • μ(Y|X) = Conditional distribution of Y given X in the joint distribution μ
  • CPD(Y|X) = Conditional probability table specified by the edge X→Y
  • β_edge = Attention weight for the edge (typically 1.0, meaning "fully attend")
  • D_KL = Kullback-Leibler divergence (measures difference between distributions)

D_KL(P || Q) = Σ_outcomes P(outcome) × log(P(outcome) / Q(outcome))

STEP-BY-STEP COMPUTATION:
━━━━━━━━━━━━━━━━━━━━━━━━━━

1. START with the PDG structure (variables + edges + CPDs)

2. FIND the optimal joint distribution μ* that minimizes inconsistency:
   μ* = argmin_μ Σ_edges β × D_KL(μ(Y|X) || CPD(Y|X))
   
   This is done using gradient descent optimization (opt_joint function)
   with γ=0.0 (no entropy regularization) for 50 iterations.

3. For EACH edge X→Y in the PDG:
   a) Extract μ(Y|X) from the optimal joint distribution μ*
   b) Compare it to the CPD(Y|X) specified by the edge using KL divergence
   c) Weight by β (attention weight, typically 1.0)

4. SUM all the weighted divergences across edges to get total inconsistency

INTERPRETATION:
━━━━━━━━━━━━━━━

• Inconsistency = 0: Perfect consistency (all CPDs agree with the joint distribution)
• Inconsistency > 0: Conflicts exist (CPDs cannot all be satisfied simultaneously)
• Higher values = More severe conflicts between the CPDs

WHY INCONSISTENCY ARISES:
━━━━━━━━━━━━━━━━━━━━━━━━━

1. CYCLES: If B→C→B, then P(C|B) and P(B|C) create circular constraints
2. MULTIPLE PARENTS: If A→C and B→C with conflicting CPDs
3. OVER-SPECIFICATION: More constraints than degrees of freedom
4. MEASUREMENT ERROR: CPDs from different sources with inherent noise

In this experiment, all PDGs have deliberately introduced conflicts via:
  - Multiple edges pointing to the same target node
  - Random CPD initialization (unlikely to be mutually consistent)
  - Cycles in the graph structure
    """)
    
    # Define PDGs
    pdg_configs = [
        (4, 3, "chain_4v_3e", 104),
        (5, 4, "chain_5v_4e", 105),
        (6, 5, "chain_6v_5e", 106),
        (7, 6, "chain_7v_6e", 107),
    ]
    
    all_pdg_data = []
    
    for num_vars, num_edges, pdg_name, seed in pdg_configs:
        print(f"\nProcessing {pdg_name}...")
        pdg = generate_pdg(num_vars, num_edges, seed=seed)
        inconsistency = compute_inconsistency(pdg)
        
        # Collect data
        vars_info = []
        for v in pdg.vars.values():
            if v.name != "1":
                vars_info.append({
                    'name': v.name,
                    'domain_size': len(v),
                    'domain': list(v)
                })
        
        edges_info = []
        target_counts = {}
        
        for L, X, Y, α, β, P in pdg.edges("l,X,Y,α,β,P"):
            if X.name != "1" and Y.name != "1":
                edges_info.append({
                    'label': L,
                    'source': X.name,
                    'target': Y.name,
                    'alpha': α,
                    'beta': β,
                    'cpd': P
                })
                
                if Y.name not in target_counts:
                    target_counts[Y.name] = 0
                target_counts[Y.name] += 1
        
        conflict_nodes = [node for node, count in target_counts.items() if count > 1]
        
        all_pdg_data.append({
            'name': pdg_name,
            'num_vars': num_vars,
            'num_edges': num_edges,
            'vars': vars_info,
            'edges': edges_info,
            'conflict_nodes': conflict_nodes,
            'target_counts': target_counts,
            'inconsistency': inconsistency
        })
    
    # Print comprehensive tables
    for data in all_pdg_data:
        print("\n" + "="*120)
        print(f"PDG: {data['name']}")
        print("="*120)
        
        # Summary
        print(f"\n📋 SUMMARY")
        print(f"  Number of Variables:     {data['num_vars']}")
        print(f"  Number of Edges:         {data['num_edges']}")
        print(f"  Conflict Nodes:          {len(data['conflict_nodes'])} nodes → {', '.join(data['conflict_nodes']) if data['conflict_nodes'] else 'None'}")
        print(f"  Initial Inconsistency:   {data['inconsistency']:.6f}")
        
        # Variables
        print(f"\n📍 VARIABLES")
        print(f"  {'Variable':<12} {'Domain Size':<15} {'Domain Values':<60}")
        print(f"  {'-'*12} {'-'*15} {'-'*60}")
        for var in data['vars']:
            domain_str = ', '.join(var['domain'])
            print(f"  {var['name']:<12} {var['domain_size']:<15} {domain_str:<60}")
        
        # Graph structure
        print(f"\n🌐 GRAPH STRUCTURE")
        edge_strs = [f"{e['source']}→{e['target']}" for e in data['edges']]
        print(f"  {', '.join(edge_strs)}")
        
        # Edge details
        print(f"\n🔗 EDGES SUMMARY")
        print(f"  {'Edge':<8} {'Source':<10} {'Target':<10} {'CPD Shape':<12} {'Conflict?':<40}")
        print(f"  {'-'*8} {'-'*10} {'-'*10} {'-'*12} {'-'*40}")
        for edge in data['edges']:
            cpd_shape = f"{edge['cpd'].shape}"
            conflict = "⚠️  CONFLICT" if edge['target'] in data['conflict_nodes'] else ""
            if conflict:
                conflict += f" ({data['target_counts'][edge['target']]} incoming edges)"
            print(f"  {edge['label']:<8} {edge['source']:<10} {edge['target']:<10} {cpd_shape:<12} {conflict:<40}")
        
        # CPDs
        print(f"\n📊 CONDITIONAL PROBABILITY DISTRIBUTIONS (CPDs)")
        print(f"  Each CPD represents P(Target | Source). Rows sum to 1.0.")
        
        for i, edge in enumerate(data['edges'], 1):
            cpd = edge['cpd']
            print(f"\n  {'─'*100}")
            print(f"  Edge {i}: {edge['label']} — P({edge['target']} | {edge['source']})")
            if edge['target'] in data['conflict_nodes']:
                print(f"          ⚠️  This edge contributes to conflict at node {edge['target']}")
            print(f"  {'─'*100}")
            
            cpd_array = cpd.to_numpy()
            
            # Column headers
            col_width = 12
            header = "  " + " " * 12
            for col in cpd.columns:
                header += f"{str(col):>{col_width}}"
            header += f"{'Sum':>{col_width}}"
            print(header)
            print("  " + " " * 12 + "─" * (col_width * (len(cpd.columns) + 1)))
            
            # Rows
            for idx, row_name in enumerate(cpd.index):
                row_str = f"  {str(row_name):>12} │"
                for val in cpd_array[idx]:
                    row_str += f"{val:>{col_width}.4f}"
                row_sum = cpd_array[idx].sum()
                row_str += f"{row_sum:>{col_width}.4f}"
                print(row_str)
            
            print()
    
    # Final comparison table
    print("\n" + "="*120)
    print("COMPARATIVE SUMMARY TABLE")
    print("="*120)
    print(f"\n{'PDG':<15} {'Vars':<6} {'Edges':<7} {'Conflicts':<11} {'Avg Domain':<12} "
          f"{'Inconsistency':<18} {'Structure':<40}")
    print(f"{'-'*15} {'-'*6} {'-'*7} {'-'*11} {'-'*12} {'-'*18} {'-'*40}")
    
    for data in all_pdg_data:
        avg_domain = np.mean([v['domain_size'] for v in data['vars']])
        structure = ', '.join([f"{e['source']}→{e['target']}" for e in data['edges'][:3]])
        if len(data['edges']) > 3:
            structure += f" (+{len(data['edges'])-3})"
        
        print(f"{data['name']:<15} {data['num_vars']:<6} {data['num_edges']:<7} "
              f"{len(data['conflict_nodes']):<11} {avg_domain:<12.2f} "
              f"{data['inconsistency']:<18.6f} {structure:<40}")
    
    # Analysis
    print("\n" + "="*120)
    print("KEY INSIGHTS")
    print("="*120)
    
    print("\n1. INCONSISTENCY RANKING (Highest to Lowest):")
    sorted_data = sorted(all_pdg_data, key=lambda x: x['inconsistency'], reverse=True)
    for i, data in enumerate(sorted_data, 1):
        print(f"   {i}. {data['name']}: {data['inconsistency']:.6f}")
    
    print("\n2. STRUCTURAL FEATURES:")
    for data in all_pdg_data:
        cycles = []
        edge_dict = {(e['source'], e['target']) for e in data['edges']}
        for e in data['edges']:
            if (e['target'], e['source']) in edge_dict:
                cycle = f"{e['source']}↔{e['target']}"
                if cycle not in cycles and f"{e['target']}↔{e['source']}" not in cycles:
                    cycles.append(cycle)
        
        print(f"\n   {data['name']}:")
        print(f"     • Conflict nodes: {', '.join(data['conflict_nodes']) if data['conflict_nodes'] else 'None'}")
        print(f"     • Cycles: {', '.join(cycles) if cycles else 'None'}")
        print(f"     • Max incoming edges: {max(data['target_counts'].values()) if data['target_counts'] else 0}")


def main():
    print_comprehensive_report()


if __name__ == "__main__":
    main()

