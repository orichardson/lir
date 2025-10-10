#!/usr/bin/env python3
"""
Inspect the exact structure of PDGs used in the final experiment.
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
from pdg.dist import CPT


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
        
        # Choose a target that already has edges (to create conflict)
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


def analyze_pdg_structure(pdg: PDG, pdg_name: str) -> dict:
    """Analyze the structure of a PDG."""
    
    # Get variables
    vars_info = []
    for v in pdg.vars.values():
        if v.name != "1":
            vars_info.append({
                'name': v.name,
                'domain_size': len(v),
                'domain': list(v)
            })
    
    # Get edges
    edges_info = []
    target_counts = {}  # Count incoming edges per target
    
    for L, X, Y, α, β, P in pdg.edges("l,X,Y,α,β,P"):
        if X.name != "1" and Y.name != "1":
            edges_info.append({
                'label': L,
                'source': X.name,
                'target': Y.name,
                'alpha': α,
                'beta': β,
                'cpd_shape': P.shape if P is not None else None,
                'cpd': P  # Store the actual CPD object
            })
            
            # Count targets
            if Y.name not in target_counts:
                target_counts[Y.name] = 0
            target_counts[Y.name] += 1
    
    # Identify nodes with multiple incoming edges (conflict points)
    conflict_nodes = [node for node, count in target_counts.items() if count > 1]
    
    return {
        'name': pdg_name,
        'num_vars': len(vars_info),
        'num_edges': len(edges_info),
        'vars': vars_info,
        'edges': edges_info,
        'conflict_nodes': conflict_nodes,
        'target_counts': target_counts
    }


def print_pdg_table(pdg_analyses):
    """Print detailed table of PDG structures."""
    
    print("\n" + "="*120)
    print("DETAILED PDG STRUCTURE ANALYSIS")
    print("="*120)
    
    for analysis in pdg_analyses:
        print(f"\n{'='*120}")
        print(f"PDG: {analysis['name']}")
        print(f"{'='*120}")
        
        # Overview
        print(f"\n📊 OVERVIEW:")
        print(f"  • Variables: {analysis['num_vars']}")
        print(f"  • Edges: {analysis['num_edges']}")
        print(f"  • Conflict Nodes (multiple incoming edges): {len(analysis['conflict_nodes'])}")
        if analysis['conflict_nodes']:
            print(f"    → Nodes with conflicts: {', '.join(analysis['conflict_nodes'])}")
        
        # Variables table
        print(f"\n📍 VARIABLES:")
        print(f"  {'Name':<8} {'Domain Size':<15} {'Domain Values':<50}")
        print(f"  {'-'*8} {'-'*15} {'-'*50}")
        for var in analysis['vars']:
            domain_str = ', '.join(var['domain'][:5])
            if len(var['domain']) > 5:
                domain_str += ', ...'
            print(f"  {var['name']:<8} {var['domain_size']:<15} {domain_str:<50}")
        
        # Edges table
        print(f"\n🔗 EDGES:")
        print(f"  {'Label':<10} {'Source':<10} {'Target':<10} {'α':<8} {'β':<8} {'CPD Shape':<15} {'Notes':<30}")
        print(f"  {'-'*10} {'-'*10} {'-'*10} {'-'*8} {'-'*8} {'-'*15} {'-'*30}")
        
        for edge in analysis['edges']:
            target = edge['target']
            notes = ""
            if target in analysis['conflict_nodes']:
                notes = f"⚠️  Conflict! ({analysis['target_counts'][target]} incoming edges)"
            
            cpd_shape_str = f"{edge['cpd_shape']}" if edge['cpd_shape'] else "None"
            print(f"  {edge['label']:<10} {edge['source']:<10} {edge['target']:<10} "
                  f"{edge['alpha']:<8.2f} {edge['beta']:<8.2f} {cpd_shape_str:<15} {notes:<30}")
        
        # Edge list visualization
        print(f"\n🌐 GRAPH STRUCTURE:")
        edge_strs = [f"{e['source']}→{e['target']}" for e in analysis['edges']]
        print(f"  {', '.join(edge_strs)}")
        
        # Incoming edges per node
        print(f"\n📥 INCOMING EDGES PER NODE:")
        for var_name in sorted(analysis['target_counts'].keys()):
            count = analysis['target_counts'][var_name]
            marker = " ⚠️  CONFLICT POINT" if count > 1 else ""
            print(f"  {var_name}: {count} incoming edge(s){marker}")
        
        # CPD Values
        print(f"\n📊 CONDITIONAL PROBABILITY DISTRIBUTIONS (CPDs):")
        print(f"\n  Each CPD represents P(Target | Source)")
        print(f"  Rows = Source variable values, Columns = Target variable values")
        print(f"  Each row sums to 1.0 (probability distribution)\n")
        
        for i, edge in enumerate(analysis['edges'], 1):
            cpd = edge['cpd']
            if cpd is not None:
                print(f"  {'─'*80}")
                print(f"  Edge {i}: {edge['label']} — {edge['source']} → {edge['target']}")
                if edge['target'] in analysis['conflict_nodes']:
                    print(f"         ⚠️  This edge contributes to a conflict at node {edge['target']}")
                print(f"  {'─'*80}")
                
                # Convert CPD to numpy array and display
                cpd_array = cpd.to_numpy()
                
                # Create a nice formatted table
                print(f"\n  P({edge['target']} | {edge['source']}):")
                print()
                
                # Column headers (target variable values)
                col_headers = "  " + " " * 10  # Indent for row labels
                for col_name in cpd.columns:
                    col_headers += f"{str(col_name):>10}"
                print(col_headers)
                print("  " + " " * 10 + "─" * (10 * len(cpd.columns)))
                
                # Rows (source variable values)
                for idx, row_name in enumerate(cpd.index):
                    row_str = f"  {str(row_name):>10} │"
                    for val in cpd_array[idx]:
                        row_str += f"{val:>10.4f}"
                    
                    # Add row sum as verification
                    row_sum = cpd_array[idx].sum()
                    row_str += f"  │ Σ={row_sum:.4f}"
                    print(row_str)
                
                print()  # Empty line between CPDs


def main():
    """Main function."""
    print("Inspecting PDG structures from the final experiment...")
    
    # Define PDGs to test (same as in experiment)
    pdg_configs = [
        (4, 3, "chain_4v_3e", 104),
        (5, 4, "chain_5v_4e", 105),
        (6, 5, "chain_6v_5e", 106),
        (7, 6, "chain_7v_6e", 107),
    ]
    
    pdg_analyses = []
    
    for num_vars, num_edges, pdg_name, seed in pdg_configs:
        print(f"\nGenerating {pdg_name}...")
        pdg = generate_pdg(num_vars, num_edges, seed=seed)
        analysis = analyze_pdg_structure(pdg, pdg_name)
        pdg_analyses.append(analysis)
    
    # Print detailed table
    print_pdg_table(pdg_analyses)
    
    # Summary comparison table
    print("\n" + "="*120)
    print("SUMMARY COMPARISON TABLE")
    print("="*120)
    print(f"\n{'PDG Name':<15} {'Variables':<12} {'Edges':<10} {'Conflict Nodes':<18} {'Avg Domain Size':<18} {'Graph Pattern':<40}")
    print(f"{'-'*15} {'-'*12} {'-'*10} {'-'*18} {'-'*18} {'-'*40}")
    
    for analysis in pdg_analyses:
        avg_domain = np.mean([v['domain_size'] for v in analysis['vars']])
        num_conflicts = len(analysis['conflict_nodes'])
        edge_strs = [f"{e['source']}→{e['target']}" for e in analysis['edges'][:3]]
        pattern = ', '.join(edge_strs)
        if len(analysis['edges']) > 3:
            pattern += f", ... (+{len(analysis['edges'])-3} more)"
        
        print(f"{analysis['name']:<15} {analysis['num_vars']:<12} {analysis['num_edges']:<10} "
              f"{num_conflicts:<18} {avg_domain:<18.2f} {pattern:<40}")


if __name__ == "__main__":
    main()

