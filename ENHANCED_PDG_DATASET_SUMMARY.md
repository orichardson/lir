# Enhanced PDG Dataset Summary

## Overview

This document summarizes the enhanced PDG dataset that was generated with 20 PDGs featuring multiple edges between nodes and different CPDs.

## Dataset Composition

### **Total PDGs: 20**
- **10 Chain PDGs**: Structured chain patterns with potential for multiple edges
- **10 Random PDGs**: Random structure patterns with multiple edges

### **Key Features**
- ✅ **Multiple edges between same nodes** (different CPDs)
- ✅ **Varying numbers of variables** (4-8 variables)
- ✅ **Varying numbers of edges** (3-10 edges)
- ✅ **Different domain sizes** (2-4 values per variable)
- ✅ **Controlled randomness** with fixed seeds for reproducibility

## Detailed Analysis

### **Edge Statistics**
- **Average edges per PDG**: 6.3
- **Edge range**: 3 - 10 edges
- **Average multi-edge pairs**: 0.8
- **Multi-edge range**: 0 - 2 pairs

### **Pattern Comparison**
- **Chain PDGs**: Average 6.0 edges
- **Random PDGs**: Average 6.6 edges

### **Multi-Edge Analysis**
Several PDGs successfully created multiple edges between the same nodes:
- `chain_04`: 2 multi-edge pairs (max 2 edges between same nodes)
- `chain_09`: 1 multi-edge pair (max 3 edges between same nodes)
- `chain_10`: 1 multi-edge pair (max 2 edges between same nodes)
- `random_01`: 1 multi-edge pair (max 2 edges between same nodes)
- `random_02`: 1 multi-edge pair (max 2 edges between same nodes)
- `random_04`: 2 multi-edge pairs (max 2 edges between same nodes)
- `random_05`: 1 multi-edge pair (max 2 edges between same nodes)
- `random_06`: 2 multi-edge pairs (max 2 edges between same nodes)
- `random_07`: 1 multi-edge pair (max 2 edges between same nodes)
- `random_09`: 2 multi-edge pairs (max 2 edges between same nodes)
- `random_10`: 2 multi-edge pairs (max 2 edges between same nodes)

## Individual PDG Specifications

### **Chain PDGs**

| Name | Variables | Edges | Multi-Edge Pairs | Max Edges/Pair | Description |
|------|-----------|-------|------------------|----------------|-------------|
| chain_01 | 4 | 3 | 0 | 1 | Small chain with multiple edges |
| chain_02 | 5 | 4 | 0 | 1 | Medium chain with multiple edges |
| chain_03 | 6 | 5 | 0 | 1 | Large chain with multiple edges |
| chain_04 | 4 | 5 | 2 | 2 | Dense small chain |
| chain_05 | 5 | 6 | 0 | 1 | Dense medium chain |
| chain_06 | 6 | 7 | 0 | 1 | Dense large chain |
| chain_07 | 7 | 6 | 0 | 1 | Long sparse chain |
| chain_08 | 8 | 7 | 0 | 1 | Very long sparse chain |
| chain_09 | 5 | 8 | 1 | 3 | Highly connected medium chain |
| chain_10 | 6 | 9 | 1 | 2 | Highly connected large chain |

### **Random PDGs**

| Name | Variables | Edges | Multi-Edge Pairs | Max Edges/Pair | Description |
|------|-----------|-------|------------------|----------------|-------------|
| random_01 | 4 | 4 | 1 | 2 | Small random with multiple edges |
| random_02 | 5 | 5 | 1 | 2 | Medium random with multiple edges |
| random_03 | 6 | 6 | 0 | 1 | Large random with multiple edges |
| random_04 | 4 | 6 | 2 | 2 | Dense small random |
| random_05 | 5 | 7 | 1 | 2 | Dense medium random |
| random_06 | 6 | 8 | 2 | 2 | Dense large random |
| random_07 | 7 | 5 | 1 | 2 | Sparse large random |
| random_08 | 8 | 6 | 0 | 1 | Sparse very large random |
| random_09 | 5 | 9 | 2 | 2 | Highly connected medium random |
| random_10 | 6 | 10 | 2 | 2 | Highly connected large random |

## Files Generated

### **Dataset Files**
- `enhanced_pdg_dataset/specs.json`: Complete specifications for all 20 PDGs
- `enhanced_pdg_dataset/pdgs.pkl`: Pickled PDG objects for direct use

### **Visualization Files**
- `enhanced_pdg_dataset.png`: Grid visualization of all 20 PDGs
- `enhanced_pdg_statistics.png`: Statistical analysis and comparisons

### **Scripts**
- `src/generate_enhanced_pdg_dataset.py`: Main dataset generator
- `src/view_enhanced_pdg_visualizations.py`: Visualization viewer

## Key Achievements

### **✅ Multiple Edges Between Nodes**
Successfully created PDGs where the same pair of nodes can have multiple edges with different CPDs. This is a significant advancement over the previous dataset.

### **✅ Diverse Structures**
- **Chain patterns**: Linear structures with potential for additional connections
- **Random patterns**: Unstructured connections with multiple edges between nodes

### **✅ Controlled Complexity**
- Variables: 4-8 (manageable but diverse)
- Edges: 3-10 (from sparse to dense)
- Domain sizes: 2-4 (realistic but not overwhelming)

### **✅ Reproducibility**
All PDGs generated with fixed seeds, ensuring consistent results across runs.

## Usage

### **Loading the Dataset**
```python
import pickle
import json

# Load specifications
with open('enhanced_pdg_dataset/specs.json', 'r') as f:
    specs = json.load(f)

# Load PDGs
with open('enhanced_pdg_dataset/pdgs.pkl', 'rb') as f:
    pdgs = pickle.load(f)
```

### **Running Experiments**
This dataset is ready for use with the experimental setup in `src/experimental_setup.py` to test different attention strategies on PDGs with multiple edges between nodes.

## Next Steps

1. **Run LIR experiments** on this enhanced dataset
2. **Compare performance** between chain and random patterns
3. **Analyze multi-edge effects** on attention strategies
4. **Investigate** how multiple CPDs between same nodes affect learning

## Conclusion

The enhanced PDG dataset successfully addresses the requirements:
- ✅ 10 chain PDGs with varying structures
- ✅ 10 random structure PDGs  
- ✅ Multiple edges between same nodes with different CPDs
- ✅ Diverse complexity levels (4-8 variables, 3-10 edges)
- ✅ Controlled randomness for reproducibility

This dataset provides a rich foundation for studying how attention strategies perform on more complex PDG structures with multiple relationships between the same variables.
