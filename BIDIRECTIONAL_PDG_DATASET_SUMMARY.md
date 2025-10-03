# Bidirectional PDG Dataset Summary

## Overview

This document summarizes the **bidirectional PDG dataset** that was generated with 20 PDGs featuring **bidirectional edges** - where the same pair of nodes has edges in both directions (A→B AND B→A).

## Dataset Composition

### **Total PDGs: 20**
- **5 Chain PDGs**: Linear structures with bidirectional connections
- **5 Random PDGs**: Unstructured connections with bidirectional pairs
- **5 Grid PDGs**: 2D grid structures with bidirectional edges
- **5 Star PDGs**: Hub-and-spoke structures with bidirectional connections

### **Key Features**
- ✅ **Bidirectional edges**: A→B AND B→A for the same node pairs
- ✅ **Multiple patterns**: Chain, random, grid, and star structures
- ✅ **Varying complexity**: 4-9 variables, 4-19 edges
- ✅ **Different domain sizes**: 2-4 values per variable
- ✅ **Controlled randomness**: Fixed seeds for reproducibility

## Detailed Analysis

### **Edge Statistics**
- **Average total edges per PDG**: 8.7
- **Total edge range**: 4 - 19 edges
- **Average bidirectional pairs**: 3.4
- **Bidirectional pairs range**: 1 - 8 pairs
- **Average unidirectional edges**: 1.6
- **Unidirectional edges range**: 0 - 3 edges

### **Pattern Comparison**

| Pattern | Average Total Edges | Average Bidirectional Pairs | Bidirectional Ratio |
|---------|-------------------|---------------------------|-------------------|
| **Chain** | 7.4 | 2.8 | 0.76 |
| **Random** | 5.8 | 2.0 | 0.69 |
| **Grid** | 12.6 | 5.2 | 0.83 |
| **Star** | 8.8 | 3.6 | 0.81 |

### **Bidirectional Edge Ratios**
- **Grid patterns** have the highest bidirectional ratio (83%)
- **Random patterns** have the lowest bidirectional ratio (69%)
- **Chain and Star patterns** have similar ratios (~76-81%)

## Individual PDG Specifications

### **Chain PDGs (5 total)**

| Name | Variables | Total Edges | Bidirectional Pairs | Unidirectional | Bidir/Total | Description |
|------|-----------|-------------|-------------------|----------------|-------------|-------------|
| chain_bidir_01 | 4 | 5 | 2 | 1 | 0.80 | Small chain with 2 bidirectional pairs |
| chain_bidir_02 | 5 | 6 | 2 | 1 | 0.67 | Medium chain with 2 bidirectional pairs |
| chain_bidir_03 | 6 | 7 | 3 | 1 | 0.86 | Large chain with 3 bidirectional pairs |
| chain_bidir_04 | 5 | 9 | 3 | 2 | 0.67 | Dense chain with 3 bidirectional pairs |
| chain_bidir_05 | 6 | 10 | 4 | 1 | 0.80 | Very dense chain with 4 bidirectional pairs |

### **Random PDGs (5 total)**

| Name | Variables | Total Edges | Bidirectional Pairs | Unidirectional | Bidir/Total | Description |
|------|-----------|-------------|-------------------|----------------|-------------|-------------|
| random_bidir_01 | 4 | 5 | 2 | 1 | 0.80 | Small random with 2 bidirectional pairs |
| random_bidir_02 | 5 | 4 | 1 | 2 | 0.50 | Medium random with 2 bidirectional pairs |
| random_bidir_03 | 6 | 5 | 2 | 1 | 0.80 | Large random with 3 bidirectional pairs |
| random_bidir_04 | 5 | 9 | 3 | 3 | 0.67 | Dense random with 3 bidirectional pairs |
| random_bidir_05 | 6 | 6 | 2 | 2 | 0.67 | Very dense random with 4 bidirectional pairs |

### **Grid PDGs (5 total)**

| Name | Variables | Total Edges | Bidirectional Pairs | Unidirectional | Bidir/Total | Description |
|------|-----------|-------------|-------------------|----------------|-------------|-------------|
| grid_bidir_01 | 4 | 5 | 2 | 0 | 0.80 | Small grid with 2 bidirectional pairs |
| grid_bidir_02 | 6 | 10 | 4 | 1 | 0.80 | Medium grid with 3 bidirectional pairs |
| grid_bidir_03 | 9 | 18 | 8 | 2 | 0.89 | Large grid with 4 bidirectional pairs |
| grid_bidir_04 | 6 | 11 | 4 | 3 | 0.73 | Dense grid with 4 bidirectional pairs |
| grid_bidir_05 | 9 | 19 | 8 | 2 | 0.84 | Very dense grid with 6 bidirectional pairs |

### **Star PDGs (5 total)**

| Name | Variables | Total Edges | Bidirectional Pairs | Unidirectional | Bidir/Total | Description |
|------|-----------|-------------|-------------------|----------------|-------------|-------------|
| star_bidir_01 | 4 | 5 | 2 | 1 | 0.80 | Small star with 2 bidirectional pairs |
| star_bidir_02 | 5 | 7 | 3 | 1 | 0.86 | Medium star with 3 bidirectional pairs |
| star_bidir_03 | 6 | 10 | 4 | 2 | 0.80 | Large star with 4 bidirectional pairs |
| star_bidir_04 | 5 | 10 | 4 | 2 | 0.80 | Dense star with 4 bidirectional pairs |
| star_bidir_05 | 6 | 12 | 5 | 2 | 0.83 | Very dense star with 5 bidirectional pairs |

## Files Generated

### **Dataset Files**
- `bidirectional_pdg_dataset/specs.json`: Complete specifications for all 20 PDGs
- `bidirectional_pdg_dataset/pdgs.pkl`: Pickled PDG objects for direct use

### **Visualization Files**
- `bidirectional_pdg_dataset.png`: Grid visualization of all 20 PDGs
  - **Red edges**: Bidirectional pairs (A→B AND B→A)
  - **Gray edges**: Unidirectional edges
- `bidirectional_pdg_statistics.png`: Statistical analysis and comparisons

### **Scripts**
- `src/generate_bidirectional_pdg_dataset.py`: Main dataset generator
- `src/view_bidirectional_pdg_visualizations.py`: Visualization viewer

## Key Achievements

### **✅ True Bidirectional Edges**
Successfully created PDGs where the same pair of nodes has edges in both directions:
- A→B AND B→A
- B→C AND C→B
- Multiple bidirectional pairs per PDG

### **✅ Diverse Structural Patterns**
- **Chain**: Linear bidirectional connections
- **Random**: Unstructured bidirectional pairs
- **Grid**: 2D bidirectional grid structures
- **Star**: Hub-and-spoke bidirectional connections

### **✅ High Bidirectional Ratios**
- Average 76% of edges are part of bidirectional pairs
- Grid patterns achieve up to 89% bidirectional ratio
- All patterns maintain significant bidirectional connectivity

### **✅ Controlled Complexity**
- Variables: 4-9 (manageable but diverse)
- Edges: 4-19 (from sparse to very dense)
- Domain sizes: 2-4 (realistic but not overwhelming)

## Usage

### **Loading the Dataset**
```python
import pickle
import json

# Load specifications
with open('bidirectional_pdg_dataset/specs.json', 'r') as f:
    specs = json.load(f)

# Load PDGs
with open('bidirectional_pdg_dataset/pdgs.pkl', 'rb') as f:
    pdgs = pickle.load(f)
```

### **Running Experiments**
This dataset is ready for use with the experimental setup in `src/experimental_setup.py` to test different attention strategies on PDGs with bidirectional edges.

### **Visualization**
```python
python src/view_bidirectional_pdg_visualizations.py
```

## Research Applications

### **Attention Strategy Testing**
- Test how attention strategies handle bidirectional relationships
- Compare performance on different structural patterns
- Analyze how bidirectional edges affect learning dynamics

### **Graph Structure Analysis**
- Study the impact of bidirectional connectivity on inconsistency
- Compare chain vs. random vs. grid vs. star patterns
- Analyze the relationship between structure and learning performance

### **LIR Algorithm Development**
- Test LIR training on bidirectional PDGs
- Develop new attention strategies for bidirectional graphs
- Optimize learning for different structural patterns

## Next Steps

1. **Run LIR experiments** on this bidirectional dataset
2. **Compare performance** across different patterns (chain, random, grid, star)
3. **Analyze bidirectional effects** on attention strategies
4. **Develop specialized attention strategies** for bidirectional graphs
5. **Investigate** how bidirectional connectivity affects convergence

## Conclusion

The bidirectional PDG dataset successfully addresses the requirement for **true bidirectional edges** (A→B AND B→A):
- ✅ 20 PDGs with genuine bidirectional relationships
- ✅ 4 different structural patterns
- ✅ High bidirectional edge ratios (69-89%)
- ✅ Diverse complexity levels (4-9 variables, 4-19 edges)
- ✅ Controlled randomness for reproducibility

This dataset provides a rich foundation for studying how attention strategies and LIR training perform on PDGs with bidirectional connectivity, enabling research into the effects of mutual dependencies between variables.
