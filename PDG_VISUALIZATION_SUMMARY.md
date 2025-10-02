# PDG Visualization Summary

## Overview

Successfully created comprehensive visualizations of the PDGs used in the LIR experimental setup. The visualizations show the structure, properties, and attention strategy effects on the generated PDGs.

## Generated Visualizations

### 1. **PDG Structures and Details** (`pdg_visualizations.png`)
- **Grid layout**: 6 PDGs × 2 views (structure + details)
- **Structure view**: NetworkX graph showing nodes, edges, and connectivity
- **Details view**: Tabular information about each edge (α, β, domain sizes)
- **PDGs included**:
  - `small_chain` (3 vars, 2 edges)
  - `small_star` (4 vars, 3 edges) 
  - `medium_chain` (5 vars, 4 edges)
  - `medium_tree` (6 vars, 5 edges)
  - `large_chain` (7 vars, 6 edges)
  - `large_complex` (8 vars, 7 edges)

### 2. **Attention Strategy Effects** (`attention_strategy_visualizations.png`)
- **2×2 grid**: Shows how each attention strategy affects edge weights
- **Edge thickness**: Proportional to β values
- **Strategies visualized**:
  - **Global**: β = 1 for all edges (uniform thickness)
  - **Local**: β = 1 for some edges, 0 for others (mixed thickness)
  - **Node-based**: β = 1 for edges connected to focus node (selective thickness)
  - **Exponential**: β sampled from exponential distribution (variable thickness)

### 3. **PDG Properties Analysis** (`pdg_properties_analysis.png`)
- **2×2 grid**: Statistical analysis of PDG properties
- **Plots included**:
  - Variables vs Edges scatter plot
  - Graph density by PDG
  - Average domain size by PDG
  - Edge-to-variable ratio by PDG

### 4. **Experimental Results** (`lir_experimental_results.png`)
- **2×2 grid**: Results from the LIR experiments
- **Shows**: Performance by strategy, PDG size, and strategy interactions

## PDG Properties Summary

| Name | Variables | Edges | Density | Avg Domain | Description |
|------|-----------|-------|---------|------------|-------------|
| small_chain | 3 | 2 | 0.333 | 2.0 | Small chain structure |
| small_star | 4 | 3 | 0.250 | 2.5 | Small star structure |
| medium_chain | 5 | 4 | 0.200 | 2.4 | Medium chain structure |
| medium_tree | 6 | 5 | 0.167 | 2.5 | Medium tree structure |
| large_chain | 7 | 6 | 0.143 | 2.3 | Large chain structure |
| large_complex | 8 | 7 | 0.125 | 2.6 | Large complex structure |

## Key Observations

### Graph Structure
- **Chain structures**: Linear connectivity patterns
- **Star structures**: Hub-and-spoke connectivity
- **Tree structures**: Hierarchical connectivity
- **Complex structures**: More random connectivity patterns

### Graph Properties
- **Density decreases** with size (larger graphs are sparser)
- **Domain sizes** range from 2-3 values per variable
- **Edge-to-variable ratios** vary by structure type

### Attention Strategy Effects
- **Global strategy**: Uniform attention across all edges
- **Local strategy**: Selective attention to subset of edges
- **Node-based strategy**: Focused attention around specific nodes
- **Exponential strategy**: Probabilistic attention weighting

## Usage

### View Visualizations
```bash
# Run the visualization generator
python src/visualize_pdgs.py

# View all visualizations
python src/view_pdg_visualizations.py
```

### Files Generated
- `pdg_visualizations.png` - Main PDG structure and details
- `attention_strategy_visualizations.png` - Attention strategy effects
- `pdg_properties_analysis.png` - Statistical analysis
- `lir_experimental_results.png` - Experimental results

## Technical Details

### Visualization Tools
- **NetworkX**: Graph structure visualization
- **Matplotlib**: Plotting and layout
- **Seaborn**: Statistical visualizations

### Graph Layout
- **Spring layout**: Force-directed positioning
- **Fixed seed**: Reproducible layouts
- **Node sizing**: Proportional to domain size
- **Edge thickness**: Proportional to attention weights

### Color Scheme
- **Nodes**: Light blue with variable sizing
- **Edges**: Gray with thickness variation
- **Labels**: Black text with clear contrast

## Integration with Experimental Setup

These visualizations complement the experimental setup by providing:
1. **Visual verification** of PDG generation
2. **Understanding** of attention strategy effects
3. **Analysis** of graph properties and their impact on results
4. **Documentation** of the experimental dataset

The visualizations help researchers understand the relationship between graph structure, attention strategies, and experimental outcomes in the LIR framework.
