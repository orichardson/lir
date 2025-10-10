# Attention Strategy Experiments - Quick Start Guide

This repository contains the essential files to reproduce the attention strategy experiments on PDGs (Probabilistic Dependency Graphs).

## 📁 Repository Structure

```
.
├── README.md                                  # Main project documentation
├── environment.yaml                           # Conda environment specification
├── conftest.py                               # Testing configuration
├── COMPREHENSIVE_PDG_REPORT.txt              # Detailed PDG analysis report
├── strategy_resolution_visualization.png     # Main results visualization
│
├── src/
│   ├── pdg/                                  # Core PDG library (essential!)
│   ├── lir__simpler.py                       # LIR training implementation
│   ├── run_strategies_on_pdgs.py            # Main experiment script
│   ├── create_individual_panels.py           # Generate individual panel figures
│   ├── comprehensive_pdg_report.py           # Generate comprehensive PDG report
│   └── inspect_pdg_structures.py             # Inspect PDG structures in detail
│
└── individual_panels/                         # Individual panel outputs (PNG + PDF)
    ├── panel_1_initial_vs_final.{png,pdf}
    ├── panel_2_average_resolution.{png,pdf}
    ├── panel_3_heatmap.{png,pdf}
    └── panel_4_{chain_*}.{png,pdf}           # Individual PDG results
```

## 🚀 Quick Start

### 1. Setup Environment

```bash
conda env create -f environment.yaml
conda activate lir
```

### 2. Run Experiments

#### Run all strategies on all PDGs (generates main visualization):
```bash
python src/run_strategies_on_pdgs.py
```
**Output**: `strategy_resolution_visualization.png`

#### Create individual panel figures (PNG + PDF):
```bash
python src/create_individual_panels.py
```
**Output**: `individual_panels/` directory with 14 files

#### Generate comprehensive PDG analysis report:
```bash
python src/comprehensive_pdg_report.py
```
**Output**: `COMPREHENSIVE_PDG_REPORT.txt`

#### Inspect PDG structures in detail:
```bash
python src/inspect_pdg_structures.py
```

## 📊 Experiments Overview

### PDGs Tested
- **chain_4v_3e**: 4 variables, 3 edges (simple cycle)
- **chain_5v_4e**: 5 variables, 4 edges (highest initial inconsistency: 0.5731)
- **chain_6v_5e**: 6 variables, 5 edges (3-way conflict)
- **chain_7v_6e**: 7 variables, 6 edges (lowest initial inconsistency: 0.0503)

### Attention Strategies
1. **Global** (β=1 for all edges): Optimizes all edges simultaneously
2. **Local** (β=1 for half edges): Focuses on subset of edges
3. **Node-based** (β=1 for edges connected to focus node): Local neighborhood optimization

### Key Results
- **Node-based**: 78.2% average resolution (best)
- **Global**: 78.0% average resolution
- **Local**: 43.8% average resolution (struggles on small PDGs)

## 📈 Output Files

### Main Visualization
`strategy_resolution_visualization.png` - Comprehensive 6-panel visualization showing:
1. Initial vs final inconsistency comparison
2. Average resolution by strategy
3. Resolution heatmap (strategy × PDG)
4. Individual PDG results (4 panels)

### Individual Panels (in `individual_panels/`)
All panels available in both PNG (300 DPI) and PDF formats:
- `panel_1_initial_vs_final`: Bar chart comparing initial and final inconsistency
- `panel_2_average_resolution`: Horizontal bar chart of average resolution by strategy
- `panel_3_heatmap`: Color-coded heatmap of resolution percentages
- `panel_4_chain_4v_3e`: Individual results for chain_4v_3e
- `panel_4_chain_5v_4e`: Individual results for chain_5v_4e
- `panel_4_chain_6v_5e`: Individual results for chain_6v_5e
- `panel_4_chain_7v_6e`: Individual results for chain_7v_6e

### Comprehensive Report
`COMPREHENSIVE_PDG_REPORT.txt` - Detailed analysis including:
- Explanation of inconsistency computation
- Full PDG structures (variables, edges, CPDs)
- Initial inconsistency values
- Conflict node analysis
- Key insights

## 🔬 Understanding the Results

### Inconsistency Computation
```
Inconsistency(μ, PDG) = Σ_edges β × D_KL(μ(Y|X) || CPD(Y|X))
```

Where:
- μ = Inferred joint probability distribution
- D_KL = Kullback-Leibler divergence
- β = Attention weight for each edge

### Why Different Strategies Perform Differently
1. **Small PDGs** (chain_4v_3e): Local strategy fails (8.5%) due to tight coupling
2. **Large PDGs** (chain_7v_6e): All strategies succeed (>85%) due to flexibility
3. **Node-based** slightly outperforms global by focusing computational resources

## 📝 Notes

- All experiments use γ=0.0 (no entropy regularization)
- Training: 20 time steps, 10 outer iterations, 20 inner iterations
- Learning rate: 0.05
- Random seed: 42 (for reproducibility)
- PDG generation seeds: 104, 105, 106, 107 (for chain_4v_3e through chain_7v_6e)

## 🛠️ Customization

To run experiments with different parameters, edit the relevant script:
- PDG configurations: Modify `pdg_configs` list
- Training parameters: Adjust `lir_train()` arguments
- Strategies: Implement new strategy functions

## 📚 Citation

If you use this code, please cite the original LIR work and this experimental setup.

---

**Last Updated**: October 10, 2024
**Repository Status**: Clean - contains only essential files for reproducing experiments

