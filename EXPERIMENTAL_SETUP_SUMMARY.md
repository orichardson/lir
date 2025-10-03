# Experimental Setup Summary

## Overview

This document provides a comprehensive summary of the current experimental setup for Local Inconsistency Resolution (LIR) training with attention strategies on Probabilistic Dependency Graphs (PDGs).

## Core Components

### 1. **Main Experimental Framework**
- **File**: `src/experimental_setup.py`
- **Purpose**: Centralized experimental framework for LIR training with attention strategies
- **Key Classes**:
  - `PDGGenerator`: Generates PDGs with different patterns and structures
  - `AttentionStrategy`: Collection of attention strategies for beta parameter control
  - `ExperimentalSetup`: Main orchestrator for running experiments

### 2. **LIR Training Implementation**
- **File**: `src/lir__simpler.py`
- **Purpose**: Local Inconsistency Resolution training algorithm
- **Key Functions**:
  - `lir_train()`: Main training function with attention strategy support
  - `_collect_learnables()`: Identifies learnable ParamCPDs in PDGs
  - `apply_attn_mask()`: Applies attention masks to PDG edges

### 3. **Inconsistency Computation**
- **File**: `src/pdg/alg/torch_opt.py`
- **Purpose**: Computes inconsistency scores using torch_score function
- **Key Parameters**:
  - `gamma=0.0`: Eliminates entropy term, focuses on likelihood and conditional information
  - Both global and local inconsistency use identical computation with gamma=0.0

## Attention Strategies

### 1. **Global Strategy**
```python
def global_strategy(pdg, t):
    # β = 1 for all edges
    # α = 0 for all edges
```
- **Description**: All edges are active during training
- **Performance**: Consistently best performing strategy
- **Use Case**: Baseline for comparison

### 2. **Local Strategy**
```python
def local_strategy(pdg, t):
    # β = 1 for randomly selected half of edges
    # β = 0 for remaining edges
    # α = 0 for all edges
```
- **Description**: Randomly selects half the edges to be active
- **Performance**: Variable, often poor performance
- **Use Case**: Tests effect of partial edge activation

### 3. **Node-based Strategy**
```python
def node_based_strategy(pdg, t):
    # β = 1 for edges connected to randomly selected node
    # β = 0 for all other edges
    # α = 0 for all edges
```
- **Description**: Focuses attention on edges connected to a single node
- **Performance**: Frequently fails due to gradient starvation
- **Use Case**: Tests localized attention patterns

### 4. **Exponential Strategy**
```python
def exponential_strategy(pdg, t):
    # β drawn from exponential distribution
    # α = 0 for all edges
```
- **Description**: Uses probabilistic edge weights
- **Performance**: Second best after global strategy
- **Use Case**: Tests probabilistic attention mechanisms

## PDG Generation

### 1. **Pattern Types**
- **Chain**: Linear structures with sequential connections
- **Random**: Unstructured connections between nodes
- **Grid**: 2D grid-like structures
- **Star**: Hub-and-spoke patterns

### 2. **Dataset Variants**

#### **Original Dataset** (`src/test/simple_pdg_dataset/`)
- 6 PDGs with varying complexity
- 3-8 variables, 2-7 edges
- Used in initial experiments

#### **Enhanced Dataset** (`enhanced_pdg_dataset/`)
- 20 PDGs (10 chain + 10 random)
- Multiple edges between same nodes with different CPDs
- 4-8 variables, 3-10 edges
- 11/20 PDGs have multi-edge pairs

#### **Bidirectional Dataset** (`bidirectional_pdg_dataset/`)
- 20 PDGs with true bidirectional edges (A→B AND B→A)
- 4 patterns: chain, random, grid, star (5 each)
- 4-9 variables, 4-19 edges
- 69-89% bidirectional edge ratios

## Experimental Parameters

### **Core Parameters**
- **Gamma (γ)**: 0.0 (eliminates entropy term)
- **Alpha (α)**: 0.0 for all edges (as specified)
- **Beta (β)**: Varies by attention strategy
- **Learning Rate**: 0.01
- **Training Iterations**: T=10, outer_iters=5, inner_iters=10

### **Inconsistency Measurement**
- **Global Inconsistency**: `torch_score(pdg, mu, γ=0.0)`
- **Local Inconsistency**: `torch_score(pdg, mu, γ=0.0)` (identical to global)
- **Improvement**: `(initial - final) / initial * 100`

## Key Experimental Results

### 1. **Gamma=0.0 Impact**
- **Before**: Global used γ=0.001, Local used γ=0.0001
- **After**: Both use γ=0.0 (identical computation)
- **Result**: Global and local inconsistency are now identical measures

### 2. **Strategy Performance Rankings**
1. **Global Strategy**: 24.98% average improvement
2. **Exponential Strategy**: 23.12% average improvement  
3. **Local Strategy**: 8.38% average improvement
4. **Node-based Strategy**: 6.40% average improvement (frequent failures)

### 3. **Chain-specific Results**
- **Global Strategy**: 19.20% average improvement
- **Local Strategy**: 0.49% average improvement
- **Global advantage**: ~39x better performance
- **Larger chains**: Show bigger performance gaps

### 4. **Node-based Strategy Issues**
- **Failure Rate**: 83.3% (5/6 PDG sizes fail)
- **Root Cause**: Gradient starvation when most edges have β=0
- **Exception**: Only succeeds on `medium_tree` (6 variables, 5 edges)

## File Structure

```
src/
├── experimental_setup.py          # Main experimental framework
├── lir__simpler.py               # LIR training implementation
├── chain_global_vs_local_comparison.py  # Chain-specific comparison
├── simple_chain_comparison.py    # Simplified chain experiments
├── generate_enhanced_pdg_dataset.py     # Enhanced dataset generator
├── generate_bidirectional_pdg_dataset.py # Bidirectional dataset generator
├── visualize_pdgs.py             # PDG visualization tools
└── test_experimental_setup.py    # Test script for validation

Datasets/
├── simple_pdg_dataset/           # Original 6 PDGs
├── enhanced_pdg_dataset/         # 20 PDGs with multi-edges
└── bidirectional_pdg_dataset/    # 20 PDGs with bidirectional edges

Results/
├── lir_experimental_results.json # Main experimental results
├── chain_comparison_results.json # Chain-specific results
├── simple_chain_results.json     # Simplified chain results
└── Various visualization files (.png)
```

## Usage Examples

### **Running Full Experiments**
```python
from experimental_setup import ExperimentalSetup

setup = ExperimentalSetup()
results = setup.run_full_experiment()
```

### **Running Chain Comparison**
```python
python src/simple_chain_comparison.py
```

### **Generating New Datasets**
```python
python src/generate_enhanced_pdg_dataset.py
python src/generate_bidirectional_pdg_dataset.py
```

### **Visualizing Results**
```python
python src/view_pdg_visualizations.py
```

## Key Insights

### 1. **Attention Strategy Effectiveness**
- **Global attention** (all edges active) consistently performs best
- **Random edge selection** (local strategy) significantly degrades performance
- **Probabilistic attention** (exponential) provides good alternative to global

### 2. **Graph Structure Impact**
- **Chain graphs**: Show clear global vs local differences
- **Larger graphs**: Exhibit bigger performance gaps
- **Bidirectional edges**: Enable more complex relationship modeling

### 3. **Training Dynamics**
- **Gradient starvation**: Major issue when too many edges are deactivated
- **Consistency**: Global strategy provides deterministic results
- **Stability**: Local strategy shows high variance across iterations

### 4. **Parameter Sensitivity**
- **Gamma=0.0**: Eliminates entropy term, focuses on core inconsistency
- **Alpha=0.0**: As specified, no regularization on conditional information
- **Beta values**: Critical for determining which edges contribute to learning

## Future Directions

### 1. **Strategy Improvements**
- Fix node-based strategy to prevent gradient starvation
- Develop adaptive attention mechanisms
- Explore learned attention patterns

### 2. **Dataset Expansion**
- Add more complex graph structures
- Include temporal dynamics
- Test on real-world probabilistic models

### 3. **Analysis Enhancements**
- Convergence analysis across iterations
- Attention pattern visualization
- Theoretical analysis of strategy effectiveness

## Conclusion

The current experimental setup provides a comprehensive framework for studying attention strategies in LIR training. Key findings show that **global attention consistently outperforms local attention**, with the gap being particularly pronounced in chain structures. The framework is extensible and ready for further research into attention mechanisms for probabilistic dependency graphs.
