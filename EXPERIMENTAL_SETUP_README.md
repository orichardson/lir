# LIR Experimental Setup - Restructured

This document describes the restructured experimental setup for Local Inconsistency Resolution (LIR) research.

## Overview

The codebase has been streamlined to provide a single, clean experimental framework that implements the following requirements:

1. **Single Experimental Function**: All experiments are run through one main function
2. **Flexible PDG Generation**: Creates PDGs with varying numbers of variables and edges, supporting multiple edges between the same nodes
3. **Alpha = 0**: All experiments use α = 0 as specified
4. **Four Attention Strategies**: Implements different beta-based attention strategies
5. **Global vs Local Inconsistency**: Computes both global and local inconsistency measures

## Key Components

### 1. PDG Generation (`PDGGenerator`)

- Generates PDGs with varying structures (3-8 variables, 2-7 edges)
- Supports multiple edges between the same node pairs (up to 3 edges per pair)
- Each edge has different CPDs (Conditional Probability Distributions)
- Uses random domain sizes (2-3 values per variable)

### 2. Attention Strategies (`AttentionStrategy`)

Four different attention strategies for beta values:

#### Global Strategy
- **β = 1** for all edges
- Represents global inconsistency measurement

#### Local Strategy  
- **β = 1** for some edges, **β = 0** for others
- Randomly selects half of the edges to have β = 1

#### Node-Based Strategy
- **β = 1** for edges connected to a randomly selected focus node
- **β = 0** for all other edges
- Tests local attention around specific nodes

#### Exponential Strategy
- **β** drawn from exponential distribution with rate **1/n_edges**
- Tests probabilistic attention weighting

### 3. Inconsistency Measures

#### Global and Local Inconsistency
- Both use `torch_score(pdg, mu, γ=0.0)`
- With gamma=0, the entropy term is eliminated
- Focuses purely on likelihood and conditional information terms

### 4. Experimental Results

The system tracks:
- Initial and final inconsistency scores
- Improvement percentages for both global and local measures
- Success/failure status for each experiment
- Detailed error reporting

## Usage

### Running the Full Experimental Setup

```python
from experimental_setup import main

# Run all experiments
results = main()
```

### Running Individual Components

```python
from experimental_setup import ExperimentalSetup

# Create setup
setup = ExperimentalSetup(seed=42)

# Run experiments
results = setup.run_experiments()

# Analyze results
setup.analyze_results(results)

# Create visualizations
setup.create_visualizations(results)

# Save results
setup.save_results(results)
```

### Testing the Setup

```python
# Run the test script to validate functionality
python test_experimental_setup.py
```

## Results Summary

The experimental setup successfully runs 24 experiments (6 PDGs × 4 strategies) with the following key findings:

### Performance by Strategy
- **Global Strategy**: 43.47% average global improvement
- **Exponential Strategy**: 37.31% average global improvement  
- **Local Strategy**: 16.84% average global improvement
- **Node-Based Strategy**: 6.75% average global improvement

### Performance by PDG Size
- **Medium-sized PDGs** (5 variables, 4 edges) show the best improvement
- **Small PDGs** (3 variables, 2 edges) show minimal improvement
- **Large PDGs** (8 variables, 7 edges) show good improvement but with higher variance

### Best Performing Combination
- **PDG**: medium_chain (5 variables, 4 edges)
- **Strategy**: global strategy
- **Improvement**: 64.8% global, 63.9% local

## File Structure

```
src/
├── experimental_setup.py          # Main experimental framework
├── test_experimental_setup.py     # Test script for validation
└── lir_experimental_results.json  # Results from experiments
```

## Key Features

1. **Streamlined**: Single file contains all experimental logic
2. **Flexible**: Easy to modify PDG generation and attention strategies
3. **Robust**: Comprehensive error handling and validation
4. **Reproducible**: Fixed random seeds for consistent results
5. **Extensible**: Easy to add new attention strategies or PDG types

## Dependencies

- Python 3.9+
- PyTorch 2.3+
- NumPy, Pandas, Matplotlib
- pgmpy (for probabilistic graphical models)
- NetworkX (for graph operations)

## Next Steps

The restructured experimental setup provides a solid foundation for LIR research. Future enhancements could include:

1. **Additional Attention Strategies**: More sophisticated beta weighting schemes
2. **Larger PDG Datasets**: Support for more complex graph structures
3. **Hyperparameter Optimization**: Automated tuning of learning rates and training parameters
4. **Statistical Analysis**: More detailed statistical comparisons between strategies
5. **Visualization Enhancements**: Interactive plots and detailed result exploration
