# Gamma=0.0 Experimental Results Analysis

## Overview

This document analyzes the experimental results after setting gamma=0.0 throughout the LIR experimental setup. The key change is that both global and local inconsistency now use identical scoring functions, eliminating the entropy regularization term.

## Key Changes with Gamma=0.0

### Before (Gamma ≠ 0)
- **Global inconsistency**: `torch_score(pdg, mu, γ=0.001)`
- **Local inconsistency**: `torch_score(pdg, mu, γ=0.0001)`
- **Different entropy regularization** levels
- **Different focus**: Global vs local structural consistency

### After (Gamma=0.0)
- **Both global and local inconsistency**: `torch_score(pdg, mu, γ=0.0)`
- **Identical scores**: Global and local inconsistency are now the same
- **No entropy regularization**: `loss -= γ * μ.H(...)` becomes `loss -= 0 * μ.H(...) = 0`
- **Pure likelihood focus**: Only likelihood and conditional information terms

## Experimental Results Summary

### Overall Performance
- **Total experiments**: 24 (6 PDGs × 4 strategies)
- **Successful experiments**: 19/24 (79.2%)
- **Failed experiments**: 5 (mostly node-based strategy)

### Strategy Performance Rankings

| Strategy | Average Improvement | Std Dev | Notes |
|----------|-------------------|---------|-------|
| **Global** | 24.98% | 19.65% | Best overall performance |
| **Exponential** | 23.12% | 17.73% | Second best, more consistent |
| **Local** | 8.38% | 10.16% | Moderate performance |
| **Node-based** | 6.40% | NaN | Limited success (fewer successful runs) |

### PDG Size Performance

| PDG Size | Variables | Edges | Average Improvement | Notes |
|----------|----------|-------|-------------------|-------|
| **medium_chain** | 5 | 4 | 36.97% | **Best performing size** |
| **large_complex** | 8 | 7 | 28.41% | Good performance, low variance |
| **large_chain** | 7 | 6 | 26.27% | Good performance |
| **medium_tree** | 6 | 5 | 16.62% | Moderate performance |
| **small_chain** | 3 | 2 | 4.39% | Minimal improvement |
| **small_star** | 4 | 3 | -3.10% | **Negative improvement** |

### Best Performing Combination
- **PDG**: medium_chain (5 variables, 4 edges)
- **Strategy**: global strategy
- **Improvement**: 47.8% (both global and local identical)

## Key Observations

### 1. Identical Global and Local Scores
With gamma=0.0, the global and local inconsistency scores are now identical:
```json
"initial_global_inconsistency": 0.6953479647636414,
"initial_local_inconsistency": 0.6953479647636414,
"final_global_inconsistency": 0.6586747765541077,
"final_local_inconsistency": 0.6586747765541077
```

### 2. Strategy Effectiveness
- **Global strategy** remains the most effective
- **Exponential strategy** shows good performance with lower variance
- **Local strategy** shows reduced effectiveness
- **Node-based strategy** has more failures

### 3. PDG Size Effects
- **Medium-sized PDGs** (5 variables, 4 edges) perform best
- **Small PDGs** show minimal improvement
- **Large PDGs** show good but variable performance
- **One small PDG** (small_star) shows negative improvement

### 4. Training Stability
- **79.2% success rate** indicates good training stability
- **Node-based strategy** has more gradient-related failures
- **Other strategies** are more robust

## Comparison with Previous Results (Gamma ≠ 0)

### Performance Changes
- **Overall performance** appears similar or slightly lower
- **Strategy rankings** remain consistent (global > exponential > local > node-based)
- **PDG size effects** remain similar

### Key Differences
- **Identical scores**: Global and local inconsistency are now the same
- **Simplified interpretation**: No distinction between global and local measures
- **Pure likelihood focus**: Entropy regularization eliminated

## Implications

### 1. Theoretical
- **Simplified model**: With gamma=0.0, the model focuses purely on likelihood terms
- **No entropy regularization**: The entropy term `γ * μ.H(...)` is eliminated
- **Unified measure**: Global and local inconsistency are now identical

### 2. Practical
- **Easier interpretation**: Single inconsistency measure instead of two
- **Computational efficiency**: No entropy calculations needed
- **Consistent results**: Global and local improvements are always identical

### 3. Research
- **Cleaner experiments**: No need to distinguish between global and local measures
- **Focus on attention strategies**: The main difference is in the β weighting
- **Simplified analysis**: Single improvement metric to track

## Recommendations

### 1. For Further Research
- **Focus on attention strategies**: The main variation is in β weighting schemes
- **Investigate PDG size effects**: Why do medium-sized PDGs perform best?
- **Study strategy failures**: Why does node-based strategy fail more often?

### 2. For Experimental Design
- **Use medium-sized PDGs**: 5 variables, 4 edges show best performance
- **Prefer global strategy**: Most reliable and effective
- **Consider exponential strategy**: Good performance with lower variance

### 3. For Analysis
- **Single metric**: Track only one improvement measure (global = local)
- **Focus on strategy comparison**: The main source of variation
- **Monitor training stability**: Track success rates across strategies

## Conclusion

Setting gamma=0.0 simplifies the experimental setup by:
1. **Eliminating entropy regularization**
2. **Unifying global and local inconsistency measures**
3. **Focusing purely on likelihood-based terms**
4. **Maintaining similar overall performance patterns**

The results show that the attention strategies remain the primary source of variation, with global strategy performing best and medium-sized PDGs being most effective for training.
