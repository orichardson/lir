
### 🎯 Quick Start for Experiments

1. **Setup Environment:**
   ```bash
   conda env create -f environment.yaml
   conda activate lir
   ```

2. **Run Experiments:**
   ```bash
   # Run all strategies on all PDGs (generates main visualization)
   python src/run_strategies_on_pdgs.py
   
   # Create individual panel figures (PNG + PDF)
   python src/create_individual_panels.py
   
   # Generate comprehensive PDG analysis report
   python src/comprehensive_pdg_report.py
   ```

### Experiment Overview

**PDGs Tested:**
- `chain_4v_3e`: 4 variables, 3 edges (simple cycle, inconsistency: 0.2938)
- `chain_5v_4e`: 5 variables, 4 edges (highest inconsistency: 0.5731)
- `chain_6v_5e`: 6 variables, 5 edges (3-way conflict, inconsistency: 0.2047)
- `chain_7v_6e`: 7 variables, 6 edges (lowest inconsistency: 0.0503)

**Attention Strategies Compared:**
1. **Global** (β=1 for all edges): Optimize all edges simultaneously
2. **Local** (β=1 for half edges): Focus on subset, ignore others
3. **Node-based** (β=1 for edges connected to focus node): Neighborhood optimization

**Key Results:**
- **Node-based**: 78.2% average resolution (best overall)
- **Global**: 78.0% average resolution
- **Local**: 43.8% average resolution (struggles on small PDGs)

### 📁 Experiment Files

**Core Scripts:**
- `src/run_strategies_on_pdgs.py` - Main experiment runner
- `src/create_individual_panels.py` - Generate individual figures (PNG + PDF)
- `src/comprehensive_pdg_report.py` - Detailed PDG analysis with all CPD values
- `src/inspect_pdg_structures.py` - Display PDG structures (variables, edges, conflicts) without running experiments
- `src/lir__simpler.py` - LIR training implementation

**Results:**
- `strategy_resolution_visualization.png` - Main 6-panel visualization
- `individual_panels/` - 14 files (7 panels × 2 formats)
- `COMPREHENSIVE_PDG_REPORT.txt` - Complete analysis report

**Documentation:**
- `EXPERIMENTS_README.md` - Detailed guide for reproducing experiments

For complete documentation on running experiments, see [EXPERIMENTS_README.md](EXPERIMENTS_README.md).

## Index of Important Files (i.e., where to start)

**For LIR Attention Strategy Experiments:**
 * [`EXPERIMENTS_README.md`](EXPERIMENTS_README.md) --- Complete guide for reproducing attention strategy experiments
 * [`src/run_strategies_on_pdgs.py`](src/run_strategies_on_pdgs.py) --- Main experiment script
 * [`strategy_resolution_visualization.png`](strategy_resolution_visualization.png) --- Main results visualization

**For General LIR Development:**
 * `src/pdg/` --- Core PDG library (git submodule)
 * `src/lir__simpler.py` --- LIR training implementation

In general: prototypes and general functions that might ideally be integrated into the pdg repository start in `src/`. Once they are stable, we can merge them into the `pdg` submodule. Experiments and applications should go in their appropriate sub-folders.

