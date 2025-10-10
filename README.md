# Quickstart

To install:

```
git clone --recurse-submodules git@github.com:orichardson/lir.git
git submodule update --init
git submodule update --remote
```

also ensure `torch`, `pgmpy`, and `numpy` are installed.

to update this repo and all submodules in sync:

```
git pull --recurse-submodules


# Overview of LIR

This repo contains code and writing for the LIR project.

_Local Inconsistency Resolution (LIR)_ is a generic recipe that can be used to derive many algorithms in machine learning. At a high level, the idea is very simple: restrict your attention to a small part of your relevant beliefs, calculate their inconsistency in context, and then resolve that inconsistency by changning each parameter in proportion to the control you have in it. For a more detailed mathematical picture, check out the [most recent draft of the paper]() in the`TeX/` folder.

Here is an inscruitable one-line summary to remind those who have seen this before. Given a parametric model $\mathcal M(\theta)$, attention $\varphi$, control $\chi$, one (repeatedly) makes the following update to the parameter settings $\theta$:

$$
  \theta_{\mathrm{new}} \gets \exp_{\theta_{\mathrm{old}}}\Bigg( -\chi \odot \nabla_\theta \mathllap{\Big\langle~}\Big\langle \varphi \odot \mathcal{M}(\theta) \mathllap{\Big\rangle~}\Big\rangle\Bigg)
$$

The approach is based on the theory of [probabilistic dependenency graphs (PDGs)](https://arxiv.org/abs/2012.10800), which provide a [natural way of measuring inconsistency](http://cs.cornell.edu/~oli/files/oli-dissertation.pdf)
(denoted $\mathllap{\langle}\langle\cdot\mathllap{\rangle}\rangle$),
that [captures and explains many objectives in machine learninng](https://arxiv.org/abs/2202.11862)
as well as graphical models and many other modeling tools in the AI literature.
The present project (LIR) operationalizes this idea, aiming to augment this explanation of what inconsistency is and how to measure it, with an account of how one goes about resolving it. The result unifies a great deal that is know about learning, inference, and decision making.


# Technical Notes for Contributors

The overall structure is roughly as follows:

```
README.md
TeX/
lir/
|-- expts/
|-- pdg/
|-- test/
```

Fragments of math can be found in the `TeX` folder.
Code that aims to test or apply LIR lives in `code/expts/`.
The folder `code/pdg/` is a git submodule that points to the main [PDG repository](https://github.com/orichardson/pdg).
Submodules can sometimes be confusing; what you need to know is summarized below.

## Submodules

The node `pdg/` in this repository is actually [git submodule](https://git-scm.com/book/en/v2/Git-Tools-Submodules); think of it a pointer to the `/lir` branch of the (distinct) [`pdg` repository](https://github.com/orichardson/pdg).

* To start: either clone the repository using
  ```
  git clone --recurse-submodules git@github.com:orichardson/lir.git
  ```
  or run `git submodule update --init` after cloning, to integrate the files from the pdg repository to your local filesystem.
* To update (pull) the submodules:
  ```
  git submodule update --remote
  ```
  To update both this repo and the submodlue, `git pull --recurse-submodules`.
* To push your work on the submodule, commit and push as usual from within the submodule `pdg`. If the change touches anything important and has the possibility of breaking things, do this on a new branch and open a pull request for review. Finally: from this outer repository, run `git add pdg` and commit/push as usual.

***Detached HEAD?***
Submodules have many conceptual and practical benefits. The drawback: git configuration issues can get nastier.
The most common problem is that is easy to get the submodule into a state that refers to a specific commit but does not track a branch. This situation is called a detached HEAD, and can happen whenever pulling a change that includes a new submodule pointer. The danger is that commits to a detached HEAD can easily be lost.

If you see a detached head, run `git submodule update --remote`; the project configuration should re-attach the head.


## Attention Strategy Experiments

This repository now includes comprehensive experiments comparing different attention strategies for Local Inconsistency Resolution on Probabilistic Dependency Graphs (PDGs).

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

### 📊 Experiment Overview

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

