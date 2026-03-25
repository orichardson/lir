## Project Overview

LIR (Local Inconsistency Resolution) is a research project implementing a generic recipe for deriving machine learning algorithms. The core idea: restrict attention to a small part of beliefs, calculate inconsistency in context via PDGs, and resolve it by updating parameters proportional to control. Built on the theory of Probabilistic Dependency Graphs (PDGs).

## Setup

```bash
git clone --recurse-submodules git@github.com:orichardson/lir.git
conda env create -f environment.yaml
conda activate lir
```

Key dependencies: PyTorch (≥2.3), pgmpy, numpy, scipy, pyro-ppl, torchgfn (≥2.3.1).

## Commands

```bash
# Run all tests
pytest

# Run a single test file
pytest lir/test/test_lir_grads.py

# Run tests excluding slow ones
pytest -m "not slow"

# Run the GFlowNet benchmark
python -m lir.gflownet.tb_normalize
```

No formal linting or build system is configured.

## Architecture

### Module Layout

- **`lir/lir__simpler.py`** — Core LIR training loop. Implements `lir_train()` for parameter updates, attention masking (`apply_attn_mask`), and PDG cleanup utilities.
- **`lir/pdg/`** — Git submodule pointing to [orichardson/pdg](https://github.com/orichardson/pdg). Contains the PDG framework:
  - `pdg.py` — Main `PDG` class (graph structure, edges, weights)
  - `dist.py` — Distribution classes: `CPT`, `ParamCPD` (learnable logits), `RawJointDist`
  - `alg/torch_opt.py` — PyTorch-based optimization (`opt_joint`, `torch_score`)
  - `alg/interior_pt.py`, `alg/bp.py`, `alg/cd.py`, `alg/mcmc.py` — Other inference methods
  - `fg.py` — Factor graph representation
  - `rv.py` — Random variable classes
- **`lir/gflownet/`** — GFlowNet benchmark experiments:
  - `tb_normalize.py` — Trajectory Balance orchestration
  - `hypergrid.py` — Modified HyperGrid environment
  - `checkpoint.py` — Run checkpointing and resume
- **`lir/test/`** — Core tests (gradient verification, convergence, pruning)
- **`tests/`** — Additional pytest tests (checkpointing)

### Key Abstractions

- **PDG**: Graph where edges carry distributions (`ParamCPD` with learnable logits) and weights (α for shape, β for scale). The inconsistency measure `⟨⟨·⟩⟩` serves as the loss function.
- **LIR training loop** (`lir_train`): Two-phase optimization — inner step solves for μ* (optimal joint given fixed parameters), outer step updates parameters with μ* detached.
- **Attention masks**: Per-edge α/β weights that restrict which parts of the PDG are considered during updates. Edges with both weights at 0 are removed.
- **`ParamCPD`**: Parametric conditional probability distribution with `logits` tensor that `requires_grad`. This is the main learnable object.

### Import Conventions

The `conftest.py` adds `lir/` to `sys.path`, so imports within `lir/` use bare module names (e.g., `from pdg.dist import ParamCPD`, not `from lir.pdg.dist import ...`). The `lir.gflownet` subpackage uses standard package imports.

### Submodule (`lir/pdg`)

This is a git submodule. After pulling changes that update the submodule pointer, run `git submodule update --remote` to re-sync. Watch for detached HEAD states in the submodule. There is also a legacy `code/pdg` submodule entry.

### Device Handling

Code supports CPU, CUDA, and MPS (Apple Silicon). Device is typically auto-detected. When writing new code, ensure tensors are created on the correct device rather than hardcoding.
