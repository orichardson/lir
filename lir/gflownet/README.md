# lir/gflownet

GFlowNet benchmark suite for comparing trajectory-balance algorithms on HyperGrid environments.

## Files

### `checkpoint.py`

Run bookkeeping and persistence. Provides:

- **`RunState`** — dataclass holding paths for config JSON, checkpoint JSON, CSV results, and plot PNG for a single benchmark run.
- **`prepare_run_state()`** — creates a timestamped `RunState`, snapshots the effective config (base config + env/algo lists) to disk, and computes a SHA-1 config hash for deduplication.
- **`write_completed_checkpoint()`** — writes a completion-marker JSON at the end of a successful run.
- **`build_effective_config()`** — assembles the config snapshot that uniquely identifies a run.

All JSON writes use atomic rename (with retry for NFS/Lustre) to avoid partial files.

### `hypergrid.py`

Defines **`ModifiedHyperGrid`**, a subclass of `torchgfn`'s `gfn.gym.hypergrid.HyperGrid` that adds:

- **Four reward functions**: `original`, `cosine`, `bitwise_xor`, `multiplicative_coprime`. The latter two are rewards with tiered structure and configurable difficulty.
- **Mode analysis** — `mode_mask()`, `mode_ids()`, `n_modes`, `n_mode_states` for tracking how well the policy covers the reward's modal structure. Supports exact enumeration or approximate sampling.
- **Mode-existence validation** — fast constructive checks (`_modes_exist_quick_check`) per reward type to verify at init that at least one state reaches the mode threshold, avoiding silent misconfiguration.
- **GF(2) solver** for bitwise XOR feasibility checks.

### `tb_normalize.py`

Main training and benchmarking script. Run via:

```bash
python -m lir.gflownet.tb_normalize [OPTIONS]
```

**What it does:**

1. Sweeps over (environment, algorithm, seed) combinations.
2. Trains each combo for `n_iterations` steps, logging loss, L1 distance, JSD, and mode coverage at regular intervals.
3. Produces a CSV of all metrics and a multi-panel comparison plot (loss, L1, JSD, mode coverage per environment).

**Key components:**

- **Four algorithms** registered in `ALGORITHM_REGISTRY`:
  - `TBGFlowNet` / `LogPartitionVarianceGFlowNet` — upstream baselines from `torchgfn`.
  - `ModifiedTBGFlowNet` / `ModifiedLogPartitionVarianceGFlowNet` — variants that normalize the TB/VarGrad loss by trajectory length (divide squared scores by T_i) to prevent long trajectories from dominating.
- **`validate()`** — draws fresh policy samples (not training states) and computes L1 distance and JSD against the true reward distribution.
- **`CONFIG` dict** — all hyperparameters with defaults (grid size, lr, schedule, optimizer, replay buffer, grad clipping, etc.). Overridable via CLI args.
- **Replay buffer** support with configurable capacity and batch fraction.
- **`--render_results`** mode to re-plot from an existing CSV without retraining.

**CLI examples:**

```bash
# Run all environments and algorithms with defaults
python -m lir.gflownet.tb_normalize

# Specific env/algo subset with progress bar
python -m lir.gflownet.tb_normalize --envs original cosine --algos TBGFlowNet ModifiedTBGFlowNet --show-progress

# Override training hyperparameters
python -m lir.gflownet.tb_normalize --lr 5e-4 --n_iterations 10000 --batch-size 256

# Re-render plots from saved results
python -m lir.gflownet.tb_normalize --render_results path/to/results.csv
```
