# experiments

Optuna-based hyperparameter search and confirmation pipeline for the GFlowNet trajectory-balance normalization experiments.

## Workflow

**Phase 1 — HP search:** 50 Optuna trials per (algorithm, environment) pair, 1 seed each.

```bash
bash experiments/launch_optuna_sweep.sh          # submit SLURM jobs
python experiments/optuna_sweep.py summary --all  # inspect results
```

**Phase 2 — Confirmation:** Top-3 configs per pair, 5 seeds × 4000 iterations each.

```bash
bash experiments/launch_optuna_confirmation.sh
```

**Analysis:** Open `gflownet_results.ipynb` to reproduce all figures and tables.

## Files

| File | Description |
|------|-------------|
| `optuna_sweep.py` | Two-phase Optuna driver (search + confirm). Can also run locally without SLURM. |
| `run_optuna.sh` | SLURM wrapper that activates conda and forwards args to `optuna_sweep.py`. |
| `launch_optuna_sweep.sh` | Submits Phase 1 jobs (4 jobs, one per algorithm). |
| `launch_optuna_confirmation.sh` | Submits Phase 2 jobs (16 jobs, one per algorithm × environment). |
| `gflownet_results.ipynb` | Analysis notebook: tables, training curves, HP landscape plots. |

## Cluster Setup

The shell scripts contain a config block at the top for SLURM partition, wall time, and email. Edit these for your cluster before submitting.
