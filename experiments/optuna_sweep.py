#!/usr/bin/env python3
"""Optuna-based hyperparameter search for GFlowNet experiments.

Phase 1 (search): Run 50 Optuna trials per (algorithm, environment) pair.
  Each trial runs 1 seed, bs=1024, 2000 iterations (~8 min/trial on L40S).
  A single job can search multiple envs for one algo (packs 4 envs per GPU).

Phase 2 (confirm): Take the top-3 HP configs per (algo, env) from the Optuna
  DB and run 5 additional seeds at 5000 iterations each.

Usage:
    # Phase 1: search all envs for one algo (one SLURM job)
    python experiments/optuna_sweep.py search --algo TBGFlowNet --envs original cosine bitwise_xor multiplicative_coprime

    # Phase 2: confirm (after all search jobs finish)
    python experiments/optuna_sweep.py confirm --algo TBGFlowNet --env original

    # Phase 2: confirm all combos at once
    python experiments/optuna_sweep.py confirm --all
"""

from argparse import ArgumentParser
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import json
import subprocess
import sys

import optuna
import pandas as pd

# ---------------------------------------------------------------------------
# Constants (mirrored from tb_normalize.py)
# ---------------------------------------------------------------------------

ALGORITHMS = (
    "TBGFlowNet",
    "ModifiedTBGFlowNet",
    "LogPartitionVarianceGFlowNet",
    "ModifiedLogPartitionVarianceGFlowNet",
)
ENVIRONMENTS = ("original", "cosine", "bitwise_xor", "multiplicative_coprime")
TB_ALGORITHMS = ("TBGFlowNet", "ModifiedTBGFlowNet")

REPO_DIR = Path(__file__).resolve().parents[1]
OPTUNA_DIR = REPO_DIR / "experiments" / "optuna_results"

# Fixed training hyperparameters (same as grid sweep).
FIXED_ARGS = [
    "--height", "24",
    "--ndim", "4",
    "--batch-size", "1024",
    "--replay-capacity", "10000",
    "--replay-batch-frac", "0.5",
    "--loss-clamp", "100.0",
    "--lr-schedule", "linear",
    "--device", "cuda",
    "--final-validation-samples", "0",  # skip 10M final JSD during HP search
]

# Search phase settings.
SEARCH_SEED = 42
SEARCH_ITERS = 2000

# Confirmation phase settings.
CONFIRM_SEEDS = 5
CONFIRM_ITERS = 4000


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def study_name(algo: str, env: str) -> str:
    return f"{algo}__{env}"


def db_path(algo: str, env: str) -> Path:
    """One SQLite DB per (algo, env) to avoid NFS locking issues."""
    return OPTUNA_DIR / "db" / f"{algo}__{env}.db"


def storage_url(algo: str, env: str) -> str:
    return f"sqlite:///{db_path(algo, env)}"


def parse_final_l1(output_dir: Path) -> float:
    """Read the last L1 distance from a run's results CSV."""
    csvs = sorted(output_dir.glob("*_results.csv"))
    if not csvs:
        raise FileNotFoundError(f"No results CSV found in {output_dir}")
    df = pd.read_csv(csvs[-1])
    # Get the last recorded L1 (validated at the end of training).
    l1_values = df["l1_dist"].dropna()
    if l1_values.empty:
        return float("inf")
    return float(l1_values.iloc[-1])



def run_training(
    algo: str,
    env: str,
    n_iterations: int,
    n_seeds: int,
    output_dir: Path,
    lr: float,
    beta2: float,
    grad_clip: float,
    lr_logz_multiplier: float,
    seed_offset: int = 0,
) -> None:
    """Launch tb_normalize.py as a subprocess."""
    output_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable, "-u", "-m", "lir.gflownet.tb_normalize",
        "--algos", algo,
        "--envs", env,
        "--n_iterations", str(n_iterations),
        "--n-seeds", str(n_seeds),
        "--lr", str(lr),
        "--beta2", str(beta2),
        "--grad-clip", str(grad_clip),
        "--lr-logz-multiplier", str(lr_logz_multiplier),
        "--output-dir", str(output_dir),
        "--show-progress",
    ] + FIXED_ARGS

    result = subprocess.run(cmd, cwd=str(REPO_DIR), capture_output=True, text=True)
    if result.returncode != 0:
        print(f"STDOUT:\n{result.stdout[-2000:]}", file=sys.stderr)
        print(f"STDERR:\n{result.stderr[-2000:]}", file=sys.stderr)
        raise RuntimeError(
            f"Training failed (exit {result.returncode}) for "
            f"{algo}/{env} lr={lr} beta2={beta2} gc={grad_clip} lzm={lr_logz_multiplier}"
        )


# ---------------------------------------------------------------------------
# Phase 1: Optuna search
# ---------------------------------------------------------------------------

def create_objective(algo: str, env: str, trials_dir: Path):
    """Return an Optuna objective function for the given (algo, env) pair."""
    is_tb = algo in TB_ALGORITHMS

    def objective(trial: optuna.Trial) -> float:
        # --- Suggest hyperparameters ---
        lr = trial.suggest_float("lr", 1e-5, 1e-1, log=True)
        beta2 = trial.suggest_categorical("beta2", [0.99, 0.999, 0.9999])
        grad_clip = trial.suggest_float("grad_clip", 0.01, 10.0, log=True)

        if is_tb:
            lr_logz_multiplier = trial.suggest_float(
                "lr_logz_multiplier", 10.0, 1000.0, log=True
            )
        else:
            lr_logz_multiplier = 1.0

        trial_dir = trials_dir / f"trial_{trial.number:04d}"

        try:
            run_training(
                algo, env,
                n_iterations=SEARCH_ITERS,
                n_seeds=1,
                output_dir=trial_dir,
                lr=lr, beta2=beta2, grad_clip=grad_clip,
                lr_logz_multiplier=lr_logz_multiplier,
            )
            l1_final = parse_final_l1(trial_dir)
        except Exception as e:
            print(f"Trial {trial.number} failed: {e}", file=sys.stderr)
            return float("inf")

        print(
            f"Trial {trial.number} done: L1={l1_final:.4f} "
            f"(lr={lr:.2e}, beta2={beta2}, gc={grad_clip:.3f}"
            + (f", lzm={lr_logz_multiplier:.1f}" if is_tb else "")
            + ")"
        )
        return l1_final

    return objective


def run_search(algo: str, env: str, n_trials: int = 50) -> None:
    """Run Phase 1 Optuna search for a single (algo, env) pair."""
    sname = study_name(algo, env)
    surl = storage_url(algo, env)
    trials_dir = OPTUNA_DIR / "trials" / sname
    trials_dir.mkdir(parents=True, exist_ok=True)
    db_path(algo, env).parent.mkdir(parents=True, exist_ok=True)

    print(f"=== Optuna Search: {algo} / {env} ===")
    print(f"Study: {sname}")
    print(f"DB: {surl}")
    print(f"Trials dir: {trials_dir}")
    print(f"N trials: {n_trials}, Iters per trial: {SEARCH_ITERS}")
    print()

    sampler = optuna.samplers.TPESampler(seed=SEARCH_SEED)

    study = optuna.create_study(
        study_name=sname,
        storage=surl,
        direction="minimize",
        sampler=sampler,
        load_if_exists=True,
    )

    # If resuming, account for already-completed trials (ignore stale RUNNING).
    n_complete = len([
        t for t in study.trials
        if t.state == optuna.trial.TrialState.COMPLETE
    ])
    n_remaining = max(0, n_trials - n_complete)
    if n_complete > 0:
        print(f"Resuming: {n_complete} completed trials, running {n_remaining} more.")

    if n_remaining > 0:
        study.optimize(
            create_objective(algo, env, trials_dir),
            n_trials=n_remaining,
        )

    # Print summary.
    print(f"\n=== Search Complete: {algo} / {env} ===")
    complete = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    print(f"Total trials: {len(study.trials)}, Complete: {len(complete)}")

    if complete:
        print(f"\nBest trial (#{study.best_trial.number}):")
        print(f"  L1 = {study.best_trial.value:.4f}")
        for k, v in study.best_trial.params.items():
            print(f"  {k} = {v}")

        print(f"\nTop 3 trials:")
        sorted_trials = sorted(complete, key=lambda t: t.value)
        for i, t in enumerate(sorted_trials[:3]):
            print(f"  #{t.number}: L1={t.value:.4f} params={t.params}")


# ---------------------------------------------------------------------------
# Phase 2: Confirmation runs
# ---------------------------------------------------------------------------

def get_top_k_configs(algo: str, env: str, top_k: int = 3) -> list[dict]:
    """Load the top-k HP configs from the Optuna study."""
    sname = study_name(algo, env)
    surl = storage_url(algo, env)

    study = optuna.load_study(study_name=sname, storage=surl)

    complete = [
        t for t in study.trials
        if t.state == optuna.trial.TrialState.COMPLETE
    ]
    if not complete:
        raise ValueError(f"No completed trials for {sname}")

    sorted_trials = sorted(complete, key=lambda t: t.value)
    configs = []
    for t in sorted_trials[:top_k]:
        params = dict(t.params)
        params["search_l1"] = t.value
        params["search_trial"] = t.number
        configs.append(params)

    return configs


def run_confirmation(
    algo: str,
    env: str,
    top_k: int = 3,
    n_seeds: int = CONFIRM_SEEDS,
    n_iterations: int = CONFIRM_ITERS,
) -> None:
    """Run Phase 2 confirmation for a single (algo, env) pair."""
    configs = get_top_k_configs(algo, env, top_k)

    print(f"=== Confirmation: {algo} / {env} ===")
    print(f"Top {len(configs)} configs, {n_seeds} seeds each, {n_iterations} iters")
    print()

    confirm_dir = OPTUNA_DIR / "confirm" / study_name(algo, env)
    confirm_dir.mkdir(parents=True, exist_ok=True)

    # Save the selected configs for reproducibility.
    meta_path = confirm_dir / "selected_configs.json"
    with meta_path.open("w") as f:
        json.dump({"algo": algo, "env": env, "configs": configs}, f, indent=2)
    print(f"Saved selected configs to {meta_path}")

    for rank, config in enumerate(configs):
        lr = config["lr"]
        beta2 = config["beta2"]
        grad_clip = config["grad_clip"]
        lr_logz_mult = config.get("lr_logz_multiplier", 1.0)

        print(
            f"\nConfig rank {rank + 1}/{len(configs)} "
            f"(search trial #{config['search_trial']}, L1={config['search_l1']:.4f}):"
        )
        print(
            f"  lr={lr:.2e}, beta2={beta2}, grad_clip={grad_clip:.3f}, "
            f"lr_logz_mult={lr_logz_mult:.1f}"
        )

        run_dir = confirm_dir / f"rank{rank + 1}_trial{config['search_trial']}"

        # Skip if a completed checkpoint already exists (resume-safe).
        checkpoint_files = list(run_dir.glob("*_checkpoint.json"))
        already_done = False
        for cp in checkpoint_files:
            try:
                status = json.loads(cp.read_text()).get("status", "")
                if status == "completed":
                    already_done = True
                    break
            except (json.JSONDecodeError, OSError):
                pass

        if already_done:
            final_l1 = parse_final_l1(run_dir)
            print(f"  SKIP (already complete): L1={final_l1:.4f}")
            continue

        # Remove partial results from a preempted run before re-running.
        if run_dir.exists():
            import shutil
            print(f"  Cleaning up partial results in {run_dir}")
            shutil.rmtree(run_dir)

        run_training(
            algo, env,
            n_iterations=n_iterations,
            n_seeds=n_seeds,
            output_dir=run_dir,
            lr=lr, beta2=beta2, grad_clip=grad_clip,
            lr_logz_multiplier=lr_logz_mult,
        )

        final_l1 = parse_final_l1(run_dir)
        print(f"  Final L1 (mean over {n_seeds} seeds): {final_l1:.4f}")

    print(f"\n=== Confirmation complete: {algo} / {env} ===")
    print(f"Results in: {confirm_dir}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = ArgumentParser(description="Optuna HP search for GFlowNet experiments.")
    subparsers = parser.add_subparsers(dest="mode", required=True)

    # --- search ---
    sp_search = subparsers.add_parser("search", help="Phase 1: Optuna HP search")
    sp_search.add_argument("--algo", required=True, choices=ALGORITHMS)
    sp_search.add_argument("--envs", nargs="+", required=True, choices=ENVIRONMENTS,
                           help="One or more environments to search (run sequentially).")
    sp_search.add_argument("--n-trials", type=int, default=50)
    sp_search.add_argument("--parallel-envs", action="store_true",
                           help="Run environments in parallel (one thread each, shared GPU).")

    # --- confirm ---
    sp_confirm = subparsers.add_parser("confirm", help="Phase 2: confirm top-k configs")
    sp_confirm.add_argument("--algo", choices=ALGORITHMS, default=None)
    sp_confirm.add_argument("--envs", nargs="+", choices=ENVIRONMENTS, default=None,
                            help="One or more environments to confirm.")
    sp_confirm.add_argument("--all", action="store_true", help="Run all (algo, env) pairs")
    sp_confirm.add_argument("--top-k", type=int, default=3)
    sp_confirm.add_argument("--n-seeds", type=int, default=CONFIRM_SEEDS)
    sp_confirm.add_argument("--n-iterations", type=int, default=CONFIRM_ITERS)

    # --- summary ---
    sp_summary = subparsers.add_parser("summary", help="Print search results summary")
    sp_summary.add_argument("--algo", choices=ALGORITHMS, default=None)
    sp_summary.add_argument("--env", choices=ENVIRONMENTS, default=None)
    sp_summary.add_argument("--all", action="store_true")

    args = parser.parse_args()

    if args.mode == "search":
        if args.parallel_envs and len(args.envs) > 1:
            print(f"Running {len(args.envs)} envs in parallel for {args.algo}")
            with ThreadPoolExecutor(max_workers=len(args.envs)) as pool:
                futures = {
                    pool.submit(run_search, args.algo, env, args.n_trials): env
                    for env in args.envs
                }
                for fut in as_completed(futures):
                    env = futures[fut]
                    try:
                        fut.result()
                    except Exception as e:
                        print(f"ERROR: {args.algo}/{env}: {e}", file=sys.stderr)
        else:
            for env in args.envs:
                run_search(args.algo, env, args.n_trials)

    elif args.mode == "confirm":
        if args.all:
            pairs = [(a, e) for a in ALGORITHMS for e in ENVIRONMENTS]
        elif args.algo and args.envs:
            pairs = [(args.algo, e) for e in args.envs]
        else:
            parser.error("Specify --algo and --envs, or use --all")
            return

        for algo, env in pairs:
            try:
                run_confirmation(algo, env, args.top_k, args.n_seeds, args.n_iterations)
            except Exception as e:
                print(f"ERROR: {algo}/{env}: {e}", file=sys.stderr)

    elif args.mode == "summary":
        if args.all:
            pairs = [(a, e) for a in ALGORITHMS for e in ENVIRONMENTS]
        elif args.algo and args.env:
            pairs = [(args.algo, args.env)]
        else:
            parser.error("Specify --algo and --env, or use --all")
            return

        for algo, env in pairs:
            sname = study_name(algo, env)
            surl = storage_url(algo, env)
            if not db_path(algo, env).exists():
                print(f"{sname}: no DB found")
                continue
            study = optuna.load_study(study_name=sname, storage=surl)
            complete = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
            print(f"\n{sname}: {len(complete)} complete / {len(study.trials)} total")
            if complete:
                best = sorted(complete, key=lambda t: t.value)[:3]
                for i, t in enumerate(best):
                    print(f"  #{i+1} trial {t.number}: L1={t.value:.4f} {t.params}")


if __name__ == "__main__":
    main()
