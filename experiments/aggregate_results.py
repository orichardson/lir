#!/usr/bin/env python3
"""Aggregate per-job results CSVs from a sweep into a single dataset and plots.

Usage:
    python experiments/aggregate_results.py                          # default dir
    python experiments/aggregate_results.py --input-dir path/to/dir  # custom dir
    python experiments/aggregate_results.py --completed-only         # skip in-progress
"""

from argparse import ArgumentParser
from pathlib import Path
import json
import re

import matplotlib.pyplot as plt
import pandas as pd

_RUN_CSV_RE = re.compile(r"\d{8}_\d{6}_results\.csv")


def find_result_csvs(results_dir: Path, completed_only: bool = False):
    """Yield (config_dict, csv_path) for each finished run in results_dir."""
    for csv_path in sorted(results_dir.glob("*_results.csv")):
        # Only match timestamped run CSVs (skip aggregated_results.csv etc.).
        if not _RUN_CSV_RE.match(csv_path.name):
            continue
        # Skip partial results files.
        if "_partial_" in csv_path.name:
            continue

        # Try to find the matching config JSON.
        run_id = csv_path.name.replace("_results.csv", "")
        config_path = results_dir / f"{run_id}_config.json"
        checkpoint_path = results_dir / f"{run_id}_checkpoint.json"

        config = {}
        if config_path.exists():
            try:
                with config_path.open() as f:
                    raw = json.load(f)
                    config = raw.get("config_snapshot", {}).get("config", {})
            except json.JSONDecodeError:
                pass  # corrupted config from filesystem race; skip metadata

        if completed_only and checkpoint_path.exists():
            with checkpoint_path.open() as f:
                status = json.load(f).get("status", "")
                if status != "completed":
                    continue

        yield config, csv_path


def augment_df(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """Add hyperparameter columns from the config to the dataframe."""
    for key in ("lr", "beta2", "grad_clip_max_norm", "optimizer",
                "loss_clamp", "replay_capacity", "replay_batch_frac",
                "cosine_schedule", "batch_size", "height", "ndim"):
        if key in config:
            df[key] = config[key]
    return df


def plot_sweep(df: pd.DataFrame, output_path: Path) -> None:
    """Plot loss and L1 distance for each (algorithm, environment), best HP per algo."""
    metrics = [("loss", "Loss"), ("l1_dist", "L1 Distance")]
    envs = sorted(df["environment"].unique())
    algos = sorted(df["algorithm"].unique())

    fig, axes = plt.subplots(
        len(envs), len(metrics),
        figsize=(7 * len(metrics), 4 * len(envs)),
        sharex="col", squeeze=False,
    )

    for row, env_name in enumerate(envs):
        env_df = df[df["environment"] == env_name]
        for col, (metric, label) in enumerate(metrics):
            ax = axes[row, col]
            if env_df.empty:
                ax.set_title(f"{env_name} (no data)")
                continue

            for algo in algos:
                algo_df = env_df[env_df["algorithm"] == algo]
                if algo_df.empty:
                    continue

                # Pick the HP config with the best final L1 distance.
                hp_cols = [c for c in ("lr", "beta2", "grad_clip_max_norm")
                           if c in algo_df.columns]
                if hp_cols:
                    final = (algo_df.groupby(hp_cols)["l1_dist"]
                             .apply(lambda g: g.iloc[-1] if len(g) > 0 else float("inf")))
                    best_idx = final.idxmin()
                    if not isinstance(best_idx, tuple):
                        best_idx = (best_idx,)
                    mask = pd.Series(True, index=algo_df.index)
                    for c, v in zip(hp_cols, best_idx):
                        mask &= algo_df[c] == v
                    best_df = algo_df[mask]
                else:
                    best_df = algo_df

                grouped = (
                    best_df.groupby("iteration")[metric]
                    .agg(["mean", "std"]).sort_index()
                )
                ax.plot(grouped.index, grouped["mean"], label=algo)
                if not grouped["std"].isna().all():
                    ax.fill_between(
                        grouped.index,
                        grouped["mean"] - grouped["std"],
                        grouped["mean"] + grouped["std"],
                        alpha=0.2,
                    )

            if row == 0:
                ax.set_title(label)
            if col == 0:
                ax.set_ylabel(env_name)
            if row == len(envs) - 1:
                ax.set_xlabel("Iteration")

    handles, labels = axes[0, 0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center",
                   ncol=min(len(algos), 4), fontsize="small")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    print(f"Saved plot to {output_path}")


def _deduplicate(combined: pd.DataFrame) -> pd.DataFrame:
    """Remove duplicate rows from configs that were run multiple times.

    For each (config, seed, iteration), keeps the row from the run with
    the most data (rows). Reports duplicates to the user.
    """
    hp_cols = ["algorithm", "environment", "lr", "beta2", "grad_clip_max_norm"]
    available = [c for c in hp_cols if c in combined.columns]
    if not available:
        return combined

    dedup_key = available + ["seed"]
    before = len(combined)

    # Group by config+seed+run, rank runs by row count (most data wins).
    combined["_rank"] = (
        combined.groupby(dedup_key + ["run_csv"])["iteration"]
        .transform("count")
    )
    combined = (
        combined.sort_values("_rank", ascending=False)
        .drop_duplicates(subset=dedup_key + ["iteration"], keep="first")
        .drop(columns=["_rank"])
    )
    after = len(combined)

    if before != after:
        n_dupes = before - after
        dup_runs = (
            combined.groupby(available)["run_csv"]
            .nunique()
            .reset_index(name="n_runs")
        )
        dup_runs = dup_runs[dup_runs["n_runs"] > 1]
        print(f"\nWARNING: Removed {n_dupes} duplicate rows from "
              f"{len(dup_runs)} configs with multiple runs:")
        for _, row in dup_runs.iterrows():
            config_str = ", ".join(f"{c}={row[c]}" for c in available)
            print(f"  {config_str}: {row['n_runs']} runs")

    return combined


def main():
    parser = ArgumentParser(description="Aggregate sweep results.")
    parser.add_argument(
        "--input-dir", type=str,
        default="experiments/sweep_results",
        help="Directory containing per-job result CSVs.",
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Output CSV path (default: <input-dir>/aggregated_results.csv).",
    )
    parser.add_argument(
        "--completed-only", action="store_true",
        help="Only include runs whose checkpoint shows 'completed'.",
    )
    args = parser.parse_args()

    results_dir = Path(args.input_dir).expanduser().resolve()
    if not results_dir.exists():
        raise FileNotFoundError(f"Results directory not found: {results_dir}")

    frames = []
    n_runs = 0
    for config, csv_path in find_result_csvs(results_dir, args.completed_only):
        try:
            df = pd.read_csv(csv_path)
        except pd.errors.EmptyDataError:
            continue
        df = augment_df(df, config)
        df["run_csv"] = csv_path.name
        frames.append(df)
        n_runs += 1

    if not frames:
        print(f"No result CSVs found in {results_dir}")
        return

    combined = pd.concat(frames, ignore_index=True)
    combined = _deduplicate(combined)

    out_csv = Path(args.output) if args.output else results_dir / "aggregated_results.csv"
    combined.to_csv(out_csv, index=False)
    print(f"Aggregated {n_runs} runs ({len(combined)} rows) -> {out_csv}")

    # Summary: best final L1 per (algorithm, environment, lr, beta2, grad_clip).
    hp_cols = [c for c in ("algorithm", "environment", "lr", "beta2",
                           "grad_clip_max_norm") if c in combined.columns]
    if hp_cols:
        summary = (
            combined.groupby(hp_cols)
            .agg(
                final_l1=("l1_dist", "last"),
                final_loss=("loss", "last"),
                max_modes=("n_modes_found", "max"),
                n_rows=("iteration", "count"),
            )
            .sort_values("final_l1")
        )
        summary_path = out_csv.with_name("sweep_summary.csv")
        summary.to_csv(summary_path)
        print(f"Saved summary to {summary_path}")
        print("\nTop 10 configurations by final L1:")
        print(summary.head(10).to_string())

    # Plot best-HP-per-algo comparison.
    plot_path = out_csv.with_suffix(".png")
    plot_sweep(combined, plot_path)


if __name__ == "__main__":
    main()
