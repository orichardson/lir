from argparse import ArgumentParser
from pathlib import Path
import sys
from datetime import datetime
from typing import cast

import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm

import torch
from gfn.containers.trajectories import Trajectories
from gfn.utils.common import set_seed
from gfn.env import Env
from gfn.estimators import DiscretePolicyEstimator, ScalarEstimator
from gfn.gflownet.base import loss_reduce
from gfn.gflownet.trajectory_balance import (
    LogPartitionVarianceGFlowNet,
    TBGFlowNet,
)
from gfn.utils.modules import MLP, DiscreteUniform
from gfn.preprocessors import KHotPreprocessor
from gfn.utils.training import validate
from gfn.utils.handlers import (
    is_callable_exception_handler,
    warn_about_recalculating_logprobs,
)

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from lir.gflownet.hypergrid import ModifiedHyperGrid


# Static configuration for running the script without argparse.
CONFIG = {
    "device": "mps",
    "height": 16,
    "ndim": 2,
    "uniform_pb": False,
    "lr": 1e-3,
    "lr_logz": 1e-3,
    "n_iterations": 100,
    "batch_size": 32,
    "epsilon": 0.0,
    "validation_interval": 25,
    "validation_samples": 2048,
    "grad_clip_max_norm": 1.0,
    "n_seeds": 5,
    "show_progress": False,
}

DEVICE = torch.device(CONFIG["device"])
EPS = 10**-6


def _maybe_to_device(module: torch.nn.Module | object) -> object:
    """Moves torch modules to the configured device when possible."""
    if hasattr(module, "to"):
        return module.to(DEVICE)
    return module


class ModifiedTBGFlowNet(TBGFlowNet):

    def loss(
        self,
        env: Env,
        trajectories: Trajectories,
        recalculate_all_logprobs: bool = True,
        reduction: str = "mean",
    ) -> torch.Tensor:
        """Computes the trajectory balance loss.

        The trajectory balance loss is described in section 2.3 of
        [Trajectory balance: Improved credit assignment in GFlowNets](
        https://arxiv.org/abs/2201.13259), normalized by the length of the trajectory.

        Args:
            env: The environment where the trajectories are sampled from (unused).
            trajectories: The Trajectories object to compute the loss with.
            recalculate_all_logprobs: Whether to re-evaluate all logprobs.
            reduction: The reduction method to use ('mean', 'sum', or 'none').

        Returns:
            The computed trajectory balance loss as a tensor. The shape depends on the
            reduction method.
        """
        del env  # unused
        warn_about_recalculating_logprobs(trajectories, recalculate_all_logprobs)
        scores = self.get_scores(
            trajectories, recalculate_all_logprobs=recalculate_all_logprobs
        )

        # If the conditions values exist, we pass them to self.logZ
        # (should be a ScalarEstimator or equivalent).
        if trajectories.conditioning is not None:
            with is_callable_exception_handler("logZ", self.logZ):
                assert isinstance(self.logZ, ScalarEstimator)
                logZ = self.logZ(trajectories.conditioning)
        else:
            logZ = self.logZ

        logZ = cast(torch.Tensor, logZ)

        # Calculate the length of each trajectory (+ EPS to avoid divide by zero).
        is_not_sink = ~trajectories.states.is_sink_state
        is_not_initial = ~trajectories.states.is_initial_state
        traj_len = (is_not_sink & is_not_initial).sum(0) + EPS

        # Normalize each batch element by the number of non-terminal, non-dummy,
        # non-initial states.
        scores = (scores + logZ.squeeze()).pow(2) / traj_len

        loss = loss_reduce(scores, reduction)
        if torch.isnan(loss).any():
            raise ValueError("loss is nan")

        return loss


class ModifiedLogPartitionVarianceGFlowNet(LogPartitionVarianceGFlowNet):
    """GFlowNet for the Log Partition Variance loss.

    The log partition variance loss is described in section 3.2 of
    [Robust Scheduling with GFlowNets](https://arxiv.org/abs/2302.05446),
    normalized by the length of the trajectory.

    Attributes:
        pf: The forward policy estimator.
        pb: The backward policy estimator.
        constant_pb: Whether to ignore pb e.g., the GFlowNet DAG is a tree, and pb
            is therefore always 1. Must be set explicitly by user to ensure that pb
            is an Estimator except under this special case.
        log_reward_clip_min: If finite, clips log rewards to this value.
    """

    def loss(
        self,
        env: Env,
        trajectories: Trajectories,
        recalculate_all_logprobs: bool = True,
        reduction: str = "mean",
    ) -> torch.Tensor:
        """Computes the log partition variance loss.

        The log partition variance loss is described in section 3.2 of
        [Robust Scheduling with GFlowNets](https://arxiv.org/abs/2302.05446).

        Args:
            env: The environment where the trajectories are sampled from (unused).
            trajectories: The Trajectories object to compute the loss with.
            recalculate_all_logprobs: Whether to re-evaluate all logprobs.
            reduction: The reduction method to use ('mean', 'sum', or 'none').

        Returns:
            The computed log partition variance loss as a tensor. The shape depends on
            the reduction method.
        """
        del env  # unused
        warn_about_recalculating_logprobs(trajectories, recalculate_all_logprobs)
        scores = self.get_scores(
            trajectories, recalculate_all_logprobs=recalculate_all_logprobs
        )

        # Calculate the length of each trajectory (+ EPS to avoid divide by zero).
        is_not_sink = ~trajectories.states.is_sink_state
        is_not_initial = ~trajectories.states.is_initial_state
        traj_len = (is_not_sink & is_not_initial).sum(0) + EPS

        # Normalize each batch element by the number of non-terminal, non-dummy,
        # non-initial states.
        scores = (scores - scores.mean()).pow(2) / traj_len

        loss = loss_reduce(scores, reduction)
        if torch.isnan(loss).any():
            raise ValueError("loss is NaN.")

        return loss


ENVIRONMENTS = (
    "original",
    "cosine",
    "bitwise_xor",
    "multiplicative_coprime",
)
ALGORITHM_ORDER = (
    "LogPartitionVarianceGFlowNet",
    "ModifiedLogPartitionVarianceGFlowNet",
    "TBGFlowNet",
    "ModifiedTBGFlowNet",
)
RESULTS_DIR = ROOT_DIR / "gflownet" / "gflownet" / "results"


def _build_env(reward_fn_str: str) -> ModifiedHyperGrid:
    """Instantiate a ModifiedHyperGrid configured for benchmarking."""
    return ModifiedHyperGrid(
        ndim=CONFIG["ndim"],
        height=CONFIG["height"],
        reward_fn_str=reward_fn_str,
        reward_fn_kwargs=None,
        device=CONFIG["device"],
        calculate_partition=True,
        store_all_states=True,
        check_action_validity=True,
        validate_modes=True,
        mode_stats="exact",
        mode_stats_samples=20000,
    )


def _build_estimators(env: ModifiedHyperGrid) -> tuple[DiscretePolicyEstimator, DiscretePolicyEstimator]:
    """Build forward and backward estimators tied to the environment."""
    preprocessor = KHotPreprocessor(height=env.height, ndim=env.ndim)
    module_pf = _maybe_to_device(
        MLP(
            input_dim=preprocessor.output_dim,
            output_dim=env.n_actions,
        )
    )
    if not CONFIG["uniform_pb"]:
        module_pb = _maybe_to_device(
            MLP(
                input_dim=preprocessor.output_dim,
                output_dim=env.n_actions - 1,
                trunk=module_pf.trunk,
            )
        )
    else:
        module_pb = _maybe_to_device(DiscreteUniform(output_dim=env.n_actions - 1))

    pf_estimator = DiscretePolicyEstimator(
        module_pf, env.n_actions, preprocessor=preprocessor, is_backward=False
    )
    pb_estimator = DiscretePolicyEstimator(
        module_pb, env.n_actions, preprocessor=preprocessor, is_backward=True
    )
    return pf_estimator, pb_estimator


ALGORITHM_REGISTRY = {
    "LogPartitionVarianceGFlowNet": LogPartitionVarianceGFlowNet,
    "ModifiedLogPartitionVarianceGFlowNet": ModifiedLogPartitionVarianceGFlowNet,
    "TBGFlowNet": TBGFlowNet,
    "ModifiedTBGFlowNet": ModifiedTBGFlowNet,
}


def _instantiate_gflownet(
    algorithm_name: str,
    pf_estimator: DiscretePolicyEstimator,
    pb_estimator: DiscretePolicyEstimator,
) -> torch.nn.Module:
    """Factory for GFlowNet instances."""
    gflownet_cls = ALGORITHM_REGISTRY[algorithm_name]
    if issubclass(gflownet_cls, TBGFlowNet):
        gflownet = gflownet_cls(
            pf=pf_estimator,
            pb=pb_estimator,
            init_logZ=0.0,
        )
    else:
        gflownet = gflownet_cls(
            pf=pf_estimator,
            pb=pb_estimator,
        )
    return cast(torch.nn.Module, _maybe_to_device(gflownet))


def _train_single_run(
    env_name: str,
    algorithm_name: str,
    seed: int,
) -> list[dict[str, float | int | str]]:
    """Train a single (environment, algorithm, seed) combo and capture metrics."""
    set_seed(seed)
    env = _build_env(env_name)
    pf_estimator, pb_estimator = _build_estimators(env)
    gflownet = _instantiate_gflownet(algorithm_name, pf_estimator, pb_estimator)

    optimizer = torch.optim.Adam(gflownet.pf_pb_parameters(), lr=CONFIG["lr"])
    logz_params = (
        list(gflownet.logz_parameters()) if hasattr(gflownet, "logz_parameters") else []
    )
    if logz_params:
        optimizer.add_param_group({"params": logz_params, "lr": CONFIG["lr_logz"]})

    visited_terminating_states = env.states_from_batch_shape((0,))
    discovered_modes: set[int] = set()
    current_l1 = float("inf")
    records: list[dict[str, float | int | str]] = []

    iterator = tqdm(
        range(CONFIG["n_iterations"]),
        dynamic_ncols=True,
        disable=not CONFIG["show_progress"],
        desc=f"{env_name}-{algorithm_name}-seed{seed}",
    )

    for it in iterator:
        trajectories = gflownet.sample_trajectories(
            env,
            n=CONFIG["batch_size"],
            save_logprobs=False,
            save_estimator_outputs=False,
            epsilon=CONFIG["epsilon"],
        )
        visited_terminating_states.extend(trajectories.terminating_states)

        optimizer.zero_grad()
        loss = gflownet.loss_from_trajectories(
            env, trajectories, recalculate_all_logprobs=False
        )
        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            gflownet.parameters(), CONFIG["grad_clip_max_norm"]
        )
        gflownet.assert_finite_gradients()
        optimizer.step()
        gflownet.assert_finite_parameters()

        if (it + 1) % CONFIG["validation_interval"] == 0:
            validation_info, _ = validate(
                env,
                gflownet,
                CONFIG["validation_samples"],
                visited_terminating_states,
            )
            current_l1 = validation_info.get("l1_dist", current_l1)

        modes_found = env.modes_found(visited_terminating_states)
        discovered_modes.update(modes_found)
        n_modes_found = len(discovered_modes)
        n_terminating_states = len(visited_terminating_states)

        records.append(
            {
                "environment": env_name,
                "algorithm": algorithm_name,
                "seed": seed,
                "iteration": it + 1,
                "loss": loss.item(),
                "l1_dist": current_l1,
                "n_modes_found": n_modes_found,
                "n_terminating_states": n_terminating_states,
            }
        )

        if CONFIG["show_progress"]:
            iterator.set_postfix({"loss": loss.item(), "n_modes": n_modes_found})

    return records


def _plot_results(results_df: pd.DataFrame, output_path: Path) -> None:
    """Create comparison plots (loss, L1 distance, modes) per environment."""
    metrics = [
        ("loss", "Loss"),
        ("l1_dist", "L1 distance"),
        ("n_modes_found", "Modes found"),
    ]
    envs = list(ENVIRONMENTS)
    fig, axes = plt.subplots(
        len(envs),
        len(metrics),
        figsize=(6 * len(metrics), 4 * len(envs)),
        sharex="col",
        squeeze=False,
    )

    for row_idx, env_name in enumerate(envs):
        env_df = results_df[results_df["environment"] == env_name]
        for col_idx, (metric_key, metric_label) in enumerate(metrics):
            ax = axes[row_idx, col_idx]
            if env_df.empty:
                ax.set_title(f"{env_name} (no data)")
                ax.set_xlabel("Iteration")
                continue

            for algo_name in ALGORITHM_ORDER:
                algo_df = env_df[env_df["algorithm"] == algo_name]
                if algo_df.empty:
                    continue
                grouped = (
                    algo_df.groupby("iteration")[metric_key]
                    .agg(["mean", "std"])
                    .sort_index()
                )
                ax.plot(grouped.index, grouped["mean"], label=algo_name)
                if not grouped["std"].isna().all():
                    ax.fill_between(
                        grouped.index,
                        grouped["mean"] - grouped["std"],
                        grouped["mean"] + grouped["std"],
                        alpha=0.2,
                    )

            if row_idx == 0:
                ax.set_title(metric_label)
            if col_idx == 0:
                ax.set_ylabel(env_name)
            if row_idx == len(envs) - 1:
                ax.set_xlabel("Iteration")

    handles, labels = axes[0, 0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=len(ALGORITHM_ORDER))
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def run_benchmark(
    env_names: tuple[str, ...] | list[str] | None = None,
) -> tuple[pd.DataFrame, Path, Path]:
    """Execute the benchmark sweep over the requested environments and persist results."""
    active_envs = tuple(env_names) if env_names is not None else ENVIRONMENTS
    if not active_envs:
        raise ValueError("No environments specified for benchmarking.")

    all_records: list[dict[str, float | int | str]] = []
    for env_name in active_envs:
        for algo_name in ALGORITHM_ORDER:
            # Lucky number 7 lets go!
            for seed in range(7, 7 * CONFIG["n_seeds"] + 1, 7):
                all_records.extend(_train_single_run(env_name, algo_name, seed))

    results_df = pd.DataFrame(all_records)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = RESULTS_DIR / f"{timestamp}_results.csv"
    results_df.to_csv(csv_path, index=False)

    plot_path = RESULTS_DIR / f"{timestamp}_results.png"
    _plot_results(results_df, plot_path)
    return results_df, csv_path, plot_path


def main():
    parser = ArgumentParser(description="Run HyperGrid GFlowNet benchmark.")
    parser.add_argument(
        "--n_iterations",
        type=int,
        default=None,
        help="Override number of training iterations per run.",
    )
    parser.add_argument(
        "--envs",
        nargs="+",
        choices=ENVIRONMENTS,
        help="Subset of environments to benchmark (default: all).",
    )
    parser.add_argument(
        "--show-progress",
        action="store_true",
        help="Display tqdm progress bars during training.",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=None,
        help="Override HyperGrid height.",
    )
    parser.add_argument(
        "--ndim",
        type=int,
        default=None,
        help="Override HyperGrid dimensionality.",
    )
    parser.add_argument(
        "--render_results",
        type=str,
        default=None,
        help="Path to an existing results CSV to render plots (skips training).",
    )
    args = parser.parse_args()

    if args.render_results is not None:
        csv_path = Path(args.render_results).expanduser().resolve()
        if not csv_path.exists():
            raise FileNotFoundError(f"Results CSV not found: {csv_path}")
        results_df = pd.read_csv(csv_path)
        plot_path = csv_path.with_suffix(".png")
        _plot_results(results_df, plot_path)
        print(f"Rendered plots to {plot_path}")
        return

    if args.n_iterations is not None:
        CONFIG["n_iterations"] = args.n_iterations
    CONFIG["show_progress"] = args.show_progress
    if args.height is not None:
        CONFIG["height"] = args.height
    if args.ndim is not None:
        CONFIG["ndim"] = args.ndim

    selected_envs = tuple(args.envs) if args.envs is not None else None

    _, csv_path, plot_path = run_benchmark(env_names=selected_envs)
    print(f"Saved benchmark table to {csv_path}")
    print(f"Saved benchmark plot to {plot_path}")


if __name__ == "__main__":
    main()
