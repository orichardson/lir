from argparse import ArgumentParser
from pathlib import Path
import sys
from typing import cast

import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm

import torch
from gfn.containers.replay_buffer import ReplayBuffer
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
from gfn.utils.handlers import (
    is_callable_exception_handler,
    warn_about_recalculating_logprobs,
)

from gfn.env import DiscreteEnv
from gfn.states import DiscreteStates
from gfn.gflownet.base import GFlowNet


def validate(
    env: DiscreteEnv,
    gflownet: GFlowNet,
    n_validation_samples: int = 100_000,
) -> dict[str, float]:
    """Compute L1 distance between fresh policy samples and the true reward distribution.

    Draws ``n_validation_samples`` fresh trajectories from the current policy
    (rather than reusing training states) to get an unbiased estimate.

    Fixes two issues in upstream ``gfn.utils.training.validate``:
      1. Uses ``.sum()`` instead of ``.mean()`` for proper L1 distance.
      2. Samples fresh from the policy instead of reusing training states.
    """
    true_dist = env.true_dist
    if not isinstance(true_dist, torch.Tensor):
        return {}

    true_dist = true_dist.cpu()

    sampled = gflownet.sample_terminating_states(env, n_validation_samples)
    assert isinstance(sampled, DiscreteStates)

    final_states_dist = env.get_terminating_state_dist(sampled)
    if final_states_dist.numel() == 0:
        return {}

    l1_dist = (final_states_dist - true_dist).abs().sum().item()

    validation_info: dict[str, float] = {"l1_dist": l1_dist}

    # Report logZ difference if available.
    if hasattr(gflownet, "logZ") and isinstance(gflownet.logZ, torch.Tensor):
        try:
            true_logZ = env.log_partition
            print("true_logZ: {}".format(true_logZ))
            validation_info["logZ_diff"] = abs(gflownet.logZ.item() - true_logZ)
        except NotImplementedError:
            raise Exception("failed to calculate logz-diff - true={}".format(env.log_partition))

    return validation_info


ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from lir.gflownet.checkpoint import prepare_run_state, write_completed_checkpoint
from lir.gflownet.hypergrid import ModifiedHyperGrid

RESULTS_DIR = ROOT_DIR / "gflownet" / "gflownet" / "results"

# Static configuration for running the script without argparse.
CONFIG = {
    "device": "auto",
    "height": 16,
    "ndim": 4,
    "uniform_pb": False,
    "lr": 1e-3,
    "lr_logz": 1e-3,
    "n_iterations": 20000,
    "batch_size": 128,
    "epsilon": 0.0,
    "validation_interval": 200,
    "validation_samples": 100_000,
    "grad_clip_max_norm": 1.0,
    "n_seeds": 5,
    "show_progress": False,
    # New hyperparameters for experiment sweep.
    "optimizer": "adamw",
    "beta2": 0.999,
    "lr_schedule": "linear",  # "cosine", "linear", or "none"
    "lr_end_factor": 0.01,    # for linear schedule: final_lr = initial_lr * end_factor
    "loss_clamp": 0.0,        # 0 = disabled; positive = clamp per-trajectory loss
    "replay_capacity": 0,     # 0 = disabled; positive = replay buffer size
    "replay_batch_frac": 0.5, # fraction of batch drawn from replay buffer
}
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
ALGORITHM_REGISTRY: dict[str, type[torch.nn.Module]] = {}
DEVICE = torch.device("cpu")


def _resolve_device_name(preference: str) -> str:
    """Resolve preferred device string to an available concrete device."""
    pref = preference.lower()
    if pref == "auto":
        for candidate in ("cuda", "mps", "cpu"):
            if _device_is_available(candidate):
                return candidate
        return "cpu"
    if pref not in {"cpu", "cuda", "mps"}:
        raise ValueError(f"Unsupported device preference '{preference}'.")
    if not _device_is_available(pref):
        raise ValueError(f"Requested device '{pref}' is not available on this machine.")
    return pref


def _device_is_available(name: str) -> bool:
    if name == "cpu":
        return True
    if name == "cuda":
        return torch.cuda.is_available()
    if name == "mps":
        return torch.backends.mps.is_available()  # type: ignore[attr-defined]
    return False


def _set_runtime_device(preference: str) -> None:
    """Set CONFIG device and global torch device from preference."""
    resolved = _resolve_device_name(preference)
    CONFIG["device"] = resolved
    global DEVICE
    DEVICE = torch.device(resolved)


_set_runtime_device(CONFIG["device"])


def _maybe_to_device(module: torch.nn.Module | object) -> object:
    """Moves torch modules to the configured device when possible."""
    if hasattr(module, "to"):
        return module.to(DEVICE)
    return module


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
        validate_modes=False,
        mode_stats="none",
    )


def _build_estimators(
    env: ModifiedHyperGrid,
) -> tuple[DiscretePolicyEstimator, DiscretePolicyEstimator]:
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


def _build_optimizer(params, lr: float) -> torch.optim.Optimizer:
    """Build optimizer from CONFIG settings."""
    beta2 = float(CONFIG["beta2"])
    name = str(CONFIG["optimizer"]).lower()
    if name == "adamw":
        return torch.optim.AdamW(params, lr=lr, betas=(0.9, beta2))
    if name == "adam":
        return torch.optim.Adam(params, lr=lr, betas=(0.9, beta2))
    if name == "sgd":
        return torch.optim.SGD(params, lr=lr, momentum=0.9)
    raise ValueError(f"Unknown optimizer: {CONFIG['optimizer']}")


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

    optimizer = _build_optimizer(gflownet.pf_pb_parameters(), lr=CONFIG["lr"])
    logz_params = (
        list(gflownet.logz_parameters()) if hasattr(gflownet, "logz_parameters") else []
    )
    if logz_params:
        optimizer.add_param_group({"params": logz_params, "lr": CONFIG["lr_logz"]})

    scheduler = None
    schedule = str(CONFIG["lr_schedule"]).lower()
    if schedule == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=CONFIG["n_iterations"]
        )
    elif schedule == "linear":
        scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=1.0,
            end_factor=float(CONFIG["lr_end_factor"]),
            total_iters=CONFIG["n_iterations"],
        )

    replay_buffer = None
    replay_cap = int(CONFIG["replay_capacity"])
    if replay_cap > 0:
        replay_buffer = ReplayBuffer(env, capacity=replay_cap)

    loss_clamp = float(CONFIG["loss_clamp"])

    # Precompute the full set of mode states for precise coverage tracking.
    all_states = env.all_states
    mode_mask = env.mode_mask(all_states)
    all_mode_states: set[tuple[int, ...]] = {
        tuple(s.tolist()) for s in all_states.tensor[mode_mask]
    }
    total_mode_states = len(all_mode_states)

    visited_terminating_states = env.states_from_batch_shape((0,))
    discovered_mode_states: set[tuple[int, ...]] = set()
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
            save_logprobs=True if CONFIG["epsilon"] == 0 else False,
            save_estimator_outputs=True if CONFIG["epsilon"] > 0 else False,
            epsilon=CONFIG["epsilon"],
        )
        visited_terminating_states.extend(trajectories.terminating_states)

        if replay_buffer is not None:
            replay_buffer.add(trajectories)

        optimizer.zero_grad()

        # Build the training batch.  When a replay buffer is active and
        # sufficiently populated, keep half the fresh on-policy trajectories
        # and replace the other half with off-policy samples from the buffer.
        # All log-probs are recalculated so the two halves are on equal
        # footing under the current policy.
        use_replay = (
            replay_buffer is not None
            and len(replay_buffer) >= CONFIG["batch_size"]
        )
        if use_replay:
            n_total = len(trajectories)
            n_replay = max(1, int(n_total * CONFIG["replay_batch_frac"]))
            n_keep = n_total - n_replay
            # Slice fresh batch and extend with replay samples.
            train_trajs = trajectories[:n_keep]
            train_trajs.extend(replay_buffer.sample(n_replay))
            recalc = True  # must recalculate for mixed on/off-policy
        else:
            train_trajs = trajectories
            recalc = False

        loss = gflownet.loss_from_trajectories(
            env, train_trajs, recalculate_all_logprobs=recalc
        )

        if loss_clamp > 0:
            loss = loss.clamp(max=loss_clamp)

        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            gflownet.parameters(), CONFIG["grad_clip_max_norm"]
        )
        gflownet.assert_finite_gradients()
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
        gflownet.assert_finite_parameters()

        if (it + 1) % CONFIG["validation_interval"] == 0:
            validation_info = validate(
                env,
                gflownet,
                CONFIG["validation_samples"],
            )
            current_l1 = validation_info.get("l1_dist", current_l1)

        # Track unique mode states discovered so far.
        new_terms = trajectories.terminating_states.tensor
        for s in new_terms:
            key = tuple(s.tolist())
            if key in all_mode_states:
                discovered_mode_states.add(key)
        mode_coverage = len(discovered_mode_states) / total_mode_states if total_mode_states > 0 else 0.0
        n_terminating_states = len(visited_terminating_states)

        records.append(
            {
                "environment": env_name,
                "algorithm": algorithm_name,
                "seed": seed,
                "iteration": it + 1,
                "loss": loss.item(),
                "l1_dist": current_l1,
                "mode_coverage": mode_coverage,
                "n_mode_states_found": len(discovered_mode_states),
                "total_mode_states": total_mode_states,
                "n_terminating_states": n_terminating_states,
            }
        )

        if CONFIG["show_progress"]:
            iterator.set_postfix({"loss": loss.item(), "cov": f"{mode_coverage:.1%}"})

    # Always compute a final validation so the last record has a fresh L1.
    n_iters = CONFIG["n_iterations"]
    if n_iters % CONFIG["validation_interval"] != 0:
        validation_info = validate(
            env,
            gflownet,
            CONFIG["validation_samples"],
        )
        current_l1 = validation_info.get("l1_dist", current_l1)
        if records:
            records[-1]["l1_dist"] = current_l1

    return records


def _plot_results(results_df: pd.DataFrame, output_path: Path) -> None:
    """Create comparison plots (loss, L1 distance, modes) per environment."""
    metrics = [
        ("loss", "Loss"),
        ("l1_dist", "L1 distance"),
        ("mode_coverage", "Mode coverage"),
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

        # If conditioning values exist, we pass them to self.logZ
        # (should be a ScalarEstimator or equivalent).
        if trajectories.conditioning is not None:
            with is_callable_exception_handler("logZ", self.logZ):
                assert isinstance(self.logZ, ScalarEstimator)
                logZ = self.logZ(trajectories.conditioning)
        else:
            logZ = self.logZ

        logZ = cast(torch.Tensor, logZ)

        # Number of actual actions (transitions) per trajectory.
        traj_len = (~trajectories.actions.is_dummy).sum(0)

        # Per-step normalized TB loss.  The raw score
        #   s_i = log P_F(τ_i) − log P_B(τ_i|x_i) − log R(x_i)
        # is a sum over T_i action log-probs, so |s_i| ~ O(T_i).
        # We square first, then divide by T_i, giving per-step
        # mean squared error:
        #
        #   L = (1/n) Σ_i  (s_i + log Z)² / T_i
        #
        # Dividing *after* squaring avoids the 1/T² gradient
        # attenuation that dividing before squaring would cause
        # (which starves long trajectories of learning signal).
        # Fixed point is unchanged: s_i + log Z = 0  ⟹  loss = 0.
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

        # Number of actual actions (transitions) per trajectory.
        traj_len = (~trajectories.actions.is_dummy).sum(0)

        # Per-step normalized VarGrad loss.  The raw score
        #   s_i = log P_F(τ_i) − log P_B(τ_i|x_i) − log R(x_i)
        # is a sum over T_i action log-probs, so |s_i| ~ O(T_i).
        # We center by the batch mean (as in standard VarGrad), then
        # square and divide by T_i to get a per-step mean squared
        # deviation:
        #
        #   L = (1/n) Σ_i  (s_i − s̄)² / T_i     where s̄ = (1/n) Σ_j s_j
        #
        # Dividing *after* squaring avoids the 1/T² gradient
        # attenuation that dividing before squaring would cause.
        # Fixed point is unchanged: all s_i equal  ⟹  s_i − s̄ = 0  ⟹  loss = 0.
        scores = (scores - scores.mean()).pow(2) / traj_len

        loss = loss_reduce(scores, reduction)
        if torch.isnan(loss).any():
            raise ValueError("loss is NaN.")

        return loss


ALGORITHM_REGISTRY.update(
    {
        "LogPartitionVarianceGFlowNet": LogPartitionVarianceGFlowNet,
        "ModifiedLogPartitionVarianceGFlowNet": ModifiedLogPartitionVarianceGFlowNet,
        "TBGFlowNet": TBGFlowNet,
        "ModifiedTBGFlowNet": ModifiedTBGFlowNet,
    }
)


def run_benchmark(
    env_names: tuple[str, ...] | list[str] | None = None,
    algo_names: tuple[str, ...] | list[str] | None = None,
) -> tuple[pd.DataFrame, Path, Path]:
    """Execute the benchmark sweep, collecting all results in memory."""
    active_envs = tuple(env_names) if env_names is not None else ENVIRONMENTS
    active_algos = tuple(algo_names) if algo_names is not None else ALGORITHM_ORDER
    if not active_envs:
        raise ValueError("No environments specified for benchmarking.")
    if not active_algos:
        raise ValueError("No algorithms specified for benchmarking.")

    run_state = prepare_run_state(RESULTS_DIR, CONFIG, active_envs, active_algos)
    print(
        f"Starting run '{run_state.run_id}'. "
        f"Config saved to {run_state.config_path}"
    )

    all_records: list[dict] = []
    for env_name in active_envs:
        for algo_name in active_algos:
            for seed in range(7, 7 * CONFIG["n_seeds"] + 1, 7):
                records = _train_single_run(env_name, algo_name, seed)
                all_records.extend(records)

    results_df = pd.DataFrame(all_records)
    results_df.to_csv(run_state.csv_path, index=False)
    _plot_results(results_df, run_state.plot_path)
    write_completed_checkpoint(run_state)

    return results_df, run_state.csv_path, run_state.plot_path


def main():
    parser = ArgumentParser(description="Run HyperGrid GFlowNet benchmark.")

    # --- Environment & grid ---
    parser.add_argument(
        "--envs", nargs="+", choices=ENVIRONMENTS,
        help="Subset of environments to benchmark (default: all).",
    )
    parser.add_argument(
        "--algos", nargs="+", choices=ALGORITHM_ORDER,
        help="Subset of algorithms to benchmark (default: all).",
    )
    parser.add_argument("--height", type=int, default=None)
    parser.add_argument("--ndim", type=int, default=None)
    parser.add_argument(
        "--device", type=str, choices=("auto", "cpu", "cuda", "mps"),
        default="auto",
    )

    # --- Training ---
    parser.add_argument("--n_iterations", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--lr-logz", type=float, default=None,
                        help="Absolute logZ learning rate (overrides --lr-logz-multiplier).")
    parser.add_argument("--lr-logz-multiplier", type=float, default=None,
                        help="Set logZ lr = lr * multiplier (e.g. 10, 100).")
    parser.add_argument(
        "--optimizer", type=str, choices=("adamw", "adam", "sgd"),
        default=None,
    )
    parser.add_argument("--beta2", type=float, default=None)
    parser.add_argument("--lr-schedule", type=str,
                        choices=("cosine", "linear", "none"), default=None,
                        help="LR schedule type (default: linear).")
    parser.add_argument("--grad-clip", type=float, default=None,
                        help="Max norm for gradient clipping.")
    parser.add_argument("--loss-clamp", type=float, default=None,
                        help="Clamp per-trajectory loss (0=disabled).")

    # --- Replay buffer ---
    parser.add_argument("--replay-capacity", type=int, default=None,
                        help="Replay buffer capacity (0=disabled).")
    parser.add_argument("--replay-batch-frac", type=float, default=None,
                        help="Fraction of batch drawn from replay buffer.")

    # --- Seeds & output ---
    parser.add_argument("--n-seeds", type=int, default=None)
    parser.add_argument("--show-progress", action="store_true")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Override results directory.")
    parser.add_argument("--render_results", type=str, default=None,
                        help="Path to existing CSV to render plots (skips training).")

    args = parser.parse_args()

    # --- Render-only mode ---
    if args.render_results is not None:
        csv_path = Path(args.render_results).expanduser().resolve()
        if not csv_path.exists():
            raise FileNotFoundError(f"Results CSV not found: {csv_path}")
        results_df = pd.read_csv(csv_path)
        plot_path = csv_path.with_suffix(".png")
        _plot_results(results_df, plot_path)
        print(f"Rendered plots to {plot_path}")
        return

    # --- Apply CLI overrides to CONFIG ---
    _cli_overrides = {
        "n_iterations": args.n_iterations,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "lr_logz": args.lr_logz,
        "optimizer": args.optimizer,
        "beta2": args.beta2,
        "lr_schedule": args.lr_schedule,
        "grad_clip_max_norm": args.grad_clip,
        "loss_clamp": args.loss_clamp,
        "replay_capacity": args.replay_capacity,
        "replay_batch_frac": args.replay_batch_frac,
        "n_seeds": args.n_seeds,
        "height": args.height,
        "ndim": args.ndim,
    }
    for key, val in _cli_overrides.items():
        if val is not None:
            CONFIG[key] = val

    # --lr-logz-multiplier sets lr_logz relative to lr (overridden by --lr-logz).
    if args.lr_logz is None and args.lr_logz_multiplier is not None:
        CONFIG["lr_logz_multiplier"] = args.lr_logz_multiplier
        CONFIG["lr_logz"] = CONFIG["lr"] * args.lr_logz_multiplier
    CONFIG["show_progress"] = args.show_progress
    if args.device is not None:
        _set_runtime_device(args.device)
    if args.output_dir is not None:
        global RESULTS_DIR
        RESULTS_DIR = Path(args.output_dir).expanduser().resolve()

    selected_envs = tuple(args.envs) if args.envs is not None else None
    selected_algos = tuple(args.algos) if args.algos is not None else None

    _, csv_path, plot_path = run_benchmark(
        env_names=selected_envs,
        algo_names=selected_algos,
    )
    print(f"Saved benchmark table to {csv_path}")
    print(f"Saved benchmark plot to {plot_path}")


if __name__ == "__main__":
    main()
