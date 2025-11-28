from typing import cast
import tqdm

import torch
from gfn.containers.trajectories import Trajectories
from gfn.env import Env
from gfn.estimators import DiscretePolicyEstimator, ScalarEstimator
from gfn.gflownet.base import loss_reduce
from gfn.gflownet.trajectory_balance import (
    LogPartitionVarianceGFlowNet,
    TrajectoryBalanceGFlowNet,
)
from gfn.modules import MLP, DiscreteUniform
from gfn.preprocessors import KHotPreprocessor
from gfn.utils.handlers import (
    is_callable_exception_handler,
    warn_about_recalculating_logprobs,
)

from .hypergrid import ModifiedHyperGrid


class ModifiedTrajectoryBalanceGFlowNet(TrajectoryBalanceGFlowNet):

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
        if trajectories.conditions is not None:
            with is_callable_exception_handler("logZ", self.logZ):
                assert isinstance(self.logZ, ScalarEstimator)
                logZ = self.logZ(trajectories.conditions)
        else:
            logZ = self.logZ

        logZ = cast(torch.Tensor, logZ)
        scores = (scores + logZ.squeeze()).pow(2)
        # TODO: normalize each batch element by the number of non-terminal, non-dummy,
        # non-initial states.
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
        scores = (scores - scores.mean()).pow(2)
        # TODO: normalize each batch element by the number of non-terminal, non-dummy,
        # non-initial states.
        loss = loss_reduce(scores, reduction)
        if torch.isnan(loss).any():
            raise ValueError("loss is NaN.")

        return loss


def main():
    # Envs: original, cosine, bitwise_xor, multiplicative_coprime
    env = ModifiedHyperGrid(
        ndim=2,
        height=16,
        reward_fn_str="original",
        reward_fn_kwargs=None,
        device="mps",
        calculate_partition=True,
        store_all_states=True,
        check_action_validity=True,
        validate_modes=True,
        mode_stats="exact",
        mode_stats_samples=20000,
    )
    preprocessor = KHotPreprocessor(height=env.height, ndim=env.ndim)

    # Build the GFlowNet.
    module_PF = MLP(
        input_dim=preprocessor.output_dim,
        output_dim=env.n_actions,
    )
    if not args.uniform_pb:
        module_PB = MLP(
            input_dim=preprocessor.output_dim,
            output_dim=env.n_actions - 1,
            trunk=module_PF.trunk,
        )
    else:
        module_PB = DiscreteUniform(output_dim=env.n_actions - 1)

    pf_estimator = DiscretePolicyEstimator(
        module_PF, env.n_actions, preprocessor=preprocessor, is_backward=False
    )
    pb_estimator = DiscretePolicyEstimator(
        module_PB, env.n_actions, preprocessor=preprocessor, is_backward=True
    )

    gflownet = TrajectoryBalanceGFlowNet(
        pf=pf_estimator, pb=pb_estimator, init_logZ=0.0
    )
    optimizer = torch.optim.Adam(gflownet.pf_pb_parameters(), lr=args.lr)
    if isinstance(gflownet, (TrajectoryBalanceGFlowNet, ModifiedTrajectoryBalanceGFlowNet)):
        optimizer.add_param_group(
            {"params": gflownet.logz_parameters(), "lr": args.lr_logz}
        )

    validation_info = {"l1_dist": float("inf")}
    visited_terminating_states = env.states_from_batch_shape((0,))
    discovered_modes = set()

    for it in (pbar := tqdm(range(args.n_iterations), dynamic_ncols=True)):
        trajectories = sampler.sample_trajectories(
            env,
            n=args.batch_size,
            save_logprobs=False,
            save_estimator_outputs=False,
            epsilon=args.epsilon,
        )
        visited_terminating_states.extend(
            cast(DiscreteStates, trajectories.terminating_states)
        )

        optimizer.zero_grad()
        loss = gflownet.loss_from_trajectories(
            env, trajectories, recalculate_all_logprobs=False
        )
        loss.backward()

        gflownet.assert_finite_gradients()
        torch.nn.utils.clip_grad_norm_(gflownet.parameters(), 1.0)
        optimizer.step()
        gflownet.assert_finite_parameters()

        if (it + 1) % args.validation_interval == 0:
            validation_info, _ = validate(
                env,
                gflownet,
                args.validation_samples,
                visited_terminating_states,
            )

            assert isinstance(visited_terminating_states, DiscreteStates)
            modes_found = env.modes_found(visited_terminating_states)
            discovered_modes.update(modes_found)

            str_info = f"Iter {it + 1}: "
            if "l1_dist" in validation_info:
                str_info += f"L1 distance={validation_info['l1_dist']:.8f} "
            str_info += f"modes discovered={len(discovered_modes)} / {env.n_modes} "
            str_info += f"n terminating states {len(visited_terminating_states)}"
            print(str_info)

        pbar.set_postfix(
            {"loss": loss.item(), "trajectories_sampled": (it + 1) * args.batch_size}
        )



if __name__ == "__main__":
    main()
