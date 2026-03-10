# Ideas to Try (Deferred)

Stabilization and loss modification ideas deferred from the initial experiment sweep. These change the math of what we're evaluating, so they should be tried only after the baseline comparison is complete.

## Loss Modifications

- **Log-space normalization**: Replace `/ traj_len` with `/ log(1 + traj_len)`. Still monotonically rewards shorter trajectories but with a softer curve (3x ratio instead of 10x for length 2 vs 20).

- **Warmup blending**: Start training with the base (unmodified) loss, then linearly blend in the length normalization over the first N iterations. Lets the policy learn to find modes before the short-trajectory incentive kicks in.

- **Minimum denominator floor**: `traj_len.clamp(min=k)` for some k (e.g., ndim). Prevents 1/traj_len amplification for very short trajectories. Principled: a trajectory shorter than ndim steps can't have visited all dimensions.

- **Median reduction**: Use median instead of mean to reduce per-trajectory losses. More robust to outlier trajectories from the heavy tail of 1/traj_len.

## Reward-Gated Loss

- Gate the length normalization behind a check for whether the trajectory reached a high-reward state. Short trajectories that miss modes would use the base loss; short trajectories that hit modes get the bonus. Prevents penalizing exploration while rewarding efficiency.
