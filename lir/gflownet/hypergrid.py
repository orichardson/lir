import itertools
import multiprocessing
import platform
import warnings
from abc import ABC, abstractmethod
from decimal import Decimal
from functools import reduce
from math import gcd, log, pi, sqrt
from time import time
from typing import List, Literal, Tuple

import torch
from gfn.actions import Actions
from gfn.env import DiscreteEnv
from gfn.gflownet.trajectory_balance import TBGFlowNet
from gfn.gym.hypergrid import (
    CosineReward,
    GridReward,
    HyperGrid,
    OriginalReward,
    lcm,
    lcm_multiple,
    smallest_multiplier_to_integers,
)
from gfn.states import DiscreteStates
from gfn.utils.common import ensure_same_device

if platform.system() == "Windows":
    multiprocessing.set_start_method("spawn", force=True)
else:
    multiprocessing.set_start_method("fork", force=True)


# Numerical tolerances used by quick mode-existence checks in this module.
#
# - EPS_REWARD_CMP: tolerance for comparing scalar rewards to thresholds. It
#   guards against small floating-point rounding errors when checking
#   inequalities like r >= thr.
# - EPS_INDEX_CMP: tolerance for floating-point-to-index boundary calculations,
#   used when turning fractional bands into integer indices.
EPS_REWARD_CMP = 1e-6
EPS_INDEX_CMP = 1e-9


class ModifiedHyperGrid(HyperGrid):
    """HyperGrid environment from the GFlowNets paper.

    The states are represented as 1-d tensors of length `ndim` with values in
    `{0, 1, ..., height - 1}`.

    Attributes:
        ndim: The dimension of the grid.
        height: The height of the grid.
        reward_fn: The reward function.
        calculate_partition: Whether to calculate the log partition function.
        store_all_states: Whether to store all states.
        validate_modes: Whether to check that at least one state reaches the
            mode threshold at init; raises if not.
        mode_stats: One of {"none", "approx", "exact"}. If not "none",
            computes (exact or approximate) `n_modes` and `n_mode_states`.
            "exact" requires `store_all_states=True` and enumerates all states.
        mode_stats_samples: Number of random samples when `mode_stats="approx"`.
    """

    def __init__(
        self,
        ndim: int = 2,
        # Smallest height that satisfies `validate_modes=True` for the original reward.
        height: int = 8,
        reward_fn_str: str = "original",
        reward_fn_kwargs: dict | None = None,
        device: Literal["cpu", "cuda"] | torch.device = "cpu",
        calculate_partition: bool = False,
        store_all_states: bool = False,
        check_action_validity: bool = True,
        validate_modes: bool = True,
        mode_stats: Literal["none", "approx", "exact"] = "none",
        mode_stats_samples: int = 20000,
    ):
        """Initializes the HyperGrid environment.

        Args:
            ndim: The dimension of the grid.
            height: The height of the grid. The default value is the smallest height
                that satisfies `validate_modes=True` for the original reward.
            reward_fn_str: The reward function string to use.
            reward_fn_kwargs: The keyword arguments for the reward function.
            device: The device to use.
            calculate_partition: Whether to calculate the log partition function.
                store_all_states: Whether to store all states. If True, the true
                distribution can be accessed via the `true_dist` property. Note that
                this is not always possible for some reward functions.
            check_action_validity: Whether to check the action validity.
            validate_modes: If True, verifies at initialization that at least one
                state achieves the reward-defined mode threshold; raises
                `ValueError` when no such state is found.
            mode_stats: Level of mode statistics to compute: "none" (disabled),
                "approx" (uniform sampling), or "exact" (full enumeration).
                "exact" requires `store_all_states=True` and may be expensive for
                large `height` or `ndim`.
            mode_stats_samples: Number of random samples used when
                `mode_stats="approx"`.
        """
        reward_functions = {
            "original": OriginalReward,
            "cosine": CosineReward,
            # New compositional environments (see classes below)
            "bitwise_xor": BitwiseXORReward,
            "multiplicative_coprime": MultiplicativeCoprimeReward,
        }

        self.ndim = ndim
        self.height = height

        # Default reward function kwargs.
        if reward_fn_kwargs is None:
            reward_fn_kwargs = {"R0": 0.1, "R1": 0.5, "R2": 2.0}
        self.reward_fn_kwargs = reward_fn_kwargs
        assert (
            reward_fn_str in reward_functions
        ), f"Invalid reward function string: {reward_fn_str} not in {reward_functions.keys()}"
        self.reward_fn = reward_functions[reward_fn_str](
            height, ndim, **reward_fn_kwargs
        )

        self._all_states_tensor = None  # Populated optionally in init.
        self._log_partition = None  # Populated optionally in init.
        self._true_dist = None  # Populated at first request.
        self.store_all_states = store_all_states

        # If we store the all states, the partition function is calculated automatically.
        self.calculate_partition = calculate_partition or store_all_states

        # Pre-computes these values when printing.
        if self.store_all_states:
            self._store_all_states_tensor()
            assert self._all_states_tensor is not None
            print(f"+ Environment has {len(self._all_states_tensor)} states")

        if self.calculate_partition:
            self._calculate_log_partition()
            assert self._log_partition is not None
            print(f"+ Environment log partition is {self._log_partition}")

        if isinstance(device, str):
            device = torch.device(device)

        s0 = torch.zeros(ndim, dtype=torch.long, device=device)
        sf = torch.full((ndim,), fill_value=-1, device=device)
        n_actions = ndim + 1

        state_shape = (self.ndim,)

        # HyperGrid -> DiscreteEnv
        DiscreteEnv.__init__(
            self,
            n_actions=n_actions,
            s0=s0,
            state_shape=state_shape,
            sf=sf,
            check_action_validity=check_action_validity,
        )
        self.States: type[DiscreteStates] = self.States  # for type checking

        # Optionally validate that modes exist under the configured reward.
        if validate_modes:
            ok, msg = self._modes_exist_quick_check_info()
            if not ok:
                raise ValueError(msg)

        # Optional mode statistics (expensive when exact)
        self._n_mode_states_exact: int | None = None
        self._n_modes_via_ids_exact: int | None = None
        self._n_mode_states_estimate: float | None = None
        self._n_modes_via_ids_estimate: float | None = None
        self._mode_stats_kind: str = "none"

        if mode_stats != "none":
            try:
                if mode_stats == "exact":
                    if not self.store_all_states:
                        raise ValueError(
                            "Exact mode_stats requires store_all_states=True to enumerate states."
                        )
                    all_states = self.all_states
                    if all_states is None:
                        raise ValueError(
                            "Failed to access all_states for exact mode_stats."
                        )
                    mask = self.mode_mask(all_states)
                    self._n_mode_states_exact = int(mask.sum().item())
                    # Use mode_ids grouping as a simple label over halves of the space
                    ids = self.mode_ids(all_states)
                    ids = ids[mask]
                    ids = ids[ids >= 0]
                    self._n_modes_via_ids_exact = int(torch.unique(ids).numel())
                    self._mode_stats_kind = "exact"
                else:
                    # Approximate via uniform sampling.
                    with torch.no_grad():
                        B = int(max(1, mode_stats_samples))
                        xs = self.make_random_states((B,)).tensor
                        mask = self.mode_mask(self.States(xs))
                        frac = float(mask.float().mean().item())
                        self._n_mode_states_estimate = frac * float(self.n_states)
                        ids = self.mode_ids(self.States(xs))
                        ids = ids[mask]
                        ids = ids[ids >= 0]
                        self._n_modes_via_ids_estimate = float(
                            torch.unique(ids).numel()
                        )
                        self._mode_stats_kind = "approx"
            except Exception:
                warnings.warn("+ Warning: Failed to compute mode_stats (skipping).")

    # Mode utilities.
    def _mode_reward_threshold(self) -> float:
        """Returns the reward threshold used to define a mode.

        By default, a state is considered in a mode if its reward is at least
        the schema-defined threshold derived from the configured reward.
        """
        # We branch on the concrete reward to derive a principled threshold.

        # Original reward: ring band adds R2 on top of base R0 and outer ring R1.
        if isinstance(self.reward_fn, OriginalReward):
            for key in ("R0", "R1", "R2"):
                if key not in self.reward_fn_kwargs:
                    raise ValueError(
                        f"Missing '{key}' in reward_fn_kwargs for Original reward; "
                        "please provide R0, R1, and R2."
                    )
            r0 = float(self.reward_fn_kwargs["R0"])  # type: ignore[index]
            r1 = float(self.reward_fn_kwargs["R1"])  # type: ignore[index]
            r2 = float(self.reward_fn_kwargs["R2"])  # type: ignore[index]
            # Modes are the thin ring where both outer ring and band conditions hold.
            return r0 + r1 + r2

        # Cosine reward: peak at center with oscillatory local maxima along each axis.
        # Treat "modes" as states whose per-dimension factor is near its theoretical
        # maximum f_max = 2 / sqrt(2*pi). We allow a tunable closeness factor
        # `mode_gamma` (default 0.8). The product structure implies a threshold of
        # (gamma*f_max)^ndim.
        if isinstance(self.reward_fn, CosineReward):
            r0 = float(self.reward_fn_kwargs.get("R0", 0.1))
            r1 = float(self.reward_fn_kwargs.get("R1", 0.5))
            gamma = float(self.reward_fn_kwargs.get("mode_gamma", 0.8))
            # Use discrete per-dimension maximum on sufficiently fine grids;
            # fall back to theoretical peak on very coarse grids to preserve
            # the intended strictness of high gamma settings.
            per_dim_peak = 2.0 / sqrt(2 * pi)
            Hm1 = max(1, self.height - 1)
            idx = torch.arange(0, self.height, dtype=torch.get_default_dtype())
            ax = (idx / Hm1 - 0.5).abs()
            pdf = (1.0 / sqrt(2 * pi)) * torch.exp(-0.5 * (5 * ax) ** 2)
            per_dim_discrete = float(((torch.cos(50 * ax) + 1.0) * pdf).max())
            per_dim_base = per_dim_discrete if self.height > 4 else per_dim_peak
            return r0 + (gamma * per_dim_base) ** self.ndim * r1

        # Other reward schemas are not supported for mode counting via threshold.
        # For compositional rewards with tiered structure, mark mode as achieving
        # the highest tier. We assume the classes expose `R0` and `tier_weights`.
        if isinstance(
            self.reward_fn,
            (
                BitwiseXORReward,
                MultiplicativeCoprimeReward,
            ),
        ):
            r0 = float(getattr(self.reward_fn, "R0", 0.0))
            tw = getattr(self.reward_fn, "tier_weights", [])
            if not isinstance(tw, (list, tuple)) or len(tw) == 0:
                raise ValueError(
                    "Tiered reward missing `tier_weights`; cannot derive mode threshold."
                )
            # For others, cumulative structure applies
            return r0 + float(sum(tw))

        raise NotImplementedError(
            "Mode threshold is only defined for known reward schemas."
        )

    @property
    def n_mode_states(self) -> int | float | None:
        """Number of states inside a mode (exact, approx, or None).

        - If mode_stats="exact", returns an exact integer count.
        - If mode_stats="approx", returns a floating-point estimate.
        - Otherwise, returns None.
        """
        if self._mode_stats_kind == "exact" and self._n_mode_states_exact is not None:
            return int(self._n_mode_states_exact)
        if (
            self._mode_stats_kind == "approx"
            and self._n_mode_states_estimate is not None
        ):
            return float(self._n_mode_states_estimate)
        return None

    @property
    def n_modes(self) -> int:
        """Returns the total number of distinct modes for this environment.

        For compositional or non-uniform rewards, this property reflects the
        cardinality of mode-IDs if available; otherwise, it returns a schema
        heuristic (default: 2**ndim) consistent with the original ring-based
        hypergrid definition.
        """
        # Prefer an exact, on-demand computation when all states are available.
        # This makes the property robust to preset changes and matches the
        # test enumeration that uses `mode_ids` over `mode_mask`.
        try:
            all_states = self.all_states
            if all_states is not None:
                mask = self.mode_mask(all_states)
                ids = self.mode_ids(all_states)
                ids = ids[mask]
                ids = ids[ids >= 0]
                return int(torch.unique(ids).numel())
        except Exception:
            pass
        if self._mode_stats_kind == "exact" and self._n_modes_via_ids_exact is not None:
            return int(self._n_modes_via_ids_exact)
        if (
            self._mode_stats_kind == "approx"
            and self._n_modes_via_ids_estimate is not None
        ):
            return int(self._n_modes_via_ids_estimate)

        return 2**self.ndim

    # Mode existence validation.
    def _modes_exist_quick_check(self) -> bool:
        """Lightweight check that a mode-level state exists.

        In simple terms, this answers: "Is there at least one state whose reward
        reaches the mode threshold?" without enumerating all states. It proceeds
        in three stages:
        1) If the grid is small (or pre-enumerated), it computes rewards exactly
           and checks against the threshold.
        2) Otherwise, it dispatches to reward-specific constructive tests that
           are sufficient to guarantee at least one state reaches the threshold.
        3) As a last resort, it samples a small batch of random states.
        """
        thr = self._mode_reward_threshold()

        # If the grid is small enough, prefer an exact check to avoid fragile heuristics.
        # Also prefer exact when all states are already stored.
        try:
            if self.store_all_states and self.all_states is not None:
                rewards = self.reward(self.all_states)
                # Compare with a small tolerance to avoid missing near-boundary cases.
                return bool((rewards >= thr - EPS_REWARD_CMP).any().item())
            # Cheap exact threshold (up to ~200k states)
            if self.n_states <= 200_000:
                axes = [
                    torch.arange(self.height, dtype=torch.long)
                    for _ in range(self.ndim)
                ]
                grid = torch.cartesian_prod(*axes)
                rewards = self.reward_fn(grid)
                return bool((rewards >= thr - EPS_REWARD_CMP).any().item())
        except Exception:
            # Fall back to heuristic paths below
            pass
        if isinstance(self.reward_fn, OriginalReward):
            return self._exists_original_or_deceptive(thr)
        if isinstance(self.reward_fn, CosineReward):
            return self._exists_cosine(thr)
        if isinstance(self.reward_fn, BitwiseXORReward):
            return self._exists_bitwise_xor(thr)
        if isinstance(self.reward_fn, MultiplicativeCoprimeReward):
            return self._exists_multiplicative_coprime(thr)
        return self._exists_fallback_random(thr)

    def _modes_exist_quick_check_info(self) -> tuple[bool, str]:
        """User-friendly wrapper for ``_modes_exist_quick_check``.

        Returns ``(ok, message)`` where:
        - ``ok`` is True if a mode-level state is found by the quick check.
        - ``message`` is "OK" on success or a short explanation of why the
          quick check failed and, for TemplateMinkowski, guidance to adjust
          parameters when r_bands are unreachable given the L1 budget
          cap_sum = (H-1)*|dims_subset|.
        """
        try:
            ok = self._modes_exist_quick_check()
            if ok:
                return True, "OK"
        except Exception:
            pass

        return (
            False,
            "No states satisfy the mode threshold for the current reward and parameters.",
        )

    def _exists_original_or_deceptive(self, thr: float) -> bool:
        """Constructive check for ``OriginalReward``.

        Intuition:
        - These rewards form rings/bands around the center when each coordinate
          is normalized to [0,1]. The mode lies on a thin band at specific
          normalized distances from the center.
        - We translate those fractional band boundaries into integer indices via
          small inside/outside nudges (using ``EPS_INDEX_CMP``) and test one
          candidate index from any non-empty feasible interval.
        - If the reward at that candidate exceeds the threshold (with
          ``EPS_REWARD_CMP`` tolerance), we return True.
        """
        Hm1 = self.height - 1
        if Hm1 <= 0:
            return False
        lows = []
        highs = []
        # Convert fractional bounds to integer index ranges with a small tolerance.
        lows.append(int((0.1 + EPS_INDEX_CMP) * Hm1) + 1)
        highs.append(int((0.2 - EPS_INDEX_CMP) * Hm1))
        lows.append(int((0.8 + EPS_INDEX_CMP) * Hm1) + 1)
        highs.append(int((0.9 - EPS_INDEX_CMP) * Hm1))
        candidate_idxs: list[int] = []
        for lo, hi in zip(lows, highs):
            if lo <= hi:
                candidate_idxs.append(lo)
        if not candidate_idxs:
            return False
        i = candidate_idxs[0]
        x = torch.full((self.ndim,), i, dtype=torch.long)
        r = float(self.reward_fn(x.unsqueeze(0))[0])
        return r >= thr - EPS_REWARD_CMP

    def _exists_cosine(self, thr: float) -> bool:
        """Analytic upper-bound check for ``CosineReward``.

        Idea:
        - The per-dimension factor is ``(cos(50·ax) + 1) · N(0,1)(5·ax)`` with
          ax in [0,0.5]. We estimate its maximum over the discrete grid by
          evaluating all candidate ax and taking the maximum value ``m``.
        - The full reward upper bound is ``R0 + m^D * R1``. If this is at least
          the mode target and the given threshold, a mode-level state must exist.
        - We also compute a theoretical per-dimension peak (at ax≈0) to form a
          slightly conservative target scaled by ``mode_gamma``.
        """
        R0 = float(self.reward_fn.kwargs.get("R0", 0.1))
        R1 = float(self.reward_fn.kwargs.get("R1", 0.5))
        gamma = float(self.reward_fn.kwargs.get("mode_gamma", 0.8))
        Hm1 = max(1, self.height - 1)
        idx = torch.arange(0, self.height, dtype=torch.get_default_dtype())
        ax = (idx / Hm1 - 0.5).abs()
        pdf = (1.0 / sqrt(2 * pi)) * torch.exp(-0.5 * (5 * ax) ** 2)
        per_dim = (torch.cos(50 * ax) + 1.0) * pdf
        m = float(per_dim.max())
        # Compute a gamma-scaled target using the theoretical per-dimension peak.
        per_dim_peak = 2.0 / sqrt(2 * pi)
        target = R0 + (gamma * per_dim_peak) ** self.ndim * R1
        rmax = R0 + (m**self.ndim) * R1
        return rmax >= target - EPS_REWARD_CMP and rmax >= thr - EPS_REWARD_CMP

    def _exists_bitwise_xor(self, thr: float) -> bool:
        """Feasibility and constructive check for ``BitwiseXORReward``.

        Steps:
        - For each tier, verify the GF(2) parity system has at least one
          solution using Gaussian elimination modulo 2. If any tier is
          infeasible, no mode exists.
        - The all-zero configuration satisfies even-parity constraints, so if
          tiers are feasible we evaluate that point against the threshold with
          tolerance.
        """
        if self.reward_fn.parity_checks is not None:
            for t in range(len(self.reward_fn.tier_weights)):
                cfg = self.reward_fn.parity_checks[t]
                if cfg is None:
                    continue
                A = cfg.get("A", None)
                c = cfg.get("c", None)
                if A is None or c is None:
                    continue
                if not self._solve_gf2_has_solution(A, c):
                    return False

        x = torch.zeros(self.ndim, dtype=torch.long)
        r = float(self.reward_fn(x.unsqueeze(0))[0])
        return r >= thr - EPS_REWARD_CMP

    def _exists_multiplicative_coprime(self, thr: float) -> bool:
        """Number-theoretic constructive check for ``MultiplicativeCoprimeReward``.

        Outline:
        - If a target LCM is specified for the last tier, factor it over the
          allowed primes and ensure exponents do not exceed caps and are
          representable as grid coordinates (< H).
        - Place prime powers on designated active dimensions and ensure required
          coprime relations hold between specified pairs.
        - If the constructed state fits within the grid, evaluate it and compare
          to the threshold with tolerance.
        """
        primes: list[int] = [int(p) for p in self.reward_fn.primes]
        caps: list[int] = [int(c) for c in self.reward_fn.exponent_caps]
        active = list(self.reward_fn.active_dims)
        copairs = self.reward_fn.coprime_pairs or []
        cap = int(caps[-1])
        target_lcms = self.reward_fn.target_lcms
        target = None if target_lcms is None else target_lcms[-1]
        x = torch.ones(self.ndim, dtype=torch.long)
        if target is not None:
            target = int(target)
            need: list[tuple[int, int]] = []
            tmp = target
            for p in primes:
                e = 0
                while tmp % p == 0:
                    tmp //= p
                    e += 1
                if e > 0:
                    if e > cap or (p**e) > (self.height - 1):
                        return False
                    need.append((p, e))
            if tmp != 1 or len(need) > len(active):
                return False
            for (p, e), dim in zip(need, active):
                x[dim] = p**e
            for i, j in copairs:
                if torch.gcd(x[active[i]], x[active[j]]).item() != 1:
                    return False
        if int(x.max()) >= self.height:
            return False
        r = float(self.reward_fn(x.unsqueeze(0))[0])
        return r >= thr - EPS_REWARD_CMP

    def _exists_fallback_random(self, thr: float) -> bool:
        """Random sampling fallback.

        Draw a modest batch of random states on CPU and accept if any exceed the
        threshold with a small tolerance. This is a last resort to avoid
        expensive enumeration for large grids.
        """
        with torch.no_grad():
            device = torch.device("cpu")
            B = min(2048, max(128, 8 * self.ndim))
            xs = torch.randint(0, self.height, (B, self.ndim), device=device)
            rr = self.reward_fn(xs)
            return bool((rr >= thr - EPS_REWARD_CMP).any().item())

    @staticmethod
    def _solve_gf2_has_solution(A: torch.Tensor, c: torch.Tensor) -> bool:
        """Return True if A x = c over GF(2) has at least one solution.

        Performs Gaussian elimination modulo 2 without constructing a specific solution.
        """
        if A.numel() == 0:
            # No constraints
            return True
        A = A.clone().detach().to(dtype=torch.uint8, device=torch.device("cpu")) & 1
        c = c.clone().detach().to(dtype=torch.uint8, device=torch.device("cpu")) & 1
        k, m = A.shape
        row = 0
        for col in range(m):
            # Find pivot
            piv = None
            for r in range(row, k):
                if A[r, col]:
                    piv = r
                    break
            if piv is None:
                continue
            # Swap
            if piv != row:
                A[[row, piv]] = A[[piv, row]]
                c[[row, piv]] = c[[piv, row]]
            # Eliminate below
            for r in range(row + 1, k):
                if A[r, col]:
                    A[r, :] ^= A[row, :]
                    c[r] ^= c[row]
            row += 1
            if row == k:
                break
        # Check for inconsistency: 0 = 1 rows
        for r in range(k):
            if not A[r, :].any() and c[r]:
                return False
        return True


class BitwiseXORReward(GridReward):
    """Tiered, compositional reward based on bitwise XOR/parity constraints.

    This class implements the "Bitwise/XOR fractal" environment family: where tiers
    progressively constrain bit-planes across a subset of dimensions via linear parity
    checks over GF(2). It supports easy sharding by high-bit prefixes, and difficulty
    control by adjusting which bit-planes and how many dimensions are constrained per tier.

    GF(2) is the finite field with two elements {0, 1}, where addition and
    multiplication are performed modulo 2. In this context, vector addition is
    equivalent to bitwise XOR, and matrix–vector products (A @ b) are evaluated
    entrywise modulo 2.

    Reward form:
        R(s) = R0 + Σ_t tier_weights[t] · 1[ state satisfies all constraints up to tier t ]

    Key kwargs (with reasonable defaults):
        - R0: float, base reward (default 0.0)
        - tier_weights: list[float], strictly increasing weights for each tier
        - dims_constrained: Optional[list[int]] subset of dims to constrain
          (default: all dims)
        - bits_per_tier: list[tuple[int,int]]; for each tier t, inclusive bit range
          (low_bit, high_bit). Example: [(0,5), (0,7), (0,9)].
        - parity_checks: Optional[list[dict]]; per tier, optional parity system:
            Each entry may contain:
              { "A": IntTensor[num_checks, m], "c": IntTensor[num_checks] }
            where m = len(dims_constrained). Constraints apply identically to every
            bit-plane specified for that tier: A @ b(mod2) == c, where b are the
            bit values across constrained dimensions at the tested bit-plane.
            If omitted for a tier, a single even-parity check across all constrained
            dims is used by default: sum(b) mod 2 == 0.

    Difficulty presets align with step ranges by controlling the highest bit used
    and the number of constrained dimensions. Typical distance from origin for
    valid modes scales roughly like (constrained_dims · 2^{highest_bit}).
    """

    def __init__(self, height: int, ndim: int, **kwargs):
        super().__init__(height, ndim, **kwargs)
        self.R0: float = float(kwargs.get("R0", 0.0))
        tier_weights = kwargs.get("tier_weights", [1.0, 10.0, 100.0])
        assert isinstance(tier_weights, (list, tuple)) and len(tier_weights) > 0
        self.tier_weights: list[float] = [float(w) for w in tier_weights]

        dims_constrained = kwargs.get("dims_constrained", None)
        if dims_constrained is None:
            dims_constrained = list(range(ndim))
        assert len(dims_constrained) > 0
        self.dims_constrained: list[int] = list(map(int, dims_constrained))

        bits_per_tier = kwargs.get("bits_per_tier", None)
        if bits_per_tier is None:
            # Default: widen the bit window gradually
            bits_per_tier = [(0, 5), (0, 7), (0, 9)]
        assert len(bits_per_tier) == len(self.tier_weights)
        self.bits_per_tier: list[tuple[int, int]] = [
            (int(lo), int(hi)) for (lo, hi) in bits_per_tier
        ]

        self.parity_checks = kwargs.get("parity_checks", None)
        if self.parity_checks is not None:
            assert len(self.parity_checks) == len(self.tier_weights)

    def _even_parity_mask(self, bits: torch.Tensor) -> torch.Tensor:
        """bits: (..., m) int/bool -> returns (...,) bool for even parity."""
        if bits.dtype != torch.long:
            bits = bits.long()
        return (bits.sum(dim=-1) & 1) == 0

    def _apply_parity_checks(
        self, bits_plane: torch.Tensor, tier_idx: int
    ) -> torch.Tensor:
        """Apply GF(2) linear parity checks at a single bit-plane.

        bits_plane: (..., m) with m=len(dims_constrained), integer in {0,1}.
        Returns mask (...,) bool.
        """
        if self.parity_checks is None or self.parity_checks[tier_idx] is None:
            return self._even_parity_mask(bits_plane)

        cfg = self.parity_checks[tier_idx]
        A: torch.Tensor | None = cfg.get("A")
        c: torch.Tensor | None = cfg.get("c")
        if A is None or c is None:
            return self._even_parity_mask(bits_plane)

        # Ensure device/dtype
        A = A.to(bits_plane.device).long()
        c = c.to(bits_plane.device).long()
        # Compute (A @ bits) mod 2 for each batch element
        # reshape bits to (..., m, 1) for bmm if needed, but here we can do matmul
        # by flattening the batch.
        flat = bits_plane.reshape(-1, bits_plane.shape[-1]).long()
        prod = (flat @ A.t()) & 1  # shape (B, num_checks)
        target = c.unsqueeze(0).expand_as(prod)
        ok = (prod == target).all(dim=-1)
        return ok.reshape(bits_plane.shape[:-1])

    def __call__(self, states_tensor: torch.Tensor) -> torch.Tensor:
        R = torch.zeros(
            states_tensor.shape[:-1],
            device=states_tensor.device,
            dtype=torch.get_default_dtype(),
        )
        if self.R0 != 0.0:
            R += self.R0

        # Select constrained dims
        x = states_tensor.index_select(
            dim=-1,
            index=torch.tensor(self.dims_constrained, device=states_tensor.device),
        )

        # Progressive, compositional tiers: a state gets tier t reward only if it
        # satisfies all constraints up to t.
        valid_up_to_t = torch.ones(x.shape[:-1], device=x.device, dtype=torch.bool)
        for t, w in enumerate(self.tier_weights):
            lo_b, hi_b = self.bits_per_tier[t]
            tier_mask = torch.ones_like(valid_up_to_t)
            for b in range(lo_b, hi_b + 1):
                bits = ((x >> b) & 1).long()
                plane_ok = self._apply_parity_checks(bits, t)
                tier_mask = tier_mask & plane_ok
            valid_up_to_t = valid_up_to_t & tier_mask
            R = R + (valid_up_to_t.to(R.dtype) * float(w))

        return R


class MultiplicativeCoprimeReward(GridReward):
    """Tiered reward based on prime-support and coprimality/lcm composition.

    Each tier enforces that per-dimension values use only a small shared prime set
    with bounded exponents, plus optional cross-dimension constraints (pairwise
    coprime pairs and/or target lcm). Higher tiers tighten exponent caps or add
    additional global targets. This encourages information sharing to learn the
    latent prime/exponent structure.

    Reward form:
        R(s) = R0 + Σ_t tier_weights[t] · 1[ constraints_0..t all satisfied ]

    Key kwargs:
        - R0: float, base reward (default 0.0)
        - tier_weights: list[float]
        - primes: list[int], e.g., [2,3,5,7,11]
        - exponent_caps: list[int], same length as tier_weights. Cap for every prime
          at tier t (uniform cap across primes for simplicity).
        - active_dims: Optional[list[int]]; constraints only apply to these dims
          (default: all dims). Other dims are ignored in constraints.
        - coprime_pairs: Optional[list[tuple[int,int]]]; indices relative to active_dims.
        - target_lcms: Optional[list[int | None]]; per-tier target lcm across active dims.

    Notes:
    - Values 0 are treated as invalid for prime-support constraints (cannot factorize);
      value 1 is valid with all-zero exponents.
    - Implementation removes primes up to the current tier cap and checks residue == 1.
      Exponent counts are accumulated to evaluate LCM targets.
    """

    def __init__(self, height: int, ndim: int, **kwargs):
        super().__init__(height, ndim, **kwargs)
        self.R0: float = float(kwargs.get("R0", 0.0))
        tier_weights = kwargs.get("tier_weights", [1.0, 10.0, 100.0])
        assert isinstance(tier_weights, (list, tuple)) and len(tier_weights) > 0
        self.tier_weights: list[float] = [float(w) for w in tier_weights]

        primes = kwargs.get("primes", [2, 3, 5])
        assert isinstance(primes, (list, tuple)) and len(primes) > 0
        self.primes: list[int] = [int(p) for p in primes]

        exponent_caps = kwargs.get("exponent_caps", [2] * len(self.tier_weights))
        assert len(exponent_caps) == len(self.tier_weights)
        self.exponent_caps: list[int] = [int(c) for c in exponent_caps]

        active_dims = kwargs.get("active_dims", None)
        if active_dims is None:
            active_dims = list(range(ndim))
        self.active_dims: list[int] = list(map(int, active_dims))

        self.coprime_pairs = kwargs.get("coprime_pairs", None)
        self.target_lcms = kwargs.get("target_lcms", [None] * len(self.tier_weights))
        assert isinstance(self.target_lcms, (list, tuple)) and len(
            self.target_lcms
        ) == len(self.tier_weights)

    def _factor_exponents_up_to_cap(self, v: torch.Tensor, cap: int):
        """Return (residue, exponents) after dividing by allowed primes up to `cap`.

        v: (...,) LongTensor, non-negative.
        residue: (...,) after stripping primes up to `cap` times each.
        exps: tensor [num_primes, ...] of exponent counts per prime (capped by `cap`).
        """
        residue = v.clone()
        exps = []
        for p in self.primes:
            p = int(p)
            count = torch.zeros_like(residue)
            # Repeatedly divide by p but not more than cap times
            for _ in range(cap):
                divisible = (residue % p) == 0
                if not torch.any(divisible):
                    break
                residue = torch.where(divisible, residue // p, residue)
                count = count + divisible.long()
            exps.append(count)
        exps = torch.stack(exps, dim=0)  # [num_primes, ...]
        return residue, exps

    def _pairwise_coprime_ok(self, v: torch.Tensor) -> torch.Tensor:
        """Check pairwise coprime on configured pairs using prime divisibility.

        v: (..., m) with m = len(active_dims).
        Returns (...,) bool.
        """
        if not self.coprime_pairs:
            return torch.ones(v.shape[:-1], dtype=torch.bool, device=v.device)
        ok = torch.ones(v.shape[:-1], dtype=torch.bool, device=v.device)
        for p in self.primes:
            div = (v % int(p)) == 0  # (..., m)
            for i, j in self.coprime_pairs:
                m = div.shape[-1]
                if i >= m or j >= m:
                    continue
                both = div[..., i] & div[..., j]
                ok = ok & (~both)
        return ok

    def _lcm_ok(self, exps: torch.Tensor, target_lcm: int) -> torch.Tensor:
        """Check whether max exponents across dims match target LCM's exponents.

        exps: [num_primes, ..., m]
        target_lcm: int
        Returns (...,) bool.
        """
        # Factor target_lcm fully over allowed primes; reject if leftover > 1
        remaining = int(target_lcm)
        target_counts: list[int] = []
        for p in self.primes:
            p = int(p)
            c = 0
            while remaining % p == 0:
                remaining //= p
                c += 1
            target_counts.append(c)
        if remaining != 1:
            # Target contains primes outside allowed set -> impossible
            shape = exps.shape[1:-1]  # broadcast shape (...)
            return torch.zeros(shape, dtype=torch.bool, device=exps.device)
        target = torch.tensor(target_counts, dtype=torch.long, device=exps.device)

        max_exp = exps.max(dim=-1).values  # [num_primes, ...]
        # Broadcast compare to target per prime
        while target.dim() < max_exp.dim():
            target = target.unsqueeze(-1)
        return (max_exp == target).all(dim=0)

    def __call__(self, states_tensor: torch.Tensor) -> torch.Tensor:
        R = torch.zeros(
            states_tensor.shape[:-1],
            device=states_tensor.device,
            dtype=torch.get_default_dtype(),
        )
        if self.R0 != 0.0:
            R += self.R0

        x_full = states_tensor
        x = x_full.index_select(
            dim=-1, index=torch.tensor(self.active_dims, device=states_tensor.device)
        )
        # Disallow zeros at the outset for constraints (they cannot have finite prime support)
        base_valid = (x != 0).all(dim=-1)

        valid_up_to_t = base_valid
        for t, w in enumerate(self.tier_weights):
            cap = self.exponent_caps[t]
            residue, exps = self._factor_exponents_up_to_cap(x.reshape(-1), cap)
            residue = residue.reshape(x.shape)
            exps = exps.reshape((len(self.primes),) + x.shape)

            # Prime-support with bounded exponents: residue must be 1 or original value 1
            support_ok = (residue == 1) | (x == 1)
            support_ok = support_ok.all(dim=-1)

            pairs_ok = self._pairwise_coprime_ok(x)
            lcm_ok = torch.ones_like(pairs_ok)
            if self.target_lcms[t] is not None:
                lcm_ok = self._lcm_ok(exps, int(self.target_lcms[t]))

            tier_ok = support_ok & pairs_ok & lcm_ok
            valid_up_to_t = valid_up_to_t & tier_ok
            R = R + (valid_up_to_t.to(R.dtype) * float(w))

        return R


# -------------------------
# Difficulty preset factories
# -------------------------


def get_bitwise_xor_presets(ndim: int, height: int) -> dict:
    """Return five difficulty presets for BitwiseXORReward.

    The presets target approximate L1 distance bands by selecting the highest
    constrained bit and number of constrained dimensions. Typical distance scales
    like m · 2^b, where m is the number of constrained dims and b the highest bit.

    Bands (steps from s0):
      - easy:        ~50–100
      - medium:      ~250–500
      - hard:        ~1k–2.5k
      - challenging: ~2.5k–5k
      - impossible:  5k+

    Notes
    - You may tweak m (dims) and bit windows to fine-tune distances for your D,H.
    - Tier weights are geometric to encourage reaching higher tiers.
    - Parity checks default to even parity across constrained dims per bit-plane.
    """

    # Choose contiguous first m dims for simplicity; users can override.
    def dims(m: int) -> list[int]:
        m = min(max(1, m), ndim)
        return list(range(m))

    presets = {
        "easy": dict(
            R0=0.0,
            tier_weights=[1.0, 10.0, 100.0],
            dims_constrained=dims(3),
            bits_per_tier=[(0, 4), (0, 5), (0, 5)],
        ),
        "medium": dict(
            R0=0.0,
            tier_weights=[1.0, 10.0, 100.0, 1000.0],
            dims_constrained=dims(4),
            bits_per_tier=[(0, 6), (0, 7), (0, 7), (0, 7)],
        ),
        "hard": dict(
            R0=0.0,
            tier_weights=[1.0, 10.0, 100.0, 1000.0],
            dims_constrained=dims(8),
            bits_per_tier=[(0, 8), (0, 8), (0, 8), (0, 8)],
        ),
        "challenging": dict(
            R0=0.0,
            tier_weights=[1.0, 10.0, 100.0, 1000.0],
            dims_constrained=dims(6),
            bits_per_tier=[(0, 9), (0, 9), (0, 9), (0, 9)],
        ),
        "impossible": dict(
            R0=0.0,
            tier_weights=[1.0, 10.0, 100.0, 1000.0, 10000.0],
            dims_constrained=dims(12),
            bits_per_tier=[(0, 9), (0, 10), (0, 10), (0, 10), (0, 10)],
        ),
    }
    return presets


def get_multiplicative_coprime_presets(ndim: int, height: int) -> dict:
    """Return five difficulty presets for MultiplicativeCoprimeReward.

    Bands (steps from s0):
      - easy:        ~50–100 (small primes, small exponents, few active dims)
      - medium:      ~250–500 (adds one prime, caps=2, more dims, light coupling)
      - hard:        ~1k–2.5k (primes up to 11, caps=3, more dims, LCM target)
      - challenging: ~2.5k–5k (primes up to 13, caps=3–4, 10–12 dims, tighter)
      - impossible:  5k+ (primes up to 29, caps=4, 12–16 dims, multiple targets)

    Notes
    - Distances are approximate; increase primes and exponent caps to push further.
    - `active_dims` indexes are relative to state dims; we pick first k for simplicity.
    - `coprime_pairs` are pairs within `active_dims` index space.
    - Tier weights are geometric.
    """

    def dims(k: int) -> list[int]:
        k = min(max(1, k), ndim)
        return list(range(k))

    def chain_pairs(k: int) -> list[tuple[int, int]]:
        return [(i, i + 1) for i in range(max(0, k - 1))]

    presets = {
        "easy": dict(
            R0=0.0,
            tier_weights=[1.0, 10.0, 100.0],
            primes=[2, 3, 5],
            exponent_caps=[2, 2, 2],
            active_dims=dims(3),
            coprime_pairs=chain_pairs(3),
            target_lcms=[None, None, None],
        ),
        "medium": dict(
            R0=0.0,
            tier_weights=[1.0, 10.0, 100.0, 1000.0],
            primes=[2, 3, 5, 7],
            exponent_caps=[2, 2, 2, 2],
            active_dims=dims(5),
            coprime_pairs=chain_pairs(5),
            target_lcms=[None, None, None, None],
        ),
        "hard": dict(
            R0=0.0,
            tier_weights=[1.0, 10.0, 100.0, 1000.0],
            primes=[2, 3, 5, 7, 11],
            exponent_caps=[3, 3, 3, 3],
            active_dims=dims(8),
            coprime_pairs=chain_pairs(8),
            # Example LCM target encourages compositional reasoning
            target_lcms=[
                None,
                None,
                2**3 * 3**2 * 5 * 7 * 11,
                2**3 * 3**2 * 5 * 7 * 11,
            ],
        ),
        "challenging": dict(
            R0=0.0,
            tier_weights=[1.0, 10.0, 100.0, 1000.0],
            primes=[2, 3, 5, 7, 11, 13],
            exponent_caps=[3, 3, 4, 4],
            active_dims=dims(10),
            coprime_pairs=chain_pairs(10),
            target_lcms=[None, None, None, 2**3 * 3**2 * 5**2 * 13],
        ),
        "impossible": dict(
            R0=0.0,
            tier_weights=[1.0, 10.0, 100.0, 1000.0, 10000.0],
            primes=[2, 3, 5, 7, 11, 13, 17, 19, 23, 29],
            exponent_caps=[4, 4, 4, 4, 4],
            active_dims=dims(12),
            coprime_pairs=chain_pairs(12),
            target_lcms=[None, None, None, None, 2**4 * 3**3 * 5**2 * 7 * 11],
        ),
    }
    return presets


def get_original_presets(ndim: int, height: int) -> dict:
    """Return five presets for OriginalReward.

    These presets primarily control the relative importance of the outer ring (R1)
    and thin band (R2). Exploration difficulty (distance from s0) is more a function
    of (D, H) than of these weights; tune D and H externally to match your distance
    bands.
    """
    presets = {
        "easy": dict(R0=0.1, R1=0.3, R2=1.0),
        "medium": dict(R0=0.1, R1=0.5, R2=2.0),
        "hard": dict(R0=0.05, R1=0.6, R2=3.0),
        "challenging": dict(R0=0.01, R1=0.6, R2=4.0),
        "impossible": dict(R0=0.0, R1=0.7, R2=5.0),
    }
    return presets


def get_cosine_presets(ndim: int, height: int) -> dict:
    """Return five presets for CosineReward.

    R1 scales the oscillatory product, and `mode_gamma` (used only for mode
    detection thresholding) tightens what is considered a "mode-like" maximum.
    """
    presets = {
        "easy": dict(R0=0.1, R1=0.3, mode_gamma=0.7),
        "medium": dict(R0=0.1, R1=0.5, mode_gamma=0.8),
        "hard": dict(R0=0.05, R1=0.6, mode_gamma=0.85),
        "challenging": dict(R0=0.01, R1=0.7, mode_gamma=0.9),
        "impossible": dict(R0=0.0, R1=0.8, mode_gamma=0.92),
    }
    return presets


def get_sparse_presets(ndim: int, height: int) -> dict:
    """Return five presets for SparseReward.

    SparseReward has built-in targets; it ignores most kwargs. Presets are provided
    for API symmetry and future extensibility.
    """
    empty: dict = {}
    presets = {
        "easy": empty,
        "medium": empty,
        "hard": empty,
        "challenging": empty,
        "impossible": empty,
    }
    return presets


def get_reward_presets(reward_fn_str: str, ndim: int, height: int) -> dict:
    """Return presets for a given reward name: 'bitwise_xor', 'multiplicative_coprime', 'template_minkowski'.

    Usage
    ----
    presets = get_reward_presets("bitwise_xor", D, H)
    kwargs = presets["hard"]
    env = HyperGrid(ndim=D, height=H, reward_fn_str="bitwise_xor", reward_fn_kwargs=kwargs)
    """
    if reward_fn_str == "bitwise_xor":
        return get_bitwise_xor_presets(ndim, height)
    if reward_fn_str == "multiplicative_coprime":
        return get_multiplicative_coprime_presets(ndim, height)
    if reward_fn_str == "original":
        return get_original_presets(ndim, height)
    if reward_fn_str == "cosine":
        return get_cosine_presets(ndim, height)
    raise ValueError(f"Unknown reward_fn_str for presets: {reward_fn_str}")


class BitwiseXORReward(GridReward):
    """Tiered, compositional reward based on bitwise XOR/parity constraints.

    This class implements the "Bitwise/XOR fractal" environment family: where tiers
    progressively constrain bit-planes across a subset of dimensions via linear parity
    checks over GF(2). It supports easy sharding by high-bit prefixes, and difficulty
    control by adjusting which bit-planes and how many dimensions are constrained per tier.

    GF(2) is the finite field with two elements {0, 1}, where addition and
    multiplication are performed modulo 2. In this context, vector addition is
    equivalent to bitwise XOR, and matrix–vector products (A @ b) are evaluated
    entrywise modulo 2.

    Reward form:
        R(s) = R0 + Σ_t tier_weights[t] · 1[ state satisfies all constraints up to tier t ]

    Key kwargs (with reasonable defaults):
        - R0: float, base reward (default 0.0)
        - tier_weights: list[float], strictly increasing weights for each tier
        - dims_constrained: Optional[list[int]] subset of dims to constrain
          (default: all dims)
        - bits_per_tier: list[tuple[int,int]]; for each tier t, inclusive bit range
          (low_bit, high_bit). Example: [(0,5), (0,7), (0,9)].
        - parity_checks: Optional[list[dict]]; per tier, optional parity system:
            Each entry may contain:
              { "A": IntTensor[num_checks, m], "c": IntTensor[num_checks] }
            where m = len(dims_constrained). Constraints apply identically to every
            bit-plane specified for that tier: A @ b(mod2) == c, where b are the
            bit values across constrained dimensions at the tested bit-plane.
            If omitted for a tier, a single even-parity check across all constrained
            dims is used by default: sum(b) mod 2 == 0.

    Difficulty presets align with step ranges by controlling the highest bit used
    and the number of constrained dimensions. Typical distance from origin for
    valid modes scales roughly like (constrained_dims · 2^{highest_bit}).
    """

    def __init__(self, height: int, ndim: int, **kwargs):
        super().__init__(height, ndim, **kwargs)
        self.R0: float = float(kwargs.get("R0", 0.0))
        tier_weights = kwargs.get("tier_weights", [1.0, 10.0, 100.0])
        assert isinstance(tier_weights, (list, tuple)) and len(tier_weights) > 0
        self.tier_weights: list[float] = [float(w) for w in tier_weights]

        dims_constrained = kwargs.get("dims_constrained", None)
        if dims_constrained is None:
            dims_constrained = list(range(ndim))
        assert len(dims_constrained) > 0
        self.dims_constrained: list[int] = list(map(int, dims_constrained))

        bits_per_tier = kwargs.get("bits_per_tier", None)
        if bits_per_tier is None:
            # Default: widen the bit window gradually
            bits_per_tier = [(0, 5), (0, 7), (0, 9)]
        assert len(bits_per_tier) == len(self.tier_weights)
        self.bits_per_tier: list[tuple[int, int]] = [
            (int(lo), int(hi)) for (lo, hi) in bits_per_tier
        ]

        self.parity_checks = kwargs.get("parity_checks", None)
        if self.parity_checks is not None:
            assert len(self.parity_checks) == len(self.tier_weights)

    def _even_parity_mask(self, bits: torch.Tensor) -> torch.Tensor:
        """bits: (..., m) int/bool -> returns (...,) bool for even parity."""
        if bits.dtype != torch.long:
            bits = bits.long()
        return (bits.sum(dim=-1) & 1) == 0

    def _apply_parity_checks(
        self, bits_plane: torch.Tensor, tier_idx: int
    ) -> torch.Tensor:
        """Apply GF(2) linear parity checks at a single bit-plane.

        bits_plane: (..., m) with m=len(dims_constrained), integer in {0,1}.
        Returns mask (...,) bool.
        """
        if self.parity_checks is None or self.parity_checks[tier_idx] is None:
            return self._even_parity_mask(bits_plane)

        cfg = self.parity_checks[tier_idx]
        A: torch.Tensor | None = cfg.get("A")
        c: torch.Tensor | None = cfg.get("c")
        if A is None or c is None:
            return self._even_parity_mask(bits_plane)

        # Ensure device/dtype
        A = A.to(bits_plane.device).long()
        c = c.to(bits_plane.device).long()
        # Compute (A @ bits) mod 2 for each batch element
        # reshape bits to (..., m, 1) for bmm if needed, but here we can do matmul
        # by flattening the batch.
        flat = bits_plane.reshape(-1, bits_plane.shape[-1]).long()
        prod = (flat @ A.t()) & 1  # shape (B, num_checks)
        target = c.unsqueeze(0).expand_as(prod)
        ok = (prod == target).all(dim=-1)
        return ok.reshape(bits_plane.shape[:-1])

    def __call__(self, states_tensor: torch.Tensor) -> torch.Tensor:
        R = torch.zeros(
            states_tensor.shape[:-1],
            device=states_tensor.device,
            dtype=torch.get_default_dtype(),
        )
        if self.R0 != 0.0:
            R += self.R0

        # Select constrained dims
        x = states_tensor.index_select(
            dim=-1,
            index=torch.tensor(self.dims_constrained, device=states_tensor.device),
        )

        # Progressive, compositional tiers: a state gets tier t reward only if it
        # satisfies all constraints up to t.
        valid_up_to_t = torch.ones(x.shape[:-1], device=x.device, dtype=torch.bool)
        for t, w in enumerate(self.tier_weights):
            lo_b, hi_b = self.bits_per_tier[t]
            tier_mask = torch.ones_like(valid_up_to_t)
            for b in range(lo_b, hi_b + 1):
                bits = ((x >> b) & 1).long()
                plane_ok = self._apply_parity_checks(bits, t)
                tier_mask = tier_mask & plane_ok
            valid_up_to_t = valid_up_to_t & tier_mask
            R = R + (valid_up_to_t.to(R.dtype) * float(w))

        return R


class MultiplicativeCoprimeReward(GridReward):
    """Tiered reward based on prime-support and coprimality/lcm composition.

    Each tier enforces that per-dimension values use only a small shared prime set
    with bounded exponents, plus optional cross-dimension constraints (pairwise
    coprime pairs and/or target lcm). Higher tiers tighten exponent caps or add
    additional global targets. This encourages information sharing to learn the
    latent prime/exponent structure.

    Reward form:
        R(s) = R0 + Σ_t tier_weights[t] · 1[ constraints_0..t all satisfied ]

    Key kwargs:
        - R0: float, base reward (default 0.0)
        - tier_weights: list[float]
        - primes: list[int], e.g., [2,3,5,7,11]
        - exponent_caps: list[int], same length as tier_weights. Cap for every prime
          at tier t (uniform cap across primes for simplicity).
        - active_dims: Optional[list[int]]; constraints only apply to these dims
          (default: all dims). Other dims are ignored in constraints.
        - coprime_pairs: Optional[list[tuple[int,int]]]; indices relative to active_dims.
        - target_lcms: Optional[list[int | None]]; per-tier target lcm across active dims.

    Notes:
    - Values 0 are treated as invalid for prime-support constraints (cannot factorize);
      value 1 is valid with all-zero exponents.
    - Implementation removes primes up to the current tier cap and checks residue == 1.
      Exponent counts are accumulated to evaluate LCM targets.
    """

    def __init__(self, height: int, ndim: int, **kwargs):
        super().__init__(height, ndim, **kwargs)
        self.R0: float = float(kwargs.get("R0", 0.0))
        tier_weights = kwargs.get("tier_weights", [1.0, 10.0, 100.0])
        assert isinstance(tier_weights, (list, tuple)) and len(tier_weights) > 0
        self.tier_weights: list[float] = [float(w) for w in tier_weights]

        primes = kwargs.get("primes", [2, 3, 5])
        assert isinstance(primes, (list, tuple)) and len(primes) > 0
        self.primes: list[int] = [int(p) for p in primes]

        exponent_caps = kwargs.get("exponent_caps", [2] * len(self.tier_weights))
        assert len(exponent_caps) == len(self.tier_weights)
        self.exponent_caps: list[int] = [int(c) for c in exponent_caps]

        active_dims = kwargs.get("active_dims", None)
        if active_dims is None:
            active_dims = list(range(ndim))
        self.active_dims: list[int] = list(map(int, active_dims))

        self.coprime_pairs = kwargs.get("coprime_pairs", None)
        self.target_lcms = kwargs.get("target_lcms", [None] * len(self.tier_weights))
        assert isinstance(self.target_lcms, (list, tuple)) and len(
            self.target_lcms
        ) == len(self.tier_weights)

    def _factor_exponents_up_to_cap(self, v: torch.Tensor, cap: int):
        """Return (residue, exponents) after dividing by allowed primes up to `cap`.

        v: (...,) LongTensor, non-negative.
        residue: (...,) after stripping primes up to `cap` times each.
        exps: tensor [num_primes, ...] of exponent counts per prime (capped by `cap`).
        """
        residue = v.clone()
        exps = []
        for p in self.primes:
            p = int(p)
            count = torch.zeros_like(residue)
            # Repeatedly divide by p but not more than cap times
            for _ in range(cap):
                divisible = (residue % p) == 0
                if not torch.any(divisible):
                    break
                residue = torch.where(divisible, residue // p, residue)
                count = count + divisible.long()
            exps.append(count)
        exps = torch.stack(exps, dim=0)  # [num_primes, ...]
        return residue, exps

    def _pairwise_coprime_ok(self, v: torch.Tensor) -> torch.Tensor:
        """Check pairwise coprime on configured pairs using prime divisibility.

        v: (..., m) with m = len(active_dims).
        Returns (...,) bool.
        """
        if not self.coprime_pairs:
            return torch.ones(v.shape[:-1], dtype=torch.bool, device=v.device)
        ok = torch.ones(v.shape[:-1], dtype=torch.bool, device=v.device)
        for p in self.primes:
            div = (v % int(p)) == 0  # (..., m)
            for i, j in self.coprime_pairs:
                m = div.shape[-1]
                if i >= m or j >= m:
                    continue
                both = div[..., i] & div[..., j]
                ok = ok & (~both)
        return ok

    def _lcm_ok(self, exps: torch.Tensor, target_lcm: int) -> torch.Tensor:
        """Check whether max exponents across dims match target LCM's exponents.

        exps: [num_primes, ..., m]
        target_lcm: int
        Returns (...,) bool.
        """
        # Factor target_lcm fully over allowed primes; reject if leftover > 1
        remaining = int(target_lcm)
        target_counts: list[int] = []
        for p in self.primes:
            p = int(p)
            c = 0
            while remaining % p == 0:
                remaining //= p
                c += 1
            target_counts.append(c)
        if remaining != 1:
            # Target contains primes outside allowed set -> impossible
            shape = exps.shape[1:-1]  # broadcast shape (...)
            return torch.zeros(shape, dtype=torch.bool, device=exps.device)
        target = torch.tensor(target_counts, dtype=torch.long, device=exps.device)

        max_exp = exps.max(dim=-1).values  # [num_primes, ...]
        # Broadcast compare to target per prime
        while target.dim() < max_exp.dim():
            target = target.unsqueeze(-1)
        return (max_exp == target).all(dim=0)

    def __call__(self, states_tensor: torch.Tensor) -> torch.Tensor:
        R = torch.zeros(
            states_tensor.shape[:-1],
            device=states_tensor.device,
            dtype=torch.get_default_dtype(),
        )
        if self.R0 != 0.0:
            R += self.R0

        x_full = states_tensor
        x = x_full.index_select(
            dim=-1, index=torch.tensor(self.active_dims, device=states_tensor.device)
        )
        # Disallow zeros at the outset for constraints (they cannot have finite prime support)
        base_valid = (x != 0).all(dim=-1)

        valid_up_to_t = base_valid
        for t, w in enumerate(self.tier_weights):
            cap = self.exponent_caps[t]
            residue, exps = self._factor_exponents_up_to_cap(x.reshape(-1), cap)
            residue = residue.reshape(x.shape)
            exps = exps.reshape((len(self.primes),) + x.shape)

            # Prime-support with bounded exponents: residue must be 1 or original value 1
            support_ok = (residue == 1) | (x == 1)
            support_ok = support_ok.all(dim=-1)

            pairs_ok = self._pairwise_coprime_ok(x)
            lcm_ok = torch.ones_like(pairs_ok)
            if self.target_lcms[t] is not None:
                lcm_ok = self._lcm_ok(exps, int(self.target_lcms[t]))

            tier_ok = support_ok & pairs_ok & lcm_ok
            valid_up_to_t = valid_up_to_t & tier_ok
            R = R + (valid_up_to_t.to(R.dtype) * float(w))

        return R
