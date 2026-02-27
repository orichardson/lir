import torch
from torch import nn
from torch.autograd import grad
from torch.nn.utils import parameters_to_vector, vector_to_parameters
from torchdiffeq import odeint

from typing import Callable, Iterable, Optional, Union, Sequence


class Optimizer:
    """
    A unified optimizer interface.

    Modes
    -----
    backend="ode":
        Integrates the gradient flow dθ/dt = -lr * ∇_θ L(θ) over t ∈ [0, 1].
        Supports adaptive (e.g., 'dopri5') or fixed-step (e.g., 'rk4') solvers.

    backend="standard":
        Wraps a standard torch.optim optimizer (e.g., 'sgd', 'adam', 'lbfgs').

    API
    ---
    step(loss_closure=...)
        The closure should recompute and return a scalar loss tensor given the
        current parameters. It must NOT call backward() itself; this class
        handles gradients appropriately for each backend.
    """

    def __init__(self,
                 params: Iterable[torch.nn.Parameter],
                 backend: str = "ode",  # "ode" or "standard"
                 # --- ODE backend options ---
                 ode_solve_mode: str = "adaptive",     # "adaptive" or "fixed"
                 ode_method: str = "dopri5",
                 ode_fixed_method: str = "rk4",
                 atol: float = 1e-4,
                 rtol: float = 1e-6,
                 max_num_steps: Optional[int] = None,
                 n_steps: Optional[int] = None,
                 lr: Union[float, Sequence[float]] = 1e-2,
                 grad_clip_norm: Optional[float] = 5.0,
                 preconditioner: Optional[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = None,
                 # --- Standard backend options ---
                 standard_type: Optional[str] = None,  # "sgd", "adam", "adamw", "lbfgs"; or None if passing instance
                 standard_kwargs: Optional[dict] = None,
                 standard_instance: Optional[torch.optim.Optimizer] = None):
        super().__init__()

        # Collect params in a stable order used by parameters_to_vector
        self.params = list(params)
        if len(self.params) == 0:
            raise ValueError("params must be a non-empty iterable of Tensors requiring grad.")

        self.backend = backend.lower().strip()
        if self.backend not in {"ode", "standard"}:
            raise ValueError("backend must be 'ode' or 'standard'.")

        # Device inferred from first parameter
        self.device = self.params[0].device

        # -----------------------------
        # Per-parameter LR processing
        # -----------------------------
        # Save raw lr for standard backend fallback
        self._lr_input = lr  # could be float or iterable

        # Build a per-element LR vector for ODE backend *after* knowing numels
        self._per_param_numels = [p.numel() for p in self.params]
        self._theta_numel = sum(self._per_param_numels)

        if isinstance(lr, (int, float)):
            self._lr_is_scalar = True
            self._lr_scalar = float(lr)
            self._lr_flat = None  # built on demand if needed
        else:
            lr_list = list(lr)
            if len(lr_list) != len(self.params):
                raise ValueError(f"Iterable lr length ({len(lr_list)}) must equal number of params ({len(self.params)}).")
            if any((not isinstance(x, (int, float))) for x in lr_list):
                raise TypeError("Iterable lr must contain floats.")
            self._lr_is_scalar = False
            self._lr_scalar = None
            # Expand per-tensor LRs to a flat vector aligned with parameters_to_vector order
            pieces = []
            for lr_i, n_i in zip(lr_list, self._per_param_numels):
                pieces.append(torch.full((n_i,), float(lr_i), device=self.device))
            self._lr_flat = torch.cat(pieces, dim=0) if pieces else torch.tensor([], device=self.device)

        # -----------------------------
        # ODE-specific configuration
        # -----------------------------
        if self.backend == "ode":
            self.ode_solve_mode = ode_solve_mode
            self.ode_method = ode_method
            self.ode_fixed_method = ode_fixed_method
            self.atol = atol
            self.rtol = rtol
            self.max_num_steps = max_num_steps
            self.grad_clip_norm = grad_clip_norm
            self.preconditioner = preconditioner

            # Fixed [0, 1] horizon
            self.t_span = torch.tensor([0.0, 1.0], device=self.device)
            horizon = float(self.t_span[-1] - self.t_span[0])

            if n_steps is None:
                self.step_size = horizon / 5.0 # default to 5 steps
                # print(f"[Optimizer/ODE] Fixed-step: defaulting to 5 steps")
            else:
                self.step_size = horizon / float(n_steps)
                # print(f"[Optimizer/ODE] Fixed-step: n_steps={n_steps}")

            # if self.backend == "ode" and self.ode_solve_mode == "adaptive":
            #     print(f"[Optimizer/ODE] Adaptive: method={self.ode_method}, rtol={self.rtol}, atol={self.atol}, max_num_steps={self.max_num_steps}")

        # -----------------------------
        # Standard optimizer backend
        # -----------------------------
        self.std_optim = None
        if self.backend == "standard":
            if standard_instance is not None:
                # Respect user-supplied optimizer; ignore iterable lr to avoid conflicts.
                if not isinstance(lr, (int, float)):
                    print("[Optimizer/Standard] Note: received iterable lr but a custom standard_instance was provided; "
                          "per-parameter LR will be taken from the instance, not from `lr`.")
                self.std_optim = standard_instance
            else:
                if standard_type is None:
                    raise ValueError("For backend='standard', provide standard_type or standard_instance.")
                standard_type = standard_type.lower().strip()
                standard_kwargs = dict(standard_kwargs or {})

                # If lr is iterable, we will create param groups with individual lrs
                if isinstance(self._lr_input, (int, float)):
                    # single group; ensure lr present unless user already set it
                    standard_kwargs.setdefault("lr", float(self._lr_input))
                    param_groups = [{"params": self.params}]
                else:
                    # per-parameter param groups (ignore any 'lr' inside standard_kwargs)
                    if "lr" in standard_kwargs:
                        standard_kwargs = {k: v for k, v in standard_kwargs.items() if k != "lr"}
                        print("[Optimizer/Standard] Removed 'lr' from standard_kwargs "
                              "since an iterable lr was provided (using per-parameter groups).")
                    param_groups = [{"params": [p], "lr": float(lr_i)}
                                    for p, lr_i in zip(self.params, self._lr_input)]

                if standard_type == "sgd":
                    self.std_optim = torch.optim.SGD(params=param_groups, **standard_kwargs)
                elif standard_type == "adam":
                    self.std_optim = torch.optim.Adam(params=param_groups, **standard_kwargs)
                elif standard_type == "adamw":
                    self.std_optim = torch.optim.AdamW(params=param_groups, **standard_kwargs)
                elif standard_type == "lbfgs":
                    self.std_optim = torch.optim.LBFGS(params=param_groups, **standard_kwargs)
                else:
                    raise ValueError(f"Unsupported standard_type='{standard_type}'. Try 'sgd', 'adam', 'adamw', or 'lbfgs'.")

    # -----------------------------
    # ODE helpers
    # -----------------------------
    @torch.no_grad()
    def _write_params(self, theta_vec: torch.Tensor) -> None:
        vector_to_parameters(vec=theta_vec, parameters=self.params)

    def _grad_at(self, theta_vec: torch.Tensor, loss_closure: Callable[[], torch.Tensor]) -> torch.Tensor:
        """Load θ, compute loss, and return ∇_θ L(θ) as a flat vector without
        permanently changing requires_grad on params."""
        self._write_params(theta_vec=theta_vec)

        # Save current flags
        orig_flags = [p.requires_grad for p in self.params]
        for p in self.params:
            p.requires_grad_(True)

        try:
            with torch.enable_grad():
                loss = loss_closure()

            g_list = grad(outputs=loss,
                        inputs=self.params,
                        create_graph=False,
                        retain_graph=False,
                        allow_unused=False)
            g_vec = torch.cat([g.reshape(-1) for g in g_list])

            # Optional grad clipping
            if self.grad_clip_norm is not None:
                norm = g_vec.norm(p=2)
                if torch.isfinite(norm) and norm > self.grad_clip_norm:
                    g_vec = g_vec * (self.grad_clip_norm / (norm + 1e-12))

            # Optional preconditioning
            if self.preconditioner is not None:
                with torch.no_grad():
                    g_vec = self.preconditioner(g_vec, theta_vec)

            return g_vec
        finally:
            # Restore original flags
            for p, flag in zip(self.params, orig_flags):
                p.requires_grad_(flag)

    def _ode_rhs(self, t: torch.Tensor, theta_vec: torch.Tensor, loss_closure: Callable[[], torch.Tensor]) -> torch.Tensor:
        # dθ/dt = - LR ⊙ ∇_θ L(θ), where LR is scalar or flat vector
        g = self._grad_at(theta_vec=theta_vec, loss_closure=loss_closure)
        if self._lr_is_scalar:
            return -self._lr_scalar * g
        else:
            # elementwise multiply by lr vector
            return -(self._lr_flat * g)

    def _integrate_adaptive(self, theta_start: torch.Tensor, loss_closure: Callable[[], torch.Tensor]) -> torch.Tensor:
        options = {"max_num_steps": self.max_num_steps} if self.max_num_steps else None
        theta_path = odeint(func=lambda tt, th: self._ode_rhs(t=tt, theta_vec=th, loss_closure=loss_closure),
                            y0=theta_start,
                            t=self.t_span,
                            method=self.ode_method,
                            rtol=self.rtol,
                            atol=self.atol,
                            options=options)
        print(f"[Optimizer/ODE] Adaptive steps taken: {len(theta_path)}")
        return theta_path[-1]

    def _integrate_fixed(self, theta_start: torch.Tensor, loss_closure: Callable[[], torch.Tensor]) -> torch.Tensor:
        theta_path = odeint(func=lambda tt, th: self._ode_rhs(t=tt, theta_vec=th, loss_closure=loss_closure),
                            y0=theta_start,
                            t=self.t_span,
                            method=self.ode_fixed_method,
                            options={"step_size": self.step_size})
        return theta_path[-1]

    # -----------------------------
    # Public API
    # -----------------------------
    def step(self, loss_closure: Callable[[], torch.Tensor]) -> float:
        """
        Perform one optimization step using the selected backend.

        loss_closure:
            Recomputes and returns the scalar loss (no .backward() inside).
            NOTE: your inner solver may call .backward() internally; grads must be enabled.
        """
        if self.backend == "standard":
            if self.std_optim is None:
                raise RuntimeError("Standard optimizer backend is not initialized.")
            self.std_optim.zero_grad(set_to_none=True)
            with torch.enable_grad():
                loss = loss_closure()
            loss.backward()
            self.std_optim.step()
            return float(loss.detach().item())

        # ODE backend
        theta_start = parameters_to_vector(self.params).to(device=self.device)

        if self.ode_solve_mode == "fixed":
            theta_end = self._integrate_fixed(theta_start=theta_start, loss_closure=loss_closure)
        else:
            try:
                theta_end = self._integrate_adaptive(theta_start=theta_start, loss_closure=loss_closure)
            except AssertionError:
                print("[Optimizer/ODE] Adaptive solver exceeded step budget — falling back to fixed RK4.")
                theta_end = self._integrate_fixed(theta_start=theta_start, loss_closure=loss_closure)

        # Write back θ(1) and report loss at new params.
        # IMPORTANT: do NOT disable grad here — inner solvers may call backward().
        self._write_params(theta_vec=theta_end)
        loss = loss_closure()
        return float(loss.detach().item())
