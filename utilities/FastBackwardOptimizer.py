# FastBackwardOptimizer.py
"""
Fast backward trajectory optimizer using shooting method with NN warm-start.

Key insight: For backward dynamics, we only need to find the oldest state x(t-H).
Given x(t-H) and controls, forward dynamics uniquely determines all intermediate states.

Two modes:
1. Shooting: Optimize only x(t-H), roll forward to verify
2. Interleaved: NN predicts one step, optimizer refines, repeat

Approach:
1. Quick sequential solve to get initial guess for x(t-H)
2. Shooting optimization: find x(t-H) such that forward_rollout ends at anchor x(t)
3. Roll forward to get full trajectory
"""

import numpy as np
from scipy.optimize import minimize
from typing import Optional, Tuple, Callable
import time

from SI_Toolkit_ASF.car_model import car_model
from SI_Toolkit.computation_library import NumpyLibrary
from utilities.Settings import Settings
from utilities.state_utilities import (
    POSE_THETA_IDX, POSE_THETA_SIN_IDX, POSE_THETA_COS_IDX,
    POSE_X_IDX, POSE_Y_IDX,
    LINEAR_VEL_X_IDX, ANGULAR_VEL_Z_IDX, STEERING_ANGLE_IDX,
    STATE_VARIABLES,
)

STATE_DIM = len(STATE_VARIABLES)


def _fix_sin_cos(x: np.ndarray) -> np.ndarray:
    """Ensure sin/cos are consistent with pose_theta."""
    x = x.copy()
    theta = x[POSE_THETA_IDX]
    x[POSE_THETA_SIN_IDX] = np.sin(theta)
    x[POSE_THETA_COS_IDX] = np.cos(theta)
    return x


def _angle_wrap(angle: float) -> float:
    """Wrap angle to [-pi, pi]."""
    return np.arctan2(np.sin(angle), np.cos(angle))


class FastBackwardOptimizer:
    """
    Backward optimizer using single-shooting method.
    
    Optimizes only the oldest state x(t-H), then rolls forward to verify
    we reach the anchor x(t). This is fast and guarantees dynamical consistency.
    """
    
    def __init__(
        self,
        dt: float = 0.01,
        mu: Optional[float] = None,
        max_iter: int = 200,
        tol: float = 1e-7,
        verbose: bool = False,
    ):
        self.dt = abs(dt)
        self.max_iter = max_iter
        self.tol = tol
        self.verbose = verbose
        
        self.lib = NumpyLibrary()
        self.car_model = car_model(
            model_of_car_dynamics=Settings.ODE_MODEL_OF_CAR_DYNAMICS,
            batch_size=1,
            car_parameter_file=Settings.CONTROLLER_CAR_PARAMETER_FILE,
            dt=self.dt,
            intermediate_steps=1,
            computation_lib=self.lib,
        )
        
        if mu is not None:
            self.car_model.change_friction_coefficient(mu)
        
        # State scaling for optimization conditioning
        self.state_scale = np.array([
            1.0,   # angular_vel_z
            1.0,   # linear_vel_x
            0.2,   # linear_vel_y
            np.pi, # pose_theta
            1.0,   # pose_theta_sin
            1.0,   # pose_theta_cos
            5.0,   # pose_x
            5.0,   # pose_y
            0.1,   # slip_angle
            0.3,   # steering_angle
        ], dtype=np.float32)
        
        # Indices to optimize (skip sin/cos as they're derived)
        self.opt_indices = [i for i in range(STATE_DIM) 
                           if i not in {POSE_THETA_SIN_IDX, POSE_THETA_COS_IDX}]
        self.n_opt = len(self.opt_indices)
        
        self.stats = {}
    
    def set_mu(self, mu: float):
        self.car_model.change_friction_coefficient(mu)
    
    def forward_step(self, x: np.ndarray, u: np.ndarray) -> np.ndarray:
        """Single forward step with PID."""
        x = np.asarray(x, dtype=np.float32).reshape(1, -1)
        u = np.asarray(u, dtype=np.float32).reshape(1, -1)
        return self.car_model.step_dynamics(x, u).flatten()
    
    def forward_rollout(self, x_start: np.ndarray, controls: np.ndarray) -> np.ndarray:
        """Roll forward from x_start using controls [H, 2]."""
        H = len(controls)
        traj = np.zeros((H + 1, STATE_DIM), dtype=np.float32)
        traj[0] = _fix_sin_cos(x_start)
        
        x = traj[0].copy()
        for i in range(H):
            x = self.forward_step(x, controls[i])
            traj[i + 1] = _fix_sin_cos(x)
        
        return traj
    
    def _kinematic_backward_rollout(self, x_current: np.ndarray, controls: np.ndarray) -> np.ndarray:
        """
        Generate a kinematic backward rollout as initial guess.
        
        Uses simple kinematics (velocity-based integration with negative dt)
        to estimate where the car was at t-H given where it is now.
        
        Args:
            x_current: Current/anchor state [state_dim]
            controls: [H, 2] controls, oldest to newest
            
        Returns:
            x_oldest_guess: Estimated state at t-H [state_dim]
        """
        H = len(controls)
        dt = self.dt
        
        # Start from current state and integrate backwards
        x = x_current.copy()
        
        # Go backwards: from newest control to oldest
        for i in range(H - 1, -1, -1):
            # Extract relevant state variables
            vx = x[LINEAR_VEL_X_IDX]
            theta = x[POSE_THETA_IDX]
            steering = x[STEERING_ANGLE_IDX]
            angular_vel_z = x[ANGULAR_VEL_Z_IDX]
            
            # Simple kinematic backward step (negative dt)
            # dx/dt = vx * cos(theta), dy/dt = vx * sin(theta), dtheta/dt = angular_vel_z
            x[POSE_X_IDX] -= vx * np.cos(theta) * dt
            x[POSE_Y_IDX] -= vx * np.sin(theta) * dt
            x[POSE_THETA_IDX] -= angular_vel_z * dt
            
            # Update sin/cos
            x[POSE_THETA_COS_IDX] = np.cos(x[POSE_THETA_IDX])
            x[POSE_THETA_SIN_IDX] = np.sin(x[POSE_THETA_IDX])
            
            # Velocity and steering: use control as hint for what they were
            # (this is approximate but helps find the right basin)
            u_angular = controls[i, 0]
            u_trans = controls[i, 1]
            
            # Steering tends to follow angular control with some lag
            # Linear velocity follows translational control
            # These are rough approximations
            x[STEERING_ANGLE_IDX] = u_angular * 0.4  # Approximate steering
            x[LINEAR_VEL_X_IDX] = u_trans  # Approximate velocity
        
        return _fix_sin_cos(x)
    
    def _shooting_objective(self, x0_opt: np.ndarray, x_anchor: np.ndarray,
                            controls: np.ndarray,
                            x_prior_opt: Optional[np.ndarray] = None,
                            prior_weight: float = 0.0,
                            traj_ref: Optional[np.ndarray] = None,
                            traj_weight: float = 0.0) -> float:
        """
        Shooting objective: error at anchor after forward rollout + regularization.
        
        cost = 0.5 * ||rollout(x0) - anchor||² 
             + prior_weight * 0.5 * ||x0 - x_prior||²
             + traj_weight * 0.5 * Σ_h ||x_h - x_ref_h||²
        
        x0_opt: Scaled oldest state (optimizable components only)
        x_anchor: Target state at end of rollout
        controls: [H, 2] controls, oldest to newest
        x_prior_opt: Optional prior state (scaled, opt indices only) for oldest state
        prior_weight: Weight for prior term on oldest state
        traj_ref: Optional reference trajectory [H, state_dim] to regularize towards
        traj_weight: Weight for trajectory regularization (per-step)
        """
        # Reconstruct full state
        x_start = np.zeros(STATE_DIM, dtype=np.float32)
        x_start[self.opt_indices] = x0_opt * self.state_scale[self.opt_indices]
        x_start = _fix_sin_cos(x_start)
        
        # Roll forward
        traj = self.forward_rollout(x_start, controls)
        x_final = traj[-1]
        
        # Compute error at anchor
        diff = (x_final - x_anchor) / self.state_scale
        diff[POSE_THETA_IDX] = _angle_wrap(x_final[POSE_THETA_IDX] - x_anchor[POSE_THETA_IDX])
        
        # Anchor term (skip sin/cos)
        anchor_cost = 0.5 * np.sum(diff[self.opt_indices] ** 2)
        
        # Prior regularization term (oldest state only)
        prior_cost = 0.0
        if x_prior_opt is not None and prior_weight > 0:
            prior_diff = x0_opt - x_prior_opt
            prior_cost = prior_weight * 0.5 * np.sum(prior_diff ** 2)
        
        # Trajectory regularization term (all timesteps)
        traj_cost = 0.0
        if traj_ref is not None and traj_weight > 0:
            # traj[:-1] is [H, state_dim], traj_ref is [H, state_dim]
            for h in range(len(controls)):
                diff_h = (traj[h] - traj_ref[h]) / self.state_scale
                diff_h[POSE_THETA_IDX] = _angle_wrap(traj[h, POSE_THETA_IDX] - traj_ref[h, POSE_THETA_IDX])
                traj_cost += 0.5 * np.sum(diff_h[self.opt_indices] ** 2)
            traj_cost *= traj_weight
        
        return anchor_cost + prior_cost + traj_cost
    
    def _shooting_gradient(self, x0_opt: np.ndarray, x_anchor: np.ndarray,
                           controls: np.ndarray,
                           x_prior_opt: Optional[np.ndarray] = None,
                           prior_weight: float = 0.0,
                           traj_ref: Optional[np.ndarray] = None,
                           traj_weight: float = 0.0) -> np.ndarray:
        """Numerical gradient for shooting objective with regularization."""
        eps = 1e-5
        grad = np.zeros_like(x0_opt)
        f0 = self._shooting_objective(x0_opt, x_anchor, controls, x_prior_opt, prior_weight, traj_ref, traj_weight)
        
        for i in range(len(x0_opt)):
            x_plus = x0_opt.copy()
            x_plus[i] += eps
            grad[i] = (self._shooting_objective(x_plus, x_anchor, controls, x_prior_opt, prior_weight, traj_ref, traj_weight) - f0) / eps
        
        return grad
    
    def _quick_sequential_init(self, x_anchor: np.ndarray, controls: np.ndarray) -> np.ndarray:
        """Quick sequential backward solve for initial guess."""
        # For now, just use anchor as initial guess for oldest state
        # Future: could use kinematic backward integration
        _ = controls  # Unused for now
        return x_anchor
    
    def single_step_backward(
        self,
        x_current: np.ndarray,
        u: np.ndarray,
        x_init: Optional[np.ndarray] = None,
        max_iter: int = 20,
        tol: float = 1e-8,
    ) -> Tuple[np.ndarray, bool, dict]:
        """
        Find x_prev such that step(x_prev, u) ≈ x_current.
        
        This is a single-step inverse dynamics problem.
        Uses L-BFGS-B with warm start from x_init (typically from NN prediction).
        
        Args:
            x_current: Current state [state_dim]
            u: Control input [2] (angular_control, translational_control)
            x_init: Initial guess for x_prev (from NN), defaults to x_current
            max_iter: Maximum iterations (will stop early if converged)
            tol: Convergence tolerance (ftol and gtol for L-BFGS-B)
            
        Returns:
            x_prev: Previous state [state_dim]
            converged: Whether optimization converged to tolerance
            stats: dict with 'niter', 'cost', 'success'
        """
        x_current = np.asarray(x_current, dtype=np.float32).flatten()
        x_current = _fix_sin_cos(x_current)
        u = np.asarray(u, dtype=np.float32).flatten()
        
        if x_init is not None:
            x0_guess = np.asarray(x_init, dtype=np.float32).flatten()
        else:
            x0_guess = x_current.copy()
        
        x0_guess = _fix_sin_cos(x0_guess)
        x0_opt = x0_guess[self.opt_indices] / self.state_scale[self.opt_indices]
        
        def objective(x_opt):
            # Reconstruct full state
            x_prev = np.zeros(STATE_DIM, dtype=np.float32)
            x_prev[self.opt_indices] = x_opt * self.state_scale[self.opt_indices]
            x_prev = _fix_sin_cos(x_prev)
            
            # Forward step
            x_next = self.forward_step(x_prev, u)
            x_next = _fix_sin_cos(x_next)
            
            # Error
            diff = (x_next - x_current) / self.state_scale
            diff[POSE_THETA_IDX] = _angle_wrap(x_next[POSE_THETA_IDX] - x_current[POSE_THETA_IDX])
            
            return 0.5 * np.sum(diff[self.opt_indices] ** 2)
        
        def gradient(x_opt):
            eps = 1e-5
            grad = np.zeros_like(x_opt)
            f0 = objective(x_opt)
            for i in range(len(x_opt)):
                x_plus = x_opt.copy()
                x_plus[i] += eps
                grad[i] = (objective(x_plus) - f0) / eps
            return grad
        
        result = minimize(
            objective,
            x0_opt,
            method='L-BFGS-B',
            jac=gradient,
            options={'maxiter': max_iter, 'ftol': tol, 'gtol': tol}
        )
        
        # Reconstruct optimal previous state
        x_prev = np.zeros(STATE_DIM, dtype=np.float32)
        x_prev[self.opt_indices] = result.x * self.state_scale[self.opt_indices]
        x_prev = _fix_sin_cos(x_prev)
        
        converged = result.success and result.fun < 1e-4
        stats = {'niter': result.nit, 'cost': result.fun, 'success': result.success}
        
        return x_prev, converged, stats
    
    def predict_interleaved(
        self,
        x_current: np.ndarray,
        controls: np.ndarray,
        nn_step_fn: Callable[[np.ndarray, np.ndarray], np.ndarray],
        max_iter_per_step: int = 10,
    ) -> Tuple[np.ndarray, np.ndarray, dict]:
        """
        Predict backward trajectory using interleaved NN + optimizer.
        
        For each step:
        1. NN predicts x(t-h) from x(t-h+1) and u(t-h)
        2. Optimizer refines to ensure step(x(t-h), u(t-h)) ≈ x(t-h+1)
        
        Args:
            x_current: Anchor state (current/newest) [state_dim]
            controls: [H, 2] controls, oldest to newest
            nn_step_fn: Function that takes (x_current, u) and returns x_prev estimate
            max_iter_per_step: Max optimizer iterations per step (should be low with NN init)
            
        Returns:
            past_states: [H, state_dim], oldest to newest
            converged: [H] bool array
            stats: dict
        """
        t_start = time.time()
        
        x_current = np.asarray(x_current, dtype=np.float32).flatten()
        x_current = _fix_sin_cos(x_current)
        controls = np.asarray(controls, dtype=np.float32)
        H = len(controls)
        
        # Work backwards from anchor
        past_states = np.zeros((H, STATE_DIM), dtype=np.float32)
        converged = np.zeros(H, dtype=bool)
        
        x = x_current
        
        # Controls are [oldest, ..., newest], we process backwards
        # i.e., first find x(t-1) using u(t-1) which is controls[-1]
        for h in range(H):
            # Control for this backward step
            u = controls[H - 1 - h]  # newest to oldest
            
            # Step 1: NN predicts x_prev
            x_nn = nn_step_fn(x, u)
            
            # Step 2: Optimizer refines
            x_refined, conv = self.single_step_backward(x, u, x_init=x_nn, max_iter=max_iter_per_step)
            
            converged[H - 1 - h] = conv
            past_states[H - 1 - h] = x_refined
            
            # Use refined state for next step
            x = x_refined
        
        elapsed = time.time() - t_start
        
        self.stats = {
            'time_ms': elapsed * 1000,
            'success': np.all(converged),
            'n_failed': int(np.sum(~converged)),
        }
        
        if self.verbose:
            n_ok = np.sum(converged)
            print(f"[Interleaved] H={H}, converged={n_ok}/{H}, time={elapsed*1000:.1f}ms")
        
        return past_states, converged, self.stats
    
    def _continuation_optimization(
        self,
        x_current: np.ndarray,
        controls: np.ndarray,
        traj_ref: np.ndarray,
        lambda_start: float = 1.0,
        lambda_end: float = 1e-6,
        n_stages: int = 5,
        max_iter_per_stage: int = 50,
    ) -> Tuple[np.ndarray, dict]:
        """
        Continuation method: gradually reduce regularization.
        
        1. Start with high λ to find solution near reference
        2. Reduce λ in stages, using previous solution as warm start
        3. Final solution has minimal anchor error while staying in correct basin
        
        Args:
            x_current: Anchor state
            controls: Controls array [H, 2]
            traj_ref: Reference trajectory [H, state_dim]
            lambda_start: Initial regularization weight (high)
            lambda_end: Final regularization weight (low/zero)
            n_stages: Number of reduction stages
            max_iter_per_stage: Max iterations per stage
            
        Returns:
            x0_opt: Optimal oldest state (scaled)
            stats: dict with per-stage info
        """
        H = len(controls)
        
        # Compute lambda schedule (geometric decay)
        if lambda_end > 0:
            lambdas = np.geomspace(lambda_start, lambda_end, n_stages)
        else:
            lambdas = np.geomspace(lambda_start, lambda_start / (10 ** n_stages), n_stages)
            lambdas[-1] = 0.0  # Final stage with no regularization
        
        # Start with reference's oldest state
        x0_opt = traj_ref[0, self.opt_indices] / self.state_scale[self.opt_indices]
        
        stage_stats = []
        total_iters = 0
        
        for stage, lam in enumerate(lambdas):
            result = minimize(
                lambda x: self._shooting_objective(x, x_current, controls, None, 0.0, traj_ref, lam),
                x0_opt,
                method='L-BFGS-B',
                jac=lambda x: self._shooting_gradient(x, x_current, controls, None, 0.0, traj_ref, lam),
                options={'maxiter': max_iter_per_stage, 'ftol': self.tol, 'gtol': self.tol}
            )
            
            x0_opt = result.x  # Warm start for next stage
            total_iters += result.nit
            
            # Compute anchor cost (without regularization)
            anchor_cost = self._shooting_objective(result.x, x_current, controls, None, 0.0, None, 0.0)
            
            stage_stats.append({
                'lambda': lam,
                'iters': result.nit,
                'total_cost': result.fun,
                'anchor_cost': anchor_cost,
            })
            
            if self.verbose:
                print(f"  Stage {stage+1}/{n_stages}: λ={lam:.2e}, iters={result.nit}, anchor={anchor_cost:.2e}")
        
        stats = {
            'stages': stage_stats,
            'total_iters': total_iters,
            'lambdas': lambdas.tolist(),
        }
        
        return x0_opt, stats
    
    def _auto_tune_prior_weight(
        self,
        x_current: np.ndarray,
        controls: np.ndarray,
        x_prior_opt: np.ndarray,
        anchor_cost_threshold: float = 1e-6,
        max_lambda: float = 1.0,
    ) -> float:
        """
        Automatically tune prior weight using binary search.
        
        Finds the largest λ where anchor_cost < threshold.
        This gives maximum regularization while maintaining dynamics consistency.
        
        Args:
            x_current: Anchor state
            controls: Controls array
            x_prior_opt: Prior state (scaled, opt indices)
            anchor_cost_threshold: Maximum acceptable anchor error
            max_lambda: Upper bound for λ search
            
        Returns:
            Optimal λ value
        """
        # Binary search for optimal λ
        lambda_low = 0.0
        lambda_high = max_lambda
        best_lambda = 0.0
        
        for _ in range(15):  # ~15 iterations gives precision of max_lambda/32768
            lambda_mid = (lambda_low + lambda_high) / 2
            
            # Quick optimization with this λ
            x0_opt = x_current[self.opt_indices] / self.state_scale[self.opt_indices]
            
            result = minimize(
                lambda x: self._shooting_objective(x, x_current, controls, x_prior_opt, lambda_mid),
                x0_opt,
                method='L-BFGS-B',
                jac=lambda x: self._shooting_gradient(x, x_current, controls, x_prior_opt, lambda_mid),
                options={'maxiter': 50, 'ftol': 1e-8, 'gtol': 1e-8}
            )
            
            # Check anchor cost (without prior term)
            anchor_cost = self._shooting_objective(result.x, x_current, controls, None, 0.0)
            
            if anchor_cost < anchor_cost_threshold:
                # This λ is acceptable, try higher
                best_lambda = lambda_mid
                lambda_low = lambda_mid
            else:
                # λ too high, anchor violated
                lambda_high = lambda_mid
        
        return best_lambda
    
    def predict_backward_shooting(
        self,
        x_current: np.ndarray,
        controls: np.ndarray,
        x_init: Optional[np.ndarray] = None,
        x_prior: Optional[np.ndarray] = None,
        prior_weight: float = 0.0,
        traj_ref: Optional[np.ndarray] = None,
        traj_weight: float = 0.0,
        continuation: bool = False,
        continuation_stages: int = 5,
        spread_anchor_error: bool = False,
        auto_tune_prior: bool = False,
        anchor_cost_threshold: float = 1e-6,
    ) -> Tuple[np.ndarray, np.ndarray, dict]:
        """
        Predict backward trajectory using SINGLE-SHOOTING method with optional regularization.
        
        Optimizes only x(t-H), then rolls forward to verify.
        
        cost = anchor_error 
             + prior_weight * ||x_oldest - x_prior||²
             + traj_weight * Σ_h ||x_h - traj_ref_h||²
        
        Args:
            x_current: Anchor state (current/newest) [state_dim]
            controls: [H, 2] controls, oldest to newest
            x_init: Optional initial guess [H, state_dim] - only uses oldest state
            x_prior: Prior state for oldest only [state_dim]. Options:
                     - None: no regularization
                     - 'stationary': car at rest (zeros for velocities)
                     - np.ndarray: specific state to bias towards
            prior_weight: Weight for oldest-state prior term.
            traj_ref: Reference trajectory [H, state_dim] to regularize entire trajectory towards.
                      This keeps the solution close to a reference at ALL timesteps.
            traj_weight: Weight for trajectory regularization (per-step).
            continuation: Use continuation method (gradually reduce regularization).
            continuation_stages: Number of stages for continuation (default 5).
            spread_anchor_error: If True, spread anchor error across trajectory via linear
                                 interpolation. Makes endpoint match anchor exactly, but
                                 introduces small dynamics violation at each step.
            auto_tune_prior: If True, automatically find optimal λ using binary search.
            anchor_cost_threshold: Max anchor error for auto-tuning (default 1e-6).
            
        Returns:
            past_states: [H, state_dim], oldest to newest (NOT including anchor)
            converged: [H] bool array
            stats: dict with regularization info
        """
        t_start = time.time()
        
        x_current = np.asarray(x_current, dtype=np.float32).flatten()
        x_current = _fix_sin_cos(x_current)
        controls = np.asarray(controls, dtype=np.float32)
        H = len(controls)
        
        # Initial guess for oldest state
        if x_init is not None:
            x0_guess = np.asarray(x_init[0], dtype=np.float32)
        else:
            # Use kinematic backward rollout for better initial guess
            x0_guess = self._kinematic_backward_rollout(x_current, controls)
        
        x0_opt = x0_guess[self.opt_indices] / self.state_scale[self.opt_indices]
        
        # Process prior state (oldest only)
        x_prior_opt = None
        x_prior_state = None
        if x_prior is not None:
            if isinstance(x_prior, str) and x_prior == 'stationary':
                x_prior_state = x_current.copy()
                x_prior_state[0] = 0.0  # angular_vel_z
                x_prior_state[1] = 0.0  # linear_vel_x
                x_prior_state[2] = 0.0  # linear_vel_y
                x_prior_state[8] = 0.0  # slip_angle
            else:
                x_prior_state = np.asarray(x_prior, dtype=np.float32).flatten()
            
            x_prior_opt = x_prior_state[self.opt_indices] / self.state_scale[self.opt_indices]
        
        # Process trajectory reference
        traj_ref_arr = None
        if traj_ref is not None and (traj_weight > 0 or continuation):
            traj_ref_arr = np.asarray(traj_ref, dtype=np.float32)
            if traj_ref_arr.shape[0] != H:
                raise ValueError(f"traj_ref must have shape [H, state_dim], got {traj_ref_arr.shape}")
        
        # Auto-tune prior weight if requested
        if auto_tune_prior and x_prior_opt is not None:
            prior_weight = self._auto_tune_prior_weight(
                x_current, controls, x_prior_opt, anchor_cost_threshold
            )
            if self.verbose:
                print(f"[Auto-tune] Found optimal λ = {prior_weight:.6f}")
        
        # Use continuation method if requested
        continuation_stats = None
        if continuation and traj_ref_arr is not None:
            if self.verbose:
                print(f"[Continuation] Starting with {continuation_stages} stages...")
            
            x0_opt, continuation_stats = self._continuation_optimization(
                x_current, controls, traj_ref_arr,
                lambda_start=1.0,
                lambda_end=0.0,
                n_stages=continuation_stages,
                max_iter_per_stage=50,
            )
            # Final optimization with no regularization (just anchor)
            traj_weight = 0.0
        
        # Shooting optimization with regularization
        result = minimize(
            lambda x: self._shooting_objective(x, x_current, controls, x_prior_opt, prior_weight, traj_ref_arr, traj_weight),
            x0_opt,
            method='L-BFGS-B',
            jac=lambda x: self._shooting_gradient(x, x_current, controls, x_prior_opt, prior_weight, traj_ref_arr, traj_weight),
            options={'maxiter': self.max_iter, 'ftol': self.tol, 'gtol': self.tol}
        )
        
        # Reconstruct optimal oldest state
        x_oldest = np.zeros(STATE_DIM, dtype=np.float32)
        x_oldest[self.opt_indices] = result.x * self.state_scale[self.opt_indices]
        x_oldest = _fix_sin_cos(x_oldest)
        
        # Roll forward to get full trajectory
        traj = self.forward_rollout(x_oldest, controls)
        
        # Check anchor error before any spreading
        x_t_predicted = traj[-1]
        anchor_error_vec = x_current - x_t_predicted
        anchor_cost = self._shooting_objective(result.x, x_current, controls, None, 0.0, None, 0.0)
        
        # Optionally spread anchor error across trajectory
        if spread_anchor_error and np.linalg.norm(anchor_error_vec) > 1e-8:
            # Linear interpolation: fraction goes from 0 at h=0 to 1 at h=H
            for h in range(H + 1):
                fraction = h / H
                traj[h] += fraction * anchor_error_vec
            
            # Fix sin/cos consistency after spreading
            traj[:, POSE_THETA_SIN_IDX] = np.sin(traj[:, POSE_THETA_IDX])
            traj[:, POSE_THETA_COS_IDX] = np.cos(traj[:, POSE_THETA_IDX])
        
        # Past states are traj[0:H] (excluding the final anchor position)
        past_states = traj[:-1].copy()
        
        elapsed = time.time() - t_start
        
        converged = np.full(H, result.success and anchor_cost < 0.01, dtype=bool)
        
        total_iters = result.nit
        if continuation_stats:
            total_iters += continuation_stats['total_iters']
        
        method_str = 'shooting'
        if continuation:
            method_str += '+continuation'
        if spread_anchor_error:
            method_str += '+spread'
        
        self.stats = {
            'time_ms': elapsed * 1000,
            'niter': total_iters,
            'final_cost': result.fun,
            'anchor_cost': anchor_cost,
            'anchor_cost_after_spread': 0.0 if spread_anchor_error else anchor_cost,
            'success': result.success,
            'method': method_str,
            'prior_weight': prior_weight,
            'traj_weight': traj_weight,
            'continuation': continuation_stats,
            'spread_anchor_error': spread_anchor_error,
        }
        
        if self.verbose:
            reg_str = ""
            if prior_weight > 0:
                reg_str += f", prior_λ={prior_weight}"
            if traj_weight > 0:
                reg_str += f", traj_λ={traj_weight}"
            if continuation:
                reg_str += f", continuation={continuation_stages}stages"
            if spread_anchor_error:
                reg_str += ", spread"
            print(f"[Shooting] H={H}, iters={total_iters}, cost={result.fun:.6f}, "
                  f"anchor={anchor_cost:.6f}{reg_str}, time={elapsed*1000:.1f}ms")
        
        return past_states, converged, self.stats
    
    def predict_backward_sequential(
        self,
        x_current: np.ndarray,
        controls: np.ndarray,
        x_init: Optional[np.ndarray] = None,
        max_iter_per_step: int = 20,
    ) -> Tuple[np.ndarray, np.ndarray, dict]:
        """
        Predict backward trajectory using SEQUENTIAL per-step method.
        
        Solves H independent single-step problems, each with 1-step rollout.
        Much faster than shooting for long horizons.
        
        For each step h:
            Find x(t-h) such that step(x(t-h), u(t-h)) = x(t-h+1)
        
        Args:
            x_current: Anchor state (current/newest) [state_dim]
            controls: [H, 2] controls, oldest to newest
            x_init: Optional initial guess [H, state_dim]
            max_iter_per_step: Max iterations per single-step solve
            
        Returns:
            past_states: [H, state_dim], oldest to newest
            converged: [H] bool array
            stats: dict
        """
        t_start = time.time()
        
        x_current = np.asarray(x_current, dtype=np.float32).flatten()
        x_current = _fix_sin_cos(x_current)
        controls = np.asarray(controls, dtype=np.float32)
        H = len(controls)
        
        past_states = np.zeros((H, STATE_DIM), dtype=np.float32)
        converged = np.zeros(H, dtype=bool)
        total_iters = 0
        
        # Work backwards from anchor
        x = x_current
        for h in range(H):
            # Control for step from x(t-h-1) to x(t-h)
            u = controls[H - 1 - h]  # newest to oldest
            
            # Initial guess: use provided init or previous solution
            if x_init is not None:
                x_guess = x_init[H - 1 - h]
            else:
                x_guess = x  # Previous refined state as guess
            
            # Single-step optimization
            x_refined, conv, stats = self.single_step_backward(
                x, u, x_init=x_guess, max_iter=max_iter_per_step, tol=self.tol
            )
            
            past_states[H - 1 - h] = x_refined
            converged[H - 1 - h] = conv
            total_iters += stats['niter']
            
            # Use refined state for next step
            x = x_refined
        
        elapsed = time.time() - t_start
        
        self.stats = {
            'time_ms': elapsed * 1000,
            'niter': total_iters,
            'avg_iter_per_step': total_iters / H,
            'success': np.all(converged),
            'n_failed': int(np.sum(~converged)),
        }
        
        if self.verbose:
            print(f"[Sequential] H={H}, iters={total_iters} ({total_iters/H:.1f}/step), "
                  f"time={elapsed*1000:.1f}ms")
        
        return past_states, converged, self.stats
    
    def predict_backward_growing(
        self,
        x_current: np.ndarray,
        controls: np.ndarray,
        max_iter_per_horizon: int = 20,
        n_warmup_steps: int = 5,
    ) -> Tuple[np.ndarray, np.ndarray, dict]:
        """
        Predict backward trajectory using SPARSE GROWING + FINAL SHOOTING.
        
        Uses a few warmup shooting problems to build a good initial guess,
        then does one final full-horizon shooting.
        
        Warmup horizons: [1, 2, 4, 8, ..., H/2] (logarithmic growth)
        Final: Full H-step shooting with excellent initial guess
        
        Args:
            x_current: Anchor state (current/newest) [state_dim]
            controls: [H, 2] controls, oldest to newest
            max_iter_per_horizon: Max iterations per shooting sub-problem
            n_warmup_steps: Number of warmup horizons (default 5)
            
        Returns:
            past_states: [H, state_dim], oldest to newest
            converged: [H] bool array
            stats: dict
        """
        t_start = time.time()
        
        x_current = np.asarray(x_current, dtype=np.float32).flatten()
        x_current = _fix_sin_cos(x_current)
        controls = np.asarray(controls, dtype=np.float32)
        H = len(controls)
        
        total_iters = 0
        
        # Build sparse warmup horizons: [1, 2, 4, 8, ..., H/2]
        warmup_horizons = []
        h = 1
        while h < H and len(warmup_horizons) < n_warmup_steps:
            warmup_horizons.append(h)
            h *= 2
        
        # Start with anchor as initial guess
        x_oldest_guess = x_current.copy()
        
        # Warmup phase: sparse growing
        for h in warmup_horizons:
            sub_controls = controls[H - h:]
            x0_opt = x_oldest_guess[self.opt_indices] / self.state_scale[self.opt_indices]
            
            result = minimize(
                lambda x: self._shooting_objective(x, x_current, sub_controls),
                x0_opt,
                method='L-BFGS-B',
                jac=lambda x: self._shooting_gradient(x, x_current, sub_controls),
                options={'maxiter': max_iter_per_horizon, 'ftol': self.tol, 'gtol': self.tol}
            )
            
            total_iters += result.nit
            
            x_oldest = np.zeros(STATE_DIM, dtype=np.float32)
            x_oldest[self.opt_indices] = result.x * self.state_scale[self.opt_indices]
            x_oldest = _fix_sin_cos(x_oldest)
            x_oldest_guess = x_oldest
        
        # Final phase: full H-step shooting with warm start
        x0_opt = x_oldest_guess[self.opt_indices] / self.state_scale[self.opt_indices]
        
        result = minimize(
            lambda x: self._shooting_objective(x, x_current, controls),
            x0_opt,
            method='L-BFGS-B',
            jac=lambda x: self._shooting_gradient(x, x_current, controls),
            options={'maxiter': self.max_iter, 'ftol': self.tol, 'gtol': self.tol}
        )
        
        total_iters += result.nit
        
        # Reconstruct optimal oldest state
        x_oldest = np.zeros(STATE_DIM, dtype=np.float32)
        x_oldest[self.opt_indices] = result.x * self.state_scale[self.opt_indices]
        x_oldest = _fix_sin_cos(x_oldest)
        
        # Roll forward to get full trajectory
        traj = self.forward_rollout(x_oldest, controls)
        past_states = traj[:-1].copy()
        
        elapsed = time.time() - t_start
        
        converged = np.full(H, result.success and result.fun < 0.01, dtype=bool)
        
        self.stats = {
            'time_ms': elapsed * 1000,
            'niter': total_iters,
            'warmup_horizons': warmup_horizons,
            'final_cost': result.fun,
            'success': result.success,
            'method': 'growing',
        }
        
        if self.verbose:
            print(f"[Growing] H={H}, warmup={warmup_horizons}, iters={total_iters}, "
                  f"cost={result.fun:.6f}, time={elapsed*1000:.1f}ms")
        
        return past_states, converged, self.stats
    
    def predict_backward(
        self,
        x_current: np.ndarray,
        controls: np.ndarray,
        x_init: Optional[np.ndarray] = None,
        method: str = 'shooting',
        traj_ref: Optional[np.ndarray] = None,
        traj_weight: float = 0.0,
        continuation: bool = False,
        continuation_stages: int = 5,
        spread_anchor_error: bool = False,
    ) -> Tuple[np.ndarray, np.ndarray, dict]:
        """
        Predict backward trajectory.
        
        Args:
            x_current: Anchor state (current/newest) [state_dim]
            controls: [H, 2] controls, oldest to newest
            x_init: Optional initial guess [H, state_dim]
            method: 'shooting' (recommended), 'sequential', or 'growing'
            traj_ref: Reference trajectory [H, state_dim] to regularize towards
            traj_weight: Weight for trajectory regularization
            continuation: Use continuation method (gradually reduce regularization)
            continuation_stages: Number of stages for continuation
            spread_anchor_error: Spread anchor error across trajectory
            
        Returns:
            past_states: [H, state_dim], oldest to newest
            converged: [H] bool array
            stats: dict
        """
        if method == 'shooting':
            return self.predict_backward_shooting(
                x_current, controls, x_init,
                traj_ref=traj_ref,
                traj_weight=traj_weight,
                continuation=continuation,
                continuation_stages=continuation_stages,
                spread_anchor_error=spread_anchor_error,
            )
        elif method == 'sequential':
            return self.predict_backward_sequential(x_current, controls, x_init)
        else:  # 'growing'
            return self.predict_backward_growing(x_current, controls)


class FastBackwardPredictor:
    """
    Predictor API wrapper for backward trajectory optimization.
    
    Configurable parameters for trading speed vs precision:
    - max_iter: For shooting method (optimizer mode)
    - max_iter_per_step: For hybrid mode's per-step refinement
    - tol: Convergence tolerance (lower = more precise but slower)
    """
    
    def __init__(
        self,
        dt: float = 0.01,
        mu: Optional[float] = None,
        max_iter: int = 200,
        max_iter_per_step: int = 10,
        tol: float = 1e-7,
        verbose: bool = False,
        **kwargs
    ):
        self.optimizer = FastBackwardOptimizer(
            dt=dt,
            mu=mu,
            max_iter=max_iter,
            tol=tol,
            verbose=verbose,
        )
        self.dt = dt
        self.max_iter_per_step = max_iter_per_step
        self.tol = tol
        self._last_stats = {}
    
    def configure(self, batch_size: int = 1, dt: float = None, **kwargs):
        if dt is not None:
            self.dt = abs(dt)
            self.optimizer.dt = abs(dt)
    
    def set_mu(self, mu: float):
        self.optimizer.set_mu(mu)
    
    def predict(
        self,
        initial_state: np.ndarray,
        controls: np.ndarray,
        X_init: Optional[np.ndarray] = None,
        traj_ref: Optional[np.ndarray] = None,
        continuation: bool = False,
        continuation_stages: int = 5,
        spread_anchor_error: bool = False,
    ) -> Optional[np.ndarray]:
        """
        Predict using shooting method (full trajectory optimization).
        
        Args:
            initial_state: Anchor state [state_dim]
            controls: [H, 2] controls, oldest to newest
            X_init: Optional initial guess [H, state_dim]
            traj_ref: Reference trajectory [H, state_dim] to stay close to
            continuation: Use continuation method (gradually reduce regularization)
            continuation_stages: Number of stages for continuation
            spread_anchor_error: Spread anchor error across trajectory
            
        Returns:
            past_states: [H, state_dim], oldest to newest
        """
        initial_state = np.asarray(initial_state, dtype=np.float32)
        if initial_state.ndim == 2:
            initial_state = initial_state.flatten()
        
        controls = np.asarray(controls, dtype=np.float32)
        
        past_states, converged, stats = self.optimizer.predict_backward(
            initial_state, controls, X_init,
            method='shooting',
            traj_ref=traj_ref,
            continuation=continuation,
            continuation_stages=continuation_stages,
            spread_anchor_error=spread_anchor_error,
        )
        
        self._last_stats = stats
        return past_states
    
    def predict_interleaved(
        self,
        initial_state: np.ndarray,
        controls: np.ndarray,
        nn_step_fn: Callable[[np.ndarray, np.ndarray], np.ndarray],
    ) -> Optional[np.ndarray]:
        """
        Predict using interleaved NN + optimizer (one step at a time).
        
        Args:
            initial_state: Current/anchor state [state_dim]
            controls: [H, 2] controls, oldest to newest
            nn_step_fn: Function (x_current, u) -> x_prev_estimate from neural network
            
        Returns:
            past_states: [H, state_dim], oldest to newest
        """
        initial_state = np.asarray(initial_state, dtype=np.float32)
        if initial_state.ndim == 2:
            initial_state = initial_state.flatten()
        
        controls = np.asarray(controls, dtype=np.float32)
        
        past_states, converged, stats = self.optimizer.predict_interleaved(
            initial_state, controls, nn_step_fn, max_iter_per_step=self.max_iter_per_step
        )
        
        self._last_stats = stats
        return past_states
    
    @property
    def last_stats(self) -> dict:
        return self._last_stats
