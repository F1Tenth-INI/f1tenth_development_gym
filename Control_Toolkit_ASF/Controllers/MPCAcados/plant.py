"""Simulator-equivalent plant model and command maps.

Used to predict the state at which a new command takes effect (control delay),
to visualise plans and in the headless closed-loop test.  Rollouts call
``car_dynamics_pacejka_jax`` with the simulator ODE, so the friction ellipse,
steered single-track terms, CoG kinematic blend and ESC scales stay those of
the simulator.  The OCP in ``model.py`` is the same ODE without the ESC scales.
It is not part of the safety argument, which rests on the OCP alone.
"""

from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp
import numpy as np

from sim.f110_sim.envs.dynamic_model_pacejka_jax import car_dynamics_pacejka_jax
from utilities.Settings import Settings
from utilities.state_utilities import POSE_THETA_IDX, POSE_X_IDX, POSE_Y_IDX


@partial(jax.jit, static_argnames=("n_sub", "dt_sim", "ode_model"))
def _rollout_jax(car_params, state0, commands, n_sub, dt_sim, ode_model):
    def step(state, command):
        nxt = car_dynamics_pacejka_jax(state, command, car_params, dt_sim * n_sub, n_sub, ode_model=ode_model)
        return nxt, nxt

    _, traj = jax.lax.scan(step, state0, commands)
    return traj


class Plant:
    """Rollouts on gym states with raw simulator commands, exactly as the simulator integrates them."""

    def __init__(self, veh, dt_sim: float, n_sub: int):
        self.car_params = jnp.asarray(veh.to_np_array(), dtype=jnp.float32)
        self.dt_sim = float(dt_sim)
        self.n_sub = int(n_sub)
        self.ode_model = Settings.ODE_MODEL_OF_CAR_DYNAMICS

    def rollout(self, state0: np.ndarray, commands: np.ndarray, n_sub: int | None = None) -> np.ndarray:
        """Apply each raw command for ``n_sub`` simulator steps; returns the states after each command."""
        commands = np.asarray(commands, dtype=np.float32).reshape(-1, 2)
        if len(commands) == 0:
            return np.zeros((0, len(state0)), dtype=np.float32)
        traj = _rollout_jax(self.car_params, jnp.asarray(state0, dtype=jnp.float32), jnp.asarray(commands),
                            self.n_sub if n_sub is None else int(n_sub), self.dt_sim, self.ode_model)
        return np.asarray(traj)


# ------------------------------------------------------------------ geometry helpers
def border_clearance(track, veh, s, n, mu) -> np.ndarray:
    """Minimum corner clearance to either border (>= 0 inside the track)."""
    s, n, mu = (np.atleast_1d(np.asarray(a, dtype=np.float64)) for a in (s, n, mu))
    d_l, d_r = track.d_left_at(s), track.d_right_at(s)
    half_l, half_w = 0.5 * veh.length, 0.5 * veh.width
    clear = np.full(s.shape, np.inf)
    for cx in (half_l, -half_l):
        for cy in (half_w, -half_w):
            nc = n + cx * np.sin(mu) + cy * np.cos(mu)
            clear = np.minimum(clear, np.minimum(d_l - nc, d_r + nc))
    return clear


def gym_to_frenet(track, states: np.ndarray, s_hint: float | None):
    states = np.atleast_2d(states)
    return track.project(states[:, [POSE_X_IDX, POSE_Y_IDX]], states[:, POSE_THETA_IDX], s_hint,
                         window_ahead=20.0)
