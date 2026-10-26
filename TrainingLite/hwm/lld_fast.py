"""LLD mean rollout with every constant resolved up front, so ``torch.compile`` can fuse it on CPU.

The generic helpers (``lld_rows_for_super``, ``lld_delta_scale``, ``reconstruct_next_state``) read settings and
numpy index constants while they run, which torch.compile traces wrongly or breaks on. Here the same arithmetic is
written with plain tensors and python ints, with the constants taken from those helpers once. Only the default
configuration is covered (no ``HWM_LLD_REFLECT``, no displacement channels); ``supported()`` says so.
"""

from __future__ import annotations

import torch

from TrainingLite.hwm.shared.observation import (
    _scale,
    lld_delta_scale,
    lld_direct_delta,
    lld_displacement_channels,
    lld_kept_state_index,
    lld_kinematic_residual,
    lld_reflect_on,
)
from utilities.Settings import Settings
from utilities.state_utilities import (
    ANGULAR_VEL_Z_IDX,
    LINEAR_VEL_X_IDX,
    LINEAR_VEL_Y_IDX,
    POSE_THETA_COS_IDX,
    POSE_THETA_IDX,
    POSE_THETA_SIN_IDX,
    POSE_X_IDX,
    POSE_Y_IDX,
    SLIP_ANGLE_IDX,
    STEERING_ANGLE_IDX,
)


def supported() -> bool:
    return not (lld_reflect_on() or lld_displacement_channels() or lld_kinematic_residual())


class FastLLDRollout:
    def __init__(self, lld: torch.nn.Module, compile_mode: str = "step") -> None:
        """``compile_mode``: ``"step"`` compiles one rollout step (python loop of 25 calls), ``"horizon"`` the
        whole unrolled loop, ``"none"`` leaves it eager (for checking)."""
        self.lld = lld
        self.compile_mode = compile_mode
        ref = torch.zeros(int(Settings.HWM_STATE_DIM))
        self.state_dim = int(Settings.HWM_STATE_DIM)
        self.dt = float(Settings.TIMESTEP_CONTROL)
        self.direct = bool(lld_direct_delta())
        self.scale = _scale(ref).clone()
        self.delta_scale = lld_delta_scale(ref).clone()
        keep = [int(i) for i in lld_kept_state_index(ref.device).tolist()]
        self.columns = torch.tensor(keep + [self.state_dim, self.state_dim + 1], dtype=torch.long)
        self.idx = {
            name: int(value)
            for name, value in dict(
                omega=ANGULAR_VEL_Z_IDX, vx=LINEAR_VEL_X_IDX, vy=LINEAR_VEL_Y_IDX, psi=POSE_THETA_IDX,
                cos=POSE_THETA_COS_IDX, sin=POSE_THETA_SIN_IDX, x=POSE_X_IDX, y=POSE_Y_IDX,
                slip=SLIP_ANGLE_IDX, steer=STEERING_ANGLE_IDX,
            ).items()
        }
        step = self._step
        if compile_mode in ("step", "horizon"):
            step = torch.compile(self._step, dynamic=False)
        self._compiled_step = step
        self._horizon = torch.compile(self._horizon_eager, dynamic=False) if compile_mode == "horizon" else None

    # -- one rollout step -------------------------------------------------------------------------------
    def _step(self, s, action, history, key_emb, tokens, density, valid):
        i = self.idx
        rows = torch.cat((s / self.scale, action), dim=-1).index_select(-1, self.columns)
        query = torch.cat((history.flatten(start_dim=1), rows), dim=-1)
        delta, n_soft, n_soft_norm = self.lld._mean_delta_core(query, key_emb, tokens, density, valid)

        raw = delta * self.delta_scale
        omega, vx, vy, steer = s[:, i["omega"]], s[:, i["vx"]], s[:, i["vy"]], s[:, i["steer"]]
        psi = s[:, i["psi"]]
        omega_n = omega + raw[:, 0]
        vx_n = vx + raw[:, 1]
        vy_n = vy + raw[:, 2]
        steer_n = steer + raw[:, 3]
        d_psi = 0.5 * (omega + omega_n) * self.dt if self.direct else raw[:, 4]
        psi_n = psi + d_psi
        psi_n = torch.atan2(torch.sin(psi_n), torch.cos(psi_n))
        mid = psi + 0.5 * d_psi
        cos_m, sin_m = torch.cos(mid), torch.sin(mid)
        ax, ay = 0.5 * (vx + vx_n), 0.5 * (vy + vy_n)
        x_n = s[:, i["x"]] + (ax * cos_m - ay * sin_m) * self.dt
        y_n = s[:, i["y"]] + (ax * sin_m + ay * cos_m) * self.dt
        vx_safe = torch.where(vx_n < 1.0e-3, torch.ones_like(vx_n) * 1.0e-3, vx_n)
        slip_n = torch.atan(vy_n / vx_safe)

        columns = [None] * self.state_dim
        columns[i["omega"]], columns[i["vx"]], columns[i["vy"]], columns[i["steer"]] = omega_n, vx_n, vy_n, steer_n
        columns[i["psi"]], columns[i["cos"]], columns[i["sin"]] = psi_n, torch.cos(psi_n), torch.sin(psi_n)
        columns[i["x"]], columns[i["y"]], columns[i["slip"]] = x_n, y_n, slip_n
        nxt = torch.stack(columns, dim=-1)
        history = torch.cat((history[:, 1:], rows.unsqueeze(1)), dim=1)
        return nxt, history, n_soft, n_soft_norm

    def _horizon_eager(self, s, actions, history, key_emb, tokens, density, valid):
        states, n_softs, norms = [s], [], []
        for k in range(int(actions.shape[1])):
            s, history, n_soft, n_soft_norm = self._step(s, actions[:, k], history, key_emb, tokens, density, valid)
            states.append(s)
            n_softs.append(n_soft)
            norms.append(n_soft_norm)
        return torch.stack(states, dim=1), torch.stack(n_softs, dim=1), torch.stack(norms, dim=1)

    @torch.no_grad()
    def run(self, s, actions, history, supers, deltas, density, valid):
        """``(states (B,H+1,D), n_soft (B,H), n_soft_norm (B,H))`` for ``actions (B,H,2)``."""
        key_emb = self.lld.attention_norm_projector(supers)
        tokens = deltas - self.lld._correct(supers)
        if self._horizon is not None:
            return self._horizon(s, actions, history, key_emb, tokens, density, valid)
        states, n_softs, norms = [s], [], []
        for k in range(int(actions.shape[1])):
            s, history, n_soft, n_soft_norm = self._compiled_step(
                s, actions[:, k], history, key_emb, tokens, density, valid
            )
            states.append(s)
            n_softs.append(n_soft)
            norms.append(n_soft_norm)
        return torch.stack(states, dim=1), torch.stack(n_softs, dim=1), torch.stack(norms, dim=1)
