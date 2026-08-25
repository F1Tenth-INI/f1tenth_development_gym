#!/usr/bin/env python3
"""Offline FineSafeEllipsoid computation (Tearle et al. Sec. V).

Usage::

    python -m utilities.mpc_safety_filter.compute_fine_safe_ellipsoid
"""

from __future__ import annotations

from utilities.Settings import Settings
from utilities.car_files.vehicle_parameters import VehicleParameters
from utilities import mpc_safety_filter as mpc_sf
from .psf_terminal_set import compute_fine_safe_ellipsoid, save_fine_safe_ellipsoid


def main() -> None:
    veh = VehicleParameters(Settings.CONTROLLER_CAR_PARAMETER_FILE)
    params = veh.to_np_array().astype(float)
    path = str(mpc_sf.ELLIPSOID_PATH)
    ell = compute_fine_safe_ellipsoid(
        params,
        v_x=float(mpc_sf.VX_SS),
        c_max=float(mpc_sf.C_MAX),
        n_curvature=int(mpc_sf.N_CURVATURE),
        dt=float(mpc_sf.DT),
        t_half=float(mpc_sf.T_HALF),
        verify_restarts=20,
        max_shrink_iters=12,
    )
    save_fine_safe_ellipsoid(path, ell)
    print(
        f"Saved FineSafeEllipsoid to {path}\n"
        f"  verified={ell.verified_objective:.4f}  scale={ell.scale:.4e}  "
        f"v_x={ell.v_x:.3f}  c_max={ell.c_grid[-1]:.3f}"
    )


if __name__ == "__main__":
    main()
