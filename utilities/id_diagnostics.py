# utilities/id_diagnostics.py
import os
import math
import numpy as np

from typing import Dict, Optional, Tuple

from utilities.InverseDynamics import (
    HARD_CODED_STATE_STD,
    KEEP_IDX,
    POSE_THETA_IDX,
    SLIP_ANGLE_IDX,
    POSE_THETA_SIN_IDX,
    POSE_THETA_COS_IDX,
    _angle_wrap_diff as _tf_angle_wrap_diff,   # tf version
)

# --- small NumPy helpers (no TF needed for metrics) -----------------------------------

# Module-level caches
_SWEEP_REFINER_CACHE = {}   # key -> ProgressiveWindowRefiner
_REF_WARMED = set()         # keys warmed once


def _angle_wrap_diff_np(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.arctan2(np.sin(a - b), np.cos(a - b))

def _get_sweep_refiner(dt: float, mu: float | None, pid: bool, W: int, O: int, prior_kind: str | None):
    key = (float(dt), None if mu is None else float(mu), bool(pid), int(W), int(O), str(prior_kind or ""))
    r = _SWEEP_REFINER_CACHE.get(key)
    if r is None:
        from utilities.InverseDynamics import ProgressiveWindowRefiner
        r = ProgressiveWindowRefiner(
            mu=mu, controls_are_pid=pid, dt=dt,
            window_size=W, overlap=O,
            smoothing_window=None, smoothing_overlap=None,
            prior_kind=prior_kind,   # pass through so cache separates by prior
        )
        _SWEEP_REFINER_CACHE[key] = r
    # one-time warm to prevent retracing/compile on every call
    if key not in _REF_WARMED:
        import numpy as np
        xT = np.zeros((1, r.refiner.state_dim), np.float32)
        Qs = np.zeros((min(5, r.refiner.MAX_T), r.refiner.control_dim), np.float32)
        r.refiner.inverse_entire_trajectory(xT, Qs, X_init=np.zeros((Qs.shape[0], r.refiner.state_dim), np.float32))
        _REF_WARMED.add(key)
    return r

def _scaled_rmse(pred: np.ndarray, gt: np.ndarray) -> float:
    keep = KEEP_IDX.numpy() if hasattr(KEEP_IDX, "numpy") else KEEP_IDX
    p, g = pred.copy(), gt.copy()
    p[:, POSE_THETA_IDX] = np.unwrap(p[:, POSE_THETA_IDX])
    g[:, POSE_THETA_IDX] = np.unwrap(g[:, POSE_THETA_IDX])
    dif = p - g
    dif[:, POSE_THETA_IDX] = _angle_wrap_diff_np(p[:, POSE_THETA_IDX], g[:, POSE_THETA_IDX])
    scale = np.maximum(HARD_CODED_STATE_STD.astype(np.float32), 1e-12)[keep]
    Z = dif[:, keep] / scale[None, :]
    mask = np.isfinite(Z).all(axis=1)
    if not np.any(mask): return float("nan")
    Z = Z[mask]
    return float(np.sqrt(np.mean(Z * Z)))

def _scaled_rmse_curve(pred: np.ndarray, gt: np.ndarray) -> np.ndarray:
    # per-step scaled RMSE over kept dims
    keep = KEEP_IDX.numpy() if hasattr(KEEP_IDX, "numpy") else KEEP_IDX
    p, g = pred.copy(), gt.copy()
    p[:, POSE_THETA_IDX] = np.unwrap(p[:, POSE_THETA_IDX])
    g[:, POSE_THETA_IDX] = np.unwrap(g[:, POSE_THETA_IDX])
    dif = p - g
    dif[:, POSE_THETA_IDX] = _angle_wrap_diff_np(p[:, POSE_THETA_IDX], g[:, POSE_THETA_IDX])
    scale = np.maximum(HARD_CODED_STATE_STD.astype(np.float32), 1e-12)[keep]
    Z = dif[:, keep] / scale[None, :]
    rmse = np.full((Z.shape[0],), np.nan, np.float32)
    rows = np.isfinite(Z).all(axis=1)
    if np.any(rows):
        rmse[rows] = np.sqrt(np.mean(Z[rows]*Z[rows], axis=1))
    return rmse

def _dim_names():
    # Order matches HARD_CODED_STATE_STD indices
    return [
        "omega_z", "vx", "vy", "psi",
        "sin_psi", "cos_psi", "x", "y", "beta", "delta"
    ]

# --------------------------------------------------------------------------------------
# 1) Time-shift / alignment probe
# --------------------------------------------------------------------------------------

def probe_time_shift(pred_past: np.ndarray, gt_past: np.ndarray, max_shift: int = 6) -> Dict[str, float]:
    """
    Tests small integer shifts between predicted and measured past to detect off-by-one
    timing between states and control holds. Returns best shift and improvement evidence.
    """
    out = {}
    L = min(pred_past.shape[0], gt_past.shape[0])
    p = pred_past[-L:]
    g = gt_past[-L:]

    rmse0 = _scaled_rmse(p, g)
    out["shift_rmse_0"] = rmse0

    best_rmse = rmse0
    best_k = 0
    for k in range(-max_shift, max_shift+1):
        if k == 0: continue
        if k < 0:
            # pred older vs gt newer
            rm = _scaled_rmse(p[:L+k], g[-(L+k):])
        else:
            # pred newer vs gt older
            rm = _scaled_rmse(p[k:], g[:L-k])
        if not math.isnan(rm) and rm < best_rmse:
            best_rmse = rm
            best_k = k

    out["shift_best_k"] = float(best_k)
    out["shift_best_rmse"] = float(best_rmse)
    out["shift_improvement"] = float(rmse0 - best_rmse) if (not math.isnan(rmse0) and not math.isnan(best_rmse)) else float("nan")
    # Interpretation hint
    #  k>0 (pred newer) → your predicted history is shifted forward: check ts or control sampling
    #  k<0 (pred older) → your predicted history lags: check 'last env tick' vs 'mean/first'
    return out

# --------------------------------------------------------------------------------------
# 2) Per-dimension residuals & scale audit
# --------------------------------------------------------------------------------------

def per_dim_residuals(pred_past: np.ndarray, gt_past: np.ndarray) -> Dict[str, float]:
    keep = KEEP_IDX.numpy() if hasattr(KEEP_IDX, "numpy") else KEEP_IDX
    p, g = pred_past.copy(), gt_past.copy()
    p[:, POSE_THETA_IDX] = np.unwrap(p[:, POSE_THETA_IDX])
    g[:, POSE_THETA_IDX] = np.unwrap(g[:, POSE_THETA_IDX])
    dif = p - g
    dif[:, POSE_THETA_IDX] = _angle_wrap_diff_np(p[:, POSE_THETA_IDX], g[:, POSE_THETA_IDX])
    scale = np.maximum(HARD_CODED_STATE_STD.astype(np.float32), 1e-12)
    Z = dif[:, keep] / scale[keep][None, :]
    mask = np.isfinite(Z).all(axis=1)
    out = {}
    if np.any(mask):
        rms_dim = np.sqrt(np.mean(Z[mask]*Z[mask], axis=0))
        names = _dim_names()
        kept_names = [names[i] for i in keep]
        for n, v in zip(kept_names, rms_dim):
            out[f"dim_rmse_{n}"] = float(v)
    out["dim_scale_min"] = float(np.min(scale[keep]))
    out["dim_scale_max"] = float(np.max(scale[keep]))
    return out

# --------------------------------------------------------------------------------------
# 3) Control-hold aggregation mismatch
# --------------------------------------------------------------------------------------

def _compress(controls_env: np.ndarray, ts: int, mode: str) -> np.ndarray:
    # controls_env: [H*ts, 2], oldest->newest
    H = controls_env.shape[0] // ts
    blocks = controls_env.reshape(H, ts, -1)
    if mode == "last":   return blocks[:, -1, :]
    if mode == "first":  return blocks[:,  0, :]
    if mode == "mean":   return np.mean(blocks, axis=1)
    if mode == "median": return np.median(blocks, axis=1)
    raise ValueError("mode must be last/first/mean/median")

def compare_control_aggregations(refiner_prog, x_next: np.ndarray, controls_env: np.ndarray, ts: int,
                                 gt_past: Optional[np.ndarray]) -> Dict[str, float]:
    """
    Re-run the inverse solver with several compressions of env-rate controls to controller-rate.
    Reports which aggregation gives lower RMSE (if GT given) or better dynamics residual.
    """
    out = {}
    modes = ["last", "mean", "first", "median"]
    for m in modes:
        Q = _compress(controls_env, ts, m).astype(np.float32)  # [H,2]
        states, flags, st = refiner_prog.refine_stats(x_next, Q)
        # sample at controller boundaries is already in this cadence; states[1:][::-1] oldest->newest
        pred_past = states[1:][::-1]
        if gt_past is not None and pred_past.shape == gt_past.shape:
            out[f"agg_{m}_rmse"] = _scaled_rmse(pred_past, gt_past)
        out[f"agg_{m}_conv_rate"] = float(np.mean(flags.astype(np.float32)))
        out[f"agg_{m}_inv_ms"] = float(st.get("inv_ms", float("nan")))
    # A large gap between 'last' and 'mean' strongly hints your controller-hold sampling is off.
    return out

# --------------------------------------------------------------------------------------
# 4) dt / rounding mismatch check
# --------------------------------------------------------------------------------------

def dt_alignment_check(dt_ctrl: float, dt_env: float, ts: int) -> Dict[str, float]:
    ideal = dt_env * ts
    out = {
        "dt_ctrl": float(dt_ctrl),
        "dt_env": float(dt_env),
        "ts": float(ts),
        "dt_env_x_ts": float(ideal),
        "dt_ctrl_minus_env_ts": float(dt_ctrl - ideal),
        "dt_rel_error_pct": float(100.0 * (dt_ctrl - ideal) / max(ideal, 1e-12)),
    }
    # If |dt_rel_error_pct| > ~2-3%, expect a persistent timing drift across history.
    return out

# --------------------------------------------------------------------------------------
# 5) Prior influence audit
# --------------------------------------------------------------------------------------

def prior_influence_audit(refiner_prog, gt_past: np.ndarray, out_stride_ctrl: int) -> Dict[str, float]:
    """
    Compares the enforced prior vs refined solution against ground truth at controller cadence.
    """
    out = {}
    prior_all = getattr(refiner_prog, "_last_prior_newest_first", None)  # [T+1,10] newest->older
    if not isinstance(prior_all, np.ndarray):
        out["prior_audit_has_prior"] = float(0.0)
        return out

    out["prior_audit_has_prior"] = float(1.0)

    # Controller boundaries sampling and chronological flip
    prior_at_ctrl = prior_all[::out_stride_ctrl, :]
    prior_past = prior_at_ctrl[1:][::-1, :]
    if prior_past.shape != gt_past.shape:
        return out

    rmse_prior = _scaled_rmse(prior_past, gt_past)
    out["prior_rmse"] = rmse_prior

    # Also publish per-step improvement (AUC)
    curve_prior = _scaled_rmse_curve(prior_past, gt_past)
    out["prior_auc"] = float(np.trapz(np.nan_to_num(curve_prior, nan=0.0), dx=1.0))
    return out

# --------------------------------------------------------------------------------------
# 6) Dynamics consistency residual (forward model check)
# --------------------------------------------------------------------------------------

def dynamics_consistency(states_newest_first: np.ndarray, Q_old2new: np.ndarray, refiner_prog) -> Dict[str, float]:
    """
    Roll forward the car model over the returned states, measure robust dynamics cost per step.
    Matches your flag logic but returns summary stats for the whole horizon.
    """
    out = {}
    try:
        f = refiner_prog.refiner._f
        import tensorflow as tf
        from utilities.InverseDynamics import _partial_state_huber
        state_scale = refiner_prog.refiner.state_scale

        T = Q_old2new.shape[0]
        costs = []
        Q_newest_first = Q_old2new[::-1, :]
        for i in range(T):
            x_i = states_newest_first[0] if i == 0 else states_newest_first[i]
            x_im1 = states_newest_first[i+1]
            qi = Q_newest_first[i]
            x_pred = f(tf.convert_to_tensor(x_im1, tf.float32)[tf.newaxis, :],
                       tf.convert_to_tensor(qi, tf.float32)[tf.newaxis, :])[0]
            # robust cost
            c = _partial_state_huber(x_pred, tf.convert_to_tensor(x_i, tf.float32),
                                     delta=refiner_prog.refiner.RES_DELTA,
                                     state_scale=state_scale,
                                     slip_lambda=refiner_prog.refiner.SLIP_PRIOR)
            costs.append(float(c.numpy()))
        arr = np.asarray(costs, np.float32)
        out["dyn_cost_mean"] = float(np.mean(arr))
        out["dyn_cost_max"] = float(np.max(arr))
        out["dyn_cost_p95"] = float(np.percentile(arr, 95.0))
    except Exception:
        pass
    return out

# --------------------------------------------------------------------------------------
# 7) Window & overlap sweep (coarse; opt-in via env)
# --------------------------------------------------------------------------------------

def window_overlap_sweep(x_next: np.ndarray, Q_ctrl: np.ndarray, refiner_prog, gt_past: Optional[np.ndarray]) -> Dict[str, float]:
    results = {}
    Ws = [10, 15, 20, 30]
    Os = [5, 10, 15]
    dt = refiner_prog.dt
    mu = getattr(refiner_prog.refiner.car_model.car_parameters, "mu", None)
    prior_kind = getattr(refiner_prog, "prior_kind", None)

    for W in Ws:
        for O in Os:
            if O >= W: continue
            r = _get_sweep_refiner(dt, mu, True, W, O, prior_kind)
            states, flags, _ = r.refine_stats(x_next, Q_ctrl)
            pred_past = states[1:][::-1]
            rm = _scaled_rmse(pred_past, gt_past) if (gt_past is not None and pred_past.shape==gt_past.shape) else float("nan")
            results[f"sweep_W{W}_O{O}_rmse"] = rm
            results[f"sweep_W{W}_O{O}_conv"] = float(np.mean(flags.astype(np.float32)))
    return results


# --------------------------------------------------------------------------------------
# 8) Model mismatch sweep (μ, PID/core; opt-in via env)
# --------------------------------------------------------------------------------------

def model_param_sweep(x_next: np.ndarray, Q_ctrl: np.ndarray, gt_past: Optional[np.ndarray]) -> Dict[str, float]:
    results = {}
    # keep grid small; you can expand later
    mus = [0.6, 0.8, 1.0, 1.2]
    pid_flags = [True, False]
    dt = float(getattr(x_next, "dt", 0.01))  # or pass dt in explicitly if you prefer
    prior_kind = "nn"
    for mu in mus:
        for pid in pid_flags:
            r = _get_sweep_refiner(dt, mu, pid, 15, 10, prior_kind)
            states, flags, _ = r.refine_stats(x_next, Q_ctrl)
            pred_past = states[1:][::-1]
            rm = _scaled_rmse(pred_past, gt_past) if (gt_past is not None and pred_past.shape==gt_past.shape) else float("nan")
            key = f"mu{mu:.2f}_{'pid' if pid else 'core'}"
            results[f"sweep_{key}_rmse"] = rm
            results[f"sweep_{key}_conv"] = float(np.mean(flags.astype(np.float32)))
    return results

# --------------------------------------------------------------------------------------
# 9) Heading wrap / sin-cos / slip sanity
# --------------------------------------------------------------------------------------

def angle_and_slip_sanity(pred_past: np.ndarray, gt_past: np.ndarray) -> Dict[str, float]:
    out = {}
    # sin^2 + cos^2 consistency for GT: if sensor feed is mis-wired or stale, this will flag it
    def _sincos_err(X):
        s2c2 = X[:, POSE_THETA_SIN_IDX]**2 + X[:, POSE_THETA_COS_IDX]**2
        return float(np.nanmax(np.abs(s2c2 - 1.0)))
    out["gt_sincos_max_err"] = _sincos_err(gt_past)
    out["pred_sincos_max_err"] = _sincos_err(pred_past)

    # Wrapped heading error distribution
    ang = _angle_wrap_diff_np(pred_past[:, POSE_THETA_IDX], gt_past[:, POSE_THETA_IDX])
    out["psi_err_mean_deg"] = float(np.rad2deg(np.nanmean(ang)))
    out["psi_err_p95_deg"]  = float(np.rad2deg(np.nanpercentile(ang, 95)))
    # Slip-angle observability: if slip not in KEEP_IDX, big drift may suggest you need stronger SLIP_PRIOR
    if SLIP_ANGLE_IDX is not None and SLIP_ANGLE_IDX >= 0:
        d_beta = pred_past[:, SLIP_ANGLE_IDX] - gt_past[:, SLIP_ANGLE_IDX]
        out["beta_err_rms"] = float(np.sqrt(np.nanmean(d_beta*d_beta)))
    return out

# --------------------------------------------------------------------------------------
# 10) Waypoint association & tangent alignment
# --------------------------------------------------------------------------------------

def waypoint_alignment_check(past_states: np.ndarray, waypoint_utils) -> Dict[str, float]:
    """
    Checks that headings align with local waypoint tangents and that the nearest waypoint
    indices move mostly monotonically. Large misalignments → wrong nearest-waypoint search window.
    """
    out = {}
    wps = waypoint_utils.waypoints  # [N,2] expected
    N = len(wps)
    if N < 3 or past_states.shape[0] == 0:
        return out

    # Recompute nearest waypoint indices for the chronological past
    idxs = []
    idx = waypoint_utils.nearest_waypoint_index
    for i in range(past_states.shape[0]-1, -1, -1):
        state = past_states[i]
        idx, _ = waypoint_utils.get_nearest_waypoint(
            state, wps, idx, lower_search_limit=-5, upper_search_limit=3
        )
        idxs.append(idx)
    idxs = np.array(idxs, dtype=int)

    # Tangent at each index
    def wp_tangent(i):
        i0 = i % N
        i1 = (i+1) % N
        v = wps[i1] - wps[i0]
        return math.atan2(v[1], v[0])

    tangents = np.array([wp_tangent(i) for i in idxs], dtype=np.float32)
    psi = past_states[:, POSE_THETA_IDX]
    mis = np.rad2deg(np.abs(_angle_wrap_diff_np(psi, tangents)))
    out["wp_heading_misalign_mean_deg"] = float(np.nanmean(mis))
    out["wp_heading_misalign_p95_deg"]  = float(np.nanpercentile(mis, 95))

    # Monotonicity of index progression (should be mostly non-decreasing on-track)
    dif = np.diff(idxs)
    out["wp_idx_nonmonotone_frac"] = float(np.mean(dif < -1))  # allow small back-and-forth; tweak threshold
    return out
