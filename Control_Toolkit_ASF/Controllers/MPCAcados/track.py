"""Closed raceline geometry, speed profiles and Frenet projection for the safe MPC.

Heading and curvature are recomputed from the x/y polyline because the CSV
``psi`` column is not consistent with the waypoint geometry on every map.
The Frenet frame is attached to the raceline: ``n > 0`` is left, ``d_left`` /
``d_right`` are the border distances along the raceline normal.
"""

from __future__ import annotations

import hashlib

import numpy as np

from utilities.waypoint_utils import WP_D_LEFT_IDX, WP_D_RIGHT_IDX, WP_VX_IDX, WP_X_IDX, WP_Y_IDX


def wrap_angle(a):
    return (np.asarray(a) + np.pi) % (2.0 * np.pi) - np.pi


def _circular_smooth(values: np.ndarray, window: int) -> np.ndarray:
    if window <= 1:
        return values.copy()
    kernel = np.ones(window) / window
    pad = window // 2
    ext = np.concatenate([values[-pad:], values, values[:pad]])
    return np.convolve(ext, kernel, mode="valid")[: len(values)]


class RacelineTrack:
    def __init__(
        self,
        waypoints: np.ndarray,
        *,
        mu: float,
        g: float,
        lat_acc_factor_ref: float,
        kappa_smoothing_window: int,
    ):
        wps = np.asarray(waypoints, dtype=np.float64)
        xy = wps[:, [WP_X_IDX, WP_Y_IDX]]
        if np.linalg.norm(xy[-1] - xy[0]) < 1e-6:
            wps, xy = wps[:-1], xy[:-1]

        seg = np.roll(xy, -1, axis=0) - xy
        seg_len = np.linalg.norm(seg, axis=1)
        keep = seg_len > 1e-6
        wps, xy = wps[keep], xy[keep]
        seg = np.roll(xy, -1, axis=0) - xy
        seg_len = np.linalg.norm(seg, axis=1)

        self.xy = xy
        self.n_points = len(xy)
        self.seg = seg
        self.seg_len = seg_len
        self.seg_heading = np.arctan2(seg[:, 1], seg[:, 0])
        self.s = np.concatenate([[0.0], np.cumsum(seg_len[:-1])])
        self.length = float(np.sum(seg_len))

        # Vertex tangent = mean of adjacent segment headings; curvature = d(heading)/ds.
        prev_heading = np.roll(self.seg_heading, 1)
        dpsi = wrap_angle(self.seg_heading - prev_heading)
        self.tangent = wrap_angle(prev_heading + 0.5 * dpsi)
        kappa_raw = dpsi / (0.5 * (seg_len + np.roll(seg_len, 1)))
        self.kappa = _circular_smooth(kappa_raw, kappa_smoothing_window)

        self.d_left = wps[:, WP_D_LEFT_IDX].copy()
        self.d_right = wps[:, WP_D_RIGHT_IDX].copy()
        self.v_wp = np.maximum(wps[:, WP_VX_IDX], 0.0)

        mu_g = mu * g
        abs_k = np.maximum(np.abs(self.kappa), 1e-6)
        self.v_ref = np.minimum(self.v_wp, np.sqrt(lat_acc_factor_ref * mu_g / abs_k))

        self.fingerprint = hashlib.sha1(
            np.concatenate([xy.ravel(), self.d_left, self.d_right, self.v_wp]).round(5).tobytes()
            + repr((mu, g, lat_acc_factor_ref, kappa_smoothing_window)).encode()
        ).hexdigest()[:12]

    # ------------------------------------------------------------------ profiles
    def periodic_grid(self, s_before: float, s_after: float):
        """Knots and values over ``[-s_before, L + s_after]`` for smooth interpolants."""
        L = self.length
        s3 = np.concatenate([self.s - L, self.s, self.s + L, self.s + 2 * L])
        mask = (s3 >= -s_before) & (s3 <= L + s_after)
        tile = lambda a: np.tile(a, 4)[mask]
        return s3[mask], {
            "kappa": tile(self.kappa),
            "d_left": tile(self.d_left),
            "d_right": tile(self.d_right),
            "v_ref": tile(self.v_ref),
            "v_wp": tile(self.v_wp),
        }

    def _interp(self, values: np.ndarray, s_query):
        s_mod = np.mod(np.asarray(s_query, dtype=np.float64), self.length)
        s_ext = np.concatenate([self.s, [self.length]])
        v_ext = np.concatenate([values, values[:1]])
        return np.interp(s_mod, s_ext, v_ext)

    def kappa_at(self, s):
        return self._interp(self.kappa, s)

    def d_left_at(self, s):
        return self._interp(self.d_left, s)

    def d_right_at(self, s):
        return self._interp(self.d_right, s)

    def v_ref_at(self, s):
        return self._interp(self.v_ref, s)

    def v_wp_at(self, s):
        return self._interp(self.v_wp, s)

    # ---------------------------------------------------------------- projection
    def project(self, xy: np.ndarray, theta: np.ndarray, s_hint: float | None = None,
                window_back: float = 3.0, window_ahead: float | None = None):
        """Project points onto the raceline. Returns ``s`` in [0, L), ``n`` and heading error ``mu``.

        With ``s_hint`` only segments in ``[s_hint - window_back, s_hint + window_ahead]`` are
        searched, which avoids jumping to a nearby but different part of the track.
        """
        pts = np.asarray(xy, dtype=np.float64).reshape(-1, 2)
        theta = np.asarray(theta, dtype=np.float64).reshape(-1)
        n_pts = self.n_points
        if s_hint is None:
            idx = np.arange(n_pts)
        else:
            if window_ahead is None:
                window_ahead = 15.0
            ds = float(np.median(self.seg_len))
            i0 = int(np.floor(np.mod(s_hint, self.length) / ds))
            idx = np.arange(i0 - int(window_back / ds) - 1, i0 + int(window_ahead / ds) + 2) % n_pts

        p0 = self.xy[idx]
        v = self.seg[idx]
        l2 = self.seg_len[idx] ** 2
        w = pts[:, None, :] - p0[None, :, :]
        t = np.clip(np.einsum("msi,si->ms", w, v) / l2[None, :], 0.0, 1.0)
        diff = w - t[:, :, None] * v[None, :, :]
        best = np.argmin(np.einsum("msi,msi->ms", diff, diff), axis=1)
        rows = np.arange(len(pts))
        seg_idx = idx[best]
        t_best = t[rows, best]
        d = diff[rows, best]
        h = self.seg_heading[seg_idx]
        n = -d[:, 0] * np.sin(h) + d[:, 1] * np.cos(h)
        s = np.mod(self.s[seg_idx] + t_best * self.seg_len[seg_idx], self.length)
        tan0 = self.tangent[seg_idx]
        tan1 = self.tangent[(seg_idx + 1) % n_pts]
        tangent = tan0 + t_best * wrap_angle(tan1 - tan0)
        mu = wrap_angle(theta - tangent)
        return s, n, mu
