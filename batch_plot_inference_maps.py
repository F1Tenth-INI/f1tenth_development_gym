#!/usr/bin/env python3
"""
Batch plot trajectory overlays for inference recordings.

Usage:
    python batch_plot_inference_maps.py --inference-dir ExperimentRecordings/2026-04-22_INFERENCE --map-name IPZ38
"""

import argparse
import os
from pathlib import Path

import matplotlib

# Select matplotlib backend early.
_backend_parser = argparse.ArgumentParser(add_help=False)
_backend_group = _backend_parser.add_mutually_exclusive_group()
_backend_group.add_argument("--interactive-plots", action="store_true")
_backend_group.add_argument("--non-interactive-plots", action="store_true")
_backend_args, _ = _backend_parser.parse_known_args()

if _backend_args.interactive_plots:
    matplotlib.use("TkAgg")
else:
    matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.ticker import FuncFormatter
import numpy as np
import pandas as pd

from plot_sample_frequency import load_map_image


root_dir = os.path.dirname(os.path.abspath(__file__))


class InferenceTrajectoryPlotter:
    MAP_METERS_PER_PIXEL = 0.05

    def __init__(
        self,
        inference_dir: str,
        map_name: str = "RCA1",
        output_dir: str = None,
        color_by: str = "speed",
        extra_color_by=None,
        stride: int = 1,
        linewidth: float = 2.0,
        alpha: float = 0.9,
        map_overlay: bool = True,
        show_waypoints: bool = True,
        waypoints_file: str = None,
    ):
        self.inference_dir = Path(inference_dir).expanduser().resolve()
        self.map_name = map_name
        self.color_by = color_by
        self.extra_color_by = list(extra_color_by or [])
        self.stride = max(1, int(stride))
        self.linewidth = float(max(0.1, linewidth))
        self.alpha = float(np.clip(alpha, 0.05, 1.0))
        self.map_overlay = bool(map_overlay)
        self.show_waypoints = bool(show_waypoints)
        self.waypoints_file = Path(waypoints_file).expanduser().resolve() if waypoints_file else None

        if output_dir is None:
            output_dir = Path(root_dir) / "batch_plot_inference" / f"batch_inference_{self.map_name}"
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.trajectory_dir = self.output_dir / "11_inference_trajectories"
        self.trajectory_dir.mkdir(exist_ok=True)

    @staticmethod
    def _sanitize_name(name: str) -> str:
        out = []
        for ch in name:
            if ch.isalnum() or ch in ("-", "_"):
                out.append(ch)
            else:
                out.append("_")
        return "".join(out)

    @staticmethod
    def _inference_label_from_path(csv_path: Path) -> str:
        stem = csv_path.stem
        parts = stem.split("_")
        if len(parts) >= 3 and len(parts[0]) == 10 and len(parts[1]) == 8:
            return "_".join(parts[2:])
        return stem

    @staticmethod
    def _style_axes_for_map(ax, img_array, meters_per_pixel: float = 0.05):
        h, w = img_array.shape[:2]
        ax.set_xlim(0, w)
        ax.set_ylim(h, 0)
        ax.set_aspect("equal")
        ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x * meters_per_pixel:.1f}"))
        ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f"{y * meters_per_pixel:.1f}"))

    @staticmethod
    def _extract_waypoint_xy_columns(df: pd.DataFrame):
        known_pairs = [
            ("x_m", "y_m"),
            ("x", "y"),
            ("X", "Y"),
            ("pose_x", "pose_y"),
        ]
        for x_col, y_col in known_pairs:
            if x_col in df.columns and y_col in df.columns:
                x = pd.to_numeric(df[x_col], errors="coerce").to_numpy(dtype=float)
                y = pd.to_numeric(df[y_col], errors="coerce").to_numpy(dtype=float)
                return x, y

        if df.shape[1] >= 2:
            x = pd.to_numeric(df.iloc[:, 0], errors="coerce").to_numpy(dtype=float)
            y = pd.to_numeric(df.iloc[:, 1], errors="coerce").to_numpy(dtype=float)
            return x, y

        raise ValueError("Waypoint file has fewer than 2 columns")

    @staticmethod
    def _world_to_image_pixel(x: float, y: float, origin, resolution: float, image_height: int):
        """Convert world coordinates to image pixel coordinates.

        ROS map origin is bottom-left in world frame; image origin is top-left.
        Also account for map yaw in origin[2].
        """
        ox, oy = float(origin[0]), float(origin[1])
        oyaw = float(origin[2]) if len(origin) >= 3 else 0.0

        dx = float(x) - ox
        dy = float(y) - oy

        # Transform world displacement into map-frame displacement.
        c = np.cos(-oyaw)
        s = np.sin(-oyaw)
        mx = c * dx - s * dy
        my = s * dx + c * dy

        px = int(np.round(mx / float(resolution)))
        py_map = int(np.round(my / float(resolution)))
        py_img = image_height - 1 - py_map
        return px, py_img

    def _load_map_waypoints_world(self):
        if self.waypoints_file is not None:
            if not self.waypoints_file.exists():
                raise FileNotFoundError(f"Waypoints file override not found: {self.waypoints_file}")
            candidate_paths = [self.waypoints_file]
        else:
            map_dir = Path(root_dir) / "utilities" / "maps" / self.map_name
            candidate_paths = [
                map_dir / f"{self.map_name}_wp.csv",
                map_dir / "centerline.csv",
                map_dir / f"{self.map_name}_trajectory.csv",
            ]

        for wp_path in candidate_paths:
            if not wp_path.exists():
                continue
            for read_kwargs in ({"comment": "#"}, {"comment": "#", "header": None}):
                try:
                    df_wp = pd.read_csv(wp_path, **read_kwargs)
                    if len(df_wp) == 0:
                        continue
                    xw, yw = self._extract_waypoint_xy_columns(df_wp)
                    mask = np.isfinite(xw) & np.isfinite(yw)
                    xw = xw[mask]
                    yw = yw[mask]
                    if len(xw) >= 2:
                        return xw, yw, wp_path
                except Exception:
                    continue

        return None, None, None

    def _build_values(self, df: pd.DataFrame):
        return self._build_values_for_spec(df, self.color_by)

    def _resolve_color_spec(self, color_spec: str):
        presets = {
            "speed": ("linear_vel_x", "Speed (linear_vel_x)"),
            "steering": ("steering_angle", "Steering Angle"),
            "reward": ("reward", "Reward"),
            "angular_vel_z": ("angular_vel_z", "Angular Velocity Z"),
        }
        if color_spec in presets:
            return presets[color_spec]

        # Fallback: treat as a raw CSV column name.
        return color_spec, color_spec

    def _build_values_for_spec(self, df: pd.DataFrame, color_spec: str):
        col_name, label = self._resolve_color_spec(color_spec)
        if col_name not in df.columns:
            raise ValueError(f"Missing column '{col_name}' for color-by {color_spec}")
        return pd.to_numeric(df[col_name], errors="coerce"), label

    def _plot_colored_trajectory(self, ax, x, y, values, vmin, vmax):
        pts = np.column_stack([x, y])
        if len(pts) < 2:
            return None

        seg = np.stack([pts[:-1], pts[1:]], axis=1)
        seg_values = np.asarray(values[:-1], dtype=float)
        mask = np.isfinite(seg_values)
        seg = seg[mask]
        seg_values = seg_values[mask]
        if len(seg) == 0:
            return None

        lc = LineCollection(
            seg,
            cmap="viridis",
            norm=plt.Normalize(vmin=vmin, vmax=vmax),
            linewidths=self.linewidth,
            alpha=self.alpha,
        )
        lc.set_array(seg_values)
        ax.add_collection(lc)
        return lc

    def _find_csvs(self):
        if not self.inference_dir.exists():
            raise FileNotFoundError(f"Inference directory not found: {self.inference_dir}")
        return sorted(self.inference_dir.glob("*.csv"))

    def run(self):
        csv_files = self._find_csvs()
        if not csv_files:
            print("No inference CSV files found.")
            return

        print(f"Generating inference trajectories from {len(csv_files)} recording files")

        runs = []

        map_img_array = None
        map_cfg = None
        use_map_overlay = self.map_overlay
        waypoint_pixels = None

        if use_map_overlay:
            try:
                map_img_array, map_cfg = load_map_image(self.map_name)
                map_height, _ = map_img_array.shape[:2]

                if self.show_waypoints:
                    wp_x_world, wp_y_world, waypoint_source = self._load_map_waypoints_world()
                    if wp_x_world is not None and wp_y_world is not None:
                        origin = map_cfg["origin"]
                        resolution = map_cfg["resolution"]
                        wp_pixels = [
                            self._world_to_image_pixel(xw, yw, origin, resolution, map_height)
                            for xw, yw in zip(wp_x_world, wp_y_world)
                        ]
                        waypoint_pixels = (
                            np.array([p[0] for p in wp_pixels], dtype=float),
                            np.array([p[1] for p in wp_pixels], dtype=float),
                        )
                        print(f"Waypoints overlay source: {waypoint_source}")
                    else:
                        print("Waypoints overlay: no waypoint file found for this map")
            except Exception as e:
                print(f"Warning: failed to load map '{self.map_name}' for overlay: {e}")
                print("Falling back to XY world-coordinate trajectory plots.")
                use_map_overlay = False

        for csv_path in csv_files:
            try:
                df = pd.read_csv(csv_path, comment="#")
                if "pose_x" not in df.columns or "pose_y" not in df.columns:
                    print(f"Skipping {csv_path.name}: missing pose_x/pose_y")
                    continue

                if self.stride > 1:
                    df = df.iloc[::self.stride].copy()

                x = pd.to_numeric(df["pose_x"], errors="coerce").to_numpy()
                y = pd.to_numeric(df["pose_y"], errors="coerce").to_numpy()

                mask = np.isfinite(x) & np.isfinite(y)
                x = x[mask]
                y = y[mask]
                df_xy = df.loc[mask].reset_index(drop=True)

                if len(x) < 2:
                    print(f"Skipping {csv_path.name}: not enough valid points")
                    continue

                if use_map_overlay:
                    origin = map_cfg["origin"]
                    resolution = map_cfg["resolution"]
                    map_height, _ = map_img_array.shape[:2]
                    pixels = [
                        self._world_to_image_pixel(px, py, origin, resolution, map_height)
                        for px, py in zip(x, y)
                    ]
                    pix_x = np.array([p[0] for p in pixels], dtype=float)
                    pix_y = np.array([p[1] for p in pixels], dtype=float)
                else:
                    pix_x = None
                    pix_y = None

                run_label = self._inference_label_from_path(csv_path)
                runs.append(
                    {
                        "path": csv_path,
                        "label": run_label,
                        "x": x,
                        "y": y,
                        "pixel_x": pix_x,
                        "pixel_y": pix_y,
                        "df_xy": df_xy,
                    }
                )
            except Exception as e:
                print(f"Skipping {csv_path.name}: {e}")

        if not runs:
            print("No valid runs found for trajectory plotting.")
            return

        if use_map_overlay:
            fig, ax = plt.subplots(figsize=(10, 8))
            ax.imshow(map_img_array, cmap="gray", alpha=0.95)
            if waypoint_pixels is not None:
                ax.plot(waypoint_pixels[0], waypoint_pixels[1], color="deepskyblue", linewidth=1.2, alpha=0.95, label="waypoints")
                ax.scatter(waypoint_pixels[0], waypoint_pixels[1], s=4, c="deepskyblue", alpha=0.75)
            ax.set_title(f"Map + Waypoints Only ({self.map_name})")
            ax.set_xlabel("map x [m]")
            ax.set_ylabel("map y [m]")
            self._style_axes_for_map(ax, map_img_array, self.MAP_METERS_PER_PIXEL)
            ax.grid(alpha=0.25)
            if waypoint_pixels is not None:
                ax.legend(loc="best")
            fig.tight_layout()
            map_only_path = self.trajectory_dir / "map_waypoints_only.png"
            fig.savefig(map_only_path, dpi=200)
            plt.close(fig)

        color_specs = []
        for spec in [self.color_by] + self.extra_color_by:
            if spec not in color_specs:
                color_specs.append(spec)

        for color_spec in color_specs:
            plot_runs = []
            value_ranges = []
            color_label = None

            for run in runs:
                values_series, label = self._build_values_for_spec(run["df_xy"], color_spec)
                values = values_series.to_numpy(dtype=float)

                vmask = np.isfinite(values)
                if not np.any(vmask):
                    print(f"Skipping {run['label']} for color '{color_spec}': no finite values")
                    continue

                if use_map_overlay:
                    x_series = run["pixel_x"][vmask]
                    y_series = run["pixel_y"][vmask]
                else:
                    x_series = run["x"][vmask]
                    y_series = run["y"][vmask]

                values = values[vmask]
                if len(values) < 2:
                    continue

                plot_runs.append(
                    {
                        "label": run["label"],
                        "x": x_series,
                        "y": y_series,
                        "values": values,
                    }
                )
                value_ranges.append((float(np.nanmin(values)), float(np.nanmax(values))))
                color_label = label

            if not plot_runs:
                print(f"No valid runs for color-by '{color_spec}'")
                continue

            global_vmin = min(v[0] for v in value_ranges)
            global_vmax = max(v[1] for v in value_ranges)
            if np.isclose(global_vmin, global_vmax):
                global_vmin -= 1e-6
                global_vmax += 1e-6

            color_tag = self._sanitize_name(color_spec)
            per_run_dir = self.trajectory_dir / f"per_run_colored_by_{color_tag}"
            per_run_dir.mkdir(exist_ok=True, parents=True)

            for run in plot_runs:
                fig, ax = plt.subplots(figsize=(8, 7))
                if use_map_overlay:
                    ax.imshow(map_img_array, cmap="gray", alpha=0.92)
                    if waypoint_pixels is not None:
                        ax.plot(waypoint_pixels[0], waypoint_pixels[1], color="deepskyblue", linewidth=1.0, alpha=0.9, label="waypoints")
                        ax.scatter(waypoint_pixels[0], waypoint_pixels[1], s=4, c="deepskyblue", alpha=0.7)

                lc = self._plot_colored_trajectory(ax, run["x"], run["y"], run["values"], global_vmin, global_vmax)
                if lc is not None:
                    cbar = fig.colorbar(lc, ax=ax, fraction=0.046, pad=0.04)
                    cbar.set_label(color_label)

                ax.scatter(run["x"][0], run["y"][0], marker="o", s=30, c="lime", edgecolors="black", label="start")
                ax.scatter(run["x"][-1], run["y"][-1], marker="X", s=36, c="red", edgecolors="black", label="end")
                map_suffix = f" on {self.map_name}" if use_map_overlay else ""
                ax.set_title(f"Trajectory: {run['label']} (colored by {color_spec}){map_suffix}")
                if use_map_overlay:
                    ax.set_xlabel("map x [m]")
                    ax.set_ylabel("map y [m]")
                    self._style_axes_for_map(ax, map_img_array, self.MAP_METERS_PER_PIXEL)
                else:
                    ax.set_xlabel("pose_x")
                    ax.set_ylabel("pose_y")
                    ax.axis("equal")
                ax.grid(alpha=0.3)
                ax.legend(loc="best")
                fig.tight_layout()

                out_name = f"{self._sanitize_name(run['label'])}_trajectory_{color_tag}.png"
                fig.savefig(per_run_dir / out_name, dpi=180)
                plt.close(fig)

            fig, ax = plt.subplots(figsize=(10, 8))
            if use_map_overlay:
                ax.imshow(map_img_array, cmap="gray", alpha=0.9)
                if waypoint_pixels is not None:
                    ax.plot(waypoint_pixels[0], waypoint_pixels[1], color="deepskyblue", linewidth=1.1, alpha=0.9, label="waypoints")
            cmap = plt.get_cmap("tab10")
            for i, run in enumerate(plot_runs):
                ax.plot(
                    run["x"],
                    run["y"],
                    color=cmap(i % 10),
                    linewidth=max(1.0, self.linewidth * 0.9),
                    alpha=min(1.0, self.alpha + 0.05),
                    label=run["label"],
                )

            ax.set_title(
                f"All Inference Runs: Racing Line Overlay"
                f"{' on ' + self.map_name if use_map_overlay else ''}"
                f" [{color_spec}]"
            )
            if use_map_overlay:
                ax.set_xlabel("map x [m]")
                ax.set_ylabel("map y [m]")
                self._style_axes_for_map(ax, map_img_array, self.MAP_METERS_PER_PIXEL)
            else:
                ax.set_xlabel("pose_x")
                ax.set_ylabel("pose_y")
                ax.axis("equal")
            ax.grid(alpha=0.3)
            ax.legend(loc="best", fontsize=8)
            fig.tight_layout()
            overlay_path = self.trajectory_dir / f"all_runs_overlay_by_run_{color_tag}.png"
            fig.savefig(overlay_path, dpi=200)
            plt.close(fig)

            fig, ax = plt.subplots(figsize=(10, 8))
            if use_map_overlay:
                ax.imshow(map_img_array, cmap="gray", alpha=0.9)
                if waypoint_pixels is not None:
                    ax.plot(waypoint_pixels[0], waypoint_pixels[1], color="deepskyblue", linewidth=1.1, alpha=0.9)

            last_lc = None
            for run in plot_runs:
                last_lc = self._plot_colored_trajectory(ax, run["x"], run["y"], run["values"], global_vmin, global_vmax)

            if last_lc is not None:
                cbar = fig.colorbar(last_lc, ax=ax, fraction=0.046, pad=0.04)
                cbar.set_label(color_label)

            ax.set_title(
                f"All Inference Runs: Overlay Colored by {color_spec}"
                f"{' on ' + self.map_name if use_map_overlay else ''}"
            )
            if use_map_overlay:
                ax.set_xlabel("map x [m]")
                ax.set_ylabel("map y [m]")
                self._style_axes_for_map(ax, map_img_array, self.MAP_METERS_PER_PIXEL)
            else:
                ax.set_xlabel("pose_x")
                ax.set_ylabel("pose_y")
                ax.axis("equal")
            ax.grid(alpha=0.3)
            fig.tight_layout()
            fig.savefig(self.trajectory_dir / f"all_runs_overlay_colored_by_{color_tag}.png", dpi=200)
            plt.close(fig)

        print(f"Saved outputs to: {self.trajectory_dir}")


def main():
    parser = argparse.ArgumentParser(description="Plot inference trajectories with optional map and waypoint overlays")
    parser.add_argument("--inference-dir", required=True, help="Directory containing inference CSV recordings")
    parser.add_argument("--map-name", type=str, default="RCA1", help="Map name to load from utilities/maps")
    parser.add_argument("--output-dir", type=str, default=None, help="Output directory root")
    parser.add_argument(
        "--color-by",
        type=str,
        default="speed",
        help=(
            "Primary trajectory color metric. Presets: speed, steering, reward, angular_vel_z. "
            "You can also pass any raw CSV column name."
        ),
    )
    parser.add_argument(
        "--extra-color-by",
        type=str,
        nargs="*",
        default=[],
        help=(
            "Additional color metrics/columns to render in the same run, e.g. --extra-color-by angular_vel_z reward"
        ),
    )
    parser.add_argument("--stride", type=int, default=1, help="Use every Nth row from each CSV")
    parser.add_argument("--linewidth", type=float, default=2.0, help="Trajectory line width")
    parser.add_argument("--alpha", type=float, default=0.9, help="Trajectory alpha")
    parser.add_argument("--map-overlay", action=argparse.BooleanOptionalAction, default=True, help="Draw trajectories over map image")
    parser.add_argument("--show-waypoints", action=argparse.BooleanOptionalAction, default=True, help="Draw waypoint overlay on map plots")
    parser.add_argument("--waypoints-file", type=str, default=None, help="Optional explicit waypoint CSV")

    backend_group = parser.add_mutually_exclusive_group()
    backend_group.add_argument("--interactive-plots", action="store_true", help="Use interactive matplotlib backend (TkAgg)")
    backend_group.add_argument("--non-interactive-plots", action="store_true", help="Force non-interactive backend (Agg)")

    args = parser.parse_args()

    plotter = InferenceTrajectoryPlotter(
        inference_dir=args.inference_dir,
        map_name=args.map_name,
        output_dir=args.output_dir,
        color_by=args.color_by,
        extra_color_by=args.extra_color_by,
        stride=args.stride,
        linewidth=args.linewidth,
        alpha=args.alpha,
        map_overlay=args.map_overlay,
        show_waypoints=args.show_waypoints,
        waypoints_file=args.waypoints_file,
    )
    plotter.run()


if __name__ == "__main__":
    main()
