#!/usr/bin/env python3
"""Test-track run that trains the LLD, then plots encoder distance on a steering/vx grid.

The client drives ``Settings.TEST_TRACK`` (current accel and steer slope) and streams
rows. The learner keeps ``TEST_TRACK`` off, so the usual skip does not apply and the
LLD is trained. Weights are saved at 0 train steps, at the first 1000 ingested rows,
and at 10000 rows.

Each snapshot encodes a 100x100 grid of superstates: steering on x, vx on y, every
other state channel 0, the same row stacked ``HWM_SUPER_STATE_SIZE`` times, action 0.
The plotted value is the L2 distance between that encoding and the encoding of the
all-zero superstate.
"""

from __future__ import annotations

import os
import sys

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["PYTHONUNBUFFERED"] = "1"

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

OUT_DIR = Path(__file__).resolve().parent
ROW_MARKS = (1000, 10000)
MODEL_NAME = "lld_encoder_grid"
PORT = 5567
GRID = 100


def client_main() -> None:
    sys.argv = [sys.argv[0], *[arg for arg in sys.argv[1:] if arg != "--client"]]
    from utilities.parser_utilities import parse_settings_args

    parse_settings_args(description="Test-track client for the LLD encoder grid", save_snapshot=False)
    from utilities.test_track import prepare

    prepare()
    from run.run_simulation import RacingSimulation

    RacingSimulation().run_experiments()


def _start_client() -> "subprocess.Popen":
    import signal
    import subprocess

    def _die_with_parent() -> None:
        if os.name != "posix":
            return
        try:
            import ctypes
            import ctypes.util

            libc = ctypes.CDLL(ctypes.util.find_library("c") or "libc.so.6", use_errno=True)
            libc.prctl(1, signal.SIGKILL)
        except Exception:
            pass

    cmd = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--client",
        "--TEST_TRACK",
        "True",
        "--TEST_TRACK_STORE_MEMORY",
        "True",
        "--CONTROLLER",
        "HirarchicalPlanner",
        "--RENDER_MODE",
        "None",
        "--WEB_RENDER_AUTO_OPEN",
        "False",
        "--SAVE_RECORDINGS",
        "False",
        "--SAVE_PLOTS",
        "False",
        "--SAVE_REWARDS",
        "False",
        "--HWM_LLD_PREVIEW",
        "False",
        "--LEARNER_TCP_PORT",
        str(PORT),
    ]
    proc = subprocess.Popen(cmd, cwd=str(ROOT), preexec_fn=_die_with_parent if os.name == "posix" else None)
    print(f"[encoder-grid] client pid {proc.pid}", flush=True)
    return proc


def server_main() -> None:
    import asyncio
    import shutil
    import threading

    import torch

    from TrainingLite.hwm.comm.protocol import MSG_TERMINATE, pack_simple
    from TrainingLite.hwm.learner.server import HWMLearnerServer
    from TrainingLite.hwm.paths import resolve_model_dir
    from TrainingLite.rl_racing.tcp_utilities import pack_frame
    from utilities.Settings import Settings

    model_dir = resolve_model_dir(MODEL_NAME)
    if model_dir.exists():
        shutil.rmtree(model_dir)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    class SnapshotServer(HWMLearnerServer):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self._weight_lock = threading.Lock()
            self._saved_marks: set[int] = set()
            self._snapshots: list[dict] = []
            self._write_snapshot(0)

        def train_round(self, n_steps: int) -> dict:
            with self._weight_lock:
                return super().train_round(n_steps)

        def _write_snapshot(self, mark: int) -> None:
            sd = self.models.state_dicts(["LLD"])["LLD"]
            sd = {key: value.clone() for key, value in sd.items()}
            payload = {
                "state_dict": sd,
                "mark": int(mark),
                "total_rows": int(self.total_rows),
                "total_updates": int(self.total_updates),
            }
            path = OUT_DIR / f"lld_rows_{mark}.pt"
            torch.save(payload, path)
            self._snapshots.append({k: payload[k] for k in ("mark", "total_rows", "total_updates")})
            print(
                f"[encoder-grid] snapshot {mark}: rows={payload['total_rows']} "
                f"updates={payload['total_updates']} -> {path}",
                flush=True,
            )

        def _try_snapshot(self) -> None:
            if not any(mark not in self._saved_marks and self.total_rows >= mark for mark in ROW_MARKS):
                return
            if not self._weight_lock.acquire(blocking=False):
                return
            try:
                for mark in ROW_MARKS:
                    if mark not in self._saved_marks and self.total_rows >= mark:
                        self._write_snapshot(mark)
                        self._saved_marks.add(mark)
            finally:
                self._weight_lock.release()

        async def _watch_rows(self) -> None:
            next_log = 1000
            while not self._should_terminate:
                self._try_snapshot()
                if self.total_rows >= next_log:
                    info = self._latest_training_info or {}
                    loss = info.get("lld_loss", float("nan"))
                    print(
                        f"[encoder-grid] rows={self.total_rows} updates={self.total_updates} lld_loss={loss}",
                        flush=True,
                    )
                    next_log += 1000
                if ROW_MARKS[-1] in self._saved_marks:
                    await self._broadcast_raw(
                        pack_frame(pack_simple(MSG_TERMINATE, reason="encoder_grid_collected_10000_rows"))
                    )
                    self._should_terminate = True
                    self._train_event.set()
                    return
                await asyncio.sleep(0.05)

        async def run(self) -> None:
            watch = asyncio.create_task(self._watch_rows())
            try:
                await super().run()
            finally:
                watch.cancel()
                try:
                    await watch
                except asyncio.CancelledError:
                    pass

    torch.manual_seed(0)
    server = SnapshotServer(
        host="0.0.0.0",
        port=PORT,
        save_model_name=MODEL_NAME,
        load_model_name=None,
        device="cuda" if torch.cuda.is_available() else "cpu",
        train_every_seconds=0.0,
        grad_steps=32,
        learning_starts=500,
    )
    client = None

    async def _run() -> None:
        nonlocal client
        async def _launch() -> None:
            nonlocal client
            await asyncio.sleep(1.0)
            client = _start_client()

        launcher = asyncio.create_task(_launch())
        try:
            await server.run()
        finally:
            launcher.cancel()
            if client is not None and client.poll() is None:
                client.terminate()
                try:
                    client.wait(timeout=5)
                except Exception:
                    client.kill()
                    client.wait()

    try:
        asyncio.run(_run())
    except KeyboardInterrupt:
        print("[encoder-grid] interrupted", flush=True)
    missing = [mark for mark in (0, *ROW_MARKS) if not (OUT_DIR / f"lld_rows_{mark}.pt").is_file()]
    if missing:
        raise SystemExit(f"missing snapshots: {missing}")
    pdf = plot_snapshots(Settings)
    print(f"[encoder-grid] wrote {pdf}", flush=True)


def plot_snapshots(Settings) -> Path:
    """Distance grids whose horizontal axis is a mirror-odd mixture, not steering alone.

    ``x = 0`` sets every channel except ``vx`` to 0. Negating ``x`` flips every channel that
    changes sign when the world is mirrored about the car's longitudinal axis, and leaves
    ``vx`` unchanged. Pose is dropped before the encoder, so the mixtures use only the odd
    channels the encoder actually sees: yaw rate, lateral velocity, slip and steering.
    Each mixture is a unit vector in that normalized space (channel / ``HWM_STATE_SCALE``),
    and the axis runs from -1 to 1.
    """
    import math

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import torch

    from TrainingLite.hwm.models.networks import LLD
    from TrainingLite.hwm.shared.observation import _normalize_rows, lld_rows_for_super
    from utilities.state_utilities import (
        ANGULAR_VEL_Z_IDX,
        LINEAR_VEL_X_IDX,
        LINEAR_VEL_Y_IDX,
        POSE_THETA_IDX,
        POSE_THETA_SIN_IDX,
        POSE_Y_IDX,
        SLIP_ANGLE_IDX,
        STEERING_ANGLE_IDX,
    )

    scale = [float(value) for value in Settings.HWM_STATE_SCALE]
    state_dim = int(Settings.HWM_STATE_DIM)
    action_dim = int(Settings.HWM_ACTION_DIM)
    repeats = int(Settings.HWM_SUPER_STATE_SIZE)
    vx_lim = scale[LINEAR_VEL_X_IDX]
    odd = (
        ANGULAR_VEL_Z_IDX,
        LINEAR_VEL_Y_IDX,
        SLIP_ANGLE_IDX,
        STEERING_ANGLE_IDX,
        POSE_Y_IDX,
        POSE_THETA_IDX,
        POSE_THETA_SIN_IDX,
    )
    short = {
        ANGULAR_VEL_Z_IDX: f"yaw/{scale[ANGULAR_VEL_Z_IDX]:.2f}",
        LINEAR_VEL_Y_IDX: f"vy/{scale[LINEAR_VEL_Y_IDX]:.0f}",
        SLIP_ANGLE_IDX: f"slip/{scale[SLIP_ANGLE_IDX]:.0f}",
        STEERING_ANGLE_IDX: f"steer/{scale[STEERING_ANGLE_IDX]:.1f}",
    }

    def unit(weights: list[tuple[int, float]]) -> tuple[tuple[int, float], ...]:
        norm = math.sqrt(sum(coef * coef for _, coef in weights))
        return tuple((index, coef / norm) for index, coef in weights)

    mixtures = (
        ("yaw rate", unit([(ANGULAR_VEL_Z_IDX, 1.0)])),
        ("lateral velocity", unit([(LINEAR_VEL_Y_IDX, 1.0)])),
        ("slip angle", unit([(SLIP_ANGLE_IDX, 1.0)])),
        ("steer + yaw", unit([(STEERING_ANGLE_IDX, 1.0), (ANGULAR_VEL_Z_IDX, 1.0)])),
        (
            "steer + yaw + vy + slip",
            unit(
                [
                    (STEERING_ANGLE_IDX, 1.0),
                    (ANGULAR_VEL_Z_IDX, 1.0),
                    (LINEAR_VEL_Y_IDX, 1.0),
                    (SLIP_ANGLE_IDX, 1.0),
                ]
            ),
        ),
    )

    def states_of(x: torch.Tensor, vx: torch.Tensor, direction: tuple[tuple[int, float], ...]) -> torch.Tensor:
        shape = torch.broadcast_shapes(x.shape, vx.shape)
        states = torch.zeros(*shape, state_dim)
        states[..., LINEAR_VEL_X_IDX] = vx
        for index, coef in direction:
            states[..., index] = x * coef * scale[index]
        return states

    def superstates_of(states: torch.Tensor) -> torch.Tensor:
        actions = torch.zeros(*states.shape[:-1], action_dim)
        rows = torch.cat((states, actions), dim=-1).unsqueeze(-2).expand(
            *states.shape[:-1], repeats, state_dim + action_dim
        )
        packed = lld_rows_for_super(_normalize_rows(rows.contiguous(), state_dim), state_dim)
        return packed.reshape(-1, packed.shape[-2] * packed.shape[-1])

    probe = torch.tensor([0.4, -0.8])
    probe_vx = torch.tensor([1.5, 6.0])
    for name, direction in mixtures:
        made = states_of(probe, probe_vx, direction)
        mirrored = made.clone()
        for index in odd:
            mirrored[..., index] *= -1
        if not torch.allclose(mirrored, states_of(-probe, probe_vx, direction)):
            raise RuntimeError(f"mixture {name} does not mirror about the car x-axis")
        at_zero = states_of(torch.zeros(1), torch.zeros(1), direction)
        if not torch.allclose(at_zero, torch.zeros_like(at_zero)):
            raise RuntimeError(f"mixture {name} is not zero at x = 0")

    x_axis = torch.linspace(-1.0, 1.0, GRID)
    vx_axis = torch.linspace(0.0, vx_lim, GRID)
    x_grid, vx_grid = torch.meshgrid(x_axis, vx_axis, indexing="xy")
    zero_super = superstates_of(torch.zeros(1, state_dim))

    loaded = []
    for mark in (0, *ROW_MARKS):
        payload = torch.load(OUT_DIR / f"lld_rows_{mark}.pt", weights_only=False)
        lld = LLD()
        lld.load_state_dict(payload["state_dict"])
        lld.eval()
        loaded.append((mark, payload, lld))

    fig, axes = plt.subplots(
        len(mixtures), 3, figsize=(12.4, 2.55 * len(mixtures)), sharey=True, constrained_layout=True
    )
    for row, (name, direction) in enumerate(mixtures):
        grid_super = superstates_of(states_of(x_grid, vx_grid, direction))
        distances = []
        for mark, payload, lld in loaded:
            with torch.no_grad():
                encoded = lld.attention_norm_projector(grid_super)
                origin = lld.attention_norm_projector(zero_super)
                distance = torch.linalg.vector_norm(encoded - origin, ord=2, dim=-1).reshape(GRID, GRID)
            distances.append((mark, payload, distance.cpu().numpy()))
        vmin = min(float(distance.min()) for _, _, distance in distances)
        vmax = max(float(distance.max()) for _, _, distance in distances)
        formula = " + ".join(f"{coef:.2f}·{short[index]}" for index, coef in direction)
        image = None
        for col, (mark, payload, distance) in enumerate(distances):
            ax = axes[row, col]
            image = ax.imshow(
                distance,
                origin="lower",
                extent=[-1.0, 1.0, 0.0, vx_lim],
                vmin=vmin,
                vmax=vmax,
                aspect="auto",
                interpolation="nearest",
                cmap="viridis",
            )
            if mark == 0:
                snap = "0 train steps"
            else:
                snap = f"after {mark} rows"
            ax.set_title(
                f"{name}:  {snap}\nrows={payload['total_rows']}  updates={payload['total_updates']}",
                fontsize=8,
            )
            if row == len(mixtures) - 1:
                ax.set_xlabel("x", fontsize=8)
            ax.tick_params(labelsize=7)
        axes[row, 0].set_ylabel("vx [m/s]", fontsize=8)
        fig.colorbar(image, ax=axes[row, :], fraction=0.046, pad=0.02).ax.tick_params(labelsize=7)
        axes[row, 1].set_xlabel(f"x = {formula}", fontsize=7)
    fig.suptitle("LLD encoder L2 distance to the zero superstate", fontsize=12)
    pdf = OUT_DIR / "encoder_l2_mixtures.pdf"
    fig.savefig(pdf)
    plt.close(fig)
    return pdf


def plot_symmetry(Settings) -> Path:
    """Mirror symmetry of the encoder, from the saved weight snapshots.

    Page 1: each latent coordinate of a real memory window against the same window
    mirrored through the car's longitudinal axis. A coordinate that flips under that
    mirror lies on the anti-diagonal.

    Page 2: the same coordinates on a steering–vx grid, diverging colormap centered
    at 0. The flipping coordinate changes sign across x = 0; the even coordinate does not.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    import torch

    from TrainingLite.hwm.models.networks import LLD
    from TrainingLite.hwm.paths import resolve_model_dir
    from TrainingLite.hwm.shared.observation import _normalize_rows, lld_rows_for_super
    from utilities.state_utilities import (
        ANGULAR_VEL_Z_IDX,
        LINEAR_VEL_X_IDX,
        LINEAR_VEL_Y_IDX,
        POSE_THETA_IDX,
        POSE_THETA_SIN_IDX,
        POSE_Y_IDX,
        SLIP_ANGLE_IDX,
        STEERING_ANGLE_IDX,
    )

    scale = [float(value) for value in Settings.HWM_STATE_SCALE]
    state_dim = int(Settings.HWM_STATE_DIM)
    action_dim = int(Settings.HWM_ACTION_DIM)
    repeats = int(Settings.HWM_SUPER_STATE_SIZE)
    odd_state = (
        ANGULAR_VEL_Z_IDX,
        LINEAR_VEL_Y_IDX,
        SLIP_ANGLE_IDX,
        STEERING_ANGLE_IDX,
        POSE_Y_IDX,
        POSE_THETA_IDX,
        POSE_THETA_SIN_IDX,
    )
    marks = (("0 train steps", OUT_DIR / "lld_rows_0.pt"), ("after 1000 rows", OUT_DIR / "lld_rows_1000.pt"), ("after 10000 rows", OUT_DIR / "lld_rows_10000.pt"))

    memory = torch.load(resolve_model_dir(MODEL_NAME) / "memory.pt", weights_only=False)
    stored = memory["rows"]
    run = memory["run_len"].reshape(-1)
    ends = [
        index
        for index in range(repeats - 1, int(run.shape[0]))
        if all(int(run[index - lag]) == int(run[index]) - lag for lag in range(repeats))
    ]
    pick = torch.tensor(ends)[torch.randperm(len(ends), generator=torch.Generator().manual_seed(0))[:2000]]
    windows = torch.stack([stored[int(index) - repeats + 1 : int(index) + 1] for index in pick])
    mirrored = windows.clone()
    for index in odd_state:
        mirrored[..., index] *= -1.0
    mirrored[..., state_dim] *= -1.0  # steering action

    def pack(batch: torch.Tensor) -> torch.Tensor:
        return lld_rows_for_super(_normalize_rows(batch, state_dim), state_dim).flatten(1)

    real = pack(windows)
    real_mirror = pack(mirrored)

    x_axis = torch.linspace(-1.0, 1.0, GRID)
    vx_axis = torch.linspace(0.0, scale[LINEAR_VEL_X_IDX], GRID)
    x_grid, vx_grid = torch.meshgrid(x_axis, vx_axis, indexing="xy")
    grid_states = torch.zeros(*x_grid.shape, state_dim)
    grid_states[..., LINEAR_VEL_X_IDX] = vx_grid
    grid_states[..., STEERING_ANGLE_IDX] = x_grid * scale[STEERING_ANGLE_IDX]
    grid_actions = torch.zeros(*x_grid.shape, action_dim)
    grid_rows = torch.cat((grid_states, grid_actions), dim=-1).unsqueeze(-2).expand(
        *x_grid.shape, repeats, state_dim + action_dim
    )
    grid_super = pack(grid_rows.reshape(-1, repeats, state_dim + action_dim).contiguous())

    panels = []
    for title, path in marks:
        payload = torch.load(path, weights_only=False)
        lld = LLD()
        lld.load_state_dict(payload["state_dict"])
        lld.eval()
        with torch.no_grad():
            coded = lld.attention_norm_projector(real)
            coded_mirror = lld.attention_norm_projector(real_mirror)
            field = lld.attention_norm_projector(grid_super).reshape(*x_grid.shape, -1)
        stats = []
        for dim in range(int(coded.shape[-1])):
            left, right = coded[:, dim], coded_mirror[:, dim]
            energy = float(0.5 * (left.pow(2).mean() + right.pow(2).mean()))
            if float(left.std()) < 1e-8 or float(right.std()) < 1e-8:
                corr = 0.0
            else:
                corr = float(torch.corrcoef(torch.stack((left, right)))[0, 1])
            stats.append((dim, corr, energy))
        flipping = max(stats, key=lambda item: item[2] * max(-item[1], 0.0))
        staying = max(stats, key=lambda item: item[2] * max(item[1], 0.0))
        if staying[1] <= 0.0 or staying[0] == flipping[0]:
            staying = max((item for item in stats if item[0] != flipping[0]), key=lambda item: item[1])
        panels.append({"title": title, "payload": payload, "coded": coded, "mirror": coded_mirror, "field": field, "flip": flipping, "even": staying})

    pdf_path = OUT_DIR / "encoder_symmetry.pdf"
    with PdfPages(pdf_path) as pdf:
        fig, axes = plt.subplots(2, 3, figsize=(12.2, 8.0), constrained_layout=True)
        for column, panel in enumerate(panels):
            for row, (kind, chosen) in enumerate((("flips", panel["flip"]), ("stays", panel["even"]))):
                dim, corr, _energy = chosen
                ax = axes[row, column]
                left = panel["coded"][:, dim].numpy()
                right = panel["mirror"][:, dim].numpy()
                ax.scatter(left, right, s=6, alpha=0.35, c="#1f4e79", linewidths=0, rasterized=True)
                lo = float(min(left.min(), right.min()))
                hi = float(max(left.max(), right.max()))
                pad = 0.08 * (hi - lo + 1e-6)
                lo, hi = lo - pad, hi + pad
                ax.plot([lo, hi], [lo, hi], color="0.75", lw=1.0, zorder=0)
                ax.plot([lo, hi], [hi, lo], color="0.15", lw=1.0, ls="--", zorder=0)
                ax.set_xlim(lo, hi)
                ax.set_ylim(lo, hi)
                ax.set_aspect("equal", adjustable="box")
                ax.set_title(f"{panel['title']}\nlatent {dim}  corr {corr:+.2f}", fontsize=9)
                ax.set_xlabel(f"z{dim}(window)", fontsize=8)
                ax.set_ylabel(f"z{dim}(mirror)", fontsize=8)
                ax.tick_params(labelsize=7)
            axes[0, column].text(0.02, 0.98, "flips", transform=axes[0, column].transAxes, va="top", fontsize=8, color="0.25")
            axes[1, column].text(0.02, 0.98, "stays", transform=axes[1, column].transAxes, va="top", fontsize=8, color="0.25")
        fig.suptitle("Real windows: latent coordinate against its mirror", fontsize=13)
        pdf.savefig(fig)
        plt.close(fig)

        fig, axes = plt.subplots(2, 3, figsize=(12.2, 7.2), sharey=True, constrained_layout=True)
        for column, panel in enumerate(panels):
            for row, chosen in enumerate((panel["flip"], panel["even"])):
                dim, corr, _energy = chosen
                image = panel["field"][..., dim].numpy()
                limit = float(max(abs(image.min()), abs(image.max()), 1e-6))
                ax = axes[row, column]
                shown = ax.imshow(
                    image,
                    origin="lower",
                    extent=[-1.0, 1.0, 0.0, scale[LINEAR_VEL_X_IDX]],
                    vmin=-limit,
                    vmax=limit,
                    cmap="RdBu_r",
                    aspect="auto",
                    interpolation="nearest",
                )
                ax.set_title(f"{panel['title']}\nlatent {dim}  corr {corr:+.2f}", fontsize=9)
                ax.tick_params(labelsize=7)
                fig.colorbar(shown, ax=ax, fraction=0.046, pad=0.04).ax.tick_params(labelsize=7)
            axes[1, column].set_xlabel("steering / 0.4", fontsize=8)
        axes[0, 0].set_ylabel("vx [m/s]", fontsize=8)
        axes[1, 0].set_ylabel("vx [m/s]", fontsize=8)
        fig.suptitle("Steering–speed grid: same latent coordinates", fontsize=13)
        pdf.savefig(fig)
        plt.close(fig)
    return pdf_path


def plot_sign_grid(Settings) -> Path:
    """Steering–vx grid of one flipping latent coordinate and one even coordinate.

    The top row is the odd part of the coordinate that flips the most under steering
    reversal, ``(z(s) - z(-s)) / 2``. It is zero on x = 0. The bottom row is the raw
    value of the coordinate that stays even, so both sides share a color.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import torch

    from TrainingLite.hwm.models.networks import LLD
    from TrainingLite.hwm.shared.observation import _normalize_rows, lld_rows_for_super
    from utilities.state_utilities import LINEAR_VEL_X_IDX, STEERING_ANGLE_IDX

    scale = [float(value) for value in Settings.HWM_STATE_SCALE]
    state_dim = int(Settings.HWM_STATE_DIM)
    action_dim = int(Settings.HWM_ACTION_DIM)
    repeats = int(Settings.HWM_SUPER_STATE_SIZE)
    x_axis = torch.linspace(-1.0, 1.0, GRID)
    vx_axis = torch.linspace(0.0, scale[LINEAR_VEL_X_IDX], GRID)
    x_grid, vx_grid = torch.meshgrid(x_axis, vx_axis, indexing="xy")
    states = torch.zeros(*x_grid.shape, state_dim)
    states[..., LINEAR_VEL_X_IDX] = vx_grid
    states[..., STEERING_ANGLE_IDX] = x_grid * scale[STEERING_ANGLE_IDX]
    rows = torch.cat((states, torch.zeros(*x_grid.shape, action_dim)), dim=-1)
    rows = rows.unsqueeze(-2).expand(*x_grid.shape, repeats, state_dim + action_dim)
    packed = lld_rows_for_super(
        _normalize_rows(rows.reshape(-1, repeats, state_dim + action_dim).contiguous(), state_dim),
        state_dim,
    ).flatten(1)

    marks = (
        ("0 train steps", OUT_DIR / "lld_rows_0.pt"),
        ("after 1000 rows", OUT_DIR / "lld_rows_1000.pt"),
        ("after 10000 rows", OUT_DIR / "lld_rows_10000.pt"),
    )
    chosen = []
    for title, path in marks:
        payload = torch.load(path, weights_only=False)
        lld = LLD()
        lld.load_state_dict(payload["state_dict"])
        lld.eval()
        with torch.no_grad():
            field = lld.attention_norm_projector(packed).reshape(*x_grid.shape, -1)
        scored = []
        for dim in range(int(field.shape[-1])):
            values = field[..., dim]
            mirrored = torch.flip(values, dims=[1])
            half = values[:, GRID // 2 :]
            other = mirrored[:, GRID // 2 :]
            if float(half.std()) < 1e-8 or float(other.std()) < 1e-8:
                corr = 0.0
            else:
                corr = float(torch.corrcoef(torch.stack((half.reshape(-1), other.reshape(-1))))[0, 1])
            odd_energy = float((values - mirrored).pow(2).mean())
            even_energy = float((values + mirrored).pow(2).mean())
            scored.append((dim, corr, odd_energy, even_energy, values))
        flipping = max((item for item in scored if item[1] < -0.5), key=lambda item: item[2], default=None)
        if flipping is None:
            flipping = max(scored, key=lambda item: item[2])
        staying = max((item for item in scored if item[1] > 0.5 and item[0] != flipping[0]), key=lambda item: item[3], default=None)
        if staying is None:
            staying = max((item for item in scored if item[0] != flipping[0]), key=lambda item: item[3])
        chosen.append((title, flipping, staying))

    fig, axes = plt.subplots(2, 3, figsize=(12.2, 7.0), sharey=True, constrained_layout=True)
    for column, (title, flipping, staying) in enumerate(chosen):
        dim, corr, _odd_energy, _even_energy, values = flipping
        odd = 0.5 * (values - torch.flip(values, dims=[1]))
        limit = float(max(odd.abs().max(), torch.tensor(1e-6)))
        show = axes[0, column].imshow(
            odd.numpy(),
            origin="lower",
            extent=[-1.0, 1.0, 0.0, scale[LINEAR_VEL_X_IDX]],
            vmin=-limit,
            vmax=limit,
            cmap="RdBu_r",
            aspect="auto",
            interpolation="nearest",
        )
        axes[0, column].axvline(0.0, color="0.3", lw=0.6)
        axes[0, column].set_title(f"{title}\nodd part of latent {dim}  corr {corr:+.2f}", fontsize=9)
        fig.colorbar(show, ax=axes[0, column], fraction=0.046, pad=0.04).ax.tick_params(labelsize=7)

        dim, corr, _odd_energy, _even_energy, values = staying
        limit = float(max(values.abs().max(), torch.tensor(1e-6)))
        show = axes[1, column].imshow(
            values.numpy(),
            origin="lower",
            extent=[-1.0, 1.0, 0.0, scale[LINEAR_VEL_X_IDX]],
            vmin=-limit,
            vmax=limit,
            cmap="RdBu_r",
            aspect="auto",
            interpolation="nearest",
        )
        axes[1, column].axvline(0.0, color="0.3", lw=0.6)
        axes[1, column].set_title(f"latent {dim}  corr {corr:+.2f}", fontsize=9)
        axes[1, column].set_xlabel("steering / 0.4", fontsize=8)
        fig.colorbar(show, ax=axes[1, column], fraction=0.046, pad=0.04).ax.tick_params(labelsize=7)
        for row in (0, 1):
            axes[row, column].tick_params(labelsize=7)
    axes[0, 0].set_ylabel("vx [m/s]", fontsize=8)
    axes[1, 0].set_ylabel("vx [m/s]", fontsize=8)
    fig.suptitle("Blue–white–red latent coordinate on the steering–speed grid", fontsize=13)
    pdf = OUT_DIR / "encoder_sign_grid.pdf"
    fig.savefig(pdf)
    plt.close(fig)
    return pdf


def plot_pca_rgb(Settings) -> Path:
    """Color the steering–yaw / speed grid by the top three principal components.

    PCA is fit on the encoder outputs of every real 3-row window in memory.
    A grid point sets steering and yaw rate from the horizontal coordinate
    (both zero on the vertical axis, both negated when that coordinate flips)
    and vx from the vertical coordinate. The action is zero steering and one
    fixed acceleration, and that row is repeated three times.
    """
    import math

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import torch

    from TrainingLite.hwm.models.networks import LLD
    from TrainingLite.hwm.paths import resolve_model_dir
    from TrainingLite.hwm.shared.observation import _normalize_rows, lld_rows_for_super
    from utilities.state_utilities import ANGULAR_VEL_Z_IDX, LINEAR_VEL_X_IDX, STEERING_ANGLE_IDX

    scale = [float(value) for value in Settings.HWM_STATE_SCALE]
    state_dim = int(Settings.HWM_STATE_DIM)
    action_dim = int(Settings.HWM_ACTION_DIM)
    repeats = int(Settings.HWM_SUPER_STATE_SIZE)
    mix = 1.0 / math.sqrt(2.0)

    memory = torch.load(resolve_model_dir(MODEL_NAME) / "memory.pt", weights_only=False)
    stored = memory["rows"]
    run = memory["run_len"].reshape(-1)
    ends = [
        index
        for index in range(repeats - 1, int(run.shape[0]))
        if all(int(run[index - lag]) == int(run[index]) - lag for lag in range(repeats))
    ]
    windows = torch.stack([stored[index - repeats + 1 : index + 1] for index in ends])
    accel = float(stored[:, state_dim + 1].median())
    if abs(accel) < 0.05:
        positive = stored[:, state_dim + 1]
        positive = positive[positive > 0]
        accel = float(positive.median()) if positive.numel() else 0.3

    def pack(batch: torch.Tensor) -> torch.Tensor:
        return lld_rows_for_super(_normalize_rows(batch, state_dim), state_dim).flatten(1)

    def grid_superstates() -> torch.Tensor:
        x_axis = torch.linspace(-1.0, 1.0, GRID)
        vx_axis = torch.linspace(0.0, scale[LINEAR_VEL_X_IDX], GRID)
        x_grid, vx_grid = torch.meshgrid(x_axis, vx_axis, indexing="xy")
        states = torch.zeros(*x_grid.shape, state_dim)
        states[..., LINEAR_VEL_X_IDX] = vx_grid
        states[..., STEERING_ANGLE_IDX] = x_grid * mix * scale[STEERING_ANGLE_IDX]
        states[..., ANGULAR_VEL_Z_IDX] = x_grid * mix * scale[ANGULAR_VEL_Z_IDX]
        actions = torch.zeros(*x_grid.shape, action_dim)
        actions[..., 1] = accel
        rows = torch.cat((states, actions), dim=-1).unsqueeze(-2).expand(
            *x_grid.shape, repeats, state_dim + action_dim
        )
        return pack(rows.reshape(-1, repeats, state_dim + action_dim).contiguous())

    probe = torch.tensor([0.4, -0.7])
    made = torch.zeros(2, state_dim)
    made[:, STEERING_ANGLE_IDX] = probe * mix * scale[STEERING_ANGLE_IDX]
    made[:, ANGULAR_VEL_Z_IDX] = probe * mix * scale[ANGULAR_VEL_Z_IDX]
    mirrored = made.clone()
    mirrored[:, STEERING_ANGLE_IDX] *= -1
    mirrored[:, ANGULAR_VEL_Z_IDX] *= -1
    neg = torch.zeros(2, state_dim)
    neg[:, STEERING_ANGLE_IDX] = -probe * mix * scale[STEERING_ANGLE_IDX]
    neg[:, ANGULAR_VEL_Z_IDX] = -probe * mix * scale[ANGULAR_VEL_Z_IDX]
    if not torch.allclose(mirrored, neg):
        raise RuntimeError("steering–yaw axis does not mirror across the vertical axis")

    real = pack(windows)
    grid = grid_superstates()
    marks = (
        ("0 train steps", OUT_DIR / "lld_rows_0.pt"),
        ("after 1000 rows", OUT_DIR / "lld_rows_1000.pt"),
        ("after 10000 rows", OUT_DIR / "lld_rows_10000.pt"),
    )
    fig, axes = plt.subplots(1, 3, figsize=(12.4, 4.2), sharey=True, constrained_layout=True)
    for ax, (title, path) in zip(axes, marks):
        payload = torch.load(path, weights_only=False)
        lld = LLD()
        lld.load_state_dict(payload["state_dict"])
        lld.eval()
        with torch.no_grad():
            coded = lld.attention_norm_projector(real)
            mean = coded.mean(dim=0)
            _u, s, v = torch.pca_lowrank(coded - mean, q=3)
            scores = (lld.attention_norm_projector(grid) - mean) @ v
        variance = (s.pow(2) / (coded.shape[0] - 1)).numpy()
        total = float((coded - mean).pow(2).sum() / (coded.shape[0] - 1))
        share = 100.0 * variance.sum() / total
        rgb = scores.reshape(GRID, GRID, 3).numpy()
        for channel in range(3):
            lo, hi = np_percentile(rgb[:, :, channel], 2.0), np_percentile(rgb[:, :, channel], 98.0)
            if hi <= lo:
                hi = lo + 1e-6
            rgb[:, :, channel] = ((rgb[:, :, channel] - lo) / (hi - lo)).clip(0.0, 1.0)
        ax.imshow(
            rgb,
            origin="lower",
            extent=[-1.0, 1.0, 0.0, scale[LINEAR_VEL_X_IDX]],
            aspect="auto",
            interpolation="nearest",
        )
        ax.axvline(0.0, color="white", lw=0.6, alpha=0.8)
        ax.set_title(f"{title}\nPC1–3 explain {share:.0f}%", fontsize=10)
        ax.set_xlabel("x = (steer/0.4 + yaw/3.33) / √2", fontsize=8)
        ax.tick_params(labelsize=7)
    axes[0].set_ylabel("vx [m/s]", fontsize=8)
    physical = accel * float(Settings.HWM_ACTION_DENORM[1])
    fig.suptitle(f"Top 3 principal components as RGB, acceleration {physical:.2f} m/s², steering action 0", fontsize=11)
    pdf = OUT_DIR / "encoder_pca_rgb.pdf"
    fig.savefig(pdf)
    plt.close(fig)
    return pdf


def plot_axis_distance(Settings) -> Path:
    """L2 distance between two states on the steering–yaw axis.

    Both plot axes use the same mixture as the horizontal axis of the PCA figure.
    The color at ``(x, y)`` is the distance between those two encodings. A sign flip
    in the encoder makes this picture unchanged under a half turn.
    """
    import math

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import torch

    from TrainingLite.hwm.models.networks import LLD
    from TrainingLite.hwm.shared.observation import _normalize_rows, lld_rows_for_super
    from utilities.state_utilities import ANGULAR_VEL_Z_IDX, STEERING_ANGLE_IDX

    scale = [float(value) for value in Settings.HWM_STATE_SCALE]
    state_dim = int(Settings.HWM_STATE_DIM)
    action_dim = int(Settings.HWM_ACTION_DIM)
    repeats = int(Settings.HWM_SUPER_STATE_SIZE)
    mix = 1.0 / math.sqrt(2.0)
    accel = 1.5 / float(Settings.HWM_ACTION_DENORM[1])
    axis = torch.linspace(-1.0, 1.0, GRID)

    states = torch.zeros(GRID, state_dim)
    states[:, STEERING_ANGLE_IDX] = axis * mix * scale[STEERING_ANGLE_IDX]
    states[:, ANGULAR_VEL_Z_IDX] = axis * mix * scale[ANGULAR_VEL_Z_IDX]
    actions = torch.zeros(GRID, action_dim)
    actions[:, 1] = accel
    rows = torch.cat((states, actions), dim=-1).unsqueeze(1).expand(GRID, repeats, state_dim + action_dim).contiguous()
    packed = lld_rows_for_super(_normalize_rows(rows, state_dim), state_dim).flatten(1)

    marks = (
        ("0 train steps", OUT_DIR / "lld_rows_0.pt"),
        ("after 1000 rows", OUT_DIR / "lld_rows_1000.pt"),
        ("after 10000 rows", OUT_DIR / "lld_rows_10000.pt"),
    )
    panels = []
    for title, path in marks:
        payload = torch.load(path, weights_only=False)
        lld = LLD()
        lld.load_state_dict(payload["state_dict"])
        lld.eval()
        with torch.no_grad():
            coded = lld.attention_norm_projector(packed)
            distance = torch.cdist(coded, coded)
        turned = torch.flip(distance, dims=(0, 1))
        mismatch = float((distance - turned).abs().max())
        panels.append((title, distance.numpy(), mismatch))
    vmax = max(float(distance.max()) for _title, distance, _mismatch in panels)
    fig, axes = plt.subplots(1, 3, figsize=(12.4, 4.3), constrained_layout=True)
    image = None
    for ax, (title, distance, mismatch) in zip(axes, panels):
        image = ax.imshow(
            distance,
            origin="lower",
            extent=[-1.0, 1.0, -1.0, 1.0],
            vmin=0.0,
            vmax=vmax,
            cmap="magma",
            aspect="equal",
            interpolation="nearest",
        )
        ax.set_title(f"{title}\nmax |D − rot180 D| = {mismatch:.3f}", fontsize=9)
        ax.set_xlabel("state x", fontsize=8)
        ax.tick_params(labelsize=7)
    axes[0].set_ylabel("state y", fontsize=8)
    fig.colorbar(image, ax=axes, fraction=0.046, pad=0.03, label="L2 distance between encodings")
    fig.suptitle("Both axes: (steer/0.4 + yaw/3.33) / √2,   acceleration 1.5 m/s², steering action 0", fontsize=11)
    pdf = OUT_DIR / "encoder_axis_distance.pdf"
    fig.savefig(pdf)
    plt.close(fig)
    return pdf


def _span_channels(image):
    """Min–max each RGB channel on its own so it fills ``[0, 1]``."""
    out = image.copy()
    for channel in range(out.shape[-1]):
        values = out[..., channel]
        lo = float(values.min())
        hi = float(values.max())
        out[..., channel] = 0.0 if hi <= lo else (values - lo) / (hi - lo)
    return out


def plot_pca_diff_rgb(Settings) -> Path:
    """Pairwise plot with RGB equal to the difference of principal components 1–3.

    Same two states as the L2 distance figure. Each snapshot is scaled on its own,
    and each color channel is scaled on its own.
    """
    import math

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import torch

    from TrainingLite.hwm.models.networks import LLD
    from TrainingLite.hwm.paths import resolve_model_dir
    from TrainingLite.hwm.shared.observation import _normalize_rows, lld_rows_for_super
    from utilities.state_utilities import ANGULAR_VEL_Z_IDX, STEERING_ANGLE_IDX

    scale = [float(value) for value in Settings.HWM_STATE_SCALE]
    state_dim = int(Settings.HWM_STATE_DIM)
    action_dim = int(Settings.HWM_ACTION_DIM)
    repeats = int(Settings.HWM_SUPER_STATE_SIZE)
    mix = 1.0 / math.sqrt(2.0)
    accel = 1.5 / float(Settings.HWM_ACTION_DENORM[1])
    axis = torch.linspace(-1.0, 1.0, GRID)
    states = torch.zeros(GRID, state_dim)
    states[:, STEERING_ANGLE_IDX] = axis * mix * scale[STEERING_ANGLE_IDX]
    states[:, ANGULAR_VEL_Z_IDX] = axis * mix * scale[ANGULAR_VEL_Z_IDX]
    actions = torch.zeros(GRID, action_dim)
    actions[:, 1] = accel
    rows = torch.cat((states, actions), dim=-1).unsqueeze(1).expand(GRID, repeats, state_dim + action_dim).contiguous()
    packed = lld_rows_for_super(_normalize_rows(rows, state_dim), state_dim).flatten(1)

    memory = torch.load(resolve_model_dir(MODEL_NAME) / "memory.pt", weights_only=False)
    stored, run = memory["rows"], memory["run_len"].reshape(-1)
    ends = [
        index
        for index in range(repeats - 1, int(run.shape[0]))
        if all(int(run[index - lag]) == int(run[index]) - lag for lag in range(repeats))
    ]
    windows = torch.stack([stored[index - repeats + 1 : index + 1] for index in ends])
    real = lld_rows_for_super(_normalize_rows(windows, state_dim), state_dim).flatten(1)

    marks = (
        ("0 train steps", OUT_DIR / "lld_rows_0.pt"),
        ("after 1000 rows", OUT_DIR / "lld_rows_1000.pt"),
        ("after 10000 rows", OUT_DIR / "lld_rows_10000.pt"),
    )
    fig, axes = plt.subplots(3, 3, figsize=(12.2, 11.2), constrained_layout=True)
    channel_names = ("PC1 difference", "PC2 difference", "PC3 difference")
    for column, (title, path) in enumerate(marks):
        payload = torch.load(path, weights_only=False)
        lld = LLD()
        lld.load_state_dict(payload["state_dict"])
        lld.eval()
        with torch.no_grad():
            basis = lld.attention_norm_projector(real)
            mean = basis.mean(dim=0)
            _u, _s, directions = torch.pca_lowrank(basis - mean, q=3)
            scores = (lld.attention_norm_projector(packed) - mean) @ directions
        diff = _span_channels((scores[None, :, :] - scores[:, None, :]).numpy())
        for row, name in enumerate(channel_names):
            ax = axes[row, column]
            ax.imshow(
                diff[..., row],
                origin="lower",
                extent=[-1.0, 1.0, -1.0, 1.0],
                vmin=0.0,
                vmax=1.0,
                cmap="magma",
                aspect="equal",
                interpolation="nearest",
            )
            if row == 0:
                ax.set_title(title, fontsize=9)
            if column == 0:
                ax.set_ylabel(name, fontsize=8)
            if row == 2:
                ax.set_xlabel("state x", fontsize=8)
            ax.tick_params(labelsize=7)
    fig.suptitle("Each principal-component difference on its own, scaled to the full range", fontsize=12)
    pdf = OUT_DIR / "encoder_pca_diff_rgb.pdf"
    fig.savefig(pdf)
    plt.close(fig)
    return pdf


def plot_delta_rgb(Settings) -> Path:
    """Steering–yaw versus speed, colored by the predicted step.

    Red is the predicted change in vx, green the predicted change in vy, blue the
    standard deviation of the predicted heading change. Each snapshot and each
    channel is scaled to the full color range on its own.
    """
    import math

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import torch

    from TrainingLite.hwm.models.networks import LLD, MultivariateStudentT
    from TrainingLite.hwm.paths import resolve_model_dir
    from TrainingLite.hwm.shared.memory_manager import MemoryManager
    from TrainingLite.hwm.shared.observation import (
        _normalize_rows,
        lld_dynamic_indices,
        lld_rows_for_super,
        retrieve_LLD_neighbors,
    )
    from utilities.state_utilities import (
        ANGULAR_VEL_Z_IDX,
        LINEAR_VEL_X_IDX,
        LINEAR_VEL_Y_IDX,
        POSE_THETA_IDX,
        STEERING_ANGLE_IDX,
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    scale = [float(value) for value in Settings.HWM_STATE_SCALE]
    state_dim = int(Settings.HWM_STATE_DIM)
    action_dim = int(Settings.HWM_ACTION_DIM)
    repeats = int(Settings.HWM_SUPER_STATE_SIZE)
    mix = 1.0 / math.sqrt(2.0)
    accel = 1.5 / float(Settings.HWM_ACTION_DENORM[1])
    dynamic = lld_dynamic_indices()
    vx_channel = dynamic.index(LINEAR_VEL_X_IDX)
    vy_channel = dynamic.index(LINEAR_VEL_Y_IDX)
    heading_channel = dynamic.index(POSE_THETA_IDX)

    x_axis = torch.linspace(-1.0, 1.0, GRID)
    vx_axis = torch.linspace(0.0, scale[LINEAR_VEL_X_IDX], GRID)
    x_grid, vx_grid = torch.meshgrid(x_axis, vx_axis, indexing="xy")
    states = torch.zeros(*x_grid.shape, state_dim)
    states[..., LINEAR_VEL_X_IDX] = vx_grid
    states[..., STEERING_ANGLE_IDX] = x_grid * mix * scale[STEERING_ANGLE_IDX]
    states[..., ANGULAR_VEL_Z_IDX] = x_grid * mix * scale[ANGULAR_VEL_Z_IDX]
    actions = torch.zeros(*x_grid.shape, action_dim)
    actions[..., 1] = accel
    rows = torch.cat((states, actions), dim=-1).unsqueeze(-2).expand(*x_grid.shape, repeats, state_dim + action_dim)
    packed = lld_rows_for_super(
        _normalize_rows(rows.reshape(-1, repeats, state_dim + action_dim).contiguous(), state_dim),
        state_dim,
    ).flatten(1)

    stored_memory = torch.load(resolve_model_dir(MODEL_NAME) / "memory.pt", weights_only=False)
    marks = (
        ("0 train steps", OUT_DIR / "lld_rows_0.pt"),
        ("after 1000 rows", OUT_DIR / "lld_rows_1000.pt"),
        ("after 10000 rows", OUT_DIR / "lld_rows_10000.pt"),
    )
    channel_names = ("predicted Δvx", "predicted Δvy", "std of predicted Δheading")
    fig, axes = plt.subplots(3, 3, figsize=(12.2, 10.4), sharey=True, constrained_layout=True)
    for column, (title, path) in enumerate(marks):
        payload = torch.load(path, weights_only=False)
        lld = LLD().to(device)
        lld.load_state_dict(payload["state_dict"])
        lld.eval()
        memory = MemoryManager(device=device, keep_all_rows=True)
        memory.attach_projector(lld.attention_norm_projector)
        saved_drop = Settings.HWM_MEMORY_DROP_FINAL_STEPS
        Settings.HWM_MEMORY_DROP_FINAL_STEPS = 0
        try:
            memory.import_rows(
                {
                    "rows": stored_memory["rows"],
                    "run_len": stored_memory["run_len"],
                    "bank_ends": stored_memory["bank_ends"],
                }
            )
        finally:
            Settings.HWM_MEMORY_DROP_FINAL_STEPS = saved_drop
        parts = []
        with torch.no_grad():
            for start in range(0, int(packed.shape[0]), 128):
                batch = packed[start : start + 128].to(device)
                neighbors = retrieve_LLD_neighbors(batch, memory)
                dist = lld(
                    {
                        "current_super_state": batch,
                        "similar_superstates": neighbors["similar_superstates"],
                        "similar_superstates_diff": neighbors["similar_superstates_diff"],
                        "similar_valid": neighbors["valid"],
                        "similar_density": neighbors["density"],
                    }
                )
                variance = dist.scale_tril.pow(2).sum(dim=-1)
                if isinstance(dist, MultivariateStudentT):
                    variance = variance * (dist.df / (dist.df - 2).clamp_min(1e-3)).unsqueeze(-1)
                std = variance.clamp_min(0).sqrt()
                parts.append(
                    torch.stack(
                        (dist.loc[:, vx_channel], dist.loc[:, vy_channel], std[:, heading_channel]),
                        dim=-1,
                    ).cpu()
                )
        image = _span_channels(torch.cat(parts, dim=0).reshape(GRID, GRID, 3).numpy())
        for row, name in enumerate(channel_names):
            ax = axes[row, column]
            ax.imshow(
                image[..., row],
                origin="lower",
                extent=[-1.0, 1.0, 0.0, scale[LINEAR_VEL_X_IDX]],
                vmin=0.0,
                vmax=1.0,
                cmap="magma",
                aspect="auto",
                interpolation="nearest",
            )
            ax.axvline(0.0, color="white", lw=0.5, alpha=0.7)
            if row == 0:
                ax.set_title(title, fontsize=9)
            if column == 0:
                ax.set_ylabel(f"vx [m/s]\n{name}", fontsize=8)
            if row == 2:
                ax.set_xlabel("x = (steer/0.4 + yaw/3.33) / √2", fontsize=8)
            ax.tick_params(labelsize=7)
        del memory, lld
        if device == "cuda":
            torch.cuda.empty_cache()
    fig.suptitle("Each predicted channel on its own, scaled to the full range", fontsize=12)
    pdf = OUT_DIR / "encoder_delta_rgb.pdf"
    fig.savefig(pdf)
    plt.close(fig)
    return pdf


def np_percentile(values, q: float) -> float:
    flat = values.reshape(-1)
    order = flat.argsort()
    pos = (len(flat) - 1) * q / 100.0
    lo = int(pos)
    hi = min(lo + 1, len(flat) - 1)
    return float(flat[order[lo]] * (1.0 - (pos - lo)) + flat[order[hi]] * (pos - lo))


if __name__ == "__main__":
    if "--client" in sys.argv:
        client_main()
    elif "--plot-only" in sys.argv:
        from utilities.Settings import Settings

        print(plot_snapshots(Settings), flush=True)
    elif "--symmetry" in sys.argv:
        from utilities.Settings import Settings

        print(plot_symmetry(Settings), flush=True)
    elif "--sign" in sys.argv:
        from utilities.Settings import Settings

        print(plot_sign_grid(Settings), flush=True)
    elif "--pca" in sys.argv:
        from utilities.Settings import Settings

        print(plot_pca_rgb(Settings), flush=True)
    elif "--axis-distance" in sys.argv:
        from utilities.Settings import Settings

        print(plot_axis_distance(Settings), flush=True)
    elif "--pca-diff" in sys.argv:
        from utilities.Settings import Settings

        print(plot_pca_diff_rgb(Settings), flush=True)
    elif "--delta-rgb" in sys.argv:
        from utilities.Settings import Settings

        print(plot_delta_rgb(Settings), flush=True)
    else:
        server_main()
