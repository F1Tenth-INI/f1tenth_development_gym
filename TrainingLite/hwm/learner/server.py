"""HWM learner server.

Skeleton adapted from ``TrainingLite/rl_racing/learner_server.py`` with the SAC
specifics removed. It accepts planner connections, ingests ``[state, action]``
rows into a ``MemoryManager`` (the same class the planner uses), runs training
rounds in a worker thread, saves checkpoints and broadcasts weights.

Insertion points (the only methods meant to be edited):

    build_models()                  -> ModelBundle      (shared with the planner)
    client_modules()                -> list[str]        which modules to broadcast
    on_ingest(states, actions, episode_id, episode_end)
    train_round(n_steps)            -> dict             metrics; called in a worker thread
"""

from __future__ import annotations

import asyncio
import time
from pathlib import Path
from typing import Any, Callable, Optional

import torch

from TrainingLite.hwm.comm.protocol import (
    MSG_ACK,
    MSG_BATCH_ACK,
    MSG_CLEAR_BUFFER,
    MSG_CLEAR_BUFFER_ACK,
    MSG_RAW_BATCH,
    MSG_TERMINATE,
    MSG_TERMINATE_ACK,
    MSG_TRAINING_INFO,
    pack_handshake,
    pack_simple,
    pack_weights,
    unpack_raw_batch,
)
from TrainingLite.hwm.models.bundle import ModelBundle
from TrainingLite.hwm.models.networks import build_models as build_model_bundle
from TrainingLite.hwm.paths import resolve_model_dir
from utilities.Settings import Settings
from TrainingLite.hwm.shared.memory_manager import MemoryManager
from TrainingLite.hwm.shared.observation import (
    build_LLD_observation_inference,
    lld_correction_alone,
    lld_delta,
)
from utilities.test_track import enabled as test_track_enabled
from TrainingLite.hwm.shared.wall_geometry import WallGeometry
from TrainingLite.rl_racing.tcp_utilities import pack_frame, read_frame


class HWMLearnerServer:
    def __init__(
        self,
        host: str,
        port: int,
        save_model_name: str,
        load_model_name: Optional[str] = None,
        device: str = "cpu",
        train_every_seconds: float = 0.0,
        grad_steps: int = 32,
        learning_starts: int = 500,
        autosave_interval_s: float = 60.0,
        status_line_callback: Optional[Callable[[str], None]] = None,
    ):
        self.host = host
        self.port = port
        self.save_model_name = str(save_model_name)
        self.load_model_name = load_model_name
        self.device = device
        self.train_every_seconds = float(train_every_seconds)
        self.grad_steps = int(grad_steps)
        self.learning_starts = int(learning_starts)
        self.autosave_interval_s = float(autosave_interval_s)
        self._status_cb = status_line_callback

        self.model_dir: Path = resolve_model_dir(self.save_model_name, create=True)

        # Shared components (identical construction on the planner, both from Settings).
        self.wall_geometry = WallGeometry(device=self.device)
        self.models: ModelBundle = self.build_models().to(self.device)
        if self.load_model_name is not None:
            loaded = self.models.load(resolve_model_dir(self.load_model_name), device=self.device)
            print(f"[server] Loaded modules {loaded} from '{self.load_model_name}'")
            self._weights_blob: Optional[bytes] = pack_frame(pack_weights(self.models.state_dicts(self.client_modules())))
        else:
            # Random init: first broadcast happens in _train_loop once learning_starts is reached.
            self._weights_blob = None
        self.memory = MemoryManager(device=self.device)
        lld = self.models["LLD"]
        self.memory.attach_projector(lld.attention_norm_projector)
        memory_path = resolve_model_dir(self.load_model_name or self.save_model_name) / "memory.pt"
        if memory_path.is_file():
            imported = self.memory.import_rows(torch.load(memory_path, weights_only=True))
            print(f"[server] Loaded {imported} memory rows from {memory_path}")
        # After .to(device) and checkpoint load, so each Adam tracks the tensors that training updates.
        # Only the LLD is trained here. The actor and the rest keep the weights they were
        # loaded with (or their random initialization).
        lr = float(Settings.HWM_LEARNING_RATE)
        self.optimizers: dict[str, torch.optim.Optimizer] = {
            "LLD": torch.optim.Adam(self.models["LLD"].parameters(), lr=lr)
        }

        # Networking / lifecycle.
        self._clients: set[asyncio.StreamWriter] = set()
        self._client_lock = asyncio.Lock()
        self._should_terminate = False
        self._train_event = asyncio.Event()
        self._pending_rows = 0
        self._train_lock = asyncio.Lock()
        self.total_rows = 0
        self.total_updates = 0
        self._last_autosave = time.time()
        self._latest_training_info: Optional[dict] = None

    # ------------------------------------------------------------------
    # INSERTION POINTS
    # ------------------------------------------------------------------
    def build_models(self) -> ModelBundle:
        return build_model_bundle()

    def client_modules(self) -> list[str]:
        modules = Settings.HWM_CLIENT_MODULES
        if isinstance(modules, str):
            return [modules]
        return list(modules)

    def on_ingest(self, states, actions, episode_id: int, episode_end: bool) -> int:
        """Called for every received raw batch. Default: append to memory."""
        return self.memory.add_batch(states, actions, episode_id, episode_end=episode_end)

    def train_round(self, n_steps: int) -> dict[str, Any]:
        """Run ``n_steps`` LLD gradient steps using ``self.memory``.

        Each step is one Adam update on ``HWM_LLD_BATCH_SIZE`` independent random
        transitions (whole-memory attention needs no sequence). With
        ``HWM_LLD_CORRECTION_ALONE`` the correction net is trained to match the delta on
        its own, and the retrieval encoder is trained to help that prediction. No other
        module is trained.
        Memory embeddings are refreshed once, after the last step. Runs in a worker thread;
        do not touch asyncio state here. Return metrics (JSON-serialisable) that are
        logged and sent to the planner as training_info.
        """
        n_steps = int(n_steps)
        if n_steps <= 0:
            return {"steps": 0}

        lld_opt = self.optimizers["LLD"]
        totals: dict[str, float] = {}
        for _ in range(n_steps):
            current, nxt, steps = self.memory.sample_transitions(
                int(Settings.HWM_LLD_BATCH_SIZE), int(Settings.HWM_LLD_SKIP_START_ROWS)
            )
            lld_loss = self._lld_nll(current, nxt, steps)
            lld_opt.zero_grad(set_to_none=True)
            lld_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.models["LLD"].parameters(), float(Settings.HWM_LLD_GRAD_CLIP))
            lld_opt.step()

            metrics = {
                "lld_loss": float(lld_loss.detach()),
                "lld_n_soft": float(self.models["LLD"].last_n_soft),
                "lld_top_keys": float(self.models["LLD"].last_top_keys),
                "lld_n_soft_norm": float(self.models["LLD"].last_n_soft_norm),
            }
            if self.models["LLD"].cov_head:
                metrics["lld_cov_n_soft"] = float(self.models["LLD"].last_cov_n_soft)
                metrics["lld_cov_top_keys"] = float(self.models["LLD"].last_cov_top_keys)
            for name, value in metrics.items():
                totals[name] = totals.get(name, 0.0) + value
        self.memory.refresh_embeddings()
        return {"steps": n_steps, **{name: value / n_steps for name, value in totals.items()}}

    def _lld_nll(self, current: torch.Tensor, nxt: torch.Tensor, steps: torch.Tensor) -> torch.Tensor:
        """Negative log-likelihood of the next-state delta of independent random transitions.

        ``current`` and ``nxt`` are ``(B, row_dim)`` rows ``t`` and ``t+1`` and ``steps`` the
        global memory indices of the current rows (``MemoryManager.sample_transitions``).
        Neighbors for each sample are the windows stored before its superstate, and the
        windows that start more than ``HWM_LLD_TRAIN_FUTURE_MARGIN`` rows after its next row.
        Neither the sample nor its immediate future can be retrieved. Each sample's history is
        read from memory, so samples near an episode start still see the rows before it;
        only rows before the episode start are zero, as at inference. Steps from the first
        ``HWM_LLD_SKIP_START_ROWS`` rows of an episode are not drawn.
        The predicted channels are yaw rate, body-frame velocity, steering, and
        heading. Position and slip angle are not targets: they follow from the
        current state together with those changes.

        With ``HWM_LLD_CORRECTION_ALONE`` the correction net is scored by the squared
        error of ``g(q)`` against that delta, and the retrieval encoder by the likelihood
        of the residual prediction with ``g`` held fixed.
        """
        if lld_correction_alone():
            return self._lld_alone_loss(current, nxt, steps)
        state_dim = int(Settings.HWM_STATE_DIM)
        hist_len = int(Settings.HWM_SUPER_STATE_SIZE) - 1
        past, _ = self.memory.preceding(steps, hist_len)

        obs = build_LLD_observation_inference(
            current[:, :state_dim],
            current[:, state_dim:],
            self.memory,
            episode_id=0,
            past_states=past,
            before_steps=steps,
            future_margin=int(Settings.HWM_LLD_TRAIN_FUTURE_MARGIN),
        )
        device = next(self.models["LLD"].parameters()).device
        obs = {name: value.to(device) for name, value in obs.items()}
        target = self._lld_target(current[:, :state_dim].to(device), nxt[:, :state_dim].to(device))
        lld = self.models["LLD"]
        dist = lld(obs)
        return -dist.log_prob(target).mean() + lld.encoder_nll(target) + lld.encoder_l1(target)

    def _lld_alone_loss(self, current: torch.Tensor, nxt: torch.Tensor, steps: torch.Tensor) -> torch.Tensor:
        """``g(q)`` matches the delta; the retrieval encoder is trained to help, with ``g`` held fixed.

        Neighbors are retrieved as in ``_lld_nll``. The squared error trains only ``g``. The
        likelihood trains the encoder through the attention weights of
        ``g(q) + sum_j w_j (delta_j - g(k_j))``, and ``g`` is detached there so it cannot
        rely on the retrieved samples.
        """
        state_dim = int(Settings.HWM_STATE_DIM)
        hist_len = int(Settings.HWM_SUPER_STATE_SIZE) - 1
        past, _ = self.memory.preceding(steps, hist_len)
        obs = build_LLD_observation_inference(
            current[:, :state_dim],
            current[:, state_dim:],
            self.memory,
            episode_id=0,
            past_states=past,
            before_steps=steps,
            future_margin=int(Settings.HWM_LLD_TRAIN_FUTURE_MARGIN),
        )
        device = next(self.models["LLD"].parameters()).device
        obs = {name: value.to(device) for name, value in obs.items()}
        target = self._lld_target(current[:, :state_dim].to(device), nxt[:, :state_dim].to(device))
        lld = self.models["LLD"]
        residual = lld._correct(obs["current_super_state"])
        dist = lld(obs, detach_correction=True)
        return lld.alone_loss(residual, dist, target) - dist.log_prob(target).mean() + lld.encoder_nll(target) + lld.encoder_l1(target)

    def _lld_target(self, states: torch.Tensor, next_states: torch.Tensor) -> torch.Tensor:
        return lld_delta(states, next_states)

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------
    async def _train_loop(self) -> None:
        while not self._should_terminate:
            if self.train_every_seconds > 0:
                await asyncio.sleep(self.train_every_seconds)
            else:
                await self._train_event.wait()
                self._train_event.clear()
            if self._should_terminate:
                break
            if test_track_enabled():
                # Rows may still be stored. No gradient step, so the LLD weights are not broadcast.
                self._pending_rows = 0
                continue
            if self.memory.total_steps < self.learning_starts:
                continue
            n_new = self._pending_rows
            self._pending_rows = 0
            if n_new <= 0 and self.train_every_seconds <= 0:
                continue

            t0 = time.time()
            async with self._train_lock:
                metrics = await asyncio.to_thread(self._train_round_safe, self.grad_steps)
            dt = time.time() - t0
            self.total_updates += int(metrics.get("steps", 0))

            payload = {
                "total_rows": int(self.total_rows),
                "total_updates": int(self.total_updates),
                "episodes": int(self.memory.num_episodes),
                "train_time_s": float(dt),
                **{k: v for k, v in metrics.items() if isinstance(v, (int, float, str, bool))},
            }
            self._latest_training_info = payload
            loss_text = "".join(
                f" {label}={float(payload[key]):.4f}"
                for key, label in (
                    ("lld_loss", "loss"),
                    ("lld_n_soft", "n_soft"),
                    ("lld_top_keys", "keys50"),
                    ("lld_n_soft_norm", "n_soft_norm"),
                    ("lld_cov_n_soft", "cov_n_soft"),
                    ("lld_cov_top_keys", "cov_keys50"),
                )
                if isinstance(payload.get(key), (int, float))
            )
            self._status(
                f"rows={payload['total_rows']} eps={payload['episodes']} "
                f"updates={payload['total_updates']}{loss_text} round={dt:.2f}s"
            )

            self._weights_blob = pack_frame(pack_weights(self.models.state_dicts(self.client_modules())))
            await self._broadcast_raw(self._weights_blob)
            await self._broadcast_raw(pack_frame(pack_simple(MSG_TRAINING_INFO, **payload)))

            if time.time() - self._last_autosave >= self.autosave_interval_s:
                self._save_models()

    def _train_round_safe(self, n_steps: int) -> dict[str, Any]:
        try:
            self.models.train()
            out = self.train_round(n_steps)
            return out if isinstance(out, dict) else {}
        except Exception as e:
            print(f"[server] train_round failed: {e}")
            return {"steps": 0, "error": str(e)}
        finally:
            self.models.eval()

    def _save_models(self) -> None:
        try:
            self.models.save(self.model_dir)
            # An LLD-sim run records its own predictions. Saving those would replace the
            # real transitions the model reads with a car that never got up to speed.
            snapshot = None if Settings.HWM_SIM_USE_LLD else self.memory.export_rows()
            if snapshot is not None:
                tmp = self.model_dir / "memory.pt.tmp"
                torch.save(snapshot, tmp)
                tmp.replace(self.model_dir / "memory.pt")
            self._last_autosave = time.time()
            print(f"[server] Saved models to {self.model_dir}")
        except Exception as e:
            print(f"[server] Failed to save models: {e}")

    def _status(self, msg: str) -> None:
        if self._status_cb is not None:
            self._status_cb(msg)
        else:
            print(f"[server] {msg}")

    # ------------------------------------------------------------------
    # Networking
    # ------------------------------------------------------------------
    async def _broadcast_raw(self, frame_bytes: bytes, exclude: Optional[asyncio.StreamWriter] = None) -> None:
        async with self._client_lock:
            dead = []
            for w in self._clients:
                if w is exclude:
                    continue
                try:
                    w.write(frame_bytes)
                    await w.drain()
                except Exception:
                    dead.append(w)
            for w in dead:
                self._clients.discard(w)

    async def handle_client(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        addr = writer.get_extra_info("peername")
        print(f"[server] Client connected: {addr}")
        async with self._client_lock:
            self._clients.add(writer)

        try:
            writer.write(pack_frame(pack_simple(MSG_ACK, msg="connected")))
            writer.write(pack_frame(pack_handshake(self.save_model_name, str(Settings.MAP_NAME))))
            if self._weights_blob is not None:
                writer.write(self._weights_blob)
            if self._latest_training_info is not None:
                writer.write(pack_frame(pack_simple(MSG_TRAINING_INFO, **self._latest_training_info)))
            await writer.drain()
        except Exception as e:
            print(f"[server] Failed initial send to {addr}: {e}")

        try:
            while not self._should_terminate:
                msg = await read_frame(reader)
                typ = msg.get("type")
                data = msg.get("data", {}) or {}

                if typ == MSG_RAW_BATCH:
                    states, actions, episode_id, episode_end = unpack_raw_batch(data)
                    n = self.on_ingest(states, actions, episode_id, episode_end)
                    self.total_rows += n
                    self._pending_rows += n
                    self._train_event.set()
                    try:
                        writer.write(pack_frame(pack_simple(MSG_BATCH_ACK, n=n, episode_id=episode_id)))
                        await writer.drain()
                    except Exception:
                        pass

                elif typ == MSG_CLEAR_BUFFER:
                    self.memory.clear()
                    self._pending_rows = 0
                    print("[server] Cleared memory (requested by planner)")
                    try:
                        writer.write(pack_frame(pack_simple(MSG_CLEAR_BUFFER_ACK)))
                        await writer.drain()
                    except Exception:
                        pass

                elif typ == MSG_TERMINATE:
                    print("[server] Received terminate from planner")
                    async with self._train_lock:
                        self._save_models()
                    # Ack the requester first (it stops reading once it sees a terminate),
                    # then tell any other connected planners to stop.
                    try:
                        writer.write(pack_frame(pack_simple(MSG_TERMINATE_ACK, msg="terminating")))
                        await writer.drain()
                    except Exception:
                        pass
                    await self._broadcast_raw(
                        pack_frame(pack_simple(MSG_TERMINATE, reason="terminate_requested_by_planner")),
                        exclude=writer,
                    )
                    self._should_terminate = True
                    self._train_event.set()
                    break
        except (asyncio.IncompleteReadError, ConnectionResetError):
            pass
        except Exception as e:
            print(f"[server] Client loop error: {e}")
        finally:
            try:
                writer.close()
                await writer.wait_closed()
            except Exception:
                pass
            async with self._client_lock:
                self._clients.discard(writer)
            print(f"[server] Client disconnected: {addr}")

    async def run(self) -> None:
        train_task = asyncio.create_task(self._train_loop())
        server = await asyncio.start_server(self.handle_client, self.host, self.port)
        addrs = ", ".join(str(s.getsockname()) for s in server.sockets or [])
        print(
            f"[server] Listening on {addrs} | save_model='{self.save_model_name}' "
            f"| load_model='{self.load_model_name}' | device='{self.device}' | map='{Settings.MAP_NAME}'"
        )

        async def monitor() -> None:
            while not self._should_terminate:
                await asyncio.sleep(0.5)

        try:
            async with server:
                await monitor()
        except (asyncio.CancelledError, KeyboardInterrupt):
            print("\n[server] Interrupted, shutting down...")
        finally:
            self._should_terminate = True
            self._train_event.set()
            if not train_task.done():
                train_task.cancel()
                try:
                    await train_task
                except asyncio.CancelledError:
                    pass
            async with self._client_lock:
                for w in list(self._clients):
                    try:
                        w.close()
                        await w.wait_closed()
                    except Exception:
                        pass
                self._clients.clear()
            server.close()
            await server.wait_closed()
            self._save_models()
            print("[server] Shutdown complete")
