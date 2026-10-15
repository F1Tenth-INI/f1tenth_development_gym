"""Hierarchical world-model planner (client side).

Mirrors the SAC actor lifecycle in ``TrainingLite/rl_racing/sac_agent_planner.py``:

* ``Settings.HWM_INFERENCE_MODEL_NAME`` is None  -> TRAINING: connect to the learner,
  stream ``[state, action]`` rows (10-d car state + the raw model action, before
  clipping or denormalization, nothing else from the observation dict), apply broadcast
  weights, Pure Pursuit until the first weights arrive.
* ``Settings.HWM_INFERENCE_MODEL_NAME`` is a name -> INFERENCE: load
  ``TrainingLite/hwm/models/<name>/`` from disk, no TCP.

Insertion points (the only methods meant to be edited):

    build_observation(state)          -> torch.Tensor      (delegates to shared/observation.py)
    select_action(obs, state)         -> np.ndarray [-1,1]^2

Everything else is transport, bookkeeping and fallback plumbing.
"""

from __future__ import annotations

import copy
import os
import signal
import sys
import threading
from typing import Any, Dict, Optional

import numpy as np
import torch

from Control_Toolkit_ASF.Controllers import template_planner
from Control_Toolkit_ASF.Controllers.PurePursuit.pp_planner import PurePursuitPlanner
from TrainingLite.hwm.comm.tcp_client import HWMTCPClient
from TrainingLite.hwm.lld_preview import BackgroundPreview, LLDPreview
from TrainingLite.hwm.paths import resolve_model_dir
from TrainingLite.hwm.models.bundle import ModelBundle
from TrainingLite.hwm.models.networks import build_models
from TrainingLite.hwm.shared.memory_manager import MemoryManager
from TrainingLite.hwm.shared.observation import (
    _normalize_rows,
    _normalize_states,
    build_actor_observation,
    build_LLD_observation_inference,
    lld_rows_for_super,
)
from TrainingLite.hwm.shared.wall_geometry import WallGeometry
from utilities.Settings import Settings
from utilities.test_track import command as test_track_command
from utilities.test_track import enabled as test_track_enabled
from utilities.test_track import store_memory as test_track_store_memory

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

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True)

try:  # keep the sim process single-threaded for torch; may already be fixed by earlier imports
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
except RuntimeError:
    pass


class HirarchicalPlanner(template_planner):
    BOOTSTRAP_ROWS = 2

    # ------------------------------------------------------------------
    # Logging
    # ------------------------------------------------------------------
    def _log_info(self, message: str) -> None:
        print(message)

    def _log_debug(self, message: str) -> None:
        if Settings.HWM_AGENT_DEBUG:
            print(message)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    def __init__(self):
        super().__init__()
        self._log_info("Initializing HirarchicalPlanner")

        self.inference_model_name: Optional[str] = Settings.HWM_INFERENCE_MODEL_NAME
        self.training_mode = self.inference_model_name is None

        # Shared components (identical construction on the learner, both from Settings).
        self.wall_geometry = WallGeometry()
        self.models: ModelBundle = build_models().to("cpu").eval()
        self.memory = MemoryManager(device="cpu")
        lld = self.models["LLD"]
        self.memory.attach_projector(lld.attention_norm_projector)
        # Bumped at every episode start so preview results from older episodes are ignored.
        self._generation = 0
        # The background preview rolls out its own copies of the actor and the LLD. The control thread
        # also runs the LLD every step, and a shared module produces garbage paths.
        self._display_lock = threading.Lock()
        self._display_actor = copy.deepcopy(self.models["actor"]).eval()
        self._display_lld = copy.deepcopy(self.models["LLD"]).eval()

        self.client: Optional[HWMTCPClient] = None
        self.latest_training_info: Optional[Dict[str, Any]] = None
        self._received_weights = not self.training_mode
        self._warned_no_weights = False
        self._server_handshake: Optional[dict] = None

        if self.training_mode:
            self._log_info("[HirarchicalPlanner] Mode: TRAINING (weights from learner over TCP)")
            host = str(getattr(Settings, "LEARNER_TCP_HOST", "127.0.0.1"))
            port = int(getattr(Settings, "LEARNER_TCP_PORT", 5555))
            self.client = HWMTCPClient(host=host, port=port)
            self.client.start()
            self._log_info(f"[HirarchicalPlanner] TCP client -> {host}:{port}")
        else:
            self._log_info(f"[HirarchicalPlanner] Mode: INFERENCE (model: {self.inference_model_name})")
            self._init_inference_models()
        if Settings.HWM_SIM_SPEED is not None:
            speed = float(Settings.HWM_SIM_SPEED)
            if speed <= 0:
                raise ValueError(f"HWM_SIM_SPEED must be positive or None, got {speed}")
            # The sim loop sleeps each control step up to 1 / MAX_SIM_FREQUENCY.
            Settings.MAX_SIM_FREQUENCY = speed / float(Settings.TIMESTEP_CONTROL)
            self._log_info(
                f"[HirarchicalPlanner] Simulation capped at {speed:g}x real time ({Settings.MAX_SIM_FREQUENCY:g} Hz)"
            )

        # Control plumbing (same as SAC actor).
        self.action_denormalization_array = np.asarray(Settings.HWM_ACTION_DENORM, dtype=np.float32)
        self.lowpass_alpha = 1.0
        self.angular_control = 0.0
        self.translational_control = 0.0
        self.prev_angular_control = 0.0
        self.prev_translational_control = 0.0
        self.terminate_server_after_simulation = True
        self.autonomous_driving = not Settings.HWM_PP_DRIVES

        self.fallback_planner = PurePursuitPlanner()
        self._controller_observation: Optional[Dict[str, Any]] = None
        self.lld_preview: Optional[LLDPreview] = None
        self.preview_worker: Optional[BackgroundPreview] = None
        self._preview_submitted = -int(Settings.HWM_LLD_PREVIEW_EVERY)
        if Settings.HWM_LLD_PREVIEW:
            self.lld_preview = LLDPreview(self._display_lld, self._display_actor, self.memory, self.wall_geometry)
            self.preview_worker = BackgroundPreview(self._preview_job)

        # Episode / streaming bookkeeping.
        self.control_index = 0
        self._episode_id = 0
        # Test track without memory storage: the drive's own [state, action] rows, which the
        # preview needs for its history and the control-delay line.
        self._test_rows: list[np.ndarray] = []
        self._stream_send_idx = 0
        self._stream_batch_size = int(Settings.HWM_STREAM_BATCH_SIZE)
        self._bootstrap_sent = False

        self.reset()

    def reset(self):
        self.control_index = 0
        self.angular_control = 0.0
        self.translational_control = 0.0
        self.prev_angular_control = 0.0
        self.prev_translational_control = 0.0
        # Episode rows are kept until on_step_end() sees done; see _reset_episode_state().
        self._generation += 1
        self._preview_submitted = -int(Settings.HWM_LLD_PREVIEW_EVERY)
    def _ensure_waypoints(self) -> None:
        waypoints = getattr(self.waypoint_utils, "waypoints", None)
        if waypoints is None or getattr(self, "_waypoints_n", 0) == len(waypoints):
            return
        self.waypoints = torch.as_tensor(np.asarray(waypoints), dtype=torch.float32)  # (N, 10)

    def _refresh_display_models(self) -> None:
        """Copy the live actor and LLD into the modules the background paths roll out."""
        with self._display_lock:
            self._display_actor.load_state_dict(self.models["actor"].state_dict())
            self._display_lld.load_state_dict(self.models["LLD"].state_dict())

    def _preview_job(self, **kwargs: Any) -> tuple[np.ndarray, torch.Tensor, torch.Tensor]:
        self._refresh_display_models()
        return self.lld_preview.compute(**kwargs)
    
    def nearest_waypoint_index(self, state: torch.Tensor) -> torch.Tensor:
        """Closest raceline index for each state ``(B, 10)``. Returns ``(B,)`` long."""
        #self._ensure_waypoints()
        pos = state[:, [POSE_X_IDX, POSE_Y_IDX]]                 # (B, 2)
        dist = (self._waypoint_xy.unsqueeze(0) - pos.unsqueeze(1)).pow(2).sum(-1)  # (B, N)
        return dist.argmin(dim=-1)

    # ------------------------------------------------------------------
    # INSERTION POINTS
    # ------------------------------------------------------------------
    def build_observation(self, state: np.ndarray) -> dict[str, torch.Tensor]:
        """Actor input for the current raw state. Shared implementation with the learner."""
        return build_actor_observation(
            state=state,
            memory=self.memory,
            wall_geometry=self.wall_geometry,
            episode_id=self._episode_id,
        )
    

    def select_action(self, obs: dict[str, torch.Tensor], state: np.ndarray) -> np.ndarray:
        """Policy in network output units. ``±1`` is ``±action_denorm`` on the car.

        Default: one sample of the actor (``HWM_SAMPLE_ACTIONS``) or its squashed
        Gaussian mean once weights are available, Pure Pursuit before that when
        ``HWM_PP_UNTIL_WEIGHTS`` is set. Pure Pursuit is not clipped, so it can leave ``[-1, 1]``.
        """
        if not self._received_weights and Settings.HWM_PP_UNTIL_WEIGHTS:
            self._ensure_waypoints()
            if not self._warned_no_weights:
                self._log_info("[HirarchicalPlanner] No weights yet; using Pure Pursuit fallback")
                self._warned_no_weights = True
            return self._fallback_action()

        with torch.no_grad():
            actor = self.models["actor"]
            dist = actor(obs)
            if Settings.HWM_SAMPLE_ACTIONS:
                action = dist.sample()[0]
            else:
                action = actor.deterministic_action(dist)[0]
        return action.cpu().numpy().astype(np.float32).reshape(-1)

    @torch.no_grad()
    def rollout_lld(self, state, actions):
        """LLD mean rollout of action sequences.

        ``actions`` is ``(B, N, 2)`` or ``(N, 2)``, the command chosen at each step, in network
        units. ``state`` is the shared start ``(state_dim,)`` or one start per sequence ``(B, state_dim)``.

        Returns ``states`` ``(B, N + 1, state_dim)``, and ``n_soft`` and ``n_soft_norm`` ``(B, N)``.
        Row ``k`` of those two is the attention mass at ``states[:, k]`` under ``actions[:, k]``.
        ``n_soft`` counts keys by ``exp(-distance / temperature)``, plus 1 for the dummy.
        ``n_soft_norm`` divides each key's kernel by that key's stored n_soft before summing.

        The key bank is read once, as views. Imagined rows stay in a small history and are not
        written into memory. The control delay is already in the training pairs, so ``actions``
        is not shifted.
        """
        actions = torch.as_tensor(actions, dtype=torch.float32, device=self.memory.device)
        if actions.ndim == 2:
            actions = actions.unsqueeze(0)
        batch, horizon, _ = (int(actions.shape[0]), int(actions.shape[1]), int(actions.shape[2]))
        state_dim = int(Settings.HWM_STATE_DIM)
        width = int(Settings.HWM_SUPER_STATE_SIZE)
        s = torch.as_tensor(state, dtype=torch.float32, device=self.memory.device)
        if s.ndim == 1:
            s = s.reshape(1, -1).expand(batch, -1).contiguous()
        elif int(s.shape[0]) != batch:
            raise ValueError(f"state batch {int(s.shape[0])} does not match actions batch {batch}")

        lld = self.models["LLD"]
        if lld.extra_dim:
            raise ValueError(
                "rollout_lld assumes the residual net has no retrieved inputs "
                "(HWM_LLD_CORRECTION_INPUT_RESIDUAL and HWM_LLD_CORRECTION_INPUT_COV are off)"
            )
        past, _ = self.memory.recent(self._episode_id, width - 1)
        history = lld_rows_for_super(_normalize_rows(past.unsqueeze(0), state_dim), state_dim)
        history = history.expand(batch, -1, -1).contiguous()
        bank = self.memory.lld_bank()
        if bank is None:
            key_emb = s.new_zeros((0, lld.attention_norm_projector[-1].out_features))
            tokens = lld.dummy_value.unsqueeze(0)
            density = s.new_zeros((0,))
            valid = torch.zeros((0,), dtype=torch.bool, device=s.device)
        else:
            supers, deltas, density, rows = bank
            cutoff = int(self.memory.next_step) - (width - 1)
            valid = rows + 1 < cutoff
            key_emb = lld.attention_norm_projector(supers)
            residuals = deltas - lld._correct(supers)
            tokens = torch.cat((residuals, lld.dummy_value.unsqueeze(0)), dim=0)

        states = [s]
        n_softs = []
        n_soft_norms = []
        for k in range(horizon):
            a = actions[:, k]
            current = lld_rows_for_super(
                torch.cat((_normalize_states(s), a), dim=-1), state_dim
            )
            query = torch.cat((history.flatten(start_dim=1), current), dim=-1)
            delta, n_soft, n_soft_norm = lld.mean_delta(query, key_emb, tokens, density, valid)
            s = lld.reconstruct_next_state(s, delta)
            history = torch.cat((history[:, 1:], current.unsqueeze(1)), dim=1)
            states.append(s)
            n_softs.append(n_soft)
            n_soft_norms.append(n_soft_norm)
        n_soft = torch.stack(n_softs, dim=1) if n_softs else s.new_zeros((batch, 0))
        n_soft_norm = torch.stack(n_soft_norms, dim=1) if n_soft_norms else n_soft
        return torch.stack(states, dim=1), n_soft, n_soft_norm

    def rollout_pacejka(self, state, actions):
        """states (B, N+1, 10) for the same actions. Applies the command from 2 control steps ago. For testing the Planner with ground truth model in simulation."""
        import jax
        actions = np.asarray(actions, dtype=np.float32)
        if actions.ndim == 2:
            actions = actions[None]
        batch, horizon, _ = actions.shape
        delay = int(round(Settings.CONTROL_DELAY / Settings.TIMESTEP_CONTROL))  # 2
        rows, _ = self.memory.recent(self._episode_id, delay)
        pending = np.broadcast_to(rows[:, int(Settings.HWM_STATE_DIM):].numpy(), (batch, delay, 2)).copy()
        s = np.broadcast_to(np.asarray(state, dtype=np.float32), (batch, int(Settings.HWM_STATE_DIM))).copy()
        denorm = self.action_denormalization_array
        cap = float(Settings.HWM_MAX_TRANSLATIONAL_CONTROL)
        step = self._pacejka_batch_step()  # jax.jit(jax.vmap(one control step))
        states = [s.copy()]
        for k in range(horizon):
            pending = np.concatenate((pending, actions[:, k, None, :]), axis=1)
            applied = pending[:, 0] * denorm
            pending = pending[:, 1:]
            applied[:, 1] = np.clip(applied[:, 1], -cap, cap)
            s = np.asarray(step(jnp.asarray(s), jnp.asarray(applied)), dtype=np.float32)
            s[:, POSE_THETA_IDX] = np.arctan2(s[:, POSE_THETA_SIN_IDX], s[:, POSE_THETA_COS_IDX])
            states.append(s.copy())
        return torch.as_tensor(np.stack(states, axis=1))

    def _log_attention(self) -> None:
        """Log the attention statistics of the last LLD forward pass (main head, and the covariance head if on)."""
        lld = self.models["LLD"]
        tag = " (single encoder)" if lld.standalone else ""
        text = f"[attention]{tag} keys for top 50% weight: {float(lld.last_top_keys):.1f} n_soft: {float(lld.last_n_soft):.2f}"
        text += f" n_soft_norm: {float(lld.last_n_soft_norm):.2f} dummy: {100.0 * float(lld.last_dummy_share):.1f}%"
        if lld.cov_head:
            text += f" | cov head: keys50 {float(lld.last_cov_top_keys):.1f} n_soft {float(lld.last_cov_n_soft):.2f} dummy {100.0 * float(lld.last_cov_dummy_share):.1f}%"
        self._log_info(text)

    @torch.no_grad()
    def _log_attention_test_track(self) -> None:
        """Test track only: one LLD pass on the last stored row, to log the attention statistics every step.

        Needs the drive in memory (``TEST_TRACK_STORE_MEMORY``); the output is not used otherwise.
        """
        width = int(Settings.HWM_SUPER_STATE_SIZE)
        state_dim = int(Settings.HWM_STATE_DIM)
        rows, mask = self.memory.recent(self._episode_id, width)
        if not bool(mask[-1]):
            return
        obs = build_LLD_observation_inference(
            rows[-1:, :state_dim],
            rows[-1:, state_dim:],
            self.memory,
            self._episode_id,
            past_states=rows[None, :-1],
            before_steps=torch.tensor([self.memory.next_step - 1]),
        )
        self.models["LLD"](obs)
        self._log_attention()

    @torch.no_grad()
    def lld_dynamics_step(self, state: np.ndarray, physical_control: np.ndarray) -> np.ndarray:
        """Next car state from the LLD, one control step.

        ``state`` is the simulator state before the step. The action is the one just
        stored with it (network units), which is the pair the LLD is trained on.
        ``HWM_SIM_LLD_SAMPLE`` draws one delta; otherwise the mean is used.
        """
        state_dim = int(Settings.HWM_STATE_DIM)
        width = int(Settings.HWM_SUPER_STATE_SIZE)
        rows, mask = self.memory.recent(self._episode_id, width)
        state_t = torch.as_tensor(np.asarray(state, dtype=np.float32), dtype=torch.float32).reshape(1, -1)
        if bool(mask[-1]):
            action_t = rows[-1:, state_dim:]
            past = rows[None, :-1]
            before = torch.tensor([self.memory.next_step - 1])
        else:
            action_t = torch.as_tensor(
                np.asarray(physical_control, dtype=np.float32) / self.action_denormalization_array,
                dtype=torch.float32,
            ).reshape(1, -1)
            past = torch.zeros(1, width - 1, state_dim + int(Settings.HWM_ACTION_DIM))
            before = None
        obs = build_LLD_observation_inference(
            state_t, action_t, self.memory, self._episode_id, past_states=past, before_steps=before
        )
        prediction = self.models["LLD"](obs)
        delta = prediction.sample() if Settings.HWM_SIM_LLD_SAMPLE else prediction.mean
        nxt = self.models["LLD"].reconstruct_next_state(state_t, delta)
        return nxt[0].cpu().numpy().astype(np.float64)

    # ------------------------------------------------------------------
    # Control loop
    # ------------------------------------------------------------------
    def process_observation(self, controller_observation: Dict[str, Any]) -> tuple[float, float]:
        self._controller_observation = controller_observation
        if test_track_enabled():
            return self._test_track_step(controller_observation)
        self._maybe_handle_server_terminate()
        self._sync_from_server()

        if not self.autonomous_driving:
            action = self._fallback_action(controller_observation)
            self._apply_normalized_action(action)
            state = self.get_car_state(controller_observation)
            self._maybe_update_lld_preview(state, controller_observation)
            self._record_step(state, action)
            return self.angular_control, self.translational_control

        state = self.get_car_state(controller_observation)
        obs = self.build_observation(state)
        action = np.asarray(self.select_action(obs, state), dtype=np.float32).reshape(-1)
        self._apply_normalized_action(action)
        self._maybe_update_lld_preview(state, controller_observation)
        # Network-scale action, unclipped. Pure Pursuit fallback is physical / action_denorm.
        self._record_step(state, action)
        return self.angular_control, self.translational_control

    def _test_track_step(self, controller_observation: Dict[str, Any]) -> tuple[float, float]:
        """Constant acceleration and a growing sine steer. The LLD weights are left as they are."""
        self._maybe_handle_server_terminate()
        self._sync_from_server()
        state = self.get_car_state(controller_observation)
        steer, accel = test_track_command(self.control_index * float(Settings.TIMESTEP_CONTROL))
        action = np.array([steer, accel], dtype=np.float32) / self.action_denormalization_array
        self.angular_control = float(steer)
        self.translational_control = float(accel)
        self.prev_angular_control = self.angular_control
        self.prev_translational_control = self.translational_control
        self._maybe_update_lld_preview(state, controller_observation)
        if test_track_store_memory():
            self._log_attention_test_track()
            self._record_step(state, action)
        else:
            keep = max(int(Settings.HWM_SUPER_STATE_SIZE), 2 * self.lld_preview.delay + 1) if self.lld_preview else 0
            row = np.concatenate((np.asarray(state, dtype=np.float32), action.astype(np.float32)))
            self._test_rows = (self._test_rows + [row])[-keep:] if keep else []
            self.control_index += 1
        return self.angular_control, self.translational_control

    def _apply_normalized_action(self, action: np.ndarray) -> None:
        steering, accel = np.asarray(action, dtype=np.float32).reshape(-1) * self.action_denormalization_array
        self.angular_control = self.lowpass_alpha * float(steering) + (1 - self.lowpass_alpha) * self.prev_angular_control
        self.translational_control = self.lowpass_alpha * float(accel) + (1 - self.lowpass_alpha) * self.prev_translational_control
        max_translational = float(Settings.HWM_MAX_TRANSLATIONAL_CONTROL)
        self.translational_control = float(np.clip(self.translational_control, -max_translational, max_translational))
        self.prev_angular_control = self.angular_control
        self.prev_translational_control = self.translational_control

    def _maybe_update_lld_preview(self, state: np.ndarray, controller_observation: Dict[str, Any]) -> None:
        """Draw the newest background preview and start the next one; before the step is recorded.

        A new preview starts at most every ``HWM_LLD_PREVIEW_EVERY`` steps, once LLD
        weights exist and the previous one has finished.
        """
        if self.preview_worker is None or not (self._received_weights or test_track_enabled()):
            return
        result = self.preview_worker.pop_result()
        if result is not None and result.generation != self._generation:
            result = None
        if result is not None and self.render_utils is not None:
            age = self.control_index - result.control_index
            plan_states, lld_states, mean_states, residual_states = result.value
            self.lld_preview.render(self.render_utils, plan_states, lld_states, age, mean_states, residual_states)
            self.render_utils.set_label_dict({"LLD preview of": result.source})
        else:
            self.lld_preview.advance(self.render_utils)
        if self.control_index - self._preview_submitted < int(Settings.HWM_LLD_PREVIEW_EVERY):
            return
        on_test_track = test_track_enabled()
        if on_test_track:
            source = "test track"
        elif self.autonomous_driving:
            source = "actor"
        else:
            source = "Pure Pursuit"
        started = self.preview_worker.submit(
            self._generation,
            self.control_index,
            source,
            state=np.array(state, dtype=np.float32),
            past=self.lld_preview.past_rows(
                self._episode_id, self._test_rows if on_test_track and not test_track_store_memory() else None
            ),
            controller_observation=dict(controller_observation),
            waypoint_utils=self.waypoint_utils,
            nearest_index=int(self.waypoint_utils.nearest_waypoint_index or 0),
            correcting_index=self.fallback_planner.correcting_index,
            use_actor=self.autonomous_driving and not on_test_track,
            scripted_time=self.control_index * float(Settings.TIMESTEP_CONTROL) if on_test_track else None,
        )
        if started:
            self._preview_submitted = self.control_index

    def _record_step(self, state: np.ndarray, action: np.ndarray) -> None:
        """Append one ``[state, action]`` row in network output units and stream it."""
        self.memory.add_step(state, np.asarray(action, dtype=np.float32).reshape(-1), self._episode_id)
        self._maybe_bootstrap_send()
        self._maybe_stream_send()
        self.control_index += 1

    def on_step_end(self, driver_obs: Dict[str, Any]) -> None:
        """Called by CarSystem after the physics step; only episode boundaries matter here."""
        self._maybe_handle_server_terminate()

        done = bool(driver_obs.get("done", False))
        if done or self.control_index >= Settings.MAX_EPISODE_LENGTH:
            n_rows = self.memory.open_length(self._episode_id)
            self._flush_stream_send(episode_end=True)
            self.memory.end_episode(self._episode_id)
            self._log_debug(f"[HirarchicalPlanner] Episode {self._episode_id} done: {n_rows} rows")
            self._reset_episode_state()

    def on_simulation_end(self, collision: bool = False) -> None:
        if self.terminate_server_after_simulation and self.training_mode and self.client is not None:
            self._terminate_server_with_retry()

    def close(self) -> None:
        if self.preview_worker is not None:
            self.preview_worker.close()
        if self.client is not None:
            try:
                self.client.close()
            except Exception:
                pass

    # ------------------------------------------------------------------
    # Fallback
    # ------------------------------------------------------------------
    def _fallback_action(self, controller_observation: Optional[Dict[str, Any]] = None) -> np.ndarray:
        self.fallback_planner.waypoint_utils = self.waypoint_utils
        self.fallback_planner.lidar_utils = self.lidar_utils
        if controller_observation is None:
            controller_observation = self._controller_observation
        control = np.asarray(self.fallback_planner.process_observation(controller_observation), dtype=np.float32)
        return (control / self.action_denormalization_array).astype(np.float32)

    # ------------------------------------------------------------------
    # Models
    # ------------------------------------------------------------------
    def _init_inference_models(self) -> None:
        model_dir = resolve_model_dir(self.inference_model_name)
        loaded = self.models.load(model_dir, device="cpu")
        self.models.eval()
        if "LLD" in loaded:
            self.memory.refresh_embeddings()
        if "actor" not in loaded:
            raise FileNotFoundError(f"[HirarchicalPlanner] actor.pt not found in {model_dir}")
        self._seed_memory(self.inference_model_name)
        self._log_info(f"[HirarchicalPlanner] Loaded modules {loaded} from {model_dir}")

    def _seed_memory(self, model_name: Optional[str]) -> None:
        """Load the learner's saved neighbor bank once, while this memory is still empty.

        The LLD's prediction is the average of stored transitions. With an empty bank the
        only token is the zero dummy, the predicted step is zero, and a car that starts at
        rest never moves. Driving that zero step back into memory keeps it there.
        """
        if not model_name or self.memory.total_steps > 0:
            return
        path = resolve_model_dir(str(model_name)) / "memory.pt"
        if not path.is_file():
            if Settings.HWM_SIM_USE_LLD:
                self._log_info(
                    "[HirarchicalPlanner] LLD sim memory is empty, so the predicted step is zero and the car stays still. "
                    "memory.pt is written when the learner saves a run driven by the real dynamics."
                )
            return
        imported = self.memory.import_rows(torch.load(path, weights_only=True))
        self.memory.refresh_embeddings()
        self._log_info(f"[HirarchicalPlanner] Loaded {imported} memory rows from {path.name}")

    def _apply_weights(self, sds: dict) -> None:
        # The test track keeps the LLD that was loaded at the start. A later broadcast is a training update.
        if test_track_enabled() and self._received_weights:
            return
        with self._display_lock:
            loaded = self.models.load_state_dicts(sds, strict=True)
            self.models.eval()
            if "LLD" in loaded:
                self.memory.refresh_embeddings()
        if loaded:
            self._received_weights = True
            self._log_debug(f"[HirarchicalPlanner] Weights updated: {loaded}")

    # ------------------------------------------------------------------
    # Networking
    # ------------------------------------------------------------------
    def _sync_from_server(self) -> None:
        if not self.training_mode or self.client is None:
            return
        handshake = self.client.pop_handshake()
        if isinstance(handshake, dict):
            self._server_handshake = handshake
            self._seed_memory(handshake.get("model_name"))
            if handshake.get("map_name") and handshake["map_name"] != str(Settings.MAP_NAME):
                self._log_info(
                    f"[HirarchicalPlanner] WARNING: learner map '{handshake['map_name']}' "
                    f"!= client map '{Settings.MAP_NAME}'"
                )
        info = self.client.pop_latest_training_info()
        if isinstance(info, dict):
            self.latest_training_info = info
        sds = self.client.pop_latest_weights()
        if sds:
            self._apply_weights(sds)

    def _can_stream(self) -> bool:
        return self.training_mode and self.client is not None

    def _send_rows(self, start: int, end: int, episode_end: bool = False) -> bool:
        states, actions = self.memory.slice_open(self._episode_id, start, end)
        if states.shape[0] == 0:
            return True
        return self.client.send_raw_batch(states, actions, self._episode_id, episode_end=episode_end)

    def _maybe_bootstrap_send(self) -> None:
        """Send the first rows early (once per planner lifetime) so the learner can init."""
        if not self._can_stream() or self._bootstrap_sent:
            return
        n = self.memory.open_length(self._episode_id)
        if n < self.BOOTSTRAP_ROWS:
            return
        if self._send_rows(0, n):
            self._stream_send_idx = n
            self._bootstrap_sent = True

    def _maybe_stream_send(self) -> None:
        if not self._can_stream():
            return
        n = self.memory.open_length(self._episode_id)
        if n - self._stream_send_idx < self._stream_batch_size:
            return
        end = self._stream_send_idx + self._stream_batch_size
        if self._send_rows(self._stream_send_idx, end):
            self._stream_send_idx = end

    def _flush_stream_send(self, episode_end: bool = False) -> None:
        if not self._can_stream():
            return
        n = self.memory.open_length(self._episode_id)
        min_len = int(Settings.HWM_MIN_EPISODE_END_BATCH_SIZE)
        if episode_end and n < min_len:
            self._log_debug(f"[HirarchicalPlanner] Skipping short episode ({n} rows, min={min_len})")
            self._stream_send_idx = n
            return
        if self._send_rows(self._stream_send_idx, n, episode_end=episode_end):
            self._stream_send_idx = n

    def _reset_episode_state(self) -> None:
        self._episode_id += 1
        self._stream_send_idx = 0
        self.control_index = 0
        self._generation += 1
        self._preview_submitted = -int(Settings.HWM_LLD_PREVIEW_EVERY)
        self._test_rows = []

    def _terminate_server_with_retry(self) -> None:
        delivered = False
        try:
            for _ in range(2):
                delivered = bool(self.client.send_terminate(wait_for_ack=True, ack_timeout=2.0, urgent=True))
                if delivered:
                    break
            if not delivered:
                self.client.send_terminate(wait_for_ack=False, urgent=True)
            self._log_info("[HirarchicalPlanner] Sent terminate message to server")
        except Exception as e:
            self._log_info(f"[HirarchicalPlanner] Failed to send terminate message: {e}")
        finally:
            self.close()

    def _maybe_handle_server_terminate(self) -> None:
        if self.client is None:
            return
        payload = self.client.pop_server_terminate()
        if payload is None:
            return
        reason = payload.get("reason", "server_requested_terminate")
        self._log_info(f"[HirarchicalPlanner] Received terminate from server: {reason}. Stopping client process.")
        self.close()
        os.kill(os.getpid(), signal.SIGTERM)
