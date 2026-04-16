#!/usr/bin/env python3
"""
Agent client that replaces the PurePursuit planner and communicates with the
learner_server over TCP to:
  - receive policy weights
  - compute actions locally (CPU) using SB3 SAC policy
  - stream experience tuples (s, a, r, s', done, info) back to the learner

Drop-in replacement for your controller class (planner): implements
`process_observation(...)` and exposes `on_step_end(reward, done, info)` which
you should call from your env step once reward/done are known.

Integration (minimal diff):
1) Construct the planner:
     driver.planner = RLAgentPlanner(host="127.0.0.1", port=5555, actor_id=0)
2) In `RacingEnv.step` *after* you compute `reward`, `terminated`, `truncated`,
   and `info`, add:
     if hasattr(driver, "planner") and hasattr(driver.planner, "on_step_end"):
         driver.planner.on_step_end(float(reward), bool(terminated or truncated), info)

This mirrors the actor→learner design we discussed.
"""
from __future__ import annotations
import os
import sys
import importlib.util

# Configure unbuffered output for immediate print statements (important for ROS nodes)
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(line_buffering=True)
else:
    # Fallback for older Python versions
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, line_buffering=True)

from typing import Optional, Dict, Any, Callable
from collections import deque
import numpy as np
import torch

from stable_baselines3 import SAC

from Control_Toolkit_ASF.Controllers import template_planner
from Control_Toolkit_ASF.Controllers.PurePursuit.pp_planner import PurePursuitPlanner

from utilities.waypoint_utils import *

root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(root_dir)
        
# Your project imports
from utilities.state_utilities import *  # indices like LINEAR_VEL_X_IDX, etc.


try:
    from TrainingLite.rl_racing.tcp_client import _TCPActorClient
    from TrainingLite.rl_racing.sac_utilities import SacUtilities, TransitionLogger
except ModuleNotFoundError:
    from f1tenth_development_gym.TrainingLite.rl_racing.tcp_client import _TCPActorClient
    from f1tenth_development_gym.TrainingLite.rl_racing.sac_utilities import SacUtilities, TransitionLogger
from utilities.CurriculumSupervisor import CurriculumSupervisor
torch.set_num_threads(1)          # intra-op parallelism
torch.set_num_interop_threads(1)  # inter-op parallelism

# ------------------------
# RL Agent Planner (drop-in replacement for PurePursuitPlanner)
# ------------------------
class RLAgentPlanner(template_planner):
    HISTORY_LEN = 10
    STATE_HISTORY_LEN = 25
    BOOTSTRAP_TRANSITIONS = 2
    ACTION_DENORM = np.array([0.4, 3.0], dtype=np.float32)

    def __init__(self):
        super().__init__()
        print("Initializing RLAgentPlanner (actor client)")

        # Training vs Inference   
        self.inference_model_name = Settings.SAC_INFERENCE_MODEL_NAME  # Model name thats loaded: if none: use weights from server
        self.training_mode = (self.inference_model_name is None)  # Training mode if no inference model specified
        
        if self.training_mode:
            print(f"[RLAgentPlanner] Mode: TRAINING (receiving weights from server)")
        else:
            print(f"[RLAgentPlanner] Mode: INFERENCE (using model: {self.inference_model_name})")


        self.clear_buffer_on_reset = True
        self.terminate_server_after_simulation = True
        self.observation_builder_fn: Optional[Callable[[Dict[str, np.ndarray], Any], np.ndarray]] = None
        self.model: Optional[SAC] = None
        self.client: Optional[_TCPActorClient] = None
        self.latest_training_info: Optional[Dict[str, Any]] = None
        # Upper bound for UDT-driven MAX_SIM_FREQUENCY tweaks: never exceed initial Settings value.
        self._udt_sim_frequency_ceiling: Optional[float] = (
            float(Settings.MAX_SIM_FREQUENCY)
            if self.training_mode and Settings.MAX_SIM_FREQUENCY is not None
            else None
        )

        # --- networking ---

        if self.training_mode:
            # self.client = _TCPActorClient(host="192.168.194.226", port=5555, actor_id=2)
            self.client = _TCPActorClient(host="127.0.0.1", port=5555, actor_id=1)
            self.client.start()
            
            # Send clear buffer message on initialization
            if self.clear_buffer_on_reset:
                self.client.send_clear_buffer()
                print("[RLAgentPlanner] Sent clear buffer message to server")

        if self.training_mode:
            # For training we infer obs_dim lazily from the first built observation.
            self._received_weights = False
            self._warned_no_weights = False
        else:
            self._init_inference_model()

            self._received_weights = True  # no waiting for weights in inference
            self._warned_no_weights = False

        # constant warmup action in policy space [-1, 1]: [steer, accel]
        # no VecNormalize in this setup
        self.vec_normalize = None

        # planner state
        self.angular_control = 0.0
        self.translational_control = 0.0
        self.action_history_queue = deque([np.zeros(2) for _ in range(self.HISTORY_LEN)], maxlen=self.HISTORY_LEN)
        self.state_history = deque([np.zeros(10) for _ in range(self.STATE_HISTORY_LEN)], maxlen=self.STATE_HISTORY_LEN)

        # Lowpass filter state for control outputs
        self.prev_angular_control = 0.0
        self.prev_translational_control = 0.0
        self.lowpass_alpha = 1.0  # Filter coefficient: 0 = no passing, 1 = no smoothing

        self.prev_obs_raw: Optional[np.ndarray] = None
        self.prev_action: Optional[np.ndarray] = None

        self.action_denormalization_array = self.ACTION_DENORM.copy()

        # episode accumulation
        self._episode: list[dict] = []
        self.fallback_planner: PurePursuitPlanner = PurePursuitPlanner()
        self.fallback_planner.waypoint_utils = self.waypoint_utils
        self.fallback_planner.lidar_utils = self.lidar_utils
        self.transition_logger = TransitionLogger()
        self.save_transitions = False
        self.autonomous_driving = True
        self.control_index = 0
        # For fast smoke tests: send the first couple transitions early so the learner can infer obs_dim
        # and broadcast weights without waiting for an episode to end.
        self._bootstrap_sent = False
        # Init Curriculum Supervisor (when any curriculum feature is enabled)
        self.curriculum_supervisor = None
        curriculum_enabled = Settings.SAC_CURRICULUM_ENABLED
        if self.training_mode and curriculum_enabled:
            self.curriculum_supervisor = CurriculumSupervisor()

        self.reset()


    def reset(self):
        self.action_history_queue.clear()
        self.action_history_queue.extend([np.zeros(2) for _ in range(self.HISTORY_LEN)])
        self.state_history.clear()
        self.state_history.extend([np.zeros(10) for _ in range(self.STATE_HISTORY_LEN)])
        
        self.transition_logger.clear()
        self.control_index = 0

        self.prev_obs_raw = None
        self.prev_action = None
        self.angular_control = 0.0
        self.translational_control = 0.0
        self.prev_angular_control = 0.0
        self.prev_translational_control = 0.0
        # do not clear self._episode here; do it on episode end

    def process_observation(self, ranges=None, ego_odom=None, observation=None):

        if not self.autonomous_driving:
            return self._fallback_action()
        
        sd_to_load = self._sync_from_server()
        

        self.fallback_action = self._fallback_action()


        # --- build raw obs (manual normalization happens inside) ---
        raw_obs = self._build_observation()
        self._ensure_model_and_apply_weights(raw_obs, sd_to_load)

        action = self._select_action(raw_obs)

        # scale to simulator units
        action = np.clip(action, -1, 1)
        steering, accel = action * self.action_denormalization_array

        # remember pre-normalized obs & raw action for transition building
        self.prev_obs_raw = raw_obs
        self.prev_action  = action

        # Apply lowpass filter to control outputs
        # filtered = alpha * new_value + (1 - alpha) * previous_value
        self.angular_control = self.lowpass_alpha * steering + (1 - self.lowpass_alpha) * self.prev_angular_control
        self.translational_control = self.lowpass_alpha * accel + (1 - self.lowpass_alpha) * self.prev_translational_control
        
        
        max_translational_control = float(
            getattr(Settings, "SAC_MAX_TRANSLATIONAL_CONTROL", 6.0)
        )
        self.translational_control = float(
            np.clip(
                self.translational_control,
                -max_translational_control,
                max_translational_control,
            )
        )
        
        # Update previous values for next iteration
        self.prev_angular_control = self.angular_control
        self.prev_translational_control = self.translational_control
        
        self.action_history_queue.append(action)
        self.state_history.append(self.car_state)
        
        self.control_index += 1
        return self.angular_control, self.translational_control


    def on_step_end(self, driver_obs:Dict[str, Any]) -> None:

        reward = driver_obs['reward']
        done = driver_obs['done']
        info = driver_obs['info']

        # if(done and self.autonomous_driving):
        #     print(f"DONE CALLED.")

        """Called by env AFTER stepping. Pass the obs returned by env.step"""
        if self.prev_obs_raw is None or self.prev_action is None:
            return  # first step guard


        next_obs = self._build_observation()

        # Add curriculum difficulty to info for learner_server plotting (clamp to [0,1] to avoid float noise)
        info_out = dict(info or {})
        if self.curriculum_supervisor is not None:
            info_out["difficulty"] = float(np.clip(np.round(self.curriculum_supervisor.get_difficulty(), 4), 0.0, 1.0))
        transition = {
            "obs":      self.prev_obs_raw.astype(np.float32),
            "action":   self.prev_action.astype(np.float32),
            "next_obs": next_obs.astype(np.float32),
            "reward":   float(reward),
            "done":     bool(done),
            "info":     info_out,
        }
        self._episode.append(transition)

        self._maybe_bootstrap_send()
        if self.save_transitions:
            self.transition_logger.log(transition)

        if done or self.control_index >= Settings.MAX_EPISODE_LENGTH:
            total_reward = sum(t["reward"] for t in self._episode) if self._episode else 0.0
            #Update Curriculum Supervisor
            if self.curriculum_supervisor is not None:
                self.curriculum_supervisor.on_episode_end(total_reward, len(self._episode))
            #Save Transitions
            if self.save_transitions:
                self.transition_logger.save_csv()
            #Send Transitions to Learner Server
            if self.training_mode and self.autonomous_driving:
                try:
                    if len(self._episode) > 10:
                        self.client.send_transition_batch(self._episode)
                        if Settings.SAC_AGENT_DEBUG:
                            print(f"[RLAgentPlanner] Sending episode with {len(self._episode)} transitions with total reward {total_reward}.")
                except Exception as e:
                    print(f"[RLAgentPlanner] Failed to send episode: {e}")
            self._reset_episode_state()

    def on_simulation_end(self, collision=False):
        """Called when the simulation ends. Sends a terminate message to the server."""
        if self.terminate_server_after_simulation:
            if self.training_mode and self.client is not None:
                try:
                    self.client.send_terminate()
                    print("[RLAgentPlanner] Sent terminate message to server")
                except Exception as e:
                    print(f"[RLAgentPlanner] Failed to send terminate message: {e}")
       
    def close(self):
        pass
    
    def _fallback_action(self) -> np.ndarray:

        self.fallback_planner.lidar_utils = self.lidar_utils
        self.fallback_planner.car_state = self.car_state
        self.fallback_planner.waypoint_utils = self.waypoint_utils

        fallback_control = self.fallback_planner.process_observation()
        fallback_action = fallback_control / self.action_denormalization_array
        # return [0., 0.]
        return fallback_action

  
    
    def _build_super_observation(self) -> Dict[str, np.ndarray]:
        """ 
        Builds the super observation dictionary.
        This dict should contain all the information that the observation available in the environment.
        This dict is then used to build the observation array for the policy.
        """
        car_state = self.car_state
        last_actions = np.asarray(list(self.action_history_queue)[-3:], dtype=np.float32).reshape(-1)
        state_history = np.asarray(list(self.state_history), dtype=np.float32)

        # Get border points relative to the car's position
        border_points = self.waypoint_utils.get_track_border_positions_relative(
            self.waypoint_utils.next_waypoints, car_state
        )

        fallback_action = self._fallback_action()
        return {
            "car_state": self.car_state,
            "state_history": state_history,
            "next_waypoints": np.asarray(self.waypoint_utils.next_waypoints, dtype=np.float32),
            "border_points": np.asarray(border_points, dtype=np.float32),
            "lidar_ranges": np.asarray(self.lidar_utils.processed_ranges, dtype=np.float32),
            "last_actions": np.asarray(last_actions, dtype=np.float32),
            "frenet_coordinates": (np.asarray(self.waypoint_utils.frenet_coordinates, dtype=np.float32)),
            "global_waypoint_vel_factor": np.array([Settings.GLOBAL_WAYPOINT_VEL_FACTOR], dtype=np.float32),
            "pp_action": np.asarray(fallback_action, dtype=np.float32),
            "fallback_action": np.asarray(fallback_action, dtype=np.float32),
        }

    def _build_observation(self) -> np.ndarray:
        if self.observation_builder_fn is None:
            raise RuntimeError(
                "No observation builder loaded from model folder. "
                "Expected '<model_dir>/client/observation_builder.py'."
            )
        super_obs = self._build_super_observation()
        obs = self.observation_builder_fn(super_obs, self)
        return np.asarray(obs, dtype=np.float32).reshape(-1)
  

    # ---- helpers ----
    def _init_inference_model(self) -> None:
        """
        Load a saved SAC model for inference and attach a dummy env with matching obs_dim.
        """
        model_path_root, model_dir = SacUtilities.resolve_model_paths(self.inference_model_name)
        model_zip_path = model_path_root + ".zip"

        print(f"[RLAgentPlanner] Loading inference model from: {model_zip_path}")
        self.model = SAC.load(model_zip_path, device="cpu")

        try:
            obs_dim = int(self.model.observation_space.shape[0])
            dummy_env = SacUtilities.create_vec_env_from_obs_dim(obs_dim)
            if hasattr(self.model, "set_env"):
                self.model.set_env(dummy_env)
        except Exception:
            pass

        print(f"[Agent] Success: Loaded SAC model: {self.inference_model_name} from {model_zip_path}")
        self._load_observation_builder(os.path.join(model_dir, "client"), required=True)

    def _load_observation_builder(self, client_model_dir: str, required: bool = False) -> bool:
        builder_path = os.path.join(client_model_dir, "observation_builder.py")
        if not os.path.isfile(builder_path):
            msg = f"[RLAgentPlanner] observation_builder.py not found at {builder_path}"
            if required:
                raise FileNotFoundError(msg)
            print(msg)
            return False
        try:
            module_name = f"observation_builder_{abs(hash(builder_path))}"
            spec = importlib.util.spec_from_file_location(module_name, builder_path)
            if spec is None or spec.loader is None:
                if required:
                    raise RuntimeError(f"Failed to load spec for {builder_path}")
                return False
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            fn = getattr(module, "build_observation", None)
            if callable(fn):
                self.observation_builder_fn = fn
                print(f"[RLAgentPlanner] Loaded observation builder from {builder_path}")
                return True
            if required:
                raise RuntimeError(f"'build_observation' is not callable in {builder_path}")
        except Exception as e:
            if required:
                raise
            print(f"[RLAgentPlanner] Failed to load observation builder from {builder_path}: {e}")
        return False


    def _init_model_for_obs_dim(self, obs_dim: int) -> None:
        """
        Initialize SB3 SAC with a dummy env whose observation_space matches `obs_dim`.
        """
        if self.model is not None:
            return
        dummy_env = SacUtilities.create_vec_env_from_obs_dim(obs_dim)
        self.model = SacUtilities.create_model(dummy_env, device="cpu")
        # When training, the actor waits for weights broadcast from the learner.
        self._received_weights = False
        self._warned_no_weights = False

    def _ensure_model_and_apply_weights(self, raw_obs: np.ndarray, sd_to_load: Optional[dict]) -> None:
        """
        Ensure local SB3 model exists, then apply incoming actor weights if available.
        """
        if self.model is None:
            obs_dim = int(np.asarray(raw_obs, dtype=np.float32).reshape(-1).shape[0])
            self._init_model_for_obs_dim(obs_dim)

        if sd_to_load is not None and self.model is not None:
            self.model.policy.actor.load_state_dict(sd_to_load, strict=True)
            self.model.policy.actor.eval()
            self._received_weights = True
            if Settings.SAC_AGENT_DEBUG:
                print("[RLAgentPlanner] ✅ Actor weights updated.")

    def _apply_udt_training_info(self, training_info: Dict[str, Any]) -> None:
        """
        If server sent udt_control (SAC_TARGET_UDT set on learner), tune MAX_SIM_FREQUENCY:
        UDT too low -> decrease Hz; UDT too high -> increase Hz.
        """
        metrics = training_info.get("metrics")
        if not isinstance(metrics, dict):
            return
        current_udt = metrics.get("current_udt")
        if not isinstance(current_udt, (int, float)):
            return

        udt_control = training_info.get("udt_control")
        if isinstance(udt_control, dict) and udt_control.get("target_udt") is not None:
            target_udt = float(udt_control["target_udt"])
            deadband = float(udt_control.get("deadband_ratio", Settings.SAC_UDT_DEADBAND_RATIO))
            step_ratio = float(udt_control.get("freq_adjust_step_ratio", Settings.SAC_UDT_FREQ_ADJUST_STEP_RATIO))
            fmin = float(udt_control.get("min_sim_frequency_hz", Settings.SAC_MIN_SIM_FREQUENCY))
        elif Settings.SAC_TARGET_UDT is not None:
            target_udt = float(Settings.SAC_TARGET_UDT)
            deadband = float(Settings.SAC_UDT_DEADBAND_RATIO)
            step_ratio = float(Settings.SAC_UDT_FREQ_ADJUST_STEP_RATIO)
            fmin = float(Settings.SAC_MIN_SIM_FREQUENCY)
        else:
            return

        if target_udt <= 0 or step_ratio <= 0:
            return
        if Settings.MAX_SIM_FREQUENCY is None:
            return
        fmax = self._udt_sim_frequency_ceiling
        if fmax is None:
            return

        lower = target_udt * (1.0 - deadband)
        upper = target_udt * (1.0 + deadband)
        prev = float(Settings.MAX_SIM_FREQUENCY)
        updated = prev
        reason = None
        if float(current_udt) < lower:
            updated = prev * (1.0 - step_ratio)
            reason = "udt_low_decrease_freq"
        elif float(current_udt) > upper:
            updated = prev * (1.0 + step_ratio)
            reason = "udt_high_increase_freq"

        updated = float(np.clip(updated, fmin, fmax))
        if reason is None or abs(updated - prev) < 1e-9:
            return

        Settings.MAX_SIM_FREQUENCY = updated
        if Settings.SAC_AGENT_DEBUG:
            print(
                "[RLAgentPlanner] UDT control: "
                f"MAX_SIM_FREQUENCY {prev:.2f} -> {updated:.2f} Hz "
                f"(UDT={float(current_udt):.4f}, target={target_udt:.4f}, reason={reason})"
            )

    def _select_action(self, raw_obs: np.ndarray) -> np.ndarray:
        """
        Select action from policy when weights are available, otherwise use fallback.
        """
        obs_for_policy = raw_obs
        if self.vec_normalize is not None:
            obs_for_policy = self.vec_normalize.normalize_obs(raw_obs[None, :])[0]

        if self._received_weights:
            action, _ = self.model.predict(obs_for_policy, deterministic=(not self.training_mode))
            return np.asarray(action, dtype=np.float32).reshape(-1)

        if not self._warned_no_weights:
            print("[RLAgentPlanner] ⚠️ No weights yet; using warmup strategy ")
            self._warned_no_weights = True
        return self._fallback_action()

    def _apply_control_filter(self, steering: float, accel: float) -> None:
        """
        Apply low-pass filtering to control outputs and update filter state.
        """
        # filtered = alpha * new_value + (1 - alpha) * previous_value
        self.angular_control = self.lowpass_alpha * steering + (1 - self.lowpass_alpha) * self.prev_angular_control
        self.translational_control = self.lowpass_alpha * accel + (1 - self.lowpass_alpha) * self.prev_translational_control
        self.prev_angular_control = self.angular_control
        self.prev_translational_control = self.translational_control

    def _maybe_bootstrap_send(self) -> None:
        """
        Send the first couple transitions early so the learner can infer obs_dim quickly.
        Only runs once per planner start.
        """
        if not (self.training_mode and self.autonomous_driving and not self._bootstrap_sent):
            return
        if len(self._episode) < self.BOOTSTRAP_TRANSITIONS:
            return

        try:
            self.client.send_transition_batch(self._episode[:self.BOOTSTRAP_TRANSITIONS])
            self._bootstrap_sent = True
            # Keep remaining transitions so the end-of-episode batch is still mostly correct.
            self._episode = self._episode[self.BOOTSTRAP_TRANSITIONS:]
        except Exception as e:
            print(f"[RLAgentPlanner] Bootstrap send failed: {e}")

    def _reset_episode_state(self) -> None:
        self._episode = []
        self.control_index = 0
        self.prev_obs_raw = None
        self.prev_action = None
        
       