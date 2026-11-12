"""Trace of the original snapshot code: true car state and command per control step.
python ref_trace.py CONTROLLER(hw|mppi) MAP WP MU NOISE(on|off) SEED STEPS ROLLOUTS OUT.npz
Run with cwd = snapshot root."""
import os, sys, json
os.environ["OMP_NUM_THREADS"] = "1"; os.environ["MKL_NUM_THREADS"] = "1"; os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ["XLA_FLAGS"] = "--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1 inter_op_parallelism_threads=1"
os.environ["JAX_PLATFORMS"] = "cpu"; os.environ["CUDA_VISIBLE_DEVICES"] = ""
ctrl, mapn, wp, mu, noise, seed, steps, rollouts, out = sys.argv[1:10]
seed, steps, rollouts = int(seed), int(steps), int(rollouts)
sys.path.insert(0, os.getcwd())
from utilities.Settings import Settings
TUNED = {"HWM_MIN_BORDER_DISTANCE": 0.07, "MPC_DRIVE_WALL_CLEARANCE": 0.25, "MPC_DRIVE_WALL_WEIGHT": 0.2,
         "MPC_WAYPOINT_WINDOW_MARGIN": 0.1, "MPC_WAYPOINT_WINDOW_MIN_MARGIN": 0.08,
         "MPC_DRIVE_TARGET_SPEED_FACTOR": 1.03, "HWM_ROLLOUT_THREADS": 1}
for k, v in TUNED.items():
    setattr(Settings, k, v)
Settings.MAP_NAME = mapn; Settings.WAYPOINT_SOURCE = wp; Settings.recalculate_paths()
Settings.CONTROLLER = {"hw": "HirarchicalPlanner", "mppi": "mppi-lite-jax"}[ctrl]
Settings.SURFACE_FRICTION = float(mu)
if noise == "off":
    Settings.NOISE_LEVEL_CAR_STATE = [0.0] * 10
Settings.RANDOM_SEED = seed
if ctrl == "hw":
    Settings.MPC_NUM_OF_TRAJECTORIES = rollouts
Settings.LEARNER_TCP_HOST = "127.0.0.1"; Settings.LEARNER_TCP_PORT = 1
Settings.SIMULATION_LENGTH = steps
Settings.TRUNCATE_ON_LEAVE_TRACK = False; Settings.MAX_EPISODE_LENGTH = 0; Settings.EXPERIMENT_MAX_LENGTH = 10 ** 12
Settings.RESET_ON_DONE = True; Settings.RESPAWN_ON_RESET = False; Settings.STOP_AFTER_N_LAPS = None
Settings.RENDER_MODE = None; Settings.SAVE_RECORDINGS = False; Settings.SAVE_PLOTS = False; Settings.SAVE_REWARDS = False
Settings.HWM_SIM_SPEED = None; Settings.MAX_SIM_FREQUENCY = None
import random, numpy as np, torch
random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.set_num_threads(1)
from utilities.episode_termination import EpisodeTerminator
orig = EpisodeTerminator.evaluate
EpisodeTerminator.evaluate = lambda self, c, d, **_: orig(self, c, d, ignore_leave_track=True, ignore_episode_length=True, crash_only=True)
from utilities.car_system import CarSystem
from run.run_simulation import RacingSimulation
oi = CarSystem.initialize_controller
def init(self, name):
    oi(self, name)
    if ctrl == "mppi" and self.planner is not None:
        import jax
        self.planner.key = jax.random.PRNGKey(seed); self.planner.batch_size = rollouts; self.planner.noise_std = (0.2, 0.6)
CarSystem.initialize_controller = init
log = {"state": [], "control": [], "laps": [], "crash": []}
ol = CarSystem.lap_complete_cb
def lap(self, t, *r):
    log["laps"].append(float(t)); return ol(self, t, *r)
CarSystem.lap_complete_cb = lap
od = RacingSimulation.handle_done
def done(self):
    log["crash"].append(len(log["state"])); return od(self)
RacingSimulation.handle_done = done
os_ = RacingSimulation.simulation_step
def step(self):
    os_(self)
    d = self.drivers[0]
    log["state"].append(np.array(self.world_sim.agents[0].state))
    log["control"].append([d.angular_control, d.translational_control])
RacingSimulation.simulation_step = step
sim = RacingSimulation(); sim.prepare_simulation()
try:
    sim.run_simulation()
finally:
    np.savez(out, state=np.array(log["state"]), control=np.array(log["control"]), laps=np.array(log["laps"]), crash=np.array(log["crash"]))
    print("LAPS", log["laps"], "CRASH", log["crash"], flush=True)
    os._exit(0)
