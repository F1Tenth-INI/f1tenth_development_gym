"""Trace of the new repo. Same args as ref_trace.py. QUIRKS=1 emulates the old harness's quirks."""
import os, sys
os.environ["OMP_NUM_THREADS"] = "1"; os.environ["JAX_PLATFORMS"] = "cpu"
os.environ["XLA_FLAGS"] = "--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1 inter_op_parallelism_threads=1"
import os
REPO = os.environ.get("ATMPPI_REPO", "../atmppi")  # the standalone ATMPPI repo (github.com/F1Tenth-INI/L4DC)
sys.path.insert(0, REPO)
ctrl, mapn, wp, mu, noise, seed, steps, rollouts, out = sys.argv[1:10]
import numpy as np
from race.config import load_settings
from race import simulation as S
cfg = load_settings(os.path.join(REPO, "settings", "settings.yml"))
cfg.controller = {"hw": "atmppi", "mppi": "mppi"}[ctrl]
sim = cfg.simulation
sim.map = mapn; sim.map_dir = os.path.join(REPO, "maps", mapn); sim.waypoints = wp; sim.friction = float(mu)
sim.state_noise = 0.0 if noise == "off" else 0.05; sim.seed = int(seed); sim.render = False
sim.duration = int(steps) * sim.timestep_control
getattr(cfg, cfg.controller).rollouts = int(rollouts)
s = S.Simulation(cfg)
log = {"state": [], "control": [], "laps": [], "crash": []}
quirks = os.environ.get("QUIRKS", "1") == "1"
original_wp = s.track.waypoints
episodes = [0]
orig_reset = s.reset
def reset():
    if quirks and cfg.controller == "atmppi":
        orig_reset()
        episodes[0] += 1
        # Episode 1: the old track table was the controller's memory (shared array); later a fresh copy.
        s.track.waypoints = s.controller.waypoints.numpy()[:, :10] if episodes[0] == 1 else original_wp
    else:
        orig_reset()
s.reset = reset
if quirks:
    np.random.uniform(-0.05, 0.05)  # the old harness drew an opponent speed factor at start
    s.track.random_start()  # the old harness drew one start pose before the first reset
    # The old CarSystem drew control noise (zero) from numpy every 10 steps.
    ctrl_step = s.controller.step
    count = [0]
    def step_with_draw(obs):
        r = ctrl_step(obs)
        if count[0] % 10 == 0:
            np.random.uniform(-1, 1, 2)
        count[0] += 1
        return r
    s.controller.step = step_with_draw
    orig_crash = s.controller.on_crash
    def on_crash():
        count[0] = 0
        orig_crash()
    s.controller.on_crash = on_crash
    # At a restart the old harness drew the pose from the table as it was then (the memory, in episode 1).
    rs = s.track.random_start
    def random_start():
        return rs()
    s.track.random_start = random_start
ol = s._lap_complete
def lap(t):
    log["laps"].append(float(t)); ol(t)
s._lap_complete = lap
s.lap_timer.callback = lap
ostep = s.step
def step():
    n = len(s.crashes)
    ostep()
    if len(s.crashes) > n:
        log["crash"].append(len(log["state"]) + 1)
        pass
    log["state"].append(np.array(s.state)); log["control"].append([0, 0])
s.step = step
s.reset()
for _ in range(int(steps)):
    s.step()
np.savez(out, state=np.array(log["state"]), laps=np.array(log["laps"]), crash=np.array(log["crash"]))
print("LAPS", log["laps"], "CRASH", log["crash"])
