"""Physical sanity test of the sim dynamics model: sliding must never gain energy.

300 random slide / reverse / brake states are simulated for 3 s. When coasting without
rolling or cornering resistance, the kinetic energy must not grow (apart from the small
explicit-Euler error), and the lateral acceleration must stay within mu*g.

Run from the repo root:
    python AnalyseData/DynamicsModelFit/stress_test.py [car.yml] [model_file.py]
"""
import os
import sys

import jax
import jax.numpy as jnp
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from fitlib import car_params, load_model  # noqa: E402

yml = sys.argv[1] if len(sys.argv) > 1 else "yokomo1_car_parameters.yml"
f = (load_model(sys.argv[2]) if len(sys.argv) > 2 else load_model()).car_dynamics_pacejka_jax
p = car_params(yml)
p_nores = p.copy()
p_nores[26] = 0  # no rolling resistance
p_nores[28] = 0  # no cornering resistance
step = jax.jit(lambda s, u, pp: f(s, u, pp, 0.01, 1, ode_model="ODE:ks_pacejka"))


def sim(s0, steer, acc, pp, T=3.0):
    s = jnp.array(s0, dtype=jnp.float32)
    out = [np.array(s)]
    for _ in range(int(T / 0.01)):
        s = step(s, jnp.array([steer, acc], dtype=jnp.float32), jnp.array(pp))
        out.append(np.array(s))
    return np.array(out)


def kinetic_energy(o):
    return 0.5 * p[4] * (o[:, 1] ** 2 + o[:, 2] ** 2) + 0.5 * p[5] * o[:, 0] ** 2


def world_acceleration(o):
    """|dV/dt| of the CoG velocity in the world frame (v_x * r is only a_y in steady cornering)."""
    c, s = np.cos(o[:, 3]), np.sin(o[:, 3])
    V = np.c_[o[:, 1] * c - o[:, 2] * s, o[:, 1] * s + o[:, 2] * c]
    return np.linalg.norm(np.diff(V, axis=0), axis=1) / 0.01


rng = np.random.default_rng(0)
runs = []
for _ in range(300):
    s0 = [rng.uniform(-8, 8), rng.uniform(-5, 8), rng.uniform(-3, 3), 0, 1, 0, 0, 0, 0, rng.uniform(-0.6, 0.6)]
    steer, acc = rng.uniform(-0.4, 0.4), rng.choice([0.0, 0.0, -5.0, 5.0])
    o = sim(s0, steer, acc, p_nores if acc == 0 else p)
    finite = np.all(np.isfinite(o))
    ke = kinetic_energy(o) if finite else None
    gain = (ke.max() - ke[0]) / max(ke[0], 1e-3) if finite and acc == 0 else 0.0
    runs.append(dict(finite=finite, gain=gain, coast=acc == 0, s0=s0, steer=steer,
                     max_vy=np.abs(o[:, 2]).max() if finite else np.inf,
                     max_acc=world_acceleration(o)[5:].max() if finite else np.inf))  # skip initial transient

coast = [r for r in runs if r["coast"] and r["finite"]]
print(f"non-finite runs: {sum(not r['finite'] for r in runs)}/{len(runs)}")
print(f"coasting runs with energy gain > 1%: {sum(r['gain'] > 0.01 for r in coast)}/{len(coast)}, "
      f"worst gain {100 * max(r['gain'] for r in coast):.1f}%")
print(f"max |v_y| reached: {max(r['max_vy'] for r in runs if r['finite']):.2f} m/s")
for r in sorted(coast, key=lambda r: -r["gain"])[:3]:
    print(f"  worst coast: gain {100 * r['gain']:.1f}%  start (r, v_x, v_y) = {np.round(r['s0'][:3], 2)}  steer {r['steer']:.2f}")

acc = np.array([r["max_acc"] for r in runs if r["finite"]])
print(f"max horizontal acceleration per run [m/s^2]: median {np.median(acc):.1f}, 95th pct {np.percentile(acc, 95):.1f}, "
      f"max {acc.max():.1f}  (mu*g = {p[0] * 9.81:.1f}; random start states are not in equilibrium)")
o = sim([0, 6, 0, 0, 1, 0, 0, 0, 0, 0], 0.4, 10.0, p, T=4.0)
print(f"full throttle on full lock: max horizontal acceleration {world_acceleration(o).max():.1f} m/s^2 (mu*g = {p[0] * 9.81:.1f})")
