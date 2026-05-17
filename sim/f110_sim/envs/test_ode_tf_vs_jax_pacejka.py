#!/usr/bin/env python3
"""Feed-forward comparison: ODE_TF (SI_Toolkit car_model) vs jax_pacejka."""

import numpy as np
import jax.numpy as jnp
import pytest

from utilities.Settings import Settings
from utilities.car_files.vehicle_parameters import VehicleParameters
from utilities.state_utilities import POSE_THETA_IDX
from sim.f110_sim.envs.base_classes import wrap_angle_rad
from SI_Toolkit_ASF.car_model import car_model
from SI_Toolkit.computation_library import NumpyLibrary
from sim.f110_sim.envs.dynamic_model_pacejka_jax import car_dynamics_pacejka_jax


def _ode_tf_step(car_model, state, steer, acc):
    u = np.array([[steer, acc]], dtype=np.float32)
    s_batch = np.expand_dims(state, 0).astype(np.float32)
    u_pid = car_model.pid(s_batch, u)
    u_c = car_model.apply_constrains(s_batch, u_pid)
    s_next = car_model.step_dynamics_core(s_batch, u_c)[0]
    s_next[POSE_THETA_IDX] = wrap_angle_rad(s_next[POSE_THETA_IDX])
    return s_next


def _jax_step(state, steer, acc, car_params_jax, ode_model):
    s_next = car_dynamics_pacejka_jax(
        jnp.array(state, dtype=jnp.float32),
        jnp.array([steer, acc], dtype=jnp.float32),
        car_params_jax,
        Settings.TIMESTEP_SIM,
        intermediate_steps=1,
        ode_model=ode_model,
    )
    s_next = np.array(s_next, dtype=np.float32)
    s_next[POSE_THETA_IDX] = wrap_angle_rad(s_next[POSE_THETA_IDX])
    return s_next


@pytest.fixture
def models():
    Settings.MOTOR_PID_IN_CAR_MODEL = False
    car_params = VehicleParameters(Settings.ENV_CAR_PARAMETER_FILE).to_np_array()
    cm = car_model(
        model_of_car_dynamics=Settings.ODE_MODEL_OF_CAR_DYNAMICS,
        batch_size=1,
        car_parameter_file=Settings.ENV_CAR_PARAMETER_FILE,
        dt=Settings.TIMESTEP_SIM,
        intermediate_steps=1,
        computation_lib=NumpyLibrary(),
    )
    return cm, jnp.array(car_params)


def test_single_step_matches_ode_tf(models):
    cm, car_params_jax = models
    s0 = np.array([0.0, 2.0, 0.0, 0.1, np.cos(0.1), np.sin(0.1), 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
    s_ode = _ode_tf_step(cm, s0, 0.1, 1.0)
    s_jax = _jax_step(s0, 0.1, 1.0, car_params_jax, Settings.ODE_MODEL_OF_CAR_DYNAMICS)
    np.testing.assert_allclose(s_ode, s_jax, rtol=1e-5, atol=1e-5)


def test_feedforward_trajectory_matches_ode_tf(models):
    cm, car_params_jax = models
    s0 = np.array([0.0, 2.0, 0.0, 0.1, np.cos(0.1), np.sin(0.1), 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
    controls = [
        (0.1, 1.0), (0.2, 2.0), (-0.1, 0.5), (0.0, -1.0),
        (0.15, 0.0), (0.3, 3.0), (-0.2, -2.0),
    ]
    s_ode, s_jax = s0.copy(), s0.copy()
    max_diff = 0.0
    for steer, acc in controls * 20:
        s_ode = _ode_tf_step(cm, s_ode, steer, acc)
        s_jax = _jax_step(s_jax, steer, acc, car_params_jax, Settings.ODE_MODEL_OF_CAR_DYNAMICS)
        max_diff = max(max_diff, float(np.max(np.abs(s_ode - s_jax))))
    assert max_diff < 1e-4, f"ODE_TF vs jax_pacejka max diff {max_diff:.2e} after feedforward"


def test_low_speed_ks_blend_region(models):
    """Exercise ks_pacejka blend around v_x ~ 1.75 m/s."""
    cm, car_params_jax = models
    s0 = np.array([0.0, 1.5, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.05], dtype=np.float32)
    s_ode = _ode_tf_step(cm, s0, 0.05, 0.2)
    s_jax = _jax_step(s0, 0.05, 0.2, car_params_jax, Settings.ODE_MODEL_OF_CAR_DYNAMICS)
    np.testing.assert_allclose(s_ode, s_jax, rtol=1e-5, atol=1e-5)


if __name__ == "__main__":
    m = models()
    test_single_step_matches_ode_tf(m)
    test_feedforward_trajectory_matches_ode_tf(m)
    test_low_speed_ks_blend_region(m)
    print("All ODE_TF vs jax_pacejka tests passed.")
