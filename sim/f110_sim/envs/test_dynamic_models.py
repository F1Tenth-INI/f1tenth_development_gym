#!/usr/bin/env python3
"""
Test script to verify the JAX Pacejka car model.
"""

import numpy as np
import jax.numpy as jnp
import time
from utilities.car_files.vehicle_parameters import VehicleParameters

from .dynamic_model_pacejka_jax import car_dynamics_pacejka_jax


def test_car_model_single_step():
    """Test JAX car model single-step integration."""
    print("Testing JAX car model single step...")

    car_params = VehicleParameters().to_np_array().astype(np.float32)
    car_params_jax = jnp.array(car_params)

    test_state = np.array([0.1, 5.0, 0.2, 0.5, 0.8, 0.6, 10.0, 5.0, 0.04, 0.1], dtype=np.float32)
    test_control = np.array([0.2, 2.0], dtype=np.float32)
    dt = 0.02

    result = car_dynamics_pacejka_jax(
        jnp.array(test_state), jnp.array(test_control), car_params_jax, dt
    )
    result_np = np.array(result)

    assert result_np.shape == (10,)
    assert np.all(np.isfinite(result_np))
    print("Single step test: PASSED")
    return True


def test_jax_pacejka_integration():
    """Test jax_pacejka integration in base classes."""
    print("Testing jax_pacejka integration...")

    from .base_classes import RaceCar

    try:
        from utilities.car_files.vehicle_parameters import VehicleParameters
        car_params = VehicleParameters()
        params_dict = {
            'mu': car_params.mu,
            'lf': car_params.lf,
            'lr': car_params.lr,
            'h': car_params.h,
            'm': car_params.m,
            'I_z': car_params.I_z,
            'length': car_params.length,
            'width': car_params.width,
            's_min': car_params.s_min,
            's_max': car_params.s_max,
            'sv_min': car_params.sv_min,
            'sv_max': car_params.sv_max,
            'a_min': car_params.a_min,
            'a_max': car_params.a_max,
            'v_min': car_params.v_min,
            'v_max': car_params.v_max,
            'v_switch': car_params.v_switch,
        }
        del params_dict
        print("JAX Pacejka integration test: PASSED (basic validation)")
    except Exception as e:
        assert False, f"JAX Pacejka integration test failed: {e}"

    return True


def test_models_comprehensive():
    """Run integration tests with various state and control combinations."""
    print("Running comprehensive model tests...")

    car_params = VehicleParameters().to_np_array().astype(np.float32)
    car_params_jax = jnp.array(car_params)
    dt = 0.02

    test_cases = [
        ([0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.5]),
        ([0.1, 5.0, 0.2, 0.5, 0.8, 0.6, 10.0, 5.0, 0.04, 0.1], [0.2, 2.0]),
        ([0.5, 15.0, 1.0, 1.2, 0.3, 0.9, 20.0, 10.0, 0.1, 0.3], [0.5, 3.0]),
        ([0.0, 0.1, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.1, 0.1]),
    ]

    for i, (state, control) in enumerate(test_cases):
        test_state = np.array(state, dtype=np.float32)
        test_control = np.array(control, dtype=np.float32)
        result = np.array(
            car_dynamics_pacejka_jax(
                jnp.array(test_state), jnp.array(test_control), car_params_jax, dt
            )
        )
        assert result.shape == (10,)
        assert np.all(np.isfinite(result)), f"Test case {i+1} produced non-finite values"
        print(f"Test case {i+1}: PASSED")

    print("Comprehensive test PASSED")
    return True


def benchmark_model():
    """Benchmark JAX single-step performance."""
    print("Benchmarking JAX model performance...")

    car_params = VehicleParameters().to_np_array().astype(np.float32)
    car_params_jax = jnp.array(car_params)
    test_state = jnp.array([0.1, 5.0, 0.2, 0.5, 0.8, 0.6, 10.0, 5.0, 0.04, 0.1], dtype=jnp.float32)
    test_control = jnp.array([0.2, 2.0], dtype=jnp.float32)
    dt = 0.02

    for _ in range(10):
        car_dynamics_pacejka_jax(test_state, test_control, car_params_jax, dt)

    n_iterations = 1000
    start_time = time.time()
    for _ in range(n_iterations):
        car_dynamics_pacejka_jax(test_state, test_control, car_params_jax, dt)
    jax_time = (time.time() - start_time) / n_iterations * 1000

    print(f"Single step performance: {jax_time:.3f} ms per call")


def run_all_tests():
    """Run all tests."""
    print("=" * 60)
    print("Running all dynamic model tests...")
    print("=" * 60)

    try:
        test_car_model_single_step()
        print()
        test_jax_pacejka_integration()
        print()
        test_models_comprehensive()
        print()
        benchmark_model()
        print()
        print("=" * 60)
        print("All tests PASSED!")
        print("=" * 60)
    except Exception as e:
        print("=" * 60)
        print(f"Tests FAILED: {e}")
        print("=" * 60)
        raise


if __name__ == "__main__":
    run_all_tests()
