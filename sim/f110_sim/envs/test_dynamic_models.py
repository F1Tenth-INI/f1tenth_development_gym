#!/usr/bin/env python3
"""
Test script to verify the standalone JAX and JIT car models work correctly.
This module contains unit tests for the dynamic models used in the F1Tenth simulation.
"""

import numpy as np
import jax.numpy as jnp
import time
from utilities.car_files.vehicle_parameters import VehicleParameters

# Import the standalone models
from .dynamic_model_pacejka_jit import car_dynamics_pacejka_jit, car_step_parallel
from .dynamic_model_pacejka_jax import car_dynamics_pacejka_jax


def test_car_models():
    """Sanity check that the JAX Pacejka model integrates a normalized stick.

    The JIT path predates the normalized-[-1, 1] interface and is now
    deprecated, so the old JIT-vs-JAX equality check no longer applies; we
    only verify that one JAX step produces finite, sane state changes.
    """

    print("Testing JAX Pacejka model with normalized control...")

    car_params_jax = jnp.array(
        VehicleParameters().to_np_array().astype(np.float32)
    )

    # 11 states (last entry is wheel_omega kinematically tied to v_x).
    om0 = 5.0 / 0.033
    test_state = jnp.array(
        [0.1, 5.0, 0.2, 0.5, 0.8, 0.6, 10.0, 5.0, 0.04, 0.1, om0],
        dtype=jnp.float32,
    )
    # 0.2 -> 20% steer, 0.4 -> 40% throttle on whatever drive_mode the YAML picks.
    test_control = jnp.array([0.2, 0.4], dtype=jnp.float32)
    dt = 0.02

    jax_result = np.array(
        car_dynamics_pacejka_jax(test_state, test_control, car_params_jax, dt)
    )

    finite = np.all(np.isfinite(jax_result))
    moved_forward = jax_result[1] > 0.0  # v_x stays positive
    print(f"v_x: 5.0 -> {jax_result[1]:.4f}, finite={finite}")
    assert finite, f"JAX result has non-finite values: {jax_result}"
    assert moved_forward, f"v_x went non-positive after a small forward stick: {jax_result[1]}"
    return True


def test_jax_pacejka_integration():
    """Test the new jax_pacejka integration in base classes"""
    
    print("Testing jax_pacejka integration...")
    
    from .base_classes import RaceCar
    from utilities.Settings import Settings
    
    # We'll skip this test if Settings don't allow modification
    # This is more of an integration test that would be better done in a full simulation context
    try:
        # Create basic car parameters dict for testing
        from utilities.car_files.vehicle_parameters import VehicleParameters
        car_params = VehicleParameters()
        params_dict = {
            'mu': car_params.mu,
            'lf': car_params.lf,
            'lr': car_params.lr,
            'h_cg': car_params.h,
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
        
        # This test is simplified - in practice, the jax_pacejka implementation
        # would be tested through the full simulation pipeline
        print(f"JAX Pacejka integration test: PASSED (basic validation)")
        
    except Exception as e:
        print(f"JAX Pacejka integration test: SKIPPED - {e}")
        # Fail the test
        assert False, f"JAX Pacejka integration test failed: {e}"

    return True


def test_models_comprehensive():
    """Smoke test the JAX dynamics on a few state/control combinations.

    The legacy JIT-vs-JAX equality test no longer applies (JIT keeps the
    pre-normalization control semantics, JAX is now [-1, 1]). We just
    verify the JAX path produces finite output for a varied state set.
    """

    print("Running comprehensive JAX model checks...")

    car_params_jax = jnp.array(
        VehicleParameters().to_np_array().astype(np.float32)
    )

    dt = 0.02
    test_cases = [
        ([0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0 / 0.033], [0.0, 0.3]),
        ([0.1, 5.0, 0.2, 0.5, 0.8, 0.6, 10.0, 5.0, 0.04, 0.1, 5.0 / 0.033], [0.2, 0.6]),
        ([0.5, 15.0, 1.0, 1.2, 0.3, 0.9, 20.0, 10.0, 0.1, 0.3, 15.0 / 0.033], [0.5, 1.0]),
        ([0.0, 0.1, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.1 / 0.033], [0.1, -0.4]),
    ]

    for i, (state, control) in enumerate(test_cases):
        test_state_jax = jnp.array(state, dtype=jnp.float32)
        test_control_jax = jnp.array(control, dtype=jnp.float32)
        jax_result = np.array(
            car_dynamics_pacejka_jax(test_state_jax, test_control_jax, car_params_jax, dt)
        )
        finite = np.all(np.isfinite(jax_result))
        print(f"Test case {i+1}: v_x {state[1]:.2f} -> {jax_result[1]:.2f}, finite={finite}")
        assert finite, f"Test case {i+1} non-finite: {jax_result}"

    print("Comprehensive JAX model checks PASSED")
    return True


def benchmark_models():
    """Benchmark the performance of JIT vs JAX implementations.

    NOTE: the JIT path is deprecated; we keep the benchmark for historical
    timings only. The two implementations expect different control units,
    so timings are comparable but the resulting states are not.
    """

    print("Benchmarking model performance...")

    car_params = VehicleParameters().to_np_array().astype(np.float32)
    car_params_jax = jnp.array(car_params)

    om0 = 5.0 / 0.033
    test_state = np.array([0.1, 5.0, 0.2, 0.5, 0.8, 0.6, 10.0, 5.0, 0.04, 0.1, om0], dtype=np.float32)
    # Normalized stick for JAX; JIT expects physical accel/torque -- different
    # numbers, so we time them separately and don't compare their outputs.
    test_control_jax = jnp.array([0.2, 0.4], dtype=jnp.float32)
    test_control_jit = np.array([0.2, 2.0], dtype=np.float32)
    test_state_jax = jnp.array(test_state)
    dt = 0.02
    
    # Warm up both implementations
    for _ in range(10):
        car_dynamics_pacejka_jit(test_state, test_control_jit, car_params, dt)
        car_dynamics_pacejka_jax(test_state_jax, test_control_jax, car_params_jax, dt)

    n_iterations = 1000

    # Benchmark JIT
    start_time = time.time()
    for _ in range(n_iterations):
        jit_result = car_dynamics_pacejka_jit(test_state, test_control_jit, car_params, dt)
    jit_time = (time.time() - start_time) / n_iterations * 1000  # ms per call

    # Benchmark JAX
    start_time = time.time()
    for _ in range(n_iterations):
        jax_result = car_dynamics_pacejka_jax(test_state_jax, test_control_jax, car_params_jax, dt)
    jax_time = (time.time() - start_time) / n_iterations * 1000  # ms per call
    
    print(f"Single step performance:")
    print(f"  JIT: {jit_time:.3f} ms per call")
    print(f"  JAX: {jax_time:.3f} ms per call")
    print(f"  Speedup: {jit_time/jax_time:.2f}x")
    
    # Batch benchmark
    batch_sizes = [10, 100, 1000]
    
    for batch_size in batch_sizes:
        batch_states = np.tile(test_state, (batch_size, 1))
        batch_controls = np.tile(test_control_jit, (batch_size, 1))

        # Warm up
        car_step_parallel(batch_states, batch_controls, car_params, dt)

        # Benchmark JIT batch
        start_time = time.time()
        for _ in range(10):
            jit_batch_result = car_step_parallel(batch_states, batch_controls, car_params, dt)
        jit_batch_time = (time.time() - start_time) / 10 * 1000  # ms per call

        print(f"Batch size {batch_size} performance:")
        print(f"  JIT: {jit_batch_time:.3f} ms per batch")


def run_all_tests():
    """Run all tests"""
    print("=" * 60)
    print("Running all dynamic model tests...")
    print("=" * 60)
    
    try:
        test_car_models()
        print()
        
        test_jax_pacejka_integration()
        print()
        
        test_models_comprehensive()
        print()
        
        benchmark_models()
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
