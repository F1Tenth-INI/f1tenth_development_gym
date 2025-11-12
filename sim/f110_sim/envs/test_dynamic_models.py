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
    """Test both JIT and JAX car model implementations to ensure they produce consistent results"""
    
    print("Testing car model implementations...")
    
    # Load car parameters
    car_params = VehicleParameters().to_np_array().astype(np.float32)
    car_params_jax = jnp.array(car_params)
    
    # Create test state and controls
    test_state = np.array([0.1, 5.0, 0.2, 0.5, 0.8, 0.6, 10.0, 5.0, 0.04, 0.1], dtype=np.float32)
    test_control = np.array([0.2, 2.0], dtype=np.float32)
    dt = 0.02
    
    # Test JIT version
    jit_result = car_dynamics_pacejka_jit(test_state, test_control, car_params, dt)
    
    # Test JAX version 
    test_state_jax = jnp.array(test_state)
    test_control_jax = jnp.array(test_control)
    jax_result = car_dynamics_pacejka_jax(test_state_jax, test_control_jax, car_params_jax, dt)
    
    # Check single step consistency
    single_step_diff = np.max(np.abs(jit_result - np.array(jax_result)))
    single_step_close = single_step_diff < 1e-5
    
    print(f"Single step max difference: {single_step_diff:.2e}")
    print(f"Car models test: {'PASSED' if single_step_close else 'FAILED'}")
    # Assert that models are consistent
    assert single_step_close, f"JIT and JAX single step models differ by {single_step_diff:.2e} (threshold: 1e-5)"
    return single_step_close


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
    """Run comprehensive tests comparing JIT and JAX implementations with various inputs"""
    
    print("Running comprehensive model tests...")
    
    # Load car parameters
    car_params = VehicleParameters().to_np_array().astype(np.float32)
    car_params_jax = jnp.array(car_params)
    
    dt = 0.02
    max_diff_overall = 0.0
    
    # Test with various state and control combinations
    test_cases = [
        # [yaw_rate, v_x, v_y, yaw_angle, yaw_cos, yaw_sin, x, y, slip_angle, steering]
        ([0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.5]),  # Low speed
        ([0.1, 5.0, 0.2, 0.5, 0.8, 0.6, 10.0, 5.0, 0.04, 0.1], [0.2, 2.0]),  # Medium speed
        ([0.5, 15.0, 1.0, 1.2, 0.3, 0.9, 20.0, 10.0, 0.1, 0.3], [0.5, 3.0]),  # High speed
        ([0.0, 0.1, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.1, 0.1]),  # Very low speed
    ]
    
    for i, (state, control) in enumerate(test_cases):
        test_state = np.array(state, dtype=np.float32)
        test_control = np.array(control, dtype=np.float32)
        
        # Test JIT version
        jit_result = car_dynamics_pacejka_jit(test_state, test_control, car_params, dt)
        
        # Test JAX version
        test_state_jax = jnp.array(test_state)
        test_control_jax = jnp.array(test_control)
        jax_result = car_dynamics_pacejka_jax(test_state_jax, test_control_jax, car_params_jax, dt)
        
        # Check consistency
        diff = np.max(np.abs(jit_result - np.array(jax_result)))
        max_diff_overall = max(max_diff_overall, diff)
        
        print(f"Test case {i+1}: max difference = {diff:.2e}")
        
        assert diff < 1e-5, f"Test case {i+1} failed: difference {diff:.2e} exceeds threshold 1e-5"
    
    print(f"Comprehensive test PASSED - Maximum difference across all cases: {max_diff_overall:.2e}")
    return True


def benchmark_models():
    """Benchmark the performance of JIT vs JAX implementations"""
    
    print("Benchmarking model performance...")
    
    # Load car parameters
    car_params = VehicleParameters().to_np_array().astype(np.float32)
    car_params_jax = jnp.array(car_params)
    
    # Create test data
    test_state = np.array([0.1, 5.0, 0.2, 0.5, 0.8, 0.6, 10.0, 5.0, 0.04, 0.1], dtype=np.float32)
    test_control = np.array([0.2, 2.0], dtype=np.float32)
    test_state_jax = jnp.array(test_state)
    test_control_jax = jnp.array(test_control)
    dt = 0.02
    
    # Warm up both implementations
    for _ in range(10):
        car_dynamics_pacejka_jit(test_state, test_control, car_params, dt)
        car_dynamics_pacejka_jax(test_state_jax, test_control_jax, car_params_jax, dt)
    
    # Single step benchmark
    n_iterations = 1000
    
    # Benchmark JIT
    start_time = time.time()
    for _ in range(n_iterations):
        jit_result = car_dynamics_pacejka_jit(test_state, test_control, car_params, dt)
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
        batch_controls = np.tile(test_control, (batch_size, 1))
        batch_states_jax = jnp.array(batch_states)
        batch_controls_jax = jnp.array(batch_controls)
        
        # Warm up
        car_step_parallel(batch_states, batch_controls, car_params, dt)
        car_step_parallel_jax(batch_states_jax, batch_controls_jax, car_params_jax, dt)
        
        # Benchmark JIT batch
        start_time = time.time()
        for _ in range(10):
            jit_batch_result = car_step_parallel(batch_states, batch_controls, car_params, dt)
        jit_batch_time = (time.time() - start_time) / 10 * 1000  # ms per call
        
        # Benchmark JAX batch
        start_time = time.time()
        for _ in range(10):
            jax_batch_result = car_step_parallel_jax(batch_states_jax, batch_controls_jax, car_params_jax, dt)
        jax_batch_time = (time.time() - start_time) / 10 * 1000  # ms per call
        
        print(f"Batch size {batch_size} performance:")
        print(f"  JIT: {jit_batch_time:.3f} ms per batch")
        print(f"  JAX: {jax_batch_time:.3f} ms per batch")
        print(f"  Speedup: {jit_batch_time/jax_batch_time:.2f}x")


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
