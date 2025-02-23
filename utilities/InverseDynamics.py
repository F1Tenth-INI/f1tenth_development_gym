# InverseDynamics.py

import tensorflow as tf
import numpy as np

from SI_Toolkit.computation_library import TensorFlowLibrary
from SI_Toolkit_ASF.car_model import car_model
from utilities.Settings import Settings
from utilities.state_utilities import (
    POSE_THETA_IDX,
    SLIP_ANGLE_IDX,
    POSE_THETA_SIN_IDX,
    POSE_THETA_COS_IDX,
    ANGULAR_VEL_Z_IDX,
)


class CarInverseDynamics:
    """
    Performs a backward pass from final state x_next with a history of controls Q.
    Uses Newton iteration for each step, pinned shapes, and a scalar 'converged' flag.
    """

    def __init__(self, mu=None):
        self.car_model = car_model(
            model_of_car_dynamics=Settings.ODE_MODEL_OF_CAR_DYNAMICS,
            batch_size=1,  # typically 1
            car_parameter_file=Settings.CONTROLLER_CAR_PARAMETER_FILE,
            dt=0.01,
            intermediate_steps=1,
            computation_lib=TensorFlowLibrary(),
        )
        if mu is not None:
            self.car_model.change_friction_coefficient(mu)

        # Make sure these match your actual model's shape
        self.state_dim = 10
        self.control_dim = 2

        # Our forward step function
        self._f = self.car_model.step_dynamics_core

        # Build a single @tf.function for the entire backward pass
        self.inverse_trajectory_tf = self._create_inverse_trajectory_fn()

    def change_friction_coefficient(self, mu):
        self.car_model.change_friction_coefficient(mu)

    def _create_inverse_trajectory_fn(self):
        @tf.function(
            input_signature=[
                tf.TensorSpec([1, None], dtype=tf.float32),  # x_next: [1, state_dim]
                tf.TensorSpec([None, None], dtype=tf.float32)  # Q: [T, control_dim]
            ]
        )
        def run_inverse_trajectory(x_next_tf, Q_tf):
            """
            x_next_tf: [1, state_dim]
            Q_tf:      [T, control_dim]

            Returns:
              states:  [T+1, state_dim]
              conv:    [T] bool
            """
            T = tf.shape(Q_tf)[0]

            states_ta = tf.TensorArray(dtype=tf.float32, size=T + 1)
            states_ta = states_ta.write(0, x_next_tf[0])  # Store current final state at index 0

            conv_ta = tf.TensorArray(dtype=tf.bool, size=T)

            def solve_single_step(x_next, q, tol=1e-6, max_iter=20):
                """
                Newton solver for x_prev s.t. f(x_prev, q) = x_next.
                """
                i0 = tf.constant(0)
                converged0 = tf.constant(False, dtype=tf.bool)  # shape=()

                x_prev0 = x_next  # initial guess, shape [1, state_dim]

                def cond(i, x_prev, converged):
                    # Keep looping while i < max_iter and not converged
                    return tf.logical_and(tf.less(i, max_iter),
                                          tf.logical_not(converged))

                def body(i, x_prev, converged):
                    # Pin shapes to keep TF happy
                    x_prev = tf.ensure_shape(x_prev, [1, self.state_dim])
                    q_fixed = tf.ensure_shape(q, [1, self.control_dim])

                    # 1) Compute forward pass, diff
                    x_pred = self._f(x_prev, q_fixed)  # shape [1, state_dim]
                    diff = x_pred - x_next  # shape [1, state_dim]

                    # 2) Adjust the angle difference
                    def angle_diff(th):
                        return tf.atan2(tf.sin(th), tf.cos(th))

                    corrected_theta_diff = tf.abs(angle_diff(diff[:, POSE_THETA_IDX]))
                    diff_abs = tf.abs(diff)

                    # 3) Zero out slip angle difference, etc.
                    diff_abs = tf.concat([
                        diff_abs[:, :POSE_THETA_IDX],
                        tf.reshape(corrected_theta_diff, [1, 1]),
                        diff_abs[:, POSE_THETA_IDX + 1: SLIP_ANGLE_IDX],
                        tf.zeros_like(diff_abs[:, SLIP_ANGLE_IDX: SLIP_ANGLE_IDX + 1]),
                        diff_abs[:, SLIP_ANGLE_IDX + 1:],
                    ], axis=1)

                    # 4) Newton update requires gradient
                    with tf.GradientTape() as tape:
                        tape.watch(x_prev)
                        x_pred2 = self._f(x_prev, q_fixed)
                        diff2 = x_pred2 - x_next

                        # same angle logic
                        corrected_theta_diff2 = tf.abs(angle_diff(diff2[:, POSE_THETA_IDX]))
                        diff_abs2 = tf.abs(diff2)
                        diff_abs2 = tf.concat([
                            diff_abs2[:, :POSE_THETA_IDX],
                            tf.reshape(corrected_theta_diff2, [1, 1]),
                            diff_abs2[:, POSE_THETA_IDX + 1: SLIP_ANGLE_IDX],
                            tf.zeros_like(diff_abs2[:, SLIP_ANGLE_IDX: SLIP_ANGLE_IDX + 1]),
                            diff_abs2[:, SLIP_ANGLE_IDX + 1:],
                        ], axis=1)

                    grad_jac = tape.batch_jacobian(diff_abs2, x_prev)  # shape [1, dim, dim]
                    grad_val = tf.linalg.diag_part(grad_jac)[0]  # shape [dim]

                    eps = 1e-12
                    x_prev_new = x_prev - diff_abs2 / (grad_val + eps)

                    # 5) Recompute sin/cos for updated angle
                    angle_updated = x_prev_new[:, POSE_THETA_IDX]
                    sin_angle = tf.sin(angle_updated)
                    cos_angle = tf.cos(angle_updated)

                    x_prev_new = tf.tensor_scatter_nd_update(
                        x_prev_new, [[0, POSE_THETA_SIN_IDX]], [sin_angle[0]]
                    )
                    x_prev_new = tf.tensor_scatter_nd_update(
                        x_prev_new, [[0, POSE_THETA_COS_IDX]], [cos_angle[0]]
                    )

                    # 6) Check convergence
                    x_pred_new = self._f(x_prev_new, q_fixed)
                    diff_new = x_pred_new - x_next
                    diff_abs_new = tf.abs(diff_new)

                    # Omit slip angle dimension from the check
                    diff_check = tf.concat([
                        diff_abs_new[:, :SLIP_ANGLE_IDX],
                        diff_abs_new[:, SLIP_ANGLE_IDX + 1:],
                    ], axis=1)

                    # IMPORTANT: make c1, c2 into scalars
                    c1 = tf.reduce_all(diff_check < tol)  # shape=()
                    # might be shape(1,) if we don't reduce_all
                    c2 = tf.reduce_all(tf.abs(diff_new[:, ANGULAR_VEL_Z_IDX]) < (tol / 100.0))  # shape=()
                    new_converged = tf.logical_and(c1, c2)  # shape=()

                    return i + 1, x_prev_new, new_converged

                # shape invariants must match actual shapes
                shape_invariants = (
                    tf.TensorShape([]),  # i is scalar int
                    tf.TensorShape([1, self.state_dim]),  # x_prev is [1, state_dim]
                    tf.TensorShape([]),  # converged is scalar bool
                )
                loop_vars = (i0, x_prev0, converged0)

                # Run up to max_iter
                _, x_prev_final, conv_final = tf.while_loop(
                    cond, body,
                    loop_vars=loop_vars,
                    shape_invariants=shape_invariants
                )

                # --- Fallback acceptance if strict tol not met ---
                #
                # Match old code: after the loop, if conv_final is False,
                # check if final error is below 1e-3 (ignoring slip angle).
                if tf.logical_not(conv_final):
                    x_pred_final = self._f(x_prev_final, q)
                    diff_final = x_pred_final - x_next
                    diff_abs_final = tf.abs(diff_final)

                    # omit slip angle dimension
                    diff_check_final = tf.concat([
                        diff_abs_final[:, :SLIP_ANGLE_IDX],
                        diff_abs_final[:, SLIP_ANGLE_IDX + 1:]
                    ], axis=1)

                    # if under 1e-3, accept anyway
                    if tf.reduce_max(diff_check_final) < 1e-3:
                        conv_final = True

                return x_prev_final, conv_final

            # main backward loop for T steps
            x_current = x_next_tf
            for i in tf.range(T):
                # slice control for step i
                q_i = Q_tf[i: i + 1, :]
                q_i = tf.ensure_shape(q_i, [1, self.control_dim])
                x_current = tf.ensure_shape(x_current, [1, self.state_dim])

                x_prev, conv = solve_single_step(x_current, q_i)
                states_ta = states_ta.write(i + 1, x_prev[0])
                conv_ta = conv_ta.write(i, conv)
                x_current = x_prev

            return states_ta.stack(), conv_ta.stack()

        return run_inverse_trajectory

    def inverse_entire_trajectory(self, x_next, Q):
        """
        Public function to do the entire backward pass in one call.
        x_next: shape [1, state_dim]
        Q: shape [T, control_dim]

        Returns states_np [T+1, state_dim], conv_np [T] bool
        """
        x_next_tf = tf.convert_to_tensor(x_next, dtype=tf.float32)
        Q_tf = tf.convert_to_tensor(Q, dtype=tf.float32)

        states_tf, conv_tf = self.inverse_trajectory_tf(x_next_tf, Q_tf)
        return states_tf.numpy(), conv_tf.numpy()
