import tensorflow as tf
import numpy as np

###############################################################################
# Some placeholders for your original imports.
# Make sure the following symbols are appropriately defined in your environment:
# - TensorFlowLibrary
# - car_model
# - Settings
# - POSE_THETA_IDX, SLIP_ANGLE_IDX, POSE_THETA_SIN_IDX, POSE_THETA_COS_IDX, ANGULAR_VEL_Z_IDX
###############################################################################
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

###############################################################################
# OLD CODE (step-by-step Newton), renamed to CarInverseDynamicsOld
###############################################################################
class CarInverseDynamicsOld:
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
            states_ta = states_ta.write(0, x_next_tf[0])  # Store final/current state at index 0

            conv_ta = tf.TensorArray(dtype=tf.bool, size=T)

            def solve_single_step(x_next, q, tol=1e-6, max_iter=20):
                """
                Newton solver for x_prev s.t. f(x_prev, q) = x_next.
                """
                i0 = tf.constant(0)
                converged0 = tf.constant(False, dtype=tf.bool)  # shape=()

                x_prev0 = x_next  # initial guess, shape [1, state_dim]

                def cond(i, x_prev, converged):
                    return tf.logical_and(tf.less(i, max_iter),
                                          tf.logical_not(converged))

                def body(i, x_prev, converged):
                    x_prev = tf.ensure_shape(x_prev, [1, self.state_dim])
                    q_fixed = tf.ensure_shape(q, [1, self.control_dim])

                    # 1) Compute forward pass
                    x_pred = self._f(x_prev, q_fixed)  # shape [1, state_dim]
                    diff = x_pred - x_next  # shape [1, state_dim]

                    # 2) Adjust the angle difference
                    def angle_diff(th):
                        return tf.atan2(tf.sin(th), tf.cos(th))

                    corrected_theta_diff = tf.abs(angle_diff(diff[:, POSE_THETA_IDX]))
                    diff_abs = tf.abs(diff)

                    # 3) Zero out slip angle difference
                    diff_abs = tf.concat([
                        diff_abs[:, :POSE_THETA_IDX],
                        tf.reshape(corrected_theta_diff, [1, 1]),
                        diff_abs[:, POSE_THETA_IDX + 1: SLIP_ANGLE_IDX],
                        tf.zeros_like(diff_abs[:, SLIP_ANGLE_IDX: SLIP_ANGLE_IDX + 1]),
                        diff_abs[:, SLIP_ANGLE_IDX + 1:],
                    ], axis=1)

                    # 4) Newton update
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

                    c1 = tf.reduce_all(diff_check < tol)
                    # The old code also has a stricter check for ANGULAR_VEL_Z_IDX
                    c2 = tf.reduce_all(tf.abs(diff_new[:, ANGULAR_VEL_Z_IDX]) < (tol / 100.0))
                    new_converged = tf.logical_and(c1, c2)

                    return i + 1, x_prev_new, new_converged

                shape_invariants = (
                    tf.TensorShape([]),
                    tf.TensorShape([1, self.state_dim]),
                    tf.TensorShape([]),
                )
                loop_vars = (i0, x_prev0, converged0)

                _, x_prev_final, conv_final = tf.while_loop(
                    cond, body,
                    loop_vars=loop_vars,
                    shape_invariants=shape_invariants
                )

                # fallback acceptance if under 1e-3 ignoring slip angle
                if tf.logical_not(conv_final):
                    x_pred_final = self._f(x_prev_final, q)
                    diff_final = x_pred_final - x_next
                    diff_abs_final = tf.abs(diff_final)
                    diff_check_final = tf.concat([
                        diff_abs_final[:, :SLIP_ANGLE_IDX],
                        diff_abs_final[:, SLIP_ANGLE_IDX + 1:]
                    ], axis=1)
                    if tf.reduce_max(diff_check_final) < 1e-3:
                        conv_final = True

                return x_prev_final, conv_final

            # main backward loop for T steps
            x_current = x_next_tf
            for i in tf.range(T):
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


###############################################################################
# Some helper functions for the new solver
###############################################################################
def _angle_wrap_diff(thA, thB):
    """Return angle difference (wrapped to [-pi, pi])."""
    return tf.atan2(tf.sin(thA - thB), tf.cos(thA - thB))

def _partial_state_diff(xA, xB):
    """
    Computes a difference measure between two states xA, xB ignoring slip angle,
    but wrapping the heading angle. Returns a scalar squared-error.
    xA, xB: shape [10].
    """
    POSE_THETA = xA[POSE_THETA_IDX]
    POSE_THETA_B = xB[POSE_THETA_IDX]

    # sum of squares for the part before heading
    d = tf.reduce_sum((xA[:POSE_THETA_IDX] - xB[:POSE_THETA_IDX])**2)

    # heading angle difference
    angle_err = _angle_wrap_diff(POSE_THETA, POSE_THETA_B)
    d += angle_err**2

    # intermediate block between heading and slip angle
    if SLIP_ANGLE_IDX > (POSE_THETA_IDX + 1):
        d += tf.reduce_sum(
            (xA[POSE_THETA_IDX+1:SLIP_ANGLE_IDX] - xB[POSE_THETA_IDX+1:SLIP_ANGLE_IDX])**2
        )

    # skip slip angle dimension
    # sum of squares for the rest
    d += tf.reduce_sum((xA[SLIP_ANGLE_IDX+1:] - xB[SLIP_ANGLE_IDX+1:])**2)

    return d

def _fix_sin_cos(x):
    """
    Returns a copy of x (shape [10]) with the sin/cos entries made consistent
    with the heading angle entry.
    """
    angle = x[POSE_THETA_IDX]
    sin_val = tf.sin(angle)
    cos_val = tf.cos(angle)
    x_new = tf.tensor_scatter_nd_update(
        x, [[POSE_THETA_SIN_IDX], [POSE_THETA_COS_IDX]],
        [sin_val, cos_val]
    )
    return x_new


###############################################################################
# NEW CODE (all-at-once MHE style), with fixes
###############################################################################
class CarInverseDynamics:
    """
    Single “MHE-like” optimization for the entire backward horizon, corrected to:
      - Use the OLD solver as the initial guess (ensuring good start).
      - Keep sin/cos consistent with angle during each gradient iteration.
      - Same partial difference ignoring slip angle, with angle wrap.
    """

    def __init__(self, mu=None):
        # 1) Car model setup
        self.computation_lib = TensorFlowLibrary()
        self.car_model = car_model(
            model_of_car_dynamics=Settings.ODE_MODEL_OF_CAR_DYNAMICS,
            batch_size=1,
            car_parameter_file=Settings.CONTROLLER_CAR_PARAMETER_FILE,
            dt=0.01,
            intermediate_steps=1,
            computation_lib=self.computation_lib,
        )
        if mu is not None:
            self.car_model.change_friction_coefficient(mu)

        self.state_dim = 10
        self.control_dim = 2
        self._f = self.car_model.step_dynamics_core  # single-step forward dynamics

        # 2) Max horizon length
        self.MAX_T = 5000

        # 3) Persistent TF Variables outside tf.function
        self.X_var = tf.Variable(
            tf.zeros([self.MAX_T, self.state_dim], dtype=tf.float32),
            trainable=True,
            name='X_var'
        )
        self.Q_var = tf.Variable(
            tf.zeros([self.MAX_T, self.control_dim], dtype=tf.float32),
            trainable=False,
            name='Q_var'
        )
        self.x_next_var = tf.Variable(
            tf.zeros([self.state_dim], dtype=tf.float32),
            trainable=False,
            name='x_next_var'
        )
        self.T_var = tf.Variable(0, dtype=tf.int32, trainable=False, name='T_var')

        # 4) Adam optimizer
        self.base_optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=1e-2)

        # 5) Additional hyperparams
        self.REG_TEMPORAL = 0.001   # weight for temporal smoothness
        self.TOL = 1e-2            # tolerance for dynamic consistency
        self.PHASE1_STEPS = 50    # first optimization phase
        self.PHASE2_STEPS = 50    # second optimization phase
        self.LR1 = 1e-2            # LR for phase 1
        self.LR2 = 1e-3            # LR for phase 2

        # 6) Create an instance of the old solver for the initial guess
        self.old_solver = CarInverseDynamicsOld(mu=mu)

    def change_friction_coefficient(self, mu):
        self.car_model.change_friction_coefficient(mu)

    ###########################################################################
    # Public entry point
    ###########################################################################
    def inverse_entire_trajectory(self, x_next, Q):
        """
        x_next: shape [1, 10], final state
        Q: shape [T, 2], reversed in time

        Returns:
          states_np: shape [T+1, 10]
            states_np[0] = x_next
            states_np[i+1] = older states
          conv_np: shape [T], booleans
        """
        # 1) Check shapes
        if x_next.shape != (1, 10):
            raise ValueError(f"x_next must be [1, 10]. Got {x_next.shape}")
        if Q.ndim != 2 or Q.shape[1] != 2:
            raise ValueError(f"Q must be [T, 2]. Got {Q.shape}")

        T = Q.shape[0]
        if T > self.MAX_T:
            raise ValueError(f"Requested horizon T={T} exceeds MAX_T={self.MAX_T}.")

        # 2) Copy data into persistent variables
        self.x_next_var.assign(x_next[0])   # shape [10]
        self.Q_var[:T].assign(Q)            # shape [T, 2]
        self.T_var.assign(T)

        # 3) Use the OLD solver result as the initial guess
        #    The old solver also expects x_next: shape [1, 10], Q: shape [T, 2].
        states_old, _ = self.old_solver.inverse_entire_trajectory(
            x_next, Q
        )  # shape [T+1, 10], states_old[0] = x_next, states_old[i+1] = older state
        # Fill X_var: we want X_var[i] = states_old[i+1], i in [0..T-1]
        self.X_var[:T].assign(states_old[1:T+1])

        # 4) Run the compiled multi-phase optimization
        states_tf, conv_tf = self._run_inverse_trajectory_compiled()

        # Convert to numpy
        return states_tf.numpy(), conv_tf.numpy()

    ###########################################################################
    # The compiled multi-phase optimization
    ###########################################################################
    @tf.function
    def _run_inverse_trajectory_compiled(self):
        T = self.T_var

        # cost function with sin/cos fix
        def cost_fn():
            cost_val = 0.0

            # Loop over steps i=0..T-1:
            #   x_i is the "newer" state (closer to final),
            #   x_im1 is the "older" state we want to find.
            for i in tf.range(T):
                if i == 0:
                    # the newest (final) state is x_next_var
                    x_i_fixed = _fix_sin_cos(self.x_next_var)
                else:
                    x_i_fixed = _fix_sin_cos(self.X_var[i-1, :])

                x_im1_fixed = _fix_sin_cos(self.X_var[i, :])
                q_i = self.Q_var[i, :]

                # Forward dynamics from x_im1 -> x_pred
                x_pred = self._f(x_im1_fixed[tf.newaxis, :],
                                 q_i[tf.newaxis, :])[0]
                # Add partial-state difference
                cost_val += _partial_state_diff(x_pred, x_i_fixed)

            # Temporal regularization: encourage X[i] ~ X[i-1]
            for i in tf.range(1, T):
                x_cur = _fix_sin_cos(self.X_var[i, :])
                x_prev = _fix_sin_cos(self.X_var[i-1, :])
                cost_val += self.REG_TEMPORAL * _partial_state_diff(x_cur, x_prev)

            return cost_val

        # Phase 1: set LR1
        self.base_optimizer.learning_rate.assign(self.LR1)
        for _ in tf.range(self.PHASE1_STEPS):
            with tf.GradientTape() as tape:
                c = cost_fn()
            grads = tape.gradient(c, [self.X_var])
            self.base_optimizer.apply_gradients(zip(grads, [self.X_var]))

        # Phase 2: set LR2
        self.base_optimizer.learning_rate.assign(self.LR2)
        for _ in tf.range(self.PHASE2_STEPS):
            with tf.GradientTape() as tape:
                c = cost_fn()
            grads = tape.gradient(c, [self.X_var])
            self.base_optimizer.apply_gradients(zip(grads, [self.X_var]))

        #######################################################################
        # Build final output states and check consistency
        #######################################################################
        states_ta = tf.TensorArray(dtype=tf.float32, size=T+1)
        # First entry is the pinned final state (with sin/cos fixed)
        states_ta = states_ta.write(0, _fix_sin_cos(self.x_next_var))

        # Next T entries are the older states
        for i in tf.range(T):
            x_candidate = _fix_sin_cos(self.X_var[i, :])
            # save it back into X_var for consistency
            self.X_var[i, :].assign(x_candidate)
            states_ta = states_ta.write(i+1, x_candidate)

        # Evaluate dynamic consistency => converged_flags
        conv_ta = tf.TensorArray(dtype=tf.bool, size=T)
        for i in tf.range(T):
            if i == 0:
                x_i = states_ta.read(0)
            else:
                x_i = states_ta.read(i)
            x_im1 = states_ta.read(i+1)
            q_i = self.Q_var[i, :]
            x_pred = self._f(x_im1[tf.newaxis, :], q_i[tf.newaxis, :])[0]
            err = _partial_state_diff(x_pred, x_i)
            conv_ta = conv_ta.write(i, err < self.TOL)

        return states_ta.stack(), conv_ta.stack()
