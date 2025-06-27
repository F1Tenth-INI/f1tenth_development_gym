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
# All-at-once MHE style
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
