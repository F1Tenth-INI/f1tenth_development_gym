import tensorflow as tf
import numpy as np

from SI_Toolkit.computation_library import TensorFlowLibrary

from SI_Toolkit_ASF.car_model import car_model
from utilities.Settings import Settings
from utilities.state_utilities import POSE_THETA_IDX, SLIP_ANGLE_IDX, POSE_THETA_SIN_IDX, POSE_THETA_COS_IDX, ANGULAR_VEL_Z_IDX


class CarInverseDynamics:
    def __init__(self, mu=None):
        self.car_model = car_model(
            model_of_car_dynamics=Settings.ODE_MODEL_OF_CAR_DYNAMICS,
            batch_size=1,
            car_parameter_file=Settings.CONTROLLER_CAR_PARAMETER_FILE,
            dt=0.01,
            intermediate_steps=1,
            computation_lib=TensorFlowLibrary()
        )

        if mu is not None:
            self.car_model.car_parameters.mu = mu
        self.inverse_dynamics = create_inverse_function_tf(self.car_model.step_dynamics)
        self.inverse_dynamics_core = create_inverse_function_tf(self.car_model.step_dynamics_core)

    def change_friction_coefficient(self, mu):
        self.car_model.change_friction_coefficient(mu)
        self.inverse_dynamics = create_inverse_function_tf(self.car_model.step_dynamics)
        self.inverse_dynamics_core = create_inverse_function_tf(self.car_model.step_dynamics_core)

    def step(self, s, Q):
        return self.inverse_dynamics(s, Q)

    def step_core(self, s, Q):
        return self.inverse_dynamics_core(s, Q)



def create_inverse_function_tf(f, tol=1e-6, max_iter=20):
    """
    Creates an inverse function solver that, given y and q,
    finds x satisfying f(x, q) = y using Newton's method in TensorFlow.

    Parameters:
      f      : A function that accepts (x, q) and returns a Tensor.
      tol    : Tolerance for convergence.
      max_iter: Maximum number of iterations.

    Returns:
      A function inverse_f_param_tf(y, q) that returns the computed x.
    """

    @tf.function
    def newton_step_param_tf(x, y, q, alpha):
        """
        Performs one Newton update step for solving f(x, q) = y.

        The update is:
          x_new = x - alpha * err / (grad_val + 1e-12)
        where err is the error computed by g(x) and grad_val is the
        element-wise derivative (diagonal of the Jacobian) of g with respect to x.

        Parameters:
          x, y, q: Tensors for the current guess, target value, and parameter.
          alpha : Damping factor controlling the step size.

        Returns:
          The updated x after one Newton step.
        """

        # Error function g: computes a modified absolute difference between f(x, q) and y.
        # (Don't change function g as per your instruction.)
        def g(x_proposed):
            diff = f(x_proposed, q) - y
            corrected_theta_diff = tf.abs(
                tf.atan2(tf.sin(diff[:, POSE_THETA_IDX]), tf.cos(diff[:, POSE_THETA_IDX]))
            )
            diff_abs = tf.abs(diff)
            diff_abs = tf.concat(
                [
                    diff_abs[:, :POSE_THETA_IDX],
                    tf.expand_dims(corrected_theta_diff, axis=1),
                    diff_abs[:, POSE_THETA_IDX + 1:SLIP_ANGLE_IDX],
                    tf.expand_dims(diff_abs[:, SLIP_ANGLE_IDX]*0.0, axis=1),
                    diff_abs[:, SLIP_ANGLE_IDX+1:],
                 ],
                axis=1
            )
            return diff_abs

        # Compute the error vector using g.
        # We then need its derivative (Jacobian) with respect to x to form the Newton update.
        with tf.GradientTape() as tape:
            tape.watch(x)
            err = g(x)  # Error vector: ideally, we want err = 0.
        # Compute the full Jacobian of err with respect to x.
        # The shape is [batch_size, dim, dim] if x has shape [batch_size, dim].
        grad_jac = tape.batch_jacobian(err, x)
        # For a decoupled update, we assume the influence is primarily on the diagonal.
        # Extract the diagonal elements to get an element-wise derivative.
        grad_val = tf.linalg.diag_part(grad_jac)

        # Perform the Newton update.
        # The term (grad_val + 1e-12) safeguards against division by zero.
        x_new = x - alpha * err / (grad_val + 1e-12)

        # --- Post-update correction for angular components ---
        # The updated x may have changed the angle (at POSE_THETA_IDX),
        # so we recompute its sine and cosine values.
        batch_size_x = tf.shape(x_new)[0]
        batch_size_x_int64 = tf.cast(batch_size_x, tf.int64)

        # Build indices for updating the sin(theta) component.
        sin_indices = tf.concat([
            tf.expand_dims(tf.range(batch_size_x_int64, dtype=tf.int64), axis=-1),
            tf.fill([batch_size_x_int64, 1], tf.cast(POSE_THETA_SIN_IDX, tf.int64))
        ], axis=1)
        # Build indices for updating the cos(theta) component.
        cos_indices = tf.concat([
            tf.expand_dims(tf.range(batch_size_x_int64, dtype=tf.int64), axis=-1),
            tf.fill([batch_size_x_int64, 1], tf.cast(POSE_THETA_COS_IDX, tf.int64))
        ], axis=1)

        # Compute new sin and cos values from the updated angle.
        sin_values = tf.sin(x_new[:, POSE_THETA_IDX])
        cos_values = tf.cos(x_new[:, POSE_THETA_IDX])

        # Update the corresponding indices in x_new.
        x_new = tf.tensor_scatter_nd_update(x_new, sin_indices, sin_values)
        x_new = tf.tensor_scatter_nd_update(x_new, cos_indices, cos_values)

        return x_new

    def inverse_f_param_tf(y, q):
        """
        Solves f(x, q) = y for x using Newton's method.

        Parameters:
          y : The target Tensor.
          q : An additional parameter Tensor for f.

        Returns:
          A NumPy array containing the solution x.
        """
        y = tf.convert_to_tensor(y, dtype=tf.float32)
        q = tf.convert_to_tensor(q, dtype=tf.float32)

        # Use y as the initial guess for x.
        x = y

        # A damping factor that can be adjusted.
        alpha = tf.constant(1.0, dtype=tf.float32)

        diff0 = tf.abs(f(x, q) - y)
        diff0_without_slip = tf.concat([diff0[:, :SLIP_ANGLE_IDX], diff0[:, SLIP_ANGLE_IDX + 1:]], axis=1)
        diffs_without_slip = [diff0_without_slip]
        for i in range(max_iter):
            x_new = newton_step_param_tf(x, y, q, alpha)
            # Compute the raw difference for convergence checking.
            diff = tf.abs(f(x_new, q) - y)
            # Uncomment the following line for debugging:
            # tf.print("Iter", i, ": Error =", norm_diff, "; Alpha =", alpha)
            # Check if all components are below the tolerance.
            diff_without_slip = tf.concat([diff[:, :SLIP_ANGLE_IDX], diff[:, SLIP_ANGLE_IDX + 1:]], axis=1)
            diffs_without_slip.append(diff_without_slip.numpy())
            if tf.reduce_all(diff_without_slip < tol) and tf.reduce_all(diff_without_slip[:, ANGULAR_VEL_Z_IDX] < tol/100.0):
                # tf.print("\nConverged after", i, "iterations.\n")
                return x_new.numpy(), True
            x = x_new
        # Return the best estimate if max iterations are reached.
        diffs_without_slip = np.array(diffs_without_slip)
        if np.max(diffs_without_slip[-1]) > 1.e-3:
            print("NO CONVERGENCE REACHED!")
            print("Max error: ", diffs_without_slip[-1])
            return x.numpy(), False
        return x.numpy(), True

    return inverse_f_param_tf