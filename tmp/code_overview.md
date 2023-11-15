# MPC code

Note: `{...}` denotes arguments not listed for brevity.

File run_simulation.py

```python
driver = CarSystem(Settings.CONTROLLER)

obs, step_reward, done, info = env.reset(
        np.array(starting_positions) )

for simulation_index in trange(experiment_length):
    odom = get_odom(obs, index)
    # Add noise to car state
    ranges = obs['scans']
    angular_control, translational_control = driver.process_observation(ranges[index], odom)
    # TODO
```

File car_system.py
```python
class CarSystem():
    def __init__(self, controller, save_recording):
        self.waypoints_planner = mpc_planner()
        if controller == 'mpc':
            self.planner = mpc_planner()

    def process_observation()
        if self.use_waypoints_from_mpc:
            # Get optimal trajectory
            pass_data_to_planner(self.waypoints_planner, self.waypoint_utils.next_waypoints, car_state, obstacles)
            self.waypoints_planner.process_observation(ranges, ego_odom)
            self.waypoins_for_controller = self.waypoints_planner.mpc.optimizer.optimal_trajectory
        pass_data_to_planner(self.planner, self.waypoints_for_controller, car_state, obstacles)
        self.angular_control, self.translational_control = self.planner.process_observation(ranges, ego_odom)
        if hasattr(self.planner, 'optimal_control_sequence'):
            self.optimal_control_sequence = self.planner.optimal_control_sequence
            next_control_step = self.optimal_control_sequence[self.control_index % Settings.OPTIMIZE_EVERY_N_STEPS]
            self.angular_control = next_control_step[0]
            self.translational_control = next_control_step[1]          
        return self.angular_control, self.translational_control
```

Function mpc_planner.py
```python
class mpc_planner(template_planner):
    def __init__(self):
        self.mpc = controller_mpc(
            dt=Settings.TIMESTEP_PLANNER,
            environment_name="Car",
            initial_environment_attributes={
                "obstacles": self.obstacles,
                "lidar_points": self.lidar_points,
                "next_waypoints": self.waypoints,
                "target_point": self.target_point

            },
            control_limits=(control_limits_low, control_limits_high),
        )
        self.mpc.configure()

    def process_observation(self, ranges=None, ego_odom=None):
        angular_control, translational_control  = self.mpc.step(self.car_state, self.time,
            {
                "obstacles": self.obstacles,
                "lidar_points": self.lidar_points,
                "next_waypoints": self.waypoint_utils.next_waypoints,
                "target_point": self.target_point,
            })
        return angular_control, translational_control
```

Function controller_mpc.py

```python
class controller_mpc(template_controller:)
    def configure(self, optimizer_name: Optional[str]=None, predictor_specification: Optional[str]=None):
        self.predictor = PredictorWrapper()
        self.optimizer = Optimizer(
            predictor=self.predictor,
            cost_function=self.cost_function,
            control_limits=self.control_limits,
            optimizer_logging=self.controller_logging,
            computation_library=self.computation_library,
            calculate_optimal_trajectory=self.config_controller.get('calculate_optimal_trajectory'),
            **config_optimizer,
        )

        self.predictor.configure(
            batch_size=self.optimizer.num_rollouts,
            horizon=self.optimizer.mpc_horizon,
            dt=self.config_controller["dt"],
            computation_library=self.computation_library,
            variable_parameters=self.variable_parameters,
            predictor_specification=predictor_specification,
        )
        
        # Setup cost function: self.cost_function.configure(...)

        self.optimizer.configure(
            dt=self.config_controller["dt"],
            predictor_specification=predictor_specification,
            num_states=self.predictor.num_states,
            num_control_inputs=self.predictor.num_control_inputs,
        )

    def step(self, s, time, update_attributes):
        u = self.optimizer.step(s, time)
        # Online learning here
```

```python
class Optimizer:
    def __init__(self, predictor, {...}, num_rollout, mpc_horizon):
        # From super.__init__
        self.num_rollouts = num_rollouts
        self.mpc_horizon = mpc_horizon
        self.predictor = predictor

        # __init__
        self.predictor_single_trajectory = self.predictor.copy()

    def configure(self, num_states, num_control_inputs, dt, predictor_specification):
        # super.configure, not too interesting
        # configure
        self.predictor_single_trajectory.configure(
            batch_size=1, horizon=self.mpc_horizon, dt=dt,  # TF requires constant batch size
            predictor_specification=predictor_specification,
        )

    def step(self, s, time):
        self.u, self.u_nom, self.rollout_trajectories, traj_cost, u_run = self.predict_and_cost(s, self.u_nom, self.rng, self.u)
        if self.calculate_optimal_trajectory:
            self.optimal_trajectory = self.lib.to_numpy(self.predict_optimal_trajectory(s, self.u_nom))
        
    def _predict_and_cost(self, s, u_nom, random_gen, u_old):
        rollout_trajectory = self.predictor.predict_tf(s, u_run)
        traj_cost = self.get_mppi_trajectory_cost(rollout_trajectory, u_run, u_old, delta_u)
        u_nom = self.lib.clip(u_nom + self.reward_weighted_average(traj_cost, delta_u), self.action_low, self.action_high)
        self.update_internal_state(s, u_nom)
        return self.mppi_output(u, u_nom, rollout_trajectory, traj_cost, u_run) #Function not important

    def _predict_optimal_trajectory(self, s, u_nom):
        optimal_trajectory = self.predictor_single_trajectory.predict_tf(s, u_nom)
        self.predictor_single_trajectory.update(s=s, Q0=u_nom[:, :1, :])
        return optimal_trajectory
```

```python
class PredictorWrapper:
    def __init__(self): # Not too interesting

    def configure():
        if self.predictor_type == 'neural':
            from SI_Toolkit.Predictors.predictor_autoregressive_neural import predictor_autoregressive_neural
            self.predictor = predictor_autoregressive_neural(horizon=self.horizon, batch_size=self.batch_size, variable_parameters=variable_parameters, dt=dt, mode=mode, hls=hls, **self.predictor_config, **compile_standalone)
        elif self.predictor_type == 'ODE_TF':
            from SI_Toolkit.Predictors.predictor_ODE_tf import predictor_ODE_tf
            self.predictor = predictor_ODE_tf(horizon=self.horizon, dt=dt, batch_size=self.batch_size, variable_parameters=variable_parameters, **self.predictor_config, **compile_standalone)
```

```python
class predictor_autoregressive_neural(template_predictor):
    def __init__(self, model_name, path_to_model, horizon, dt, batch_size, variable_parameters, disable_individual_compilation, update_before_predicting, mode, hls):
        self.net, self.net_info = get_net(a, time_series_length=1, batch_size=self.batch_size, stateful=True)
        if np.any(['D_' in output_name for output_name in self.net_info.outputs]):
            self.differential_network = True
        self.normalization_info = get_norm_info_for_net(self.net_info)
        self.AL = autoregression_loop(
            model_inputs_len=len(self.net_info.inputs),
            model_outputs_len=len(self.net_info.outputs),
            batch_size=self.batch_size,
            lib=self.lib,
            differential_model_autoregression_helper_instance=self.dmah,
        )

    def _predict_tf(self, initial_state, Q):
        outputs = self.AL.run(
            model=self.net,
            horizon=self.horizon,
            initial_input=self.model_initial_input_normed,
            external_input_left=model_external_input_normed,
        )
        return outputs_augmented # Add some indices etc.

```

```python
class autoregression_loop:
    def __init__(self, model_inputs_len, model_outputs_len, batch_size, lib, differential_model_autoregression_helper_instance):


        def run(self, model, horizon, initial_input, external_input_left, external_input_right, predictor='neural'):
            model_output = self.evaluate_model(model, model_input)
            return outputs

        def evaluate_model(self, model, model_input):
            return model(model_input)
```

```python
class predictor_ODE_tf(template_predictor):
    def __init__(self, horizon, dt, intermediate_steps, disable_individual_compilation, batch_size, variable_parameters):
        self.next_step_predictor = next_state_predictor_ODE_tf(
            dt,
            intermediate_steps,
            self.batch_size,
            variable_parameters=variable_parameters,
            disable_individual_compilation=True,
        )

    def _predict_tf(self, initial_state, Q):
        for k in tf.range(self.horizon):
            next_state = self.next_step_predictor.step(next_state, Q[:, k, :])
        return self.output # All predictions in a single tensorarray

class next_state_predictor_ODE_tf():
    def __init__(self, dt, intermediate_steps, batch_size, variable_parameters, disable_individual_compilation):
        if environment_name == 'Car':
            from SI_Toolkit_ASF.car_model import car_model
            self.env = car_model(
                model_of_car_dynamics=model_of_car_dynamics,
                with_pid=with_pid,
                batch_size=batch_size,
                car_parameter_file=car_parameter_file,
                dt=dt,
                intermediate_steps=intermediate_steps,
            )  # Environment model, keeping car ODEs

    def _step(self, s, Q):
        return self.env.step_dynamics(s, Q, None)


class car_model:
    def __init__(self, model_of_car_dynamics, with_pid, batch_size, car_parameter_file,  dt, intermediate_steps, computation_lib):
        # Advances one step according to ODE.
        #!! Problem: Can only predict for a given batch_size! The functions do not work for different sizes of inputs. Only one input size per class.
```


# Environment stepping

## Gym environment:
```python
for i in range(int(Settings.TIMESTEP_CONTROL/env.timestep)):
    if Settings.WITH_PID:
        accl, sv = pid(...)  # From f110_gym.envs.dynamic_models
    else:
        accl, sv = translational_control_with_noise, angular_control_with_noise
    
    obs, step_reward, done, info = env.step(np.array(controlls))
    laptime += step_reward

class F110Env(gym.Env):
    def step(self, action):
        self.sim.step(action)

class Simulator(object):
    def step(self, control_inputs):
        current_scan = agent.update_pose(control_inputs[i, 0], control_inputs[i, 1])

class RaceCar(Object):
    def update_pose(self, raw_steer, vel):
        f = vehicle_dynamics_st(...)
        self.state = self.state + f * self.time_step
        self.state[4] = wrap_angle_rad(self.state[4])  # Bounds angle by [0, 2 * pi]
```

## Brunton Test:
```python
output = predictor.predict(initial_input, external_input)

class Predictor_ODE_tf():
    def predict(self, inital_state, Q)
        return self.predict_tf(self.initial_state, Q)
    
    def _predict_tf(self, intial_state, Q):
        for k in tf.range(self.horizon):
            next_state = self.next_step_predictor.step(next_state, Q[:, k, :])

class next_state_predictor_ODE_tf():
    def __init__(...):
        self.env = car_model()
    def _step(self, s, Q):
        return self.env.step_dynamics(s, Q, None)

class car_model():
    def _step_model_with_servo_and_motor_pid_(self, model):
        def _step_model_with_servo_and_motor_pid(s, Q, params):
            delta_dot = self.servo_proportional(desired_angle, delta)
            vel_x_dot = self.motor_controller_pid(desired_speed, vel_x)
            return model(s, Q_pid, params)  # Here model is ST Model
        return _step_model_with_servo_and_motor_pid
    
    def _step_dynamics_st(self, s, Q, params):
        for _ in range(self.intermediate_steps):
            step_states without PID
```
## All in one
```python
for i in range(int(Settings.TIMESTEP_CONTROL/env.timestep)):
    accl, sv = pid(...)
    f = vehicle_dynamics_st(...)
    self.state += f * self.time_step  # dt 0.01
    self.state[4] = wrap_angle_rad(self.state[4])
```

```python
for k in tf.range(self.horizon):
    delta_dot = self.servo_proportional(desired_angle, delta)
    vel_x_dot = self.motor_controller_pid(desired_speed, vel_x)
    step_states_without PID()
```

# PID

## Steering

### F1TENTH
```python
steer_diff = steer - current_steer
if np.fabs(steer_diff) > 1e-4:
    sv = (steer_diff / np.fabs(steer_diff)) * max_sv  # max_sv = 3.2
else:
    sv = 0.0
```

### Predictor
```python
steering_diff_low = self.car_parameters['steering_diff_low']
servo_p = self.car_parameters['servo_p']

steering_angle_difference = desired_steering_angle - current_steering_angle

steering_angle_difference_not_too_low_indices = tf.math.greater(tf.math.abs(steering_angle_difference), steering_diff_low)

steering_velocity = steering_angle_difference_not_too_low_indices * (steering_angle_difference * servo_p)

return steering_velocity
```


# State overview

## F1TENTH dynamics
```python
state = [pose_x, pose_y, steering_angle, linear_vel, pose_theta, angular_vel, slip_angle]
```

## Predictor dynamics
```python
state = [('angular_control', 'translational_control'), 'angular_vel_z', 'linear_vel_x', 'pose_theta', 'pose_theta_cos', 'pose_theta_sin', 'pose_x', 'pose_y', 'slip_angle', 'steering_angle']
```

Pose theta is defined as: Along x is zero, then it increases clockwise (I think)

Dimensions:
    - Input = [batch_size, timesteps, states]
    - Output = [batch_size, states]

# Parameters

Dense
    - 32-128-32: 8968
    - 64-64-64: 9673
    - 64-64-64-64: 13704
    - 64-128-256-128-64: 83720
    - 64-128-64: 17800
    - 128-128: 19000
    - 128-128-128: 35464
    - 128-128-128-128:
    - 128-128-128-128-128: 68500
    - 256-256:
    - 256-256-256: 136500
    - 128-256-256-128:
    - 256: 4800

LSTM
    - 32-32-32: 22408
    - 64-64: 52744
    - 32-64-32: 43016
    - 16-32-16: 11272
    - 8-32-8: 7240
