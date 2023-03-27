from Control_Toolkit_ASF.Controllers.MPC.mpc_planner import mpc_planner
from Control_Toolkit.Controllers.controller_mpc import controller_mpc
from utilities.Settings import Settings
import yaml, os
import numpy as np


class mpc_multiplanner(mpc_planner):
    """
    Multi Planner
    """

    def __init__(self):
        super().__init__()

        if Settings.ENVIRONMENT_NAME == 'Car':
            num_states = 9
            num_control_inputs = 2
            if not Settings.WITH_PID:  # MPC return velocity and steering angle
                control_limits_low, control_limits_high = self.get_control_limits([[-3.2, -9.5], [3.2, 9.5]])
            else:  # MPC returns acceleration and steering velocity
                control_limits_low, control_limits_high = self.get_control_limits([1.066, 20])
        else:
            raise NotImplementedError('{} mpc not implemented yet'.format(Settings.ENVIRONMENT_NAME))
        config_controllers = yaml.load(
            open(os.path.join("Control_Toolkit_ASF", "config_controllers.yml")),
            Loader=yaml.FullLoader
        )
        optimizers_names = config_controllers['multimpc']['optimizers']
        self.redundant_controllers = []
        for optimizer_name in optimizers_names:
            redundant_controller = controller_mpc(
                dt=Settings.TIMESTEP_CONTROL,
                environment_name="Car",
                initial_environment_attributes={
                    "lidar_points": self.lidar_points,
                    "next_waypoints": self.waypoint_utils.next_waypoints,
                    "target_point": self.target_point

                },
                num_states=num_states,
                num_control_inputs=num_control_inputs,
                control_limits=(control_limits_low, control_limits_high),
            )
            redundant_controller.configure(optimizer_name=optimizer_name)
            self.redundant_controllers += [redundant_controller]
        pass

    def process_observation(self, ranges=None, ego_odom=None):
        super().process_observation(ranges, ego_odom)

        s = self.car_state
        rollout_trajectories_tuple = (self.mpc.optimizer.rollout_trajectories,)

        for redundant_controller in self.redundant_controllers:
            redundant_controller.step(s,
                                      self.time,
                                      {
                                           "lidar_points": self.lidar_points,
                                           "next_waypoints": self.waypoint_utils.next_waypoints,
                                           "target_point": self.target_point,
                                        })
            if hasattr(redundant_controller.optimizer, 'rollout_trajectories'):
                rollout_trajectories_tuple += (redundant_controller.optimizer.rollout_trajectories,)

        rollout_trajectories = np.concatenate(rollout_trajectories_tuple, axis=0)

        # TODO: pass optimal trajectory
        self.Render.update(
            lidar_points=self.lidar_points,
            rollout_trajectory=rollout_trajectories,
            next_waypoints=self.waypoint_utils.next_waypoint_positions,
            car_state=s
        )