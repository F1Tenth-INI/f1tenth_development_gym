from SI_Toolkit.computation_library import ComputationLibrary

from utilities.Settings import Settings
from utilities.state_utilities import *

from Control_Toolkit_ASF.Cost_Functions.Car.f1t_cost_function import f1t_cost_function

from types import SimpleNamespace

class racing(f1t_cost_function):

    def __init__(self, variable_parameters: SimpleNamespace, ComputationLib: "type[ComputationLibrary]"):
        super().__init__(variable_parameters=variable_parameters, ComputationLib=ComputationLib)
        self.cost_components = SimpleNamespace()
        self.total_stage_cost = None
        self.cost_components.crash_cost = None
        self.cost_components.cc = None
        self.cost_components.ccrc = None
        self.cost_components.ccocrc = None
        self.cost_components.icdc = None
        self.cost_components.angular_velocity_cost = None
        self.cost_components.angle_difference_to_wp_cost = None
        self.cost_components.speed_control_difference_to_wp_cost = None
        self.cost_components.distance_to_wp_segments_cost = None
        self.cost_components.velocity_difference_to_wp_cost = None
        self.single_step = False

    def configure(
            self,
            batch_size: int,
            horizon: int,
    ):
        super().configure(batch_size=batch_size, horizon=horizon)
        cost_vector = self.lib.zeros((batch_size, horizon))
        self.total_stage_cost = self.lib.to_variable(cost_vector, dtype=self.lib.float32)
        self.cost_components.crash_cost = self.lib.to_variable(cost_vector, dtype=self.lib.float32)
        self.cost_components.cc = self.lib.to_variable(cost_vector, dtype=self.lib.float32)
        self.cost_components.ccrc = self.lib.to_variable(cost_vector, dtype=self.lib.float32)
        self.cost_components.ccocrc = self.lib.to_variable(cost_vector, dtype=self.lib.float32)
        self.cost_components.icdc = self.lib.to_variable(cost_vector, dtype=self.lib.float32)
        self.cost_components.angular_velocity_cost = self.lib.to_variable(cost_vector, dtype=self.lib.float32)
        self.cost_components.angle_difference_to_wp_cost = self.lib.to_variable(cost_vector, dtype=self.lib.float32)
        self.cost_components.speed_control_difference_to_wp_cost = self.lib.to_variable(cost_vector, dtype=self.lib.float32)
        self.cost_components.distance_to_wp_segments_cost = self.lib.to_variable(cost_vector, dtype=self.lib.float32)
        self.cost_components.velocity_difference_to_wp_cost = self.lib.to_variable(cost_vector, dtype=self.lib.float32)
        
        if horizon == 1:
            self.single_step = True

        self.set_logged_attributes({
            "total_stage_cost": lambda: float(self.total_stage_cost),
            "crash_cost": lambda: float(self.cost_components.crash_cost),
            "cc": lambda: float(self.cost_components.cc),
            "ccrc": lambda: float(self.cost_components.ccrc),
            "ccocrc": lambda: float(self.cost_components.ccocrc),
            "icdc": lambda: float(self.cost_components.icdc),
            "angular_velocity_cost": lambda: float(self.cost_components.angular_velocity_cost),
            "angle_difference_to_wp_cost": lambda: float(self.cost_components.angle_difference_to_wp_cost),
            "speed_control_difference_to_wp_cost": lambda: float(self.cost_components.speed_control_difference_to_wp_cost),
            "distance_to_wp_segments_cost": lambda: float(self.cost_components.distance_to_wp_segments_cost),
            "velocity_difference_to_wp_cost": lambda: float(self.cost_components.velocity_difference_to_wp_cost),
        })



    def get_terminal_cost(self, terminal_state):
        terminal_speed_cost = self.get_terminal_speed_cost(terminal_state)
        terminal_cost = terminal_speed_cost

        return terminal_cost

    def get_stage_cost(self, s, u, u_prev):

        # It is not used while writing...
        cc = self.get_actuation_cost(u)
        ccrc = self.get_control_change_rate_cost(u, u_prev)
        if not self.single_step:
            ccrh = self.get_control_change_rate_within_horizon_cost(u)
            ccocrc = self.get_control_change_of_change_rate_cost(u, u_prev)
            icdc = self.get_immediate_control_discontinuity_cost(u, u_prev)
        else:
            ccrh = self.lib.zeros_like(cc)
            ccocrc = self.lib.zeros_like(cc)
            icdc = self.lib.zeros_like(cc)

        ## Crash cost: comment out for faster calculation...
        car_positions = s[:, :, self.POSE_X_IDX:self.POSE_Y_IDX + 1]
        if hasattr(self.variable_parameters, 'lidar_points'):
            crash_cost = self.get_crash_cost(car_positions, self.variable_parameters.lidar_points)
        else:
            crash_cost = self.lib.zeros_like(cc)

        # Cost related to state
        angular_velocity_cost = self.get_angular_velocity_cost(s)
        slipping_cost = self.get_slipping_cost(s)

        # Costs related to waypoints 
        if hasattr (self.variable_parameters.next_waypoints, 'shape'):
            # TODO: calculate closest waypoints only once
            waypoints = self.variable_parameters.next_waypoints #np.array(self.controller.next_waypoints).astype(np.float32) #self.lib.constant(self.controller.next_waypoints, dtype=self.lib.float32)
            waypoint_positions = waypoints[:,1:3]

            # Reuse, dont calculate twice...
            nearest_waypoint_indices = self.get_nearest_waypoints_indices(car_positions, waypoint_positions[:-1])

            # distance_to_wp_segments_cost = self.get_distance_to_wp_cost(s, waypoints, nearest_waypoint_indices)
            distance_to_wp_segments_cost = self.get_distance_to_wp_segments_cost(s, waypoints, nearest_waypoint_indices)
            velocity_difference_to_wp_cost = self.get_velocity_difference_to_wp_cost(s, waypoints, nearest_waypoint_indices)
            speed_control_difference_to_wp_cost = self.get_speed_control_difference_to_wp_cost(u, s, waypoints, nearest_waypoint_indices)
            angle_difference_to_wp_cost = self.get_angle_difference_to_wp_cost(s, waypoints, nearest_waypoint_indices)

        else:
            pass
            # distance_to_wp_segments_cost = self.lib.zeros_like(acceleration_cost)
            # velocity_difference_to_wp_cost = self.lib.zeros_like(acceleration_cost)
            # speed_control_difference_to_wp_cost = self.lib.zeros_like(acceleration_cost)


        speed_control_difference_to_wp_cost = self.normed_discount(speed_control_difference_to_wp_cost, s[0, :, 0], 0.95)
        # distance_to_wp_segments_cost = self.normed_discount(distance_to_wp_segments_cost, s[0, :, 0], 1.5)
        # distance_final = self.lib.concat((self.lib.zeros_like(distance_to_wp_segments_cost)[:, :-1], distance_to_wp_segments_cost[:, -2:-1]), 1)
        # distance_to_wp_segments_cost = distance_final

        stage_cost = (
                distance_to_wp_segments_cost
                + velocity_difference_to_wp_cost
                + crash_cost
                + cc
                + ccrc
                + ccrh
                # + ccocrc
                # + icdc
                + angular_velocity_cost
                # + angle_difference_to_wp_cost
                # + speed_control_difference_to_wp_cost
                # + acceleration_cost
                # + slipping_cost
                # + cost_for_stopping
            )

        if (Settings.ANALYZE_COST or self.cost_function_for_state_metric) and Settings.ROS_BRIDGE is False:
            if cc.shape == self.cost_components.cc.shape:
                self.lib.assign(self.total_stage_cost, stage_cost)
                self.lib.assign(self.cost_components.crash_cost, crash_cost)
                self.lib.assign(self.cost_components.cc, cc)
                self.lib.assign(self.cost_components.ccrc, ccrc)
                self.lib.assign(self.cost_components.ccocrc, ccocrc)
                self.lib.assign(self.cost_components.icdc, icdc)
                self.lib.assign(self.cost_components.angular_velocity_cost, angular_velocity_cost)
                self.lib.assign(self.cost_components.angle_difference_to_wp_cost, angle_difference_to_wp_cost)
                self.lib.assign(self.cost_components.speed_control_difference_to_wp_cost, speed_control_difference_to_wp_cost)
                self.lib.assign(self.cost_components.distance_to_wp_segments_cost, distance_to_wp_segments_cost)
                self.lib.assign(self.cost_components.velocity_difference_to_wp_cost, velocity_difference_to_wp_cost)


        discount_vector = self.lib.ones_like(s[0, :, 0])*1.00 #nth wypt has wheight factor^n, if no wheighting required use factor=1.00
        discount_vector = self.lib.cumprod(discount_vector, 0)
        discount_vector = self.lib.cast(discount_vector, self.lib.float32)
        # Read out values for cost weight callibration: Uncomment for debugging

        # distance_to_waypoints_cost_numpy = distance_to_waypoints_cost.numpy()[:20]
        # acceleration_cost_numpy = acceleration_cost.numpy()[:20]
        # steering_cost_numpy= steering_cost.numpy()[:20]
        # crash_penelty_numpy= crash_penelty.numpy()[:20]
        # cc_numpy= cc.numpy()[:20]
        # ccrc_numpy= ccrc.numpy()[:20]
        # stage_cost_numpy= stage_cost.numpy()[:20]

        return stage_cost*discount_vector