import tensorflow as tf

from SI_Toolkit.TF.TF_Functions.Compile import Compile
import numpy as np


STATE_INDICES = {} # This could be imported


class next_state_predictor_ODE_tf():

    def __init__(self, dt, intermediate_steps, disable_individual_compilation=False):
        self.s = None

        self.intermediate_steps = tf.convert_to_tensor(intermediate_steps, dtype=tf.int32)
        self.intermediate_steps_float = tf.convert_to_tensor(intermediate_steps, dtype=tf.float32)
        self.t_step = tf.convert_to_tensor(dt / float(self.intermediate_steps), dtype=tf.float32)

        if disable_individual_compilation:
            # self.step = self._step
            self.step = self._step_ks
        else:
            self.step = Compile(self._step)

    # @Compile
    def _step(self, s, Q, params):
        '''
        Parallaley executes steps frim initial state s[i] with control input Q[i] for every i
        @param s: (2000, 3) all initial states for every step 
        @param s: (2000, 2) all control inputs for every step
        @param params: TODO: Parameters of the car
        returns s_next: (2000, 3) all nexts states 
        '''

        pose_x = s[:, 0]
        pose_y = s[:, 1]
        pose_theta = s[:, 4]

        speed = Q[:, 0]
        steering = Q[:, 1]

        for _ in tf.range(self.intermediate_steps):
            pose_theta = pose_theta + 0.5*(steering/self.intermediate_steps_float)
            pose_x = pose_x + self.t_step * speed * tf.math.cos(pose_theta)
            pose_y = pose_y + self.t_step * speed * tf.math.sin(pose_theta)

        # s_next = tf.stack([pose_x, pose_y, pose_theta], axis=1)
        
        s_next = tf.stack([pose_x, pose_y, tf.zeros([2000]), tf.zeros([2000]), pose_theta, tf.zeros([2000]), tf.zeros([2000])], axis=1)
        

        return s_next
    
    def _step_st(self, s, Q, params):
        '''
        Parallaley executes steps frim initial state s[i] with control input Q[i] for every i
        @param s: (2000, 3) all initial states for every step 
        @param s: (2000, 2) all control inputs for every step
        @param params: TODO: Parameters of the car
        returns s_next: (2000, 3) all nexts states 
        '''
        
        # params
        mu = 1.0489
        C_Sf = 21.92/1.0489
        C_Sr = 21.92/1.0489
        lf = 0.3048*3.793293
        lr = 0.3048*4.667707
        h = 0.3048*2.01355
        m = 4.4482216152605/0.3048*74.91452
        I = 4.4482216152605*0.3048*1321.416
        g = 9.81

        # State
        s_x = s[:, 0]       # Pose X
        s_y = s[:, 1]       # Pose Y
        delta = s[:, 2]     # Fron Wheel steering angle
        theta = s[:, 3]     # Speed
        psi = s[:, 4]       # Yaw Angle
        psi_dot = s[:, 5]   # Yaw Rate
        beta = s[:, 6]      # Slipping Angle

        # Control Input
        delta_dot = Q[:, 1] # steering angle velocity of front wheels
        theta_dot = Q[:, 0] # longitudinal acceleration
        
        # Constaints
        theta_dot = self.accl_constraints(theta, theta_dot)
        delta_dot = self.steering_constraints(delta, delta_dot)

        # switch to kinematic model for small velocities
        min_speed_st = 0.1
        speed_too_low_for_st_indices = tf.math.less(theta, min_speed_st)
        speed_not_too_low_for_st_indices = tf.math.logical_not(speed_too_low_for_st_indices)
        
        speed_too_low_for_st_indices = tf.cast(speed_too_low_for_st_indices, tf.float32)
        speed_not_too_low_for_st_indices = tf.cast(speed_not_too_low_for_st_indices, tf.float32)
        
        

        # if abs(x[3]) < 0.1:
        # # wheelbase
        # lwb = lf + lr

        # # system dynamics
        # x_ks = x[0:5]
        # f_ks = vehicle_dynamics_ks(x_ks, u, mu, C_Sf, C_Sr, lf, lr, h, m, I, s_min, s_max, sv_min, sv_max, v_switch, a_max, v_min, v_max)
        # # f = np.hstack((f_ks, np.array([u[1]/lwb*np.tan(x[2])+x[3]/(lwb*np.cos(x[2])**2)*u[0],
        # # 0])))
        
        # f = np.hstack((f_ks, np.array([0,0])))
        
        for _ in tf.range(self.intermediate_steps):
            s_x_dot = tf.multiply(theta,tf.cos(tf.add(psi, beta)))
            s_y_dot = tf.multiply(theta,tf.sin(tf.add(psi, beta)))
            # delta_dot = delta_dot
            # theta_dot = theta_dot
            psi_dot = psi_dot
            psi_dot_dot = tf.zeros([2000])
            #* tf.multiply(delta, (C_Sf * (tf.fill([2000], g*lr) + theta_dot * h))) 
            
            # p = psi_dot.numpy()[:10]
            # print("p", p)
            #tf.zeros([2000]) #-mu*m/(theta*I*(lr+lf))*(lf**2*C_Sf*(g*lr-theta_dot*h))*theta #+ lr**2*C_Sr*(g*lf + theta_dot*h))*psi_dot+mu*m/(I*(lr+lf))*(lr*C_Sr*(g*lf + theta_dot*h) - lf*C_Sf*(g*lr - theta_dot*h))*beta+mu*m/(I*(lr+lf))*lf*C_Sf*(g*lr - theta_dot*h)*delta,
            
            # -mu*m/(x[3]*I*(lr+lf))*(lf**2*C_Sf*(g*lr-u[1]*h) + lr**2*C_Sr*(g*lf + u[1]*h))*x[5] \
            #     +mu*m/(I*(lr+lf))*(lr*C_Sr*(g*lf + u[1]*h) - lf*C_Sf*(g*lr - u[1]*h))*x[6] \
            #     +mu*m/(I*(lr+lf))*lf*C_Sf*(g*lr - u[1]*h)*x[2],
            beta_dot = tf.zeros([2000]) 
            #mu/(theta**2*(lr+lf))*(C_Sr*(g*lf + theta_dot*h)*lr)*theta #- C_Sf*(g*lr - theta_dot*h)*lf)-1)*psi_dot -mu/(theta*(lr+lf))*(C_Sr*(g*lf + theta_dot*h) + C_Sf*(g*lr-theta_dot*h))*beta +mu/(theta*(lr+lf))*(C_Sf*(g*lr-theta_dot*h))*delta
                
            #  (mu/(x[3]**2*(lr+lf))*(C_Sr*(g*lf + u[1]*h)*lr - C_Sf*(g*lr - u[1]*h)*lf)-1)*x[5] \
            #     -mu/(x[3]*(lr+lf))*(C_Sr*(g*lf + u[1]*h) + C_Sf*(g*lr-u[1]*h))*x[6] \
            #     +mu/(x[3]*(lr+lf))*(C_Sf*(g*lr-u[1]*h))*x[2]])
           
            s_x = tf.add(s_x, tf.multiply(self.t_step, s_x_dot))
            s_y = s_y + self.t_step * s_y_dot
            delta = delta + self.t_step * delta_dot
            theta = theta + self.t_step * theta_dot
            psi = psi + self.t_step * psi_dot
            psi_dot = psi_dot + self.t_step * psi_dot_dot 
            beta = beta + self.t_step * beta_dot
           

        s_next = tf.stack([s_x, s_y, delta,theta,psi,psi_dot,beta ], axis=1)

        return s_next

    # @Compile
    def steering_constraints (self, steering_angle, steering_velocity):
        s_min = tf.constant([-0.4189])
        s_max = tf.constant([0.4189])
        sv_min = tf.constant([-3.2]) 
        sv_max = tf.constant([3.2]) 
        
    
        # Steering angle constraings
        steering_angle_not_too_low_indices = tf.math.greater(steering_angle, s_min)
        steering_angle_not_too_low_indices = tf.cast(steering_angle_not_too_low_indices, tf.float32)
        
        steering_angle_not_too_high_indices = tf.math.less(steering_angle, s_max)
        steering_angle_not_too_high_indices = tf.cast(steering_angle_not_too_high_indices, tf.float32)
        
        steering_velocity = tf.multiply(steering_angle_not_too_low_indices, steering_velocity)
        steering_velocity = tf.multiply(steering_angle_not_too_high_indices, steering_velocity)
        
        # Steering velocity is constrainted
        steering_velocity = tf.clip_by_value(steering_velocity, clip_value_min=sv_min, clip_value_max=sv_max)
        
        
        return steering_velocity
    
    def accl_constraints (self, vel, accl):
        v_switch = tf.constant([7.319])
        a_max = tf.constant([9.51])
        v_min = tf.constant([-5.0]) 
        v_max = tf.constant([20.0]) 
        
        # positive accl limit
        velocity_too_high_indices = tf.math.greater(vel, v_switch)
        velocity_not_too_high_indices = tf.math.logical_not(velocity_too_high_indices)
        velocity_too_high_indices = tf.cast(velocity_too_high_indices, tf.float32)
        velocity_not_too_high_indices = tf.cast(velocity_not_too_high_indices, tf.float32)
        
        pos_limit_velocity_too_high = tf.math.divide(a_max * v_switch, vel)
        pos_limit_velocity_not_too_high = tf.tile(a_max, [2000])
        
        pos_limit = tf.multiply(velocity_too_high_indices, pos_limit_velocity_too_high) + tf.multiply(velocity_not_too_high_indices, pos_limit_velocity_not_too_high)
        
        
        # accl limit reached?
        velocity_not_over_max_indices = tf.math.less(vel, v_max)
        velocity_not_over_max_indices = tf.cast(velocity_not_over_max_indices, tf.float32)
        
        velocity_not_under_min_indices = tf.math.greater(vel, v_min)
        velocity_not_under_min_indices = tf.cast(velocity_not_under_min_indices, tf.float32)
        
        accl = tf.multiply(velocity_not_over_max_indices, accl)
        accl = tf.multiply(velocity_not_under_min_indices, accl)
        
        accl = tf.clip_by_value(accl, clip_value_min=-a_max, clip_value_max=10000)
        accl = tf.clip_by_value(accl, clip_value_min=-100000, clip_value_max=pos_limit)
        
        #Tested until here
        return accl
        
            
         
    def _step_ks(self, s, Q, params):
        '''
        Parallaley executes steps frim initial state s[i] with control input Q[i] for every i
        @param s: (2000, 3) all initial states for every step 
        @param s: (2000, 2) all control inputs for every step
        @param params: TODO: Parameters of the car
        returns s_next: (2000, 3) all nexts states 
        '''
        lf = 0.15875
        lr = 0.17145
        lwb = lf + lr
        
        
        s_x = s[:, 0]           # Pose X
        s_y = s[:, 1]           # Pose Y
        delta = s[:, 2]         # Fron Wheel steering angle
        theta = s[:, 3]         # Speed
        psi = s[:, 4]           # Yaw Angle

        delta_dot = Q[:, 1]     # steering angle velocity of front wheels
        theta_dot = Q[:, 0]     # longitudinal acceleration
        
        # Constaints
        theta_dot = self.accl_constraints(theta, theta_dot)
        delta_dot = self.steering_constraints(delta, delta_dot)
   
        # Euler stepping
        for _ in tf.range(self.intermediate_steps):
            s_x_dot = tf.multiply(theta,tf.cos(psi))
            s_y_dot = tf.multiply(theta,tf.sin(psi))
            # delta_dot = delta_dot
            # theta_dot = theta_dot
            psi_dot = tf.divide(theta, lwb)*tf.tan(delta)
            
            s_x = s_x + self.t_step * s_x_dot
            s_y = s_y + self.t_step * s_y_dot
            delta = delta + self.t_step * delta_dot
            theta = theta + self.t_step * theta_dot
            psi = psi + self.t_step * psi_dot

        s_next = tf.stack([s_x, s_y, delta,theta, psi,tf.zeros([2000]),tf.zeros([2000])], axis=1)

        return s_next
    

class predictor_output_augmentation_tf:
    def __init__(self, net_info):
        self.net_output_indices = {key: value for value, key in enumerate(net_info.outputs)}
        indices_augmentation = []
        features_augmentation = []
        # if 'sin(x)' not in net_info.outputs:
        #     indices_augmentation.append(STATE_INDICES['sin(x)'])
        #     features_augmentation.append('sin(x)')
        #
        self.indices_augmentation = indices_augmentation
        self.features_augmentation = features_augmentation
        self.augmentation_len = len(self.indices_augmentation)

        if 'x' in net_info.outputs:
            self.index_x = tf.convert_to_tensor(self.net_output_indices['x'])

    def get_indices_augmentation(self):
        return self.indices_augmentation

    def get_features_augmentation(self):
        return self.features_augmentation

    @Compile
    def augment(self, net_output):

        output = net_output
        # if 'sin(x)' in self.features_augmentation:
        #     sin_x = tf.math.sin(net_output[..., self.index_x])[:, :, tf.newaxis]
        #     output = tf.concat([output, sin_x], axis=-1)

        return output
