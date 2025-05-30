import sys
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from f110_sim.envs.dynamic_model_pacejka_jit import car_dynamics_pacejka_jit
from utilities.car_files.vehicle_parameters import VehicleParameters


""" This file loads hardware data (either manual or auto driven) and prepares a csv file ready to be fed
    into any train_NN files. Note that the train_NN files only load 8 states from the 10 measured
    
    The format of the outputted csv is as follows: 
    -10 states that were directly measured (including current position x,y)
    -2 control input that were applied
    -8 states of the error between real and simulating the states for 1 timestep

    angular_vel_z,linear_vel_x,linear_vel_y,pose_theta,pose_theta_cos,pose_theta_sin,pose_x,pose_y,slip_angle,steering_angle,
    manual_steering_angle,manual_acceleration,
    err_x,err_y,
    sim_angular_vel_z,sim_linear_vel_x,sim_linear_vel_y,sim_pose_theta,sim_pose_theta_cos,sim_pose_theta_sin,sim_slip_angle,sim_steering_angle
    
    If use_history is true (for SI_tool), then becomes 8-2-8 states
    
    If flie is meant to be used on personal NN_train, keep use_history False.
    If feeding csv for SI_toolkit, use_history to make csv with columns input_dim * history
"""
using_manual = False
save_csv=True
plotting= False
use_history= True

training= True
validate= False
test=False

#Make sure to change the input columns below!!
# input_folder = "F1tenth_data/hardware_data/sysid/manual_control"
input_folder = "F1tenth_data/hardware_data/sysid/auto_control"

"""File for loading experiment and sim csv files for analyzing the differenece if plotting"""
input_file = "F1tenth_data/hardware_data/validation_2.csv"

"""Simulate the ODE as in the simulation. Depending on number of steps of prediction"""
def simulate_ODE(state,control,predict):
    sim_states=[]
    tot_data=state.shape[0]
    loop_count= tot_data // predict
    if predict ==1:
        loop_count -=1
    
    vehicle_parm=VehicleParameters()
    car_params=vehicle_parm.to_np_array()

    for i in range(loop_count):
        
        index= i*predict
        #reading real data again
        s=state.iloc[index, :].to_numpy(dtype=np.float32)
        Q=control.iloc[index, :].to_numpy(dtype=np.float32)

        for j in range(predict):
            predicted_state=car_dynamics_pacejka_jit(s,Q,car_params,dt)
            s=predicted_state
            sim_states.append(s)
            Q=control.iloc[index+j+1, :].to_numpy(dtype=np.float32)
    
    sim_df = pd.DataFrame(sim_states)
    #name the columns appropriately
    sim_df.columns = ["angular_vel_z", "linear_vel_x", "linear_vel_y", "pose_theta", "pose_theta_cos", "pose_theta_sin", "pose_x", "pose_y", "slip_angle","steering_angle"] 

    return sim_df    


"""The real part of the code: First Load CSV files, process them and save the results"""
csv_files = [f for f in os.listdir(input_folder) if f.endswith('.csv')]

for csv_file in csv_files:
    input_path = os.path.join(input_folder, csv_file)
    print(f"Processing: {input_path}")

    real_data=pd.read_csv(input_path, comment = "#")

    #extract manual and autonomous control
    manual_control= real_data[["manual_steering_angle","manual_acceleration"]]
    auto_control = real_data[["angular_control","translational_control"]]

    real_state=real_data[["time","angular_vel_z","linear_vel_x","linear_vel_y","pose_theta",
                        "pose_theta_cos","pose_theta_sin","pose_x","pose_y","slip_angle","steering_angle"]]

    #Start analyzing data only when acceleration was applied in both cases
    if using_manual:
        input_column="manual_acceleration"
        first_nonzero= manual_control[manual_control[input_column] !=0].index.min()
        clean_control = manual_control.loc[first_nonzero:].reset_index(drop=True)
        clean_real_state = real_state.loc[first_nonzero:].reset_index(drop=True)

        input_steer=clean_control["manual_steering_angle"]
        input_acc=clean_control["manual_acceleration"]

    else:
        input_column="translational_control"
        first_nonzero= auto_control[auto_control[input_column] !=0].index.min()
        clean_control = auto_control.loc[first_nonzero:].reset_index(drop=True)
        clean_real_state = real_state.loc[first_nonzero:].reset_index(drop=True)
        input_steer=clean_control["angular_control"]
        input_acc=clean_control["translational_control"]


    """"Now calculate the change in state per every timestep"""
    real_state_change=clean_real_state.diff(1, axis=0)
    real_state_change = real_state_change.dropna(axis=0) #remove first row which has NaN


    """define variables that will be used to simulate in simulation for later comparison of error in delx
        maybe make different versions, one for computing 1 steps ahead, another for 10 (error builds up)"""

    dt= 0.02
    true_delx=real_state_change["pose_x"]
    true_dely=real_state_change["pose_y"]

    """Call the simulation with varying simulation timesteps"""
    notime_clean_real_state= clean_real_state.drop(["time"],axis=1)
    onestep_sim=simulate_ODE(notime_clean_real_state,clean_control,1)
    # tenstep_sim=simulate_ODE(notime_clean_real_state,clean_control,10)


    """Calculate the difference between real and sim x,y positions"""
    sim_state_change = onestep_sim.diff(1, axis=0).dropna(axis=0)
    sim_state_change = sim_state_change.reset_index(drop=True)
    sim_delx = sim_state_change["pose_x"]
    sim_dely = sim_state_change["pose_y"]

    #ensure same length of data with sim and real
    true_delx = true_delx.iloc[:len(sim_delx)].reset_index(drop=True)
    true_dely = true_dely.iloc[:len(sim_dely)].reset_index(drop=True)

    sim_error_x= true_delx - sim_delx
    sim_error_y= true_dely - sim_dely

    """calculate the rest of error in state between real and sim"""
    sim_error_state=notime_clean_real_state - onestep_sim
    sim_error_state= sim_error_state.drop(["pose_x", "pose_y"],axis=1)


    if plotting== True:
        """Plotting the results of sim and real data for sanity check"""
        time_pred1 = np.arange(0, len(onestep_sim) * dt, dt)
        time_pred10 = np.arange(0, len(tenstep_sim) * dt, dt)
        time_real = np.arange(0, len(clean_real_state) * dt, dt)
        time_real_ctrl = np.arange(0, len(clean_control) * dt, dt)


        fig, (ax1, ax2,ax3,ax4, ax5, ax6, ax7) = plt.subplots(7, 1, figsize=(14, 12))
        ax1.plot(time_pred1, onestep_sim["pose_y"], label='Predicted pose_y 1', linestyle='--')
        ax1.plot(time_pred10, tenstep_sim['pose_y'], label='Predicted pose_y 10', linestyle='-')
        ax1.plot(time_real, clean_real_state['pose_y'], label='Actual pose_y', linestyle='-')

        ax1.set_xlabel("Time (s)")
        ax1.set_ylabel("Y Position")
        ax1.set_title("Predicted vs Actual Y Position Over Time")
        ax1.legend()
        ax1.grid(True)

        ax2.plot(time_pred1, onestep_sim["pose_x"], label='Predicted pose_x', linestyle='--')
        ax2.plot(time_pred10, tenstep_sim["pose_x"], label='Predicted pose_x', linestyle='--')
        ax2.plot(time_real, clean_real_state["pose_x"], label='Actual pose_x', linestyle='-')
        ax2.set_xlabel("Time (s)")
        ax2.set_ylabel("X Position")
        ax2.set_title("Predicted vs Actual X Position Over Time")
        ax2.legend()
        ax2.grid(True)

        ax3.plot(time_pred1, onestep_sim["angular_vel_z"], label='Predicted angular_vel_z', linestyle='--')
        ax3.plot(time_pred10, tenstep_sim["angular_vel_z"], label='Predicted angular_vel_z', linestyle='--')
        ax3.plot(time_real, clean_real_state["angular_vel_z"], label='Actual angular_vel_z', linestyle='-')
        ax3.set_xlabel("Time (s)")
        ax3.set_ylabel("angular_vel_z")
        ax3.set_title("Predicted vs Actual angular_vel_z Over Time")
        ax3.legend()
        ax3.grid(True)
        ax3.set_ylim(-2,2)


        ax4.plot(time_real_ctrl, input_steer, label='Input steer', linestyle='-')
        ax4.plot(time_real_ctrl, input_acc, label='Input acc', linestyle='-')
        ax4.set_xlabel("Time (s)")
        ax4.set_ylabel("input")
        ax4.set_title("Control Input applied Over Time")
        ax4.legend()
        ax4.grid(True)

        ax5.plot(time_pred1, onestep_sim["pose_theta"], label='Predicted pose_theta', linestyle='--')
        ax5.plot(time_pred10, tenstep_sim["pose_theta"], label='Predicted pose_theta', linestyle='--')
        ax5.plot(time_real, clean_real_state["pose_theta"], label='Actual pose_theta', linestyle='-')
        ax5.set_xlabel("Time (s)")
        ax5.set_ylabel("pose_theta (rad)")
        ax5.set_title("Predicted vs Actual pose_theta Over Time")
        ax5.legend()
        ax5.grid(True)

        ax6.plot(time_pred1, onestep_sim["linear_vel_x"], label='Predicted vel x', linestyle='--')
        ax6.plot(time_pred10, tenstep_sim["linear_vel_x"], label='Predicted vel x', linestyle='--')
        ax6.plot(time_real, clean_real_state["linear_vel_x"], label='Actual vel x', linestyle='-')
        ax6.set_xlabel("Time (s)")
        ax6.set_ylabel("velocity (m/s)")
        ax6.set_title("Predicted vs Actual x velocity Over Time")
        ax6.legend()
        ax6.grid(True)

        ax7.plot(time_pred1, onestep_sim["linear_vel_y"], label='Predicted vel y', linestyle='--')
        ax7.plot(time_pred10, tenstep_sim["linear_vel_y"], label='Predicted vel y', linestyle='--')
        ax7.plot(time_real, clean_real_state["linear_vel_y"], label='Actual vel y', linestyle='-')
        ax7.set_xlabel("Time (s)")
        ax7.set_ylabel("velocity (m/s)")
        ax7.set_title("Predicted vs Actual y velocity Over Time")
        ax7.legend()
        ax7.grid(True)

        
        plt.show()


    """Now saving all the neccesary data as a single csv file"""
    if save_csv==True and use_history==False:
        #making sure that every data is matched to shortest length
        cutlength = min(
            len(notime_clean_real_state),
            len(clean_control),
            len(sim_error_x),
            len(sim_error_y),
            len(sim_error_state)
        )

        trimmed_real_state = notime_clean_real_state.iloc[:cutlength].reset_index(drop=True)
        trimmed_control     = clean_control.iloc[:cutlength].reset_index(drop=True)
        trimmed_error_x     = sim_error_x.iloc[:cutlength].reset_index(drop=True)
        trimmed_error_y     = sim_error_y.iloc[:cutlength].reset_index(drop=True)
        trimmed_error_state = sim_error_state.iloc[:cutlength].reset_index(drop=True)

        combined_df = pd.concat([
            trimmed_real_state,
            trimmed_control,
            trimmed_error_x,
            trimmed_error_y,
            trimmed_error_state
        ], axis=1)

        combined_df.columns = ["angular_vel_z", "linear_vel_x", "linear_vel_y", "pose_theta", "pose_theta_cos", "pose_theta_sin", "pose_x", "pose_y", "slip_angle","steering_angle",
                                "manual_steering_angle","manual_acceleration",
                                "err_x","err_y",
                                "sim_angular_vel_z", "sim_linear_vel_x", "sim_linear_vel_y", "sim_pose_theta", "sim_pose_theta_cos", "sim_pose_theta_sin", "sim_slip_angle","sim_steering_angle"
                                ]

        filename = os.path.basename(input_file)  
        
        output_filename = f"processed_{filename}" 

        if training==True:
            folder_path = 'training_data'
        elif validate==True:
            folder_path = "validation_data"
        elif test==True:
            folder_path = "test_data"
        os.makedirs(folder_path, exist_ok=True)

        save_path = os.path.join(folder_path, output_filename)
        combined_df.to_csv(save_path, index=False)

        print(f"CSV saved to: {save_path}")

    if save_csv==True and use_history==True:
        history= 8
        inputs = []
        targets = []

        cutlength = min(
        len(notime_clean_real_state),
        len(clean_control),
        len(sim_error_x),
        len(sim_error_y),
        len(sim_error_state)
        )

        trimmed_real_state = notime_clean_real_state.iloc[:cutlength].reset_index(drop=True)
        trimmed_control     = clean_control.iloc[:cutlength].reset_index(drop=True)
        trimmed_error_x     = sim_error_x.iloc[:cutlength].reset_index(drop=True)
        trimmed_error_y     = sim_error_y.iloc[:cutlength].reset_index(drop=True)
        trimmed_error_state = sim_error_state.iloc[:cutlength].reset_index(drop=True)
        
        #We don't need the poes x,y as input to NN so remove them
        # trimmed_real_state = trimmed_real_state.drop(["pose_x","pose_y"],axis=1)

        
        combined_df = pd.concat([
            trimmed_real_state,
            trimmed_control,
        ], axis=1)


        for i in range(cutlength):
            if i < history:
                continue 
            else:
                # Get the past `history` rows and flatten them into one row
                past_states = combined_df.iloc[i - history:i].values.flatten()

                #target is the change in position and state 
                target = [trimmed_error_x.iloc[i], trimmed_error_y.iloc[i], *trimmed_error_state.iloc[i].values]

                inputs.append(past_states)
                targets.append(target)

        # Convert to dataframe
        inputs_df = pd.DataFrame(inputs)
        targets_df = pd.DataFrame(targets)

        #renaming the columns iteratively
        base_features = [
            "angular_vel_z",
            "linear_vel_x",
            "linear_vel_y",
            "pose_theta",
            "pose_theta_cos",
            "pose_theta_sin",
            #Inserting pose here for trial
            "pose_x",
            "pose_y",
            #######
            "slip_angle",
            "steering_angle",
            "angular_control",
            "translational_control"
        ]


        # Generate column names
        stacked_columns = [f"{feat}_{t+1}" for t in range(history) for feat in base_features]

        # Assign to your DataFrame
        inputs_df.columns = stacked_columns
        targets_df.columns = ["err_x","err_y","err_angular_vel_z", "err_linear_vel_x", "err_linear_vel_y", "err_pose_theta", "err_pose_theta_cos", "err_pose_theta_sin","err_slip_angle","err_steering_angle"] 

        # Combine them
        final_df = pd.concat([inputs_df, targets_df], axis=1)

        #saving the csv
        filename = os.path.basename(input_path)  
        output_filename = f"history_{filename}" 
        if using_manual==True:
            folder_path = 'history_manual_data'
        else:
            folder_path = 'history_auto_data'
        os.makedirs(folder_path, exist_ok=True)

        save_path = os.path.join(folder_path, output_filename)
        final_df.to_csv(save_path, index=False)

        print(f"CSV saved to: {save_path}")