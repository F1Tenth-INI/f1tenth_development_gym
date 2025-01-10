import os 
import shutil
import zipfile
import glob
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import yaml
from typing import Optional
from joblib import load
from matplotlib import pyplot as plt
from joblib import dump
import time
import torch

class TrainingHelper:
    def __init__(self, experiment_path: str, model_name: str, dataset_name: Optional[str] = None):
        
        self.model_name = model_name
        self.experiment_path = experiment_path
        self.model_dir = os.path.join(experiment_path,'models', model_name)
        if dataset_name:
            self.dataset_dir = os.path.join(experiment_path, '..','Datasets', dataset_name)
  
        return
    
    def create_and_clear_model_folder(self, model_dir: str):
        if os.path.exists(model_dir):
            shutil.rmtree(model_dir)
        os.makedirs(model_dir, exist_ok=True)
        
    def save_training_scripts(self, training_file_path: str):
        # Zip the training script for reconstruction
        this_file_path = os.path.realpath(__file__)
        
        zip_file_path = os.path.join(self.model_dir, 'TrainingHelper.py.zip')
        with zipfile.ZipFile(zip_file_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            zipf.write(training_file_path, arcname=os.path.basename(this_file_path))

        zip_file_path = os.path.join(self.model_dir, 'train.py.zip')
        with zipfile.ZipFile(zip_file_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            zipf.write(this_file_path, arcname=os.path.basename(this_file_path))
            
        return 
 
    def load_dataset(self, reduce_size_by: int = 1):
        csv_files = glob.glob(self.dataset_dir + '/*.csv')
        df_list = []
        for i, file in enumerate(csv_files):
            df = pd.read_csv(file, comment='#')
            df['pose_theta_cos'] = np.cos(df['pose_theta'])
            df['pose_theta_sin'] = np.sin(df['pose_theta'])
            df['d_time'] = df['time'].diff()

            state_variables = ['angular_vel_z', 'linear_vel_x', 'pose_theta', 'pose_theta_cos', 'pose_theta_sin', 'pose_theta', 'pose_x', 'pose_y', 'slip_angle', 'steering_angle']
            for var in state_variables:
                df['d_' + var] = df[var].diff() / df['d_time']
                
            control_variables = ['angular_control_calculated', 'translational_control_calculated']
            for var in control_variables:
                df['prev_' + var] = df[var].shift(1)
            

            df = df[df['d_angular_vel_z'] <= 60]
            df = df[df['linear_vel_x'] <= 20]
            df = df[df['d_pose_x'] <= 20.]
            df = df[df['d_pose_y'] <= 20.]
            
            df = df[df['imu_a_x'] <= 20.]
            df = df[df['imu_a_y'] <= 20.]
            df = df[df['imu_av_z'] <= 20.]
            
            df = df[df['imu_a_x'] >= -20.]
            df = df[df['imu_a_y'] >= -20.]
            df = df[df['imu_av_z'] >= -20.]

            df['source'] = file
            
            if i % reduce_size_by == 0:
                df_list.append(df)
            # df_list.append(df)


        df = pd.concat(df_list, ignore_index=True)
        df = df.dropna()
        file_change_indices = df.index[df['source'].ne(df['source'].shift())].tolist()
        return df, file_change_indices
    
    def create_histograms(self, df, input_cols, output_cols):
        
        print("Creating histograms, might take a while...")
        for col in df[input_cols]:
            plt.figure()
            df[col].hist(bins=100)  # Increase the number of bins to 100
            plt.title(col)
            plt.savefig(self.experiment_path + '/figures/' + col + '.png')

            time.sleep(0.15)  # Sleep for 50 milliseconds
            
        for col in df[output_cols]:
            plt.figure()
            df[col].hist(bins=100)  # Increase the number of bins to 100
            plt.title(col)
            plt.savefig(self.experiment_path + '/figures/' + col + '.png')

            time.sleep(0.15)  # Sleep for 150 milliseconds
        print("Histograms created.")
    
    def shuffle_dataset_by_files(self, X, y, file_change_indices):
 
        # Split X_train and Y_train at file_change_indices
        X_splits = np.split(X, file_change_indices)
        y_splits = np.split(y, file_change_indices)

        # Shuffle the sequences
        shuffled_indices = np.random.permutation(len(X_splits))
        X_shuffled = [X_splits[i] for i in shuffled_indices]
        y_shuffled = [y_splits[i] for i in shuffled_indices]

        # Concatenate the shuffled sequences back together
        X = np.concatenate(X_shuffled)
        y = np.concatenate(y_shuffled)

        # Update file_change_indices
        file_change_indices = np.cumsum([len(seq) for seq in X_shuffled[:-1]])
        
        return X, y, file_change_indices
    
    
    def fit_trainsform_save_scalers(self, X, y):
        
        input_scaler = MinMaxScaler(feature_range=(-1, 1))
        output_scaler = MinMaxScaler(feature_range=(-1, 1))

        X = input_scaler.fit_transform(X)
        y = output_scaler.fit_transform(y)


        # save the scaler for denormalization
        dump(input_scaler, self.model_dir+'/input_scaler.joblib')
        dump(output_scaler, self.model_dir+'/output_scaler.joblib')
        
        return X, y
                
    def save_network_metadata(self, input_cols, output_cols, model):
            
        cols_dict = {"input_cols": input_cols, 
                     "output_cols": output_cols,
                     "input_size": model.input_size,
                     "output_size": model.output_size, 
                     "hidden_size": model.hidden_size, 
                     "num_layers": model.num_layers}
        with open(self.model_dir+'/network.yaml', 'w') as file:
            yaml.dump(cols_dict, file)
            


        return
    
    def load_network_meta_data_and_scalers(self):
        input_scaler = load(self.experiment_path + '/models/' + self.model_name + '/input_scaler.joblib')
        output_scaler = load(self.experiment_path + '/models/' + self.model_name + '/output_scaler.joblib')
        with open(self.experiment_path + '/models/' + self.model_name + '/network.yaml', 'r') as file:
            network_yaml = yaml.safe_load(file)
        return network_yaml, input_scaler, output_scaler
    
 
    def save_torch_model(self, model, train_losses, val_losses):
        torch.save(model.state_dict(), os.path.join(self.model_dir, "model.pth"))

        # Plot the loss values
        plt.figure(figsize=(10, 5))
        plt.plot(train_losses, label='Training Loss')
        plt.plot(val_losses, label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.title('Training and Validation Loss')
        plt.grid(True)
        plt.savefig(os.path.join(self.model_dir, 'loss_plot.png'))