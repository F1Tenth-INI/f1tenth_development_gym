import os 
import shutil
import zipfile
import glob
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

from joblib import dump


class TrainingHelper:
    def __init__(self, experiment_path: str, model_name: str, dataset_name: str):
        
        self.experiment_path = experiment_path
        self.model_dir = os.path.join(experiment_path,'models', model_name)
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
 
 
    def load_dataset(self):
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

            df = df[df['d_angular_vel_z'] <= 60]
            df = df[df['linear_vel_x'] <= 20]
            df = df[df['d_pose_x'] <= 20.]
            df = df[df['d_pose_y'] <= 20.]

            df['source'] = file
            
            # if i % 4 == 0:
            #     df_list.append(df)
            df_list.append(df)


        df = pd.concat(df_list, ignore_index=True)
        file_change_indices = df.index[df['source'].ne(df['source'].shift())].tolist()
        return df, file_change_indices
    
    
    def fit_trainsform_save_scalers(self, X, y):
        
        input_scaler = MinMaxScaler(feature_range=(-1, 1))
        output_scaler = MinMaxScaler(feature_range=(-1, 1))

        X = input_scaler.fit_transform(X)
        y = output_scaler.fit_transform(y)


        # save the scaler for denormalization
        dump(input_scaler, self.model_dir+'/input_scaler.joblib')
        dump(output_scaler, self.model_dir+'/output_scaler.joblib')
        
        return X, y
                

        