import os 
import shutil
import zipfile
import glob
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, Dataset, TensorDataset

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
        # if os.path.exists(model_dir):
        #     shutil.rmtree(model_dir)

        if not os.path.exists(model_dir):
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
 
    def load_dataset(self, dataset_dir: str, reduce_size_by: int = 1):
        csv_files = glob.glob(dataset_dir + '/*.csv')
        print(dataset_dir)
        df_list = []
        for i, file in enumerate(csv_files):
            if i % reduce_size_by != 0: 
                continue

            df = pd.read_csv(file, comment='#')
            df['pose_theta_cos'] = np.cos(df['pose_theta'])
            df['pose_theta_sin'] = np.sin(df['pose_theta'])
            df['d_time'] = df['time'].diff()

            state_variables = ['angular_vel_z', 'linear_vel_x', 'pose_theta', 'pose_theta_cos', 'pose_theta_sin', 'pose_theta', 'pose_x', 'pose_y', 'slip_angle', 'steering_angle']
            # for var in state_variables:
            #     df['d_' + var] = df[var].diff() / df['d_time']
                
            control_variables = ['angular_control_calculated', 'translational_control_calculated']
            for var in control_variables:
                df['prev_' + var] = df[var].shift(1)
            

            # df = df[df['d_angular_vel_z'] <= 60]
            # df = df[df['linear_vel_x'] <= 20]
            # df = df[df['d_pose_x'] <= 20.]
            # df = df[df['d_pose_y'] <= 20.]
            
            # df = df[df['imu_a_x'] <= 20.]
            # df = df[df['imu_a_y'] <= 20.]
            # df = df[df['imu_av_z'] <= 20.]
            
            # df = df[df['imu_a_x'] >= -20.]
            # df = df[df['imu_a_y'] >= -20.]
            # df = df[df['imu_av_z'] >= -20.]

            df['source'] = file
            df_list.append(df)
            # df_list.append(df)


        df = pd.concat(df_list, ignore_index=True)
        df = df.dropna()
        file_change_indices = df.index[df['source'].ne(df['source'].shift())].tolist()
        return df, file_change_indices
    
    def create_histograms(self, df, input_cols, output_cols):
        print("Creating histograms, might take a while...")
        # Create figures folder
        os.makedirs(self.experiment_path + '/figures', exist_ok=True)

        # Process input columns
        for col in input_cols:
            if pd.api.types.is_numeric_dtype(df[col]):  # Check if the column is numeric
                plt.figure()
                df[col].hist(bins=100)  # Increase the number of bins to 100
                plt.title(col)
                plt.savefig(self.experiment_path + '/figures/' + col + '.png')
                time.sleep(0.15)  # Sleep for 150 milliseconds
            else:
                print(f"Skipping non-numeric column: {col}")

        # Process output columns
        for col in output_cols:
            if pd.api.types.is_numeric_dtype(df[col]):  # Check if the column is numeric
                plt.figure()
                df[col].hist(bins=100)  # Increase the number of bins to 100
                plt.title(col)
                plt.savefig(self.experiment_path + '/figures/' + col + '.png')
                time.sleep(0.15)  # Sleep for 150 milliseconds
            else:
                print(f"Skipping non-numeric column: {col}")

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
        
    def shuffle_dataset_by_chunks(self, X, y, file_change_indices, window_size, step_size):
        """
        Splits each file into smaller overlapping chunks, shuffles these chunks, 
        and reconstructs the dataset while updating file_change_indices.

        Parameters:
        - X: numpy array of input data (samples, features)
        - y: numpy array of target data (samples, outputs)
        - file_change_indices: indices where files change in the original dataset
        - window_size: size of each chunk
        - step_size: step size between chunks (controls overlap)

        Returns:
        - X_shuffled: shuffled input data (num_chunks, seq_length, features)
        - y_shuffled: shuffled target data (num_chunks, seq_length, outputs)
        - new_file_change_indices: updated indices reflecting chunk boundaries
        """

        # ✅ Use np.array_split to avoid errors with uneven splits
        X_splits = np.array_split(X, file_change_indices)
        y_splits = np.array_split(y, file_change_indices)

        chunks_X = []
        chunks_y = []

        for i in range(len(X_splits)):
            seq_X = X_splits[i]
            seq_y = y_splits[i]

            num_chunks = max(1, (len(seq_X) - window_size) // step_size + 1)
            for j in range(num_chunks):
                start_idx = j * step_size
                end_idx = start_idx + window_size
                if end_idx > len(seq_X):
                    break  # Stop if we exceed sequence length

                chunks_X.append(seq_X[start_idx:end_idx])
                chunks_y.append(seq_y[start_idx:end_idx])  # Ensure y has the same shape

        # Convert to numpy arrays
        if len(chunks_X) == 0 or len(chunks_y) == 0:
            raise ValueError("No valid chunks created! Check window_size and step_size.")

        chunks_X = np.array(chunks_X)  # Shape: (num_chunks, seq_length, features)
        chunks_y = np.array(chunks_y)  # Shape: (num_chunks, seq_length, outputs)

        # Shuffle the chunks
        shuffled_indices = np.random.permutation(len(chunks_X))
        X_shuffled = chunks_X[shuffled_indices]
        y_shuffled = chunks_y[shuffled_indices]

        # ✅ Ensure new file_change_indices are valid
        new_file_change_indices = np.cumsum([len(chunk) for chunk in X_shuffled[:-1]])
        new_file_change_indices = new_file_change_indices[new_file_change_indices < len(X_shuffled)]  # Remove out-of-range indices

        return X_shuffled, y_shuffled, new_file_change_indices

    
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





# Custom streamed windowing dataset
class SequenceWindowDataset(Dataset):
    def __init__(self, X, y, window_size, step_size=1):
        self.X = X
        self.y = y
        self.window_size = window_size
        self.step_size = step_size
        self.indices = list(range(0, len(X) - window_size + 1, step_size))

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        i = self.indices[idx]
        x_window = self.X[i:i + self.window_size]
        y_window = self.y[i:i + self.window_size]
        return torch.tensor(x_window, dtype=torch.float32), torch.tensor(y_window, dtype=torch.float32)