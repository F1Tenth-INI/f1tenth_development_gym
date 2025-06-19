import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np



import matplotlib.pyplot as plt

from joblib import load

from utilities.state_utilities import *
from utilities.waypoint_utils import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")

current_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(current_dir)
from TrainingHelper import TrainingHelper
from TorchNetworks import GRU as Network

class ControlPredictor:
    def __init__(self, sequence_length=100):
        self.model_name = "04_08_RCA1_noise"
        self.experiment_path = os.path.dirname(os.path.realpath(__file__))

        self.training_helper = TrainingHelper(self.experiment_path, self.model_name)
        self.network_yaml, self.input_scaler, self.output_scaler = self.training_helper.load_network_meta_data_and_scalers()

        input_size = len(self.network_yaml["input_cols"])
        output_size = len(self.network_yaml["output_cols"])
        hidden_size = self.network_yaml["hidden_size"]
        num_layers = self.network_yaml["num_layers"]

        self.model = Network(input_size, hidden_size, output_size, num_layers)
        self.model.load_state_dict(
            torch.load(
                os.path.join(self.training_helper.model_dir, "model.pth"),
                map_location=device
                )
            )
        self.model.to(device)  # Move the model to the correct device

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = self.model.to(self.device)
        self.model.eval()

        self.sequence_length = sequence_length
        self.history = []
        self.hidden = self.model.reset_hidden_state(batch_size=1)

    def predict_next_control(self, s, waypoints_relative, waypoints, ranges):
        state = [s[LINEAR_VEL_X_IDX]]
        state = [ s[ANGULAR_VEL_Z_IDX],s[LINEAR_VEL_X_IDX],s[LINEAR_VEL_Y_IDX], s[STEERING_ANGLE_IDX]]
        waypoints_x = waypoints_relative[:30, 0]
        waypoints_y = waypoints_relative[:30, 1]
        waypoints_vx = waypoints[:30, WP_VX_IDX]
        
        # plot wayoints
        # plt.clf()
        # plt.plot(-waypoints_y[:10], waypoints_x[:10])
        # plt.gca().set_aspect('equal', adjustable='box')
        # plt.xlim(-max(abs(waypoints_y[:10])), max(abs(waypoints_y[:10])))

        # plt.savefig('waypoints.png')
        
        # plt.clf()
        # plt.plot(waypoints_vx)
        # plt.savefig('waypoints_vx.png')
        # plt.clf()
        # time.sleep(0.1)    

        current_input = np.concatenate((state, waypoints_x, waypoints_y, waypoints_vx))

        self.history.append(current_input)
        if len(self.history) > self.sequence_length:
            self.history.pop(0)

        if len(self.history) < self.sequence_length:
            padding = [self.history[0]] * (self.sequence_length - len(self.history))
            input_sequence = padding + self.history
        else:
            input_sequence = self.history

        X_scaled = self.input_scaler.transform(input_sequence)
        X_tensor = torch.tensor(X_scaled, dtype=torch.float32).unsqueeze(0).to(self.device)

        with torch.no_grad():
            output, self.hidden = self.model(X_tensor, self.hidden)

        prediction = self.output_scaler.inverse_transform(output[:, -1, :].cpu().numpy())
        return prediction.squeeze()

if __name__ == "__main__":
    predictor = ControlPredictor(sequence_length=100)

    # Example input data placeholders (adjust these according to your actual data structure)
    s = [0, 0., 0., 0.3232363]
    waypoints_relative = np.zeros((30, 2))
    waypoints = np.zeros((30, WP_VX_IDX + 1))
    ranges = np.zeros(30)

    prediction = predictor.predict_next_control(s, waypoints_relative, waypoints, ranges)
    print("Prediction:", prediction)