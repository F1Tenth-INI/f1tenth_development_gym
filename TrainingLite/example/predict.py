
from tensorflow.keras.models import load_model
from joblib import load
import numpy as np
import os
import math
import tensorflow as tf
import yaml

experiment_path =os.path.dirname(os.path.realpath(__file__))


# Load the model
model = load_model(experiment_path + '/models/my_model.h5')
input_scaler = load(experiment_path + '/models/input_scaler.joblib')
output_scaler = load(experiment_path + '/models/output_scaler.joblib')
with open(experiment_path + '/models/network.yaml', 'r') as file:
    network_yaml = yaml.safe_load(file)



def predict(input):
    
    example_input = [input]
    example_input = input_scaler.transform(example_input)
    example_input = tf.convert_to_tensor(example_input, dtype=tf.float32)
    prediction = _predict(example_input)
    prediction = prediction.numpy()
    prediction = output_scaler.inverse_transform(prediction)
    
    return prediction


@tf.function
def _predict(input):

    # Make a prediction
    prediction = model(input)
    return prediction




if __name__ == "__main__":
    
    print("inputs" , network_yaml["input_cols"])
    print("outputs", network_yaml["output_cols"])
    
    example_input = np.array([-0.154,3.525,-0.983,-0.184,0.0,0.0,0.015,3.844]) 
    # should return -0.070	-0.013
    
    prediction = predict(example_input)
    print(prediction)
