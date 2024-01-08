
from tensorflow.keras.models import load_model
from joblib import load
import numpy as np
import os
import math
import tensorflow as tf

experiment_path = os.path.join("TrainingLite","NN_lidar_odom")


# Load the model
model = load_model(experiment_path + '/models/my_model.h5')
input_scaler = load(experiment_path + '/models/input_scaler.joblib')
output_scaler = load(experiment_path + '/models/output_scaler.joblib')



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



# example_input = np.array([[1.972,1.821,1.723,1.685,1.63,1.627,1.676,1.65,1.731,1.709,1.825,2.114,4.011,4.075,4.027,3.975,4.051,4.144,4.311,4.534,4.826,5.289,3.938,2.827,2.191,1.875,1.624,1.434,1.258,1.223,1.135,1.094,1.032,0.98,1.01,1.02,1.031,1.043,1.074,1.12, 1.988,1.832,1.72,1.686,1.596,1.622,1.668,1.644,1.729,1.722,1.837,2.119,4.008,4.077,4.028,3.962,4.058,4.114,4.3,4.51,4.877,5.278,3.957,2.858,2.196,1.885,1.631,1.43,1.273,1.201,1.106,1.072,1.019,0.956,1.019,1.022,1.017,1.044,1.07,1.129]])  # replace with your actual input
# prediction = predict(example_input)
# print(prediction)
