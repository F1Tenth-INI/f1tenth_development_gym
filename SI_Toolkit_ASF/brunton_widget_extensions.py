import numpy as np

def get_feature_label(feature):

    if feature == 'angle':
        label = "Pole's Angle [deg]"
    elif feature == 'angleD':
        label = "Pole's Angular Velocity [deg/s]"
    elif feature == 'angle_cos':
        ...
    else:
        label = feature

    return label


def convert_units_inplace(ground_truth, predictions_list):

    pass
