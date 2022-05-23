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


def convert_units_inplace(ground_truth, predictions_list, features):

    # Convert ground truth
    for feature in features:
        feature_idx = features.index(feature)

        if feature == 'angle':
            ground_truth[:, feature_idx] *= 180.0 / np.pi
        elif feature == 'angleD':
            ...
        else:
            pass

    # Convert predictions
    for i in range(len(predictions_list)):
        for feature in features:
            feature_idx = features.index(feature)

            predictions_array = predictions_list[i]

            if feature == 'angle':
                predictions_array[:, :, feature_idx] *= 180.0/np.pi
            elif feature == 'angle_cos':
                pass
            elif feature == 'angle_sin':
                ...
            else:
                pass

            predictions_list[i] = predictions_array
