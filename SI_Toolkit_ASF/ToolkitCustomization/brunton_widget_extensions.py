"""
Adds application specific additions to brunton GUI.
get_feature_label allows replacing feature names with more descriptive labels.
convert_units_inplace converts units of the data to the desired units for display only.
All changes are done for GUI only, no changes to underlying dataset.
"""

import numpy as np


def get_feature_label(feature):

    if feature == 'angle':
        label = "Pole's Angle [deg]"
    elif feature == 'angleD':
        label = "Pole's Angular Velocity [deg/s]"
    elif feature == 'angle_cos':
        ...
    elif feature == 'mu':
        label = "Friction coefficient"
    else:
        label = feature

    return label


def convert_units_inplace(ground_truth, predictions_list):
    pass
