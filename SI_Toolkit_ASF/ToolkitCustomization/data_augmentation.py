"""
A class for data augmentation.
The class is used to modify the input features and output targets of the dataset
for neural network training.
"""

import tensorflow as tf


class DataAugmentation:
    def __init__(self, inputs, outputs, config_series_modification) -> None:

        self.inputs = inputs
        self.outputs = outputs

        self.noise_level_features = config_series_modification['NOISE_LEVEL']['FEATURES']

        self.columns_created = []


    def series_modification(self, features, targets):

        # Convert to tensors in case numpy arrays were passed
        features = tf.convert_to_tensor(features)
        targets = tf.convert_to_tensor(targets)

        # —————— Ensure we have a batch axis ——————
        # If features is [T, n_in] then insert a leading dim so B=1
        print("APPLYING DATA AUGMENTATION")
        _squeeze_batch = False
        if features.shape.ndims == 2:
            features = tf.expand_dims(features, axis=0)   # now [1, T, n_in]
            targets  = tf.expand_dims(targets,  axis=0)   # now [1, T, n_out]
            _squeeze_batch = True

        # Feature‑wise multiplicative noise
        noise = 1.0 + self.noise_level_features * tf.random.uniform(
            tf.shape(features), minval=-1.0, maxval=1.0, dtype=features.dtype
        )
        features = features * noise

        # —————— Remove artificial batch axis if we added one ——————
        if _squeeze_batch:
            features = tf.squeeze(features, axis=0)
            targets  = tf.squeeze(targets,  axis=0)

        return features, targets


