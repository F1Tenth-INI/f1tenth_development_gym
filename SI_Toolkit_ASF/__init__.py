from utilities.Settings import Settings


GLOBALLY_DISABLE_COMPILATION = Settings.GLOBALLY_DISABLE_COMPILATION   # Set to False to use tf.function
USE_JIT_COMPILATION = True  # XLA ignores random seeds. Set to False for reproducibility
