import os
from typing import Optional

import numpy as np

from SI_Toolkit.computation_library import TensorType
from SI_Toolkit.load_and_normalize import load_yaml
from SI_Toolkit.General.variable_parameters import VariableParameters
from SI_Toolkit.computation_library import (ComputationLibrary, ComputationClasses,
                                            NumpyLibrary, PyTorchLibrary, TensorFlowLibrary,)

from Control_Toolkit.others.globals_and_utils import get_logger
from Control_Toolkit.Cost_Functions.cost_function_wrapper import CostFunctionWrapper


config_cost_function = load_yaml(os.path.join("Control_Toolkit_ASF", "config_cost_function.yml"))


logger = get_logger(__name__)


class StateMetricCalculator:
    def __init__(self,
                 environment_name: str,
                 initial_environment_attributes: "dict[str, TensorType]",
                 recorder_base_dict,
                 ):

        # Load controller config and select the entry for the current controller
        config_controllers = load_yaml(os.path.join("Control_Toolkit_ASF", "config_controllers.yml"))
        # self.controller_name is inferred from the class name, which is the class being instantiated
        # Example: If you create a controller_mpc, this controller_template.__init__ will be called
        # but the class name will be controller_mpc, not template_controller.
        self.config_controller = dict(config_controllers['mpc'])

        # Set computation library
        computation_library_name = str(self.config_controller.get("computation_library", ""))

        if computation_library_name:
            # Assign computation library from config
            logger.info(f"Found library {computation_library_name} for MPC controller.")
            if "tensorflow" in computation_library_name.lower():
                self._computation_library = TensorFlowLibrary()
            elif "pytorch" in computation_library_name.lower():
                self._computation_library = PyTorchLibrary()
            elif "numpy" in computation_library_name.lower():
                self._computation_library = NumpyLibrary()
            else:
                raise ValueError(f"Computation library {computation_library_name} could not be interpreted.")
        else:
            # Try using default computation library set as class attribute
            if not isinstance(self.computation_library, ComputationClasses):
                raise ValueError(
                    f"{self.__class__.__name__} does not have a default computation library set. You have to define one in this controller's config.")
            else:
                logger.info(
                    f"No computation library specified in controller config. Using default {self.computation_library} for class.")

        # Environment-related parameters
        self.environment_name = environment_name

        if "device" in self.config_controller:
            device = str(self.config_controller["device"])
        else:
            device = None

        self.initial_environment_attributes = {key: self.lib.to_variable(value, self.lib.float32) for key, value in
                                               initial_environment_attributes.items()}
        self.variable_parameters = VariableParameters(self.lib)
        self.variable_parameters.set_attributes(self.initial_environment_attributes, device=device)

        self.controller_data_for_csv = {}

        # Create cost function
        cost_function_specification = self.config_controller.get("cost_function_specification", None)
        self.cost_function = CostFunctionWrapper()
        self.cost_function.configure(
            batch_size=1,
            horizon=1,
            variable_parameters=self.variable_parameters,
            environment_name=self.environment_name,
            computation_library=self.computation_library,
            cost_function_specification=cost_function_specification
        )
        self.cost_function.cost_function.set_cost_function_for_state_metric()

        self.dict_for_csv = self.cost_function.cost_function.logged_attributes
        recorder_base_dict.update(self.dict_for_csv)

        self.past_control = None

    @property
    def controller_name(self):
        name = self.__class__.__name__
        if name != "template_controller":
            return name.replace("controller_", "").replace("_", "-").lower()
        else:
            raise AttributeError()

    @property
    def computation_library(self) -> "type[ComputationLibrary]":
        if self._computation_library is None:
            raise NotImplementedError("Controller class needs to specify its computation library")
        return self._computation_library

    @property
    def lib(self) -> "type[ComputationLibrary]":
        """Shortcut to make easy using functions from computation library, this is also used by CompileAdaptive to recognize library"""
        return self.computation_library

    def update_attributes(self, updated_attributes: "dict[str, TensorType]"):
        self.variable_parameters.update_attributes(updated_attributes)


    def calculate_metrics(self, current_state, current_control, updated_attributes):

        if self.past_control is None:
            self.past_control = current_control
            return None
        self.update_attributes(updated_attributes)
        current_state = current_state[np.newaxis, np.newaxis, :]
        current_control = current_control[np.newaxis, np.newaxis, :]
        stage_cost = self.cost_function.get_stage_cost(current_state, current_control, self.past_control)

        self.past_control = current_control

        return