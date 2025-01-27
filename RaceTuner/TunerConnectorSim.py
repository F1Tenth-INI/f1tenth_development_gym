# TunerConnectorSim.py

from RaceTuner.TunerConnector import TunerConnector


class TunerConnectorSim(TunerConnector):
    """
    TunerConnector subclass that provides an interface for the Simulator to send data.
    """
    def __init__(self, host='localhost', port=5005):
        super().__init__(host, port)

    def update_car_state(self, car_state):

        car_state_copy = car_state.copy()

        self.car_state = {
            'car_x': car_state_copy.get('car_x', None),
            'car_y': car_state_copy.get('car_y', None),
            'car_v': car_state_copy.get('car_v', None),
            'idx_global': car_state_copy.get('idx_global', None),
            'time': car_state_copy.get('time', None),
        }
