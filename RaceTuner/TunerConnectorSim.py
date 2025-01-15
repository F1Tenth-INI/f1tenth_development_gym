# TunerConnectorSim.py

from RaceTuner.TunerConnector import TunerConnector
import time


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
            'idx_global': car_state_copy.get('idx_global', None)
        }


def main():
    """
    Example usage of TunerConnectorSim.
    This should be integrated into your Simulator's main loop or relevant function.
    """
    connector = TunerConnectorSim(host='localhost', port=5005)
    try:
        for i in range(100):
            # Simulate car state and waypoints
            car_state = {
                'car_x': 10.0 + i * 0.1,
                'car_y': 5.0 + i * 0.05,
                'car_v': 15.0 + i * 0.2,
            }
            waypoints = {
                'idx_global': 42 + 2*i,
                'x_m': 10.0 + i * 0.1,
                'y_m': 5.0 + i * 0.05
                # Add more waypoint data as needed
            }

            # Send data to TunerConnector server
            connector.update_car_state(car_state, waypoints)

            # Wait before sending the next update
            time.sleep(1)  # Adjust the interval as needed
    except KeyboardInterrupt:
        print("Shutting down TunerConnectorSim.")
    finally:
        connector.shutdown()


if __name__ == "__main__":
    main()
