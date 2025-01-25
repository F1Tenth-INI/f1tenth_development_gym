# TunerConnectorROS.py

import rospy
from rospy.msg import AnyMsg
import struct
from TunerConnector import TunerConnector
import threading
import os


class TunerConnectorROS(TunerConnector):
    """
    TunerConnector subclass that subscribes to ROS topics and updates car state.
    """
    def __init__(self, host='localhost', port=5005):
        # Set ROS environment variables
        os.environ['ROS_MASTER_URI'] = 'http://ini-nuc.local:11311'
        os.environ['ROS_IP'] = '192.168.194.233'
        os.environ['ROS_LOG_DIR'] = '/tmp'

        super().__init__(host, port)
        self.initialize_ros()

    def initialize_ros(self):
        """Initialize ROS node and subscribe to necessary topics."""
        rospy.init_node('tuner_connector_ros', anonymous=True, log_level=rospy.INFO)

        # Specifying queue_size=1 ensures that we only keep the latest message,
        # preventing a backlog if your callbacks can't keep up with the incoming data rate.
        rospy.Subscriber("/car_state/state", AnyMsg, self.car_state_callback, queue_size=1)
        rospy.Subscriber("/local_waypoints", AnyMsg, self.local_waypoints_callback, queue_size=1)
        rospy.loginfo("TunerConnectorROS subscribed to ROS topics with queue_size=1.")

        # Spin in a separate thread to process callbacks asynchronously
        self.spin_thread = threading.Thread(target=self._ros_spin)
        self.spin_thread.daemon = True
        self.spin_thread.start()

    def _ros_spin(self):
        """Run rospy.spin() in a separate thread to handle callbacks."""
        try:
            rospy.spin()
        except rospy.ROSInterruptException:
            pass

    def car_state_callback(self, msg):
        """
        Callback function for /car_state/state topic.
        Decodes the message and updates the car state in the base class.
        """
        try:
            # 22 float32 fields, each 4 bytes, so 88 bytes total expected from the message buffer
            expected_size = 22 * 4
            if len(msg._buff) < expected_size:
                rospy.logerr(f"TunerConnectorROS: Buffer size ({len(msg._buff)}) "
                             f"smaller than expected for /car_state/state ({expected_size}).")
                return

            # Unpack raw buffer into 22 float values
            unpacked_data = struct.unpack('22f', msg._buff[:expected_size])

            # The first 11 floats are relevant measurements (x, y, v_x, etc.)
            # The next 11 floats are variance information
            decoded_message = {
                'x': unpacked_data[0],
                'y': unpacked_data[1],
                'v_x': unpacked_data[2],
                'v_y': unpacked_data[3],
                'steering_angle': unpacked_data[4],
                'yaw': unpacked_data[5],
                'yaw_rate': unpacked_data[6],
                'pitch': unpacked_data[7],
                'pitch_rate': unpacked_data[8],
                'roll': unpacked_data[9],
                'roll_rate': unpacked_data[10],
                'variance': unpacked_data[11:22]
            }

            # Construct a minimal dict representing the new car state
            car_state = {
                'car_x': decoded_message['x'],
                'car_y': decoded_message['y'],
                'car_v': decoded_message['v_x'],  # Using v_x as the main velocity
                # If idx_global not yet set, default to 0 (the .get(...) call is for safety)
                'idx_global': self.get_car_state().get('idx_global', 0)
            }

            # Update the car state in your TunerConnector base class
            self.update_car_state(car_state)

            rospy.loginfo(f"TunerConnectorROS updated car state: {car_state}")

        except struct.error as e:
            rospy.logerr(f"TunerConnectorROS error decoding /car_state/state message: {e}")
        except Exception as e:
            rospy.logerr(f"TunerConnectorROS unexpected error in car_state_callback: {e}")

    def local_waypoints_callback(self, msg):
        """
        Callback function for /local_waypoints topic.
        Extracts idx_global where id == 0 and updates the car state accordingly.
        """
        try:
            # Example header size for your custom message layout (adjust if needed)
            header_size = 12  # bytes to skip before reading waypoints

            # Each waypoint: 2 int32 + 10 float64, which is 2*4 + 10*8 = 88 bytes
            waypoint_format = 'ii10d'
            waypoint_size = struct.calcsize(waypoint_format)

            # Ensure buffer is large enough for the header
            if len(msg._buff) < header_size:
                rospy.logerr("TunerConnectorROS: Buffer size smaller than the expected header size for /local_waypoints.")
                return

            current_offset = 0
            # Read waypoints until we run out of buffer
            while current_offset + waypoint_size <= len(msg._buff):
                unpacked_data = struct.unpack_from(waypoint_format, msg._buff, current_offset)
                current_offset += waypoint_size

                waypoint = {
                    'id': unpacked_data[0],
                    'idx_global': unpacked_data[1],
                    's_m': unpacked_data[2],
                    'd_m': unpacked_data[3],
                    'x_m': unpacked_data[4],
                    'y_m': unpacked_data[5],
                    'd_right': unpacked_data[6],
                    'd_left': unpacked_data[7],
                    'psi_rad': unpacked_data[8],
                    'kappa_radpm': unpacked_data[9],
                    'vx_mps': unpacked_data[10],
                    'ax_mps2': unpacked_data[11],
                }

                # We only need to update idx_global if waypoint id is 0
                if waypoint['id'] == 0:
                    # print(waypoint)
                    current_car_state = self.get_car_state()

                    # Keep other car_state fields consistent with what's already stored
                    car_state = {
                        'car_x': current_car_state.get('car_x', 0.0),
                        'car_y': current_car_state.get('car_y', 0.0),
                        'car_v': current_car_state.get('car_v', 0.0),
                        'idx_global': waypoint['idx_global']
                    }
                    self.update_car_state(car_state)
                    rospy.loginfo(f"TunerConnectorROS updated idx_global: {waypoint['idx_global']}")
                    # Stop after the first waypoint with id==0
                    break

        except struct.error as e:
            rospy.logerr(f"TunerConnectorROS error unpacking /local_waypoints message: {e}")
        except Exception as e:
            rospy.logerr(f"TunerConnectorROS unexpected error in local_waypoints_callback: {e}")

def main():
    connector = TunerConnectorROS(host='localhost', port=5005)
    try:
        while not rospy.is_shutdown():
            rospy.sleep(1)
    except KeyboardInterrupt:
        rospy.loginfo("Shutting down TunerConnectorROS.")
    finally:
        connector.shutdown()

if __name__ == "__main__":
    main()
