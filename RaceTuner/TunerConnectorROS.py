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
        # Initialize ROS node
        rospy.init_node('tuner_connector_ros', anonymous=True, log_level=rospy.INFO)

        # Subscribe to ROS topics
        rospy.Subscriber("/car_state/state", AnyMsg, self.car_state_callback)
        rospy.Subscriber("/local_waypoints", AnyMsg, self.local_waypoints_callback)
        rospy.loginfo("TunerConnectorROS subscribed to ROS topics.")

        # Start ROS spin in a separate thread
        self.spin_thread = threading.Thread(target=self._ros_spin)
        self.spin_thread.daemon = True
        self.spin_thread.start()

    def _ros_spin(self):
        """Run rospy.spin() in a separate thread."""
        try:
            rospy.spin()
        except rospy.ROSInterruptException:
            pass

    def car_state_callback(self, msg):
        """
        Callback function for /car_state/state topic.
        Decodes the message and updates the car state.
        """
        try:
            # Total fields: 11 float32 + 11 float32 (variance) = 22 float32
            expected_size = 22 * 4  # 22 float32 fields, 4 bytes each
            if len(msg._buff) < expected_size:
                rospy.logerr(f"TunerConnectorROS: Buffer size ({len(msg._buff)}) smaller than expected for /car_state/state ({expected_size}).")
                return

            # Unpack the raw buffer into 22 float32 values
            unpacked_data = struct.unpack('22f', msg._buff[:expected_size])

            # Map the unpacked values to their respective fields
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

            car_state = {
                'car_x': decoded_message['x'],
                'car_y': decoded_message['y'],
                'car_v': decoded_message['v_x'],  # Assuming v_x is the velocity of interest
                'idx_global': self.get_car_state().get('idx_global', 0)  # Default to 0 if not set
            }

            self.update_car_state(car_state)

            rospy.loginfo(f"TunerConnectorROS updated car state: {car_state}")

        except struct.error as e:
            rospy.logerr(f"TunerConnectorROS error decoding /car_state/state message: {e}")
        except Exception as e:
            rospy.logerr(f"TunerConnectorROS unexpected error in car_state_callback: {e}")

    def local_waypoints_callback(self, msg):
        """
        Callback function for /local_waypoints topic.
        Extracts idx_global where id == 0 and updates the car state.
        """
        try:
            # Define header size and waypoint size based on updated Wpnt.msg
            header_size = 12  # Example header size (adjust as needed)
            waypoint_format = 'ii10d'  # 2 int32 + 10 float64
            waypoint_size = struct.calcsize(waypoint_format)  # 2*4 + 10*8 = 88 bytes

            # Check if buffer size is sufficient
            if len(msg._buff) < header_size:
                rospy.logerr("TunerConnectorROS: Buffer size smaller than expected header size for /local_waypoints.")
                return

            # Start reading after the header
            current_offset = header_size
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

                # Check for id == 0
                if waypoint['id'] == 0:
                    current_car_state = self.get_car_state()
                    car_state = {
                        'car_x': current_car_state.get('car_x', 0.0),
                        'car_y': current_car_state.get('car_y', 0.0),
                        'car_v': current_car_state.get('car_v', 0.0),
                        'idx_global': waypoint['idx_global']
                    }
                    self.update_car_state(car_state)
                    rospy.loginfo(f"TunerConnectorROS updated idx_global: {waypoint['idx_global']}")
                    break  # Exit after finding the first waypoint with id == 0

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
