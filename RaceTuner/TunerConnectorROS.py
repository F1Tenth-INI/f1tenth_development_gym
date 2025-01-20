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
        rospy.Subscriber("/car_state/tum_state", AnyMsg, self.car_state_callback)
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
        Callback function for /car_state/tum_state topic.
        Decodes the message and updates the car state.
        """
        try:
            # Unpack the raw buffer into float32 values
            unpacked_data = struct.unpack('7f', msg._buff)

            # Map the unpacked values to their respective fields
            decoded_message = {
                's_x': unpacked_data[0],
                's_y': unpacked_data[1],
                'yaw': unpacked_data[2],
                'yaw_rate': unpacked_data[3],
                'velocity': unpacked_data[4],
                'steering_angle': unpacked_data[5],
                'slipping_angle': unpacked_data[6],
            }

            car_state = {
                'car_x': decoded_message['s_x'],
                'car_y': decoded_message['s_y'],
                'car_v': decoded_message['velocity'],
                'idx_global': self.get_car_state()['idx_global']
            }

            self.update_car_state(car_state)

            rospy.loginfo(f"TunerConnectorROS updated car state: {car_state}")

        except struct.error as e:
            rospy.logerr(f"TunerConnectorROS error decoding message: {e}")
        except Exception as e:
            rospy.logerr(f"TunerConnectorROS unexpected error: {e}")

    def local_waypoints_callback(self, msg):
        """
        Callback function for /local_waypoints topic.
        Extracts idx_global where id == 0 and updates the car state.
        """
        try:
            # Define header size and waypoint size
            header_size = 12  # Example header size (adjust as needed)
            waypoint_size = struct.calcsize('ii10f')  # 2 int32 + 10 float32 fields

            # Check if buffer size is sufficient
            if len(msg._buff) < header_size:
                rospy.logerr("TunerConnectorROS: Buffer size smaller than expected header size for /local_waypoints.")
                return

            # Start reading after the header
            current_offset = header_size
            while current_offset + waypoint_size <= len(msg._buff):
                unpacked_data = struct.unpack_from('ii10f', msg._buff, current_offset)
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
                    car_state = {
                        'car_x': self.get_car_state()['car_x'],
                        'car_y': self.get_car_state()['car_y'],
                        'car_v': self.get_car_state()['car_v'],
                        'idx_global': waypoint['idx_global']
                    }
                    self.update_car_state(car_state)
                    rospy.loginfo(f"TunerConnectorROS updated idx_global: {waypoint['idx_global']}")
                    break

        except struct.error as e:
            rospy.logerr(f"TunerConnectorROS error unpacking message: {e}")
        except Exception as e:
            rospy.logerr(f"TunerConnectorROS unexpected error: {e}")

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
