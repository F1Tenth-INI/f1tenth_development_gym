import os
import rospy
from rospy.msg import AnyMsg
import struct
import threading


class CarStateListener:
    def __init__(self):
        # Set ROS environment variables
        os.environ['ROS_MASTER_URI'] = 'http://192.168.116.46:11311'  # Ubuntu machine with ROS master
        os.environ['ROS_IP'] = '192.168.194.233'  # MacOS IP where this script is running
        os.environ['ROS_LOG_DIR'] = '/tmp'  # Replace '/tmp' with your preferred log directory

        # Initialize ROS node
        rospy.init_node('car_listener', anonymous=True, log_level=rospy.INFO)

        # Initialize car state variables
        self.car_x = None
        self.car_y = None
        self.car_v = None
        self.idx_global = None  # For /local_waypoints.wpnts[:]{id==0}.idx_global

        # Initialize a lock for thread-safe operations
        self.lock = threading.Lock()

        # Subscribe to the car state topic
        rospy.Subscriber("/car_state/tum_state", AnyMsg, self.car_state_callback)
        rospy.loginfo("CarStateListener subscribed to /car_state/tum_state.")

        # Subscribe to /local_waypoints topic for idx_global
        rospy.Subscriber("/local_waypoints", AnyMsg, self.local_waypoints_callback)
        rospy.loginfo("CarStateListener subscribed to /local_waypoints.")

        # Start ROS spin in a separate thread to prevent blocking
        self.spin_thread = threading.Thread(target=self._ros_spin)
        self.spin_thread.daemon = True  # Daemonize thread to exit when main program exits
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
        Decodes the message based on the known structure.
        """
        try:
            # Unpack the raw buffer into float32 values
            # Adjust the format string based on actual message structure
            # Example assumes 7 float32 values
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

            with self.lock:
                self.car_x = decoded_message['s_x']
                self.car_y = decoded_message['s_y']
                self.car_v = decoded_message['velocity']

            rospy.loginfo(f"Updated car state: x={self.car_x}, y={self.car_y}, v={self.car_v}")

        except struct.error as e:
            rospy.logerr(f"Error decoding message: {e}")

    def local_waypoints_callback(self, msg):
        """
        Callback function for /local_waypoints topic.
        Extracts idx_global where id == 0.
        """
        try:
            # Debug: Print buffer size
            rospy.loginfo(f"Received buffer length: {len(msg._buff)}")

            # Define header size and waypoint size
            header_size = 12  # Example header size (adjust as needed)
            waypoint_size = struct.calcsize('ii8f')  # 2 int32 + 8 float32 fields

            # Check if buffer size is sufficient
            if len(msg._buff) < header_size:
                rospy.logerr("Buffer size is smaller than expected header size.")
                return

            # Start reading after the header
            current_offset = header_size
            while current_offset + waypoint_size <= len(msg._buff):
                unpacked_data = struct.unpack_from('ii10f', msg._buff, current_offset)
                current_offset += waypoint_size

                # Debug: Print unpacked waypoint data
                rospy.loginfo(f"Waypoint unpacked: {unpacked_data}")

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
                    with self.lock:
                        self.idx_global = waypoint['idx_global']
                    rospy.loginfo(f"Updated idx_global: {self.idx_global}")
                    break

        except struct.error as e:
            rospy.logerr(f"Error unpacking /local_waypoints message: {e}")
        except Exception as e:
            rospy.logerr(f"Unexpected error in /local_waypoints callback: {e}")

    def get_car_state(self):
        with self.lock:
            return self.car_x, self.car_y, self.car_v, self.idx_global

    def shutdown(self):
        """Shutdown the ROS node gracefully."""
        rospy.signal_shutdown('Shutting down CarStateListener')
        self.spin_thread.join()
