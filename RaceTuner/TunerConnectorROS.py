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
                'idx_global': self.get_car_state().get('idx_global', 0),
                'time': self.get_car_state().get('time', 0),  # Adding time in seconds as float32
            }

            # Update the car state in your TunerConnector base class
            self.update_car_state(car_state)

            rospy.loginfo(f"TunerConnectorROS updated car state: {car_state}")

        except struct.error as e:
            rospy.logerr(f"TunerConnectorROS error decoding /car_state/state message: {e}")
        except Exception as e:
            rospy.logerr(f"TunerConnectorROS unexpected error in car_state_callback: {e}")

    def local_waypoints_callback(self, msg):
        try:
            buffer = msg._buff
            offset = 0

            # Parse Header
            seq, stamp_sec, stamp_nsec = struct.unpack_from('III', buffer, offset)
            offset += 12

            # Parse frame_id string
            frame_id_length = struct.unpack_from('I', buffer, offset)[0]
            offset += 4
            frame_id = struct.unpack_from(f'{frame_id_length}s', buffer, offset)[0].decode('utf-8')
            offset += frame_id_length

            # Parse wpnts_length
            wpnts_length = struct.unpack_from('I', buffer, offset)[0]
            offset += 4

            # Each waypoint is 88 bytes (2 int32 + 10 float64)
            waypoint_format = 'ii10d'
            waypoint_size = struct.calcsize(waypoint_format)

            # Compute the timestamp from the header
            message_time = stamp_sec + stamp_nsec * 1e-9  # Time in seconds as float
            # Convert to float32 using struct
            message_time_float32 = struct.unpack('f', struct.pack('f', message_time))[0]

            for _ in range(wpnts_length):
                if offset + waypoint_size > len(buffer):
                    rospy.logerr("TunerConnectorROS: Buffer size smaller than expected for all waypoints.")
                    break

                unpacked_data = struct.unpack_from(waypoint_format, buffer, offset)
                offset += waypoint_size

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

                if waypoint['id'] == 0:
                    current_car_state = self.get_car_state()
                    car_state = {
                        'car_x': current_car_state.get('car_x', 0.0),
                        'car_y': current_car_state.get('car_y', 0.0),
                        'car_v': current_car_state.get('car_v', 0.0),
                        'idx_global': waypoint['idx_global'],
                        'time': message_time_float32  # Adding time in seconds as float32
                    }
                    self.update_car_state(car_state)
                    rospy.loginfo(f"TunerConnectorROS updated idx_global: {waypoint['idx_global']} at time {message_time_float32}")
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
