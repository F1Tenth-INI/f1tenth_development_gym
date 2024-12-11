# car_state_listener.py
import os
import rospy
from rospy.msg import AnyMsg
import struct
import threading

class CarStateListener:
    def __init__(self):

        os.environ['ROS_MASTER_URI'] = 'http://192.168.116.46:11311'  # Ubuntu machine with ros master
        os.environ['ROS_IP'] = '192.168.194.233'  # MacOs IP which I am running this script on
        os.environ['ROS_LOG_DIR'] = '/tmp'  # Replace '/tmp' with your preferred log directory

        rospy.init_node('car_listener', anonymous=True, log_level=rospy.DEBUG)

        self.car_x = None
        self.car_y = None
        self.car_v = None

        rospy.Subscriber("/car_state/tum_state", AnyMsg, self.car_state_callback)
        rospy.loginfo("CarStateListener subscribed to /car_state/tum_state.")

        rospy.spin()

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

            self.car_x = decoded_message['s_x']
            self.car_y = decoded_message['s_y']
            self.car_v = decoded_message['velocity']

            rospy.loginfo(f"Updated car state: x={self.car_x}, y={self.car_y}, v={self.car_v}")

        except struct.error as e:
            rospy.logerr(f"Error decoding message: {e}")

    def get_car_state(self):
        with self.lock:
            return self.car_x, self.car_y, self.car_v
