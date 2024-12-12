# car_state_listener.py

import os
import rospy
from rospy.msg import AnyMsg
import struct
import threading
import socket
import json

class CarStateListener:
    def __init__(self, host='localhost', port=5005):
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

        # Start the socket server in a separate thread
        self.server_thread = threading.Thread(target=self._start_socket_server, args=(host, port))
        self.server_thread.daemon = True
        self.server_thread.start()

    def _ros_spin(self):
        """Run rospy.spin() in a separate thread."""
        try:
            rospy.spin()
        except rospy.ROSInterruptException:
            pass

    def _start_socket_server(self, host, port):
        """Starts a TCP socket server to serve car state data."""
        server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server_socket.bind((host, port))
        server_socket.listen(5)  # Allow up to 5 pending connections
        rospy.loginfo(f"CarStateListener socket server started on {host}:{port}")

        while not rospy.is_shutdown():
            try:
                client_socket, addr = server_socket.accept()
                rospy.loginfo(f"Accepted connection from {addr}")
                client_handler = threading.Thread(
                    target=self._handle_client,
                    args=(client_socket,)
                )
                client_handler.daemon = True
                client_handler.start()
            except socket.error as e:
                rospy.logerr(f"Socket error: {e}")
                break

        server_socket.close()

    def _handle_client(self, client_socket):
        """Handles client requests."""
        with client_socket:
            while not rospy.is_shutdown():
                try:
                    data = client_socket.recv(1024).decode('utf-8').strip()
                    if not data:
                        break  # No more data from client
                    rospy.loginfo(f"Received request: {data}")
                    if data.upper() == "GET_CAR_STATE":
                        with self.lock:
                            car_state = {
                                'car_x': self.car_x,
                                'car_y': self.car_y,
                                'car_v': self.car_v,
                                'idx_global': self.idx_global
                            }
                        response = json.dumps(car_state)
                        client_socket.sendall(response.encode('utf-8'))
                        rospy.loginfo("Sent car state data to client.")
                    else:
                        rospy.logwarn(f"Unknown request: {data}")
                        client_socket.sendall(b"UNKNOWN_REQUEST")
                except socket.error as e:
                    rospy.logerr(f"Socket communication error: {e}")
                    break
                except Exception as e:
                    rospy.logerr(f"Unexpected error: {e}")
                    break

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
            waypoint_size = struct.calcsize('ii10f')  # 2 int32 + 10 float32 fields

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
            return {
                'car_x': self.car_x,
                'car_y': self.car_y,
                'car_v': self.car_v,
                'idx_global': self.idx_global
            }

    def shutdown(self):
        """Shutdown the ROS node gracefully."""
        rospy.signal_shutdown('Shutting down CarStateListener')
        self.spin_thread.join()
        rospy.loginfo("CarStateListener has been shut down.")

if __name__ == "__main__":
    listener = CarStateListener()
    try:
        while not rospy.is_shutdown():
            rospy.sleep(1)
    except KeyboardInterrupt:
        rospy.loginfo("KeyboardInterrupt received. Shutting down CarStateListener.")
    finally:
        listener.shutdown()
