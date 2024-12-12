# TunerConnector.py

import socket
import threading
import json
import rospy


class TunerConnector:
    def __init__(self, host='localhost', port=5005):
        self.host = host
        self.port = port
        self.car_state = {
            'car_x': None,
            'car_y': None,
            'car_v': None,
            'idx_global': None
        }
        self.clients = []
        self.lock = threading.Lock()
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_socket.bind((self.host, self.port))
        self.server_socket.listen(5)  # Allow up to 5 pending connections
        rospy.loginfo(f"TunerConnector server started on {self.host}:{self.port}")

        self.server_thread = threading.Thread(target=self.start_server)
        self.server_thread.daemon = True
        self.server_thread.start()

    def start_server(self):
        """Starts the socket server to listen for incoming connections."""
        while not rospy.is_shutdown():
            try:
                client_socket, addr = self.server_socket.accept()
                rospy.loginfo(f"Accepted connection from {addr}")
                client_thread = threading.Thread(target=self.handle_client, args=(client_socket, addr))
                client_thread.daemon = True
                client_thread.start()
            except socket.error as e:
                rospy.logerr(f"Socket error: {e}")
                break

    def handle_client(self, client_socket, addr):
        """Handles incoming client connections."""
        with client_socket:
            self.clients.append(client_socket)
            rospy.loginfo(f"Client {addr} added.")
            while not rospy.is_shutdown():
                try:
                    data = client_socket.recv(1024).decode('utf-8').strip()
                    if not data:
                        rospy.loginfo(f"Client {addr} disconnected.")
                        break

                    rospy.loginfo(f"Received data from {addr}: {data}")

                    if data.upper() == "GET_CAR_STATE":
                        with self.lock:
                            response = json.dumps(self.car_state)
                        client_socket.sendall(response.encode('utf-8'))
                        rospy.loginfo(f"Sent car state to {addr}: {response}")

                    else:
                        rospy.logwarn(f"Unknown command received from {addr}: {data}")
                        client_socket.sendall(b"UNKNOWN_REQUEST\n")

                except ConnectionResetError:
                    rospy.logwarn(f"Connection with {addr} was reset.")
                    break
                except Exception as e:
                    rospy.logerr(f"Error handling client {addr}: {e}")
                    break

            self.clients.remove(client_socket)
            rospy.loginfo(f"Client {addr} removed.")

    def update_car_state(self, new_state):
        """Updates the car state in a thread-safe manner."""
        with self.lock:
            self.car_state.update(new_state)
        rospy.loginfo(f"Updated car state: {self.car_state}")

    def get_car_state(self):
        """Retrieves the current car state in a thread-safe manner."""
        with self.lock:
            return self.car_state.copy()

    def shutdown(self):
        """Shutdown the server gracefully."""
        try:
            self.server_socket.close()
            rospy.loginfo("TunerConnector server shut down.")
        except Exception as e:
            rospy.logerr(f"Error shutting down server: {e}")
