import json
import socket
import threading


class SocketWatpointEditor:
    def __init__(self, host='localhost', port=5005):
        self.host = host
        self.port = port
        self.sock = None
        self.lock = threading.Lock()

    def connect(self):
        """Establish a connection to the socket server."""
        try:
            self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.sock.connect((self.host, self.port))
            print(f"Connected to CarStateListener at {self.host}:{self.port}")
        except socket.error as e:
            print(f"Failed to connect to CarStateListener: {e}")
            self.sock = None

    def get_car_state(self):
        """Request and receive the latest car state."""
        if self.sock is None:
            self.connect()
            if self.sock is None:
                return None  # Connection failed

        try:
            with self.lock:
                self.sock.sendall(b"GET_CAR_STATE\n")
                received = self.sock.recv(4096).decode('utf-8')
            if received:
                car_state = json.loads(received)
                return car_state
            else:
                print("No data received from CarStateListener.")
                return None
        except socket.error as e:
            print(f"Socket error during communication: {e}")
            self.sock.close()
            self.sock = None
            return None
        except json.JSONDecodeError as e:
            print(f"JSON decode error: {e}")
            return None

    def close(self):
        """Close the socket connection."""
        if self.sock:
            self.sock.close()
            self.sock = None
            print("Socket connection closed.")
