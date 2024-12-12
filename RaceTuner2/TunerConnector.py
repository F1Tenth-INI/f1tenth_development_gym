# TunerConnector.py

import socket
import threading
import json
import logging

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
        self.shutdown_event = threading.Event()

        # Configure logging
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.setLevel(logging.INFO)
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            '%(asctime)s [%(levelname)s] %(name)s: %(message)s'
        )
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)

        try:
            self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.server_socket.bind((self.host, self.port))
            self.server_socket.listen(5)  # Allow up to 5 pending connections
            self.logger.info(f"TunerConnector server started on {self.host}:{self.port}")
        except socket.error as e:
            self.logger.error(f"Failed to start server on {self.host}:{self.port}: {e}")
            raise

        self.server_thread = threading.Thread(target=self.start_server)
        self.server_thread.daemon = True
        self.server_thread.start()

    def start_server(self):
        """Starts the socket server to listen for incoming connections."""
        while not self.shutdown_event.is_set():
            try:
                self.server_socket.settimeout(1.0)  # Timeout to check shutdown_event periodically
                client_socket, addr = self.server_socket.accept()
                self.logger.info(f"Accepted connection from {addr}")
                client_thread = threading.Thread(target=self.handle_client, args=(client_socket, addr))
                client_thread.daemon = True
                client_thread.start()
            except socket.timeout:
                continue  # Continue the loop to check for shutdown_event
            except socket.error as e:
                if not self.shutdown_event.is_set():
                    self.logger.error(f"Socket error: {e}")
                break

    def handle_client(self, client_socket, addr):
        """Handles incoming client connections."""
        with client_socket:
            self.clients.append(client_socket)
            self.logger.info(f"Client {addr} added.")
            while not self.shutdown_event.is_set():
                try:
                    client_socket.settimeout(1.0)  # Timeout to check shutdown_event periodically
                    data = client_socket.recv(1024).decode('utf-8').strip()
                    if not data:
                        self.logger.info(f"Client {addr} disconnected.")
                        break

                    self.logger.info(f"Received data from {addr}: {data}")

                    if data.upper() == "GET_CAR_STATE":
                        with self.lock:
                            response = json.dumps(self.car_state)
                        client_socket.sendall(response.encode('utf-8'))
                        self.logger.info(f"Sent car state to {addr}: {response}")

                    else:
                        self.logger.warning(f"Unknown command received from {addr}: {data}")
                        client_socket.sendall(b"UNKNOWN_REQUEST\n")

                except socket.timeout:
                    continue  # Continue the loop to check for shutdown_event
                except ConnectionResetError:
                    self.logger.warning(f"Connection with {addr} was reset.")
                    break
                except Exception as e:
                    self.logger.error(f"Error handling client {addr}: {e}")
                    break

            self.clients.remove(client_socket)
            self.logger.info(f"Client {addr} removed.")

    def update_car_state(self, new_state):
        """Updates the car state in a thread-safe manner."""
        with self.lock:
            self.car_state.update(new_state)
        self.logger.info(f"Updated car state: {self.car_state}")

    def get_car_state(self):
        """Retrieves the current car state in a thread-safe manner."""
        with self.lock:
            return self.car_state.copy()

    def shutdown(self):
        """Shutdown the server gracefully."""
        self.logger.info("Shutting down TunerConnector server...")
        self.shutdown_event.set()

        # Close all client sockets
        for client in self.clients:
            try:
                client.shutdown(socket.SHUT_RDWR)
                client.close()
                self.logger.info("Closed client socket.")
            except Exception as e:
                self.logger.error(f"Error closing client socket: {e}")

        # Close the server socket
        try:
            self.server_socket.close()
            self.logger.info("TunerConnector server shut down.")
        except Exception as e:
            self.logger.error(f"Error shutting down server: {e}")

        # Wait for the server thread to finish
        self.server_thread.join(timeout=2)
        if self.server_thread.is_alive():
            self.logger.warning("Server thread did not terminate within the timeout.")
