# TunerConnectorROS2.py

import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
import threading
import numpy as np

from RaceTuner.TunerConnector import TunerConnector

# Import the CarState message from f110_msgs
# Make sure f110_msgs is built and sourced in your ROS2 workspace
from f110_msgs.msg import CarState


class TunerConnectorROS2(TunerConnector):
    """
    TunerConnector subclass that subscribes to ROS2 topics and updates car state.
    
    This connector subscribes to /car_state/state (f110_msgs/CarState) and
    optionally computes the nearest waypoint index (idx_global) if waypoints are provided.
    """
    
    def __init__(self, host='localhost', port=5005, waypoints_x=None, waypoints_y=None):
        """
        Initialize the ROS2 TunerConnector.
        
        Args:
            host: Socket server host
            port: Socket server port
            waypoints_x: Optional numpy array of waypoint x coordinates for idx_global computation
            waypoints_y: Optional numpy array of waypoint y coordinates for idx_global computation
        """
        super().__init__(host, port)
        
        # Store waypoints for local idx_global computation
        self.waypoints_x = waypoints_x
        self.waypoints_y = waypoints_y
        self.last_idx_global = 0  # For optimized search
        
        self.time_start = None
        
        # Initialize ROS2
        self.initialize_ros2()
    
    def initialize_ros2(self):
        """Initialize ROS2 node and subscriptions."""
        # Guard against double initialization
        if not rclpy.ok():
            rclpy.init()
        self.node = TunerNode(self)
        
        # Create executor for spinning in background thread
        self.executor = MultiThreadedExecutor()
        self.executor.add_node(self.node)
        
        # Spin in a separate thread
        self.spin_thread = threading.Thread(target=self._ros2_spin, daemon=True)
        self.spin_thread.start()
        
        self.logger.info("TunerConnectorROS2 initialized and subscribed to /car_state/state")
    
    def _ros2_spin(self):
        """Run executor.spin() in a separate thread."""
        try:
            self.executor.spin()
        except Exception as e:
            self.logger.error(f"ROS2 spin error: {e}")
    
    def set_waypoints(self, waypoints_x, waypoints_y):
        """
        Set or update waypoints for idx_global computation.
        
        Args:
            waypoints_x: numpy array of waypoint x coordinates
            waypoints_y: numpy array of waypoint y coordinates
        """
        self.waypoints_x = np.asarray(waypoints_x)
        self.waypoints_y = np.asarray(waypoints_y)
        self.logger.info(f"Waypoints set: {len(self.waypoints_x)} points")
    
    def compute_nearest_waypoint(self, car_x, car_y):
        """
        Compute the nearest waypoint index given car position.
        
        Uses an optimized search starting from the last known index,
        which is efficient for sequential waypoint following.
        
        Args:
            car_x: Car x position
            car_y: Car y position
            
        Returns:
            Nearest waypoint index, or None if waypoints not set
        """
        if self.waypoints_x is None or self.waypoints_y is None:
            return None
        
        n_waypoints = len(self.waypoints_x)
        
        # Optimized search: check nearby waypoints first (within ±50 of last index)
        search_radius = 50
        start_idx = max(0, self.last_idx_global - search_radius)
        end_idx = min(n_waypoints, self.last_idx_global + search_radius)
        
        # Search in the local window first
        local_indices = np.arange(start_idx, end_idx)
        if len(local_indices) > 0:
            local_distances = (
                (self.waypoints_x[local_indices] - car_x) ** 2 +
                (self.waypoints_y[local_indices] - car_y) ** 2
            )
            local_min_idx = local_indices[np.argmin(local_distances)]
            local_min_dist = local_distances.min()
        else:
            local_min_dist = float('inf')
            local_min_idx = 0
        
        # If the car might have jumped (e.g., lap completion), do a full search
        # Threshold: if local min distance > 5m, search globally
        if local_min_dist > 25.0:  # 5m squared
            all_distances = (
                (self.waypoints_x - car_x) ** 2 +
                (self.waypoints_y - car_y) ** 2
            )
            nearest_idx = int(np.argmin(all_distances))
        else:
            nearest_idx = int(local_min_idx)
        
        self.last_idx_global = nearest_idx
        return nearest_idx
    
    def process_car_state(self, msg, timestamp_sec):
        """
        Process incoming CarState message and update internal state.
        
        Args:
            msg: CarState message
            timestamp_sec: Message timestamp in seconds
        """
        # Compute time since start
        if self.time_start is None:
            self.time_start = timestamp_sec
        relative_time = timestamp_sec - self.time_start
        
        # Compute idx_global from position if waypoints are available
        idx_global = self.compute_nearest_waypoint(msg.x, msg.y)
        
        # Build car state dict
        car_state = {
            'car_x': float(msg.x),
            'car_y': float(msg.y),
            'car_v': float(msg.v_x),  # Using v_x as main velocity
            'idx_global': idx_global,
            'time': relative_time,
        }
        
        # Update the car state in base class
        self.update_car_state(car_state)
    
    def shutdown(self):
        """Shutdown ROS2 and the connector gracefully."""
        self.logger.info("Shutting down TunerConnectorROS2...")
        
        # Shutdown ROS2
        try:
            self.executor.shutdown()
            self.node.destroy_node()
            rclpy.shutdown()
        except Exception as e:
            self.logger.error(f"Error during ROS2 shutdown: {e}")
        
        # Shutdown base class (socket server)
        super().shutdown()


class TunerNode(Node):
    """ROS2 Node for subscribing to car state topics."""
    
    def __init__(self, connector: TunerConnectorROS2):
        super().__init__('tuner_connector_ros2')
        self.connector = connector
        
        # Subscribe to car state
        self.car_state_sub = self.create_subscription(
            CarState,
            '/car_state/state',
            self.car_state_callback,
            10  # QoS depth
        )
        
        self.get_logger().info("TunerNode subscribed to /car_state/state")
    
    def car_state_callback(self, msg: CarState):
        """Callback for /car_state/state topic."""
        try:
            # Get current timestamp
            timestamp_sec = self.get_clock().now().nanoseconds / 1e9
            
            # Process the message
            self.connector.process_car_state(msg, timestamp_sec)
            
            # Log periodically (every ~100 messages to avoid spam)
            if not hasattr(self, '_msg_count'):
                self._msg_count = 0
            self._msg_count += 1
            if self._msg_count % 100 == 1:
                self.get_logger().info(
                    f"Car state: x={msg.x:.2f}, y={msg.y:.2f}, v_x={msg.v_x:.2f}"
                )
            
        except Exception as e:
            self.get_logger().error(f"Error in car_state_callback: {e}")


def main():
    """Main entry point for running TunerConnectorROS2 standalone."""
    import time
    
    connector = TunerConnectorROS2(host='localhost', port=5005)
    
    print("TunerConnectorROS2 running. Press Ctrl+C to stop.")
    print("Socket server listening on localhost:5005")
    print("Subscribed to /car_state/state")
    
    try:
        # Keep the main thread alive while background thread handles ROS2
        while rclpy.ok():
            time.sleep(1.0)
    except KeyboardInterrupt:
        print("\nShutting down TunerConnectorROS2...")
    finally:
        connector.shutdown()


if __name__ == "__main__":
    main()


