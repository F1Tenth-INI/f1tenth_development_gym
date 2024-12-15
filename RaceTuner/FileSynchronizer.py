# FileSynchronizer.py

import hashlib
import os
import threading
import time

import paramiko

from utilities.Settings import Settings

from TunerSettings import (
    USE_REMOTE_FILES,
    MAP_NAME,
    REMOTE_MAP_DIR,
    REMOTE_AT_LOCAL_DIR,
    REMOTE_CONFIG,
)


class FileSynchronizer(threading.Thread):
    """Handles initial and periodic synchronization of files from remote server."""

    def __init__(self, waypoint_manager, reload_event, interval=5):
        super().__init__(daemon=True)
        self.waypoint_manager = waypoint_manager
        self.reload_event = reload_event
        self.interval = interval  # in seconds
        self.running = True

        # Define download directory
        self.download_dir = REMOTE_AT_LOCAL_DIR
        os.makedirs(self.download_dir, exist_ok=True)

        # Initialize previous hashes
        self.previous_hashes = self._compute_waypoints_hashes()

    def _compute_file_hash(self, file_path):
        """Compute MD5 hash of the given file."""
        hash_md5 = hashlib.md5()
        try:
            with open(file_path, "rb") as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    hash_md5.update(chunk)
            return hash_md5.hexdigest()
        except FileNotFoundError:
            return None

    def _compute_waypoints_hashes(self):
        """Compute hashes for all relevant waypoint files."""
        wp_file = os.path.join(self.download_dir, MAP_NAME, f"{MAP_NAME}_wp.csv")
        wp_reverse_file = os.path.join(self.download_dir, MAP_NAME, f"{MAP_NAME}_wp_reverse.csv")
        speed_scaling_file = os.path.join(self.download_dir, MAP_NAME, f"{MAP_NAME}_speed_scaling.csv")
        wp_hash = self._compute_file_hash(wp_file)
        wp_rev_hash = self._compute_file_hash(wp_reverse_file)
        speed_hash = self._compute_file_hash(speed_scaling_file)
        return (wp_hash, wp_rev_hash, speed_hash)

    def run(self):

        while self.running:
            time.sleep(self.interval)

            if USE_REMOTE_FILES:
                download_map_files_via_sftp(MAP_NAME, REMOTE_MAP_DIR, REMOTE_AT_LOCAL_DIR, reverse_direction=Settings.REVERSE_DIRECTION)

                # Compute current hashes of downloaded files
                current_hashes = self._compute_waypoints_hashes()

                # Compare with previous hashes
                if current_hashes != self.previous_hashes:
                    print("Detected changes in remote waypoint files.")
                    # Update the waypoint manager with new waypoints
                    self.waypoint_manager.load_waypoints_from_file()

                    # Signal the UI to reload/redraw
                    self.reload_event.set()

                    # Update previous hashes
                    self.previous_hashes = current_hashes

    def stop(self):
        self.running = False


def upload_to_remote_via_sftp(local_path, remote_path):
    config = REMOTE_CONFIG
    try:
        transport = paramiko.Transport((config["host"], config["port"]))
        transport.connect(username=config["username"], password=config["password"])
        sftp = paramiko.SFTPClient.from_transport(transport)

        sftp.put(local_path, remote_path)
        print(f"File {local_path} synchronized to {remote_path} on remote server.")

        sftp.close()
        transport.close()
    except Exception as e:
        print(f"Failed to synchronize file: {str(e)}")


def download_map_files_via_sftp(map_name, remote_dir, local_dir, mode=None, reverse_direction=False):
    """
    Downloads the .png and .yaml map files from the remote server to the local directory.
    """
    try:
        transport = paramiko.Transport((REMOTE_CONFIG["host"], REMOTE_CONFIG["port"]))
        transport.connect(username=REMOTE_CONFIG["username"], password=REMOTE_CONFIG["password"])
        sftp = paramiko.SFTPClient.from_transport(transport)

        # Ensure the local directory exists
        os.makedirs(os.path.join(local_dir, map_name), exist_ok=True)

        # Define remote and local file paths
        file_names_initial_only = [
            f"{map_name}.png",
            f"{map_name}.yaml",
        ]

        file_names_forward = [
            f"{map_name}_wp.csv",
            f"{map_name}_speed_scaling.csv",
        ]

        file_names_reverse = [
            f"{map_name}_wp_reverse.csv",
            f"{map_name}_speed_scaling.csv",
        ]

        if mode == "initial":
            file_names = file_names_initial_only + file_names_forward + file_names_reverse
        elif reverse_direction is True:
            file_names = file_names_reverse
        else:
            file_names = file_names_forward

        for file in file_names:
            remote_file = os.path.join(remote_dir, map_name, file)
            local_file = os.path.join(local_dir, map_name, file)
            sftp.get(remote_file, local_file)
            print(f"Downloaded {remote_file} to {local_file}.")

        if mode == "inital":
            backup_files = [f"{map_name}_wp_backup.csv", f"{map_name}_wp_reverse_backup.csv"]

            for file in backup_files:
                remote_file = os.path.join(remote_dir, map_name, file)
                local_file = os.path.join(local_dir, map_name, file)
                try:
                    sftp.get(remote_file, local_file)
                    print(f"Downloaded {remote_file} to {local_file}.")
                except FileNotFoundError:
                    print(f"Optional file {remote_file} does not exist. Skipping.")

        sftp.close()
        transport.close()
    except Exception as e:
        print(f"Failed to download map files: {str(e)}")
