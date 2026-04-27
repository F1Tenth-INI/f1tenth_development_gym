import os
import shutil
import subprocess

from utilities.Settings import Settings
from utilities.RecordingToVideoConverter import RecordingToVideoConverter

def move_csv_to_crash_folder(csv_filepath, path_to_plots):
    import os
    import shutil

    path_to_experiment_recordings, _ = os.path.split(csv_filepath)

    # Check if the crash directory exists, if not create it
    dir_path = os.path.join(path_to_experiment_recordings, "crashes")
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    # Move the file to the crash directory
    shutil.move(csv_filepath, dir_path)

    # Move the plot directory to the crash directory
    if path_to_plots is not None and os.path.isdir(path_to_plots):
        shutil.move(path_to_plots, dir_path)


def save_experiment_data(csv_filepath):
    """
    Analyze a recording using the new ExperimentAnalyzer CLI.
    """
    path_to_experiment_recordings, experiment_filename = os.path.split(csv_filepath)
    experiment_name = experiment_filename[:-4]  # Remove .csv
    save_path = os.path.join(path_to_experiment_recordings, f"{experiment_name}_analysis")

    try:
        subprocess.run(
            [
                "python",
                "utilities/ExperimentAnalyzer.py",
                "--csv_path",
                csv_filepath,
                "--output_dir",
                save_path,
            ],
            check=True,
        )
    except Exception as e:
        print(f"Warning: CSV analysis did not work. Error: {e}")
        save_path = None
        
    if Settings.SAVE_VIDEOS:
        try:
            recording_to_video_converter = RecordingToVideoConverter(path_to_experiment_recordings, experiment_name, Settings.MAP_NAME)
            recording_to_video_converter.render_video()
        except Exception as e:
            print(f'Warning: video conversion did not work. Error: {e}')

    return save_path