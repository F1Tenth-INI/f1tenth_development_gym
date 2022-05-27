import pandas as pd
import matplotlib.pyplot as plt


# load csv file with experiment recording (e.g. for replay)
def load_csv_recording(file_path):
    if isinstance(file_path, list):
        file_path = file_path[0]

    # Get race recording
    print('Loading file {}'.format(file_path))
    try:
        data: pd.DataFrame = pd.read_csv(file_path, comment='#')  # skip comment lines starting with #
    except Exception as e:
        print('Cannot load: Caught {} trying to read CSV file {}'.format(e, file_path))
        return False

    # Change to float32 wherever numeric column
    cols = data.columns
    data[cols] = data[cols].apply(pd.to_numeric, errors='ignore', downcast='float')

    return data



data = load_csv_recording('ExperimentRecordings/F1TENTH_Blank-MPPI-0__2022-05-23_21-13-11.csv')

pose_x = data['pose_x'].to_numpy()
pose_y = data['pose_y'].to_numpy()
speed = data['speed']

plt.figure()
plt.scatter(pose_x, pose_y, c=speed)
plt.show()