from utilities.lidar_utils import LidarHelper
from tqdm import trange
from time import sleep

Lidar = LidarHelper()
CORRUPT_LIDAR_FOR_TRAINING = False


def augment_data(data, labels):

    if CORRUPT_LIDAR_FOR_TRAINING:
        print('Augmenting data...')
        sleep(0.002)
        for i in trange(len(data)):
            data[i] = Lidar.corrupt_datafile(data[i])

    return data, labels
