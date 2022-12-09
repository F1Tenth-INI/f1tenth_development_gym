#Program to delete the first 0.12 Second of recordings, so the RNN doesn't learn the Schupf behaviour

import os
import pandas as pd
from glob import glob


file_list = glob('/home/marcin/PycharmProjects/f1tenth_development_gym_Jago/ExperimentRecordings/F1TENTH*.csv')

for i in range(len(file_list)):
    file = file_list[i]

    df_head = pd.read_csv(file, nrows=7)
    df_content = pd.read_csv(file, sep=",", header=8, float_precision='round_trip')
    df_content.drop(df_content.head(4).index, inplace=True)
    #os.remove(file)
    df_head.to_csv(file, index=False, header=True)
    df_content.to_csv(file,mode="a", index=False, header=True)