import os
import time
import matplotlib.pyplot as plt
import numpy as np
from tqdm import trange

# working_dir = '/home/marcin/mpac_a1'
# os.chdir(working_dir)

from mpac_cmd import walk_idqp, lie, get_tlm_data

# tlm[5] - state
# state[0] =
# state[1] = x m
# state[2] = y m
# state[6] = theta rad


time.sleep(1)
walk_idqp(vx=0.2, vrz=np.pi/5)
tlm = get_tlm_data()
q_list = []
for i in trange(10000):
    time.sleep(0.001)
    q = get_tlm_data()[5]
    q_list.append(q)
walk_idqp(vx=0.2, vrz=-np.pi/5)
for i in trange(10000):
    time.sleep(0.001)
    q = get_tlm_data()[5]
    q_list.append(q)
lie()
q_list = np.array(q_list)
for i in range(q_list.shape[1]):

    plt.figure()
    plt.plot(q_list[:, i])
    plt.title(i)
    plt.show()


# # time.sleep(10)
# walk_idqp(vx=-0.5)
# # time.sleep(10)
# for i in range(10):
#     print(get_tlm_data())
#     time.sleep(1)
# lie()
# # time.sleep(10)
# for i in range(10):
#     print(get_tlm_data())
#     time.sleep(1)
# time.sleep(10)

