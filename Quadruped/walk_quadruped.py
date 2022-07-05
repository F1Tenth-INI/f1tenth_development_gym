import os
import time

# working_dir = '/home/marcin/mpac_a1'
# os.chdir(working_dir)

from mpac_cmd import walk_idqp, lie

walk_idqp(vx=0.5)
time.sleep(10)
walk_idqp(vx=-0.5)
time.sleep(10)
lie()
time.sleep(10)

