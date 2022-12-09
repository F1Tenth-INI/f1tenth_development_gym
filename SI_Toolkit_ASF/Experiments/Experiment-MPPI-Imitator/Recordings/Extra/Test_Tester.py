import numpy as np

number_scans = 61

lower_bound = 500
upper_bound = 580

ranges = np.arange(lower_bound, upper_bound)

lower_bound = 500
upper_bound = 580
number_deleted = (upper_bound-lower_bound) - number_scans
bound_correction = number_deleted - (upper_bound - lower_bound) % number_deleted
#bound_correction = number_deleted - (upper_bound - lower_bound) % number_deleted                        #length of lidar options must fit the number of scans
upper_bound = upper_bound + round(bound_correction/2)
lower_bound = lower_bound - round(bound_correction/2) + (bound_correction % 2)
del_every_nth = int((upper_bound-lower_bound)/number_deleted)

rangesi = np.delete(ranges, np.arange(0, ranges.size, del_every_nth))

""" Option with  transformation to List

ranges = ranges.tolist()
del ranges[0::del_every_nth]
ranges = np.array(ranges)
"""

print((rangesi))
print(len(rangesi))
breakpoint()
''' Problem: number of deleted element is exact, number of remaining elements is wrong

import numpy as np

number_scans = 70

lower_bound = 500
upper_bound = 580
number_deleted = (upper_bound-lower_bound) - number_scans
bound_correction = number_deleted - (upper_bound - lower_bound) % number_deleted                        #length of lidar options must fit the number of scans
upper_bound = upper_bound + round(bound_correction/2)
lower_bound = lower_bound - round(bound_correction/2) + (bound_correction % 2)
slicer = int((upper_bound-lower_bound)/number_deleted)


elements_to_delete = np.arange(lower_bound,upper_bound, slicer)


elements_to_keep = [elements for elements in range(lower_bound, upper_bound+1) if elements not in elements_to_delete]

print(len(elements_to_keep))
print(elements_to_keep)
print(elements_to_delete)
print(len(elements_to_delete))
breakpoint()

'''