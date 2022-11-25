
import numpy as np


number_scans = 68

ranges = np.arange(0,1080)
lower_bound = 200
upper_bound = 880

number_deleted = 1080 - number_scans
bound_correction = number_deleted - (upper_bound - lower_bound) % number_deleted                        #length of lidar options must fit the number of scans
upper_bound = upper_bound + round(bound_correction/2)
lower_bound = lower_bound - round(bound_correction/2) + (bound_correction % 2)
slicer = int((upper_bound-lower_bound)/number_deleted)


print (slicer)

elements_to_delete = np.arange(lower_bound,upper_bound, slicer)

elements_to_keep = [elements for elements in range(lower_bound, upper_bound+1) if elements not in elements_to_delete]


#ranges_new = np.array(ranges_new)
print(elements_to_keep)




#print(ranges)


#print(np.arange(lower_bound,upper_bound, 3))


"""
if number_scans <= 540:
    bound_correction = number_scans - (upper_bound - lower_bound) % number_scans                        #length of lidar options must fit the number of scans
    upper_bound = upper_bound + round(bound_correction/2)
    lower_bound = lower_bound - round(bound_correction/2) + (bound_correction % 2)
    slicer = round((upper_bound-lower_bound)/number_scans)
    ranges = ranges[lower_bound:upper_bound:slicer]
else:
    ranges = ranges[lower_bound:upper_bound]
    np.delete(ranges, slice(None, None, slicer))


"""
#print(ranges_new)
#print(len(ranges_new))
#breakpoint()


