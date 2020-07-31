import os
import numpy as np

original_root=''
whole_dataset_root='/data2/KITTI_RAW'

for root, dirs, files in os.walk(whole_dataset_root, topdown=False):
    for filename in files:
        if filename.endswith('.bin'):
            fullpath = os.path.join(root, filename)
            pc=np.fromfile(fullpath, dtype=np.float32) #.reshape(-1, 4)






