import numpy as np
from collections import defaultdict
from functools import partial


def create_calib_dict(calib_paths):
    input_calib = defaultdict(str)
    output_calib = defaultdict(partial(np.ndarray, 0, dtype=np.float32))

    for calib_path in calib_paths:
        with open(calib_path) as f:
            lines = f.readlines()

        for line in lines:
            input_calib[calib_path.split('/')[-1].replace('.txt', ''), (line.split(': ')[0])] = line.split(': ')[
                -1].replace('\n', '')

    output_calib['P0'] = np.array(input_calib[('calib_cam_to_cam', 'P_rect_00')].split(' '), dtype=np.float32)
    output_calib['P1'] = np.array(input_calib[('calib_cam_to_cam', 'P_rect_01')].split(' '), dtype=np.float32)
    output_calib['P2'] = np.array(input_calib[('calib_cam_to_cam', 'P_rect_02')].split(' '), dtype=np.float32)
    output_calib['P3'] = np.array(input_calib[('calib_cam_to_cam', 'P_rect_03')].split(' '), dtype=np.float32)
    output_calib['R0_rect'] = np.array(input_calib[('calib_cam_to_cam', 'R_rect_00')].split(' '), dtype=np.float32)

    output_calib['Tr_velo_to_cam'] = np.hstack((np.array(input_calib[('calib_velo_to_cam', 'R')].split(' '),
                                                         dtype=np.float32).reshape(3, 3),
                                                np.array(input_calib[('calib_velo_to_cam', 'T')].split(' '),
                                                         dtype=np.float32).reshape(3, 1))).flatten()
    output_calib['Tr_imu_to_velo'] = np.hstack((np.array(input_calib[('calib_imu_to_velo', 'R')].split(' '),
                                                         dtype=np.float32).reshape(3, 3),
                                                np.array(input_calib[('calib_velo_to_cam', 'T')].split(' '),
                                                         dtype=np.float32).reshape(3, 1))).flatten()

    return output_calib
