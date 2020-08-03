import os
from utils.create_position import Oxts_object


def write_txt_to_file(ouput_txt, output_dir, output_filename):
    with open(os.path.join(output_dir, output_filename), 'w') as file:
        file.write(ouput_txt)


def write_calib_file(calib_dict, output_dir, output_filename):
    P0 = " ".join(map(str, calib_dict['P0'].tolist()))
    P1 = " ".join(map(str, calib_dict['P1'].tolist()))
    P2 = " ".join(map(str, calib_dict['P2'].tolist()))  # " ".join(map(str, ECP_JSON['cam_info']['P']))
    P3 = " ".join(map(str, calib_dict['P3'].tolist()))
    R0 = " ".join(map(str, calib_dict['R0'].tolist()))  # " ".join(map(str, ECP_JSON['cam_info']['R']))
    tr_velo = " ".join(map(str, calib_dict['tr_velo'].tolist()))  # " ".join(map(str, kitti_tf))
    tr_imu = " ".join(map(str, calib_dict['tr_imu'].tolist()))  # ""
    kitti_format = "P0: %s\n" \
                   "P1: %s\n" \
                   "P2: %s\n" \
                   "P3: %s\n" \
                   "R0_rect: %s\n" \
                   "Tr_velo_to_cam: %s\n" \
                   "Tr_imu_to_velo: %s\n"

    output_calib = kitti_format % (P0, P1, P2, P3, R0, tr_velo, tr_imu)
    write_txt_to_file(ouput_txt=output_calib, output_dir=output_dir, output_filename=output_filename)


def write_pose_file(oxts: Oxts_object, output_dir: str, output_filename: str):
    Pose2Star = " ".join(map(str, oxts.pose_to_start.flatten().tolist()))  # transform back by .reshape(4,4)
    Pose2Prev = " ".join(map(str, oxts.pose_to_prev.flatten().tolist()))
    Curr_Pose = " ".join(map(str, oxts.current_pose.flatten().tolist()))

    kitti_format = "Pose2Star: %s\n" \
                   "Pose2Prev: %s\n" \
                   "Curr_Pose: %s\n"

    output_calib = kitti_format % (Pose2Star, Pose2Prev, Curr_Pose)
    write_txt_to_file(ouput_txt=output_calib, output_dir=output_dir, output_filename=output_filename)
