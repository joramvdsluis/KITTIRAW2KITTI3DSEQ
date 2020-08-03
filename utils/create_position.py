# TODO: get posisition in lidar frame, not imu frame! --> https://github.com/utiasSTARS/pykitti/blob/master/pykitti/utils.py
import numpy as np
import os
from typing import List
import matplotlib.pyplot as plt
import pptk


class Oxts_object:
    def __init__(self, file_path: str):
        file_content_str = read_text_file(file_path).replace('\n', '').split(' ')
        file_content = [float(content_str) for content_str in file_content_str]
        idx = int(file_path.split('/')[-1].split('.txt')[0])
        timestaps_str_list = read_text_file(file_path.split('/data/')[0] + '/timestamps.txt').split('\n')

        self.timestamp_start_sequence = timestaps_str_list[0]
        self.timestamp = timestaps_str_list[idx]
        self.path = file_path
        self.idx = idx

        self.lat = file_content[0]  # latitude of the oxts-unit (deg)
        self.lon = file_content[1]  # longitude of the oxts-unit (deg)
        self.alt = file_content[2]  # altitude of the oxts-unit (m)
        self.roll = file_content[3]  # roll angle (rad),  0 = level, positive = left side up (-pi..pi)
        self.pitch = file_content[4]  # pitch angle (rad), 0 = level, positive = front down (-pi/2..pi/2)
        self.yaw = file_content[5]  # heading (rad),     0 = east,  positive = counter clockwise (-pi..pi)
        self.vn = file_content[6]  # velocity towards north (m/s)
        self.ve = file_content[7]  # velocity towards east (m/s)
        self.vf = file_content[8]  # forward velocity, i.e. parallel to earth-surface (m/s)
        self.vl = file_content[9]  # leftward velocity, i.e. parallel to earth-surface (m/s)
        self.vu = file_content[10]  # upward velocity, i.e. perpendicular to earth-surface (m/s)
        self.ax = file_content[11]  # acceleration in x, i.e. in direction of vehicle front (m/s^2)
        self.ay = file_content[12]  # acceleration in y, i.e. in direction of vehicle left (m/s^2)
        self.az = file_content[13]  # acceleration in z, i.e. in direction of vehicle top (m/s^2)
        self.af = file_content[14]  # forward acceleration (m/s^2)
        self.al = file_content[15]  # leftward acceleration (m/s^2)
        self.au = file_content[16]  # upward acceleration (m/s^2)
        self.wx = file_content[17]  # angular rate around x (rad/s)
        self.wy = file_content[18]  # angular rate around y (rad/s)
        self.wz = file_content[19]  # angular rate around z (rad/s)
        self.wf = file_content[20]  # angular rate around forward axis (rad/s)
        self.wl = file_content[21]  # angular rate around leftward axis (rad/s)
        self.wu = file_content[22]  # angular rate around upward axis (rad/s)
        self.posacc = file_content[23]  # velocity accuracy (north/east in m)
        self.velacc = file_content[24]  # velocity accuracy (north/east in m/s)
        self.navstat = int(file_content[25])  # navigation status
        self.numsats = int(file_content[26])  # number of satellites tracked by primary GPS receiver
        self.posmode = int(file_content[27])  # position mode of primary GPS receiver
        self.velmode = int(file_content[28])  # velocity mode of primary GPS receiver
        self.orimode = int(file_content[29])  # orientation mode of primary GPS receiver

        self.pose_to_start = None
        self.pose_to_prev = None
        self.current_pose = None

    def add_poses(self, pose_to_start: np.array, pose_to_prev: np.array, current_pose: np.array):
        self.pose_to_start = pose_to_start
        self.pose_to_prev = pose_to_prev
        self.current_pose = current_pose


def read_text_file(path: str):
    with open(path, 'r') as file:
        file_content = file.read()
    return file_content


# converts a list of oxts measurements into metric poses, starting at (0,0,0) meters, OXTS coordinates are defined as
# x = forward, y = right, z = down (see OXTS RT3000 user manual) afterwards, pose{i} contains the transformation which
# takes a 3D point in the i'th frame and projects it into the oxts coordinates of the first frame

# Translated from the official KITTI convertOxtsToPose matlab files to python.
def convertoxtstopose(oxts: List[Oxts_object]) -> List[Oxts_object]:
    """
    :param oxts:
    :return:
    """
    # compute scale from first lat value
    scale = latToScale(lat=oxts[0].lat)

    # init pose
    # pose_to_start = []
    # pose_to_prev = []
    tr_0_inv = []
    previous_pose = []
    # current_pose = []

    # for all oxts packets do
    for i, oxts_single in enumerate(oxts):

        # if there is no data => no pose
        if oxts_single == []:  # same as == [] // maybe == None
            # pose_to_start.append([])
            oxts_single.add_poses(pose_to_start=np.array([]), pose_to_prev=np.array([]), current_pose=np.array([]))
            continue

        # translation vector
        t = np.zeros((3, 1))
        t[0, 0], t[1, 0] = latlonToMercator(lat=oxts_single.lat, lon=oxts_single.lon, scale=scale)
        t[2, 0] = oxts_single.lon

        # rotation matrix (OXTS RT3000 user manual, page 71/92)
        rx = oxts_single.roll  # roll
        ry = oxts_single.pitch  # pitch
        rz = oxts_single.yaw  # heading
        rot_x = np.array(([[1, 0, 0], [0, np.cos(rx), -np.sin(rx)],
                           [0, np.sin(rx), np.cos(rx)]]))  # base => nav  (level oxts => rotated oxts)
        rot_y = np.array(([[np.cos(ry), 0, np.sin(ry)], [0, 1, 0],
                           [-np.sin(ry), 0, np.cos(ry)]]))  # base => nav  (level oxts => rotated oxts)
        rot_z = np.array(([[np.cos(rz), -np.sin(rz), 0], [np.sin(rz), np.cos(rz), 0],
                           [0, 0, 1]]))  # base => nav  (level oxts => rotated oxts)
        rot_total = np.matmul(np.matmul(rot_z, rot_y), rot_x)

        # normalize translation and rotation (start at 0/0/0)
        if len(tr_0_inv) == 0:  # == []:
            tr_0_inv = np.linalg.inv(np.vstack((np.hstack((rot_total, t)), [0, 0, 0, 1])))

        # add pose
        pose = np.matmul(tr_0_inv, np.vstack((np.hstack((rot_total, t)), [0, 0, 0, 1])))
        # pose_to_start.append(pose)

        if len(previous_pose) == 0:
            pose2prev = np.identity(4)
            # pose_to_prev.append(pose2prev)

        else:
            prev_pose_inv = np.linalg.inv(previous_pose)
            pose2prev = np.matmul(prev_pose_inv, np.vstack((np.hstack((rot_total, t)), [0, 0, 0, 1])))
            # pose_to_prev.append(pose2prev)

        previous_pose = np.vstack((np.hstack((rot_total, t)), [0, 0, 0, 1]))
        # current_pose.append(np.vstack((np.hstack((rot_total, t)), [0, 0, 0, 1])))

        oxts_single.add_poses(pose_to_start=pose, pose_to_prev=pose2prev,
                              current_pose=np.vstack((np.hstack((rot_total, t)), [0, 0, 0, 1])))

    return oxts


def latToScale(lat):
    # compute mercator scale from latitude
    scale = np.cos(lat * np.pi / 180.0)

    return scale


def latlonToMercator(lat, lon, scale):
    # converts lat/lon coordinates to mercator coordinates using mercator scale
    er = 6378137
    mx = scale * lon * np.pi * er / 180
    my = scale * er * np.log(np.tan((90 + lat) * np.pi / 360))

    return mx, my


def load_oxts_files(paths: List[str]) -> List[Oxts_object]:
    oxt_files = []
    for path in paths:
        oxt_files.append(Oxts_object(file_path=path))

    return oxt_files


def transform_pointcloud_example(oxts_objects: List[Oxts_object], pc_idx: int):
    pose_to_start = []
    pose_to_prev = []

    for oxts in oxts_objects:
        pose_to_start.append(oxts.pose_to_start)
        pose_to_prev.append(oxts.pose_to_prev)

    # visualize data
    plt.plot([test1[0, 3] for test1 in pose_to_start], label='x start')
    plt.plot([test1[1, 3] for test1 in pose_to_start], label='y start')
    plt.plot([test1[2, 3] for test1 in pose_to_start], label='z start')
    plt.xlabel('# measurement')
    plt.ylabel('Distance (m)')
    plt.legend()
    plt.show()

    plt.plot([test1[0, 3] for test1 in pose_to_prev], label='x prev')
    plt.plot([test1[1, 3] for test1 in pose_to_prev], label='y prev')
    plt.plot([test1[2, 3] for test1 in pose_to_prev], label='z prev')
    plt.xlabel('# measurement')
    plt.ylabel('Distance (m)')
    plt.legend()
    plt.show()

    start_pc_path = oxts_objects[0].path.replace('oxts', 'velodyne_points').replace('txt', 'bin')
    end_pc_path = oxts_objects[pc_idx].path.replace('oxts', 'velodyne_points').replace('txt', 'bin')
    pose_config = oxts_objects[pc_idx].pose_to_start  # pose_to_start[pc_idx]
    start_pc = np.fromfile(start_pc_path, dtype=np.float32).reshape(-1, 4)[:, :3]
    end_pc = np.fromfile(end_pc_path, dtype=np.float32).reshape(-1, 4)[:, :3]

    new_pc = np.matmul(np.linalg.inv(pose_config), np.hstack((start_pc[:, :3], np.ones((start_pc.shape[0], 1)))).T).T[:,
             :3]

    colors_start = np.ones_like(new_pc)
    colors_end = np.ones_like(end_pc) * 0.4

    points = np.concatenate((new_pc, end_pc))
    colors = np.concatenate((colors_start, colors_end))

    v = pptk.viewer(points, colors)
    v.set(point_size=0.05)



def create_position_info(drive_path: str) -> List[Oxts_object]:
    folder_path = drive_path + '/oxts/data/'
    file_list = sorted([folder_path + path for path in os.listdir(folder_path)])
    oxts_objects = load_oxts_files(paths=file_list)
    oxts_with_pose = convertoxtstopose(oxts=oxts_objects)

    return oxts_with_pose

    # transform_pointcloud_example(oxts_objects=oxts_with_pose, pc_idx=40)
