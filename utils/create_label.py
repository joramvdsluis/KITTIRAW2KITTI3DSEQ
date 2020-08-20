import numpy as np
from utils.parseTrackletXML import parseXML
from collections import defaultdict


def xyz_camera_to_lidar_coordinate(calib_dict, lidar_points):
    lidar_points_hom = np.hstack((lidar_points, np.ones((lidar_points.shape[0], 1), dtype=np.float32)))

    R0_ext = np.hstack((calib_dict['R0_rect'].reshape(3, 3), np.zeros((3, 1), dtype=np.float32)))  # (3, 4)
    R0_ext = np.vstack((R0_ext, np.zeros((1, 4), dtype=np.float32)))  # (4, 4)
    R0_ext[3, 3] = 1

    V2C_ext = np.vstack((calib_dict['Tr_velo_to_cam'].reshape(3, 4), np.zeros((1, 4), dtype=np.float32)))  # (4, 4)
    V2C_ext[3, 3] = 1
    pts_lidar = np.dot(lidar_points_hom, np.linalg.inv(np.dot(R0_ext, V2C_ext).T))

    return pts_lidar[:, 0:3]


def xyz_lidar_to_camera_coordinate(calib_dict, lidar_points):
    """
    input: (N,3) array where N is the number of xyz lidar labels (example: lidar_points=np.array([[15.497332336633916, 8.9680576169198734, -1.7421763681726727]]) )
    output: (N,3) array where N is the number of xyz camera labels (example: array([[-8.94831176,  1.92356391, 15.20725978]])
    """

    lidar_points_hom = np.hstack((lidar_points, np.ones((lidar_points.shape[0], 1), dtype=np.float32)))
    camera_coord_lidar_points = np.dot(lidar_points_hom, np.dot(calib_dict['Tr_velo_to_cam'].reshape(3, 4).T,
                                                                calib_dict['R0_rect'].reshape(3, 3).T))

    return camera_coord_lidar_points


# Adapted from PCDet box_utils.py
def boxes3d_to_corners3d_camera(boxes3d, bottom_center=True):
    """
    :param boxes3d: (N, 7) [x, y, z, l, h, w, ry] in camera coords, see the definition of ry in KITTI dataset
    :param bottom_center: whether y is on the bottom center of object
    :return: corners3d: (N, 8, 3)
        7 -------- 4
       /|         /|
      6 -------- 5 .
      | |        | |
      . 3 -------- 0
      |/         |/
      2 -------- 1
    """
    boxes_num = boxes3d.shape[0]
    l, h, w = boxes3d[:, 3], boxes3d[:, 4], boxes3d[:, 5]
    x_corners = np.array([l / 2., l / 2., -l / 2., -l / 2., l / 2., l / 2., -l / 2., -l / 2], dtype=np.float32).T
    z_corners = np.array([w / 2., -w / 2., -w / 2., w / 2., w / 2., -w / 2., -w / 2., w / 2.], dtype=np.float32).T
    if bottom_center:
        y_corners = np.zeros((boxes_num, 8), dtype=np.float32)
        y_corners[:, 0:4] = -h.reshape(boxes_num, 1).repeat(4, axis=1)  # (N, 8)
    else:
        z_corners = np.array([h / 2., h / 2., h / 2., h / 2., -h / 2., -h / 2., -h / 2., -h / 2.], dtype=np.float32).T

    ry = boxes3d[:, 6]
    zeros, ones = np.zeros(ry.size, dtype=np.float32), np.ones(ry.size, dtype=np.float32)
    rot_list = np.array([[np.cos(ry), zeros, -np.sin(ry)],
                         [zeros, ones, zeros],
                         [np.sin(ry), zeros, np.cos(ry)]])  # (3, 3, N)
    R_list = np.transpose(rot_list, (2, 0, 1))  # (N, 3, 3)

    temp_corners = np.concatenate((x_corners.reshape(-1, 8, 1), y_corners.reshape(-1, 8, 1),
                                   z_corners.reshape(-1, 8, 1)), axis=2)  # (N, 8, 3)
    rotated_corners = np.matmul(temp_corners, R_list)  # (N, 8, 3)
    x_corners, y_corners, z_corners = rotated_corners[:, :, 0], rotated_corners[:, :, 1], rotated_corners[:, :, 2]

    x_loc, y_loc, z_loc = boxes3d[:, 0], boxes3d[:, 1], boxes3d[:, 2]

    x = x_loc.reshape(-1, 1) + x_corners.reshape(-1, 8)
    y = y_loc.reshape(-1, 1) + y_corners.reshape(-1, 8)
    z = z_loc.reshape(-1, 1) + z_corners.reshape(-1, 8)

    corners = np.concatenate((x.reshape(-1, 8, 1), y.reshape(-1, 8, 1), z.reshape(-1, 8, 1)), axis=2)

    return corners.astype(np.float32)


# Adapted from PCDet box_utils.py
def boxes3d_camera_to_imageboxes(boxes3d, calib_dict, image_shape=None):
    """
    :param boxes3d: (N, 7) [x, y, z, l, h, w, r] in rect camera coords
    :param calib:
    :return:
        box_2d_preds: (N, 4) [x1, y1, x2, y2]
    """
    corners3d = boxes3d_to_corners3d_camera(boxes3d)
    pts_img, _ = rect_to_img(corners3d.reshape(-1, 3), calib_dict)
    corners_in_image = pts_img.reshape(-1, 8, 2)

    min_uv = np.min(corners_in_image, axis=1)  # (N, 2)
    max_uv = np.max(corners_in_image, axis=1)  # (N, 2)
    boxes2d_image = np.concatenate([min_uv, max_uv], axis=1)
    if image_shape is not None:
        boxes2d_image[:, 0] = np.clip(boxes2d_image[:, 0], a_min=0, a_max=image_shape[1] - 1)
        boxes2d_image[:, 1] = np.clip(boxes2d_image[:, 1], a_min=0, a_max=image_shape[0] - 1)
        boxes2d_image[:, 2] = np.clip(boxes2d_image[:, 2], a_min=0, a_max=image_shape[1] - 1)
        boxes2d_image[:, 3] = np.clip(boxes2d_image[:, 3], a_min=0, a_max=image_shape[0] - 1)

    return boxes2d_image


# Adapted from PCDet box_utils.py
def rect_to_img(pts_rect, calib_dict):
    """
    :param pts_rect: (N, 3)
    :return pts_img: (N, 2)
    """
    pts_rect_hom = np.hstack((pts_rect, np.ones((pts_rect.shape[0], 1), dtype=np.float32)))
    pts_2d_hom = np.dot(pts_rect_hom, calib_dict['P2'].reshape(3, 4).T)
    pts_img = (pts_2d_hom[:, 0:2].T / pts_rect_hom[:, 2]).T  # (N, 2)
    pts_rect_depth = pts_2d_hom[:, 2] - calib_dict['P2'].reshape(3, 4).T[3, 2]  # depth in rect camera coord
    return pts_img, pts_rect_depth


def label_2_text(kitti_labels_dicts, frame_nr: int) -> str:
    create_kitti_labels = ''

    # loop over all labels in one frame:
    for kitti_label_dict in kitti_labels_dicts:
        assert frame_nr == kitti_label_dict['frame_nr'], "WARNING: frame of label and frame do not match!!"
        create_kitti_labels += ('{} {} {} {} {} {} {} {} {} {} {} {} {} {} {} {} {} {}\n'.format(
            kitti_label_dict['type'],
            kitti_label_dict['truncated'],
            kitti_label_dict['occluded'],
            kitti_label_dict['alpha'],
            kitti_label_dict['bbox'][0],
            kitti_label_dict['bbox'][1],
            kitti_label_dict['bbox'][2],
            kitti_label_dict['bbox'][3],
            kitti_label_dict['dimensions'][0],
            kitti_label_dict['dimensions'][1],
            kitti_label_dict['dimensions'][2],
            kitti_label_dict['location'][0],
            kitti_label_dict['location'][1],
            kitti_label_dict['location'][2],
            kitti_label_dict['rotation_y'],
            kitti_label_dict['score'],
            kitti_label_dict['id'],
            kitti_label_dict['frame_nr']
        ))

    # return frame
    return create_kitti_labels


def get_labels_from_drive_xml(xml_path: str, calib_dict):
    create_kitti_labels = ''
    TRUNC_IN_IMAGE = 0
    TRUNC_TRUNCATED = 1
    label_per_frame = defaultdict(list)
    kitti_to_ecp_classes = {
        'Car': 'Car',
        'Van': 'Van',
        'Truck': 'Truck',
        'Pedestrian': 'Pedestrian',
        'Person_sitting': 'Pedestrian',
        'Cyclist': 'Cyclist',
        'Tram': 'Tram',
        'Misc': 'Misc',
        'DontCare': 'DontCare',
        "Person (sitting)":  'Pedestrian'

    }

    tracklets = parseXML(xml_path)

    # loop over tracklets per drive
    for iTracklet, tracklet in enumerate(tracklets):
        # print('tracklet {0: 3d}: {1}'.format(iTracklet, tracklet))

        # this part is inspired by kitti object development kit matlab code: computeBox3D
        h, w, l = tracklet.size

        """"
                #Values    Name      Description
        ----------------------------------------------------------------------------
           1    type         Describes the type of object: 'Car', 'Van', 'Truck',
                             'Pedestrian', 'Person_sitting', 'Cyclist', 'Tram',
                             'Misc' or 'DontCare'
           1    truncated    Float from 0 (non-truncated) to 1 (truncated), where
                             truncated refers to the object leaving image boundaries
           1    occluded     Integer (0,1,2,3) indicating occlusion state:
                             0 = fully visible, 1 = partly occluded
                             2 = largely occluded, 3 = unknown
           1    alpha        Observation angle of object, ranging [-pi..pi]
           4    bbox         2D bounding box of object in the image (0-based index):
                             contains left, top, right, bottom pixel coordinates
           3    dimensions   3D object dimensions: height, width, length (in meters)
           3    location     3D object location x,y,z in camera coordinates (in meters)
           1    rotation_y   Rotation ry around Y-axis in camera coordinates [-pi..pi]
           1    score        Only for results: Float, indicating confidence in
                             detection, needed for p/r curves, higher is better.
        """

        # type
        type_object = kitti_to_ecp_classes[tracklet.objectType]

        # dimensions
        dims = tracklet.size

        # add: ID
        id = hash(str(tracklet.size) + str(xml_path) + str(iTracklet))

        for tracklet_nr in range(0, tracklet.trans.shape[0]):
            # truncated
            truncated = tracklet.truncs[tracklet_nr]

            # occluded [occlusion state, is this an occlusion keyframe]
            occluded = tracklet.occs[tracklet_nr, 0]

            # location
            loc_lidar = tracklet.trans[tracklet_nr, :]
            x_lidar, y_lidar, z_lidar = loc_lidar
            [loc_camera] = xyz_lidar_to_camera_coordinate(calib_dict, np.array([loc_lidar]))

            # rotation_y
            rot_y_cam = ((np.matmul(np.array([tracklet.rots[tracklet_nr, :]]),
                                    calib_dict['Tr_velo_to_cam'].reshape(3, 4)) + np.pi / 2)[0][
                             1] + np.pi) % (2 * np.pi) - np.pi

            # alpha
            alpha = (rot_y_cam - np.arctan2(-y_lidar, x_lidar) + np.pi) % (2. * np.pi) - np.pi

            # score: to make clear this one is not used
            score = -1

            # add: frame number
            current_frame = tracklet.firstFrame + tracklet_nr

            # bbox
            # TODO convert to official kitti conversion instead of PCDet conversion (https://github.com/open-mmlab/OpenPCDet)
            x_cam, y_cam, z_cam = loc_camera
            boxes3d_cam = np.array([[x_cam, y_cam, z_cam, l, h, w, rot_y_cam]])
            cam_2d_box = boxes3d_camera_to_imageboxes(boxes3d=boxes3d_cam, calib_dict=calib_dict)

            label_dict = {
                "type": type_object,
                "truncated": truncated,
                "occluded": occluded,
                "alpha": alpha,
                "bbox": cam_2d_box.flatten(),
                "dimensions": dims,
                "location": loc_camera,
                "rotation_y": rot_y_cam,
                "score": score,
                "id": id,
                "frame_nr": current_frame
            }

            label_per_frame[current_frame].append(label_dict)


    # loop over frames
    label_list = defaultdict(list)
    for frame_number, labels in label_per_frame.items():
        label_list[frame_number] = label_2_text(kitti_labels_dicts=labels, frame_nr=frame_number)

    return label_list
