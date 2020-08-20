import os
from collections import defaultdict
from utils.create_calib import create_calib_dict
from utils.write_files import write_calib_file, write_txt_to_file, write_pose_file
from utils.create_label import get_labels_from_drive_xml
from utils.create_position import create_position_info
from shutil import copyfile
from tqdm import tqdm


def main(input_main_dir, output_main_dir, only_with_labels, emtpy_labels_if_no_label):
    # Add name behind dataset to make clear which of the two possible datasets is created
    if only_with_labels:
        output_main_dir += '_only_drives_with_label'
    else:
        output_main_dir += '_whole'

    # Initialize defaultdicts
    calib_paths = defaultdict(list)
    calib = defaultdict(list)
    oxts_dict = defaultdict(list)
    output_dirs = {
        "label_2": output_main_dir + '/object/training/label_2',
        "calib": output_main_dir + '/object/training/calib',
        "image_2": output_main_dir + '/object/training/image_2',
        "planes": output_main_dir + '/object/training/planes',
        "velodyne": output_main_dir + '/object/training/velodyne',
        "oxts": output_main_dir + '/object/training/oxts',
        "object": output_main_dir + '/object'
    }

    # Create folder structure
    for output_dir in output_dirs.values():
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

    # Loop over files and count them, this is required for tqdm
    nr_of_files = 0
    for dirpath, dirnames, filenames in os.walk(input_main_dir):
        if not ('image_00' in dirpath or 'image_01' in dirpath or 'image_03' in dirpath):
            nr_of_files += len(filenames)

    with tqdm(total=nr_of_files) as pbar:

        # loop through recording date directories
        for recoding_date in sorted(
                [date for date in os.listdir(input_main_dir) if os.path.isdir(os.path.join(input_main_dir, date))]):

            # loop through drives and save files which are no directory (those are the calibration files)
            for drive_number in sorted(
                    [drive_nr for drive_nr in os.listdir(os.path.join(input_main_dir, recoding_date)) if
                     not os.path.isdir(os.path.join(input_main_dir, recoding_date, drive_nr))]):
                orig_calib_path = os.path.join(input_main_dir, recoding_date, drive_number)
                calib_paths[recoding_date].append(orig_calib_path)
                pbar.update(1)

            # Determine the calib file per recording date:
            calib[recoding_date] = create_calib_dict(calib_paths=calib_paths[recoding_date])

            # Loop through drives
            for drive_number in sorted(
                    [drive_nr for drive_nr in os.listdir(os.path.join(input_main_dir, recoding_date)) if
                     os.path.isdir(os.path.join(input_main_dir, recoding_date, drive_nr))]):

                # Get label path if there
                if os.path.exists(os.path.join(input_main_dir, recoding_date, drive_number, 'tracklet_labels.xml')):
                    # get label path
                    orig_tracklet_label_path = os.path.join(input_main_dir, recoding_date, drive_number,
                                                            'tracklet_labels.xml')
                    # Extract labels from xml format
                    label_list = get_labels_from_drive_xml(xml_path=orig_tracklet_label_path,
                                                           calib_dict=calib[recoding_date])

                    # Loop over all frames in the labels
                    for frame_number, labels_per_frame in label_list.items():
                        # Create output name of label file
                        output_label_path = os.path.join(output_dirs['label_2'],
                                                         drive_number + '_' + str(frame_number).zfill(10) + '.txt')

                        # If file does not already exist, write file
                        if not os.path.exists(output_label_path):
                            write_txt_to_file(ouput_txt=labels_per_frame, output_dir=output_dirs['label_2'],
                                              output_filename=str(
                                                  drive_number + '_' + str(frame_number).zfill(10) + '.txt'))
                    pbar.update(1)
                else:
                    # If true, create dataset with only labeled drives, break if no label files in drive
                    if only_with_labels:
                        break
                    else:
                        pass

                # Get poses if available for whole sequence (drive)
                if os.path.exists(os.path.join(input_main_dir, recoding_date, drive_number, 'oxts', 'data')):
                    # Get path file of location information
                    orig_oxts_drive_path = os.path.join(input_main_dir, recoding_date, drive_number)

                    # Convert location information to poses for drive
                    oxts_with_pose = create_position_info(orig_oxts_drive_path)

                    # Save location infrormation per frame
                    for sinle_oxts_with_pose in oxts_with_pose:
                        oxts_dict[sinle_oxts_with_pose.path] = sinle_oxts_with_pose

                    pbar.update(1)

                # loop through images if there
                for image in sorted([drive_nr for drive_nr in os.listdir(
                        os.path.join(input_main_dir, recoding_date, drive_number, 'image_02', 'data'))]):
                    # Get image path
                    orig_image_path = os.path.join(input_main_dir, recoding_date, drive_number, 'image_02', 'data',
                                                   image)
                    # Create output path
                    output_image_path = os.path.join(output_dirs['image_2'], drive_number + '_' + image)

                    # If not already existing, copy file.
                    if not os.path.exists(output_image_path):
                        copyfile(orig_image_path, output_image_path)

                    pbar.update(1)

                # Loop through velodyne point clouds if there
                for point_cloud in sorted([drive_nr for drive_nr in os.listdir(
                        os.path.join(input_main_dir, recoding_date, drive_number, 'velodyne_points', 'data'))]):
                    # Get original file name
                    orig_velo_path = os.path.join(input_main_dir, recoding_date, drive_number, 'velodyne_points',
                                                  'data', point_cloud)
                    # Get output path
                    output_velo_path = os.path.join(output_dirs['velodyne'], drive_number + '_' + point_cloud)

                    # Save file to output path
                    if not os.path.exists(output_velo_path):
                        copyfile(orig_velo_path, output_velo_path)

                    pbar.update(1)

                    # For every point cloud: write a calib file for the corresponding drive
                    output_calib_path = os.path.join(output_dirs['calib'],
                                                     drive_number + '_' + point_cloud.replace('.bin', '.txt'))
                    if not os.path.exists(output_calib_path):
                        write_calib_file(calib_dict=calib[recoding_date], output_dir=output_dirs['calib'],
                                         output_filename=drive_number + '_' + point_cloud.replace('.bin', '.txt'))

                # Loop through oxts if there
                for oxts_file in sorted([drive_nr for drive_nr in os.listdir(
                        os.path.join(input_main_dir, recoding_date, drive_number, 'oxts', 'data'))]):
                    # Get original path
                    orig_oxts_path = os.path.join(input_main_dir, recoding_date, drive_number, 'oxts', 'data',
                                                  oxts_file)
                    # Get output path
                    output_oxts_path = os.path.join(output_dirs['oxts'], drive_number + oxts_file)
                    # Write pose if not already exists
                    if not os.path.exists(output_oxts_path):
                        write_pose_file(oxts=oxts_dict[orig_oxts_path], output_dir=str(output_dirs['oxts']),
                                        output_filename=str(drive_number + '_' + oxts_file))

                    pbar.update(1)

        # Create emtpy label files if emtpy_labels_if_no_label is True
        if not only_with_labels:
            if emtpy_labels_if_no_label:
                label_files = [os.path.join(output_dirs['label_2'], item.replace('.bin', '.txt')) for item in
                               sorted(os.listdir(output_dirs['velodyne']))]
                for label_file in label_files:
                    if not os.path.exists(label_file):
                        with open(label_file, 'w') as file:
                            pass

        # Check lists of all avaialble frames.
        val_list_label = [item.replace('.txt', '') for item in sorted(os.listdir(output_dirs['label_2']))]
        val_list_img = [item.replace('.png', '') for item in sorted(os.listdir(output_dirs['image_2']))]
        val_list_velo = [item.replace('.bin', '') for item in sorted(os.listdir(output_dirs['velodyne']))]
        val_list_calib = [item.replace('.txt', '') for item in sorted(os.listdir(output_dirs['calib']))]
        val_list_oxts = [item.replace('.txt', '') for item in sorted(os.listdir(output_dirs['oxts']))]

        # Determine by intersection of the file list for which frame all information is available, if so, add to val.txt
        if only_with_labels:
            val_list = sorted(set(val_list_img).intersection(val_list_velo).intersection(val_list_calib).intersection(
                val_list_oxts).intersection(val_list_label))
            val_list = [val.replace('.txt', '') for val in val_list]
        else:
            val_list = sorted(set(val_list_img).intersection(val_list_velo).intersection(val_list_calib).intersection(
                val_list_oxts))
            val_list = [val.replace('.bin', '') for val in val_list]

        with open(os.path.join(output_dirs['object'], 'val.txt'),
                  'w') as file:  # can also be saved as test.txt or train.txt or trainval.txt
            [file.write("%s\n" % item) for item in val_list]

        with open(os.path.join(output_dirs['object'], 'trainval.txt'), 'w') as file:
            [file.write("%s\n" % item) for item in val_list]

        with open(os.path.join(output_dirs['object'], 'train.txt'), 'w') as file:
            pass

        with open(os.path.join(output_dirs['object'], 'test.txt'), 'w') as file:
            pass


if __name__ == '__main__':
    # Define KITTI RAW dir
    input_main_dir = '/data2/KITTI_RAW'

    # Define output dir
    output_main_dir = '/data2/KITTI_RAW_KITTI_FORMAT'

    # Only generate dataset with labels
    only_with_labels = True

    # create empty label files for frames without labels:
    emtpy_labels_if_no_label = True

    main(input_main_dir=input_main_dir, output_main_dir=output_main_dir, only_with_labels=only_with_labels,
         emtpy_labels_if_no_label=emtpy_labels_if_no_label)
