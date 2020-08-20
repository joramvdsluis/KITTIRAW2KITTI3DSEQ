#!/usr/bin/env python
# if running in py3, change the shebang, drop the next import for readability (it does no harm in py3)
from __future__ import print_function   # py2 compatibility
from collections import defaultdict
import hashlib
import os
import sys
from tqdm import tqdm
import pickle



def chunk_reader(fobj, chunk_size=1024):
    """Generator that reads a file in chunks of bytes"""
    while True:
        chunk = fobj.read(chunk_size)
        if not chunk:
            return
        yield chunk


def get_hash(filename, first_chunk_only=False, hash=hashlib.sha1):
    hashobj = hash()
    file_object = open(filename, 'rb')

    if first_chunk_only:
        hashobj.update(file_object.read(612))
    else:
        for chunk in chunk_reader(file_object):
            hashobj.update(chunk)
    hashed = hashobj.digest()

    file_object.close()
    return hashed


def check_for_duplicates(paths, filetype='.bin', hash=hashlib.sha1,pickle_path=None):
    duplicate_dict=defaultdict(list)
    hashes_by_size = defaultdict(list)  # dict of size_in_bytes: [full_path_to_file1, full_path_to_file2, ]
    hashes_on_1k = defaultdict(list)  # dict of (hash1k, size_in_bytes): [full_path_to_file1, full_path_to_file2, ]
    hasesh_by_name = defaultdict(list)
    hashes_full = {}   # dict of full_file_hash: full_path_to_file_string

    nr_of_files=0
    for path in paths:
        for dirpath, dirnames, filenames in os.walk(path):
            # print(dirpath)
            if ('0001' in dirpath and 'image_02' in dirpath) or 'image_2' in dirpath:
                nr_of_files += len(filenames)

    with tqdm(total=nr_of_files) as pbar:
        for path in paths:
            for dirpath, dirnames, filenames in os.walk(path):
                # get all files that have the same size - they are the collision candidates
                if 'image_02' in dirpath or 'image_2' in dirpath:
                    for filename in filenames:
                        pbar.update(1)
                        if filename.endswith(filetype):
                            full_path = os.path.join(dirpath, filename)
                            try:
                                # if the target is a symlink (soft one), this will
                                # dereference it - change the value to the actual target file
                                full_path = os.path.realpath(full_path)
                                file_size = os.path.getsize(full_path)
                                hashes_by_size[file_size].append(full_path)
                            except (OSError,):
                                # not accessible (permissions, etc) - pass on
                                continue
        print('done with identifying file sizes')


    # For all files with the same file size, get their hash on the 1st 1024 bytes only
    with tqdm(total=len(hashes_by_size.items())) as pbar2:
        for size_in_bytes, files in hashes_by_size.items():
            pbar2.update(1)
            if len(files) < 2:
                continue    # this file size is unique, no need to spend CPU cycles on it

            for filename in files:
                try:
                    small_hash = get_hash(filename, first_chunk_only=True)
                    # the key is the hash on the first 1024 bytes plus the size - to
                    # avoid collisions on equal hashes in the first part of the file
                    # credits to @Futal for the optimization
                    hashes_on_1k[(small_hash, size_in_bytes)].append(filename)
                    hasesh_by_name[filename].append(small_hash)
                except (OSError,):
                    # the file access might've changed till the exec point got here
                    continue

    print('done with identifying file sizes and getting hash')

    # For all files with the hash on the 1st 1024 bytes, get their hash on the full file - collisions will be duplicates
    with tqdm(total=len(hashes_on_1k.items())) as pbar3:
        for __, files_list in hashes_on_1k.items():
            pbar3.update(1)
            if len(files_list) < 2:
                continue    # this hash of fist 1k file bytes is unique, no need to spend cpy cycles on it

            for filename in files_list:
                try:
                    full_hash = get_hash(filename, first_chunk_only=False)
                    duplicate = hashes_full.get(full_hash)
                    if duplicate:
                        print("Duplicate found: {} and {}".format(filename, duplicate))
                        duplicate_dict[duplicate] = filename

                    else:
                        hashes_full[full_hash] = filename
                except (OSError,):
                    # the file access might've changed till the exec point got here
                    continue

    if pickle_path is not None:
        with open(pickle_path + 'duplicates.pkl', 'wb') as f:
            pickle.dump(duplicate_dict, f)

    return duplicate_dict


if __name__ == "__main__":
    # original_root = '/data/Datasets/KITTI_3D_Object_Detection/object/training/image_2/000045.png'
    original_root = '/data/Datasets/KITTI_3D_Object_Detection/object/training/image_2/'
    # original_root = '/home/jrvandersluis/Downloads/test2/'

    # whole_dataset_root = '/data2/KITTI_RAW_sync/2011_09_26/2011_09_26_drive_0001_sync/image_02/data/0000000021.png'
    whole_dataset_root = '/data2/KITTI_RAW_KITTI_FORMAT_whole/object/training/image_2/'
    # whole_dataset_root = '/home/jrvandersluis/Downloads/test1/'
    filetype='.png'
    pickle_path = os.path.dirname(os.path.dirname(os.path.dirname(whole_dataset_root))) + '/'
    duplicate_dict= check_for_duplicates(paths=[original_root, whole_dataset_root], filetype=filetype,pickle_path=pickle_path)