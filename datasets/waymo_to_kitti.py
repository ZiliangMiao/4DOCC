import multiprocessing
import os
import click
import tensorflow.compat.v1 as tf
import math
import numpy as np
from tqdm import tqdm
from multiprocessing import Process
import itertools
import matplotlib.pyplot as plt

tf.enable_eager_execution()

from waymo_open_dataset.utils import range_image_utils
from waymo_open_dataset.utils import transform_utils
from waymo_open_dataset.utils import frame_utils
from waymo_open_dataset import dataset_pb2 as open_dataset

def save_poses(seq_idx):
    print("Save poses of seq" + str(seq_idx) + " to poses.txt files")
    tfrecord_path = os.path.join("/home/mars/4DMOS/data/Waymo_M/sequences", str(seq_idx).zfill(4), "tfrecord")
    tfrecord_file = os.listdir(tfrecord_path)[0]
    tfrecord_file = os.path.join(tfrecord_path, tfrecord_file)
    poses_file = os.path.join("/home/mars/4DMOS/data/Waymo_M/sequences", str(seq_idx).zfill(4), "poses.txt")
    seq_data = tf.data.TFRecordDataset(tfrecord_file, compression_type='')

    poses_list = []
    for scan_idx, scan_data in enumerate(tqdm(seq_data)):
        frame = open_dataset.Frame()
        frame.ParseFromString(bytearray(scan_data.numpy()))  # frame has the timestamp_micros attribute
        # pose of frame
        pose = np.array(frame.pose.transform)[0:12].reshape(1, 12)
        poses_list.append(pose)
    poses = np.concatenate(poses_list, axis=0)
    np.savetxt(poses_file, poses, delimiter=' ')
    # verify
    # kitti_pose = np.loadtxt("/home/mars/4DMOS/data/SeKITTI/sequences/0000/poses.txt", delimiter=' ')
    # kitti_scan_pose = kitti_pose[0].reshape(-1, 4)
    # kitti_scan_pose = np.vstack((kitti_scan_pose, np.array([0, 0, 0, 1])))
    # waymo_pose = np.loadtxt("/home/mars/4DMOS/data/Waymo_M/sequences/0000/poses.txt", delimiter=' ')
    # waymo_scan_pose = waymo_pose[0].reshape(-1, 4)
    # waymo_scan_pose = np.vstack((waymo_scan_pose, np.array([0, 0, 0, 1])))

def transfer_to_bin(seq_idx):
    print("Transfer seq" + str(seq_idx) + " to .bin files")
    tfrecord_path = os.path.join("/home/mars/4DMOS/data/Waymo_M/sequences", str(seq_idx).zfill(4), "tfrecord")
    tfrecord_file = os.listdir(tfrecord_path)[0]
    tfrecord_file = os.path.join(tfrecord_path, tfrecord_file)
    bin_path = os.path.join("/home/mars/4DMOS/data/Waymo_M/sequences", str(seq_idx).zfill(4), "lidar")
    seq_data = tf.data.TFRecordDataset(tfrecord_file, compression_type='')

    for scan_idx, scan_data in enumerate(tqdm(seq_data)):
        frame = open_dataset.Frame()
        frame.ParseFromString(bytearray(scan_data.numpy()))  # frame has the timestamp_micros attribute
        # load point clouds
        range_images, camera_projections, _, range_image_top_pose = (
            frame_utils.parse_range_image_and_camera_projection(frame))
        frame.lasers.sort(key=lambda laser: laser.name)
        points, _ = frame_utils.convert_range_image_to_point_cloud(
            frame, range_images, camera_projections, range_image_top_pose)  # points contains point clouds of both 5 lidars

        # Points in vehicle frame
        # scan_points = np.concatenate(points, axis=0)  # register all points of five lidars
        top_lidar_points = points[0]
        bin_file = os.path.join(bin_path, str(scan_idx).zfill(6) + ".bin")
        top_lidar_points.tofile(bin_file)

def check_labels_to_points(seq_idx):
    print("Transfer seq" + str(seq_idx) + " to .bin files")
    dataset_path = "/home/mars/4DMOS/data/Waymo_M/sequences"
    # points
    tfrecord_path = os.path.join(dataset_path, str(seq_idx).zfill(4), "tfrecord")
    tfrecord_file = os.listdir(tfrecord_path)[0]
    tfrecord_file = os.path.join(tfrecord_path, tfrecord_file)
    seq_data = tf.data.TFRecordDataset(tfrecord_file, compression_type='')

    # labels
    semantic_label_path = os.path.join(dataset_path, str(seq_idx).zfill(4), "labels")
    semantic_label_files = [os.path.join(dp, f) for dp, dn, fn in os.walk(os.path.expanduser(semantic_label_path)) for f
                            in fn]
    semantic_label_files.sort()

    for scan_idx, scan_data in enumerate(tqdm(seq_data)):
        frame = open_dataset.Frame()
        frame.ParseFromString(bytearray(scan_data.numpy()))  # frame has the timestamp_micros attribute
        # load point clouds
        range_images, camera_projections, _, range_image_top_pose = (
            frame_utils.parse_range_image_and_camera_projection(frame))
        frame.lasers.sort(key=lambda laser: laser.name)
        # double returns
        points_0, cp_points_0 = frame_utils.convert_range_image_to_point_cloud(
            frame, range_images, camera_projections, range_image_top_pose,
            ri_index=0)  # points contains point clouds of both 5 lidars
        points_1, cp_points_1 = frame_utils.convert_range_image_to_point_cloud(
            frame, range_images, camera_projections, range_image_top_pose,
            ri_index=1)  # points contains point clouds of both 5 lidars

        top_points_0 = points_0[0]
        top_points_1 = points_1[0]

        all_points_0 = np.concatenate(points_0, axis=0)
        all_points_1 = np.concatenate(points_1, axis=0)

        all_points = np.concatenate([all_points_0, all_points_1], axis=0)  # register all points of five lidars

        # points
        top_lidar_points = np.concatenate([top_points_0, top_points_1], axis=0)  # double returns, top lidar points
        num_top_points = top_lidar_points.shape[0]
        num_all_points = all_points.shape[0]

        # labels
        semantic_label_file = semantic_label_files[scan_idx]
        labels = np.fromfile(semantic_label_file, dtype=np.uint32)
        uint_labels = np.fromfile(semantic_label_file, dtype=np.uint32).reshape((-1)) & 0xFFFF
        int_labels = np.fromfile(semantic_label_file, dtype=np.int32).reshape((-1)) & 0xFFFF
        num_uint_labels = uint_labels.shape[0]
        num_int_labels = int_labels.shape[0]
        a = 1

def num_points_in_scan0():
    dataset_path = "/home/mars/4DMOS/data/Waymo_M/sequences"
    num_all_points_list = []
    num_top_points_list = []
    num_labels_list = []

    for seq_idx in range(44):
        # points
        tfrecord_path = os.path.join(dataset_path, str(seq_idx).zfill(4), "tfrecord")
        tfrecord_file = os.listdir(tfrecord_path)[0]
        tfrecord_file = os.path.join(tfrecord_path, tfrecord_file)
        seq_data = tf.data.TFRecordDataset(tfrecord_file, compression_type='')
        # labels
        semantic_label_path = os.path.join(dataset_path, str(seq_idx).zfill(4), "labels")
        semantic_label_files = [os.path.join(dp, f) for dp, dn, fn in os.walk(os.path.expanduser(semantic_label_path))
                                for f
                                in fn]
        semantic_label_files.sort()

        # points
        for scan0_data in seq_data:
            frame = open_dataset.Frame()
            frame.ParseFromString(bytearray(scan0_data.numpy()))  # frame has the timestamp_micros attribute

            range_images, camera_projections, _, range_image_top_pose = (
                frame_utils.parse_range_image_and_camera_projection(frame))
            frame.lasers.sort(key=lambda laser: laser.name)
            points, _ = frame_utils.convert_range_image_to_point_cloud(
                frame, range_images, camera_projections,
                range_image_top_pose)  # points contains point clouds of both 5 lidars

            all_points = np.concatenate(points, axis=0)  # register all points of five lidars
            top_lidar_points = points[0]
            num_top_points = top_lidar_points.shape[0]
            num_all_points = all_points.shape[0]
            num_top_points_list.append(num_top_points)
            num_all_points_list.append(num_all_points)
            break

        # labels
        semantic_label_file = semantic_label_files[0]
        labels = np.fromfile(semantic_label_file, dtype=np.uint32).reshape((-1)) & 0xFFFF
        num_labels = labels.shape[0]
        num_labels_list.append(num_labels)
    a = 1

def read_bin_file(filename):
    """Load point clouds from .bin file"""
    point_cloud = np.fromfile(filename, dtype=np.float32).reshape((-1, 4))
    point_cloud = point_cloud[:, :3]
    return point_cloud

if __name__ == "__main__":
    check_labels_to_points(43)
    seqs_num = 44
    # multi-processing loop
    pool = multiprocessing.Pool(processes=44)
    for seq_idx in range(seqs_num):
        pool.apply_async(func=transfer_to_bin, args=(seq_idx,))
        pool.apply_async(func=save_poses, args=(seq_idx,))
    pool.close()
    pool.join()
    print("multi-processing success!")

