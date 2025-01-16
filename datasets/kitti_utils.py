import os
import numpy as np
import torch


def load_poses(pose_path):
    """Load ground truth poses (T_w_cam0) from file.
    Args:
      pose_path: (Complete) filename for the pose file
    Returns:
      A numpy array of size nx4x4 with n poses as 4x4 transformation
      matrices
    """
    # Read and parse the poses
    poses = []
    try:
        if ".txt" in pose_path:
            with open(pose_path, "r") as f:
                lines = f.readlines()
                for line in lines:
                    T_w_cam0 = np.fromstring(line, dtype=float, sep=" ")

                    if len(T_w_cam0) == 12:
                        T_w_cam0 = T_w_cam0.reshape(3, 4)
                        T_w_cam0 = np.vstack((T_w_cam0, [0, 0, 0, 1]))
                    elif len(T_w_cam0) == 16:
                        T_w_cam0 = T_w_cam0.reshape(4, 4)
                    poses.append(T_w_cam0)
        else:
            poses = np.load(pose_path)["arr_0"]

    except FileNotFoundError:
        print("Ground truth poses are not avaialble.")

    return np.array(poses)


def load_calib(calib_path):
    """Load calibrations (T_cam_velo) from file."""
    # Read and parse the calibrations
    T_cam_velo = []
    try:
        with open(calib_path, "r") as f:
            lines = f.readlines()
            for line in lines:
                if "Tr:" in line:
                    line = line.replace("Tr:", "")
                    T_cam_velo = np.fromstring(line, dtype=float, sep=" ")
                    T_cam_velo = T_cam_velo.reshape(3, 4)
                    T_cam_velo = np.vstack((T_cam_velo, [0, 0, 0, 1]))

    except FileNotFoundError:
        print("Calibrations are not avaialble.")

    return np.array(T_cam_velo)


def load_timestamp(path_to_seq):
    timestamp_file = os.path.join(path_to_seq, "times.txt")
    timestamp_list = []
    try:
        with open(timestamp_file, "r") as f:
            lines = f.readlines()
            for line in lines:
                timestamp = float(line.split("\n")[0])
                timestamp_list.append(timestamp)
    except FileNotFoundError:
        print("Timestamps are not avaialble.")
    return timestamp_list


def read_kitti_poses(path_to_seq):
    pose_file = os.path.join(path_to_seq, "poses.txt")
    calib_file = os.path.join(path_to_seq, "calib.txt")
    poses = np.array(load_poses(pose_file))
    inv_frame0 = np.linalg.inv(poses[0])

    # load calibrations
    T_cam_velo = load_calib(calib_file)
    T_cam_velo = np.asarray(T_cam_velo).reshape((4, 4))
    T_velo_cam = np.linalg.inv(T_cam_velo)

    # convert kitti poses from camera coord to LiDAR coord
    new_poses = []
    for pose in poses:
        new_poses.append(T_velo_cam.dot(inv_frame0).dot(pose).dot(T_cam_velo))
    poses = np.array(new_poses)
    return poses


def add_timestamp(tensor, ts):
    """Add time as additional column to tensor"""
    n_points = tensor.shape[0]
    ts = ts * torch.ones((n_points, 1))
    tensor_with_ts = torch.hstack([tensor, ts])
    return tensor_with_ts


def load_files(folder, dataset_root, seq_idx):
    """Load all files path in a folder and sort."""
    # file_paths = [os.path.join(dp, f) for dp, dn, fn in os.walk(os.path.expanduser(folder)) for f in fn]
    # file_paths.sort()
    file_paths = []
    label_paths = []
    for root, dirs, files in os.walk(os.path.expanduser(folder)):
        for f in files:
            file_paths.append(os.path.join(root, f))
            scan_idx = f.split(".")[0]
            label_dir = os.path.join(dataset_root, str(seq_idx).zfill(2), "labels")
            assert os.path.exists(label_dir)
            label_file = os.path.join(label_dir, str(scan_idx + ".label"))
            label_paths.append(label_file)
    file_paths.sort()
    label_paths.sort()
    return file_paths, label_paths


def get_transformed_pcd(cfg, path_to_seq, ref_scan_idx, scan_idx):
    # get pcd
    pcd_file = os.path.join(path_to_seq, "velodyne", str(scan_idx).zfill(6) + ".bin")
    pcd = np.fromfile(pcd_file, dtype=np.float32)
    pcd = torch.tensor(pcd.reshape((-1, 4)))[:, :3]
    # get pose
    poses = read_kitti_poses(path_to_seq)
    # get timestamps
    timestamps_list = load_timestamp(path_to_seq)

    if cfg["transform"]:
        curr_pose = poses[scan_idx]
        ref_pose = poses[ref_scan_idx]
        points_tf = transform_point_cloud(pcd, curr_pose, ref_pose)  # curr point cloud, at {ref lidar} frame
    # true relative timestamp
    ts_rela = timestamps_list[scan_idx] - timestamps_list[ref_scan_idx]

    # transformed sensor origin and points, at {ref lidar} frame
    origin_tf = torch.tensor(curr_pose[:3, 3], dtype=torch.float32)  # curr sensor location, at {ref lidar} frame

    # filter ego points and outside scene bbox points
    valid_mask = torch.squeeze(torch.full((len(points_tf), 1), True))
    if cfg['ego_mask']:
        ego_mask = get_ego_mask(points_tf)
        valid_mask = torch.logical_and(valid_mask, ~ego_mask)
    if cfg['outside_scene_mask']:
        outside_scene_mask = get_outside_scene_mask(points_tf, cfg["scene_bbox"], cfg['outside_scene_mask_z'], cfg['outside_scene_mask_ub'])
        valid_mask = torch.logical_and(valid_mask, ~outside_scene_mask)
    points_tf = points_tf[valid_mask]
    return origin_tf, points_tf, ts_rela, valid_mask


def get_ego_mask(pcd):  # mask the points of ego vehicle: x [-0.8, 0.8], y [-1.5, 2.5]
    # kitti car: length (4.084 m), width (1.730 m), height (1.562 m)
    # https://www.cvlibs.net/datasets/kitti/setup.php
    ego_mask = torch.logical_and(
        torch.logical_and(-0.760 - 0.8 <= pcd[:, 0], pcd[:, 0] <= 1.950 + 0.8),
        torch.logical_and(-0.850 - 0.2 <= pcd[:, 1], pcd[:, 1] <= 0.850 + 0.2),
    )
    return ego_mask


def get_outside_scene_mask(pcd, scene_bbox, mask_z: bool, upper_bound: bool):  # TODO: note, preprocessing use both <=
    if upper_bound: # TODO: for mop pre-processing, keep ray index unchanged
        inside_scene_mask = torch.logical_and(scene_bbox[0] <= pcd[:, 0], pcd[:, 0] <= scene_bbox[3])
        inside_scene_mask = torch.logical_and(inside_scene_mask,
                                              torch.logical_and(scene_bbox[1] <= pcd[:, 1], pcd[:, 1] <= scene_bbox[4]))
    else: # TODO: for uno, avoid index out of bound
        inside_scene_mask = torch.logical_and(scene_bbox[0] <= pcd[:, 0], pcd[:, 0] < scene_bbox[3])
        inside_scene_mask = torch.logical_and(inside_scene_mask,
                                              torch.logical_and(scene_bbox[1] <= pcd[:, 1], pcd[:, 1] < scene_bbox[4]))
    if mask_z:
        inside_scene_mask = torch.logical_and(inside_scene_mask,
                                              torch.logical_and(scene_bbox[2] <= pcd[:, 2], pcd[:, 2] < scene_bbox[5]))
    return ~inside_scene_mask


def transform_point_cloud(pcd, from_pose, to_pose):
    transformation = torch.Tensor(np.linalg.inv(to_pose) @ from_pose)
    NP = pcd.shape[0]
    xyz1 = torch.hstack([pcd, torch.ones(NP, 1)]).T
    pcd = (transformation @ xyz1).T[:, :3]
    return pcd


def load_mos_labels(filename):
    """Load moving object labels from .label file"""
    semantic_labels = np.fromfile(filename, dtype=np.int32).reshape((-1))
    semantic_labels = semantic_labels & 0xFFFF  # Mask semantics in lower half
    mos_labels = np.ones_like(semantic_labels)
    mos_labels[semantic_labels <= 1] = 0  # Unlabeled (0), outlier (1)
    mos_labels[semantic_labels > 250] = 2  # Moving
    mos_labels = torch.tensor(mos_labels.astype(dtype=np.uint8).reshape(-1))
    return mos_labels


def get_mutual_scans_dict(seq_scans_list, path_to_seq, cfg):
    key_sd_toks_dict = {}
    timestamps_list = load_timestamp(path_to_seq)

    # for scan skip, determine the valid scans
    valid_scans = []
    skip_cnt = 0
    for i in range(len(seq_scans_list)):
        if skip_cnt < cfg["n_skip"]:
            skip_cnt += 1
            continue
        skip_cnt = 0
        valid_scans.append(int(seq_scans_list[i]))

    past_scans = []
    past_scans_cnt = 0
    for i, scan_idx in enumerate(valid_scans):
        # get input scans
        if past_scans_cnt < cfg["n_input"] - 1:
            past_scans.append(scan_idx)
            past_scans_cnt += 1
            continue
        if past_scans_cnt == cfg["n_input"] - 1:
            if i >= len(valid_scans):
                break
            ref_scan_idx = valid_scans[i]
            ref_ts = timestamps_list[ref_scan_idx]

            # get future scans
            future_scans = valid_scans[i+1:i+6]
            if len(future_scans) < 5:
                continue
            if len(future_scans) == 5:
                key_sd_toks_dict[ref_scan_idx] = past_scans + future_scans
                past_scans = []
                past_scans_cnt = 0
    return key_sd_toks_dict