import open3d as o3d
import numpy as np
import struct
import os

def find_matching_indices(root, type_lidar):
    assert type_lidar in ["Velodyne", "Ouster", "Aeva", "Avia"]
    dir1 = os.path.join(root, type_lidar, "velodyne")
    dir2 = os.path.join(root, type_lidar, "labels")
    index_set1 = set(os.path.splitext(f)[0] for f in os.listdir(dir1))
    index_set2 = set(os.path.splitext(f)[0] for f in os.listdir(dir2))
    common_indices = index_set1.intersection(index_set2)
    common_indices = sorted(common_indices, key=int)
    return common_indices


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


def read_lidar_file(root, type_lidar, file_index):
    assert type_lidar in ["Velodyne", "Ouster", "Aeva", "Avia"]
    filename_bin = os.path.join(root, type_lidar, "velodyne", f"{file_index}.bin")
    filename_label = os.path.join(root, type_lidar, "labels", f"{file_index}.label")
    points = np.fromfile(filename_bin, dtype=np.float32).reshape(-1, 4)
    labels = np.fromfile(filename_label, dtype=np.uint32)
    return points[:, :3], labels


def read_kitti_poses(path_to_seq):
    pose_file = os.path.join(path_to_seq, 'poses.txt')
    calib_file = os.path.join(path_to_seq, 'calib.txt')
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


def get_transform_from_ref_scan(poses, ref_scan_idx, curr_scan_idx):
    ref_pose = poses[ref_scan_idx]
    curr_pose = poses[curr_scan_idx]
    return np.linalg.inv(ref_pose) @ curr_pose


def transfrom_points(points, transform):
    assert points.shape[1] == 3
    pts_homo = np.hstack([points, np.ones((points.shape[0], 1))]).T
    return (transform @ pts_homo).T[:, :3]


def visualize_point_cloud(points, labels):
    vis = o3d.visualization.VisualizerWithKeyCallback()    
    sample_idx = 0

    mos_colormap = {
        0: (255/255, 255/255, 255/255),  # unknown: white
        9: (25/255, 80/255, 25/255),    # static: green
        251: (255/255, 20/255, 20/255)     # moving: red
    }

    def draw_sample(vis):
        nonlocal sample_idx
        print("Rendering sample: " + str(sample_idx))
        sample_points = points[sample_idx]
        sample_labels = labels[sample_idx].astype(np.uint8)

        # clear geometry
        vis.clear_geometries()

        # draw origin
        axis_pcd = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0, origin=[0, 0, 0])
        vis.add_geometry(axis_pcd)

        # draw points
        pts = o3d.geometry.PointCloud()
        pts.points = o3d.utility.Vector3dVector(sample_points)
        vis.add_geometry(pts)

        # draw points label
        vfunc = np.vectorize(mos_colormap.get)
        points_color = np.array(vfunc(sample_labels)).T
        pts.colors = o3d.utility.Vector3dVector(points_color)

        # view settings
        vis.get_render_option().point_size = 3.0
        vis.get_render_option().background_color = np.zeros(3)

        view_ctrl = vis.get_view_control()
        # view_ctrl.set_front((0.75263429526187886, -0.13358133681379755, 0.64474618575893383))
        # view_ctrl.set_lookat((16.206845402638745, -3.8676194858766819, 15.365323753623207))
        # view_ctrl.set_up((-0.64932205862151104, 0.011806106960120792, 0.76042190922274799))
        # view_ctrl.set_zoom((0.19999999999999998))

        view_ctrl.set_front((-1.0, 0.0, 0.2))
        view_ctrl.set_lookat((-2.0, 0.0, 0.2))
        view_ctrl.set_up((0.0, 0.0, 1.0))
        view_ctrl.set_zoom((0.05))

        # update vis
        vis.poll_events()
        vis.update_renderer()

    def render_next(vis):
        nonlocal sample_idx
        sample_idx += 1
        if sample_idx >= len(labels):
            sample_idx = len(labels) - 1
        draw_sample(vis)

    def render_prev(vis):
        nonlocal sample_idx
        sample_idx -= 1
        if sample_idx < 0:
            sample_idx = 0
        draw_sample(vis)

    vis.create_window()
    vis.register_key_callback(ord('D'), render_next)
    vis.register_key_callback(ord('A'), render_prev)
    draw_sample(vis)
    vis.run()
    


if __name__ == '__main__':
    # Example Usage
    # root = input("Enter the root path of dataset: ")
    # type_lidar = input("Enter the LiDAR type (Avia, Aeva, Ouster, Velodyne): ")  # Change as per your LiDAR type: "Velodyne", "Ouster", "Aeva", or "Livox"

    root = "/home/ubuntu/mos/helimos/HeLiMOS/KAIST05/Deskewed_LiDAR"
    type_lidar = "Aeva"  # ["Velodyne", "Ouster", "Aeva", "Avia"]
    index_range = [2015, 11662] # Note! Labels are avaliable from 001015 to 011662!

    points = []
    labels = []
    poses = read_kitti_poses(os.path.join(root, type_lidar))

    for file_index in find_matching_indices(root, type_lidar):
      if int(file_index) < index_range[1] and int(file_index) > index_range[0]:
        sample_points, sample_labels = read_lidar_file(root, type_lidar, file_index)
        sample_points = transfrom_points(sample_points, poses[int(file_index)])
        points.append(sample_points)
        labels.append(sample_labels)
    visualize_point_cloud(points, labels)