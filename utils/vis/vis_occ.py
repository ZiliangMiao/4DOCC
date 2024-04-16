import os
import open3d
import numpy as np
from matplotlib import pyplot as plt


def color_by_z(pts, pc_range, cmap):
    cmap = plt.get_cmap(cmap)
    z = pts[:, 2]
    z_min, z_max = pc_range[2], pc_range[5]
    n_z = (z - z_min) / (z_max - z_min)
    n_z = np.maximum(0, np.minimum(1, n_z))
    return cmap(n_z)[:, :3]

def color_by_density(density, cmap):
    cmap = plt.get_cmap(cmap)
    d_min, d_max = np.min(density), np.max(density)
    n_d = (density - d_min) / (d_max - d_min)  # (0, 1)
    n_d = np.maximum(0, np.minimum(1, n_d))    # (0, 1)
    return cmap(n_d)[:, :3]

def get_occupancy_as_pcd(pog, thresh, voxel_size, pc_range, cmap, filename):
    x_min, y_min, z_min = pc_range[:3]
    for t in range(len(pog)):  # time loop
        coord_min = np.array([z_min, y_min, x_min]).reshape(1, 3)
        coords = np.argwhere(pog[t] > thresh) * voxel_size + coord_min

        density = pog[t].reshape(-1, 1)
        density = np.squeeze(density[density > thresh])
        assert len(density) == len(coords)

        z = coords[:, 0].reshape(-1, 1)
        y = coords[:, 1].reshape(-1, 1)
        x = coords[:, 2].reshape(-1, 1)
        pts = np.concatenate((x, y, z), axis=1)
        colors = color_by_z(pts, pc_range, cmap)
        # colors = color_by_density(density, cmap)

        # write occ pcd
        pred_pcd_file = os.path.join(f"{filename}_occ-{t}.pcd")
        occ_pred_pcd = open3d.geometry.PointCloud()
        occ_pred_pcd.points = open3d.utility.Vector3dVector(pts)
        occ_pred_pcd.colors = open3d.utility.Vector3dVector(colors)
        open3d.io.write_point_cloud(pred_pcd_file, occ_pred_pcd)

        # # visualize
        # open3d.geometry.VoxelGrid.create_from_point_cloud_within_bounds(occ_pred_pcd, voxel_size,
        #                                                                 pc_range[:3], pc_range[3:])



if __name__ == '__main__':
    pog_rand = np.random.rand(6, 45, 700, 700)
    voxel_size = 0.2
    pc_range = [-70.0, -70.0, -4.5, 70.0, 70.0, 4.5]
    cmap = "Oranges"
    threshold = 0.01
    occ_pcd = get_occupancy_as_pcd(pog_rand, threshold, voxel_size, pc_range, cmap)
