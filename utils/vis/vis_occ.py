import numpy as np
import open3d
from matplotlib import pyplot as plt


def color_by_z(pts, pc_range, cmap):
    cmap = plt.get_cmap(cmap)
    z = pts[:, 2]
    z_min, z_max = pc_range[2], pc_range[5]
    n_z = (z - z_min) / (z_max - z_min)
    n_z = np.maximum(0, np.minimum(1, n_z))
    return cmap(n_z)[:, :3]

def get_occupancy_as_pcd(pog, thresh, voxel_size, pc_range, cmap):
    def grid_to_pts(X, x_min, y_min, z_min, voxel_size):
        pz, py, px = np.nonzero(X >= thresh)
        xx = px * voxel_size + x_min
        yy = py * voxel_size + y_min
        zz = pz * voxel_size + z_min
        pts = np.stack((xx, yy, zz)).T
        return pts

    x_min, y_min, z_min = pc_range[:3]
    pcds = []
    for t in range(len(pog)):
        pred = (pog[t] >= thresh)
        pts = grid_to_pts(pred, x_min, y_min, z_min, voxel_size)
        colors = color_by_z(pts, pc_range, cmap)
        pcds.append((pts, colors))
    return pcds