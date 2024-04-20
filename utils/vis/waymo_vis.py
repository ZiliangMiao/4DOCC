import os
import click
import numpy as np
import open3d as o3d

# visualize the point cloud using open3d
@click.command()
@click.option(
    "--seq_idx",
    "-seq",
    type=int,
    help="sequence index",
    default=0,
    required=True,
)
@click.option(
    "--scan_idx",
    "-scan",
    type=int,
    help="scan index",
    default=0,
    required=True,
)
@click.option(
    "--dataset_path",
    "-p",
    type=str,
    help="dataset path",
    default="/home/mars/4DMOS/data/Waymo_M/sequences",
)
def main(seq_idx, scan_idx, dataset_path):
    pcd_path = os.path.join(dataset_path, str(seq_idx).zfill(4), "lidar", str(scan_idx).zfill(6)+".bin")
    pcd_np = np.fromfile(pcd_path, dtype=np.float32).reshape((-1, 3))  #pcd numpy

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pcd_np)

    o3d.visualization.draw_geometries([pcd],
                                      zoom=0.22119999999999995,
                                      front=[ -0.13364572203040562, 0.11076436056229091, 0.98481981977019462 ],
                                      lookat=[ -9.3558874944560735, 8.8461631564860888, 5.022112913845012 ],
                                      up=[ -0.056526044061650522, 0.99126468140363277, -0.11916013487947659 ])
    # press h to see the helper
    # ctrl+c to copy the current view status info into clipboard
    # press +- to increase or decrease the size of points
    return None

if __name__ == "__main__":
    main()