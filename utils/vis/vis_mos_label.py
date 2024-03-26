import open3d
import numpy as np

mos_colormap = {
        0: (255/255, 255/255, 255/255),  # unknown: white
        1: (25/255, 80/255, 25/255),    # static: green
        2: (255/255, 20/255, 20/255)     # moving: red
    }

check_colormap = {
        0: (255/255, 20/255, 20/255),     # moving: red
        1: (255/255, 255/255, 255/255),  # unknown: white
    }


def draw(vis, points, labels):
    # draw origin
    axis_pcd = open3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0, origin=[0, 0, 0])
    vis.add_geometry(axis_pcd)
    # draw points
    pts = open3d.geometry.PointCloud()
    pts.points = open3d.utility.Vector3dVector(points)
    vis.add_geometry(pts)
    # draw points label
    vfunc = np.vectorize(mos_colormap.get)
    points_color = np.array(vfunc(labels)).T
    pts.colors = open3d.utility.Vector3dVector(points_color)

def render_mos_pointcloud(points, gt_labels, pred_labels):
    # render gt labels
    vis_gt = open3d.visualization.Visualizer()
    vis_gt.create_window()
    draw(vis_gt, points, gt_labels)

    # render pred labels
    vis_pred = open3d.visualization.Visualizer()
    vis_pred.create_window()
    draw(vis_pred, points, pred_labels)

    # view settings
    vis_gt.get_render_option().point_size = 3.0
    vis_gt.get_render_option().background_color = np.zeros(3)
    view_ctrl_gt = vis_gt.get_view_control()
    view_ctrl_gt.set_front((0.75263429526187886, -0.13358133681379755, 0.64474618575893383))
    view_ctrl_gt.set_lookat((16.206845402638745, -3.8676194858766819, 15.365323753623207))
    view_ctrl_gt.set_up((-0.64932205862151104, 0.011806106960120792, 0.76042190922274799))
    view_ctrl_gt.set_zoom((0.19999999999999998))

    vis_pred.get_render_option().point_size = 3.0
    vis_pred.get_render_option().background_color = np.zeros(3)
    view_ctrl_pred = vis_pred.get_view_control()
    view_ctrl_pred.set_front((0.75263429526187886, -0.13358133681379755, 0.64474618575893383))
    view_ctrl_pred.set_lookat((16.206845402638745, -3.8676194858766819, 15.365323753623207))
    view_ctrl_pred.set_up((-0.64932205862151104, 0.011806106960120792, 0.76042190922274799))
    view_ctrl_pred.set_zoom((0.19999999999999998))

    # run vis
    vis_gt.run()
    vis_pred.run()

def render_mos_comparison(points, gt_labels, pred_labels):
    # open3d vis (for gt labels and pred labels)
    vis_gt = open3d.visualization.Visualizer()
    vis_gt.create_window(window_name='ground truth label', width=1200, height=1600, left=0, top=150)
    vis_pred = open3d.visualization.Visualizer()
    vis_pred.create_window(window_name='predicted label', width=1200, height=1600, left=1200, top=150)
    # vis render option
    vis_gt.get_render_option().point_size = 3.0
    vis_gt.get_render_option().background_color = np.zeros(3)
    vis_pred.get_render_option().point_size = 3.0
    vis_pred.get_render_option().background_color = np.zeros(3)

    # vis view control
    ctrl_gt = vis_gt.get_view_control()
    ctrl_pred = vis_pred.get_view_control()
    # sync camera parameters of two vis
    ctrl_gt.set_front((0.75263429526187886, -0.13358133681379755, 0.64474618575893383))
    ctrl_gt.set_lookat((16.206845402638745, -3.8676194858766819, 15.365323753623207))
    ctrl_gt.set_up((-0.64932205862151104, 0.011806106960120792, 0.76042190922274799))
    ctrl_gt.set_zoom((0.19999999999999998))
    cam_params_gt = ctrl_gt.convert_to_pinhole_camera_parameters()
    ctrl_pred.convert_from_pinhole_camera_parameters(cam_params_gt)

    # draw point cloud
    draw(vis_gt, points, gt_labels)
    draw(vis_pred, points, pred_labels)
    vis_gt.poll_events()
    vis_gt.update_renderer()
    vis_pred.poll_events()
    vis_pred.update_renderer()