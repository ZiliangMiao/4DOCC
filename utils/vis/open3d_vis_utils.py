import os
import matplotlib as plt
import open3d
import numpy as np
from nuscenes.utils.splits import create_splits_logs


nusc_classname_to_color = {  # RGB.
        "noise": (255, 255, 255),  # White: noise

        "animal": (100, 149, 237),  # Cornflowerblue: movable people or animals or stuff
        "human.pedestrian.adult": (100, 149, 237),
        "human.pedestrian.child": (100, 149, 237),
        "human.pedestrian.construction_worker": (100, 149, 237),
        "human.pedestrian.personal_mobility": (100, 149, 237),
        "human.pedestrian.police_officer": (100, 149, 237),
        "human.pedestrian.stroller": (100, 149, 237),
        "human.pedestrian.wheelchair": (100, 149, 237),
        "movable_object.barrier": (100, 149, 237),
        "movable_object.debris": (100, 149, 237),
        "movable_object.pushable_pullable": (100, 149, 237),
        "movable_object.trafficcone": (100, 149, 237),

        "static_object.bicycle_rack": (0, 207, 191),  # nuTonomy green: static stuff

        "vehicle.bicycle": (255, 127, 80),  # Coral: movable vehicles
        "vehicle.bus.bendy": (255, 127, 80),
        "vehicle.bus.rigid": (255, 127, 80),
        "vehicle.car": (255, 127, 80),
        "vehicle.construction": (255, 127, 80),
        "vehicle.emergency.ambulance": (255, 127, 80),
        "vehicle.emergency.police": (255, 127, 80),
        "vehicle.motorcycle": (255, 127, 80),
        "vehicle.trailer": (255, 127, 80),
        "vehicle.truck": (255, 127, 80),

        "flat.driveable_surface": (0, 207, 191),  # nuTonomy green: static stuff
        "flat.other": (0, 207, 191),
        "flat.sidewalk": (0, 207, 191),
        "flat.terrain": (0, 207, 191),
        "static.manmade": (0, 207, 191),
        "static.other": (0, 207, 191),
        "static.vegetation": (0, 207, 191),

        "vehicle.ego": (255, 127, 80)  # Coral: movable vehicles
    }

my_cmap = plt.cm.get_cmap('tab10')
my_cmap = my_cmap(np.arange(5))[:, :3]
color_first_return = my_cmap[0]
color_second_return = my_cmap[1]

occ_colormap = {
    0: (255 / 255, 255 / 255, 255 / 255),  # unknown: black
    1: (0 / 255, 153 / 255, 41 / 255),  # tp (free -> free): green #009929
    2: (230 / 255, 172 / 255, 0 / 255),  # tn (occ -> occ): yellow #E6AC00
    3: (0 / 255, 163 / 255, 166 / 255),  # fp (free -> occ): cyan #00A3A6
    4: (204 / 255, 92 / 255, 0 / 255),  # fn (occ -> free): orange #CC5C00
}

mos_colormap = {
    0: (255 / 255, 255 / 255, 255 / 255),  # unknown: black
    1: (63 / 255, 79 / 255, 153 / 255),  # tp (sta -> sta): blue #3F4F99
    2: (153 / 255, 50 / 255, 50 / 255),  # tn (mov -> mov): red #993232
    3: (63 / 255, 79 / 255, 153 / 255),  # fp (sta -> mov): blue #3F4F99
    4: (130 / 255, 49 / 255, 153 / 255),  # fn (mov -> sta): purple #823199
}

lidarseg_colormap = {  # RGB.
        0: (0, 0, 0),  # Black.
        1: (70, 130, 180),  # Steelblue
        2: (0, 0, 230),  # Blue
        3: (135, 206, 235),  # Skyblue,
        4: (100, 149, 237),  # Cornflowerblue
        5: (219, 112, 147),  # Palevioletred
        6: (0, 0, 128),  # Navy,
        7: (240, 128, 128),  # Lightcoral
        8: (138, 43, 226),  # Blueviolet
        9: (112, 128, 144),  # Slategrey
        10: (210, 105, 30),  # Chocolate
        11: (105, 105, 105),  # Dimgrey
        12: (47, 79, 79),  # Darkslategrey
        13: (188, 143, 143),  # Rosybrown
        14: (220, 20, 60),  # Crimson
        15: (255, 127, 80),  # Coral
        16: (255, 69, 0),  # Orangered
        17: (255, 158, 0),  # Orange
        18: (233, 150, 70),  # Darksalmon
        19: (255, 83, 0),
        20: (255, 215, 0),  # Gold
        21: (255, 61, 99),  # Red
        22: (255, 140, 0),  # Darkorange
        23: (255, 99, 71),  # Tomato
        24: (0, 207, 191),  # nuTonomy green
        25: (175, 0, 75),
        26: (75, 0, 75),
        27: (112, 180, 60),
        28: (222, 184, 135),  # Burlywood
        29: (255, 228, 196),  # Bisque
        30: (0, 175, 0),  # Green
        31: (255, 240, 245)
    }

occ_color_func = np.vectorize(occ_colormap.get)

mos_color_func = np.vectorize(mos_colormap.get)

def get_confusion_color(gt_labels, pred_labels):
    # confusion status mask
    unk_mask = gt_labels == 0
    minus_labels = gt_labels - pred_labels
    true_mask = np.logical_and(~unk_mask, minus_labels == 0)
    tp_mask = np.logical_and(true_mask, gt_labels == 1)
    tn_mask = np.logical_and(true_mask, gt_labels == 2)
    fp_mask = np.logical_and(~unk_mask, minus_labels == -1)
    fn_mask = np.logical_and(~unk_mask, minus_labels == 1)
    # colors
    color_indices = np.zeros_like(pred_labels)
    color_indices[tp_mask] = 1
    color_indices[tn_mask] = 2
    color_indices[fp_mask] = 3
    color_indices[fn_mask] = 4
    return color_indices

def translate_boxes_to_open3d_instance(gt_boxes):
    """
             4-------- 6
           /|         /|
          5 -------- 3 .
          | |        | |
          . 7 -------- 1
          |/         |/
          2 -------- 0
    """
    center = gt_boxes[0:3]
    lwh = gt_boxes[3:6]
    axis_angles = np.array([0, 0, gt_boxes[6] + 1e-10])
    rot = open3d.geometry.get_rotation_matrix_from_axis_angle(axis_angles)
    box3d = open3d.geometry.OrientedBoundingBox(center, rot, lwh)

    line_set = open3d.geometry.LineSet.create_from_oriented_bounding_box(box3d)

    # import ipdb; ipdb.set_trace(context=20)
    lines = np.asarray(line_set.lines)
    lines = np.concatenate([lines, np.array([[1, 4], [7, 6]])], axis=0)

    line_set.lines = open3d.utility.Vector2iVector(lines)

    return line_set, box3d


def translate_nusc_boxes_to_open3d_instance(gt_boxes):
    """
                 4-------- 6
               /|         /|
              5 -------- 3 .
              | |        | |
              . 7 -------- 1
              |/         |/
              2 -------- 0
        """
    center = gt_boxes.center
    w = gt_boxes.wlh[0]
    l = gt_boxes.wlh[1]
    h = gt_boxes.wlh[2]
    lwh = np.array([l, w, h])
    rot = gt_boxes.rotation_matrix

    box3d = open3d.geometry.OrientedBoundingBox(center, rot, lwh)

    line_set = open3d.geometry.LineSet.create_from_oriented_bounding_box(box3d)

    # import ipdb; ipdb.set_trace(context=20)
    lines = np.asarray(line_set.lines)
    lines = np.concatenate([lines, np.array([[1, 4], [7, 6]])], axis=0)

    line_set.lines = open3d.utility.Vector2iVector(lines)
    return line_set, box3d


def draw_box(vis, boxes: list):
    for box in boxes:
        line_set, box3d = translate_nusc_boxes_to_open3d_instance(box)
        class_name = box.name
        color = nusc_classname_to_color[class_name]
        color = np.array([color[0]/256, color[1]/256, color[2]/256])
        line_set.paint_uniform_color(color)
        vis.add_geometry(line_set)
    return vis


def get_coor_colors(obj_labels):
    """
    Args:
        obj_labels: 1 is ground, labels > 1 indicates different instance cluster

    Returns:
        rgb: [N, 3]. color for each point.
    """
    colors = plt.colors.XKCD_COLORS.values()
    max_color_num = obj_labels.max()

    color_list = list(colors)[:max_color_num+1]
    colors_rgba = [plt.colors.to_rgba_array(color) for color in color_list]
    label_rgba = np.array(colors_rgba)[obj_labels]
    label_rgba = label_rgba.squeeze()[:, :3]

    return label_rgba


######################################### Get Nusc Sample Data Tokens for Vis ##########################################
def split_to_samples(nusc, split_logs):
    sample_tokens = []  # store the sample tokens
    sample_data_tokens = []
    for sample in nusc.sample:
        sample_data_token = sample['data']['LIDAR_TOP']
        scene = nusc.get('scene', sample['scene_token'])
        log = nusc.get('log', scene['log_token'])
        logfile = log['logfile']
        if logfile in split_logs:
            sample_data_tokens.append(sample_data_token)
            sample_tokens.append(sample['token'])
    return sample_tokens, sample_data_tokens


def split_to_samples_singapore(nusc, split_logs):
    sample_tokens = []  # store the sample tokens
    sample_data_tokens = []
    for sample in nusc.sample:
        sample_data_token = sample['data']['LIDAR_TOP']
        scene = nusc.get('scene', sample['scene_token'])
        log = nusc.get('log', scene['log_token'])
        logfile = log['logfile']
        if logfile in split_logs:
            if log["location"].startswith("singapore"):
                sample_data_tokens.append(sample_data_token)
                sample_tokens.append(sample['token'])
    return sample_tokens, sample_data_tokens


def split_to_samples_boston(nusc, split_logs):
    sample_tokens = []  # store the sample tokens
    sample_data_tokens = []
    for sample in nusc.sample:
        sample_data_token = sample['data']['LIDAR_TOP']
        scene = nusc.get('scene', sample['scene_token'])
        log = nusc.get('log', scene['log_token'])
        logfile = log['logfile']
        if logfile in split_logs:
            if log["location"].startswith("boston"):
                sample_data_tokens.append(sample_data_token)
                sample_tokens.append(sample['token'])
    return sample_tokens, sample_data_tokens


def get_vis_sd_toks(nusc, source: str, split: str, bg_label_dir, label_suffix: str):
    if source == 'all':
        split_logs = create_splits_logs(split, nusc)
        sample_tokens, sd_toks_list = split_to_samples(nusc, split_logs)

        # list predicted labels to get all valid sd tokens
        mos_pred_file_list = os.listdir(bg_label_dir)
        for i in range(len(mos_pred_file_list)):
            mos_pred_file_list[i] = mos_pred_file_list[i].replace(label_suffix, '')
        valid_sd_toks_list = mos_pred_file_list

        # all valid sd tokens, and remain the order
        sd_toks_list = [sd_tok for sd_tok in sd_toks_list if sd_tok in valid_sd_toks_list]
    elif source == 'singapore':
        split_logs = create_splits_logs(split, nusc)
        sample_tokens, sd_toks_list = split_to_samples_singapore(nusc, split_logs)

        # list predicted labels to get all valid sd tokens
        mos_pred_file_list = os.listdir(bg_label_dir)
        for i in range(len(mos_pred_file_list)):
            mos_pred_file_list[i] = mos_pred_file_list[i].replace(label_suffix, '')
        valid_sd_toks_list = mos_pred_file_list

        # all valid sd tokens, and remain the order
        sd_toks_list = [sd_tok for sd_tok in sd_toks_list if sd_tok in valid_sd_toks_list]
    elif source == 'boston':
        split_logs = create_splits_logs(split, nusc)
        sample_tokens, sd_toks_list = split_to_samples_boston(nusc, split_logs)

        # list predicted labels to get all valid sd tokens
        mos_pred_file_list = os.listdir(bg_label_dir)
        for i in range(len(mos_pred_file_list)):
            mos_pred_file_list[i] = mos_pred_file_list[i].replace(label_suffix, '')
        valid_sd_toks_list = mos_pred_file_list

        # all valid sd tokens, and remain the order
        sd_toks_list = [sd_tok for sd_tok in sd_toks_list if sd_tok in valid_sd_toks_list]
    elif source == 'given':
        sd_toks_list = [
            '8d629893d7424fed9568d55ec97e495a',
            '5719a3aad77a4243887b631b9748fc48',
            'db164fdd8aab4e61aa6d5287a585e821',
            'd1fda35ed35f43bdb6a87656d7469449',
            '42aaba29d851496abb5af7109650b8dc',
            '2a3303da6f86442692342764bc5df7a4',

            '67e2205659b249ac85c2288faa487e01',
            'e4a9c18ebfb049bbb40a24a5653752f2',
            '43ebc6eacd1d4b7694102ed6b72ef07b',
            'f9d5160a4bda45929c70eb8a7ce2252c',
            'e76fdb2eddd1451f893dea97741c4966',
            '5f8393250fae4960b501cb6055614547',
            'ccc5156611314b489b52e831b3fbfdd4',
        ]  # TODO: load txt file
    else:
        sd_toks_list = None
    return sd_toks_list