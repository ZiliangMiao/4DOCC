import os
import math
import argparse
from dataclasses import asdict
import matplotlib.pyplot as plt
import numpy as np
from sympy import print_tree
from tqdm import tqdm
import multiprocessing
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.geometry_utils import points_in_box
from nuscenes.utils.data_io import load_bin_file
from nuscenes.utils.data_classes import LidarPointCloud, LidarSegPointCloud


def generate_mos_labels(sample, nusc):
    sd_tok = sample['data']['LIDAR_TOP']
    sample_data = nusc.get('sample_data', sd_tok)
    lidar_pcd = LidarPointCloud.from_file(os.path.join(nusc.dataroot, sample_data['filename'])).points[:3, :]

    # get lidarseg labels
    lidarseg_file = os.path.join(nusc.dataroot, nusc.get('lidarseg', sd_tok)['filename'])
    lidarseg_labels = load_bin_file(lidarseg_file, 'lidarseg')  # load lidarseg label

    # initialize mos labels, 0: unknown; 1: static; 2: moving
    mos_labels = np.zeros_like(lidarseg_labels, dtype=np.uint8)

    # speed thresholds
    veh_spd_lb = 0.5  # vehicle speed lower bound (1.8 km/h)
    veh_spd_ub = 1  # vehicle speed upper bound (3.6 km/h)

    hum_spd_lb = 0.375  # human speed lower bound (1.35 km/h)
    hum_spd_ub = 0.6  # human speed upper bound (2.16 km/h)

    cyc_spd_lb = 0.375  # cycle speed lower bound (1.35 km/h)
    cyc_spd_ub = 1  # cycle speed upper bound (3.6 km/h)

    ########################################## static objects and noise ################################################
    # 30: static.vegetation, 29: static.other, 28: static.manmade
    # 27: flat.driveable_surface, 26: flat.terrain, 25: flat.sidewalk, 24: flat.other
    # 13: static_object.bicycle_rack
    # 0: noise
    stat_mask = np.isin(lidarseg_labels, [30, 29, 28, 27, 26, 25, 24, 13])
    mos_labels[stat_mask] = 1
    noise_mask = lidarseg_labels == 0
    mos_labels[noise_mask] = 0
    ####################################################################################################################

    ############################################### movable objects ####################################################
    # speed of ego vehicle
    ego_pose_tok = sample_data['ego_pose_token']
    ego_pos_curr = np.array(nusc.get('ego_pose', ego_pose_tok)['translation'])

    # calculate ego vehicle speed by current and the next (or prev) ego_pose
    t_curr = sample_data['timestamp']
    if sample_data['prev'] != '':
        sd_tok_adja = sample_data['prev']
    else:
        sd_tok_adja = sample_data['next']
    sample_data_adja = nusc.get('sample_data', sd_tok_adja)

    # ego vehicle speed -> mos labels
    t_adja = sample_data_adja['timestamp']
    t_rela = np.abs(t_curr - t_adja) / 1000000
    ego_pos_adja = np.array(nusc.get('ego_pose', sample_data_adja['ego_pose_token'])['translation'])
    pos_rela = ego_pos_curr - ego_pos_adja
    ego_spd = np.linalg.norm(pos_rela / t_rela)
    ego_mask = lidarseg_labels == 31
    if ego_spd > veh_spd_ub:
        mos_labels[ego_mask] = 2
    elif ego_spd < veh_spd_lb:
        mos_labels[ego_mask] = 1
    else:
        mos_labels[ego_mask] = 0

    # speed of object bboxes
    for ann_token in sample['anns']:
        # box
        ann = nusc.get('sample_annotation', ann_token)
        _, box, _ = nusc.get_sample_data(sd_tok, selected_anntokens=[ann_token], use_flat_vehicle_coordinates=False)
        box = box[0]
        box_label = name2idx_mapping[ann['category_name']]
        box_spd = np.linalg.norm(nusc.box_velocity(box.token))

        # lidar points inside the box (with consistent label)
        pts_in_box_mask = points_in_box(box, lidar_pcd)
        consistent_mask = lidarseg_labels == box_label

        # object attribute
        has_attr = len(ann["attribute_tokens"]) > 0
        if has_attr:
            attr = nusc.get("attribute", ann["attribute_tokens"][0])["name"]
            # spd_file.write(f"{attr} {box_spd}\n")
        else:
            attr = None

        # vehicles
        # 23: vehicle.truck, 22: vehicle.trailer, 20: vehicle.emergency.police,
        # 19: vehicle.emergency.ambulance, 18: vehicle.construction, 17: vehicle.car, 16: vehicle.bus.rigid,
        # 15: vehicle.bus.bendy]
        # criterion-1: movable semantic classes
        if box_label in [23, 22, 20, 19, 18, 17, 16, 15]:
            # criterion-2: object attribute
            if attr == "vehicle.moving":
                # criterion-3: speed threshold
                if box_spd > veh_spd_ub:
                    mos_labels[np.logical_and(pts_in_box_mask, consistent_mask)] = 2
                else:
                    mos_labels[np.logical_and(pts_in_box_mask, consistent_mask)] = 0  # todo: criterion-2 and 3 not consistent
            elif attr in ["vehicle.stopped", "vehicle.parked"]: # todo: must be static
                if box_spd < veh_spd_lb:
                    mos_labels[np.logical_and(pts_in_box_mask, consistent_mask)] = 1
                else:
                    mos_labels[np.logical_and(pts_in_box_mask, consistent_mask)] = 0
            else: # todo: no attribute, only criterion-3
                if box_spd > veh_spd_ub:
                    mos_labels[np.logical_and(pts_in_box_mask, consistent_mask)] = 2
                elif box_spd < veh_spd_lb:
                    mos_labels[np.logical_and(pts_in_box_mask, consistent_mask)] = 1
                else:
                    mos_labels[np.logical_and(pts_in_box_mask, consistent_mask)] = 0

        # motor-cycle, 21: vehicle.motorcycle
        # bi-cycle, 14: vehicle.bicycle
        elif box_label in [21, 14]:
            if attr == "cycle.without_rider":
                if box_spd < cyc_spd_lb:
                    mos_labels[np.logical_and(pts_in_box_mask, consistent_mask)] = 1
                else:
                    mos_labels[np.logical_and(pts_in_box_mask, consistent_mask)] = 0
            else:
                if box_spd > cyc_spd_ub:
                    mos_labels[np.logical_and(pts_in_box_mask, consistent_mask)] = 2
                elif box_spd < cyc_spd_lb:
                    mos_labels[np.logical_and(pts_in_box_mask, consistent_mask)] = 1
                else:
                    mos_labels[np.logical_and(pts_in_box_mask, consistent_mask)] = 0

        # human
        # 8: human.pedestrian.wheelchair, 7: human.pedestrian.stroller, 6: human.pedestrian.police_officer
        # 5: human.pedestrian.personal_mobility, 4: human.pedestrian.construction_worker, 3: human.pedestrian.child, 2: human.pedestrian.adult
        elif box_label in [8, 7, 6, 5, 4, 3, 2]:
            if attr == "pedestrian.moving":
                if box_spd > hum_spd_ub:
                    mos_labels[np.logical_and(pts_in_box_mask, consistent_mask)] = 2
                else:
                    mos_labels[np.logical_and(pts_in_box_mask, consistent_mask)] = 0
            elif attr in ["pedestrian.standing", "pedestrian.sitting_lying_down"]:
                if box_spd < hum_spd_lb:
                    mos_labels[np.logical_and(pts_in_box_mask, consistent_mask)] = 1
                else:
                    mos_labels[np.logical_and(pts_in_box_mask, consistent_mask)] = 0
            else:
                if box_spd > hum_spd_ub:
                    mos_labels[np.logical_and(pts_in_box_mask, consistent_mask)] = 2
                elif box_spd < hum_spd_lb:
                    mos_labels[np.logical_and(pts_in_box_mask, consistent_mask)] = 1
                else:
                    mos_labels[np.logical_and(pts_in_box_mask, consistent_mask)] = 0

        # movable objects
        # 12: movable_object.trafficcone, 11: movable_object.pushable_pullable, 10: movable_object.debris, 9: movable_object.barrier, 1: animal
        elif box_label in [12, 11, 10, 9, 1]:
            if box_spd > hum_spd_ub:
                mos_labels[np.logical_and(pts_in_box_mask, consistent_mask)] = 2
            elif box_spd < hum_spd_lb:
                mos_labels[np.logical_and(pts_in_box_mask, consistent_mask)] = 1
            else:
                mos_labels[np.logical_and(pts_in_box_mask, consistent_mask)] = 0
    ####################################################################################################################

    # save mos labels
    mos_dir = os.path.join(nusc.dataroot, "mos_labels", nusc.version)
    os.makedirs(mos_dir, exist_ok=True)
    mos_file = os.path.join(mos_dir, sd_tok + "_mos.label")
    mos_labels.tofile(mos_file)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate nuScenes MOS labels.')
    parser.add_argument('--root_dir', type=str, default='/home/ziliang/Datasets_0/nuScenes')
    parser.add_argument('--version', type=str, default='v1.0-trainval')
    parser.add_argument('--verbose', type=bool, default=True)
    args = parser.parse_args()

    print(f'Start mos label generation... \n Arguments: {args}')
    nusc = NuScenes(version=args.version, dataroot=args.root_dir, verbose=args.verbose)
    if not hasattr(nusc, "lidarseg") or len(getattr(nusc, 'lidarseg')) == 0:
        raise RuntimeError(f"No nuscenes-lidarseg annotations found in {nusc.version}")
    name2idx_mapping = nusc.lidarseg_name2idx_mapping  # idx2name_mapping = nusc.lidarseg_idx2name_mapping
    print(f'There are {len(nusc.sample)} samples.')

    # with open('box_speed.txt', 'w') as spd_file:
    for sample in tqdm(nusc.sample):
        generate_mos_labels(sample, nusc)
    print('MOS label generation finished.')

    # # speed statistics of objects with attributes
    # veh_mov_spd_list = []
    # veh_sta_spd_list = []
    # hum_mov_spd_list = []
    # hum_sta_spd_list = []
    # cyc_w_spd_list = []
    # cyc_wo_spd_list = []
    # with open('box_speed.txt', 'r') as f:
    #     for line in f:
    #         if line.strip(): # skip the empty line
    #             attr, spd = line.strip().split()
    #             if attr == "vehicle.moving":
    #                 veh_mov_spd_list.append(float(spd))
    #             elif attr in  ["vehicle.stopped", "vehicle.parked"]:
    #                 veh_sta_spd_list.append(float(spd))
    #             elif attr == "pedestrian.moving":
    #                 hum_mov_spd_list.append(float(spd))
    #             elif attr in ["pedestrian.standing", "pedestrian.sitting_lying_down"]:
    #                 hum_sta_spd_list.append(float(spd))
    #             elif attr == "cycle.with_rider":
    #                 cyc_w_spd_list.append(float(spd))
    #             elif attr == "cycle.without_rider":
    #                 cyc_wo_spd_list.append(float(spd))
    #
    # fig, axs = plt.subplots(2, 3, figsize=(30, 20))
    # fig.suptitle('speed distribution of objects with different attributes')
    #
    # axs[0, 0].hist(veh_mov_spd_list, bins=100, range=(0,5), log=False, color='skyblue', alpha=0.9)
    # axs[0, 0].set_title('moving vehicles')
    # axs[0, 0].set_xlabel('speed (m/s)')
    # axs[0, 0].set_ylabel('frequency')
    # axs[0, 0].grid(True, alpha=0.3)
    #
    # axs[0, 1].hist(veh_sta_spd_list, bins=100, range=(0,2), log=False, color='lightgreen', alpha=0.9)
    # axs[0, 1].set_title('static vehicles')
    # axs[0, 1].set_xlabel('speed (m/s)')
    # axs[0, 1].set_ylabel('frequency')
    # axs[0, 1].grid(True, alpha=0.3)
    #
    # axs[0, 2].hist(hum_mov_spd_list, bins=100, range=(0,1), log=False, color='salmon', alpha=0.9)
    # axs[0, 2].set_title('moving humans')
    # axs[0, 2].set_xlabel('speed (m/s)')
    # axs[0, 2].set_ylabel('frequency')
    # axs[0, 2].grid(True, alpha=0.3)
    #
    # axs[1, 0].hist(hum_sta_spd_list, bins=100, range=(0,2), log=False, color='purple', alpha=0.9)
    # axs[1, 0].set_title('static humans')
    # axs[1, 0].set_xlabel('speed (m/s)')
    # axs[1, 0].set_ylabel('frequency')
    # axs[1, 0].grid(True, alpha=0.3)
    #
    # axs[1, 1].hist(cyc_w_spd_list, bins=100, range=(0,2), log=False, color='orange', alpha=0.9)
    # axs[1, 1].set_title('cycle with rider')
    # axs[1, 1].set_xlabel('speed (m/s)')
    # axs[1, 1].set_ylabel('frequency')
    # axs[1, 1].grid(True, alpha=0.3)
    #
    # axs[1, 2].hist(cyc_wo_spd_list, bins=100, range=(0,2), log=False, color='brown', alpha=0.9)
    # axs[1, 2].set_title('cycle without rider')
    # axs[1, 2].set_xlabel('speed (m/s)')
    # axs[1, 2].set_ylabel('frequency')
    # axs[1, 2].grid(True, alpha=0.3)
    #
    # plt.tight_layout()
    #
    # plt.savefig('category_velocity_distributions_min.png', dpi=500)
    # plt.close()


