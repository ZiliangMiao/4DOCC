import torch
import torch.nn as nn

import time
from tqdm import tqdm
import numpy as np

from .rays import RayBundle, RaySamples
from .ray_samplers import UniformSampler
from .scene_colliders import AABBBoxCollider as SceneCollider
from .density_field import DensityField
from .rays import Rays


# memory debug
# from lib.pytorch_memory_track.gpu_mem_track import MemTracker
# gpu_tracker = MemTracker()
# ######### memory tracker #########
# gpu_tracker.track()
# ##################################

class DepthRenderer(nn.Module):
    """Calculate depth along ray."""

    def __init__(self, **kwargs):
        super().__init__()

    def forward(self, ray_samples, weights):
        """
        Composite samples along ray and calculate depths.

        Args:
            weights: Weights for each sample. [N, K, 1]
            ray_samples: Set of ray samples.
        Returns:
            depth: volume rendered depth value [N, 1]
        """
        eps = 1e-10
        # steps = (ray_samples.frustums.starts + ray_samples.frustums.ends) / 2
        # TODO: why starts, not (stars + ends) / 2
        steps = ray_samples.frustums.starts
        depth = torch.sum(weights * steps, dim=-2) / (torch.sum(weights, -2) + eps)
        depth = torch.clip(depth, steps.min(), steps.max())
        return depth


class OcclusionRenderer():
    def __init__(self):
        a = 1


class FutureDenseConv3D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, stride=1):
        super(FutureDenseConv3D, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                padding=padding,
                stride=stride,
            ),
            # nn.BatchNorm3d(out_channels),
            # nn.ReLU(inplace=True),
            # nn.Conv3d(
            #     in_channels,
            #     out_channels,
            #     kernel_size=kernel_size,
            #     padding=padding,
            #     stride=stride,
            # ),
        )

    def forward(self, x):
        out = self.conv(x)
        return out


class OcclusionDecoder(nn.Module):
    def __init__(self, cfg, feat_volume_shape):
        super().__init__()
        # feature grid
        self.scene_bbox = cfg["data"]["scene_bbox"]
        self.feat_volume_shape = feat_volume_shape
        [B, F, T, Z, Y, X] = self.feat_volume_shape
        self.future_dense_conv = FutureDenseConv3D(in_channels=F * Z, out_channels=F * Z)

        # rays
        self.n_rays_scan = cfg["data"]["n_rays_scan"]
        self.rays = Rays(n_rays_scan=self.n_rays_scan, scene_bbox=self.scene_bbox)

        # sampler
        self.n_points_ray = cfg["data"]["n_points_ray"]
        self.ray_sampler = UniformSampler(num_samples=self.n_points_ray, train_stratified=True, single_jitter=False)

        # density field
        self.density_field = DensityField(feat_dim=F)

        # renderer
        self.depth_renderer = DepthRenderer()
        self.occlusion_renderer = OcclusionRenderer()

    def forward(self, batch, curr_feat_volume):
        """
        Occlusion Decoder

        Args:
            ray_dict: dict includes "ray_start" "ray_end" "ray_direction" "ray_depth", normalized to scene,
                      under reference lidar coordinate frame
            curr_feat_volume: shape [B, F, T, Z, Y, X]

        Returns:
            rendered_depth:
            rendered_occlusion_depth:

        """
        [B, F, T, Z, Y, X] = self.feat_volume_shape
        # TODO: check whether F * Z is better than F * T
        # feature volume: B, F, T, Z, Y, X -> permute and reshape to [B, F * Z, T, Y, X]
        curr_feat_volume = curr_feat_volume.permute(0, 1, 3, 2, 4, 5).contiguous().reshape(B, F * Z, T, Y, X)
        future_feat_volume = self.future_dense_conv(curr_feat_volume)
        # TODO: feature volume at time t for grid_sample, [B, T (t) = 1, F, Z, Y, X]
        future_feat_volume = future_feat_volume.reshape(B, F, Z, T, Y, X).permute(0, 3, 1, 2, 4, 5).contiguous()

        # get rays from ray_dict
        # TODO: rays have been normalize to 0-1, why?
        ray_dict = self.rays(batch=batch, feat_volume_shape=self.feat_volume_shape)

        rendered_depth_list, rendered_occlusion_list = [], []
        for time_idx in range(T):
            rays_o_t = ray_dict["ray_start"][time_idx]
            rays_d_t = ray_dict["ray_direction"][time_idx]

            # initialize ray_bundle from ray_dict, and get ray nears and fars
            ray_bundle = RayBundle(origins=rays_o_t, directions=rays_d_t)
            scene_collider = SceneCollider(bbox=[0.0, 0.0, 0.0, 1.0, 1.0, 1.0], near_plane=0.01)
            ray_bundle = scene_collider.set_nears_and_fars(ray_bundle)

            # sample points on the rays
            ray_samples = self.ray_sampler.generate_ray_samples(ray_bundle)

            # get density from density MLP field
            densities = self.density_field(ray_samples, future_feat_volume[0][time_idx])

            # get weights from densities
            weights, transmittance = ray_samples.get_weights_and_transmittance(densities)

            # volume rendering: get depth from ray_samples
            rendered_depth = self.depth_renderer(ray_samples=ray_samples, weights=weights)

            # TODO: occlusion depth rendering
            rendered_occlusion_depth = 0.0

            # append to list
            rendered_depth_list.append(rendered_depth)
            rendered_occlusion_list.append(rendered_occlusion_depth)

        rendered_dict = dict(
            depth=rendered_depth_list,
            occlusion=rendered_occlusion_list,
            gt_depth=ray_dict["ray_depth"],
            gt_occlusion=ray_dict["ray_occlusion"]
        )
        return rendered_dict