import torch
import torch.nn as nn

# by ziliang
class Rays(object):
    def __init__(self, n_rays_scan, scene_bbox):
        self.n_rays_scan = n_rays_scan  # randomly subsample a certain number of rays in each scan
        self.scene_bbox = scene_bbox  # scene bounding box, meters


    def ray_normalize_to_scene(self, coord, scene_bbox):
        scene_bbox = torch.tensor(scene_bbox).to(coord)
        norm_coord = (coord - scene_bbox[:3]) / (scene_bbox[3:] - scene_bbox[:3])
        return norm_coord


    def __call__(self, batch, feat_volume_shape):
        [B, F, T, Z, Y, X] = feat_volume_shape
        (meta_info, in_points_4d, out_origin, out_points_4d) = batch

        # get ray_start, ray_end, ray_direction and ray_depth
        ray_start, ray_end, ray_direction, ray_depth, ray_occlusion = [], [], [], [], []
        for b in range(B):
            origin = out_origin[b]
            points_4d = in_points_4d[b]
            for t in range(T):
                # time mask
                origin_t = origin[t]
                t_mask = points_4d[:, -1] == t

                # get rays
                ray_end_t = points_4d[t_mask][:, :-1]
                if self.n_rays_scan > 0:  # random ray subsample
                    weights = torch.ones(len(ray_end_t))
                    valid_ray_idx = torch.multinomial(weights, num_samples=self.n_rays_scan, replacement=False)
                    ray_end_t = ray_end_t[valid_ray_idx]
                ray_start_t = torch.zeros_like(ray_end_t) + origin_t

                # scene normalization
                ray_start_t = self.ray_normalize_to_scene(ray_start_t, self.scene_bbox)
                ray_end_t = self.ray_normalize_to_scene(ray_end_t, self.scene_bbox)

                # ray direction and depth
                ray_direction_t = nn.functional.normalize(ray_end_t - ray_start_t, p=2, dim=1)
                ray_depth_t = torch.linalg.norm(ray_end_t - ray_start_t, dim=1, keepdim=True)

                # TODO: get ray occlusion depth
                ray_occlusion_t = torch.zeros_like(ray_depth_t)

                # append to batch ray list
                ray_start.append(ray_start_t)
                ray_end.append(ray_end_t)
                ray_direction.append(ray_direction_t)
                ray_depth.append(ray_depth_t)
                ray_occlusion.append(ray_occlusion_t)

        # ray return dictionary
        ray_dict = dict(ray_start=ray_start,
                        ray_end=ray_end,
                        ray_direction=ray_direction,
                        ray_depth=ray_depth,
                        ray_occlusion=ray_occlusion)
        return ray_dict


class Frustums(nn.Module):
    def __init__(self, origins, directions, starts, ends, **kwargs):
        super().__init__()
        self.origins = origins
        self.directions = directions
        self.starts = starts
        self.ends = ends

    def get_positions(self):
        """Calulates "center" position of frustum. Not weighted by mass.

        Returns:
            xyz positions: (num_rays, num_samples, 3)
        """
        pos = self.origins + self.directions * (self.starts + self.ends) / 2
        return pos

    def get_start_positions(self):
        """Calulates "start" position of frustum. We use start positions for MonoSDF
        because when we use error bounded sampling, we need to upsample many times.
        It's hard to merge two set of ray samples while keeping the mid points fixed.
        Every time we up sample the points the mid points will change and
        therefore we need to evaluate all points again which is 3 times slower.
        But we can skip the evaluation of sdf value if we use start position instead of mid position
        because after we merge the points, the starting point is the same and only the delta is changed.

        Returns:
            xyz positions: (num_rays, num_samples, 3)
        """
        return self.origins + self.directions * self.starts


class RaySamples(nn.Module):
    """Samples along a ray"""

    def __init__(
        self,
        frustums,
        deltas,
        spacing_starts,
        spacing_ends,
        spacing_to_euclidean_fn,
        **kwargs
    ):
        super().__init__()
        self.frustums = frustums
        self.deltas = deltas
        self.spacing_starts = spacing_starts
        self.spacing_ends = spacing_ends
        self.spacing_to_euclidean_fn = spacing_to_euclidean_fn

    def get_weights_and_transmittance(self, densities):
        """
        Return weights based on predicted densities

        Args:
            densities: predicted densities of each point [N, K, 1]

        Returns:
            weights: weights of each point [N, K, 1]
        """
        delta_density = self.deltas * densities
        alphas = 1 - torch.exp(-delta_density)

        transmittance = torch.cumsum(delta_density[..., :-1, :], dim=-2)
        transmittance = torch.cat(
            [
                torch.zeros((*transmittance.shape[:1], 1, 1), device=densities.device),
                transmittance,
            ],
            dim=-2,
        )
        transmittance = torch.exp(-transmittance)  # [..., "num_samples"]

        weights = alphas * transmittance  # [..., "num_samples"]

        return weights, transmittance

    def get_weights_and_transmittance_from_alphas(self, alphas):
        """Return weights based on predicted alphas

        Args:
            alphas: Predicted alphas (maybe from sdf) for samples along ray

        Returns:
            Weights for each sample
        """
        transmittance = torch.cumprod(
            torch.cat(
                [
                    torch.ones((*alphas.shape[:1], 1, 1), device=alphas.device),
                    1.0 - alphas + 1e-7,
                ],
                1,
            ),
            1,
        )  # [..., "num_samples"]

        weights = alphas * transmittance[:, :-1, :]  # [num_rays, num_samples, 1]

        return weights, transmittance


class RayBundle(nn.Module):
    """A bundle of ray parameters."""

    def __init__(self, origins, directions, nears=None, fars=None, **kwargs):
        super().__init__(**kwargs)
        self.origins = origins  # (num_rays, 3)
        self.directions = directions  # (num_rays, 3)
        self.nears = nears  # (num_rays, 1)
        self.fars = fars  # (num_rays, 1)

    def merge_ray_samples(self, ray_samples_1, ray_samples_2):
        """Merge two set of ray samples and return sorted index which can be used to merge sdf values

        Args:
            ray_samples_1 : ray_samples to merge
            ray_samples_2 : ray_samples to merge
        """

        starts_1 = ray_samples_1.spacing_starts[..., 0]
        starts_2 = ray_samples_2.spacing_starts[..., 0]

        ends = torch.maximum(
            ray_samples_1.spacing_ends[..., -1:, 0],
            ray_samples_2.spacing_ends[..., -1:, 0],
        )

        bins, sorted_index = torch.sort(torch.cat([starts_1, starts_2], -1), -1)

        bins = torch.cat([bins, ends], dim=-1)

        # Stop gradients
        bins = bins.detach()

        euclidean_bins = ray_samples_1.spacing_to_euclidean_fn(bins)

        ray_samples = self.get_ray_samples(
            bin_starts=euclidean_bins[
                ..., :-1, None
            ],  # (num_rays, num_samples + num_importance, 1)
            bin_ends=euclidean_bins[..., 1:, None],
            spacing_starts=bins[..., :-1, None],
            spacing_ends=bins[..., 1:, None],
            spacing_to_euclidean_fn=ray_samples_1.spacing_to_euclidean_fn,
        )

        return ray_samples, sorted_index

    def merge_ray_samples_in_eculidean(self, ray_samples_1, ray_samples_2):
        """Merge two set of ray samples and return sorted index which can be used to merge sdf values

        Args:
            ray_samples_1 : ray_samples to merge
            ray_samples_2 : ray_samples to merge
        """
        starts_1 = ray_samples_1.frustums.starts[..., 0]
        starts_2 = ray_samples_2.frustums.starts[..., 0]

        end_1 = ray_samples_1.frustums.ends[:, -1:, 0]
        end_2 = ray_samples_2.frustums.ends[:, -1:, 0]

        end = torch.maximum(end_1, end_2)

        euclidean_bins, _ = torch.sort(torch.cat([starts_1, starts_2], -1), -1)

        euclidean_bins = torch.cat([euclidean_bins, end], dim=-1)

        # Stop gradients
        euclidean_bins = euclidean_bins.detach()

        # TODO convert euclidean bins to spacing bins
        bins = euclidean_bins

        ray_samples = self.get_ray_samples(
            bin_starts=euclidean_bins[..., :-1, None],
            bin_ends=euclidean_bins[..., 1:, None],
            spacing_starts=None,
            spacing_ends=None,
            spacing_to_euclidean_fn=None,  # near and far are different
        )

        return ray_samples

    def get_ray_samples(
        self,
        bin_starts,
        bin_ends,
        spacing_starts,
        spacing_ends,
        spacing_to_euclidean_fn,
    ):
        """Produces samples for each ray by projection points along the ray direction. Currently samples uniformly.

        Args:
            bin_starts: Distance from origin to start of bin.
            bin_ends: Distance from origin to end of bin.

        Returns:
            Samples projected along ray.
        """
        deltas = bin_ends - bin_starts
        broadcast_size = [*deltas.shape[:-1], -1]
        frustums = Frustums(
            origins=self.origins[..., None, :].expand(
                broadcast_size
            ),  # (num_rays, num_samples, 3)
            directions=self.directions[..., None, :].expand(
                broadcast_size
            ),  # (num_rays, num_samples, 3)
            starts=bin_starts,  # (num_rays, num_samples, 1)
            ends=bin_ends,
        )
        ray_samples = RaySamples(
            frustums=frustums,
            deltas=deltas,  # [..., num_samples, 1]
            spacing_starts=spacing_starts,  # [..., num_samples, 1]
            spacing_ends=spacing_ends,  # [..., num_samples, 1]
            spacing_to_euclidean_fn=spacing_to_euclidean_fn,
        )

        return ray_samples