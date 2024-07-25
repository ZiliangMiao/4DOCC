import torch
import torch.nn.functional as F
from smooth_sampler import SmoothSampler
from torch import nn

class DensityMLP(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_size, n_blocks, points_factor=1.0, **kwargs):
        super().__init__()

        dims = [hidden_size] + [hidden_size for _ in range(n_blocks)] + [out_dim]
        self.num_layers = len(dims)

        for l in range(self.num_layers - 1):
            lin = nn.Linear(dims[l], dims[l + 1])
            setattr(self, "lin" + str(l), lin)

        self.fc_c = nn.ModuleList(
            [nn.Linear(in_dim, hidden_size) for i in range(self.num_layers - 1)]
        )
        self.fc_p = nn.Linear(3, hidden_size)

        self.activation = nn.Softplus(beta=100)  # TODO: what is soft plus

        self.points_factor = points_factor

    def forward(self, points, point_feats):
        x = self.fc_p(points) * self.points_factor
        for l in range(self.num_layers - 1):
            x = x + self.fc_c[l](point_feats)
            lin = getattr(self, "lin" + str(l))
            x = lin(x)
            if l < self.num_layers - 2:
                x = self.activation(x)
        return x


class DensityField(nn.Module):
    def __init__(
        self,
        feat_dim,
        padding_mode="zeros",
        share_volume=True,
    ):
        super().__init__()
        self.feat_dim = feat_dim
        self.padding_mode = padding_mode
        self.share_volume = share_volume
        self.density_mlp = DensityMLP(in_dim=self.feat_dim, out_dim=self.feat_dim + 1, hidden_size=16, n_blocks=5)


    def get_occupancy_from_density(self, density):
        """
        Compute occupancy from density value

        Args:
            density: predicted density value

        Returns:

        """
        occupancy = torch.sigmoid(-10.0 * density)
        return occupancy


    def feature_sampling(self, points_norm, feat_volume):
        """
        Tri-linearly query points' features along rays

        Args:
            points_norm: (N, K, 3), [x, y, z], scaled in 0-1
            feat_volume: (C, Z, Y, X) at a specific time t

        Returns:
            feats: (N, K, C)
        """
        points_norm = points_norm * 2 - 1  # re-scale range from [0, 1] -> [-1, 1]
        # TODO: torch grid_sample, input [N, C, D_in, H_in, W_in], grid [N, D_out, H_out, W_out, 3]
        feat_volume = feat_volume.unsqueeze(0).to(points_norm.dtype)
        points_norm = points_norm[None, None]
        queried_feats = SmoothSampler.apply(feat_volume, points_norm, self.padding_mode, True, False)
        ret_feats = queried_feats.squeeze(0).squeeze(1).permute(1, 2, 0)  # (1, C, 1, N, K) -> (N, K, C)
        return ret_feats


    def forward(self, ray_samples, volume_feature):
        """
        Predict the density value for ray samples

        Args:
            points:
            volume_feature:

        Returns:
            (MLP outputs dim = 1 + feat_dim)
            density: dim=1, the first digit of MLP output
            output_feats: dim=feat_dim, the other digits of MLP output
            point_feats: tri-linear queried features
        """
        points = ray_samples.frustums.get_start_positions()
        points.requires_grad_(True)
        with torch.enable_grad():
            point_feats = self.feature_sampling(points, volume_feature)
            mlp_output = self.density_mlp(
                points,
                (point_feats if self.share_volume else torch.chunk(point_feats, 2, dim=-1)[0]),
            )
            density, output_feats = mlp_output[..., :1], mlp_output[..., 1:]

        ### Others ###
        # output_dict = {}
        # d_output = torch.ones_like(density, requires_grad=False, device=density.device)
        # gradients = torch.autograd.grad(
        #     outputs=density,
        #     inputs=points,
        #     grad_outputs=d_output,
        #     create_graph=True,
        #     retain_graph=True,
        #     only_inputs=True,
        # )[0]
        # directions = ray_samples.frustums.directions  # (num_rays, num_samples, 3)
        return density

# class LaplaceDensity(nn.Module):  # alpha * Laplace(loc=0, scale=beta).cdf(-sdf)
#     """Laplace density from VolSDF, convert sdf to density"""
#
#     def __init__(self, init_val, beta_min=0.0001):
#         super().__init__()
#         self.register_parameter(
#             "beta_min", nn.Parameter(beta_min * torch.ones(1), requires_grad=False)
#         )
#         self.register_parameter(
#             "beta", nn.Parameter(init_val * torch.ones(1), requires_grad=True)
#         )
#
#     def forward(self, sdf, beta=None):
#         """convert sdf value to density value with beta, if beta is missing, then use learable beta"""
#         if beta is None:
#             beta = self.get_beta()
#
#         alpha = 1.0 / beta
#
#         density = alpha * (0.5 + 0.5 * sdf.sign() * torch.expm1(-sdf.abs() / beta))
#         return density
#
#     def get_beta(self):
#         """return current beta value"""
#         beta = self.beta.abs() + self.beta_min
#         return beta
#
#
# class SingleVarianceNetwork(nn.Module):
#     """Variance network in NeuS"""
#
#     def __init__(self, init_val):
#         super(SingleVarianceNetwork, self).__init__()
#         self.register_parameter(
#             "variance", nn.Parameter(init_val * torch.ones(1), requires_grad=True)
#         )
#
#     def forward(self, x):
#         """Returns current variance value"""
#         return torch.ones([len(x), 1], device=x.device) * torch.exp(
#             self.variance * 10.0
#         )
#
#     def get_variance(self):
#         """return current variance value"""
#         return torch.exp(self.variance * 10.0).clip(1e-6, 1e6)