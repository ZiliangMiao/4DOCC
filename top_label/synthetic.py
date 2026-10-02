"""Synthetic lidar scans for tests and benchmarks (analytic ray casting, no dataset needed).

Scene: ground plane z = -ground_h, a vertical cylinder wall of radius wall_r around the world
origin and a few axis-aligned boxes. Sensors move along +x without rotation, so the scans of
all frames are already aligned; everything is returned in the current frame (current sensor
at the origin).
"""
from __future__ import annotations

import math
from typing import List, Tuple

import torch


def _ray_cast(org: torch.Tensor, d: torch.Tensor, boxes: torch.Tensor, ground_h: float, wall_r: float) -> torch.Tensor:
    n = d.shape[0]
    t_best = torch.full((n,), math.inf, dtype=d.dtype)
    # ground plane
    dz = d[:, 2]
    t = torch.where(dz < -1e-9, (-ground_h - org[2]) / torch.where(dz < -1e-9, dz, -torch.ones_like(dz)),
                    torch.full_like(dz, math.inf))
    t_best = torch.minimum(t_best, torch.where(t > 0, t, torch.full_like(t, math.inf)))
    # cylinder wall x^2 + y^2 = R^2 (sensor inside, take the positive root)
    a2 = d[:, 0] ** 2 + d[:, 1] ** 2
    b2 = 2 * (org[0] * d[:, 0] + org[1] * d[:, 1])
    c2 = org[0] ** 2 + org[1] ** 2 - wall_r ** 2
    disc = b2 ** 2 - 4 * a2 * c2
    a2s = torch.where(a2 > 1e-12, a2, torch.ones_like(a2))
    t = (-b2 + torch.sqrt(torch.clamp(disc, min=0))) / (2 * a2s)
    t = torch.where((a2 > 1e-12) & (disc >= 0) & (t > 0), t, torch.full_like(t, math.inf))
    t_best = torch.minimum(t_best, t)
    # boxes (slab test)
    for bx in boxes:
        lo, hi = bx[:3], bx[3:]
        inv = 1.0 / torch.where(d.abs() > 1e-12, d, torch.full_like(d, 1e-12))
        t0 = (lo - org) * inv
        t1 = (hi - org) * inv
        tmin = torch.minimum(t0, t1).max(dim=1).values
        tmax = torch.maximum(t0, t1).min(dim=1).values
        hit = (tmax >= tmin) & (tmin > 0)
        t_best = torch.minimum(t_best, torch.where(hit, tmin, torch.full_like(tmin, math.inf)))
    return t_best


def lidar_directions(n_rings: int, n_az: int, az_offset: float = 0.0,
                     elev_min_deg: float = -30.67, elev_max_deg: float = 10.67,
                     dtype=torch.float64) -> torch.Tensor:
    elev = torch.deg2rad(torch.linspace(elev_min_deg, elev_max_deg, n_rings, dtype=dtype))
    az = torch.arange(n_az, dtype=dtype) * (2 * math.pi / n_az) + az_offset
    e, z = torch.meshgrid(elev, az, indexing="ij")
    d = torch.stack([torch.cos(e) * torch.cos(z), torch.cos(e) * torch.sin(z), torch.sin(e)], dim=-1)
    return d.reshape(-1, 3)


def make_sequence(n_rings: int = 32, n_az: int = 1080, n_past: int = 6, n_future: int = 6,
                  step: float = 3.0, max_range: float = 100.0, range_noise: float = 0.0,
                  seed: int = 0, dtype=torch.float32, n_boxes: int = 12,
                  ground_h: float = 1.84, wall_r: float = 45.0, az_jitter: bool = True
                  ) -> Tuple[torch.Tensor, torch.Tensor, List[torch.Tensor], List[torch.Tensor]]:
    """Current scan plus n_past + n_future neighbor scans, all in the current frame.

    step: ego displacement between consecutive frames [m] (0 gives a static ego).
    az_jitter: random azimuth offset per frame (within one azimuth step); without it, a static
    ego re-fires exactly the same directions.
    Returns cur_origin (3,), cur_points (N, 3), nbr_origins [T x (3,)], nbr_points [T x (M_t, 3)].
    """
    g = torch.Generator().manual_seed(seed)
    boxes = []
    for _ in range(n_boxes):
        cx = (torch.rand(1, generator=g) * 70 - 35).item()
        cy = (torch.rand(1, generator=g) * 2 - 1).sign().item() * (4 + torch.rand(1, generator=g).item() * 20)
        sx, sy, sz = (1 + torch.rand(3, generator=g, dtype=torch.float64) * torch.tensor([3.0, 2.0, 2.0], dtype=torch.float64)).tolist()
        boxes.append([cx - sx, cy - sy, -ground_h, cx + sx, cy + sy, -ground_h + sz])
    boxes_t = torch.tensor(boxes, dtype=torch.float64) if boxes else torch.zeros(0, 6, dtype=torch.float64)

    offsets = [k for k in range(-n_past, n_future + 1)]
    cur_world = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float64)
    scans = {}
    for k in offsets:
        org = torch.tensor([k * step, 0.0, 0.0], dtype=torch.float64)
        az_off = (torch.rand(1, generator=g).item() - 0.5) * (2 * math.pi / n_az) if az_jitter else 0.0
        d = lidar_directions(n_rings, n_az, az_off)
        t = _ray_cast(org, d, boxes_t, ground_h, wall_r)
        if range_noise > 0:
            t = t + range_noise * torch.randn(t.shape, generator=g, dtype=t.dtype)
        ok = torch.isfinite(t) & (t <= max_range) & (t > 0.5)
        pts = org + t[ok, None] * d[ok]
        scans[k] = ((org - cur_world).to(dtype), (pts - cur_world).to(dtype))
    cur_o, cur_p = scans[0]
    nbr_o = [scans[k][0] for k in offsets if k != 0]
    nbr_p = [scans[k][1] for k in offsets if k != 0]
    return cur_o, cur_p, nbr_o, nbr_p
