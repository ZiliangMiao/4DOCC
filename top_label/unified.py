"""Unified (angle-independent) TOP label computation, device agnostic.

Model
-----
Every lidar beam is a cone with apex at its sensor origin, unit axis d and
half-angle theta_dvg / 2 (radius at axial depth s is tau * s, tau = tan(theta_dvg / 2)).
Work in the current frame with the current origin moved to 0. For a current
beam i (direction d_i, hit range r_i) and a neighbor beam j (origin a, direction
d_j, hit range r_j), the point x(s) = s * d_i is called overlapping at depth s if

    kappa * dist(x(s), line_j) <= R(s) = tau * s + tau * t(s),
    t(s) = (x(s) - a) . d_j            (foot of the perpendicular on beam j),

i.e. the distance between the two axes is at most the sum of the two radii
(kappa = 1 is the geometric choice, kappa = 0.5 reproduces Eq. 14 of the paper).
Both sides squared are quadratics in s, so the overlapping depths form a single
interval [s_lo, s_hi] that is solved in closed form. No angle based case split,
no explicit coplanarity test, and it stays defined when a == 0 or d_i || d_j.

Every sample depth s in the interval is projected onto beam j (t(s)) and
labelled with Eq. 6-7 of the paper using delta = t(s) - r_j:
free if delta < 0 (w = 1), occupied if exp(-delta) >= lambda_occ,
unknown otherwise (w = exp(-delta)).

Candidate pairs come from a lossless 1-D index on the azimuth around the
baseline axis (see `_phi_candidates` and the design document), so the work per
frame pair is O((N + M) log M + K) instead of O(N * M).
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Iterator, List, Optional, Sequence, Tuple

import torch

STATE_UNKNOWN = 0
STATE_FREE = 1
STATE_OCCUPIED = 2

_F64 = torch.float64


@dataclass
class TopLabelConfig:
    theta_dvg: float = 0.003  # full beam divergence angle [rad], paper value for nuScenes
    lambda_occ: float = 0.9  # occupied if w_conf >= lambda_occ
    w_min: float = 0.1  # drop samples with w_conf < w_min (0 disables, paper has no truncation)
    min_range: float = 1.0  # s and t must be >= min_range, points closer are ignored
    max_range: float = 100.0  # s and t must be <= max_range, points farther are ignored
    sep_scale: float = 1.0  # kappa, 1.0 = geometric, 0.5 = paper Eq. 14 (half chord)
    cur_hit_margin: float = math.inf  # s <= r_i + cur_hit_margin (inf: only max_range)
    sampling: str = "paper"  # "paper" (closest point + paper 5 points) or "uniform"
    uniform_step: float = 0.5  # [m] spacing of "uniform" samples
    uniform_max: int = 8  # max number of "uniform" samples per pair (plus the closest point)
    prune: str = "phi"  # "phi" (lossless azimuth index) or "none" (all N x M pairs)
    pair_chunk: int = 2_000_000  # max candidate pairs evaluated at once (memory bound)
    baseline_eps: float = 1e-6  # [m] below this the origins are treated as coincident
    dedup_tol: float = 1e-6  # [m] samples of one pair closer than this are merged
    hit_tol: float = 1e-6  # [m] |t - r_j| below this counts as "at the hit point" (occupied)

    @property
    def tau(self) -> float:
        return math.tan(self.theta_dvg / 2.0)

    @property
    def unk_margin(self) -> float:
        # samples beyond r_j + unk_margin have w_conf < w_min
        return -math.log(self.w_min) if self.w_min > 0 else math.inf

    @property
    def num_slots(self) -> int:
        return 6 if self.sampling == "paper" else 1 + self.uniform_max


# ----------------------------------------------------------------------------------------------
# labels
# ----------------------------------------------------------------------------------------------
def label_from_depth(t: torch.Tensor, r_j: torch.Tensor, lambda_occ: float,
                     hit_tol: float = 0.0) -> Tuple[torch.Tensor, torch.Tensor]:
    """Eq. 6-7 with the depth measured along the neighbor beam.

    delta = t - r_j. free if delta < -hit_tol (w = 1); otherwise w = exp(-max(delta, 0)),
    occupied if w >= lambda_occ, unknown else. hit_tol keeps a sample that lies on the hit
    point up to round-off from flipping to free.
    Returns (state uint8, w_conf same dtype as t).
    """
    delta = t - r_j
    free = delta < -hit_tol
    w = torch.where(free, torch.ones_like(delta), torch.exp(-torch.clamp(delta, min=0.0)))
    state = torch.full_like(delta, STATE_UNKNOWN, dtype=torch.uint8)
    state = torch.where(w >= lambda_occ, torch.full_like(state, STATE_OCCUPIED), state)
    state = torch.where(free, torch.full_like(state, STATE_FREE), state)
    return state, w


# ----------------------------------------------------------------------------------------------
# stage 2: exact overlap interval per pair (elementwise)
# ----------------------------------------------------------------------------------------------
def overlap_intervals(d_i: torch.Tensor, d_j: torch.Tensor, a: torch.Tensor,
                      s_cap: torch.Tensor, t_cap: torch.Tensor, cfg: TopLabelConfig) -> Dict[str, torch.Tensor]:
    """Overlap interval on the current beam for K pairs.

    Args:
        d_i: (K, 3) unit directions of current beams (current origin at 0).
        d_j: (K, 3) unit directions of neighbor beams.
        a: (3,) neighbor origin in the current frame relative to the current origin.
        s_cap: (K,) upper bound of s, t_cap: (K,) upper bound of t.
    Returns dict with s_lo, s_hi, valid, s_c (closest approach clamped into the interval, s_lo if parallel),
    q (unclamped closest approach = Eq. 5, inf if parallel), c (cos alpha), u, v.
    """
    tau = cfg.tau
    k2 = cfg.sep_scale ** 2
    s_min = t_min = cfg.min_range
    a = a.to(d_i).reshape(1, 3)

    c = (d_i * d_j).sum(-1)
    m = torch.cross(d_i, d_j, dim=-1)
    sin2 = (m * m).sum(-1)  # accurate for small angles, unlike 1 - c^2
    u = (d_i * a).sum(-1)
    v = (d_j * a).sum(-1)
    axd = torch.cross(a.expand_as(d_j), d_j, dim=-1)
    perp2 = (axd * axd).sum(-1)  # |a|^2 - v^2, squared distance of the current origin to line j

    # kappa^2 dist^2(s) = k2 * (sin2 s^2 - 2 (u - c v) s + perp2),  R(s) = g s + h
    g = tau * (1.0 + c)
    h = -tau * v
    A = k2 * sin2 - g * g
    B = -2.0 * (k2 * (u - c * v) + g * h)
    C = k2 * perp2 - h * h

    inf = torch.full_like(c, math.inf)
    ninf = -inf

    # roots of A s^2 + B s + C (numerically stable form)
    disc = B * B - 4.0 * A * C
    sq = torch.sqrt(torch.clamp(disc, min=0.0))
    qq = -0.5 * (B + torch.where(B >= 0, sq, -sq))
    qq_safe = torch.where(qq == 0, torch.ones_like(qq), qq)
    A_safe = torch.where(A == 0, torch.ones_like(A), A)
    ra = torch.where(qq == 0, torch.zeros_like(qq), qq / A_safe)
    rb = torch.where(qq == 0, torch.zeros_like(qq), C / qq_safe)
    r1 = torch.minimum(ra, rb)
    r2 = torch.maximum(ra, rb)

    # set {f <= 0} as an interval [f_lo, f_hi] restricted to where R >= 0 (see module doc)
    f_lo = ninf.clone()
    f_hi = inf.clone()
    empty = torch.zeros_like(c, dtype=torch.bool)
    pos = A > 0
    neg = A < 0
    lin = A == 0
    # A > 0: bounded
    f_lo = torch.where(pos, r1, f_lo)
    f_hi = torch.where(pos, r2, f_hi)
    empty = empty | (pos & (disc < 0))
    # A < 0: two rays; the one with R >= 0 is on the right if g >= 0
    two_rays = neg & (disc >= 0)
    f_lo = torch.where(two_rays & (g >= 0), r2, f_lo)
    f_hi = torch.where(two_rays & (g < 0), r1, f_hi)
    # A == 0: B s + C <= 0
    B_safe = torch.where(B == 0, torch.ones_like(B), B)
    lin_root = -C / B_safe
    f_hi = torch.where(lin & (B > 0), lin_root, f_hi)
    f_lo = torch.where(lin & (B < 0), lin_root, f_lo)
    empty = empty | (lin & (B == 0) & (C > 0))

    # domain: s in [s_min, s_cap], t(s) = c s - v in [t_min, t_cap]
    d_lo = torch.full_like(c, s_min)
    d_hi = s_cap.to(c)
    t_cap = t_cap.to(c)
    c_eps = 1e-12
    cp = c > c_eps
    cn = c < -c_eps
    cz = ~(cp | cn)
    c_safe = torch.where(cz, torch.ones_like(c), c)
    lo_t = (t_min + v) / c_safe
    hi_t = (t_cap + v) / c_safe
    d_lo = torch.where(cp, torch.maximum(d_lo, lo_t), d_lo)
    d_hi = torch.where(cp, torch.minimum(d_hi, hi_t), d_hi)
    d_lo = torch.where(cn, torch.maximum(d_lo, hi_t), d_lo)
    d_hi = torch.where(cn, torch.minimum(d_hi, lo_t), d_hi)
    empty = empty | (cz & ((-v < t_min) | (-v > t_cap)))

    s_lo = torch.maximum(f_lo, d_lo)
    s_hi = torch.minimum(f_hi, d_hi)
    valid = (~empty) & (s_lo <= s_hi) & torch.isfinite(s_lo) & torch.isfinite(s_hi)

    # closest approach of the two axes (Eq. 5), inf when parallel. The anchor s_c is q clamped
    # into the interval; parallel axes have no closest point and use the interval start instead.
    sin2_safe = torch.where(sin2 > 1e-24, sin2, torch.ones_like(sin2))
    q = torch.where(sin2 > 1e-24, (u - c * v) / sin2_safe, inf)
    s_c = torch.where(torch.isfinite(q), torch.minimum(torch.maximum(q, s_lo), s_hi), s_lo)
    return {"s_lo": s_lo, "s_hi": s_hi, "valid": valid, "s_c": s_c, "q": q, "c": c, "u": u, "v": v}


def _sample_depths(iv: Dict[str, torch.Tensor], r_i: torch.Tensor, r_j: torch.Tensor,
                   cfg: TopLabelConfig) -> Tuple[torch.Tensor, torch.Tensor]:
    """Fixed number of sample slots per pair. Returns depths (K, S) and mask (K, S)."""
    s_lo, s_hi, s_c = iv["s_lo"], iv["s_hi"], iv["s_c"]
    if cfg.sampling == "paper":
        # o_k1 = p_i, o_k2 = projection of p_j on d_i, o_k3..5 = midpoints with the closest point
        s_pj = iv["u"] + r_j * iv["c"]
        s = torch.stack([s_c, r_i, s_pj, 0.5 * (r_i + s_pj), 0.5 * (r_i + s_c), 0.5 * (s_pj + s_c)], dim=1)
        mask = (s >= s_lo[:, None]) & (s <= s_hi[:, None])
    elif cfg.sampling == "uniform":
        length = s_hi - s_lo
        n = torch.clamp(torch.ceil(length / cfg.uniform_step), min=1, max=cfg.uniform_max)
        k = torch.arange(cfg.uniform_max, device=s_lo.device, dtype=s_lo.dtype)
        s_u = s_lo[:, None] + (k[None, :] + 0.5) * (length / n)[:, None]
        s = torch.cat([s_c[:, None], s_u], dim=1)
        mask = torch.cat([torch.ones_like(s_c[:, None], dtype=torch.bool), k[None, :] < n[:, None]], dim=1)
    else:
        raise ValueError(f"unknown sampling mode {cfg.sampling}")
    mask = mask & iv["valid"][:, None]
    # merge duplicates inside one pair (keep the first slot)
    S = s.shape[1]
    for k in range(1, S):
        dup = ((s[:, :k] - s[:, k:k + 1]).abs() <= cfg.dedup_tol) & mask[:, :k]
        mask[:, k] &= ~dup.any(dim=1)
    return s, mask


def _evaluate_pairs(i_idx: torch.Tensor, j_idx: torch.Tensor, cur: Dict[str, torch.Tensor],
                    nbr: Dict[str, torch.Tensor], a: torch.Tensor, cfg: TopLabelConfig) -> Dict[str, torch.Tensor]:
    d_i, r_i = cur["dir"][i_idx], cur["rng"][i_idx]
    d_j, r_j = nbr["dir"][j_idx], nbr["rng"][j_idx]
    s_cap = torch.clamp(r_i + cfg.cur_hit_margin, max=cfg.max_range)
    t_cap = torch.clamp(r_j + cfg.unk_margin, max=cfg.max_range)
    iv = overlap_intervals(d_i, d_j, a, s_cap, t_cap, cfg)
    # compact to overlapping pairs before the (K, S) sampling tensors are built
    keep = torch.nonzero(iv["valid"], as_tuple=True)[0]
    iv = {k: v[keep] for k, v in iv.items()}
    i_idx, j_idx, r_i, r_j = i_idx[keep], j_idx[keep], r_i[keep], r_j[keep]
    s, mask = _sample_depths(iv, r_i, r_j, cfg)
    t = iv["c"][:, None] * s - iv["v"][:, None]
    state, w = label_from_depth(t, r_j[:, None].expand_as(t), cfg.lambda_occ, cfg.hit_tol)
    mask = mask & (w >= cfg.w_min)
    pair, slot = torch.nonzero(mask, as_tuple=True)
    return {
        "cur_idx": i_idx[pair], "nbr_idx": j_idx[pair], "slot": slot,
        "depth": s[pair, slot], "nbr_depth": t[pair, slot],
        "state": state[pair, slot], "w_conf": w[pair, slot],
    }


# ----------------------------------------------------------------------------------------------
# stage 1: lossless candidate generation
# ----------------------------------------------------------------------------------------------
def _axis_frame(axis: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    e3 = axis / torch.linalg.norm(axis)
    helper = torch.tensor([1.0, 0.0, 0.0], dtype=axis.dtype, device=axis.device)
    if abs(float(e3[0])) > 0.9:
        helper = torch.tensor([0.0, 1.0, 0.0], dtype=axis.dtype, device=axis.device)
    e1 = helper - (helper @ e3) * e3
    e1 = e1 / torch.linalg.norm(e1)
    e2 = torch.cross(e3, e1, dim=0)
    return e1, e2


def _azimuth(dirs: torch.Tensor, e1: torch.Tensor, e2: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """psi in [0, pi) (azimuth around the axis modulo pi) and sin(gamma) (distance to the axis)."""
    x = dirs @ e1
    y = dirs @ e2
    sg = torch.hypot(x, y)
    psi = torch.remainder(torch.atan2(y, x), math.pi)
    psi = torch.where(psi >= math.pi, psi - math.pi, psi)
    return psi, sg


def _windows(sg: torch.Tensor, thr: float) -> torch.Tensor:
    ratio = torch.where(sg > 0, thr / torch.where(sg > 0, sg, torch.ones_like(sg)), torch.full_like(sg, 2.0))
    return torch.asin(torch.clamp(ratio, max=1.0)) + 1e-9


def _expand(lo: torch.Tensor, cnt: torch.Tensor, q_ids: torch.Tensor, perm: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Expand per-query ranges [lo, lo + cnt) of an extended sorted array into (query, target) pairs."""
    M = perm.numel()
    total = int(cnt.sum())
    if total == 0:
        e = torch.empty(0, dtype=torch.long, device=perm.device)
        return e, e
    rep = torch.repeat_interleave(torch.arange(q_ids.numel(), device=perm.device), cnt)
    start = torch.cumsum(cnt, 0) - cnt
    off = torch.arange(total, device=perm.device) - start[rep]
    pos = lo[rep] + off
    return q_ids[rep], perm[torch.remainder(pos, M)]


def _chunk_bounds(cnt: torch.Tensor, chunk: int) -> List[Tuple[int, int]]:
    """Split queries into consecutive groups whose total count is <= chunk (a single larger query stays alone)."""
    if cnt.numel() == 0:
        return []
    csum = torch.cumsum(cnt, 0).cpu()
    bounds = []
    start, base = 0, 0
    n = cnt.numel()
    while start < n:
        end = int(torch.searchsorted(csum, torch.tensor(base + chunk, dtype=csum.dtype), right=True))
        end = max(end, start + 1)
        bounds.append((start, end))
        base = int(csum[end - 1])
        start = end
    return bounds


def _side_ranges(psi_q: torch.Tensor, w_q: torch.Tensor, psi_t: torch.Tensor):
    """Window ranges of every query into the extended sorted target azimuths."""
    order = torch.argsort(psi_t)
    srt = psi_t[order]
    M = srt.numel()
    ext = torch.cat([srt - math.pi, srt, srt + math.pi])
    lo = torch.searchsorted(ext, psi_q - w_q, right=False)
    hi = torch.searchsorted(ext, psi_q + w_q, right=True)
    cnt = torch.clamp(hi - lo, min=0, max=M)
    return lo, cnt, order


def _phi_candidates(cur_dir: torch.Tensor, cur_ids: torch.Tensor, nbr_dir: torch.Tensor, nbr_ids: torch.Tensor,
                    a: torch.Tensor, cfg: TopLabelConfig) -> Iterator[Tuple[torch.Tensor, torch.Tensor]]:
    """Yield chunks of candidate pairs (indices into the full current / neighbor arrays).

    Plane mode (|a| > baseline_eps): every plane through the current origin and the baseline a is
    parameterized by its azimuth psi around the axis a/|a|. Overlap implies
        sin(gamma_j) |sin(psi_i - psi_j)| <= 2 tau / kappa   (t >= s case)  OR
        sin(gamma_i) |sin(psi_i - psi_j)| <= 2 tau / kappa   (s >= t case),
    which is exact (no approximation beyond the cone model). Angular mode (|a| <= baseline_eps):
    both inequalities must hold with the threshold bounding sin(alpha).
    """
    tau, kap = cfg.tau, cfg.sep_scale
    a_norm = float(torch.linalg.norm(a))
    if a_norm > cfg.baseline_eps:
        axis, thr, mode = a, 2.0 * tau / kap, "or"
    else:
        axis = torch.tensor([0.0, 0.0, 1.0], dtype=a.dtype, device=a.device)
        thr = 2.0 * tau / kap + a_norm * (1.0 + tau / kap) / cfg.min_range
        mode = "and"
    thr = min(thr * (1.0 + 1e-9) + 1e-12, 1.0)
    e1, e2 = _axis_frame(axis)
    psi_i, sg_i = _azimuth(cur_dir[cur_ids], e1, e2)
    psi_j, sg_j = _azimuth(nbr_dir[nbr_ids], e1, e2)

    def pred_b(li, lj):  # s >= t branch, window from the current beam
        return sg_i[li] * torch.sin(psi_i[li] - psi_j[lj]).abs() <= thr

    def pred_a(li, lj):  # t >= s branch, window from the neighbor beam
        return sg_j[lj] * torch.sin(psi_i[li] - psi_j[lj]).abs() <= thr

    # branch B: query = current beams
    lo, cnt, order = _side_ranges(psi_i, _windows(sg_i, thr), psi_j)
    q_all = torch.arange(psi_i.numel(), device=psi_i.device)
    for s0, s1 in _chunk_bounds(cnt, cfg.pair_chunk):
        li, lj = _expand(lo[s0:s1], cnt[s0:s1], q_all[s0:s1], order)
        keep = pred_b(li, lj)
        if mode == "and":
            keep &= pred_a(li, lj)
        yield cur_ids[li[keep]], nbr_ids[lj[keep]]
    if mode == "and":
        return
    # branch A: query = neighbor beams, drop pairs already produced by branch B
    lo, cnt, order = _side_ranges(psi_j, _windows(sg_j, thr), psi_i)
    q_all = torch.arange(psi_j.numel(), device=psi_j.device)
    for s0, s1 in _chunk_bounds(cnt, cfg.pair_chunk):
        lj, li = _expand(lo[s0:s1], cnt[s0:s1], q_all[s0:s1], order)
        keep = pred_a(li, lj) & ~pred_b(li, lj)
        yield cur_ids[li[keep]], nbr_ids[lj[keep]]


def _dense_candidates(cur_ids: torch.Tensor, nbr_ids: torch.Tensor,
                      cfg: TopLabelConfig) -> Iterator[Tuple[torch.Tensor, torch.Tensor]]:
    M = nbr_ids.numel()
    rows = max(1, cfg.pair_chunk // max(M, 1))
    for s0 in range(0, cur_ids.numel(), rows):
        ci = cur_ids[s0:s0 + rows]
        yield ci.repeat_interleave(M), nbr_ids.repeat(ci.numel())


# ----------------------------------------------------------------------------------------------
# public API
# ----------------------------------------------------------------------------------------------
def _prepare(origin: torch.Tensor, points: torch.Tensor, cur_origin: torch.Tensor,
             cfg: TopLabelConfig) -> Dict[str, torch.Tensor]:
    pts = points.to(_F64)
    org = origin.to(_F64).reshape(3)
    vec = pts - org
    rng = torch.linalg.norm(vec, dim=1)
    ok = torch.isfinite(rng) & (rng >= cfg.min_range) & (rng <= cfg.max_range)
    rng_safe = torch.where(ok, rng, torch.ones_like(rng))
    d = vec / rng_safe[:, None]
    return {"dir": d, "rng": rng, "ids": torch.nonzero(ok, as_tuple=True)[0], "a": org - cur_origin}


def compute_frame_pair_labels(cur: Dict[str, torch.Tensor], nbr: Dict[str, torch.Tensor],
                              cfg: TopLabelConfig, stats: Optional[dict] = None) -> Dict[str, torch.Tensor]:
    """Labels of one (current, neighbor) frame pair from prepared rays (see `_prepare`)."""
    a = nbr["a"]
    if cfg.prune == "phi":
        gen = _phi_candidates(cur["dir"], cur["ids"], nbr["dir"], nbr["ids"], a, cfg)
    elif cfg.prune == "none":
        gen = _dense_candidates(cur["ids"], nbr["ids"], cfg)
    else:
        raise ValueError(f"unknown prune mode {cfg.prune}")
    outs = []
    for i_idx, j_idx in gen:
        if stats is not None:
            stats["candidates"] = stats.get("candidates", 0) + int(i_idx.numel())
        if i_idx.numel() == 0:
            continue
        outs.append(_evaluate_pairs(i_idx, j_idx, cur, nbr, a, cfg))
    keys = ["cur_idx", "nbr_idx", "slot", "depth", "nbr_depth", "state", "w_conf"]
    if not outs:
        dev = cur["dir"].device
        empty = {k: torch.empty(0, dtype=torch.long, device=dev) for k in ["cur_idx", "nbr_idx", "slot"]}
        empty.update({k: torch.empty(0, dtype=_F64, device=dev) for k in ["depth", "nbr_depth", "w_conf"]})
        empty["state"] = torch.empty(0, dtype=torch.uint8, device=dev)
        return empty
    return {k: torch.cat([o[k] for o in outs]) for k in keys}


def compute_top_labels(cur_origin: torch.Tensor, cur_points: torch.Tensor,
                       nbr_origins: Sequence[torch.Tensor], nbr_points: Sequence[torch.Tensor],
                       cfg: Optional[TopLabelConfig] = None, sort: bool = True,
                       stats: Optional[dict] = None) -> Dict[str, torch.Tensor]:
    """Temporal overlapping points and their labels for one current scan.

    All inputs are already expressed in the current frame; tensors may live on any device
    (the computation runs on the device of `cur_points`, internally in float64).

    Args:
        cur_origin: (3,) current sensor origin. cur_points: (N, 3) current hit points.
        nbr_origins: T tensors (3,). nbr_points: T tensors (M_t, 3).
        cfg: parameters. sort: sort the output by (frame, cur_idx, nbr_idx, slot).
    Returns dict of 1-D tensors of equal length:
        cur_idx (long, index into cur_points), frame_idx (long), nbr_idx (long, index into
        nbr_points[frame_idx]), depth (float32, along the current beam from cur_origin),
        nbr_depth (float32, along the neighbor beam), state (uint8, 0 unknown / 1 free /
        2 occupied), w_conf (float32).
    """
    cfg = cfg or TopLabelConfig()
    dev = cur_points.device
    cur_o = cur_origin.to(device=dev, dtype=_F64).reshape(3)
    cur = _prepare(cur_o, cur_points.to(dev), cur_o, cfg)
    outs = []
    for f, (o, p) in enumerate(zip(nbr_origins, nbr_points)):
        nbr = _prepare(o.to(dev), p.to(dev), cur_o, cfg)
        res = compute_frame_pair_labels(cur, nbr, cfg, stats)
        res["frame_idx"] = torch.full_like(res["cur_idx"], f)
        outs.append(res)
    keys = ["cur_idx", "frame_idx", "nbr_idx", "slot", "depth", "nbr_depth", "state", "w_conf"]
    if not outs:
        raise ValueError("at least one neighbor frame is required")
    out = {k: torch.cat([o[k] for o in outs]) for k in keys}
    if sort and out["cur_idx"].numel() > 0:
        n_cur = cur_points.shape[0]
        m_max = max([int(p.shape[0]) for p in nbr_points] + [1])
        key = ((out["frame_idx"] * n_cur + out["cur_idx"]) * m_max + out["nbr_idx"]) * cfg.num_slots + out["slot"]
        order = torch.argsort(key)
        out = {k: v[order] for k, v in out.items()}
    out["depth"] = out["depth"].float()
    out["nbr_depth"] = out["nbr_depth"].float()
    out["w_conf"] = out["w_conf"].float()
    return out
