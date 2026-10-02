"""CPU unit tests of the unified TOP label computation (synthetic data only).

Run: python -m pytest top_label/tests -q
"""
import math

import pytest
import torch

from top_label import (STATE_FREE, STATE_OCCUPIED, STATE_UNKNOWN, TopLabelConfig, compute_top_labels,
                       label_from_depth, overlap_intervals)
from top_label.synthetic import make_sequence

F64 = torch.float64
BIG = 1e3


def _unit(*v):
    t = torch.tensor(v, dtype=F64)
    return t / torch.linalg.norm(t)


def _iv(d_i, d_j, a, cfg, s_cap=BIG, t_cap=BIG):
    out = overlap_intervals(d_i.reshape(1, 3), d_j.reshape(1, 3), torch.as_tensor(a, dtype=F64),
                            torch.tensor([s_cap], dtype=F64), torch.tensor([t_cap], dtype=F64), cfg)
    return {k: v[0] for k, v in out.items()}


def _eq5(d_i, d_j, a):
    """Closest point on the current beam as written in the paper / original code (Eq. 5)."""
    m = torch.cross(d_i, d_j, dim=0)
    return float(torch.dot(torch.cross(a, d_j, dim=0), m) / torch.dot(m, m))


def _eq15(q, t_q, alpha, theta):
    s, tn = math.sin(alpha / 2), math.tan(theta / 2)
    return (q * (s + tn) - t_q * tn) / (s + 2 * tn)


def _assert_same(o1, o2):
    assert o1.keys() == o2.keys()
    for k in o1:
        assert torch.equal(o1[k], o2[k]), k


# ----------------------------------------------------------------------------------------------
# geometry of a single pair
# ----------------------------------------------------------------------------------------------
def test_large_angle_matches_eq5_and_is_short():
    cfg = TopLabelConfig()
    a = torch.tensor([10.0, 0.0, 0.0], dtype=F64)
    d_i = _unit(1, 1, 0)
    Q = 20.0 * d_i
    d_j = (Q - a) / torch.linalg.norm(Q - a)
    t_q = float(torch.linalg.norm(Q - a))
    iv = _iv(d_i, d_j, a, cfg)
    assert bool(iv["valid"])
    q = _eq5(d_i, d_j, a)
    assert abs(q - 20.0) < 1e-9
    assert abs(float(iv["q"]) - q) < 1e-9
    assert abs(float(iv["s_c"]) - q) < 1e-9
    assert float(iv["s_lo"]) < q < float(iv["s_hi"])
    # half length ~ R(q) / sin(alpha): short region around the paper's intersection point
    alpha = math.acos(float(torch.dot(d_i, d_j)))
    half = cfg.tau * (q + t_q) / math.sin(alpha)
    length = float(iv["s_hi"] - iv["s_lo"])
    assert length < 0.3  # alpha ~ 0.5 rad, R(q) ~ 0.05 m
    assert abs(length / (2 * half) - 1) < 0.02


@pytest.mark.parametrize("delta,state,w", [(-5.0, STATE_FREE, 1.0),
                                           (0.05, STATE_OCCUPIED, math.exp(-0.05)),
                                           (1.0, STATE_UNKNOWN, math.exp(-1.0))])
def test_large_angle_labels(delta, state, w):
    cfg = TopLabelConfig()
    a = torch.tensor([10.0, 0.0, 0.0], dtype=F64)
    d_i = _unit(1, 1, 0)
    Q = 20.0 * d_i
    d_j = (Q - a) / torch.linalg.norm(Q - a)
    t_q = float(torch.linalg.norm(Q - a))
    p_i = (30.0 * d_i).reshape(1, 3)
    p_j = (a + (t_q - delta) * d_j).reshape(1, 3)
    out = compute_top_labels(torch.zeros(3, dtype=F64), p_i, [a], [p_j], cfg)
    # slot 0 is the closest point of the two axes (= Eq. 5 intersection)
    assert int(out["slot"][0]) == 0
    assert abs(float(out["depth"][0]) - 20.0) < 1e-4
    assert int(out["state"][0]) == state
    assert abs(float(out["w_conf"][0]) - w) < 1e-5
    # any extra sample (projection of p_j, midpoints) lies inside the short overlap region
    assert bool(((out["depth"] - 20.0).abs() < 0.15).all())


def test_small_angle_start_matches_eq15_with_paper_scale():
    theta = 0.003
    alpha = 0.5 * theta
    q, t_q = 40.0, 35.0
    d_i = _unit(1, 0, 0)
    d_j = torch.tensor([math.cos(alpha), math.sin(alpha), 0.0], dtype=F64)
    a = q * d_i - t_q * d_j
    assert abs(_eq5(d_i, d_j, a) - q) < 1e-6
    b_paper = _eq15(q, t_q, alpha, theta)

    # kappa = 0.5 reproduces Eq. 15 up to O(alpha^2)
    iv = _iv(d_i, d_j, a, TopLabelConfig(theta_dvg=theta, sep_scale=0.5))
    assert bool(iv["valid"])
    assert abs(float(iv["s_lo"]) - b_paper) < 1e-4 * q
    # alpha <= theta_dvg: the cones never separate again, the interval runs to the cap
    assert float(iv["s_hi"]) == BIG
    assert abs(float(iv["s_c"]) - q) < 1e-6

    # kappa = 1 (full chord): closed form of the same construction, starts later than Eq. 15
    cfg = TopLabelConfig(theta_dvg=theta)
    iv1 = _iv(d_i, d_j, a, cfg)
    tau, c, s = cfg.tau, math.cos(alpha), math.sin(alpha)
    b_full = (q * s - tau * t_q + tau * q * c) / (s + tau * (1 + c))
    assert abs(float(iv1["s_lo"]) - b_full) < 1e-6
    assert float(iv1["s_lo"]) > b_paper


def test_intermediate_angle_is_long_interval():
    """theta < alpha < ~30 theta: the paper treats it as one point, the cones overlap over meters."""
    theta = 0.003
    alpha = 3 * theta
    q, t_q = 40.0, 35.0
    d_i = _unit(1, 0, 0)
    d_j = torch.tensor([math.cos(alpha), math.sin(alpha), 0.0], dtype=F64)
    a = q * d_i - t_q * d_j
    iv = _iv(d_i, d_j, a, TopLabelConfig(theta_dvg=theta))
    assert bool(iv["valid"]) and float(iv["s_hi"]) < BIG
    assert float(iv["s_hi"] - iv["s_lo"]) > 10.0
    assert float(iv["s_lo"]) < q < float(iv["s_hi"])


def test_exactly_parallel_offset():
    cfg = TopLabelConfig()
    d = _unit(1, 0, 0)
    a = torch.tensor([0.0, 0.1, 0.0], dtype=F64)
    iv = _iv(d, d, a, cfg)
    assert bool(iv["valid"])
    assert abs(float(iv["s_lo"]) - 0.1 / (2 * cfg.tau)) < 1e-6
    assert float(iv["s_hi"]) == BIG
    assert math.isinf(float(iv["q"]))
    assert float(iv["s_c"]) == float(iv["s_lo"])


def test_coincident_origins():
    cfg = TopLabelConfig()
    a = torch.zeros(3, dtype=F64)
    d = _unit(1, 2, 0.3)
    iv = _iv(d, d, a, cfg)
    assert bool(iv["valid"])
    assert float(iv["s_lo"]) == cfg.min_range and float(iv["s_hi"]) == BIG
    for alpha, expect in [(0.5 * cfg.theta_dvg, True), (2.0 * cfg.theta_dvg, False)]:
        d_j = _unit(math.cos(alpha), math.sin(alpha), 0)
        iv = _iv(_unit(1, 0, 0), d_j, a, cfg)
        assert bool(iv["valid"]) == expect


def test_coincident_origins_end_to_end():
    """Static ego: every beam re-observes itself, occupied exactly at its own hit point."""
    cfg = TopLabelConfig()
    co, cp, _, _ = make_sequence(n_rings=4, n_az=60, n_past=0, n_future=0, seed=1)
    out = compute_top_labels(co, cp, [co.clone()], [cp.clone()], cfg)
    rng = torch.linalg.norm(cp.double(), dim=1).float()
    self_hit = (out["cur_idx"] == out["nbr_idx"]) & (out["state"] == STATE_OCCUPIED)
    idx = out["cur_idx"][self_hit]
    assert torch.equal(torch.unique(idx), torch.arange(cp.shape[0]))
    assert torch.allclose(out["depth"][self_hit], rng[idx], atol=1e-4)
    # free samples in front of the hit point exist as well
    assert bool(((out["cur_idx"] == out["nbr_idx"]) & (out["state"] == STATE_FREE)).any())


def test_skew_lines():
    cfg = TopLabelConfig()
    d_i = _unit(1, 0, 0)
    d_j = _unit(0, 1, 0)
    far = _iv(d_i, d_j, torch.tensor([10.0, -10.0, 1.0], dtype=F64), cfg)
    assert not bool(far["valid"])
    near = _iv(d_i, d_j, torch.tensor([10.0, -10.0, 0.01], dtype=F64), cfg)  # 0.01 < r_i + r_j = 0.03
    assert bool(near["valid"])
    assert abs(float(near["s_c"]) - 10.0) < 1e-9


def test_behind_sensor_rejected():
    cfg = TopLabelConfig()
    d_i = _unit(1, 0, 0)
    # centerlines cross at s = -5 (behind the current sensor)
    a = torch.tensor([-5.0, -5.0, 0.0], dtype=F64)
    d_j = _unit(0, -1, 0)
    assert not bool(_iv(d_i, d_j, a, cfg)["valid"])


def test_label_rule():
    r = torch.full((5,), 10.0, dtype=F64)
    t = torch.tensor([5.0, 10.0 - 1e-9, 10.1, 10.2, 13.0], dtype=F64)
    state, w = label_from_depth(t, r, 0.9, hit_tol=1e-6)
    assert state.tolist() == [STATE_FREE, STATE_OCCUPIED, STATE_OCCUPIED, STATE_UNKNOWN, STATE_UNKNOWN]
    assert torch.allclose(w, torch.tensor([1.0, 1.0, math.exp(-0.1), math.exp(-0.2), math.exp(-3.0)], dtype=F64))


# ----------------------------------------------------------------------------------------------
# chunking and pruning do not change the result
# ----------------------------------------------------------------------------------------------
def _random_scene(seed, n=300, m=300, baseline=(3.0, 1.0, 0.2)):
    g = torch.Generator().manual_seed(seed)
    cur_o = torch.randn(3, generator=g, dtype=F64)
    nbr_o = cur_o + torch.tensor(baseline, dtype=F64)
    def pts(o, k):
        d = torch.randn(k, 3, generator=g, dtype=F64)
        d[:, 2] *= 0.15  # lidar-like: mostly horizontal
        d = d / torch.linalg.norm(d, dim=1, keepdim=True)
        return o + d * (2 + 30 * torch.rand(k, 1, generator=g, dtype=F64))
    return cur_o, pts(cur_o, n), [nbr_o], [pts(nbr_o, m)]


SCENES = {
    "moving": lambda: make_sequence(n_rings=6, n_az=72, n_past=2, n_future=2, step=3.0, seed=3),
    "static": lambda: make_sequence(n_rings=6, n_az=72, n_past=1, n_future=1, step=0.0, seed=4,
                                    az_jitter=False, range_noise=0.05),
    "tiny_baseline": lambda: make_sequence(n_rings=6, n_az=72, n_past=1, n_future=1, step=1e-4, seed=5,
                                           az_jitter=False, range_noise=0.05),
    "random": lambda: _random_scene(6),
    "random_vertical": lambda: _random_scene(7, baseline=(0.0, 0.0, 2.0)),
}


@pytest.mark.parametrize("scene", list(SCENES))
@pytest.mark.parametrize("cfg_kw", [{}, {"sep_scale": 0.5, "theta_dvg": 0.02}, {"sampling": "uniform", "w_min": 0.0}])
def test_pruned_equals_dense_and_chunk_invariant(scene, cfg_kw):
    co, cp, no, np_ = SCENES[scene]()
    ref = compute_top_labels(co, cp, no, np_, TopLabelConfig(prune="none", pair_chunk=10 ** 8, **cfg_kw))
    assert ref["cur_idx"].numel() > 0
    for prune in ["none", "phi"]:
        for chunk in [10 ** 8, 997]:
            out = compute_top_labels(co, cp, no, np_, TopLabelConfig(prune=prune, pair_chunk=chunk, **cfg_kw))
            _assert_same(ref, out)


def test_phi_prunes_most_pairs():
    co, cp, no, np_ = make_sequence(n_rings=8, n_az=180, n_past=1, n_future=1, seed=8)
    st = {}
    compute_top_labels(co, cp, no, np_, TopLabelConfig(), stats=st)
    dense = sum(cp.shape[0] * p.shape[0] for p in np_)
    assert st["candidates"] < 0.05 * dense


def test_uniform_samples_inside_interval_and_capped():
    cfg = TopLabelConfig(sampling="uniform", uniform_step=0.5, uniform_max=4)
    co, cp, no, np_ = make_sequence(n_rings=6, n_az=72, n_past=1, n_future=1, seed=9)
    out = compute_top_labels(co, cp, no, np_, cfg)
    per_pair = torch.unique(torch.stack([out["frame_idx"], out["cur_idx"], out["nbr_idx"]]), dim=1, return_counts=True)[1]
    assert int(per_pair.max()) <= 1 + cfg.uniform_max
    assert bool((out["depth"] >= cfg.min_range).all()) and bool((out["depth"] <= cfg.max_range).all())


@pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA device")
def test_cuda_matches_cpu():
    co, cp, no, np_ = make_sequence(n_rings=8, n_az=180, n_past=2, n_future=2, seed=10)
    ref = compute_top_labels(co, cp, no, np_, TopLabelConfig())
    out = compute_top_labels(co.cuda(), cp.cuda(), [o.cuda() for o in no], [p.cuda() for p in np_], TopLabelConfig())
    for k in ["cur_idx", "frame_idx", "nbr_idx", "slot", "state"]:
        assert torch.equal(ref[k], out[k].cpu()), k
    for k in ["depth", "nbr_depth", "w_conf"]:
        assert torch.allclose(ref[k], out[k].cpu(), atol=1e-5), k
