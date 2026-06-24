"""Characterization tests for :mod:`gvhmr.utils.geo.hmr_cam`.

These pin the pure-torch camera math (intrinsics construction, K resizing,
bbox-from-xyxy, perspective projection, and the pred_cam <-> camera-translation
inverse pair) so the reorg/rename/typing passes cannot silently change numerics.
"""

from __future__ import annotations

import pytest
import torch

from gvhmr.utils.geo import hmr_cam

# ====== intrinsics: estimate_K / convert_f_to_K ===== #


def test_estimate_focal_length_is_image_diagonal() -> None:
    # Documented behaviour: focal length == sqrt(w^2 + h^2) (image diagonal).
    assert hmr_cam.estimate_focal_length(640, 480) == pytest.approx(800.0)
    assert hmr_cam.estimate_focal_length(3, 4) == pytest.approx(5.0)


def test_estimate_K_matches_golden() -> None:
    K = hmr_cam.estimate_K(640, 480)
    expected = torch.tensor(
        [
            [800.0, 0.0, 320.0],
            [0.0, 800.0, 240.0],
            [0.0, 0.0, 1.0],
        ]
    )
    assert torch.equal(K, expected)
    assert K.dtype == torch.float32


def test_estimate_K_structure_invariants() -> None:
    w, h = 1280, 720
    K = hmr_cam.estimate_K(w, h)
    f = hmr_cam.estimate_focal_length(w, h)
    # Square pixels: fx == fy.
    assert K[0, 0] == K[1, 1] == pytest.approx(f)
    # Principal point is the image centre.
    assert K[0, 2] == pytest.approx(w / 2.0)
    assert K[1, 2] == pytest.approx(h / 2.0)
    # Homogeneous bottom-right and zero skew.
    assert K[2, 2] == 1.0
    assert K[0, 1] == 0.0 and K[1, 0] == 0.0


def test_convert_f_to_K_matches_golden() -> None:
    K = hmr_cam.convert_f_to_K(1000.0, 800, 600)
    expected = torch.tensor(
        [
            [1000.0, 0.0, 400.0],
            [0.0, 1000.0, 300.0],
            [0.0, 0.0, 1.0],
        ]
    )
    assert torch.equal(K, expected)
    assert K.dtype == torch.float32


def test_convert_f_to_K_agrees_with_estimate_K_at_diagonal_focal() -> None:
    # estimate_K is convert_f_to_K specialised to the diagonal focal length.
    w, h = 960, 540
    f = hmr_cam.estimate_focal_length(w, h)
    assert torch.equal(hmr_cam.convert_f_to_K(f, w, h), hmr_cam.estimate_K(w, h))


def test_convert_K_to_K4_packs_fx_fy_cx_cy() -> None:
    K = hmr_cam.convert_f_to_K(1000.0, 800, 600)
    k4 = hmr_cam.convert_K_to_K4(K)
    assert torch.equal(k4, torch.tensor([1000.0, 1000.0, 400.0, 300.0]))
    assert k4.dtype == torch.float32


# ====== resize_K ===== #


def test_resize_K_matches_golden() -> None:
    K = hmr_cam.convert_f_to_K(1000.0, 800, 600)
    Kr = hmr_cam.resize_K(K, f=0.5)
    expected = torch.tensor(
        [
            [500.0, 0.0, 200.0],
            [0.0, 500.0, 150.0],
            [0.0, 0.0, 1.0],
        ]
    )
    assert torch.equal(Kr, expected)


def test_resize_K_scales_focal_and_principal_keeps_homogeneous() -> None:
    K = hmr_cam.convert_f_to_K(1234.0, 1920, 1080)
    f = 0.37
    Kr = hmr_cam.resize_K(K, f=f)
    assert Kr[0, 0] == pytest.approx(K[0, 0] * f)
    assert Kr[1, 1] == pytest.approx(K[1, 1] * f)
    assert Kr[0, 2] == pytest.approx(K[0, 2] * f)
    assert Kr[1, 2] == pytest.approx(K[1, 2] * f)
    # The (2,2) entry is forced back to 1 regardless of the scale.
    assert Kr[2, 2] == 1.0


def test_resize_K_identity_scale_only_resets_corner() -> None:
    # f=1 leaves fx/fy/cx/cy untouched; (2,2) is reassigned to 1.0 (already 1).
    K = hmr_cam.convert_f_to_K(700.0, 512, 384)
    assert torch.equal(hmr_cam.resize_K(K, f=1.0), K)


def test_resize_K_does_not_mutate_input() -> None:
    K = hmr_cam.convert_f_to_K(700.0, 512, 384)
    K_copy = K.clone()
    _ = hmr_cam.resize_K(K, f=0.25)
    assert torch.equal(K, K_copy)


def test_resize_K_is_batched() -> None:
    # The (..., 2, 2) indexing means resize_K accepts a leading batch dim.
    K = torch.stack([hmr_cam.convert_f_to_K(800.0, 640, 480) for _ in range(4)])
    Kr = hmr_cam.resize_K(K, f=0.5)
    assert Kr.shape == (4, 3, 3)
    assert torch.allclose(Kr[..., 2, 2], torch.ones(4))
    assert torch.allclose(Kr[:, 0, 0], torch.full((4,), 400.0))


# ====== get_bbx_xys_from_xyxy (aspect 192:256, base_enlarge 1.2) ===== #


def test_get_bbx_xys_from_xyxy_tall_box_golden() -> None:
    # Box w=200, h=300; aspect 192/256=0.75 -> w padded to 0.75*h=225;
    # size = max(300, 225) * 1.2 = 360.
    box = torch.tensor([[100.0, 200.0, 300.0, 500.0]])
    out = hmr_cam.get_bbx_xys_from_xyxy(box)
    assert torch.allclose(out, torch.tensor([[200.0, 350.0, 360.0]]))


def test_get_bbx_xys_from_xyxy_square_box_golden() -> None:
    # Square box: w=h=100; w > 0.75*h so h padded to w/0.75=133.33;
    # size = max(133.33, 100) * 1.2 = 160.
    box = torch.tensor([[0.0, 0.0, 100.0, 100.0]])
    out = hmr_cam.get_bbx_xys_from_xyxy(box)
    assert torch.allclose(out, torch.tensor([[50.0, 50.0, 160.0]]))


def test_get_bbx_xys_from_xyxy_wide_box_golden() -> None:
    # Wide box: w=400, h=100; w > 0.75*h so h padded to 400/0.75=533.33;
    # size = max(533.33, 400) * 1.2 = 640.
    box = torch.tensor([[0.0, 0.0, 400.0, 100.0]])
    out = hmr_cam.get_bbx_xys_from_xyxy(box)
    assert torch.allclose(out, torch.tensor([[200.0, 50.0, 640.0]]))


def test_get_bbx_xys_from_xyxy_center_invariants() -> None:
    box = torch.tensor([[10.0, 20.0, 50.0, 120.0]])
    out = hmr_cam.get_bbx_xys_from_xyxy(box)
    # Centre is always the midpoint of the input corners regardless of padding.
    assert out[0, 0] == pytest.approx((10.0 + 50.0) / 2.0)
    assert out[0, 1] == pytest.approx((20.0 + 120.0) / 2.0)


def test_get_bbx_xys_from_xyxy_size_scales_with_base_enlarge() -> None:
    box = torch.tensor([[0.0, 0.0, 300.0, 400.0]])
    out_12 = hmr_cam.get_bbx_xys_from_xyxy(box, base_enlarge=1.2)
    out_10 = hmr_cam.get_bbx_xys_from_xyxy(box, base_enlarge=1.0)
    # Only the size is multiplied by base_enlarge; the centre is unchanged.
    assert torch.allclose(out_12[:, :2], out_10[:, :2])
    assert out_12[0, 2] == pytest.approx(out_10[0, 2] * 1.2)


def test_get_bbx_xys_from_xyxy_resulting_box_has_target_aspect() -> None:
    # After padding, the (w, h) implied by the algorithm matches the 192:256 aspect.
    box = torch.tensor([[100.0, 200.0, 340.0, 460.0]])  # w=240, h=260
    out = hmr_cam.get_bbx_xys_from_xyxy(box)
    size = out[0, 2].item()
    # size is the *enlarged* longer side; the padded square has aspect 1 here
    # because get_bbx_xys takes max(h, w). Pin the exact size instead.
    w, h = 240.0, 260.0
    aspect = 192 / 256
    # w < aspect*h?  240 < 0.75*260=195 -> no.  w > aspect*h? 240>195 -> yes: h=w/aspect=320.
    expected_size = max(320.0, 240.0) * 1.2
    assert size == pytest.approx(expected_size)


def test_get_bbx_xys_from_xyxy_batched() -> None:
    boxes = torch.tensor(
        [
            [100.0, 200.0, 300.0, 500.0],
            [0.0, 0.0, 100.0, 100.0],
            [0.0, 0.0, 400.0, 100.0],
        ]
    )
    out = hmr_cam.get_bbx_xys_from_xyxy(boxes)
    assert out.shape == (3, 3)
    expected = torch.tensor(
        [
            [200.0, 350.0, 360.0],
            [50.0, 50.0, 160.0],
            [200.0, 50.0, 640.0],
        ]
    )
    assert torch.allclose(out, expected)


# ====== perspective_projection ===== #


def test_perspective_projection_matches_golden() -> None:
    pts = torch.tensor([[[[1.0, 2.0, 4.0], [0.0, 0.0, 2.0]]]])  # (1, 1, 2, 3)
    K = hmr_cam.convert_f_to_K(1000.0, 800, 600)[None, None]  # (1, 1, 3, 3)
    out = hmr_cam.perspective_projection(pts, K)
    # x' = f * X/Z + cx, y' = f * Y/Z + cy.
    expected = torch.tensor([[[[650.0, 800.0], [400.0, 300.0]]]])
    assert torch.allclose(out, expected)


def test_perspective_projection_pinhole_formula() -> None:
    # Cross-check against the textbook pinhole formula for random points.
    torch.manual_seed(0)
    pts = torch.randn(2, 3, 5, 3)
    pts = pts + torch.tensor([0.0, 0.0, 5.0])  # push in front of the camera (z>0)
    K = hmr_cam.convert_f_to_K(900.0, 1024, 768)[None, None].expand(2, 3, 3, 3)
    out = hmr_cam.perspective_projection(pts, K)
    fx, fy = K[..., 0, 0], K[..., 1, 1]
    cx, cy = K[..., 0, 2], K[..., 1, 2]
    x = fx[..., None] * pts[..., 0] / pts[..., 2] + cx[..., None]
    y = fy[..., None] * pts[..., 1] / pts[..., 2] + cy[..., None]
    assert torch.allclose(out[..., 0], x, atol=1e-3)
    assert torch.allclose(out[..., 1], y, atol=1e-3)


def test_perspective_projection_centre_ray_hits_principal_point() -> None:
    # A point on the optical axis (X=Y=0) projects to the principal point.
    pts = torch.tensor([[[[0.0, 0.0, 3.0]]]])
    K = hmr_cam.convert_f_to_K(1000.0, 800, 600)[None, None]
    out = hmr_cam.perspective_projection(pts, K)
    assert torch.allclose(out, torch.tensor([[[[400.0, 300.0]]]]))


def test_perspective_projection_scale_invariance_along_ray() -> None:
    # Scaling a point along its viewing ray leaves the 2D projection unchanged.
    K = hmr_cam.convert_f_to_K(800.0, 640, 480)[None, None]
    p = torch.tensor([[[[1.5, -0.5, 2.0]]]])
    out1 = hmr_cam.perspective_projection(p, K)
    out2 = hmr_cam.perspective_projection(p * 3.0, K)
    assert torch.allclose(out1, out2, atol=1e-4)


# ====== compute_transl_full_cam <-> get_a_pred_cam (inverse pair) ===== #


def test_compute_transl_full_cam_matches_golden() -> None:
    pred_cam = torch.tensor([1.2, 0.05, -0.03])
    bbx_xys = torch.tensor([400.0, 300.0, 250.0])
    K = hmr_cam.estimate_K(800, 600)
    transl = hmr_cam.compute_transl_full_cam(pred_cam, bbx_xys, K)
    # With bbx centre == principal point, the (cx, cy) correction vanishes, so
    # tx/ty pass straight through and tz = 2f / (s * b).
    f = K[0, 0]
    expected_tz = 2 * f / (1.2 * 250.0)
    assert torch.allclose(transl, torch.tensor([0.05, -0.03, expected_tz.item()]), atol=1e-4)


def test_pred_cam_transl_roundtrip() -> None:
    # get_a_pred_cam is documented as the inverse of compute_transl_full_cam.
    pred_cam = torch.tensor([[1.2, 0.05, -0.03], [0.8, -0.1, 0.2]])
    bbx_xys = torch.tensor([[400.0, 300.0, 250.0], [350.0, 280.0, 180.0]])
    K = torch.stack([hmr_cam.estimate_K(800, 600), hmr_cam.estimate_K(800, 600)])
    transl = hmr_cam.compute_transl_full_cam(pred_cam, bbx_xys, K)
    recovered = hmr_cam.get_a_pred_cam(transl, bbx_xys, K)
    assert torch.allclose(recovered, pred_cam, atol=1e-4)


def test_transl_pred_cam_roundtrip_offcentre_bbx() -> None:
    # Off-centre bbox so the (cx, cy) correction terms are genuinely exercised.
    transl = torch.tensor([[0.17, -0.42, 4.5]])
    bbx_xys = torch.tensor([[512.0, 211.0, 333.0]])
    K = hmr_cam.estimate_K(1280, 720)[None]
    pred_cam = hmr_cam.get_a_pred_cam(transl, bbx_xys, K)
    transl_back = hmr_cam.compute_transl_full_cam(pred_cam, bbx_xys, K)
    assert torch.allclose(transl_back, transl, atol=1e-3)


def test_compute_transl_full_cam_tz_is_positive_and_depth_like() -> None:
    # tz must be a positive depth, decreasing as the subject scale s grows.
    bbx_xys = torch.tensor([400.0, 300.0, 250.0])
    K = hmr_cam.estimate_K(800, 600)
    tz_small = hmr_cam.compute_transl_full_cam(torch.tensor([0.5, 0.0, 0.0]), bbx_xys, K)[2]
    tz_large = hmr_cam.compute_transl_full_cam(torch.tensor([2.0, 0.0, 0.0]), bbx_xys, K)[2]
    assert tz_small > 0 and tz_large > 0
    assert tz_large < tz_small  # larger apparent scale -> closer to camera
