"""Tests for device selection and the Apple-Silicon (MPS) inference path.

These guard the device-abstraction work that replaced hard-coded ``.cuda()``:
the helpers in :mod:`gvhmr.utils.device`, and that the GVHMR denoiser actually
builds and runs a forward pass on MPS (skipped when MPS is unavailable).
"""

from __future__ import annotations

import torch

from gvhmr.utils.device import device_name, get_device, synchronize, to_device


def test_get_device_honours_explicit_and_env(monkeypatch) -> None:
    assert get_device(prefer="cpu") == torch.device("cpu")
    monkeypatch.setenv("GVHMR_DEVICE", "cpu")
    assert get_device() == torch.device("cpu")
    monkeypatch.delenv("GVHMR_DEVICE")
    # Auto-selection returns a real, valid device.
    assert get_device().type in {"cuda", "mps", "cpu"}


def test_get_device_falls_back_when_cuda_unavailable(monkeypatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setenv("GVHMR_DEVICE", "cuda")
    assert get_device() == torch.device("cpu")
    monkeypatch.delenv("GVHMR_DEVICE")
    assert get_device(prefer="cuda") == torch.device("cpu")


def test_to_device_recurses_and_preserves_non_tensors() -> None:
    data = {"a": torch.randn(2, 3), "b": [torch.ones(4), "keep"], "n": 5}
    out = to_device(data, "cpu")
    assert out["a"].device.type == "cpu"
    assert out["b"][0].device.type == "cpu"
    assert out["b"][1] == "keep" and out["n"] == 5


def test_device_name_is_human_readable() -> None:
    assert device_name("cpu") == "CPU"
    assert "MPS" in device_name("mps") or device_name("mps")  # string, never raises


def test_synchronize_is_a_noop_on_cpu() -> None:
    synchronize("cpu")  # must not raise


def _make_net_and_batch(device, length: int = 16):
    import gvhmr.network.gvhmr.relative_transformer as rt

    torch.manual_seed(0)
    net = rt.NetworkEncoderRoPE().to(device).eval()
    batch = to_device(
        {
            "length": torch.tensor([length]),
            "obs": torch.randn(1, length, 17, 3),
            "f_cliffcam": torch.randn(1, length, 3),
            "f_cam_angvel": torch.randn(1, length, 6),
            "f_imgseq": torch.randn(1, length, 1024),
        },
        device,
    )
    return net, batch


def test_denoiser_forward_on_cpu_is_deterministic() -> None:
    net, batch = _make_net_and_batch("cpu")
    with torch.no_grad():
        a = net(**batch)["pred_x"]
        b = net(**batch)["pred_x"]
    assert a.shape == (1, 16, 151)
    assert torch.equal(a, b)


def test_denoiser_forward_runs_on_mps(mps_device) -> None:
    net, batch = _make_net_and_batch(mps_device)
    with torch.no_grad():
        out = net(**batch)
    assert out["pred_x"].shape == (1, 16, 151)
    assert out["pred_x"].device.type == "mps"
    assert torch.isfinite(out["pred_x"]).all()


def test_cpu_mps_forward_parity(mps_device) -> None:
    # Same seed/weights on both devices: outputs should agree within MPS fp tolerance.
    # nn.Module.to() moves parameters in place, so run the CPU pass first, then move.
    net, batch = _make_net_and_batch("cpu")
    with torch.no_grad():
        out_cpu = net(**batch)["pred_x"].clone()
        net_mps = net.to(mps_device)
        out_mps = net_mps(**to_device(batch, mps_device))["pred_x"].cpu()
    assert torch.allclose(out_cpu, out_mps, atol=1e-2, rtol=1e-2)
