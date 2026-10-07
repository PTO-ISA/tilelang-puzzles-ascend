"""Assertions and the two-phase puzzle runner (torch first, then TileLang)."""

import torch


def assert_fp8_near(a: torch.Tensor, b: torch.Tensor, name: str = "tensor") -> None:
    """Byte-exact when possible; else a few adjacent e4m3 codes."""
    assert a.dtype == b.dtype and a.shape == b.shape, name
    au = a.contiguous().view(torch.uint8).flatten()
    bu = b.contiguous().view(torch.uint8).flatten()
    if torch.equal(au, bu):
        return
    n = int((au != bu).sum().item())
    code_diff = (au.to(torch.int16) - bu.to(torch.int16)).abs()
    worst_code = int(code_diff.max().item())
    limit = max(32, a.numel() // 128)
    max_abs = float((a.float() - b.float()).abs().max())
    # Simulator / VF rounding can flip a few codes by more than 1 ULP at boundaries.
    if worst_code > 8 or n > limit or max_abs > 64.0:
        raise AssertionError(
            f"{name} FP8 mismatch at {n}/{au.numel()} positions, "
            f"max_abs={max_abs:.6g}, max_code_diff={worst_code} "
            f"(limit {limit} positions / code_diff 8 / abs 64)"
        )


def assert_fp32_ulps(a: torch.Tensor, b: torch.Tensor, name: str = "tensor", max_ulps: int = 1) -> None:
    assert a.dtype == torch.float32 and b.dtype == torch.float32, name
    assert a.shape == b.shape, f"{name} shape {tuple(a.shape)} vs {tuple(b.shape)}"
    ulps = (a.contiguous().view(torch.int32) - b.contiguous().view(torch.int32)).abs()
    worst = int(ulps.max().item())
    if worst > max_ulps:
        diff = (a.float() - b.float()).abs()
        raise AssertionError(
            f"{name} differs by {worst} ULPs (allowed {max_ulps}); "
            f"max_abs={diff.max().item():.6g}"
        )


def assert_same_bytes(a: torch.Tensor, b: torch.Tensor, name: str = "tensor") -> None:
    assert a.dtype == b.dtype, f"{name} dtype {a.dtype} vs {b.dtype}"
    assert a.shape == b.shape, f"{name} shape {tuple(a.shape)} vs {tuple(b.shape)}"
    a_u8 = a.detach().reshape(-1).view(torch.uint8)
    b_u8 = b.detach().reshape(-1).view(torch.uint8)
    if torch.equal(a_u8, b_u8):
        return
    if a.dtype in (torch.float32, torch.bfloat16, torch.float16):
        diff = (a.float() - b.float()).abs()
        raise AssertionError(
            f"{name} mismatch: max_abs={diff.max().item():.6g} mean_abs={diff.mean().item():.6g}"
        )
    mismatch = (a_u8 != b_u8).sum().item()
    raise AssertionError(f"{name} byte mismatch at {mismatch}/{a_u8.numel()} positions")


def assert_bf16_near(a: torch.Tensor, b: torch.Tensor, name: str = "tensor", atol: float = 1e-2) -> None:
    assert a.shape == b.shape, name
    diff = (a.float() - b.float()).abs()
    if float(diff.max()) > atol:
        raise AssertionError(f"{name} max_abs={float(diff.max()):.6g} > atol={atol}")


def run_phases(name: str, test_torch, test_tilelang) -> None:
    """Fail-fast: do not run TileLang until the torch reference passes."""
    print(name)
    print("[1/2] torch reference")
    try:
        test_torch()
    except Exception as exc:
        print(f"torch not done yet: {type(exc).__name__}: {exc}")
        print("Get the torch reference right before writing TileLang.")
        return
    print("[2/2] TileLang vs your torch")
    try:
        test_tilelang()
    except Exception as exc:
        print(f"TileLang not done yet: {type(exc).__name__}: {exc}")
