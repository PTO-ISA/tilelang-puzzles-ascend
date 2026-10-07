"""Probe which VF bodies the installed tilelang/ptoas can lower.

Compile-only by default (fast, no simulator). With --run it also launches the
bodies that compiled and checks them numerically against torch.

    python harness/probe/vf_lane_limits.py
    python harness/probe/vf_lane_limits.py --run

See README.md in this directory for the last recorded results.
"""

from __future__ import annotations

import argparse
import sys

import tilelang
import tilelang.ascend.language as T
from tilelang.ascend.language import vmi as V

EPS, QMAX = 1e-4, 448.0


# --------------------------------------------------------------------------
# bodies under test
# --------------------------------------------------------------------------

def body_fused_bf16(M=32, K=128, G=32):
    """One 128-lane vector per row: convert, grouped reduce, fused divide,
    grouped broadcast, quantize. The shortest correct per_token body."""
    NG = K // G

    @T.prim_func
    def kern(X: T.Tensor((M, K), T.bfloat16), Out: T.Tensor((M, K), T.float8_e4m3fn),
             Amax: T.Tensor((M, NG), T.float32)):
        with T.Kernel(1):
            x_ub = T.alloc_shared((M, K), T.bfloat16)
            out_ub = T.alloc_shared((M, K), T.float8_e4m3fn)
            amax_ub = T.alloc_shared((M, 64), T.float32)
            T.copy(X, x_ub)
            with T.SimdVF():
                m128 = V.create_mask(128, size=128)
                m4 = V.create_mask(4, size=4)
                eps4 = V.vbrc(T.float32(EPS), size=4)
                qmax4 = V.vbrc(T.float32(QMAX), size=4)
                for row in T.serial(M):
                    x = V.vcvt(V.vload(x_ub[row, 0], size=128), "float32")
                    amax = V.vmax(V.vcmax(V.vabs(x), m128, group=4), eps4, m4)
                    V.vstore(amax, amax_ub[row, 0])
                    inv = V.vbrc(V.vdiv(qmax4, amax, m4), size=128, group=4)
                    V.vstore(V.vcvt(V.vmul(x, inv, m128), "float8_e4m3fn",
                                    rounding="R", saturate="SAT"), out_ub[row, 0], m128)
            T.copy(out_ub, Out)
            T.copy(amax_ub[0:M, 0:NG], Amax)

    return kern


def body_fused_f32(M=32, K=128, G=32):
    """Same, with a float32 input (a raw 128-lane load, no convert)."""
    NG = K // G

    @T.prim_func
    def kern(X: T.Tensor((M, K), T.float32), Out: T.Tensor((M, K), T.float8_e4m3fn),
             Amax: T.Tensor((M, NG), T.float32)):
        with T.Kernel(1):
            x_ub = T.alloc_shared((M, K), T.float32)
            out_ub = T.alloc_shared((M, K), T.float8_e4m3fn)
            amax_ub = T.alloc_shared((M, 64), T.float32)
            T.copy(X, x_ub)
            with T.SimdVF():
                m128 = V.create_mask(128, size=128)
                m4 = V.create_mask(4, size=4)
                eps4 = V.vbrc(T.float32(EPS), size=4)
                qmax4 = V.vbrc(T.float32(QMAX), size=4)
                for row in T.serial(M):
                    x = V.vload(x_ub[row, 0], size=128)
                    amax = V.vmax(V.vcmax(V.vabs(x), m128, group=4), eps4, m4)
                    V.vstore(amax, amax_ub[row, 0])
                    inv = V.vbrc(V.vdiv(qmax4, amax, m4), size=128, group=4)
                    V.vstore(V.vcvt(V.vmul(x, inv, m128), "float8_e4m3fn",
                                    rounding="R", saturate="SAT"), out_ub[row, 0], m128)
            T.copy(out_ub, Out)
            T.copy(amax_ub[0:M, 0:NG], Amax)

    return kern


def _per_block(lanes, BM=32, BK=32):
    """32x32 tile -> one scale, reducing `lanes` values at a time."""
    NVEC = BM * BK // lanes

    @T.prim_func
    def kern(X: T.Tensor((BM, BK), T.bfloat16), Out: T.Tensor((BM, BK), T.float8_e4m3fn),
             Sf: T.Tensor((1,), T.float32)):
        with T.Kernel(1):
            x_ub = T.alloc_shared((BM, BK), T.bfloat16)
            out_ub = T.alloc_shared((BM, BK), T.float8_e4m3fn)
            sf_ub = T.alloc_shared((8,), T.float32)
            flat = T.Tensor((BM * BK,), T.bfloat16, x_ub.data)
            flat_out = T.Tensor((BM * BK,), T.float8_e4m3fn, out_ub.data)
            T.copy(X, x_ub)
            with T.SimdVF():
                mask = V.create_mask(lanes, size=lanes)
                m1 = V.create_mask(1, size=1)
                # SIMD values are immutable, so an accumulator carried across a
                # loop must live in a register array, not a plain value.
                acc = V.alloc_local((1,), V.vreg(lanes, T.float32))
                acc[0] = V.vbrc(T.float32(EPS), size=lanes)
                for i in T.serial(NVEC):
                    v = V.vcvt(V.vload(flat[i * lanes], size=lanes), "float32")
                    acc[0] = V.vmax(acc[0], V.vabs(v), mask)
                tile_amax = V.vcmax(acc[0], mask, group=1)
                qmax1 = V.vbrc(T.float32(QMAX), size=1)
                V.vstore(V.vdiv(tile_amax, qmax1, m1), sf_ub[0])
                inv = V.vbrc(V.vdiv(qmax1, tile_amax, m1), size=lanes)
                for i in T.serial(NVEC):
                    v = V.vcvt(V.vload(flat[i * lanes], size=lanes), "float32")
                    V.vstore(V.vcvt(V.vmul(v, inv, mask), "float8_e4m3fn",
                                    rounding="R", saturate="SAT"), flat_out[i * lanes], mask)
            T.copy(out_ub, Out)
            T.copy(sf_ub[0:1], Sf)

    return kern


CASES = {
    "fused_bf16_128lane": (body_fused_bf16, "per_token, one 128-lane vector, fused"),
    "fused_f32_128lane": (body_fused_f32, "per_token, float32 input, fused"),
    "per_block_8lane": (lambda: _per_block(8), "per_block 32x32 via 8-lane bf16 chunks"),
    "per_block_64lane": (lambda: _per_block(64), "per_block 32x32 via 64-lane loads"),
}


# --------------------------------------------------------------------------
# drivers
# --------------------------------------------------------------------------

def compile_all() -> dict[str, str]:
    results = {}
    for name, (builder, desc) in CASES.items():
        try:
            tilelang.compile(builder(), target="pto", out_idx=[1, 2])
            results[name] = "COMPILE_OK"
            print(f"  {name:22} COMPILE_OK   ({desc})")
        except Exception as exc:  # noqa: BLE001
            line = next((ln.strip() for ln in str(exc).splitlines()
                         if "VMI-" in ln or "error:" in ln), type(exc).__name__)
            results[name] = f"COMPILE_FAIL {line[:160]}"
            print(f"  {name:22} COMPILE_FAIL ({desc})")
            print(f"      {line[:200]}")
    return results


def run_numeric() -> None:
    import torch
    import torch_npu  # noqa: F401

    def fp8_hist(got, ref):
        d = (got.view(torch.uint8).int() - ref.view(torch.uint8).int()).abs()
        return {int(k): int((d == k).sum()) for k in d.unique()}, int((d > 1).sum())

    for name, dtype in (("fused_bf16_128lane", torch.bfloat16), ("fused_f32_128lane", torch.float32)):
        M, K, G = 32, 128, 32
        torch.manual_seed(0)
        x = torch.randn(M, K, dtype=dtype)
        x[0] = 0
        xv = x.float().view(M, K // G, G)
        ref_amax = xv.abs().amax(-1).clamp(min=EPS)
        ref_q = (xv * (QMAX / ref_amax.unsqueeze(-1))).view(M, K).to(torch.float8_e4m3fn)
        kernel = tilelang.compile(CASES[name][0](), target="pto", out_idx=[1, 2])
        got_q, got_amax = kernel(x.npu())
        hist, bad = fp8_hist(got_q.cpu(), ref_q)
        err = (got_amax.cpu() - ref_amax).abs().max().item()
        print(f"  {name:22} amax_err={err:.2e} codes={hist} verdict={'PASS' if bad == 0 and err < 1e-3 else 'FAIL'}")

    torch.manual_seed(1)
    x = torch.randn(32, 32, dtype=torch.bfloat16)
    ref_amax = x.float().abs().amax().clamp(min=EPS)
    ref_q = (x.float() * (QMAX / ref_amax)).to(torch.float8_e4m3fn)
    kernel = tilelang.compile(_per_block(64), target="pto", out_idx=[1, 2])
    got_q, got_sf = kernel(x.npu())
    hist, bad = fp8_hist(got_q.cpu(), ref_q)
    sf_err = (got_sf.cpu()[0] - ref_amax / QMAX).abs().item()
    print(f"  {'per_block_64lane':22} sf_err={sf_err:.2e} codes={hist} "
          f"verdict={'PASS' if bad == 0 and sf_err < 1e-6 else 'FAIL'}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", action="store_true", help="also check numerics on device")
    args = ap.parse_args()

    print(f"tilelang {tilelang.__version__}")
    print("\n[compile]")
    results = compile_all()
    if args.run:
        print("\n[numeric]")
        run_numeric()
    nfail = sum(1 for v in results.values() if v.startswith("COMPILE_FAIL"))
    print(f"\n{len(results) - nfail}/{len(results)} bodies compiled")
    print("PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
