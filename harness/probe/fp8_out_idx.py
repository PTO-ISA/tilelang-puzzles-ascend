"""Probe: can tilelang allocate a float8_e4m3fn output itself via out_idx?

Answer on tilelang 0.1.15 / CANN 9.2.0-beta.2: **no.**

    MemoryError: Unsupported code 10

Passing an explicitly allocated float8 tensor as a kernel argument works fine, and
so does allocating one directly with torch (`torch.empty(..., dtype=
torch.float8_e4m3fn, device="npu")`). The failure is specific to the
`out_idx` auto-allocation path.

Consequence for this repo: every quantize kernel (per_token, per_block,
per_channel) allocates its FP8/FP4 output in `launch()` and passes it in, rather
than using `out_idx`. That is also how the production kernels are called, so it
costs nothing pedagogically. cast_back is unaffected -- its outputs are bfloat16
or float32.

Re-run this after a tilelang upgrade; if it starts passing, out_idx can be used
uniformly again.

    python harness/probe/fp8_out_idx.py        # needs the simulator or a device
"""

import sys

import torch
import torch_npu  # noqa: F401
import tilelang
import tilelang.ascend.language as T
from tilelang.ascend.language import simd as S

M, K = 8, 128


def build(out_idx):
    @tilelang.jit(target="ascend", out_idx=out_idx)
    def _build():
        @T.prim_func
        def kern(X: T.Tensor((M, K), T.bfloat16), Q: T.Tensor((M, K), T.float8_e4m3fn)):
            with T.Kernel(1):
                x_ub = T.alloc_shared((K,), T.bfloat16)
                q_ub = T.alloc_shared((K,), T.float8_e4m3fn)
                for t in T.serial(M):
                    T.copy(X[t, 0], x_ub)
                    with T.SimdVF():
                        for s in T.serial(K // 64):
                            v = S.vcvt(S.vld(x_ub[s * 64], dist="UNPK_B16"), T.float32)
                            S.vsts(q_ub[s * 64], S.vcvt(v, T.float8_e4m3fn),
                                   dist="PK4_B32")
                    T.copy(q_ub, Q[t, 0])

        return kern

    return _build()


def main() -> int:
    x = torch.randn(M, K, dtype=torch.bfloat16).npu()

    print("[1] torch can allocate float8 on the device:")
    try:
        torch.empty((M, K), dtype=torch.float8_e4m3fn, device="npu")
        print("    OK")
    except Exception as exc:  # noqa: BLE001
        print(f"    FAIL {type(exc).__name__}: {exc}")

    print("[2] tilelang out_idx allocating a float8 output:")
    out_idx_ok = False
    try:
        build([1])(x)
        out_idx_ok = True
        print("    OK  <- this now works; out_idx can be used for FP8 again")
    except Exception as exc:  # noqa: BLE001
        print(f"    FAIL {type(exc).__name__}: {str(exc).splitlines()[0]}")

    print("[3] passing an explicitly allocated float8 output:")
    q = torch.empty((M, K), dtype=torch.float8_e4m3fn, device="npu")
    build(None)(x, q)
    nonzero = q.cpu().float().abs().max().item() > 0
    print(f"    OK, kernel wrote values: {nonzero}")

    print(f"\nout_idx float8 support: {'YES' if out_idx_ok else 'NO'}")
    print("PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
