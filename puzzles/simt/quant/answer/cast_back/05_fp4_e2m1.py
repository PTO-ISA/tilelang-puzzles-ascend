"""cast_back 05 (SIMT). See doc/quant/cast_back/05_fp4_e2m1.md"""

import torch
import tilelang
import tilelang.ascend.language as T

from harness import status
from harness.consts import CANONICAL_G

THREADS = 128


@tilelang.jit(target="ascend", out_idx=[2])
def compile_kernel(hidden: int, group_size: int = CANONICAL_G):
    """Dequantize packed FP4 (e2m1) -> bfloat16.

    The one place the SIMT prim_func signature differs from ASC and PTO: they
    declare `Q` as `float4_e2m1fn` and let a hardware convert widen it, but that
    path is not reachable from a SimtVF region. So the kernel takes the raw
    bytes as `uint8` and decodes the nibbles itself. `launch` has the same
    signature as the other tiers, so the harness cannot tell.
    """
    assert hidden % 128 == 0 and group_size == 32
    num_groups = hidden // group_size
    num_bytes = hidden // 2                 # two e2m1 values per byte
    num_tokens = T.dynamic("num_tokens")

    @T.prim_func
    def cast_back(
        Q: T.Tensor((num_tokens, num_bytes), T.uint8),
        Sf: T.Tensor((num_tokens, num_groups), T.float32),
        Out: T.Tensor((num_tokens, hidden), T.bfloat16),
    ):
        with T.Kernel(1):
            q_ub = T.alloc_shared((num_bytes,), T.uint8)
            sf_ub = T.alloc_shared((num_groups,), T.float32)
            out_ub = T.alloc_shared((hidden,), T.bfloat16)

            for token in T.serial(num_tokens):
                T.copy(Q[token, 0], q_ub)
                T.copy(Sf[token, 0], sf_ub)

                # --- BEGIN SOLUTION hint="decode the nibbles by hand -- there is no FP4 convert reachable from SIMT. One T.Parallel over hidden: byte = T.Cast('int32', q_ub[i // 2]); nib = (byte >> (4 * (i % 2))) & 0x0F, low nibble first. Split it as sgn = (nib >> 3) & 1, e = (nib >> 1) & 3, m = nib & 1. With e == 0 the value is subnormal, m * 0.5; otherwise it is (1 + m*0.5) * 2**(e-1), and that power of two is T.reinterpret('float32', (126 + e) << 23). Pick between them with T.if_then_else, apply the sign the same way, then scale by sf_ub[i // group_size]."
                with T.SimtVF(threads=THREADS):
                    o_l = T.alloc_fragment((hidden,), T.float32)

                    for i in T.Parallel(hidden):
                        byte = T.Cast("int32", q_ub[i // 2])
                        nib = (byte >> (4 * (i % 2))) & 0x0F
                        sgn = (nib >> 3) & 1
                        e = (nib >> 1) & 3
                        m = T.Cast("float32", nib & 1)
                        # e == 0 is the subnormal row: 0 or 0.5, no implicit 1.
                        sub = m * 0.5
                        nrm = (1.0 + m * 0.5) * T.reinterpret("float32", (126 + e) << 23)
                        mag = T.if_then_else(e == 0, sub, nrm)
                        val = T.if_then_else(sgn == 1, -mag, mag)
                        o_l[i] = val * sf_ub[i // group_size]

                    T.copy(o_l, out_ub)
                # --- END SOLUTION

                T.copy(out_ub, Out[token, 0])

    return cast_back


def launch(q_packed: torch.Tensor, sf: torch.Tensor) -> torch.Tensor:
    """`q_packed` is (M, K/2) int8 -- the SIMT kernel reads it as raw bytes."""
    hidden = q_packed.shape[1] * 2
    out = compile_kernel(hidden)(q_packed.view(torch.uint8), sf)
    status.assert_on_device("cast_back 05", out)
    return out

if __name__ == "__main__":        # not a script -- see the module docstring
    raise SystemExit("Run it through the harness:  python -m harness.check simt/cast_back/05")
