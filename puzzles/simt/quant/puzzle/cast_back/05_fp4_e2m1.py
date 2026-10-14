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

                # TODO: decode the nibbles by hand -- there is no FP4 convert
                #       reachable from SIMT. One T.Parallel over hidden: byte =
                #       T.Cast('int32', q_ub[i // 2]); nib = (byte >> (4 * (i %
                #       2))) & 0x0F, low nibble first. Split it as sgn = (nib >>
                #       3) & 1, e = (nib >> 1) & 3, m = nib & 1. With e == 0 the
                #       value is subnormal, m * 0.5; otherwise it is (1 + m*0.5) *
                #       2**(e-1), and that power of two is
                #       T.reinterpret('float32', (126 + e) << 23). Pick between
                #       them with T.if_then_else, apply the sign the same way,
                #       then scale by sf_ub[i // group_size].
                raise NotImplementedError("simt/cast_back/05_fp4_e2m1: implement cast_back")

                T.copy(out_ub, Out[token, 0])

    return cast_back


def launch(q_packed: torch.Tensor, sf: torch.Tensor) -> torch.Tensor:
    """`q_packed` is (M, K/2) int8 -- the SIMT kernel reads it as raw bytes."""
    hidden = q_packed.shape[1] * 2
    out = compile_kernel(hidden)(q_packed.view(torch.uint8), sf)
    status.assert_on_device("cast_back 05", out)
    return out

if __name__ == "__main__":        # not a script -- see the module docstring
    raise SystemExit("Run it through the harness:  python -m harness.check simt/cast_back/05 --role puzzle")
