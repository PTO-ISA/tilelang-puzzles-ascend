"""cast_back 02 (PTO). See doc/quant/cast_back/02_fp32_out.md"""

import torch
import tilelang
import tilelang.ascend.language as T
from tilelang.ascend.language import vmi as V

from harness import status
from harness.consts import CANONICAL_G

LANES = 64
SF_PAD = 64


@tilelang.jit(target="pto", out_idx=[2])
def compile_kernel(hidden: int, group_size: int = CANONICAL_G):
    """Dequantize FP8 -> float32."""
    assert hidden % 128 == 0 and group_size == 32
    num_groups = hidden // group_size
    num_tokens = T.dynamic("num_tokens")

    @T.prim_func
    def cast_back(
        Q: T.Tensor((num_tokens, hidden), T.float8_e4m3fn),
        Sf: T.Tensor((num_tokens, num_groups), T.float32),
        Out: T.Tensor((num_tokens, hidden), T.float32),
    ):
        with T.Kernel(1):
            q_ub = T.alloc_shared((hidden,), T.float8_e4m3fn)
            sf_ub = T.alloc_shared((SF_PAD,), T.float32)
            out_ub = T.alloc_shared((hidden,), T.float32)
            for token in T.serial(num_tokens):
                T.copy(Q[token, 0], q_ub)
                T.copy(Sf[token, 0], sf_ub[0:num_groups])
                # --- BEGIN SOLUTION hint="same as variant 01, except V.vstore(scaled, out_ub[col]) with no vcvt -- the destination buffer is float32 so VMI stores 32-bit lanes directly"
                with T.SimdVF():
                    mask = V.create_mask(LANES, size=LANES)
                    groups_per_strip = LANES // group_size
                    for strip in T.serial(hidden // LANES):
                        col = strip * LANES
                        group = strip * groups_per_strip
                        values = V.vcvt(V.vload(q_ub[col], size=LANES), "float32")
                        scale = V.vload(sf_ub[group], size=LANES, stride=1,
                                        dist_mode="brc", group=groups_per_strip)
                        scaled = V.vmul(values, scale, mask)
                        # Identical spelling to variant 01's store; only the
                        # convert is gone, because out_ub is already float32.
                        V.vstore(scaled, out_ub[col])
                # --- END SOLUTION
                T.copy(out_ub, Out[token, 0])

    return cast_back


def launch(q: torch.Tensor, sf: torch.Tensor) -> torch.Tensor:
    out = compile_kernel(q.shape[1])(q, sf)
    status.assert_on_device("cast_back 02", out)
    return out

if __name__ == "__main__":        # not a script -- see the module docstring
    raise SystemExit("Run it through the harness:  python -m harness.check pto/cast_back/02")
