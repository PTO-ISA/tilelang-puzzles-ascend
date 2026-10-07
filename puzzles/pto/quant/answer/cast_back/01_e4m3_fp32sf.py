"""cast_back 01 (PTO). See doc/quant/cast_back/01_e4m3_fp32sf.md"""

import torch
import tilelang
import tilelang.ascend.language as T
from tilelang.ascend.language import vmi as V

from common import status
from common.consts import CANONICAL_G

LANES = 64
SF_PAD = 64


@tilelang.jit(target="pto", out_idx=[2])
def compile_kernel(hidden: int, group_size: int = CANONICAL_G):
    """Dequantize FP8 -> bfloat16 with one FP32 scale per `group_size` channels."""
    assert hidden % 128 == 0, "this teaching kernel steps two 64-lane strips at a time"
    assert group_size == 32, "Ascend quant granularity is fixed at 32"
    num_groups = hidden // group_size
    groups_per_strip = LANES // group_size      # 2
    num_tokens = T.dynamic("num_tokens")

    @T.prim_func
    def cast_back(
        Q: T.Tensor((num_tokens, hidden), T.float8_e4m3fn),
        Sf: T.Tensor((num_tokens, num_groups), T.float32),
        Out: T.Tensor((num_tokens, hidden), T.bfloat16),
    ):
        with T.Kernel(1):
            q_ub = T.alloc_shared((hidden,), T.float8_e4m3fn)
            sf_ub = T.alloc_shared((SF_PAD,), T.float32)
            out_ub = T.alloc_shared((hidden,), T.bfloat16)

            for token in T.serial(num_tokens):
                T.copy(Q[token, 0], q_ub)
                T.copy(Sf[token, 0], sf_ub[0:num_groups])

                # --- BEGIN SOLUTION hint="open `with T.SimdVF():`; mask = V.create_mask(64, size=64); loop strip over hidden//64; values = V.vcvt(V.vload(q_ub[col], size=64), 'float32'); scale = V.vload(sf_ub[group], size=64, stride=1, dist_mode='brc', group=2) -- one load, no select; then V.vmul and V.vstore(V.vcvt(scaled, 'bfloat16'), out_ub[col])"
                with T.SimdVF():
                    mask = V.create_mask(LANES, size=LANES)
                    for strip in T.serial(hidden // LANES):
                        col = strip * LANES
                        group = strip * groups_per_strip

                        # Load 64 FP8 values and widen. No distribution-mode
                        # catalogue to consult: the convert says what it does.
                        values = V.vcvt(V.vload(q_ub[col], size=LANES), "float32")

                        # One load builds the whole scale vector: 64 lanes, two
                        # segments, each broadcast from a consecutive scalar.
                        # The ASC file needs two loads and a select for this.
                        scale = V.vload(sf_ub[group], size=LANES, stride=1,
                                        dist_mode="brc", group=groups_per_strip)

                        scaled = V.vmul(values, scale, mask)
                        V.vstore(V.vcvt(scaled, "bfloat16"), out_ub[col])
                # --- END SOLUTION

                T.copy(out_ub, Out[token, 0])

    return cast_back


def launch(q: torch.Tensor, sf: torch.Tensor) -> torch.Tensor:
    """Run the kernel. Returns a device tensor -- never a host-computed answer."""
    kernel = compile_kernel(q.shape[1])
    out = kernel(q, sf)
    status.assert_on_device("cast_back 01", out)
    return out
