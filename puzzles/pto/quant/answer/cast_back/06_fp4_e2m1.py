"""cast_back 06 (PTO). See doc/quant/cast_back/06_fp4_e2m1.md"""

import torch
import tilelang
import tilelang.ascend.language as T
from tilelang.ascend.language import vmi as V

from common import status
from common.consts import CANONICAL_G

FP4_STRIP = 128


@tilelang.jit(target="pto", out_idx=[2])
def compile_kernel(hidden: int, group_size: int = CANONICAL_G):
    """Dequantize packed FP4 -> bfloat16 with one FP32 scale per 32 channels."""
    assert hidden % FP4_STRIP == 0 and group_size == 32
    num_groups = hidden // group_size
    groups_per_strip = FP4_STRIP // group_size      # 4
    num_tokens = T.dynamic("num_tokens")
    sf_pad = max(num_groups, 64)

    @T.prim_func
    def cast_back(
        Q: T.Tensor((num_tokens, hidden), T.float4_e2m1fn),
        Sf: T.Tensor((num_tokens, num_groups), T.float32),
        Out: T.Tensor((num_tokens, hidden), T.bfloat16),
    ):
        with T.Kernel(1):
            q_ub = T.alloc_shared((hidden,), T.float4_e2m1fn)
            sf_ub = T.alloc_shared((sf_pad,), T.float32)
            out_ub = T.alloc_shared((hidden,), T.bfloat16)
            for token in T.serial(num_tokens):
                T.copy(Q[token, 0], q_ub)
                T.copy(Sf[token, 0], sf_ub[0:num_groups])
                # --- BEGIN SOLUTION hint="stay at 128 lanes throughout: x_f32 = V.vzip(V.vbrc(T.bfloat16(0.0), size=128), V.vcvt(V.vload(q_ub[col], size=128), 'bfloat16'), 'float32'); scale = V.vload(sf_ub[group], size=128, stride=1, dist_mode='brc', group=4); then one vmul and one vstore -- no splitting into 64-lane halves"
                with T.SimdVF():
                    mask = V.create_mask(FP4_STRIP, size=FP4_STRIP)
                    zero_bf16 = V.vbrc(T.bfloat16(0.0), size=FP4_STRIP)
                    for strip in T.serial(hidden // FP4_STRIP):
                        col = strip * FP4_STRIP
                        group = strip * groups_per_strip

                        # 128 FP4 -> 128 bfloat16 -> 128 float32, one name each.
                        x_bf16 = V.vcvt(V.vload(q_ub[col], size=FP4_STRIP), "bfloat16")
                        x_f32 = V.vzip(zero_bf16, x_bf16, "float32")

                        # group=4 instead of group=2: the width is an argument.
                        scale = V.vload(sf_ub[group], size=FP4_STRIP, stride=1,
                                        dist_mode="brc", group=groups_per_strip)

                        scaled = V.vmul(x_f32, scale, mask)
                        V.vstore(V.vcvt(scaled, "bfloat16"), out_ub[col])
                # --- END SOLUTION
                T.copy(out_ub, Out[token, 0])

    return cast_back


def launch(q_packed: torch.Tensor, sf: torch.Tensor) -> torch.Tensor:
    hidden = q_packed.shape[1] * 2
    q = q_packed.view(torch.uint8).view(torch.float4_e2m1fn_x2)
    out = compile_kernel(hidden)(q, sf)
    status.assert_on_device("cast_back 06", out)
    return out
