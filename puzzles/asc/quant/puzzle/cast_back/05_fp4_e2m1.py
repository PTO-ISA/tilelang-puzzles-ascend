"""cast_back 05 (ASC). See doc/quant/cast_back/04_fp4_e2m1.md"""

import torch
import tilelang
import tilelang.ascend.language as T
from tilelang.ascend.language import simd as S

from harness import status
from harness.consts import CANONICAL_G

LANES = 64          # float32 lanes per register
FP4_STRIP = 128     # FP4 values produced by one unpacking load


@tilelang.jit(target="ascend", out_idx=[2])
def compile_kernel(hidden: int, group_size: int = CANONICAL_G):
    """Dequantize packed FP4 -> bfloat16 with one FP32 scale per 32 channels."""
    assert hidden % FP4_STRIP == 0, "the FP4 path steps 128 values at a time"
    assert group_size == 32
    num_groups = hidden // group_size
    num_tokens = T.dynamic("num_tokens")
    sf_pad = max(num_groups, LANES)

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
                # TODO: one S.vld(q_ub[col], dist='UNPK4_B8') yields 128 FP4
                #       values; S.vcvt them to bfloat16; widen+split with x_lo,
                #       x_hi = S.vintlv(S.vdup(0.0, T.bfloat16), x_bf16) and
                #       T.reinterpret each half to 'float32x64'; build two scale
                #       vectors (groups g,g+1 for the low half and g+2,g+3 for the
                #       high) the same way as variant 01, multiply, and store both
                #       halves
                raise NotImplementedError("asc/cast_back/05_fp4_e2m1: implement cast_back")
                T.copy(out_ub, Out[token, 0])

    return cast_back


def launch(q_packed: torch.Tensor, sf: torch.Tensor) -> torch.Tensor:
    """`q_packed` is (M, K/2) int8; tilelang wants it as float4_e2m1fn_x2."""
    hidden = q_packed.shape[1] * 2
    q = q_packed.view(torch.uint8).view(torch.float4_e2m1fn_x2)
    out = compile_kernel(hidden)(q, sf)
    status.assert_on_device("cast_back 05", out)
    return out

if __name__ == "__main__":        # not a script -- see the module docstring
    raise SystemExit("Run it through the harness:  python -m harness.check asc/cast_back/05")
