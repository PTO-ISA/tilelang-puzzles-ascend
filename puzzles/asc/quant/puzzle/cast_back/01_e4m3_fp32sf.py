"""cast_back 01 (ASC). See doc/quant/cast_back/01_e4m3_fp32sf.md"""

import torch
import tilelang
import tilelang.ascend.language as T
from tilelang.ascend.language import simd as S

from common import status
from common.consts import CANONICAL_G

LANES = 64          # float32 lanes in one 256-byte vector register
SF_PAD = 64         # pad the scale buffer out to a whole register


@tilelang.jit(target="ascend", out_idx=[2])
def compile_kernel(hidden: int, group_size: int = CANONICAL_G):
    """Dequantize FP8 -> bfloat16 with one FP32 scale per `group_size` channels."""
    assert hidden % 128 == 0, "this teaching kernel steps two 64-lane strips at a time"
    assert group_size == 32, "Ascend quant granularity is fixed at 32"
    num_groups = hidden // group_size
    num_tokens = T.dynamic("num_tokens")

    @T.prim_func
    def cast_back(
        Q: T.Tensor((num_tokens, hidden), T.float8_e4m3fn),
        Sf: T.Tensor((num_tokens, num_groups), T.float32),
        Out: T.Tensor((num_tokens, hidden), T.bfloat16),
    ):
        # One vector core. Production runs T.Persistent over many cores; that is
        # scheduling, and it would make the simulator far slower without
        # teaching anything new about the vector unit.
        with T.Kernel(1):
            q_ub = T.alloc_shared((hidden,), T.float8_e4m3fn)
            sf_ub = T.alloc_shared((SF_PAD,), T.float32)
            out_ub = T.alloc_shared((hidden,), T.bfloat16)

            for token in T.serial(num_tokens):
                # GM -> UB, on the DMA engine.
                T.copy(Q[token, 0], q_ub)
                T.copy(Sf[token, 0], sf_ub[0:num_groups])

                # TODO: open `with T.SimdVF():`; make a mask with S.pset(32,
                #       'PAT_VL32'); loop strip over hidden//64; load 64 FP8
                #       values with S.vld(q_ub[col], dist='UNPK4_B8') and S.vcvt
                #       to float32; build the scale with two S.vld(...,
                #       dist='BRC_B32') and S.vsel(lo, hi, mask_low); S.vmul;
                #       store with S.vsts(..., S.vcvt(x, T.bfloat16),
                #       dist='PK_B32')
                raise NotImplementedError("asc/cast_back/01_e4m3_fp32sf: implement cast_back")

                # UB -> GM.
                T.copy(out_ub, Out[token, 0])

    return cast_back


def launch(q: torch.Tensor, sf: torch.Tensor) -> torch.Tensor:
    """Run the kernel. Returns a device tensor -- never a host-computed answer."""
    kernel = compile_kernel(q.shape[1])
    out = kernel(q, sf)
    status.assert_on_device("cast_back 01", out)
    return out
