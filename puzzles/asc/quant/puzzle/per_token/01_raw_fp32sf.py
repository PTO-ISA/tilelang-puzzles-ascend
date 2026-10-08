"""per_token 01 (ASC). See doc/quant/per_token/01_raw_fp32sf.md"""

import torch
import tilelang
import tilelang.ascend.language as T
from tilelang.ascend.language import simd as S

from harness import status
from harness.consts import CANONICAL_G, E4M3_CLAMP_MIN, E4M3_MAX

LANES = 64
PAIR = 128          # two 64-lane strips: the smallest unit holding 4 groups
SF_PAD = 64


# No out_idx: tilelang's automatic output allocation rejects float8_e4m3fn with
# "MemoryError: Unsupported code 10" on this toolchain, so launch() allocates the
# outputs and passes them in. See harness/probe/fp8_out_idx.py, which reproduces
# the failure and will start reporting YES if a later tilelang fixes it. This is
# also how the production kernels are called.
@tilelang.jit(target="ascend")
def compile_kernel(hidden: int, group_size: int = CANONICAL_G):
    """Quantize bfloat16 -> FP8 e4m3 with one FP32 scale per 32 channels."""
    assert hidden % PAIR == 0, "this kernel steps 128 channels (4 groups) at a time"
    assert group_size == 32
    num_groups = hidden // group_size
    assert num_groups <= SF_PAD, "one register must hold all of a token's scales"
    num_tokens = T.dynamic("num_tokens")

    @T.prim_func
    def per_token_cast(
        X: T.Tensor((num_tokens, hidden), T.bfloat16),
        Q: T.Tensor((num_tokens, hidden), T.float8_e4m3fn),
        Sf: T.Tensor((num_tokens, num_groups), T.float32),
    ):
        with T.Kernel(1):
            x_ub = T.alloc_shared((hidden,), T.bfloat16)
            q_ub = T.alloc_shared((hidden,), T.float8_e4m3fn)
            amax_ub = T.alloc_shared((SF_PAD,), T.float32)
            sf_ub = T.alloc_shared((SF_PAD,), T.float32)
            inv_ub = T.alloc_shared((SF_PAD,), T.float32)

            for token in T.serial(num_tokens):
                T.copy(X[token, 0], x_ub)
                # TODO: three passes. (1) per 128-channel pair, load two 64-lane
                #       strips with S.vld(x_ub[col], dist='UNPK_B16') + S.vcvt to
                #       float32, S.vabs, then for each strip store S.vcmax(abs,
                #       mask_low) and S.vcmax(abs, mask_high) to amax_ub with
                #       dist='ONEPT_B32'. (2) S.mem_bar('VST_VLD'); load 64 amax
                #       values at once, S.vmaxs by E4M3_CLAMP_MIN, then sf =
                #       S.vdiv(clamped, 448) and inv = S.vdiv(448, clamped), store
                #       both. (3) S.mem_bar('VST_VLD'); reload x, build the
                #       inverse vector from four BRC_B32 loads plus two S.vsel,
                #       multiply, S.vcvt to float8_e4m3fn and store with
                #       dist='PK4_B32'
                raise NotImplementedError("asc/per_token/01_raw_fp32sf: implement per_token_cast")
                T.copy(q_ub, Q[token, 0])
                T.copy(sf_ub[0:num_groups], Sf[token, 0])

    return per_token_cast


def launch(x: torch.Tensor):
    m, hidden = x.shape
    num_groups = hidden // CANONICAL_G
    q = torch.empty((m, hidden), dtype=torch.float8_e4m3fn, device=x.device)
    sf = torch.empty((m, num_groups), dtype=torch.float32, device=x.device)
    compile_kernel(hidden)(x, q, sf)
    status.assert_on_device("per_token 01", q, sf)
    return q, sf

if __name__ == "__main__":        # not a script -- see the module docstring
    raise SystemExit("Run it through the harness:  python -m harness.check asc/per_token/01")
