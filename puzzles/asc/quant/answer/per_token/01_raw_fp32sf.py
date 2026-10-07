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
                # --- BEGIN SOLUTION hint="three passes. (1) per 128-channel pair, load two 64-lane strips with S.vld(x_ub[col], dist='UNPK_B16') + S.vcvt to float32, S.vabs, then for each strip store S.vcmax(abs, mask_low) and S.vcmax(abs, mask_high) to amax_ub with dist='ONEPT_B32'. (2) S.mem_bar('VST_VLD'); load 64 amax values at once, S.vmaxs by E4M3_CLAMP_MIN, then sf = S.vdiv(clamped, 448) and inv = S.vdiv(448, clamped), store both. (3) S.mem_bar('VST_VLD'); reload x, build the inverse vector from four BRC_B32 loads plus two S.vsel, multiply, S.vcvt to float8_e4m3fn and store with dist='PK4_B32'"
                with T.SimdVF():
                    mask_low = S.pset(32, "PAT_VL32")
                    # S.pset must be bound to a name before use: nesting it
                    # inside another call leaves `tl.simd.pset` unresolved at
                    # lowering time.
                    mask_all = S.pset(32, "PAT_ALL")
                    mask_high = S.pnot(mask_low, mask_all)
                    qmax = S.vdup(E4M3_MAX, T.float32)

                    # ---- pass 1: reduce |x| over each group of 32 ----
                    for pair in T.serial(hidden // PAIR):
                        col = pair * PAIR
                        group = pair * (PAIR // group_size)     # 4 groups per pair
                        a0 = S.vabs(S.vcvt(S.vld(x_ub[col], dist="UNPK_B16"), T.float32))
                        a1 = S.vabs(S.vcvt(S.vld(x_ub[col + LANES], dist="UNPK_B16"),
                                           T.float32))
                        # Each register holds two groups, so reduce half at a time.
                        S.vsts(amax_ub[group], S.vcmax(a0, mask_low), dist="ONEPT_B32")
                        S.vsts(amax_ub[group + 1], S.vcmax(a0, mask_high),
                               dist="ONEPT_B32")
                        S.vsts(amax_ub[group + 2], S.vcmax(a1, mask_low),
                               dist="ONEPT_B32")
                        S.vsts(amax_ub[group + 3], S.vcmax(a1, mask_high),
                               dist="ONEPT_B32")

                    # Without this, pass 2 may read amax values pass 1 has not
                    # yet written: UB aliasing is not tracked by the hardware.
                    S.mem_bar("VST_VLD")

                    # ---- pass 2: amax -> scale, 64 groups at a time ----
                    clamped = S.vmaxs(S.vld(amax_ub[0]), E4M3_CLAMP_MIN)
                    S.vsts(sf_ub[0], S.vdiv(clamped, qmax))
                    S.vsts(inv_ub[0], S.vdiv(qmax, clamped))

                    S.mem_bar("VST_VLD")

                    # ---- pass 3: apply the inverse and convert ----
                    for pair in T.serial(hidden // PAIR):
                        col = pair * PAIR
                        group = pair * (PAIR // group_size)
                        i0 = S.vld(inv_ub[group], dist="BRC_B32")
                        i1 = S.vld(inv_ub[group + 1], dist="BRC_B32")
                        i2 = S.vld(inv_ub[group + 2], dist="BRC_B32")
                        i3 = S.vld(inv_ub[group + 3], dist="BRC_B32")
                        x0 = S.vcvt(S.vld(x_ub[col], dist="UNPK_B16"), T.float32)
                        x1 = S.vcvt(S.vld(x_ub[col + LANES], dist="UNPK_B16"),
                                    T.float32)
                        q0 = S.vmul(x0, S.vsel(i0, i1, mask_low))
                        q1 = S.vmul(x1, S.vsel(i2, i3, mask_low))
                        S.vsts(q_ub[col], S.vcvt(q0, T.float8_e4m3fn), dist="PK4_B32")
                        S.vsts(q_ub[col + LANES], S.vcvt(q1, T.float8_e4m3fn),
                               dist="PK4_B32")
                # --- END SOLUTION
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
