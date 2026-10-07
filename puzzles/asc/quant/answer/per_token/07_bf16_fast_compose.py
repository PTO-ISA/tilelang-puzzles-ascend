"""per_token 07 (ASC). See doc/quant/per_token/07_bf16_fast_compose.md"""

import torch
import tilelang
import tilelang.ascend.language as T
from tilelang.ascend.language import simd as S

from harness import status
from harness.consts import CANONICAL_G, E4M3_CLAMP_MIN, E4M3_MAX

LANES = 64
STRIP = 256            # the bf16 fast path's step: 8 groups of 32
SF_PAD = 64
BF16_K = 256           # this variant needs hidden % 256 == 0


@tilelang.jit(target="ascend")
def compile_kernel(hidden: int, group_size: int = CANONICAL_G):
    """bfloat16 compute + power-of-two scale + packed UE8M0 + FP8 output."""
    assert hidden % STRIP == 0, "the bfloat16 fast path steps 256 values at a time"
    assert group_size == 32
    num_groups = hidden // group_size
    groups_per_strip = STRIP // group_size          # 8
    num_tokens = T.dynamic("num_tokens")
    inv_qmax = 1.0 / E4M3_MAX

    @T.prim_func
    def per_token_cast(
        X: T.Tensor((num_tokens, hidden), T.bfloat16),
        Q: T.Tensor((num_tokens, hidden), T.float8_e4m3fn),
        Sf: T.Tensor((num_tokens, num_groups), T.uint8),
    ):
        with T.Kernel(1):
            x_ub = T.alloc_shared((hidden,), T.bfloat16)
            q_ub = T.alloc_shared((hidden,), T.float8_e4m3fn)
            amax_ub = T.alloc_shared((SF_PAD,), T.float32)
            sf_ub = T.alloc_shared((SF_PAD,), T.uint8)
            inv_ub = T.alloc_shared((SF_PAD,), T.float32)

            for token in T.serial(num_tokens):
                T.copy(X[token, 0], x_ub)
                # --- BEGIN SOLUTION hint="bf16 reduce: per 256-value strip, x0, x1 = S.vld2(x_ub[col], dist='DINTLV_B16'); abs via S.vand(T.reinterpret(x, 'uint16x128'), S.vdup(0x7FFF, T.uint16)); pair them with S.vmax; S.vcgmax gives 8 grouped maxima; widen with dense, _ = S.vintlv(S.vdup(0.0, T.bfloat16), T.reinterpret(maxima, 'bfloat16x128')) and store 8 elements with mask S.pset(32, 'PAT_VL8'), dist='NORM_B32', extent=8. Then the exponent trick from variant 02/03 and the apply pass from variant 01."
                with T.SimdVF():
                    mask_low = S.pset(32, "PAT_VL32")
                    mask_all = S.pset(32, "PAT_ALL")
                    mask_high = S.pnot(mask_low, mask_all)
                    mask_vl8 = S.pset(32, "PAT_VL8")
                    abs_mask = S.vdup(0x7FFF, T.uint16)
                    zero_bf16 = S.vdup(0.0, T.bfloat16)

                    # ---- pass 1: reduce in bfloat16, 256 values per step ----
                    for strip in T.serial(hidden // STRIP):
                        col = strip * STRIP
                        group = strip * groups_per_strip
                        # deinterleave 256 bf16 into evens and odds
                        x0, x1 = S.vld2(x_ub[col], dist="DINTLV_B16")
                        # clearing the sign bit is abs, on the integer unit
                        a0 = S.vand(T.reinterpret(x0, "uint16x128"), abs_mask)
                        a1 = S.vand(T.reinterpret(x1, "uint16x128"), abs_mask)
                        # each lane now covers 2 originals; vcgmax reduces 16
                        # lanes per hardware group -> 8 results of 32 originals
                        maxima = S.vcgmax(S.vmax(a0, a1))
                        # widen the 8 bf16 maxima to float32 by interleaving zeros
                        dense, _ = S.vintlv(zero_bf16,
                                            T.reinterpret(maxima, "bfloat16x128"))
                        S.vsts(amax_ub[group], T.reinterpret(dense, "float32x64"),
                               mask_vl8, dist="NORM_B32", extent=groups_per_strip)
                    S.mem_bar("VST_VLD")

                    # ---- pass 2: exponent arithmetic (variants 02/03) ----
                    clamped = S.vmaxs(S.vld(amax_ub[0]), E4M3_CLAMP_MIN)
                    one = S.vdup(1, T.uint32)
                    bits = T.reinterpret(S.vmuls(clamped, inv_qmax), "uint32x64")
                    biased = S.vadds(S.vshrs(S.vsub(bits, one), 23), 1)
                    S.vsts(sf_ub[0], T.reinterpret(biased, "uint8x256"), dist="PK4_B32")
                    S.vsts(inv_ub[0], T.reinterpret(
                        S.vshls(S.vsub(S.vdup(254, T.uint32), biased), 23),
                        "float32x64"))
                    S.mem_bar("VST_VLD")

                    # ---- pass 3: apply, in float32 (the scale is exact) ----
                    for pair in T.serial(hidden // 128):
                        col = pair * 128
                        group = pair * 4
                        for half in range(2):
                            c = col + half * LANES
                            g = group + half * 2
                            i0 = S.vld(inv_ub[g], dist="BRC_B32")
                            i1 = S.vld(inv_ub[g + 1], dist="BRC_B32")
                            xv = S.vcvt(S.vld(x_ub[c], dist="UNPK_B16"), T.float32)
                            S.vsts(q_ub[c],
                                   S.vcvt(S.vmul(xv, S.vsel(i0, i1, mask_low)),
                                          T.float8_e4m3fn), dist="PK4_B32")
                # --- END SOLUTION
                T.copy(q_ub, Q[token, 0])
                T.copy(sf_ub[0:num_groups], Sf[token, 0])

    return per_token_cast


def launch(x: torch.Tensor):
    m, hidden = x.shape
    ng = hidden // CANONICAL_G
    q = torch.empty((m, hidden), dtype=torch.float8_e4m3fn, device=x.device)
    sf_bytes = torch.empty((m, ng), dtype=torch.uint8, device=x.device)
    compile_kernel(hidden)(x, q, sf_bytes)
    status.assert_on_device("per_token 07", q, sf_bytes)
    return q, sf_bytes.view(torch.int16)
