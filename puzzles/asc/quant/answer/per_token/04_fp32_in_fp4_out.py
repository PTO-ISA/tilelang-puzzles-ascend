"""per_token 04 (ASC). See doc/quant/per_token/04_fp32_in_fp4_out.md"""

import torch
import tilelang
import tilelang.ascend.language as T
from tilelang.ascend.language import simd as S

from harness import status
from harness.consts import CANONICAL_G, E2M1_CLAMP_MIN, E2M1_MAX

LANES = 64
PAIR = 128
SF_PAD = 64


@tilelang.jit(target="ascend")
def compile_kernel(hidden: int, group_size: int = CANONICAL_G):
    """Quantize float32 -> packed FP4 e2m1 with one FP32 scale per 32 channels."""
    assert hidden % PAIR == 0 and group_size == 32
    num_groups = hidden // group_size
    num_tokens = T.dynamic("num_tokens")

    @T.prim_func
    def per_token_cast(
        X: T.Tensor((num_tokens, hidden), T.float32),
        Q: T.Tensor((num_tokens, hidden), T.float4_e2m1fn),
        Sf: T.Tensor((num_tokens, num_groups), T.float32),
    ):
        with T.Kernel(1):
            x_ub = T.alloc_shared((hidden,), T.float32)
            q_ub = T.alloc_shared((hidden,), T.float4_e2m1fn)
            amax_ub = T.alloc_shared((SF_PAD,), T.float32)
            sf_ub = T.alloc_shared((SF_PAD,), T.float32)
            inv_ub = T.alloc_shared((SF_PAD,), T.float32)

            for token in T.serial(num_tokens):
                T.copy(X[token, 0], x_ub)
                # --- BEGIN SOLUTION hint="float32 input needs no unpacking load: just S.vld(x_ub[col]). Use E2M1_MAX/E2M1_CLAMP_MIN instead of the e4m3 constants. For the store, there is no float32->e2m1 convert: deinterleave the two quantized halves with low, high = S.vdintlv(T.reinterpret(q0, 'uint16x128'), T.reinterpret(q1, 'uint16x128')), round to odd with S.vor(high, S.vmins(low, 1)), reinterpret to 'bfloat16x128', then S.vcvt to T.float4_e2m1fn and store with dist='PK4_B32'"
                with T.SimdVF():
                    mask_low = S.pset(32, "PAT_VL32")
                    mask_all = S.pset(32, "PAT_ALL")
                    mask_high = S.pnot(mask_low, mask_all)
                    qmax = S.vdup(E2M1_MAX, T.float32)

                    for pair in T.serial(hidden // PAIR):
                        col = pair * PAIR
                        group = pair * (PAIR // group_size)
                        # float32 input: a plain load, no convert.
                        a0 = S.vabs(S.vld(x_ub[col]))
                        a1 = S.vabs(S.vld(x_ub[col + LANES]))
                        S.vsts(amax_ub[group], S.vcmax(a0, mask_low), dist="ONEPT_B32")
                        S.vsts(amax_ub[group + 1], S.vcmax(a0, mask_high), dist="ONEPT_B32")
                        S.vsts(amax_ub[group + 2], S.vcmax(a1, mask_low), dist="ONEPT_B32")
                        S.vsts(amax_ub[group + 3], S.vcmax(a1, mask_high), dist="ONEPT_B32")
                    S.mem_bar("VST_VLD")

                    clamped = S.vmaxs(S.vld(amax_ub[0]), E2M1_CLAMP_MIN)
                    S.vsts(sf_ub[0], S.vdiv(clamped, qmax))
                    S.vsts(inv_ub[0], S.vdiv(qmax, clamped))
                    S.mem_bar("VST_VLD")

                    one_u16 = S.vdup(1, T.uint16)
                    for pair in T.serial(hidden // PAIR):
                        col = pair * PAIR
                        group = pair * (PAIR // group_size)
                        i0 = S.vld(inv_ub[group], dist="BRC_B32")
                        i1 = S.vld(inv_ub[group + 1], dist="BRC_B32")
                        i2 = S.vld(inv_ub[group + 2], dist="BRC_B32")
                        i3 = S.vld(inv_ub[group + 3], dist="BRC_B32")
                        q0 = S.vmul(S.vld(x_ub[col]), S.vsel(i0, i1, mask_low))
                        q1 = S.vmul(S.vld(x_ub[col + LANES]), S.vsel(i2, i3, mask_low))
                        # float32 -> bfloat16 (round to odd) -> e2m1.
                        low, high = S.vdintlv(T.reinterpret(q0, "uint16x128"),
                                              T.reinterpret(q1, "uint16x128"))
                        odd = S.vor(high, S.vmin(low, one_u16))
                        S.vsts(q_ub[col],
                               S.vcvt(T.reinterpret(odd, "bfloat16x128"),
                                      T.float4_e2m1fn), dist="PK4_B32")
                # --- END SOLUTION
                T.copy(q_ub, Q[token, 0])
                T.copy(sf_ub[0:num_groups], Sf[token, 0])

    return per_token_cast


def launch(x: torch.Tensor):
    m, hidden = x.shape
    q = torch.empty((m, hidden // 2), dtype=torch.uint8,
                    device=x.device).view(torch.float4_e2m1fn_x2)
    sf = torch.empty((m, hidden // CANONICAL_G), dtype=torch.float32, device=x.device)
    compile_kernel(hidden)(x, q, sf)
    status.assert_on_device("per_token 04", q, sf)
    return q.view(torch.uint8).view(torch.int8), sf

if __name__ == "__main__":        # not a script -- see the module docstring
    raise SystemExit("Run it through the harness:  python -m harness.check asc/per_token/04")
