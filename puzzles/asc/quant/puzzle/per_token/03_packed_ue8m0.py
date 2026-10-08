"""per_token 03 (ASC). See doc/quant/per_token/03_packed_ue8m0.md"""

import torch
import tilelang
import tilelang.ascend.language as T
from tilelang.ascend.language import simd as S

from harness import status
from harness.consts import CANONICAL_G, E4M3_CLAMP_MIN, E4M3_MAX

LANES = 64
PAIR = 128
SF_PAD = 64


@tilelang.jit(target="ascend")
def compile_kernel(hidden: int, group_size: int = CANONICAL_G):
    """Quantize with power-of-two scales stored as UE8M0 bytes."""
    assert hidden % PAIR == 0 and group_size == 32
    num_groups = hidden // group_size
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
                with T.SimdVF():
                    mask_low = S.pset(32, "PAT_VL32")
                    mask_all = S.pset(32, "PAT_ALL")
                    mask_high = S.pnot(mask_low, mask_all)

                    for pair in T.serial(hidden // PAIR):
                        col = pair * PAIR
                        group = pair * (PAIR // group_size)
                        a0 = S.vabs(S.vcvt(S.vld(x_ub[col], dist="UNPK_B16"), T.float32))
                        a1 = S.vabs(S.vcvt(S.vld(x_ub[col + LANES], dist="UNPK_B16"),
                                           T.float32))
                        S.vsts(amax_ub[group], S.vcmax(a0, mask_low), dist="ONEPT_B32")
                        S.vsts(amax_ub[group + 1], S.vcmax(a0, mask_high), dist="ONEPT_B32")
                        S.vsts(amax_ub[group + 2], S.vcmax(a1, mask_low), dist="ONEPT_B32")
                        S.vsts(amax_ub[group + 3], S.vcmax(a1, mask_high), dist="ONEPT_B32")
                    S.mem_bar("VST_VLD")

                    clamped = S.vmaxs(S.vld(amax_ub[0]), E4M3_CLAMP_MIN)
                    one = S.vdup(1, T.uint32)
                    bits = T.reinterpret(S.vmuls(clamped, inv_qmax), "uint32x64")
                    biased = S.vadds(S.vshrs(S.vsub(bits, one), 23), 1)
                    inv = T.reinterpret(
                        S.vshls(S.vsub(S.vdup(254, T.uint32), biased), 23), "float32x64")
                    # TODO: `biased` already IS the UE8M0 byte (ceil_exp + 127),
                    #       so store its low byte: S.vsts(sf_ub[0],
                    #       T.reinterpret(biased, 'uint8x256'), dist='PK4_B32').
                    #       PK4_B32 narrows each 32-bit lane to one byte.
                    raise NotImplementedError("asc/per_token/03_packed_ue8m0: implement per_token_cast")
                    S.vsts(inv_ub[0], inv)
                    S.mem_bar("VST_VLD")

                    for pair in T.serial(hidden // PAIR):
                        col = pair * PAIR
                        group = pair * (PAIR // group_size)
                        i0 = S.vld(inv_ub[group], dist="BRC_B32")
                        i1 = S.vld(inv_ub[group + 1], dist="BRC_B32")
                        i2 = S.vld(inv_ub[group + 2], dist="BRC_B32")
                        i3 = S.vld(inv_ub[group + 3], dist="BRC_B32")
                        x0 = S.vcvt(S.vld(x_ub[col], dist="UNPK_B16"), T.float32)
                        x1 = S.vcvt(S.vld(x_ub[col + LANES], dist="UNPK_B16"), T.float32)
                        S.vsts(q_ub[col], S.vcvt(S.vmul(x0, S.vsel(i0, i1, mask_low)),
                                                 T.float8_e4m3fn), dist="PK4_B32")
                        S.vsts(q_ub[col + LANES],
                               S.vcvt(S.vmul(x1, S.vsel(i2, i3, mask_low)),
                                      T.float8_e4m3fn), dist="PK4_B32")
                T.copy(q_ub, Q[token, 0])
                T.copy(sf_ub[0:num_groups], Sf[token, 0])

    return per_token_cast


def launch(x: torch.Tensor):
    """Returns (q, sf_packed) where sf_packed is int16, two exponents per word."""
    m, hidden = x.shape
    num_groups = hidden // CANONICAL_G
    q = torch.empty((m, hidden), dtype=torch.float8_e4m3fn, device=x.device)
    sf_bytes = torch.empty((m, num_groups), dtype=torch.uint8, device=x.device)
    compile_kernel(hidden)(x, q, sf_bytes)
    status.assert_on_device("per_token 03", q, sf_bytes)
    # The int16 packing is a reinterpretation of the same bytes, not kernel work.
    return q, sf_bytes.view(torch.int16)

if __name__ == "__main__":        # not a script -- see the module docstring
    raise SystemExit("Run it through the harness:  python -m harness.check asc/per_token/03")
