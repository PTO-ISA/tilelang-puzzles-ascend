"""per_token 03 (PTO). See doc/quant/per_token/03_packed_ue8m0.md"""

import torch
import tilelang
import tilelang.ascend.language as T
from tilelang.ascend.language import vmi as V

from harness import status
from harness.consts import CANONICAL_G, E4M3_CLAMP_MIN, E4M3_MAX

LANES = 64
PAIR = 128
SF_PAD = 64


@tilelang.jit(target="pto")
def compile_kernel(hidden: int, group_size: int = CANONICAL_G):
    """Quantize with a power-of-two scale. Scales are returned as float32."""
    assert hidden % PAIR == 0 and group_size == 32
    num_groups = hidden // group_size
    groups_per_pair = PAIR // group_size
    num_tokens = T.dynamic("num_tokens")
    inv_qmax = 1.0 / E4M3_MAX

    @T.macro
    def compute_scale(amax, lanes, packed: bool):
        """amax -> (scale, 1/scale), at any lane count.

        With `packed`, the first result is the *biased exponent* -- the UE8M0
        byte -- instead of the float32 scale. Same helper, one flag, which is the
        shape production's compute_scale has and which only works because the
        body is width-independent.
        """
        mask = V.create_mask(lanes, size=lanes)
        clamped = V.vmax(amax, V.vbrc(T.float32(E4M3_CLAMP_MIN), size=lanes), mask)
        one = V.vbrc(T.uint32(1), size=lanes)
        shift = V.vbrc(T.uint32(23), size=lanes)
        bits = V.vinterpret_cast(
            V.vmul(clamped, V.vbrc(T.float32(inv_qmax), size=lanes), mask), "uint32")
        # ((bits - 1) >> 23) + 1  ==  ceil(log2(v)) + 127
        biased = V.vadd(V.vshr(V.vsub(bits, one), shift), one)
        sf = biased if packed else V.vinterpret_cast(V.vshl(biased, shift), "float32")
        # 254 - biased == 127 - ceil_exp: the negated exponent, still biased.
        inv = V.vinterpret_cast(
            V.vshl(V.vsub(V.vbrc(T.uint32(254), size=lanes), biased), shift), "float32")
        return sf, inv

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
                    mask = V.create_mask(PAIR, size=PAIR)

                    for pair in T.serial(hidden // PAIR):
                        col = pair * PAIR
                        group = pair * groups_per_pair
                        x = V.vcvt(V.vload(x_ub[col], size=PAIR), "float32")
                        V.vstore(V.vcmax(V.vabs(x, mask), mask, group=groups_per_pair),
                                 amax_ub[group])
                    T.simd.mem_bar("VST_VLD")

                    # TODO: call compute_scale(..., packed=True) so it returns the
                    #       biased exponent, then store it as bytes with one
                    #       conversion: V.vstore(V.vcvt(exponent, 'uint8'),
                    #       sf_ub[0]). No reinterpret and no PK4_B32 store mode --
                    #       the destination buffer dtype decides the packing.
                    raise NotImplementedError("pto/per_token/03_packed_ue8m0: implement per_token_cast")
                    T.simd.mem_bar("VST_VLD")

                    for pair in T.serial(hidden // PAIR):
                        col = pair * PAIR
                        group = pair * groups_per_pair
                        inv_v = V.vload(inv_ub[group], size=PAIR, stride=1,
                                        dist_mode="brc", group=groups_per_pair)
                        x = V.vcvt(V.vload(x_ub[col], size=PAIR), "float32")
                        V.vstore(V.vcvt(V.vmul(x, inv_v, mask), "float8_e4m3fn",
                                        rounding="R", saturate="SAT"), q_ub[col])
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
    return q, sf_bytes.view(torch.int16)
