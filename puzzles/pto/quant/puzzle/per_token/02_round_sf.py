"""per_token 02 (PTO). See doc/quant/per_token/02_round_sf.md"""

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
    def compute_scale(amax, lanes):
        """amax -> (power-of-two scale, its reciprocal), at any lane count.

        The same body serves 4, 64, 128 or 256 lanes because `size=` is an
        argument and the reinterprets carry no lane count. The ASC spelling of
        this cannot be written once: "uint32x64" fixes the width.
        """
        mask = V.create_mask(lanes, size=lanes)
        clamped = V.vmax(amax, V.vbrc(T.float32(E4M3_CLAMP_MIN), size=lanes), mask)
        one = V.vbrc(T.uint32(1), size=lanes)
        shift = V.vbrc(T.uint32(23), size=lanes)
        bits = V.vinterpret_cast(
            V.vmul(clamped, V.vbrc(T.float32(inv_qmax), size=lanes), mask), "uint32")
        # ((bits - 1) >> 23) + 1  ==  ceil(log2(v)) + 127
        biased = V.vadd(V.vshr(V.vsub(bits, one), shift), one)
        sf = V.vinterpret_cast(V.vshl(biased, shift), "float32")
        # 254 - biased == 127 - ceil_exp: the negated exponent, still biased.
        inv = V.vinterpret_cast(
            V.vshl(V.vsub(V.vbrc(T.uint32(254), size=lanes), biased), shift), "float32")
        return sf, inv

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
                with T.SimdVF():
                    mask = V.create_mask(PAIR, size=PAIR)

                    for pair in T.serial(hidden // PAIR):
                        col = pair * PAIR
                        group = pair * groups_per_pair
                        x = V.vcvt(V.vload(x_ub[col], size=PAIR), "float32")
                        V.vstore(V.vcmax(V.vabs(x, mask), mask, group=groups_per_pair),
                                 amax_ub[group])
                    T.simd.mem_bar("VST_VLD")

                    # TODO: the compute_scale(amax, lanes) macro is ALREADY
                    #       WRITTEN above this region -- read it first, because
                    #       the exponent trick it contains is the whole point of
                    #       this variant: clamp with V.vmax, bits =
                    #       V.vinterpret_cast(V.vmul(clamped, 1/448), 'uint32'),
                    #       biased = V.vadd(V.vshr(V.vsub(bits, one), shift), one)
                    #       with shift = 23, then sf =
                    #       V.vinterpret_cast(V.vshl(biased, shift), 'float32')
                    #       and inv the same with V.vsub(254, biased). Because VMI
                    #       takes the lane count as an argument, that one macro
                    #       serves every width, which is why it can be factored
                    #       out at all -- the ASC file spells the arithmetic
                    #       inline instead. Your job here is the three lines that
                    #       use it: call it on V.vload(amax_ub[0], size=LANES) and
                    #       V.vstore both results.
                    raise NotImplementedError("pto/per_token/02_round_sf: implement per_token_cast")
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
    m, hidden = x.shape
    q = torch.empty((m, hidden), dtype=torch.float8_e4m3fn, device=x.device)
    sf = torch.empty((m, hidden // CANONICAL_G), dtype=torch.float32, device=x.device)
    compile_kernel(hidden)(x, q, sf)
    status.assert_on_device("per_token 02", q, sf)
    return q, sf

if __name__ == "__main__":        # not a script -- see the module docstring
    raise SystemExit("Run it through the harness:  python -m harness.check pto/per_token/02 --role puzzle")
