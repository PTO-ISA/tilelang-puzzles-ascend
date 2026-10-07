"""per_token 07 (PTO). See doc/quant/per_token/07_bf16_fast_compose.md"""

import torch
import tilelang
import tilelang.ascend.language as T
from tilelang.ascend.language import vmi as V

from harness import status
from harness.consts import CANONICAL_G, E4M3_CLAMP_MIN, E4M3_MAX

LANES = 64
STRIP = 256
SF_PAD = 64
BF16_K = 256


@tilelang.jit(target="pto")
def compile_kernel(hidden: int, group_size: int = CANONICAL_G):
    """bfloat16 compute + power-of-two scale + packed UE8M0 + FP8 output."""
    assert hidden % STRIP == 0 and group_size == 32
    num_groups = hidden // group_size
    groups_per_strip = STRIP // group_size          # 8
    num_tokens = T.dynamic("num_tokens")
    inv_qmax = 1.0 / E4M3_MAX

    @T.macro
    def compute_scale(amax, lanes):
        """amax -> (UE8M0 exponent byte value, bfloat16 reciprocal)."""
        mask = V.create_mask(lanes, size=lanes)
        clamped = V.vmax(amax, V.vbrc(T.float32(E4M3_CLAMP_MIN), size=lanes), mask)
        one = V.vbrc(T.uint32(1), size=lanes)
        shift = V.vbrc(T.uint32(23), size=lanes)
        bits = V.vinterpret_cast(
            V.vmul(clamped, V.vbrc(T.float32(inv_qmax), size=lanes), mask), "uint32")
        biased = V.vadd(V.vshr(V.vsub(bits, one), shift), one)
        inv = V.vinterpret_cast(
            V.vshl(V.vsub(V.vbrc(T.uint32(254), size=lanes), biased), shift), "float32")
        return biased, inv

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
            inv_ub = T.alloc_shared((SF_PAD,), T.bfloat16)

            for token in T.serial(num_tokens):
                T.copy(X[token, 0], x_ub)
                # TODO: bf16 reduce in four operations: raw = V.vload(x_ub[col],
                #       size=256); abs_u = V.vand(V.vinterpret_cast(raw,
                #       'uint16'), V.vbrc(T.uint16(0x7FFF), size=256)); amax =
                #       V.vcmax(abs_u, mask256, group=8); then
                #       V.vstore(V.vcvt(V.vinterpret_cast(amax, 'bfloat16'),
                #       'float32'), amax_ub[group], group=8, stride=1). No
                #       deinterleave and no pairing -- group=8 asks for exactly 8
                #       segments of 32. Then compute_scale, and an apply pass that
                #       multiplies in bfloat16 at 256 lanes with a brc group=8
                #       inverse.
                raise NotImplementedError("pto/per_token/07_bf16_fast_compose: implement per_token_cast")
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
