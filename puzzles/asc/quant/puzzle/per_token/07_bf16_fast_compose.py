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
                # TODO: bf16 reduce: per 256-value strip, x0, x1 =
                #       S.vld2(x_ub[col], dist='DINTLV_B16'); abs via
                #       S.vand(T.reinterpret(x, 'uint16x128'), S.vdup(0x7FFF,
                #       T.uint16)); pair them with S.vmax; S.vcgmax gives 8
                #       grouped maxima; widen with dense, _ = S.vintlv(S.vdup(0.0,
                #       T.bfloat16), T.reinterpret(maxima, 'bfloat16x128')) and
                #       store 8 elements with mask S.pset(32, 'PAT_VL8'),
                #       dist='NORM_B32', extent=8. Then the exponent trick from
                #       variant 02/03 and the apply pass from variant 01.
                raise NotImplementedError("asc/per_token/07_bf16_fast_compose: implement per_token_cast")
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

if __name__ == "__main__":        # not a script -- see the module docstring
    raise SystemExit("Run it through the harness:  python -m harness.check asc/per_token/07 --role puzzle")
