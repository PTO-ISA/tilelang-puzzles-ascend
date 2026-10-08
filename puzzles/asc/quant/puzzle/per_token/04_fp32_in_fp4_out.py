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
                # TODO: float32 input needs no unpacking load: just
                #       S.vld(x_ub[col]). Use E2M1_MAX/E2M1_CLAMP_MIN instead of
                #       the e4m3 constants. For the store, there is no
                #       float32->e2m1 convert: deinterleave the two quantized
                #       halves with low, high = S.vdintlv(T.reinterpret(q0,
                #       'uint16x128'), T.reinterpret(q1, 'uint16x128')), round to
                #       odd with S.vor(high, S.vmins(low, 1)), reinterpret to
                #       'bfloat16x128', then S.vcvt to T.float4_e2m1fn and store
                #       with dist='PK4_B32'
                raise NotImplementedError("asc/per_token/04_fp32_in_fp4_out: implement per_token_cast")
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
