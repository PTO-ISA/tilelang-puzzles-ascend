"""per_token 07 (SIMT). See doc/quant/per_token/07_bf16_fast_compose.md"""

import torch
import tilelang
import tilelang.ascend.language as T

from harness import status
from harness.consts import CANONICAL_G, E4M3_CLAMP_MIN, E4M3_MAX

THREADS = 128
BF16_K = 256


@tilelang.jit(target="ascend")
def compile_kernel(hidden: int, group_size: int = CANONICAL_G):
    """Power-of-two scale + packed UE8M0 + FP8 output, at K=256.

    This is the one variant whose *lesson does not survive the port*, and the
    docs say so rather than inventing an advantage. In the SIMD tiers the point
    is that bfloat16 halves the element width, so one instruction covers 256
    values instead of 128, and `group=8` reduces eight groups per step. SIMT has
    no lane width to exploit: parallelism is threads, `T.copy` widens bfloat16 to
    float32 on the way into a fragment, and a cross-thread reduction in bfloat16
    ICEs upstream anyway. So the body below is variant 03's body with a larger
    `hidden` -- the composition is free, but so is nothing else.

    It is still numerically identical: the scale is a power of two, so scaling a
    bfloat16 value by it is exact in either precision, and the reduction uses
    only the exponent, which bfloat16 keeps exactly.
    """
    assert hidden % group_size == 0 and group_size == 32
    num_groups = hidden // group_size
    num_tokens = T.dynamic("num_tokens")

    @T.prim_func
    def per_token_cast(
        X: T.Tensor((num_tokens, hidden), T.bfloat16),
        Q: T.Tensor((num_tokens, hidden), T.float8_e4m3fn),
        Sf: T.Tensor((num_tokens, num_groups), T.uint8),
    ):
        with T.Kernel(1):
            x_ub = T.alloc_shared((hidden,), T.bfloat16)
            q_ub = T.alloc_shared((hidden,), T.float8_e4m3fn)
            sf_ub = T.alloc_shared((num_groups,), T.uint8)

            for token in T.serial(num_tokens):
                T.copy(X[token, 0], x_ub)

                # TODO: this is variant 03's body unchanged, at hidden=256 and
                #       num_groups=8. That is the honest answer: the SIMD tiers
                #       earn this variant by computing in bfloat16 so one
                #       instruction covers 256 values and `group=8` folds eight
                #       reductions into one op, but SIMT has no lane width to
                #       trade -- threads are the parallelism, T.copy widens
                #       bfloat16 to float32 on the way into a fragment, and a
                #       bfloat16 cross-thread reduction ICEs. Write the float32
                #       path: reduce per group, exponent trick, store the biased
                #       exponent as a uint8, apply. The result is bit-identical
                #       because a power-of-two scale multiplies exactly in either
                #       precision.
                raise NotImplementedError("simt/per_token/07_bf16_fast_compose: implement per_token_cast")

                T.copy(q_ub, Q[token, 0])
                T.copy(sf_ub, Sf[token, 0])

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
    raise SystemExit("Run it through the harness:  python -m harness.check simt/per_token/07 --role puzzle")
