"""per_token 02 (SIMT). See doc/quant/per_token/02_round_sf.md"""

import torch
import tilelang
import tilelang.ascend.language as T

from harness import status
from harness.consts import CANONICAL_G, E4M3_CLAMP_MIN, E4M3_MAX

THREADS = 128


@tilelang.jit(target="ascend")
def compile_kernel(hidden: int, group_size: int = CANONICAL_G):
    """Quantize bfloat16 -> FP8 e4m3 with a power-of-two scale per 32 channels.

    The exponent trick is the same arithmetic as the SIMD tiers, written on
    scalars: there is no lane count to carry, so no `size=` argument and no
    `V.vbrc` to lift a constant into a vector. What ASC spells as six vector
    ops and PTO as a width-independent macro is four lines of plain integer
    arithmetic here -- the one place where SIMT's scalar view is a clear win.
    """
    assert hidden % group_size == 0 and group_size == 32
    num_groups = hidden // group_size
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
            sf_ub = T.alloc_shared((num_groups,), T.float32)

            for token in T.serial(num_tokens):
                T.copy(X[token, 0], x_ub)

                # --- BEGIN SOLUTION hint="variant 01's two passes, with the division replaced by the exponent trick. Per group: reduce amax as before, then bits = T.reinterpret('int32', amax / E4M3_MAX); biased = ((bits - 1) >> 23) + 1 is ceil(log2(amax/448)) + 127; sf = T.reinterpret('float32', biased << 23) rebuilds the power of two by writing an exponent back, and inv = T.reinterpret('float32', (254 - biased) << 23) is its reciprocal, since 254 - biased negates the unbiased exponent. No lane count and no broadcast of constants -- compare the ASC file's six vector ops, or PTO's compute_scale macro, which only has to be a macro because `size=` has to be threaded through it."
                with T.SimtVF(threads=THREADS):
                    q_l = T.alloc_fragment((hidden,), T.float32)
                    inv_l = T.alloc_fragment((num_groups,), T.float32)
                    sf_l = T.alloc_fragment((num_groups,), T.float32)

                    for g in T.Parallel(num_groups):
                        amax = T.alloc_var(T.float32)
                        amax = E4M3_CLAMP_MIN
                        for j in T.serial(group_size):
                            v = T.Cast("float32", x_ub[g * group_size + j])
                            amax = T.max(amax, T.abs(v))

                        bits = T.reinterpret("int32", amax / E4M3_MAX)
                        biased = ((bits - 1) >> 23) + 1
                        sf_l[g] = T.reinterpret("float32", biased << 23)
                        inv_l[g] = T.reinterpret("float32", (254 - biased) << 23)

                    for i in T.Parallel(hidden):
                        q_l[i] = T.Cast("float32", x_ub[i]) * inv_l[i // group_size]

                    T.copy(q_l, q_ub)
                    T.copy(sf_l, sf_ub)
                # --- END SOLUTION

                T.copy(q_ub, Q[token, 0])
                T.copy(sf_ub, Sf[token, 0])

    return per_token_cast


def launch(x: torch.Tensor):
    m, hidden = x.shape
    q = torch.empty((m, hidden), dtype=torch.float8_e4m3fn, device=x.device)
    sf = torch.empty((m, hidden // CANONICAL_G), dtype=torch.float32, device=x.device)
    compile_kernel(hidden)(x, q, sf)
    status.assert_on_device("per_token 02", q, sf)
    return q, sf

if __name__ == "__main__":        # not a script -- see the module docstring
    raise SystemExit("Run it through the harness:  python -m harness.check simt/per_token/02")
