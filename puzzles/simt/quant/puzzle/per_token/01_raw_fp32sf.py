"""per_token 01 (SIMT). See doc/quant/per_token/01_raw_fp32sf.md"""

import torch
import tilelang
import tilelang.ascend.language as T

from harness import status
from harness.consts import CANONICAL_G, E4M3_CLAMP_MIN, E4M3_MAX

THREADS = 128


# No out_idx, for the same reason as the ASC and PTO tiers: tilelang's automatic
# output allocation rejects float8_e4m3fn on this toolchain, so launch() allocates
# the outputs. See harness/probe/fp8_out_idx.py.
@tilelang.jit(target="ascend")
def compile_kernel(hidden: int, group_size: int = CANONICAL_G):
    """Quantize bfloat16 -> FP8 e4m3 with one FP32 scale per 32 channels."""
    assert hidden % group_size == 0
    assert group_size == 32
    num_groups = hidden // group_size
    num_tokens = T.dynamic("num_tokens")

    @T.prim_func
    def per_token_cast(
        X: T.Tensor((num_tokens, hidden), T.bfloat16),
        Q: T.Tensor((num_tokens, hidden), T.float8_e4m3fn),
        Sf: T.Tensor((num_tokens, num_groups), T.float32),
    ):
        with T.Kernel(1):
            # Flat, like the ASC and PTO tiers. A 2-D UB buffer here would be
            # wrong: `T.copy(X[token, 0], x_ub)` into a (num_groups, group_size)
            # buffer copies the 2-D REGION X[token:token+num_groups, 0:group_size],
            # not the token's 128 contiguous channels. The group structure belongs
            # in the index arithmetic instead, which shared buffers allow freely.
            x_ub = T.alloc_shared((hidden,), T.bfloat16)
            q_ub = T.alloc_shared((hidden,), T.float8_e4m3fn)
            sf_ub = T.alloc_shared((num_groups,), T.float32)

            for token in T.serial(num_tokens):
                T.copy(X[token, 0], x_ub)

                # TODO: open `with T.SimtVF(threads=128):` and allocate a float32
                #       fragment shaped (num_groups, group_size) -- making the
                #       group the fast axis is what turns the quant group into a
                #       reduction axis, the same reshape the CUDA reference does
                #       before its own T.reduce_absmax. Fill it with one thread
                #       per CHANNEL -- T.Parallel(num_groups, group_size) widening
                #       x_ub -- and fold it with T.reduce_absmax(x_l, amax_l,
                #       dim=1). Do NOT write the reduction as
                #       T.Parallel(num_groups) with an inner T.serial: one thread
                #       per GROUP is 4 threads of 128, and it measures 8.73x
                #       slower (harness/probe/simt_reduce_max.py). Clamp after the
                #       fold, since a library reduction cannot be seeded, then
                #       form sf and its reciprocal. Then one T.Parallel(hidden)
                #       applying inv_l[i // group_size]; indexing a (num_groups,)
                #       fragment inside a differently-shaped Parallel is fine,
                #       since the one-pattern rule is per buffer. T.copy the
                #       fragments out to shared.
                raise NotImplementedError("simt/per_token/01_raw_fp32sf: implement per_token_cast")

                T.copy(q_ub, Q[token, 0])
                T.copy(sf_ub, Sf[token, 0])

    return per_token_cast


def launch(x: torch.Tensor):
    m, hidden = x.shape
    num_groups = hidden // CANONICAL_G
    q = torch.empty((m, hidden), dtype=torch.float8_e4m3fn, device=x.device)
    sf = torch.empty((m, num_groups), dtype=torch.float32, device=x.device)
    compile_kernel(hidden)(x, q, sf)
    status.assert_on_device("per_token 01", q, sf)
    return q, sf

if __name__ == "__main__":        # not a script -- see the module docstring
    raise SystemExit("Run it through the harness:  python -m harness.check simt/per_token/01 --role puzzle")
