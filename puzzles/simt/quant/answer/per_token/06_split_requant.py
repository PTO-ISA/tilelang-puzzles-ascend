"""per_token 06 (SIMT). See doc/quant/per_token/06_split_requant.md"""

import torch
import tilelang
import tilelang.ascend.language as T

from harness import status
from harness.consts import CANONICAL_G, E4M3_CLAMP_MIN, E4M3_MAX

THREADS = 128


@tilelang.jit(target="ascend")
def compile_kernel(hidden: int, mode: str = "full", group_size: int = CANONICAL_G):
    """mode in {"full", "sf_only", "cast_only", "requant"}.

    The mode flags are plain Python read at trace time, exactly as in the SIMD
    tiers, so only the selected passes are ever emitted. This variant ports with
    no thought at all -- which is itself the observation: composition is a
    property of tilelang's tracing, not of the vector style underneath.
    """
    assert hidden % group_size == 0 and group_size == 32
    assert mode in ("full", "sf_only", "cast_only", "requant")
    num_groups = hidden // group_size
    num_tokens = T.dynamic("num_tokens")

    need_amax = mode in ("full", "sf_only", "requant")
    need_quant = mode in ("full", "cast_only", "requant")
    sf_is_input = mode == "cast_only"
    x_dtype = T.float8_e4m3fn if mode == "requant" else T.bfloat16

    @T.prim_func
    def per_token_cast(
        X: T.Tensor((num_tokens, hidden), x_dtype),
        XSf: T.Tensor((num_tokens, num_groups), T.float32),
        Q: T.Tensor((num_tokens, hidden), T.float8_e4m3fn),
        Sf: T.Tensor((num_tokens, num_groups), T.float32),
    ):
        with T.Kernel(1):
            x_ub = T.alloc_shared((hidden,), x_dtype)
            q_ub = T.alloc_shared((hidden,), T.float8_e4m3fn)
            val_ub = T.alloc_shared((hidden,), T.float32)   # dequantized scratch
            sf_ub = T.alloc_shared((num_groups,), T.float32)
            xsf_ub = T.alloc_shared((num_groups,), T.float32)

            for token in T.serial(num_tokens):
                T.copy(X[token, 0], x_ub)
                if mode in ("cast_only", "requant"):
                    T.copy(XSf[token, 0], xsf_ub)

                # --- BEGIN SOLUTION hint="gate the passes on the mode, the same way the SIMD tiers do -- `mode` is a Python string read at trace time, so an `if` here chooses what gets emitted, not what gets executed. requant first dequantizes into val_ub with one T.Parallel(hidden) doing val_ub[i] = T.Cast('float32', x_ub[i]) * xsf_ub[i // group_size], which is the whole of variant cast_back/01. sf_only emits the reduction and the scale only; cast_only skips the reduction and uses inv = 1.0 / xsf_ub[g]. Everything else is variant 01. Read the dequantized values back out of val_ub rather than a fragment: the two passes index it differently, and only shared buffers may be touched with more than one index pattern."
                with T.SimtVF(threads=THREADS):
                    q_l = T.alloc_fragment((hidden,), T.float32)
                    inv_l = T.alloc_fragment((num_groups,), T.float32)
                    sf_l = T.alloc_fragment((num_groups,), T.float32)

                    if mode == "requant":
                        # stage 0: dequantize into the scratch buffer. This is
                        # cast_back/01 inlined -- one line, where ASC needs four
                        # loads and two vsel to broadcast the scale.
                        for i in T.Parallel(hidden):
                            val_ub[i] = (T.Cast("float32", x_ub[i])
                                         * xsf_ub[i // group_size])

                    if need_amax:
                        for g in T.Parallel(num_groups):
                            amax = T.alloc_var(T.float32)
                            amax = E4M3_CLAMP_MIN
                            for j in T.serial(group_size):
                                if mode == "requant":
                                    v = val_ub[g * group_size + j]
                                else:
                                    v = T.Cast("float32", x_ub[g * group_size + j])
                                amax = T.max(amax, T.abs(v))
                            inv_l[g] = E4M3_MAX / amax
                            sf_l[g] = amax / E4M3_MAX
                    elif sf_is_input:
                        # cast_only: invert the scale we were handed. This
                        # reciprocal is why cast_only can differ from the fused
                        # path by a single code.
                        for g in T.Parallel(num_groups):
                            inv_l[g] = 1.0 / xsf_ub[g]

                    if need_quant:
                        for i in T.Parallel(hidden):
                            if mode == "requant":
                                v = val_ub[i]
                            else:
                                v = T.Cast("float32", x_ub[i])
                            q_l[i] = v * inv_l[i // group_size]
                        T.copy(q_l, q_ub)

                    if need_amax:
                        T.copy(sf_l, sf_ub)
                # --- END SOLUTION

                if need_quant:
                    T.copy(q_ub, Q[token, 0])
                if need_amax:
                    T.copy(sf_ub, Sf[token, 0])

    return per_token_cast


def launch(x: torch.Tensor, mode: str = "full", x_sf: torch.Tensor | None = None):
    m, hidden = x.shape
    ng = hidden // CANONICAL_G
    dev = x.device
    xsf = x_sf if x_sf is not None else torch.zeros((m, ng), dtype=torch.float32, device=dev)
    q = torch.empty((m, hidden), dtype=torch.float8_e4m3fn, device=dev)
    sf = torch.empty((m, ng), dtype=torch.float32, device=dev)
    compile_kernel(hidden, mode)(x, xsf, q, sf)
    status.assert_on_device(f"per_token 06 {mode}", q, sf)
    return q, sf

if __name__ == "__main__":        # not a script -- see the module docstring
    raise SystemExit("Run it through the harness:  python -m harness.check simt/per_token/06")
