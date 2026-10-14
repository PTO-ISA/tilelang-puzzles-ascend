"""cast_back 01 (SIMT). See doc/quant/cast_back/01_e4m3_fp32sf.md"""

import torch
import tilelang
import tilelang.ascend.language as T

from harness import status
from harness.consts import CANONICAL_G

THREADS = 128       # one SIMT workgroup; T.Parallel spreads its extent over these


@tilelang.jit(target="ascend", out_idx=[2])
def compile_kernel(hidden: int, out_dtype: str = "bfloat16",
                   group_size: int = CANONICAL_G):
    """Dequantize FP8 with one FP32 scale per `group_size` channels.

    The SIMT tier targets the **ascend** backend, not `pto`: `T.SimtVF` lowers
    through the Ascend path, and the PTO backend has known SimtVF gaps.

    `out_dtype` is still a parameter, to stay 1:1 with the other tiers -- but
    notice the kernel body never branches on it. ASC has to pick a distribution
    mode (`PK_B32` against `NORM_B32`) and PTO has to add or drop a convert;
    here the destination buffer's dtype carries the whole config.
    """
    assert hidden % 128 == 0, "this teaching kernel steps 128 values at a time"
    assert group_size == 32, "Ascend quant granularity is fixed at 32"
    assert out_dtype in ("bfloat16", "float32"), out_dtype
    out_t = T.bfloat16 if out_dtype == "bfloat16" else T.float32
    num_groups = hidden // group_size
    num_tokens = T.dynamic("num_tokens")

    @T.prim_func
    def cast_back(
        Q: T.Tensor((num_tokens, hidden), T.float8_e4m3fn),
        Sf: T.Tensor((num_tokens, num_groups), T.float32),
        Out: T.Tensor((num_tokens, hidden), out_t),
    ):
        # One vector core, as in the ASC and PTO tiers.
        with T.Kernel(1):
            q_ub = T.alloc_shared((hidden,), T.float8_e4m3fn)
            sf_ub = T.alloc_shared((num_groups,), T.float32)
            out_ub = T.alloc_shared((hidden,), out_t)

            for token in T.serial(num_tokens):
                T.copy(Q[token, 0], q_ub)
                T.copy(Sf[token, 0], sf_ub)

                # --- BEGIN SOLUTION hint="open `with T.SimtVF(threads=128):`, bring the FP8 values into a float32 fragment with T.copy (it converts), then write the whole kernel as one T.Parallel over hidden: o_l[i] = q_l[i] * sf_ub[i // group_size]. Reading the scale straight out of shared memory is allowed -- the one-index-pattern rule applies to fragments, not to shared buffers -- so no broadcast, no mask and no select is needed. T.copy back to out_ub converts to the output dtype."
                with T.SimtVF(threads=THREADS):
                    q_l = T.alloc_fragment((hidden,), T.float32)
                    o_l = T.alloc_fragment((hidden,), T.float32)
                    T.copy(q_ub, q_l)       # FP8 -> float32, per element

                    # The whole kernel. `i // group_size` picks the scale, so the
                    # broadcast that costs ASC two BRC_B32 loads and a vsel is
                    # just an index expression here.
                    for i in T.Parallel(hidden):
                        o_l[i] = q_l[i] * sf_ub[i // group_size]

                    T.copy(o_l, out_ub)     # float32 -> out_dtype
                # --- END SOLUTION

                T.copy(out_ub, Out[token, 0])

    return cast_back


def launch(q: torch.Tensor, sf: torch.Tensor,
           out_dtype: str = "bfloat16") -> torch.Tensor:
    """Run the kernel. Returns a device tensor -- never a host-computed answer."""
    kernel = compile_kernel(q.shape[1], out_dtype)
    out = kernel(q, sf)
    status.assert_on_device(f"cast_back 01 {out_dtype}", out)
    return out

if __name__ == "__main__":        # not a script -- see the module docstring
    raise SystemExit("Run it through the harness:  python -m harness.check simt/cast_back/01")
