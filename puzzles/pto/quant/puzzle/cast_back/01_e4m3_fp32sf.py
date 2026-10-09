"""cast_back 01 (PTO). See doc/quant/cast_back/01_e4m3_fp32sf.md"""

import torch
import tilelang
import tilelang.ascend.language as T
from tilelang.ascend.language import vmi as V

from harness import status
from harness.consts import CANONICAL_G

LANES = 64
SF_PAD = 64


@tilelang.jit(target="pto", out_idx=[2])
def compile_kernel(hidden: int, out_dtype: str = "bfloat16",
                   group_size: int = CANONICAL_G):
    """Dequantize FP8 with one FP32 scale per `group_size` channels.

    `out_dtype` is a Python string, so the branch on it below is taken at trace
    time and only the chosen store is emitted.
    """
    assert hidden % 128 == 0, "this teaching kernel steps two 64-lane strips at a time"
    assert group_size == 32, "Ascend quant granularity is fixed at 32"
    assert out_dtype in ("bfloat16", "float32"), out_dtype
    out_t = T.bfloat16 if out_dtype == "bfloat16" else T.float32
    num_groups = hidden // group_size
    groups_per_strip = LANES // group_size      # 2
    num_tokens = T.dynamic("num_tokens")

    @T.prim_func
    def cast_back(
        Q: T.Tensor((num_tokens, hidden), T.float8_e4m3fn),
        Sf: T.Tensor((num_tokens, num_groups), T.float32),
        Out: T.Tensor((num_tokens, hidden), out_t),
    ):
        with T.Kernel(1):
            q_ub = T.alloc_shared((hidden,), T.float8_e4m3fn)
            sf_ub = T.alloc_shared((SF_PAD,), T.float32)
            out_ub = T.alloc_shared((hidden,), out_t)

            for token in T.serial(num_tokens):
                T.copy(Q[token, 0], q_ub)
                T.copy(Sf[token, 0], sf_ub[0:num_groups])

                # TODO: open `with T.SimdVF():`; mask = V.create_mask(64,
                #       size=64); loop strip over hidden//64; values =
                #       V.vcvt(V.vload(q_ub[col], size=64), 'float32'); scale =
                #       V.vload(sf_ub[group], size=64, stride=1, dist_mode='brc',
                #       group=2) -- one load, no select; then V.vmul, and branch
                #       on out_dtype for the store: V.vstore(V.vcvt(scaled,
                #       'bfloat16'), out_ub[col]) or V.vstore(scaled,
                #       out_ub[col]). Note the call is spelled the same either way
                #       -- the destination buffer's dtype decides the packing, so
                #       only the convert appears or disappears. Compare the ASC
                #       file, which also has to swap the distribution mode.
                raise NotImplementedError("pto/cast_back/01_e4m3_fp32sf: implement cast_back")

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
    raise SystemExit("Run it through the harness:  python -m harness.check pto/cast_back/01 --role puzzle")
