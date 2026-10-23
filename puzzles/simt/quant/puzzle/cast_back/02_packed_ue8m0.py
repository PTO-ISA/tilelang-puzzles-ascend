"""cast_back 02 (SIMT). See doc/quant/cast_back/02_packed_ue8m0.md"""

import torch
import tilelang
import tilelang.ascend.language as T

from harness import status
from harness.consts import CANONICAL_G, PACK_FACTOR

THREADS = 128


@tilelang.jit(target="ascend", out_idx=[2])
def compile_kernel(hidden: int, group_size: int = CANONICAL_G):
    """Dequantize FP8 -> bfloat16 with packed-UE8M0 scales."""
    assert hidden % 128 == 0 and group_size == 32
    num_groups = hidden // group_size
    num_words = num_groups // PACK_FACTOR       # two exponents per int16
    num_tokens = T.dynamic("num_tokens")

    @T.prim_func
    def cast_back(
        Q: T.Tensor((num_tokens, hidden), T.float8_e4m3fn),
        Sf: T.Tensor((num_tokens, num_words), T.uint16),
        Out: T.Tensor((num_tokens, hidden), T.bfloat16),
    ):
        with T.Kernel(1):
            q_ub = T.alloc_shared((hidden,), T.float8_e4m3fn)
            sf_ub = T.alloc_shared((num_words,), T.uint16)
            out_ub = T.alloc_shared((hidden,), T.bfloat16)

            for token in T.serial(num_tokens):
                T.copy(Q[token, 0], q_ub)
                T.copy(Sf[token, 0], sf_ub)

                # TODO: one T.Parallel over hidden does the whole thing. For
                #       element i: g = i // group_size is its group, w = g // 2
                #       the int16 that holds it and half = g % 2 which byte. Pull
                #       the byte out with ((T.Cast('int32', sf_ub[w]) >> (8 *
                #       half)) & 0xFF) -- low byte first -- and a UE8M0 byte e
                #       decodes as T.reinterpret('float32', e << 23), since the
                #       format is a bare exponent. Multiply the widened FP8 value
                #       by it. No vector-lane bookkeeping: ASC needs a shift, a
                #       mask and a select to get the right byte into the right
                #       lanes.
                raise NotImplementedError("simt/cast_back/02_packed_ue8m0: implement cast_back")

                T.copy(out_ub, Out[token, 0])

    return cast_back


def launch(q: torch.Tensor, sf_packed: torch.Tensor) -> torch.Tensor:
    """`sf_packed` is int16; the kernel reads it as uint16 for the shifts."""
    out = compile_kernel(q.shape[1])(q, sf_packed.view(torch.uint16))
    status.assert_on_device("cast_back 02", out)
    return out

if __name__ == "__main__":        # not a script -- see the module docstring
    raise SystemExit("Run it through the harness:  python -m harness.check simt/cast_back/02 --role puzzle")
