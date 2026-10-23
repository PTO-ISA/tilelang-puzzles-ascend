"""cast_back 06 (SIMT). See doc/quant/cast_back/06_col_major_compose.md"""

import torch
import tilelang
import tilelang.ascend.language as T

from harness import status
from harness.consts import BLOCK_MN, CANONICAL_G, PACK_FACTOR

THREADS = 128


@tilelang.jit(target="ascend", out_idx=[2])
def compile_kernel(hidden: int, group_size: int = CANONICAL_G,
                   token_block: int = BLOCK_MN):
    """Dequantize FP4 values with column-major packed-UE8M0 scales.

    Both layout configs at once. As in variant 05 the kernel takes the FP4
    values as raw `uint8`, since no FP4 convert is reachable from SimtVF.
    """
    assert hidden % 128 == 0 and group_size == 32 and token_block == 32
    num_groups = hidden // group_size
    num_words = num_groups // PACK_FACTOR
    num_bytes = hidden // 2
    num_tokens = T.dynamic("num_tokens")
    num_blocks = T.ceildiv(num_tokens, token_block)

    @T.prim_func
    def cast_back(
        Q: T.Tensor((num_tokens, num_bytes), T.uint8),
        SfCm: T.Tensor((num_words, num_tokens), T.uint16),
        Out: T.Tensor((num_tokens, hidden), T.bfloat16),
    ):
        with T.Kernel(1):
            q_ub = T.alloc_shared((num_bytes,), T.uint8)
            # A (num_words, token_block) REGION of the transposed scale array.
            # Consuming a transposed layout costs nothing: the DMA takes the
            # block and the kernel just indexes it by [word, token].
            sfcm_ub = T.alloc_shared((num_words, token_block), T.uint16)
            out_ub = T.alloc_shared((hidden,), T.bfloat16)

            for blk in T.serial(num_blocks):
                T.copy(SfCm[0, blk * token_block], sfcm_ub)

                for sub in T.serial(token_block):
                    T.copy(Q[blk * token_block + sub, 0], q_ub)

                    # --- BEGIN SOLUTION hint="variant 05's nibble decode and variant 02's byte decode, in one expression. The scale now comes from the transposed block: g = i // group_size, then word = T.Cast('int32', sfcm_ub[g // 2, sub]) and byte = (word >> (8 * (g % 2))) & 0xFF, decoded as T.reinterpret('float32', byte << 23). Nothing about consuming a transposed layout costs anything here -- the index just has its two axes the other way round."
                    with T.SimtVF(threads=THREADS):
                        o_l = T.alloc_fragment((hidden,), T.float32)

                        for i in T.Parallel(hidden):
                            # the FP4 value
                            byte = T.Cast("int32", q_ub[i // 2])
                            nib = (byte >> (4 * (i % 2))) & 0x0F
                            sgn = (nib >> 3) & 1
                            e = (nib >> 1) & 3
                            m = T.Cast("float32", nib & 1)
                            sub_v = m * 0.5
                            nrm = (1.0 + m * 0.5) * T.reinterpret("float32", (126 + e) << 23)
                            mag = T.if_then_else(e == 0, sub_v, nrm)
                            val = T.if_then_else(sgn == 1, -mag, mag)

                            # the scale: one exponent byte, from the transposed block
                            g = i // group_size
                            word = T.Cast("int32", sfcm_ub[g // PACK_FACTOR, sub])
                            ebyte = (word >> (8 * (g % PACK_FACTOR))) & 0xFF
                            o_l[i] = val * T.reinterpret("float32", ebyte << 23)

                        T.copy(o_l, out_ub)
                    # --- END SOLUTION

                    T.copy(out_ub, Out[blk * token_block + sub, 0])

    return cast_back


def launch(q_packed: torch.Tensor, sf_cm: torch.Tensor) -> torch.Tensor:
    hidden = q_packed.shape[1] * 2
    out = compile_kernel(hidden)(q_packed.view(torch.uint8), sf_cm.view(torch.uint16))
    status.assert_on_device("cast_back 06", out)
    return out

if __name__ == "__main__":        # not a script -- see the module docstring
    raise SystemExit("Run it through the harness:  python -m harness.check simt/cast_back/06")
