"""cast_back 06 (PTO). See doc/quant/cast_back/05_col_major_compose.md"""

import torch
import tilelang
import tilelang.ascend.language as T
from tilelang.ascend.language import vmi as V

from harness import status
from harness.consts import BLOCK_MN, CANONICAL_G, PACK_FACTOR

FP4_STRIP = 128
EXP_MASK = 0x7F800000


@tilelang.jit(target="pto", out_idx=[2])
def compile_kernel(hidden: int, group_size: int = CANONICAL_G,
                   token_block: int = BLOCK_MN):
    """Packed UE8M0 + column-major scales + FP4 values -> bfloat16."""
    assert hidden % FP4_STRIP == 0 and group_size == 32
    num_groups = hidden // group_size
    num_words = num_groups // PACK_FACTOR
    words_per_strip = FP4_STRIP // (group_size * PACK_FACTOR)    # 2
    num_tokens = T.dynamic("num_tokens")
    num_blocks = T.ceildiv(num_tokens, token_block)

    @T.prim_func
    def cast_back(
        Q: T.Tensor((num_tokens, hidden), T.float4_e2m1fn),
        SfCm: T.Tensor((num_words, num_tokens), T.uint16),
        Out: T.Tensor((num_tokens, hidden), T.bfloat16),
    ):
        with T.Kernel(1):
            q_ub = T.alloc_shared((hidden,), T.float4_e2m1fn)
            sf_ub = T.alloc_shared((num_words, token_block), T.uint16)
            out_ub = T.alloc_shared((hidden,), T.bfloat16)
            for blk in T.serial(num_blocks):
                T.copy(SfCm[0, blk * token_block], sf_ub)
                for row in T.serial(token_block):
                    token = blk * token_block + row
                    T.copy(Q[token, 0], q_ub)
                    # --- BEGIN SOLUTION hint="stay at 128 lanes: x_f32 = V.vzip(zero_bf16, V.vcvt(V.vload(q_ub[col], size=128), 'bfloat16'), 'float32'); fetch both packed words in one go with V.vload(sf_ub[strip*2, row], size=256, stride=token_block, dist_mode='brc', group=2); build sf_shift from V.create_mask(32, size=128, group=2); then shift, mask, reinterpret to float32, multiply and store once"
                    with T.SimdVF():
                        mask = V.create_mask(FP4_STRIP, size=FP4_STRIP)
                        # "first 32 of every 64 lanes", repeating -- there is no
                        # 128-lane equivalent of ASC's PAT_VL32.
                        mask_low = V.create_mask(32, size=FP4_STRIP, group=2)
                        sf_shift = V.vsel(mask_low,
                                          V.vbrc(T.uint32(23), size=FP4_STRIP),
                                          V.vbrc(T.uint32(15), size=FP4_STRIP))
                        exp_mask = V.vbrc(T.uint32(EXP_MASK), size=FP4_STRIP)
                        zero_bf16 = V.vbrc(T.bfloat16(0.0), size=FP4_STRIP)
                        for strip in T.serial(hidden // FP4_STRIP):
                            col = strip * FP4_STRIP
                            word = strip * words_per_strip

                            # 128 FP4 -> 128 bfloat16 -> 128 float32, one name.
                            x_bf16 = V.vcvt(V.vload(q_ub[col], size=FP4_STRIP),
                                            "bfloat16")
                            x_f32 = V.vzip(zero_bf16, x_bf16, "float32")

                            # Two packed words, each broadcast over 64 lanes, in
                            # one load. The transposed layout is just the index.
                            packed = V.vload(sf_ub[word, row], size=FP4_STRIP * 2,
                                             stride=token_block, dist_mode="brc",
                                             group=words_per_strip)
                            bits = V.vand(
                                V.vshl(V.vinterpret_cast(packed, "uint32"), sf_shift),
                                exp_mask)
                            scale = V.vinterpret_cast(bits, "float32")

                            scaled = V.vmul(x_f32, scale, mask)
                            V.vstore(V.vcvt(scaled, "bfloat16"), out_ub[col])
                    # --- END SOLUTION
                    T.copy(out_ub, Out[token, 0])

    return cast_back


def launch(q_packed: torch.Tensor, sf_cm: torch.Tensor) -> torch.Tensor:
    hidden = q_packed.shape[1] * 2
    q = q_packed.view(torch.uint8).view(torch.float4_e2m1fn_x2)
    out = compile_kernel(hidden)(q, sf_cm.view(torch.uint16))
    status.assert_on_device("cast_back 06", out)
    return out

if __name__ == "__main__":        # not a script -- see the module docstring
    raise SystemExit("Run it through the harness:  python -m harness.check pto/cast_back/06")
