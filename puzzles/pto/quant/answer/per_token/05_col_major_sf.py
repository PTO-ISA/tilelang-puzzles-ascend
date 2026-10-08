"""per_token 05 (PTO). See doc/quant/per_token/05_col_major_sf.md"""

import torch
import tilelang
import tilelang.ascend.language as T
from tilelang.ascend.language import vmi as V

from harness import status
from harness.consts import BLOCK_MN, CANONICAL_G, E4M3_CLAMP_MIN, E4M3_MAX

LANES = 64
PAIR = 128
SF_STRIDE = 64          # padded row length of the token-major scale buffer


@tilelang.jit(target="pto")
def compile_kernel(hidden: int, group_size: int = CANONICAL_G,
                   token_block: int = BLOCK_MN):
    """Quantize, writing the scales transposed as (num_groups, num_tokens)."""
    assert hidden % PAIR == 0 and group_size == 32 and token_block == 32
    num_groups = hidden // group_size
    log2_block = token_block.bit_length() - 1          # 5
    num_out_values = num_groups * token_block
    groups_per_gather = LANES // token_block           # 2
    assert num_out_values % LANES == 0, "this teaching kernel wants whole gathers"
    num_gathers = num_out_values // LANES
    num_tokens = T.dynamic("num_tokens")
    num_blocks = T.ceildiv(num_tokens, token_block)
    inv_qmax = 1.0 / E4M3_MAX

    @T.prim_func
    def per_token_cast(
        X: T.Tensor((num_tokens, hidden), T.bfloat16),
        Q: T.Tensor((num_tokens, hidden), T.float8_e4m3fn),
        SfCm: T.Tensor((num_groups, num_tokens), T.float32),
    ):
        with T.Kernel(1):
            x_ub = T.alloc_shared((hidden,), T.bfloat16)
            q_ub = T.alloc_shared((hidden,), T.float8_e4m3fn)
            amax_ub = T.alloc_shared((SF_STRIDE,), T.float32)
            inv_ub = T.alloc_shared((SF_STRIDE,), T.float32)
            # token-major scales for a block of tokens, then their transpose
            sf_dense_ub = T.alloc_shared((token_block, SF_STRIDE), T.float32)
            sf_out_ub = T.alloc_shared((num_groups, token_block), T.float32)
            idx_ub = T.alloc_shared((LANES,), T.uint32)

            # --- BEGIN SOLUTION hint="build the gather index vector once with V.vci(T.int32(0), size=64): token = lane & (token_block-1), group = lane >> log2(token_block), idx = token*SF_STRIDE + group -- note there is no scalar-operand vmul, so broadcast the stride. Then per token compute scales into sf_dense_ub[row, :], and after the block transpose with V.vgather(sf_dense_ub[0, base], idx, mask) -> V.vstore(..., sf_out_ub[base, 0])"
            # One lane-index vector serves every block; compute it once.
            with T.SimdVF():
                lane = V.vci(T.int32(0), size=LANES)
                mask64 = V.create_mask(LANES, size=LANES)
                token_of_lane = V.vand(V.vinterpret_cast(lane, "uint32"),
                                       V.vbrc(T.uint32(token_block - 1), size=LANES))
                group_of_lane = V.vinterpret_cast(
                    V.vshrs(lane, log2_block, mask64), "uint32")
                # No scalar-operand vmul in VMI: the stride must be a vector.
                stride_v = V.vbrc(T.uint32(SF_STRIDE), size=LANES)
                V.vstore(V.vadd(V.vmul(token_of_lane, stride_v, mask64),
                                group_of_lane), idx_ub[0])

            for blk in T.serial(num_blocks):
                for row in T.serial(token_block):
                    token = blk * token_block + row
                    T.copy(X[token, 0], x_ub)
                    with T.SimdVF():
                        mask = V.create_mask(PAIR, size=PAIR)
                        m64 = V.create_mask(LANES, size=LANES)
                        shift = V.vbrc(T.uint32(23), size=LANES)
                        one = V.vbrc(T.uint32(1), size=LANES)
                        for pair in T.serial(hidden // PAIR):
                            col = pair * PAIR
                            group = pair * (PAIR // group_size)
                            xv = V.vcvt(V.vload(x_ub[col], size=PAIR), "float32")
                            V.vstore(V.vcmax(V.vabs(xv, mask), mask,
                                             group=PAIR // group_size), amax_ub[group])
                        T.simd.mem_bar("VST_VLD")

                        clamped = V.vmax(V.vload(amax_ub[0], size=LANES),
                                         V.vbrc(T.float32(E4M3_CLAMP_MIN), size=LANES),
                                         m64)
                        bits = V.vinterpret_cast(
                            V.vmul(clamped, V.vbrc(T.float32(inv_qmax), size=LANES), m64),
                            "uint32")
                        biased = V.vadd(V.vshr(V.vsub(bits, one), shift), one)
                        # Scales land token-major; transposed after the block.
                        V.vstore(V.vinterpret_cast(V.vshl(biased, shift), "float32"),
                                 sf_dense_ub[row, 0])
                        V.vstore(V.vinterpret_cast(
                            V.vshl(V.vsub(V.vbrc(T.uint32(254), size=LANES), biased),
                                   shift), "float32"), inv_ub[0])
                        T.simd.mem_bar("VST_VLD")

                        for pair in T.serial(hidden // PAIR):
                            col = pair * PAIR
                            group = pair * (PAIR // group_size)
                            inv = V.vload(inv_ub[group], size=PAIR, stride=1,
                                          dist_mode="brc", group=PAIR // group_size)
                            xv = V.vcvt(V.vload(x_ub[col], size=PAIR), "float32")
                            V.vstore(V.vcvt(V.vmul(xv, inv, mask), "float8_e4m3fn",
                                            rounding="R", saturate="SAT"), q_ub[col])
                    T.copy(q_ub, Q[token, 0])

                # transpose the block's scales in register, then one DMA out
                with T.SimdVF():
                    T.simd.mem_bar("VST_VLD")
                    gmask = V.create_mask(LANES, size=LANES)
                    idx = V.vload(idx_ub[0], size=LANES)
                    for g in T.serial(num_gathers):
                        base = g * groups_per_gather
                        # vgather requires the mask; vgather2 takes it optionally.
                        V.vstore(V.vgather(sf_dense_ub[0, base], idx, gmask),
                                 sf_out_ub[base, 0])
                T.copy(sf_out_ub, SfCm[0, blk * token_block])
            # --- END SOLUTION

    return per_token_cast


def launch(x: torch.Tensor):
    m, hidden = x.shape
    num_groups = hidden // CANONICAL_G
    q = torch.empty((m, hidden), dtype=torch.float8_e4m3fn, device=x.device)
    sf_cm = torch.empty((num_groups, m), dtype=torch.float32, device=x.device)
    compile_kernel(hidden)(x, q, sf_cm)
    status.assert_on_device("per_token 05", q, sf_cm)
    return q, sf_cm

if __name__ == "__main__":        # not a script -- see the module docstring
    raise SystemExit("Run it through the harness:  python -m harness.check pto/per_token/05")
