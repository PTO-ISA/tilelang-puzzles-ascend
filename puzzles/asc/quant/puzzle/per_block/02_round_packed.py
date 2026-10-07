"""per_block 02 (ASC). See doc/quant/per_block/02_round_packed.md"""

import torch
import tilelang
import tilelang.ascend.language as T
from tilelang.ascend.language import simd as S

from harness import status
from harness.consts import BLOCK_K, BLOCK_MN, E4M3_CLAMP_MIN, E4M3_MAX

LANES = 64


@tilelang.jit(target="ascend")
def compile_kernel(hidden: int, block: tuple = (BLOCK_MN, BLOCK_K)):
    """Quantize bfloat16 -> FP8 e4m3 with one FP32 scale per `block` tile."""
    bm, bk = block
    assert bm == 32 and bk == 32, "Ascend per_block granularity is 32x32"
    assert hidden % bk == 0
    tile_values = bm * bk                       # 1024
    num_chunks = tile_values // LANES           # 16 reduction steps
    num_k_blocks = hidden // bk
    num_tokens = T.dynamic("num_tokens")
    num_m_blocks = T.ceildiv(num_tokens, bm)
    run_pad = max(num_chunks, LANES)

    @T.prim_func
    def per_block_cast(
        X: T.Tensor((num_tokens, hidden), T.bfloat16),
        Q: T.Tensor((num_tokens, hidden), T.float8_e4m3fn),
        Sf: T.Tensor((num_m_blocks, num_k_blocks), T.uint8),
    ):
        with T.Kernel(1):
            x_ub = T.alloc_shared((bm, bk), T.bfloat16)
            q_ub = T.alloc_shared((bm, bk), T.float8_e4m3fn)
            run_ub = T.alloc_shared((run_pad,), T.float32)   # per-chunk maxima
            sf_ub = T.alloc_shared((LANES,), T.uint8)
            inv_ub = T.alloc_shared((LANES,), T.float32)
            # Flat views of the UB tiles: contiguous, so 64-lane ops can walk them.
            flat_x = T.Tensor((tile_values,), T.bfloat16, x_ub.data)
            flat_q = T.Tensor((tile_values,), T.float8_e4m3fn, q_ub.data)

            for mb in T.serial(num_m_blocks):
                for kb in T.serial(num_k_blocks):
                    T.copy(X[mb * bm, kb * bk], x_ub)
                    # TODO: two-level reduction. (1) for each of the 16 chunks of
                    #       64, load from flat_x with dist='UNPK_B16', vcvt to
                    #       float32, vabs, and store S.vcmax(..., mask_all) to
                    #       run_ub[chunk] with dist='ONEPT_B32'. (2) barrier, then
                    #       reduce the 16 partials with S.vcmax(S.vld(run_ub[0]),
                    #       mask_vl16), clamp, divide both ways, store the scale
                    #       and keep the inverse. (3) barrier, then reload each
                    #       chunk, multiply by the broadcast inverse and store FP8
                    #       with dist='PK4_B32'.
                    raise NotImplementedError("asc/per_block/02_round_packed: implement per_block_cast")
                    T.copy(q_ub, Q[mb * bm, kb * bk])
                    T.copy(sf_ub[0:1], Sf[mb, kb:kb + 1])

    return per_block_cast


def launch(x: torch.Tensor):
    """Returns (q, sf_packed) with sf_packed int16, two tile exponents per word."""
    m, hidden = x.shape
    q = torch.empty((m, hidden), dtype=torch.float8_e4m3fn, device=x.device)
    sf_bytes = torch.empty((m // BLOCK_MN, hidden // BLOCK_K), dtype=torch.uint8,
                           device=x.device)
    compile_kernel(hidden)(x, q, sf_bytes)
    status.assert_on_device("per_block 02", q, sf_bytes)
    return q, sf_bytes.view(torch.int16)
