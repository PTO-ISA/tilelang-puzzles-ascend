"""per_block 04 (ASC). See doc/quant/per_block/04_col_major_tma.md"""

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
        SfCm: T.Tensor((num_k_blocks, num_m_blocks), T.float32),
    ):
        with T.Kernel(1):
            x_ub = T.alloc_shared((bm, bk), T.bfloat16)
            q_ub = T.alloc_shared((bm, bk), T.float8_e4m3fn)
            run_ub = T.alloc_shared((run_pad,), T.float32)   # per-chunk maxima
            sf_ub = T.alloc_shared((LANES,), T.float32)
            # Flat views of the UB tiles: contiguous, so 64-lane ops can walk them.
            flat_x = T.Tensor((tile_values,), T.bfloat16, x_ub.data)
            flat_q = T.Tensor((tile_values,), T.float8_e4m3fn, q_ub.data)

            for mb in T.serial(num_m_blocks):
                for kb in T.serial(num_k_blocks):
                    T.copy(X[mb * bm, kb * bk], x_ub)
                    # --- BEGIN SOLUTION hint="two-level reduction. (1) for each of the 16 chunks of 64, load from flat_x with dist='UNPK_B16', vcvt to float32, vabs, and store S.vcmax(..., mask_all) to run_ub[chunk] with dist='ONEPT_B32'. (2) barrier, then reduce the 16 partials with S.vcmax(S.vld(run_ub[0]), mask_vl16), clamp, divide both ways, store the scale and keep the inverse. (3) barrier, then reload each chunk, multiply by the broadcast inverse and store FP8 with dist='PK4_B32'."
                    with T.SimdVF():
                        mask_all = S.pset(32, "PAT_ALL")
                        mask_chunks = S.pset(32, f"PAT_VL{num_chunks}")
                        qmax = S.vdup(E4M3_MAX, T.float32)

                        # ---- pass 1: 16 partial maxima ----
                        for chunk in T.serial(num_chunks):
                            v = S.vabs(S.vcvt(S.vld(flat_x[chunk * LANES],
                                                    dist="UNPK_B16"), T.float32))
                            S.vsts(run_ub[chunk], S.vcmax(v, mask_all),
                                   dist="ONEPT_B32")
                        S.mem_bar("VST_VLD")

                        # ---- pass 2: reduce the partials, then one scale ----
                        tile_amax = S.vcmax(S.vld(run_ub[0]), mask_chunks)
                        clamped = S.vmaxs(tile_amax, E4M3_CLAMP_MIN)
                        S.vsts(sf_ub[0], S.vdiv(clamped, qmax), dist="ONEPT_B32")
                        S.vsts(sf_ub[1], S.vdiv(qmax, clamped), dist="ONEPT_B32")
                        S.mem_bar("VST_VLD")

                        # ---- pass 3: one scale for the whole tile ----
                        inv = S.vld(sf_ub[1], dist="BRC_B32")
                        for chunk in T.serial(num_chunks):
                            v = S.vcvt(S.vld(flat_x[chunk * LANES], dist="UNPK_B16"),
                                       T.float32)
                            S.vsts(flat_q[chunk * LANES],
                                   S.vcvt(S.vmul(v, inv), T.float8_e4m3fn),
                                   dist="PK4_B32")
                    # --- END SOLUTION
                    T.copy(q_ub, Q[mb * bm, kb * bk])
                    # the transposed index is the entire change
                    T.copy(sf_ub[0:1], SfCm[kb, mb:mb + 1])

    return per_block_cast


def launch(x: torch.Tensor):
    m, hidden = x.shape
    q = torch.empty((m, hidden), dtype=torch.float8_e4m3fn, device=x.device)
    sf = torch.empty((hidden // BLOCK_K, m // BLOCK_MN), dtype=torch.float32,
                     device=x.device)
    compile_kernel(hidden)(x, q, sf)
    status.assert_on_device("per_block 04", q, sf)
    return q, sf
