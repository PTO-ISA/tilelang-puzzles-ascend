"""per_block 01 (SIMT). See doc/quant/per_block/01_raw_32x32.md"""

import torch
import tilelang
import tilelang.ascend.language as T

from harness import status
from harness.consts import BLOCK_K, BLOCK_MN, E4M3_CLAMP_MIN, E4M3_MAX

THREADS = 128


@tilelang.jit(target="ascend")
def compile_kernel(hidden: int, block: tuple = (BLOCK_MN, BLOCK_K)):
    """Quantize bfloat16 -> FP8 e4m3 with one FP32 scale per `block` tile.

    One scale for 1024 values means a reduction all the way down to a scalar.
    The SIMD tiers need two levels for it -- a vector reduction stops at one
    value per register, so the partials have to meet in UB -- and so does the
    shape a GPU programmer reaches for first: a partial per thread into shared
    memory, then one thread finishing the job.

    Neither is written here. `T.reduce_absmax` over the tile laid out as a
    single row does the whole fold in one call, and that is both shorter than
    the two-level scan and 7.24x fewer cycles, because the scan's second level
    runs on one thread of 128. `harness/probe/simt_reduce_max.py` measures the
    same choice on `per_token/01`.
    """
    bm, bk = block
    assert bm == 32 and bk == 32, "Ascend per_block granularity is 32x32"
    assert hidden % bk == 0
    num_k_blocks = hidden // bk
    num_tokens = T.dynamic("num_tokens")
    num_m_blocks = T.ceildiv(num_tokens, bm)

    @T.prim_func
    def per_block_cast(
        X: T.Tensor((num_tokens, hidden), T.bfloat16),
        Q: T.Tensor((num_tokens, hidden), T.float8_e4m3fn),
        Sf: T.Tensor((num_m_blocks, num_k_blocks), T.float32),
    ):
        with T.Kernel(1):
            # A 2-D UB tile is right here, unlike per_token: the tile really is
            # 32 tokens by 32 channels, so T.copy's 2-D region is what we want.
            x_ub = T.alloc_shared((bm, bk), T.bfloat16)
            q_ub = T.alloc_shared((bm, bk), T.float8_e4m3fn)
            sf_ub = T.alloc_shared((1,), T.float32)
            inv_ub = T.alloc_shared((1,), T.float32)

            for mb in T.serial(num_m_blocks):
                for kb in T.serial(num_k_blocks):
                    T.copy(X[mb * bm, kb * bk], x_ub)

                    # TODO: ONE reduction, not the two levels the SIMD tiers need.
                    #       (1) T.Parallel(bm, bk) copies the tile into a fragment
                    #       declared (1, bm * bk) -- a single row, so the whole
                    #       tile is one reduction axis -- and
                    #       T.reduce_absmax(flat_l, tile_l, dim=1) folds it. Do
                    #       not write the GPU shape of a partial per row into
                    #       row_ub and then one thread folding the partials: it is
                    #       longer and 7.24x slower, because that second level
                    #       runs on 1 thread of 128. (2) T.Parallel(1) clamps
                    #       tile_l[0] at E4M3_CLAMP_MIN -- a library reduction
                    #       cannot be seeded -- and writes sf = amax/448 and inv =
                    #       448/amax into one-element shared buffers, since a
                    #       scalar result belongs in shared rather than a
                    #       fragment. (3) T.Parallel(bm, bk): apply, reading the
                    #       single inv_ub[0] that every thread shares. Note there
                    #       is no index arithmetic anywhere in this variant: the
                    #       tile is already the reduction domain.
                    raise NotImplementedError("simt/per_block/01_raw_32x32: implement per_block_cast")

                    T.copy(q_ub, Q[mb * bm, kb * bk])
                    T.copy(sf_ub[0:1], Sf[mb, kb:kb + 1])

    return per_block_cast


def launch(x: torch.Tensor):
    m, hidden = x.shape
    q = torch.empty((m, hidden), dtype=torch.float8_e4m3fn, device=x.device)
    sf = torch.empty((m // BLOCK_MN, hidden // BLOCK_K), dtype=torch.float32,
                     device=x.device)
    compile_kernel(hidden)(x, q, sf)
    status.assert_on_device("per_block 01", q, sf)
    return q, sf

if __name__ == "__main__":        # not a script -- see the module docstring
    raise SystemExit("Run it through the harness:  python -m harness.check simt/per_block/01 --role puzzle")
