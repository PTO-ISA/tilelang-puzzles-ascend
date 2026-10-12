"""per_block 02 (SIMT). See doc/quant/per_block/02_round_packed.md"""

import torch
import tilelang
import tilelang.ascend.language as T

from harness import status
from harness.consts import BLOCK_K, BLOCK_MN, E4M3_CLAMP_MIN, E4M3_MAX

THREADS = 128


@tilelang.jit(target="ascend")
def compile_kernel(hidden: int, block: tuple = (BLOCK_MN, BLOCK_K)):
    """One power-of-two scale per tile, returned as a single UE8M0 byte.

    This is the variant where the SIMD tiers pay most for working on vectors:
    there is exactly one scale per tile, so ASC and PTO do the exponent
    arithmetic on a one-lane vector and still have to spell every constant as a
    `size=1` broadcast. Here it is scalar arithmetic on a scalar, which is what
    it always was.
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
        Sf: T.Tensor((num_m_blocks, num_k_blocks), T.uint8),
    ):
        with T.Kernel(1):
            x_ub = T.alloc_shared((bm, bk), T.bfloat16)
            q_ub = T.alloc_shared((bm, bk), T.float8_e4m3fn)
            row_ub = T.alloc_shared((bm,), T.float32)
            sf_ub = T.alloc_shared((1,), T.uint8)
            inv_ub = T.alloc_shared((1,), T.float32)

            for mb in T.serial(num_m_blocks):
                for kb in T.serial(num_k_blocks):
                    T.copy(X[mb * bm, kb * bk], x_ub)

                    # TODO: variant 01's three levels, with the division in level
                    #       2 replaced by the exponent trick and the result stored
                    #       as a byte: bits = T.reinterpret('int32', tile /
                    #       E4M3_MAX); biased = ((bits - 1) >> 23) + 1; sf_ub[0] =
                    #       T.Cast('uint8', biased) and inv_ub[0] =
                    #       T.reinterpret('float32', (254 - biased) << 23).
                    #       Compare the SIMD tiers, which must do this on a
                    #       one-lane vector -- every constant becomes a
                    #       V.vbrc(..., size=1) and the byte store needs a convert
                    #       (PTO) or a reinterpret plus a PK4_B32 store mode
                    #       (ASC). A scalar scale is a scalar here.
                    raise NotImplementedError("simt/per_block/02_round_packed: implement per_block_cast")

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

if __name__ == "__main__":        # not a script -- see the module docstring
    raise SystemExit("Run it through the harness:  python -m harness.check simt/per_block/02 --role puzzle")
