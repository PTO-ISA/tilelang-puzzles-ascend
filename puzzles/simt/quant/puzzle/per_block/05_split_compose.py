"""per_block 05 (SIMT). See doc/quant/per_block/05_split_compose.md"""

import torch
import tilelang
import tilelang.ascend.language as T

from harness import status
from harness.consts import BLOCK_K, BLOCK_MN, E4M3_CLAMP_MIN, E4M3_MAX

THREADS = 128


@tilelang.jit(target="ascend")
def compile_kernel(hidden: int, mode: str = "full",
                   block: tuple = (BLOCK_MN, BLOCK_K)):
    """mode in {"full", "sf_only", "cast_only"}.

    As with per_token/06, the modes are plain Python read at trace time, so the
    port is mechanical. The one thing worth noticing is that `cast_only` has to
    rebuild the inverse from the given UE8M0 byte -- `(254 - byte) << 23`
    reinterpreted -- and because that is exact, `cast_only` must come out
    *bit-identical* to the fused path. The harness asserts exactly that, which
    is a stronger check than per_token/06 can make, since its scale is an
    arbitrary float32 and `1/sf` is not exact.
    """
    bm, bk = block
    assert bm == 32 and bk == 32 and hidden % bk == 0
    assert mode in ("full", "sf_only", "cast_only")
    need_amax = mode in ("full", "sf_only")
    need_quant = mode in ("full", "cast_only")
    num_k_blocks = hidden // bk
    num_tokens = T.dynamic("num_tokens")
    num_m_blocks = T.ceildiv(num_tokens, bm)

    @T.prim_func
    def per_block_cast(
        X: T.Tensor((num_tokens, hidden), T.bfloat16),
        SfIn: T.Tensor((num_m_blocks, num_k_blocks), T.uint8),
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
                    if mode == "cast_only":
                        T.copy(SfIn[mb, kb], sf_ub[0:1])

                    # TODO: gate the stages on the mode -- `mode` is a Python
                    #       string read at trace time, so these `if`s choose what
                    #       is emitted. need_amax emits variant 02's two reduction
                    #       levels and the exponent trick, writing the byte to
                    #       sf_ub[0]. cast_only instead reads the byte it was
                    #       handed: biased = T.Cast('int32', sf_ub[0]). Either way
                    #       the inverse is inv_ub[0] = T.reinterpret('float32',
                    #       (254 - biased) << 23) -- exact, because negating a
                    #       power of two's exponent is exact, which is why the
                    #       harness can demand that cast_only be bit-identical to
                    #       the fused path. need_quant emits the apply.
                    raise NotImplementedError("simt/per_block/05_split_compose: implement per_block_cast")

                    if need_quant:
                        T.copy(q_ub, Q[mb * bm, kb * bk])
                    if need_amax:
                        T.copy(sf_ub[0:1], Sf[mb, kb:kb + 1])

    return per_block_cast


def launch(x: torch.Tensor, mode: str = "full", sf_in: torch.Tensor | None = None):
    m, hidden = x.shape
    nm, nk = m // BLOCK_MN, hidden // BLOCK_K
    dev = x.device
    given = sf_in if sf_in is not None else torch.zeros((nm, nk), dtype=torch.uint8,
                                                        device=dev)
    q = torch.empty((m, hidden), dtype=torch.float8_e4m3fn, device=dev)
    sf = torch.empty((nm, nk), dtype=torch.uint8, device=dev)
    compile_kernel(hidden, mode)(x, given, q, sf)
    status.assert_on_device(f"per_block 05 {mode}", q, sf)
    return q, sf

if __name__ == "__main__":        # not a script -- see the module docstring
    raise SystemExit("Run it through the harness:  python -m harness.check simt/per_block/05 --role puzzle")
