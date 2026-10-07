"""cast_back 04 (ASC) -- one scale per 32x32 tile.

The scale index gains a row term: `sf[m // 32, k // 32]` instead of
`sf[m, k // 32]`. Numerically that is all (see the torch variant). But it changes
the *schedule*, and that is the point of this variant.

### A coarser scale axis means less DMA

With per-token scales, every token needed its own scale row, so the scale copy sat
inside the token loop. Now 32 consecutive tokens share one scale row, so the copy
hoists out:

    for token_block in ...:
        T.copy(Sf[token_block, 0], sf_ub)      <- once per 32 tokens
        for row in T.serial(32):
            ... reuse sf_ub ...

That is 32x fewer scale DMAs. Production expresses the same thing by giving the
scale buffer a different multi-buffer depth from the value buffer
(`sf_stages = num_stages if num_per_tokens == 1 else 1` in `cast_back_asc.py`):
when the scale changes once per tile there is no point double-buffering it.

Restructuring the loop nest to match the scale granularity is the kind of change
that does not exist at all in the torch tier, where the scale is just an index
expression. It is most of what writing these kernels consists of.

Run:  python puzzles/asc/quant/answer/cast_back/04_block_sf.py
"""

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[5]))

import torch
import tilelang
import tilelang.ascend.language as T
from tilelang.ascend.language import simd as S

from common import oracle, sim, status
from common.check import assert_bf16_near
from common.consts import BLOCK_K, BLOCK_MN

VARIANT = "asc/cast_back/04_block_sf"
LANES = 64


@tilelang.jit(target="ascend", out_idx=[2])
def compile_kernel(hidden: int, block: tuple = (BLOCK_MN, BLOCK_K)):
    """Dequantize FP8 -> bfloat16 with one scale per `block`-shaped tile."""
    bm, bk = block
    assert hidden % 128 == 0 and bm == 32 and bk == 32
    num_k_blocks = hidden // bk
    num_tokens = T.dynamic("num_tokens")
    num_m_blocks = T.ceildiv(num_tokens, bm)
    sf_pad = max(num_k_blocks, 16)

    @T.prim_func
    def cast_back(
        Q: T.Tensor((num_tokens, hidden), T.float8_e4m3fn),
        Sf: T.Tensor((num_m_blocks, num_k_blocks), T.float32),
        Out: T.Tensor((num_tokens, hidden), T.bfloat16),
    ):
        with T.Kernel(1):
            q_ub = T.alloc_shared((hidden,), T.float8_e4m3fn)
            sf_ub = T.alloc_shared((sf_pad,), T.float32)
            out_ub = T.alloc_shared((hidden,), T.bfloat16)
            for m_block in T.serial(num_m_blocks):
                # Hoisted: one scale row serves all 32 tokens below.
                T.copy(Sf[m_block, 0], sf_ub[0:num_k_blocks])
                for row in T.serial(bm):
                    token = m_block * bm + row
                    T.copy(Q[token, 0], q_ub)
                    # TODO: identical VF body to variant 01 -- the only change is
                    #       that sf_ub was filled once per 32 tokens instead of
                    #       once per token
                    raise NotImplementedError("asc/cast_back/04_block_sf: implement cast_back")
                    T.copy(out_ub, Out[token, 0])

    return cast_back


def launch(q: torch.Tensor, sf: torch.Tensor) -> torch.Tensor:
    out = compile_kernel(q.shape[1])(q, sf)
    status.assert_on_device("cast_back 04", out)
    return out


def demo_numbers() -> None:
    print("[demo] scale DMAs for M=32, K=128:")
    print("[demo]   per-token scales  (variant 01): 32 copies, one per token")
    print("[demo]   per-block scales  (this one)  :  1 copy, reused by 32 tokens")
    print("[demo] the loop nest follows the scale granularity, not the data")


def test_correctness() -> None:
    m, k = sim.sim_shapes()
    assert m % BLOCK_MN == 0, f"per_block needs M to be a multiple of {BLOCK_MN}"
    torch.manual_seed(0)
    q = (torch.randn(m, k) * 100).to(torch.float8_e4m3fn)
    sf = torch.rand(m // BLOCK_MN, k // BLOCK_K) * 0.01 + 1e-4
    ref = oracle.cast_back(q, sf, (BLOCK_MN, BLOCK_K), out_dtype=torch.bfloat16)
    got = launch(q.npu(), sf.npu()).cpu()
    assert_bf16_near(got, ref, f"cast_back_block({m},{k})", atol=0.0)
    print(f"[check] shape=({m},{k}) sf={tuple(sf.shape)} matches the oracle exactly")


def main() -> int:
    m, k = sim.sim_shapes()
    if status.unimplemented(VARIANT, lambda: compile_kernel(k)):
        return 0
    sim.maybe_reexec()
    sim.print_banner("asc", "cast_back", "04_block_sf")

    def body():
        demo_numbers()
        test_correctness()

    return status.run_variant(VARIANT, body)


if __name__ == "__main__":
    sys.exit(main())
