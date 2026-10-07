"""per_token 05 (ASC) -- produce column-major scales with an in-register transpose.

New config: `use_tma_aligned_col_major_sf`. This is the hardest variant in the
ladder, and the reason is worth stating up front: **a transpose inside a vector
register is not a memory operation.** You cannot restride a register. The kernel
has to compute, for every lane, which source element that lane should receive, and
then perform a gather.

### What is being transposed, and why

The scales are written as `sf_cm[group, token]` instead of `sf[token, group]`, so
that the GEMM consuming them can fetch one tile's scales contiguously with a bulk
async copy. cast_back/07 showed that *consuming* this layout is free -- a
broadcast load does not care about stride. Producing it is where the work is.

Note that a strided DMA could also do this (write each token's scales down a
column, stride = num_tokens). That is one instruction but a terrible access
pattern: 4-byte elements scattered with a large stride, which is why production
transposes in-register and then writes one contiguous block.

### Lane arithmetic

The kernel accumulates scales for a block of 32 tokens into `sf_dense_ub` with
shape (32, 64) -- token-major, one padded row per token. The output wants
(num_groups, 32) -- group-major. For output flat index `i`:

    token = i & 31              (i % 32, since the output row is 32 tokens wide)
    group = i >> 5              (i / 32)
    source flat index = token * 64 + group

`S.vci` gives each lane its own index, so all of that is computed as a vector,
once, outside the token loop:

    lane   = S.vci(0, T.int32)
    token  = S.vand(reinterpret(lane, "uint32x64"), S.vdup(31, T.uint32))
    group  = reinterpret(S.vshrs(lane, 5), "uint32x64")
    idx    = S.vadd(S.vmuls(token, 64), group)

Then one `S.vgather2(src, idx)` fetches 64 elements in the output's order. A
64-lane gather covers 64 output elements = 2 groups x 32 tokens, so a
4-group scale array needs two gathers.

`S.vci(0, ...)` means "lane index starting at 0" -- the vector equivalent of
`threadIdx.x`, and the only way to get per-lane varying data without reading
memory.

### GPU vs NPU

On a GPU this config is nearly free: you write `out_sf[g, m] = ...` and the
compiler assigns elements to threads however it likes; a transposed store is just
a different index expression, possibly with a shared-memory staging step to keep
the writes coalesced. The thread model lets any element go anywhere. A register
lane cannot move, so on the NPU the same change costs an index computation and a
gather.

Run:  python puzzles/asc/quant/answer/per_token/05_col_major_sf.py
"""

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[5]))

import torch
import tilelang
import tilelang.ascend.language as T
from tilelang.ascend.language import simd as S

from common import oracle, sim, status
from common.check import assert_fp8_near, assert_fp32_ulps
from common.consts import BLOCK_MN, CANONICAL_G, E4M3_CLAMP_MIN, E4M3_MAX
from common.demo import randn_with_zero_row

VARIANT = "asc/per_token/05_col_major_sf"
LANES = 64
PAIR = 128
SF_STRIDE = 64          # padded row length of the token-major scale buffer


@tilelang.jit(target="ascend")
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

            # TODO: build the gather index vector once with S.vci: token = lane &
            #       (token_block-1), group = lane >> log2(token_block), idx =
            #       token*SF_STRIDE + group. Then per token compute scales into
            #       sf_dense_ub[row, :] as usual, and after the block transpose
            #       with S.vgather2(sf_dense_ub[0, base], idx) ->
            #       S.vsts(sf_out_ub[base, 0], ...)
            raise NotImplementedError("asc/per_token/05_col_major_sf: implement per_token_cast")

    return per_token_cast


def launch(x: torch.Tensor):
    m, hidden = x.shape
    num_groups = hidden // CANONICAL_G
    q = torch.empty((m, hidden), dtype=torch.float8_e4m3fn, device=x.device)
    sf_cm = torch.empty((num_groups, m), dtype=torch.float32, device=x.device)
    compile_kernel(hidden)(x, q, sf_cm)
    status.assert_on_device("per_token 05", q, sf_cm)
    return q, sf_cm


def demo_numbers() -> None:
    tb, stride = BLOCK_MN, SF_STRIDE
    print(f"[demo] transposing a ({tb}, groups) scale block to (groups, {tb}):")
    print("[demo] for output flat index i, the source element is:")
    print(f"[demo]   token = i & {tb - 1}      group = i >> {tb.bit_length() - 1}"
          f"      source = token * {stride} + group")
    for i in (0, 1, 31, 32, 33, 63):
        token, group = i & (tb - 1), i >> (tb.bit_length() - 1)
        print(f"[demo]   lane {i:3d} -> out[g={group}, t={token:2d}] "
              f"<- src flat {token * stride + group}")
    print("[demo] S.vci gives each lane its index, so that whole table is one")
    print("[demo] vector computed once; S.vgather2 then fetches 64 elements in")
    print("[demo] the output's order. A register lane cannot move on its own.")


def test_correctness() -> None:
    m, k = sim.sim_shapes()
    assert m % BLOCK_MN == 0, f"the transpose works on blocks of {BLOCK_MN} tokens"
    torch.manual_seed(0)
    x = randn_with_zero_row(m, k, torch.device("cpu"))
    ref_q, ref_sf = oracle.per_token(x, CANONICAL_G, round_sf=True)
    q, sf_cm = launch(x.npu())
    assert sf_cm.shape == (k // CANONICAL_G, m), sf_cm.shape
    assert_fp32_ulps(sf_cm.cpu().T.contiguous(), ref_sf, f"sf_cm.T({m},{k})", max_ulps=0)
    assert_fp8_near(q.cpu(), ref_q, f"q({m},{k})")
    print(f"[check] shape=({m},{k}) sf_cm={tuple(sf_cm.shape)} transposes back "
          f"to {tuple(ref_sf.shape)} bit-exactly")


def main() -> int:
    m, k = sim.sim_shapes()
    if status.unimplemented(VARIANT, lambda: compile_kernel(k)):
        return 0
    sim.maybe_reexec()
    sim.print_banner("asc", "per_token", "05_col_major_sf")

    def body():
        demo_numbers()
        test_correctness()

    return status.run_variant(VARIANT, body)


if __name__ == "__main__":
    sys.exit(main())
