"""per_block 05 (ASC) -- the sf_only / cast_only split, fully composed.

Last per_block variant. Same trace-time mode flags as per_token/06: `mode` picks
which passes are emitted, so one source serves the full kernel, the scales-only
kernel and the apply-only kernel.

    mode="full"       tile amax -> power-of-two scale -> quantize
    mode="sf_only"    tile amax -> scale, no quantized output
    mode="cast_only"  the scale byte is given; apply it, no reduction at all

### Composed config

    bfloat16 input -> 32x32 blocks -> power-of-two scale -> packed UE8M0
    -> FP8 e4m3 output

That is production's weight-quantization configuration. What remains between this
file and `per_block_cast_asc.py` is scheduling: multiple vector cores, a
persistent loop over tiles, double-buffered UB, and the `token_group = 4` widening
discussed in variant 04. None of it changes a number computed here.

### cast_only is exact here, unlike per_token/06

per_token/06 had to explain that `cast_only` can differ from the fused path by an
FP8 code, because it only has the rounded scale and must compute `1/sf`.

Here the scale is always a power of two, so the reciprocal is formed by *negating
the exponent field* -- `(254 - biased) << 23` -- which is exact. No division, no
rounding, so `cast_only` is bit-identical to the fused kernel. The test asserts
byte equality rather than closeness. That is a concrete payoff of `round_sf`
beyond the memory saving.

Run:  python puzzles/asc/quant/answer/per_block/05_split_compose.py
"""

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[5]))

import torch
import tilelang
import tilelang.ascend.language as T
from tilelang.ascend.language import simd as S

from common import oracle, sim, status
from common.check import assert_fp8_near, assert_same_bytes
from common.consts import BLOCK_K, BLOCK_MN, E4M3_CLAMP_MIN, E4M3_MAX
from common.demo import randn_with_zero_row
from common.math_ops import decode_packed_ue8m0

VARIANT = "asc/per_block/05_split_compose"
LANES = 64


@tilelang.jit(target="ascend")
def compile_kernel(hidden: int, mode: str = "full",
                   block: tuple = (BLOCK_MN, BLOCK_K)):
    """mode in {"full", "sf_only", "cast_only"}."""
    bm, bk = block
    assert bm == 32 and bk == 32 and hidden % bk == 0
    assert mode in ("full", "sf_only", "cast_only")
    need_amax = mode in ("full", "sf_only")
    need_quant = mode in ("full", "cast_only")
    tile_values = bm * bk
    num_chunks = tile_values // LANES
    num_k_blocks = hidden // bk
    num_tokens = T.dynamic("num_tokens")
    num_m_blocks = T.ceildiv(num_tokens, bm)
    run_pad = max(num_chunks, LANES)

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
            run_ub = T.alloc_shared((run_pad,), T.float32)
            sf_ub = T.alloc_shared((LANES,), T.uint8)
            inv_ub = T.alloc_shared((LANES,), T.float32)
            flat_x = T.Tensor((tile_values,), T.bfloat16, x_ub.data)
            flat_q = T.Tensor((tile_values,), T.float8_e4m3fn, q_ub.data)

            for mb in T.serial(num_m_blocks):
                for kb in T.serial(num_k_blocks):
                    T.copy(X[mb * bm, kb * bk], x_ub)
                    if mode == "cast_only":
                        T.copy(SfIn[mb, kb], sf_ub[0:1])
                    # TODO: gate the passes on the mode. need_amax emits the
                    #       two-level reduction and the exponent trick; cast_only
                    #       instead loads the given byte with S.vld(sf_ub[0],
                    #       dist='BRC_B32'), converts it to uint32, and forms the
                    #       inverse by negating the exponent: (254 - byte) << 23
                    #       reinterpreted to float32 -- exact, no division.
                    #       need_quant emits the apply pass.
                    raise NotImplementedError("asc/per_block/05_split_compose: implement per_block_cast")
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


def demo_numbers() -> None:
    print("[demo] modes, decided at trace time:")
    print("[demo]   full      reduce -> scale -> quantize")
    print("[demo]   sf_only   reduce -> scale")
    print("[demo]   cast_only given scale -> quantize   (no reduction at all)")
    print("[demo] inverting a power-of-two scale is exact: negate the exponent.")
    for e in (127, 120, 136):
        inv_bits = (254 - e) << 23
        inv = torch.tensor([inv_bits], dtype=torch.int32).view(torch.float32).item()
        assert abs(inv - 2.0 ** -(e - 127)) < 1e-30
        print(f"[demo]   byte {e} = 2^{e - 127:+d}  ->  (254-{e})<<23 = {inv:g}")
    print("[demo] so cast_only is bit-identical to the fused kernel here, unlike")
    print("[demo] per_token/06 where the scale was an arbitrary float32.")


def test_correctness() -> None:
    m, k = sim.sim_shapes()
    torch.manual_seed(0)
    x = randn_with_zero_row(m, k, torch.device("cpu"))
    ref_q, ref_packed = oracle.per_block(x, (BLOCK_MN, BLOCK_K),
                                        round_sf=True, packed=True)

    _, sf = launch(x.npu(), "sf_only")
    assert_same_bytes(sf.cpu().view(torch.int16), ref_packed, "sf_only")
    print("[check] sf_only produces the oracle's packed scales byte-exactly")

    q_full, _ = launch(x.npu(), "full")
    assert_fp8_near(q_full.cpu(), ref_q, "full q")
    print("[check] full matches the oracle")

    q_co, _ = launch(x.npu(), "cast_only", sf_in=sf)
    assert torch.equal(q_co.cpu().view(torch.uint8), q_full.cpu().view(torch.uint8)), (
        "with a power-of-two scale, cast_only must be bit-identical to fused"
    )
    print("[check] cast_only is bit-identical to the fused kernel "
          "(exact reciprocal of a power of two)")
    _, ref_f32 = oracle.per_block(x, (BLOCK_MN, BLOCK_K), round_sf=True)
    assert torch.equal(decode_packed_ue8m0(sf.cpu().view(torch.int16)), ref_f32)
    print(f"[check] shape=({m},{k}) scales decode to the float32 scales")


def main() -> int:
    m, k = sim.sim_shapes()
    if status.unimplemented(VARIANT, lambda: compile_kernel(k, "full")):
        return 0
    sim.maybe_reexec()
    sim.print_banner("asc", "per_block", "05_split_compose")

    def body():
        demo_numbers()
        test_correctness()

    return status.run_variant(VARIANT, body)


if __name__ == "__main__":
    sys.exit(main())
