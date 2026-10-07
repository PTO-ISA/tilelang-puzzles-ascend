"""per_block 05 (PTO) -- the split, fully composed, in VMI.

Read the ASC variant for the mode flags and for why `cast_only` is bit-identical
to the fused kernel here (a power-of-two scale is inverted by negating its
exponent, which is exact).

### PTO vs ASC

Two small things, both already seen:

**Reading the given scale byte.** ASC's `BRC_B32` broadcasts the 32 bits at an
address, so loading one byte out of a uint8 buffer picks up its three neighbours
and they have to be masked away:

    raw    = T.reinterpret(S.vld(sf_ub[0], dist="BRC_B32"), "uint32x64")
    biased = S.vand(raw, S.vdup(0xFF, T.uint32))

VMI widens on conversion instead, so the element boundary is respected:

    biased = V.vcvt(V.vload(sf_ub[0], size=64), "uint32")

This is the same difference as the scale *store* in variant 02 -- ASC works in
whole machine words and masks, VMI works in elements and converts -- and it is the
one that most often hides a bug, because a forgotten mask still compiles and
usually still produces plausible numbers.

**The exponent arithmetic** reinterprets without a lane count, as in per_token/02,
so the sequence is not pinned to 64 lanes.

### Composed config

    bfloat16 input -> 32x32 blocks -> power-of-two scale -> packed UE8M0
    -> FP8 e4m3 output

Production's weight-quantization configuration. The rest of the distance to
`per_block_cast_asc.py` is scheduling only.

Run:  python puzzles/pto/quant/answer/per_block/05_split_compose.py
"""

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[5]))

import torch
import tilelang
import tilelang.ascend.language as T
from tilelang.ascend.language import vmi as V

from common import oracle, sim, status
from common.check import assert_fp8_near, assert_same_bytes
from common.consts import BLOCK_K, BLOCK_MN, E4M3_CLAMP_MIN, E4M3_MAX
from common.demo import randn_with_zero_row
from common.math_ops import decode_packed_ue8m0

VARIANT = "pto/per_block/05_split_compose"
LANES = 128


@tilelang.jit(target="pto")
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
                    # --- BEGIN SOLUTION hint="gate the passes on the mode. need_amax emits the two-level reduction at 128 lanes and the exponent trick; cast_only instead reads the given byte with V.vcvt(V.vload(sf_ub[0], size=128), 'uint32') -- widening on conversion needs no mask -- and forms the inverse by negating the exponent: (254 - byte) << 23 reinterpreted to float32. need_quant emits the apply pass."
                    with T.SimdVF():
                        mask = V.create_mask(LANES, size=LANES)
                        mask_chunks = V.create_mask(num_chunks, size=LANES)
                        one1 = V.create_mask(1, size=1)
                        # The scale math works on a single lane: there is one
                        # scale per tile. VMI needs every operand at that width,
                        # so the constants are size=1 too.
                        shift = V.vbrc(T.uint32(23), size=1)
                        one_u = V.vbrc(T.uint32(1), size=1)
                        bias254 = V.vbrc(T.uint32(254), size=1)

                        if need_amax:
                            for chunk in T.serial(num_chunks):
                                v = V.vabs(V.vcvt(V.vload(flat_x[chunk * LANES],
                                                          size=LANES), "float32"), mask)
                                V.vstore(V.vcmax(v, mask, group=1), run_ub[chunk])
                            T.simd.mem_bar("VST_VLD")

                            partials = V.vload(run_ub[0], size=LANES)
                            tile_amax = V.vcmax(partials, mask_chunks, group=1)
                            clamped = V.vmax(
                                tile_amax,
                                V.vbrc(T.float32(E4M3_CLAMP_MIN), size=1), one1)
                            bits = V.vinterpret_cast(
                                V.vmul(clamped,
                                       V.vbrc(T.float32(1.0 / E4M3_MAX), size=1),
                                       one1), "uint32")
                            biased = V.vadd(V.vshr(V.vsub(bits, one_u), shift), one_u)
                            V.vstore(V.vcvt(biased, "uint8"), sf_ub[0])
                        else:
                            # cast_only: widening on conversion respects the
                            # element boundary, so no mask is needed.
                            biased = V.vcvt(V.vload(sf_ub[0], size=1), "uint32")

                        if need_quant:
                            V.vstore(V.vinterpret_cast(
                                V.vshl(V.vsub(bias254, biased), shift), "float32"),
                                inv_ub[0])
                            T.simd.mem_bar("VST_VLD")
                            inv = V.vbrc(V.vload(inv_ub[0], size=1), size=LANES)
                            for chunk in T.serial(num_chunks):
                                v = V.vcvt(V.vload(flat_x[chunk * LANES], size=LANES),
                                           "float32")
                                V.vstore(V.vcvt(V.vmul(v, inv, mask), "float8_e4m3fn",
                                                rounding="R", saturate="SAT"),
                                         flat_q[chunk * LANES])
                    # --- END SOLUTION
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
    print("[demo] reading the byte: ASC broadcasts 32 bits and masks 0xFF; VMI")
    print("[demo] widens on conversion, so the element boundary is respected.")


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
    sim.print_banner("pto", "per_block", "05_split_compose")

    def body():
        demo_numbers()
        test_correctness()

    return status.run_variant(VARIANT, body)


if __name__ == "__main__":
    sys.exit(main())
