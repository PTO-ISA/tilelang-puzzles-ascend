"""cast_back 01 (PTO) -- the same kernel in logical VMI instead of Ascend SIMD.

Read the torch variant first for the maths, then the ASC variant for the hardware
model. This file changes neither: same schedule, same memory hierarchy, same
64-lane strips. Only the vector instruction set is different.

    out[m, k] = float(q[m, k]) * sf[m, k // 32]

### What PTO / VMI is

PTO is a second vector IR for the same chip. Where Ascend SIMD (`T.simd`) names
*physical* operations -- a named distribution pattern, a lane count baked into a
dtype string -- VMI (`T.vmi`) names the *intent* and lets the assembler choose the
encoding. Both compile to the same hardware; both live inside `with T.SimdVF()`.

In tilelang they differ only by the namespace and the jit target:

    ASC:  from tilelang.ascend.language import simd as S   @tilelang.jit(target="ascend")
    PTO:  from tilelang.ascend.language import vmi  as V   @tilelang.jit(target="pto")

### PTO vs ASC

**1. The scale broadcast: three operations become one.**

A 64-lane float32 strip spans two 32-channel groups, so it needs two scale
values in one register. ASC has to broadcast each one separately and stitch them
with a predicate:

    mask_low = S.pset(32, "PAT_VL32")
    lo = S.vld(sf_ub[group],     dist="BRC_B32")
    hi = S.vld(sf_ub[group + 1], dist="BRC_B32")
    scale = S.vsel(lo, hi, mask_low)

VMI says what is actually wanted -- "fill 64 lanes from consecutive scalars, in 2
segments" -- and the assembler emits whatever that takes:

    scale = V.vload(sf_ub[group], size=64, dist_mode="brc", group=2, stride=1)

One operation, no mask, and nothing in it is specific to the number 32. That
`group=` parameter is the single most useful idea in VMI: it makes *segmented*
behaviour a first-class argument on loads, stores, reduces and broadcasts.

**2. Conversions read like conversions.**

ASC spells a widening load as a distribution mode (`dist="UNPK4_B8"`, "unpack 4x
from bytes") and a narrowing store as another (`dist="PK_B32"`). You have to know
the catalogue to recognise that those two mean "widen" and "narrow".

    raw = S.vld(q_ub[col], dist="UNPK4_B8");  values = S.vcvt(raw, T.float32)
    S.vsts(out_ub[col], S.vcvt(scaled, T.bfloat16), dist="PK_B32")

VMI loads a vector and converts it, and the packing is implied by the buffer's
dtype:

    values = V.vcvt(V.vload(q_ub[col], size=64), "float32")
    V.vstore(V.vcvt(scaled, "bfloat16"), out_ub[col])

This gap gets much wider in later variants. In production's per_token, the ASC
FP8 store is four `S.vcvt(..., part=0..3)` calls OR-ed together, because one
convert only fills every fourth byte; in PTO it is a single `V.vcvt`.

**3. Lane count is a value, not part of a type.**

ASC encodes width in dtype strings -- `'float32x64'`, `'uint16x128'` -- so a
64-lane and a 128-lane version of the same arithmetic are *different code*. VMI
passes `size=` as an ordinary argument, so one helper serves 4, 64, 128 and 256
lanes. The later variants in this ladder reuse a single `compute_scale` helper at
several widths for exactly this reason; the ASC files cannot.

**4. Masks are counts, not pattern names.**

`S.pset(32, "PAT_VL32")` names a hardware predicate pattern. `V.create_mask(32,
size=64)` says "32 of 64 lanes active". The second composes with arbitrary
widths; the first is drawn from a fixed catalogue.

**Where PTO is not simpler.** It is not torch. You still open `T.SimdVF()`, still
place memory barriers by hand, still live inside explicit UB buffers, and `size=`
is mandatory on nearly every call -- so the token count per line sometimes goes
*up* even as the operation count goes down. The honest summary is "torch-like
vector operations inside an NPU schedule".

**Evidence at production scale.** Porting all four quant kernels from ASC to
VMI (TileKernels commit 5395526, "Port four quant kernels to PTO/VMI") changed
only the `T.SimdVF` bodies -- every `@T.prim_func` schedule and host contract
stayed byte-identical -- and came to 312 insertions against 395 deletions:
**83 fewer lines**, concentrated exactly in the mask/select/part= machinery
described above.

### GPU vs NPU

Unchanged from the ASC variant: `T.Parallel` names one element of an index space
and the compiler vectorises; `T.SimdVF` names one vector register and you do.
VMI narrows the gap in *expressiveness* -- `group=` is close to a reshape-and-
reduce -- but not in *structure*. The strip loop, the UB staging and the DMA are
all still explicit. GPU-level brevity would need tilelang to emit VMI from
`T.Parallel`, which it does not yet do.

### Simulation

Expect roughly 15-25 s at M=32, K=128 under `msprof op simulator`.

Run:  python puzzles/pto/quant/answer/cast_back/01_e4m3_fp32sf.py
"""

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[5]))

import torch
import tilelang
import tilelang.ascend.language as T
from tilelang.ascend.language import vmi as V

from common import oracle, sim, status
from common.check import assert_bf16_near
from common.consts import CANONICAL_G
from common.demo import print_example

VARIANT = "pto/cast_back/01_e4m3_fp32sf"
LANES = 64
SF_PAD = 64


@tilelang.jit(target="pto", out_idx=[2])
def compile_kernel(hidden: int, group_size: int = CANONICAL_G):
    """Dequantize FP8 -> bfloat16 with one FP32 scale per `group_size` channels."""
    assert hidden % 128 == 0, "this teaching kernel steps two 64-lane strips at a time"
    assert group_size == 32, "Ascend quant granularity is fixed at 32"
    num_groups = hidden // group_size
    groups_per_strip = LANES // group_size      # 2
    num_tokens = T.dynamic("num_tokens")

    @T.prim_func
    def cast_back(
        Q: T.Tensor((num_tokens, hidden), T.float8_e4m3fn),
        Sf: T.Tensor((num_tokens, num_groups), T.float32),
        Out: T.Tensor((num_tokens, hidden), T.bfloat16),
    ):
        with T.Kernel(1):
            q_ub = T.alloc_shared((hidden,), T.float8_e4m3fn)
            sf_ub = T.alloc_shared((SF_PAD,), T.float32)
            out_ub = T.alloc_shared((hidden,), T.bfloat16)

            for token in T.serial(num_tokens):
                T.copy(Q[token, 0], q_ub)
                T.copy(Sf[token, 0], sf_ub[0:num_groups])

                # --- BEGIN SOLUTION hint="open `with T.SimdVF():`; mask = V.create_mask(64, size=64); loop strip over hidden//64; values = V.vcvt(V.vload(q_ub[col], size=64), 'float32'); scale = V.vload(sf_ub[group], size=64, stride=1, dist_mode='brc', group=2) -- one load, no select; then V.vmul and V.vstore(V.vcvt(scaled, 'bfloat16'), out_ub[col])"
                with T.SimdVF():
                    mask = V.create_mask(LANES, size=LANES)
                    for strip in T.serial(hidden // LANES):
                        col = strip * LANES
                        group = strip * groups_per_strip

                        # Load 64 FP8 values and widen. No distribution-mode
                        # catalogue to consult: the convert says what it does.
                        values = V.vcvt(V.vload(q_ub[col], size=LANES), "float32")

                        # One load builds the whole scale vector: 64 lanes, two
                        # segments, each broadcast from a consecutive scalar.
                        # The ASC file needs two loads and a select for this.
                        scale = V.vload(sf_ub[group], size=LANES, stride=1,
                                        dist_mode="brc", group=groups_per_strip)

                        scaled = V.vmul(values, scale, mask)
                        V.vstore(V.vcvt(scaled, "bfloat16"), out_ub[col])
                # --- END SOLUTION

                T.copy(out_ub, Out[token, 0])

    return cast_back


def launch(q: torch.Tensor, sf: torch.Tensor) -> torch.Tensor:
    """Run the kernel. Returns a device tensor -- never a host-computed answer."""
    kernel = compile_kernel(q.shape[1])
    out = kernel(q, sf)
    status.assert_on_device("cast_back 01", out)
    return out


def demo_numbers() -> None:
    q = torch.zeros(1, 64)
    q[0, 0:4] = torch.tensor([112.0, 224.0, -448.0, 56.0])
    q = q.to(torch.float8_e4m3fn)
    sf = torch.full((1, 2), 4.0 / 448.0)
    expect = oracle.cast_back(q, sf, (1, CANONICAL_G), out_dtype=torch.bfloat16)
    print("[demo] q[0,0:4] = [112, 224, -448, 56], sf = 4/448")
    print_example("expected", out=expect[:, :4])
    print("[demo] the 64-lane strip spans groups 0 and 1, and one brc load with")
    print("[demo] group=2 fills it -- compare the ASC file's vsel.")


def test_correctness() -> None:
    m, k = sim.sim_shapes()
    torch.manual_seed(0)
    q_host = (torch.randn(m, k) * 100).to(torch.float8_e4m3fn)
    sf_host = torch.rand(m, k // CANONICAL_G) * 0.01 + 1e-4

    ref = oracle.cast_back(q_host, sf_host, (1, CANONICAL_G), out_dtype=torch.bfloat16)
    got = launch(q_host.npu(), sf_host.npu()).cpu()
    assert_bf16_near(got, ref, f"cast_back({m},{k})", atol=0.0)
    print(f"[check] shape=({m},{k}) matches the torch oracle exactly")


def main() -> int:
    # Fail fast on an unwritten kernel: tracing happens on the host, so there is
    # no need to pay for a simulator launch to discover the body is missing.
    m, k = sim.sim_shapes()
    if status.unimplemented(VARIANT, lambda: compile_kernel(k)):
        return 0
    sim.maybe_reexec()
    sim.print_banner("pto", "cast_back", "01_e4m3_fp32sf")

    def body():
        demo_numbers()
        test_correctness()

    return status.run_variant(VARIANT, body)


if __name__ == "__main__":
    sys.exit(main())
