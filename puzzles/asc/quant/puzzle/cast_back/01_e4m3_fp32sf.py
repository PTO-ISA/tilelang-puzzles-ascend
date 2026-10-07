"""cast_back 01 (ASC) -- the Ascend SIMD data path, end to end.

Read puzzles/torch/quant/answer/cast_back/01_e4m3_fp32sf.py first. The maths is
identical; everything here is about how the hardware is addressed.

    out[m, k] = float(q[m, k]) * sf[m, k // 32]

### The four levels of memory

A tilelang Ascend kernel moves data through a fixed hierarchy, and every level is
explicit in the source:

    Global Memory (GM)      the tensors the host sees
        | T.copy                      DMA, runs on the MTE2 engine
    Unified Buffer (UB)     256 KB of on-chip SRAM, T.alloc_shared
        | S.vld                       vector load
    Vector Register (VRF)   256 bytes each, the only thing compute can touch
        | S.vsts                      vector store
    Unified Buffer
        | T.copy                      DMA out, on MTE3
    Global Memory

`T.alloc_shared` is UB here, not CUDA shared memory -- the name is inherited from
the GPU backend and is a little misleading.

### Lanes: why the loop steps by 64

One register is 256 bytes. In float32 that is **64 lanes**, so compute proceeds
64 values at a time and a 128-wide row is two strips. That number drives
everything about the loop structure, and it is the single most important thing to
internalise about this backend.

### Distribution modes

An Ascend load or store is not just an address; it carries a `dist=` describing
how memory maps onto lanes. This kernel uses three:

    UNPK4_B8    load 64 one-byte values and spread them across 64 four-byte
                lanes, ready to be converted. "unpack 4x from bytes."
    BRC_B32     load *one* float32 and broadcast it to all 64 lanes.
    PK_B32      on store, narrow each 32-bit lane to 16 bits and pack them
                together -- the bfloat16 output.

These are hardware capabilities, not DSL conveniences, and they are why the
kernel never needs a separate "broadcast" or "pack" instruction.

### The awkward part: 32-wide groups in a 64-lane register

A scale covers 32 channels, but a register holds 64 float32 lanes -- so one strip
spans **two** groups and needs two different scales. There is no 32-lane vector
type (the legal lane counts are {1, 2, 4, 8, 64, 128, 256}; see
doc/vf-lane-widths-and-limits.md). The ASC answer is a predicate register and a
select:

    mask_low = S.pset(32, "PAT_VL32")   # a mask: lanes 0-31 on, 32-63 off
    lo = S.vld(sf_ub[g],     dist="BRC_B32")
    hi = S.vld(sf_ub[g + 1], dist="BRC_B32")
    scale = S.vsel(lo, hi, mask_low)    # mask ? lo : hi

Two broadcast loads and a select, to build one vector holding two scale values.
Watch what the PTO version of this file does with the same problem -- it is one
load.

### GPU vs NPU

The CUDA version of this kernel (`cast_back_cuda.py` in TileKernels) is 87 lines
and its entire body is:

    for i, j in T.Parallel(TILE_M, TILE_K):
        out_fragment[i, j] = x_shared[i, j] * sf_shared[i // num_per_tokens,
                                                        j // num_per_channels]

One statement over an index space; the compiler assigns elements to threads and
picks the load width. The NPU version is 200+ lines for the same contract.

The difference is the unit of work. `T.Parallel` names **one element** and lets
the compiler build the vectorisation. `T.SimdVF` names **one vector register**,
and you write the strip-mining, the lane arithmetic, the distribution modes and
the UB staging yourself. It is much closer to writing AVX intrinsics than to
writing a CUDA kernel. The payoff is that nothing is hidden; the cost is
everything above.

Specific mappings:

    GPU                                NPU
    T.alloc_fragment (registers)       T.alloc_shared (UB) + S.vld into VRF
    T.alloc_shared   (SMEM)            T.alloc_shared (UB) -- same call, different hw
    T.Parallel(i, j)                   T.serial strip loop + 64-lane ops
    implicit vectorisation             explicit dist= on every load/store
    sf_shared[i // 32] indexing        BRC_B32 load + mask + vsel
    no barrier needed for registers    S.mem_bar when UB is reused

### Simulation

No NPU is present, so main() re-launches this file under
`msprof op simulator`. Expect roughly 15-25 s at M=32, K=128.

Run:  python puzzles/asc/quant/answer/cast_back/01_e4m3_fp32sf.py
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
from common.consts import CANONICAL_G
from common.demo import print_example

VARIANT = "asc/cast_back/01_e4m3_fp32sf"
LANES = 64          # float32 lanes in one 256-byte vector register
SF_PAD = 64         # pad the scale buffer out to a whole register


@tilelang.jit(target="ascend", out_idx=[2])
def compile_kernel(hidden: int, group_size: int = CANONICAL_G):
    """Dequantize FP8 -> bfloat16 with one FP32 scale per `group_size` channels."""
    assert hidden % 128 == 0, "this teaching kernel steps two 64-lane strips at a time"
    assert group_size == 32, "Ascend quant granularity is fixed at 32"
    num_groups = hidden // group_size
    num_tokens = T.dynamic("num_tokens")

    @T.prim_func
    def cast_back(
        Q: T.Tensor((num_tokens, hidden), T.float8_e4m3fn),
        Sf: T.Tensor((num_tokens, num_groups), T.float32),
        Out: T.Tensor((num_tokens, hidden), T.bfloat16),
    ):
        # One vector core. Production runs T.Persistent over many cores; that is
        # scheduling, and it would make the simulator far slower without
        # teaching anything new about the vector unit.
        with T.Kernel(1):
            q_ub = T.alloc_shared((hidden,), T.float8_e4m3fn)
            sf_ub = T.alloc_shared((SF_PAD,), T.float32)
            out_ub = T.alloc_shared((hidden,), T.bfloat16)

            for token in T.serial(num_tokens):
                # GM -> UB, on the DMA engine.
                T.copy(Q[token, 0], q_ub)
                T.copy(Sf[token, 0], sf_ub[0:num_groups])

                # TODO: open `with T.SimdVF():`; make a mask with S.pset(32,
                #       'PAT_VL32'); loop strip over hidden//64; load 64 FP8
                #       values with S.vld(q_ub[col], dist='UNPK4_B8') and S.vcvt
                #       to float32; build the scale with two S.vld(...,
                #       dist='BRC_B32') and S.vsel(lo, hi, mask_low); S.vmul;
                #       store with S.vsts(..., S.vcvt(x, T.bfloat16),
                #       dist='PK_B32')
                raise NotImplementedError("asc/cast_back/01_e4m3_fp32sf: implement cast_back")

                # UB -> GM.
                T.copy(out_ub, Out[token, 0])

    return cast_back


def launch(q: torch.Tensor, sf: torch.Tensor) -> torch.Tensor:
    """Run the kernel. Returns a device tensor -- never a host-computed answer."""
    kernel = compile_kernel(q.shape[1])
    out = kernel(q, sf)
    status.assert_on_device("cast_back 01", out)
    return out


def demo_numbers() -> None:
    """The same worked example as the torch variant, on the host."""
    q = torch.zeros(1, 64)
    q[0, 0:4] = torch.tensor([112.0, 224.0, -448.0, 56.0])
    q = q.to(torch.float8_e4m3fn)
    sf = torch.full((1, 2), 4.0 / 448.0)
    expect = oracle.cast_back(q, sf, (1, CANONICAL_G), out_dtype=torch.bfloat16)
    print("[demo] q[0,0:4] = [112, 224, -448, 56], sf = 4/448")
    print_example("expected", out=expect[:, :4])
    print("[demo] one 64-lane strip covers channels 0-63, i.e. groups 0 and 1,")
    print("[demo] so it needs both scales in one register -- hence the vsel.")


def test_correctness() -> None:
    m, k = sim.sim_shapes()
    torch.manual_seed(0)
    q_host = (torch.randn(m, k) * 100).to(torch.float8_e4m3fn)
    sf_host = (torch.rand(m, k // CANONICAL_G) * 0.01 + 1e-4)

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
    sim.print_banner("asc", "cast_back", "01_e4m3_fp32sf")

    def body():
        demo_numbers()
        test_correctness()

    return status.run_variant(VARIANT, body)


if __name__ == "__main__":
    sys.exit(main())
