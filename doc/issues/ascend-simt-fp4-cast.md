# [Ascend] `float4_e2m1fn` output cannot be expressed inside a `T.SimtVF` region

*A bug report for the tilelang Ascend backend. Everything below is reproducible
with the single script in "Reproduction". No NPU and no simulator are needed —
the three failing cases fail during lowering, before codegen — but a tilelang
build with the Ascend target and a CANN install are, since the three passing
cases do reach the Ascend compiler.*

## Summary

On the CUDA backend an FP4 (`float4_e2m1fn`) result is written the obvious way —
a per-element `T.cast` into an fp4 fragment inside `T.Parallel`, then `T.copy`
out — and that is how [deepseek-ai/TileKernels][tk-per-token] writes its
quantization kernels.

The same spelling cannot be expressed inside a `T.SimtVF` region on the Ascend
backend. Every variant fails during lowering, before codegen, with one of three
internal errors. The underlying reason is that **the Ascend path has no scalar
FP4 representation at all**: `float4_e2m1fn` is only ever materialised as the
packed pair `float4_e2m1fnx2`, and per-thread SIMT code has a single lane to
offer.

The same kernel with an **FP8** output compiles, and the FP4 convert itself
compiles inside a `T.SimdVF` region, so this is specific to the 4-bit type on
the thread-model surface.

This is not only a frontend-ergonomics problem. Because the convert is out of
reach, an FP4 quantization kernel written for `T.SimtVF` has to hand-write an
e2m1 encoder in integer arithmetic; in a 22-kernel quantization suite measured on
this toolchain that kernel costs **562,280** simulated VF cycles against
**7,778** for the same algorithm that enters a `T.SimdVF` region for the one
convert instruction — **72x**, with byte-identical output.

## Environment

| | |
|---|---|
| tilelang | 0.1.15, Ascend target |
| CANN | 9.2.0-beta.2 |
| SoC | `Ascend950PR_9599` |
| Python | 3.12.14 |
| torch / torch_npu | 2.9.0 / 2.9.0.post6 |

## Reproduction

Save as `repro_fp4_simt.py` and run `python repro_fp4_simt.py`. It is
compile-only: six kernels are built, three of which are expected to fail.

```python
"""tilelang/Ascend: no FP4 value can be produced inside a `T.SimtVF` region.

Compile-only; no device and no simulator are needed.

    python repro_fp4_simt.py
"""
import tilelang
import tilelang.ascend.language as T
from tilelang.ascend.language import simd as S

K = 128              # elements per row
THREADS = 128        # SIMT threads per region
LANES = 64           # float32 lanes in one Ascend vector register
PAIR = 2 * LANES     # bfloat16 lanes consumed by one packed-FP4 convert
FP4 = "float4_e2m1fn"


# --------------------------------------------------------------------------
# 1. scalar cast into an fp4 shared buffer -- the CUDA spelling
# --------------------------------------------------------------------------
def simt_scalar_cast():
    @T.prim_func
    def kern(X: T.Tensor((K,), T.float32), Q: T.Tensor((K,), T.float4_e2m1fn)):
        with T.Kernel(1):
            x_ub = T.alloc_shared((K,), T.float32)
            q_ub = T.alloc_shared((K,), T.float4_e2m1fn)
            T.copy(X, x_ub)
            with T.SimtVF(threads=THREADS):
                for i in T.Parallel(K):
                    q_ub[i] = T.Cast(FP4, x_ub[i])
            T.copy(q_ub, Q)
    return kern


# --------------------------------------------------------------------------
# 2. the same, into an fp4 FRAGMENT, then T.copy out -- TileKernels' shape
# --------------------------------------------------------------------------
def simt_fragment_cast():
    @T.prim_func
    def kern(X: T.Tensor((K,), T.float32), Q: T.Tensor((K,), T.float4_e2m1fn)):
        with T.Kernel(1):
            x_ub = T.alloc_shared((K,), T.float32)
            q_ub = T.alloc_shared((K,), T.float4_e2m1fn)
            T.copy(X, x_ub)
            with T.SimtVF(threads=THREADS):
                q_l = T.alloc_fragment((K,), T.float4_e2m1fn)
                for i in T.Parallel(K):
                    q_l[i] = T.Cast(FP4, x_ub[i])
                T.copy(q_l, q_ub)
            T.copy(q_ub, Q)
    return kern


# --------------------------------------------------------------------------
# 3. two codes per thread, so the per-thread tile is even
# --------------------------------------------------------------------------
def simt_two_codes_per_thread():
    @T.prim_func
    def kern(X: T.Tensor((K,), T.float32), Q: T.Tensor((K,), T.float4_e2m1fn)):
        with T.Kernel(1):
            x_ub = T.alloc_shared((K,), T.float32)
            q_ub = T.alloc_shared((K,), T.float4_e2m1fn)
            T.copy(X, x_ub)
            with T.SimtVF(threads=K // 2):          # 2 fp4 codes per thread
                q_l = T.alloc_fragment((K,), T.float4_e2m1fn)
                for i in T.Parallel(K):
                    q_l[i] = T.Cast(FP4, x_ub[i])
                T.copy(q_l, q_ub)
            T.copy(q_ub, Q)
    return kern


# --------------------------------------------------------------------------
# 4. CONTROL: the identical kernel with an FP8 output lowers
# --------------------------------------------------------------------------
def simt_scalar_cast_fp8():
    @T.prim_func
    def kern(X: T.Tensor((K,), T.float32), Q: T.Tensor((K,), T.float8_e4m3fn)):
        with T.Kernel(1):
            x_ub = T.alloc_shared((K,), T.float32)
            q_ub = T.alloc_shared((K,), T.float8_e4m3fn)
            T.copy(X, x_ub)
            with T.SimtVF(threads=THREADS):
                for i in T.Parallel(K):
                    q_ub[i] = T.Cast("float8_e4m3fn", x_ub[i])
            T.copy(q_ub, Q)
    return kern


# --------------------------------------------------------------------------
# 5. CONTROL: the FP4 convert in a SimdVF region lowers
# --------------------------------------------------------------------------
def simd_convert():
    @T.prim_func
    def kern(X: T.Tensor((K,), T.bfloat16), Q: T.Tensor((K,), T.float4_e2m1fn)):
        with T.Kernel(1):
            x_ub = T.alloc_shared((K,), T.bfloat16)
            q_ub = T.alloc_shared((K,), T.float4_e2m1fn)
            T.copy(X, x_ub)
            with T.SimdVF():
                for pair in T.serial(K // PAIR):
                    S.vsts(q_ub[pair * PAIR],
                           S.vcvt(S.vld(x_ub[pair * PAIR]), T.float4_e2m1fn),
                           dist="PK4_B32")
            T.copy(q_ub, Q)
    return kern


# --------------------------------------------------------------------------
# 6. WORKAROUND: SimtVF for the arithmetic, SimdVF for the convert
# --------------------------------------------------------------------------
def hybrid():
    @T.prim_func
    def kern(X: T.Tensor((K,), T.float32), Q: T.Tensor((K,), T.float4_e2m1fn)):
        with T.Kernel(1):
            x_ub = T.alloc_shared((K,), T.float32)
            bf_ub = T.alloc_shared((K,), T.bfloat16)
            q_ub = T.alloc_shared((K,), T.float4_e2m1fn)
            T.copy(X, x_ub)
            with T.SimtVF(threads=THREADS):
                for i in T.Parallel(K):             # round to odd, per thread
                    bits = T.reinterpret("uint32", x_ub[i])
                    sticky = T.min(bits & 0xFFFF, T.uint32(1))
                    bf_ub[i] = T.reinterpret(
                        "bfloat16", T.Cast("uint16", (bits >> 16) | sticky))
            with T.SimdVF():                        # the one vector instruction
                for pair in T.serial(K // PAIR):
                    S.vsts(q_ub[pair * PAIR],
                           S.vcvt(S.vld(bf_ub[pair * PAIR]), T.float4_e2m1fn),
                           dist="PK4_B32")
            T.copy(q_ub, Q)
    return kern


CASES = [
    ("1  SimtVF, scalar cast into fp4 shared", simt_scalar_cast),
    ("2  SimtVF, scalar cast into an fp4 fragment", simt_fragment_cast),
    ("3  SimtVF, 2 fp4 codes per thread", simt_two_codes_per_thread),
    ("4  SimtVF, same kernel with an FP8 output", simt_scalar_cast_fp8),
    ("5  SimdVF, the FP4 convert", simd_convert),
    ("6  SimtVF + SimdVF in one kernel", hybrid),
]

print(f"tilelang {tilelang.__version__}\n")
for name, build in CASES:
    try:
        tilelang.compile(build(), target="ascend")
        print(f"{name:46} OK")
    except Exception as exc:                                     # noqa: BLE001
        first = str(exc).strip().splitlines()[0] if str(exc).strip() else type(exc).__name__
        print(f"{name:46} FAIL\n     {first}")
```

### Output

```text
tilelang 0.1.15
1  SimtVF, scalar cast into fp4 shared         FAIL
     Check failed: index_lanes * buffer_lanes == value_dtype_lanes (2 vs. 1) : Cannot store value with 1, expected value with 2 (1 index lanes * 2 buffer element lanes)
2  SimtVF, scalar cast into an fp4 fragment    FAIL
     Check failed: imm->value % 2 == 0 (1 vs. 0) : RewriteFp4ToFp4x2: fp4 buffer last-dim shape must be even (fp4 is packed 2 codes/byte), got 1
3  SimtVF, 2 fp4 codes per thread              FAIL
     Check failed: (lanes_match) is false: Cannot match type float4_e2m1fnx2 vs float4_e2m1fnx4
4  SimtVF, same kernel with an FP8 output      OK
5  SimdVF, the FP4 convert                     OK
6  SimtVF + SimdVF in one kernel               OK
```

## What each case shows

| # | spelling | result |
|---|---|---|
| 1 | scalar `T.Cast` into an fp4 **shared** buffer, inside `T.SimtVF` | `Cannot store value with 1, expected value with 2 (1 index lanes * 2 buffer element lanes)` |
| 2 | the same into an fp4 **fragment**, then `T.copy` | `RewriteFp4ToFp4x2: fp4 buffer last-dim shape must be even (fp4 is packed 2 codes/byte), got 1` |
| 3 | as (2) but with **two fp4 codes per thread**, so the per-thread tile is even | `Cannot match type float4_e2m1fnx2 vs float4_e2m1fnx4` |
| 4 | **control:** the case-1 kernel with an `float8_e4m3fn` output | compiles |
| 5 | **control:** the FP4 convert inside a `T.SimdVF` region | compiles |
| 6 | **workaround:** `T.SimtVF` for the arithmetic + `T.SimdVF` for the convert | compiles |

Cases 4 and 5 are what make this a report rather than a question: the thread
model converts to FP8 happily, and the FP4 convert exists — it is reaching it
from per-thread code that is impossible.

Other spellings tried, all failing with one of the same three errors: a
`T.vectorized(2)` pair store; an fp4 fragment at 4, 8 and 16 codes per thread;
`T.copy` of a float32 fragment straight into fp4 storage; an explicit
`T.Cast("float4_e2m1fnx2", ...)` (`lane count 2 vs 1`); and
`T.reinterpret("uint8", <fp4 code>)`, which is rejected as
`reinterpret expects source and target to have the same number of bits` (4 vs 8).

## Where it fails in the framework

**Case 1** surfaces in a Python-level pass rebuilding the store after the fp4
buffer has been retyped:

```text
File "tilelang/transform/hoist_broadcast_values.py", line 39, in visit_buffer_store_
  new_stmt = BufferStore(op.buffer, self.visit_expr(op.value), new_indices)
File "src/tirx/ir/stmt.cc", line 375, in tvm::tirx::BufferStore::BufferStore(...)
  TVM_FFI_ICHECK_EQ(index_lanes * buffer_lanes, value_dtype_lanes)
tvm.error.InternalError: Check failed: index_lanes * buffer_lanes == value_dtype_lanes (2 vs. 1)
```

* the store is built in [`tilelang/transform/hoist_broadcast_values.py`][hoist]
  (`visit_buffer_store_`)
* the check is [`src/tirx/ir/stmt.cc`][stmt] in the bundled TVM

By that point [`src/ascend/transform/rewrite_fp4_to_fp4x2.cc`][rewrite] has
retyped the buffer. That pass describes itself as:

> Rewrite packed 4-bit float buffers (float4_e2m1fn, lanes=1) into the 1-byte
> packed-pair form (float4_e2m1fnx2, lanes=2) before Ascend codegen.

and its `VisitStmt_(const BufferStoreNode *)` retypes `node->buffer` and halves
`node->indices[0]` — but leaves **`node->value` untouched**. Nothing then widens
a per-thread scalar value to match the buffer it is being stored into, so a
1-lane value meets a 2-lane buffer and the `BufferStore` constructor rejects it.
The vector paths that do work (cases 5 and 6) never build such a store: they
reach the packed type through the `S.vsts` / `S.vcvt` intrinsics instead.

**Case 2** fails earlier, in the same pass: `RemapBuffer` halves the innermost
dimension through `HalveEven`, and by the time it runs the fp4 fragment's
last dimension is 1 — one code per thread — which is odd, so the even check
fires.

**Case 3** gets past both and fails in the type checker, because the copy out of
the fragment wants `float4_e2m1fnx4` where the rewritten buffer offers
`float4_e2m1fnx2`.

**Codegen could not accept a scalar anyway.** In
[`src/ascend/codegen/codegen_ascend.cc`][cg-ascend] the only FP4 branch is

```cpp
if (t.is_float4_e2m1fn() && lanes == 2) {
  os << "float4_e2m1x2_t";
  return;
}
```

with no `lanes == 1` case; anything else reaches

```cpp
LOG(FATAL) << VFModeName(vf_mode_) << " cannot convert vector type " << t
           << " to an Ascend type";
```

where `VFModeName` prints `SimtVF` or `SimdVF`. Underneath tilelang, the CANN
A5 headers agree: the only FP4 types are the vector ones (`vector_f4e2m1x2`,
`float4_e2m1x2_t`), and the conversion is a vector instruction over them —
`vcvt(vector_f4e2m1x2, vector_bf16, preg, ROUND_R, PART_P0)`. There is no scalar
FP4 type on this hardware to convert into.

## Comparison with the CUDA backend

CUDA has a scalar FP4 type, so the same source lowers. In
[`src/target/cuda/codegen_cuda.cc`][cg-cuda]:

```cpp
decl_stream << "using fp4_e2_t = __nv_fp4_e2m1;\n";
decl_stream << "using fp4_e2x2_t = __nv_fp4x2_e2m1;\n";
decl_stream << "using fp4_e2x4_t = __nv_fp4x4_e2m1;\n";
```

```cpp
} else if (t.is_float4_e2m1fn()) {
  enable_fp4_ = true;
  if (t.lanes() <= 4) {
    os << GetFP4Type(t);          // lanes == 1 is accepted
  } else {
    fail = true;
  }
```

and `CastNode` with a `kFloat4_e2m1fn` target emits a plain C++ cast,
`(__nv_fp4_e2m1)(expr)`, letting the CUDA headers do the conversion. So on CUDA
a *value* of type fp4 exists, one thread can hold one, and the packing into
bytes is a storage concern rather than an expressibility one.

That is what [TileKernels][tk-per-token] relies on — an fp4 fragment written
per element inside `T.Parallel`, then copied out:

```python
out_fragment_or_shared = T.alloc_fragment((block_m, block_k), out_config.dtype)
...
out_fragment_or_shared[i, j * 4 + k] = T.cast(x_fp32_local[k], out_config.dtype,
                                              round='rs', rbits=rbits)
...
T.copy(out_fragment_or_shared, out[pid_token * block_m, pid_hidden * block_k],
       disable_tma=True)
```

It handles the two-codes-per-byte packing by *choosing a block size*, with the
comment "Must ensure each thread has elements as multiple of 2 for FP4" — the
same constraint the Ascend pass tries to enforce with `HalveEven`, but resolved
in the kernel's tiling rather than rejected during lowering. Note that case 3
above satisfies exactly that constraint and still fails.

*The CUDA path is quoted from the sources above, not executed: the machine this
report was produced on has no CUDA device, and the tilelang build is the Ascend
one.*

## Workaround

A `T.SimtVF` region and a `T.SimdVF` region can live in the same kernel, so the
thread model can keep the arithmetic and the vector model can be entered for the
one instruction (case 6 above):

```python
with T.SimtVF(threads=THREADS):
    for i in T.Parallel(K):                 # scale + round to odd, per thread
        bits = T.reinterpret("uint32", x_ub[i] * inv)
        sticky = T.min(bits & 0xFFFF, T.uint32(1))
        bf_ub[i] = T.reinterpret("bfloat16",
                                 T.Cast("uint16", (bits >> 16) | sticky))

with T.SimdVF():                            # the one vector instruction
    for pair in T.serial(K // PAIR):
        S.vsts(q_ub[pair * PAIR],
               S.vcvt(S.vld(bf_ub[pair * PAIR]), T.float4_e2m1fn),
               dist="PK4_B32")
```

The round-to-odd step is needed because `S.vcvt` rounds again; without the
sticky bit the result double-rounds.

This is what the measurement in the summary compares: the same quantization
kernel, 562,280 simulated VF cycles with a hand-written e2m1 encoder inside the
SIMT region against 7,778 with this three-line block, byte-identical output,
verified against a PyTorch reference packer.

## Expected behaviour

Any one of these would resolve it, in decreasing order of preference:

1. **Support a scalar fp4 store from per-thread code.** `RewriteFp4ToFp4x2`
   knows the buffer is packed; what it does not do is pair the *values*. Two
   adjacent per-thread stores, or a store whose index is even and whose value is
   a 2-lane build, could be folded into one `float4_e2m1fnx2` store.
2. **Accept the case-3 shape**, where the user has already arranged an even
   number of codes per thread — that is the contract TileKernels satisfies on
   CUDA, and it fails here only on a lane-count mismatch in the copy.
3. **Diagnose it clearly.** If FP4 output genuinely requires `T.SimdVF`, the
   three internal `ICHECK` failures should become one frontend error naming the
   restriction and the workaround, rather than a lane-arithmetic message from
   inside a broadcast-hoisting pass.

## References

* [`src/ascend/transform/rewrite_fp4_to_fp4x2.cc`][rewrite] — the fp4 → fp4x2 pass (`RemapBuffer`, `HalveEven`, `VisitStmt_(BufferStoreNode)`)
* [`src/ascend/codegen/codegen_ascend.cc`][cg-ascend] — `float4_e2m1fn` only at `lanes == 2`; `VFModeName`
* [`tilelang/transform/hoist_broadcast_values.py`][hoist] — where case 1 surfaces
* [`src/tirx/ir/stmt.cc`][stmt] — the store lane check
* [`src/target/cuda/codegen_cuda.cc`][cg-cuda] — the CUDA FP4 types and cast
* [`tile_kernels/quant/per_token_cast_cuda.py`][tk-per-token] — the CUDA spelling in practice

[rewrite]: https://github.com/tile-ai/tilelang/blob/main/src/ascend/transform/rewrite_fp4_to_fp4x2.cc
[cg-ascend]: https://github.com/tile-ai/tilelang/blob/main/src/ascend/codegen/codegen_ascend.cc
[hoist]: https://github.com/tile-ai/tilelang/blob/main/tilelang/transform/hoist_broadcast_values.py
[stmt]: https://github.com/tile-ai/tvm/blob/tilelang_main/src/tirx/ir/stmt.cc
[cg-cuda]: https://github.com/tile-ai/tvm/blob/tilelang_main/src/target/cuda/codegen_cuda.cc
[tk-per-token]: https://github.com/deepseek-ai/TileKernels/blob/main/tile_kernels/quant/per_token_cast_cuda.py
