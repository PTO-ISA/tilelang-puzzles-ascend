# Ladder audit — does each variant make a meaningful, incremental change?

Each variant should add one production config and teach one idea. This page
measures whether each one does, and records which ones do not. It is a **report**: the only
structural change made from it so far is the `cast_back` output-dtype merge
described below.

## How this is measured, and what the numbers cannot tell you

- **Solution lines** — substantive (non-blank, non-comment) lines inside the
  `BEGIN/END SOLUTION` region, which is exactly what the student writes.
- **Delta** — changed lines against the *preceding* variant in the same kernel
  and tier.
- **New intrinsics** — `S.*` operations used in this variant's region that no
  earlier variant in the same kernel used. Zero means it teaches no new
  instruction.

Two caveats that matter more than the numbers:

1. **The ladder is not monotone.** Several variants revert to an earlier body,
   so "delta against the predecessor" overstates them. `per_block/04`'s delta
   against its predecessor is 22 lines, but against `01` it is **zero** — the
   regions are byte-identical. Where that happens it is called out.
2. **Region size is not information content.** `per_token/03`'s ASC region is one
   line, but that line is the packed UE8M0 store and the rest of the change lives
   outside the region in buffer dtypes. It teaches a real thing through a weak
   exercise. The fix for that class is to *widen the region*, not to merge.

## The table

| variant | torch | ASC | PTO | Δ ASC | Δ PTO | new ASC intrinsics |
|---|---:|---:|---:|---:|---:|---:|
| cast_back/01_e4m3_fp32sf | 7 | 17 | 13 | — | — | 6 |
| cast_back/02_packed_ue8m0 | 10 | 13 | 17 | 20 | 16 | 3 |
| cast_back/03_block_sf | 6 | 10 | 11 | 11 | 14 | **0** |
| cast_back/04_per_channel_sf | 5 | 7 | 8 | 7 | 5 | **0** |
| cast_back/05_fp4_e2m1 | 19 | 19 | 12 | 24 | 16 | 1 |
| cast_back/06_col_major_compose | 10 | 22 | 23 | 27 | 19 | **0** |
| per_token/01_raw_fp32sf | 7 | 38 | 25 | — | — | 13 |
| per_token/02_round_sf | 10 | 10 | **3** | 46 | 28 | 5 |
| per_token/03_packed_ue8m0 | 13 | **1** | 4 | 11 | 5 | **0** |
| per_token/04_fp32_in_fp4_out | 30 | 35 | 24 | 36 | 28 | 3 |
| per_token/05_col_major_sf | 10 | **63** | **57** | 56 | 63 | 4 |
| per_token/06_split_requant | 17 | **58** | 45 | **93** | **72** | **0** |
| per_token/07_bf16_fast_compose | 14 | 40 | 25 | **68** | **62** | 4 |
| per_block/01_raw_32x32 | 9 | 22 | 26 | — | — | 11 |
| per_block/02_round_packed | 15 | 29 | 34 | 13 | 16 | 5 |
| per_block/03_fp4_e2m1 | 8 | 30 | 30 | 35 | 28 | 3 |
| per_block/04_col_major_tma | 9 | 22 | 26 | 22 | 14 | **0** |
| per_block/05_split_compose | 19 | 34 | 38 | 24 | 28 | 1 |
| per_channel/01_raw_32tokens | 7 | 22 | 22 | — | — | 10 |
| per_channel/02_round_packed_m | 13 | **8** | **9** | 26 | 27 | 1 |
| per_channel/03_requant_bf16 | 10 | 30 | 36 | 34 | 41 | 2 |
| per_channel/04_compose | 17 | **49** | **54** | 59 | 68 | 7 |

## When to merge two variants

**Merge only when the change is trivial in all three tiers.** Triviality in torch
proves nothing, because a torch parameter routinely becomes a schedule change and
then an instruction change on the way down.

### Merged: the output dtype (was `cast_back/02_fp32_out`)

| | torch | ASC | PTO |
|---|---|---|---|
| the whole change | drop `.to(bfloat16)` | **one store line** | **one store line** |

Trivial in all three, so it is now a trace-time `out_dtype` switch on
[cast_back/01](quant/cast_back/01_e4m3_fp32sf.md). The two store paths sit side by
side, which reads better than a two-file diff: `NORM_B32` against `PK_B32`, where
the *wider* dtype takes the *simpler* instruction. `cast_back` renumbered to six
variants.

### Not merged: the `sf_block` family (`cast_back/03`, `/04`)

These look trivial from torch and are not:

| step | torch | ASC | PTO |
|---|---|---|---|
| `(1,32)` → `(32,32)` | a different `.view()` | region identical, but `Sf` changes shape and the loop nest becomes `for m_block / for row` with the scale DMA **hoisted** | same schedule change |
| `(32,32)` → `(32,1)` | a different broadcast | **loses `S.pset`, one `BRC_B32` load and `S.vsel`** — 8 ops to 6 | loses `dist_mode='brc', group=` |

So the first step is a *schedule* change and the second an *instruction* change.
They stay, and the asymmetry is itself the lesson.

### Meets the rule but not yet changed: `per_block/04_col_major_tma`

The strongest remaining candidate, and stronger than the one already merged:

- the ASC and PTO regions are **byte-identical to `per_block/01`** — 0 lines, 0
  tokens. The student retypes 22 (ASC) / 26 (PTO) lines verbatim;
- the hint was a **verbatim copy** of `01`'s, so nothing told them so (now fixed —
  see below);
- torch is 2 real lines (`return q, sf.T.contiguous()`);
- the entire change is outside the region: `SfCm` is `(num_k_blocks,
  num_m_blocks)` and the final `T.copy(sf_ub[0:1], SfCm[kb, mb:mb + 1])` swaps a
  pair of indices.

Trivial in all three tiers, so by the rule it should fold into `per_block/01` as a
`col_major` switch. Acting on it means renumbering `per_block`, so it is listed
rather than done.

## Variants that teach a real idea through a weak exercise

Here the region boundary is drawn so the student does not write the new thing.
**The fix is to widen the region, not to merge** — and in the PTO cases it also
restores 1:1:1 alignment with ASC, where the student *does* write the arithmetic.

| variant | the problem |
|---|---|
| `per_token/03` (ASC) | the region is **1 line of 80** in the file: the `PK4_B32` packed store. Everything else is given. |
| `per_token/02` (PTO) | the region is 3 lines calling a `compute_scale` macro **defined above it**. The ASC student writes the ceil-log2 trick by hand; the PTO student does not. |
| `per_token/04` (PTO) | same shape: `to_bf16_round_odd` is provided above the region. |
| `per_token/06` (PTO) | novelty routed through provided macros; zero new intrinsics in the region. |
| `per_channel/02` | 8 (ASC) / 9 (PTO) lines, against 13 in torch — the NPU exercise is thinner than the torch one. |

## Steps that are too large

Four consecutive-pair deltas exceed 55 changed lines in both vector tiers:

| step | Δ ASC | Δ PTO | what arrives at once |
|---|---:|---:|---|
| `per_token/04` → `/05` | 56 | 63 | `S.vci`, `S.vgather2`, the index-vector transpose |
| `per_token/05` → `/06` | **93** | **72** | four compile-time modes, a scratch buffer, two barriers |
| `per_token/06` → `/07` | 68 | 62 | `S.vld2`, `S.vcgmax`, bfloat16 compute at 128 lanes |
| `per_channel/03` → `/04` | 59 | 68 | 7 new ASC intrinsics in one variant |

`per_token/01` → `/02` is also 46 lines in ASC: the exponent trick arrives as five
new intrinsics together. Splitting any of these means inventing a new intermediate
variant — new material rather than restructuring — so none is recommended without
a decision on what the intermediate step should teach.

## Fixed in this pass: seven misleading hints

A hint that describes code other than the answer is worse than no hint. Found by
comparing every hint's named operations against its own answer body:

| variant | was |
|---|---|
| `per_block/02` (ASC, PTO) | a **verbatim copy of `01`'s hint**, telling the student to "divide both ways" where the answer uses the exponent trick (`vmuls`/`vshrs`/`vadds`/`vshls`, and no `vdiv` at all) |
| `per_block/03` (ASC, PTO) | the same copied hint, with no mention of the FP4 path the answer actually takes (`vdintlv`/`vor`/`vmin`) |
| `per_block/04` (ASC, PTO) | the same copied hint again — accurate by luck, since the body *is* `01`'s, but it never said so |
| `per_token/04` (ASC) | told the student to use `S.vmins`, which **does not compile** on this toolchain (`asc_min_scalar` has no uint16 overload) |
| `per_token/02`, `/04` (PTO) | told the student to "write a macro" that is already provided above the region |

All now describe their own answer. The check is mechanical and worth re-running
after any edit to a hint or a solution body.
