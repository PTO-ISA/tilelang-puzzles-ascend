# Working the puzzles — recommended order

This directory is **code only**. The algorithms are explained in
[`doc/quant/`](../doc/quant/README.md); this page is about what to implement, in
what order, and what actually depends on what.

```
puzzles/torch/quant/   plain PyTorch, CPU, instant     <- start here
puzzles/asc/quant/     Ascend SIMD   (T.simd, "S")     ) either order,
puzzles/pto/quant/     PTO VMI       (T.vmi,  "V")     ) independent
```

## What depends on what

```
            ┌─> asc/   (23 variants)
torch/ ─────┤
            └─> pto/   (23 variants)

torch -> NPU : recommended, not required
asc <-> pto  : no dependency in either direction
```

**Torch first** is a recommendation about learning, not a technical constraint.
Each torch exercise is the same arithmetic as its NPU counterpart with none of the
hardware: no lane widths, no distribution modes, no memory barriers. Getting the
algorithm wrong there costs a second; getting it wrong in a vector kernel costs
thirty, and you will not know which of the two mistakes you made.

**ASC and PTO do not depend on each other.** They are two instruction sets over
the same hardware at different levels of abstraction — ASC names the physical
operation, VMI names the intent — and neither is built on the other. The puzzle
hints are self-contained in both directions: no ASC hint mentions VMI, and 22 of
23 PTO hints are pure VMI. The exception is `pto/per_token/03`, which names an ASC
store mode (`PK4_B32`) only to say you do **not** need one — nothing to act on if
the name means nothing to you.

So **starting with PTO is a legitimate path**, and for some people the better one:
`group=`, `size=` and `dist_mode=` describe what you want, where ASC's `BRC_B32`
and `"float32x64"` require knowing what the machine provides. If you came for the
higher-level surface, go straight to it — you are not missing a prerequisite.

The one thing that *is* ordered is the **comparison**. Each variant page carries a
`PTO vs ASC` section, and those read better once you have written at least one of
the two. They are commentary, not instructions: skipping them costs you the
argument, not the kernel.

### What the checks compare against

All three tiers are checked against the same oracle,
[`harness/oracle.py`](../harness/oracle.py), which is a standalone PyTorch
reference — it does **not** import anything from this directory. That is why the
tier order is free: an ASC variant is verified whether or not you ever wrote the
torch one.

## Order within a tier

Follow the numbering. Each variant adds exactly one production config to the one
before it, and later variants compose earlier ones.

Start with **`cast_back`** — it is the only kernel with no reduction, since its
scale factors are an input. That makes it the only place to learn the vector data
path (load, convert, broadcast, store) without also learning reduce-and-broadcast
machinery. Then:

| order | kernel | what it is for | variants |
|---|---|---|---|
| 1 | [`cast_back`](../doc/quant/cast_back/README.md) | the data path, no reduction | 7 |
| 2 | [`per_token`](../doc/quant/per_token/README.md) | segmented reduction along the fast axis | 7 |
| 3 | [`per_block`](../doc/quant/per_block/README.md) | 2-D reduction, and the lane-width rule | 5 |
| 4 | [`per_channel`](../doc/quant/per_channel/README.md) | reduction along the slow axis | 4 |

`per_token` before `per_block` because a segment reduction inside a register is the
harder idea and everything after reuses it. `per_channel` last because its
reduction is the *simplest* of the three — once you have seen why, you understand
the whole axis argument.

### If you only do a few

Six variants carry most of the transferable ideas:

| variant | the idea you cannot get elsewhere |
|---|---|
| [`cast_back/01`](../doc/quant/cast_back/01_e4m3_fp32sf.md) | the vector data path, and broadcast loads |
| [`per_token/01`](../doc/quant/per_token/01_raw_fp32sf.md) | segmented reduction, masks, the three-pass shape |
| [`per_token/02`](../doc/quant/per_token/02_round_sf.md) | the ceil-log2 bit trick, exact power-of-two scales |
| [`per_token/05`](../doc/quant/per_token/05_col_major_sf.md) | you cannot restride a register — index arithmetic and `vgather` |
| [`per_block/01`](../doc/quant/per_block/01_raw_32x32.md) | pick the lane width from the hardware, reshape the problem to fit |
| [`per_channel/02`](../doc/quant/per_channel/02_round_packed_m.md) | a vector load's width is a property of the register, not the request |

## The loop

```bash
# 1. read the variant's page
less doc/quant/per_token/02_round_sf.md

# 2. open the puzzle and fill in the TODO between the SOLUTION sentinels
$EDITOR puzzles/asc/quant/puzzle/per_token/02_round_sf.py

# 3. check it -- ~6 s while unimplemented, ~30 s once it compiles
python -m harness.check asc/per_token/02 --role puzzle

# 4. compare against the reference
diff puzzles/asc/quant/{puzzle,answer}/per_token/02_round_sf.py
```

An unimplemented variant reports `TODO` in about 6 s: the harness traces the kernel
on the host and stops before launching the simulator, so you are not waiting on a
simulator for an answer you have not written yet.

Once it is right, `--role puzzle` and the answer give the same verdict:

```bash
python -m harness.check asc/per_token/02                # the reference
python -m harness.check per_token/02                    # all three tiers
python -m harness.check per_token/02 --list             # what that would run
```

## Reading the same variant three ways

The numbering is identical across tiers, so the same variant is directly
diffable — this is the comparison the repo is built for:

```bash
diff puzzles/asc/quant/answer/per_token/02_round_sf.py \
     puzzles/pto/quant/answer/per_token/02_round_sf.py
```

`tools/vf_lines.py` counts operations and lines for every such pair, and the
`PTO vs ASC` section of each variant page explains the measured difference. It is
not uniform — VMI collapses segmented reduce-and-broadcast and width changes, and
gains nothing at all on gathers or whole-vector reductions. `per_token/02` is the
widest gap in the ladder (45 operations against 14), while
[`per_channel/01`](../doc/quant/per_channel/01_raw_32tokens.md) comes out level:
17 operations against 16, and 21 source lines each.

## Time budget

| | per variant | whole tier |
|---|---|---|
| torch | ~5 s | 1.9 min |
| ASC | 18–32 s (one at 64 s) | 10.6 min |
| PTO | 17–31 s (one at 61 s) | 10.4 min |
| unsolved puzzle, any tier | ~6 s | 6.3 min |

Measured end to end including compilation, on the CPU simulator. The per-variant
table is in the [root README](../README.md).

## The files

```
<tier>/quant/answer/<kernel>/NN_name.py    the reference: a complete kernel
<tier>/quant/puzzle/<kernel>/NN_name.py    the same file, kernel body removed,
                                           a TODO hint left in its place
```

Both are generated from the answer — `tools/make_puzzles.py` replaces the body
between the `BEGIN/END SOLUTION` sentinels and leaves everything else identical, so
a diff between them is exactly the work you have to do.

Neither is a script. A variant file holds `compile_kernel` and `launch` and nothing
else — no `main`, no tests, no worked examples — which is why one is ~60 lines
rather than ~230. Run them through the harness; running one directly exits with the
command you wanted.
