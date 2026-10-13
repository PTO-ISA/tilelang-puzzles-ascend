---
name: writing-technical-slides
description: >-
  Write engineering slide decks and technical reports that read as
  human-written rather than model-generated. Use when authoring or editing
  slides, decks, presentations, Marp/Beamer/reveal.js markdown, technical
  reports, benchmark write-ups, or any document that compares implementations
  and quotes measurements.
---

# Writing technical slides

Rules derived from two reviews of a benchmark deck, where the content was
correct and first the writing, then the structure, was rejected. Every rule
below has a before/after taken from those reviews.

The underlying test: **a reader should not be able to delete a sentence without
losing a fact.**

## 1. Titles name a finding or a mechanism

A title is the most-read line on a slide. Abstract nouns and evaluations waste
it.

| rejected | reason | replacement |
|---|---|---|
| `One algorithm, three frontends` | names a shape, not a subject | `tilelang frontend comparison` |
| `The headline: two axes that disagree` | "axes" names nothing | `ASC vs PTO: fewer operations, comparable cycles` |
| `Method, and why the numbers can be trusted` | defensive | `Benchmark setup (and caveats resolved)` |
| `Pattern 2 -- the trap underneath, and the cost` | mystery-novel phrasing | `per_channel/02: performance penalty of SIMT` |
| `Pattern 3 -- the sharpest single trade` | superlative, no content | `per_token/05: SIMT's performance penalty on transpose` |
| `Pattern 4 -- a genuine draw` | sports metaphor + hedge | `per_channel/01: reduction along M needs no cross-lane step` |

Prefer `<subject>: <finding>`. If the title would survive being pasted onto a
different slide, it is too vague.

## 2. No second person

"You" turns a report into a chat reply.

- `what you name` -> `abstraction level`
- `the unit you think in` -> `smallest unit of data`
- `you cannot restride a register` -> `lane assignment inside a vector register is fixed`
- `Where PTO costs you` -> `Where PTO's explicitness costs source length`
- `measure how much you write` -> `measure source size`

## 3. Every sentence carries a fact

Delete sentences whose only function is to announce that something matters.

- `This row sets the theme.` -> delete, or `Observations:`
- `Neither axis predicts the other.` -> `Operations written and cycles executed correlate at r = -0.09 (n = 22).`
- `Nothing here says "SIMT is slow" in general.` -> state the measured range instead.
- `It is tempting to assume ... Across these 22 variants, it does not:` -> drop the set-up, open with the table.

Watch for the scaffolding tic: `Two things are worth separating out`,
`The rule:`, `worth stating`, `it is worth noticing`.

## 4. Replace adjectives with the measurement

If a number exists, the adjective is a worse version of it.

- `among the worst cycle ratios` -> `13.74x, the 14th lowest of 22` (the original was also false: 8 variants were worse)
- `an order of magnitude throughout` -> `3.20x to 21.18x; under 6x on 8 of 22`
- `roughly an order of magnitude` -> `10.44x mean`
- `two axes that disagree` -> `Pearson r = -0.09`

Hedges to cut: `genuinely`, `simply`, `merely`, `quietly`, `plainly`,
`essentially`, `marginally`, `nonetheless`.

## 5. Name the concrete object

- `the vector body` -> `the T.SimdVF / T.SimtVF region`
- `the frontends differ by up to 99.62x in cycles` -> `msprof-simulated VF cycles differ by 10-100x between the SIMT and SIMD implementations`
- `all passing the same checks` -> `all passing python -m harness.check`

Units and measurement method travel with every quantity. `99.62x` alone does not
say *of what*, *measured how*, or *on which variant*.

## 6. Do not anthropomorphise the system under test

`PTO says`, `ASC cannot write`, `a thread knows which element it owns`,
`SIMT wrote a third of the code`, `VMI's wins come from`. Rewrite with the
engineer as the agent, or with no agent: `PTO declares the segment structure on
the load`.

Avoid the scoring frame too — `win`, `lose`, `draw`, `pays off` — unless
reporting an actual contest.

## 7. A slide that concludes about N implementations shows N implementations

The most damaging defect found in the review, because it reads as authoritative:

- a single ASC snippet captioned **`ASC / PTO`** when PTO's code used different
  APIs (`V.vload`/`V.vintlv` vs `S.vld`/`S.vintlv`)
- a slide concluding `PTO is marginally longer` with no PTO code on it
- `SIMT's 12 lines buy notation, not structure` with no SIMT code on it
- `the code shows why` where the referenced code was not on that slide

Before shipping a comparison slide, list the implementations the prose mentions
and the implementations shown. They must match.

## 8. Check that quoted code contains the thing being discussed

Excerpts cut by line count drift away from their point:

- a slide reading "that is an interleave instruction" whose excerpt stopped two
  lines before `S.vintlv`
- a reduction slide whose excerpt stopped before the `vmax`
- an excerpt ending on `inv = T.reinterpret(` — an unterminated open paren
- "an extract of 18 lines" printed above a 6-line block

Anchor excerpts to a *pattern* rather than absolute line numbers, require the
anchor to be unique, and re-verify extraction in CI.

## 9. Re-derive every number; never retype one

Hardcoded figures in prose rot silently and contradict each other. Real examples
from one deck:

- `PTO writes more operations on all four ... and runs fewer cycles on three` —
  arithmetically impossible: the family had five members, the claim silently
  excluded one to make "all four" true and re-included it to make "three" true
- `the largest static gap in the set` on one slide vs `PTO's largest static win`
  on another, naming different variants
- `runs 1.78x fewer cycles` — not a valid ratio; write `runs at 0.56x the cycles`

Generate the document from the measurements, and add a checker that re-derives
each quoted figure and fails when the prose no longer matches.

## 10. Charts for anything over ~10 rows

A 13-row by 8-column table is not readable from a projector. Keep the table for
the record and add a chart that makes the single claim: sorted horizontal bars
for per-item ratios, a log axis when one point is 5x the rest.

## 11. A chart has to carry what the prose cannot

A figure whose whole content fits in one sentence is wasted space, and a
figure confirming what the audience already expects is worse than none.

| rejected | reason | replacement |
|---|---|---|
| scatter of source size vs cycles, annotated `r = -0.09` | the reader already expects no relationship, and the number says it in nine characters | grouped bars of the three implementations' line counts, which no sentence reproduces |

Test: write the chart's finding as one sentence. If the sentence is as
informative as the picture, ship the sentence.

## 12. A run of examples needs an outline and an order

Nine worked examples named `per_token/05`, `per_channel/02`, `per_token/01`
arrive as unconnected pieces: the reader cannot remember what each name means,
why `05` came before `01`, or what any of them was supposed to settle.

Three things fix it, and all three are needed:

1. **One outline slide** mapping each example to its pattern and to the
   question it decides — before the first example.
2. **An order with a reason**, stated in a word or two. This deck's examples
   were reordered into four parts along one axis (how far the data sits from
   the vector register's lane structure): reduce, place the scale, place the
   output, convert it. Difficulty, pipeline order and increasing cost are other
   defensible axes; file name is not one.
3. **A key message per example**, in a fixed position and form. Here: a
   one-sentence takeaway block closing each example, labelled with the
   example's index in the outline (`A1`, `A2`, ...), so a slide seen alone
   still says where it sits.

Carry those labels into the conclusions, so the closing argument points back at
the evidence by name.

## Quick self-check before shipping

1. Does every title name a subject and a finding?
2. Any `you` / `your`?
3. Any sentence that could be deleted without losing a fact?
4. Any adjective where a measured number exists?
5. Does each comparison slide show every implementation it mentions?
6. Does each excerpt contain the operation the caption claims?
7. Is every figure derived from data, and checked?
8. Any table over ~10 rows without a chart beside it?
9. Does every chart carry more than its one-sentence summary?
10. Is there an outline before the examples, a stated reason for their order,
    and a key message closing each one?
