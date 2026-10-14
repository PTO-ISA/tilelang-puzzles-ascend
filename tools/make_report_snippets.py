"""Extract the code fragments that presentation/ quotes, verbatim from the kernels.

    python tools/make_report_snippets.py            # write presentation/snippets/
    python tools/make_report_snippets.py --check    # fail if any has drifted
    python tools/make_report_snippets.py --show NAME

The report and the deck make claims about how three vector frontends express the
same algorithm. Those claims are only worth reading if the code shown is the code
that actually ran, so no snippet is typed by hand: each is cut from a kernel file
that `python -m harness.check` passes and `tools/vf_cycles.py` has measured.

Fragments are located by an **anchor** -- the one line matching a regex -- plus a
line count, rather than by absolute line numbers, so ordinary edits above a
fragment cannot silently shift what gets quoted. An anchor that matches more than
once is an error, not a guess; `after=` narrows the search when a line repeats.
`--check` re-resolves every anchor and fails on any difference.

Each written file carries a header naming its source path and the resolved line
range, so a reader can go and look.
"""

from __future__ import annotations

import argparse
import re
import sys
import textwrap
from pathlib import Path
from typing import NamedTuple

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "presentation" / "snippets"
ANS = "puzzles/{tier}/quant/answer/{variant}.py"
# One pseudo-tier: a fragment quoted from a probe rather than from a ladder
# kernel. Probes run and assert like anything else, so a fragment cut from one
# is still code that executed -- which is the only property that matters here.
PROBE = "harness/probe/{variant}.py"


class Cut(NamedTuple):
    """Where a fragment lives, described so edits elsewhere cannot move it."""

    tier: str
    variant: str
    anchor: str                 # regex; must match exactly one line in the window
    nlines: int
    after: str | None = None    # window starts below the first line matching this
    before: str | None = None   # window ends at the first such line after `after`


# Keep fragments SHORT: they are quoted side by side on slides.
SNIPPETS: dict[str, Cut] = {
    # -- the algorithm, in torch --------------------------------------------
    "torch_dequant": Cut("torch", "cast_back/01_e4m3_fp32sf",
                         r"grouped = q\.float\(\)", 3),
    "torch_per_token": Cut("torch", "per_token/01_raw_fp32sf",
                           r"grouped = x\.float\(\)", 4),

    # -- pattern 1: the per-group scale broadcast ---------------------------
    "broadcast_asc": Cut("asc", "cast_back/01_e4m3_fp32sf",
                         r"lo = S\.vld\(sf_ub\[group\]", 5),
    "broadcast_pto": Cut("pto", "cast_back/01_e4m3_fp32sf",
                         r"scale = V\.vload\(sf_ub\[group\]", 4),
    "broadcast_simt": Cut("simt", "cast_back/01_e4m3_fp32sf",
                          r"for i in T\.Parallel\(hidden\):", 2),
    "outdtype_asc": Cut("asc", "cast_back/01_e4m3_fp32sf",
                        r'if out_dtype == "bfloat16":', 6),
    "outdtype_pto": Cut("pto", "cast_back/01_e4m3_fp32sf",
                        r'if out_dtype == "bfloat16":', 4),

    # -- pattern 2: SIMT wins -- the M-pack interleave ----------------------
    "interleave_asc": Cut("asc", "per_channel/02_round_packed_m",
                          r"a = S\.vld\(sf_rows_ub\[0, col\]", 8),
    "interleave_pto": Cut("pto", "per_channel/02_round_packed_m",
                          r"a = V\.vload\(sf_rows_ub\[0, col\]", 8),
    "interleave_simt": Cut("simt", "per_channel/02_round_packed_m",
                           r"for c in T\.Parallel\(hidden\):", 3,
                           after=r"BEGIN SOLUTION"),
    "pad_asc": Cut("asc", "per_channel/02_round_packed_m",
                   r"sf_rows_ub = T\.alloc_shared", 1),
    "pad_pto": Cut("pto", "per_channel/02_round_packed_m",
                   r"sf_rows_ub = T\.alloc_shared", 1),
    "pad_simt": Cut("simt", "per_channel/02_round_packed_m",
                    r"sf_rows_ub = T\.alloc_shared", 1),

    # -- pattern 3: SIMT wins -- the in-register transpose ------------------
    "transpose_asc": Cut("asc", "per_token/05_col_major_sf",
                         r"lane = S\.vci\(0, T\.int32\)", 7),
    "transpose_asc_gather": Cut("asc", "per_token/05_col_major_sf",
                                r"S\.vsts\(sf_out_ub\[base, 0\],", 2),
    "transpose_pto": Cut("pto", "per_token/05_col_major_sf",
                         r"lane = V\.vci\(T\.int32\(0\), size=LANES\)", 10),
    "transpose_pto_gather": Cut("pto", "per_token/05_col_major_sf",
                                r"V\.vstore\(V\.vgather\(sf_dense_ub", 2),
    # the whole T.Parallel body, so that g, biased and the store's two indices
    # are all defined where the reader can see them
    "transpose_simt": Cut("simt", "per_token/05_col_major_sf",
                          r"for g in T\.Parallel\(num_groups\):", 14),
    # ...and the declaration that makes the store a transpose: the buffer is
    # (groups, tokens) where the input is read token by token.
    "transpose_simt_decl": Cut("simt", "per_token/05_col_major_sf",
                               r"# Already transposed: one column per token", 2),

    # -- pattern 4: where PTO is already clean ------------------------------
    "chanreduce_asc": Cut("asc", "per_channel/01_raw_32tokens",
                          r"acc = S\.alloc_local", 10),
    "chanreduce_pto": Cut("pto", "per_channel/01_raw_32tokens",
                          r"acc = V\.alloc_local", 10),
    "chanreduce_simt": Cut("simt", "per_channel/01_raw_32tokens",
                           r"for c in T\.Parallel\(hidden\):", 8),

    # -- pattern 5: PTO's real win -- the segmented reduction ---------------
    "segreduce_asc": Cut("asc", "per_token/01_raw_fp32sf",
                         r"a0 = S\.vabs\(S\.vcvt\(S\.vld\(x_ub\[col\]", 12),
    # pass 3 loads x the same way, so start the search inside pass 1.
    "segreduce_pto": Cut("pto", "per_token/01_raw_fp32sf",
                         r"x = V\.vcvt\(V\.vload\(x_ub\[col\], size=PAIR\)", 3,
                         after=r"pass 1: segmented reduce",
                         before=r"T\.simd\.mem_bar"),
    "segreduce_simt": Cut("simt", "per_token/01_raw_fp32sf",
                          r"for g in T\.Parallel\(num_groups\):", 7),

    # -- pattern 5b: width as an argument -----------------------------------
    "exponent_asc": Cut("asc", "per_token/02_round_sf",
                        r"one = S\.vdup\(1, T\.uint32\)", 9),
    "exponent_pto": Cut("pto", "per_token/02_round_sf",
                        r"one = V\.vbrc\(T\.uint32\(1\), size=lanes\)", 10),
    # spans the BEGIN SOLUTION marker, which resolve() drops
    "exponent_simt": Cut("simt", "per_token/02_round_sf",
                         r"for g in T\.Parallel\(num_groups\):", 12),

    # -- pattern 6: where PTO costs you -------------------------------------
    "ue8m0_asc": Cut("asc", "cast_back/02_packed_ue8m0",
                     r"packed = S\.vld\(sf_ub\[strip\]", 5),
    "ue8m0_pto": Cut("pto", "cast_back/02_packed_ue8m0",
                     r"packed = V\.vload\(sf_ub\[strip\]", 5),
    "ue8m0_simt": Cut("simt", "cast_back/02_packed_ue8m0",
                      r"for i in T\.Parallel\(hidden\):", 5),

    # -- pattern 7: where SIMT loses ----------------------------------------
    "fp4_asc": Cut("asc", "per_token/04_fp32_in_fp4_out",
                   r"low, high = S\.vdintlv", 6),
    "fp4_pto": Cut("pto", "per_token/04_fp32_in_fp4_out",
                   r'low, high = V\.vunzip\(x, "uint16"\)', 3),
    "fp4_pto_call": Cut("pto", "per_token/04_fp32_in_fp4_out",
                        r"odd = to_bf16_round_odd\(q, PAIR\)", 2),
    "fp4_simt": Cut("simt", "per_token/04_fp32_in_fp4_out",
                    r'bits = T\.reinterpret\("int32", v\)', 6),

    # -- A2's alternative reduction, measured by the probe it comes from -----
    "reduce_max_simt": Cut("probe", "simt_reduce_max",
                           r"for g, j in T\.Parallel\(num_groups, group_size\):", 4),

    # -- pattern 8: the static/dynamic inversion ----------------------------
    # Both per_block loops repeat `for chunk in ...`, so start the search at
    # the reduction pass's own first line.
    "blockreduce_asc": Cut("asc", "per_block/01_raw_32x32",
                           r"for chunk in T\.serial\(num_chunks\):", 5,
                           after=r"mask_chunks = S\.pset",
                           before=r"S\.mem_bar"),
    "blockreduce_simt": Cut("simt", "per_block/01_raw_32x32",
                            r"# level 1: one thread per row of the tile", 8),
    "blockreduce_pto": Cut("pto", "per_block/01_raw_32x32",
                           r"for chunk in T\.serial\(num_chunks\):", 4,
                           after=r"one1 = V\.create_mask",
                           before=r"T\.simd\.mem_bar"),
}


def resolve(name: str) -> tuple[Path, int, int, str]:
    cut = SNIPPETS[name]
    tmpl = PROBE if cut.tier == "probe" else ANS
    path = ROOT / tmpl.format(tier=cut.tier, variant=cut.variant)
    lines = path.read_text().splitlines()

    start, stop = 0, len(lines)
    if cut.after is not None:
        where = [i for i, ln in enumerate(lines) if re.search(cut.after, ln)]
        if not where:
            raise SystemExit(f"{name}: after=/{cut.after}/ not found in "
                             f"{path.relative_to(ROOT)}")
        start = where[0] + 1
    if cut.before is not None:
        where = [i for i, ln in enumerate(lines)
                 if i >= start and re.search(cut.before, ln)]
        if not where:
            raise SystemExit(f"{name}: before=/{cut.before}/ not found below "
                             f"after= in {path.relative_to(ROOT)}")
        stop = where[0]

    pat = re.compile(cut.anchor)
    hits = [i for i, ln in enumerate(lines)
            if start <= i < stop and pat.search(ln) and "hint=" not in ln]
    if not hits:
        raise SystemExit(f"{name}: anchor /{cut.anchor}/ not found in "
                         f"{path.relative_to(ROOT)}")
    if len(hits) > 1:
        raise SystemExit(f"{name}: anchor /{cut.anchor}/ matches {len(hits)} lines "
                         f"in {path.relative_to(ROOT)} (lines "
                         f"{[h + 1 for h in hits]}); make it unique, or bound "
                         f"the window with after= / before=")
    first = hits[0]
    # `# --- BEGIN/END SOLUTION` are puzzle scaffolding, not part of any
    # kernel's algorithm, and the BEGIN marker carries a long hint string. A
    # fragment that spans one drops it, and `nlines` counts the source lines,
    # marker included -- so the count still matches the file.
    kept = [ln for ln in lines[first:first + cut.nlines]
            if "BEGIN SOLUTION" not in ln and "END SOLUTION" not in ln]
    body = textwrap.dedent("\n".join(kept)).rstrip() + "\n"
    return path, first + 1, first + cut.nlines, body


def render(name: str) -> str:
    path, lo, hi, body = resolve(name)
    return (f"# {path.relative_to(ROOT)}:{lo}-{hi}\n"
            f"# extracted by tools/make_report_snippets.py -- do not edit\n"
            f"{body}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check", action="store_true",
                    help="exit non-zero if any snippet differs from its source")
    ap.add_argument("--show", metavar="NAME", help="print one snippet and exit")
    args = ap.parse_args()

    if args.show:
        if args.show not in SNIPPETS:
            print(f"unknown snippet {args.show!r}; have:\n  "
                  + "\n  ".join(sorted(SNIPPETS)))
            return 2
        print(render(args.show), end="")
        return 0

    OUT.mkdir(parents=True, exist_ok=True)
    stale, written = [], 0
    manifest = ["# generated by tools/make_report_snippets.py",
                "# snippet, source, lines"]
    for name in sorted(SNIPPETS):
        path, lo, hi, _ = resolve(name)
        text = render(name)
        dest = OUT / f"{name}.py"
        manifest.append(f"{name}, {path.relative_to(ROOT)}, {lo}-{hi}")
        if args.check:
            if not dest.exists() or dest.read_text() != text:
                stale.append(name)
        else:
            dest.write_text(text)
            written += 1

    if args.check:
        extra = sorted(p.stem for p in OUT.glob("*.py") if p.stem not in SNIPPETS)
        if stale or extra:
            for n in stale:
                print(f"STALE   {n}  (source changed, or the file was hand-edited)")
            for n in extra:
                print(f"ORPHAN  {n}  (no longer in the table)")
            print(f"\n{len(stale) + len(extra)} problem(s); "
                  f"re-run without --check to regenerate")
            return 1
        print(f"all {len(SNIPPETS)} snippets match their sources")
        return 0

    (OUT / "MANIFEST.txt").write_text("\n".join(manifest) + "\n")
    print(f"wrote {written} snippets + MANIFEST.txt to {OUT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
