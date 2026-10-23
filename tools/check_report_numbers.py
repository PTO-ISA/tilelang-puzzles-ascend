"""Re-derive every number and code block in presentation/ from the measured data.

    python tools/check_report_numbers.py
    python tools/check_report_numbers.py --list     # show every claim checked

The report argues from measurement, so a stale figure in it is worse than no
figure. Nothing in it is trusted: this re-reads `perf/quant_vf_cycles.json` and
`tools/vf_lines.py`, rebuilds each claim as a string, and fails if the report
does not contain that exact string. When a kernel is edited and re-measured, the
derived string changes and this check goes red -- which is the point.

Four groups are checked:

* **report code blocks** -- every ``<!-- snippet: NAME -->`` block must be
  byte-identical to what ``tools/make_report_snippets.py`` extracts today.
* **deck code blocks** -- every fenced Python block in ``presentation/slides.md``
  must be one of those extracts. The deck carries no snippet markers, so this is
  the check that nothing in it was typed by hand.
* **the aggregate table** -- all 22 rows are reconstructed cell by cell.
* **inline claims** -- the per-pattern cycle counts, ratios, line and operation
  counts quoted in the prose.

What it cannot check: prose. "the clearest win" is a judgement; the numbers
beside it are not.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "presentation" / "quant_kernel_style_compare.md"
SLIDES = ROOT / "presentation" / "slides.md"
sys.path.insert(0, str(ROOT / "tools"))

from make_report_snippets import SNIPPETS, resolve                # noqa: E402


def _cite(name: str) -> str:
    """`path:first-last` for one snippet, as the report prints it."""
    path, first, last, _ = resolve(name)
    return f"{path.relative_to(ROOT)}:{first}-{last}"


def load() -> tuple[dict, dict]:
    cycles = json.loads((ROOT / "perf/quant_vf_cycles.json").read_text())
    out = subprocess.run([sys.executable, str(ROOT / "tools/vf_lines.py")],
                         capture_output=True, text=True, cwd=ROOT).stdout
    static = {}
    for ln in out.splitlines():
        p = ln.split()
        if len(p) >= 9 and "/" in p[0]:
            static[p[0]] = dict(ops_asc=int(p[1]), ops_pto=int(p[2]),
                                ln_asc=int(p[3]), ln_pto=int(p[4]),
                                ln_simt=int(p[5]))
    if not static:
        raise SystemExit("tools/vf_lines.py produced no rows")
    return cycles, static


def build_claims(cycles: dict, static: dict) -> list[tuple[str, str]]:
    """(description, exact string the report must contain)."""
    c = lambda t, v: cycles[f"{t}/{v}"]["cycles"]
    claims: list[tuple[str, str]] = []

    # --- the aggregate table, row by row -------------------------------
    for v, r in static.items():
        a, p, s = c("asc", v), c("pto", v), c("simt", v)
        pa, sa = p / a, s / a
        bold = "**" if pa < 0.98 else ""
        claims.append((f"table row {v}",
                       f"| `{v}` | {r['ops_asc']} | {r['ops_pto']} | "
                       f"{r['ln_asc']} | {r['ln_pto']} | {r['ln_simt']} | "
                       f"{a:,.0f} | {p:,.0f} | {s:,.0f} | "
                       f"{bold}{pa:.2f}x{bold} | {sa:.2f}x |"))

    # --- aggregates ----------------------------------------------------
    tot = lambda k: sum(r[k] for r in static.values())
    pr = [c("pto", v) / c("asc", v) for v in static]
    sr = [c("simt", v) / c("asc", v) for v in static]
    nofp4 = [x for v, x in zip(static, sr) if "per_token/04" not in v]
    claims += [
        ("PTO ops share",   f"**{tot('ops_pto') / tot('ops_asc'):.0%}**"),
        ("PTO line share",  f"**{tot('ln_pto') / tot('ln_asc'):.0%}**"),
        ("SIMT line share", f"**{tot('ln_simt') / tot('ln_asc'):.0%}**"),
        ("PTO cycle mean",  f"**{sum(pr) / len(pr):.2f}x mean**"),
        ("PTO cycle range", f"({min(pr):.2f}x–{max(pr):.2f}x)"),
        ("PTO wins",        f"**{sum(1 for x in pr if x < 0.98)} of 22**"),
        ("SIMT cycle mean", f"{sum(sr) / len(sr):.2f}x mean"),
        ("SIMT ex-FP4",     f"**{min(nofp4):.2f}x–{max(nofp4):.2f}x, mean "
                            f"{sum(nofp4) / len(nofp4):.2f}x**"),
    ]
    allln = [n for r in static.values()
             for n in (r["ln_asc"], r["ln_pto"], r["ln_simt"])]
    claims.append(("vector body line span",
                   f"**{min(allln)} to {max(allln)} lines**"))

    # --- the per-pattern tables ----------------------------------------
    # Each pattern quotes ASC/PTO/SIMT cycles and the two ratios.
    for label, v in [("pattern 1 cast_back/01", "cast_back/01_e4m3_fp32sf"),
                     ("pattern 3.1 per_channel/02", "per_channel/02_round_packed_m"),
                     ("pattern 3.2 per_token/05", "per_token/05_col_major_sf"),
                     ("pattern 4 per_channel/01", "per_channel/01_raw_32tokens"),
                     ("pattern 5.1 per_token/01", "per_token/01_raw_fp32sf"),
                     ("pattern 5.2 per_token/02", "per_token/02_round_sf"),
                     ("pattern 7 per_token/04", "per_token/04_fp32_in_fp4_out")]:
        a, p, s = c("asc", v), c("pto", v), c("simt", v)
        claims.append((f"{label} cycle row",
                       f"| 1.00x | {p / a:.2f}x | "
                       f"{'**' if s / a > 50 else ''}{s / a:.2f}x"
                       f"{'**' if s / a > 50 else ''} |"))
        claims.append((f"{label} lines",
                       f"| {static[v]['ln_asc']} | {static[v]['ln_pto']} | "))

    # --- individual figures quoted in prose ----------------------------
    bf = cycles["simt/per_token/04_fp32_in_fp4_out+branch_free"]["cycles"]
    hw = cycles["simt/per_token/04_fp32_in_fp4_out+hw_convert"]["cycles"]
    fp4 = c("simt", "per_token/04_fp32_in_fp4_out")
    claims += [
        ("branch-free cycles", f"**{bf:,.0f}**"),
        # the third spelling: the e2m1 encoder replaced by a SimdVF convert
        ("hw_convert cycles", f"| three lines of `T.SimdVF` | **{hw:,.0f}** | "
                              f"**{hw / c('asc', 'per_token/04_fp32_in_fp4_out'):.2f}x** |"),
        ("hw_convert speed-up", f"**{fp4 / hw:.2f}x**, bit-exact"),
        ("branch-free vs ASC",
         f"{bf / c('asc', 'per_token/04_fp32_in_fp4_out'):.1f}x ASC rather than "
         f"{fp4 / c('asc', 'per_token/04_fp32_in_fp4_out'):.1f}x"),
        ("per_block/01 SIMT fold",
         f"**{c('simt', 'per_block/01_raw_32x32') / c('asc', 'per_block/01_raw_32x32'):.2f}x**: "
         f"{c('simt', 'per_block/01_raw_32x32'):,.0f} cycles against ASC's "
         f"{c('asc', 'per_block/01_raw_32x32'):,.0f}, from "
         f"{static['per_block/01_raw_32x32']['ln_simt']} source lines against "
         f"{static['per_block/01_raw_32x32']['ln_asc']}"),
        ("per_block/03 SIMT ratio",
         f"{c('simt', 'per_block/03_fp4_e2m1') / c('asc', 'per_block/03_fp4_e2m1'):.2f}x"),
        ("cast_back/05 PTO regression",
         f"worst\ncycle regression in the set, "
         f"{c('pto', 'cast_back/05_fp4_e2m1') / c('asc', 'cast_back/05_fp4_e2m1'):.2f}x"),
        ("per_channel/04 draw",
         f"**{c('pto', 'per_channel/04_compose') / c('asc', 'per_channel/04_compose'):.2f}x**"),
        ("per_channel/04 lines",
         f"**{static['per_channel/04_compose']['ln_pto']} lines against ASC's "
         f"{static['per_channel/04_compose']['ln_asc']}**"),
        ("per_channel/01 ops",
         f"**{static['per_channel/01_raw_32tokens']['ops_asc']} operations against "
         f"{static['per_channel/01_raw_32tokens']['ops_pto']}"),
        ("cast_back/02 ops tie",
         f"{static['cast_back/02_packed_ue8m0']['ops_asc']} each"),
        ("cast_back/02 PTO longer",
         f"({static['cast_back/02_packed_ue8m0']['ln_pto']} against "
         f"{static['cast_back/02_packed_ue8m0']['ln_asc']})"),
        ("cast_back/05 static win",
         f"({static['cast_back/05_fp4_e2m1']['ops_asc']} against "
         f"{static['cast_back/05_fp4_e2m1']['ops_pto']}, "
         f"{static['cast_back/05_fp4_e2m1']['ln_asc']} lines against "
         f"{static['cast_back/05_fp4_e2m1']['ln_pto']})"),
    ]

    # the per_block inversion table
    for v in ["per_block/01_raw_32x32", "per_block/02_round_packed",
              "per_block/04_col_major_tma", "per_block/05_split_compose"]:
        r = static[v]
        pa = c("pto", v) / c("asc", v)
        bold = "**" if pa < 0.98 else ""
        claims.append((f"inversion row {v}",
                       f"| `{v}` | {r['ops_asc']} | {r['ops_pto']} | "
                       f"{bold}{pa:.2f}x{bold} |"))
    return claims


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--list", action="store_true",
                    help="print every claim and whether it was found")
    ap.add_argument("--fix-snippets", action="store_true",
                    help="rewrite stale embedded code blocks in the report from "
                         "their kernels (numbers and prose are never touched)")
    args = ap.parse_args()

    if not REPORT.exists():
        print(f"{REPORT.relative_to(ROOT)} does not exist yet")
        return 2
    text = REPORT.read_text()
    bad = 0

    if args.fix_snippets:
        # Cut lengths change when a slide needs to show more of an instruction.
        # The report embeds the same extracts inline, so re-sync them here
        # rather than by hand; prose and numbers are left alone.
        def _sub(m):
            name = m.group(1)
            if name not in SNIPPETS:
                return m.group(0)
            return f"<!-- snippet: {name} -->\n```python\n{resolve(name)[3]}```"
        fixed = re.sub(r"<!-- snippet: (\w+) -->\n```python\n.*?```", _sub,
                       text, flags=re.S)

        # The `*source:*` line above each block names the file and the line
        # range. It moves whenever a cut is widened, so re-derive it too --
        # a stale range is a reader following a pointer to the wrong code.
        def _src(m):
            name = m.group(2)
            if name not in SNIPPETS:
                return m.group(0)
            return f"*source:* `{_cite(name)}`\n\n<!-- snippet: {name} -->"
        fixed = re.sub(r"\*source:\* `[^`]+`\n\n<!-- snippet: (?P<n>\w+) -->",
                       lambda m: _src(type("M", (), {"group": lambda s, i: m.group("n")})()),
                       fixed)
        if fixed != text:
            REPORT.write_text(fixed)
            print(f"re-synced embedded code blocks in "
                  f"{REPORT.relative_to(ROOT)}")
            text = fixed
        else:
            print("embedded code blocks were already current")

    # 1. code blocks
    blocks = re.findall(r"<!-- snippet: (\w+) -->\n```python\n(.*?)```",
                        text, re.S)
    seen = set()
    for name, body in blocks:
        seen.add(name)
        if name not in SNIPPETS:
            print(f"UNKNOWN SNIPPET  {name}")
            bad += 1
            continue
        _, _, _, want = resolve(name)
        if body != want:
            print(f"STALE SNIPPET    {name}  (report differs from the kernel)")
            bad += 1
    print(f"report code blocks: {len(blocks)} embedded, {len(seen)} distinct, "
          f"{len(SNIPPETS) - len(seen)} extracted but unused")

    # 1a. the `*source:*` pointer above each block
    for cited, name in re.findall(
            r"\*source:\* `([^`]+)`\n\n<!-- snippet: (\w+) -->", text):
        if name in SNIPPETS and cited != _cite(name):
            print(f"STALE SOURCE LINE {name}: report says {cited}, "
                  f"now {_cite(name)}")
            bad += 1

    # 1b. the deck: every fenced block must be a known extract
    if SLIDES.exists():
        known = {resolve(n)[3] for n in SNIPPETS}
        deck = re.findall(r"```python\n(.*?)```", SLIDES.read_text(), re.S)
        unknown = [b for b in deck if b not in known]
        for b in unknown:
            first = b.splitlines()[0] if b.splitlines() else "(empty)"
            print(f"HAND-WRITTEN CODE IN DECK: {first[:70]}")
        bad += len(unknown)
        print(f"deck code blocks:   {len(deck)} embedded, "
              f"{len(deck) - len(unknown)} matched to an extract")

    # 2 + 3. numbers
    cycles, static = load()
    claims = build_claims(cycles, static)
    missing = []
    for desc, s in claims:
        ok = s in text
        if args.list:
            print(f"  {'ok ' if ok else 'BAD'} {desc:38} {s[:72]}")
        if not ok:
            missing.append((desc, s))
    if missing:
        print(f"\n{len(missing)} claim(s) not found in the report as derived:")
        for desc, s in missing:
            print(f"  {desc}\n    expected to find: {s}")
        bad += len(missing)

    print(f"numbers: {len(claims) - len(missing)}/{len(claims)} claims re-derived "
          f"and found")
    if bad:
        print(f"\nFAIL: {bad} problem(s)")
        return 1
    print("\nevery embedded snippet matches its kernel; every checked number "
          "re-derives from the measured data")
    return 0


if __name__ == "__main__":
    sys.exit(main())
