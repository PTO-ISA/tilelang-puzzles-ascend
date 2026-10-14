"""Verify the VF line counts quoted in the variant pages against the source.

    python tools/check_doc_numbers.py

Each variant page's SIMT section quotes a triple in one fixed phrasing:

    <n> lines against ASC's <a> and PTO's <p>

Those numbers come from `tools/vf_lines.py`, which counts non-comment lines
inside the VF region. Prose goes stale silently when a kernel is edited, so this
re-derives every triple from the files and fails on any mismatch.

The phrasing is load-bearing: if you reword one of those sentences, either keep
the pattern or this check stops seeing it. `--list` shows what it matched, so a
sentence that quietly stopped being checked is visible.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

from vf_lines import TIERS, vf_body                           # noqa: E402

# "16 lines against ASC's 21 and PTO's 22", with the numbers optionally spelled
# in words where the prose reads better that way.
WORDS = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6,
         "seven": 7, "eight": 8, "nine": 9, "ten": 10, "eleven": 11,
         "twelve": 12, "thirteen": 13, "fourteen": 14, "fifteen": 15,
         "sixteen": 16, "seventeen": 17, "eighteen": 18, "nineteen": 19,
         "twenty": 20}
NUM = r"(\d+|" + "|".join(WORDS) + r")"
# Whitespace-tolerant, because the prose wraps mid-sentence.
CLAIM = re.compile(NUM + r"\s+lines against ASC's\s+" + NUM
                   + r"\s+and PTO's\s+" + NUM, re.I)


def val(tok: str) -> int:
    return int(tok) if tok.isdigit() else WORDS[tok.lower()]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--list", action="store_true",
                    help="print every claim found and its verified value")
    args = ap.parse_args()

    pages = sorted((ROOT / "doc/quant").glob("*/[0-9]*.md"))
    bad, found = [], 0
    for page in pages:
        text = page.read_text()
        rel = page.relative_to(ROOT)
        kernel, stem = page.parent.name, page.stem
        want = {}
        for tier, (_, opener, op_re) in TIERS.items():
            src = ROOT / "puzzles" / tier / "quant/answer" / kernel / f"{stem}.py"
            want[tier] = vf_body(src, opener, op_re)[0] if src.exists() else None
        for m in CLAIM.finditer(text):
            found += 1
            got = (val(m.group(1)), val(m.group(2)), val(m.group(3)))
            exp = (want["simt"], want["asc"], want["pto"])
            mark = "ok " if got == exp else "BAD"
            if got != exp:
                bad.append((rel, m.group(0), got, exp))
            if args.list:
                print(f"  {mark} {rel}: {m.group(0)!r} -> SIMT/ASC/PTO {exp}")

    print(f"\nchecked {found} line-count claims across {len(pages)} variant pages")
    missing = [p.relative_to(ROOT) for p in pages
               if "## SIMT" in p.read_text() and not CLAIM.search(p.read_text())]
    if missing:
        print(f"{len(missing)} page(s) have a SIMT section with no checkable "
              f"line-count claim:")
        for m in missing:
            print(f"  - {m}")
    if bad:
        print(f"\n{len(bad)} STALE claim(s):")
        for rel, txt, got, exp in bad:
            print(f"  {rel}\n    says  {txt!r}  (SIMT, ASC, PTO) = {got}\n"
                  f"    files {exp}")
        return 1
    print("every quoted triple matches the source")
    return 0


if __name__ == "__main__":
    sys.exit(main())
