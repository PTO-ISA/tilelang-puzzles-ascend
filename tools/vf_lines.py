"""Count the non-comment lines of each kernel's T.SimdVF body.

The PTO-vs-ASC argument in the docs rests on PTO needing fewer vector operations
for the same work. This measures that instead of asserting it, so the numbers in
the README are generated rather than remembered.

    python tools/vf_lines.py
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def vf_body_lines(path: Path) -> int:
    """Lines of code inside `with T.SimdVF():` blocks, ignoring comments/blanks."""
    lines = path.read_text().splitlines()
    # skip the module docstring, which also mentions T.SimdVF
    start = 0
    if lines and lines[0].lstrip().startswith('"""'):
        for i in range(1, len(lines)):
            if '"""' in lines[i]:
                start = i + 1
                break
    total, i = 0, start
    while i < len(lines):
        m = re.match(r"^(\s*)with T\.SimdVF\(", lines[i])
        if not m:
            i += 1
            continue
        indent = len(m.group(1))
        i += 1
        while i < len(lines):
            line = lines[i]
            if line.strip() and (len(line) - len(line.lstrip())) <= indent:
                break
            if line.strip() and not line.strip().startswith("#"):
                total += 1
            i += 1
    return total


def main() -> int:
    rows = []
    for kernel in ("cast_back", "per_token", "per_block", "per_channel"):
        for variant in sorted((ROOT / "puzzles/asc/quant/answer" / kernel).glob("[0-9]*.py")):
            pto = ROOT / "puzzles/pto/quant/answer" / kernel / variant.name
            if not pto.exists():
                continue
            a, p = vf_body_lines(variant), vf_body_lines(pto)
            rows.append((f"{kernel}/{variant.stem}", a, p))
    if not rows:
        print("no paired asc/pto variants yet")
        return 0
    w = max(len(r[0]) for r in rows)
    print(f"{'variant'.ljust(w)}  {'ASC':>5} {'PTO':>5}  {'delta':>6}")
    print("-" * (w + 22))
    ta = tp = 0
    for name, a, p in rows:
        ta += a
        tp += p
        print(f"{name.ljust(w)}  {a:5} {p:5}  {p - a:+6}")
    print("-" * (w + 22))
    print(f"{'total'.ljust(w)}  {ta:5} {tp:5}  {tp - ta:+6}")
    if ta:
        print(f"\nPTO vector-body size relative to ASC: {tp / ta:.0%}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
