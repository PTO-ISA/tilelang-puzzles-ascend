"""Generate every puzzle/ file from the matching answer/ file.

An answer file marks each region a student is meant to write with a sentinel
pair, carrying the hint that should appear in its place:

    # --- BEGIN SOLUTION hint="what to do, in one line"
    <reference implementation>
    # --- END SOLUTION

This script copies the answer file verbatim except that each such region becomes

    # TODO: <hint>
    raise NotImplementedError("<variant>: implement <function>")

so a puzzle and its answer can never drift apart: the docstring, the worked
example, and the tests are literally the same text. Run it after editing any
answer file.

    python tools/make_puzzles.py           # write puzzle/ files
    python tools/make_puzzles.py --check   # verify they are up to date (CI)
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BEGIN = re.compile(r'^(\s*)# --- BEGIN SOLUTION(?:\s+hint="([^"]*)")?\s*$')
END = re.compile(r'^\s*# --- END SOLUTION\s*$')
DEF = re.compile(r'^\s*def\s+(\w+)')


def to_puzzle(text: str, variant: str) -> str:
    out: list[str] = []
    lines = text.splitlines()
    i = 0
    current_def = "this function"
    while i < len(lines):
        line = lines[i]
        m = DEF.match(line)
        if m:
            current_def = m.group(1)
        begin = BEGIN.match(line)
        if not begin:
            out.append(line)
            i += 1
            continue
        indent, hint = begin.group(1), begin.group(2) or "implement this"
        # skip to the matching END
        j = i + 1
        while j < len(lines) and not END.match(lines[j]):
            j += 1
        if j >= len(lines):
            raise SystemExit(f"{variant}: BEGIN SOLUTION at line {i + 1} has no END SOLUTION")
        for n, chunk in enumerate(_wrap(hint, 74 - len(indent))):
            prefix = "# TODO: " if n == 0 else "#       "
            out.append(f"{indent}{prefix}{chunk}")
        out.append(f'{indent}raise NotImplementedError("{variant}: implement {current_def}")')
        i = j + 1
    return "\n".join(out) + "\n"


def _wrap(hint: str, width: int) -> list[str]:
    words, lines, cur = hint.split(), [], ""
    for w in words:
        if cur and len(cur) + 1 + len(w) > width:
            lines.append(cur)
            cur = w
        else:
            cur = f"{cur} {w}".strip()
    if cur:
        lines.append(cur)
    return lines or [hint]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--check", action="store_true", help="fail if any puzzle is stale")
    args = ap.parse_args()

    answers = sorted(ROOT.glob("puzzles/*/quant/answer/*/*.py"))
    if not answers:
        print("no answer files found")
        return 1

    stale, written = [], 0
    for ans in answers:
        rel = ans.relative_to(ROOT)
        variant = f"{rel.parts[1]}/{ans.parent.name}/{ans.stem}"
        puzzle = Path(str(ans).replace("/answer/", "/puzzle/"))
        text = to_puzzle(ans.read_text(), variant)
        if "# TODO:" not in text:
            print(f"warning: {rel} has no BEGIN SOLUTION region")
        if args.check:
            if not puzzle.exists() or puzzle.read_text() != text:
                stale.append(str(puzzle.relative_to(ROOT)))
        else:
            puzzle.parent.mkdir(parents=True, exist_ok=True)
            if not puzzle.exists() or puzzle.read_text() != text:
                puzzle.write_text(text)
                written += 1

    if args.check:
        if stale:
            print("stale puzzle files (re-run tools/make_puzzles.py):")
            for s in stale:
                print(f"  {s}")
            return 1
        print(f"all {len(answers)} puzzle files up to date")
        return 0
    print(f"{written} written / {len(answers)} answer files")
    return 0


if __name__ == "__main__":
    sys.exit(main())
