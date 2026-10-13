"""Turn Marp-dialect slides.md into something pandoc's beamer writer accepts.

    python tools/slides_strip_marp.py slides.md body.md

`presentation/slides.md` is authored in Marp dialect so that a later
`marp slides.md --pdf --pptx` needs no edits. Pandoc cuts slides at headings
instead of at `---`, so this does two things and nothing else:

* drops the Marp YAML front matter (pandoc would render it as a title block);
* drops the body-level `---` separators, which pandoc would otherwise turn into
  horizontal rules and stray empty frames.

Every slide in slides.md starts with an `##` heading, so `--slide-level=2`
reproduces exactly the breaks the `---` markers describe. Keeping this transform
this small is deliberate: the Marp file stays the single source.
"""

from __future__ import annotations

import sys
from pathlib import Path


def strip(text: str) -> str:
    lines = text.splitlines()
    out: list[str] = []
    i = 0
    # front matter, if present
    if lines and lines[0].strip() == "---":
        i = 1
        while i < len(lines) and lines[i].strip() != "---":
            i += 1
        i += 1
    for ln in lines[i:]:
        if ln.strip() == "---":
            continue                 # a Marp slide break; pandoc breaks at ##
        out.append(ln)
    return "\n".join(out).strip() + "\n"


def main() -> int:
    if len(sys.argv) != 3:
        print(__doc__)
        return 2
    src, dst = Path(sys.argv[1]), Path(sys.argv[2])
    dst.write_text(strip(src.read_text()))
    n = sum(1 for ln in dst.read_text().splitlines() if ln.startswith("## "))
    print(f"{src.name} -> {dst.name}: {n} slides")
    return 0


if __name__ == "__main__":
    sys.exit(main())
