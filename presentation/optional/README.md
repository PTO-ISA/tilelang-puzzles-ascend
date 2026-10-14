# Marp build (secondary)

`quant_kernel_style_compare_marp.pdf` is the same deck rendered by Marp CLI
instead of pandoc + Beamer, from the same `../slides.md`. It is kept for
comparison; the Beamer PDF one directory up is the deliverable.

```bash
make -C .. marp        # needs node >= 18 and a Chrome binary
```

## Why it is secondary

Two layout properties are not reachable from the front-matter CSS in
`../slides.md`, and both cost usable slide area:

- **Content is centred vertically.** Beamer top-aligns. Setting
  `justify-content`, `display: block` or `align-content` on `section` does not
  move it — measured by comparing the heading's y-position across all three,
  which is identical in each case.
- **A wide right margin on every page**, which the 52px `section` padding does
  not account for.

Marp also does not shrink content to fit a slide. The CSS in the front matter
(font sizes, padding, table and code-block sizing) exists to keep the dense
slides inside the frame; without it the code slides clip mid-block and the
13-row table loses four rows.

## What it is good for

One Markdown source produces both decks, so the Marp path stays available for
PPTX (`marp ../slides.md --pptx`) and for a self-hosted HTML deck, neither of
which the Beamer build offers.
