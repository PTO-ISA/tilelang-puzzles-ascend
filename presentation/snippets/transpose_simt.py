# puzzles/simt/quant/answer/per_token/05_col_major_sf.py:62-63
# extracted by tools/make_report_snippets.py -- do not edit
# The whole column-major config: a different index.
sf_out_ub[g, sub] = T.reinterpret("float32", biased << 23)
