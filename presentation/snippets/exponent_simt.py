# puzzles/simt/quant/answer/per_token/02_round_sf.py:54-57
# extracted by tools/make_report_snippets.py -- do not edit
bits = T.reinterpret("int32", amax / E4M3_MAX)
biased = ((bits - 1) >> 23) + 1
sf_l[g] = T.reinterpret("float32", biased << 23)
inv_l[g] = T.reinterpret("float32", (254 - biased) << 23)
