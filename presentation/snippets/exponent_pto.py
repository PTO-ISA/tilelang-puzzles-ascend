# puzzles/pto/quant/answer/per_token/02_round_sf.py:35-44
# extracted by tools/make_report_snippets.py -- do not edit
one = V.vbrc(T.uint32(1), size=lanes)
shift = V.vbrc(T.uint32(23), size=lanes)
bits = V.vinterpret_cast(
    V.vmul(clamped, V.vbrc(T.float32(inv_qmax), size=lanes), mask), "uint32")
# ((bits - 1) >> 23) + 1  ==  ceil(log2(v)) + 127
biased = V.vadd(V.vshr(V.vsub(bits, one), shift), one)
sf = V.vinterpret_cast(V.vshl(biased, shift), "float32")
# 254 - biased == 127 - ceil_exp: the negated exponent, still biased.
inv = V.vinterpret_cast(
    V.vshl(V.vsub(V.vbrc(T.uint32(254), size=lanes), biased), shift), "float32")
