# puzzles/asc/quant/answer/cast_back/02_packed_ue8m0.py:49-52
# extracted by tools/make_report_snippets.py -- do not edit
packed = S.vld(sf_ub[strip], dist="BRC_B16")
bits = S.vand(S.vshl(T.reinterpret(packed, "uint32x64"),
                     sf_shift), exp_mask)
# An exponent field with a zero mantissa *is* 2^(e-127).
