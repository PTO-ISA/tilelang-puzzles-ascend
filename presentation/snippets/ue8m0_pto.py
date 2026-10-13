# puzzles/pto/quant/answer/cast_back/02_packed_ue8m0.py:52-56
# extracted by tools/make_report_snippets.py -- do not edit
packed = V.vload(sf_ub[strip], size=LANES * 2, stride=1,
                 dist_mode="brc", group=1)
bits = V.vand(V.vshl(V.vinterpret_cast(packed, "uint32"),
                     sf_shift), exp_mask)
scale = V.vinterpret_cast(bits, "float32")
