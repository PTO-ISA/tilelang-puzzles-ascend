# puzzles/asc/quant/answer/cast_back/01_e4m3_fp32sf.py:66-68
# extracted by tools/make_report_snippets.py -- do not edit
lo = S.vld(sf_ub[group], dist="BRC_B32")
hi = S.vld(sf_ub[group + 1], dist="BRC_B32")
scale = S.vsel(lo, hi, mask_low)
