# puzzles/pto/quant/answer/cast_back/01_e4m3_fp32sf.py:60-61
# extracted by tools/make_report_snippets.py -- do not edit
scale = V.vload(sf_ub[group], size=LANES, stride=1,
                dist_mode="brc", group=groups_per_strip)
