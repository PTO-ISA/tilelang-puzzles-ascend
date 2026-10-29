# puzzles/pto/quant/answer/cast_back/01_e4m3_fp32sf.py:60-64
# extracted by tools/make_report_snippets.py -- do not edit
slots = V.vload(sf_ub[group], size=groups_per_strip,
                stride=1, group=groups_per_strip)
scale = V.vbrc(slots, size=LANES, group=groups_per_strip)

scaled = V.vmul(values, scale, mask)
