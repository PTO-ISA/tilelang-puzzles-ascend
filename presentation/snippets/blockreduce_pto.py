# puzzles/pto/quant/answer/per_block/01_raw_32x32.py:53-56
# extracted by tools/make_report_snippets.py -- do not edit
for chunk in T.serial(num_chunks):
    v = V.vabs(V.vcvt(V.vload(flat_x[chunk * LANES],
                              size=LANES), "float32"), mask)
    V.vstore(V.vcmax(v, mask, group=1), run_ub[chunk])
