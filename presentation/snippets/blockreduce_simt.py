# puzzles/simt/quant/answer/per_block/01_raw_32x32.py:56-63
# extracted by tools/make_report_snippets.py -- do not edit
# level 1: one thread per row of the tile
for t in T.Parallel(bm):
    amax = T.alloc_var(T.float32)
    amax = E4M3_CLAMP_MIN
    for j in T.serial(bk):
        amax = T.max(amax,
                     T.abs(T.Cast("float32", x_ub[t, j])))
    row_ub[t] = amax
