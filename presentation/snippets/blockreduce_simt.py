# puzzles/simt/quant/answer/per_block/01_raw_32x32.py:65-67
# extracted by tools/make_report_snippets.py -- do not edit
for i, j in T.Parallel(bm, bk):
    flat_l[0, i * bk + j] = T.Cast("float32", x_ub[i, j])
T.reduce_absmax(flat_l, tile_l, dim=1)
