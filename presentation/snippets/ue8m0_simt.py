# puzzles/simt/quant/answer/cast_back/02_packed_ue8m0.py:42-46
# extracted by tools/make_report_snippets.py -- do not edit
for i in T.Parallel(hidden):
    g = i // group_size
    word = T.Cast("int32", sf_ub[g // PACK_FACTOR])
    byte = (word >> (8 * (g % PACK_FACTOR))) & 0xFF
    o_l[i] = q_l[i] * T.reinterpret("float32", byte << 23)
