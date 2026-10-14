# puzzles/simt/quant/answer/per_token/05_col_major_sf.py:38-39
# extracted by tools/make_report_snippets.py -- do not edit
# Already transposed: one column per token of the current block.
sf_out_ub = T.alloc_shared((num_groups, token_block), T.float32)
