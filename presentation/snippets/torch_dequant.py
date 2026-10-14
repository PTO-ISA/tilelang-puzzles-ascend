# puzzles/torch/quant/answer/cast_back/01_e4m3_fp32sf.py:31-33
# extracted by tools/make_report_snippets.py -- do not edit
grouped = q.float().view(m, k // group_size, group_size)
out = grouped * sf.unsqueeze(-1)
target = torch.bfloat16 if out_dtype == "bfloat16" else torch.float32
