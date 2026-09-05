"""Oracle parity report over a decode-parity directory tree (#11, 2026-09-05): <dir>/tf298 (prefill path + ref-logits.f32 from oracle/ref_engine_logits.py with CROW_PARITY_DIR), <dir>/prompt235 (prompt-only parity), optional <dir>/R1_prod_D (decode path).
Usage (repo root, oracle venv): .venv-oracle/Scripts/python.exe tools/oracle_parity_report.py decode_out/oracle-t2w"""
import json, sys, os
import numpy as np
from transformers import AutoTokenizer

D = sys.argv[1]
V = 248320
tok = AutoTokenizer.from_pretrained("models/Qwen3.8-Flash-Next-original")
seq = json.load(open(os.path.join(D, "tf298", "gen-sequence.json")))
rows = seq["rows"]; plen = 235
ids = seq["all_ids"]
gpu = np.fromfile(os.path.join(D, "tf298", "gpu-logits.f32"), dtype=np.float32).reshape(rows, V)
ref = np.fromfile(os.path.join(D, "tf298", "ref-logits.f32"), dtype=np.float32).reshape(rows, V)

def top(row, k=3):
    idx = np.argsort(row)[::-1][:k]
    return [(int(i), repr(tok.decode([int(i)])), round(float(row[i]), 2)) for i in idx]

def hit(a, b):
    return int(np.argmax(a)) == int(np.argmax(b))

pre = [hit(gpu[t], ref[t]) for t in range(plen)]
gen = [hit(gpu[t], ref[t]) for t in range(plen, rows)]
print(f"argmax engine==oracle: prompt rows {sum(pre)}/{len(pre)}, generated rows {sum(gen)}/{len(gen)}")
# oracle's own greedy continuation vs the engine trace (what the f32 model wants at each generated position)
print("\nper generated position: pos | next token in trace | engine top-3 | oracle top-3 | oracle margin")
first_dev = None
for t in range(plen - 1, rows):
    nxt = ids[t + 1] if t + 1 < len(ids) else None
    ra = int(np.argmax(ref[t])); ga = int(np.argmax(gpu[t]))
    srt = np.sort(ref[t]); margin = srt[-1] - srt[-2]
    flag = "" if ra == ga else "  <-- DIFF"
    if first_dev is None and ra != ga:
        first_dev = t
    print(f"{t:4d} | {nxt} {tok.decode([nxt])!r} | {top(gpu[t])} | {top(ref[t])} | {margin:.3f}{flag}")
print("\nfirst argmax deviation at row", first_dev)
# prefill on 235 vs prefill on 298: same rows 0..234 must agree (deterministic chunk <512), and its 4 decode
# rows 235..238 (decode path) vs the tf298 prefill rows 235..238 (prefill path) vs oracle
seqA = json.load(open(os.path.join(D, "prompt235", "gen-sequence.json")))
gA = np.fromfile(os.path.join(D, "prompt235", "gpu-logits.f32"), dtype=np.float32).reshape(seqA["rows"], V)
d_prompt = np.abs(gA[:plen] - gpu[:plen]).max()
print(f"\nprompt rows 0..234: prefill(235) vs prefill(298) max_abs delta {d_prompt:.3e}")
for t in range(plen, plen + 4):
    print(f"row {t}: decode-path top-3 {top(gA[t])} | prefill-path top-3 {top(gpu[t])} | oracle top-3 {top(ref[t])} | dec-vs-pre max_abs {np.abs(gA[t]-gpu[t]).max():.3e}")
# logit-delta statistics by region
for name, sl in (("prompt", slice(0, plen)), ("generated", slice(plen, rows))):
    d = np.abs(gpu[sl] - ref[sl]).max(axis=1)
    print(f"{name}: max_abs per row median {np.median(d):.2f}, max {d.max():.2f}")

# decode path (R1: prefill 235 + teacher-forced decode) vs oracle
R = os.path.join(D, "R1_prod_D")
if os.path.exists(R):
    sr = json.load(open(os.path.join(R, "gen-sequence.json")))
    gr = np.fromfile(os.path.join(R, "gpu-logits.f32"), dtype=np.float32).reshape(sr["rows"], V)
    hd = [hit(gr[t], ref[t]) for t in range(plen, rows)]
    hp = [hit(gpu[t], ref[t]) for t in range(plen, rows)]
    print(f"\ngenerated rows {plen}..{rows-1}: argmax==oracle  prefill-path {sum(hp)}/{len(hp)}  decode-path {sum(hd)}/{len(hd)}")
    dp = np.abs(gpu[plen:rows] - ref[plen:rows]).max(axis=1); dd = np.abs(gr[plen:rows] - ref[plen:rows]).max(axis=1)
    print(f"max_abs vs oracle, generated rows: prefill-path median {np.median(dp):.2f} max {dp.max():.2f}; decode-path median {np.median(dd):.2f} max {dd.max():.2f}")
    print("\nrows where the decode path disagrees with the oracle but the prefill path agrees (pos, oracle top, oracle margin, decode top):")
    for t in range(plen, rows):
        ra, pa, da = int(ref[t].argmax()), int(gpu[t].argmax()), int(gr[t].argmax())
        if da != ra and pa == ra:
            srt = np.sort(ref[t]); print(f"  {t}: oracle {ra} {tok.decode([ra])!r} margin {srt[-1]-srt[-2]:.2f} | decode {da} {tok.decode([da])!r}")
    # oracle's own greedy continuation: does the f32 model want the newline/EOS collapse?
    print("\noracle greedy at the first generated positions (what f32 would emit given the same prefix):")
    for t in range(plen-1, plen+12):
        print(f"  row {t}: fed-next {ids[t+1]} {tok.decode([ids[t+1]])!r} | oracle top {top(ref[t],3)}")
