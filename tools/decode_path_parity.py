"""Decode-path rows (decode parity with CROW_PARITY_PREFILL=n) vs prefill-path rows (plain decode parity) over the SAME ids: per row max_abs, argmax, margins (#11, 2026-09-05).
Usage (repo root): python tools/decode_path_parity.py <dir_D> <dir_P> [prefill_len=235]"""
V = 248320
D, P = sys.argv[1], sys.argv[2]
n0 = int(sys.argv[3]) if len(sys.argv) > 3 else 235
sd = json.load(open(os.path.join(D, "gen-sequence.json"))); sp = json.load(open(os.path.join(P, "gen-sequence.json")))
gd = np.fromfile(os.path.join(D, "gpu-logits.f32"), dtype=np.float32).reshape(sd["rows"], V)
gp = np.fromfile(os.path.join(P, "gpu-logits.f32"), dtype=np.float32).reshape(sp["rows"], V)
n = min(sd["rows"], sp["rows"], len(sd["all_ids"]), len(sp["all_ids"]))
same = [sd["all_ids"][i] == sp["all_ids"][i] for i in range(n)]
last = n if all(same) else same.index(False)
print(f"D rows {sd['rows']} P rows {sp['rows']}; processed ids identical for the first {last} rows")
print(f"prompt rows 0..{n0-1}: max_abs D-P {np.abs(gd[:n0]-gp[:n0]).max():.3e}")
print("row | step | pos%4 | max_abs | argmax D | argmax P | eq | P margin | D margin")
by_mod = {0: [], 1: [], 2: [], 3: []}
for t in range(n0, last):
    d = np.abs(gd[t] - gp[t]).max()
    ad, ap = int(gd[t].argmax()), int(gp[t].argmax())
    sp_ = np.sort(gp[t]); sd_ = np.sort(gd[t])
    by_mod[t % 4].append(d)
    print(f"{t:4d} | {t-n0+1:3d} | {t%4} | {d:7.3f} | {ad:6d} | {ap:6d} | {'=' if ad==ap else 'X'} | {sp_[-1]-sp_[-2]:6.3f} | {sd_[-1]-sd_[-2]:6.3f}")
print("median max_abs by pos%4:", {k: round(float(np.median(v)), 3) for k, v in by_mod.items() if v})
print("trace D:", sd.get("tf_trace"))
