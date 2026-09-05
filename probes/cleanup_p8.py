# One-shot: remove debug instrumentation from p8_full_layer.rs (probe should
# stay minimal like p6/p7). Run once, then delete this script.
import io

p = 'src/bin/p8_full_layer.rs'
s = io.open(p, encoding='utf-8').read()


def cut(marker_start, marker_end, repl=''):
    global s
    i = s.index(marker_start)
    j = s.index(marker_end, i)
    s = s[:i] + repl + s[j:]


# 1. hc debug closure
cut("                let dbg_in = |t2: &str",
    "                let mut normed = alloc_zeroed(T * HCT * 4);",
    "                let mut normed = alloc_zeroed(T * HCT * 4);")

# 2. the three dbg_in calls
for tag in ['hc-normed', 'hc-mixw', 'hc-injw']:
    needle = 'dbg_in(&format!("' + tag
    line = [l for l in s.splitlines() if needle in l]
    assert len(line) == 1, tag
    s = s.replace(line[0] + "\n", "", 1)

# 3. tag param off again
s = s.replace("let mut run_hc = move |tag: &str,\n                      mut norm: CUdeviceptr,",
              "let mut run_hc = move |mut norm: CUdeviceptr,", 1)
s = s.replace('run_hc("a", w_ahc_norm', 'run_hc(w_ahc_norm', 1)
s = s.replace('run_hc("m", w_mhc_norm', 'run_hc(w_mhc_norm', 1)

# 4. staged debug block: dbg closure definition
cut("        // staged debug: intermediates for the python stage-wise comparator\n",
    "        let mut gdn_out = run_gdn(mixed_a);", "")
for call in ['dbg("gpu-mixed-a", mixed_a, T * H);',
             'dbg("gpu-gdn-out", gdn_out, T * H);',
             'dbg("gpu-x1", x1, T * HCT);',
             'dbg("gpu-mixed-m", mixed_m, T * H);']:
    line = [l for l in s.splitlines() if call in l]
    assert len(line) == 1, call
    s = s.replace(line[0] + "\n", "", 1)
cut("        let dbg = |tag: &str, src: CUdeviceptr, n: usize| {", "        };\n", "")

# 5. routing/logits dump block
cut("        {\n            let mut idb: Vec<u8> = Vec::new();",
    "        // shared expert per token", "")

# 6. lg2_bytes helper
cut("unsafe fn lg2_bytes(d: &CUdeviceptr, n: usize) -> Vec<u8> {", "fn main() {", "")

# 7. shared dump machinery (dtoh block at loop tail + dump blocks)
cut("            let sv = dtoh(sdown, H);",
    "        let mut h1 = alloc_zeroed(2 * INTER * 4);", "")
s = s.replace("""        let mut moe_out = alloc_zeroed(T * H * 4);
        let mut shared_acc = vec![0f32; T * H];
        let mut all_sg1: Vec<f32> = Vec::new();
        let mut all_su1: Vec<f32> = Vec::new();
        let mut all_sdown: Vec<f32> = Vec::new();
        let mut all_sgv: Vec<f32> = Vec::new();""",
    """        let mut moe_out = alloc_zeroed(T * H * 4);""", 1)

# 8. moe dbg call
line = [l for l in s.splitlines() if 'dbg("gpu-moe"' in l]
assert len(line) == 1
s = s.replace(line[0] + "\n", "", 1)

io.open(p, 'w', encoding='utf-8').write(s)
print("cleaned")
