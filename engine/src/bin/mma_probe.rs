//! SF-fragment layout pin for `mma.sync...mxf4nvf4.block_scale.scale_vec::4X`
//! m16n8k64 (probe-2 methodology, differential form).
//!
//! Probe 2 (probes/src/bin/p2_nvfp4_mma.rs) proved: LSB-first e2m1 nibbles in
//! both operands, A/B/D fragment layouts, scale u32 byte i -> k-sub-block i —
//! but only under LANE-UNIFORM scale operands. The engine's GEMV needs per-row
//! scales (B = weights: per (n, ksub) ue4m3 from the 36-byte container blocks;
//! A = quantized activations: per (m, ksub)). This probe pins which lane's
//! scale register feeds which (row, ksub) pair.
//!
//! Method: uniform 1.0 data everywhere (D = 16 * sum_i sfa(m,i)*sfb(n,i), all
//! dyadic => every delta vs the all-1.0 baseline is exact). Warp 0 re-runs the
//! mma 64 times; run L sets EXACTLY lane L's sf_a (pass A) / sf_b (pass B)
//! bytes to [2.0, 0.5, 1.5, 1.0] (all other lanes 1.0). Any output cell that
//! moves tells us lane L controls (row m, ksub i) resp. (col n, ksub i) —
//! bit-exact, no candidate guessing. Deltas: ksub0 +16, ksub1 -8, ksub2 +8
//! (16 * (value - 1.0) * 1.0 * 16 elements).

use crow_nest_engine::cuda;

const SRC: &str = r#"
extern "C" __global__ void mma_sf_probe(const unsigned int* a_frag,
                                        const unsigned int* b_frag,
                                        const unsigned int* sfa_lanes,
                                        const unsigned int* sfb_lanes,
                                        float* d_out) {
    const unsigned int a0 = a_frag[threadIdx.x * 4 + 0];
    const unsigned int a1 = a_frag[threadIdx.x * 4 + 1];
    const unsigned int a2 = a_frag[threadIdx.x * 4 + 2];
    const unsigned int a3 = a_frag[threadIdx.x * 4 + 3];
    const unsigned int b0 = b_frag[threadIdx.x * 2 + 0];
    const unsigned int b1 = b_frag[threadIdx.x * 2 + 1];
    const unsigned int sa = sfa_lanes[threadIdx.x];
    const unsigned int sb = sfb_lanes[threadIdx.x];

    float d0 = 0.0f, d1 = 0.0f, d2 = 0.0f, d3 = 0.0f;
    asm volatile(
        "mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 "
        "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, "
        "%10, {0, 0}, %11, {0, 0};"
        : "+f"(d0), "+f"(d1), "+f"(d2), "+f"(d3)
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1), "r"(sa), "r"(sb));

    const int g = threadIdx.x >> 2;
    const int t = threadIdx.x & 3;
    float* out = d_out + blockIdx.x * 128;
    out[g * 8 + 2 * t + 0] = d0;
    out[g * 8 + 2 * t + 1] = d1;
    out[(g + 8) * 8 + 2 * t + 0] = d2;
    out[(g + 8) * 8 + 2 * t + 1] = d3;
}
"#;

#[allow(dead_code)]
fn dec_ue4m3(b: u32) -> f32 {
    let e = (b >> 3) & 0xF;
    let m = b & 7;
    if e == 0 {
        (m as f32) * 1.953125e-3
    } else {
        (1.0 + (m as f32) / 8.0) * (2.0f32).powi(e as i32 - 7)
    }
}

const ONE: u32 = 0x38; // ue4m3 1.0
const TWO: u32 = 0x40; // 2.0
const HALF: u32 = 0x30; // 0.5
const ONE5: u32 = 0x3C; // 1.5

fn enc(v: f32) -> u32 {
    if v == 2.0 {
        TWO
    } else if v == 0.5 {
        HALF
    } else if v == 1.5 {
        ONE5
    } else {
        ONE
    }
}

fn packed(t: &[f32; 4]) -> u32 {
    enc(t[0]) | enc(t[1]) << 8 | enc(t[2]) << 16 | enc(t[3]) << 24
}

fn pack_nib(nibs: &[u32]) -> u32 {
    nibs.iter().enumerate().map(|(i, &n)| n << (4 * i)).sum()
}


fn main() {
    // #13: the logging subscriber of this process. Every library line this bin
    // triggers (`[prefill]`, `[load]`, `[budget]`, `[ple]`, ...) is a `tracing`
    // event now, so without this call they go nowhere. The guard drains the two
    // writer threads when `main` returns; an `exit` below calls `shutdown` first.
    let _log = crow_nest_engine::log::init();
    unsafe {
        let _ctx = cuda::Ctx::init();
        let module = cuda::compile(SRC);
        let f = module.get("mma_sf_probe");

        // uniform 1.0 data: every element = 1.0 (nibble 2)
        let mut a_frag = [0u32; 32 * 4];
        let mut b_frag = [0u32; 32 * 2];
        for lane in 0..32usize {
            for reg in a_frag[lane * 4..lane * 4 + 4].iter_mut() {
                *reg = pack_nib(&[2, 2, 2, 2, 2, 2, 2, 2]);
            }
            for reg in b_frag[lane * 2..lane * 2 + 2].iter_mut() {
                *reg = pack_nib(&[2, 2, 2, 2, 2, 2, 2, 2]);
            }
        }
        let a_dev = cuda::to_dev(&a_frag);
        let b_dev = cuda::to_dev(&b_frag);
        let n_warps = 32usize;

        let run = |sfa: &[u32; 32], sfb: &[u32; 32]| -> Vec<f32> {
            let sfa_dev = cuda::to_dev(sfa);
            let sfb_dev = cuda::to_dev(sfb);
            let out = cuda::alloc_zeroed(n_warps * 128 * 4);
            // every kernel arg is a device address — the driver reads the VALUES
            // from these host slots (gen.rs launch_v pattern)
            let vals: [u64; 5] = [a_dev as u64, b_dev as u64, sfa_dev as u64, sfb_dev as u64, out as u64];
            let mut ptrs: Vec<*mut std::ffi::c_void> = vals
                .iter()
                .map(|v| v as *const u64 as *mut std::ffi::c_void)
                .collect();
            cuda::launch(f, n_warps as u32, 1, 32, 0, &mut ptrs);
            let mut f1 = sfa_dev;
            let mut f2 = sfb_dev;
            cuda::free_dev(&mut f1);
            cuda::free_dev(&mut f2);
            let v = cuda::dtoh(out, n_warps * 128);
            let mut o = out;
            cuda::free_dev(&mut o);
            v
        };

        let base = run(&[ONE; 32], &[ONE; 32]);
        let tuple = [2.0f32, 0.5, 1.5, 1.0];
        let expect = [16.0f32 * (tuple[0] - 1.0), 16.0 * (tuple[1] - 1.0), 16.0 * (tuple[2] - 1.0), 0.0];

        // For each lane L: warp 0 runs with sfa[L] (pass A) / sfb[L] (pass B) changed.
        // a_map[m][i] = lanes controlling row m, k-sub-block i
        let mut a_map: [[Vec<u32>; 4]; 16] = Default::default();
        let mut b_map: [[Vec<u32>; 4]; 8] = Default::default();
        for l in 0..32u32 {
            let mut sfa = [ONE; 32];
            sfa[l as usize] = packed(&tuple);
            let ra = run(&sfa, &[ONE; 32]);
            for m in 0..16usize {
                for n in 0..8usize {
                    let d = ra[m * 8 + n] - base[m * 8 + n];
                    for i in 0..4usize {
                        if (d - expect[i]).abs() < 1e-4 {
                            a_map[m][i].push(l);
                        }
                    }
                }
            }
            let mut sfb = [ONE; 32];
            sfb[l as usize] = packed(&tuple);
            let rb = run(&[ONE; 32], &sfb);
            for m in 0..16usize {
                for n in 0..8usize {
                    let d = rb[m * 8 + n] - base[m * 8 + n];
                    for i in 0..4usize {
                        if (d - expect[i]).abs() < 1e-4 {
                            b_map[n][i].push(l);
                        }
                    }
                }
            }
        }

        println!("=== SF-A lane map: row m, ksub i -> lanes ===");
        for m in 0..16usize {
            let cells: Vec<String> = (0..4)
                .map(|i| {
                    let ls: Vec<String> = a_map[m][i].iter().map(|l| format!("L{l:02}")).collect();
                    if ls.is_empty() { "--".into() } else { ls.join(",") }
                })
                .collect();
            println!("  row {m:2}: [{}]", cells.join(" | "));
        }
        println!("=== SF-B lane map: col n, ksub i -> lanes ===");
        for n in 0..8usize {
            let cells: Vec<String> = (0..4)
                .map(|i| {
                    let ls: Vec<String> = b_map[n][i].iter().map(|l| format!("L{l:02}")).collect();
                    if ls.is_empty() { "--".into() } else { ls.join(",") }
                })
                .collect();
            println!("  col {n:2}: [{}]", cells.join(" | "));
        }

        let mut f1 = a_dev;
        let mut f2 = b_dev;
        cuda::free_dev(&mut f1);
        cuda::free_dev(&mut f2);
    }
}
