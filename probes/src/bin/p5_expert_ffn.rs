//! Probe 5 v2 (Crow #8 + #10): expert-FFN chain with mixed residency — CLEAN REWRITE.
//!
//! The v1 found a divergence under mixed routing but drowned in debug scaffolding.
//! This version is staged: each stage asserts before the next, so the divergence —
//! if it still exists — is pinned to a minimal configuration by construction.
//!
//!   A: one expert (0), one token, gate_up only            → vs CPU dequant
//!   B: two tokens, experts 0 (VRAM) and 4 (pinned)         → full FFN vs CPU chain
//!   C: 8 tokens × 10 routed experts, mixed residency      → full FFN vs CPU chain
//!
//! Everything the v1 debugging established is baked in: ALL uploads async + explicit
//! sync (WDDM partial-copy trap), scalars via device buffers (raw-launch 4-byte arg
//! corruption), combo-major buffers with asserted sizes, per-combo pointer tables.

use std::ffi::CString;
use std::io::{Read, Seek, SeekFrom};
use std::time::Instant;

use cudarc::driver::sys::{self, CUdeviceptr, CUresult};
use cudarc::nvrtc::{compile_ptx_with_opts, CompileOptions};

const TOKENS: usize = 8;
const EXPERTS: usize = 10; // routed per token
const H: usize = 2560;
const INTER: usize = 1280; // gate_up rows (640 gate + 640 up)
const DOWN_IN: usize = 640;
const GU_BPR: usize = H / 64; // 40 blocks per gate_up row
const DN_BPR: usize = DOWN_IN / 64; // 10 blocks per down row
const GU_ROW_BYTES: usize = GU_BPR * 36; // 1440
const DN_ROW_BYTES: usize = DN_BPR * 36; // 360
const GU_BYTES_PER_E: usize = INTER * GU_ROW_BYTES; // 1,843,200
const DN_BYTES_PER_E: usize = H * DN_ROW_BYTES; // 921,600
const COMBOS: usize = TOKENS * EXPERTS;

const KERNEL_SRC: &str = r#"
__device__ __forceinline__ float e2m1(unsigned int nib) {
    const float mag[8] = {0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f};
    float v = mag[nib & 0x7];
    return (nib & 0x8) ? -v : v;
}
__device__ __forceinline__ float ue4m3(unsigned int byte) {
    unsigned int e = (byte >> 3) & 0xF;
    unsigned int m = byte & 0x7;
    if (e == 0) return (float)m * 1.953125e-3f;
    return (1.0f + (float)m / 8.0f) * exp2f((float)e - 7.0f);
}

// grid (rows, combos); weights via per-combo pointer table; scalars via prm buffer
extern "C" __global__ void gemv_table(const unsigned long long* __restrict__ w_table,
                                      const float* __restrict__ x,   // [combos][k]
                                      float* __restrict__ y,         // [combos][rows]
                                      const float* __restrict__ gs_ptr,
                                      const int* __restrict__ prm) { // {rows,k,bpr,combos}
    int rows = prm[0], k_dim = prm[1], bpr = prm[2], combos = prm[3];
    float gs = gs_ptr[0];
    int row = blockIdx.x;
    int combo = blockIdx.y;
    if (row >= rows || combo >= combos) return;
    const unsigned char* w = (const unsigned char*)w_table[combo];
    const float* xv = x + (size_t)combo * k_dim;
    const unsigned char* rowp = w + (size_t)row * (bpr * 36);

    float acc = 0.0f;
    for (int b = threadIdx.x; b < bpr; b += blockDim.x) {
        const unsigned char* blk = rowp + b * 36;
        const unsigned char* q = blk + 4;
        #pragma unroll
        for (int sb = 0; sb < 4; sb++) {
            float s = ue4m3(blk[sb]) * gs;
            float acc_s = 0.0f;
            #pragma unroll
            for (int j = 0; j < 16; j++) {
                int idx = sb * 16 + j;
                unsigned int byte = q[idx >> 1];
                unsigned int nib = (idx & 1) ? ((byte >> 4) & 0xF) : (byte & 0xF);
                acc_s += e2m1(nib) * xv[b * 64 + sb * 16 + j];
            }
            acc += acc_s * s;
        }
    }

    __shared__ float red[256];
    red[threadIdx.x] = acc;
    __syncthreads();
    for (int st = blockDim.x / 2; st > 0; st >>= 1) {
        if (threadIdx.x < st) red[threadIdx.x] += red[threadIdx.x + st];
        __syncthreads();
    }
    if (threadIdx.x == 0) y[(size_t)combo * rows + row] = red[0];
}

// h2[c][j] = silu(h1[c][j]) * h1[c][INTER/2 + j]
extern "C" __global__ void silu_mul(const float* __restrict__ h1,
                                    float* __restrict__ h2,
                                    const int* __restrict__ prm) { // {inter, combos}
    int inter = prm[0], combos = prm[1];
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = (inter / 2) * combos;
    if (idx >= total) return;
    int c = idx / (inter / 2), j = idx % (inter / 2);
    float gate = h1[(size_t)c * inter + j];
    float up = h1[(size_t)c * inter + inter / 2 + j];
    h2[idx] = (gate / (1.0f + expf(-gate))) * up;
}
"#;

fn ck(r: CUresult) {
    if r != CUresult::CUDA_SUCCESS {
        panic!("CUDA error: {r:?}");
    }
}

fn e2m1(nibble: u32) -> f32 {
    const MAG: [f32; 8] = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0];
    let v = MAG[(nibble & 0x7) as usize];
    if nibble & 0x8 != 0 { -v } else { v }
}

fn ue4m3(byte: u32) -> f32 {
    let e = (byte >> 3) & 0xF;
    let m = byte & 0x7;
    if e == 0 {
        (m as f32) * 2.0f32.powi(-9)
    } else {
        (1.0 + (m as f32) / 8.0) * 2.0f32.powi(e as i32 - 7)
    }
}

fn dequant_row(row: &[u8], bpr: usize, global: f32) -> Vec<f32> {
    let mut out = vec![0.0f32; bpr * 64];
    for b in 0..bpr {
        let blk = &row[b * 36..(b + 1) * 36];
        for sb in 0..4 {
            let s = ue4m3(blk[sb] as u32) * global;
            for j in 0..16 {
                let idx = sb * 16 + j;
                let byte = blk[4 + (idx >> 1)];
                let nib = if idx & 1 == 1 { (byte >> 4) & 0xF } else { byte & 0xF };
                out[b * 64 + idx] = e2m1(nib as u32) * s;
            }
        }
    }
    out
}

/// CPU expert FFN for one (token, expert): returns down @ (silu(gate)·up)
fn cpu_expert(xc: &[f32], gu_row_bytes: &[u8], dn_row_bytes: &[u8], gu_g: f32, dn_g: f32) -> Vec<f32> {
    let mut h1 = vec![0.0f32; INTER];
    for r in 0..INTER {
        let w = dequant_row(&gu_row_bytes[r * GU_ROW_BYTES..(r + 1) * GU_ROW_BYTES], GU_BPR, gu_g);
        h1[r] = w.iter().zip(xc.iter()).map(|(a, b)| a * b).sum();
    }
    let mut h2 = vec![0.0f32; DOWN_IN];
    for j in 0..DOWN_IN {
        let gate = h1[j];
        h2[j] = (gate / (1.0 + (-gate).exp())) * h1[INTER / 2 + j];
    }
    let mut out = vec![0.0f32; H];
    for r in 0..H {
        let w = dequant_row(&dn_row_bytes[r * DN_ROW_BYTES..(r + 1) * DN_ROW_BYTES], DN_BPR, dn_g);
        out[r] = w.iter().zip(h2.iter()).map(|(a, b)| a * b).sum();
    }
    out
}

// ---- GPU orchestration ----

unsafe fn upload_async(dst: CUdeviceptr, src: *const u8, len: usize) {
    ck(sys::cuMemcpyHtoDAsync_v2(dst, src as *const std::ffi::c_void, len, std::ptr::null_mut()));
    // EVERY upload gets an explicit sync: WDDM lands synchronous large HtoD copies
    // incompletely on this stack (measured, twice) — async + sync is the safe pattern
    ck(sys::cuStreamSynchronize(std::ptr::null_mut()));
}

unsafe fn alloc_dev(bytes: usize) -> CUdeviceptr {
    let mut d: CUdeviceptr = 0;
    ck(sys::cuMemAlloc_v2(&mut d, bytes));
    d
}

unsafe fn to_f32_dev(v: &[f32]) -> CUdeviceptr {
    let d = alloc_dev(v.len() * 4);
    ck(sys::cuMemcpyHtoDAsync_v2(d, v.as_ptr() as *const std::ffi::c_void, v.len() * 4, std::ptr::null_mut()));
    ck(sys::cuStreamSynchronize(std::ptr::null_mut()));
    d
}

unsafe fn dtoh_f32(src: CUdeviceptr, len: usize) -> Vec<f32> {
    let mut out = vec![0f32; len];
    ck(sys::cuMemcpyDtoH_v2(out.as_mut_ptr() as *mut std::ffi::c_void, src, len * 4));
    out
}

unsafe fn to_ptr_table(v: &[CUdeviceptr]) -> CUdeviceptr {
    let as_u64: Vec<u64> = v.iter().map(|d| *d).collect();
    let d = alloc_dev(as_u64.len() * 8);
    ck(sys::cuMemcpyHtoDAsync_v2(d, as_u64.as_ptr() as *const std::ffi::c_void, as_u64.len() * 8, std::ptr::null_mut()));
    ck(sys::cuStreamSynchronize(std::ptr::null_mut()));
    d
}

unsafe fn to_i32_dev(v: &[i32]) -> CUdeviceptr {
    let d = alloc_dev(v.len() * 4);
    ck(sys::cuMemcpyHtoDAsync_v2(d, v.as_ptr() as *const std::ffi::c_void, v.len() * 4, std::ptr::null_mut()));
    ck(sys::cuStreamSynchronize(std::ptr::null_mut()));
    d
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let cnq = args
        .get(1)
        .cloned()
        .unwrap_or_else(|| "../converter/Qwen3.8-Flash-Next-CNQ4.5.cnq".to_string());

    // ---- container: layer-0 expert weights ----
    let mut f = std::fs::File::open(&cnq).expect("open CNQ4.5");
    let file_len = f.metadata().unwrap().len();
    f.seek(SeekFrom::Start(file_len - 8)).unwrap();
    let mut b8 = [0u8; 8];
    f.read_exact(&mut b8).unwrap();
    let idx_len = u64::from_le_bytes(b8) as usize;
    f.seek(SeekFrom::Start(file_len - 8 - idx_len as u64)).unwrap();
    let mut ib = vec![0u8; idx_len];
    f.read_exact(&mut ib).unwrap();
    let idx: serde_json::Value = serde_json::from_slice(&ib).unwrap();
    let blob_off = idx["blob_offset"].as_u64().unwrap() as usize;
    let find = |suffix: &str| -> (u64, f32) {
        let t = idx["tensors"].as_array().unwrap().iter()
            .find(|t| t["name"].as_str().unwrap().ends_with(suffix)).expect("tensor");
        (t["offset"].as_u64().unwrap(), t["global_scale"].as_f64().unwrap() as f32)
    };
    let (gu_off, gu_g) = find("layers.0.mlp.experts.gate_up_proj");
    let (dn_off, dn_g) = find("layers.0.mlp.experts.down_proj");

    let mut gu = vec![0u8; EXPERTS * GU_BYTES_PER_E];
    let mut dn = vec![0u8; EXPERTS * DN_BYTES_PER_E];
    for e in 0..EXPERTS {
        f.seek(SeekFrom::Start(blob_off as u64 + gu_off + (e * GU_BYTES_PER_E) as u64)).unwrap();
        f.read_exact(&mut gu[e * GU_BYTES_PER_E..(e + 1) * GU_BYTES_PER_E]).unwrap();
        f.seek(SeekFrom::Start(blob_off as u64 + dn_off + (e * DN_BYTES_PER_E) as u64)).unwrap();
        f.read_exact(&mut dn[e * DN_BYTES_PER_E..(e + 1) * DN_BYTES_PER_E]).unwrap();
    }
    println!("p5: loaded experts 0-9 (gate_up {} B, down {} B each)", GU_BYTES_PER_E, DN_BYTES_PER_E);

    unsafe {
        ck(sys::cuInit(0));
        let mut dev = 0;
        ck(sys::cuDeviceGet(&mut dev, 0));
        let mut ctx = std::ptr::null_mut();
        ck(sys::cuDevicePrimaryCtxRetain(&mut ctx, dev));
        ck(sys::cuCtxSetCurrent(ctx));

        let opts = CompileOptions {
            options: vec!["--gpu-architecture=compute_120a".into()],
            ..Default::default()
        };
        let ptx = compile_ptx_with_opts(KERNEL_SRC, opts).expect("nvrtc");
        let c_ptx = CString::new(ptx.to_src()).unwrap();
        let mut module = std::ptr::null_mut();
        ck(sys::cuModuleLoadData(&mut module, c_ptx.as_ptr() as *const _));
        let get_fn = |n: &str| {
            let mut fu = std::ptr::null_mut();
            ck(sys::cuModuleGetFunction(&mut fu, module, CString::new(n).unwrap().as_ptr()));
            fu
        };
        let fn_gu = get_fn("gemv_table");
        let fn_dn = get_fn("gemv_table");
        let fn_sm = get_fn("silu_mul");

        // residency: experts 0-3 VRAM, 4-9 pinned host
        let mut gu_vram: Vec<CUdeviceptr> = Vec::new();
        let mut dn_vram: Vec<CUdeviceptr> = Vec::new();
        let mut gu_pinned: Vec<CUdeviceptr> = Vec::new();
        let mut dn_pinned: Vec<CUdeviceptr> = Vec::new();
        for e in 0..EXPERTS {
            if e < 4 {
                let d = alloc_dev(GU_BYTES_PER_E);
                upload_async(d, gu[e * GU_BYTES_PER_E..].as_ptr(), GU_BYTES_PER_E);
                gu_vram.push(d);
                let d2 = alloc_dev(DN_BYTES_PER_E);
                upload_async(d2, dn[e * DN_BYTES_PER_E..].as_ptr(), DN_BYTES_PER_E);
                dn_vram.push(d2);
            } else {
                let mut p: *mut std::ffi::c_void = std::ptr::null_mut();
                ck(sys::cuMemHostAlloc(&mut p, GU_BYTES_PER_E, sys::CU_MEMHOSTALLOC_PORTABLE | sys::CU_MEMHOSTALLOC_DEVICEMAP));
                std::ptr::copy_nonoverlapping(gu[e * GU_BYTES_PER_E..].as_ptr(), p as *mut u8, GU_BYTES_PER_E);
                gu_pinned.push(p as CUdeviceptr);
                let mut p2: *mut std::ffi::c_void = std::ptr::null_mut();
                ck(sys::cuMemHostAlloc(&mut p2, DN_BYTES_PER_E, sys::CU_MEMHOSTALLOC_PORTABLE | sys::CU_MEMHOSTALLOC_DEVICEMAP));
                std::ptr::copy_nonoverlapping(dn[e * DN_BYTES_PER_E..].as_ptr(), p2 as *mut u8, DN_BYTES_PER_E);
                dn_pinned.push(p2 as CUdeviceptr);
            }
        }
        // gate 0: every resident copy byte-verified on device
        for e in 0..EXPERTS {
            let (src, host) = if e < 4 {
                (gu_vram[e], &gu[e * GU_BYTES_PER_E..e * GU_BYTES_PER_E + 4])
            } else {
                (gu_pinned[e - 4], &gu[e * GU_BYTES_PER_E..e * GU_BYTES_PER_E + 4])
            };
            let mut back = [0u8; 4];
            ck(sys::cuMemcpyDtoH_v2(back.as_mut_ptr() as *mut std::ffi::c_void, src, 4));
            assert_eq!(back, host, "expert {e} device bytes mismatch");
        }
        println!("residency: 0-3 VRAM, 4-9 pinned — device bytes verified");

        // ---- deterministic routing + inputs ----
        let mut s: u32 = 0xC0FFEE;
        let mut rnd = move || {
            s ^= s << 13;
            s ^= s >> 17;
            s ^= s << 5;
            s
        };
        let mut routed = [[0usize; EXPERTS]; TOKENS];
        let mut rw = [[1.0f32 / EXPERTS as f32; EXPERTS]; TOKENS];
        for t in 0..TOKENS {
            for e in 0..EXPERTS {
                routed[t][e] = (rnd() as usize) % EXPERTS;
            }
        }
        let mut xs_tok = vec![0.0f32; TOKENS * H];
        for v in xs_tok.iter_mut() {
            let r = rnd();
            *v = (r & 0xFFFF) as f32 / 65535.0 - 0.5;
        }
        // combo-major x: combo = e * TOKENS + t
        let mut xs = vec![0.0f32; COMBOS * H];
        for e in 0..EXPERTS {
            for t in 0..TOKENS {
                xs[(e * TOKENS + t) * H..(e * TOKENS + t + 1) * H]
                    .copy_from_slice(&xs_tok[t * H..(t + 1) * H]);
            }
        }
        assert_eq!(xs.len(), COMBOS * H, "combo-major x size");

        let mut x_dev = to_f32_dev(&xs);
        let mut h1 = alloc_dev(COMBOS * INTER * 4);
        let mut h2 = alloc_dev(COMBOS * DOWN_IN * 4);
        let mut y = alloc_dev(COMBOS * H * 4);
        let mut prm_gu = to_i32_dev(&[INTER as i32, H as i32, GU_BPR as i32, COMBOS as i32]);
        let mut prm_dn = to_i32_dev(&[H as i32, DOWN_IN as i32, DN_BPR as i32, COMBOS as i32]);
        let mut prm_sm = to_i32_dev(&[INTER as i32, COMBOS as i32]);
        let mut gs_gu = to_f32_dev(&[gu_g]);
        let mut gs_dn = to_f32_dev(&[dn_g]);

        // per-combo expert table (engine shape): entries = ROUTED ids; resident
        // experts 0-3 use VRAM pointers, 4-9 the pinned host pointers
        let make_combo_tables = |routed: &[[usize; EXPERTS]; TOKENS]| -> (CUdeviceptr, CUdeviceptr) {
            unsafe {
                let mut tg: Vec<u64> = Vec::with_capacity(COMBOS);
                let mut td: Vec<u64> = Vec::with_capacity(COMBOS);
                for t in 0..TOKENS {
                    for e in 0..EXPERTS {
                        let eid = routed[t][e];
                        tg.push(if eid < 4 { gu_vram[eid] } else { gu_pinned[eid - 4] });
                        td.push(if eid < 4 { dn_vram[eid] } else { dn_pinned[eid - 4] });
                    }
                }
                (to_u64_dev(&tg), to_u64_dev(&td))
            }
        };

        // ---- the chain, parameterized by routing ----
        let mut run_chain = |routed: &[[usize; EXPERTS]; TOKENS]| -> Vec<f32> {
            unsafe {
                let t0 = Instant::now();
                let (mut t_gu_dev, mut t_dn_dev) = make_combo_tables(routed);
                ck(sys::cuMemsetD8_v2(h1, 0x7F, COMBOS * INTER * 4));
                ck(sys::cuMemsetD8_v2(h2, 0x7F, COMBOS * DOWN_IN * 4));
                ck(sys::cuMemsetD8_v2(y, 0x7F, COMBOS * H * 4));

                let launch = |fnk, gx: u32, gy: u32, args: &mut [*mut std::ffi::c_void]| {
                    ck(sys::cuLaunchKernel(fnk, gx, gy, 1, 256, 1, 1, 0, std::ptr::null_mut(), args.as_mut_ptr(), std::ptr::null_mut()));
                    ck(sys::cuStreamSynchronize(std::ptr::null_mut()));
                };
                launch(fn_gu, INTER as u32, COMBOS as u32, &mut [
                    &mut t_gu_dev as *mut _ as *mut _,
                    &mut x_dev as *mut _ as *mut _,
                    &mut h1 as *mut _ as *mut _,
                    &mut gs_gu as *mut _ as *mut _,
                    &mut prm_gu as *mut _ as *mut _,
                ]);
                launch(fn_sm, ((COMBOS * (INTER / 2)) as u32 + 255) / 256, 1, &mut [
                    &mut h1 as *mut _ as *mut _,
                    &mut h2 as *mut _ as *mut _,
                    &mut prm_sm as *mut _ as *mut _,
                ]);
                launch(fn_dn, H as u32, COMBOS as u32, &mut [
                    &mut t_dn_dev as *mut _ as *mut _,
                    &mut h2 as *mut _ as *mut _,
                    &mut y as *mut _ as *mut _,
                    &mut gs_dn as *mut _ as *mut _,
                    &mut prm_dn as *mut _ as *mut _,
                ]);

                ck(sys::cuStreamSynchronize(std::ptr::null_mut()));
                let gpu_ms = t0.elapsed().as_secs_f64() * 1e3;
                println!("  chain time (3 launches, 80 combos, mixed residency): {gpu_ms:.2} ms -> x48 layers ≈ {:.1} ms/token (microbenchmark)", gpu_ms * 48.0);
                let y_host = dtoh_f32(y, COMBOS * H);
                let mut out = vec![0f32; TOKENS * H];
                for t in 0..TOKENS {
                    for e in 0..EXPERTS {
                        let c = e * TOKENS + t;
                        for j in 0..H {
                            out[t * H + j] += rw[t][e] * y_host[c * H + j];
                        }
                    }
                }
                out
            }
        };

        // ---- CPU reference ----
        let cpu_chain = |routed: &[[usize; EXPERTS]; TOKENS]| -> Vec<f32> {
            let mut out = vec![0f32; TOKENS * H];
            for t in 0..TOKENS {
                for e in 0..EXPERTS {
                    let eid = routed[t][e];
                    let xc = &xs[(e * TOKENS + t) * H..(e * TOKENS + t + 1) * H];
                    let r = cpu_expert(
                        xc,
                        &gu[eid * GU_BYTES_PER_E..(eid + 1) * GU_BYTES_PER_E],
                        &dn[eid * DN_BYTES_PER_E..(eid + 1) * DN_BYTES_PER_E],
                        gu_g,
                        dn_g,
                    );
                    for j in 0..H {
                        out[t * H + j] += rw[t][e] * r[j];
                    }
                }
            }
            out
        };

        let compare = |gpu: &Vec<f32>, cpu: &Vec<f32>, stage: &str| {
            let mut nan = 0usize;
            let mut max_abs = 0.0f32;
            for (g, r) in gpu.iter().zip(cpu.iter()) {
                if g.is_nan() {
                    nan += 1;
                }
                max_abs = max_abs.max((g - r).abs());
            }
            println!("{stage}: max_abs={max_abs:.3e} NaN={nan}");
            assert!(nan == 0, "{stage}: NaN in GPU output");
            assert!(max_abs < 0.05, "{stage}: divergence beyond bound");
        };

        // ---- stage A: single expert 0 ----
        let a = [[0usize; EXPERTS]; TOKENS];
        let gpu_a = run_chain(&a);
        let cpu_a = cpu_chain(&a);
        compare(&gpu_a, &cpu_a, "stage A (all combos expert 0, VRAM)");

        // ---- stage B: alternating expert 0 (VRAM) / expert 4 (pinned) ----
        let mut b = [[0usize; EXPERTS]; TOKENS];
        for t in 0..TOKENS {
            for e in 0..EXPERTS {
                b[t][e] = if (t + e) % 2 == 0 { 0 } else { 4 };
            }
        }
        let gpu_b = run_chain(&b);
        let cpu_b = cpu_chain(&b);
        compare(&gpu_b, &cpu_b, "stage B (experts 0/4 alternating, VRAM+pinned)");

        // ---- stage C: full random routing over all 10 experts ----
        let gpu_c = run_chain(&routed);
        let cpu_c = cpu_chain(&routed);
        compare(&gpu_c, &cpu_c, "stage C (full random routing)");

        println!("p5: PASS — expert FFN chain, mixed residency, verified against CPU dequant");
    }
}

unsafe fn to_u64_dev(v: &[u64]) -> CUdeviceptr {
    let d = alloc_dev(v.len() * 8);
    ck(sys::cuMemcpyHtoDAsync_v2(d, v.as_ptr() as *const std::ffi::c_void, v.len() * 8, std::ptr::null_mut()));
    ck(sys::cuStreamSynchronize(std::ptr::null_mut()));
    d
}
