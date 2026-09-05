//! Probe 7 (Crow #10): full-attention mixer sub-block on GPU, f32, vs the
//! transformers reference golden (`oracle/golden/layer3-attn-input.f32` /
//! `-output.f32`, exported by `oracle/export_attention_golden.py`). Math ported
//! 1:1 from `Qwen4ExpTextAttention` + `eager_attention_forward`
//! (transformers 5.16.1, layer 3 = first full_attention layer).
//!
//! Reference formulas (cross-checked against the source):
//! - q_proj: 2560 → 24 heads × 512, chunk → query [24×256] + gate [24×256]
//! - q_norm/k_norm: RMSNorm per head over 256, eps 1e-6, weight applied as (1+w)
//! - partial rotary 0.25: rotate first 64 dims (rotate_half within [0..32|32..64]),
//!   interleaved mrope with identical T/H/W streams = plain rope, theta 1e7
//! - GQA 24 q-heads / 2 kv-heads (12:1, repeat_interleave), scaling 1/sqrt(256)
//! - eager attention, softmax f32, causal; QSA indexer DENSE by construction at
//!   T=8 (2 complete blocks of 4, block_topk 512 ≥ 2 → all visible selected)
//! - output gate: attn_out × sigmoid(gate), then o_proj 6144 → 2560
//!
//! Scalar kernel params travel in device i32 buffers (p5 lesson). Every HtoD
//! upload async + explicit sync (WDDM rule).

use std::ffi::CString;
use std::io::{Read, Seek, SeekFrom};
use std::time::Instant;

use cudarc::driver::sys::{self, CUdeviceptr, CUfunction, CUresult};
use cudarc::nvrtc::{compile_ptx_with_opts, CompileOptions};

const T: usize = 8;
const H: usize = 2560;
const NQ: usize = 24;
const NKV: usize = 2;
const HD: usize = 256;
const ROT: usize = 64; // partial_rotary_factor 0.25 × 256
const Q_ROWS: usize = NQ * HD * 2; // 12288 (query + gate)
const KV_ROWS: usize = NKV * HD; // 512
const CORE: usize = NQ * HD; // 6144

const KERNEL_SRC: &str = r#"
extern "C" __global__ void gemv_f32(const float* __restrict__ w,
                                    const float* __restrict__ x,
                                    float* __restrict__ y,
                                    const int* __restrict__ k_dim_p) {
    int k_dim = *k_dim_p;
    int row = blockIdx.x;
    const float* wp = w + (size_t)row * k_dim;
    float acc = 0.0f;
    for (int i = threadIdx.x; i < k_dim; i += blockDim.x) acc += wp[i] * x[i];
    __shared__ float red[256];
    red[threadIdx.x] = acc;
    __syncthreads();
    for (int st = blockDim.x / 2; st > 0; st >>= 1) {
        if (threadIdx.x < st) red[threadIdx.x] += red[threadIdx.x + st];
        __syncthreads();
    }
    if (threadIdx.x == 0) y[row] = red[0];
}

// q_proj output [T][12288] -> query [T][24][256] compact + gate [T][6144] compact
extern "C" __global__ void split_qg(const float* __restrict__ qg,
                                    float* __restrict__ q,
                                    float* __restrict__ gate) {
    int head = blockIdx.x;
    int t = blockIdx.y;
    int d = threadIdx.x;
    int src = t * 12288 + head * 512 + d;
    q[(t * 24 + head) * 256 + d] = qg[src];
    gate[t * 6144 + head * 256 + d] = qg[src + 256];
}

// RMSNorm per head, weight applied as (1 + w) — checkpoint stores the delta
extern "C" __global__ void rmsnorm_1pw(const float* __restrict__ x,
                                       const float* __restrict__ w,
                                       float* __restrict__ out) {
    int head = blockIdx.x;
    int t = blockIdx.y;
    int d = threadIdx.x;
    const float* xp = x + (t * gridDim.x + head) * 256;
    __shared__ float red[256];
    red[d] = xp[d] * xp[d];
    __syncthreads();
    for (int st = 128; st > 0; st >>= 1) {
        if (d < st) red[d] += red[d + st];
        __syncthreads();
    }
    float rms = rsqrtf(red[0] / 256.0f + 1e-6f);
    out[(t * gridDim.x + head) * 256 + d] = xp[d] * rms * (1.0f + w[d]);
}

// partial rotary: rotate dims [0..32) against [32..64), rest passthrough
// (rotate_half style; interleaved mrope degenerates to plain rope for text-only)
extern "C" __global__ void rope(const float* __restrict__ x,
                                const float* __restrict__ cos_,
                                const float* __restrict__ sin_,
                                float* __restrict__ out) {
    int head = blockIdx.x;
    int t = blockIdx.y;
    int d = threadIdx.x;
    const float* xp = x + (t * gridDim.x + head) * 256;
    float* op = out + (t * gridDim.x + head) * 256;
    if (d >= 32) {
        op[d] = xp[d];
        return;
    }
    float a = xp[d], b = xp[d + 32];
    float c = cos_[t * 32 + d], s = sin_[t * 32 + d];
    op[d] = a * c - b * s;
    op[d + 32] = b * c + a * s;
}

// dense causal attention, block per q-head, thread d owns output dim d.
// Each thread computes the ≤8 scores redundantly (deterministic, no divgence).
extern "C" __global__ void attn_dense(const float* __restrict__ q,
                                      const float* __restrict__ k,
                                      const float* __restrict__ v,
                                      float* __restrict__ out,
                                      const int* __restrict__ t_p) {
    int steps = *t_p;
    int head = blockIdx.x;
    int kvh = head / 12;
    int d = threadIdx.x;
    const float scale = 0.0625f; // 1/sqrt(256)
    for (int t = 0; t < steps; t++) {
        const float* qt = q + (t * 24 + head) * 256;
        float p[8];
        float mx = -3.0e38f;
        for (int s = 0; s <= t; s++) {
            const float* ks = k + (s * 2 + kvh) * 256;
            float acc = 0.0f;
            for (int j = 0; j < 256; j++) acc += qt[j] * ks[j];
            acc *= scale;
            p[s] = acc;
            if (acc > mx) mx = acc;
        }
        float sum = 0.0f;
        for (int s = 0; s <= t; s++) { p[s] = expf(p[s] - mx); sum += p[s]; }
        float o = 0.0f;
        for (int s = 0; s <= t; s++) o += p[s] * v[(s * 2 + kvh) * 256 + d];
        out[(t * 24 + head) * 256 + d] = o / sum;
    }
}

// attn_out × sigmoid(gate), both [T][6144] compact; T*6144 divisible by 256
extern "C" __global__ void gate_mul(const float* __restrict__ core,
                                    const float* __restrict__ gate,
                                    float* __restrict__ out) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    float g = gate[i];
    out[i] = core[i] / (1.0f + expf(-g));
}
"#;

fn ck(r: CUresult) {
    if r != CUresult::CUDA_SUCCESS {
        panic!("CUDA error: {r:?}");
    }
}

fn bf16_file_to_f32(models: &str, name: &str, shard: &str) -> Vec<f32> {
    let mut fh = std::fs::File::open(format!("{models}/{shard}")).unwrap();
    let mut n8 = [0u8; 8];
    fh.read_exact(&mut n8).unwrap();
    let hl = u64::from_le_bytes(n8) as usize;
    let mut hb = vec![0u8; hl];
    fh.read_exact(&mut hb).unwrap();
    let hdr: serde_json::Value = serde_json::from_slice(&hb).unwrap();
    let info = &hdr[name];
    let dt = info["dtype"].as_str().unwrap();
    let off = info["data_offsets"][0].as_u64().unwrap() as usize;
    let end = info["data_offsets"][1].as_u64().unwrap() as usize;
    fh.seek(SeekFrom::Start((8 + hl + off) as u64)).unwrap();
    let mut raw = vec![0u8; end - off];
    fh.read_exact(&mut raw).unwrap();
    match dt {
        "F32" => raw
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect(),
        _ => raw
            .chunks_exact(2)
            .map(|c| f32::from_bits((u16::from_le_bytes([c[0], c[1]]) as u32) << 16))
            .collect(),
    }
}

fn read_bin_f32(path: &str, expect_len: usize) -> Vec<f32> {
    let mut f = std::fs::File::open(path).unwrap();
    let mut v = Vec::new();
    f.read_to_end(&mut v).unwrap();
    assert_eq!(v.len(), expect_len * 4, "{path}: size mismatch");
    v.chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

unsafe fn alloc_zeroed(bytes: usize) -> CUdeviceptr {
    let mut d: CUdeviceptr = 0;
    ck(sys::cuMemAlloc_v2(&mut d, bytes));
    ck(sys::cuMemsetD8_v2(d, 0, bytes));
    d
}

unsafe fn upload_dev(v: &[u8]) -> CUdeviceptr {
    let d = alloc_zeroed(v.len());
    ck(sys::cuMemcpyHtoDAsync_v2(
        d,
        v.as_ptr() as *const std::ffi::c_void,
        v.len(),
        std::ptr::null_mut(),
    ));
    ck(sys::cuStreamSynchronize(std::ptr::null_mut()));
    d
}

unsafe fn to_f32_dev(v: &[f32]) -> CUdeviceptr {
    upload_dev(std::slice::from_raw_parts(v.as_ptr() as *const u8, v.len() * 4))
}

unsafe fn to_i32_dev(v: &[i32]) -> CUdeviceptr {
    upload_dev(std::slice::from_raw_parts(v.as_ptr() as *const u8, v.len() * 4))
}

unsafe fn dtoh(src: CUdeviceptr, n: usize) -> Vec<f32> {
    let mut out = vec![0f32; n];
    ck(sys::cuMemcpyDtoH_v2(
        out.as_mut_ptr() as *mut std::ffi::c_void,
        src,
        n * 4,
    ));
    out
}

unsafe fn launch(
    f: CUfunction,
    gx: u32,
    gy: u32,
    bx: u32,
    by: u32,
    smem: u32,
    args: &mut [*mut std::ffi::c_void],
) {
    ck(sys::cuLaunchKernel(
        f,
        gx,
        gy,
        1,
        bx,
        by,
        1,
        smem,
        std::ptr::null_mut(),
        args.as_mut_ptr(),
        std::ptr::null_mut(),
    ));
    ck(sys::cuStreamSynchronize(std::ptr::null_mut()));
}

fn main() {
    let models = "../models/Qwen3.8-Flash-Next-original";
    let index: serde_json::Value = serde_json::from_slice(
        &std::fs::read(format!("{models}/model.safetensors.index.json")).unwrap(),
    )
    .unwrap();
    let wm = &index["weight_map"];
    let p = |sub: &str| -> String {
        wm[format!("model.language_model.layers.3.self_attn.{sub}")]
            .as_str()
            .unwrap()
            .to_string()
    };

    let qw = bf16_file_to_f32(models, "model.language_model.layers.3.self_attn.q_proj.weight", &p("q_proj.weight"));
    let kw = bf16_file_to_f32(models, "model.language_model.layers.3.self_attn.k_proj.weight", &p("k_proj.weight"));
    let vw = bf16_file_to_f32(models, "model.language_model.layers.3.self_attn.v_proj.weight", &p("v_proj.weight"));
    let ow = bf16_file_to_f32(models, "model.language_model.layers.3.self_attn.o_proj.weight", &p("o_proj.weight"));
    let q_norm_w = bf16_file_to_f32(models, "model.language_model.layers.3.self_attn.q_norm.weight", &p("q_norm.weight"));
    let k_norm_w = bf16_file_to_f32(models, "model.language_model.layers.3.self_attn.k_norm.weight", &p("k_norm.weight"));
    assert_eq!(qw.len(), Q_ROWS * H);
    assert_eq!(kw.len(), KV_ROWS * H);
    assert_eq!(vw.len(), KV_ROWS * H);
    assert_eq!(ow.len(), H * CORE);
    assert_eq!(q_norm_w.len(), HD);
    assert_eq!(k_norm_w.len(), HD);
    println!("p7: layer-3 attention weights loaded (6 tensors, f32)");

    let x_in = read_bin_f32("../oracle/golden/layer3-attn-input.f32", T * H);
    let golden = read_bin_f32("../oracle/golden/layer3-attn-output.f32", T * H);

    // cos/sin host-side, mirroring the reference f32 computation: inv_freq over
    // 32 pairs, theta 1e7, positions 0..T-1 (text-only: T/H/W streams identical)
    let mut cos_h = vec![0f32; T * 32];
    let mut sin_h = vec![0f32; T * 32];
    for t in 0..T {
        for j in 0..32 {
            let inv = 10_000_000f32.powf(-(2.0 * j as f32) / ROT as f32);
            let f = t as f32 * inv;
            cos_h[t * 32 + j] = f.cos();
            sin_h[t * 32 + j] = f.sin();
        }
    }

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
            let mut fu: CUfunction = std::ptr::null_mut();
            ck(sys::cuModuleGetFunction(
                &mut fu,
                module,
                CString::new(n).unwrap().as_ptr(),
            ));
            fu
        };
        let fn_gemv = get_fn("gemv_f32");
        let fn_split = get_fn("split_qg");
        let fn_norm = get_fn("rmsnorm_1pw");
        let fn_rope = get_fn("rope");
        let fn_attn = get_fn("attn_dense");
        let fn_gate = get_fn("gate_mul");

        let mut kdim_h = to_i32_dev(&[H as i32]);
        let mut kdim_v = to_i32_dev(&[CORE as i32]);
        let mut t_param = to_i32_dev(&[T as i32]);

        let mut wq = to_f32_dev(&qw);
        let mut wk = to_f32_dev(&kw);
        let mut wv = to_f32_dev(&vw);
        let mut wo = to_f32_dev(&ow);
        let mut wqn = to_f32_dev(&q_norm_w);
        let mut wkn = to_f32_dev(&k_norm_w);
        let mut cos_dev = to_f32_dev(&cos_h);
        let mut sin_dev = to_f32_dev(&sin_h);
        let x_dev = to_f32_dev(&x_in);

        let t_start = Instant::now();

        // 1. q_proj per token → qg [T][12288]
        let mut qg_dev = alloc_zeroed(T * Q_ROWS * 4);
        for t in 0..T {
            let mut xt = x_dev + (t * H * 4) as u64;
            let mut yt = qg_dev + (t * Q_ROWS * 4) as u64;
            launch(
                fn_gemv,
                Q_ROWS as u32,
                1,
                256,
                1,
                0,
                &mut [
                    &mut wq as *mut _ as *mut _,
                    &mut xt as *mut _ as *mut _,
                    &mut yt as *mut _ as *mut _,
                    &mut kdim_h as *mut _ as *mut _,
                ],
            );
        }

        // 2. split query/gate into compact layouts
        let mut q_dev = alloc_zeroed(T * CORE * 4);
        let mut gate_dev = alloc_zeroed(T * CORE * 4);
        launch(
            fn_split,
            NQ as u32,
            T as u32,
            HD as u32,
            1,
            0,
            &mut [
                &mut qg_dev as *mut _ as *mut _,
                &mut q_dev as *mut _ as *mut _,
                &mut gate_dev as *mut _ as *mut _,
            ],
        );

        // 3. q_norm then rope
        let mut qn_dev = alloc_zeroed(T * CORE * 4);
        launch(
            fn_norm,
            NQ as u32,
            T as u32,
            HD as u32,
            1,
            0,
            &mut [
                &mut q_dev as *mut _ as *mut _,
                &mut wqn as *mut _ as *mut _,
                &mut qn_dev as *mut _ as *mut _,
            ],
        );
        let mut qr_dev = alloc_zeroed(T * CORE * 4);
        launch(
            fn_rope,
            NQ as u32,
            T as u32,
            HD as u32,
            1,
            0,
            &mut [
                &mut qn_dev as *mut _ as *mut _,
                &mut cos_dev as *mut _ as *mut _,
                &mut sin_dev as *mut _ as *mut _,
                &mut qr_dev as *mut _ as *mut _,
            ],
        );

        // 4. k_proj per token → k_norm → rope
        let mut k_dev = alloc_zeroed(T * KV_ROWS * 4);
        for t in 0..T {
            let mut xt = x_dev + (t * H * 4) as u64;
            let mut yt = k_dev + (t * KV_ROWS * 4) as u64;
            launch(
                fn_gemv,
                KV_ROWS as u32,
                1,
                256,
                1,
                0,
                &mut [
                    &mut wk as *mut _ as *mut _,
                    &mut xt as *mut _ as *mut _,
                    &mut yt as *mut _ as *mut _,
                    &mut kdim_h as *mut _ as *mut _,
                ],
            );
        }
        let mut kn_dev = alloc_zeroed(T * KV_ROWS * 4);
        launch(
            fn_norm,
            NKV as u32,
            T as u32,
            HD as u32,
            1,
            0,
            &mut [
                &mut k_dev as *mut _ as *mut _,
                &mut wkn as *mut _ as *mut _,
                &mut kn_dev as *mut _ as *mut _,
            ],
        );
        let mut kr_dev = alloc_zeroed(T * KV_ROWS * 4);
        launch(
            fn_rope,
            NKV as u32,
            T as u32,
            HD as u32,
            1,
            0,
            &mut [
                &mut kn_dev as *mut _ as *mut _,
                &mut cos_dev as *mut _ as *mut _,
                &mut sin_dev as *mut _ as *mut _,
                &mut kr_dev as *mut _ as *mut _,
            ],
        );

        // 5. v_proj per token
        let mut v_dev = alloc_zeroed(T * KV_ROWS * 4);
        for t in 0..T {
            let mut xt = x_dev + (t * H * 4) as u64;
            let mut yt = v_dev + (t * KV_ROWS * 4) as u64;
            launch(
                fn_gemv,
                KV_ROWS as u32,
                1,
                256,
                1,
                0,
                &mut [
                    &mut wv as *mut _ as *mut _,
                    &mut xt as *mut _ as *mut _,
                    &mut yt as *mut _ as *mut _,
                    &mut kdim_h as *mut _ as *mut _,
                ],
            );
        }

        // 6. dense causal attention (QSA dense by construction at T=8)
        let mut core_dev = alloc_zeroed(T * CORE * 4);
        launch(
            fn_attn,
            NQ as u32,
            1,
            HD as u32,
            1,
            0,
            &mut [
                &mut qr_dev as *mut _ as *mut _,
                &mut kr_dev as *mut _ as *mut _,
                &mut v_dev as *mut _ as *mut _,
                &mut core_dev as *mut _ as *mut _,
                &mut t_param as *mut _ as *mut _,
            ],
        );

        // 7. output gate
        let mut gated_dev = alloc_zeroed(T * CORE * 4);
        launch(
            fn_gate,
            ((T * CORE) as u32 + 255) / 256,
            1,
            256,
            1,
            0,
            &mut [
                &mut core_dev as *mut _ as *mut _,
                &mut gate_dev as *mut _ as *mut _,
                &mut gated_dev as *mut _ as *mut _,
            ],
        );

        // 8. o_proj per token
        let y_dev = alloc_zeroed(T * H * 4);
        for t in 0..T {
            let mut xt = gated_dev + (t * CORE * 4) as u64;
            let mut yt = y_dev + (t * H * 4) as u64;
            launch(
                fn_gemv,
                H as u32,
                1,
                256,
                1,
                0,
                &mut [
                    &mut wo as *mut _ as *mut _,
                    &mut xt as *mut _ as *mut _,
                    &mut yt as *mut _ as *mut _,
                    &mut kdim_v as *mut _ as *mut _,
                ],
            );
        }

        let gpu_ms = t_start.elapsed().as_secs_f64() * 1e3;

        // 9. compare vs golden
        let gpu = dtoh(y_dev, T * H);
        let mut nan = 0usize;
        let mut max_abs = 0.0f32;
        let mut argmax = 0usize;
        for i in 0..T * H {
            if gpu[i].is_nan() {
                nan += 1;
            }
            let d = (gpu[i] - golden[i]).abs();
            if d > max_abs {
                max_abs = d;
                argmax = i;
            }
        }
        for t in 0..T {
            let mut tmax = 0.0f32;
            for j in 0..H {
                tmax = tmax.max((gpu[t * H + j] - golden[t * H + j]).abs());
            }
            println!("  token {t}: max_abs={tmax:.3e}");
        }
        println!(
            "p7: max_abs={max_abs:.3e} at ({}, {}) NaN={nan} tol=5e-3",
            argmax / H,
            argmax % H
        );
        println!("p7: pipeline (all launches, sync-per-launch) T=8: {gpu_ms:.2} ms");
        if nan == 0 && max_abs < 5e-3 {
            println!("p7: PASS — attention sub-block matches the transformers reference golden");
        } else {
            let mut worst: Vec<usize> = (0..T * H).collect();
            worst.sort_by(|&i, &j| {
                ((gpu[j] - golden[j]).abs())
                    .partial_cmp(&(gpu[i] - golden[i]).abs())
                    .unwrap()
            });
            println!("p7: FAIL — worst 8 positions (gpu vs golden):");
            for &i in worst.iter().take(8) {
                println!(
                    "  ({}, {}): gpu={:+.6} golden={:+.6}",
                    i / H,
                    i % H,
                    gpu[i],
                    golden[i]
                );
            }
            std::process::exit(1);
        }
    }
}
