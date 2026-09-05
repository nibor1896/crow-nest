//! Probe 16 (Crow #10): the ATTENTION sub-block at PRODUCTION weight precision
//! — the p10 pattern applied to the 12 full-attention layers. Layer 3 (first
//! full_attention layer), tensors read straight from the real CNQ4.5 container
//! (`converter/Qwen3.8-Flash-Next-CNQ4.5.cnq`, section `text`), dequantized on
//! the fly inside the GEMV kernels (p2 fragment layouts). q_norm/k_norm are
//! BF16 keeps → f32. Activations stay f32 (FP4 is a weight format).
//!
//! Stage A: gemv_fp4 vs CPU dequant of the same bytes (transport + decode
//! exactness, first 128 rows of each projection).
//! Stage B: batched sub-block (T=8, p7 chain) vs `oracle/golden/
//! layer3-attn-output.f32` — MEASUREMENT of the 4.5-bpw RTN delta (not a
//! pass/fail gate; quality belongs to the ten-task gate at #11).
//! Stage C: 8 decode steps (p12 chain: persistent KV cache, rotary at pos t)
//! vs the same golden rows — the decode-shape FP4 delta.
//!
//! Scalar kernel args via device buffers (p5 lesson); HtoD async + sync.

use std::ffi::CString;
use std::io::{Read, Seek, SeekFrom};
use std::time::Instant;

use cudarc::driver::sys::{self, CUdeviceptr, CUfunction, CUresult};
use cudarc::nvrtc::{compile_ptx_with_opts, CompileOptions};

const T: usize = 8;
const T_MAX: usize = 8;
const H: usize = 2560;
const NQ: usize = 24;
const NKV: usize = 2;
const HD: usize = 256;
const Q_ROWS: usize = NQ * HD * 2; // 12288
const KV_ROWS: usize = NKV * HD; // 512
const CORE: usize = NQ * HD; // 6144
const LAYER: usize = 3;

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

// FP4 GEMV single token: y[row] = W[row]·x, dequant on the fly
extern "C" __global__ void gemv_fp4(const unsigned char* __restrict__ w,
                                    const float* __restrict__ x,
                                    const float* __restrict__ gs_ptr,
                                    float* __restrict__ y,
                                    const int* __restrict__ k_dim_p) {
    int k_dim = *k_dim_p;
    int bpr = k_dim >> 6;
    int row = blockIdx.x;
    const unsigned char* rowp = w + (size_t)row * bpr * 36;
    float gs = gs_ptr[0];
    float acc = 0.0f;
    for (int b = threadIdx.x; b < bpr; b += blockDim.x) {
        const unsigned char* blk = rowp + b * 36;
        #pragma unroll
        for (int sb = 0; sb < 4; sb++) {
            float s = ue4m3(blk[sb]) * gs;
            float part = 0.0f;
            #pragma unroll
            for (int j = 0; j < 16; j++) {
                int idx = sb * 16 + j;
                unsigned int byte = blk[4 + (idx >> 1)];
                unsigned int nib = (idx & 1) ? ((byte >> 4) & 0xF) : (byte & 0xF);
                part += e2m1(nib) * x[b * 64 + sb * 16 + j];
            }
            acc += part * s;
        }
    }
    __shared__ float red[256];
    red[threadIdx.x] = acc;
    __syncthreads();
    for (int st = blockDim.x / 2; st > 0; st >>= 1) {
        if (threadIdx.x < st) red[threadIdx.x] += red[threadIdx.x + st];
        __syncthreads();
    }
    if (threadIdx.x == 0) y[row] = red[0];
}

// flat dequant for reference chains: one thread per value
extern "C" __global__ void dequant_fp4_flat(const unsigned char* __restrict__ w,
                                            const float* __restrict__ gs_ptr,
                                            float* __restrict__ out,
                                            const int* __restrict__ n_p) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= *n_p) return;
    int b = i >> 6, idx = i & 63;
    const unsigned char* blk = w + (size_t)b * 36;
    float s = ue4m3(blk[idx >> 4]) * gs_ptr[0];
    unsigned int byte = blk[4 + (idx >> 1)];
    unsigned int nib = (idx & 1) ? ((byte >> 4) & 0xF) : (byte & 0xF);
    out[i] = e2m1(nib) * s;
}

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

// dense causal attention over the prompt (T=8, block per q-head)
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

// single-token attention over the persistent KV cache
extern "C" __global__ void attn_step(const float* __restrict__ q,
                                     const float* __restrict__ k_cache,
                                     const float* __restrict__ v_cache,
                                     const int* __restrict__ pos_p,
                                     const int* __restrict__ tmax_p,
                                     float* __restrict__ out) {
    int head = blockIdx.x;
    int kvh = head / 12;
    int d = threadIdx.x;
    int len = *pos_p + 1;
    int tmax = *tmax_p;
    const float* qt = q + head * 256;
    const float scale = 0.0625f;
    float p[8];
    float mx = -3.0e38f;
    for (int s = 0; s < len; s++) {
        const float* ks = k_cache + (kvh * tmax + s) * 256;
        float acc = 0.0f;
        for (int j = 0; j < 256; j++) acc += qt[j] * ks[j];
        acc *= scale;
        p[s] = acc;
        if (acc > mx) mx = acc;
    }
    float sum = 0.0f;
    for (int s = 0; s < len; s++) { p[s] = expf(p[s] - mx); sum += p[s]; }
    float o = 0.0f;
    for (int s = 0; s < len; s++) o += p[s] * v_cache[(kvh * tmax + s) * 256 + d];
    out[head * 256 + d] = o / sum;
}

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

// ---- CNQ container (p10 reader) ----

struct Cnq {
    file: std::fs::File,
    blob_offset: u64,
    index: serde_json::Value,
}

impl Cnq {
    fn open(path: &str) -> Self {
        let mut f = std::fs::File::open(path).unwrap();
        let file_len = f.metadata().unwrap().len();
        f.seek(SeekFrom::Start(file_len - 8)).unwrap();
        let mut b8 = [0u8; 8];
        f.read_exact(&mut b8).unwrap();
        let idx_len = u64::from_le_bytes(b8) as usize;
        f.seek(SeekFrom::Start(file_len - 8 - idx_len as u64)).unwrap();
        let mut ib = vec![0u8; idx_len];
        f.read_exact(&mut ib).unwrap();
        let index: serde_json::Value = serde_json::from_slice(&ib).unwrap();
        let blob_offset = index["blob_offset"].as_u64().unwrap();
        Self { file: f, blob_offset, index }
    }

    fn find(&self, name: &str, section: &str) -> &serde_json::Value {
        self.index["tensors"]
            .as_array()
            .unwrap()
            .iter()
            .find(|t| t["name"].as_str().unwrap() == name && t["section"].as_str().unwrap() == section)
            .unwrap_or_else(|| panic!("tensor not found: {name} [{section}]"))
    }

    fn read_bytes(&mut self, t: &serde_json::Value) -> Vec<u8> {
        let off = self.blob_offset + t["offset"].as_u64().unwrap();
        let n_values = t["n_values"].as_u64().unwrap() as usize;
        let len = if t["dtype"].as_str().unwrap() == "bf16" {
            n_values * 2
        } else {
            (n_values + 63) / 64 * 36
        };
        self.file.seek(SeekFrom::Start(off)).unwrap();
        let mut raw = vec![0u8; len];
        self.file.read_exact(&mut raw).unwrap();
        raw
    }
}

fn bf16_bytes_to_f32(raw: &[u8]) -> Vec<f32> {
    raw.chunks_exact(2)
        .map(|c| f32::from_bits((u16::from_le_bytes([c[0], c[1]]) as u32) << 16))
        .collect()
}

// CPU dequant (stage A reference) — same decode as the kernel
fn e2m1(nibble: u32) -> f32 {
    const MAG: [f32; 8] = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0];
    let v = MAG[(nibble & 0x7) as usize];
    if nibble & 0x8 != 0 { -v } else { v }
}

fn ue4m3(byte: u32) -> f32 {
    let e = (byte >> 3) & 0xF;
    let m = byte & 0x7;
    if e == 0 {
        (m as f32) * 1.953125e-3
    } else {
        (1.0 + (m as f32) / 8.0) * 2.0f32.powi(e as i32 - 7)
    }
}

/// CPU dequant of `rows` rows of an FP4 tensor, then dot with `x`
fn cpu_dequant_dot(raw: &[u8], gs: f32, k_dim: usize, rows: usize, x: &[f32]) -> Vec<f32> {
    let bpr = k_dim / 64;
    (0..rows)
        .map(|row| {
            let rowp = &raw[row * bpr * 36..(row + 1) * bpr * 36];
            let mut acc = 0.0f32;
            for b in 0..bpr {
                let blk = &rowp[b * 36..(b + 1) * 36];
                for sb in 0..4 {
                    let s = ue4m3(blk[sb] as u32) * gs;
                    let mut part = 0.0f32;
                    for j in 0..16 {
                        let idx = sb * 16 + j;
                        let byte = blk[4 + (idx >> 1)] as u32;
                        let nib = if idx & 1 != 0 { (byte >> 4) & 0xF } else { byte & 0xF };
                        part += e2m1(nib) * x[b * 64 + sb * 16 + j];
                    }
                    acc += part * s;
                }
            }
            acc
        })
        .collect()
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

/// one FP4 GEMV for a single-token row
unsafe fn fp4_gemv_row(
    f: CUfunction,
    w: CUdeviceptr,
    gs: CUdeviceptr,
    kdim: CUdeviceptr,
    x_row: CUdeviceptr,
    y_row: CUdeviceptr,
    rows: u32,
) {
    let mut wv = w;
    let mut xv = x_row;
    let mut yv = y_row;
    let mut gsv = gs;
    let mut kv = kdim;
    launch(f, rows, 1, 256, 1, 0, &mut [
        &mut wv as *mut _ as *mut _, &mut xv as *mut _ as *mut _,
        &mut gsv as *mut _ as *mut _, &mut yv as *mut _ as *mut _,
        &mut kv as *mut _ as *mut _]);
}

/// one GEMV row, either FP4 on-the-fly or pre-dequantized f32
unsafe fn gemv_any<FProj, FDq>(
    use_fp4: bool,
    f_gemv4: CUfunction,
    f_gemvf32: CUfunction,
    proj: &FProj,
    dq_get: &FDq,
    sub: &str,
    kdim: CUdeviceptr,
    x: CUdeviceptr,
    y: CUdeviceptr,
    rows: u32,
) where
    FProj: Fn(&str) -> (CUdeviceptr, CUdeviceptr, usize),
    FDq: Fn(&str) -> CUdeviceptr,
{
    if use_fp4 {
        let (w, gs, _) = proj(sub);
        fp4_gemv_row(f_gemv4, w, gs, kdim, x, y, rows);
    } else {
        let mut wv = dq_get(sub);
        let mut xv = x;
        let mut yv = y;
        let mut kv = kdim;
        launch(f_gemvf32, rows, 1, 256, 1, 0, &mut [
            &mut wv as *mut _ as *mut _, &mut xv as *mut _ as *mut _,
            &mut yv as *mut _ as *mut _, &mut kv as *mut _ as *mut _]);
    }
}

fn main() {
    let cnq_path = "../converter/Qwen3.8-Flash-Next-CNQ4.5.cnq";
    println!("p16: opening CNQ4.5 container …");
    let mut cnq = Cnq::open(cnq_path);
    let P = |sub: &str| format!("model.language_model.layers.{LAYER}.self_attn.{sub}");
    const SEC: &str = "text";

    let t_load = Instant::now();
    // FP4 projections: (name, raw bytes, global scale)
    let mut projs: Vec<(&str, Vec<u8>, f32, usize)> = Vec::new(); // name, raw, gs, k_dim
    for (sub, k_dim) in [
        ("q_proj.weight", H),
        ("k_proj.weight", H),
        ("v_proj.weight", H),
        ("o_proj.weight", CORE),
    ] {
        let name = P(sub);
        let t = cnq.find(&name, SEC).clone();
        assert_ne!(t["dtype"].as_str().unwrap(), "bf16", "{sub}: expected NVFP4");
        let raw = cnq.read_bytes(&t);
        let gs = t["global_scale"].as_f64().unwrap() as f32;
        projs.push((sub, raw, gs, k_dim));
    }
    // BF16 keeps: per-head norms
    let mut q_norm_f32 = Vec::new();
    let mut k_norm_f32 = Vec::new();
    for (sub, out) in [("q_norm.weight", &mut q_norm_f32), ("k_norm.weight", &mut k_norm_f32)] {
        let name = P(sub);
        let t = cnq.find(&name, SEC).clone();
        assert_eq!(t["dtype"].as_str().unwrap(), "bf16", "{sub}: expected BF16 keep");
        let raw = cnq.read_bytes(&t);
        *out = bf16_bytes_to_f32(&raw);
        assert_eq!(out.len(), HD);
    }
    println!(
        "p16: 4 FP4 projections ({} MB total) + 2 BF16 norms in {:.1} s",
        projs.iter().map(|p| p.1.len()).sum::<usize>() / (1 << 20),
        t_load.elapsed().as_secs_f64()
    );

    let x_in = read_bin_f32("../oracle/golden/layer3-attn-input.f32", T * H);
    let golden = read_bin_f32("../oracle/golden/layer3-attn-output.f32", T * H);

    let mut cos_h = vec![0f32; T * 32];
    let mut sin_h = vec![0f32; T * 32];
    for t in 0..T {
        for j in 0..32 {
            let inv = 10_000_000f32.powf(-(2.0 * j as f32) / 64.0);
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
        let f_gemv4 = get_fn("gemv_fp4");
        let f_split = get_fn("split_qg");
        let f_norm = get_fn("rmsnorm_1pw");
        let f_rope = get_fn("rope");
        let f_attnb = get_fn("attn_dense");
        let f_attns = get_fn("attn_step");
        let f_gate = get_fn("gate_mul");

        let mut kdim_h = to_i32_dev(&[H as i32]);
        let mut kdim_v = to_i32_dev(&[CORE as i32]);
        let mut tmax_p = to_i32_dev(&[T_MAX as i32]);
        let mut pos_p = to_i32_dev(&[0]);
        let mut t_param = to_i32_dev(&[T as i32]);

        // upload the four projections (raw FP4 + global scale on device)
        let mut dev_proj: Vec<(&str, CUdeviceptr, CUdeviceptr, usize)> = Vec::new();
        for (sub, raw, gs, k_dim) in &projs {
            let raw_dev = upload_dev(raw);
            let gs_dev = to_f32_dev(&[*gs]);
            dev_proj.push((*sub, raw_dev, gs_dev, *k_dim));
        }
        let proj = |sub: &str| -> (CUdeviceptr, CUdeviceptr, usize) {
            dev_proj
                .iter()
                .find(|(n, _, _, _)| *n == sub)
                .map(|(_, r, g, k)| (*r, *g, *k))
                .unwrap()
        };
        let mut w_qn = to_f32_dev(&q_norm_f32);
        let mut w_kn = to_f32_dev(&k_norm_f32);
        let mut cos_dev = to_f32_dev(&cos_h);
        let mut sin_dev = to_f32_dev(&sin_h);
        let mut x_dev = to_f32_dev(&x_in);

        // ---------- STAGE A: transport exactness (first 128 rows each) ----------
        println!("p16 stage A: gemv_fp4 vs CPU dequant (first 128 rows)");
        let mut kdim_o_dev = to_i32_dev(&[projs.iter().find(|p| p.0 == "o_proj.weight").unwrap().3 as i32]);
        let mut a_max = 0.0f32;
        let mut a_abs = 0.0f32;
        for (sub, raw, gs, k_dim) in &projs {
            let (r, g, _) = proj(sub);
            let x_row = x_dev; // golden input row 0
            let mut y_dev = alloc_zeroed(*k_dim * 4);
            let rows = 128u32;
            let kdim_a = if *sub == "o_proj.weight" { kdim_o_dev } else { kdim_h };
            fp4_gemv_row(f_gemv4, r, g, kdim_a, x_row, y_dev, rows);
            let gpu = dtoh(y_dev, *k_dim);
            free_dev(&mut y_dev);
            let y_slice = &gpu[..128.min(*k_dim)];
            let cpu = cpu_dequant_dot(raw, *gs, *k_dim, y_slice.len(), &x_in[..*k_dim]);
            for i in 0..y_slice.len() {
                let diff = (gpu[i] - cpu[i]).abs();
                a_abs = a_abs.max(diff);
                let denom = cpu[i].abs().max(1e-3);
                a_max = a_max.max(diff / denom);
            }
        }
        println!("p16 stage A: max rel vs CPU dequant = {a_max:.3e}, max_abs = {a_abs:.3e} (rel gate 1e-2)");

        // ---------- STAGE B: batched sub-block, FP4 vs dequant-f32 reference ----------
        let mut f_dqflat = get_fn("dequant_fp4_flat");
        let mut f_gemvf32 = get_fn("gemv_f32");

        // dequantized f32 copies of all four projections (reference chain)
        let skip_dq = std::env::var("P16_SKIP_DQ").is_ok();
        let mut dq_f32: Vec<(&str, CUdeviceptr)> = Vec::new();
        let rows_of = |sub: &str| -> usize {
            if sub == "q_proj.weight" { Q_ROWS } else if sub == "o_proj.weight" { H } else { KV_ROWS }
        };
        for (sub, _, _, k_dim) in &projs {
            if skip_dq { break; }
            println!("  dq {} (k_dim={}) …", sub, k_dim);
            let (mut r, mut g, _) = proj(sub);
            let rows = rows_of(sub);
            let n_vals = rows * k_dim;
            let mut d_dev = alloc_zeroed(n_vals * 4);
            let mut n_dev = to_i32_dev(&[n_vals as i32]);
            launch(f_dqflat, ((n_vals as u32) + 255) / 256, 1, 256, 1, 0, &mut [
                &mut r as *mut _ as *mut _, &mut g as *mut _ as *mut _,
                &mut d_dev as *mut _ as *mut _, &mut n_dev as *mut _ as *mut _]);
            dq_f32.push((sub, d_dev));
            println!("  dq {} ok ({} rows)", sub, rows);
        }
        ck(sys::cuCtxSynchronize());
        println!("  dq phase synced clean");
        let dq_get = |sub: &str| -> CUdeviceptr {
            dq_f32.iter().find(|(n, _)| *n == sub).map(|(_, d)| *d).unwrap()
        };

        let mut k_cache = alloc_zeroed(NKV * T_MAX * HD * 4);
        let mut v_cache = alloc_zeroed(NKV * T_MAX * HD * 4);
        let mut kdim_o_dev = to_i32_dev(&[projs.iter().find(|p| p.0 == "o_proj.weight").unwrap().3 as i32]);

        let mut k_cache = alloc_zeroed(NKV * T_MAX * HD * 4);
        let mut v_cache = alloc_zeroed(NKV * T_MAX * HD * 4);
        let (r_q, g_q, _) = proj("q_proj.weight");
        let (r_k, g_k, _) = proj("k_proj.weight");
        let (r_v, g_v, _) = proj("v_proj.weight");
        let (r_o, g_o, _) = proj("o_proj.weight");

        // the p7 chain, parameterized by weight source (FP4 on-the-fly vs pre-dequant f32)
        let mut run_batched = |use_fp4: bool| -> Vec<f32> {
            unsafe {
                let gemv_row = |use_fp4: bool, sub: &str, kdim: CUdeviceptr, x: CUdeviceptr, y: CUdeviceptr, rows: u32| {
                    if use_fp4 {
                        let (r, g, _) = proj(sub);
                        fp4_gemv_row(f_gemv4, r, g, kdim, x, y, rows);
                    } else {
                        let mut wv = dq_get(sub);
                        let mut xv = x;
                        let mut yv = y;
                        let mut kv = kdim;
                        launch(f_gemvf32, rows, 1, 256, 1, 0, &mut [
                            &mut wv as *mut _ as *mut _, &mut xv as *mut _ as *mut _,
                            &mut yv as *mut _ as *mut _, &mut kv as *mut _ as *mut _]);
                    }
                };
                let mut qg_dev = alloc_zeroed(T * Q_ROWS * 4);
                for t in 0..T {
                    let mut xt = x_dev + (t * H * 4) as u64;
                    let mut yt = qg_dev + (t * Q_ROWS * 4) as u64;
                    gemv_any(use_fp4, f_gemv4, f_gemvf32, &proj, &dq_get, "q_proj.weight", kdim_h, xt, yt, Q_ROWS as u32);
                }
                println!("    chain: q gemvs done (fp4={use_fp4})");
                let mut q_dev = alloc_zeroed(T * CORE * 4);
                let mut gate_dev = alloc_zeroed(T * CORE * 4);
                launch(f_split, NQ as u32, T as u32, HD as u32, 1, 0, &mut [
                    &mut qg_dev as *mut _ as *mut _, &mut q_dev as *mut _ as *mut _,
                    &mut gate_dev as *mut _ as *mut _]);
                let mut qn_dev = alloc_zeroed(T * CORE * 4);
                launch(f_norm, NQ as u32, T as u32, HD as u32, 1, 0, &mut [
                    &mut q_dev as *mut _ as *mut _, &mut w_qn as *mut _ as *mut _,
                    &mut qn_dev as *mut _ as *mut _]);
                let mut qr_dev = alloc_zeroed(T * CORE * 4);
                launch(f_rope, NQ as u32, T as u32, HD as u32, 1, 0, &mut [
                    &mut qn_dev as *mut _ as *mut _, &mut cos_dev as *mut _ as *mut _,
                    &mut sin_dev as *mut _ as *mut _, &mut qr_dev as *mut _ as *mut _]);

                let mut k_dev = alloc_zeroed(T * KV_ROWS * 4);
                for t in 0..T {
                    let mut xt = x_dev + (t * H * 4) as u64;
                    let mut yt = k_dev + (t * KV_ROWS * 4) as u64;
                    gemv_any(use_fp4, f_gemv4, f_gemvf32, &proj, &dq_get, "k_proj.weight", kdim_h, xt, yt, KV_ROWS as u32);
                }
                println!("    chain: k gemvs done");
                let mut kn_dev = alloc_zeroed(T * KV_ROWS * 4);
                launch(f_norm, NKV as u32, T as u32, HD as u32, 1, 0, &mut [
                    &mut k_dev as *mut _ as *mut _, &mut w_kn as *mut _ as *mut _,
                    &mut kn_dev as *mut _ as *mut _]);
                let mut kr_dev = alloc_zeroed(T * KV_ROWS * 4);
                launch(f_rope, NKV as u32, T as u32, HD as u32, 1, 0, &mut [
                    &mut kn_dev as *mut _ as *mut _, &mut cos_dev as *mut _ as *mut _,
                    &mut sin_dev as *mut _ as *mut _, &mut kr_dev as *mut _ as *mut _]);

                let mut v_dev = alloc_zeroed(T * KV_ROWS * 4);
                for t in 0..T {
                    let mut xt = x_dev + (t * H * 4) as u64;
                    let mut yt = v_dev + (t * KV_ROWS * 4) as u64;
                    gemv_any(use_fp4, f_gemv4, f_gemvf32, &proj, &dq_get, "v_proj.weight", kdim_h, xt, yt, KV_ROWS as u32);
                }
                println!("    chain: v gemvs done");

                let mut core_dev = alloc_zeroed(T * CORE * 4);
                launch(f_attnb, NQ as u32, 1, HD as u32, 1, 0, &mut [
                    &mut qr_dev as *mut _ as *mut _, &mut kr_dev as *mut _ as *mut _,
                    &mut v_dev as *mut _ as *mut _, &mut core_dev as *mut _ as *mut _,
                    &mut t_param as *mut _ as *mut _]);
                let mut gated_dev = alloc_zeroed(T * CORE * 4);
                launch(f_gate, ((T * CORE) as u32 + 255) / 256, 1, 256, 1, 0, &mut [
                    &mut core_dev as *mut _ as *mut _, &mut gate_dev as *mut _ as *mut _,
                    &mut gated_dev as *mut _ as *mut _]);
                let mut y_dev = alloc_zeroed(T * H * 4);
                for t in 0..T {
                    let mut xt = gated_dev + (t * CORE * 4) as u64;
                    let mut yt = y_dev + (t * H * 4) as u64;
                    gemv_any(use_fp4, f_gemv4, f_gemvf32, &proj, &dq_get, "o_proj.weight", kdim_o_dev, xt, yt, H as u32);
                }
                println!("    chain: o gemvs done");
                let out = dtoh(y_dev, T * H);
                free_dev(&mut y_dev);
                free_dev(&mut qg_dev);
                free_dev(&mut q_dev);
                free_dev(&mut gate_dev);
                free_dev(&mut qn_dev);
                free_dev(&mut qr_dev);
                free_dev(&mut k_dev);
                free_dev(&mut kn_dev);
                free_dev(&mut kr_dev);
                free_dev(&mut v_dev);
                free_dev(&mut core_dev);
                free_dev(&mut gated_dev);
                out
            }
        };

        let gpu_b = run_batched(true); // FP4 on-the-fly
        let gpu_b32 = run_batched(false); // pre-dequantized f32 — GEMV reference
        let mut gemv_consistency = 0.0f32;
        for i in 0..T * H {
            gemv_consistency = gemv_consistency.max((gpu_b[i] - gpu_b32[i]).abs());
        }
        let mut b_max = 0.0f32;
        let mut b_sum = 0.0f64;
        let mut b_ref_sq = 0.0f64;
        let mut b_nan = 0usize;
        for i in 0..T * H {
            if gpu_b[i].is_nan() { b_nan += 1; }
            let d = (gpu_b[i] - golden[i]).abs();
            b_max = b_max.max(d);
            b_sum += (d as f64) * (d as f64);
            b_ref_sq += golden[i] as f64 * golden[i] as f64;
        }
        let rel_l2 = (b_sum / b_ref_sq).sqrt();
        println!(
            "p16 stage B: batched vs f32 golden: max_abs={b_max:.4} mean_abs={:.3e} rel_L2={rel_l2:.4e} NaN={b_nan}",
            gpu_b.iter().zip(&golden).map(|(a, b)| (a - b).abs()).sum::<f32>() / (T * H) as f32
        );
        println!("p16 stage B': on-the-fly FP4 vs pre-dequant f32 chain: max_abs={gemv_consistency:.3e} (GEMV consistency, expect ~1e-6)");

        // ---------- STAGE C: 8 decode steps (p12 chain, FP4 weights) ----------
        // fresh caches: stepping from zero, appending per step
        let mut kc = alloc_zeroed(NKV * T_MAX * HD * 4);
        let mut vc = alloc_zeroed(NKV * T_MAX * HD * 4);
        let mut c_max = 0.0f32;
        let mut c_nan = 0usize;
        for t in 0..T {
            let mut xt = x_dev + (t * H * 4) as u64;
            // q path
            let mut qg = alloc_zeroed(Q_ROWS * 4);
            fp4_gemv_row(f_gemv4, r_q, g_q, kdim_h, xt, qg, Q_ROWS as u32);
            let mut q1 = alloc_zeroed(CORE * 4);
            let mut g1 = alloc_zeroed(CORE * 4);
            launch(f_split, NQ as u32, 1, HD as u32, 1, 0, &mut [
                &mut qg as *mut _ as *mut _, &mut q1 as *mut _ as *mut _,
                &mut g1 as *mut _ as *mut _]);
            let mut qn1 = alloc_zeroed(CORE * 4);
            launch(f_norm, NQ as u32, 1, HD as u32, 1, 0, &mut [
                &mut q1 as *mut _ as *mut _, &mut w_qn as *mut _ as *mut _,
                &mut qn1 as *mut _ as *mut _]);
            let mut cos_t = cos_dev + (t * 32 * 4) as u64;
            let mut sin_t = sin_dev + (t * 32 * 4) as u64;
            let mut qr1 = alloc_zeroed(CORE * 4);
            launch(f_rope, NQ as u32, 1, HD as u32, 1, 0, &mut [
                &mut qn1 as *mut _ as *mut _, &mut cos_t as *mut _ as *mut _,
                &mut sin_t as *mut _ as *mut _, &mut qr1 as *mut _ as *mut _]);
            // k path
            let mut k1 = alloc_zeroed(KV_ROWS * 4);
            fp4_gemv_row(f_gemv4, r_k, g_k, kdim_h, xt, k1, KV_ROWS as u32);
            let mut kn1 = alloc_zeroed(KV_ROWS * 4);
            launch(f_norm, NKV as u32, 1, HD as u32, 1, 0, &mut [
                &mut k1 as *mut _ as *mut _, &mut w_kn as *mut _ as *mut _,
                &mut kn1 as *mut _ as *mut _]);
            let mut kr1 = alloc_zeroed(KV_ROWS * 4);
            launch(f_rope, NKV as u32, 1, HD as u32, 1, 0, &mut [
                &mut kn1 as *mut _ as *mut _, &mut cos_t as *mut _ as *mut _,
                &mut sin_t as *mut _ as *mut _, &mut kr1 as *mut _ as *mut _]);
            // v path
            let mut v1 = alloc_zeroed(KV_ROWS * 4);
            fp4_gemv_row(f_gemv4, r_v, g_v, kdim_h, xt, v1, KV_ROWS as u32);
            // append to caches
            for kvh in 0..NKV {
                let sk = kr1 + (kvh * HD * 4) as u64;
                let dk = kc + ((kvh * T_MAX + t) * HD * 4) as u64;
                ck(sys::cuMemcpyDtoDAsync_v2(dk, sk, HD * 4, std::ptr::null_mut()));
                let sv = v1 + (kvh * HD * 4) as u64;
                let dv = vc + ((kvh * T_MAX + t) * HD * 4) as u64;
                ck(sys::cuMemcpyDtoDAsync_v2(dv, sv, HD * 4, std::ptr::null_mut()));
            }
            ck(sys::cuStreamSynchronize(std::ptr::null_mut()));
            let mut pos_val: u32 = t as u32;
            ck(sys::cuMemcpyHtoDAsync_v2(
                pos_p,
                &mut pos_val as *mut u32 as *const std::ffi::c_void,
                4,
                std::ptr::null_mut(),
            ));
            ck(sys::cuStreamSynchronize(std::ptr::null_mut()));
            let mut core1 = alloc_zeroed(CORE * 4);
            launch(f_attns, NQ as u32, 1, HD as u32, 1, 0, &mut [
                &mut qr1 as *mut _ as *mut _, &mut kc as *mut _ as *mut _,
                &mut vc as *mut _ as *mut _, &mut pos_p as *mut _ as *mut _,
                &mut tmax_p as *mut _ as *mut _, &mut core1 as *mut _ as *mut _]);
            let mut gated1 = alloc_zeroed(CORE * 4);
            launch(f_gate, (CORE as u32 + 255) / 256, 1, 256, 1, 0, &mut [
                &mut core1 as *mut _ as *mut _, &mut g1 as *mut _ as *mut _,
                &mut gated1 as *mut _ as *mut _]);
            let mut y1 = alloc_zeroed(H * 4);
            fp4_gemv_row(f_gemv4, r_o, g_o, kdim_o_dev, gated1, y1, H as u32);
            let y_h = dtoh(y1, H);
            for j in 0..H {
                if y_h[j].is_nan() { c_nan += 1; }
                c_max = c_max.max((y_h[j] - golden[t * H + j]).abs());
            }
            for d in [&mut qg, &mut q1, &mut g1, &mut qn1, &mut qr1, &mut k1, &mut kn1, &mut kr1, &mut v1, &mut core1, &mut gated1, &mut y1] {
                free_dev(d);
            }
        }
        println!("p16 stage C: 8 decode steps vs f32 golden: max_abs={c_max:.4} NaN={c_nan}");

        let ok = a_max < 1e-2 && b_nan == 0 && c_nan == 0;
        if ok {
            println!("p16: DONE — attention sub-block measured at production FP4 (transport exact; B/C are 4.5-bpw RTN measurements, quality at #11)");
        } else {
            println!("p16: FAIL");
            std::process::exit(1);
        }
    }
}

unsafe fn free_dev(d: &mut CUdeviceptr) {
    if *d != 0 {
        ck(sys::cuMemFree_v2(*d));
        *d = 0;
    }
}
