//! Probe 10 (Crow #10): FULL decoder layer 0 with PRODUCTION weight precision —
//! every NVFP4 tensor straight from the real CNQ4.5 container
//! (`converter/Qwen3.8-Flash-Next-CNQ4.5.cnq`), dequantized on the fly inside
//! the GEMV kernels (p2-fragment layouts, p4/p5 pattern). BF16 keeps load as
//! f32. Pipeline math identical to p8 (f32 activations; FP4 is a weight format).
//!
//! Stage A: gemv_fp4 self-check vs a CPU dequant of the same bytes (tight tol).
//! Stage B: full layer vs `oracle/golden/layer0-golden-output.f32` (the f32
//! reference). This is a MEASUREMENT, not a pass/fail gate: weight-quant
//! quality is judged by the ten-task gate at #11; per-tensor weight deltas live
//! in the converter sidecar. We assert NaN=0 and report max/mean/rel-L2.
//!
//! Container facts (verified against the index): section `text` (the `mtp`
//! section carries its own layers.0 copy), nvfp4 = 64 values / 36 B block
//! (4 ue4m3 scale bytes + 32 B LSB-first nibbles), one global f32 scale per
//! tensor, rows contiguous without padding (offset deltas prove it). BF16
//! keeps: dtype "bf16", raw little-endian bytes. conv1d [10240,4] crosses
//! 64-value blocks → flat dequant at load, f32 conv kernel unchanged.

use std::ffi::CString;
use std::io::{Read, Seek, SeekFrom};
use std::time::Instant;

use cudarc::driver::sys::{self, CUdeviceptr, CUfunction, CUresult};
use cudarc::nvrtc::{compile_ptx_with_opts, CompileOptions};

const T: usize = 8;
const H: usize = 2560;
const HC: usize = 4;
const HCT: usize = HC * H; // 10240
const LOWRANK: usize = 320;
const E: usize = 512; // num_experts
const TOPK: usize = 10;
const INTER: usize = 640; // per-expert intermediate (gate_up rows = 2*INTER)
const GDN_KEY: usize = 2048;
const GDN_VAL: usize = 6144;
const GDN_CONV: usize = GDN_KEY * 2 + GDN_VAL; // 10240

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

// FP4 GEMV: row-major packed weights (bpr*36 B per row), dequant on the fly
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
    for (int st = 128; st > 0; st >>= 1) {
        if (threadIdx.x < st) red[threadIdx.x] += red[threadIdx.x + st];
        __syncthreads();
    }
    if (threadIdx.x == 0) y[row] = red[0];
}

// flat dequant for non-GEMV shapes (conv1d): one thread per value
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

// grouped RMSNorm: groups of 2560 over [T][10240], eps 1e-6, weight (1+w)
extern "C" __global__ void rms_group(const float* __restrict__ x,
                                     const float* __restrict__ w,
                                     float* __restrict__ out) {
    int g = blockIdx.x;
    int t = blockIdx.y;
    int d = threadIdx.x;
    const float* xp = x + (t * 4 + g) * 2560;
    float part = 0.0f;
    for (int i = d; i < 2560; i += 256) part += xp[i] * xp[i];
    __shared__ float red[256];
    red[d] = part;
    __syncthreads();
    for (int st = 128; st > 0; st >>= 1) {
        if (d < st) red[d] += red[d + st];
        __syncthreads();
    }
    float rms = rsqrtf(red[0] / 2560.0f + 1e-6f);
    for (int i = d; i < 2560; i += 256) {
        out[(t * 4 + g) * 2560 + i] = xp[i] * rms * (1.0f + w[g * 2560 + i]);
    }
}

extern "C" __global__ void silu_div4(const float* __restrict__ x,
                                     float* __restrict__ out,
                                     const int* __restrict__ n_p) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= *n_p) return;
    float v = x[i] * 0.25f;
    out[i] = v / (1.0f + expf(-v));
}

extern "C" __global__ void sigmoid_el(float* __restrict__ x,
                                      const int* __restrict__ n_p) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= *n_p) return;
    x[i] = 1.0f / (1.0f + expf(-x[i]));
}

extern "C" __global__ void sig2_div4(const float* __restrict__ x,
                                     float* __restrict__ out,
                                     const int* __restrict__ n_p) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= *n_p) return;
    out[i] = 2.0f / (1.0f + expf(-x[i] * 0.25f));
}

extern "C" __global__ void mix_streams(const float* __restrict__ mixw,
                                       const float* __restrict__ normed,
                                       float* __restrict__ out) {
    int c = blockIdx.x * 256 + threadIdx.x;
    int t = blockIdx.y;
    float acc = 0.0f;
    for (int g = 0; g < 4; g++) {
        acc += mixw[t * 10240 + g * 2560 + c] * normed[t * 10240 + g * 2560 + c];
    }
    out[t * 2560 + c] = acc * 0.25f;
}

extern "C" __global__ void inject_residual(const float* __restrict__ base,
                                           const float* __restrict__ mix,
                                           const float* __restrict__ injw,
                                           float* __restrict__ out) {
    int g = blockIdx.x;
    int tc = blockIdx.y;
    int t = tc / 10;
    int c = (tc % 10) * 256 + threadIdx.x;
    out[(t * 4 + g) * 2560 + c] =
        base[(t * 4 + g) * 2560 + c] + mix[t * 2560 + c] * injw[t * 4 + g];
}

extern "C" __global__ void silu_mul640(const float* __restrict__ h1,
                                       float* __restrict__ h2) {
    int j = blockIdx.x * 256 + threadIdx.x;
    int t = blockIdx.y;
    if (j >= 640) return;
    float gate = h1[t * 1280 + j];
    h2[t * 640 + j] = (gate / (1.0f + expf(-gate))) * h1[t * 1280 + 640 + j];
}

extern "C" __global__ void acc_scale(const float* __restrict__ x,
                                     const float* __restrict__ w,
                                     float* __restrict__ y) {
    int c = blockIdx.x * 256 + threadIdx.x;
    int t = blockIdx.y;
    y[t * 2560 + c] += (*w) * x[t * 2560 + c];
}

extern "C" __global__ void gate_shared(const float* __restrict__ s,
                                       const float* __restrict__ sg,
                                       float* __restrict__ y) {
    int c = blockIdx.x * 256 + threadIdx.x;
    float g = sg[0];
    y[c] += (1.0f / (1.0f + expf(-g))) * s[c];
}

// ---- GDN (probe 6 math, unchanged) ----
extern "C" __global__ void conv_silu(const float* __restrict__ in_,
                                     const float* __restrict__ w,
                                     float* __restrict__ out,
                                     const int* __restrict__ t_p) {
    int tt = *t_p;
    int ch = blockIdx.x;
    int t = threadIdx.x;
    if (t >= tt) return;
    float acc = 0.0f;
    for (int k = 0; k < 4; k++) {
        int src_t = t + k - 3;
        if (src_t >= 0) acc += w[ch * 4 + k] * in_[ch * tt + src_t];
    }
    out[ch * tt + t] = acc / (1.0f + expf(-acc));
}

extern "C" __global__ void l2norm_repeat(const float* __restrict__ q_in,
                                         const float* __restrict__ k_in,
                                         float* __restrict__ q_out,
                                         float* __restrict__ k_out) {
    int vhead = blockIdx.x;
    int t = blockIdx.y;
    int d = threadIdx.x;
    int khead = vhead / 3;
    const float* qp = q_in + (t * 16 + khead) * 128;
    const float* kp = k_in + (t * 16 + khead) * 128;
    float qn = 0.0f, kn = 0.0f;
    for (int i = 0; i < 128; i++) { qn += qp[i] * qp[i]; kn += kp[i] * kp[i]; }
    qn = rsqrtf(qn + 1e-6f); kn = rsqrtf(kn + 1e-6f);
    q_out[(t * 48 + vhead) * 128 + d] = qp[d] * qn * rsqrtf(128.0f);
    k_out[(t * 48 + vhead) * 128 + d] = kp[d] * kn;
}

extern "C" __global__ void beta_g(const float* __restrict__ b_pr,
                                  const float* __restrict__ a_pr,
                                  const float* __restrict__ a_log,
                                  const float* __restrict__ dt_bias,
                                  float* __restrict__ beta_out,
                                  float* __restrict__ g_out) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    float beta = 1.0f / (1.0f + expf(-b_pr[i]));
    float a = a_pr[i] + dt_bias[i % 48];
    float sp = logf(1.0f + expf(a));
    beta_out[i] = beta;
    g_out[i] = -expf(a_log[i % 48]) * sp;
}

extern "C" __global__ void delta_rule(const float* __restrict__ q,
                                      const float* __restrict__ k,
                                      const float* __restrict__ v,
                                      const float* __restrict__ g,
                                      const float* __restrict__ beta,
                                      float* __restrict__ out,
                                      const int* __restrict__ steps_p) {
    int steps = *steps_p;
    int head = blockIdx.x;
    extern __shared__ float state[];
    for (int i = threadIdx.x; i < 128 * 128; i += blockDim.x) state[i] = 0.0f;
    __syncthreads();
    for (int t = 0; t < steps; t++) {
        int d = threadIdx.x;
        float g_t = expf(g[t * 48 + head]);
        float beta_t = beta[t * 48 + head];
        const float* qt = q + (t * 48 + head) * 128;
        const float* kt = k + (t * 48 + head) * 128;
        const float* vt = v + (t * 48 + head) * 128;
        for (int dk = 0; dk < 128; dk++) state[dk * 128 + d] *= g_t;
        float kv = 0.0f;
        for (int dk = 0; dk < 128; dk++) kv += state[dk * 128 + d] * kt[dk];
        float delta = (vt[d] - kv) * beta_t;
        for (int dk = 0; dk < 128; dk++) state[dk * 128 + d] += kt[dk] * delta;
        float o = 0.0f;
        for (int dk = 0; dk < 128; dk++) o += state[dk * 128 + d] * qt[dk];
        out[(t * 48 + head) * 128 + d] = o;
    }
}

extern "C" __global__ void rmsnorm_gated(const float* __restrict__ x,
                                         const float* __restrict__ z,
                                         const float* __restrict__ w,
                                         float* __restrict__ out) {
    int vhead = blockIdx.x;
    int t = blockIdx.y;
    int d = threadIdx.x;
    const float* xp = x + ((t * 48 + vhead) * 128);
    __shared__ float red[128];
    red[d] = xp[d] * xp[d];
    __syncthreads();
    for (int st = 64; st > 0; st >>= 1) {
        if (d < st) red[d] += red[d + st];
        __syncthreads();
    }
    float rms = rsqrtf(red[0] / 128.0f + 1e-6f);
    float gate = 1.0f / (1.0f + expf(-z[(t * 48 + vhead) * 128 + d]));
    out[(t * 48 + vhead) * 128 + d] = w[d] * xp[d] * rms * gate;
}
"#;

fn ck(r: CUresult) {
    if r != CUresult::CUDA_SUCCESS {
        panic!("CUDA error: {r:?}");
    }
}

// ---- CNQ container ----

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

    /// exact tensor by name within a section (the mtp section carries its own
    /// layers.0 copy — always filter by section)
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
    let cnq_path = "../converter/Qwen3.8-Flash-Next-CNQ4.5.cnq";
    println!("p10: opening CNQ4.5 container …");
    let mut cnq = Cnq::open(cnq_path);
    let L = |sub: &str| format!("model.language_model.layers.0.{sub}");
    const SEC: &str = "text";

    let t_load = Instant::now();

    // weight bundle: (name, kind, payload) — kind: Fp4(raw bytes) | F32(exact)
    enum W {
        Fp4(Vec<u8>, f32),
        F32(Vec<f32>),
    }
    let mut w: Vec<(String, W)> = Vec::new();
    let mut load = |name: String, sub: &str| {
        let t = cnq.find(&name, SEC).clone();
        let payload = cnq.read_bytes(&t);
        let kind = if t["dtype"].as_str().unwrap() == "bf16" {
            W::F32(bf16_bytes_to_f32(&payload))
        } else {
            W::Fp4(payload, t["global_scale"].as_f64().unwrap() as f32)
        };
        w.push((sub.to_string(), kind));
    };
    for hc in ["attn_hyper_connection", "mlp_hyper_connection"] {
        for t in [
            "hc_norm.weight",
            "input_mix_weight_down.weight",
            "input_mix_weight_up.weight",
            "block_inject_weight.weight",
        ] {
            load(L(&format!("{hc}.{t}")), &format!("{hc}.{t}"));
        }
    }
    for t in [
        "linear_attn.in_proj_qkv.weight",
        "linear_attn.conv1d.weight",
        "linear_attn.in_proj_z.weight",
        "linear_attn.in_proj_b.weight",
        "linear_attn.in_proj_a.weight",
        "linear_attn.A_log",
        "linear_attn.dt_bias",
        "linear_attn.norm.weight",
        "linear_attn.out_proj.weight",
        "mlp.gate.weight",
        "mlp.shared_expert.gate_proj.weight",
        "mlp.shared_expert.up_proj.weight",
        "mlp.shared_expert.down_proj.weight",
        "mlp.shared_expert_gate.weight",
    ] {
        load(L(t), t);
    }
    // expert tensors (whole slabs, FP4)
    let gt = cnq.find(&L("mlp.experts.gate_up_proj"), SEC).clone();
    let dt = cnq.find(&L("mlp.experts.down_proj"), SEC).clone();
    let gu_raw = cnq.read_bytes(&gt);
    let dn_raw = cnq.read_bytes(&dt);
    let gu_gs = gt["global_scale"].as_f64().unwrap() as f32;
    let dn_gs = dt["global_scale"].as_f64().unwrap() as f32;
    let gu_slab_bytes = gu_raw.len() / E;
    let dn_slab_bytes = dn_raw.len() / E;
    assert_eq!(gu_slab_bytes, 2 * INTER * H / 64 * 36);
    assert_eq!(dn_slab_bytes, H * INTER / 64 * 36);
    println!(
        "p10: 22 per-name tensors + expert FP4 slabs ({} MB gate_up, {} MB down) in {:.1} s",
        gu_raw.len() / (1 << 20),
        dn_raw.len() / (1 << 20),
        t_load.elapsed().as_secs_f64()
    );

    let mut x_in = read_bin_f32("../oracle/golden/layer0-input.f32", T * HCT);
    let golden = read_bin_f32("../oracle/golden/layer0-golden-output.f32", T * HCT);

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
        let f_dqflat = get_fn("dequant_fp4_flat");
        let f_gemv = get_fn("gemv_f32");
        let f_rmsg = get_fn("rms_group");
        let f_silu4 = get_fn("silu_div4");
        let f_sig = get_fn("sigmoid_el");
        let f_sig2 = get_fn("sig2_div4");
        let f_mix = get_fn("mix_streams");
        let f_inj = get_fn("inject_residual");
        let f_smul = get_fn("silu_mul640");
        let f_acc = get_fn("acc_scale");
        let f_gsh = get_fn("gate_shared");
        let f_conv = get_fn("conv_silu");
        let f_l2 = get_fn("l2norm_repeat");
        let f_bg = get_fn("beta_g");
        let f_delta = get_fn("delta_rule");
        let f_rmsgt = get_fn("rmsnorm_gated");

        // scalar params via device buffers (p5 lesson)
        let mut n320 = to_i32_dev(&[LOWRANK as i32]);
        let mut n2560 = to_i32_dev(&[H as i32]);
        let mut n640 = to_i32_dev(&[INTER as i32]);
        let mut n10240 = to_i32_dev(&[HCT as i32]);
        let mut n6144 = to_i32_dev(&[GDN_VAL as i32]);
        let mut nt_lowrank = to_i32_dev(&[(T * LOWRANK) as i32]);
        let mut nt_hct = to_i32_dev(&[(T * HCT) as i32]);
        let mut nt_hc = to_i32_dev(&[(T * HC) as i32]);
        let mut t_param = to_i32_dev(&[T as i32]);
        let mut steps_param = to_i32_dev(&[T as i32]);
        let mut n_conv = to_i32_dev(&[(GDN_CONV * CONV_K) as i32]);

        // upload bundle: fp4 bytes + gs stay on device; f32 tensors as f32
        let mut dw_fp4: Vec<(String, CUdeviceptr, CUdeviceptr)> = Vec::new(); // name, raw, gs
        let mut dw_f32: Vec<(String, CUdeviceptr)> = Vec::new();
        for (name, kind) in &w {
            match kind {
                W::Fp4(raw, gs) => {
                    let raw_dev = upload_dev(raw);
                    let gs_dev = to_f32_dev(&[*gs]);
                    dw_fp4.push((name.clone(), raw_dev, gs_dev));
                }
                W::F32(v) => dw_f32.push((name.clone(), to_f32_dev(v))),
            }
        }
        let fp4 = |name: &str| -> (CUdeviceptr, CUdeviceptr) {
            dw_fp4
                .iter()
                .find(|(n, _, _)| n == name)
                .map(|(_, r, g)| (*r, *g))
                .unwrap_or_else(|| panic!("fp4 tensor missing: {name}"))
        };
        let f32w = |name: &str| -> CUdeviceptr {
            dw_f32
                .iter()
                .find(|(n, _)| n == name)
                .map(|(_, d)| *d)
                .unwrap_or_else(|| panic!("f32 tensor missing: {name}"))
        };

        let (mut r_qkv, mut g_qkv) = fp4("linear_attn.in_proj_qkv.weight");
        let (mut r_conv, mut g_conv) = fp4("linear_attn.conv1d.weight");
        let (mut r_z, mut g_z) = fp4("linear_attn.in_proj_z.weight");
        let (mut r_b, mut g_b) = fp4("linear_attn.in_proj_b.weight");
        let (mut r_a, mut g_a) = fp4("linear_attn.in_proj_a.weight");
        let mut w_alog = f32w("linear_attn.A_log");
        let mut w_dt = f32w("linear_attn.dt_bias");
        let mut w_gnorm = f32w("linear_attn.norm.weight");
        let (mut r_gout, mut g_gout) = fp4("linear_attn.out_proj.weight");
        let mut w_router = f32w("mlp.gate.weight");
        let (mut r_sg, mut g_sg) = fp4("mlp.shared_expert.gate_proj.weight");
        let (mut r_su, mut g_su) = fp4("mlp.shared_expert.up_proj.weight");
        let (mut r_sdn, mut g_sdn) = fp4("mlp.shared_expert.down_proj.weight");
        let mut w_sgate = f32w("mlp.shared_expert_gate.weight");
        let (mut r_ahc_down, mut g_ahc_down) = fp4("attn_hyper_connection.input_mix_weight_down.weight");
        let (mut r_ahc_up, mut g_ahc_up) = fp4("attn_hyper_connection.input_mix_weight_up.weight");
        let (mut r_ahc_inj, mut g_ahc_inj) = fp4("attn_hyper_connection.block_inject_weight.weight");
        let (mut r_mhc_down, mut g_mhc_down) = fp4("mlp_hyper_connection.input_mix_weight_down.weight");
        let (mut r_mhc_up, mut g_mhc_up) = fp4("mlp_hyper_connection.input_mix_weight_up.weight");
        let (mut r_mhc_inj, mut g_mhc_inj) = fp4("mlp_hyper_connection.block_inject_weight.weight");
        let mut w_ahc_norm = f32w("attn_hyper_connection.hc_norm.weight");
        let mut w_mhc_norm = f32w("mlp_hyper_connection.hc_norm.weight");

        // conv1d: flat dequant to f32 at load (row layout [ch][4] restored)
        let mut conv_f32_dev = alloc_zeroed(GDN_CONV * CONV_K * 4);
        launch(
            f_dqflat,
            ((GDN_CONV * CONV_K) as u32 + 255) / 256,
            1,
            256,
            1,
            0,
            &mut [
                &mut r_conv as *mut _ as *mut _,
                &mut g_conv as *mut _ as *mut _,
                &mut conv_f32_dev as *mut _ as *mut _,
                &mut n_conv as *mut _ as *mut _,
            ],
        );
        let mut w_conv = conv_f32_dev;

        // experts: one device buffer each, expert e at byte offset e*slab
        let mut gu_dev = upload_dev(&gu_raw);
        let mut dn_dev = upload_dev(&dn_raw);
        let mut gu_gs_dev = to_f32_dev(&[gu_gs]);
        let mut dn_gs_dev = to_f32_dev(&[dn_gs]);
        println!("p10: weights on device ({:.0} MB FP4 experts), t={:.1} s",
            (gu_raw.len() + dn_raw.len()) as f64 / (1 << 20) as f64,
            t_load.elapsed().as_secs_f64());

        // ---- stage A: gemv_fp4 self-check vs CPU dequant ----
        {
            let row = 0usize;
            let k_dim = H;
            let bpr = k_dim / 64;
            let row_bytes = bpr * 36;
            let qkv_host = {
                let t = cnq.find(&L("linear_attn.in_proj_qkv.weight"), SEC).clone();
                let off = cnq.blob_offset + t["offset"].as_u64().unwrap() + (row * row_bytes) as u64;
                cnq.file.seek(SeekFrom::Start(off)).unwrap();
                let mut raw = vec![0u8; row_bytes];
                cnq.file.read_exact(&mut raw).unwrap();
                raw
            };
            let x: Vec<f32> = (0..k_dim).map(|i| ((i * 7 % 13) as f32 - 6.0) * 0.25).collect();
            let gs = {
                let t = cnq.find(&L("linear_attn.in_proj_qkv.weight"), SEC).clone();
                t["global_scale"].as_f64().unwrap() as f32
            };
            let mut acc = 0f32;
            for b in 0..bpr {
                let blk = &qkv_host[b * 36..(b + 1) * 36];
                for sb in 0..4 {
                    let sc = ue4m3(blk[sb] as u32) * gs;
                    let mut part = 0f32;
                    for j in 0..16 {
                        let idx = sb * 16 + j;
                        let byte = blk[4 + (idx >> 1)];
                        let nib = if idx & 1 == 1 { (byte >> 4) & 0xF } else { byte & 0xF };
                        part += e2m1(nib as u32) * x[b * 64 + sb * 16 + j];
                    }
                    acc += part * sc;
                }
            }
            let mut x_dev = to_f32_dev(&x);
            let mut y_dev = alloc_zeroed(4);
            let mut row_dev = upload_dev(&qkv_host);
            launch(
                f_gemv4,
                1,
                1,
                256,
                1,
                0,
                &mut [
                    &mut row_dev as *mut _ as *mut _,
                    &mut x_dev as *mut _ as *mut _,
                    &mut g_qkv as *mut _ as *mut _,
                    &mut y_dev as *mut _ as *mut _,
                    &mut n2560 as *mut _ as *mut _,
                ],
            );
            let gpu_y = dtoh(y_dev, 1);
            let rel = ((gpu_y[0] - acc).abs() / acc.abs().max(1e-9)) as f64;
            println!("p10: stage A gemv_fp4 vs CPU dequant (row 0, in_proj_qkv): gpu={:.6} cpu={:.6} rel={rel:.2e}", gpu_y[0], acc);
            assert!(rel < 1e-4, "gemv_fp4 diverges from CPU dequant");
            println!("p10: stage A PASS");
        }

        let t_start = Instant::now();
        let mut x0 = to_f32_dev(&x_in);

        // ---- GatedResidual (HC block) ----
        let mut run_hc = move |mut wdown: (CUdeviceptr, CUdeviceptr),
                           mut wup: (CUdeviceptr, CUdeviceptr),
                           mut winj: (CUdeviceptr, CUdeviceptr),
                           mut norm: CUdeviceptr,
                           mut x_in_dev: u64|
         -> (CUdeviceptr, CUdeviceptr) {
            unsafe {
                let mut normed = alloc_zeroed(T * HCT * 4);
                launch(
                    f_rmsg,
                    4,
                    T as u32,
                    256,
                    1,
                    0,
                    &mut [
                        &mut x_in_dev as *mut _ as *mut _,
                        &mut norm as *mut _ as *mut _,
                        &mut normed as *mut _ as *mut _,
                    ],
                );
                let mut low = alloc_zeroed(T * LOWRANK * 4);
                for t in 0..T {
                    let mut xt = normed + (t * HCT * 4) as u64;
                    let mut yt = low + (t * LOWRANK * 4) as u64;
                    launch(
                        f_gemv4,
                        LOWRANK as u32,
                        1,
                        256,
                        1,
                        0,
                        &mut [
                            &mut wdown.0 as *mut _ as *mut _,
                            &mut xt as *mut _ as *mut _,
                            &mut wdown.1 as *mut _ as *mut _,
                            &mut yt as *mut _ as *mut _,
                            &mut n10240 as *mut _ as *mut _,
                        ],
                    );
                }
                let mut sil = alloc_zeroed(T * LOWRANK * 4);
                launch(
                    f_silu4,
                    ((T * LOWRANK) as u32 + 255) / 256,
                    1,
                    256,
                    1,
                    0,
                    &mut [
                        &mut low as *mut _ as *mut _,
                        &mut sil as *mut _ as *mut _,
                        &mut nt_lowrank as *mut _ as *mut _,
                    ],
                );
                let mut mixw = alloc_zeroed(T * HCT * 4);
                for t in 0..T {
                    let mut xt = sil + (t * LOWRANK * 4) as u64;
                    let mut yt = mixw + (t * HCT * 4) as u64;
                    launch(
                        f_gemv4,
                        HCT as u32,
                        1,
                        256,
                        1,
                        0,
                        &mut [
                            &mut wup.0 as *mut _ as *mut _,
                            &mut xt as *mut _ as *mut _,
                            &mut wup.1 as *mut _ as *mut _,
                            &mut yt as *mut _ as *mut _,
                            &mut n320 as *mut _ as *mut _,
                        ],
                    );
                }
                launch(
                    f_sig,
                    ((T * HCT) as u32 + 255) / 256,
                    1,
                    256,
                    1,
                    0,
                    &mut [
                        &mut mixw as *mut _ as *mut _,
                        &mut nt_hct as *mut _ as *mut _,
                    ],
                );
                let mut mixed = alloc_zeroed(T * H * 4);
                launch(
                    f_mix,
                    10,
                    T as u32,
                    256,
                    1,
                    0,
                    &mut [
                        &mut mixw as *mut _ as *mut _,
                        &mut normed as *mut _ as *mut _,
                        &mut mixed as *mut _ as *mut _,
                    ],
                );
                let mut injr = alloc_zeroed(T * HC * 4);
                for t in 0..T {
                    let mut xt = normed + (t * HCT * 4) as u64;
                    let mut yt = injr + (t * HC * 4) as u64;
                    launch(
                        f_gemv4,
                        HC as u32,
                        1,
                        256,
                        1,
                        0,
                        &mut [
                            &mut winj.0 as *mut _ as *mut _,
                            &mut xt as *mut _ as *mut _,
                            &mut winj.1 as *mut _ as *mut _,
                            &mut yt as *mut _ as *mut _,
                            &mut n10240 as *mut _ as *mut _,
                        ],
                    );
                }
                let mut injw = alloc_zeroed(T * HC * 4);
                launch(
                    f_sig2,
                    1,
                    1,
                    (T * HC) as u32,
                    1,
                    0,
                    &mut [
                        &mut injr as *mut _ as *mut _,
                        &mut injw as *mut _ as *mut _,
                        &mut nt_hc as *mut _ as *mut _,
                    ],
                );
                (mixed, injw)
            }
        };

        // ---- GDN sub-block (p6 chain, FP4 GEMVs) ----
        let mut run_gdn = move |mixed: CUdeviceptr| -> CUdeviceptr {
            unsafe {
                let mut mq = alloc_zeroed(T * GDN_CONV * 4);
                for t in 0..T {
                    let mut xt = mixed + (t * H * 4) as u64;
                    let mut yt = mq + (t * GDN_CONV * 4) as u64;
                    launch(
                        f_gemv4,
                        GDN_CONV as u32,
                        1,
                        256,
                        1,
                        0,
                        &mut [
                            &mut r_qkv as *mut _ as *mut _,
                            &mut xt as *mut _ as *mut _,
                            &mut g_qkv as *mut _ as *mut _,
                            &mut yt as *mut _ as *mut _,
                            &mut n2560 as *mut _ as *mut _,
                        ],
                    );
                }
                let mq_host = dtoh(mq, T * GDN_CONV);
                let mut mq_t = vec![0f32; T * GDN_CONV];
                for i in 0..T {
                    for j in 0..GDN_CONV {
                        mq_t[j * T + i] = mq_host[i * GDN_CONV + j];
                    }
                }
                let mut mq_t_dev = to_f32_dev(&mq_t);
                let mut conv_out = alloc_zeroed(T * GDN_CONV * 4);
                launch(
                    f_conv,
                    GDN_CONV as u32,
                    1,
                    T as u32,
                    1,
                    0,
                    &mut [
                        &mut mq_t_dev as *mut _ as *mut _,
                        &mut w_conv as *mut _ as *mut _,
                        &mut conv_out as *mut _ as *mut _,
                        &mut t_param as *mut _ as *mut _,
                    ],
                );
                let conv_h = dtoh(conv_out, T * GDN_CONV);
                let mut conv_t = vec![0f32; T * GDN_CONV];
                for i in 0..GDN_CONV {
                    for j in 0..T {
                        conv_t[j * GDN_CONV + i] = conv_h[i * T + j];
                    }
                }
                let mut q_dev = alloc_zeroed(T * GDN_KEY * 4);
                let mut k_dev = alloc_zeroed(T * GDN_KEY * 4);
                let mut v_dev = alloc_zeroed(T * GDN_VAL * 4);
                for t in 0..T {
                    let base = t * GDN_CONV;
                    let qh = &conv_t[base..base + GDN_KEY];
                    let kh = &conv_t[base + GDN_KEY..base + 2 * GDN_KEY];
                    let vh = &conv_t[base + 2 * GDN_KEY..base + GDN_CONV];
                    ck(sys::cuMemcpyHtoDAsync_v2(
                        q_dev + (t * GDN_KEY * 4) as u64,
                        qh.as_ptr() as *const std::ffi::c_void,
                        GDN_KEY * 4,
                        std::ptr::null_mut(),
                    ));
                    ck(sys::cuMemcpyHtoDAsync_v2(
                        k_dev + (t * GDN_KEY * 4) as u64,
                        kh.as_ptr() as *const std::ffi::c_void,
                        GDN_KEY * 4,
                        std::ptr::null_mut(),
                    ));
                    ck(sys::cuMemcpyHtoDAsync_v2(
                        v_dev + (t * GDN_VAL * 4) as u64,
                        vh.as_ptr() as *const std::ffi::c_void,
                        GDN_VAL * 4,
                        std::ptr::null_mut(),
                    ));
                }
                ck(sys::cuStreamSynchronize(std::ptr::null_mut()));
                let mut z_dev = alloc_zeroed(T * GDN_VAL * 4);
                for t in 0..T {
                    let mut xt = mixed + (t * H * 4) as u64;
                    let mut yt = z_dev + (t * GDN_VAL * 4) as u64;
                    launch(
                        f_gemv4,
                        GDN_VAL as u32,
                        1,
                        256,
                        1,
                        0,
                        &mut [
                            &mut r_z as *mut _ as *mut _,
                            &mut xt as *mut _ as *mut _,
                            &mut g_z as *mut _ as *mut _,
                            &mut yt as *mut _ as *mut _,
                            &mut n2560 as *mut _ as *mut _,
                        ],
                    );
                }
                let mut b_dev = alloc_zeroed(T * 48 * 4);
                for t in 0..T {
                    let mut xt = mixed + (t * H * 4) as u64;
                    let mut yt = b_dev + (t * 48 * 4) as u64;
                    launch(
                        f_gemv4,
                        48,
                        1,
                        256,
                        1,
                        0,
                        &mut [
                            &mut r_b as *mut _ as *mut _,
                            &mut xt as *mut _ as *mut _,
                            &mut g_b as *mut _ as *mut _,
                            &mut yt as *mut _ as *mut _,
                            &mut n2560 as *mut _ as *mut _,
                        ],
                    );
                }
                let mut a_dev = alloc_zeroed(T * 48 * 4);
                for t in 0..T {
                    let mut xt = mixed + (t * H * 4) as u64;
                    let mut yt = a_dev + (t * 48 * 4) as u64;
                    launch(
                        f_gemv4,
                        48,
                        1,
                        256,
                        1,
                        0,
                        &mut [
                            &mut r_a as *mut _ as *mut _,
                            &mut xt as *mut _ as *mut _,
                            &mut g_a as *mut _ as *mut _,
                            &mut yt as *mut _ as *mut _,
                            &mut n2560 as *mut _ as *mut _,
                        ],
                    );
                }
                let mut beta_dev = alloc_zeroed(T * 48 * 4);
                let mut g_dev = alloc_zeroed(T * 48 * 4);
                launch(
                    f_bg,
                    1,
                    1,
                    (T * 48) as u32,
                    1,
                    0,
                    &mut [
                        &mut b_dev as *mut _ as *mut _,
                        &mut a_dev as *mut _ as *mut _,
                        &mut w_alog as *mut _ as *mut _,
                        &mut w_dt as *mut _ as *mut _,
                        &mut beta_dev as *mut _ as *mut _,
                        &mut g_dev as *mut _ as *mut _,
                    ],
                );
                let mut qr = alloc_zeroed(T * 48 * 128 * 4);
                let mut kr = alloc_zeroed(T * 48 * 128 * 4);
                launch(
                    f_l2,
                    48,
                    T as u32,
                    128,
                    1,
                    0,
                    &mut [
                        &mut q_dev as *mut _ as *mut _,
                        &mut k_dev as *mut _ as *mut _,
                        &mut qr as *mut _ as *mut _,
                        &mut kr as *mut _ as *mut _,
                    ],
                );
                ck(sys::cuFuncSetAttribute(
                    f_delta,
                    sys::CUfunction_attribute::CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
                    64 * 1024,
                ));
                let mut core = alloc_zeroed(T * GDN_VAL * 4);
                launch(
                    f_delta,
                    48,
                    1,
                    128,
                    1,
                    64 * 1024,
                    &mut [
                        &mut qr as *mut _ as *mut _,
                        &mut kr as *mut _ as *mut _,
                        &mut v_dev as *mut _ as *mut _,
                        &mut g_dev as *mut _ as *mut _,
                        &mut beta_dev as *mut _ as *mut _,
                        &mut core as *mut _ as *mut _,
                        &mut steps_param as *mut _ as *mut _,
                    ],
                );
                let mut normed = alloc_zeroed(T * GDN_VAL * 4);
                launch(
                    f_rmsgt,
                    48,
                    T as u32,
                    128,
                    1,
                    0,
                    &mut [
                        &mut core as *mut _ as *mut _,
                        &mut z_dev as *mut _ as *mut _,
                        &mut w_gnorm as *mut _ as *mut _,
                        &mut normed as *mut _ as *mut _,
                    ],
                );
                let mut out = alloc_zeroed(T * H * 4);
                for t in 0..T {
                    let mut xt = normed + (t * GDN_VAL * 4) as u64;
                    let mut yt = out + (t * H * 4) as u64;
                    launch(
                        f_gemv4,
                        H as u32,
                        1,
                        256,
                        1,
                        0,
                        &mut [
                            &mut r_gout as *mut _ as *mut _,
                            &mut xt as *mut _ as *mut _,
                            &mut g_gout as *mut _ as *mut _,
                            &mut yt as *mut _ as *mut _,
                            &mut n6144 as *mut _ as *mut _,
                        ],
                    );
                }
                out
            }
        };

        let (mut mixed_a, mut injw_a) = run_hc(
            (r_ahc_down, g_ahc_down),
            (r_ahc_up, g_ahc_up),
            (r_ahc_inj, g_ahc_inj),
            w_ahc_norm,
            x0,
        );
        let mut gdn_out = run_gdn(mixed_a);
        let mut x1 = alloc_zeroed(T * HCT * 4);
        launch(
            f_inj,
            4,
            (T * 10) as u32,
            256,
            1,
            0,
            &mut [
                &mut x0 as *mut _ as *mut _,
                &mut gdn_out as *mut _ as *mut _,
                &mut injw_a as *mut _ as *mut _,
                &mut x1 as *mut _ as *mut _,
            ],
        );
        let (mut mixed_m, mut injw_m) = run_hc(
            (r_mhc_down, g_mhc_down),
            (r_mhc_up, g_mhc_up),
            (r_mhc_inj, g_mhc_inj),
            w_mhc_norm,
            x1,
        );

        // ---- MoE ----
        let mut logits = alloc_zeroed(T * E * 4);
        for t in 0..T {
            let mut xt = mixed_m + (t * H * 4) as u64;
            let mut yt = logits + (t * E * 4) as u64;
            launch(
                f_gemv,
                E as u32,
                1,
                256,
                1,
                0,
                &mut [
                    &mut w_router as *mut _ as *mut _,
                    &mut xt as *mut _ as *mut _,
                    &mut yt as *mut _ as *mut _,
                    &mut n2560 as *mut _ as *mut _,
                ],
            );
        }
        let lg = dtoh(logits, T * E);
        let mut routing = vec![(0usize, 0f32); T * TOPK];
        for t in 0..T {
            let row = &lg[t * E..(t + 1) * E];
            let mx = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let mut probs: Vec<(usize, f32)> = row
                .iter()
                .enumerate()
                .map(|(i, v)| (i, (v - mx).exp()))
                .collect();
            let sum: f32 = probs.iter().map(|p| p.1).sum();
            for pr in probs.iter_mut() {
                pr.1 /= sum;
            }
            probs.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
            let sel_sum: f32 = probs[..TOPK].iter().map(|p| p.1).sum();
            for j in 0..TOPK {
                routing[t * TOPK + j] = (probs[j].0, probs[j].1 / sel_sum);
            }
        }
        drop(lg);

        let mut moe_out = alloc_zeroed(T * H * 4);
        for t in 0..T {
            let mut xt = mixed_m + (t * H * 4) as u64;
            let mut rowp = moe_out + (t * H * 4) as u64;
            let mut sg1 = alloc_zeroed(INTER * 4);
            let mut su1 = alloc_zeroed(INTER * 4);
            let mut y1 = sg1;
            launch(f_gemv4, INTER as u32, 1, 256, 1, 0, &mut [
                &mut r_sg as *mut _ as *mut _, &mut xt as *mut _ as *mut _,
                &mut g_sg as *mut _ as *mut _, &mut y1 as *mut _ as *mut _,
                &mut n2560 as *mut _ as *mut _]);
            let mut y2 = su1;
            launch(f_gemv4, INTER as u32, 1, 256, 1, 0, &mut [
                &mut r_su as *mut _ as *mut _, &mut xt as *mut _ as *mut _,
                &mut g_su as *mut _ as *mut _, &mut y2 as *mut _ as *mut _,
                &mut n2560 as *mut _ as *mut _]);
            let g1 = dtoh(sg1, INTER);
            let u1 = dtoh(su1, INTER);
            let mut h2s = vec![0f32; INTER];
            for j in 0..INTER {
                h2s[j] = (g1[j] / (1.0 + (-g1[j]).exp())) * u1[j];
            }
            let mut h2s_dev = to_f32_dev(&h2s);
            let mut sdown = alloc_zeroed(H * 4);
            launch(f_gemv4, H as u32, 1, 256, 1, 0, &mut [
                &mut r_sdn as *mut _ as *mut _, &mut h2s_dev as *mut _ as *mut _,
                &mut g_sdn as *mut _ as *mut _, &mut sdown as *mut _ as *mut _,
                &mut n640 as *mut _ as *mut _]);
            let mut sgv = alloc_zeroed(4);
            launch(f_gemv, 1, 1, 256, 1, 0, &mut [
                &mut w_sgate as *mut _ as *mut _, &mut xt as *mut _ as *mut _,
                &mut sgv as *mut _ as *mut _, &mut n2560 as *mut _ as *mut _]);
            launch(f_gsh, 10, 1, 256, 1, 0, &mut [
                &mut sdown as *mut _ as *mut _, &mut sgv as *mut _ as *mut _,
                &mut rowp as *mut _ as *mut _]);
        }

        let mut h1 = alloc_zeroed(2 * INTER * 4);
        let mut h2 = alloc_zeroed(INTER * 4);
        let mut eo = alloc_zeroed(H * 4);
        for t in 0..T {
            for j in 0..TOPK {
                let (eid, rw) = routing[t * TOPK + j];
                let mut xt = mixed_m + (t * H * 4) as u64;
                let mut rowp = moe_out + (t * H * 4) as u64;
                let mut gu_e = gu_dev + (eid * gu_slab_bytes) as u64;
                let mut y1 = h1;
                launch(f_gemv4, (2 * INTER) as u32, 1, 256, 1, 0, &mut [
                    &mut gu_e as *mut _ as *mut _, &mut xt as *mut _ as *mut _,
                    &mut gu_gs_dev as *mut _ as *mut _, &mut y1 as *mut _ as *mut _,
                    &mut n2560 as *mut _ as *mut _]);
                launch(f_smul, 3, 1, 256, 1, 0, &mut [
                    &mut h1 as *mut _ as *mut _, &mut h2 as *mut _ as *mut _]);
                let mut dn_e = dn_dev + (eid * dn_slab_bytes) as u64;
                let mut y2 = eo;
                launch(f_gemv4, H as u32, 1, 256, 1, 0, &mut [
                    &mut dn_e as *mut _ as *mut _, &mut h2 as *mut _ as *mut _,
                    &mut dn_gs_dev as *mut _ as *mut _, &mut y2 as *mut _ as *mut _,
                    &mut n640 as *mut _ as *mut _]);
                let mut wdev = to_f32_dev(&[rw]);
                launch(f_acc, 10, 1, 256, 1, 0, &mut [
                    &mut eo as *mut _ as *mut _, &mut wdev as *mut _ as *mut _,
                    &mut rowp as *mut _ as *mut _]);
            }
        }

        let mut out_dev = alloc_zeroed(T * HCT * 4);
        launch(
            f_inj,
            4,
            (T * 10) as u32,
            256,
            1,
            0,
            &mut [
                &mut x1 as *mut _ as *mut _,
                &mut moe_out as *mut _ as *mut _,
                &mut injw_m as *mut _ as *mut _,
                &mut out_dev as *mut _ as *mut _,
            ],
        );

        let gpu_ms = t_start.elapsed().as_secs_f64() * 1e3;

        // ---- MEASUREMENT vs f32 golden ----
        let gpu = dtoh(out_dev, T * HCT);
        let mut nan = 0usize;
        let mut max_abs = 0.0f32;
        let mut sum_abs = 0.0f64;
        let mut sum_sq_d = 0.0f64;
        let mut sum_sq_g = 0.0f64;
        let mut argmax = 0usize;
        for i in 0..T * HCT {
            if gpu[i].is_nan() {
                nan += 1;
            }
            let d = (gpu[i] - golden[i]).abs();
            if d > max_abs {
                max_abs = d;
                argmax = i;
            }
            sum_abs += d as f64;
            sum_sq_d += (d as f64) * (d as f64);
            sum_sq_g += (golden[i] as f64) * (golden[i] as f64);
        }
        for t in 0..T {
            let mut tmax = 0.0f32;
            for j in 0..HCT {
                tmax = tmax.max((gpu[t * HCT + j] - golden[t * HCT + j]).abs());
            }
            println!("  token {t}: max_abs={tmax:.3e}");
        }
        let rel_l2 = (sum_sq_d / sum_sq_g).sqrt();
        println!(
            "p10: FP4 layer vs f32 golden: max_abs={max_abs:.3e} at ({}, {}) mean_abs={:.3e} rel_L2={rel_l2:.3e} NaN={nan}",
            argmax / HCT,
            argmax % HCT,
            sum_abs / (T * HCT) as f64
        );
        println!("p10: layer pipeline (sync-per-launch) T=8: {gpu_ms:.2} ms");
        if nan == 0 {
            println!("p10: DONE — FP4 production-precision layer measurement (quality judged by ten-task gate at #11)");
        } else {
            println!("p10: FAIL — NaN in output");
            std::process::exit(1);
        }
    }
}

const CONV_K: usize = 4;
