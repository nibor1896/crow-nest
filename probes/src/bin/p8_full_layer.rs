//! Probe 8 (Crow #10): FULL text decoder layer 0 (GDN layer) on GPU, f32, vs the
//! transformers reference golden (`oracle/golden/layer0-input.f32` = hidden
//! [1,8,10240], `layer0-golden-output.f32`). Composition ported 1:1 from
//! `Qwen4ExpTextDecoderLayer.forward` (transformers 5.16.1):
//!
//!   x [1,8,10240] — no PLE on layer 0
//!   → attn_hyper_connection (GatedResidual): grouped RMSNorm (4×2560, eps 1e-6,
//!     weight 1+w) → low-rank mix down[10240→320], silu(·/4), up[320→10240],
//!     sigmoid → per-stream mix weights; mixed = mean over 4 streams; inject
//!     weights = 2·sigmoid(inj(·)/4)
//!   → linear_attn (p6 GDN kernels, gated delta net)
//!   → hyper_input + mixer_out ⊗ inject_weights  → [1,8,10240]
//!   → mlp_hyper_connection (same construction)
//!   → MoE: router GEMM [512,2560], softmax f32, top-10 (HOST for the probe —
//!     engine moves it on-GPU), weights normalized (norm_topk_prob=true);
//!     experts gate_up [512,1280,2560] chunk 640+640, silu·up, down
//!     [512,2560,640], weighted sum; shared expert (640) gated by
//!     sigmoid(shared_expert_gate) — weights from the REAL BF16 checkpoint → f32.
//!
//! Scalar kernel params in device buffers (p5 lesson); HtoD async + sync (WDDM).
//! Tolerance 5e-3 (f32 reorder + libm noise measured 3.6e-7 / 2.0e-6 on the
//! sub-block gates p6/p7).

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
// weight is [10240], indexed by group only (t-independent)
extern "C" __global__ void rms_group(const float* __restrict__ x,
                                     const float* __restrict__ w,
                                     float* __restrict__ out) {
    int g = blockIdx.x;      // 4 groups
    int t = blockIdx.y;
    int d = threadIdx.x;     // 256 threads
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

// out = silu(x / 4), elementwise over n (device ptr)
extern "C" __global__ void silu_div4(const float* __restrict__ x,
                                     float* __restrict__ out,
                                     const int* __restrict__ n_p) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= *n_p) return;
    float v = x[i] * 0.25f;
    out[i] = v / (1.0f + expf(-v));
}

// out = sigmoid(x), elementwise in place over n (device ptr)
extern "C" __global__ void sigmoid_el(float* __restrict__ x,
                                      const int* __restrict__ n_p) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= *n_p) return;
    x[i] = 1.0f / (1.0f + expf(-x[i]));
}

// out = 2·sigmoid(x / 4), elementwise over n (device ptr) — inject weights
extern "C" __global__ void sig2_div4(const float* __restrict__ x,
                                     float* __restrict__ out,
                                     const int* __restrict__ n_p) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= *n_p) return;
    out[i] = 2.0f / (1.0f + expf(-x[i] * 0.25f));
}

// mixed[t][c] = 0.25·Σ_g mixw[t][g·2560+c]·normed[t][g·2560+c]
extern "C" __global__ void mix_streams(const float* __restrict__ mixw,
                                       const float* __restrict__ normed,
                                       float* __restrict__ out) {
    int c = blockIdx.x * 256 + threadIdx.x;  // grid (10, T)
    int t = blockIdx.y;
    float acc = 0.0f;
    for (int g = 0; g < 4; g++) {
        acc += mixw[t * 10240 + g * 2560 + c] * normed[t * 10240 + g * 2560 + c];
    }
    out[t * 2560 + c] = acc * 0.25f;
}

// out[(t·4+g)·2560+c] = base[...] + mix[t][c]·injw[t·4+g]   grid (4, T·10)
extern "C" __global__ void inject_residual(const float* __restrict__ base,
                                           const float* __restrict__ mix,
                                           const float* __restrict__ injw,
                                           float* __restrict__ out) {
    int g = blockIdx.x;
    int tc = blockIdx.y;              // t·10 + chunk
    int t = tc / 10;
    int c = (tc % 10) * 256 + threadIdx.x;
    out[(t * 4 + g) * 2560 + c] =
        base[(t * 4 + g) * 2560 + c] + mix[t * 2560 + c] * injw[t * 4 + g];
}

// h2[t·640+j] = silu(h1[t·1280+j]) · h1[t·1280+640+j]   grid (3, T)
extern "C" __global__ void silu_mul640(const float* __restrict__ h1,
                                       float* __restrict__ h2) {
    int j = blockIdx.x * 256 + threadIdx.x;
    int t = blockIdx.y;
    if (j >= 640) return;
    float gate = h1[t * 1280 + j];
    h2[t * 640 + j] = (gate / (1.0f + expf(-gate))) * h1[t * 1280 + 640 + j];
}

// y[t·2560+c] += (*w)·x[t·2560+c]   grid (10, T); y = row pointer, t is 0
extern "C" __global__ void acc_scale(const float* __restrict__ x,
                                     const float* __restrict__ w,
                                     float* __restrict__ y) {
    int c = blockIdx.x * 256 + threadIdx.x;
    int t = blockIdx.y;
    y[t * 2560 + c] += (*w) * x[t * 2560 + c];
}

// y[·2560+c] += sigmoid(sg[0])·s[·2560+c]   grid (10, 1), y/s row pointers
extern "C" __global__ void gate_shared(const float* __restrict__ s,
                                       const float* __restrict__ sg,
                                       float* __restrict__ y) {
    int c = blockIdx.x * 256 + threadIdx.x;
    float g = sg[0];
    y[c] += (1.0f / (1.0f + expf(-g))) * s[c];
}

// ---- GDN (probe 6, unchanged math) ----
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

/// tensor location inside its safetensors shard
struct TensorRef {
    shard: String,
    dtype: String,
    start: usize, // byte offset of data within the shard
    nbytes: usize,
}

fn locate(models: &str, name: &str) -> TensorRef {
    let index: serde_json::Value = serde_json::from_slice(
        &std::fs::read(format!("{models}/model.safetensors.index.json")).unwrap(),
    )
    .unwrap();
    let shard = index["weight_map"][name].as_str().unwrap().to_string();
    let mut fh = std::fs::File::open(format!("{models}/{shard}")).unwrap();
    let mut n8 = [0u8; 8];
    fh.read_exact(&mut n8).unwrap();
    let hl = u64::from_le_bytes(n8) as usize;
    let mut hb = vec![0u8; hl];
    fh.read_exact(&mut hb).unwrap();
    let hdr: serde_json::Value = serde_json::from_slice(&hb).unwrap();
    let info = &hdr[name];
    let off = info["data_offsets"][0].as_u64().unwrap() as usize;
    let end = info["data_offsets"][1].as_u64().unwrap() as usize;
    TensorRef {
        shard,
        dtype: info["dtype"].as_str().unwrap().to_string(),
        start: 8 + hl + off,
        nbytes: end - off,
    }
}

fn read_tensor_f32(models: &str, name: &str) -> Vec<f32> {
    let r = locate(models, name);
    assert_eq!(r.dtype, "BF16", "{name}: expected BF16");
    let mut fh = std::fs::File::open(format!("{models}/{}", r.shard)).unwrap();
    fh.seek(SeekFrom::Start(r.start as u64)).unwrap();
    let mut raw = vec![0u8; r.nbytes];
    fh.read_exact(&mut raw).unwrap();
    raw.chunks_exact(2)
        .map(|c| f32::from_bits((u16::from_le_bytes([c[0], c[1]]) as u32) << 16))
        .collect()
}

/// raw BF16 bytes of one expert slab tensor (all experts, contiguous)
struct ExpertSlab {
    raw: Vec<u8>,
    bytes_per_expert: usize,
}

impl ExpertSlab {
    fn load(models: &str, name: &str, experts: usize) -> Self {
        let r = locate(models, name);
        assert_eq!(r.dtype, "BF16");
        assert_eq!(r.nbytes % experts, 0, "{name}: not expert-aligned");
        let mut fh = std::fs::File::open(format!("{models}/{}", r.shard)).unwrap();
        fh.seek(SeekFrom::Start(r.start as u64)).unwrap();
        let mut raw = vec![0u8; r.nbytes];
        fh.read_exact(&mut raw).unwrap();
        Self {
            raw,
            bytes_per_expert: r.nbytes / experts,
        }
    }
    fn expert_f32(&self, e: usize) -> Vec<f32> {
        self.raw[e * self.bytes_per_expert..(e + 1) * self.bytes_per_expert]
            .chunks_exact(2)
            .map(|c| f32::from_bits((u16::from_le_bytes([c[0], c[1]]) as u32) << 16))
            .collect()
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
    let L = |sub: &str| format!("model.language_model.layers.0.{sub}");

    println!("p8: loading layer-0 weights (BF16 → f32) …");
    let t_load = Instant::now();

    let mut w: Vec<(String, Vec<f32>)> = Vec::new();
    for hc in ["attn_hyper_connection", "mlp_hyper_connection"] {
        for t in [
            "hc_norm.weight",
            "input_mix_weight_down.weight",
            "input_mix_weight_up.weight",
            "block_inject_weight.weight",
        ] {
            w.push((format!("{hc}.{t}"), read_tensor_f32(models, &L(&format!("{hc}.{t}")))));
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
        w.push((t.to_string(), read_tensor_f32(models, &L(t))));
    }
    let gu_slab = ExpertSlab::load(models, &L("mlp.experts.gate_up_proj"), E);
    let dn_slab = ExpertSlab::load(models, &L("mlp.experts.down_proj"), E);
    assert_eq!(gu_slab.bytes_per_expert, 2 * INTER * H * 2);
    assert_eq!(dn_slab.bytes_per_expert, H * INTER * 2);
    println!(
        "p8: {} tensors + expert slabs ({} MB gate_up, {} MB down) loaded in {:.1} s",
        w.len(),
        gu_slab.raw.len() / (1 << 20),
        dn_slab.raw.len() / (1 << 20),
        t_load.elapsed().as_secs_f64()
    );

    let x_in = read_bin_f32("../oracle/golden/layer0-input.f32", T * HCT);
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
        let mut nt_lowrank = to_i32_dev(&[(T * LOWRANK) as i32]); // flat silu buffer [T*320]
        let mut nt_hct = to_i32_dev(&[(T * HCT) as i32]); // flat sigmoid buffer [T*10240]
        let mut nt_hc = to_i32_dev(&[(T * HC) as i32]); // flat injw buffer [T*4]
        let mut t_param = to_i32_dev(&[T as i32]);
        let mut steps_param = to_i32_dev(&[T as i32]);

        let mut dw: Vec<(String, CUdeviceptr)> = Vec::new();
        for (name, host) in &w {
            dw.push((name.clone(), to_f32_dev(host)));
        }
        let get = |name: &str| -> CUdeviceptr {
            dw.iter().find(|(n, _)| n == name).unwrap().1
        };
        let w_ahc_norm = get("attn_hyper_connection.hc_norm.weight");
        let w_ahc_down = get("attn_hyper_connection.input_mix_weight_down.weight");
        let w_ahc_up = get("attn_hyper_connection.input_mix_weight_up.weight");
        let w_ahc_inj = get("attn_hyper_connection.block_inject_weight.weight");
        let w_mhc_norm = get("mlp_hyper_connection.hc_norm.weight");
        let w_mhc_down = get("mlp_hyper_connection.input_mix_weight_down.weight");
        let w_mhc_up = get("mlp_hyper_connection.input_mix_weight_up.weight");
        let w_mhc_inj = get("mlp_hyper_connection.block_inject_weight.weight");
        let mut w_qkv = get("linear_attn.in_proj_qkv.weight");
        let mut w_conv = get("linear_attn.conv1d.weight");
        let mut w_z = get("linear_attn.in_proj_z.weight");
        let mut w_b = get("linear_attn.in_proj_b.weight");
        let mut w_a = get("linear_attn.in_proj_a.weight");
        let mut w_alog = get("linear_attn.A_log");
        let mut w_dt = get("linear_attn.dt_bias");
        let mut w_gnorm = get("linear_attn.norm.weight");
        let mut w_gout = get("linear_attn.out_proj.weight");
        let mut w_router = get("mlp.gate.weight");
        let mut w_sg = get("mlp.shared_expert.gate_proj.weight");
        let mut w_su = get("mlp.shared_expert.up_proj.weight");
        let mut w_sdn = get("mlp.shared_expert.down_proj.weight");
        let mut w_sgate = get("mlp.shared_expert_gate.weight");

        // all 512 experts in VRAM (f32)
        println!("p8: uploading 512 experts (f32) …");
        let t_up = Instant::now();
        let mut gu_dev: Vec<CUdeviceptr> = Vec::with_capacity(E);
        let mut dn_dev: Vec<CUdeviceptr> = Vec::with_capacity(E);
        for e in 0..E {
            gu_dev.push(to_f32_dev(&gu_slab.expert_f32(e)));
            dn_dev.push(to_f32_dev(&dn_slab.expert_f32(e)));
        }
        println!("p8: experts on device in {:.1} s", t_up.elapsed().as_secs_f64());

        let t_start = Instant::now();
        let mut x0 = to_f32_dev(&x_in);

        // ---- GatedResidual: (mixed, injw) from hyper_input ----
        let mut run_hc = move |mut norm: CUdeviceptr,
                      mut wdown: CUdeviceptr,
                      mut wup: CUdeviceptr,
                      mut winj: CUdeviceptr,
                      mut x_in_dev: CUdeviceptr|
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
                        f_gemv,
                        LOWRANK as u32,
                        1,
                        256,
                        1,
                        0,
                        &mut [
                            &mut wdown as *mut _ as *mut _,
                            &mut xt as *mut _ as *mut _,
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
                        f_gemv,
                        HCT as u32,
                        1,
                        256,
                        1,
                        0,
                        &mut [
                            &mut wup as *mut _ as *mut _,
                            &mut xt as *mut _ as *mut _,
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
                        f_gemv,
                        HC as u32,
                        1,
                        256,
                        1,
                        0,
                        &mut [
                            &mut winj as *mut _ as *mut _,
                            &mut xt as *mut _ as *mut _,
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

        // ---- GDN sub-block (p6 chain) on a [T][H] mixer input ----
        let mut run_gdn = move |mut mixed: CUdeviceptr| -> CUdeviceptr {
            unsafe {
                let mut mq = alloc_zeroed(T * GDN_CONV * 4);
                for t in 0..T {
                    let mut xt = mixed + (t * H * 4) as u64;
                    let mut yt = mq + (t * GDN_CONV * 4) as u64;
                    launch(
                        f_gemv,
                        GDN_CONV as u32,
                        1,
                        256,
                        1,
                        0,
                        &mut [
                            &mut w_qkv as *mut _ as *mut _,
                            &mut xt as *mut _ as *mut _,
                            &mut yt as *mut _ as *mut _,
                            &mut n2560 as *mut _ as *mut _,
                        ],
                    );
                }
                // conv in [chan][T] layout, transpose through host
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
                // split q/k/v (q | k | v)
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
                // z / b / a
                let mut z_dev = alloc_zeroed(T * GDN_VAL * 4);
                for t in 0..T {
                    let mut xt = mixed + (t * H * 4) as u64;
                    let mut yt = z_dev + (t * GDN_VAL * 4) as u64;
                    launch(
                        f_gemv,
                        GDN_VAL as u32,
                        1,
                        256,
                        1,
                        0,
                        &mut [
                            &mut w_z as *mut _ as *mut _,
                            &mut xt as *mut _ as *mut _,
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
                        f_gemv,
                        48,
                        1,
                        256,
                        1,
                        0,
                        &mut [
                            &mut w_b as *mut _ as *mut _,
                            &mut xt as *mut _ as *mut _,
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
                        f_gemv,
                        48,
                        1,
                        256,
                        1,
                        0,
                        &mut [
                            &mut w_a as *mut _ as *mut _,
                            &mut xt as *mut _ as *mut _,
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
                        f_gemv,
                        H as u32,
                        1,
                        256,
                        1,
                        0,
                        &mut [
                            &mut w_gout as *mut _ as *mut _,
                            &mut xt as *mut _ as *mut _,
                            &mut yt as *mut _ as *mut _,
                            &mut n6144 as *mut _ as *mut _,
                        ],
                    );
                }
                out
            }
        };

        let (mut mixed_a, mut injw_a) = run_hc(w_ahc_norm, w_ahc_down, w_ahc_up, w_ahc_inj, x0);
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
        let (mut mixed_m, mut injw_m) = run_hc(w_mhc_norm, w_mhc_down, w_mhc_up, w_mhc_inj, x1);

        // ---- MoE on mixed_m ----
        // router GEMV per token; softmax + top-10 + normalize on host (probe)
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

        // shared expert per token, then routed experts per (token, rank)
        let mut moe_out = alloc_zeroed(T * H * 4);
        for t in 0..T {
            let mut xt = mixed_m + (t * H * 4) as u64;
            let mut rowp = moe_out + (t * H * 4) as u64;
            let mut sg1 = alloc_zeroed(INTER * 4);
            let mut su1 = alloc_zeroed(INTER * 4);
            let mut y1 = sg1;
            launch(f_gemv, INTER as u32, 1, 256, 1, 0, &mut [
                &mut w_sg as *mut _ as *mut _, &mut xt as *mut _ as *mut _,
                &mut y1 as *mut _ as *mut _, &mut n2560 as *mut _ as *mut _]);
            let mut y2 = su1;
            launch(f_gemv, INTER as u32, 1, 256, 1, 0, &mut [
                &mut w_su as *mut _ as *mut _, &mut xt as *mut _ as *mut _,
                &mut y2 as *mut _ as *mut _, &mut n2560 as *mut _ as *mut _]);
            let g1 = dtoh(sg1, INTER);
            let u1 = dtoh(su1, INTER);
            let mut h2s = vec![0f32; INTER];
            for j in 0..INTER {
                h2s[j] = (g1[j] / (1.0 + (-g1[j]).exp())) * u1[j];
            }
            let mut h2s_dev = to_f32_dev(&h2s);
            let mut sdown = alloc_zeroed(H * 4);
            launch(f_gemv, H as u32, 1, 256, 1, 0, &mut [
                &mut w_sdn as *mut _ as *mut _, &mut h2s_dev as *mut _ as *mut _,
                &mut sdown as *mut _ as *mut _, &mut n640 as *mut _ as *mut _]);
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
                let mut y1 = h1;
                launch(f_gemv, (2 * INTER) as u32, 1, 256, 1, 0, &mut [
                    &mut gu_dev[eid] as *mut _ as *mut _, &mut xt as *mut _ as *mut _,
                    &mut y1 as *mut _ as *mut _, &mut n2560 as *mut _ as *mut _]);
                launch(f_smul, 3, 1, 256, 1, 0, &mut [
                    &mut h1 as *mut _ as *mut _, &mut h2 as *mut _ as *mut _]);
                let mut y2 = eo;
                launch(f_gemv, H as u32, 1, 256, 1, 0, &mut [
                    &mut dn_dev[eid] as *mut _ as *mut _, &mut h2 as *mut _ as *mut _,
                    &mut y2 as *mut _ as *mut _, &mut n640 as *mut _ as *mut _]);
                let mut wdev = to_f32_dev(&[rw]);
                launch(f_acc, 10, 1, 256, 1, 0, &mut [
                    &mut eo as *mut _ as *mut _, &mut wdev as *mut _ as *mut _,
                    &mut rowp as *mut _ as *mut _]);
            }
        }

        // final injection: layer_out = x1 + moe⊗injw_m
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

        // ---- compare vs golden ----
        let gpu = dtoh(out_dev, T * HCT);
        let mut nan = 0usize;
        let mut max_abs = 0.0f32;
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
        }
        for t in 0..T {
            let mut tmax = 0.0f32;
            for j in 0..HCT {
                tmax = tmax.max((gpu[t * HCT + j] - golden[t * HCT + j]).abs());
            }
            println!("  token {t}: max_abs={tmax:.3e}");
        }
        println!(
            "p8: max_abs={max_abs:.3e} at ({}, {}) NaN={nan} tol=5e-3",
            argmax / HCT,
            argmax % HCT
        );
        println!("p8: layer pipeline (sync-per-launch) T=8: {gpu_ms:.2} ms");
        if nan == 0 && max_abs < 5e-3 {
            println!("p8: PASS — full layer 0 matches the transformers reference golden");
        } else {
            let mut worst: Vec<usize> = (0..T * HCT).collect();
            worst.sort_by(|&i, &j| {
                ((gpu[j] - golden[j]).abs())
                    .partial_cmp(&(gpu[i] - golden[i]).abs())
                    .unwrap()
            });
            println!("p8: FAIL — worst 8 positions (gpu vs golden):");
            for &i in worst.iter().take(8) {
                println!(
                    "  ({}, {}): gpu={:+.6} golden={:+.6}",
                    i / HCT,
                    i % HCT,
                    gpu[i],
                    golden[i]
                );
            }
            std::process::exit(1);
        }
    }
}
