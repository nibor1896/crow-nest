//! Probe 13 (Crow #10 → #11 prep): the f32 GENERATION LOOP — first end-to-end
//! forward of the whole text model on GPU: embedding → 48 decoder layers
//! (dispatch by layer_types: 3× linear_attention GDN + 1× full_attention,
//! every 4th) → hyper_connection_mixer → lm_head → greedy sampling, with
//! persistent GDN state (p11 kernels) and persistent KV caches (p12 kernels).
//!
//! Prompt pass: batched kernels (p8 HC/MoE composition, p6 chain with
//! `delta_rule_persist`, p7 dense attention, K/V cached batched).
//! Decode passes: single-token kernels (p11 `conv_step`/`delta_rule_step`,
//! p12 `attn_step` with rotary at position t).
//!
//! Gate (this binary): greedy token trace + per-position top-2 margins, logits
//! dumped to `probes/p13debug/gpu-logits.f32` [12][248320]. The reference side
//! (`oracle/ref_gen_logits.py`) re-runs the full 48-layer f32 forward on CPU
//! for the SAME token sequence and compares logits — tol 5e-3.
//!
//! Probe shortcuts (documented, engine moves these):
//! - PLE (layers.1) is SKIPPED on both sides — gated separately with the PLE
//!   gather; `layer.ple = None` on the reference side mirrors this.
//! - Router softmax/top-10 on host (p8 pattern).
//! - Layer weights are STREAMED per pass from the safetensors shards
//!   (BF16 raw HtoD → on-GPU bf16→f32; f32 for all 48 layers never fits
//!   VRAM — the engine's residency/fp4 paths replace this).
//! - QSA indexer dense by construction (T=12 < 2048, p7 evidence).
//!
//! WDDM rules held: scalars via device buffers (p5 lesson), HtoD async +
//! explicit sync, elementwise guards over the full flat length.

use std::collections::HashMap;
use std::ffi::CString;
use std::io::{Read, Seek, SeekFrom};
use std::time::Instant;

use cudarc::driver::sys::{self, CUdeviceptr, CUfunction, CUresult};
use cudarc::nvrtc::{compile_ptx_with_opts, CompileOptions};

const PROMPT: usize = 8;
const GEN: usize = 4;
const T_MAX: usize = PROMPT + GEN; // 12
const H: usize = 2560;
const HC: usize = 4;
const HCT: usize = HC * H; // 10240
const LOWRANK: usize = 320;
const E: usize = 512;
const TOPK: usize = 10;
const INTER: usize = 640;
const GDN_KEY: usize = 2048;
const GDN_VAL: usize = 6144;
const GDN_CONV: usize = GDN_KEY * 2 + GDN_VAL; // 10240
const NQ: usize = 24;
const NKV: usize = 2;
const HD: usize = 256;
const Q_ROWS: usize = NQ * HD * 2; // 12288
const KV_ROWS: usize = NKV * HD; // 512
const CORE: usize = NQ * HD; // 6144
const V: usize = 248320; // vocab
const LAYERS: usize = 48;

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

// exact BF16 -> f32 (bit shift), elementwise over the full flat length
extern "C" __global__ void bf16_to_f32(const unsigned short* __restrict__ in_,
                                       float* __restrict__ out,
                                       const int* __restrict__ n_p) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= *n_p) return;
    out[i] = __int_as_float(((unsigned int)in_[i]) << 16);
}

// grouped RMSNorm: groups of 2560 over [T][10240], eps 1e-6, weight (1+w)
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

// y[t·2560+c] += (*w)·x[t·2560+c]   grid (10, T); w is a pointer INTO the
// per-layer routing-weight buffer at [t][rank]
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

// ---- GDN batched (prompt) ----
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

// batched prompt recurrence: per-step outputs AND persistent state (global)
extern "C" __global__ void delta_rule_persist(const float* __restrict__ q,
                                              const float* __restrict__ k,
                                              const float* __restrict__ v,
                                              const float* __restrict__ g,
                                              const float* __restrict__ beta,
                                              float* __restrict__ out,
                                              float* __restrict__ s_global,
                                              const int* __restrict__ steps_p) {
    int steps = *steps_p;
    int head = blockIdx.x;
    float* S = s_global + head * 128 * 128;
    int d = threadIdx.x;
    for (int dk = 0; dk < 128; dk++) S[dk * 128 + d] = 0.0f;
    for (int t = 0; t < steps; t++) {
        float g_t = expf(g[t * 48 + head]);
        float beta_t = beta[t * 48 + head];
        const float* qt = q + (t * 48 + head) * 128;
        const float* kt = k + (t * 48 + head) * 128;
        const float* vt = v + (t * 48 + head) * 128;
        for (int dk = 0; dk < 128; dk++) S[dk * 128 + d] *= g_t;
        float kv = 0.0f;
        for (int dk = 0; dk < 128; dk++) kv += S[dk * 128 + d] * kt[dk];
        float delta = (vt[d] - kv) * beta_t;
        for (int dk = 0; dk < 128; dk++) S[dk * 128 + d] += kt[dk] * delta;
        float o = 0.0f;
        for (int dk = 0; dk < 128; dk++) o += S[dk * 128 + d] * qt[dk];
        out[(t * 48 + head) * 128 + d] = o;
    }
}

// ---- GDN decode step (persistent state) ----
extern "C" __global__ void conv_step(const float* __restrict__ mq1,
                                     const float* __restrict__ w,
                                     float* __restrict__ cs,     // [10240][3]
                                     float* __restrict__ cout) { // [10240]
    int ch = blockIdx.x * blockDim.x + threadIdx.x;
    if (ch >= 10240) return;
    const float* wv = w + ch * 4;
    float acc = wv[0] * cs[ch * 3 + 0] + wv[1] * cs[ch * 3 + 1]
              + wv[2] * cs[ch * 3 + 2] + wv[3] * mq1[ch];
    cout[ch] = acc / (1.0f + expf(-acc));
    cs[ch * 3 + 0] = cs[ch * 3 + 1];
    cs[ch * 3 + 1] = cs[ch * 3 + 2];
    cs[ch * 3 + 2] = mq1[ch];
}

extern "C" __global__ void delta_rule_step(float* __restrict__ s_global,
                                           const float* __restrict__ q,
                                           const float* __restrict__ k,
                                           const float* __restrict__ v,
                                           const float* __restrict__ g,
                                           const float* __restrict__ beta,
                                           float* __restrict__ core) {
    int head = blockIdx.x;
    float* S = s_global + head * 128 * 128;
    int d = threadIdx.x;
    float g_t = expf(g[head]);
    float beta_t = beta[head];
    const float* qt = q + head * 128;
    const float* kt = k + head * 128;
    const float* vt = v + head * 128;
    for (int dk = 0; dk < 128; dk++) S[dk * 128 + d] *= g_t;
    float kv = 0.0f;
    for (int dk = 0; dk < 128; dk++) kv += S[dk * 128 + d] * kt[dk];
    float delta = (vt[d] - kv) * beta_t;
    for (int dk = 0; dk < 128; dk++) S[dk * 128 + d] += kt[dk] * delta;
    float o = 0.0f;
    for (int dk = 0; dk < 128; dk++) o += S[dk * 128 + d] * qt[dk];
    core[head * 128 + d] = o;
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

// ---- attention (full layers) ----
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

// dense causal attention over the prompt, block per q-head (T=8 fits p[8])
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

// single-token attention over the persistent KV cache; stride = tmax param
extern "C" __global__ void attn_step(const float* __restrict__ q,        // [24][256]
                                     const float* __restrict__ k_cache,  // [2][tmax][256]
                                     const float* __restrict__ v_cache,  // [2][tmax][256]
                                     const int* __restrict__ pos_p,
                                     const int* __restrict__ tmax_p,
                                     float* __restrict__ out) {          // [24][256]
    int head = blockIdx.x;
    int kvh = head / 12;
    int d = threadIdx.x;
    int len = *pos_p + 1;
    int tmax = *tmax_p;
    const float* qt = q + head * 256;
    const float scale = 0.0625f; // 1/sqrt(256)
    float p[32];
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

/// tensor location inside its safetensors shard, parsed once
struct Ckpt {
    models: String,
    refs: HashMap<String, (String, String, usize, usize)>, // shard, dtype, start, nbytes
}

impl Ckpt {
    fn new(models: &str) -> Self {
        let index: serde_json::Value = serde_json::from_slice(
            &std::fs::read(format!("{models}/model.safetensors.index.json")).unwrap(),
        )
        .unwrap();
        let mut refs = HashMap::new();
        for (name, shard) in index["weight_map"].as_object().unwrap() {
            refs.insert(name.clone(), (shard.as_str().unwrap().to_string(), String::new(), 0, 0));
        }
        Self {
            models: models.to_string(),
            refs,
        }
    }
    fn locate(&mut self, name: &str) -> (String, String, usize, usize) {
        let models = self.models.clone();
        let e = self.refs.get_mut(name).unwrap_or_else(|| panic!("missing in checkpoint: {name}"));
        if e.3 == 0 {
            let shard = e.0.clone();
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
            e.1 = info["dtype"].as_str().unwrap().to_string();
            e.2 = 8 + hl + off;
            e.3 = end - off;
        }
        e.clone()
    }
    /// BF16/F32 checkpoint tensor → f32 host vector (BF16→f32 is exact)
    fn f32(&mut self, name: &str) -> Vec<f32> {
        let (shard, dtype, start, nbytes) = self.locate(name);
        let mut fh = std::fs::File::open(format!("{}/{}", self.models, shard)).unwrap();
        fh.seek(SeekFrom::Start(start as u64)).unwrap();
        let mut raw = vec![0u8; nbytes];
        fh.read_exact(&mut raw).unwrap();
        match dtype.as_str() {
            "F32" => raw
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect(),
            "BF16" => raw
                .chunks_exact(2)
                .map(|c| f32::from_bits((u16::from_le_bytes([c[0], c[1]]) as u32) << 16))
                .collect(),
            other => panic!("{name}: unexpected dtype {other}"),
        }
    }
    /// raw BF16 bytes (for on-GPU conversion of big slabs)
    fn bf16_raw(&mut self, name: &str) -> Vec<u8> {
        let (shard, dtype, start, nbytes) = self.locate(name);
        assert_eq!(dtype, "BF16", "{name}: expected BF16");
        let mut fh = std::fs::File::open(format!("{}/{}", self.models, shard)).unwrap();
        fh.seek(SeekFrom::Start(start as u64)).unwrap();
        let mut raw = vec![0u8; nbytes];
        fh.read_exact(&mut raw).unwrap();
        raw
    }
}

struct Fx {
    gemv: CUfunction,
    b2f: CUfunction,
    rmsg: CUfunction,
    sil4: CUfunction,
    sig: CUfunction,
    sig2: CUfunction,
    mix: CUfunction,
    inj: CUfunction,
    smul: CUfunction,
    acc: CUfunction,
    gsh: CUfunction,
    conv: CUfunction,
    convstep: CUfunction,
    l2: CUfunction,
    bg: CUfunction,
    persist: CUfunction,
    step: CUfunction,
    rmsgt: CUfunction,
    split: CUfunction,
    norm1pw: CUfunction,
    rope: CUfunction,
    attnb: CUfunction,
    attns: CUfunction,
    gmul: CUfunction,
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

unsafe fn free_dev(d: &mut CUdeviceptr) {
    if *d != 0 {
        ck(sys::cuMemFree_v2(*d));
        *d = 0;
    }
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

/// upload raw BF16 bytes and convert ON GPU; returns the f32 device buffer
unsafe fn upload_bf16_to_f32_dev(fx: &Fx, raw: &[u8]) -> CUdeviceptr {
    let n = raw.len() / 2;
    assert!(n <= i32::MAX as usize, "slab too large for i32 guard");
    let mut n_dev = to_i32_dev(&[n as i32]);
    let mut staging = upload_dev(raw);
    let mut out = alloc_zeroed(n * 4);
    launch(
        fx.b2f,
        ((n as u32) + 255) / 256,
        1,
        256,
        1,
        0,
        &mut [
            &mut staging as *mut _ as *mut _,
            &mut out as *mut _ as *mut _,
            &mut n_dev as *mut _ as *mut _,
        ],
    );
    free_dev(&mut staging);
    free_dev(&mut n_dev);
    out
}

/// one gemv y = W·x for token row t (x row pointer already offset)
unsafe fn gemv_row(fx: &Fx, kdim: CUdeviceptr, w: CUdeviceptr, x_row: CUdeviceptr, y_row: CUdeviceptr, rows: u32) {
    let mut wv = w;
    let mut xv = x_row;
    let mut yv = y_row;
    let mut kv = kdim;
    launch(fx.gemv, rows, 1, 256, 1, 0, &mut [
        &mut wv as *mut _ as *mut _, &mut xv as *mut _ as *mut _,
        &mut yv as *mut _ as *mut _, &mut kv as *mut _ as *mut _]);
}

/// GatedResidual block (layer-internal): returns (mixed [T][2560], injw [T][4])
unsafe fn hc_block(
    fx: &Fx, ps: &Ps, t: usize,
    mut w_norm: CUdeviceptr, mut w_down: CUdeviceptr, mut w_up: CUdeviceptr, mut w_inj: CUdeviceptr,
    x: CUdeviceptr,
) -> (CUdeviceptr, CUdeviceptr) {
    let mut normed = alloc_zeroed(t * HCT * 4);
    {
        let mut xv = x;
        launch(fx.rmsg, 4, t as u32, 256, 1, 0, &mut [
            &mut xv as *mut _ as *mut _, &mut w_norm as *mut _ as *mut _,
            &mut normed as *mut _ as *mut _]);
    }
    let mut low = alloc_zeroed(t * LOWRANK * 4);
    for i in 0..t {
        gemv_row(fx, ps.n10240, w_down, normed + (i * HCT * 4) as u64, low + (i * LOWRANK * 4) as u64, LOWRANK as u32);
    }
    let mut sil = alloc_zeroed(t * LOWRANK * 4);
    {
        let mut n = ps.nt_low(t);
        launch(fx.sil4, ((t * LOWRANK) as u32 + 255) / 256, 1, 256, 1, 0, &mut [
            &mut low as *mut _ as *mut _, &mut sil as *mut _ as *mut _,
            &mut n as *mut _ as *mut _]);
    }
    let mut mixw = alloc_zeroed(t * HCT * 4);
    for i in 0..t {
        gemv_row(fx, ps.n320, w_up, sil + (i * LOWRANK * 4) as u64, mixw + (i * HCT * 4) as u64, HCT as u32);
    }
    {
        let mut n = ps.nt_hct(t);
        launch(fx.sig, ((t * HCT) as u32 + 255) / 256, 1, 256, 1, 0, &mut [
            &mut mixw as *mut _ as *mut _, &mut n as *mut _ as *mut _]);
    }
    let mut mixed = alloc_zeroed(t * H * 4);
    {
        let mut mw = mixw;
        let mut nd = normed;
        launch(fx.mix, 10, t as u32, 256, 1, 0, &mut [
            &mut mw as *mut _ as *mut _, &mut nd as *mut _ as *mut _,
            &mut mixed as *mut _ as *mut _]);
    }
    let mut injr = alloc_zeroed(t * HC * 4);
    for i in 0..t {
        gemv_row(fx, ps.n10240, w_inj, normed + (i * HCT * 4) as u64, injr + (i * HC * 4) as u64, HC as u32);
    }
    let mut injw = alloc_zeroed(t * HC * 4);
    {
        let mut n = ps.nt_hc(t);
        launch(fx.sig2, 1, 1, (t * HC) as u32, 1, 0, &mut [
            &mut injr as *mut _ as *mut _, &mut injw as *mut _ as *mut _,
            &mut n as *mut _ as *mut _]);
    }
    free_dev(&mut low);
    free_dev(&mut sil);
    free_dev(&mut mixw);
    free_dev(&mut injr);
    // normed stays alive (mix needs it? no — mix already done). free it too.
    free_dev(&mut normed);
    (mixed, injw)
}

/// model-level hyper_connection_mixer (use_combine=False): [T][10240] → [T][2560]
unsafe fn mixer_block(fx: &Fx, ps: &Ps, t: usize, mut w_norm: CUdeviceptr, mut w_down: CUdeviceptr, mut w_up: CUdeviceptr, x: CUdeviceptr) -> CUdeviceptr {
    let mut normed = alloc_zeroed(t * HCT * 4);
    {
        let mut xv = x;
        launch(fx.rmsg, 4, t as u32, 256, 1, 0, &mut [
            &mut xv as *mut _ as *mut _, &mut w_norm as *mut _ as *mut _,
            &mut normed as *mut _ as *mut _]);
    }
    let mut low = alloc_zeroed(t * LOWRANK * 4);
    for i in 0..t {
        gemv_row(fx, ps.n10240, w_down, normed + (i * HCT * 4) as u64, low + (i * LOWRANK * 4) as u64, LOWRANK as u32);
    }
    let mut sil = alloc_zeroed(t * LOWRANK * 4);
    {
        let mut n = ps.nt_low(t);
        launch(fx.sil4, ((t * LOWRANK) as u32 + 255) / 256, 1, 256, 1, 0, &mut [
            &mut low as *mut _ as *mut _, &mut sil as *mut _ as *mut _,
            &mut n as *mut _ as *mut _]);
    }
    let mut mixw = alloc_zeroed(t * HCT * 4);
    for i in 0..t {
        gemv_row(fx, ps.n320, w_up, sil + (i * LOWRANK * 4) as u64, mixw + (i * HCT * 4) as u64, HCT as u32);
    }
    {
        let mut n = ps.nt_hct(t);
        launch(fx.sig, ((t * HCT) as u32 + 255) / 256, 1, 256, 1, 0, &mut [
            &mut mixw as *mut _ as *mut _, &mut n as *mut _ as *mut _]);
    }
    let mut mixed = alloc_zeroed(t * H * 4);
    {
        let mut mw = mixw;
        let mut nd = normed;
        launch(fx.mix, 10, t as u32, 256, 1, 0, &mut [
            &mut mw as *mut _ as *mut _, &mut nd as *mut _ as *mut _,
            &mut mixed as *mut _ as *mut _]);
    }
    free_dev(&mut low);
    free_dev(&mut sil);
    free_dev(&mut mixw);
    free_dev(&mut normed);
    mixed
}

fn transpose_host(v: &[f32], rows: usize, cols: usize) -> Vec<f32> {
    let mut out = vec![0f32; rows * cols];
    for i in 0..rows {
        for j in 0..cols {
            out[j * rows + i] = v[i * cols + j];
        }
    }
    out
}

/// scalar-count params (all via device buffers, p5 lesson)
struct Ps {
    n320: CUdeviceptr,
    n2560: CUdeviceptr,
    n640: CUdeviceptr,
    n6144: CUdeviceptr,
    n10240: CUdeviceptr,
    n_conv: CUdeviceptr,
    tmax_p: CUdeviceptr,
    pos_p: CUdeviceptr,
    t_prompt_p: CUdeviceptr,
}
impl Ps {
    unsafe fn nt_low(&self, t: usize) -> CUdeviceptr {
        // elementwise n buffers must match the CURRENT t; small uploads per call
        to_i32_dev(&[(t * LOWRANK) as i32])
    }
    unsafe fn nt_hct(&self, t: usize) -> CUdeviceptr {
        to_i32_dev(&[(t * HCT) as i32])
    }
    unsafe fn nt_hc(&self, t: usize) -> CUdeviceptr {
        to_i32_dev(&[(t * HC) as i32])
    }
}

/// full GDN sub-block, PROMPT mode: batched chain, state persisted
unsafe fn gdn_prompt(
    fx: &Fx, ps: &Ps, t: usize, mixed: CUdeviceptr,
    w: &GdnW, mut s_state: CUdeviceptr, mut cs_state: CUdeviceptr,
) -> CUdeviceptr {
    let mut w_conv = w.conv;
    let mut w_alog = w.alog;
    let mut w_dt = w.dt;
    let mut w_nrm = w.norm;
    // in_proj_qkv per token
    let mut mq = alloc_zeroed(t * GDN_CONV * 4);
    for i in 0..t {
        gemv_row(fx, ps.n2560, w.qkv, mixed + (i * H * 4) as u64, mq + (i * GDN_CONV * 4) as u64, GDN_CONV as u32);
    }
    let mq_host = dtoh(mq, t * GDN_CONV);
    let mq_t = transpose_host(&mq_host, t, GDN_CONV); // [chan][t]
    let mut mq_t_dev = to_f32_dev(&mq_t);
    let mut conv_out = alloc_zeroed(t * GDN_CONV * 4);
    {
        let mut tp = ps.t_prompt_p;
        launch(fx.conv, GDN_CONV as u32, 1, t as u32, 1, 0, &mut [
            &mut mq_t_dev as *mut _ as *mut _, &mut w_conv as *mut _ as *mut _,
            &mut conv_out as *mut _ as *mut _, &mut tp as *mut _ as *mut _]);
    }
    let conv_t = transpose_host(&dtoh(conv_out, t * GDN_CONV), GDN_CONV, t);

    // split q/k/v rows to device
    let mut q_dev = alloc_zeroed(t * GDN_KEY * 4);
    let mut k_dev = alloc_zeroed(t * GDN_KEY * 4);
    let mut v_dev = alloc_zeroed(t * GDN_VAL * 4);
    for i in 0..t {
        let base = i * GDN_CONV;
        let qh = &conv_t[base..base + GDN_KEY];
        let kh = &conv_t[base + GDN_KEY..base + 2 * GDN_KEY];
        let vh = &conv_t[base + 2 * GDN_KEY..base + GDN_CONV];
        ck(sys::cuMemcpyHtoDAsync_v2(q_dev + (i * GDN_KEY * 4) as u64, qh.as_ptr() as *const std::ffi::c_void, GDN_KEY * 4, std::ptr::null_mut()));
        ck(sys::cuMemcpyHtoDAsync_v2(k_dev + (i * GDN_KEY * 4) as u64, kh.as_ptr() as *const std::ffi::c_void, GDN_KEY * 4, std::ptr::null_mut()));
        ck(sys::cuMemcpyHtoDAsync_v2(v_dev + (i * GDN_VAL * 4) as u64, vh.as_ptr() as *const std::ffi::c_void, GDN_VAL * 4, std::ptr::null_mut()));
    }
    ck(sys::cuStreamSynchronize(std::ptr::null_mut()));

    // z / b / a
    let mut z_dev = alloc_zeroed(t * GDN_VAL * 4);
    for i in 0..t {
        gemv_row(fx, ps.n2560, w.z, mixed + (i * H * 4) as u64, z_dev + (i * GDN_VAL * 4) as u64, GDN_VAL as u32);
    }
    let mut b_dev = alloc_zeroed(t * 48 * 4);
    for i in 0..t {
        gemv_row(fx, ps.n2560, w.b, mixed + (i * H * 4) as u64, b_dev + (i * 48 * 4) as u64, 48);
    }
    let mut a_dev = alloc_zeroed(t * 48 * 4);
    for i in 0..t {
        gemv_row(fx, ps.n2560, w.a, mixed + (i * H * 4) as u64, a_dev + (i * 48 * 4) as u64, 48);
    }
    let mut beta_dev = alloc_zeroed(t * 48 * 4);
    let mut g_dev = alloc_zeroed(t * 48 * 4);
    {
        launch(fx.bg, 1, 1, (t * 48) as u32, 1, 0, &mut [
            &mut b_dev as *mut _ as *mut _, &mut a_dev as *mut _ as *mut _,
            &mut w_alog as *mut _ as *mut _, &mut w_dt as *mut _ as *mut _,
            &mut beta_dev as *mut _ as *mut _, &mut g_dev as *mut _ as *mut _]);
    }
    let mut qr = alloc_zeroed(t * 48 * 128 * 4);
    let mut kr = alloc_zeroed(t * 48 * 128 * 4);
    {
        launch(fx.l2, 48, t as u32, 128, 1, 0, &mut [
            &mut q_dev as *mut _ as *mut _, &mut k_dev as *mut _ as *mut _,
            &mut qr as *mut _ as *mut _, &mut kr as *mut _ as *mut _]);
    }
    // persistent recurrence
    let mut core = alloc_zeroed(t * GDN_VAL * 4);
    {
        let mut steps = to_i32_dev(&[t as i32]);
        launch(fx.persist, 48, 1, 128, 1, 0, &mut [
            &mut qr as *mut _ as *mut _, &mut kr as *mut _ as *mut _,
            &mut v_dev as *mut _ as *mut _, &mut g_dev as *mut _ as *mut _,
            &mut beta_dev as *mut _ as *mut _, &mut core as *mut _ as *mut _,
            &mut s_state as *mut _ as *mut _, &mut steps as *mut _ as *mut _]);
        free_dev(&mut steps);
    }
    let mut normed = alloc_zeroed(t * GDN_VAL * 4);
    {
        launch(fx.rmsgt, 48, t as u32, 128, 1, 0, &mut [
            &mut core as *mut _ as *mut _, &mut z_dev as *mut _ as *mut _,
            &mut w_nrm as *mut _ as *mut _, &mut normed as *mut _ as *mut _]);
    }
    let mut out = alloc_zeroed(t * H * 4);
    for i in 0..t {
        gemv_row(fx, ps.n6144, w.out, normed + (i * GDN_VAL * 4) as u64, out + (i * H * 4) as u64, H as u32);
    }
    // conv_state = last 3 PRE-conv mixed_qkv rows ([chan][t] layout)
    let mut cs_host = vec![0f32; GDN_CONV * 3];
    for ch in 0..GDN_CONV {
        for (i, tt) in [t - 3, t - 2, t - 1].iter().enumerate() {
            cs_host[ch * 3 + i] = mq_t[ch * t + tt];
        }
    }
    ck(sys::cuMemcpyHtoDAsync_v2(cs_state, cs_host.as_ptr() as *const std::ffi::c_void, GDN_CONV * 3 * 4, std::ptr::null_mut()));
    ck(sys::cuStreamSynchronize(std::ptr::null_mut()));

    free_dev(&mut mq);
    free_dev(&mut mq_t_dev);
    free_dev(&mut conv_out);
    free_dev(&mut q_dev);
    free_dev(&mut k_dev);
    free_dev(&mut v_dev);
    free_dev(&mut z_dev);
    free_dev(&mut b_dev);
    free_dev(&mut a_dev);
    free_dev(&mut beta_dev);
    free_dev(&mut g_dev);
    free_dev(&mut qr);
    free_dev(&mut kr);
    free_dev(&mut core);
    free_dev(&mut normed);
    out
}

struct GdnW {
    qkv: CUdeviceptr,
    conv: CUdeviceptr,
    z: CUdeviceptr,
    b: CUdeviceptr,
    a: CUdeviceptr,
    alog: CUdeviceptr,
    dt: CUdeviceptr,
    norm: CUdeviceptr,
    out: CUdeviceptr,
}

/// full GDN sub-block, DECODE mode: single token against persistent state
unsafe fn gdn_step(
    fx: &Fx, ps: &Ps, mixed_row: CUdeviceptr,
    w: &GdnW, mut s_state: CUdeviceptr, mut cs_state: CUdeviceptr,
) -> CUdeviceptr {
    let mut w_conv = w.conv;
    let mut w_alog = w.alog;
    let mut w_dt = w.dt;
    let mut w_nrm = w.norm;
    let mut mq1 = alloc_zeroed(GDN_CONV * 4);
    gemv_row(fx, ps.n2560, w.qkv, mixed_row, mq1, GDN_CONV as u32);
    let mut cout = alloc_zeroed(GDN_CONV * 4);
    {
        launch(fx.convstep, (GDN_CONV as u32 + 255) / 256, 1, 256, 1, 0, &mut [
            &mut mq1 as *mut _ as *mut _, &mut w_conv as *mut _ as *mut _,
            &mut cs_state as *mut _ as *mut _, &mut cout as *mut _ as *mut _]);
    }
    let cout_h = dtoh(cout, GDN_CONV);
    let mut q1 = to_f32_dev(&cout_h[..GDN_KEY]);
    let mut k1 = to_f32_dev(&cout_h[GDN_KEY..2 * GDN_KEY]);
    let mut v1 = to_f32_dev(&cout_h[2 * GDN_KEY..]);
    let mut z1 = alloc_zeroed(GDN_VAL * 4);
    gemv_row(fx, ps.n2560, w.z, mixed_row, z1, GDN_VAL as u32);
    let mut b1 = alloc_zeroed(48 * 4);
    gemv_row(fx, ps.n2560, w.b, mixed_row, b1, 48);
    let mut a1 = alloc_zeroed(48 * 4);
    gemv_row(fx, ps.n2560, w.a, mixed_row, a1, 48);
    let mut beta1 = alloc_zeroed(48 * 4);
    let mut g1 = alloc_zeroed(48 * 4);
    {
        launch(fx.bg, 1, 1, 48, 1, 0, &mut [
            &mut b1 as *mut _ as *mut _, &mut a1 as *mut _ as *mut _,
            &mut w_alog as *mut _ as *mut _, &mut w_dt as *mut _ as *mut _,
            &mut beta1 as *mut _ as *mut _, &mut g1 as *mut _ as *mut _]);
    }
    let mut q1r = alloc_zeroed(48 * 128 * 4);
    let mut k1r = alloc_zeroed(48 * 128 * 4);
    {
        launch(fx.l2, 48, 1, 128, 1, 0, &mut [
            &mut q1 as *mut _ as *mut _, &mut k1 as *mut _ as *mut _,
            &mut q1r as *mut _ as *mut _, &mut k1r as *mut _ as *mut _]);
    }
    let mut core1 = alloc_zeroed(GDN_VAL * 4);
    {
        launch(fx.step, 48, 1, 128, 1, 0, &mut [
            &mut s_state as *mut _ as *mut _, &mut q1r as *mut _ as *mut _,
            &mut k1r as *mut _ as *mut _, &mut v1 as *mut _ as *mut _,
            &mut g1 as *mut _ as *mut _, &mut beta1 as *mut _ as *mut _,
            &mut core1 as *mut _ as *mut _]);
    }
    let mut normed1 = alloc_zeroed(GDN_VAL * 4);
    {
        launch(fx.rmsgt, 48, 1, 128, 1, 0, &mut [
            &mut core1 as *mut _ as *mut _, &mut z1 as *mut _ as *mut _,
            &mut w_nrm as *mut _ as *mut _, &mut normed1 as *mut _ as *mut _]);
    }
    let mut y1 = alloc_zeroed(H * 4);
    gemv_row(fx, ps.n6144, w.out, normed1, y1, H as u32);

    free_dev(&mut mq1);
    free_dev(&mut cout);
    free_dev(&mut q1);
    free_dev(&mut k1);
    free_dev(&mut v1);
    free_dev(&mut z1);
    free_dev(&mut b1);
    free_dev(&mut a1);
    free_dev(&mut beta1);
    free_dev(&mut g1);
    free_dev(&mut q1r);
    free_dev(&mut k1r);
    free_dev(&mut core1);
    free_dev(&mut normed1);
    y1
}

struct AttnW {
    q: CUdeviceptr,
    k: CUdeviceptr,
    v: CUdeviceptr,
    o: CUdeviceptr,
    qn: CUdeviceptr,
    kn: CUdeviceptr,
}

/// full-attention sub-block, PROMPT mode (p7 chain) + batched K/V cache fill
unsafe fn attn_prompt(
    fx: &Fx, ps: &Ps, t: usize, mixed: CUdeviceptr,
    w: &AttnW, k_cache: CUdeviceptr, v_cache: CUdeviceptr,
    mut cos_dev: CUdeviceptr, mut sin_dev: CUdeviceptr,
) -> CUdeviceptr {
    let mut w_qn = w.qn;
    let mut w_kn = w.kn;
    let mut qg = alloc_zeroed(t * Q_ROWS * 4);
    for i in 0..t {
        gemv_row(fx, ps.n2560, w.q, mixed + (i * H * 4) as u64, qg + (i * Q_ROWS * 4) as u64, Q_ROWS as u32);
    }
    let mut q_dev = alloc_zeroed(t * CORE * 4);
    let mut gate_dev = alloc_zeroed(t * CORE * 4);
    {
        launch(fx.split, NQ as u32, t as u32, HD as u32, 1, 0, &mut [
            &mut qg as *mut _ as *mut _, &mut q_dev as *mut _ as *mut _,
            &mut gate_dev as *mut _ as *mut _]);
    }
    let mut qn_dev = alloc_zeroed(t * CORE * 4);
    {
        launch(fx.norm1pw, NQ as u32, t as u32, HD as u32, 1, 0, &mut [
            &mut q_dev as *mut _ as *mut _, &mut w_qn as *mut _ as *mut _,
            &mut qn_dev as *mut _ as *mut _]);
    }
    let mut qr = alloc_zeroed(t * CORE * 4);
    {
        launch(fx.rope, NQ as u32, t as u32, HD as u32, 1, 0, &mut [
            &mut qn_dev as *mut _ as *mut _, &mut cos_dev as *mut _ as *mut _,
            &mut sin_dev as *mut _ as *mut _, &mut qr as *mut _ as *mut _]);
    }
    let mut k_dev = alloc_zeroed(t * KV_ROWS * 4);
    for i in 0..t {
        gemv_row(fx, ps.n2560, w.k, mixed + (i * H * 4) as u64, k_dev + (i * KV_ROWS * 4) as u64, KV_ROWS as u32);
    }
    let mut kn = alloc_zeroed(t * KV_ROWS * 4);
    {
        launch(fx.norm1pw, NKV as u32, t as u32, HD as u32, 1, 0, &mut [
            &mut k_dev as *mut _ as *mut _, &mut w_kn as *mut _ as *mut _,
            &mut kn as *mut _ as *mut _]);
    }
    let mut kr = alloc_zeroed(t * KV_ROWS * 4);
    {
        launch(fx.rope, NKV as u32, t as u32, HD as u32, 1, 0, &mut [
            &mut kn as *mut _ as *mut _, &mut cos_dev as *mut _ as *mut _,
            &mut sin_dev as *mut _ as *mut _, &mut kr as *mut _ as *mut _]);
    }
    let mut v_dev = alloc_zeroed(t * KV_ROWS * 4);
    for i in 0..t {
        gemv_row(fx, ps.n2560, w.v, mixed + (i * H * 4) as u64, v_dev + (i * KV_ROWS * 4) as u64, KV_ROWS as u32);
    }
    // batched K/V append into the persistent caches ([2][tmax][256] each)
    for i in 0..t {
        for kvh in 0..NKV {
            let src_k = kr + ((i * NKV + kvh) * HD * 4) as u64;
            let dst_k = k_cache + ((kvh * T_MAX + i) * HD * 4) as u64;
            ck(sys::cuMemcpyDtoDAsync_v2(dst_k, src_k, HD * 4, std::ptr::null_mut()));
            let src_v = v_dev + ((i * NKV + kvh) * HD * 4) as u64;
            let dst_v = v_cache + ((kvh * T_MAX + i) * HD * 4) as u64;
            ck(sys::cuMemcpyDtoDAsync_v2(dst_v, src_v, HD * 4, std::ptr::null_mut()));
        }
    }
    ck(sys::cuStreamSynchronize(std::ptr::null_mut()));
    let mut core = alloc_zeroed(t * CORE * 4);
    {
        let mut tp = ps.t_prompt_p;
        launch(fx.attnb, NQ as u32, 1, HD as u32, 1, 0, &mut [
            &mut qr as *mut _ as *mut _, &mut kr as *mut _ as *mut _,
            &mut v_dev as *mut _ as *mut _, &mut core as *mut _ as *mut _,
            &mut tp as *mut _ as *mut _]);
    }
    let mut gated = alloc_zeroed(t * CORE * 4);
    {
        launch(fx.gmul, ((t * CORE) as u32 + 255) / 256, 1, 256, 1, 0, &mut [
            &mut core as *mut _ as *mut _, &mut gate_dev as *mut _ as *mut _,
            &mut gated as *mut _ as *mut _]);
    }
    let mut out = alloc_zeroed(t * H * 4);
    for i in 0..t {
        gemv_row(fx, ps.n6144, w.o, gated + (i * CORE * 4) as u64, out + (i * H * 4) as u64, H as u32);
    }
    free_dev(&mut qg);
    free_dev(&mut q_dev);
    free_dev(&mut gate_dev);
    free_dev(&mut qn_dev);
    free_dev(&mut qr);
    free_dev(&mut k_dev);
    free_dev(&mut kn);
    free_dev(&mut kr);
    free_dev(&mut v_dev);
    free_dev(&mut core);
    free_dev(&mut gated);
    out
}

/// full-attention sub-block, DECODE mode (p12): rotary at pos, cache append, attn_step
unsafe fn attn_step_tok(
    fx: &Fx, ps: &Ps, mixed_row: CUdeviceptr,
    w: &AttnW, k_cache: CUdeviceptr, v_cache: CUdeviceptr,
    cos_dev: CUdeviceptr, sin_dev: CUdeviceptr, pos: usize,
) -> CUdeviceptr {
    let mut w_qn = w.qn;
    let mut w_kn = w.kn;
    let mut qg = alloc_zeroed(Q_ROWS * 4);
    gemv_row(fx, ps.n2560, w.q, mixed_row, qg, Q_ROWS as u32);
    let mut q_dev = alloc_zeroed(CORE * 4);
    let mut gate_dev = alloc_zeroed(CORE * 4);
    {
        launch(fx.split, NQ as u32, 1, HD as u32, 1, 0, &mut [
            &mut qg as *mut _ as *mut _, &mut q_dev as *mut _ as *mut _,
            &mut gate_dev as *mut _ as *mut _]);
    }
    let mut qn = alloc_zeroed(CORE * 4);
    {
        launch(fx.norm1pw, NQ as u32, 1, HD as u32, 1, 0, &mut [
            &mut q_dev as *mut _ as *mut _, &mut w_qn as *mut _ as *mut _,
            &mut qn as *mut _ as *mut _]);
    }
    let mut cos_t = cos_dev + (pos * 32 * 4) as u64;
    let mut sin_t = sin_dev + (pos * 32 * 4) as u64;
    let mut posp = ps.pos_p;
    let mut tmaxp = ps.tmax_p;
    let mut kc = k_cache;
    let mut vc = v_cache;
    let mut qr = alloc_zeroed(CORE * 4);
    {
        launch(fx.rope, NQ as u32, 1, HD as u32, 1, 0, &mut [
            &mut qn as *mut _ as *mut _, &mut cos_t as *mut _ as *mut _,
            &mut sin_t as *mut _ as *mut _, &mut qr as *mut _ as *mut _]);
    }
    let mut k1 = alloc_zeroed(KV_ROWS * 4);
    gemv_row(fx, ps.n2560, w.k, mixed_row, k1, KV_ROWS as u32);
    let mut kn = alloc_zeroed(KV_ROWS * 4);
    {
        launch(fx.norm1pw, NKV as u32, 1, HD as u32, 1, 0, &mut [
            &mut k1 as *mut _ as *mut _, &mut w_kn as *mut _ as *mut _,
            &mut kn as *mut _ as *mut _]);
    }
    let mut kr = alloc_zeroed(KV_ROWS * 4);
    {
        launch(fx.rope, NKV as u32, 1, HD as u32, 1, 0, &mut [
            &mut kn as *mut _ as *mut _, &mut cos_t as *mut _ as *mut _,
            &mut sin_t as *mut _ as *mut _, &mut kr as *mut _ as *mut _]);
    }
    let mut v1 = alloc_zeroed(KV_ROWS * 4);
    gemv_row(fx, ps.n2560, w.v, mixed_row, v1, KV_ROWS as u32);
    // append K/V into the persistent caches
    for kvh in 0..NKV {
        let src_k = kr + (kvh * HD * 4) as u64;
        let dst_k = k_cache + ((kvh * T_MAX + pos) * HD * 4) as u64;
        ck(sys::cuMemcpyDtoDAsync_v2(dst_k, src_k, HD * 4, std::ptr::null_mut()));
        let src_v = v1 + (kvh * HD * 4) as u64;
        let dst_v = v_cache + ((kvh * T_MAX + pos) * HD * 4) as u64;
        ck(sys::cuMemcpyDtoDAsync_v2(dst_v, src_v, HD * 4, std::ptr::null_mut()));
    }
    ck(sys::cuStreamSynchronize(std::ptr::null_mut()));
    // pos via device buffer (scalar rule)
    let mut pos_val: u32 = pos as u32;
    ck(sys::cuMemcpyHtoDAsync_v2(ps.pos_p, &mut pos_val as *mut u32 as *const std::ffi::c_void, 4, std::ptr::null_mut()));
    ck(sys::cuStreamSynchronize(std::ptr::null_mut()));
    let mut core = alloc_zeroed(CORE * 4);
    {
        launch(fx.attns, NQ as u32, 1, HD as u32, 1, 0, &mut [
            &mut qr as *mut _ as *mut _, &mut kc as *mut _ as *mut _,
            &mut vc as *mut _ as *mut _, &mut posp as *mut _ as *mut _,
            &mut tmaxp as *mut _ as *mut _, &mut core as *mut _ as *mut _]);
    }
    let mut gated = alloc_zeroed(CORE * 4);
    {
        launch(fx.gmul, (CORE as u32 + 255) / 256, 1, 256, 1, 0, &mut [
            &mut core as *mut _ as *mut _, &mut gate_dev as *mut _ as *mut _,
            &mut gated as *mut _ as *mut _]);
    }
    let mut y = alloc_zeroed(H * 4);
    gemv_row(fx, ps.n6144, w.o, gated, y, H as u32);

    free_dev(&mut qg);
    free_dev(&mut q_dev);
    free_dev(&mut gate_dev);
    free_dev(&mut qn);
    free_dev(&mut qr);
    free_dev(&mut k1);
    free_dev(&mut kn);
    free_dev(&mut kr);
    free_dev(&mut v1);
    free_dev(&mut core);
    free_dev(&mut gated);
    y
}

struct MoEW {
    gate: CUdeviceptr,
    sg: CUdeviceptr,
    su: CUdeviceptr,
    sdn: CUdeviceptr,
    sgate: CUdeviceptr,
    gu_slab: CUdeviceptr,
    dn_slab: CUdeviceptr,
}

/// MoE block (p8 composition): router GEMV on GPU, softmax/top-10/norm on host,
/// shared expert + routed experts, weighted accumulate
unsafe fn moe_block(fx: &Fx, ps: &Ps, t: usize, mixed_m: CUdeviceptr, w: &MoEW, rw_dev: CUdeviceptr) -> CUdeviceptr {
    // router logits
    let mut rlog = alloc_zeroed(t * E * 4);
    for i in 0..t {
        gemv_row(fx, ps.n2560, w.gate, mixed_m + (i * H * 4) as u64, rlog + (i * E * 4) as u64, E as u32);
    }
    let lg = dtoh(rlog, t * E);
    free_dev(&mut rlog);
    let mut routing = vec![(0usize, 0f32); t * TOPK];
    for i in 0..t {
        let row = &lg[i * E..(i + 1) * E];
        let mx = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let mut probs: Vec<(usize, f32)> = row
            .iter()
            .enumerate()
            .map(|(ei, v)| (ei, (v - mx).exp()))
            .collect();
        let sum: f32 = probs.iter().map(|p| p.1).sum();
        for pr in probs.iter_mut() {
            pr.1 /= sum;
        }
        probs.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        let sel_sum: f32 = probs[..TOPK].iter().map(|p| p.1).sum();
        for j in 0..TOPK {
            routing[i * TOPK + j] = (probs[j].0, probs[j].1 / sel_sum);
        }
    }
    drop(lg);
    // routing weights to device ([t][10] f32)
    let rw_host: Vec<f32> = routing.iter().map(|(_, w)| *w).collect();
    ck(sys::cuMemcpyHtoDAsync_v2(rw_dev, rw_host.as_ptr() as *const std::ffi::c_void, rw_host.len() * 4, std::ptr::null_mut()));
    ck(sys::cuStreamSynchronize(std::ptr::null_mut()));

    let mut moe_out = alloc_zeroed(t * H * 4);
    let mut h1 = alloc_zeroed(2 * INTER * 4);
    let mut h2 = alloc_zeroed(INTER * 4);
    let mut eo = alloc_zeroed(H * 4);
    // shared expert per token
    for i in 0..t {
        let xt = mixed_m + (i * H * 4) as u64;
        let rowp = moe_out + (i * H * 4) as u64;
        let mut sg1 = alloc_zeroed(INTER * 4);
        gemv_row(fx, ps.n2560, w.sg, xt, sg1, INTER as u32);
        let mut su1 = alloc_zeroed(INTER * 4);
        gemv_row(fx, ps.n2560, w.su, xt, su1, INTER as u32);
        let g1 = dtoh(sg1, INTER);
        let u1 = dtoh(su1, INTER);
        free_dev(&mut sg1);
        free_dev(&mut su1);
        let mut h2s = vec![0f32; INTER];
        for j in 0..INTER {
            h2s[j] = (g1[j] / (1.0 + (-g1[j]).exp())) * u1[j];
        }
        let mut h2s_dev = to_f32_dev(&h2s);
        let mut sdown = alloc_zeroed(H * 4);
        gemv_row(fx, ps.n640, w.sdn, h2s_dev, sdown, H as u32);
        free_dev(&mut h2s_dev);
        let mut sgv = alloc_zeroed(4);
        gemv_row(fx, ps.n2560, w.sgate, xt, sgv, 1);
        {
            let mut s = sdown;
            let mut g = sgv;
            let mut rp = rowp;
            launch(fx.gsh, 10, 1, 256, 1, 0, &mut [
                &mut s as *mut _ as *mut _, &mut g as *mut _ as *mut _,
                &mut rp as *mut _ as *mut _]);
        }
        free_dev(&mut sdown);
        free_dev(&mut sgv);
    }
    // routed experts per (token, rank)
    for i in 0..t {
        for j in 0..TOPK {
            let (eid, _) = routing[i * TOPK + j];
            let xt = mixed_m + (i * H * 4) as u64;
            let rowp = moe_out + (i * H * 4) as u64;
            let e_off = (eid * 2 * INTER * H * 4) as u64;
            let mut wg = w.gu_slab + e_off;
            let mut y1 = h1;
            gemv_row(fx, ps.n2560, wg, xt, y1, (2 * INTER) as u32);
            {
                let mut a = h1;
                let mut b = h2;
                launch(fx.smul, 3, 1, 256, 1, 0, &mut [
                    &mut a as *mut _ as *mut _, &mut b as *mut _ as *mut _]);
            }
            let d_off = (eid * H * INTER * 4) as u64;
            let mut wd = w.dn_slab + d_off;
            let mut y2 = eo;
            gemv_row(fx, ps.n640, wd, h2, y2, H as u32);
            let mut wrw = rw_dev + ((i * TOPK + j) * 4) as u64;
            {
                let mut x2 = eo;
                let mut rp = rowp;
                launch(fx.acc, 10, 1, 256, 1, 0, &mut [
                    &mut x2 as *mut _ as *mut _, &mut wrw as *mut _ as *mut _,
                    &mut rp as *mut _ as *mut _]);
            }
        }
    }
    free_dev(&mut h1);
    free_dev(&mut h2);
    free_dev(&mut eo);
    moe_out
}

fn main() {
    let models = "../models/Qwen3.8-Flash-Next-original";
    let dbg_dir = "../probes/p13debug";
    std::fs::create_dir_all(dbg_dir).unwrap();
    let prompt_ids: [usize; PROMPT] = [760, 3841, 13477, 37550, 33075, 888, 279, 15217];

    println!("p13: loading persistent weights (embed host, lm_head + mixer device) …");
    let t0 = Instant::now();
    let mut ckpt = Ckpt::new(models);

    // embedding stays on HOST — per-token row uploads (10 KB each)
    let embed = ckpt.f32("model.language_model.embed_tokens.weight");
    assert_eq!(embed.len(), V * H);

    let layer_prefix = |l: usize| format!("model.language_model.layers.{l}.");
    let L = |l: usize, sub: &str| format!("{}{}", layer_prefix(l), sub);

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
        let fx = Fx {
            gemv: get_fn("gemv_f32"),
            b2f: get_fn("bf16_to_f32"),
            rmsg: get_fn("rms_group"),
            sil4: get_fn("silu_div4"),
            sig: get_fn("sigmoid_el"),
            sig2: get_fn("sig2_div4"),
            mix: get_fn("mix_streams"),
            inj: get_fn("inject_residual"),
            smul: get_fn("silu_mul640"),
            acc: get_fn("acc_scale"),
            gsh: get_fn("gate_shared"),
            conv: get_fn("conv_silu"),
            convstep: get_fn("conv_step"),
            l2: get_fn("l2norm_repeat"),
            bg: get_fn("beta_g"),
            persist: get_fn("delta_rule_persist"),
            step: get_fn("delta_rule_step"),
            rmsgt: get_fn("rmsnorm_gated"),
            split: get_fn("split_qg"),
            norm1pw: get_fn("rmsnorm_1pw"),
            rope: get_fn("rope"),
            attnb: get_fn("attn_dense"),
            attns: get_fn("attn_step"),
            gmul: get_fn("gate_mul"),
        };

        // elementwise n params are uploaded per call (t-dependent); fixed ones here
        let ps = Ps {
            n320: to_i32_dev(&[LOWRANK as i32]),
            n2560: to_i32_dev(&[H as i32]),
            n640: to_i32_dev(&[INTER as i32]),
            n6144: to_i32_dev(&[GDN_VAL as i32]),
            n10240: to_i32_dev(&[HCT as i32]),
            n_conv: to_i32_dev(&[GDN_CONV as i32]),
            tmax_p: to_i32_dev(&[T_MAX as i32]),
            pos_p: to_i32_dev(&[0]),
            t_prompt_p: to_i32_dev(&[PROMPT as i32]),
        };

        // lm_head + model-level mixer, resident
        let lm_head = upload_bf16_to_f32_dev(&fx, &ckpt.bf16_raw("lm_head.weight"));
        let mx_norm = to_f32_dev(&ckpt.f32("model.language_model.hyper_connection_mixer.hc_norm.weight"));
        let mx_down = to_f32_dev(&ckpt.f32("model.language_model.hyper_connection_mixer.input_mix_weight_down.weight"));
        let mx_up = to_f32_dev(&ckpt.f32("model.language_model.hyper_connection_mixer.input_mix_weight_up.weight"));

        // rotary tables for ALL positions 0..T_MAX-1 (theta 1e7, 32 pairs)
        let mut cos_h = vec![0f32; T_MAX * 32];
        let mut sin_h = vec![0f32; T_MAX * 32];
        for t in 0..T_MAX {
            for j in 0..32 {
                let inv = 10_000_000f32.powf(-(2.0 * j as f32) / 64.0);
                let f = t as f32 * inv;
                cos_h[t * 32 + j] = f.cos();
                sin_h[t * 32 + j] = f.sin();
            }
        }
        let cos_dev = to_f32_dev(&cos_h);
        let sin_dev = to_f32_dev(&sin_h);

        // persistent states: 36 GDN (S + conv) + 12 KV caches
        let mut s_gdn: Vec<CUdeviceptr> = Vec::with_capacity(36);
        let mut cs_gdn: Vec<CUdeviceptr> = Vec::with_capacity(36);
        for _ in 0..36 {
            s_gdn.push(alloc_zeroed(48 * 128 * 128 * 4));
            cs_gdn.push(alloc_zeroed(GDN_CONV * 3 * 4));
        }
        let mut k_cache: Vec<CUdeviceptr> = Vec::with_capacity(12);
        let mut v_cache: Vec<CUdeviceptr> = Vec::with_capacity(12);
        for _ in 0..12 {
            k_cache.push(alloc_zeroed(NKV * T_MAX * HD * 4));
            v_cache.push(alloc_zeroed(NKV * T_MAX * HD * 4));
        }

        // per-layer scratch routing-weight buffer
        let mut rw_dev = alloc_zeroed(PROMPT * TOPK * 4);

        let logits_dev = alloc_zeroed(T_MAX * V * 4);
        let mut all_ids: Vec<usize> = prompt_ids.to_vec();
        println!(
            "p13: setup done in {:.1} s — running prompt pass (weights streamed per layer)",
            t0.elapsed().as_secs_f64()
        );

        // ---------- the generation loop ----------
        unsafe fn run_layer(
            fx: &Fx, ps: &Ps, ckpt: &mut Ckpt, layer: usize,
            is_prompt: bool, pos: usize, mut h: CUdeviceptr, t: usize,
            s_gdn: &[CUdeviceptr], cs_gdn: &[CUdeviceptr],
            k_cache: &[CUdeviceptr], v_cache: &[CUdeviceptr],
            cos_dev: CUdeviceptr, sin_dev: CUdeviceptr, rw_dev: CUdeviceptr,
        ) -> CUdeviceptr {
            let L = |sub: &str| format!("model.language_model.layers.{layer}.{sub}");
            let is_attn = layer % 4 == 3;
            let gdn_i = layer - layer / 4;
            let attn_i = layer / 4;

            let mut a_norm = to_f32_dev(&ckpt.f32(&L("attn_hyper_connection.hc_norm.weight")));
            let mut a_down = to_f32_dev(&ckpt.f32(&L("attn_hyper_connection.input_mix_weight_down.weight")));
            let mut a_up = to_f32_dev(&ckpt.f32(&L("attn_hyper_connection.input_mix_weight_up.weight")));
            let mut a_inj = to_f32_dev(&ckpt.f32(&L("attn_hyper_connection.block_inject_weight.weight")));
            let mut m_norm = to_f32_dev(&ckpt.f32(&L("mlp_hyper_connection.hc_norm.weight")));
            let mut m_down = to_f32_dev(&ckpt.f32(&L("mlp_hyper_connection.input_mix_weight_down.weight")));
            let mut m_up = to_f32_dev(&ckpt.f32(&L("mlp_hyper_connection.input_mix_weight_up.weight")));
            let mut m_inj = to_f32_dev(&ckpt.f32(&L("mlp_hyper_connection.block_inject_weight.weight")));

            let mut gdn_w = None;
            let mut attn_w = None;
            if is_attn {
                attn_w = Some(AttnW {
                    q: to_f32_dev(&ckpt.f32(&L("self_attn.q_proj.weight"))),
                    k: to_f32_dev(&ckpt.f32(&L("self_attn.k_proj.weight"))),
                    v: to_f32_dev(&ckpt.f32(&L("self_attn.v_proj.weight"))),
                    o: to_f32_dev(&ckpt.f32(&L("self_attn.o_proj.weight"))),
                    qn: to_f32_dev(&ckpt.f32(&L("self_attn.q_norm.weight"))),
                    kn: to_f32_dev(&ckpt.f32(&L("self_attn.k_norm.weight"))),
                });
            } else {
                gdn_w = Some(GdnW {
                    qkv: to_f32_dev(&ckpt.f32(&L("linear_attn.in_proj_qkv.weight"))),
                    conv: to_f32_dev(&ckpt.f32(&L("linear_attn.conv1d.weight"))),
                    z: to_f32_dev(&ckpt.f32(&L("linear_attn.in_proj_z.weight"))),
                    b: to_f32_dev(&ckpt.f32(&L("linear_attn.in_proj_b.weight"))),
                    a: to_f32_dev(&ckpt.f32(&L("linear_attn.in_proj_a.weight"))),
                    alog: to_f32_dev(&ckpt.f32(&L("linear_attn.A_log"))),
                    dt: to_f32_dev(&ckpt.f32(&L("linear_attn.dt_bias"))),
                    norm: to_f32_dev(&ckpt.f32(&L("linear_attn.norm.weight"))),
                    out: to_f32_dev(&ckpt.f32(&L("linear_attn.out_proj.weight"))),
                });
            }

            let (mut mixed, mut injw_a) = hc_block(fx, ps, t, a_norm, a_down, a_up, a_inj, h);
            let mut sub = match (&gdn_w, &attn_w) {
                (Some(w), _) => {
                    if is_prompt {
                        gdn_prompt(fx, ps, t, mixed, w, s_gdn[gdn_i], cs_gdn[gdn_i])
                    } else {
                        gdn_step(fx, ps, mixed, w, s_gdn[gdn_i], cs_gdn[gdn_i])
                    }
                }
                (_, Some(w)) => {
                    if is_prompt {
                        attn_prompt(fx, ps, t, mixed, w, k_cache[attn_i], v_cache[attn_i], cos_dev, sin_dev)
                    } else {
                        attn_step_tok(fx, ps, mixed, w, k_cache[attn_i], v_cache[attn_i], cos_dev, sin_dev, pos)
                    }
                }
                _ => unreachable!(),
            };

            // x1 = h + sub ⊗ injw_a
            let mut x1 = alloc_zeroed(t * HCT * 4);
            {
                let mut b = h;
                launch(fx.inj, 4, (t * 10) as u32, 256, 1, 0, &mut [
                    &mut b as *mut _ as *mut _, &mut sub as *mut _ as *mut _,
                    &mut injw_a as *mut _ as *mut _, &mut x1 as *mut _ as *mut _]);
            }
            free_dev(&mut h);
            free_dev(&mut sub);
            free_dev(&mut mixed);
            free_dev(&mut injw_a);

            // second HC + MoE
            let (mut mixed_m, mut injw_m) = hc_block(fx, ps, t, m_norm, m_down, m_up, m_inj, x1);
            let mut moe_w = MoEW {
                gate: to_f32_dev(&ckpt.f32(&L("mlp.gate.weight"))),
                sg: to_f32_dev(&ckpt.f32(&L("mlp.shared_expert.gate_proj.weight"))),
                su: to_f32_dev(&ckpt.f32(&L("mlp.shared_expert.up_proj.weight"))),
                sdn: to_f32_dev(&ckpt.f32(&L("mlp.shared_expert.down_proj.weight"))),
                sgate: to_f32_dev(&ckpt.f32(&L("mlp.shared_expert_gate.weight"))),
                gu_slab: upload_bf16_to_f32_dev(fx, &ckpt.bf16_raw(&L("mlp.experts.gate_up_proj"))),
                dn_slab: upload_bf16_to_f32_dev(fx, &ckpt.bf16_raw(&L("mlp.experts.down_proj"))),
            };
            let mut moe_out = moe_block(fx, ps, t, mixed_m, &moe_w, rw_dev);

            // h_next = x1 + moe ⊗ injw_m
            let mut h_next = alloc_zeroed(t * HCT * 4);
            {
                launch(fx.inj, 4, (t * 10) as u32, 256, 1, 0, &mut [
                    &mut x1 as *mut _ as *mut _, &mut moe_out as *mut _ as *mut _,
                    &mut injw_m as *mut _ as *mut _, &mut h_next as *mut _ as *mut _]);
            }

            free_dev(&mut a_norm);
            free_dev(&mut a_down);
            free_dev(&mut a_up);
            free_dev(&mut a_inj);
            free_dev(&mut m_norm);
            free_dev(&mut m_down);
            free_dev(&mut m_up);
            free_dev(&mut m_inj);
            if let Some(mut w) = gdn_w {
                free_dev(&mut w.qkv);
                free_dev(&mut w.conv);
                free_dev(&mut w.z);
                free_dev(&mut w.b);
                free_dev(&mut w.a);
                free_dev(&mut w.alog);
                free_dev(&mut w.dt);
                free_dev(&mut w.norm);
                free_dev(&mut w.out);
            }
            if let Some(mut w) = attn_w {
                free_dev(&mut w.q);
                free_dev(&mut w.k);
                free_dev(&mut w.v);
                free_dev(&mut w.o);
                free_dev(&mut w.qn);
                free_dev(&mut w.kn);
            }
            free_dev(&mut moe_w.gate);
            free_dev(&mut moe_w.sg);
            free_dev(&mut moe_w.su);
            free_dev(&mut moe_w.sdn);
            free_dev(&mut moe_w.sgate);
            free_dev(&mut moe_w.gu_slab);
            free_dev(&mut moe_w.dn_slab);
            free_dev(&mut mixed_m);
            free_dev(&mut injw_m);
            free_dev(&mut moe_out);
            free_dev(&mut x1);
            h_next
        }

        // ---------- PROMPT PASS (batched kernels) ----------
        let mut h_host = vec![0f32; PROMPT * HCT];
        for t in 0..PROMPT {
            let row = &embed[all_ids[t] * H..(all_ids[t] + 1) * H];
            for g in 0..HC {
                h_host[t * HCT + g * H..t * HCT + (g + 1) * H].copy_from_slice(row);
            }
        }
        let mut h = to_f32_dev(&h_host);
        let t_pass = Instant::now();
        for layer in 0..LAYERS {
            let lt = Instant::now();
            h = run_layer(&fx, &ps, &mut ckpt, layer, true, 0, h, PROMPT,
                          &s_gdn, &cs_gdn, &k_cache, &v_cache, cos_dev, sin_dev, rw_dev);
            println!(
                "  [prompt] layer {:2} ({}) {:.1} s",
                layer,
                if layer % 4 == 3 { "attn" } else { "gdn" },
                lt.elapsed().as_secs_f64()
            );
        }
        let mut mixed_final = mixer_block(&fx, &ps, PROMPT, mx_norm, mx_down, mx_up, h);
        free_dev(&mut h);
        for t in 0..PROMPT {
            gemv_row(&fx, ps.n2560, lm_head, mixed_final + (t * H * 4) as u64,
                     logits_dev + (t * V * 4) as u64, V as u32);
        }
        free_dev(&mut mixed_final);
        println!(
            "p13: prompt pass done in {:.1} s — logits for positions 0..7, greedy sampling",
            t_pass.elapsed().as_secs_f64()
        );

        let mut logits_all = vec![0f32; T_MAX * V];
        let prompt_logits = dtoh(logits_dev, PROMPT * V);
        logits_all[..PROMPT * V].copy_from_slice(&prompt_logits);

        let argmax_top = |row: &[f32]| -> (usize, f32, usize, f32) {
            let mut best = 0usize;
            let mut bv = f32::NEG_INFINITY;
            let mut second = 0usize;
            let mut sv = f32::NEG_INFINITY;
            for (i, &v) in row.iter().enumerate() {
                if v > bv {
                    sv = bv;
                    second = best;
                    bv = v;
                    best = i;
                } else if v > sv {
                    sv = v;
                    second = i;
                }
            }
            (best, bv, second, sv)
        };
        let (b, bv, s2, sv) = argmax_top(&logits_all[(PROMPT - 1) * V..PROMPT * V]);
        all_ids.push(b);
        println!("  pos 7 → token {b} (top-2 margin {bv:.4}−{sv:.4}={:.4}, runner-up {s2})", bv - sv);

        // ---------- DECODE PASSES (single-token kernels, persistent state) ----------
        for j in 1..=GEN {
            let pos = PROMPT - 1 + j;
            let t_pass = Instant::now();
            let mut h1_host = vec![0f32; HCT];
            let row = &embed[all_ids[pos] * H..(all_ids[pos] + 1) * H];
            for g in 0..HC {
                h1_host[g * H..(g + 1) * H].copy_from_slice(row);
            }
            let mut h = to_f32_dev(&h1_host);
            for layer in 0..LAYERS {
                h = run_layer(&fx, &ps, &mut ckpt, layer, false, pos, h, 1,
                              &s_gdn, &cs_gdn, &k_cache, &v_cache, cos_dev, sin_dev, rw_dev);
            }
            let mut mixed_final = mixer_block(&fx, &ps, 1, mx_norm, mx_down, mx_up, h);
            free_dev(&mut h);
            gemv_row(&fx, ps.n2560, lm_head, mixed_final,
                     logits_dev + (pos * V * 4) as u64, V as u32);
            free_dev(&mut mixed_final);
            let row_l = dtoh(logits_dev + (pos * V * 4) as u64, V);
            logits_all[pos * V..(pos + 1) * V].copy_from_slice(&row_l);
            let (b, bv, s2, sv) = argmax_top(&row_l);
            all_ids.push(b);
            println!(
                "  [decode {j}] pos {pos} → token {b} (top-2 margin {:.4}, runner-up {s2}) — {:.1} s",
                bv - sv,
                t_pass.elapsed().as_secs_f64()
            );
        }

        // ---------- artifacts for the oracle gate ----------
        let mut nan = 0usize;
        for &v in logits_all.iter() {
            if v.is_nan() {
                nan += 1;
            }
        }
        let mut bytes = Vec::with_capacity(logits_all.len() * 4);
        for v in &logits_all {
            bytes.extend_from_slice(&v.to_le_bytes());
        }
        std::fs::write(format!("{dbg_dir}/gpu-logits.f32"), bytes).unwrap();
        serde_json::to_writer(
            std::fs::File::create(format!("{dbg_dir}/gen-sequence.json")).unwrap(),
            &serde_json::json!({
                "prompt_ids": prompt_ids,
                "all_ids": all_ids,
                "prompt_len": PROMPT,
                "generated": GEN,
                "note": "gpu greedy trace; logits [12][248320] f32 in gpu-logits.f32; PLE skipped on BOTH sides (ple=None), QSA dense by construction (T=12<2048)",
            }),
        )
        .unwrap();
        println!("p13: token trace ({} tokens): {:?}", all_ids.len(), all_ids);
        println!("p13: logits [{T_MAX}][{V}] → {dbg_dir}/gpu-logits.f32 (NaN={nan})");
        println!("p13: GPU side done — run oracle/ref_gen_logits.py for the 5e-3 logit gate");
    }
}
