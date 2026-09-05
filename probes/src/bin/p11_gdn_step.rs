//! Probe 11 (Crow #10 → #11 prep): GDN DECODE STEP with persistent state —
//! the kernel shape the generation loop needs.
//!
//! Prompt phase: the p6 batched chain over T=8, but the delta rule persists its
//! recurrent state into GLOBAL memory (`delta_rule_persist`, one block per
//! v-head, thread d owns state column d — no shared memory, no atomics needed).
//! conv_state after the prompt = the last 3 pre-conv mixed_qkv rows.
//!
//! Step phase: the SAME 8 tokens re-fed one at a time from zero state through
//! the decode-step kernels — `conv_step` (state + shift + silu), single-token
//! GEMVs, `delta_rule_step` (one recurrence step against the global state).
//!
//! Gates vs `oracle/golden/layer0-gdn-step-*.f32` (exported by
//! `oracle/export_gdn_step_golden.py`, stepping == batched golden 4.8e-7):
//!   - per-step y and core outputs
//!   - final recurrent state S (stepping AND batched-persist paths)
//!   - final conv_state
//! plus sanity: prompt y vs the p6 golden.
//!
//! Scalars via device buffers (p5 lesson); HtoD async + sync (WDDM rule).

use std::ffi::CString;
use std::io::{Read, Seek, SeekFrom};
use std::time::Instant;

use cudarc::driver::sys::{self, CUdeviceptr, CUfunction, CUresult};
use cudarc::nvrtc::{compile_ptx_with_opts, CompileOptions};

const T: usize = 8;
const H: usize = 2560;
const NUM_K_HEADS: usize = 16;
const NUM_V_HEADS: usize = 48;
const DK: usize = 128;
const DV: usize = 128;
const KEY_DIM: usize = NUM_K_HEADS * DK; // 2048
const VALUE_DIM: usize = NUM_V_HEADS * DV; // 6144
const CONV_DIM: usize = KEY_DIM * 2 + VALUE_DIM; // 10240
const CONV_K: usize = 4;
const STATE_BYTES: usize = DK * DV * 4;

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

// single-token causal conv step: reads conv_state[ch][3] + new mixed value,
// writes silu(conv) for the token and shifts the state
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

// batched prompt recurrence: per-step outputs AND final state in global memory
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

// single decode step: one token against the persistent global state
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
    let off = info["data_offsets"][0].as_u64().unwrap() as usize;
    let end = info["data_offsets"][1].as_u64().unwrap() as usize;
    fh.seek(SeekFrom::Start((8 + hl + off) as u64)).unwrap();
    let mut raw = vec![0u8; end - off];
    fh.read_exact(&mut raw).unwrap();
    match info["dtype"].as_str().unwrap() {
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

fn transpose_host(v: &[f32], rows: usize, cols: usize) -> Vec<f32> {
    let mut out = vec![0f32; rows * cols];
    for i in 0..rows {
        for j in 0..cols {
            out[j * rows + i] = v[i * cols + j];
        }
    }
    out
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
        wm[format!("model.language_model.layers.0.linear_attn.{sub}")]
            .as_str()
            .unwrap()
            .to_string()
    };

    let qkv_w = bf16_file_to_f32(models, "model.language_model.layers.0.linear_attn.in_proj_qkv.weight", &p("in_proj_qkv.weight"));
    let conv_w = bf16_file_to_f32(models, "model.language_model.layers.0.linear_attn.conv1d.weight", &p("conv1d.weight"));
    let z_w = bf16_file_to_f32(models, "model.language_model.layers.0.linear_attn.in_proj_z.weight", &p("in_proj_z.weight"));
    let b_w = bf16_file_to_f32(models, "model.language_model.layers.0.linear_attn.in_proj_b.weight", &p("in_proj_b.weight"));
    let a_w = bf16_file_to_f32(models, "model.language_model.layers.0.linear_attn.in_proj_a.weight", &p("in_proj_a.weight"));
    let a_log = bf16_file_to_f32(models, "model.language_model.layers.0.linear_attn.A_log", &p("A_log"));
    let dt_bias = bf16_file_to_f32(models, "model.language_model.layers.0.linear_attn.dt_bias", &p("dt_bias"));
    let norm_w = bf16_file_to_f32(models, "model.language_model.layers.0.linear_attn.norm.weight", &p("norm.weight"));
    let out_w = bf16_file_to_f32(models, "model.language_model.layers.0.linear_attn.out_proj.weight", &p("out_proj.weight"));
    println!("p11: layer-0 GDN weights loaded (9 tensors, f32)");

    let x_in = read_bin_f32("../oracle/golden/layer0-gdn-input.f32", T * H);
    let golden_y = read_bin_f32("../oracle/golden/layer0-gdn-output.f32", T * H);
    let g_core = read_bin_f32("../oracle/golden/layer0-gdn-step-core.f32", T * NUM_V_HEADS * DV);
    let g_s = read_bin_f32("../oracle/golden/layer0-gdn-step-s.f32", NUM_V_HEADS * DK * DV);
    let g_conv = read_bin_f32("../oracle/golden/layer0-gdn-step-conv.f32", CONV_DIM * 3);
    let g_normed = read_bin_f32("../oracle/golden/layer0-gdn-step-normed.f32", T * VALUE_DIM);
    let g_y = read_bin_f32("../oracle/golden/layer0-gdn-step-y.f32", T * H);

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
        let fn_conv = get_fn("conv_silu");
        let fn_conv_step = get_fn("conv_step");
        let fn_l2 = get_fn("l2norm_repeat");
        let fn_bg = get_fn("beta_g");
        let fn_persist = get_fn("delta_rule_persist");
        let fn_step = get_fn("delta_rule_step");
        let fn_rms = get_fn("rmsnorm_gated");

        let mut kdim_h = to_i32_dev(&[H as i32]);
        let mut kdim_v = to_i32_dev(&[VALUE_DIM as i32]);
        let mut t_param = to_i32_dev(&[T as i32]);
        let mut steps_param = to_i32_dev(&[T as i32]);

        let mut wqkv = to_f32_dev(&qkv_w);
        let mut wconv = to_f32_dev(&conv_w);
        let mut wz = to_f32_dev(&z_w);
        let mut wb = to_f32_dev(&b_w);
        let mut wa = to_f32_dev(&a_w);
        let mut wa_log = to_f32_dev(&a_log);
        let mut wdt = to_f32_dev(&dt_bias);
        let mut wnorm = to_f32_dev(&norm_w);
        let mut wout = to_f32_dev(&out_w);
        let mut x_dev = to_f32_dev(&x_in);

        let t_start = Instant::now();

        // ============ PROMPT PHASE (batched, state persisted) ============
        let mut mq = alloc_zeroed(T * CONV_DIM * 4);
        for t in 0..T {
            let mut xt = x_dev + (t * H * 4) as u64;
            let mut yt = mq + (t * CONV_DIM * 4) as u64;
            launch(fn_gemv, CONV_DIM as u32, 1, 256, 1, 0, &mut [
                &mut wqkv as *mut _ as *mut _, &mut xt as *mut _ as *mut _,
                &mut yt as *mut _ as *mut _, &mut kdim_h as *mut _ as *mut _]);
        }
        let mq_host = dtoh(mq, T * CONV_DIM);
        let mq_t = transpose_host(&mq_host, T, CONV_DIM);
        let mut mq_t_dev = to_f32_dev(&mq_t);
        let mut conv_out = alloc_zeroed(T * CONV_DIM * 4);
        launch(fn_conv, CONV_DIM as u32, 1, T as u32, 1, 0, &mut [
            &mut mq_t_dev as *mut _ as *mut _, &mut wconv as *mut _ as *mut _,
            &mut conv_out as *mut _ as *mut _, &mut t_param as *mut _ as *mut _]);
        let conv_t = transpose_host(&dtoh(conv_out, T * CONV_DIM), CONV_DIM, T);

        let mut q_dev = alloc_zeroed(T * KEY_DIM * 4);
        let mut k_dev = alloc_zeroed(T * KEY_DIM * 4);
        let mut v_dev = alloc_zeroed(T * VALUE_DIM * 4);
        for t in 0..T {
            let base = t * CONV_DIM;
            let qh = &conv_t[base..base + KEY_DIM];
            let kh = &conv_t[base + KEY_DIM..base + 2 * KEY_DIM];
            let vh = &conv_t[base + 2 * KEY_DIM..base + CONV_DIM];
            ck(sys::cuMemcpyHtoDAsync_v2(q_dev + (t * KEY_DIM * 4) as u64, qh.as_ptr() as *const std::ffi::c_void, KEY_DIM * 4, std::ptr::null_mut()));
            ck(sys::cuMemcpyHtoDAsync_v2(k_dev + (t * KEY_DIM * 4) as u64, kh.as_ptr() as *const std::ffi::c_void, KEY_DIM * 4, std::ptr::null_mut()));
            ck(sys::cuMemcpyHtoDAsync_v2(v_dev + (t * VALUE_DIM * 4) as u64, vh.as_ptr() as *const std::ffi::c_void, VALUE_DIM * 4, std::ptr::null_mut()));
        }
        ck(sys::cuStreamSynchronize(std::ptr::null_mut()));

        let mut z_dev = alloc_zeroed(T * VALUE_DIM * 4);
        for t in 0..T {
            let mut xt = x_dev + (t * H * 4) as u64;
            let mut yt = z_dev + (t * VALUE_DIM * 4) as u64;
            launch(fn_gemv, VALUE_DIM as u32, 1, 256, 1, 0, &mut [
                &mut wz as *mut _ as *mut _, &mut xt as *mut _ as *mut _,
                &mut yt as *mut _ as *mut _, &mut kdim_h as *mut _ as *mut _]);
        }
        let mut b_dev = alloc_zeroed(T * NUM_V_HEADS * 4);
        for t in 0..T {
            let mut xt = x_dev + (t * H * 4) as u64;
            let mut yt = b_dev + (t * NUM_V_HEADS * 4) as u64;
            launch(fn_gemv, NUM_V_HEADS as u32, 1, 256, 1, 0, &mut [
                &mut wb as *mut _ as *mut _, &mut xt as *mut _ as *mut _,
                &mut yt as *mut _ as *mut _, &mut kdim_h as *mut _ as *mut _]);
        }
        let mut a_dev = alloc_zeroed(T * NUM_V_HEADS * 4);
        for t in 0..T {
            let mut xt = x_dev + (t * H * 4) as u64;
            let mut yt = a_dev + (t * NUM_V_HEADS * 4) as u64;
            launch(fn_gemv, NUM_V_HEADS as u32, 1, 256, 1, 0, &mut [
                &mut wa as *mut _ as *mut _, &mut xt as *mut _ as *mut _,
                &mut yt as *mut _ as *mut _, &mut kdim_h as *mut _ as *mut _]);
        }
        let mut beta_dev = alloc_zeroed(T * NUM_V_HEADS * 4);
        let mut g_dev = alloc_zeroed(T * NUM_V_HEADS * 4);
        launch(fn_bg, 1, 1, (T * NUM_V_HEADS) as u32, 1, 0, &mut [
            &mut b_dev as *mut _ as *mut _, &mut a_dev as *mut _ as *mut _,
            &mut wa_log as *mut _ as *mut _, &mut wdt as *mut _ as *mut _,
            &mut beta_dev as *mut _ as *mut _, &mut g_dev as *mut _ as *mut _]);

        let mut qr_dev = alloc_zeroed(T * NUM_V_HEADS * DK * 4);
        let mut kr_dev = alloc_zeroed(T * NUM_V_HEADS * DK * 4);
        launch(fn_l2, NUM_V_HEADS as u32, T as u32, DK as u32, 1, 0, &mut [
            &mut q_dev as *mut _ as *mut _, &mut k_dev as *mut _ as *mut _,
            &mut qr_dev as *mut _ as *mut _, &mut kr_dev as *mut _ as *mut _]);

        let mut s_prompt = alloc_zeroed(NUM_V_HEADS * DK * DV * 4);
        let mut core_p = alloc_zeroed(T * VALUE_DIM * 4);
        launch(fn_persist, NUM_V_HEADS as u32, 1, DV as u32, 1, 0, &mut [
            &mut qr_dev as *mut _ as *mut _, &mut kr_dev as *mut _ as *mut _,
            &mut v_dev as *mut _ as *mut _, &mut g_dev as *mut _ as *mut _,
            &mut beta_dev as *mut _ as *mut _, &mut core_p as *mut _ as *mut _,
            &mut s_prompt as *mut _ as *mut _, &mut steps_param as *mut _ as *mut _]);

        let mut normed_p = alloc_zeroed(T * VALUE_DIM * 4);
        launch(fn_rms, NUM_V_HEADS as u32, T as u32, DV as u32, 1, 0, &mut [
            &mut core_p as *mut _ as *mut _, &mut z_dev as *mut _ as *mut _,
            &mut wnorm as *mut _ as *mut _, &mut normed_p as *mut _ as *mut _]);
        let mut y_prompt = alloc_zeroed(T * H * 4);
        for t in 0..T {
            let mut xt = normed_p + (t * VALUE_DIM * 4) as u64;
            let mut yt = y_prompt + (t * H * 4) as u64;
            launch(fn_gemv, H as u32, 1, 256, 1, 0, &mut [
                &mut wout as *mut _ as *mut _, &mut xt as *mut _ as *mut _,
                &mut yt as *mut _ as *mut _, &mut kdim_v as *mut _ as *mut _]);
        }

        // conv_state after the prompt = last 3 pre-conv rows (mq_t is [chan][T])
        let mut cs_host = vec![0f32; CONV_DIM * 3];
        for ch in 0..CONV_DIM {
            for (i, tt) in [T - 3, T - 2, T - 1].iter().enumerate() {
                cs_host[ch * 3 + i] = mq_t[ch * T + tt];
            }
        }
        let mut cs_dev = to_f32_dev(&cs_host);

        let prompt_y = dtoh(y_prompt, T * H);
        let mut prompt_max = 0.0f32;
        for i in 0..T * H {
            prompt_max = prompt_max.max((prompt_y[i] - golden_y[i]).abs());
        }
        println!("p11: prompt (batched+persist) vs p6 golden: max_abs={prompt_max:.3e}");
        assert!(prompt_max < 5e-3, "prompt path diverged");

        // ============ STEP PHASE (8 single-token steps from zero state) ============
        let mut s_step = alloc_zeroed(NUM_V_HEADS * DK * DV * 4); // zero state
        let mut cs_step = alloc_zeroed(CONV_DIM * 3 * 4);         // zero conv state
        let mut max_y = 0.0f32;
        let mut max_core = 0.0f32;
        let mut max_normed = 0.0f32;
        for t in 0..T {
            // single-token projections
            let mut mq1 = alloc_zeroed(CONV_DIM * 4);
            let mut cout = alloc_zeroed(CONV_DIM * 4);
            let mut xt = x_dev + (t * H * 4) as u64;
            launch(fn_gemv, CONV_DIM as u32, 1, 256, 1, 0, &mut [
                &mut wqkv as *mut _ as *mut _, &mut xt as *mut _ as *mut _,
                &mut mq1 as *mut _ as *mut _, &mut kdim_h as *mut _ as *mut _]);
            launch(fn_conv_step, (CONV_DIM as u32 + 255) / 256, 1, 256, 1, 0, &mut [
                &mut mq1 as *mut _ as *mut _, &mut wconv as *mut _ as *mut _,
                &mut cs_step as *mut _ as *mut _, &mut cout as *mut _ as *mut _]);
            // split q/k/v for the single token
            let cout_h = dtoh(cout, CONV_DIM);
            let mut q1 = to_f32_dev(&cout_h[..KEY_DIM]);
            let mut k1 = to_f32_dev(&cout_h[KEY_DIM..2 * KEY_DIM]);
            let mut v1 = to_f32_dev(&cout_h[2 * KEY_DIM..]);
            // z / b / a
            let mut z1 = alloc_zeroed(VALUE_DIM * 4);
            launch(fn_gemv, VALUE_DIM as u32, 1, 256, 1, 0, &mut [
                &mut wz as *mut _ as *mut _, &mut xt as *mut _ as *mut _,
                &mut z1 as *mut _ as *mut _, &mut kdim_h as *mut _ as *mut _]);
            let mut b1 = alloc_zeroed(NUM_V_HEADS * 4);
            launch(fn_gemv, NUM_V_HEADS as u32, 1, 256, 1, 0, &mut [
                &mut wb as *mut _ as *mut _, &mut xt as *mut _ as *mut _,
                &mut b1 as *mut _ as *mut _, &mut kdim_h as *mut _ as *mut _]);
            let mut a1 = alloc_zeroed(NUM_V_HEADS * 4);
            launch(fn_gemv, NUM_V_HEADS as u32, 1, 256, 1, 0, &mut [
                &mut wa as *mut _ as *mut _, &mut xt as *mut _ as *mut _,
                &mut a1 as *mut _ as *mut _, &mut kdim_h as *mut _ as *mut _]);
            let mut beta1 = alloc_zeroed(NUM_V_HEADS * 4);
            let mut g1 = alloc_zeroed(NUM_V_HEADS * 4);
            launch(fn_bg, 1, 1, NUM_V_HEADS as u32, 1, 0, &mut [
                &mut b1 as *mut _ as *mut _, &mut a1 as *mut _ as *mut _,
                &mut wa_log as *mut _ as *mut _, &mut wdt as *mut _ as *mut _,
                &mut beta1 as *mut _ as *mut _, &mut g1 as *mut _ as *mut _]);
            // l2norm + repeat for the single token
            let mut q1r = alloc_zeroed(NUM_V_HEADS * DK * 4);
            let mut k1r = alloc_zeroed(NUM_V_HEADS * DK * 4);
            launch(fn_l2, NUM_V_HEADS as u32, 1, DK as u32, 1, 0, &mut [
                &mut q1 as *mut _ as *mut _, &mut k1 as *mut _ as *mut _,
                &mut q1r as *mut _ as *mut _, &mut k1r as *mut _ as *mut _]);
            // one recurrence step against the persistent state
            let mut core1 = alloc_zeroed(VALUE_DIM * 4);
            launch(fn_step, NUM_V_HEADS as u32, 1, DV as u32, 1, 0, &mut [
                &mut s_step as *mut _ as *mut _, &mut q1r as *mut _ as *mut _,
                &mut k1r as *mut _ as *mut _, &mut v1 as *mut _ as *mut _,
                &mut g1 as *mut _ as *mut _, &mut beta1 as *mut _ as *mut _,
                &mut core1 as *mut _ as *mut _]);
            // rmsnorm + out_proj
            let mut normed1 = alloc_zeroed(VALUE_DIM * 4);
            launch(fn_rms, NUM_V_HEADS as u32, 1, DV as u32, 1, 0, &mut [
                &mut core1 as *mut _ as *mut _, &mut z1 as *mut _ as *mut _,
                &mut wnorm as *mut _ as *mut _, &mut normed1 as *mut _ as *mut _]);
            let mut y1 = alloc_zeroed(H * 4);
            launch(fn_gemv, H as u32, 1, 256, 1, 0, &mut [
                &mut wout as *mut _ as *mut _, &mut normed1 as *mut _ as *mut _,
                &mut y1 as *mut _ as *mut _, &mut kdim_v as *mut _ as *mut _]);

            // gates for this step
            let core_h = dtoh(core1, VALUE_DIM);
            let y_h = dtoh(y1, H);
            let normed_h = dtoh(normed1, VALUE_DIM);
            for i in 0..VALUE_DIM {
                max_core = max_core.max((core_h[i] - g_core[t * VALUE_DIM + i]).abs());
                max_normed = max_normed.max((normed_h[i] - g_normed[t * VALUE_DIM + i]).abs());
            }
            for i in 0..H {
                max_y = max_y.max((y_h[i] - g_y[t * H + i]).abs());
            }
        }

        let s_step_h = dtoh(s_step, NUM_V_HEADS * DK * DV);
        let s_prompt_h = dtoh(s_prompt, NUM_V_HEADS * DK * DV);
        let cs_step_h = dtoh(cs_step, CONV_DIM * 3);
        let mut max_s_step = 0.0f32;
        let mut max_s_prompt = 0.0f32;
        let mut max_cs = 0.0f32;
        for i in 0..NUM_V_HEADS * DK * DV {
            max_s_step = max_s_step.max((s_step_h[i] - g_s[i]).abs());
            max_s_prompt = max_s_prompt.max((s_prompt_h[i] - g_s[i]).abs());
        }
        for i in 0..CONV_DIM * 3 {
            max_cs = max_cs.max((cs_step_h[i] - g_conv[i]).abs());
        }

        let gpu_ms = t_start.elapsed().as_secs_f64() * 1e3;
        println!("p11: per-step y     vs golden: max_abs={max_y:.3e}");
        println!("p11: per-step core  vs golden: max_abs={max_core:.3e}");
        println!("p11: per-step normed vs golden: max_abs={max_normed:.3e}");
        println!("p11: S stepping     vs golden: max_abs={max_s_step:.3e}");
        println!("p11: S batched-persist vs golden: max_abs={max_s_prompt:.3e}");
        println!("p11: conv_state stepping vs golden: max_abs={max_cs:.3e}");
        println!("p11: total (prompt + 8 steps, sync-per-launch): {gpu_ms:.2} ms");

        let ok = max_y < 5e-3
            && max_core < 5e-3
            && max_normed < 5e-3
            && max_s_step < 5e-3
            && max_s_prompt < 5e-3
            && max_cs < 5e-3;
        if ok {
            println!("p11: PASS — GDN decode step with persistent state matches reference stepping (batched and stepping paths agree)");
        } else {
            println!("p11: FAIL");
            std::process::exit(1);
        }
    }
}
