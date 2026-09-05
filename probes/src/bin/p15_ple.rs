//! Probe 15 (Crow #10): PLE gather + projection/gating/conv on GPU, f32, vs
//! the transformers reference (`oracle/export_ple_golden.py`: PLELayer of
//! layer index 1, real weights, input = the p13 12-token sequence, no cache —
//! n-gram history EOS-filled).
//!
//! Split of work (mirrors the engine design, spec 3.5):
//! - HOST: n-gram index math — shift-right-ignore-EOS history, per-position
//!   multiplier XOR (wrapping i64, same bit pattern as torch int64), mod by
//!   the checkpoint's prime head vocab sizes + offsets. Uses the checkpoint's
//!   stored I64 tables — nothing hashed is re-derived.
//! - GPU: row gather from a compact resident row cache (the 102 GB table
//!   never loads; host remaps ids to cache slots — the engine's hot-row
//!   pattern), key/value projections, 3 grouped RMSNorms (1+w), the
//!   sqrt-|g|·sign gate, sigmoid broadcast, dilated depthwise conv
//!   (kernel 4, dilation 3, left state 9) + silu, final add.
//!
//! Gates: ngram ids EXACT (i64), gathered embeddings exact (same bytes),
//! layer output < 5e-3, NaN=0.

use std::collections::BTreeMap;
use std::ffi::CString;
use std::io::{Read, Seek, SeekFrom};

use cudarc::driver::sys::{self, CUdeviceptr, CUfunction, CUresult};
use cudarc::nvrtc::{compile_ptx_with_opts, CompileOptions};

const T: usize = 12;
const EOS: i64 = 248044;
const NGRAM: usize = 3;
const CONTEXT: usize = NGRAM - 1;
const HEADS_PER_NGRAM: usize = 8;
const NHEADS: usize = (NGRAM - 1) * HEADS_PER_NGRAM; // 16
const EMB_DIM: usize = 160; // per head
const EMBED: usize = NHEADS * EMB_DIM; // 2560
const H: usize = 2560;
const HCT: usize = 4 * H; // 10240
const ROWS_PER_SHARD: i64 = 2_500_012;

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

// gather: out[t][h*160+d] = rows[slot[t*16+h]*160+d]; grid (16, T), block 160
extern "C" __global__ void gather_rows(const float* __restrict__ rows,
                                       const int* __restrict__ slot,
                                       float* __restrict__ out) {
    int h = blockIdx.x;
    int t = blockIdx.y;
    int d = threadIdx.x;
    int s = slot[t * 16 + h];
    out[(size_t)t * 2560 + h * 160 + d] = rows[(size_t)s * 160 + d];
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

// gate[t][s] = Σ_d key_normed[t][s·2560+d]·query_normed[t][s·2560+d] / √2560
// grid (4, T), block 256
extern "C" __global__ void gate_dot(const float* __restrict__ key,
                                    const float* __restrict__ query,
                                    float* __restrict__ gate) {
    int s = blockIdx.x;
    int t = blockIdx.y;
    int d = threadIdx.x;
    const float* kp = key + ((size_t)t * 4 + s) * 2560;
    const float* qp = query + ((size_t)t * 4 + s) * 2560;
    float acc = 0.0f;
    for (int i = d; i < 2560; i += 256) acc += kp[i] * qp[i];
    __shared__ float red[256];
    red[d] = acc;
    __syncthreads();
    for (int st = 128; st > 0; st >>= 1) {
        if (d < st) red[d] += red[d + st];
        __syncthreads();
    }
    if (d == 0) gate[t * 4 + s] = red[0] * rsqrtf(2560.0f);
}

// gate' = sqrt(max(|g|, 1e-6)) · sign(g); then per element:
// gated[t][s·2560+d] = sigmoid(gate'[t][s]) · value[t][d];  grid (4, T), block 256
extern "C" __global__ void gate_apply(const float* __restrict__ gate,
                                      const float* __restrict__ value,
                                      float* __restrict__ gate_signed,
                                      float* __restrict__ gated) {
    int s = blockIdx.x;
    int t = blockIdx.y;
    int d = threadIdx.x;
    if (d == 0) {
        float g = gate[t * 4 + s];
        gate_signed[t * 4 + s] = sqrtf(fmaxf(fabsf(g), 1e-6f)) * ((g > 0.0f) - (g < 0.0f));
    }
    __syncthreads();
    float sg = 1.0f / (1.0f + expf(-gate_signed[t * 4 + s]));
    for (int i = d; i < 2560; i += 256) {
        gated[((size_t)t * 4 + s) * 2560 + i] = sg * value[(size_t)t * 2560 + i];
    }
}

// output[t][c] = gated[t][c] + silu(Σ_k w[c·4+k]·gn[t + k·3 − 9][c])
// (left state 9 zeros = (kernel−1)·dilation, no cache)   grid (10240), block 32
extern "C" __global__ void conv_silu_add(const float* __restrict__ gn,
                                         const float* __restrict__ w,
                                         const float* __restrict__ gated,
                                         float* __restrict__ out,
                                         const int* __restrict__ t_p) {
    int tt = *t_p;
    int c = blockIdx.x;
    int t = threadIdx.x;
    if (t >= tt) return;
    float acc = 0.0f;
    for (int k = 0; k < 4; k++) {
        int src = t + k * 3 - 9;
        if (src >= 0) acc += w[c * 4 + k] * gn[(size_t)src * 10240 + c];
    }
    out[(size_t)t * 10240 + c] = gated[(size_t)t * 10240 + c]
        + acc / (1.0f + expf(-acc));
}
"#;

fn ck(r: CUresult) {
    if r != CUresult::CUDA_SUCCESS {
        panic!("CUDA error: {r:?}");
    }
}

fn read_bin(path: &str, elem: usize) -> Vec<u8> {
    let mut f = std::fs::File::open(path).unwrap();
    let mut v = Vec::new();
    f.read_to_end(&mut v).unwrap();
    assert_eq!(v.len(), elem, "{path}: size mismatch");
    v
}

fn to_f32(raw: &[u8]) -> Vec<f32> {
    raw.chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

fn to_i32(raw: &[u8]) -> Vec<i32> {
    raw.chunks_exact(4)
        .map(|c| i32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

fn to_i64(raw: &[u8]) -> Vec<i64> {
    raw.chunks_exact(8)
        .map(|c| i64::from_le_bytes([c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7]]))
        .collect()
}

/// checkpoint tensor location (safetensors)
struct TensorRef {
    shard: String,
    dtype: String,
    start: usize,
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
    let off = hdr[name]["data_offsets"][0].as_u64().unwrap() as usize;
    let end = hdr[name]["data_offsets"][1].as_u64().unwrap() as usize;
    TensorRef {
        shard,
        dtype: hdr[name]["dtype"].as_str().unwrap().to_string(),
        start: 8 + hl + off,
        nbytes: end - off,
    }
}

fn read_bytes(models: &str, name: &str) -> (Vec<u8>, String) {
    let r = locate(models, name);
    let mut fh = std::fs::File::open(format!("{models}/{}", r.shard)).unwrap();
    fh.seek(SeekFrom::Start(r.start as u64)).unwrap();
    let mut raw = vec![0u8; r.nbytes];
    fh.read_exact(&mut raw).unwrap();
    (raw, r.dtype)
}

fn read_f32(models: &str, name: &str) -> Vec<f32> {
    let (raw, dtype) = read_bytes(models, name);
    match dtype.as_str() {
        "F32" => to_f32(&raw),
        _ => raw
            .chunks_exact(2)
            .map(|c| f32::from_bits((u16::from_le_bytes([c[0], c[1]]) as u32) << 16))
            .collect(),
    }
}

fn read_i64(models: &str, name: &str) -> Vec<i64> {
    let (raw, _) = read_bytes(models, name);
    to_i64(&raw)
}

/// host-side n-gram index math (the engine does exactly this per token)
fn ngram_ids_for(ids: &[i64], multipliers: &[i64], vocab_sizes: &[i64], offsets: &[i64]) -> Vec<Vec<i64>> {
    let history: Vec<i64> = std::iter::repeat(EOS).take(CONTEXT).chain(ids.iter().copied()).collect();
    let n = history.len();
    // shifted_right_ignore_eos per shift
    let shifted: Vec<Vec<i64>> = (0..NGRAM)
        .map(|shift| {
            let mut eos_pos = vec![-1i64; n];
            for (i, &v) in history.iter().enumerate() {
                if v == EOS {
                    eos_pos[i] = i as i64;
                }
            }
            let mut prev_incl = vec![0i64; n];
            let mut run = i64::MIN;
            for i in 0..n {
                run = run.max(eos_pos[i]);
                prev_incl[i] = run;
            }
            (0..n)
                .map(|i| {
                    let prev = if i == 0 { -1 } else { prev_incl[i - 1] };
                    let segment_start = prev + 1;
                    let pos_in_seg = i as i64 - segment_start;
                    let src = i as i64 - shift as i64;
                    let gather = src.max(0) as usize;
                    let valid = pos_in_seg >= shift as i64 && src >= 0;
                    if valid { history[gather] } else { EOS }
                })
                .collect()
        })
        .collect();

    let mut blocks: Vec<Vec<Vec<i64>>> = Vec::new();
    for ngram in 2..=NGRAM {
        let start = (ngram - 2) * HEADS_PER_NGRAM;
        let mut rows_per_token = Vec::new();
        for i in 0..n {
            let mut mixed = (shifted[0][i] as u64).wrapping_mul(multipliers[0] as u64);
            for p in 1..ngram {
                mixed ^= (shifted[p][i] as u64).wrapping_mul(multipliers[p] as u64);
            }
            let mixed = mixed as i64; // torch.remainder: sign of divisor
            let mut row = vec![0i64; HEADS_PER_NGRAM];
            for h in 0..HEADS_PER_NGRAM {
                row[h] = mixed.rem_euclid(vocab_sizes[start + h]) + offsets[start + h];
            }
            rows_per_token.push(row);
        }
        blocks.push(rows_per_token);
    }
    // concat blocks in order, keep only the last T positions
    let mut out = Vec::new();
    for i in (history.len() - ids.len())..n {
        let mut row = Vec::new();
        for b in &blocks {
            row.extend_from_slice(&b[i]);
        }
        out.push(row);
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
    let dbg = "../probes/p15debug";
    let P = "model.language_model.layers.1.ple.";

    println!("p15: loading goldens + PLE weights (layer index 1) …");
    let token_ids = to_i32(&read_bin(&format!("{dbg}/ple-token-ids.i32"), T * 4));
    let golden_ngids = to_i64(&read_bin(&format!("{dbg}/ple-ngram-ids.i64"), T * NHEADS * 8));
    let golden_emb = to_f32(&read_bin(&format!("{dbg}/ple-embeddings.f32"), T * EMBED * 4));
    let golden_out = to_f32(&read_bin(&format!("{dbg}/ple-output.f32"), T * HCT * 4));
    let hidden = to_f32(&read_bin(&format!("{dbg}/ple-hidden.f32"), T * HCT * 4));

    let multipliers = read_i64(models, &format!("{P}ple_embedding.layer_multipliers"));
    let vocab_sizes = read_i64(models, &format!("{P}ple_embedding.ngram_heads_vocab_sizes"));
    let offsets = read_i64(models, &format!("{P}ple_embedding.ngram_heads_offsets"));
    let key_proj = read_f32(models, &format!("{P}key_proj.weight"));
    let value_proj = read_f32(models, &format!("{P}value_proj.weight"));
    let norm_key = read_f32(models, &format!("{P}norm_key.weight"));
    let norm_query = read_f32(models, &format!("{P}norm_query.weight"));
    let norm_conv = read_f32(models, &format!("{P}norm_conv.weight"));
    let conv1d = read_f32(models, &format!("{P}conv1d.weight"));
    assert_eq!(key_proj.len(), HCT * EMBED);
    assert_eq!(value_proj.len(), EMBED * H);
    assert_eq!(conv1d.len(), HCT * 4);

    let ids64: Vec<i64> = token_ids.iter().map(|&v| v as i64).collect();

    // ---- HOST: index math ----
    let ngids = ngram_ids_for(&ids64, &multipliers, &vocab_sizes, &offsets);
    let mut ngid_mismatch = 0usize;
    for t in 0..T {
        for h in 0..NHEADS {
            if ngids[t][h] != golden_ngids[t * NHEADS + h] {
                ngid_mismatch += 1;
            }
        }
    }
    println!("p15: ngram ids host vs golden: {ngid_mismatch} mismatches over {} (must be 0)", T * NHEADS);

    // unique rows + compact cache, remap ids to slots
    let flat: Vec<i64> = ngids.iter().flatten().copied().collect();
    let unique: BTreeMap<i64, usize> = flat.iter().copied().enumerate().map(|(i, v)| (v, i)).collect();
    let sorted_unique: Vec<i64> = unique.keys().copied().collect();
    let slot_of: BTreeMap<i64, usize> = sorted_unique.iter().enumerate().map(|(i, &v)| (v, i)).collect();
    let mut slots = vec![0i32; T * NHEADS];
    for t in 0..T {
        for h in 0..NHEADS {
            slots[t * NHEADS + h] = slot_of[&ngids[t][h]] as i32;
        }
    }
    let mut row_cache = vec![0f32; sorted_unique.len() * EMB_DIM];
    for (slot, &uid) in sorted_unique.iter().enumerate() {
        let shard_idx = uid / ROWS_PER_SHARD;
        let row = (uid % ROWS_PER_SHARD) as usize;
        let rel = {
            let index: serde_json::Value = serde_json::from_slice(
                &std::fs::read(format!("{models}/model.safetensors.index.json")).unwrap(),
            )
            .unwrap();
            index["weight_map"][format!("{P}ple_embedding.ngram_embedding.shard_{shard_idx}.weight")]
                .as_str()
                .unwrap()
                .to_string()
        };
        let mut fh = std::fs::File::open(format!("{models}/{rel}")).unwrap();
        let mut n8 = [0u8; 8];
        fh.read_exact(&mut n8).unwrap();
        let hl = u64::from_le_bytes(n8) as usize;
        let mut hb = vec![0u8; hl];
        fh.read_exact(&mut hb).unwrap();
        let hdr: serde_json::Value = serde_json::from_slice(&hb).unwrap();
        let full = format!("{P}ple_embedding.ngram_embedding.shard_{shard_idx}.weight");
        let off = hdr[&full]["data_offsets"][0].as_u64().unwrap() as usize;
        let base = 8 + hl + off;
        fh.seek(SeekFrom::Start((base + row * EMB_DIM * 2) as u64)).unwrap();
        let mut raw = vec![0u8; EMB_DIM * 2];
        fh.read_exact(&mut raw).unwrap();
        for (d, c) in raw.chunks_exact(2).enumerate() {
            let bits = (u16::from_le_bytes([c[0], c[1]]) as u32) << 16;
            row_cache[slot * EMB_DIM + d] = f32::from_bits(bits);
        }
    }
    println!("p15: row cache: {} unique rows × {} loaded", sorted_unique.len(), EMB_DIM);

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
        let fn_gather = get_fn("gather_rows");
        let fn_rmsg = get_fn("rms_group");
        let fn_gdot = get_fn("gate_dot");
        let fn_gapply = get_fn("gate_apply");
        let fn_conv = get_fn("conv_silu_add");

        let mut kdim_e = to_i32_dev(&[EMBED as i32]);
        let mut t_param = to_i32_dev(&[T as i32]);

        let mut w_key = to_f32_dev(&key_proj);
        let mut w_val = to_f32_dev(&value_proj);
        let mut w_nkey = to_f32_dev(&norm_key);
        let mut w_nquery = to_f32_dev(&norm_query);
        let mut w_nconv = to_f32_dev(&norm_conv);
        let mut w_conv = to_f32_dev(&conv1d);
        let mut rows_dev = to_f32_dev(&row_cache);
        let mut slots_dev = to_i32_dev(&slots);
        let mut hidden_dev = to_f32_dev(&hidden);

        // 1. gather embeddings [T][2560]
        let mut emb_dev = alloc_zeroed(T * EMBED * 4);
        {
            let mut e = emb_dev;
            launch(fn_gather, NHEADS as u32, T as u32, EMB_DIM as u32, 1, 0, &mut [
                &mut rows_dev as *mut _ as *mut _, &mut slots_dev as *mut _ as *mut _,
                &mut e as *mut _ as *mut _]);
        }

        // 2. key path: proj [T][10240] → norm_key
        let mut key_dev = alloc_zeroed(T * HCT * 4);
        for t in 0..T {
            let mut xt = emb_dev + (t * EMBED * 4) as u64;
            let mut yt = key_dev + (t * HCT * 4) as u64;
            launch(fn_gemv, HCT as u32, 1, 256, 1, 0, &mut [
                &mut w_key as *mut _ as *mut _, &mut xt as *mut _ as *mut _,
                &mut yt as *mut _ as *mut _, &mut kdim_e as *mut _ as *mut _]);
        }
        let mut key_normed = alloc_zeroed(T * HCT * 4);
        {
            let mut k = key_dev;
            launch(fn_rmsg, 4, T as u32, 256, 1, 0, &mut [
                &mut k as *mut _ as *mut _, &mut w_nkey as *mut _ as *mut _,
                &mut key_normed as *mut _ as *mut _]);
        }

        // 3. value path: proj [T][2560]
        let mut value_dev = alloc_zeroed(T * H * 4);
        for t in 0..T {
            let mut xt = emb_dev + (t * EMBED * 4) as u64;
            let mut yt = value_dev + (t * H * 4) as u64;
            launch(fn_gemv, H as u32, 1, 256, 1, 0, &mut [
                &mut w_val as *mut _ as *mut _, &mut xt as *mut _ as *mut _,
                &mut yt as *mut _ as *mut _, &mut kdim_e as *mut _ as *mut _]);
        }

        // 4. query path: norm_query(hidden)
        let mut query_normed = alloc_zeroed(T * HCT * 4);
        {
            let mut h = hidden_dev;
            launch(fn_rmsg, 4, T as u32, 256, 1, 0, &mut [
                &mut h as *mut _ as *mut _, &mut w_nquery as *mut _ as *mut _,
                &mut query_normed as *mut _ as *mut _]);
        }

        // 5. gate + apply: gated [T][10240] = sigmoid(√|g|·sign)·value broadcast
        let mut gate_dev = alloc_zeroed(T * 4 * 4);
        {
            let mut k = key_normed;
            let mut q = query_normed;
            launch(fn_gdot, 4, T as u32, 256, 1, 0, &mut [
                &mut k as *mut _ as *mut _, &mut q as *mut _ as *mut _,
                &mut gate_dev as *mut _ as *mut _]);
        }
        let mut gated_dev = alloc_zeroed(T * HCT * 4);
        let mut gate_signed = alloc_zeroed(T * 4 * 4);
        {
            let mut g = gate_dev;
            let mut v = value_dev;
            launch(fn_gapply, 4, T as u32, 256, 1, 0, &mut [
                &mut g as *mut _ as *mut _, &mut v as *mut _ as *mut _,
                &mut gate_signed as *mut _ as *mut _, &mut gated_dev as *mut _ as *mut _]);
        }

        // 6. norm_conv(gated) then dilated conv + silu + add
        let mut gated_normed = alloc_zeroed(T * HCT * 4);
        {
            let mut g = gated_dev;
            launch(fn_rmsg, 4, T as u32, 256, 1, 0, &mut [
                &mut g as *mut _ as *mut _, &mut w_nconv as *mut _ as *mut _,
                &mut gated_normed as *mut _ as *mut _]);
        }
        let mut out_dev = alloc_zeroed(T * HCT * 4);
        {
            let mut gn = gated_normed;
            let mut gd = gated_dev;
            launch(fn_conv, HCT as u32, 1, 32, 1, 0, &mut [
                &mut gn as *mut _ as *mut _, &mut w_conv as *mut _ as *mut _,
                &mut gd as *mut _ as *mut _, &mut out_dev as *mut _ as *mut _,
                &mut t_param as *mut _ as *mut _]);
        }

        // stage dumps for pinpointing
        let dbg_all = |tag: &str, v: &[f32]| {
            let mut b = Vec::with_capacity(v.len() * 4);
            for x in v {
                b.extend_from_slice(&x.to_le_bytes());
            }
            std::fs::write(format!("{dbg}/gpu-{tag}.f32"), b).unwrap();
        };
        dbg_all("key-normed", &dtoh(key_normed, T * HCT));
        dbg_all("value", &dtoh(value_dev, T * H));
        dbg_all("query-normed", &dtoh(query_normed, T * HCT));
        dbg_all("gate-signed", &dtoh(gate_signed, T * 4));
        dbg_all("gated", &dtoh(gated_dev, T * HCT));
        dbg_all("gated-normed", &dtoh(gated_normed, T * HCT));

        // ---- gates ----
        let gpu_emb = dtoh(emb_dev, T * EMBED);
        let mut emb_max = 0.0f32;
        let mut emb_mismatch = 0usize;
        for i in 0..T * EMBED {
            emb_max = emb_max.max((gpu_emb[i] - golden_emb[i]).abs());
            if gpu_emb[i].to_bits() != golden_emb[i].to_bits() {
                emb_mismatch += 1;
            }
        }
        let gpu_out = dtoh(out_dev, T * HCT);
        let mut nan = 0usize;
        let mut out_max = 0.0f32;
        for i in 0..T * HCT {
            if gpu_out[i].is_nan() {
                nan += 1;
            }
            out_max = out_max.max((gpu_out[i] - golden_out[i]).abs());
        }
        println!("p15: embeddings gpu vs golden: max_abs={emb_max:.3e} bitmismatches={emb_mismatch} (expect 0/0 — same bytes)");
        println!("p15: output    gpu vs golden: max_abs={out_max:.3e} (tol 5e-3) NaN={nan}");

        let ok = ngid_mismatch == 0 && emb_mismatch == 0 && out_max < 5e-3 && nan == 0;
        if ok {
            println!("p15: PASS — PLE gather + projections + gating + dilated conv match the transformers reference");
        } else {
            println!("p15: FAIL");
            std::process::exit(1);
        }
    }
}
