//! Probe 4 (Crow #8 + #10): the first engine numerics+transport gate.
//!
//! One REAL expert (expert 0 of `model.language_model.layers.0.mlp.experts.gate_up_proj`,
//! [1280, 2560]) read as NVFP4 blocks STRAIGHT from the CNQ4.5 container file, placed in
//! pinned host memory, and computed by a dequant-GEMV kernel that reads it via UVA
//! zero-copy — the strategy-Z cold path, end to end:
//!   CNQ4.5 file bytes → pinned host tier → kernel dequant (e2m1 × ue4m3 × global) → f32 out
//!
//! Verification:
//!   a) kernel result vs a CPU dequant of the SAME blocks: tight (transport/numerics exact)
//!   b) CPU dequant result vs the BF16 original matmul: within RTN error (informational + loose gate)
//!
//! Layouts per probe 2 / converter: block = 4 ue4m3 scale bytes + 32 B packed E2M1
//! (nibble j at bit 4j, LSB-first); value (sb, j) = e2m1 × ue4m3(scale[sb]) × global.

use std::ffi::CString;
use std::io::{Read, Seek, SeekFrom};
use std::time::Instant;

use cudarc::driver::sys::{self, CUdeviceptr, CUresult};
use cudarc::nvrtc::{compile_ptx_with_opts, CompileOptions};

const N_ROWS: usize = 1280; // gate+up outputs per expert
const K: usize = 2560; // input dim
const BLOCK_BYTES: usize = 36; // 4 scale bytes + 32 B packed E2M1
const BLOCKS_PER_ROW: usize = K / 64; // 40
const ROW_BYTES: usize = BLOCKS_PER_ROW * BLOCK_BYTES; // 1440
const EXPERT_BYTES: usize = N_ROWS * ROW_BYTES; // 1.844 MB

const KERNEL_SRC: &str = r#"
__device__ __forceinline__ float e2m1(unsigned int nib) {
    const float mag[8] = {0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f};
    float v = mag[nib & 0x7];
    return (nib & 0x8) ? -v : v;
}

__device__ __forceinline__ float ue4m3(unsigned int byte) {
    unsigned int e = (byte >> 3) & 0xF;
    unsigned int m = byte & 0x7;
    if (e == 0) return (float)m * 1.953125e-3f;          // m * 2^-9
    return (1.0f + (float)m / 8.0f) * exp2f((float)e - 7.0f);
}

// one thread block per output row; threads stride over the 40 NVFP4 blocks,
// dequantize and dot against x, then block-reduce
#define NR 1280
extern "C" __global__ void gemv_nvfp4(const unsigned char* __restrict__ w, // pinned host (UVA)
                                      const float* __restrict__ x,         // [2560] VRAM
                                      float* __restrict__ y,               // [NR+4] VRAM
                                      const float* __restrict__ gs_ptr) {  // [1] VRAM
    const float global_scale = gs_ptr[0];
    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    const int nthreads = blockDim.x;
    const unsigned char* rowp = w + (size_t)row * 1440u;

    float acc = 0.0f;
    for (int b = tid; b < 40; b += nthreads) {
        const unsigned char* blk = rowp + b * 36;
        float s0 = ue4m3(blk[0]) * global_scale;
        float s1 = ue4m3(blk[1]) * global_scale;
        float s2 = ue4m3(blk[2]) * global_scale;
        float s3 = ue4m3(blk[3]) * global_scale;
        const unsigned char* q = blk + 4;
        // decode 64 values: sub-block sb, element j -> byte (sb*16+j)/2, nibble (sb*16+j)%2
        #pragma unroll
        for (int sb = 0; sb < 4; sb++) {
            float s = (sb == 0) ? s0 : (sb == 1) ? s1 : (sb == 2) ? s2 : s3;
            float acc_s = 0.0f;
            #pragma unroll
            for (int j = 0; j < 16; j++) {
                int idx = sb * 16 + j;
                unsigned int byte = q[idx >> 1];
                unsigned int nib = (idx & 1) ? ((byte >> 4) & 0xF) : (byte & 0xF);
                acc_s += e2m1(nib) * x[b * 64 + sb * 16 + j];
            }
            acc += acc_s * s;
        }
    }

    __shared__ float red[256];
    red[tid] = acc;
    __syncthreads();
    for (int st = nthreads / 2; st > 0; st >>= 1) {
        if (tid < st) red[tid] += red[tid + st];
        __syncthreads();
    }
    if (tid == 0) {
        y[row] = red[0];
    }
}
"#;

fn ck(r: CUresult) {
    if r != CUresult::CUDA_SUCCESS {
        panic!("CUDA error: {r:?}");
    }
}

// ---- CPU side: identical decode for the reference ----
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

fn dequant_row(row: &[u8], global: f32) -> Vec<f32> {
    let mut out = vec![0.0f32; K];
    for b in 0..BLOCKS_PER_ROW {
        let blk = &row[b * BLOCK_BYTES..(b + 1) * BLOCK_BYTES];
        for sb in 0..4 {
            let s = ue4m3(blk[sb] as u32) * global;
            for j in 0..16 {
                let idx = sb * 16 + j;
                let byte = blk[4 + (idx >> 1)];
                let nib = if idx & 1 == 1 { (byte >> 4) & 0xF } else { byte & 0xF };
                out[b * 64 + sb * 16 + j] = e2m1(nib as u32) * s;
            }
        }
    }
    out
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let cnq = args
        .get(1)
        .cloned()
        .unwrap_or_else(|| "../converter/Qwen3.8-Flash-Next-CNQ4.5.cnq".to_string());
    let expert: usize = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(0);

    // ---- locate the expert slice inside the CNQ4.5 container ----
    let mut f = std::fs::File::open(&cnq).expect("open CNQ4.5");
    let file_len = f.metadata().unwrap().len();
    f.seek(SeekFrom::Start(file_len - 8)).unwrap();
    let mut b8 = [0u8; 8];
    f.read_exact(&mut b8).unwrap();
    let idx_len = u64::from_le_bytes(b8) as usize;
    f.seek(SeekFrom::Start(file_len - 8 - idx_len as u64)).unwrap();
    let mut idx_buf = vec![0u8; idx_len];
    f.read_exact(&mut idx_buf).unwrap();
    let idx: serde_json::Value = serde_json::from_slice(&idx_buf).unwrap();
    let blob_off = idx["blob_offset"].as_u64().unwrap() as usize;
    let tname = format!("model.language_model.layers.0.mlp.experts.gate_up_proj");
    let t = idx["tensors"]
        .as_array()
        .unwrap()
        .iter()
        .find(|t| t["name"] == tname)
        .expect("expert tensor in container");
    assert_eq!(t["dtype"], "nvfp4");
    let global = t["global_scale"].as_f64().unwrap() as f32;
    let expert_bytes_total = t["len"].as_u64().unwrap() as usize;
    assert_eq!(expert_bytes_total, 512 * EXPERT_BYTES, "shape mismatch");
    let slice_off = blob_off + t["offset"].as_u64().unwrap() as usize + expert * EXPERT_BYTES;
    println!(
        "p4: expert {expert} of {} — slice at file offset {}, {} B, global_scale={global:e}",
        tname, slice_off, EXPERT_BYTES
    );

    // read the slice, then place it in pinned device-mapped host memory
    f.seek(SeekFrom::Start(slice_off as u64)).unwrap();
    let mut slice = vec![0u8; EXPERT_BYTES];
    f.read_exact(&mut slice).unwrap();

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
        let mut fn_g = std::ptr::null_mut();
        ck(sys::cuModuleGetFunction(&mut fn_g, module, CString::new("gemv_nvfp4").unwrap().as_ptr()));

        let mut pinned: *mut std::ffi::c_void = std::ptr::null_mut();
        ck(sys::cuMemHostAlloc(
            &mut pinned,
            EXPERT_BYTES,
            sys::CU_MEMHOSTALLOC_PORTABLE | sys::CU_MEMHOSTALLOC_DEVICEMAP,
        ));
        std::ptr::copy_nonoverlapping(slice.as_ptr(), pinned as *mut u8, EXPERT_BYTES);
        let w_dev = pinned as CUdeviceptr;

        // deterministic input
        let mut x = vec![0.0f32; K];
        let mut s: u32 = 0x9E3779B9;
        for v in x.iter_mut() {
            s ^= s << 13;
            s ^= s >> 17;
            s ^= s << 5;
            *v = (s & 0xFFFF) as f32 / 65535.0 - 0.5;
        }

        let mut x_dev: CUdeviceptr = 0;
        let mut y_dev: CUdeviceptr = 0;
        ck(sys::cuMemAlloc_v2(&mut x_dev, (K * 4) as usize));
        ck(sys::cuMemAlloc_v2(&mut y_dev, (N_ROWS + 4) * 4));

        ck(sys::cuMemcpyHtoDAsync_v2(
            x_dev,
            x.as_ptr() as *const std::ffi::c_void,
            K * 4,
            std::ptr::null_mut(),
        ));
        ck(sys::cuStreamSynchronize(std::ptr::null_mut()));

        // launch: one block of 256 threads per output row
        let (grid, block): (u32, u32) = (N_ROWS as u32, 256);
        let mut gs = global;
        let mut gs_dev: CUdeviceptr = 0;
        ck(sys::cuMemAlloc_v2(&mut gs_dev, 4));
        ck(sys::cuMemcpyHtoD_v2(gs_dev, &mut gs as *mut f32 as *mut std::ffi::c_void, 4));
        let mut p_gsp = gs_dev as *mut std::ffi::c_void;
        let mut p_w = w_dev as *mut std::ffi::c_void;
        let mut p_x = x_dev as *mut std::ffi::c_void;
        let mut p_y = y_dev as *mut std::ffi::c_void;
        let args = &mut [
            &mut p_w as *mut _ as *mut _,
            &mut p_x as *mut _ as *mut _,
            &mut p_y as *mut _ as *mut _,
            &mut p_gsp as *mut _ as *mut _,
        ];
        let t = Instant::now();
        ck(sys::cuLaunchKernel(fn_g, grid, 1, 1, block, 1, 1, 0, std::ptr::null_mut(), args.as_mut_ptr(), std::ptr::null_mut()));
        ck(sys::cuStreamSynchronize(std::ptr::null_mut()));
        let wall = t.elapsed().as_secs_f64() * 1e3;

        let mut y_gpu = vec![0f32; N_ROWS + 4];
        ck(sys::cuMemcpyDtoH_v2(
            y_gpu.as_mut_ptr() as *mut std::ffi::c_void,
            y_dev,
            (N_ROWS + 4) * 4,
        ));

        // ---- CPU reference 1: dequant of the SAME blocks ----
        let t = Instant::now();
        let mut y_ref = vec![0f32; N_ROWS];
        for r in 0..N_ROWS {
            let w = dequant_row(&slice[r * ROW_BYTES..(r + 1) * ROW_BYTES], global);
            y_ref[r] = w.iter().zip(x.iter()).map(|(wv, xv)| wv * xv).sum();
        }
        let cpu_ms = t.elapsed().as_secs_f64() * 1e3;

        let mut max_rel = 0.0f32;
        for (g, r) in y_gpu.iter().zip(y_ref.iter()) {
            let rel = (g - r).abs() / r.abs().max(1e-6);
            max_rel = max_rel.max(rel);
        }
        println!(
            "a) kernel vs CPU-dequant: max_rel={max_rel:.2e} (gate 1e-2)  [gpu {:.1} ms | cpu {:.1} ms]",
            wall, cpu_ms
        );
        assert!(max_rel < 1e-2, "kernel/transport mismatch");

        // ---- CPU reference 2: BF16 original from the shard ----
        let models = "../models/Qwen3.8-Flash-Next-original";
        let midx: serde_json::Value = serde_json::from_slice(
            &std::fs::read(format!("{models}/model.safetensors.index.json")).unwrap(),
        )
        .unwrap();
        let ck_name = format!("model.language_model.layers.0.mlp.experts.gate_up_proj");
        let shard = midx["weight_map"][&ck_name].as_str().unwrap().to_string();
        let mut sf = std::fs::File::open(format!("{models}/{shard}")).unwrap();
        let mut n8 = [0u8; 8];
        sf.read_exact(&mut n8).unwrap();
        let hl = u64::from_le_bytes(n8) as usize;
        let mut hb = vec![0u8; hl];
        sf.read_exact(&mut hb).unwrap();
        let hdr: serde_json::Value = serde_json::from_slice(&hb).unwrap();
        let info = &hdr[&ck_name];
        let (off, end) = (
            info["data_offsets"][0].as_u64().unwrap() as usize,
            info["data_offsets"][1].as_u64().unwrap() as usize,
        );
        sf.seek(SeekFrom::Start(8 + hl as u64 + off as u64)).unwrap();
        let mut raw = vec![0u8; end - off];
        sf.read_exact(&mut raw).unwrap();
        // expert slice rows [0..1280) of [512*1280, 2560]
        let row_stride = K * 2;
        let mut y_bf16 = vec![0f32; N_ROWS];
        for r in 0..N_ROWS {
            let base = (expert * N_ROWS + r) * row_stride;
            let mut acc = 0.0f64;
            for kk in 0..K {
                let bits = u16::from_le_bytes([raw[base + kk * 2], raw[base + kk * 2 + 1]]);
                let w = f32::from_bits((bits as u32) << 16);
                acc += (w * x[kk]) as f64;
            }
            y_bf16[r] = acc as f32;
        }
        let mut rels = Vec::new();
        for (a, b) in y_ref.iter().zip(y_bf16.iter()) {
            rels.push((a - b).abs() / b.abs().max(1e-6));
        }
        rels.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let med = rels[rels.len() / 2];
        let p99 = rels[(rels.len() as f32 * 0.99) as usize];
        println!(
            "b) RTN vs BF16 original: median rel={med:.3} p99={p99:.3} max={:.3} (informational; loose gate 0.25)",
            rels[rels.len() - 1]
        );
        assert!(med < 0.25, "RTN output drift beyond loose gate");

        println!("p4: PASS — zero-copy NVFP4 GEMV from pinned host memory is numerically bound and transport-exact");
    }
}
