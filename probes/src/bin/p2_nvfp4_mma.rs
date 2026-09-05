//! Probe 2 (Crow #188, next step a): does ONE `mma.sync…kind::mxf4nvf4` kernel, compiled
//! per NVRTC, compute on the RTX 5090 (sm_120)?
//!
//! Instruction copied verbatim from llama.cpp `ggml/src/ggml-cuda/mma.cuh`
//! (`mma_block_scaled_fp4`, GGML_TYPE_NVFP4 branch, upstream master 2026-09-01).
//! cubecl `crates/cubecl-cpp/src/cuda/mma/manual.rs` independently lists the same
//! combination for arch >= 120 && < 130: E2M1 x E2M1, k64, scales E4M3, scales_factor 4.
//!
//! Findings kept in the code because they cost debugging rounds:
//! - The instruction is NOT accepted on the plain `sm_120` target: ptxas says
//!   "Feature '.kind::mxf4nvf4' not supported on .target 'sm_120'". NVRTC must target
//!   `compute_120a` (arch-specific), matching llama.cpp's `120a-real` build flag.
//! - The test data must be free of linear structure: the first data generator had
//!   period 16 along k (a[m][k+16]==a[m][k], b[k+16][n]==b[k][n]), which made every
//!   scale-byte assignment and every joint A/B k-permutation produce identical output.
//!   The generators below mix in `(m+3)*k` / `(k+3)*(n+3)` mod terms to break it.
//! - Permuting A and B identically along k cannot change the matmul (index relabeling),
//!   so nibble order must be tested per operand, never jointly.
//!
//! One warp computes one m16n8k64 tile; the kernel runs exactly one mma.sync. All values
//! are dyadic (e2m1 values, ue4m3 scales 0.5/1/2), so every partial sum is exactly
//! representable in f32 — the expected max delta on success is 0.0 in ANY accumulation
//! order. Each config compares against the reference computed from its own packed bits
//! under one hypothesis (hardware reads e2m1 nibbles LSB-first; scale byte i applies to
//! k-sub-block i). A match pins that hypothesis; e2/e3 swap the nibble hypothesis per
//! operand, e5 is the transport sanity check.

use cudarc::driver::result::DriverError;
use cudarc::driver::safe::{CudaContext, LaunchConfig, PushKernelArg};
use cudarc::nvrtc::{compile_ptx_with_opts, CompileOptions};

const MMA_SRC: &str = r#"
extern "C" __global__ void mma_nvfp4(const unsigned int* a_frag,
                                     const unsigned int* b_frag,
                                     const unsigned int* sf,
                                     float* d_out) {
    const unsigned int a0 = a_frag[threadIdx.x * 4 + 0];
    const unsigned int a1 = a_frag[threadIdx.x * 4 + 1];
    const unsigned int a2 = a_frag[threadIdx.x * 4 + 2];
    const unsigned int a3 = a_frag[threadIdx.x * 4 + 3];
    const unsigned int b0 = b_frag[threadIdx.x * 2 + 0];
    const unsigned int b1 = b_frag[threadIdx.x * 2 + 1];
    const unsigned int sa = sf[0];
    const unsigned int sb = sf[1];

    float d0 = 0.0f, d1 = 0.0f, d2 = 0.0f, d3 = 0.0f;
    asm volatile(
        "mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 "
        "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, "
        "%10, {0, 0}, %11, {0, 0};"
        : "+f"(d0), "+f"(d1), "+f"(d2), "+f"(d3)
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1), "r"(sa), "r"(sb));

    const int g = threadIdx.x >> 2;
    const int t = threadIdx.x & 3;
    d_out[g * 8 + 2 * t + 0] = d0;
    d_out[g * 8 + 2 * t + 1] = d1;
    d_out[(g + 8) * 8 + 2 * t + 0] = d2;
    d_out[(g + 8) * 8 + 2 * t + 1] = d3;
}
"#;

// e2m1: 1 sign, 2 exp, 1 mantissa, bias 1. Index = (sign << 3) | bits.
const E2M1: [f32; 8] = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0];

fn decode_e2m1(nibble: u32) -> f32 {
    let v = E2M1[(nibble & 0x7) as usize];
    if nibble & 0x8 != 0 { -v } else { v }
}

// ue4m3 (unsigned e4m3, bias 7). Values used here: 0.5 = 0x30, 1.0 = 0x38, 2.0 = 0x40.
fn decode_ue4m3(byte: u32) -> f32 {
    let e = (byte >> 3) & 0xF;
    let m = byte & 0x7;
    if e == 0 {
        (m as f32) * (2.0_f32).powi(-6) / 8.0
    } else {
        (1.0 + (m as f32) / 8.0) * (2.0_f32).powi(e as i32 - 7)
    }
}

const M: usize = 16;
const K: usize = 64;
const N: usize = 8;

// Pack one register: nibble i at bit position 4*i (LSB-first) or 4*(7-i) (MSB-first).
fn pack_reg(nibbles: &[u32], msb: bool) -> u32 {
    nibbles
        .iter()
        .enumerate()
        .map(|(i, &n)| n << (4 * if msb { 7 - i } else { i }))
        .sum()
}

fn main() -> Result<(), DriverError> {
    println!("p2: NVRTC compile of the llama.cpp mma.sync…mxf4nvf4 instruction (arch compute_120a) ...");
    let opts = CompileOptions {
        arch: Some("compute_120a"),
        ..Default::default()
    };
    let ptx = match compile_ptx_with_opts(MMA_SRC, opts) {
        Ok(p) => p,
        Err(e) => {
            println!("p2: NVRTC COMPILE FAILED: {e:?}");
            println!("p2: FAIL (instruction rejected by the 13.3 runtime compiler)");
            std::process::exit(2);
        }
    };
    println!("p2: NVRTC compile OK ({} bytes PTX)", ptx.to_src().len());
    std::fs::write("p2_emitted.ptx", ptx.to_src()).expect("write p2_emitted.ptx");

    let ctx = CudaContext::new(0)?;
    let stream = ctx.default_stream();
    let module = ctx.load_module(ptx)?;
    let func = module.load_function("mma_nvfp4")?;
    println!("p2: module loaded, function resolved");

    // Nonlinear, aperiodic data: the (m+3)*k and (k+3)*(n+3) mod terms kill any
    // period-8/16/32 structure along k that would make scale patterns indistinguishable.
    let mut a_nib = [[0u32; K]; M];
    let mut b_nib = [[0u32; N]; K];
    for m in 0..M {
        for k in 0..K {
            let idx = (m * 11 + k * 7 + ((m + 3) * k) % 13) % 8;
            let sign = ((m * 5 + k * 3 + ((m + 3) * k) % 7) & 1) as u32;
            a_nib[m][k] = (idx as u32) | (sign << 3);
        }
    }
    for k in 0..K {
        for n in 0..N {
            let idx = (k * 5 + n * 3 + ((k + 3) * (n + 3)) % 11) % 8;
            let sign = ((k * 3 + n * 5 + ((k + 3) * (n + 3)) % 7) & 1) as u32;
            b_nib[k][n] = (idx as u32) | (sign << 3);
        }
    }

    const ONE: u32 = 0x38; // ue4m3 1.0
    const TWO: u32 = 0x40; // ue4m3 2.0
    const HALF: u32 = 0x30; // ue4m3 0.5

    struct Cfg {
        name: &'static str,
        msb_a: bool,
        msb_b: bool,
        sfa: [u32; 4],
        sfb: [u32; 4],
        flip_a01_sign: bool,
    }
    // e0 is the full primary hypothesis: LSB nibbles both operands, scale byte i -> k-block i.
    let cfgs = [
        Cfg { name: "e0 primary: LSB nibbles, sf A[1,2,.5,1] B[1,1,1,2]", msb_a: false, msb_b: false, sfa: [ONE, TWO, HALF, ONE], sfb: [ONE, ONE, ONE, TWO], flip_a01_sign: false },
        Cfg { name: "e1 all scales 1.0",                                  msb_a: false, msb_b: false, sfa: [ONE; 4],             sfb: [ONE; 4],             flip_a01_sign: false },
        Cfg { name: "e2 A nibbles MSB (B LSB), e0 scales",                msb_a: true,  msb_b: false, sfa: [ONE, TWO, HALF, ONE], sfb: [ONE, ONE, ONE, TWO], flip_a01_sign: false },
        Cfg { name: "e3 B nibbles MSB (A LSB), e0 scales",                msb_a: false, msb_b: true,  sfa: [ONE, TWO, HALF, ONE], sfb: [ONE, ONE, ONE, TWO], flip_a01_sign: false },
        Cfg { name: "e4 sf bytes reversed: A[1,.5,2,1] B[2,1,1,1]",       msb_a: false, msb_b: false, sfa: [ONE, HALF, TWO, ONE], sfb: [TWO, ONE, ONE, ONE], flip_a01_sign: false },
        Cfg { name: "e5 transport: A[0][1] sign flipped (1.0 -> -1.0)",   msb_a: false, msb_b: false, sfa: [ONE, TWO, HALF, ONE], sfb: [ONE, ONE, ONE, TWO], flip_a01_sign: true },
    ];

    let mut a_frag = [0u32; 32 * 4];
    let mut b_frag = [0u32; 32 * 2];

    // Effective nibbles the hardware sees under the LSB-first read hypothesis, and the
    // matching reference for a config.
    let effective = |msb: bool| -> ([[f32; K]; M], [[f32; N]; K]) {
        let mut ea = [[0.0f32; K]; M];
        let mut eb = [[0.0f32; N]; K];
        for lane in 0..32usize {
            let g = lane >> 2;
            let t = lane & 3;
            let mut regs_a = [0u32; 4];
            let mut regs_b = [0u32; 2];
            for (i, reg) in regs_a.iter_mut().enumerate() {
                let row = if i & 1 == 0 { g } else { g + 8 };
                let base = if i < 2 { 8 * t } else { 32 + 8 * t };
                let nibs: Vec<u32> = (0..8).map(|j| a_nib[row][base + j]).collect();
                *reg = pack_reg(&nibs, msb);
            }
            for (i, reg) in regs_b.iter_mut().enumerate() {
                let base = if i == 0 { 8 * t } else { 32 + 8 * t };
                let nibs: Vec<u32> = (0..8).map(|j| b_nib[base + j][g]).collect();
                *reg = pack_reg(&nibs, msb);
            }
            // hardware reads nibble j of register r at bit 4*j
            for (r, reg) in regs_a.iter().enumerate() {
                let row = if r & 1 == 0 { g } else { g + 8 };
                let base = if r < 2 { 8 * t } else { 32 + 8 * t };
                for j in 0..8 {
                    ea[row][base + j] = decode_e2m1((reg >> (4 * j)) & 0xF);
                }
            }
            for (r, reg) in regs_b.iter().enumerate() {
                let base = if r == 0 { 8 * t } else { 32 + 8 * t };
                for j in 0..8 {
                    eb[base + j][g] = decode_e2m1((reg >> (4 * j)) & 0xF);
                }
            }
        }
        (ea, eb)
    };

    let mut results: Vec<(&'static str, bool)> = Vec::new();
    for cfg in &cfgs {
        let name = cfg.name;
        // Effective matrices the hardware sees if it reads nibbles LSB-first: ea from
        // A packed with cfg.msb_a, eb from B packed with cfg.msb_b — each operand's
        // reference side must mirror its own device packing.
        let (ea, _) = effective(cfg.msb_a);
        let (_, eb) = effective(cfg.msb_b);

        // reference under: hardware reads LSB-first; scale byte i applies to k-block i
        let mut reference = [[0.0f32; N]; M];
        for m in 0..M {
            for n in 0..N {
                let mut acc = 0.0f32;
                for kb in 0..4 {
                    let sf = decode_ue4m3(cfg.sfa[kb]) * decode_ue4m3(cfg.sfb[kb]);
                    let mut sub = 0.0f32;
                    for k in (kb * 16)..(kb * 16 + 16) {
                        sub += ea[m][k] * eb[k][n];
                    }
                    acc += sf * sub;
                }
                reference[m][n] = acc;
            }
        }

        // pack what goes to the device
        for lane in 0..32usize {
            let g = lane >> 2;
            let t = lane & 3;
            for (i, reg) in a_frag[lane * 4..lane * 4 + 4].iter_mut().enumerate() {
                let row = if i & 1 == 0 { g } else { g + 8 };
                let base = if i < 2 { 8 * t } else { 32 + 8 * t };
                let mut nibs: Vec<u32> = (0..8).map(|j| a_nib[row][base + j]).collect();
                if cfg.flip_a01_sign && row == 0 && base == 0 {
                    nibs[1] ^= 0x8; // A[0][1] decodes to 1.0, sign flip makes it -1.0
                }
                *reg = pack_reg(&nibs, cfg.msb_a);
            }
            for (i, reg) in b_frag[lane * 2..lane * 2 + 2].iter_mut().enumerate() {
                let base = if i == 0 { 8 * t } else { 32 + 8 * t };
                let nibs: Vec<u32> = (0..8).map(|j| b_nib[base + j][g]).collect();
                *reg = pack_reg(&nibs, cfg.msb_b);
            }
        }
        let pack_le = |s: &[u32; 4]| s[0] | (s[1] << 8) | (s[2] << 16) | (s[3] << 24);
        let sf = [pack_le(&cfg.sfa), pack_le(&cfg.sfb)];

        let a_dev = stream.clone_htod(&a_frag)?;
        let b_dev = stream.clone_htod(&b_frag)?;
        let sf_dev = stream.clone_htod(&sf)?;
        let mut d_dev = stream.alloc_zeros::<f32>(M * N)?;
        unsafe {
            stream
                .launch_builder(&func)
                .arg(&a_dev)
                .arg(&b_dev)
                .arg(&sf_dev)
                .arg(&mut d_dev)
                .launch(LaunchConfig {
                    grid_dim: (1, 1, 1),
                    block_dim: (32, 1, 1),
                    shared_mem_bytes: 0,
                })
        }?;
        let d_host: Vec<f32> = stream.clone_dtoh(&d_dev)?;

        let mut max_delta = 0.0f32;
        let mut first = Option::<(usize, usize, f32, f32)>::None;
        for m in 0..M {
            for n in 0..N {
                let gpu = d_host[m * N + n];
                let delta = (gpu - reference[m][n]).abs();
                if delta > max_delta {
                    max_delta = delta;
                }
                if first.is_none() && delta > 1e-2 {
                    first = Some((m, n, gpu, reference[m][n]));
                }
            }
        }
        let matches = max_delta <= 1e-2;
        results.push((cfg.name, matches));
        let verdict = if matches { "MATCH" } else { "no match" };
        println!(
            "p2: {name}: a_frag[0]={:#010x} sf=[{:#010x},{:#010x}] gpu[:4]={:?}  max_delta={max_delta:.4}  {verdict}",
            a_frag[0],
            sf[0],
            sf[1],
            &d_host[..4]
        );
        if let Some((m, n, gpu, want)) = first {
            println!("p2: {name}:   first off D[{m}][{n}] gpu={gpu:.4} ref={want:.4}");
        }
    }

    println!("p2: ---- summary ----");
    let r = |i: usize| results[i].1;
    println!(
        "p2: e0 (full hypothesis) match: {} | e1 scale-free: {} | e2 A-MSB: {} | e3 B-MSB: {} | e4 reversed sf bytes: {} | e5 flipped input breaks reference: {}",
        r(0), r(1), r(2), r(3), r(4), !r(5)
    );
    let computes = r(0) || (r(1) && (r(2) || r(3)));
    let transport_ok = !r(5);
    if computes && transport_ok {
        if r(0) {
            println!("p2: PASS — mxf4nvf4 mma.sync computes on sm_120; LSB-first nibbles (both operands), scale byte i -> k-sub-block i confirmed");
        } else {
            println!("p2: PASS — mxf4nvf4 mma.sync computes on sm_120 (see per-config lines for layout details)");
        }
        Ok(())
    } else {
        println!("p2: FAIL");
        std::process::exit(1);
    }
}
