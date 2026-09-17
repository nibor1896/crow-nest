//! ue4m3 scale-byte edge decode on the mma hardware (NaN hunt, #10 MMA ticket).
use crow_nest_engine::cuda;

const SRC: &str = r#"
extern "C" __global__ void mma_sf_edge(const unsigned int* a_frag, const unsigned int* b_frag,
                                       const unsigned int* sfa_lanes, const unsigned int* sfb_lanes,
                                       float* d_out) {
    const unsigned int a0 = a_frag[threadIdx.x * 4 + 0];
    const unsigned int a1 = a_frag[threadIdx.x * 4 + 1];
    const unsigned int a2 = a_frag[threadIdx.x * 4 + 2];
    const unsigned int a3 = a_frag[threadIdx.x * 4 + 3];
    const unsigned int b0 = b_frag[threadIdx.x * 2 + 0];
    const unsigned int b1 = b_frag[threadIdx.x * 2 + 1];
    const unsigned int sa = sfa_lanes[threadIdx.x];
    const unsigned int sb = sfb_lanes[threadIdx.x];
    float d0 = 0.0f, d1 = 0.0f, d2 = 0.0f, d3 = 0.0f;
    asm volatile(
        "mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 "
        "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, "
        "%10, {0, 0}, %11, {0, 0};"
        : "+f"(d0), "+f"(d1), "+f"(d2), "+f"(d3)
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1), "r"(sa), "r"(sb));
    const int g = threadIdx.x >> 2;
    const int t = threadIdx.x & 3;
    float* out = d_out + blockIdx.x * 128;
    out[g * 8 + 2 * t + 0] = d0;
    out[g * 8 + 2 * t + 1] = d1;
    out[(g + 8) * 8 + 2 * t + 0] = d2;
    out[(g + 8) * 8 + 2 * t + 1] = d3;
}
"#;

fn pack_nib(nibs: &[u32]) -> u32 {
    nibs.iter().enumerate().map(|(i, &n)| n << (4 * i)).sum()
}


fn main() {
    unsafe {
        let _ctx = cuda::Ctx::init();
        let module = cuda::compile(SRC);
        let f = module.get("mma_sf_edge");

        let mut a_frag = [0u32; 32 * 4];
        let mut b_frag = [0u32; 32 * 2];
        for lane in 0..32usize {
            for reg in a_frag[lane * 4..lane * 4 + 4].iter_mut() {
                *reg = pack_nib(&[2, 2, 2, 2, 2, 2, 2, 2]); // all 1.0
            }
            for reg in b_frag[lane * 2..lane * 2 + 2].iter_mut() {
                *reg = pack_nib(&[2, 2, 2, 2, 2, 2, 2, 2]);
            }
        }
        let a_dev = cuda::to_dev(&a_frag);
        let b_dev = cuda::to_dev(&b_frag);

        // all-0x38 (1.0) fragments; sf bytes: sfa byte0 = candidate, rest 1.0;
        // sfb all 1.0 -> D[0][0] = 16 * (1.0) * dec(candidate byte0) * ... =
        // 64 elements per ksub -> D = 64 * dec(byte0) for the sfa[0] lane row 0
        for cand in [0x38u32, 0x40, 0x78, 0x7F, 0xFF, 0x00, 0x70, 0x7E] {
            let mut sfa = [0x38383838u32; 32]; // all bytes 1.0
            sfa[0] = 0x38383800 | cand; // byte0 = candidate, bytes1-3 = 1.0
            let sfb = [0x38383838u32; 32];
            let sfa_dev = cuda::to_dev(&sfa);
            let sfb_dev = cuda::to_dev(&sfb);
            let out = cuda::alloc_zeroed(128 * 4);
            let vals: [u64; 5] = [a_dev as u64, b_dev as u64, sfa_dev as u64, sfb_dev as u64, out as u64];
            let mut ptrs: Vec<*mut std::ffi::c_void> = vals
                .iter()
                .map(|v| v as *const u64 as *mut std::ffi::c_void)
                .collect();
            cuda::launch(f, 1, 1, 32, 0, &mut ptrs);
            let d = cuda::dtoh(out, 128);
            println!("sfa byte0 = {cand:#04x}: D[0][0] = {:e}  D[1][0] = {:e}  (finite: {})",
                d[0], d[8], d[0].is_finite() && d[8].is_finite());
            let mut o = out;
            let mut f1 = sfa_dev;
            let mut f2 = sfb_dev;
            cuda::free_dev(&mut o);
            cuda::free_dev(&mut f1);
            cuda::free_dev(&mut f2);
        }
        let mut a2 = a_dev;
        let mut b2 = b_dev;
        cuda::free_dev(&mut a2);
        cuda::free_dev(&mut b2);
    }
}
