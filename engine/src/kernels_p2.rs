//! Crow #300 phase 2: the kernels only a family without hyper-connections, QSA or
//! MoE needs (the dense Qwen3.5/3.8 path).
//!
//! They live in their OWN source, not in `KERNEL_SRC`: `kernels::tests_300_c4`
//! proves the whole Flash-Next PTX module byte-identical to the PTX of record
//! (`tests/fixtures/ptx-manifest-3154b3b.txt`), entry list included, so one more
//! `.entry` there would break the proof of record. A family that needs these
//! kernels compiles `prelude + KERNEL_SRC + P2_SRC` as its one module
//! (`KernelGeo::source`), so they use `KERNEL_SRC`'s helpers (`kv_ld`,
//! `dec_e4m3`) as they are; the Flash-Next source stays `prelude + KERNEL_SRC`.

/// the entries of `P2_SRC`, resolved by `Kernels::add_p2` when the family compiles them
pub const P2_NAMES: &[&str] = &["attn_full_split", "silu_mul_n"];

/// the phase 2 kernel source, appended after `KERNEL_SRC` (`KernelGeo::source`)
pub const P2_SRC: &str = r#"
// ---------------- Attn::Full: uncapped causal attention ----------------
// Query row t of a chunk sits at position pos = *pos_base_p + t and attends the
// KV cache rows 0..=pos (the rows store_kv wrote; layout [kvh][tmax][AHD], e4m3
// or bf16 by *mode_p). The rows are split over gridDim.z blocks per head; each
// block runs an online softmax over tiles of AF_TILE rows, so the row count has
// no cap (the attn_sel family keeps p[CN_QSA_SEL_MAX] in shared memory), and
// writes one unnormalized partial (m, l, o[AHD]) in the flash-decoding form
// KERNEL_SRC's attn_merge combines. An empty split writes (-3e38, 0, 0), which
// attn_merge weighs with exp(-3e38 - m) = 0. With S = 1 (the prefill form: one
// block per query row and head already fills the card) the block writes the
// normalized o / l straight into part_o as [T][NQ][AHD] and part_ml is unused.
// grid (NQ, T, S), block CN_AHD (= 256: eight warps, one thread per o element).
#define AF_TILE 64
extern "C" __global__ void attn_full_split(const float* __restrict__ q, const unsigned char* __restrict__ kc,
                                           const unsigned char* __restrict__ vc, const int* __restrict__ pos_base_p,
                                           const int* __restrict__ tmax_p, const int* __restrict__ mode_p,
                                           float* __restrict__ part_o, float* __restrict__ part_ml) {
    int head = blockIdx.x;
    int t = blockIdx.y;
    int split = blockIdx.z, S = gridDim.z;
    int d = threadIdx.x;
    int kvh = head / CN_GQA;
    const int tmax = *tmax_p;
    int n = *pos_base_p + t + 1;
    if (n > tmax) n = tmax;
    int per = (n + S - 1) / S;
    int lo = split * per, hi = min(n, lo + per);
    __shared__ float p[AF_TILE];
    __shared__ float lut[256];
    __shared__ float qs[CN_AHD];
    const int mode = *mode_p;
    const size_t esz = mode ? 2 : 1;
    lut[d] = dec_e4m3((unsigned char)d);
    qs[d] = q[((size_t)t * CN_NQ + head) * CN_AHD + d];
    __syncthreads();
    int warp = d >> 5, lane = d & 31;
    const float scale = CN_ATTN_SCALE;
    const size_t kvbase = (size_t)kvh * tmax;
    float m = -3.0e38f, l = 0.0f, o = 0.0f;
    for (int j0 = lo; j0 < hi; j0 += AF_TILE) {
        int cnt = min(AF_TILE, hi - j0);
        for (int jj = warp; jj < cnt; jj += CN_AHD / 32) {
            const unsigned char* kp = kc + (kvbase + j0 + jj) * CN_AHD * esz;
            float acc = 0.0f;
            for (int e = lane; e < CN_AHD; e += 32) acc += qs[e] * kv_ld<1>(kp, e, mode, lut);
            for (int of = 16; of > 0; of >>= 1) acc += __shfl_down_sync(0xffffffffu, acc, of);
            if (lane == 0) p[jj] = acc * scale;
        }
        __syncthreads();
        float tm = m;
        for (int jj = 0; jj < cnt; jj++) tm = fmaxf(tm, p[jj]);
        float corr = expf(m - tm);
        l *= corr;
        o *= corr;
        for (int jj = 0; jj < cnt; jj++) {
            float e = expf(p[jj] - tm);
            l += e;
            o += e * kv_ld<1>(vc + (kvbase + j0 + jj) * CN_AHD * esz, d, mode, lut);
        }
        m = tm;
        __syncthreads(); // p is rewritten by the next tile
    }
    size_t pi = ((size_t)t * CN_NQ + head) * S + split;
    if (S == 1) {
        part_o[pi * CN_AHD + d] = o / l; // n >= 1: the row itself is always attended
        return;
    }
    part_o[pi * CN_AHD + d] = o;
    if (d == 0) {
        part_ml[pi * 2] = (hi > lo) ? m : -3.0e38f;
        part_ml[pi * 2 + 1] = (hi > lo) ? l : 0.0f;
    }
}

// ---------------- Ffn::Dense: the SwiGLU product ----------------
// g[i] = silu(g[i]) * u[i] over n = *n_p values (t rows x the dense width, one flat
// index: gate and up are separate [T][I] buffers). HF Qwen3_5MLP:
// down(act_fn(gate(x)) * up(x)), act_fn = silu.
extern "C" __global__ void silu_mul_n(float* __restrict__ g, const float* __restrict__ u,
                                      const int* __restrict__ n_p) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= *n_p) return;
    float a = g[i];
    g[i] = (a / (1.0f + expf(-a))) * u[i];
}
"#;

#[cfg(test)]
mod tests_300_p2 {
    use crate::kernels::tests_300_c4::{entries, ptx};
    use super::P2_NAMES;
    use crate::kernels::{KernelGeo, KERNEL_SRC};

    /// The 27B's kernel text is `prelude + KERNEL_SRC + P2_SRC` and compiles
    /// (NVRTC is a host compiler: no GPU); its entries are Flash-Next's 130 plus
    /// the phase 2 ones. Flash-Next's text stays `prelude + KERNEL_SRC`, so the
    /// PTX of record (`tests_300_c4`) is not touched by this module.
    #[test]
    fn the_dense_source_compiles_with_the_phase_2_entries_and_flash_next_does_not_carry_them() {
        let fx = KernelGeo::flash_next();
        assert!(!fx.p2);
        assert_eq!(fx.source(), format!("{}{}", fx.prelude(), KERNEL_SRC));
        let kg = KernelGeo::of(&crate::meta::dense_fixture_geo());
        assert!(kg.p2);
        let src = kg.source();
        assert!(src.ends_with(super::P2_SRC));
        let names: Vec<String> = entries(&ptx(&src)).into_iter().map(|(n, _)| n).collect();
        assert_eq!(names.len(), 130 + P2_NAMES.len(), "{names:?}");
        for n in P2_NAMES {
            assert!(names.iter().any(|m| m == n), "{n}");
        }
        assert!(kg.prelude().contains("#define CN_QSA_SEL_MAX 1\n"), "compile-only size for a family without QSA");
    }
}
