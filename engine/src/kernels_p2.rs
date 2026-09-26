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
pub const P2_NAMES: &[&str] = &["attn_full_split", "silu_mul_n", "gemv_nvfp4_w", "gemv_nvfp4_gu", "add_rms_1k", "gemv_bf16_ba"];

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

// ---------------- the decode GEMV of the dense family (one token) ----------------
// y[row] = gs * sum_b sum_s ue4m3(scale[b][s]) * sum_j e2m1(nib[b][s][j]) * x[64b + 16s + j]
// over NVFP4 rows (36-byte blocks: 4 ue4m3 sub-block scales, then 32 bytes of e2m1 pairs, low
// nibble = even value), on the f32 activation row itself: no activation quantization.
// One warp per row, 8 rows per 256-thread block. Both streams are read the way Flash-Next's
// gemv_bf16_w reads its BF16 lm_head (1.67 TB/s, profile 2026-09-26): the row, 32 blocks =
// 1152 B at a time, with 16-byte loads into shared memory; then the warp walks the chunk four
// blocks per step, lane l taking the 32-bit word l % 8 (8 values, sub-block (l % 8) / 2) of
// block l / 8 and the two float4 of x under it, so a warp step reads 1 KB of x contiguously.
// Both tables live in shared memory: e2m1() indexes a local array (a local-memory load per
// nibble, 232 ms/token in the first build) and ue4m3() is an exp2f per call.
// History (decode 128 tokens): one lane per block with 4-byte loads ~800 GB/s; one byte per
// lane (2 values) 20.8 ms/token with the GDN on the old path.
// NR rows per warp share every x load: gemv_nvfp4_w (NR 1) and gemv_nvfp4_gu (NR 2: the gate
// and up rows of one dense-FFN index, silu(gate) * up in the epilogue).
// grid (ceil(rows / 8)), block 256; k_dim % 256 == 0 (a 32-block chunk is then a whole number
// of 16-byte words and of 4-block steps).
template <int NR>
__device__ __forceinline__ void nvfp4_warp_dot(const unsigned char* const (&rowb)[NR], const float* __restrict__ x,
                                               int bpr, uint4 (*stage)[72], const float* lut, const float* lut8,
                                               float (&acc)[NR]) {
    const int lane = threadIdx.x & 31;
    const int wi = lane & 7, bi = lane >> 3;
    #pragma unroll
    for (int r = 0; r < NR; r++) acc[r] = 0.0f;
    for (int b0 = 0; b0 < bpr; b0 += 32) {
        const int nb = min(32, bpr - b0);
        const int nq = nb * 36 / 16;
        #pragma unroll
        for (int r = 0; r < NR; r++) {
            const uint4* src = (const uint4*)(rowb[r] + (size_t)b0 * 36);
            for (int q = lane; q < nq; q += 32) stage[r][q] = __ldg(src + q);
        }
        __syncwarp();
        for (int j = bi; j < nb; j += 4) {
            const float4* xp = (const float4*)(x + (size_t)(b0 + j) * 64 + wi * 8);
            const float4 xa = __ldg(xp), xb = __ldg(xp + 1);
            #pragma unroll
            for (int r = 0; r < NR; r++) {
                const unsigned char* blk = (const unsigned char*)stage[r] + j * 36;
                const unsigned int wd = *(const unsigned int*)(blk + 4 + 4 * wi);
                const float sc = lut8[blk[wi >> 1]];
                float part = lut[wd & 0xF] * xa.x;
                part += lut[(wd >> 4) & 0xF] * xa.y;
                part += lut[(wd >> 8) & 0xF] * xa.z;
                part += lut[(wd >> 12) & 0xF] * xa.w;
                part += lut[(wd >> 16) & 0xF] * xb.x;
                part += lut[(wd >> 20) & 0xF] * xb.y;
                part += lut[(wd >> 24) & 0xF] * xb.z;
                part += lut[wd >> 28] * xb.w;
                acc[r] += part * sc;
            }
        }
        __syncwarp(); // the next chunk overwrites the stage
    }
    #pragma unroll
    for (int r = 0; r < NR; r++) {
        #pragma unroll
        for (int o = 16; o > 0; o >>= 1) acc[r] += __shfl_down_sync(0xffffffffu, acc[r], o);
    }
}
// the two tables, filled by the first 256 threads (the block is 256)
#define NVFP4_TABLES                                              \
    __shared__ float lut[16];                                     \
    __shared__ float lut8[256];                                   \
    if (threadIdx.x < 16) lut[threadIdx.x] = e2m1(threadIdx.x);   \
    lut8[threadIdx.x] = ue4m3(threadIdx.x);                       \
    __syncthreads();
extern "C" __global__ void gemv_nvfp4_w(const unsigned char* __restrict__ w, const float* __restrict__ x,
                                        const float* __restrict__ gs_ptr, float* __restrict__ y,
                                        const int* __restrict__ k_dim_p, const int* __restrict__ rows_p) {
    NVFP4_TABLES
    __shared__ uint4 stage[8][1][72];
    const int warp = threadIdx.x >> 5;
    const int row = blockIdx.x * 8 + warp;
    if (row >= *rows_p) return; // no block-wide barrier below this line
    const int bpr = *k_dim_p >> 6;
    const unsigned char* const rowb[1] = {w + (size_t)row * bpr * 36};
    float acc[1];
    nvfp4_warp_dot<1>(rowb, x, bpr, stage[warp], lut, lut8, acc);
    if ((threadIdx.x & 31) == 0) y[row] = acc[0] * gs_ptr[0];
}
// the dense FFN's gate and up rows of index `row` in one warp: y[row] = silu(g) * u with
// g = gs_g * <gate row, x>, u = gs_u * <up row, x> (HF Qwen3_5MLP), replacing two GEMVs and
// silu_mul_n at one token
extern "C" __global__ void gemv_nvfp4_gu(const unsigned char* __restrict__ wg, const unsigned char* __restrict__ wu,
                                         const float* __restrict__ x, const float* __restrict__ gs_g,
                                         const float* __restrict__ gs_u, float* __restrict__ y,
                                         const int* __restrict__ k_dim_p, const int* __restrict__ rows_p) {
    NVFP4_TABLES
    __shared__ uint4 stage[8][2][72];
    const int warp = threadIdx.x >> 5;
    const int row = blockIdx.x * 8 + warp;
    if (row >= *rows_p) return;
    const int bpr = *k_dim_p >> 6;
    const unsigned char* const rowb[2] = {wg + (size_t)row * bpr * 36, wu + (size_t)row * bpr * 36};
    float acc[2];
    nvfp4_warp_dot<2>(rowb, x, bpr, stage[warp], lut, lut8, acc);
    if ((threadIdx.x & 31) == 0) {
        const float g = acc[0] * gs_g[0];
        y[row] = (g / (1.0f + expf(-g))) * (acc[1] * gs_u[0]);
    }
}

// ---------------- the GDN's two small BF16 projections in one launch ----------------
// in_proj_b and in_proj_a (48 rows each, BF16 keeps of the dense recipe) over one f32 row:
// rows 0..n-1 of `wb` into yb, rows n..2n-1 of `wa` into ya, one warp per row with the 16-byte
// loads of gemv_bf16_w (same per-row math, same order). Two launches of 6 blocks per GDN
// layer were 96 launches per decode token. grid (ceil(2n / 8)), block 256; k_dim % 256 == 0.
extern "C" __global__ void gemv_bf16_ba(const unsigned short* __restrict__ wb, const unsigned short* __restrict__ wa,
                                        const float* __restrict__ x, float* __restrict__ yb, float* __restrict__ ya,
                                        const int* __restrict__ k_dim_p, const int* __restrict__ n_p) {
    const int k_dim = *k_dim_p, n = *n_p;
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int r = blockIdx.x * 8 + warp;
    if (r >= 2 * n) return;
    const unsigned short* wp = (r < n) ? wb + (size_t)r * k_dim : wa + (size_t)(r - n) * k_dim;
    float acc = 0.0f;
    for (int i = lane * 8; i < k_dim; i += 256) {
        uint4 v = *(const uint4*)(wp + i);
        unsigned int u[4] = {v.x, v.y, v.z, v.w};
        #pragma unroll
        for (int j = 0; j < 4; j++) {
            float lo = __uint_as_float(u[j] << 16);
            float hi = __uint_as_float(u[j] & 0xFFFF0000u);
            acc += lo * x[i + 2 * j] + hi * x[i + 2 * j + 1];
        }
    }
    for (int o = 16; o > 0; o >>= 1) acc += __shfl_down_sync(0xffffffffu, acc, o);
    if (lane == 0) {
        if (r < n) yb[r] = acc;
        else ya[r - n] = acc;
    }
}

// ---------------- Residual::Plain: the residual add and the next pre-norm ----------------
// h[row] += y[row] (skipped when y is 0), then out[row] = h * rsqrt(mean(h^2) + eps) * (1 + w),
// HF Qwen3_5RMSNorm (zero-centred). The plain residual adds every sub-block output and
// normalizes right after; in one launch that is 128 fewer per decode token (add_flat + rms_group
// were 257 launches), and 1024 threads per row instead of rms_group's 256.
// grid (T), block 1024
extern "C" __global__ void add_rms_1k(float* __restrict__ h, const float* __restrict__ y,
                                      const float* __restrict__ w, float* __restrict__ out) {
    __shared__ float red[32];
    float* hp = h + (size_t)blockIdx.x * CN_H;
    const float* yp = y ? y + (size_t)blockIdx.x * CN_H : nullptr;
    float ss = 0.0f;
    for (int i = threadIdx.x; i < CN_H; i += 1024) {
        float v = hp[i];
        if (yp) {
            v += yp[i];
            hp[i] = v;
        }
        ss += v * v;
    }
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) ss += __shfl_xor_sync(0xffffffffu, ss, o);
    if ((threadIdx.x & 31) == 0) red[threadIdx.x >> 5] = ss;
    __syncthreads();
    ss = 0.0f;
    #pragma unroll
    for (int i = 0; i < 32; i++) ss += red[i];
    const float rms = rsqrtf(ss / (float)CN_H + CN_EPS);
    float* op = out + (size_t)blockIdx.x * CN_H;
    for (int i = threadIdx.x; i < CN_H; i += 1024) op[i] = hp[i] * rms * (1.0f + w[i]);
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
