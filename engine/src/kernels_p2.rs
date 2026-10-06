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
pub const P2_NAMES: &[&str] = &["attn_full_split", "silu_mul_n", "gemv_nvfp4_w", "gemv_nvfp4_gu", "add_rms_1k", "gemv_bf16_ba", "attn_full_fa", "gemv_nvfp4_wm", "gemv_nvfp4_gum", "gemv_bf16_wm"];

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

// ---------------- Attn::Full prefill: tiled attention on tensor cores ----------------
// FlashAttention-2 form (Dao 2023, arXiv 2307.08691 alg. 1) for the prompt chunk: one block per
// (kv head, 16 query rows), one warp per query head of the GQA group (CN_GQA warps), all sharing
// each K / V tile of 16 keys, which the block stages ONCE from the KV cache (e4m3 or bf16) into
// shared memory as f16 (K row-major, V transposed, both padded against bank conflicts).
// Per warp and tile: S = Q K^T with mma.m16n8k16 f16 (f32 accumulate; Q held as f16 A fragments),
// online softmax in f32 (exp2 with the scale folded), O += P V with P repacked from the S
// accumulators as the next A operand. Rows g and g + 8 of the m16 tile belong to lane g * 4 + t.
// The causal mask applies on the tile that holds the diagonal. attn_full_split with S = 1 read
// every K / V row once per query row and head (GQA 6): 240 tok/s at position 20k (2026-09-26).
// Split K (gridDim.z = S > 1, the decode form): split z takes an equal share of the key range
// and writes the unnormalized partial (o, m, l) in attn_merge's layout [T][NQ][S] (m in natural
// log units) to part_o / part_ml; S = 1 writes the normalized rows to part_o as [T][NQ][AHD].
// Decode reads each K / V row once per KV head instead of once per query head (GQA 6).
// grid (NKV, ceil(T / 16), S), block 32 * CN_GQA. CN_AHD % 16 == 0.
#define FA_KT 16
#define FA_KS (CN_AHD + 8)   // K_s row stride in halves (bank padding)
#define FA_VS (FA_KT + 2)    // V_t row stride in halves (bank padding)
// f16 without cuda_fp16.h (NVRTC compiles with no include path): the PTX conversions
#define FA_NEG_INF __int_as_float(0xff800000)
__device__ __forceinline__ unsigned int fa_pack_h2(float lo, float hi) {
    unsigned int r;
    asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(r) : "f"(hi), "f"(lo)); // first source -> upper half
    return r;
}
// two e4m3 bytes (low byte = first value) -> f16x2 (low half = first value), one instruction
__device__ __forceinline__ unsigned int fa_e4m3x2_h2(unsigned short v) {
    unsigned int r;
    asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(r) : "h"(v));
    return r;
}
__device__ __forceinline__ unsigned short fa_h(float x) {
    unsigned short r;
    asm("cvt.rn.f16.f32 %0, %1;" : "=h"(r) : "f"(x));
    return r;
}
__device__ __forceinline__ void mma_f16_16n8k16(float& d0, float& d1, float& d2, float& d3,
                                                unsigned int a0, unsigned int a1, unsigned int a2,
                                                unsigned int a3, unsigned int b0, unsigned int b1) {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
        "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};"
        : "+f"(d0), "+f"(d1), "+f"(d2), "+f"(d3)
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}
extern "C" __global__ void __launch_bounds__(32 * CN_GQA, 1)
attn_full_fa(const float* __restrict__ q, const unsigned char* __restrict__ kc, const unsigned char* __restrict__ vc,
             const int* __restrict__ pos_base_p, const int* __restrict__ t_p, const int* __restrict__ tmax_p,
             const int* __restrict__ mode_p, float* __restrict__ out, float* __restrict__ part_ml) {
    __shared__ unsigned short k_s[FA_KT * FA_KS]; // f16 bits
    __shared__ unsigned short v_t[CN_AHD * FA_VS];
    const int kvh = blockIdx.x;
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int g = lane >> 2, tq = lane & 3;
    const int head = kvh * CN_GQA + warp;
    const int T = *t_p, pos_base = *pos_base_p, tmax = *tmax_p, mode = *mode_p;
    const int esz = mode ? 2 : 1;
    const int r0 = blockIdx.y * 16;
    const int ra = r0 + g, rb = r0 + g + 8;            // this lane's two query rows
    const int qa = pos_base + ra, qb = pos_base + rb;   // their positions
    const float sl2 = CN_ATTN_SCALE * 1.4426950408889634f; // scale * log2(e)
    // Q A fragments for the 16 k-steps of the head dim (rows beyond T read row T - 1, never stored)
    unsigned int qf[CN_AHD / 16][4];
    {
        const float* pa = q + ((size_t)min(ra, T - 1) * CN_NQ + head) * CN_AHD;
        const float* pb = q + ((size_t)min(rb, T - 1) * CN_NQ + head) * CN_AHD;
        #pragma unroll
        for (int ks = 0; ks < CN_AHD / 16; ks++) {
            const int c = ks * 16 + 2 * tq;
            qf[ks][0] = fa_pack_h2(pa[c], pa[c + 1]);
            qf[ks][1] = fa_pack_h2(pb[c], pb[c + 1]);
            qf[ks][2] = fa_pack_h2(pa[c + 8], pa[c + 9]);
            qf[ks][3] = fa_pack_h2(pb[c + 8], pb[c + 9]);
        }
    }
    float o[CN_AHD / 8][4];
    #pragma unroll
    for (int n = 0; n < CN_AHD / 8; n++) { o[n][0] = 0.0f; o[n][1] = 0.0f; o[n][2] = 0.0f; o[n][3] = 0.0f; }
    float ma = FA_NEG_INF, mb = FA_NEG_INF, la = 0.0f, lb = 0.0f;
    // keys 0 ..= the tile's last query position
    const int kall = min(pos_base + min(r0 + 15, T - 1) + 1, tmax);
    const int S = gridDim.z, split = blockIdx.z;
    const int per = ((kall + S - 1) / S + FA_KT - 1) / FA_KT * FA_KT; // whole tiles per split
    const int kbeg = min(kall, split * per), kend = min(kall, kbeg + per);
    const size_t kvbase = (size_t)kvh * tmax;
    for (int k0 = kbeg; k0 < kend; k0 += FA_KT) {
        __syncthreads(); // the previous tile's readers are done
        // stage: 16 keys x CN_AHD values of K and V, e4m3 or bf16 -> f16
        for (int i = threadIdx.x; i < FA_KT * CN_AHD / 2; i += blockDim.x) {
            const int key = i / (CN_AHD / 2), d = (i % (CN_AHD / 2)) * 2;
            const int kk = min(k0 + key, tmax - 1);
            const unsigned char* kp = kc + (kvbase + kk) * CN_AHD * esz;
            const unsigned char* vp = vc + (kvbase + kk) * CN_AHD * esz;
            unsigned int kh, vh;
            if (mode == 0) {
                // e4m3 -> f16 is exact (f16 holds every e4m3 value); the hardware pair convert
                // replaced dec_e4m3's ldexpf (666 us per decode layer at 30k, 2026-09-26)
                kh = fa_e4m3x2_h2(*(const unsigned short*)(kp + d));
                vh = fa_e4m3x2_h2(*(const unsigned short*)(vp + d));
            } else {
                kh = fa_pack_h2(kv_load(kp, d, 1), kv_load(kp, d + 1, 1));
                vh = fa_pack_h2(kv_load(vp, d, 1), kv_load(vp, d + 1, 1));
            }
            *(unsigned int*)&k_s[key * FA_KS + d] = kh;
            v_t[d * FA_VS + key] = (unsigned short)(vh & 0xFFFF);
            v_t[(d + 1) * FA_VS + key] = (unsigned short)(vh >> 16);
        }
        __syncthreads();
        // S = Q K^T over the 16 keys: two n8 tiles
        float s[2][4];
        #pragma unroll
        for (int j = 0; j < 2; j++) {
            s[j][0] = 0.0f; s[j][1] = 0.0f; s[j][2] = 0.0f; s[j][3] = 0.0f;
            const unsigned short* kr = &k_s[(j * 8 + g) * FA_KS + 2 * tq];
            #pragma unroll
            for (int ks = 0; ks < CN_AHD / 16; ks++) {
                const unsigned int b0 = *(const unsigned int*)(kr + ks * 16);
                const unsigned int b1 = *(const unsigned int*)(kr + ks * 16 + 8);
                mma_f16_16n8k16(s[j][0], s[j][1], s[j][2], s[j][3], qf[ks][0], qf[ks][1], qf[ks][2], qf[ks][3], b0, b1);
            }
        }
        // scale (log2 domain) and the causal mask: key k0 + 8j + 2tq (+1) against qa / qb
        float mxa = ma, mxb = mb;
        #pragma unroll
        for (int j = 0; j < 2; j++) {
            const int key = k0 + j * 8 + 2 * tq;
            s[j][0] = (key <= qa && key < kend) ? s[j][0] * sl2 : FA_NEG_INF;
            s[j][1] = (key + 1 <= qa && key + 1 < kend) ? s[j][1] * sl2 : FA_NEG_INF;
            s[j][2] = (key <= qb && key < kend) ? s[j][2] * sl2 : FA_NEG_INF;
            s[j][3] = (key + 1 <= qb && key + 1 < kend) ? s[j][3] * sl2 : FA_NEG_INF;
            mxa = fmaxf(mxa, fmaxf(s[j][0], s[j][1]));
            mxb = fmaxf(mxb, fmaxf(s[j][2], s[j][3]));
        }
        mxa = fmaxf(mxa, __shfl_xor_sync(0xffffffffu, mxa, 1));
        mxa = fmaxf(mxa, __shfl_xor_sync(0xffffffffu, mxa, 2));
        mxb = fmaxf(mxb, __shfl_xor_sync(0xffffffffu, mxb, 1));
        mxb = fmaxf(mxb, __shfl_xor_sync(0xffffffffu, mxb, 2));
        // a row whose keys are all masked so far keeps m = -inf; its exps are 0 (use 0 as base)
        const float ba = (mxa == FA_NEG_INF) ? 0.0f : mxa, bb = (mxb == FA_NEG_INF) ? 0.0f : mxb;
        const float ca = exp2f(ma - ba), cb = exp2f(mb - bb);
        ma = mxa; mb = mxb;
        float p[2][4], sa = 0.0f, sb = 0.0f;
        #pragma unroll
        for (int j = 0; j < 2; j++) {
            p[j][0] = exp2f(s[j][0] - ba); p[j][1] = exp2f(s[j][1] - ba);
            p[j][2] = exp2f(s[j][2] - bb); p[j][3] = exp2f(s[j][3] - bb);
            sa += p[j][0] + p[j][1];
            sb += p[j][2] + p[j][3];
        }
        sa += __shfl_xor_sync(0xffffffffu, sa, 1); sa += __shfl_xor_sync(0xffffffffu, sa, 2);
        sb += __shfl_xor_sync(0xffffffffu, sb, 1); sb += __shfl_xor_sync(0xffffffffu, sb, 2);
        la = la * ca + sa;
        lb = lb * cb + sb;
        // P as the A operand of P V (keys 0..15 = k16): rows g / g + 8, keys 2tq (+1) and 2tq + 8 (+9)
        const unsigned int pa0 = fa_pack_h2(p[0][0], p[0][1]), pa1 = fa_pack_h2(p[0][2], p[0][3]);
        const unsigned int pa2 = fa_pack_h2(p[1][0], p[1][1]), pa3 = fa_pack_h2(p[1][2], p[1][3]);
        #pragma unroll
        for (int n = 0; n < CN_AHD / 8; n++) {
            o[n][0] *= ca; o[n][1] *= ca; o[n][2] *= cb; o[n][3] *= cb;
            const unsigned short* vr = &v_t[(n * 8 + g) * FA_VS + 2 * tq];
            const unsigned int b0 = *(const unsigned int*)vr;
            const unsigned int b1 = *(const unsigned int*)(vr + 8);
            mma_f16_16n8k16(o[n][0], o[n][1], o[n][2], o[n][3], pa0, pa1, pa2, pa3, b0, b1);
        }
    }
    if (S > 1) {
        // partial for attn_merge: o unnormalized, m in natural log units (m2 * ln 2), l
        const size_t pa = ((size_t)ra * CN_NQ + head) * S + split, pb = ((size_t)rb * CN_NQ + head) * S + split;
        #pragma unroll
        for (int n = 0; n < CN_AHD / 8; n++) {
            const int c = n * 8 + 2 * tq;
            if (ra < T) { out[pa * CN_AHD + c] = o[n][0]; out[pa * CN_AHD + c + 1] = o[n][1]; }
            if (rb < T) { out[pb * CN_AHD + c] = o[n][2]; out[pb * CN_AHD + c + 1] = o[n][3]; }
        }
        if (tq == 0) {
            // an empty split (no key of it at or below the row) weighs 0 in attn_merge
            if (ra < T) { part_ml[pa * 2] = (la > 0.0f) ? ma * 0.6931471805599453f : -3.0e38f; part_ml[pa * 2 + 1] = la; }
            if (rb < T) { part_ml[pb * 2] = (lb > 0.0f) ? mb * 0.6931471805599453f : -3.0e38f; part_ml[pb * 2 + 1] = lb; }
        }
        return;
    }
    const float ia = 1.0f / la, ib = 1.0f / lb;
    #pragma unroll
    for (int n = 0; n < CN_AHD / 8; n++) {
        const int c = n * 8 + 2 * tq;
        if (ra < T) {
            float* oa = out + ((size_t)ra * CN_NQ + head) * CN_AHD + c;
            oa[0] = o[n][0] * ia; oa[1] = o[n][1] * ia;
        }
        if (rb < T) {
            float* ob = out + ((size_t)rb * CN_NQ + head) * CN_AHD + c;
            ob[0] = o[n][2] * ib; ob[1] = o[n][3] * ib;
        }
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

// ---------------- the verify GEMVs of speculative decoding (crow-nest #95) ----------------
// gemv_nvfp4_w / gemv_nvfp4_gu over M <= 4 activation rows (x row m at x + m * k_dim, y row m
// at y + m * rows): the weight row is staged ONCE for all M rows, and each (row, m) pair runs
// the exact operation sequence of the one-row kernel (same chunks, same lane split, same
// `part` chain, same `acc += part * sc`, same shuffle tree), so y[m] is bit-identical to a
// one-row launch on x row m - the batch invariance the greedy contract of the verify needs.
#define NVFP4_MMAX 4
// Each warp takes NR weight rows over the same M activation slices (x traffic per weight byte
// divided by NR; per warp and row the x slices came from L2 again, ~28x the weight bytes at
// M = 4: kprof 2026-09-27). Staging x in shared memory per block was slower (the block barriers,
// 32.3 vs 28.7 ms per verify), so x is read per warp and shared across its rows.
template <int NR>
__device__ __forceinline__ void nvfp4_warp_dot_m(const unsigned char* const (&rowb)[NR], const float* __restrict__ x,
                                                 int bpr, int M, int k_dim, uint4 (*stage)[72], const float* lut,
                                                 const float* lut8, float (&acc)[NR][NVFP4_MMAX]) {
    const int lane = threadIdx.x & 31;
    const int wi = lane & 7, bi = lane >> 3;
    #pragma unroll
    for (int r = 0; r < NR; r++) {
        #pragma unroll
        for (int m = 0; m < NVFP4_MMAX; m++) acc[r][m] = 0.0f;
    }
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
            // the M activation slices under this lane's 8 values, loaded once for the NR rows
            float4 xa[NVFP4_MMAX], xb[NVFP4_MMAX];
            #pragma unroll
            for (int m = 0; m < NVFP4_MMAX; m++) {
                if (m < M) {
                    const float4* xp = (const float4*)(x + (size_t)m * k_dim + (size_t)(b0 + j) * 64 + wi * 8);
                    xa[m] = __ldg(xp);
                    xb[m] = __ldg(xp + 1);
                }
            }
            #pragma unroll
            for (int r = 0; r < NR; r++) {
                // the 8 weights and the sub-block scale decoded ONCE for all M rows (the table
                // values the one-row kernel multiplies: each row's products and sums are the
                // same operations in the same order)
                const unsigned char* blk = (const unsigned char*)stage[r] + j * 36;
                const unsigned int wd = *(const unsigned int*)(blk + 4 + 4 * wi);
                const float sc = lut8[blk[wi >> 1]];
                const float w0 = lut[wd & 0xF], w1 = lut[(wd >> 4) & 0xF], w2 = lut[(wd >> 8) & 0xF], w3 = lut[(wd >> 12) & 0xF];
                const float w4 = lut[(wd >> 16) & 0xF], w5 = lut[(wd >> 20) & 0xF], w6 = lut[(wd >> 24) & 0xF], w7 = lut[wd >> 28];
                #pragma unroll
                for (int m = 0; m < NVFP4_MMAX; m++) {
                    if (m < M) {
                        float part = w0 * xa[m].x;
                        part += w1 * xa[m].y;
                        part += w2 * xa[m].z;
                        part += w3 * xa[m].w;
                        part += w4 * xb[m].x;
                        part += w5 * xb[m].y;
                        part += w6 * xb[m].z;
                        part += w7 * xb[m].w;
                        acc[r][m] += part * sc;
                    }
                }
            }
        }
        __syncwarp(); // the next chunk overwrites the stage
    }
    #pragma unroll
    for (int r = 0; r < NR; r++) {
        #pragma unroll
        for (int m = 0; m < NVFP4_MMAX; m++) {
            #pragma unroll
            for (int o = 16; o > 0; o >>= 1) acc[r][m] += __shfl_down_sync(0xffffffffu, acc[r][m], o);
        }
    }
}
// two rows per warp: grid (ceil(rows / 16)), block 256; M = *m_p in 1..=4. A warp whose second
// row is past the end computes it on row 0's bytes and does not write it.
extern "C" __global__ void gemv_nvfp4_wm(const unsigned char* __restrict__ w, const float* __restrict__ x,
                                         const float* __restrict__ gs_ptr, float* __restrict__ y,
                                         const int* __restrict__ k_dim_p, const int* __restrict__ rows_p,
                                         const int* __restrict__ m_p) {
    NVFP4_TABLES
    __shared__ uint4 stage[8][2][72];
    const int warp = threadIdx.x >> 5;
    const int row = (blockIdx.x * 8 + warp) * 2;
    const int rows = *rows_p;
    if (row >= rows) return; // no block-wide barrier below this line
    const int k_dim = *k_dim_p, M = *m_p;
    const int bpr = k_dim >> 6;
    const bool two = row + 1 < rows;
    const unsigned char* const rowb[2] = {w + (size_t)row * bpr * 36, w + (size_t)(two ? row + 1 : row) * bpr * 36};
    float acc[2][NVFP4_MMAX];
    nvfp4_warp_dot_m<2>(rowb, x, bpr, M, k_dim, stage[warp], lut, lut8, acc);
    if ((threadIdx.x & 31) == 0) {
        #pragma unroll
        for (int m = 0; m < NVFP4_MMAX; m++) {
            if (m < M) {
                y[(size_t)m * rows + row] = acc[0][m] * gs_ptr[0];
                if (two) y[(size_t)m * rows + row + 1] = acc[1][m] * gs_ptr[0];
            }
        }
    }
}
// two FFN indices per warp (gate and up of each): grid (ceil(rows / 16)), block 256
extern "C" __global__ void gemv_nvfp4_gum(const unsigned char* __restrict__ wg, const unsigned char* __restrict__ wu,
                                          const float* __restrict__ x, const float* __restrict__ gs_g,
                                          const float* __restrict__ gs_u, float* __restrict__ y,
                                          const int* __restrict__ k_dim_p, const int* __restrict__ rows_p,
                                          const int* __restrict__ m_p) {
    NVFP4_TABLES
    __shared__ uint4 stage[8][4][72];
    const int warp = threadIdx.x >> 5;
    const int row = (blockIdx.x * 8 + warp) * 2;
    const int rows = *rows_p;
    if (row >= rows) return;
    const int k_dim = *k_dim_p, M = *m_p;
    const int bpr = k_dim >> 6;
    const bool two = row + 1 < rows;
    const size_t o0 = (size_t)row * bpr * 36, o1 = (size_t)(two ? row + 1 : row) * bpr * 36;
    const unsigned char* const rowb[4] = {wg + o0, wu + o0, wg + o1, wu + o1};
    float acc[4][NVFP4_MMAX];
    nvfp4_warp_dot_m<4>(rowb, x, bpr, M, k_dim, stage[warp], lut, lut8, acc);
    if ((threadIdx.x & 31) == 0) {
        #pragma unroll
        for (int m = 0; m < NVFP4_MMAX; m++) {
            if (m < M) {
                const float g0 = acc[0][m] * gs_g[0];
                y[(size_t)m * rows + row] = (g0 / (1.0f + expf(-g0))) * (acc[1][m] * gs_u[0]);
                if (two) {
                    const float g1 = acc[2][m] * gs_g[0];
                    y[(size_t)m * rows + row + 1] = (g1 / (1.0f + expf(-g1))) * (acc[3][m] * gs_u[0]);
                }
            }
        }
    }
}

// gemv_bf16_w over M <= 4 activation rows (x row m at x + m * k_dim, y row m at y + m * rows),
// the weight row read ONCE for all rows (crow-nest #95: the MTP head's catch-up rows; the
// one-row form read the 0.79 GB of BF16 MTP weights once per row). grid (ceil(rows / 8)),
// block 256; M = *m_p.
extern "C" __global__ void gemv_bf16_wm(const unsigned short* __restrict__ w, const float* __restrict__ x,
                                        float* __restrict__ y, const int* __restrict__ k_dim_p,
                                        const int* __restrict__ rows_p, const int* __restrict__ m_p) {
    const int k_dim = *k_dim_p, rows = *rows_p, M = *m_p;
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int row = blockIdx.x * 8 + warp;
    if (row >= rows) return;
    const unsigned short* wp = w + (size_t)row * k_dim;
    float acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    for (int i = lane * 8; i < k_dim; i += 256) {
        const uint4 v = *(const uint4*)(wp + i);
        const unsigned int u[4] = {v.x, v.y, v.z, v.w};
        #pragma unroll
        for (int m = 0; m < 4; m++) {
            if (m < M) {
                const float* xp = x + (size_t)m * k_dim + i;
                #pragma unroll
                for (int j = 0; j < 4; j++) {
                    const float lo = __uint_as_float(u[j] << 16);
                    const float hi = __uint_as_float(u[j] & 0xFFFF0000u);
                    acc[m] += lo * xp[2 * j] + hi * xp[2 * j + 1];
                }
            }
        }
    }
    #pragma unroll
    for (int m = 0; m < 4; m++) {
        #pragma unroll
        for (int o = 16; o > 0; o >>= 1) acc[m] += __shfl_down_sync(0xffffffffu, acc[m], o);
    }
    if (lane == 0) {
        #pragma unroll
        for (int m = 0; m < 4; m++) if (m < M) y[(size_t)m * rows + row] = acc[m];
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

/// #88: the KV kernels `Q8KV_SRC` replaces at boot under `CROW_KV=q8` (`Kernels::arm_q8_kv`):
/// (name launched by `gen.rs`, its q8 twin). Same signatures, same launch sites.
pub const Q8KV_SWAP: &[(&str, &str)] = &[
    ("store_kv", "store_kv_q8"),
    ("attn_full_split", "attn_full_split_q8"),
    ("attn_full_fa", "attn_full_fa_q8"),
];

/// #88: the q8 KV twins of `store_kv`, `attn_full_split` and `attn_full_fa`. Appended after
/// `P2_SRC` ONLY when the boot runs `CROW_KV=q8` (`KernelGeo::q8kv`), so without it the
/// module text, and with it every PTX entry, is the one of before #88. They use `P2_SRC`'s
/// helpers (`fa_pack_h2`, `fa_h`, `mma_f16_16n8k16`, `AF_TILE`, `FA_*`) as they are.
pub const Q8KV_SRC: &str = r#"
// ---------------- #88: the q8 KV cache (CROW_KV=q8, opt-in) ----------------
// q8_0 numerics (llama.cpp quantize_row_q8_0_ref in ggml-quants.c, and the CUDA KV copy
// quantize_f32_q8_0_block in ggml-cuda/cpy-utils.cuh): per block of 32 values d = amax / 127,
// q = roundf(x * (1 / d)) as int8 (0 when d = 0), d stored as f16 (round to nearest even);
// a value reads back as q * d. One cache row (token x KV head, K or V) is Q8_ROWB bytes:
// the CN_AHD int8 values, then the CN_AHD / 32 f16 scales (272 B at head dim 256, 8.5 bit
// per value), rows in store_kv's order [kvh][tmax][row]. The scale sits after the values
// rather than in front of each block (llama.cpp's 34-byte block_q8_0) so the value bytes
// keep store_kv's alignment. `mode_p` is accepted and ignored: the twins keep the
// signatures of the kernels they replace.
#define Q8_ROWB (CN_AHD + CN_AHD / 16)
static_assert(CN_AHD % 32 == 0, "q8 KV: whole 32-value blocks per row");
__device__ __forceinline__ float q8_h2f(unsigned short h) {
    float r;
    asm("cvt.f32.f16 %0, %1;" : "=f"(r) : "h"(h));
    return r;
}
// value i of a q8 row as f32: int8 x the f16 scale of its 32-value block
__device__ __forceinline__ float q8_ld(const unsigned char* row, int i) {
    return (float)(signed char)row[i] * q8_h2f(((const unsigned short*)(row + CN_AHD))[i >> 5]);
}

// store_kv's twin: grid (2 * NKV, T), block CN_AHD; one warp per 32-value block reduces the
// block's amax, every lane writes its int8, lane 0 the block's scale
extern "C" __global__ void store_kv_q8(const float* __restrict__ kr, const float* __restrict__ vr,
                                       unsigned char* __restrict__ kcache, unsigned char* __restrict__ vcache,
                                       const int* __restrict__ slot_p, const int* __restrict__ tmax_p,
                                       const int* __restrict__ mode_p) {
    int h = blockIdx.x;
    int d = threadIdx.x;
    int slot = *slot_p + (int)blockIdx.y;
    int tmax = *tmax_p;
    int bh = (h < CN_NKV) ? h : h - CN_NKV;
    size_t rowb = (size_t)(bh * tmax + slot) * Q8_ROWB;
    const float* src = (h < CN_NKV) ? kr + (size_t)(blockIdx.y * CN_NKV + bh) * CN_AHD
                                    : vr + (size_t)(blockIdx.y * CN_NKV + bh) * CN_AHD;
    unsigned char* dst = (h < CN_NKV) ? kcache + rowb : vcache + rowb;
    const float x = src[d];
    float amax = fabsf(x);
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, o));
    const float sc = amax / 127.0f;
    const float id = (sc != 0.0f) ? 1.0f / sc : 0.0f;
    dst[d] = (unsigned char)(signed char)(int)roundf(x * id);
    if ((d & 31) == 0) ((unsigned short*)(dst + CN_AHD))[d >> 5] = fa_h(sc);
}

// attn_full_split's twin (the CROW_P2_FA=0 path): the same online softmax, the K / V values
// read as int8 x scale. grid (NQ, T, S), block CN_AHD.
extern "C" __global__ void attn_full_split_q8(const float* __restrict__ q, const unsigned char* __restrict__ kc,
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
    __shared__ float qs[CN_AHD];
    qs[d] = q[((size_t)t * CN_NQ + head) * CN_AHD + d];
    __syncthreads();
    int warp = d >> 5, lane = d & 31;
    const float scale = CN_ATTN_SCALE;
    const size_t kvbase = (size_t)kvh * tmax;
    float m = -3.0e38f, l = 0.0f, o = 0.0f;
    for (int j0 = lo; j0 < hi; j0 += AF_TILE) {
        int cnt = min(AF_TILE, hi - j0);
        for (int jj = warp; jj < cnt; jj += CN_AHD / 32) {
            const unsigned char* kp = kc + (kvbase + j0 + jj) * Q8_ROWB;
            float acc = 0.0f;
            for (int e = lane; e < CN_AHD; e += 32) acc += qs[e] * q8_ld(kp, e);
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
            o += e * q8_ld(vc + (kvbase + j0 + jj) * Q8_ROWB, d);
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

// attn_full_fa's twin (prefill S = 1 and split-K decode): the same tiles, MMAs, online softmax
// and partials; only the stage differs - a K / V pair of int8 values times its block's f16
// scale (the pair never straddles a block: d is even), rounded once to f16. Every line but the
// stage is attn_full_fa's. grid (NKV, ceil(T / 16), S), block 32 * CN_GQA.
extern "C" __global__ void __launch_bounds__(32 * CN_GQA, 1)
attn_full_fa_q8(const float* __restrict__ q, const unsigned char* __restrict__ kc, const unsigned char* __restrict__ vc,
                const int* __restrict__ pos_base_p, const int* __restrict__ t_p, const int* __restrict__ tmax_p,
                const int* __restrict__ mode_p, float* __restrict__ out, float* __restrict__ part_ml) {
    __shared__ unsigned short k_s[FA_KT * FA_KS]; // f16 bits
    __shared__ unsigned short v_t[CN_AHD * FA_VS];
    const int kvh = blockIdx.x;
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int g = lane >> 2, tq = lane & 3;
    const int head = kvh * CN_GQA + warp;
    const int T = *t_p, pos_base = *pos_base_p, tmax = *tmax_p;
    const int r0 = blockIdx.y * 16;
    const int ra = r0 + g, rb = r0 + g + 8;            // this lane's two query rows
    const int qa = pos_base + ra, qb = pos_base + rb;   // their positions
    const float sl2 = CN_ATTN_SCALE * 1.4426950408889634f; // scale * log2(e)
    unsigned int qf[CN_AHD / 16][4];
    {
        const float* pa = q + ((size_t)min(ra, T - 1) * CN_NQ + head) * CN_AHD;
        const float* pb = q + ((size_t)min(rb, T - 1) * CN_NQ + head) * CN_AHD;
        #pragma unroll
        for (int ks = 0; ks < CN_AHD / 16; ks++) {
            const int c = ks * 16 + 2 * tq;
            qf[ks][0] = fa_pack_h2(pa[c], pa[c + 1]);
            qf[ks][1] = fa_pack_h2(pb[c], pb[c + 1]);
            qf[ks][2] = fa_pack_h2(pa[c + 8], pa[c + 9]);
            qf[ks][3] = fa_pack_h2(pb[c + 8], pb[c + 9]);
        }
    }
    float o[CN_AHD / 8][4];
    #pragma unroll
    for (int n = 0; n < CN_AHD / 8; n++) { o[n][0] = 0.0f; o[n][1] = 0.0f; o[n][2] = 0.0f; o[n][3] = 0.0f; }
    float ma = FA_NEG_INF, mb = FA_NEG_INF, la = 0.0f, lb = 0.0f;
    const int kall = min(pos_base + min(r0 + 15, T - 1) + 1, tmax);
    const int S = gridDim.z, split = blockIdx.z;
    const int per = ((kall + S - 1) / S + FA_KT - 1) / FA_KT * FA_KT; // whole tiles per split
    const int kbeg = min(kall, split * per), kend = min(kall, kbeg + per);
    const size_t kvbase = (size_t)kvh * tmax;
    for (int k0 = kbeg; k0 < kend; k0 += FA_KT) {
        __syncthreads(); // the previous tile's readers are done
        // stage: 16 keys x CN_AHD values of K and V, int8 x f16 scale -> f16
        for (int i = threadIdx.x; i < FA_KT * CN_AHD / 2; i += blockDim.x) {
            const int key = i / (CN_AHD / 2), d = (i % (CN_AHD / 2)) * 2;
            const int kk = min(k0 + key, tmax - 1);
            const unsigned char* kp = kc + (kvbase + kk) * Q8_ROWB;
            const unsigned char* vp = vc + (kvbase + kk) * Q8_ROWB;
            const float ksc = q8_h2f(((const unsigned short*)(kp + CN_AHD))[d >> 5]);
            const float vsc = q8_h2f(((const unsigned short*)(vp + CN_AHD))[d >> 5]);
            const unsigned short kq = *(const unsigned short*)(kp + d);
            const unsigned short vq = *(const unsigned short*)(vp + d);
            const unsigned int kh = fa_pack_h2((float)(signed char)(kq & 0xFF) * ksc, (float)(signed char)(kq >> 8) * ksc);
            const unsigned int vh = fa_pack_h2((float)(signed char)(vq & 0xFF) * vsc, (float)(signed char)(vq >> 8) * vsc);
            *(unsigned int*)&k_s[key * FA_KS + d] = kh;
            v_t[d * FA_VS + key] = (unsigned short)(vh & 0xFFFF);
            v_t[(d + 1) * FA_VS + key] = (unsigned short)(vh >> 16);
        }
        __syncthreads();
        float s[2][4];
        #pragma unroll
        for (int j = 0; j < 2; j++) {
            s[j][0] = 0.0f; s[j][1] = 0.0f; s[j][2] = 0.0f; s[j][3] = 0.0f;
            const unsigned short* kr = &k_s[(j * 8 + g) * FA_KS + 2 * tq];
            #pragma unroll
            for (int ks = 0; ks < CN_AHD / 16; ks++) {
                const unsigned int b0 = *(const unsigned int*)(kr + ks * 16);
                const unsigned int b1 = *(const unsigned int*)(kr + ks * 16 + 8);
                mma_f16_16n8k16(s[j][0], s[j][1], s[j][2], s[j][3], qf[ks][0], qf[ks][1], qf[ks][2], qf[ks][3], b0, b1);
            }
        }
        float mxa = ma, mxb = mb;
        #pragma unroll
        for (int j = 0; j < 2; j++) {
            const int key = k0 + j * 8 + 2 * tq;
            s[j][0] = (key <= qa && key < kend) ? s[j][0] * sl2 : FA_NEG_INF;
            s[j][1] = (key + 1 <= qa && key + 1 < kend) ? s[j][1] * sl2 : FA_NEG_INF;
            s[j][2] = (key <= qb && key < kend) ? s[j][2] * sl2 : FA_NEG_INF;
            s[j][3] = (key + 1 <= qb && key + 1 < kend) ? s[j][3] * sl2 : FA_NEG_INF;
            mxa = fmaxf(mxa, fmaxf(s[j][0], s[j][1]));
            mxb = fmaxf(mxb, fmaxf(s[j][2], s[j][3]));
        }
        mxa = fmaxf(mxa, __shfl_xor_sync(0xffffffffu, mxa, 1));
        mxa = fmaxf(mxa, __shfl_xor_sync(0xffffffffu, mxa, 2));
        mxb = fmaxf(mxb, __shfl_xor_sync(0xffffffffu, mxb, 1));
        mxb = fmaxf(mxb, __shfl_xor_sync(0xffffffffu, mxb, 2));
        const float ba = (mxa == FA_NEG_INF) ? 0.0f : mxa, bb = (mxb == FA_NEG_INF) ? 0.0f : mxb;
        const float ca = exp2f(ma - ba), cb = exp2f(mb - bb);
        ma = mxa; mb = mxb;
        float p[2][4], sa = 0.0f, sb = 0.0f;
        #pragma unroll
        for (int j = 0; j < 2; j++) {
            p[j][0] = exp2f(s[j][0] - ba); p[j][1] = exp2f(s[j][1] - ba);
            p[j][2] = exp2f(s[j][2] - bb); p[j][3] = exp2f(s[j][3] - bb);
            sa += p[j][0] + p[j][1];
            sb += p[j][2] + p[j][3];
        }
        sa += __shfl_xor_sync(0xffffffffu, sa, 1); sa += __shfl_xor_sync(0xffffffffu, sa, 2);
        sb += __shfl_xor_sync(0xffffffffu, sb, 1); sb += __shfl_xor_sync(0xffffffffu, sb, 2);
        la = la * ca + sa;
        lb = lb * cb + sb;
        const unsigned int pa0 = fa_pack_h2(p[0][0], p[0][1]), pa1 = fa_pack_h2(p[0][2], p[0][3]);
        const unsigned int pa2 = fa_pack_h2(p[1][0], p[1][1]), pa3 = fa_pack_h2(p[1][2], p[1][3]);
        #pragma unroll
        for (int n = 0; n < CN_AHD / 8; n++) {
            o[n][0] *= ca; o[n][1] *= ca; o[n][2] *= cb; o[n][3] *= cb;
            const unsigned short* vr = &v_t[(n * 8 + g) * FA_VS + 2 * tq];
            const unsigned int b0 = *(const unsigned int*)vr;
            const unsigned int b1 = *(const unsigned int*)(vr + 8);
            mma_f16_16n8k16(o[n][0], o[n][1], o[n][2], o[n][3], pa0, pa1, pa2, pa3, b0, b1);
        }
    }
    if (S > 1) {
        const size_t pa = ((size_t)ra * CN_NQ + head) * S + split, pb = ((size_t)rb * CN_NQ + head) * S + split;
        #pragma unroll
        for (int n = 0; n < CN_AHD / 8; n++) {
            const int c = n * 8 + 2 * tq;
            if (ra < T) { out[pa * CN_AHD + c] = o[n][0]; out[pa * CN_AHD + c + 1] = o[n][1]; }
            if (rb < T) { out[pb * CN_AHD + c] = o[n][2]; out[pb * CN_AHD + c + 1] = o[n][3]; }
        }
        if (tq == 0) {
            if (ra < T) { part_ml[pa * 2] = (la > 0.0f) ? ma * 0.6931471805599453f : -3.0e38f; part_ml[pa * 2 + 1] = la; }
            if (rb < T) { part_ml[pb * 2] = (lb > 0.0f) ? mb * 0.6931471805599453f : -3.0e38f; part_ml[pb * 2 + 1] = lb; }
        }
        return;
    }
    const float ia = 1.0f / la, ib = 1.0f / lb;
    #pragma unroll
    for (int n = 0; n < CN_AHD / 8; n++) {
        const int c = n * 8 + 2 * tq;
        if (ra < T) {
            float* oa = out + ((size_t)ra * CN_NQ + head) * CN_AHD + c;
            oa[0] = o[n][0] * ia; oa[1] = o[n][1] * ia;
        }
        if (rb < T) {
            float* ob = out + ((size_t)rb * CN_NQ + head) * CN_AHD + c;
            ob[0] = o[n][2] * ib; ob[1] = o[n][3] * ib;
        }
    }
}
"#;

/// #88: the host side of the q8 KV row (`Q8KV_SRC`): the CPU reference of `store_kv_q8` and
/// the decode `Engine::kv_rows_host` reads a q8 cache with
pub mod q8kv {
    /// f32 -> IEEE binary16 bits, round to nearest even (the PTX `cvt.rn.f16.f32` of `fa_h`);
    /// overflow to inf, NaN stays NaN
    pub fn f32_to_f16(v: f32) -> u16 {
        let x = v.to_bits();
        let sign = ((x >> 16) & 0x8000) as u16;
        let exp = ((x >> 23) & 0xFF) as i32;
        let man = x & 0x7F_FFFF;
        if exp == 0xFF {
            return sign | 0x7C00 | if man != 0 { 0x200 } else { 0 };
        }
        let e = exp - 127 + 15;
        if e >= 0x1F {
            return sign | 0x7C00;
        }
        // value = m * 2^(e - 25) with m the 24-bit significand; keep `shift` fewer bits
        let (m, shift) = if e > 0 { (man | 0x80_0000, 13u32) } else { (if exp == 0 { man } else { man | 0x80_0000 }, (14 - e) as u32) };
        if shift > 24 {
            return sign;
        }
        let q = m >> shift;
        let rem = m & ((1u32 << shift) - 1);
        let half = 1u32 << (shift - 1);
        let q = if rem > half || (rem == half && q & 1 == 1) { q + 1 } else { q };
        // normal: the exponent field plus the 10-bit mantissa (a carry out of it bumps the
        // exponent, up to inf); subnormal: q itself (q = 0x400 is the smallest normal)
        let bits = if e > 0 { ((e as u32) << 10) + (q - 0x400) } else { q };
        sign | bits as u16
    }

    /// one row of `x.len()` values (a multiple of 32) in the cache layout: the int8 values,
    /// then one f16 scale per 32 of them; the same operations in the same order as
    /// `store_kv_q8` (amax / 127, x * (1 / d), round half away from zero)
    pub fn encode_row(x: &[f32]) -> Vec<u8> {
        assert_eq!(x.len() % 32, 0, "q8 KV rows are whole 32-value blocks");
        let n = x.len();
        let mut out = vec![0u8; n + n / 16];
        for (b, blk) in x.chunks(32).enumerate() {
            let amax = blk.iter().fold(0f32, |m, v| m.max(v.abs()));
            let d = amax / 127.0;
            let id = if d != 0.0 { 1.0 / d } else { 0.0 };
            for (j, &v) in blk.iter().enumerate() {
                out[b * 32 + j] = ((v * id).round() as i32 as i8) as u8;
            }
            out[n + 2 * b..n + 2 * b + 2].copy_from_slice(&f32_to_f16(d).to_le_bytes());
        }
        out
    }

    /// the `n` values of one q8 row as f32: int8 x the f16 scale of its block (`q8_ld`)
    pub fn decode_row(row: &[u8], n: usize) -> Vec<f32> {
        (0..n)
            .map(|i| {
                let s = u16::from_le_bytes([row[n + 2 * (i / 32)], row[n + 2 * (i / 32) + 1]]);
                row[i] as i8 as f32 * crate::gguf::f16_to_f32(s)
            })
            .collect()
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        /// #88: the f16 encoder the CPU reference compares scale bits with: every finite
        /// f16 round-trips, a value halfway between two f16 neighbours goes to the even one
        #[test]
        fn f32_to_f16_round_trips_every_f16_and_ties_to_even() {
            for h in 0u16..=0xFFFF {
                let f = crate::gguf::f16_to_f32(h);
                if f.is_nan() {
                    assert!(f32_to_f16(f) & 0x7C00 == 0x7C00 && f32_to_f16(f) & 0x3FF != 0, "{h:#06x}");
                    continue;
                }
                assert_eq!(f32_to_f16(f), h, "{h:#06x} = {f:e}");
            }
            // 1 + 2^-11 lies halfway between 1.0 (0x3C00) and 1 + 2^-10 (0x3C01): even wins
            assert_eq!(f32_to_f16(1.0 + 2f32.powi(-11)), 0x3C00);
            assert_eq!(f32_to_f16(1.0 + 3.0 * 2f32.powi(-11)), 0x3C02);
            // the largest subnormal tie and an overflow
            assert_eq!(f32_to_f16(2f32.powi(-25)), 0x0000);
            assert_eq!(f32_to_f16(3.0 * 2f32.powi(-25)), 0x0002);
            assert_eq!(f32_to_f16(65520.0), 0x7C00);
            assert_eq!(f32_to_f16(-65504.0), 0xFBFF);
        }

        /// #88: one encoded row decodes to within half a scale step of every value, the
        /// block's amax hits +-127 exactly, an all-zero block stores scale 0
        #[test]
        fn a_q8_row_decodes_within_half_a_step_and_a_zero_block_has_scale_zero() {
            let mut x: Vec<f32> = (0..256).map(|i| ((i * 37 % 101) as f32 - 50.0) * 0.173).collect();
            for v in &mut x[64..96] {
                *v = 0.0;
            }
            x[5] = 97.0; // the long-context max |K| of the #90 session
            let row = encode_row(&x);
            assert_eq!(row.len(), 272);
            let back = decode_row(&row, 256);
            for (b, blk) in x.chunks(32).enumerate() {
                let amax = blk.iter().fold(0f32, |m, v| m.max(v.abs()));
                let step = crate::gguf::f16_to_f32(u16::from_le_bytes([row[256 + 2 * b], row[257 + 2 * b]]));
                assert!((step - amax / 127.0).abs() <= amax / 127.0 * 1e-3, "block {b}: scale {step} vs {}", amax / 127.0);
                for j in 0..32 {
                    let i = b * 32 + j;
                    assert!((back[i] - x[i]).abs() <= 0.5 * step + amax * 1e-3, "value {i}: {} -> {}", x[i], back[i]);
                }
                if amax > 0.0 {
                    assert!(row[b * 32..b * 32 + 32].iter().any(|&q| q as i8 == 127 || q as i8 == -127), "block {b}: amax is +-127");
                }
            }
            assert_eq!(&row[256 + 4..256 + 6], &[0, 0], "the zero block stores scale 0");
            assert!(back[64..96].iter().all(|&v| v == 0.0));
        }
    }
}

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

#[cfg(test)]
mod tests_88_q8kv {
    use super::{P2_SRC, Q8KV_SRC, Q8KV_SWAP};
    use crate::kernels::tests_300_c4::{entries, ptx};
    use crate::kernels::{KernelGeo, KERNEL_SRC};

    /// #88: without `CROW_KV=q8` the dense module text is the pre-#88 text (so its PTX, launches
    /// and logits are); with it the q8 text is appended and every entry the default module has
    /// compiles to byte-identical PTX in the q8 module, plus the three twins. NVRTC is a host
    /// compiler: no GPU.
    #[test]
    fn the_q8_kv_text_is_appended_only_under_q8_and_leaves_every_other_entry_ptx_unchanged() {
        let base = KernelGeo::of(&crate::meta::dense_fixture_geo());
        assert!(!base.q8kv, "KernelGeo::of never arms q8");
        assert_eq!(base.source(), format!("{}{}{}", base.prelude(), KERNEL_SRC, P2_SRC), "the default dense text");
        let q8 = KernelGeo { q8kv: true, ..base };
        assert_eq!(q8.source(), format!("{}{}", base.source(), Q8KV_SRC));
        let (e0, e1) = (entries(&ptx(&base.source())), entries(&ptx(&q8.source())));
        assert_eq!(e1.len(), e0.len() + Q8KV_SWAP.len());
        for (n, body) in &e0 {
            let got = e1.iter().find(|(m, _)| m == n).map(|(_, b)| b);
            assert!(got == Some(body), "{n}: its PTX differs once the q8 text is appended");
        }
        for &(orig, twin) in Q8KV_SWAP {
            assert!(e0.iter().any(|(m, _)| m == orig), "{orig} is a default entry");
            assert!(e1.iter().any(|(m, _)| m == twin) && !e0.iter().any(|(m, _)| m == twin), "{twin} only under q8");
        }
    }
}

#[cfg(test)]
mod tests_88_q8kv_gpu {
    //! #88: the q8 KV twins on synthetic data against a CPU reference (no model, no
    //! container; the `tests_300_c4_gpu` pattern). `#[ignore]`: CI has no GPU. Run with
    //! `cargo test --release tests_88_q8kv_gpu -- --ignored --nocapture`.
    use super::q8kv::{decode_row, encode_row};
    use crate::cuda;
    use crate::kernels::{launch_sync, KernelGeo};
    use crate::sample::Rng;

    fn fill(n: usize, seed: u64, scale: f32) -> Vec<f32> {
        let mut r = Rng::new(seed);
        (0..n).map(|_| ((r.next_f64() * 2.0 - 1.0) as f32) * scale).collect()
    }

    /// the dense 27B kernel shape (NKV 4, NQ 24 = GQA 6, head dim 256), q8 text appended
    fn q8_geo() -> KernelGeo {
        KernelGeo { q8kv: true, ..KernelGeo::of(&crate::meta::dense_fixture_geo()) }
    }

    /// K rows `[n][nkv][ahd]` in +-2 with two outlier channels up to +-20 (real K has outlier
    /// channels; the max |K| of the #90 session was 97) and one all-zero 32-value block per row
    fn k_rows(n: usize, nkv: usize, ahd: usize, seed: u64) -> Vec<f32> {
        let mut k = fill(n * nkv * ahd, seed, 2.0);
        for (i, row) in k.chunks_mut(ahd).enumerate() {
            row[7] = if i % 2 == 0 { 20.0 } else { -13.5 };
            row[200] = 10.0 * ((i % 5) as f32 - 2.0);
            for v in &mut row[96..128] {
                *v = 0.0;
            }
        }
        k
    }

    #[test]
    #[ignore = "needs the GPU: cargo test --release tests_88_q8kv_gpu -- --ignored --nocapture"]
    fn store_kv_q8_writes_the_cpu_reference_bytes_and_nothing_else() {
        unsafe {
            let _ctx = cuda::Ctx::init();
            let kg = q8_geo();
            let (nkv, ahd) = (kg.d.nkv, kg.d.ahd);
            let rb = crate::geo::KvDtype::Q8Block.row_bytes(ahd);
            let module = cuda::compile(&kg.source());
            let (t, tmax, slot0) = (5usize, 16usize, 3usize);
            let kr = k_rows(t, nkv, ahd, 0x88A1);
            let vr = fill(t * nkv * ahd, 0x88A2, 2.0);
            let (krd, vrd) = (cuda::to_f32_dev(&kr), cuda::to_f32_dev(&vr));
            let bytes = nkv * tmax * rb;
            let (kc, vc) = (cuda::alloc_zeroed(bytes), cuda::alloc_zeroed(bytes));
            let (slot_p, tmax_p, mode_p) = (cuda::to_i32_dev(&[slot0 as i32]), cuda::to_i32_dev(&[tmax as i32]), cuda::to_i32_dev(&[2]));
            // the store_kv launch: grid (2 * NKV, T), block AHD
            launch_sync(module.get("store_kv_q8"), (2 * nkv) as u32, t as u32, 1, ahd as u32, &[krd, vrd, kc, vc, slot_p, tmax_p, mode_p]);
            let (kb, vb): (Vec<u8>, Vec<u8>) = (cuda::dtoh_t(kc, bytes), cuda::dtoh_t(vc, bytes));
            let (mut stored, mut worst) = (0, 0f32);
            for (cache, src) in [(&kb, &kr), (&vb, &vr)] {
                for kvh in 0..nkv {
                    for slot in 0..tmax {
                        let row = &cache[(kvh * tmax + slot) * rb..(kvh * tmax + slot + 1) * rb];
                        if !(slot0..slot0 + t).contains(&slot) {
                            assert!(row.iter().all(|&b| b == 0), "kvh {kvh} slot {slot}: a row outside the store was written");
                            continue;
                        }
                        let x = &src[((slot - slot0) * nkv + kvh) * ahd..((slot - slot0) * nkv + kvh + 1) * ahd];
                        assert!(row == &encode_row(x)[..], "kvh {kvh} slot {slot}: bytes differ from the CPU reference");
                        stored += 1;
                        // read back within half a step of the block scale (+ the f16 rounding of it)
                        for (xb, yb) in x.chunks(32).zip(decode_row(row, ahd).chunks(32)) {
                            let amax = xb.iter().fold(0f32, |m, v| m.max(v.abs()));
                            for (a, b) in xb.iter().zip(yb) {
                                let e = (a - b).abs() / amax.max(1e-30);
                                worst = worst.max(e);
                                assert!(e <= 0.5 / 127.0 + 1e-3, "{a} -> {b} (block amax {amax})");
                            }
                        }
                    }
                }
            }
            assert_eq!(stored, 2 * t * nkv);
            println!("store_kv_q8: {stored} rows of {rb} B byte-identical to the CPU reference, worst |x - q*d| = {worst:.5} x block amax (half a step is {:.5})", 0.5 / 127.0);
        }
    }

    /// The prefill (S = 1) and split-K decode forms of `attn_full_fa_q8` and `attn_full_split_q8`
    /// over 300 cached rows, against an f64 CPU attention over the SAME cache values (decoded
    /// from the bytes `store_kv_q8` wrote), and `attn_full_fa_q8` against the BF16 path of
    /// record (`store_kv` mode 1 + `attn_full_fa`) on the same K / V. Tolerances fixed before
    /// the first run: the f32 split kernel 1e-4 (the `attn_sel` test); the tensor-core kernel
    /// 5e-3 (Q, K, V and P rounded to f16, rel 2^-11 each, V in +-2).
    ///
    /// AMENDED 2026-10-06, after the first run: the third tolerance was a guessed constant,
    /// q8 against bf16 <= 0.03 (V's q8 step <= amax / 254 = 0.0079, plus a score shift guessed
    /// at std 0.01). The run read 3.643e-2 (mean 1.416e-3) while both q8 kernels matched their
    /// CPU reference over the stored values (8.1e-4 / 4.9e-7): the q8_0 format's own error on
    /// rows whose 32-value K block holds a 20.0 outlier (a 10x coarser step for the other 31
    /// values), which the guess underestimated. The check is now the format's exact error
    /// bound per output, against the f64 attention over the ORIGINAL f32 K / V: with
    /// M = max_j |s'_j - s_j| (scores of the q8 K minus those of the f32 K), every softmax
    /// weight moves by at most a factor e^(+-2M), so |o' - o| <= max_j |V'_j - V_j| +
    /// (e^(2M) - 1) * (max_j V_j - min_j V_j) / 2 per output, plus the kernel's 5e-3. The q8
    /// against bf16 difference is printed, not judged.
    #[test]
    #[ignore = "needs the GPU: cargo test --release tests_88_q8kv_gpu -- --ignored --nocapture"]
    fn q8_attention_matches_the_cpu_reference_and_the_q8_format_bound() {
        unsafe {
            let _ctx = cuda::Ctx::init();
            let kg = q8_geo();
            let (nq, nkv, ahd) = (kg.d.nq, kg.d.nkv, kg.d.ahd);
            let gqa = nq / nkv;
            assert_eq!((nq, nkv, ahd, gqa), (24, 4, 256, 6), "the attention shape of the dense 27B");
            let rb = crate::geo::KvDtype::Q8Block.row_bytes(ahd);
            let module = cuda::compile(&kg.source());
            let (n, tmax) = (300usize, 512usize);
            let kr = k_rows(n, nkv, ahd, 0x88B1);
            let vr = fill(n * nkv * ahd, 0x88B2, 2.0);
            let q = fill(n * nq * ahd, 0x88B3, 1.0);
            let (krd, vrd, qd) = (cuda::to_f32_dev(&kr), cuda::to_f32_dev(&vr), cuda::to_f32_dev(&q));
            let (zero, tmax_p, t_p, one) = (cuda::to_i32_dev(&[0]), cuda::to_i32_dev(&[tmax as i32]), cuda::to_i32_dev(&[n as i32]), cuda::to_i32_dev(&[1]));
            let (mode_bf16, mode_q8) = (cuda::to_i32_dev(&[1]), cuda::to_i32_dev(&[2]));
            // the same n rows into a q8 cache and a bf16 cache (store_kv mode 1: the kernel of the default path)
            let (kq, vq) = (cuda::alloc_zeroed(nkv * tmax * rb), cuda::alloc_zeroed(nkv * tmax * rb));
            let (kb, vb) = (cuda::alloc_zeroed(nkv * tmax * ahd * 2), cuda::alloc_zeroed(nkv * tmax * ahd * 2));
            launch_sync(module.get("store_kv_q8"), (2 * nkv) as u32, n as u32, 1, ahd as u32, &[krd, vrd, kq, vq, zero, tmax_p, mode_q8]);
            launch_sync(module.get("store_kv"), (2 * nkv) as u32, n as u32, 1, ahd as u32, &[krd, vrd, kb, vb, zero, tmax_p, mode_bf16]);
            // the CPU reference over the stored q8 values: query row i at position i attends 0..=i
            let (kh, vh): (Vec<u8>, Vec<u8>) = (cuda::dtoh_t(kq, nkv * tmax * rb), cuda::dtoh_t(vq, nkv * tmax * rb));
            let dec = |c: &[u8]| -> Vec<Vec<f32>> {
                (0..nkv * n)
                    .map(|i| {
                        let (h, tok) = (i / n, i % n);
                        decode_row(&c[(h * tmax + tok) * rb..(h * tmax + tok + 1) * rb], ahd)
                    })
                    .collect()
            };
            let (kd, vd) = (dec(&kh), dec(&vh));
            // the f32 rows as they were stored ([kvh * n + tok][e], the layout of kd / vd)
            let orig = |src: &[f32]| -> Vec<Vec<f32>> {
                (0..nkv * n).map(|i| { let (h, tok) = (i / n, i % n); src[(tok * nkv + h) * ahd..(tok * nkv + h + 1) * ahd].to_vec() }).collect()
            };
            let (ko, vo) = (orig(&kr), orig(&vr));
            // one pass: the f64 attention over the stored q8 values (`reference`), over the f32
            // values (`exact`), and the format bound of every output (`bound`)
            let (mut reference, mut exact, mut bound) = (vec![0f64; n * nq * ahd], vec![0f64; n * nq * ahd], vec![0f64; n * nq * ahd]);
            let softmax = |s: &[f64]| -> Vec<f64> {
                let mx = s.iter().cloned().fold(f64::MIN, f64::max);
                let p: Vec<f64> = s.iter().map(|v| (v - mx).exp()).collect();
                let z: f64 = p.iter().sum();
                p.into_iter().map(|v| v / z).collect()
            };
            for kvh in 0..nkv {
                // running over the keys 0..=i: max |V' - V|, max V, min V per dim
                let (mut dv, mut vmax, mut vmin) = (vec![0f64; ahd], vec![f64::MIN; ahd], vec![f64::MAX; ahd]);
                for i in 0..n {
                    for e in 0..ahd {
                        let (v, v8) = (vo[kvh * n + i][e] as f64, vd[kvh * n + i][e] as f64);
                        dv[e] = dv[e].max((v8 - v).abs());
                        vmax[e] = vmax[e].max(v);
                        vmin[e] = vmin[e].min(v);
                    }
                    for h in kvh * gqa..(kvh + 1) * gqa {
                        let qh = &q[(i * nq + h) * ahd..(i * nq + h + 1) * ahd];
                        let dot = |k: &[f32]| k.iter().zip(qh).map(|(&k, &x)| k as f64 * x as f64).sum::<f64>() * 0.0625;
                        let s8: Vec<f64> = (0..=i).map(|j| dot(&kd[kvh * n + j])).collect();
                        let s: Vec<f64> = (0..=i).map(|j| dot(&ko[kvh * n + j])).collect();
                        let m = s8.iter().zip(&s).map(|(a, b)| (a - b).abs()).fold(0.0, f64::max);
                        let (p8, p) = (softmax(&s8), softmax(&s));
                        let o = (i * nq + h) * ahd;
                        for j in 0..=i {
                            for e in 0..ahd {
                                reference[o + e] += p8[j] * vd[kvh * n + j][e] as f64;
                                exact[o + e] += p[j] * vo[kvh * n + j][e] as f64;
                            }
                        }
                        for e in 0..ahd {
                            bound[o + e] = dv[e] + ((2.0 * m).exp() - 1.0) * (vmax[e] - vmin[e]) / 2.0;
                        }
                    }
                }
            }
            let err = |got: &[f32], want: &[f64]| got.iter().zip(want).map(|(&g, &w)| (g as f64 - w).abs()).fold(0.0, f64::max);
            let rows = n * nq * ahd;
            let last = &reference[(n - 1) * nq * ahd..];
            // prefill form, all n rows at pos_base 0: FA grid (NKV, ceil(T / 16), 1), block 32 * GQA; split grid (NQ, T, 1), block AHD
            let out = cuda::alloc_zeroed(rows * 4);
            launch_sync(module.get("attn_full_fa_q8"), nkv as u32, n.div_ceil(16) as u32, 1, (32 * gqa) as u32, &[qd, kq, vq, zero, t_p, tmax_p, mode_q8, out, 0]);
            let fa_pre = cuda::dtoh(out, rows);
            let outb = cuda::alloc_zeroed(rows * 4);
            launch_sync(module.get("attn_full_fa"), nkv as u32, n.div_ceil(16) as u32, 1, (32 * gqa) as u32, &[qd, kb, vb, zero, t_p, tmax_p, mode_bf16, outb, 0]);
            let bf_pre = cuda::dtoh(outb, rows);
            let outs = cuda::alloc_zeroed(rows * 4);
            launch_sync(module.get("attn_full_split_q8"), nq as u32, n as u32, 1, ahd as u32, &[qd, kq, vq, zero, tmax_p, mode_q8, outs, 0]);
            let sp_pre = cuda::dtoh(outs, rows);
            // decode form: the last row alone at position n - 1, split K + attn_merge (the gen.rs launches)
            let (pos_last, qlast) = (cuda::to_i32_dev(&[(n - 1) as i32]), qd + ((n - 1) * nq * ahd * 4) as u64);
            let merged = |name: &str, splits: usize, fa: bool| -> Vec<f32> {
                let (po, pml, o) = (cuda::alloc_zeroed(nq * splits * ahd * 4), cuda::alloc_zeroed(nq * splits * 2 * 4), cuda::alloc_zeroed(nq * ahd * 4));
                if fa {
                    launch_sync(module.get(name), nkv as u32, 1, splits as u32, (32 * gqa) as u32, &[qlast, kq, vq, pos_last, one, tmax_p, mode_q8, po, pml]);
                } else {
                    launch_sync(module.get(name), nq as u32, 1, splits as u32, ahd as u32, &[qlast, kq, vq, pos_last, tmax_p, mode_q8, po, pml]);
                }
                launch_sync(module.get("attn_merge"), nq as u32, 1, 1, ahd as u32, &[po, pml, o, cuda::to_i32_dev(&[splits as i32])]);
                cuda::dtoh(o, nq * ahd)
            };
            let fa_dec = merged("attn_full_fa_q8", crate::gen::FA_DECODE_SPLITS, true);
            let sp_dec = merged("attn_full_split_q8", 8, false);
            let (e_fa, e_sp, e_fad, e_spd) = (err(&fa_pre, &reference), err(&sp_pre, &reference), err(&fa_dec, last), err(&sp_dec, last));
            let stats = |a: &[f32], b: &[f64]| -> (f64, f64) {
                let d: Vec<f64> = a.iter().zip(b).map(|(&x, &y)| (x as f64 - y).abs()).collect();
                (d.iter().cloned().fold(0.0, f64::max), d.iter().sum::<f64>() / d.len() as f64)
            };
            let bf_pre64: Vec<f64> = bf_pre.iter().map(|&v| v as f64).collect();
            let ((d_max, d_mean), (q_max, q_mean), (b_max, b_mean)) = (stats(&fa_pre, &bf_pre64), stats(&fa_pre, &exact), stats(&bf_pre, &exact));
            // the worst output relative to its own bound (<= 1 passes)
            let (mut worst, mut at) = (0f64, 0usize);
            for (i, ((&g, &x), &bd)) in fa_pre.iter().zip(&exact).zip(&bound).enumerate() {
                let r = (g as f64 - x).abs() / (bd + 5e-3);
                if r > worst {
                    (worst, at) = (r, i);
                }
            }
            println!("q8 attention, {n} rows x {nq} heads, against the f64 reference over the stored q8 values: attn_full_fa_q8 prefill {e_fa:.3e}, decode ({} splits + attn_merge) {e_fad:.3e}; attn_full_split_q8 prefill {e_sp:.3e}, decode (8 splits) {e_spd:.3e}", crate::gen::FA_DECODE_SPLITS);
            println!("against the f64 attention over the f32 K / V: q8 path (attn_full_fa_q8) max {q_max:.3e} mean {q_mean:.3e}; bf16 path of record (store_kv mode 1 + attn_full_fa) max {b_max:.3e} mean {b_mean:.3e}; q8 against bf16 max {d_max:.3e} mean {d_mean:.3e}");
            println!("q8 format bound: worst output at {:.3} of its bound (row {}, head {}, dim {}: |err| {:.3e}, bound {:.3e} + 5e-3)", worst, at / (nq * ahd), at / ahd % nq, at % ahd, (fa_pre[at] as f64 - exact[at]).abs(), bound[at]);
            assert!(e_sp < 1e-4 && e_spd < 1e-4, "attn_full_split_q8 off by {e_sp:.3e} / {e_spd:.3e}");
            assert!(e_fa < 5e-3 && e_fad < 5e-3, "attn_full_fa_q8 off by {e_fa:.3e} / {e_fad:.3e}");
            assert!(worst <= 1.0, "q8 outside the format bound: {worst:.3} of it at output {at}");
            assert!(d_max > 0.0, "the q8 and bf16 caches cannot be told apart");
        }
    }
}
