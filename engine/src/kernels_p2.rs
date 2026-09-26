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
pub const P2_NAMES: &[&str] = &["attn_full_split", "silu_mul_n", "gemv_nvfp4_w", "gemv_nvfp4_gu", "add_rms_1k", "gemv_bf16_ba", "attn_full_fa"];

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
