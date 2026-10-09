// crow-nest #163 (GLM-5.3-Flash plan step 14, MLA/DSA part): glm5_next MLA over a BF16 latent cache
// and the DSA indexer with learned 4-token pooling. Its own NVRTC module (the #180 MUL1 pattern):
// `glm5_mla::MlaDims::prelude()` puts the `GM_*` shape defines in front of this text; no entry of
// `KERNEL_SRC` changes. The top-k selection itself is `qsa_select_fast` of `KERNEL_SRC` (unchanged:
// pools of 4 from position 0, ties to the lowest pool index, the tail tokens 4*ncb..pos appended).
//
// Math (docs/glm5-next-recipe.md sections 7 and 8, HF modeling_glm5_next.py:736-1256), per row at
// absolute position p, per head h:
//   q_resid = rms(q_a x) * w_qa                 q = q_b q_resid                  [H][NOPE]
//   c_p     = bf16(rms(kv_a x) * w_kva)          -> latent cache row p            [LAT]
//   k_p     = bf16(LN(wk x) * w + b), g_p = bf16(gate x), 1.0  -> indexer row p   [ID | ID | 1]
//   pool P  = rows 4P..4P+3, pk[c] = sum_j softmax_j(g_j[c] + ape[j][c]) * k_j[c]
//   score P = sum_ih (w_ih * IH^-0.5) * relu(iq_ih . pk * ID^-0.5),  iq = wq_b q_resid
//   sel     = top (TOPK / 4) pools by score among the (p+1)/4 complete pools + the tail tokens
//   q~_h    = W_k,h^T q_h  (W_k,h = kv_b rows h*(NOPE+V) .. +NOPE)                 [LAT]
//   u_h     = sum_{s in sel} softmax_s(q~_h . c_s * NOPE^-0.5) c_s                  [LAT]
//   o_h     = W_v,h u_h    (W_v,h = kv_b rows h*(NOPE+V)+NOPE .. +V)                [V]
// Every launch reads the call's (pos0, t) from the device word pair `st` (graph-safe, the
// d2d_block pattern); shapes that never change for a buffer are by-value `long long`.

__device__ __forceinline__ float gm_bf(unsigned short b) { return __uint_as_float(((unsigned int)b) << 16); }
// f32 -> BF16 round to nearest even (finite values only; every stored value here is finite)
__device__ __forceinline__ unsigned short gm_f2bf(float f) {
    unsigned int u = __float_as_uint(f);
    u += 0x7FFFu + ((u >> 16) & 1u);
    return (unsigned short)(u >> 16);
}
__device__ __forceinline__ float gm_warp_sum(float v) {
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    return v;
}
// block-wide sum over 256 threads; every thread gets the total
__device__ __forceinline__ float gm_block_sum(float v, float* red) {
    v = gm_warp_sum(v);
    int w = threadIdx.x >> 5, l = threadIdx.x & 31;
    __syncthreads();
    if (l == 0) red[w] = v;
    __syncthreads();
    float s = (l < (blockDim.x >> 5)) ? red[l] : 0.0f;
    if (w == 0) s = gm_warp_sum(s);
    if (threadIdx.x == 0) red[0] = s;
    __syncthreads();
    return red[0];
}

// y[tok][r] (row stride ldy) = sum_k W[r][k] * x[tok][k] (row stride ldx); W BF16 [rows][k], f32
// accumulation in ascending k, k % 32 == 0. grid (ceil(rows/64), ceil(t/16)), block 256.
extern "C" __global__ void gm_gemm(const unsigned short* __restrict__ w, const float* __restrict__ x,
                                   float* __restrict__ y, long long k, long long rows, long long ldx,
                                   long long ldy, const int* __restrict__ st) {
    int t = st[1];
    __shared__ float ws[32][65];
    __shared__ float xs[16][33];
    int r = threadIdx.x & 63, tg = threadIdx.x >> 6;
    long long row0 = (long long)blockIdx.x * 64, tok0 = (long long)blockIdx.y * 16;
    float acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    for (long long k0 = 0; k0 < k; k0 += 32) {
        for (int i = threadIdx.x; i < 64 * 32; i += 256) {
            int rr = i >> 5, kk = i & 31;
            long long row = row0 + rr;
            ws[kk][rr] = row < rows ? gm_bf(w[row * k + k0 + kk]) : 0.0f;
        }
        for (int i = threadIdx.x; i < 16 * 32; i += 256) {
            int tt = i >> 5, kk = i & 31;
            long long tok = tok0 + tt;
            xs[tt][kk] = tok < t ? x[tok * ldx + k0 + kk] : 0.0f;
        }
        __syncthreads();
        #pragma unroll 8
        for (int kk = 0; kk < 32; kk++) {
            float wv = ws[kk][r];
            #pragma unroll
            for (int j = 0; j < 4; j++) acc[j] += wv * xs[tg * 4 + j][kk];
        }
        __syncthreads();
    }
    long long row = row0 + r;
    if (row < rows)
        for (int j = 0; j < 4; j++) {
            long long tok = tok0 + tg * 4 + j;
            if (tok < t) y[tok * ldy + row] = acc[j];
        }
}

// weighted RMSNorm in place over rows of n values (HF Glm5NextTextRMSNorm: x * rsqrt(mean(x^2) + eps),
// then * w). grid t, block 256.
extern "C" __global__ void gm_rmsnorm(float* __restrict__ x, const float* __restrict__ w, long long n,
                                      const int* __restrict__ st) {
    __shared__ float red[32];
    float* xp = x + (long long)blockIdx.x * n;
    float ss = 0.0f;
    for (long long i = threadIdx.x; i < n; i += blockDim.x) ss += xp[i] * xp[i];
    float r = rsqrtf(gm_block_sum(ss, red) / (float)n + GM_EPS);
    for (long long i = threadIdx.x; i < n; i += blockDim.x) xp[i] = w[i] * (xp[i] * r);
}

// kv_a output [t][LAT] -> RMSNorm * w -> BF16 latent cache row pos0 + tok. grid t, block 256.
extern "C" __global__ void gm_latent_store(const float* __restrict__ kva, const float* __restrict__ w,
                                           unsigned short* __restrict__ lat, const int* __restrict__ st) {
    __shared__ float red[32];
    int tok = blockIdx.x;
    const float* xp = kva + (long long)tok * GM_LAT;
    float ss = 0.0f;
    for (int i = threadIdx.x; i < GM_LAT; i += blockDim.x) ss += xp[i] * xp[i];
    float r = rsqrtf(gm_block_sum(ss, red) / (float)GM_LAT + GM_EPS);
    unsigned short* out = lat + (long long)(st[0] + tok) * GM_LAT;
    for (int i = threadIdx.x; i < GM_LAT; i += blockDim.x) out[i] = gm_f2bf(w[i] * (xp[i] * r));
}

// indexer x-side projection [t][2 ID + IH] = [wk x | gate x | weights_proj x] -> the indexer cache row
// pos0 + tok, HF layout [LayerNorm(wk x) (w, b, eps 1e-6) | gate x | valid 1] in BF16 (GM_ROW values).
// grid t, block 256.
extern "C" __global__ void gm_idx_store(const float* __restrict__ ip, const float* __restrict__ nw,
                                        const float* __restrict__ nb, unsigned short* __restrict__ idx,
                                        const int* __restrict__ st) {
    __shared__ float red[32];
    int tok = blockIdx.x;
    const float* kp = ip + (long long)tok * (2 * GM_ID + GM_IH);
    float s = 0.0f;
    for (int c = threadIdx.x; c < GM_ID; c += blockDim.x) s += kp[c];
    float mean = gm_block_sum(s, red) / (float)GM_ID;
    float v = 0.0f;
    for (int c = threadIdx.x; c < GM_ID; c += blockDim.x) v += (kp[c] - mean) * (kp[c] - mean);
    float r = rsqrtf(gm_block_sum(v, red) / (float)GM_ID + GM_LN_EPS);
    unsigned short* out = idx + (long long)(st[0] + tok) * GM_ROW;
    for (int c = threadIdx.x; c < GM_ID; c += blockDim.x) {
        out[c] = gm_f2bf((kp[c] - mean) * r * nw[c] + nb[c]);
        out[GM_ID + c] = gm_f2bf(kp[GM_ID + c]);
    }
    if (threadIdx.x == 0) out[2 * GM_ID] = 0x3F80; // valid = 1.0
}

// pool scores: scores[tok][P] for every complete pool P < (pos+1)/4 of the query at pos = pos0 + tok.
// The pooled keys are recomputed from the indexer rows (HF recomputes them every call, :899-972):
// a block pools 32 pools once into shared memory and scores them for up to 16 queries.
// iq [t][IH][ID], ip [t][2 ID + IH] (weights_proj at column 2 ID), ape [4][ID] f32.
// grid (ceil(ncb_max / 32), ceil(t / 16)), block 256.
extern "C" __global__ void gm_idx_scores(const float* __restrict__ iq, const float* __restrict__ ip,
                                         const unsigned short* __restrict__ idx, const float* __restrict__ ape,
                                         float* __restrict__ scores, long long cap, const int* __restrict__ st) {
    __shared__ float pk[32][GM_ID + 1];
    __shared__ float qs[GM_IH * GM_ID];
    __shared__ float wsh[GM_IH];
    int pos0 = st[0], t = st[1];
    int p0 = blockIdx.x * 32;
    int tq0 = blockIdx.y * 16;
    int tq1 = min(tq0 + 16, t);
    int ncb_hi = (pos0 + tq1) / 4; // complete pools of the last query of this tile
    for (int i = threadIdx.x; i < 32 * GM_ID; i += blockDim.x) {
        int pp = i / GM_ID, c = i % GM_ID;
        int P = p0 + pp;
        float val = 0.0f;
        if (P < ncb_hi) {
            const unsigned short* rp = idx + (long long)P * 4 * GM_ROW;
            float g[4], m = -3.0e38f;
            #pragma unroll
            for (int j = 0; j < 4; j++) {
                g[j] = gm_bf(rp[j * GM_ROW + GM_ID + c]) + ape[j * GM_ID + c];
                m = fmaxf(m, g[j]);
            }
            float e[4], s = 0.0f;
            #pragma unroll
            for (int j = 0; j < 4; j++) {
                e[j] = expf(g[j] - m);
                s += e[j];
            }
            #pragma unroll
            for (int j = 0; j < 4; j++) val += (e[j] / s) * gm_bf(rp[j * GM_ROW + c]);
        }
        pk[pp][c] = val;
    }
    int pp = threadIdx.x >> 3, hg = threadIdx.x & 7;
    for (int tq = tq0; tq < tq1; tq++) {
        int ncb = (pos0 + tq + 1) / 4;
        __syncthreads();
        for (int i = threadIdx.x; i < GM_IH * GM_ID; i += blockDim.x) qs[i] = iq[(long long)tq * GM_IH * GM_ID + i];
        for (int i = threadIdx.x; i < GM_IH; i += blockDim.x)
            wsh[i] = ip[(long long)tq * (2 * GM_ID + GM_IH) + 2 * GM_ID + i] * GM_WSCALE;
        __syncthreads();
        if (p0 >= ncb) continue; // uniform over the block
        float part = 0.0f;
        for (int h = hg; h < GM_IH; h += 8) {
            float d = 0.0f;
            for (int c = 0; c < GM_ID; c++) d += qs[h * GM_ID + c] * pk[pp][c];
            part += wsh[h] * fmaxf(d * GM_ISCALE, 0.0f);
        }
        part += __shfl_xor_sync(0xffffffffu, part, 4);
        part += __shfl_xor_sync(0xffffffffu, part, 2);
        part += __shfl_xor_sync(0xffffffffu, part, 1);
        int P = p0 + pp;
        if (hg == 0 && P < ncb) scores[(long long)tq * cap + P] = part;
    }
}

// the per-query inputs of qsa_select_fast: complete pools and position. grid ceil(t / 256), block 256.
extern "C" __global__ void gm_sel_prep(int* __restrict__ ncb, int* __restrict__ pos, const int* __restrict__ st) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= st[1]) return;
    int p = st[0] + i;
    ncb[i] = (p + 1) / 4;
    pos[i] = p;
}

// q~[tok][h][j] = sum_i q[tok][h][i] * kv_b[h (NOPE + V) + i][j]. grid (H, ceil(t / 8)), block 256.
#define GM_JPT ((GM_LAT + 255) / 256)
extern "C" __global__ void gm_absorb(const float* __restrict__ q, const unsigned short* __restrict__ kvb,
                                     float* __restrict__ qt, const int* __restrict__ st) {
    __shared__ float qsh[8][GM_NOPE];
    int t = st[1];
    int h = blockIdx.x, tok0 = blockIdx.y * 8;
    for (int i = threadIdx.x; i < 8 * GM_NOPE; i += blockDim.x) {
        int tt = i / GM_NOPE, d = i % GM_NOPE;
        qsh[tt][d] = (tok0 + tt < t) ? q[((long long)(tok0 + tt) * GM_HEADS + h) * GM_NOPE + d] : 0.0f;
    }
    __syncthreads();
    float acc[8][GM_JPT];
    #pragma unroll
    for (int a = 0; a < 8; a++)
        #pragma unroll
        for (int b = 0; b < GM_JPT; b++) acc[a][b] = 0.0f;
    const unsigned short* wb = kvb + (long long)h * (GM_NOPE + GM_V) * GM_LAT;
    for (int i = 0; i < GM_NOPE; i++) {
        #pragma unroll
        for (int b = 0; b < GM_JPT; b++) {
            int j = threadIdx.x + 256 * b;
            if (j < GM_LAT) {
                float wv = gm_bf(wb[(long long)i * GM_LAT + j]);
                #pragma unroll
                for (int a = 0; a < 8; a++) acc[a][b] += qsh[a][i] * wv;
            }
        }
    }
    #pragma unroll
    for (int b = 0; b < GM_JPT; b++) {
        int j = threadIdx.x + 256 * b;
        if (j < GM_LAT)
            for (int a = 0; a < 8; a++)
                if (tok0 + a < t) qt[((long long)(tok0 + a) * GM_HEADS + h) * GM_LAT + j] = acc[a][b];
    }
}

// split-K attention over the selected latent rows, one warp per head: an online softmax over this
// split's share of the selection list, partials (acc[LAT], m, l) per (tok, split, head).
// grid (ceil(H / 8), t, nsplit), block 256.
#define GM_LPL (GM_LAT / 32)
extern "C" __global__ void gm_attn(const float* __restrict__ qt, const unsigned short* __restrict__ lat,
                                   const int* __restrict__ sel, const int* __restrict__ sel_n,
                                   float* __restrict__ part_o, float* __restrict__ part_ml, long long nsplit,
                                   const int* __restrict__ st) {
    int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    int h = blockIdx.x * 8 + warp;
    if (h >= GM_HEADS) return;
    int tok = blockIdx.y, sp = blockIdx.z;
    int j0 = lane * GM_LPL;
    float qv[GM_LPL], acc[GM_LPL];
    const float* qp = qt + ((long long)tok * GM_HEADS + h) * GM_LAT + j0;
    #pragma unroll
    for (int i = 0; i < GM_LPL; i++) {
        qv[i] = qp[i];
        acc[i] = 0.0f;
    }
    int n = sel_n[tok];
    int chunk = (n + (int)nsplit - 1) / (int)nsplit;
    int s0 = sp * chunk, s1 = min(n, s0 + chunk);
    float m = -__int_as_float(0x7f800000), l = 0.0f;
    const int* list = sel + (long long)tok * GM_SEL_MAX;
    for (int e = s0; e < s1; e++) {
        const unsigned short* cp = lat + (long long)list[e] * GM_LAT + j0;
        float c[GM_LPL], d = 0.0f;
        #pragma unroll
        for (int i = 0; i < GM_LPL; i++) {
            c[i] = gm_bf(cp[i]);
            d += qv[i] * c[i];
        }
        float s = gm_warp_sum(d) * GM_SCALE;
        float mn = fmaxf(m, s);
        float corr = expf(m - mn), p = expf(s - mn);
        l = l * corr + p;
        #pragma unroll
        for (int i = 0; i < GM_LPL; i++) acc[i] = acc[i] * corr + p * c[i];
        m = mn;
    }
    long long slot = ((long long)tok * nsplit + sp) * GM_HEADS + h;
    #pragma unroll
    for (int i = 0; i < GM_LPL; i++) part_o[slot * GM_LAT + j0 + i] = acc[i];
    if (lane == 0) {
        part_ml[slot * 2] = m;
        part_ml[slot * 2 + 1] = l;
    }
}

// u[tok][h][j] = sum_s exp(m_s - M) acc_s[j] / sum_s exp(m_s - M) l_s. grid (H, t), block 256.
extern "C" __global__ void gm_attn_merge(const float* __restrict__ part_o, const float* __restrict__ part_ml,
                                         float* __restrict__ u, long long nsplit, const int* __restrict__ st) {
    int h = blockIdx.x, tok = blockIdx.y;
    long long base = (long long)tok * nsplit * GM_HEADS + h;
    float M = -__int_as_float(0x7f800000);
    for (int s = 0; s < (int)nsplit; s++) M = fmaxf(M, part_ml[(base + (long long)s * GM_HEADS) * 2]);
    float L = 0.0f;
    for (int s = 0; s < (int)nsplit; s++) {
        long long sl = base + (long long)s * GM_HEADS;
        float ls = part_ml[sl * 2 + 1];
        if (ls > 0.0f) L += expf(part_ml[sl * 2] - M) * ls;
    }
    for (int j = threadIdx.x; j < GM_LAT; j += blockDim.x) {
        float a = 0.0f;
        for (int s = 0; s < (int)nsplit; s++) {
            long long sl = base + (long long)s * GM_HEADS;
            if (part_ml[sl * 2 + 1] > 0.0f) a += expf(part_ml[sl * 2] - M) * part_o[sl * GM_LAT + j];
        }
        u[((long long)tok * GM_HEADS + h) * GM_LAT + j] = a / L;
    }
}

// o[tok][h][i] = sum_j kv_b[h (NOPE + V) + NOPE + i][j] * u[tok][h][j]. grid (H, ceil(t / 8)), block 256:
// warp w takes rows i = w, w + 8, ...; a lane reads GM_LPL consecutive weights of the row.
extern "C" __global__ void gm_out_v(const float* __restrict__ u, const unsigned short* __restrict__ kvb,
                                    float* __restrict__ o, const int* __restrict__ st) {
    __shared__ float us[8][GM_LAT];
    int t = st[1];
    int h = blockIdx.x, tok0 = blockIdx.y * 8;
    for (int i = threadIdx.x; i < 8 * GM_LAT; i += blockDim.x) {
        int tt = i / GM_LAT, j = i % GM_LAT;
        us[tt][j] = (tok0 + tt < t) ? u[((long long)(tok0 + tt) * GM_HEADS + h) * GM_LAT + j] : 0.0f;
    }
    __syncthreads();
    int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    int j0 = lane * GM_LPL;
    const unsigned short* wb = kvb + ((long long)h * (GM_NOPE + GM_V) + GM_NOPE) * GM_LAT;
    for (int i = warp; i < GM_V; i += 8) {
        const unsigned short* wr = wb + (long long)i * GM_LAT + j0;
        float wv[GM_LPL];
        #pragma unroll
        for (int jj = 0; jj < GM_LPL; jj++) wv[jj] = gm_bf(wr[jj]);
        for (int a = 0; a < 8; a++) {
            float d = 0.0f;
            #pragma unroll
            for (int jj = 0; jj < GM_LPL; jj++) d += wv[jj] * us[a][j0 + jj];
            d = gm_warp_sum(d);
            if (lane == 0 && tok0 + a < t) o[((long long)(tok0 + a) * GM_HEADS + h) * GM_V + i] = d;
        }
    }
}

// ---------------- the decode path (follow-up of #191) ----------------
// gm_gemm, gm_absorb and gm_out_v above are tiled for prompt calls (16 / 8 rows per block); at one
// row they run on 5 .. 64 blocks and read their BF16 weights at 5 .. 220 GB/s (Nsight 2026-10-09:
// ~8.5 ms of a decode row). The three kernels below take calls of at most GM_DT rows (the host
// routes t <= GM_DT here, larger calls to the tiled kernels) and keep every output's summation
// chain of the kernel they replace, expression for expression, so each output is bit-identical:
//   gm_gemv     = gm_gemm:   acc += w[k] * x[k], k ascending from 0.0f (one thread per output)
//   gm_absorb1  = gm_absorb: acc += q[i] * W[i][j], i ascending from 0.0f (one thread per output)
//   gm_out_v1   = gm_out_v:  lane l: d += W[i][l LPL + jj] * u[l LPL + jj], jj ascending from
//                            0.0f, then gm_warp_sum (xor 16 .. 1)
// What changes is how the weights reach the chains: many more blocks, 16-byte loads, many loads
// in flight per thread.
#define GM_DT 4       // the most rows one decode-kernel call takes (host: glm5_mla::DECODE_T)
#define GM_GV_R 8     // output rows per gm_gemv block (host: glm5_mla::GEMV_ROWS)
#define GM_GV_KC 1024 // k values per gm_gemv stage

// y[tok][r] (row stride ldy) = sum_k W[r][k] * x[tok][k] (row stride ldx) for t <= GM_DT rows.
// Thread r + GM_GV_R * tok owns output (row0 + r, tok). The block's GM_GV_R weight rows and the
// call's x rows pass through shared memory in stages of GM_GV_KC values (16-byte loads, the next
// stage in registers while the current one is summed). grid ceil(rows / GM_GV_R), block 128.
// w, x 16-byte aligned, ldx % 4 == 0, k % 32 == 0 (host-checked).
extern "C" __global__ void __launch_bounds__(128) gm_gemv(const unsigned short* __restrict__ w, const float* __restrict__ x,
                                                          float* __restrict__ y, long long k, long long rows, long long ldx,
                                                          long long ldy, const int* __restrict__ st) {
    __shared__ __align__(16) unsigned short ws[GM_GV_R][GM_GV_KC + 8];
    __shared__ __align__(16) float xs[GM_DT][GM_GV_KC + 4];
    const int t = st[1];
    const int tid = threadIdx.x;
    const long long row0 = (long long)blockIdx.x * GM_GV_R;
    constexpr int WQ = GM_GV_KC / 8;  // uint4 per weight row of a stage
    constexpr int XQ = GM_GV_KC / 4;  // float4 per x row of a stage
    constexpr int WN = GM_GV_R * WQ / 128;
    constexpr int XN = GM_DT * XQ / 128;
    uint4 wr[WN];
    float4 xr[XN];
    auto fetch = [&](long long k0) {
        const long long kc = min((long long)GM_GV_KC, k - k0);
#pragma unroll
        for (int n = 0; n < WN; n++) {
            const int q = tid + 128 * n, r = q / WQ, c = q % WQ;
            const long long row = min(row0 + r, rows - 1);  // a tail row re-reads the last row, never stored
            wr[n] = c * 8 < kc ? *(const uint4*)(w + row * k + k0 + c * 8) : make_uint4(0, 0, 0, 0);
        }
#pragma unroll
        for (int n = 0; n < XN; n++) {
            const int q = tid + 128 * n, tt = q / XQ, c = q % XQ;
            xr[n] = (tt < t && c * 4 < kc) ? *(const float4*)(x + tt * ldx + k0 + c * 4) : make_float4(0.f, 0.f, 0.f, 0.f);
        }
    };
    const int r = tid % GM_GV_R, tok = tid / GM_GV_R;
    float acc = 0.0f;
    fetch(0);
    for (long long k0 = 0; k0 < k; k0 += GM_GV_KC) {
        __syncthreads();  // the previous stage is summed
#pragma unroll
        for (int n = 0; n < WN; n++) {
            const int q = tid + 128 * n;
            *(uint4*)&ws[q / WQ][(q % WQ) * 8] = wr[n];
        }
#pragma unroll
        for (int n = 0; n < XN; n++) {
            const int q = tid + 128 * n;
            *(float4*)&xs[q / XQ][(q % XQ) * 4] = xr[n];
        }
        __syncthreads();
        if (k0 + GM_GV_KC < k) fetch(k0 + GM_GV_KC);
        if (tok < t) {
            const int kc8 = (int)(min((long long)GM_GV_KC, k - k0) / 8);
            const uint4* wp = (const uint4*)&ws[r][0];
            const float4* xp = (const float4*)&xs[tok][0];
#pragma unroll 4
            for (int c = 0; c < kc8; c++) {
                const uint4 wv = wp[c];
                const float4 xa = xp[2 * c], xb = xp[2 * c + 1];
                acc += __uint_as_float(wv.x << 16) * xa.x;
                acc += __uint_as_float(wv.x & 0xFFFF0000u) * xa.y;
                acc += __uint_as_float(wv.y << 16) * xa.z;
                acc += __uint_as_float(wv.y & 0xFFFF0000u) * xa.w;
                acc += __uint_as_float(wv.z << 16) * xb.x;
                acc += __uint_as_float(wv.z & 0xFFFF0000u) * xb.y;
                acc += __uint_as_float(wv.w << 16) * xb.z;
                acc += __uint_as_float(wv.w & 0xFFFF0000u) * xb.w;
            }
        }
    }
    const long long row = row0 + r;
    if (tok < t && row < rows) y[tok * ldy + row] = acc;
}

// q~[tok][h][j] = sum_i q[tok][h][i] * kv_b[h (NOPE + V) + i][j] for t <= GM_DT rows. A thread owns
// the columns j, j + 1 of head h (one 32-bit load per i). grid (H, ceil(LAT / 128)), block 64.
extern "C" __global__ void __launch_bounds__(64) gm_absorb1(const float* __restrict__ q, const unsigned short* __restrict__ kvb,
                                                            float* __restrict__ qt, const int* __restrict__ st) {
    __shared__ float qsh[GM_DT][GM_NOPE];
    const int t = st[1];
    const int h = blockIdx.x;
    for (int i = threadIdx.x; i < GM_DT * GM_NOPE; i += blockDim.x) {
        const int tt = i / GM_NOPE, d = i % GM_NOPE;
        qsh[tt][d] = tt < t ? q[((long long)tt * GM_HEADS + h) * GM_NOPE + d] : 0.0f;
    }
    __syncthreads();
    const int j = 2 * (blockIdx.y * 64 + threadIdx.x);
    if (j >= GM_LAT) return;
    const unsigned int* wb = (const unsigned int*)(kvb + (long long)h * (GM_NOPE + GM_V) * GM_LAT + j);
    float acc[GM_DT][2];
#pragma unroll
    for (int a = 0; a < GM_DT; a++) acc[a][0] = acc[a][1] = 0.0f;
#pragma unroll 32
    for (int i = 0; i < GM_NOPE; i++) {
        const unsigned int wv = __ldg(wb + (long long)i * (GM_LAT / 2));
        const float w0 = __uint_as_float(wv << 16), w1 = __uint_as_float(wv & 0xFFFF0000u);
#pragma unroll
        for (int a = 0; a < GM_DT; a++) {
            acc[a][0] += qsh[a][i] * w0;
            acc[a][1] += qsh[a][i] * w1;
        }
    }
#pragma unroll
    for (int a = 0; a < GM_DT; a++)
        if (a < t) {
            float* op = qt + ((long long)a * GM_HEADS + h) * GM_LAT + j;
            op[0] = acc[a][0];
            op[1] = acc[a][1];
        }
}

// o[tok][h][i] = sum_j kv_b[h (NOPE + V) + NOPE + i][j] * u[tok][h][j] for t <= GM_DT rows with
// gm_out_v's lane split. Warp w takes the rows i = blockIdx.y * 32 + GM_OV_RW w .. + GM_OV_RW - 1,
// their weights loaded up front. grid (H, ceil(V / 32)), block 256. kv_b 16-byte aligned when
// GM_LPL % 8 == 0 (host-checked).
#define GM_OV_RW 4
extern "C" __global__ void __launch_bounds__(256) gm_out_v1(const float* __restrict__ u, const unsigned short* __restrict__ kvb,
                                                             float* __restrict__ o, const int* __restrict__ st) {
    const int t = st[1];
    const int h = blockIdx.x;
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int i0 = blockIdx.y * 32 + warp * GM_OV_RW;
    const int j0 = lane * GM_LPL;
    const unsigned short* wb = kvb + ((long long)h * (GM_NOPE + GM_V) + GM_NOPE) * GM_LAT + j0;
    float wv[GM_OV_RW][GM_LPL];
#pragma unroll
    for (int rr = 0; rr < GM_OV_RW; rr++) {
        const unsigned short* wr = wb + (long long)min(i0 + rr, GM_V - 1) * GM_LAT;
#if (GM_LPL % 8) == 0
#pragma unroll
        for (int c = 0; c < GM_LPL / 8; c++) {
            const uint4 v = __ldg((const uint4*)wr + c);
            wv[rr][8 * c] = __uint_as_float(v.x << 16);
            wv[rr][8 * c + 1] = __uint_as_float(v.x & 0xFFFF0000u);
            wv[rr][8 * c + 2] = __uint_as_float(v.y << 16);
            wv[rr][8 * c + 3] = __uint_as_float(v.y & 0xFFFF0000u);
            wv[rr][8 * c + 4] = __uint_as_float(v.z << 16);
            wv[rr][8 * c + 5] = __uint_as_float(v.z & 0xFFFF0000u);
            wv[rr][8 * c + 6] = __uint_as_float(v.w << 16);
            wv[rr][8 * c + 7] = __uint_as_float(v.w & 0xFFFF0000u);
        }
#else
#pragma unroll
        for (int jj = 0; jj < GM_LPL; jj++) wv[rr][jj] = gm_bf(wr[jj]);
#endif
    }
    for (int a = 0; a < t; a++) {
        const float* up = u + ((long long)a * GM_HEADS + h) * GM_LAT + j0;
        float uv[GM_LPL];
#pragma unroll
        for (int jj = 0; jj < GM_LPL; jj++) uv[jj] = up[jj];
#pragma unroll
        for (int rr = 0; rr < GM_OV_RW; rr++) {
            float d = 0.0f;
#pragma unroll
            for (int jj = 0; jj < GM_LPL; jj++) d += wv[rr][jj] * uv[jj];
            d = gm_warp_sum(d);
            if (lane == 0 && i0 + rr < GM_V) o[((long long)a * GM_HEADS + h) * GM_V + i0 + rr] = d;
        }
    }
}
