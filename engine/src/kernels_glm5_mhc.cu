// crow-nest #161: GLM-5.3-Flash mHC residual (manifold-constrained hyper-connections), two per
// decoder layer (attn_hc, ffn_hc). NVRTC, compute_120a; its own module (`glm5_mhc::Kernels::new`
// compiles `kernels::GLM5_MHC_SRC` alone): one more `.entry` in KERNEL_SRC would break the PTX of
// record (`kernels::tests_300_c4`). The CPU twin is `glm5_mhc.rs`; the math is HF's
// `Glm5NextTextHyperConnection.forward` (transformers 5.16.1, modeling_glm5_next.py:267-295) and the
// decoder layer's stream mix (:1316-1318), written out in docs/glm5-next-recipe.md section 5 and
// docs/glm5-mhc.md.
//
//   glm5_mhc_coeffs  grid (T), block 256, one row per block. x [T][4][H] f32 (the 4 streams),
//                    fn [24][4H] bf16, base [24] f32, scale [3] f32 ->
//                    logits [T][24], pre [T][4], post [T][4], comb [T][4][4] (row j = source
//                    stream, column i = destination stream), collapsed [T][H] = sum_s pre[s] x[s].
//   glm5_mhc_mix     #191, grid (24, T), block 256: what glm5_mhc_coeffs computes, bit for bit,
//                    over 24 blocks per row (the record's one block per row left 169 of 170 SMs
//                    idle at T = 1). Block m: the RMS factor (the record's chain, in every block)
//                    and mix row m -> logits; the last block of the row: pre, post, comb (the
//                    record's steps 3-5 on 16 lanes of warp 0, same operations in the same order)
//                    and collapsed (warps 1-7). done [T] u32 is a zeroed per-row counter the last
//                    block resets.
//   glm5_mhc_mix_norm CROW_GLM_HCFUSE: glm5_mhc_mix plus the sublayer RMSNorm (gm_rmsnorm of
//                    kernels_glm5_mla.cu, weight nw [H] f32) over collapsed in place, by the last
//                    block of the row: 2 launches per site (mix_norm, expand) instead of 3.
//   glm5_mhc_coeffs  the record (kept for the #191 bit-identity test; no longer launched).
//   glm5_mhc_expand  grid (ceil(H / 256), T), block 256. out[t][i][d] = post[i] y[t][d]
//                    + sum_j comb[j][i] x[t][j][d]. out may be x itself (one thread reads the four
//                    streams of its d before it writes them).
//
// Every product and sum is f32, in HF's order: normalise, then dot (r * x first, then fn . (r x));
// the coefficient tail runs on one thread with the HF expression order. Every rounded add,
// multiply and divide is an __f*_rn intrinsic, which NVRTC never contracts into an fma
// (-fmad=true is its default); expf is the accurate one (no --use_fast_math). The dot products
// and the sum of squares are block reductions, a different summation order than torch's (the
// goldens bound that, docs/glm5-mhc.md). Scalar arguments come in one device int buffer per
// launch (the KERNEL_SRC rule): prm[0] = H.

#define MHC_HC 4
#define MHC_MIX 24
#define MHC_THREADS 256
#define MHC_HC_EPS 1e-6f
#define MHC_RMS_EPS 1e-5f
#define MHC_SINKHORN_ITERS 20

__device__ __forceinline__ float mhc_bf16(unsigned short h) {
    return __uint_as_float(((unsigned int)h) << 16);
}

__device__ __forceinline__ float mhc_sigmoid(float v) {
    return __fdiv_rn(1.0f, __fadd_rn(1.0f, expf(-v)));
}

// the block sum of one value per thread; every thread gets the total
__device__ __forceinline__ float mhc_block_sum(float v, float* sh) {
    for (int o = 16; o > 0; o >>= 1) v = __fadd_rn(v, __shfl_xor_sync(0xffffffffu, v, o));
    const int w = threadIdx.x >> 5;
    if ((threadIdx.x & 31) == 0) sh[w] = v;
    __syncthreads();
    float s = 0.0f;
    for (int i = 0; i < MHC_THREADS / 32; i++) s = __fadd_rn(s, sh[i]);
    __syncthreads();
    return s;
}

// #191: steps 3-8 of one row t (pre, post, comb) from its 24 logits m on warp 0 of the last
// glm5_mhc_mix block of the row, bit-identical to the record's thread-0 tail in glm5_mhc_coeffs: every coefficient gets
// the record's operations in the record's order, only spread over lanes. Lanes 0-3 take pre[s]
// and post[s] (s = lane). Lane q = 4 j + i (lanes 16-31 mirror 0-15) holds comb[j][i]; a row or
// column sum gathers the four values by __shfl_sync and adds them from 0.0f in the record's
// index order, so each lane computes the very sum the record computes once; the softmax max is
// the record's sequential fmaxf from -inf. sh_pre gets pre; `store` false (warp-uniform) derives
// pre only.
__device__ __forceinline__ void mhc_coeff_tail_warp(const float* m, const float* __restrict__ base,
                                                    const float* __restrict__ scale, float* sh_pre,
                                                    float* __restrict__ pre, float* __restrict__ post,
                                                    float* __restrict__ comb, const int t, const bool store) {
    const unsigned int full = 0xffffffffu;
    const int lane = threadIdx.x & 31;
    const float s0 = scale[0], s1 = scale[1], s2 = scale[2];
    if (lane < MHC_HC) {
        const float p = __fadd_rn(mhc_sigmoid(__fadd_rn(__fmul_rn(m[lane], s0), base[lane])), MHC_HC_EPS);
        sh_pre[lane] = p;
        if (store) {
            pre[t * MHC_HC + lane] = p;
            post[t * MHC_HC + lane] = __fmul_rn(2.0f, mhc_sigmoid(__fadd_rn(__fmul_rn(m[MHC_HC + lane], s1), base[MHC_HC + lane])));
        }
    }
    if (!store) return;
    const int q = lane & 15;
    const int i = q & 3;
    const int row0 = lane & ~3;          // the lanes of row j: row0 + u
    const int col0 = (lane & 16) + i;    // the lanes of column i: col0 + 4 u
    // softmax of row j
    const int qq = 2 * MHC_HC + q;
    float l = __fadd_rn(__fmul_rn(m[qq], s2), base[qq]);
    float g[MHC_HC];
#pragma unroll
    for (int u = 0; u < MHC_HC; u++) g[u] = __shfl_sync(full, l, row0 + u);
    float mx = __int_as_float(0xff800000);  // -inf
#pragma unroll
    for (int u = 0; u < MHC_HC; u++) mx = fmaxf(mx, g[u]);
    l = expf(__fadd_rn(l, -mx));
#pragma unroll
    for (int u = 0; u < MHC_HC; u++) g[u] = __shfl_sync(full, l, row0 + u);
    float sum = 0.0f;
#pragma unroll
    for (int u = 0; u < MHC_HC; u++) sum = __fadd_rn(sum, g[u]);
    float c = __fadd_rn(__fdiv_rn(l, sum), MHC_HC_EPS);
    for (int it = 0; it < MHC_SINKHORN_ITERS; it++) {
        if (it > 0) {  // rows: sum over the destination i
#pragma unroll
            for (int u = 0; u < MHC_HC; u++) g[u] = __shfl_sync(full, c, row0 + u);
            float s = 0.0f;
#pragma unroll
            for (int u = 0; u < MHC_HC; u++) s = __fadd_rn(s, g[u]);
            s = __fadd_rn(s, MHC_HC_EPS);
            c = __fdiv_rn(c, s);
        }
        // columns: sum over the source j
#pragma unroll
        for (int u = 0; u < MHC_HC; u++) g[u] = __shfl_sync(full, c, col0 + 4 * u);
        float s = 0.0f;
#pragma unroll
        for (int u = 0; u < MHC_HC; u++) s = __fadd_rn(s, g[u]);
        s = __fadd_rn(s, MHC_HC_EPS);
        c = __fdiv_rn(c, s);
    }
    if (lane < MHC_HC * MHC_HC) comb[(size_t)t * MHC_HC * MHC_HC + q] = c;
}

extern "C" __global__ void glm5_mhc_coeffs(const float* __restrict__ x, const unsigned short* __restrict__ fn,
                                           const float* __restrict__ base, const float* __restrict__ scale,
                                           float* __restrict__ logits, float* __restrict__ pre,
                                           float* __restrict__ post, float* __restrict__ comb,
                                           float* __restrict__ collapsed, const int* __restrict__ prm) {
    __shared__ float sh_red[MHC_THREADS / 32];
    __shared__ float sh_part[MHC_THREADS / 32][MHC_MIX];
    __shared__ float sh_m[MHC_MIX];
    __shared__ float sh_pre[MHC_HC];
    const int H = prm[0];
    const int n = MHC_HC * H;
    const int t = blockIdx.x;
    const float* xr = x + (size_t)t * n;

    // 1. unweighted RMSNorm over the 4H values (eps = rms_norm_eps, not hc_eps)
    float ss = 0.0f;
    for (int k = threadIdx.x; k < n; k += MHC_THREADS) ss = fmaf(xr[k], xr[k], ss);
    ss = mhc_block_sum(ss, sh_red);
    const float r = __frsqrt_rn(__fadd_rn(__fdiv_rn(ss, (float)n), MHC_RMS_EPS));

    // 2. m = fn . (r x), 24 dot products
    float acc[MHC_MIX];
#pragma unroll
    for (int m = 0; m < MHC_MIX; m++) acc[m] = 0.0f;
    for (int k = threadIdx.x; k < n; k += MHC_THREADS) {
        const float xn = __fmul_rn(xr[k], r);
#pragma unroll
        for (int m = 0; m < MHC_MIX; m++) acc[m] = fmaf(mhc_bf16(fn[(size_t)m * n + k]), xn, acc[m]);
    }
    const int w = threadIdx.x >> 5;
#pragma unroll
    for (int m = 0; m < MHC_MIX; m++) {
        float v = acc[m];
        for (int o = 16; o > 0; o >>= 1) v = __fadd_rn(v, __shfl_xor_sync(0xffffffffu, v, o));
        if ((threadIdx.x & 31) == 0) sh_part[w][m] = v;
    }
    __syncthreads();
    if (threadIdx.x < MHC_MIX) {
        float s = 0.0f;
        for (int i = 0; i < MHC_THREADS / 32; i++) s = __fadd_rn(s, sh_part[i][threadIdx.x]);
        sh_m[threadIdx.x] = s;
        logits[(size_t)t * MHC_MIX + threadIdx.x] = s;
    }
    __syncthreads();

    // 3.-8. pre, post, comb (softmax, column normalise, 19 row + column rounds), one thread
    if (threadIdx.x == 0) {
        const float s0 = scale[0], s1 = scale[1], s2 = scale[2];
        for (int s = 0; s < MHC_HC; s++) {
            const float p = __fadd_rn(mhc_sigmoid(__fadd_rn(__fmul_rn(sh_m[s], s0), base[s])), MHC_HC_EPS);
            sh_pre[s] = p;
            pre[t * MHC_HC + s] = p;
            post[t * MHC_HC + s] = __fmul_rn(2.0f, mhc_sigmoid(__fadd_rn(__fmul_rn(sh_m[MHC_HC + s], s1), base[MHC_HC + s])));
        }
        float c[MHC_HC][MHC_HC];
        for (int j = 0; j < MHC_HC; j++) {
            float l[MHC_HC];
            float mx = __int_as_float(0xff800000);  // -inf
            for (int i = 0; i < MHC_HC; i++) {
                const int q = 2 * MHC_HC + j * MHC_HC + i;
                l[i] = __fadd_rn(__fmul_rn(sh_m[q], s2), base[q]);
                mx = fmaxf(mx, l[i]);
            }
            float sum = 0.0f;
            for (int i = 0; i < MHC_HC; i++) {
                l[i] = expf(__fadd_rn(l[i], -mx));
                sum = __fadd_rn(sum, l[i]);
            }
            for (int i = 0; i < MHC_HC; i++) c[j][i] = __fadd_rn(__fdiv_rn(l[i], sum), MHC_HC_EPS);
        }
        for (int it = 0; it < MHC_SINKHORN_ITERS; it++) {
            if (it > 0) {  // rows: sum over the destination i
                for (int j = 0; j < MHC_HC; j++) {
                    float s = 0.0f;
                    for (int i = 0; i < MHC_HC; i++) s = __fadd_rn(s, c[j][i]);
                    s = __fadd_rn(s, MHC_HC_EPS);
                    for (int i = 0; i < MHC_HC; i++) c[j][i] = __fdiv_rn(c[j][i], s);
                }
            }
            for (int i = 0; i < MHC_HC; i++) {  // columns: sum over the source j
                float s = 0.0f;
                for (int j = 0; j < MHC_HC; j++) s = __fadd_rn(s, c[j][i]);
                s = __fadd_rn(s, MHC_HC_EPS);
                for (int j = 0; j < MHC_HC; j++) c[j][i] = __fdiv_rn(c[j][i], s);
            }
        }
        for (int j = 0; j < MHC_HC; j++)
            for (int i = 0; i < MHC_HC; i++) comb[(size_t)t * MHC_HC * MHC_HC + j * MHC_HC + i] = c[j][i];
    }
    __syncthreads();

    // 9. collapsed = sum_s pre[s] x[s], f32
    const float p0 = sh_pre[0], p1 = sh_pre[1], p2 = sh_pre[2], p3 = sh_pre[3];
    for (int d = threadIdx.x; d < H; d += MHC_THREADS) {
        float v = __fmul_rn(p0, xr[d]);
        v = __fadd_rn(v, __fmul_rn(p1, xr[H + d]));
        v = __fadd_rn(v, __fmul_rn(p2, xr[2 * H + d]));
        v = __fadd_rn(v, __fmul_rn(p3, xr[3 * H + d]));
        collapsed[(size_t)t * H + d] = v;
    }
}

extern "C" __global__ void glm5_mhc_expand(const float* x, const float* __restrict__ y,
                                           const float* __restrict__ post, const float* __restrict__ comb,
                                           float* out, const int* __restrict__ prm) {
    const int H = prm[0];
    const int t = blockIdx.y;
    const int d = blockIdx.x * MHC_THREADS + threadIdx.x;
    if (d >= H) return;
    const float* xr = x + (size_t)t * MHC_HC * H;
    const float* c = comb + (size_t)t * MHC_HC * MHC_HC;
    float r[MHC_HC];
#pragma unroll
    for (int j = 0; j < MHC_HC; j++) r[j] = xr[j * H + d];
    const float yv = y[(size_t)t * H + d];
    float o[MHC_HC];
#pragma unroll
    for (int i = 0; i < MHC_HC; i++) {
        float v = __fmul_rn(c[i], r[0]);
#pragma unroll
        for (int j = 1; j < MHC_HC; j++) v = __fadd_rn(v, __fmul_rn(c[j * MHC_HC + i], r[j]));
        o[i] = __fadd_rn(__fmul_rn(post[t * MHC_HC + i], yv), v);
    }
    float* orow = out + (size_t)t * MHC_HC * H;
#pragma unroll
    for (int i = 0; i < MHC_HC; i++) orow[i * H + d] = o[i];
}

// the layer driver's block sum of gm_rmsnorm (kernels_glm5_mla.cu `gm_block_sum`), copied verbatim:
// glm5_mhc_mix_norm's norm tail must take gm_rmsnorm's reduction tree to stay bit-identical
__device__ __forceinline__ float mhc_gm_warp_sum(float v) {
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    return v;
}
__device__ __forceinline__ float mhc_gm_block_sum(float v, float* red) {
    v = mhc_gm_warp_sum(v);
    int w = threadIdx.x >> 5, l = threadIdx.x & 31;
    __syncthreads();
    if (l == 0) red[w] = v;
    __syncthreads();
    float s = (l < (blockDim.x >> 5)) ? red[l] : 0.0f;
    if (w == 0) s = mhc_gm_warp_sum(s);
    if (threadIdx.x == 0) red[0] = s;
    __syncthreads();
    return red[0];
}

// #191: steps 1-5 + 9 of one site. grid (24, T), block 256. Block m of row t: the record's RMS
// factor (every block the same chain), then the record's dot chain, xor tree and warp sum for mix
// row m -> logits[t][m]. The LAST of the 24 blocks of row t to finish (a per-row counter `done`,
// reset by that block, so the next launch finds 0) reads the row's 24 logits and runs the record's
// steps 3-5 on warp 0 (pre, post, comb stored; mhc_coeff_tail_warp) while warps 1-7 derive pre
// themselves (lanes 32-35, the same expression) and write collapsed [t] in the record's order. Only
// that block reads other blocks' logits: they are published with __threadfence before the counter
// add and read back with __ldcg (L2, not a stale L1).
// NORM (CROW_GLM_HCFUSE, glm5_mhc_mix_norm): that last block then also runs the sublayer's weighted
// RMSNorm over collapsed [t] in place, gm_rmsnorm's code (one 256-thread block per row, the same
// strided chain, the same block sum, the same expressions under NVRTC's default contraction), so
// the norm launch of the driver (MlaKernels::rmsnorm_rows) is folded into the site.
template <bool NORM>
__device__ __forceinline__ void mhc_mix_site(const float* __restrict__ x, const unsigned short* __restrict__ fn,
                                             const float* __restrict__ base, const float* __restrict__ scale,
                                             float* logits, float* __restrict__ pre, float* __restrict__ post,
                                             float* __restrict__ comb, float* collapsed,
                                             unsigned int* done, const int* __restrict__ prm,
                                             const float* __restrict__ nw) {
    __shared__ float sh_red[MHC_THREADS / 32];
    __shared__ float sh_part[MHC_THREADS / 32];
    __shared__ float sh_m[MHC_MIX];
    __shared__ float sh_pre[MHC_HC];
    __shared__ float sh_pre1[MHC_HC];
    __shared__ int sh_last;
    const int H = prm[0];
    const int n = MHC_HC * H;
    const int m = blockIdx.x;
    const int t = blockIdx.y;
    const float* xr = x + (size_t)t * n;

    // 1. the record's unweighted RMSNorm factor (every block the same chain, the same r)
    float ss = 0.0f;
#pragma unroll 16
    for (int k = threadIdx.x; k < n; k += MHC_THREADS) ss = fmaf(xr[k], xr[k], ss);
    ss = mhc_block_sum(ss, sh_red);
    const float r = __frsqrt_rn(__fadd_rn(__fdiv_rn(ss, (float)n), MHC_RMS_EPS));

    // 2. row m of m = fn . (r x): the record's acc[m] chain, xor tree, sequential warp sum
    const unsigned short* fm = fn + (size_t)m * n;
    float acc = 0.0f;
#pragma unroll 16
    for (int k = threadIdx.x; k < n; k += MHC_THREADS) {
        const float xn = __fmul_rn(xr[k], r);
        acc = fmaf(mhc_bf16(fm[k]), xn, acc);
    }
    float v = acc;
    for (int o = 16; o > 0; o >>= 1) v = __fadd_rn(v, __shfl_xor_sync(0xffffffffu, v, o));
    if ((threadIdx.x & 31) == 0) sh_part[threadIdx.x >> 5] = v;
    __syncthreads();
    if (threadIdx.x == 0) {
        float s = 0.0f;
        for (int i = 0; i < MHC_THREADS / 32; i++) s = __fadd_rn(s, sh_part[i]);
        logits[(size_t)t * MHC_MIX + m] = s;
        __threadfence();
        const unsigned int prev = atomicAdd(&done[t], 1u);
        sh_last = prev == MHC_MIX - 1;
        if (sh_last) done[t] = 0u;  // every other block of the row has counted
    }
    __syncthreads();
    if (!sh_last) return;  // block-uniform

    // 3.-5. and 9., the last block of row t only
    __threadfence();
    if (threadIdx.x < MHC_MIX) sh_m[threadIdx.x] = __ldcg(logits + (size_t)t * MHC_MIX + threadIdx.x);
    __syncthreads();
    if (threadIdx.x < 32) {
        mhc_coeff_tail_warp(sh_m, base, scale, sh_pre, pre, post, comb, t, true);
        if (!NORM) return;
    } else {
        // warps 1-7: pre for the collapse (the record's expression), then the collapse
        const int u = threadIdx.x - 32;
        if (u < MHC_HC) sh_pre1[u] = __fadd_rn(mhc_sigmoid(__fadd_rn(__fmul_rn(sh_m[u], scale[0]), base[u])), MHC_HC_EPS);
        asm volatile("bar.sync 1, %0;" ::"r"(MHC_THREADS - 32));
        const float p0 = sh_pre1[0], p1 = sh_pre1[1], p2 = sh_pre1[2], p3 = sh_pre1[3];
        for (int d = u; d < H; d += MHC_THREADS - 32) {
            float c = __fmul_rn(p0, xr[d]);
            c = __fadd_rn(c, __fmul_rn(p1, xr[H + d]));
            c = __fadd_rn(c, __fmul_rn(p2, xr[2 * H + d]));
            c = __fadd_rn(c, __fmul_rn(p3, xr[3 * H + d]));
            collapsed[(size_t)t * H + d] = c;
        }
    }
    if (!NORM) return;

    // NORM: gm_rmsnorm over collapsed [t] (n = H, eps = rms_norm_eps), all 256 threads; the
    // barrier makes the collapse of warps 1-7 visible to the whole block
    __syncthreads();
    __shared__ float red[32];
    float* xp = collapsed + (size_t)t * H;
    const long long hn = H;
    float sq = 0.0f;
    for (long long i = threadIdx.x; i < hn; i += blockDim.x) sq += xp[i] * xp[i];
    float rn = rsqrtf(mhc_gm_block_sum(sq, red) / (float)hn + MHC_RMS_EPS);
    for (long long i = threadIdx.x; i < hn; i += blockDim.x) xp[i] = nw[i] * (xp[i] * rn);
}

extern "C" __global__ void glm5_mhc_mix(const float* __restrict__ x, const unsigned short* __restrict__ fn,
                                        const float* __restrict__ base, const float* __restrict__ scale,
                                        float* logits, float* __restrict__ pre, float* __restrict__ post,
                                        float* __restrict__ comb, float* __restrict__ collapsed,
                                        unsigned int* done, const int* __restrict__ prm) {
    mhc_mix_site<false>(x, fn, base, scale, logits, pre, post, comb, collapsed, done, prm, nullptr);
}

// CROW_GLM_HCFUSE: glm5_mhc_mix, then the sublayer RMSNorm (weight nw [H] f32) over collapsed in
// place, in one launch. grid (24, T), block 256.
extern "C" __global__ void glm5_mhc_mix_norm(const float* __restrict__ x, const unsigned short* __restrict__ fn,
                                             const float* __restrict__ base, const float* __restrict__ scale,
                                             float* logits, float* __restrict__ pre, float* __restrict__ post,
                                             float* __restrict__ comb, float* collapsed,
                                             unsigned int* done, const int* __restrict__ prm,
                                             const float* __restrict__ nw) {
    mhc_mix_site<true>(x, fn, base, scale, logits, pre, post, comb, collapsed, done, prm, nw);
}

// #202 A (CROW_GLM_ATTN_FUSE): glm5_mhc_expand of the site before (its post / comb still in the
// plan, the sublayer output y) folded into glm5_mhc_mix_norm of the next site: one launch where the
// layer queued two (the attention site's expand, the FFN site's mix_norm). grid (24, T), block 256,
// H / 256 one of 1, 2, 4, 8, 16 (MHC_EXP_Q). Every block first runs glm5_mhc_expand's
// statements for its thread's columns d = threadIdx.x + 256 q of row t (the coefficients copied to
// shared memory first) into registers xe[i][q] = X'[i][d], then mhc_mix_site's chains (NORM) over
// those values in the record's k order (k = i H + d ascending per thread, the order of its strided
// loops), so every logit, coefficient, collapse and norm value has the bits of expand + mix_norm.
// The last block of the row (every other block of the row has read x and the old post / comb by
// then: each counts after its loops) writes X' over x (in place, as the expand did), then runs
// mhc_mix_site's tail, which writes the new site's coefficients and reads X' back for the collapse
// (the block's own stores, ordered by the barrier). x is not restrict here: this kernel writes it.
// Q = H / 256 is a template parameter (1, 2, 4, 8 or 16): with a run-time Q the register array's
// guarded loops ran 2.5 times slower (RTX 5090, h 4096: 37.5 us against 16.1 us per launch).
#define MHC_EXP_Q 16

template <int Q>
__device__ __forceinline__ void mhc_expand_mix_norm_q(float* x, const float* __restrict__ y,
                                                      const unsigned short* __restrict__ fn, const float* __restrict__ base,
                                                      const float* __restrict__ scale, float* logits, float* pre, float* post,
                                                      float* comb, float* collapsed, unsigned int* done,
                                                      const int* __restrict__ prm, const float* __restrict__ nw) {
    __shared__ float sh_red[MHC_THREADS / 32];
    __shared__ float sh_part[MHC_THREADS / 32];
    __shared__ float sh_m[MHC_MIX];
    __shared__ float sh_pre[MHC_HC];
    __shared__ float sh_pre1[MHC_HC];
    __shared__ float sh_ep[MHC_HC];
    __shared__ float sh_ec[MHC_HC * MHC_HC];
    __shared__ int sh_last;
    const int H = prm[0];
    const int n = MHC_HC * H;
    const int m = blockIdx.x;
    const int t = blockIdx.y;
    float* xr = x + (size_t)t * n;

    // 0. glm5_mhc_expand of row t (the previous site's post / comb), this thread's columns
    if (threadIdx.x < MHC_HC) sh_ep[threadIdx.x] = post[t * MHC_HC + threadIdx.x];
    if (threadIdx.x < MHC_HC * MHC_HC) sh_ec[threadIdx.x] = comb[(size_t)t * MHC_HC * MHC_HC + threadIdx.x];
    __syncthreads();
    float xe[MHC_HC][Q];
#pragma unroll
    for (int q = 0; q < Q; q++) {
        if (q < Q) {
            const int d = threadIdx.x + MHC_THREADS * q;
            float r[MHC_HC];
#pragma unroll
            for (int j = 0; j < MHC_HC; j++) r[j] = xr[j * H + d];
            const float yv = y[(size_t)t * H + d];
#pragma unroll
            for (int i = 0; i < MHC_HC; i++) {
                float v = __fmul_rn(sh_ec[i], r[0]);
#pragma unroll
                for (int j = 1; j < MHC_HC; j++) v = __fadd_rn(v, __fmul_rn(sh_ec[j * MHC_HC + i], r[j]));
                xe[i][q] = __fadd_rn(__fmul_rn(sh_ep[i], yv), v);
            }
        }
    }

    // 1. the record's unweighted RMSNorm factor over X'
    float ss = 0.0f;
#pragma unroll
    for (int i = 0; i < MHC_HC; i++)
#pragma unroll
        for (int q = 0; q < Q; q++)
            if (q < Q) ss = fmaf(xe[i][q], xe[i][q], ss);
    ss = mhc_block_sum(ss, sh_red);
    const float r = __frsqrt_rn(__fadd_rn(__fdiv_rn(ss, (float)n), MHC_RMS_EPS));

    // 2. row m of m = fn . (r X')
    const unsigned short* fm = fn + (size_t)m * n;
    float acc = 0.0f;
#pragma unroll
    for (int i = 0; i < MHC_HC; i++)
#pragma unroll
        for (int q = 0; q < Q; q++)
            if (q < Q) {
                const float xn = __fmul_rn(xe[i][q], r);
                acc = fmaf(mhc_bf16(fm[i * H + threadIdx.x + MHC_THREADS * q]), xn, acc);
            }
    float v = acc;
    for (int o = 16; o > 0; o >>= 1) v = __fadd_rn(v, __shfl_xor_sync(0xffffffffu, v, o));
    if ((threadIdx.x & 31) == 0) sh_part[threadIdx.x >> 5] = v;
    __syncthreads();
    if (threadIdx.x == 0) {
        float s = 0.0f;
        for (int i = 0; i < MHC_THREADS / 32; i++) s = __fadd_rn(s, sh_part[i]);
        logits[(size_t)t * MHC_MIX + m] = s;
        __threadfence();
        const unsigned int prev = atomicAdd(&done[t], 1u);
        sh_last = prev == MHC_MIX - 1;
        if (sh_last) done[t] = 0u;  // every other block of the row has counted
    }
    __syncthreads();
    if (!sh_last) return;  // block-uniform

    // the expand's output: X' over x (every block of the row has read x)
#pragma unroll
    for (int i = 0; i < MHC_HC; i++)
#pragma unroll
        for (int q = 0; q < Q; q++)
            if (q < Q) xr[i * H + threadIdx.x + MHC_THREADS * q] = xe[i][q];

    // 3.-5. and 9. (mhc_mix_site's tail with NORM), the last block of row t only
    __threadfence();
    if (threadIdx.x < MHC_MIX) sh_m[threadIdx.x] = __ldcg(logits + (size_t)t * MHC_MIX + threadIdx.x);
    __syncthreads();
    if (threadIdx.x < 32) {
        mhc_coeff_tail_warp(sh_m, base, scale, sh_pre, pre, post, comb, t, true);
    } else {
        const int u = threadIdx.x - 32;
        if (u < MHC_HC) sh_pre1[u] = __fadd_rn(mhc_sigmoid(__fadd_rn(__fmul_rn(sh_m[u], scale[0]), base[u])), MHC_HC_EPS);
        asm volatile("bar.sync 1, %0;" ::"r"(MHC_THREADS - 32));
        const float p0 = sh_pre1[0], p1 = sh_pre1[1], p2 = sh_pre1[2], p3 = sh_pre1[3];
        for (int d = u; d < H; d += MHC_THREADS - 32) {
            float c = __fmul_rn(p0, xr[d]);
            c = __fadd_rn(c, __fmul_rn(p1, xr[H + d]));
            c = __fadd_rn(c, __fmul_rn(p2, xr[2 * H + d]));
            c = __fadd_rn(c, __fmul_rn(p3, xr[3 * H + d]));
            collapsed[(size_t)t * H + d] = c;
        }
    }
    // gm_rmsnorm over collapsed [t] (mhc_mix_site's NORM part)
    __syncthreads();
    __shared__ float red[32];
    float* xp = collapsed + (size_t)t * H;
    const long long hn = H;
    float sq = 0.0f;
    for (long long i = threadIdx.x; i < hn; i += blockDim.x) sq += xp[i] * xp[i];
    float rn = rsqrtf(mhc_gm_block_sum(sq, red) / (float)hn + MHC_RMS_EPS);
    for (long long i = threadIdx.x; i < hn; i += blockDim.x) xp[i] = nw[i] * (xp[i] * rn);
}

extern "C" __global__ void glm5_mhc_expand_mix_norm(float* x, const float* __restrict__ y,
                                                    const unsigned short* __restrict__ fn, const float* __restrict__ base,
                                                    const float* __restrict__ scale, float* logits, float* pre, float* post,
                                                    float* comb, float* collapsed, unsigned int* done,
                                                    const int* __restrict__ prm, const float* __restrict__ nw) {
    switch (prm[0] / MHC_THREADS) {
#define MHC_EXP_ARM(Q)     case Q: mhc_expand_mix_norm_q<Q>(x, y, fn, base, scale, logits, pre, post, comb, collapsed, done, prm, nw); break;
        MHC_EXP_ARM(1) MHC_EXP_ARM(2) MHC_EXP_ARM(4) MHC_EXP_ARM(8) MHC_EXP_ARM(MHC_EXP_Q)
#undef MHC_EXP_ARM
        default: __trap();
    }
}
