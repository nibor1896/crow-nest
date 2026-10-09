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
