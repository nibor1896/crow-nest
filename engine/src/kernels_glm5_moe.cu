// crow-nest #164: the glm5_next FFN kernels (NVRTC, compute_120a) - the GPU half of
// `glm5_moe.rs`. Its own module (`kernels::glm5_moe::Kernels::new` compiles `GLM5_MOE_SRC`
// alone): one more `.entry` in KERNEL_SRC would break the PTX of record (`kernels::tests_300_c4`),
// and these kernels use none of its helpers.
//
// Math of record: transformers 5.16.1 `modeling_glm5_next.py` (docs/glm5-next-recipe.md
// sections 9-10): the sigmoid router with the selection-only `e_score_correction_bias`
// (Glm5NextTextTopkRouter.forward, M:158-183), the SwiGLU clamp of every FFN
// (M:98-104, M:137-142), the routed sum plus the shared expert (M:200-207).
//
//   glm5_router_sig_topk  s = sigmoid(logit), c = s + bias; K times: pick argmax c (ties -> the
//                         lowest expert, as router_top10), w = s[pick]; w = w / (sum w + 1e-20) * scale
//   glm5_moe_gather       combo c = t * K + k: ptrs[c] = table[ids[c]], xg[c] = x[t]
//   glm5_swiglu_clamp     h = silu(min(g, L)) * clamp(u, -L, L), silu(v) = v / (1 + exp(-v))
//   glm5_moe_combine      y[t] = (sum_k w[t][k] * ye[t * K + k]) + ys[t], k in pick order
//
// Every rounded add and multiply outside expf is an __fadd_rn / __fmul_rn / __fdiv_rn intrinsic,
// which NVRTC never contracts into an fma (-fmad=true is its default), so the f32 order is the
// one of the CPU twins in glm5_moe.rs. expf is the accurate one (no --use_fast_math).
// Scalar arguments come in device buffers (the KERNEL_SRC rule).

#define GLM5_ROUTER_THREADS 512
#define GLM5_MAXK 16

__device__ __forceinline__ float glm5_neg_inf() { return __int_as_float(0xff800000); }

// a better than b: larger value, ties to the smaller key (keys are unique)
__device__ __forceinline__ void glm5_best(float& v, int& i, float v2, int i2) {
    if (v2 > v || (v2 == v && i2 < i)) {
        v = v2;
        i = i2;
    }
}

// grid (T), block 512 (E <= 512, K <= 16, E >= K). prm_i = {E, K}, prm_f = {routed_scaling}.
// logits [T][E] f32, bias [E] f32 -> ids [T][K] i32, wts [T][K] f32, in pick order.
// Threads e >= E hold -inf with a key past every expert, so the power-of-two reduction stays
// valid for E = 288. A picked expert gets -inf and a key past the unpicked ones, so even a row
// of NaN choices (NaN reads as -inf) picks K distinct experts.
extern "C" __global__ void glm5_router_sig_topk(const float* __restrict__ logits, const float* __restrict__ bias,
                                                int* __restrict__ ids, float* __restrict__ wts,
                                                const int* __restrict__ prm_i, const float* __restrict__ prm_f) {
    const int E = prm_i[0], K = prm_i[1];
    const float scale = prm_f[0];
    const int t = blockIdx.x, e = threadIdx.x, lane = e & 31, warp = e >> 5;
    __shared__ float sv[GLM5_ROUTER_THREADS / 32];
    __shared__ int si[GLM5_ROUTER_THREADS / 32];
    __shared__ int pick_s;
    __shared__ float wsel[GLM5_MAXK];
    float sig = 0.0f, c = glm5_neg_inf();
    int key = (1 << 20) + e;
    if (e < E) {
        float l = logits[(size_t)t * E + e];
        sig = __fdiv_rn(1.0f, __fadd_rn(1.0f, expf(-l)));
        c = __fadd_rn(sig, bias[e]);
        if (!(c == c)) c = glm5_neg_inf();
        key = e;
    }
    for (int j = 0; j < K; j++) {
        float v = c;
        int i = key;
        #pragma unroll
        for (int o = 16; o > 0; o >>= 1) glm5_best(v, i, __shfl_xor_sync(0xffffffffu, v, o), __shfl_xor_sync(0xffffffffu, i, o));
        if (lane == 0) {
            sv[warp] = v;
            si[warp] = i;
        }
        __syncthreads();
        if (warp == 0) {
            const int nw = blockDim.x >> 5;
            v = lane < nw ? sv[lane] : glm5_neg_inf();
            i = lane < nw ? si[lane] : 0x7fffffff;
            #pragma unroll
            for (int o = 16; o > 0; o >>= 1) glm5_best(v, i, __shfl_xor_sync(0xffffffffu, v, o), __shfl_xor_sync(0xffffffffu, i, o));
            if (lane == 0) pick_s = i;
        }
        __syncthreads();
        if (key == pick_s) {
            wsel[j] = sig;
            ids[(size_t)t * K + j] = e;
            c = glm5_neg_inf();
            key = (1 << 21) + e;
        }
        __syncthreads();
    }
    if (e == 0) {
        float sum = 0.0f;
        for (int j = 0; j < K; j++) sum = __fadd_rn(sum, wsel[j]);
        const float den = __fadd_rn(sum, 1e-20f);
        for (int j = 0; j < K; j++) wts[(size_t)t * K + j] = __fmul_rn(__fdiv_rn(wsel[j], den), scale);
    }
}

// grid (ceil(H / 256), T * K), block 256. prm = {K, H}. table [E] u64 record bases (a VRAM slot
// or a pinned-host UVA address: residency is invisible here), ids [T][K] -> ptrs [T * K] u64;
// x [T][H] -> xg [T * K][H] (the MUL1 GemvPlan input, one slot per combo, one token each).
extern "C" __global__ void glm5_moe_gather(const int* __restrict__ ids, const unsigned long long* __restrict__ table,
                                           const float* __restrict__ x, unsigned long long* __restrict__ ptrs,
                                           float* __restrict__ xg, const int* __restrict__ prm) {
    const int K = prm[0], H = prm[1];
    const int c = blockIdx.y, t = c / K;
    const int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j < H) xg[(size_t)c * H + j] = x[(size_t)t * H + j];
    if (blockIdx.x == 0 && threadIdx.x == 0) ptrs[c] = table[ids[c]];
}

// grid (ceil(n / 256)), block 256. prm_i = {n}, prm_f = {_, swiglu_limit}. g, u, h [n].
// clamp as torch.clamp: a NaN stays NaN (comparisons, not fminf / fmaxf)
extern "C" __global__ void glm5_swiglu_clamp(const float* __restrict__ g, const float* __restrict__ u, float* __restrict__ h,
                                             const int* __restrict__ prm_i, const float* __restrict__ prm_f) {
    const int n = prm_i[0];
    const float L = prm_f[1];
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float gv = g[i], uv = u[i];
    gv = gv > L ? L : gv;
    uv = uv > L ? L : (uv < -L ? -L : uv);
    const float silu = __fdiv_rn(gv, __fadd_rn(1.0f, expf(-gv)));
    h[i] = __fmul_rn(silu, uv);
}

// grid (ceil(H / 256), T), block 256. prm = {K, H}. ye [T * K][H], w [T][K], ys [T][H] -> y [T][H]
extern "C" __global__ void glm5_moe_combine(const float* __restrict__ ye, const float* __restrict__ w,
                                            const float* __restrict__ ys, float* __restrict__ y,
                                            const int* __restrict__ prm) {
    const int K = prm[0], H = prm[1];
    const int t = blockIdx.y;
    const int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= H) return;
    float acc = 0.0f;
    for (int k = 0; k < K; k++) acc = __fadd_rn(acc, __fmul_rn(w[(size_t)t * K + k], ye[((size_t)t * K + k) * H + j]));
    y[(size_t)t * H + j] = __fadd_rn(acc, ys[(size_t)t * H + j]);
}
