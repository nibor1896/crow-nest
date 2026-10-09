// crow-nest #164: the glm5_next FFN kernels (NVRTC, compute_120a) - the GPU half of
// `glm5_moe.rs`. Its own module (`kernels::glm5_moe::Kernels::new` compiles `GLM5_MOE_SRC`
// alone): one more `.entry` in KERNEL_SRC would break the PTX of record (`kernels::tests_300_c4`),
// and these kernels use none of its helpers (#191's `glm5_gemv_fp4` at the end, the NVFP4 GEMV of
// every glm5 dense projection, carries its own NVFP4 decode helpers).
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

// ---------------- crow-nest #191: the glm5_next dense NVFP4 GEMV ----------------
// y[t][row] (row stride ldy) = W x[t] for one NVFP4 matrix (36-byte blocks: 4 UE4M3 scale bytes,
// 32 code bytes, 64 values; rows of K / 64 blocks; global scale gs), the dense projections of
// glm5_next (KDA q/k/v/o, MLA q_a/q_b/kv_a/o, dense FFN, shared expert). Bit-identical to the
// KERNEL_SRC record `gemv_fp4_b` / `gemv_fp4_bs` (kernels.rs), which KERNEL_SRC keeps for
// Flash-Next / 27B. The record runs one 256-thread block per row, thread i accumulating the
// blocks i, i + 256, ... as `part += e2m1 * x` over 16 values and `acc += part * scale` per
// sub-block, then the shared-memory tree red[i] += red[i + s], s = 128 .. 1. Here thread i is
// the record's thread i for GLM5_FP4_RB rows at once (one x load serves them all), with the
// same expressions in the same order, and the same tree per row: the eight values
// red[lane + 32 v] are added in-lane for s = 128, 64, 32 and by __shfl_down_sync for s = 16 .. 1
// (the same operand pairs). The block has only the warps that hold blocks (32 * ceil(bpr / 32)
// threads, at most 256): a thread the record runs without a block holds +0.0f, and so does every
// absent thread here, so the tree adds the same zeros. Memory: whole 36-byte blocks as nine
// 32-bit words, x as float4, and the `e2m1` value in registers (below) - the record's measured
// bottleneck.
//
//   glm5_gemv_fp4  grid (ceil(rows / GLM5_FP4_RB), T), block 32 * min(8, ceil(K / 2048)).
//                  w [rows][K / 64][36] u8, x [T][K] f32, gs [1] f32 -> y [T][ldy] f32 (row < rows).
//                  K % 64 == 0; k_dim_p, ldy_p, rows_p are device ints (the KERNEL_SRC rule).

#define GLM5_FP4_RB 2

// The value of the KERNEL_SRC helper `e2m1` (kernels.rs), built in registers. The record reads a
// `const float mag[8]` with a run-time index, which NVRTC places in local memory and REWRITES on
// every call (PTX of `gemv_fp4_b`: two st.local.v2.b64 of the 32-byte table and one ld.local per
// weight value, ~36 B of local traffic per 0.56 B of weight - the measured bottleneck of #191).
// The same eight values as IEEE bits: code c = nib & 7 is 0 -> +0, 1 -> 0.5 (2^-1), else
// 2^((c >> 1) - 1) * (1 + (c & 1) / 2), i.e. exponent field (c >> 1) + 126, mantissa bit 22 =
// c & 1; the sign bit of nib flips the sign exactly as `-v` does (nib 8 -> -0.0).
__device__ __forceinline__ float glm5_e2m1(unsigned int nib) {
    const unsigned int c = nib & 0x7;
    const unsigned int mag = c < 2 ? (c ? 0x3F000000u : 0u) : ((((c >> 1) + 126u) << 23) | ((c & 1u) << 22));
    return __uint_as_float(mag | ((nib & 0x8u) << 28));
}
// the KERNEL_SRC decode helper `ue4m3` (kernels.rs), expression for expression (4 per 64 values)
__device__ __forceinline__ float glm5_ue4m3(unsigned int byte) {
    unsigned int e = (byte >> 3) & 0xF;
    unsigned int m = byte & 0x7;
    if (e == 0) return (float)m * 1.953125e-3f;
    return (1.0f + (float)m / 8.0f) * exp2f((float)e - 7.0f);
}

extern "C" __global__ void glm5_gemv_fp4(const unsigned char* __restrict__ w, const float* __restrict__ x,
                                         const float* __restrict__ gs_ptr, float* __restrict__ y,
                                         const int* __restrict__ k_dim_p, const int* __restrict__ ldy_p,
                                         const int* __restrict__ rows_p) {
    __shared__ float red[GLM5_FP4_RB][256];
    const int k_dim = *k_dim_p;
    const int ldy = *ldy_p;
    const int rows = *rows_p;
    const int bpr = k_dim >> 6;
    const int i = threadIdx.x;  // the record's thread i
    const int nw = blockDim.x >> 5;
    const int r0 = blockIdx.x * GLM5_FP4_RB;
    const int t = blockIdx.y;
    const float* xp = x + (size_t)t * k_dim;
    const float gs = gs_ptr[0];
    float acc[GLM5_FP4_RB];
#pragma unroll
    for (int r = 0; r < GLM5_FP4_RB; r++) acc[r] = 0.0f;
    for (int b = i; b < bpr; b += 256) {
        unsigned int wd[GLM5_FP4_RB][9];
#pragma unroll
        for (int r = 0; r < GLM5_FP4_RB; r++) {
            const int row = min(r0 + r, rows - 1);  // a tail row re-reads the last row, never stored
            const unsigned int* bp = (const unsigned int*)(w + ((size_t)row * bpr + b) * 36);
#pragma unroll
            for (int q = 0; q < 9; q++) wd[r][q] = bp[q];
        }
#pragma unroll
        for (int sb = 0; sb < 4; sb++) {
            const float4* x4 = (const float4*)(xp + b * 64 + sb * 16);
            float xv[16];
#pragma unroll
            for (int q = 0; q < 4; q++) {
                const float4 f = x4[q];
                xv[4 * q] = f.x;
                xv[4 * q + 1] = f.y;
                xv[4 * q + 2] = f.z;
                xv[4 * q + 3] = f.w;
            }
#pragma unroll
            for (int r = 0; r < GLM5_FP4_RB; r++) {
                float s = glm5_ue4m3((wd[r][0] >> (8 * sb)) & 0xFF) * gs;
                float part = 0.0f;
#pragma unroll
                for (int j = 0; j < 16; j++) {
                    unsigned int byte = (wd[r][1 + sb * 2 + (j >> 3)] >> (((j >> 1) & 3) * 8)) & 0xFF;
                    unsigned int nib = (j & 1) ? ((byte >> 4) & 0xF) : (byte & 0xF);
                    part += glm5_e2m1(nib) * xv[j];
                }
                acc[r] += part * s;
            }
        }
    }
#pragma unroll
    for (int r = 0; r < GLM5_FP4_RB; r++) red[r][i] = acc[r];
    __syncthreads();
    // the record tree, one row per warp: s = 128 (v, v + 4), 64 (v, v + 2), 32 (v, v + 1), 16 .. 1
    const int lane = i & 31;
    for (int r = i >> 5; r < GLM5_FP4_RB; r += nw) {
        float a[8];
#pragma unroll
        for (int v = 0; v < 8; v++) a[v] = v < nw ? red[r][lane + 32 * v] : 0.0f;
        float a0 = a[0] + a[4];
        float a1 = a[1] + a[5];
        float a2 = a[2] + a[6];
        float a3 = a[3] + a[7];
        a0 = a0 + a2;
        a1 = a1 + a3;
        a0 = a0 + a1;
#pragma unroll
        for (int o = 16; o > 0; o >>= 1) a0 = a0 + __shfl_down_sync(0xffffffffu, a0, o);
        if (lane == 0 && r0 + r < rows) y[(size_t)t * ldy + r0 + r] = a0;
    }
}
