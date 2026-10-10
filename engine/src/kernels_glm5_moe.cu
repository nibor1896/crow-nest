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
//   glm5_moe_combine_rows the same, each row through a pointer (#202 CROW_GLM_RT2)
//   glm5_moe_tables_in    dst[i] = src[i], i < 2 * K: RT2's slot / row tables from mapped host memory
//                         into VRAM on the SMs (#202 CROW_GLM_RT2_TABLE_SM)
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

// #202 CROW_GLM_RT2: glm5_moe_combine with each combo's row read through a pointer:
// rows [T * K] u64 device addresses of the [H] f32 output rows (a GPU slot of ye, or a CPU-lane
// row in mapped host memory). The same expressions in the same order as glm5_moe_combine, so the
// bits are those of the combine over the same rows. grid (ceil(H / 256), T), block 256.
extern "C" __global__ void glm5_moe_combine_rows(const unsigned long long* __restrict__ rows, const float* __restrict__ w,
                                                 const float* __restrict__ ys, float* __restrict__ y,
                                                 const int* __restrict__ prm) {
    const int K = prm[0], H = prm[1];
    const int t = blockIdx.y;
    const int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= H) return;
    float acc = 0.0f;
    for (int k = 0; k < K; k++) {
        const float* r = (const float*)rows[(size_t)t * K + k];
        acc = __fadd_rn(acc, __fmul_rn(w[(size_t)t * K + k], r[j]));
    }
    y[(size_t)t * H + j] = __fadd_rn(acc, ys[(size_t)t * H + j]);
}

// #202 CROW_GLM_RT2_TABLE_SM: the [2 * K] u64 slot and row tables of GpuMoePlan::experts_rt2,
// written by the host into a mapped pinned buffer (src, its device pointer), copied into the
// device table (dst) by the SMs instead of a cuMemcpyHtoDAsync: no copy-engine command, so the
// early pass that follows on the compute stream does not queue behind the stager's landings and
// write-backs on the copy engine. Volatile loads: the host rewrites these words between calls, so
// no cache may serve a former call's value (each word is read once per launch anyway).
// prm = {K, H}. grid (1), block 32 (2 * K <= 2 * GLM5_MAXK = 32; the loop covers any block).
extern "C" __global__ void glm5_moe_tables_in(const volatile unsigned long long* src, unsigned long long* __restrict__ dst,
                                              const int* __restrict__ prm) {
    const int n = 2 * prm[0];
    for (int i = threadIdx.x; i < n; i += blockDim.x) dst[i] = src[i];
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
//   glm5_gemv_fp4_x3  three matrices of one shape in one launch (the KDA q|k|v projections).

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

// the body of glm5_gemv_fp4 for the block's rows r0 .. r0 + GLM5_FP4_RB - 1 of one matrix, row t of x
__device__ __forceinline__ void glm5_gemv_fp4_rows(const unsigned char* __restrict__ w, const float* __restrict__ x,
                                                   const float gs, float* __restrict__ y, const int k_dim,
                                                   const int ldy, const int rows, const int r0, const int t) {
    __shared__ float red[GLM5_FP4_RB][256];
    const int bpr = k_dim >> 6;
    const int i = threadIdx.x;  // the record's thread i
    const int nw = blockDim.x >> 5;
    const float* xp = x + (size_t)t * k_dim;
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

extern "C" __global__ void glm5_gemv_fp4(const unsigned char* __restrict__ w, const float* __restrict__ x,
                                         const float* __restrict__ gs_ptr, float* __restrict__ y,
                                         const int* __restrict__ k_dim_p, const int* __restrict__ ldy_p,
                                         const int* __restrict__ rows_p) {
    glm5_gemv_fp4_rows(w, x, gs_ptr[0], y, *k_dim_p, *ldy_p, *rows_p, blockIdx.x * GLM5_FP4_RB, blockIdx.y);
}

// glm5_gemv_fp4 over three matrices of the same shape in ONE launch (the KDA q|k|v projections of
// a row, which read the same x): blocks m * nb .. (m + 1) * nb - 1 (nb = ceil(rows / GLM5_FP4_RB))
// run matrix m into the columns m * rows .. (m + 1) * rows - 1 of y, each block exactly as its
// glm5_gemv_fp4 launch (bit-identical). grid (3 * nb, T), block as glm5_gemv_fp4.
extern "C" __global__ void glm5_gemv_fp4_x3(const unsigned char* __restrict__ w0, const unsigned char* __restrict__ w1,
                                            const unsigned char* __restrict__ w2, const float* __restrict__ gs0,
                                            const float* __restrict__ gs1, const float* __restrict__ gs2,
                                            const float* __restrict__ x, float* __restrict__ y,
                                            const int* __restrict__ k_dim_p, const int* __restrict__ ldy_p,
                                            const int* __restrict__ rows_p) {
    const int rows = *rows_p;
    const int nb = (rows + GLM5_FP4_RB - 1) / GLM5_FP4_RB;
    const int m = blockIdx.x / nb;
    const int b = blockIdx.x - m * nb;
    const unsigned char* w = m == 0 ? w0 : (m == 1 ? w1 : w2);
    const float gs = m == 0 ? gs0[0] : (m == 1 ? gs1[0] : gs2[0]);
    glm5_gemv_fp4_rows(w, x, gs, y + (size_t)m * rows, *k_dim_p, *ldy_p, rows, b * GLM5_FP4_RB, blockIdx.y);
}

// ---------------- crow-nest #186: the glm5_next dense NVFP4 GEMM of a prompt call ----------------
// Y[t][n] = gs * sum_k X[t][k] W[n][k] for T >= kernels::glm5_moe::TC_MIN_ROWS prompt rows on the
// FP16 tensor cores (`CROW_GLM_DENSE_GEMM=1`), the way exllamav3 runs its large-batch EXL3 GEMM
// (exl3.py: reconstruct the weight to FP16 once, hgemm with FP32 accumulate). Each NVFP4 block
// is decoded once per 128-row tile of prompt rows into shared memory as FP16: e2m1 * ue4m3 has
// at most 6 significant bits and lies in [2^-10, 2880], so the FP16 weight is EXACT (lossless);
// the global scale gs is applied once in the epilogue. X is rounded to FP16 (the one rounding of
// this path) and every product is accumulated in FP32 by mma.m16n8k16. Not bit-identical to
// glm5_gemv_fp4: the accumulation order is the tensor core's.
//
//   glm5_gemm_fp4_tc  grid (ceil(rows / 128), ceil(T / 128), M), block 256 (8 warps, 2 x 4, each
//                     64 prompt rows x 32 outputs). Matrix m (M <= 3, the KDA q|k|v of one launch)
//                     is w_m / gs_m and writes the columns m * rows .. of y (row stride ldy).
//                     w [rows][K / 64][36] u8, x [T][K] f32 -> y [T][ldy] f32; K % 64 == 0;
//                     k_dim_p, ldy_p, rows_p, t_p are device ints (the KERNEL_SRC rule).

#define GLM5_TC_BM 128
#define GLM5_TC_BN 128
#define GLM5_TC_LDS 72  // halves per shared row: 64 + 8, so ldmatrix's 8 row reads hit 32 banks

__device__ __forceinline__ unsigned int glm5_h2(float lo, float hi) {
    unsigned int r;
    asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(r) : "f"(hi), "f"(lo));
    return r;
}
__device__ __forceinline__ unsigned int glm5_smem(const void* p) {
    unsigned int r;
    asm("{ .reg .u64 a; cvta.to.shared.u64 a, %1; cvt.u32.u64 %0, a; }" : "=r"(r) : "l"(p));
    return r;
}
__device__ __forceinline__ void glm5_ldsm4(unsigned int addr, unsigned int& r0, unsigned int& r1, unsigned int& r2, unsigned int& r3) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];" : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3) : "r"(addr));
}
__device__ __forceinline__ void glm5_mma(float* c, const unsigned int* a, unsigned int b0, unsigned int b1) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};"
                 : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}

extern "C" __global__ void __launch_bounds__(256) glm5_gemm_fp4_tc(
    const unsigned char* __restrict__ w0, const unsigned char* __restrict__ w1, const unsigned char* __restrict__ w2,
    const float* __restrict__ gs0, const float* __restrict__ gs1, const float* __restrict__ gs2,
    const float* __restrict__ x, float* __restrict__ y, const int* __restrict__ k_dim_p, const int* __restrict__ ldy_p,
    const int* __restrict__ rows_p, const int* __restrict__ t_p) {
    __shared__ __align__(16) unsigned short xs[GLM5_TC_BM * GLM5_TC_LDS];
    __shared__ __align__(16) unsigned short ws[GLM5_TC_BN * GLM5_TC_LDS];
    const int k_dim = *k_dim_p, ldy = *ldy_p, rows = *rows_p, T = *t_p;
    const int m = blockIdx.z;
    const unsigned char* w = m == 0 ? w0 : (m == 1 ? w1 : w2);
    const float gs = m == 0 ? gs0[0] : (m == 1 ? gs1[0] : gs2[0]);
    y += (size_t)m * rows;
    const int bpr = k_dim >> 6;
    const int n0 = blockIdx.x * GLM5_TC_BN, t0 = blockIdx.y * GLM5_TC_BM;
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    const int wm = warp >> 2, wn = warp & 3;

    // the loads of one k-tile (64 values): X 128 x 64 f32 as 8 float4 per thread (16 per row,
    // coalesced), W one 36-byte block of row tid / 2, sub-blocks 2 (tid & 1) and 2 (tid & 1) + 1
    float4 xr[8];
    unsigned int wr[5];
    const int wrow = tid >> 1, wh = tid & 1;
    const unsigned int* wbase = (const unsigned int*)(w + (size_t)min(n0 + wrow, rows - 1) * bpr * 36);
    auto load = [&](int kt) {
#pragma unroll
        for (int j = 0; j < 8; j++) {
            const int q = tid + 256 * j, r = q >> 4, c4 = q & 15;
            xr[j] = *(const float4*)(x + (size_t)min(t0 + r, T - 1) * k_dim + kt * 64 + c4 * 4);
        }
        const unsigned int* bp = wbase + kt * 9;
        wr[0] = bp[0];
#pragma unroll
        for (int q = 0; q < 4; q++) wr[1 + q] = bp[1 + 4 * wh + q];
    };
    auto store = [&]() {
#pragma unroll
        for (int j = 0; j < 8; j++) {
            const int q = tid + 256 * j, r = q >> 4, c4 = q & 15;
            *(uint2*)(xs + r * GLM5_TC_LDS + c4 * 4) = make_uint2(glm5_h2(xr[j].x, xr[j].y), glm5_h2(xr[j].z, xr[j].w));
        }
#pragma unroll
        for (int h = 0; h < 2; h++) {
            const int sb = 2 * wh + h;
            const float s = glm5_ue4m3((wr[0] >> (8 * sb)) & 0xFF);
            unsigned int p[8];
#pragma unroll
            for (int j = 0; j < 8; j++) {
                const unsigned int byte = (wr[1 + 2 * h + (j >> 2)] >> ((j & 3) * 8)) & 0xFF;
                p[j] = glm5_h2(glm5_e2m1(byte & 0xF) * s, glm5_e2m1(byte >> 4) * s);
            }
            uint4* dst = (uint4*)(ws + wrow * GLM5_TC_LDS + sb * 16);
            dst[0] = make_uint4(p[0], p[1], p[2], p[3]);
            dst[1] = make_uint4(p[4], p[5], p[6], p[7]);
        }
    };

    float acc[4][4][4];
#pragma unroll
    for (int i = 0; i < 4; i++)
#pragma unroll
        for (int j = 0; j < 4; j++)
#pragma unroll
            for (int v = 0; v < 4; v++) acc[i][j][v] = 0.0f;

    const unsigned int xs_a = glm5_smem(xs) + 2 * ((wm * 64 + (lane & 15)) * GLM5_TC_LDS + (lane >> 4) * 8);
    const unsigned int ws_a = glm5_smem(ws) + 2 * ((wn * 32 + ((lane >> 4) << 3) + (lane & 7)) * GLM5_TC_LDS + ((lane >> 3) & 1) * 8);
    load(0);
    store();
    __syncthreads();
    for (int kt = 0; kt < bpr; kt++) {
        if (kt + 1 < bpr) load(kt + 1);
#pragma unroll
        for (int ks = 0; ks < 4; ks++) {
            unsigned int a[4][4], b[4][2];
#pragma unroll
            for (int mi = 0; mi < 4; mi++) glm5_ldsm4(xs_a + 2 * (mi * 16 * GLM5_TC_LDS + ks * 16), a[mi][0], a[mi][1], a[mi][2], a[mi][3]);
#pragma unroll
            for (int nj = 0; nj < 2; nj++) glm5_ldsm4(ws_a + 2 * (nj * 16 * GLM5_TC_LDS + ks * 16), b[2 * nj][0], b[2 * nj][1], b[2 * nj + 1][0], b[2 * nj + 1][1]);
#pragma unroll
            for (int mi = 0; mi < 4; mi++)
#pragma unroll
                for (int ni = 0; ni < 4; ni++) glm5_mma(acc[mi][ni], a[mi], b[ni][0], b[ni][1]);
        }
        __syncthreads();
        if (kt + 1 < bpr) {
            store();
            __syncthreads();
        }
    }
    const int g = lane >> 2, c = lane & 3;
#pragma unroll
    for (int mi = 0; mi < 4; mi++)
#pragma unroll
        for (int ni = 0; ni < 4; ni++)
#pragma unroll
            for (int v = 0; v < 4; v++) {
                const int t = t0 + wm * 64 + mi * 16 + g + (v >> 1) * 8;
                const int n = n0 + wn * 32 + ni * 8 + 2 * c + (v & 1);
                if (t < T && n < rows) y[(size_t)t * ldy + n] = acc[mi][ni][v] * gs;
            }
}

// glm5_gemm_fp4_tc for calls of few rows (kernels::glm5_moe::TC_SMALL_MAX_ROWS and fewer): the
// 128 x 128 tiles leave most SMs idle there (grid ceil(rows / 128) x 1). Here a block computes
// 32 prompt rows x 32 outputs and its 8 warps split K (warp w takes the k-tiles w, w + 8, ...),
// fragments straight from global memory (no shared staging): lane (g, c) decodes the bytes c of
// its four weight rows' code words, which are exactly its B fragments (e2m1 x ue4m3, exact in
// FP16), and loads its A fragments as float2 -> FP16. The warps' partial tiles are summed in
// warp order 0 .. 7 (deterministic), times gs. Same arguments as glm5_gemm_fp4_tc;
// grid (ceil(rows / 32), ceil(T / 32), M), block 256.
#define GLM5_TCS_T 32
#define GLM5_TCS_NI 4  // n8 tiles per block (2 measured slower from 16 rows on)
#define GLM5_TCS_N (8 * GLM5_TCS_NI)

extern "C" __global__ void __launch_bounds__(256) glm5_gemm_fp4_tcs(
    const unsigned char* __restrict__ w0, const unsigned char* __restrict__ w1, const unsigned char* __restrict__ w2,
    const float* __restrict__ gs0, const float* __restrict__ gs1, const float* __restrict__ gs2,
    const float* __restrict__ x, float* __restrict__ y, const int* __restrict__ k_dim_p, const int* __restrict__ ldy_p,
    const int* __restrict__ rows_p, const int* __restrict__ t_p) {
    __shared__ float red[8 * 8 * GLM5_TCS_NI * 32];
    const int k_dim = *k_dim_p, ldy = *ldy_p, rows = *rows_p, T = *t_p;
    const int m = blockIdx.z;
    const unsigned char* w = m == 0 ? w0 : (m == 1 ? w1 : w2);
    const float gs = m == 0 ? gs0[0] : (m == 1 ? gs1[0] : gs2[0]);
    y += (size_t)m * rows;
    const int bpr = k_dim >> 6;
    const int n0 = blockIdx.x * GLM5_TCS_N, t0 = blockIdx.y * GLM5_TCS_T;
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    const int g = lane >> 2, c = lane & 3;
    const unsigned int* wrow[GLM5_TCS_NI];
#pragma unroll
    for (int ni = 0; ni < GLM5_TCS_NI; ni++) wrow[ni] = (const unsigned int*)(w + (size_t)min(n0 + ni * 8 + g, rows - 1) * bpr * 36);
    const float* xrow[4];
#pragma unroll
    for (int r = 0; r < 4; r++) xrow[r] = x + (size_t)min(t0 + r * 8 + g, T - 1) * k_dim + 2 * c;  // rows g, g + 8, g + 16, g + 24
    float acc[2][GLM5_TCS_NI][4];
#pragma unroll
    for (int i = 0; i < 2; i++)
#pragma unroll
        for (int j = 0; j < GLM5_TCS_NI; j++)
#pragma unroll
            for (int v = 0; v < 4; v++) acc[i][j][v] = 0.0f;
    for (int kt = warp; kt < bpr; kt += 8) {
        unsigned int wd[GLM5_TCS_NI][9];
#pragma unroll
        for (int ni = 0; ni < GLM5_TCS_NI; ni++)
#pragma unroll
            for (int q = 0; q < 9; q++) wd[ni][q] = wrow[ni][kt * 9 + q];
        float2 xv[4][4][2];
#pragma unroll
        for (int r = 0; r < 4; r++)
#pragma unroll
            for (int ks = 0; ks < 4; ks++) {
                xv[r][ks][0] = *(const float2*)(xrow[r] + kt * 64 + ks * 16);
                xv[r][ks][1] = *(const float2*)(xrow[r] + kt * 64 + ks * 16 + 8);
            }
#pragma unroll
        for (int ks = 0; ks < 4; ks++) {
            unsigned int b[GLM5_TCS_NI][2];
#pragma unroll
            for (int ni = 0; ni < GLM5_TCS_NI; ni++) {
                const float s = glm5_ue4m3((wd[ni][0] >> (8 * ks)) & 0xFF);
#pragma unroll
                for (int h = 0; h < 2; h++) {
                    const unsigned int byte = (wd[ni][1 + 2 * ks + h] >> (8 * c)) & 0xFF;
                    b[ni][h] = glm5_h2(glm5_e2m1(byte & 0xF) * s, glm5_e2m1(byte >> 4) * s);
                }
            }
#pragma unroll
            for (int mi = 0; mi < 2; mi++) {
                const unsigned int a[4] = {glm5_h2(xv[2 * mi][ks][0].x, xv[2 * mi][ks][0].y), glm5_h2(xv[2 * mi + 1][ks][0].x, xv[2 * mi + 1][ks][0].y),
                                           glm5_h2(xv[2 * mi][ks][1].x, xv[2 * mi][ks][1].y), glm5_h2(xv[2 * mi + 1][ks][1].x, xv[2 * mi + 1][ks][1].y)};
#pragma unroll
                for (int ni = 0; ni < GLM5_TCS_NI; ni++) glm5_mma(acc[mi][ni], a, b[ni][0], b[ni][1]);
            }
        }
    }
#pragma unroll
    for (int mi = 0; mi < 2; mi++)
#pragma unroll
        for (int ni = 0; ni < GLM5_TCS_NI; ni++)
#pragma unroll
            for (int e = 0; e < 4; e++) red[(warp * 8 * GLM5_TCS_NI + (mi * GLM5_TCS_NI + ni) * 4 + e) * 32 + lane] = acc[mi][ni][e];
    __syncthreads();
    for (int o = tid; o < 8 * GLM5_TCS_NI * 32; o += 256) {
        const int v = o >> 5, l = o & 31;
        float s = 0.0f;
#pragma unroll
        for (int wi = 0; wi < 8; wi++) s += red[(wi * 8 * GLM5_TCS_NI + v) * 32 + l];
        const int mi = v / (4 * GLM5_TCS_NI), ni = (v >> 2) % GLM5_TCS_NI, e = v & 3;
        const int t = t0 + mi * 16 + (l >> 2) + (e >> 1) * 8;
        const int n = n0 + ni * 8 + 2 * (l & 3) + (e & 1);
        if (t < T && n < rows) y[(size_t)t * ldy + n] = s * gs;
    }
}
