//! Consolidated CUDA kernel source (NVRTC, compute_120a). Every family is the
//! probe-verified math (p5–p16); the engine additions are:
//!   - FP8 E4M3 KV cache cast/decode (config BF16 path kept),
//!   - exact GPU top-k block selection for QSA (radix refine, tie-safe),
//!   - GPU router (softmax + top-10 + residency bitmap + pointer gather),
//!   - pointer-table batched FP4 GEMV (zero-copy cold tier = pinned UVA ptrs),
//!   - list-based sparse attention over selected cache rows.
//! Rules held: scalar args via device buffers, full-flat guards, block-stride
//! loops over full rows, explicit stride params for slices of strided matrices.
//!
//! The host side of this file is the kernel table (`Kernels::f`), the ONE
//! `cuLaunchKernel` shim every launch goes through (`launch_v`) and the
//! per-kernel profile that shim feeds.

pub const KERNEL_SRC: &str = r#"
// ---------------- decode helpers (p2/p10-verified) ----------------
__device__ __forceinline__ float e2m1(unsigned int nib) {
    const float mag[8] = {0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f};
    float v = mag[nib & 0x7];
    return (nib & 0x8) ? -v : v;
}
__device__ __forceinline__ float ue4m3(unsigned int byte) {
    unsigned int e = (byte >> 3) & 0xF;
    unsigned int m = byte & 0x7;
    if (e == 0) return (float)m * 1.953125e-3f;
    return (1.0f + (float)m / 8.0f) * exp2f((float)e - 7.0f);
}
// FP8 E4M3 (RNE, saturating) — exact twin of engine::cnq::f32_to_e4m3
__device__ __forceinline__ unsigned char enc_e4m3(float v) {
    unsigned int bits = __float_as_uint(v);
    unsigned char s = (unsigned char)((bits >> 24) & 0x80);
    float f = fabsf(v);
    if (f != f) return (unsigned char)(s | 0x7F); // NaN check (no NAN macro in NVRTC)
    if (f >= 464.0f) return (unsigned char)(s | 0x7E);
    if (f < 0.015625f) {
        float m = nearbyintf(f * 512.0f);
        if (m >= 8.0f) return (unsigned char)(s | 0x08);
        return (unsigned char)(s | ((unsigned char)m & 0x7));
    }
    int e = (int)floorf(log2f(f));
    while (ldexpf(1.0f, e) > f) e--;
    while (ldexpf(1.0f, e + 1) <= f) e++;
    int ef = e + 7;
    if (ef > 15) return (unsigned char)(s | 0x7E);
    float q = nearbyintf(ldexpf(f, 3 - e));
    if (q >= 16.0f) {
        if (e + 8 > 15) return (unsigned char)(s | 0x7E);
        return (unsigned char)(s | ((unsigned char)(e + 8) << 3));
    }
    return (unsigned char)(s | ((unsigned char)ef << 3) | ((unsigned char)q & 0x7));
}
__device__ __forceinline__ float dec_e4m3(unsigned char b) {
    unsigned int e = (b >> 3) & 0xF;
    unsigned int m = b & 0x7;
    float v;
    if (e == 15 && m == 7) v = __int_as_float(0x7FC00000u);
    else if (e == 0) v = ldexpf((float)m, -9);
    else v = (1.0f + (float)m / 8.0f) * ldexpf(1.0f, (int)e - 7);
    return (b & 0x80) ? -v : v;
}
__device__ __forceinline__ unsigned short f32_bf16_bits(float v) {
    unsigned int x = __float_as_uint(v);
    x += 0x7FFFu + ((x >> 16) & 1u);
    return (unsigned short)(x >> 16);
}
// cache load: mode 0 = e4m3 (1 B), mode 1 = bf16 (2 B)
__device__ __forceinline__ float kv_load(const unsigned char* p, int i, int mode) {
    if (mode == 0) return dec_e4m3(p[i]);
    unsigned short b = ((const unsigned short*)p)[i];
    return __int_as_float(((unsigned int)b) << 16);
}

// ---------------- GEMV family ----------------
extern "C" __global__ void gemv_f32(const float* __restrict__ w, const float* __restrict__ x,
                                    float* __restrict__ y, const int* __restrict__ k_dim_p) {
    int k_dim = *k_dim_p;
    int row = blockIdx.x;
    const float* wp = w + (size_t)row * k_dim;
    float acc = 0.0f;
    for (int i = threadIdx.x; i < k_dim; i += blockDim.x) acc += wp[i] * x[i];
    __shared__ float red[256];
    red[threadIdx.x] = acc;
    __syncthreads();
    for (int st = blockDim.x / 2; st > 0; st >>= 1) {
        if (threadIdx.x < st) red[threadIdx.x] += red[threadIdx.x + st];
        __syncthreads();
    }
    if (threadIdx.x == 0) y[row] = red[0];
}

// batched: y[t][row] over grid (rows, T)
extern "C" __global__ void gemv_b(const float* __restrict__ w, const float* __restrict__ x,
                                  float* __restrict__ y, const int* __restrict__ k_dim_p) {
    int k_dim = *k_dim_p;
    int row = blockIdx.x;
    int t = blockIdx.y;
    const float* wp = w + (size_t)row * k_dim;
    const float* xp = x + (size_t)t * k_dim;
    float acc = 0.0f;
    for (int i = threadIdx.x; i < k_dim; i += blockDim.x) acc += wp[i] * xp[i];
    __shared__ float red[256];
    red[threadIdx.x] = acc;
    __syncthreads();
    for (int st = blockDim.x / 2; st > 0; st >>= 1) {
        if (threadIdx.x < st) red[threadIdx.x] += red[threadIdx.x + st];
        __syncthreads();
    }
    if (threadIdx.x == 0) y[(size_t)t * gridDim.x + row] = red[0];
}

// FP4 GEMV, single activation row (p10-verified numerics)
extern "C" __global__ void gemv_fp4(const unsigned char* __restrict__ w, const float* __restrict__ x,
                                    const float* __restrict__ gs_ptr, float* __restrict__ y,
                                    const int* __restrict__ k_dim_p) {
    int k_dim = *k_dim_p;
    int bpr = k_dim >> 6;
    int row = blockIdx.x;
    const unsigned char* rowp = w + (size_t)row * bpr * 36;
    float gs = gs_ptr[0];
    float acc = 0.0f;
    for (int b = threadIdx.x; b < bpr; b += blockDim.x) {
        const unsigned char* blk = rowp + b * 36;
        #pragma unroll
        for (int sb = 0; sb < 4; sb++) {
            float s = ue4m3(blk[sb]) * gs;
            float part = 0.0f;
            #pragma unroll
            for (int j = 0; j < 16; j++) {
                int idx = sb * 16 + j;
                unsigned int byte = blk[4 + (idx >> 1)];
                unsigned int nib = (idx & 1) ? ((byte >> 4) & 0xF) : (byte & 0xF);
                part += e2m1(nib) * x[b * 64 + sb * 16 + j];
            }
            acc += part * s;
        }
    }
    __shared__ float red[256];
    red[threadIdx.x] = acc;
    __syncthreads();
    for (int st = blockDim.x / 2; st > 0; st >>= 1) {
        if (threadIdx.x < st) red[threadIdx.x] += red[threadIdx.x + st];
        __syncthreads();
    }
    if (threadIdx.x == 0) y[row] = red[0];
}

// FP4 GEMV batched over token rows: grid (rows, T)
extern "C" __global__ void gemv_fp4_b(const unsigned char* __restrict__ w, const float* __restrict__ x,
                                      const float* __restrict__ gs_ptr, float* __restrict__ y,
                                      const int* __restrict__ k_dim_p) {
    int k_dim = *k_dim_p;
    int bpr = k_dim >> 6;
    int row = blockIdx.x;
    int t = blockIdx.y;
    const unsigned char* rowp = w + (size_t)row * bpr * 36;
    const float* xp = x + (size_t)t * k_dim;
    float gs = gs_ptr[0];
    float acc = 0.0f;
    for (int b = threadIdx.x; b < bpr; b += blockDim.x) {
        const unsigned char* blk = rowp + b * 36;
        #pragma unroll
        for (int sb = 0; sb < 4; sb++) {
            float s = ue4m3(blk[sb]) * gs;
            float part = 0.0f;
            #pragma unroll
            for (int j = 0; j < 16; j++) {
                int idx = sb * 16 + j;
                unsigned int byte = blk[4 + (idx >> 1)];
                unsigned int nib = (idx & 1) ? ((byte >> 4) & 0xF) : (byte & 0xF);
                part += e2m1(nib) * xp[b * 64 + sb * 16 + j];
            }
            acc += part * s;
        }
    }
    __shared__ float red[256];
    red[threadIdx.x] = acc;
    __syncthreads();
    for (int st = blockDim.x / 2; st > 0; st >>= 1) {
        if (threadIdx.x < st) red[threadIdx.x] += red[threadIdx.x + st];
        __syncthreads();
    }
    if (threadIdx.x == 0) y[(size_t)t * gridDim.x + row] = red[0];
}

// FP4 GEMV with per-combo weight POINTER (device-read): the zero-copy cold
// path — ptr is a VRAM address for hot experts or a pinned-host UVA address
// for cold ones; residency is invisible to the kernel (p5 pattern).
// grid (rows, combos); combo = t * x_div + rank  (x_div 10 routed / 1 combo-major)
extern "C" __global__ void gemv_fp4_ptrb(const unsigned long long* __restrict__ ptrs,
                                         const float* __restrict__ x, const float* __restrict__ gs_ptr,
                                         float* __restrict__ y, const int* __restrict__ k_dim_p,
                                         const int* __restrict__ x_div_p, const int* __restrict__ x_stride_p) {
    int k_dim = *k_dim_p;
    int bpr = k_dim >> 6;
    int row = blockIdx.x;
    int combo = blockIdx.y;
    const unsigned char* w = (const unsigned char*)ptrs[combo];
    int xr = combo / *x_div_p;
    const float* xp = x + (size_t)xr * *x_stride_p;
    float gs = gs_ptr[0];
    float acc = 0.0f;
    const unsigned char* rowp = w + (size_t)row * bpr * 36;
    for (int b = threadIdx.x; b < bpr; b += blockDim.x) {
        const unsigned char* blk = rowp + b * 36;
        #pragma unroll
        for (int sb = 0; sb < 4; sb++) {
            float s = ue4m3(blk[sb]) * gs;
            float part = 0.0f;
            #pragma unroll
            for (int j = 0; j < 16; j++) {
                int idx = sb * 16 + j;
                unsigned int byte = blk[4 + (idx >> 1)];
                unsigned int nib = (idx & 1) ? ((byte >> 4) & 0xF) : (byte & 0xF);
                part += e2m1(nib) * xp[b * 64 + sb * 16 + j];
            }
            acc += part * s;
        }
    }
    __shared__ float red[256];
    red[threadIdx.x] = acc;
    __syncthreads();
    for (int st = blockDim.x / 2; st > 0; st >>= 1) {
        if (threadIdx.x < st) red[threadIdx.x] += red[threadIdx.x + st];
        __syncthreads();
    }
    if (threadIdx.x == 0) y[(size_t)combo * gridDim.x + row] = red[0];
}

// ---------------- MMA FP4 (tensor-core path, #10) ----------------
// e2m1 RNE quantization with ties-to-even (lattice {0,.5,1,1.5,2,3,4,6};
// tie points .25,.75,1.25,1.75,2.5,3.5,5 -> even-mantissa side, i.e. 0,1,1,2,2,4,4)
__device__ __forceinline__ unsigned int q_e2m1(float v) { // |v| in [0,6]
    unsigned int s = (v < 0.0f) ? 8u : 0u;
    float r = fabsf(v);
    if (r <= 0.25f) return s | 0;
    if (r < 0.75f) return s | 1;
    if (r <= 1.25f) return s | 2;
    if (r < 1.75f) return s | 3;
    if (r <= 2.5f) return s | 4;
    if (r < 3.5f) return s | 5;
    if (r <= 5.0f) return s | 6;
    return s | 7;
}
// smallest ue4m3 >= s (no-overflow scale encoding; 0 iff s <= 0; saturates at
// 0x7E = 448 — 0x7F is the E4M3 NaN encoding on the tensor core, mma_probe2).
// At e = 15 the mantissa stops at 6: s in (448, 480] gives m = 7 there, i.e.
// (15 << 3) | 7 = 0x7F, which must fall through to the 0x7E saturation.
__device__ __forceinline__ unsigned char enc_ue4m3_up(float s) {
    if (!(s > 0.0f)) return 0;
    for (int e = 0; e < 16; e++) {
        float mul = (e == 0) ? 512.0f : ldexpf(1.0f, 10 - e);
        float m = ceilf(s * mul) - ((e == 0) ? 0.0f : 8.0f);
        float m_max = (e == 15) ? 6.0f : 7.0f;
        if (m >= 0.0f && m <= m_max) return (unsigned char)((e << 3) | (int)m);
    }
    return 0x7E;
}
// f32 activation rows -> NVFP4 36-byte 64-blocks, THREE levels per block
// (x ~= dec(q1)+dec(q2)+dec(q3), each level quantizing the previous residual
// with its own ue4m3 scales; the residual cascade halves the activation
// quantization error ~10x per level at the cost of one extra mma per k-block
// — the weight fragments are reused).
// Layout per row: LV * bpr*36 bytes, level L at L*bpr*36.
// grid (rows), block 128; row r reads x[r * x_stride] where r = blockIdx.x —
// callers pass the combo table decomposition like gemv_fp4_ptrb (x_div/x_stride),
// so one quantized row serves all TOPK combos of a token (gate_up) or is
// per-combo (down, x_div=1).
__device__ __forceinline__ void quant_level(const float* p, float* r, unsigned char* sc_dst,
                                            unsigned char* d_dst) {
    float amax = 0.0f;
    #pragma unroll
    for (int j = 0; j < 16; j++) amax = fmaxf(amax, fabsf(p[j]));
    unsigned char sc = enc_ue4m3_up(amax * (1.0f / 6.0f));
    float inv = 1.0f / ue4m3(sc);
    float scf = ue4m3(sc);
    *sc_dst = sc;
    #pragma unroll
    for (int j = 0; j < 8; j++) {
        unsigned int n0 = q_e2m1(p[2 * j] * inv);
        unsigned int n1 = q_e2m1(p[2 * j + 1] * inv);
        d_dst[j] = (unsigned char)(n0 | (n1 << 4));
        r[2 * j] = p[2 * j] - e2m1(n0) * scf;
        r[2 * j + 1] = p[2 * j + 1] - e2m1(n1) * scf;
    }
}
extern "C" __global__ void quant_x_fp4(const float* __restrict__ x, unsigned char* __restrict__ xq,
                                       const int* __restrict__ k_dim_p, const int* __restrict__ x_div_p,
                                       const int* __restrict__ x_stride_p) {
    int k_dim = *k_dim_p;
    int bpr = k_dim >> 6;
    int row = blockIdx.x / *x_div_p;
    const float* xp = x + (size_t)row * *x_stride_p;
    unsigned char* op = xq + (size_t)blockIdx.x * bpr * 108; // 3 levels
    for (int sb = threadIdx.x; sb < bpr * 4; sb += blockDim.x) {
        int b = sb >> 2, s = sb & 3;
        const float* p = xp + b * 64 + s * 16;
        float r0[16], r1[16];
        quant_level(p, r0, op + b * 36 + s, op + b * 36 + 4 + s * 8);
        quant_level(r0, r1, op + bpr * 36 + b * 36 + s, op + bpr * 36 + b * 36 + 4 + s * 8);
        quant_level(r1, r0, op + 2 * bpr * 36 + b * 36 + s, op + 2 * bpr * 36 + b * 36 + 4 + s * 8);
    }
}
// p2-proven instruction, wrapped for reuse (D accumulates in-place)
__device__ __forceinline__ void mma_fp4_16n8k64(float& d0, float& d1, float& d2, float& d3,
                                                unsigned int a0, unsigned int a1, unsigned int a2,
                                                unsigned int a3, unsigned int b0, unsigned int b1,
                                                unsigned int sa, unsigned int sb) {
    asm volatile(
        "mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 "
        "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, "
        "%10, {0, 0}, %11, {0, 0};"
        : "+f"(d0), "+f"(d1), "+f"(d2), "+f"(d3)
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1), "r"(sa), "r"(sb));
}
// Tensor-core batched FP4 GEMV over the routed expert pointer table — the MMA
// replacement for gemv_fp4_ptrb (p2 layout + mma_probe lane map).
// One mma.sync.m16n8k64 per (warp, 64-k-block, quant level), accumulated over
// bpr blocks and both levels:
//   A operand = 16 WEIGHT rows of the combo's expert, straight from the 36-byte
//     container blocks (a0 = row g k[8t..], a1 = row g+8, a2/a3 k[32+8t..];
//     sf_a byte i = k-subblock i, lane t=L&3 holds row g+8t, t>=2 ignored);
//   B operand = the combo's quantized activation, ONE 36-byte block broadcast
//     over all 8 n-columns (b0 = act k[8t..8t+7] etc., lane t=0 supplies the
//     sf_b bytes), level 0 + level 1 (residual) summed into the same D regs —
//     n is 8x redundant (decode combos are distinct experts, so no
//     expert-grouped batching exists; 1/8 mma utilization, still >>naive);
//   D: d0 = D[g][2t] -> y[combo][w0+g], d2 -> y[combo][w0+g+8], written by the
//     t==0 lanes; the per-tensor global scale folds into the final store.
// grid (rows/64, combos), block 128 (4 warps = 64 weight rows per block);
// y row stride = rows = gridDim.x*64.
// K-split version (deterministic): blockDim = 128 * KS; warp w -> row group
// (w & 3) of 16 rows, k-slice (w >> 2) of the bpr k-blocks; partials are
// reduced through shared memory in a FIXED order (slice 0..KS-1), so the
// result does not depend on scheduling. KS = 1 reproduces the old kernel
// bit for bit. Chosen per launch via the block size (gen::mma_bx()).
extern "C" __global__ void gemv_fp4_mma(const unsigned long long* __restrict__ ptrs,
                                        const unsigned char* __restrict__ xq,
                                        const float* __restrict__ gs_ptr,
                                        float* __restrict__ y,
                                        const int* __restrict__ k_dim_p,
                                        const int* __restrict__ x_div_p) {
    int k_dim = *k_dim_p;
    int bpr = k_dim >> 6;
    int combo = blockIdx.y;
    const unsigned char* w = (const unsigned char*)ptrs[combo];
    int xr = combo / *x_div_p;
    const unsigned char* xa = xq + (size_t)xr * bpr * 108;
    int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    int ks_n = blockDim.x >> 7;
    int rg = warp & 3, ks = warp >> 2;
    int g = lane >> 2, t = lane & 3;
    int w0 = (blockIdx.x << 6) + (rg << 4);
    const unsigned char* rowg = w + (size_t)(w0 + g) * bpr * 36;
    const unsigned char* rowg8 = rowg + 8 * bpr * 36;
    const unsigned char* sfrow = (t & 1) ? rowg8 : rowg;
    float d0 = 0.0f, d1 = 0.0f, d2 = 0.0f, d3 = 0.0f;
    int bpb = (bpr + ks_n - 1) / ks_n;
    int b_lo = ks * bpb, b_hi = min(bpr, b_lo + bpb);
    for (int b = b_lo; b < b_hi; b++) {
        const unsigned char* blk = rowg + b * 36;
        const unsigned char* blk8 = rowg8 + b * 36;
        const unsigned char* ab = xa + b * 36;
        unsigned int sa = sfrow[b * 36 + 0] | (unsigned int)sfrow[b * 36 + 1] << 8
                        | (unsigned int)sfrow[b * 36 + 2] << 16 | (unsigned int)sfrow[b * 36 + 3] << 24;
        unsigned int a0 = *(const unsigned int*)(blk + 4 + 4 * t);
        unsigned int a1 = *(const unsigned int*)(blk8 + 4 + 4 * t);
        unsigned int a2 = *(const unsigned int*)(blk + 20 + 4 * t);
        unsigned int a3 = *(const unsigned int*)(blk8 + 20 + 4 * t);
        unsigned int sb = ab[0] | (unsigned int)ab[1] << 8
                        | (unsigned int)ab[2] << 16 | (unsigned int)ab[3] << 24;
        unsigned int b0 = *(const unsigned int*)(ab + 4 + 4 * t);
        unsigned int b1 = *(const unsigned int*)(ab + 20 + 4 * t);
        mma_fp4_16n8k64(d0, d1, d2, d3, a0, a1, a2, a3, b0, b1, sa, sb);
        #pragma unroll
        for (int lv = 1; lv < 3; lv++) {
            const unsigned char* ab2 = ab + lv * bpr * 36; // residual level
            unsigned int sb2 = ab2[0] | (unsigned int)ab2[1] << 8
                             | (unsigned int)ab2[2] << 16 | (unsigned int)ab2[3] << 24;
            unsigned int c0 = *(const unsigned int*)(ab2 + 4 + 4 * t);
            unsigned int c1 = *(const unsigned int*)(ab2 + 20 + 4 * t);
            mma_fp4_16n8k64(d0, d1, d2, d3, a0, a1, a2, a3, c0, c1, sa, sb2);
        }
    }
    __shared__ float red[4][2][64];
    if (t == 0) { red[ks][0][(rg << 4) + g] = d0; red[ks][1][(rg << 4) + g] = d2; }
    __syncthreads();
    float gs = gs_ptr[0];
    if (ks == 0 && t == 0) {
        float s0 = 0.0f, s2 = 0.0f;
        for (int i = 0; i < ks_n; i++) { s0 += red[i][0][(rg << 4) + g]; s2 += red[i][1][(rg << 4) + g]; }
        y[(size_t)combo * ((size_t)gridDim.x << 6) + w0 + g] = s0 * gs;
        y[(size_t)combo * ((size_t)gridDim.x << 6) + w0 + g + 8] = s2 * gs;
    }
}

// Dense twin of gemv_fp4_mma: direct weight pointer (no residency table),
// whole-warp row-count guard (rows not a multiple of 64 supported) and an
// explicit y row stride — keeps BOTH dense output layouts ([t][rows] compact
// like gemv_fp4_b, and sh12 [t][1280] like gemv_fp4_bs). A-fragment rows past
// rows-1 are clamped (mma.sync needs all 32 lanes); their stores are masked.
// grid (ceil(rows/64), T), block 128; xq = quant_x_fp4 output, one row per
// token (token = blockIdx.y), k_dim must match the weight slab.
extern "C" __global__ void gemv_fp4_mma_d(const unsigned char* __restrict__ w,
                                          const unsigned char* __restrict__ xq,
                                          const float* __restrict__ gs_ptr,
                                          float* __restrict__ y,
                                          const int* __restrict__ k_dim_p,
                                          const int* __restrict__ rows_p,
                                          const int* __restrict__ y_stride_p) {
    int k_dim = *k_dim_p;
    int bpr = k_dim >> 6;
    int rows = *rows_p;
    int ys = *y_stride_p;
    int tok = blockIdx.y;
    int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    int ks_n = blockDim.x >> 7;
    int rg = warp & 3, ks = warp >> 2;
    int g = lane >> 2, lt = lane & 3;
    int w0 = (blockIdx.x << 6) + (rg << 4);
    bool active = w0 < rows;   // whole-warp guard; no early return (smem barrier below)
    float d0 = 0.0f, d1 = 0.0f, d2 = 0.0f, d3 = 0.0f;
    int r0 = w0 + g, r1 = w0 + g + 8;
    if (active) {
        const unsigned char* xa = xq + (size_t)tok * bpr * 108;
        const unsigned char* rowg = w + (size_t)min(r0, rows - 1) * bpr * 36;
        const unsigned char* rowg8 = w + (size_t)min(r1, rows - 1) * bpr * 36;
        const unsigned char* sfrow = (lt & 1) ? rowg8 : rowg;
        int bpb = (bpr + ks_n - 1) / ks_n;
        int b_lo = ks * bpb, b_hi = min(bpr, b_lo + bpb);
        for (int b = b_lo; b < b_hi; b++) {
            const unsigned char* blk = rowg + b * 36;
            const unsigned char* blk8 = rowg8 + b * 36;
            const unsigned char* ab = xa + b * 36;
            unsigned int sa = sfrow[b * 36 + 0] | (unsigned int)sfrow[b * 36 + 1] << 8
                            | (unsigned int)sfrow[b * 36 + 2] << 16 | (unsigned int)sfrow[b * 36 + 3] << 24;
            unsigned int a0 = *(const unsigned int*)(blk + 4 + 4 * lt);
            unsigned int a1 = *(const unsigned int*)(blk8 + 4 + 4 * lt);
            unsigned int a2 = *(const unsigned int*)(blk + 20 + 4 * lt);
            unsigned int a3 = *(const unsigned int*)(blk8 + 20 + 4 * lt);
            unsigned int sb = ab[0] | (unsigned int)ab[1] << 8
                            | (unsigned int)ab[2] << 16 | (unsigned int)ab[3] << 24;
            unsigned int b0 = *(const unsigned int*)(ab + 4 + 4 * lt);
            unsigned int b1 = *(const unsigned int*)(ab + 20 + 4 * lt);
            mma_fp4_16n8k64(d0, d1, d2, d3, a0, a1, a2, a3, b0, b1, sa, sb);
            #pragma unroll
            for (int lv = 1; lv < 3; lv++) {
                const unsigned char* ab2 = ab + lv * bpr * 36; // residual level
                unsigned int sb2 = ab2[0] | (unsigned int)ab2[1] << 8
                                 | (unsigned int)ab2[2] << 16 | (unsigned int)ab2[3] << 24;
                unsigned int c0 = *(const unsigned int*)(ab2 + 4 + 4 * lt);
                unsigned int c1 = *(const unsigned int*)(ab2 + 20 + 4 * lt);
                mma_fp4_16n8k64(d0, d1, d2, d3, a0, a1, a2, a3, c0, c1, sa, sb2);
            }
        }
    }
    __shared__ float red[4][2][64];
    if (lt == 0) { red[ks][0][(rg << 4) + g] = d0; red[ks][1][(rg << 4) + g] = d2; }
    __syncthreads();
    float gs = gs_ptr[0];
    if (ks == 0 && lt == 0 && active) {
        float s0 = 0.0f, s2 = 0.0f;
        for (int i = 0; i < ks_n; i++) { s0 += red[i][0][(rg << 4) + g]; s2 += red[i][1][(rg << 4) + g]; }
        if (r0 < rows) y[(size_t)tok * ys + r0] = s0 * gs;
        if (r1 < rows) y[(size_t)tok * ys + r1] = s2 * gs;
    }
}

// Dense FP4 GEMM with 8 tokens per mma (prefill twin of gemv_fp4_mma_d): the
// GEMV form re-reads the weight matrix once per token (t=512: 7.5 GB per
// qkv call); here each weight fragment serves 8 tokens. grid (ceil(rows/64),
// ceil(t/8)), block 128*KS; y[tok][row] with explicit stride; per-token math
// identical to gemv_fp4_mma_d (same k order, levels, KS reduce).
extern "C" __global__ void gemm_fp4_dense(const unsigned char* __restrict__ w,
                                          const unsigned char* __restrict__ xq,
                                          const float* __restrict__ gs_ptr,
                                          float* __restrict__ y,
                                          const int* __restrict__ k_dim_p,
                                          const int* __restrict__ rows_p,
                                          const int* __restrict__ y_stride_p,
                                          const int* __restrict__ t_p) {
    int k_dim = *k_dim_p;
    int bpr = k_dim >> 6;
    int rows = *rows_p;
    int ys = *y_stride_p;
    int tt = *t_p;
    int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    int ks_n = blockDim.x >> 7;
    int rg = warp & 3, ks = warp >> 2;
    int g = lane >> 2, lt = lane & 3;
    int w0 = (blockIdx.x << 6) + (rg << 4);
    bool active = w0 < rows;
    int tok_g = blockIdx.y * 8 + g;
    const unsigned char* xa = xq + (size_t)min(tok_g, tt - 1) * bpr * 108;
    float d0 = 0.0f, d1 = 0.0f, d2 = 0.0f, d3 = 0.0f;
    int r0 = w0 + g, r1 = w0 + g + 8;
    if (active) {
        const unsigned char* rowg = w + (size_t)min(r0, rows - 1) * bpr * 36;
        const unsigned char* rowg8 = w + (size_t)min(r1, rows - 1) * bpr * 36;
        const unsigned char* sfrow = (lt & 1) ? rowg8 : rowg;
        int bpb = (bpr + ks_n - 1) / ks_n;
        int b_lo = ks * bpb, b_hi = min(bpr, b_lo + bpb);
        for (int b = b_lo; b < b_hi; b++) {
            const unsigned char* blk = rowg + b * 36;
            const unsigned char* blk8 = rowg8 + b * 36;
            const unsigned char* ab = xa + b * 36;
            unsigned int sa = sfrow[b * 36 + 0] | (unsigned int)sfrow[b * 36 + 1] << 8
                            | (unsigned int)sfrow[b * 36 + 2] << 16 | (unsigned int)sfrow[b * 36 + 3] << 24;
            unsigned int a0 = *(const unsigned int*)(blk + 4 + 4 * lt);
            unsigned int a1 = *(const unsigned int*)(blk8 + 4 + 4 * lt);
            unsigned int a2 = *(const unsigned int*)(blk + 20 + 4 * lt);
            unsigned int a3 = *(const unsigned int*)(blk8 + 20 + 4 * lt);
            unsigned int sb = ab[0] | (unsigned int)ab[1] << 8
                            | (unsigned int)ab[2] << 16 | (unsigned int)ab[3] << 24;
            unsigned int b0 = *(const unsigned int*)(ab + 4 + 4 * lt);
            unsigned int b1 = *(const unsigned int*)(ab + 20 + 4 * lt);
            mma_fp4_16n8k64(d0, d1, d2, d3, a0, a1, a2, a3, b0, b1, sa, sb);
            #pragma unroll
            for (int lv = 1; lv < 3; lv++) {
                const unsigned char* ab2 = ab + lv * bpr * 36;
                unsigned int sb2 = ab2[0] | (unsigned int)ab2[1] << 8
                                 | (unsigned int)ab2[2] << 16 | (unsigned int)ab2[3] << 24;
                unsigned int c0 = *(const unsigned int*)(ab2 + 4 + 4 * lt);
                unsigned int c1 = *(const unsigned int*)(ab2 + 20 + 4 * lt);
                mma_fp4_16n8k64(d0, d1, d2, d3, a0, a1, a2, a3, c0, c1, sa, sb2);
            }
        }
    }
    __shared__ float red[4][4][32][4];
    red[ks][rg][lane][0] = d0; red[ks][rg][lane][1] = d1; red[ks][rg][lane][2] = d2; red[ks][rg][lane][3] = d3;
    __syncthreads();
    if (ks == 0 && active) {
        float s0 = 0.0f, s1 = 0.0f, s2 = 0.0f, s3 = 0.0f;
        for (int i = 0; i < ks_n; i++) {
            s0 += red[i][rg][lane][0]; s1 += red[i][rg][lane][1];
            s2 += red[i][rg][lane][2]; s3 += red[i][rg][lane][3];
        }
        float gs = gs_ptr[0];
        int n0 = blockIdx.y * 8 + 2 * lt, n1 = n0 + 1;
        if (n0 < tt) {
            if (r0 < rows) y[(size_t)n0 * ys + r0] = s0 * gs;
            if (r1 < rows) y[(size_t)n0 * ys + r1] = s2 * gs;
        }
        if (n1 < tt) {
            if (r0 < rows) y[(size_t)n1 * ys + r0] = s1 * gs;
            if (r1 < rows) y[(size_t)n1 * ys + r1] = s3 * gs;
        }
    }
}

// per-lane pick: word i (0..11) of a 48-byte window held as three uint4
__device__ __forceinline__ unsigned int wpick3(uint4 a, uint4 b, uint4 c, int i) {
    if (i < 4) { return (i == 0) ? a.x : (i == 1) ? a.y : (i == 2) ? a.z : a.w; }
    if (i < 8) { i -= 4; return (i == 0) ? b.x : (i == 1) ? b.y : (i == 2) ? b.z : b.w; }
    i -= 8; return (i == 0) ? c.x : (i == 1) ? c.y : (i == 2) ? c.z : c.w;
}

// Dense FP4 GEMM variant B (#10c, CROW_PF_GEMM_B exact 1): the 8-token
// gemm_fp4_dense form with the token tile widened 8 -> 32 (grid y = ceil(t/32))
// and the weight fragments loaded as 16-byte vectors. BIT-IDENTITY CONTRACT
// (the 8-token form's: "per-token math identical to gemv_fp4_mma_d, same k
// order, levels, KS reduce"): mma_fp4_16n8k64 is n8, so a 32-token tile is
// FOUR independent n-tiles per warp with four accumulator sets; tile j's mma
// at (block b, level lv) consumes EXACTLY the operand words the 8-token form's
// block (blockIdx.y = j) loaded: the same A words (the weight fragments load
// ONCE per b and serve all four tiles), the same B words (token 8j+g's
// quantized row), the same block scales, accumulated in the same order
// (b ascending over the same KS-split range, levels 0,1,2 per block), reduced
// through the same red[ks][rg][lane][4] smem form per tile in the same fixed
// slice order. 16-BYTE LOADS: the 36 B block layout puts block b at byte
// 36b = 16*(b/4) + 4*(b&3) inside the row, so a block is only 4B aligned;
// the vector path therefore loads the ALIGNED 48-byte window that contains
// the block (base = blk - 4*(b&3), three uint4) and picks the exact 4-byte
// fragment words out of the window (little-endian, so the packed scale word
// equals the old byte assembly). The window stays inside the row slab: it
// ends past the block only into the SAME row's next block, and the LAST
// block b = bpr-1 sits at line offset 12 (bpr % 4 == 0), where the window
// ends EXACTLY at the slab end. Requires bpr % 4 == 0 AND w 16B aligned;
// the scalar fallback keeps the 4-byte loads verbatim otherwise (no model
// k_dim hits it: every dense site is a multiple of 2560).
extern "C" __global__ void gemm_fp4_dense_b(const unsigned char* __restrict__ w,
                                            const unsigned char* __restrict__ xq,
                                            const float* __restrict__ gs_ptr,
                                            float* __restrict__ y,
                                            const int* __restrict__ k_dim_p,
                                            const int* __restrict__ rows_p,
                                            const int* __restrict__ y_stride_p,
                                            const int* __restrict__ t_p) {
    int k_dim = *k_dim_p;
    int bpr = k_dim >> 6;
    int rows = *rows_p;
    int ys = *y_stride_p;
    int tt = *t_p;
    int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    int ks_n = blockDim.x >> 7;
    int rg = warp & 3, ks = warp >> 2;
    int g = lane >> 2, lt = lane & 3;
    int w0 = (blockIdx.x << 6) + (rg << 4);
    bool active = w0 < rows;
    int tokg0 = blockIdx.y * 32;
    int r0 = w0 + g, r1 = w0 + g + 8;
    float d[4][4]; // [n-tile j][d0..d3]
    #pragma unroll
    for (int j = 0; j < 4; j++) { d[j][0] = 0.0f; d[j][1] = 0.0f; d[j][2] = 0.0f; d[j][3] = 0.0f; }
    if (active) {
        const unsigned char* rowg = w + (size_t)min(r0, rows - 1) * bpr * 36;
        const unsigned char* rowg8 = w + (size_t)min(r1, rows - 1) * bpr * 36;
        int bpb = (bpr + ks_n - 1) / ks_n;
        int b_lo = ks * bpb, b_hi = min(bpr, b_lo + bpb);
        bool va = ((bpr & 3) == 0) && ((((unsigned long long)w) & 15ull) == 0ull); // aligned-window guard
        for (int b = b_lo; b < b_hi; b++) {
            const unsigned char* blk = rowg + b * 36;
            const unsigned char* blk8 = rowg8 + b * 36;
            unsigned int a0, a1, a2, a3, sa;
            if (va) {
                // the block sits at line offset 4*(b&3); the aligned 48-byte
                // window at blk - 4*(b&3) contains it (see the header comment)
                int q = (int)(b & 3);
                const unsigned char* wb = blk - q * 4;
                const unsigned char* wb8 = blk8 - q * 4;
                uint4 v0 = *(const uint4*)(wb);
                uint4 v1 = *(const uint4*)(wb + 16);
                uint4 v2 = *(const uint4*)(wb + 32);
                uint4 u0 = *(const uint4*)(wb8);
                uint4 u1 = *(const uint4*)(wb8 + 16);
                uint4 u2 = *(const uint4*)(wb8 + 32);
                // window words: scale at q, frag k at q + 1 + k
                sa = (lt & 1) ? wpick3(u0, u1, u2, q) : wpick3(v0, v1, v2, q);
                int i1 = q + 1 + lt; // fragment lt
                int i5 = q + 5 + lt; // fragment lt + 4
                a0 = wpick3(v0, v1, v2, i1);
                a2 = wpick3(v0, v1, v2, i5);
                a1 = wpick3(u0, u1, u2, i1);
                a3 = wpick3(u0, u1, u2, i5);
            } else {
                const unsigned char* sfrow = (lt & 1) ? blk8 : blk;
                a0 = *(const unsigned int*)(blk + 4 + 4 * lt);
                a1 = *(const unsigned int*)(blk8 + 4 + 4 * lt);
                a2 = *(const unsigned int*)(blk + 20 + 4 * lt);
                a3 = *(const unsigned int*)(blk8 + 20 + 4 * lt);
                sa = sfrow[0] | (unsigned int)sfrow[1] << 8
                   | (unsigned int)sfrow[2] << 16 | (unsigned int)sfrow[3] << 24;
            }
            #pragma unroll
            for (int j = 0; j < 4; j++) {
                int tok_g = tokg0 + 8 * j + g;
                const unsigned char* ab = xq + (size_t)min(tok_g, tt - 1) * bpr * 108 + b * 36;
                unsigned int sb = *(const unsigned int*)(ab);
                unsigned int b0 = *(const unsigned int*)(ab + 4 + 4 * lt);
                unsigned int b1 = *(const unsigned int*)(ab + 20 + 4 * lt);
                mma_fp4_16n8k64(d[j][0], d[j][1], d[j][2], d[j][3], a0, a1, a2, a3, b0, b1, sa, sb);
                #pragma unroll
                for (int lv = 1; lv < 3; lv++) {
                    const unsigned char* ab2 = ab + lv * bpr * 36;
                    unsigned int sb2 = *(const unsigned int*)(ab2);
                    unsigned int c0 = *(const unsigned int*)(ab2 + 4 + 4 * lt);
                    unsigned int c1 = *(const unsigned int*)(ab2 + 20 + 4 * lt);
                    mma_fp4_16n8k64(d[j][0], d[j][1], d[j][2], d[j][3], a0, a1, a2, a3, c0, c1, sa, sb2);
                }
            }
        }
    }
    __shared__ float red[4][4][32][4];
    #pragma unroll
    for (int j = 0; j < 4; j++) {
        red[ks][rg][lane][0] = d[j][0]; red[ks][rg][lane][1] = d[j][1];
        red[ks][rg][lane][2] = d[j][2]; red[ks][rg][lane][3] = d[j][3];
        __syncthreads();
        if (ks == 0 && active) {
            float s0 = 0.0f, s1 = 0.0f, s2 = 0.0f, s3 = 0.0f;
            for (int i = 0; i < ks_n; i++) {
                s0 += red[i][rg][lane][0]; s1 += red[i][rg][lane][1];
                s2 += red[i][rg][lane][2]; s3 += red[i][rg][lane][3];
            }
            float gs = gs_ptr[0];
            int n0 = tokg0 + 8 * j + 2 * lt, n1 = n0 + 1;
            if (n0 < tt) {
                if (r0 < rows) y[(size_t)n0 * ys + r0] = s0 * gs;
                if (r1 < rows) y[(size_t)n0 * ys + r1] = s2 * gs;
            }
            if (n1 < tt) {
                if (r0 < rows) y[(size_t)n1 * ys + r0] = s1 * gs;
                if (r1 < rows) y[(size_t)n1 * ys + r1] = s3 * gs;
            }
        }
        __syncthreads();
    }
}

// Grouped dense FP4 GEMV (#62b lever 1): the four GDN decode input projections
// (qkv 10240 + z 6144 + b 48 + a 48 rows, all k 2560) in ONE launch over ONE
// shared quantized activation row. Four (w, gs, y, rows) groups; blockIdx.x
// maps into the group by row range at 16-row warp granularity, so every group
// row count must be a multiple of 16 (true for 10240/6144/48/48) and no
// warp's A fragment ever spans two groups. Per-row math is the
// gemv_fp4_mma_d form BIT FOR BIT: same bpr k loop in the same order, same
// levels, same KS smem reduce in the fixed slice order, the group's OWN
// per-slab gs at the store (a single-gs fusion would NOT be bit exact, 62a
// report C5). Only the launch shape changes (one ceil(16480/64)=258-block
// grid instead of 160/96/1/1), the KS=1 contract class. grid
// (ceil(total_rows/64), T), block 128*KS (gen::mma_bx()).
extern "C" __global__ void gemv_fp4_mma_g(const unsigned char* __restrict__ w0,
                                          const unsigned char* __restrict__ w1,
                                          const unsigned char* __restrict__ w2,
                                          const unsigned char* __restrict__ w3,
                                          const float* __restrict__ gs0,
                                          const float* __restrict__ gs1,
                                          const float* __restrict__ gs2,
                                          const float* __restrict__ gs3,
                                          float* __restrict__ y0,
                                          float* __restrict__ y1,
                                          float* __restrict__ y2,
                                          float* __restrict__ y3,
                                          const int* __restrict__ nr0_p,
                                          const int* __restrict__ nr1_p,
                                          const int* __restrict__ nr2_p,
                                          const int* __restrict__ nr3_p,
                                          const unsigned char* __restrict__ xq,
                                          const int* __restrict__ k_dim_p) {
    int k_dim = *k_dim_p;
    int bpr = k_dim >> 6;
    int nr0 = *nr0_p, nr1 = *nr1_p, nr2 = *nr2_p, nr3 = *nr3_p;
    int s1 = nr0, s2 = s1 + nr1, s3 = s2 + nr2, total = s3 + nr3;
    int tok = blockIdx.y;
    int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    int ks_n = blockDim.x >> 7;
    int rg = warp & 3, ks = warp >> 2;
    int g = lane >> 2, lt = lane & 3;
    int wr = (blockIdx.x << 6) + (rg << 4);   // this warp's 16-row base, concat row space
    const unsigned char* w; const float* gsp; float* y; int gstart, grows;
    if (wr < s1)      { w = w0; gsp = gs0; y = y0; gstart = 0;  grows = nr0; }
    else if (wr < s2) { w = w1; gsp = gs1; y = y1; gstart = s1; grows = nr1; }
    else if (wr < s3) { w = w2; gsp = gs2; y = y2; gstart = s2; grows = nr2; }
    else              { w = w3; gsp = gs3; y = y3; gstart = s3; grows = nr3; }
    int lr = wr - gstart;     // row base inside the group
    bool active = wr < total; // whole-warp guard; no early return (smem barrier below)
    float d0 = 0.0f, d1 = 0.0f, d2 = 0.0f, d3 = 0.0f;
    int r0 = lr + g, r1 = lr + g + 8;
    if (active) {
        const unsigned char* xa = xq + (size_t)tok * bpr * 108;
        const unsigned char* rowg = w + (size_t)min(r0, grows - 1) * bpr * 36;
        const unsigned char* rowg8 = w + (size_t)min(r1, grows - 1) * bpr * 36;
        const unsigned char* sfrow = (lt & 1) ? rowg8 : rowg;
        int bpb = (bpr + ks_n - 1) / ks_n;
        int b_lo = ks * bpb, b_hi = min(bpr, b_lo + bpb);
        for (int b = b_lo; b < b_hi; b++) {
            const unsigned char* blk = rowg + b * 36;
            const unsigned char* blk8 = rowg8 + b * 36;
            const unsigned char* ab = xa + b * 36;
            unsigned int sa = sfrow[b * 36 + 0] | (unsigned int)sfrow[b * 36 + 1] << 8
                            | (unsigned int)sfrow[b * 36 + 2] << 16 | (unsigned int)sfrow[b * 36 + 3] << 24;
            unsigned int a0 = *(const unsigned int*)(blk + 4 + 4 * lt);
            unsigned int a1 = *(const unsigned int*)(blk8 + 4 + 4 * lt);
            unsigned int a2 = *(const unsigned int*)(blk + 20 + 4 * lt);
            unsigned int a3 = *(const unsigned int*)(blk8 + 20 + 4 * lt);
            unsigned int sb = ab[0] | (unsigned int)ab[1] << 8
                            | (unsigned int)ab[2] << 16 | (unsigned int)ab[3] << 24;
            unsigned int b0 = *(const unsigned int*)(ab + 4 + 4 * lt);
            unsigned int b1 = *(const unsigned int*)(ab + 20 + 4 * lt);
            mma_fp4_16n8k64(d0, d1, d2, d3, a0, a1, a2, a3, b0, b1, sa, sb);
            #pragma unroll
            for (int lv = 1; lv < 3; lv++) {
                const unsigned char* ab2 = ab + lv * bpr * 36; // residual level
                unsigned int sb2 = ab2[0] | (unsigned int)ab2[1] << 8
                                 | (unsigned int)ab2[2] << 16 | (unsigned int)ab2[3] << 24;
                unsigned int c0 = *(const unsigned int*)(ab2 + 4 + 4 * lt);
                unsigned int c1 = *(const unsigned int*)(ab2 + 20 + 4 * lt);
                mma_fp4_16n8k64(d0, d1, d2, d3, a0, a1, a2, a3, c0, c1, sa, sb2);
            }
        }
    }
    __shared__ float red[4][2][64];
    if (lt == 0) { red[ks][0][(rg << 4) + g] = d0; red[ks][1][(rg << 4) + g] = d2; }
    __syncthreads();
    if (ks == 0 && lt == 0 && active) {
        float gs = gsp[0];
        float s0 = 0.0f, s2 = 0.0f;
        for (int i = 0; i < ks_n; i++) { s0 += red[i][0][(rg << 4) + g]; s2 += red[i][1][(rg << 4) + g]; }
        if (r0 < grows) y[(size_t)tok * grows + r0] = s0 * gs;
        if (r1 < grows) y[(size_t)tok * grows + r1] = s2 * gs;
    }
}

// 32-rows-per-block twin of gemv_fp4_mma_d (#62d lever 2): the GDN decode
// launches (the per-slab qkv/z fallback and the out projection) run
// ceil(rows/32) blocks instead of rows/64, so a block holds TWO 16-row
// groups instead of four and the grid doubles (160 -> 320, 96 -> 192,
// 40 -> 80). PER-ROW MATH IS BIT-IDENTICAL to the 64-row kernel: the k
// split stays CROW_MMA_KS slices (ks_n = blockDim >> 6, block 64*KS =
// gen::mma_bx32(); default KS 4 -> 256 threads), the bpr loop bounds, the
// mma order, the residual levels and the FIXED smem slice reduce order are
// the gemv_fp4_mma_d forms verbatim; only the warp map moves (rg = warp & 1,
// ks = warp >> 1, w0 = blockIdx.x*32 + rg*16). The non-GDN users of
// gemv_fp4_mma_d (attention, qsa, shared expert, PLE) keep the 64-row
// geometry, so the pairs measure the GDN lever only.
extern "C" __global__ void gemv_fp4_mma_d32(const unsigned char* __restrict__ w,
                                           const unsigned char* __restrict__ xq,
                                           const float* __restrict__ gs_ptr,
                                           float* __restrict__ y,
                                           const int* __restrict__ k_dim_p,
                                           const int* __restrict__ rows_p,
                                           const int* __restrict__ y_stride_p) {
    int k_dim = *k_dim_p;
    int bpr = k_dim >> 6;
    int rows = *rows_p;
    int ys = *y_stride_p;
    int tok = blockIdx.y;
    int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    int ks_n = blockDim.x >> 6;
    int rg = warp & 1, ks = warp >> 1;
    int g = lane >> 2, lt = lane & 3;
    int w0 = (blockIdx.x << 5) + (rg << 4);
    bool active = w0 < rows;   // whole-warp guard; no early return (smem barrier below)
    float d0 = 0.0f, d1 = 0.0f, d2 = 0.0f, d3 = 0.0f;
    int r0 = w0 + g, r1 = w0 + g + 8;
    if (active) {
        const unsigned char* xa = xq + (size_t)tok * bpr * 108;
        const unsigned char* rowg = w + (size_t)min(r0, rows - 1) * bpr * 36;
        const unsigned char* rowg8 = w + (size_t)min(r1, rows - 1) * bpr * 36;
        const unsigned char* sfrow = (lt & 1) ? rowg8 : rowg;
        int bpb = (bpr + ks_n - 1) / ks_n;
        int b_lo = ks * bpb, b_hi = min(bpr, b_lo + bpb);
        for (int b = b_lo; b < b_hi; b++) {
            const unsigned char* blk = rowg + b * 36;
            const unsigned char* blk8 = rowg8 + b * 36;
            const unsigned char* ab = xa + b * 36;
            unsigned int sa = sfrow[b * 36 + 0] | (unsigned int)sfrow[b * 36 + 1] << 8
                            | (unsigned int)sfrow[b * 36 + 2] << 16 | (unsigned int)sfrow[b * 36 + 3] << 24;
            unsigned int a0 = *(const unsigned int*)(blk + 4 + 4 * lt);
            unsigned int a1 = *(const unsigned int*)(blk8 + 4 + 4 * lt);
            unsigned int a2 = *(const unsigned int*)(blk + 20 + 4 * lt);
            unsigned int a3 = *(const unsigned int*)(blk8 + 20 + 4 * lt);
            unsigned int sb = ab[0] | (unsigned int)ab[1] << 8
                            | (unsigned int)ab[2] << 16 | (unsigned int)ab[3] << 24;
            unsigned int b0 = *(const unsigned int*)(ab + 4 + 4 * lt);
            unsigned int b1 = *(const unsigned int*)(ab + 20 + 4 * lt);
            mma_fp4_16n8k64(d0, d1, d2, d3, a0, a1, a2, a3, b0, b1, sa, sb);
            #pragma unroll
            for (int lv = 1; lv < 3; lv++) {
                const unsigned char* ab2 = ab + lv * bpr * 36; // residual level
                unsigned int sb2 = ab2[0] | (unsigned int)ab2[1] << 8
                                 | (unsigned int)ab2[2] << 16 | (unsigned int)ab2[3] << 24;
                unsigned int c0 = *(const unsigned int*)(ab2 + 4 + 4 * lt);
                unsigned int c1 = *(const unsigned int*)(ab2 + 20 + 4 * lt);
                mma_fp4_16n8k64(d0, d1, d2, d3, a0, a1, a2, a3, c0, c1, sa, sb2);
            }
        }
    }
    __shared__ float red[4][2][64];
    if (lt == 0) { red[ks][0][(rg << 4) + g] = d0; red[ks][1][(rg << 4) + g] = d2; }
    __syncthreads();
    float gs = gs_ptr[0];
    if (ks == 0 && lt == 0 && active) {
        float s0 = 0.0f, s2 = 0.0f;
        for (int i = 0; i < ks_n; i++) { s0 += red[i][0][(rg << 4) + g]; s2 += red[i][1][(rg << 4) + g]; }
        if (r0 < rows) y[(size_t)tok * ys + r0] = s0 * gs;
        if (r1 < rows) y[(size_t)tok * ys + r1] = s2 * gs;
    }
}

// 32-rows-per-block twin of gemv_fp4_mma_g (#62d lever 2): the grouped GDN
// input-projection launch runs ceil(16480/32) = 515 blocks instead of 258
// (16480 = 32*515 exactly, so every block is full; the whole-warp guard
// stays for generality). The per-warp group selection by the warp's own
// 16-row concat-space base is unchanged, so the two row groups of a block
// may sit in different groups exactly as in the 64-row form (the 48-row b
// and a slabs never aligned to blocks). Per-row math BIT-IDENTICAL: same
// k split (ks_n = blockDim >> 6), same fixed smem reduce, same per-group gs
// at the store. grid (ceil(total_rows/32), T), block 64*KS (mma_bx32()).
extern "C" __global__ void gemv_fp4_mma_g32(const unsigned char* __restrict__ w0,
                                           const unsigned char* __restrict__ w1,
                                           const unsigned char* __restrict__ w2,
                                           const unsigned char* __restrict__ w3,
                                           const float* __restrict__ gs0,
                                           const float* __restrict__ gs1,
                                           const float* __restrict__ gs2,
                                           const float* __restrict__ gs3,
                                           float* __restrict__ y0,
                                           float* __restrict__ y1,
                                           float* __restrict__ y2,
                                           float* __restrict__ y3,
                                           const int* __restrict__ nr0_p,
                                           const int* __restrict__ nr1_p,
                                           const int* __restrict__ nr2_p,
                                           const int* __restrict__ nr3_p,
                                           const unsigned char* __restrict__ xq,
                                           const int* __restrict__ k_dim_p) {
    int k_dim = *k_dim_p;
    int bpr = k_dim >> 6;
    int nr0 = *nr0_p, nr1 = *nr1_p, nr2 = *nr2_p, nr3 = *nr3_p;
    int s1 = nr0, s2 = s1 + nr1, s3 = s2 + nr2, total = s3 + nr3;
    int tok = blockIdx.y;
    int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    int ks_n = blockDim.x >> 6;
    int rg = warp & 1, ks = warp >> 1;
    int g = lane >> 2, lt = lane & 3;
    int wr = (blockIdx.x << 5) + (rg << 4);   // this warp's 16-row base, concat row space
    const unsigned char* w; const float* gsp; float* y; int gstart, grows;
    if (wr < s1)      { w = w0; gsp = gs0; y = y0; gstart = 0;  grows = nr0; }
    else if (wr < s2) { w = w1; gsp = gs1; y = y1; gstart = s1; grows = nr1; }
    else if (wr < s3) { w = w2; gsp = gs2; y = y2; gstart = s2; grows = nr2; }
    else              { w = w3; gsp = gs3; y = y3; gstart = s3; grows = nr3; }
    int lr = wr - gstart;     // row base inside the group
    bool active = wr < total; // whole-warp guard; no early return (smem barrier below)
    float d0 = 0.0f, d1 = 0.0f, d2 = 0.0f, d3 = 0.0f;
    int r0 = lr + g, r1 = lr + g + 8;
    if (active) {
        const unsigned char* xa = xq + (size_t)tok * bpr * 108;
        const unsigned char* rowg = w + (size_t)min(r0, grows - 1) * bpr * 36;
        const unsigned char* rowg8 = w + (size_t)min(r1, grows - 1) * bpr * 36;
        const unsigned char* sfrow = (lt & 1) ? rowg8 : rowg;
        int bpb = (bpr + ks_n - 1) / ks_n;
        int b_lo = ks * bpb, b_hi = min(bpr, b_lo + bpb);
        for (int b = b_lo; b < b_hi; b++) {
            const unsigned char* blk = rowg + b * 36;
            const unsigned char* blk8 = rowg8 + b * 36;
            const unsigned char* ab = xa + b * 36;
            unsigned int sa = sfrow[b * 36 + 0] | (unsigned int)sfrow[b * 36 + 1] << 8
                            | (unsigned int)sfrow[b * 36 + 2] << 16 | (unsigned int)sfrow[b * 36 + 3] << 24;
            unsigned int a0 = *(const unsigned int*)(blk + 4 + 4 * lt);
            unsigned int a1 = *(const unsigned int*)(blk8 + 4 + 4 * lt);
            unsigned int a2 = *(const unsigned int*)(blk + 20 + 4 * lt);
            unsigned int a3 = *(const unsigned int*)(blk8 + 20 + 4 * lt);
            unsigned int sb = ab[0] | (unsigned int)ab[1] << 8
                            | (unsigned int)ab[2] << 16 | (unsigned int)ab[3] << 24;
            unsigned int b0 = *(const unsigned int*)(ab + 4 + 4 * lt);
            unsigned int b1 = *(const unsigned int*)(ab + 20 + 4 * lt);
            mma_fp4_16n8k64(d0, d1, d2, d3, a0, a1, a2, a3, b0, b1, sa, sb);
            #pragma unroll
            for (int lv = 1; lv < 3; lv++) {
                const unsigned char* ab2 = ab + lv * bpr * 36; // residual level
                unsigned int sb2 = ab2[0] | (unsigned int)ab2[1] << 8
                                 | (unsigned int)ab2[2] << 16 | (unsigned int)ab2[3] << 24;
                unsigned int c0 = *(const unsigned int*)(ab2 + 4 + 4 * lt);
                unsigned int c1 = *(const unsigned int*)(ab2 + 20 + 4 * lt);
                mma_fp4_16n8k64(d0, d1, d2, d3, a0, a1, a2, a3, c0, c1, sa, sb2);
            }
        }
    }
    __shared__ float red[4][2][64];
    if (lt == 0) { red[ks][0][(rg << 4) + g] = d0; red[ks][1][(rg << 4) + g] = d2; }
    __syncthreads();
    if (ks == 0 && lt == 0 && active) {
        float gs = gsp[0];
        float s0 = 0.0f, s2 = 0.0f;
        for (int i = 0; i < ks_n; i++) { s0 += red[i][0][(rg << 4) + g]; s2 += red[i][1][(rg << 4) + g]; }
        if (r0 < grows) y[(size_t)tok * grows + r0] = s0 * gs;
        if (r1 < grows) y[(size_t)tok * grows + r1] = s2 * gs;
    }
}

// same as gemv_fp4_b but with an EXPLICIT y row stride — used for the shared
// expert's gate|up pair writing into ONE [t][1280] buffer (p13 layout, the
// contract silu_mul640 reads). The plain gemv_fp4_b keeps the [t][rows]
// compact layout used by every other call site.
extern "C" __global__ void gemv_fp4_bs(const unsigned char* __restrict__ w, const float* __restrict__ x,
                                       const float* __restrict__ gs_ptr, float* __restrict__ y,
                                       const int* __restrict__ k_dim_p, const int* __restrict__ y_stride_p) {
    int k_dim = *k_dim_p;
    int bpr = k_dim >> 6;
    int row = blockIdx.x;
    int t = blockIdx.y;
    int ys = *y_stride_p;
    const unsigned char* rowp = w + (size_t)row * bpr * 36;
    const float* xp = x + (size_t)t * k_dim;
    float gs = gs_ptr[0];
    float acc = 0.0f;
    for (int b = threadIdx.x; b < bpr; b += blockDim.x) {
        const unsigned char* blk = rowp + b * 36;
        #pragma unroll
        for (int sb = 0; sb < 4; sb++) {
            float s = ue4m3(blk[sb]) * gs;
            float part = 0.0f;
            #pragma unroll
            for (int j = 0; j < 16; j++) {
                int idx = sb * 16 + j;
                unsigned int byte = blk[4 + (idx >> 1)];
                unsigned int nib = (idx & 1) ? ((byte >> 4) & 0xF) : (byte & 0xF);
                part += e2m1(nib) * xp[b * 64 + sb * 16 + j];
            }
            acc += part * s;
        }
    }
    __shared__ float red2[256];
    red2[threadIdx.x] = acc;
    __syncthreads();
    for (int st = blockDim.x / 2; st > 0; st >>= 1) {
        if (threadIdx.x < st) red2[threadIdx.x] += red2[threadIdx.x + st];
        __syncthreads();
    }
    if (threadIdx.x == 0) y[(size_t)t * ys + row] = red2[0];
}

// BF16-weight GEMV (untied lm_head, router — BF16 keeps), single row
extern "C" __global__ void gemv_bf16(const unsigned short* __restrict__ w, const float* __restrict__ x,
                                     float* __restrict__ y, const int* __restrict__ k_dim_p) {
    int k_dim = *k_dim_p;
    int row = blockIdx.x;
    const unsigned short* wp = w + (size_t)row * k_dim;
    float acc = 0.0f;
    for (int i = threadIdx.x; i < k_dim; i += blockDim.x)
        acc += __int_as_float(((unsigned int)wp[i]) << 16) * x[i];
    __shared__ float red[256];
    red[threadIdx.x] = acc;
    __syncthreads();
    for (int st = blockDim.x / 2; st > 0; st >>= 1) {
        if (threadIdx.x < st) red[threadIdx.x] += red[threadIdx.x + st];
        __syncthreads();
    }
    if (threadIdx.x == 0) y[row] = red[0];
}

// BF16-weight GEMV batched over token rows: grid (rows, T)
extern "C" __global__ void gemv_bf16_b(const unsigned short* __restrict__ w, const float* __restrict__ x,
                                       float* __restrict__ y, const int* __restrict__ k_dim_p) {
    int k_dim = *k_dim_p;
    int row = blockIdx.x;
    int t = blockIdx.y;
    const unsigned short* wp = w + (size_t)row * k_dim;
    const float* xp = x + (size_t)t * k_dim;
    float acc = 0.0f;
    for (int i = threadIdx.x; i < k_dim; i += blockDim.x)
        acc += __int_as_float(((unsigned int)wp[i]) << 16) * xp[i];
    __shared__ float red[256];
    red[threadIdx.x] = acc;
    __syncthreads();
    for (int st = blockDim.x / 2; st > 0; st >>= 1) {
        if (threadIdx.x < st) red[threadIdx.x] += red[threadIdx.x + st];
        __syncthreads();
    }
    if (threadIdx.x == 0) y[(size_t)t * gridDim.x + row] = red[0];
}

// #77: gemv_bf16_b with an EXPLICIT y row stride — the BF16 twin of gemv_fp4_bs, for the
// shared expert's gate|up pair writing into ONE [t][1280] buffer (the layout silu_mul640
// reads) when the dense overlay shadows those two tensors. Body verbatim gemv_bf16_b; only
// the store address differs, exactly as gemv_fp4_bs differs from gemv_fp4_b.
extern "C" __global__ void gemv_bf16_bs(const unsigned short* __restrict__ w, const float* __restrict__ x,
                                        float* __restrict__ y, const int* __restrict__ k_dim_p,
                                        const int* __restrict__ y_stride_p) {
    int k_dim = *k_dim_p;
    int ys = *y_stride_p;
    int row = blockIdx.x;
    int t = blockIdx.y;
    const unsigned short* wp = w + (size_t)row * k_dim;
    const float* xp = x + (size_t)t * k_dim;
    float acc = 0.0f;
    for (int i = threadIdx.x; i < k_dim; i += blockDim.x)
        acc += __int_as_float(((unsigned int)wp[i]) << 16) * xp[i];
    __shared__ float redbs[256];
    redbs[threadIdx.x] = acc;
    __syncthreads();
    for (int st = blockDim.x / 2; st > 0; st >>= 1) {
        if (threadIdx.x < st) redbs[threadIdx.x] += redbs[threadIdx.x + st];
        __syncthreads();
    }
    if (threadIdx.x == 0) y[(size_t)t * ys + row] = redbs[0];
}

// BF16-weight GEMV, one WARP per row, 16-byte vector loads (8 bf16 per lane
// per step; k_dim % 8 == 0 for every keep shape: 320 / 2560 / 10240). Replaces
// the block-per-row 2-byte-load kernels for the BF16 keeps (HC down/up, q/k,
// lm_head). grid (ceil(rows/8), T), block 256 = 8 rows; y[t][row] compact.
extern "C" __global__ void gemv_bf16_w(const unsigned short* __restrict__ w, const float* __restrict__ x,
                                       float* __restrict__ y, const int* __restrict__ k_dim_p,
                                       const int* __restrict__ rows_p) {
    int k_dim = *k_dim_p;
    int rows = *rows_p;
    int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    int row = blockIdx.x * 8 + warp;
    int t = blockIdx.y;
    if (row >= rows) return;
    const unsigned short* wp = w + (size_t)row * k_dim;
    const float* xp = x + (size_t)t * k_dim;
    float acc = 0.0f;
    for (int i = lane * 8; i < k_dim; i += 256) {
        uint4 v = *(const uint4*)(wp + i);
        unsigned int u[4] = {v.x, v.y, v.z, v.w};
        #pragma unroll
        for (int j = 0; j < 4; j++) {
            float lo = __uint_as_float(u[j] << 16);
            float hi = __uint_as_float(u[j] & 0xFFFF0000u);
            acc += lo * xp[i + 2 * j] + hi * xp[i + 2 * j + 1];
        }
    }
    for (int o = 16; o > 0; o >>= 1) acc += __shfl_down_sync(0xffffffffu, acc, o);
    if (lane == 0) y[(size_t)t * rows + row] = acc;
}

// BF16-keep GEMM for prefill: mma.sync m16n8k16 bf16 (f32 accumulate), 8
// tokens per tile. The f32 activation enters as hi + lo bf16 pair (two mma
// per k-step) so the keep tensors stay effectively f32-exact on the
// activation side (~16 mantissa bits). grid (ceil(rows/64), ceil(t/8)),
// block 128 (4 warps x 16 rows); y[tok][row] compact (stride = rows).
__device__ __forceinline__ void mma_bf16_16n8k16(float& d0, float& d1, float& d2, float& d3,
                                                 unsigned int a0, unsigned int a1, unsigned int a2,
                                                 unsigned int a3, unsigned int b0, unsigned int b1) {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
        "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};"
        : "+f"(d0), "+f"(d1), "+f"(d2), "+f"(d3)
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}
__device__ __forceinline__ unsigned int bf16_pair_hi(float x0, float x1) {
    return (unsigned int)f32_bf16_bits(x0) | ((unsigned int)f32_bf16_bits(x1) << 16);
}
__device__ __forceinline__ float bf16_bits_f32(unsigned short b) {
    return __uint_as_float(((unsigned int)b) << 16);
}
extern "C" __global__ void gemm_bf16_dense(const unsigned short* __restrict__ w, const float* __restrict__ x,
                                           float* __restrict__ y, const int* __restrict__ k_dim_p,
                                           const int* __restrict__ rows_p, const int* __restrict__ t_p) {
    int k_dim = *k_dim_p;
    int rows = *rows_p;
    int tt = *t_p;
    int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    int g = lane >> 2, t = lane & 3;
    int w0 = (blockIdx.x << 6) + (warp << 4);
    if (w0 >= rows) return;
    int r0 = w0 + g, r1 = w0 + g + 8;
    const unsigned short* wr0 = w + (size_t)min(r0, rows - 1) * k_dim;
    const unsigned short* wr1 = w + (size_t)min(r1, rows - 1) * k_dim;
    int tok = blockIdx.y * 8 + g;
    const float* xp = x + (size_t)min(tok, tt - 1) * k_dim;
    float d0 = 0.0f, d1 = 0.0f, d2 = 0.0f, d3 = 0.0f;
    for (int kk = 0; kk < k_dim; kk += 16) {
        unsigned int a0 = *(const unsigned int*)(wr0 + kk + 2 * t);
        unsigned int a1 = *(const unsigned int*)(wr1 + kk + 2 * t);
        unsigned int a2 = *(const unsigned int*)(wr0 + kk + 2 * t + 8);
        unsigned int a3 = *(const unsigned int*)(wr1 + kk + 2 * t + 8);
        float x0 = xp[kk + 2 * t], x1 = xp[kk + 2 * t + 1];
        float x2 = xp[kk + 2 * t + 8], x3 = xp[kk + 2 * t + 9];
        unsigned short h0 = f32_bf16_bits(x0), h1 = f32_bf16_bits(x1), h2 = f32_bf16_bits(x2), h3 = f32_bf16_bits(x3);
        unsigned int b0 = (unsigned int)h0 | ((unsigned int)h1 << 16);
        unsigned int b1 = (unsigned int)h2 | ((unsigned int)h3 << 16);
        mma_bf16_16n8k16(d0, d1, d2, d3, a0, a1, a2, a3, b0, b1);
        // residual (lo) pass: what bf16 rounding of the activation dropped
        unsigned int l0 = bf16_pair_hi(x0 - bf16_bits_f32(h0), x1 - bf16_bits_f32(h1));
        unsigned int l1 = bf16_pair_hi(x2 - bf16_bits_f32(h2), x3 - bf16_bits_f32(h3));
        mma_bf16_16n8k16(d0, d1, d2, d3, a0, a1, a2, a3, l0, l1);
    }
    int n0 = blockIdx.y * 8 + 2 * t, n1 = n0 + 1;
    if (n0 < tt) {
        if (r0 < rows) y[(size_t)n0 * rows + r0] = d0;
        if (r1 < rows) y[(size_t)n0 * rows + r1] = d2;
    }
    if (n1 < tt) {
        if (r0 < rows) y[(size_t)n1 * rows + r0] = d1;
        if (r1 < rows) y[(size_t)n1 * rows + r1] = d3;
    }
}

// Dense bf16 GEMM variant B (#10c, CROW_PF_GEMM_B exact 1): the 8-token
// gemm_bf16_dense form with the token tile widened 8 -> 32 (grid y =
// ceil(t/32)) and the weight words loaded as 16-byte vectors. BIT-IDENTITY
// CONTRACT: mma_bf16_16n8k16 is k16 n8, so a 32-token tile is four
// independent n-tiles per warp with four accumulator sets; tile j's mma at
// k step kk consumes EXACTLY the operand words the 8-token form's block
// (blockIdx.y = j) loaded: the same A words (the weight uint4 pair loads
// once per kk and serves all four tiles; word t of the lo vector is a0,
// word t of the hi vector is a2, per row exactly the four 4-byte loads of
// the 8-token form) and the same per-token B words. The per-token f32-to-bf16 conversion and the
// per-token residual second mma stay per token, per tile, in the same kk
// ascending order with the main mma before the residual mma (the 8-token
// form's exact sequence). Requires k_dim % 8 == 0 for the uint4 alignment of
// wr = w + row*k_dim (every dense bf16 site is 2560 or 12288); the scalar
// fallback keeps the 4-byte loads verbatim otherwise.
extern "C" __global__ void gemm_bf16_dense_b(const unsigned short* __restrict__ w, const float* __restrict__ x,
                                             float* __restrict__ y, const int* __restrict__ k_dim_p,
                                             const int* __restrict__ rows_p, const int* __restrict__ t_p) {
    int k_dim = *k_dim_p;
    int rows = *rows_p;
    int tt = *t_p;
    int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    int g = lane >> 2, t = lane & 3;
    int w0 = (blockIdx.x << 6) + (warp << 4);
    if (w0 >= rows) return;
    int r0 = w0 + g, r1 = w0 + g + 8;
    const unsigned short* wr0 = w + (size_t)min(r0, rows - 1) * k_dim;
    const unsigned short* wr1 = w + (size_t)min(r1, rows - 1) * k_dim;
    int tokg0 = blockIdx.y * 32;
    bool va = ((k_dim & 7) == 0) && ((((unsigned long long)w) & 15ull) == 0ull); // aligned-window guard
    float d[4][4]; // [n-tile j][d0..d3]
    #pragma unroll
    for (int j = 0; j < 4; j++) { d[j][0] = 0.0f; d[j][1] = 0.0f; d[j][2] = 0.0f; d[j][3] = 0.0f; }
    for (int kk = 0; kk < k_dim; kk += 16) {
        unsigned int a0, a1, a2, a3;
        if (va) {
            uint4 vlo0 = *(const uint4*)(wr0 + kk);
            uint4 vlo1 = *(const uint4*)(wr0 + kk + 8);
            uint4 vhi0 = *(const uint4*)(wr1 + kk);
            uint4 vhi1 = *(const uint4*)(wr1 + kk + 8);
            a0 = (t == 0) ? vlo0.x : (t == 1) ? vlo0.y : (t == 2) ? vlo0.z : vlo0.w;
            a1 = (t == 0) ? vhi0.x : (t == 1) ? vhi0.y : (t == 2) ? vhi0.z : vhi0.w;
            a2 = (t == 0) ? vlo1.x : (t == 1) ? vlo1.y : (t == 2) ? vlo1.z : vlo1.w;
            a3 = (t == 0) ? vhi1.x : (t == 1) ? vhi1.y : (t == 2) ? vhi1.z : vhi1.w;
        } else {
            a0 = *(const unsigned int*)(wr0 + kk + 2 * t);
            a1 = *(const unsigned int*)(wr1 + kk + 2 * t);
            a2 = *(const unsigned int*)(wr0 + kk + 2 * t + 8);
            a3 = *(const unsigned int*)(wr1 + kk + 2 * t + 8);
        }
        #pragma unroll
        for (int j = 0; j < 4; j++) {
            int tok = tokg0 + 8 * j + g;
            const float* xp = x + (size_t)min(tok, tt - 1) * k_dim;
            float x0 = xp[kk + 2 * t], x1 = xp[kk + 2 * t + 1];
            float x2 = xp[kk + 2 * t + 8], x3 = xp[kk + 2 * t + 9];
            unsigned short h0 = f32_bf16_bits(x0), h1 = f32_bf16_bits(x1), h2 = f32_bf16_bits(x2), h3 = f32_bf16_bits(x3);
            unsigned int b0 = (unsigned int)h0 | ((unsigned int)h1 << 16);
            unsigned int b1 = (unsigned int)h2 | ((unsigned int)h3 << 16);
            mma_bf16_16n8k16(d[j][0], d[j][1], d[j][2], d[j][3], a0, a1, a2, a3, b0, b1);
            // residual (lo) pass: what bf16 rounding of the activation dropped
            unsigned int l0 = bf16_pair_hi(x0 - bf16_bits_f32(h0), x1 - bf16_bits_f32(h1));
            unsigned int l1 = bf16_pair_hi(x2 - bf16_bits_f32(h2), x3 - bf16_bits_f32(h3));
            mma_bf16_16n8k16(d[j][0], d[j][1], d[j][2], d[j][3], a0, a1, a2, a3, l0, l1);
        }
    }
    #pragma unroll
    for (int j = 0; j < 4; j++) {
        int n0 = tokg0 + 8 * j + 2 * t, n1 = n0 + 1;
        if (n0 < tt) {
            if (r0 < rows) y[(size_t)n0 * rows + r0] = d[j][0];
            if (r1 < rows) y[(size_t)n0 * rows + r1] = d[j][2];
        }
        if (n1 < tt) {
            if (r0 < rows) y[(size_t)n1 * rows + r0] = d[j][1];
            if (r1 < rows) y[(size_t)n1 * rows + r1] = d[j][3];
        }
    }
}

// gemv_fp4_b with a 1024-thread block: for tiny row counts on long rows (the
// HC block-inject, rows=4, k=10240 = 160 k-blocks) the 256-thread version
// walks the k-blocks in a serial chain; 1024 threads cover them in one pass.
extern "C" __global__ void gemv_fp4_b1k(const unsigned char* __restrict__ w, const float* __restrict__ x,
                                        const float* __restrict__ gs_ptr, float* __restrict__ y,
                                        const int* __restrict__ k_dim_p) {
    int k_dim = *k_dim_p;
    int bpr = k_dim >> 6;
    int row = blockIdx.x;
    int t = blockIdx.y;
    const unsigned char* rowp = w + (size_t)row * bpr * 36;
    const float* xp = x + (size_t)t * k_dim;
    float gs = gs_ptr[0];
    float acc = 0.0f;
    for (int b = threadIdx.x; b < bpr; b += blockDim.x) {
        const unsigned char* blk = rowp + b * 36;
        #pragma unroll
        for (int sb = 0; sb < 4; sb++) {
            float s = ue4m3(blk[sb]) * gs;
            float part = 0.0f;
            #pragma unroll
            for (int j = 0; j < 16; j++) {
                int idx = sb * 16 + j;
                unsigned int byte = blk[4 + (idx >> 1)];
                unsigned int nib = (idx & 1) ? ((byte >> 4) & 0xF) : (byte & 0xF);
                part += e2m1(nib) * xp[b * 64 + sb * 16 + j];
            }
            acc += part * s;
        }
    }
    __shared__ float red[1024];
    red[threadIdx.x] = acc;
    __syncthreads();
    for (int st = blockDim.x / 2; st > 0; st >>= 1) {
        if (threadIdx.x < st) red[threadIdx.x] += red[threadIdx.x + st];
        __syncthreads();
    }
    if (threadIdx.x == 0) y[(size_t)t * gridDim.x + row] = red[0];
}

// ---------------- 19f (CROW_QFUSE=1): fused hc_run decode launches ----------------
// ONE launch replaces the hc_run down GEMV + silu_div4 + inject GEMV +
// sig2_div4 (decode, t < 8): grid ((rows + 7) / 8 + 4, T), block 256.
//  - blocks [0, (rows + 7) / 8): the gemv_bf16_w warp-per-row body VERBATIM
//    (same lane split, same k walk => the bit-identical accumulator), with
//    the silu_div4 epilogue folded in: v = acc * 0.25f, v / (1 + expf(-v)),
//    elementwise on the finished accumulator. Writes sil[t][row] only (the
//    s.low write is dead in the fused chain).
//  - the last 4 blocks: the gemv_fp4_b1k row math with its 1024-slot red[]
//    tree EMULATED on 256 threads: the k-block loop keeps the 1024 stride,
//    so threads 0..159 fill red[d] exactly like the 1024-wide launch (the
//    rest contribute +0.0), the three extra slots are explicit zeros, and
//    the binary tree runs two slots per thread with the same pairing and one
//    __syncthreads per level, so every FP32 add is the one gemv_fp4_b1k makes,
//    so the reduce is bit-identical by construction. The sig2_div4 epilogue
//    folds in: 2 / (1 + expf(-red[0] * 0.25f)).
// Both sides read x = s.normed (the hc down/inj k_dim = 10240); rows_p is
// the bf16 row count (320). A launch with grid.x == (rows + 7) / 8 (the
// head_run mixer) has no inj blocks and never dereferences the fp4 params.
extern "C" __global__ void hc_down_inj(const unsigned short* __restrict__ w,
                                       const float* __restrict__ x, float* __restrict__ sil,
                                       const unsigned char* __restrict__ wfp4,
                                       const float* __restrict__ gs_ptr, float* __restrict__ injw,
                                       const int* __restrict__ k_dim_p, const int* __restrict__ rows_p) {
    int k_dim = *k_dim_p;
    int rows = *rows_p;
    int bb = (rows + 7) >> 3;
    int t = blockIdx.y;
    if (blockIdx.x < bb) {
        int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
        int row = blockIdx.x * 8 + warp;
        if (row >= rows) return;
        const unsigned short* wp = w + (size_t)row * k_dim;
        const float* xp = x + (size_t)t * k_dim;
        float acc = 0.0f;
        for (int i = lane * 8; i < k_dim; i += 256) {
            uint4 v = *(const uint4*)(wp + i);
            unsigned int u[4] = {v.x, v.y, v.z, v.w};
            #pragma unroll
            for (int j = 0; j < 4; j++) {
                float lo = __uint_as_float(u[j] << 16);
                float hi = __uint_as_float(u[j] & 0xFFFF0000u);
                acc += lo * xp[i + 2 * j] + hi * xp[i + 2 * j + 1];
            }
        }
        for (int o = 16; o > 0; o >>= 1) acc += __shfl_down_sync(0xffffffffu, acc, o);
        if (lane == 0) {
            float v4 = acc * 0.25f;                  // silu_div4 epilogue
            sil[(size_t)t * rows + row] = v4 / (1.0f + expf(-v4));
        }
    } else {
        int row = blockIdx.x - bb;                   // 0..3, the inject rows
        int bpr = k_dim >> 6;
        const unsigned char* rowp = wfp4 + (size_t)row * bpr * 36;
        const float* xp = x + (size_t)t * k_dim;
        float gs = gs_ptr[0];
        float acc = 0.0f;
        for (int b = threadIdx.x; b < bpr; b += 1024) {   // 1024: the b1k stride
            const unsigned char* blk = rowp + b * 36;
            #pragma unroll
            for (int sb = 0; sb < 4; sb++) {
                float s = ue4m3(blk[sb]) * gs;
                float part = 0.0f;
                #pragma unroll
                for (int j = 0; j < 16; j++) {
                    int idx = sb * 16 + j;
                    unsigned int byte = blk[4 + (idx >> 1)];
                    unsigned int nib = (idx & 1) ? ((byte >> 4) & 0xF) : (byte & 0xF);
                    part += e2m1(nib) * xp[b * 64 + sb * 16 + j];
                }
                acc += part * s;
            }
        }
        __shared__ float red[1024];
        red[threadIdx.x] = acc;
        red[threadIdx.x + 256] = 0.0f;
        red[threadIdx.x + 512] = 0.0f;
        red[threadIdx.x + 768] = 0.0f;
        __syncthreads();
        for (int st = 512; st > 0; st >>= 1) {       // the b1k tree, 2 slots/thread
            if (threadIdx.x < st) red[threadIdx.x] += red[threadIdx.x + st];
            if (threadIdx.x + 256 < st) red[threadIdx.x + 256] += red[threadIdx.x + 256 + st];
            __syncthreads();
        }
        if (threadIdx.x == 0) {
            float v = red[0];                        // sig2_div4 epilogue
            injw[(size_t)t * 4 + row] = 2.0f / (1.0f + expf(-v * 0.25f));
        }
    }
}

// 19f: gemv_bf16_w with the sigmoid_el epilogue folded in (CROW_QFUSE=1 hc
// chain): the standalone sigmoid_el reads the finished accumulator back and
// writes 1 / (1 + expf(-x)) elementwise, so applying it at the lane-0 store
// is bit-identical. Body otherwise verbatim gemv_bf16_w.
extern "C" __global__ void gemv_bf16_ws(const unsigned short* __restrict__ w, const float* __restrict__ x,
                                        float* __restrict__ y, const int* __restrict__ k_dim_p,
                                        const int* __restrict__ rows_p) {
    int k_dim = *k_dim_p;
    int rows = *rows_p;
    int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    int row = blockIdx.x * 8 + warp;
    int t = blockIdx.y;
    if (row >= rows) return;
    const unsigned short* wp = w + (size_t)row * k_dim;
    const float* xp = x + (size_t)t * k_dim;
    float acc = 0.0f;
    for (int i = lane * 8; i < k_dim; i += 256) {
        uint4 v = *(const uint4*)(wp + i);
        unsigned int u[4] = {v.x, v.y, v.z, v.w};
        #pragma unroll
        for (int j = 0; j < 4; j++) {
            float lo = __uint_as_float(u[j] << 16);
            float hi = __uint_as_float(u[j] & 0xFFFF0000u);
            acc += lo * xp[i + 2 * j] + hi * xp[i + 2 * j + 1];
        }
    }
    for (int o = 16; o > 0; o >>= 1) acc += __shfl_down_sync(0xffffffffu, acc, o);
    if (lane == 0) y[(size_t)t * rows + row] = 1.0f / (1.0f + expf(-acc));
}

extern "C" __global__ void dequant_fp4_flat(const unsigned char* __restrict__ w,
                                            const float* __restrict__ gs_ptr, float* __restrict__ out,
                                            const int* __restrict__ n_p) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= *n_p) return;
    int b = i >> 6, idx = i & 63;
    const unsigned char* blk = w + (size_t)b * 36;
    float s = ue4m3(blk[idx >> 4]) * gs_ptr[0];
    unsigned int byte = blk[4 + (idx >> 1)];
    unsigned int nib = (idx & 1) ? ((byte >> 4) & 0xF) : (byte & 0xF);
    out[i] = e2m1(nib) * s;
}

extern "C" __global__ void bf16_to_f32(const unsigned short* __restrict__ in_, float* __restrict__ out,
                                       const int* __restrict__ n_p) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= *n_p) return;
    out[i] = __int_as_float(((unsigned int)in_[i]) << 16);
}

// ---------------- elementwise + norms (p8/p13-verified) ----------------
extern "C" __global__ void rms_group(const float* __restrict__ x, const float* __restrict__ w,
                                     float* __restrict__ out) {
    int g = blockIdx.x;
    int t = blockIdx.y;
    int d = threadIdx.x;
    const float* xp = x + (t * 4 + g) * 2560;
    float part = 0.0f;
    for (int i = d; i < 2560; i += 256) part += xp[i] * xp[i];
    __shared__ float red[256];
    red[d] = part;
    __syncthreads();
    for (int st = 128; st > 0; st >>= 1) {
        if (d < st) red[d] += red[d + st];
        __syncthreads();
    }
    float rms = rsqrtf(red[0] / 2560.0f + 1e-6f);
    for (int i = d; i < 2560; i += 256)
        out[(t * 4 + g) * 2560 + i] = xp[i] * rms * (1.0f + w[g * 2560 + i]);
}

extern "C" __global__ void rmsnorm_1pw(const float* __restrict__ x, const float* __restrict__ w,
                                       float* __restrict__ out) {
    int head = blockIdx.x;
    int t = blockIdx.y;
    int d = threadIdx.x;
    const float* xp = x + (t * gridDim.x + head) * 256;
    __shared__ float red[256];
    red[d] = xp[d] * xp[d];
    __syncthreads();
    for (int st = 128; st > 0; st >>= 1) {
        if (d < st) red[d] += red[d + st];
        __syncthreads();
    }
    float rms = rsqrtf(red[0] / 256.0f + 1e-6f);
    out[(t * gridDim.x + head) * 256 + d] = xp[d] * rms * (1.0f + w[d]);
}

extern "C" __global__ void silu_div4(const float* __restrict__ x, float* __restrict__ out,
                                     const int* __restrict__ n_p) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= *n_p) return;
    float v = x[i] * 0.25f;
    out[i] = v / (1.0f + expf(-v));
}
extern "C" __global__ void sigmoid_el(float* __restrict__ x, const int* __restrict__ n_p) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= *n_p) return;
    x[i] = 1.0f / (1.0f + expf(-x[i]));
}
extern "C" __global__ void sig2_div4(const float* __restrict__ x, float* __restrict__ out,
                                     const int* __restrict__ n_p) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= *n_p) return;
    out[i] = 2.0f / (1.0f + expf(-x[i] * 0.25f));
}
extern "C" __global__ void mix_streams(const float* __restrict__ mixw, const float* __restrict__ normed,
                                       float* __restrict__ out) {
    int c = blockIdx.x * 256 + threadIdx.x;
    int t = blockIdx.y;
    float acc = 0.0f;
    for (int g = 0; g < 4; g++)
        acc += mixw[t * 10240 + g * 2560 + c] * normed[t * 10240 + g * 2560 + c];
    out[t * 2560 + c] = acc * 0.25f;
}
extern "C" __global__ void inject_residual(const float* __restrict__ base, const float* __restrict__ mix,
                                           const float* __restrict__ injw, float* __restrict__ out) {
    int g = blockIdx.x;
    int tc = blockIdx.y;
    int t = tc / 10;
    int c = (tc % 10) * 256 + threadIdx.x;
    out[(t * 4 + g) * 2560 + c] =
        base[(t * 4 + g) * 2560 + c] + mix[t * 2560 + c] * injw[t * 4 + g];
}
extern "C" __global__ void silu_mul640(const float* __restrict__ h1, float* __restrict__ h2) {
    int j = blockIdx.x * 256 + threadIdx.x;
    int t = blockIdx.y;
    if (j >= 640) return;
    float gate = h1[t * 1280 + j];
    h2[t * 640 + j] = (gate / (1.0f + expf(-gate))) * h1[t * 1280 + 640 + j];
}
// combo-major silu·gate over [C][1280] -> [C][640], full-flat guard
extern "C" __global__ void silu_mul_combo(const float* __restrict__ h1, float* __restrict__ h2,
                                          const int* __restrict__ n640_p) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= *n640_p) return;
    int c = i / 640;
    int j = i % 640;
    float gate = h1[(size_t)c * 1280 + j];
    h2[i] = (gate / (1.0f + expf(-gate))) * h1[(size_t)c * 1280 + 640 + j];
}
// y[t][c] += (*w)·x[t][c]  (w pointer INTO the routing-weight row, p13)
extern "C" __global__ void acc_scale(const float* __restrict__ x, const float* __restrict__ w,
                                     float* __restrict__ y) {
    int c = blockIdx.x * 256 + threadIdx.x;
    int t = blockIdx.y;
    y[t * 2560 + c] += (*w) * x[t * 2560 + c];
}
// batched ranks: y[t][c] += rw[t*10+j] · x[(t*10+j)][c], grid (10, T)
// deterministic per-token reduction over the 10 routed ranks (fixed order,
// ONE launch). The previous version did y += from 10 blocks (blockIdx.x =
// rank) on the same addresses without atomics — updates were lost and the
// result differed per warp (shared-expert reconstruction, 2026-09-03).
extern "C" __global__ void acc_combo(const float* __restrict__ x, const float* __restrict__ rw,
                                     float* __restrict__ y) {
    int c = blockIdx.x * 256 + threadIdx.x;
    int t = blockIdx.y;
    float acc = 0.0f;
    for (int j = 0; j < 10; j++) {
        acc += rw[t * 10 + j] * x[((size_t)t * 10 + j) * 2560 + c];
    }
    y[t * 2560 + c] += acc;
}
extern "C" __global__ void gate_shared(const float* __restrict__ s, const float* __restrict__ sg,
                                       float* __restrict__ y) {
    int c = blockIdx.x * 256 + threadIdx.x;
    int t = blockIdx.y;
    float g = sg[t];
    y[t * 2560 + c] = (1.0f / (1.0f + expf(-g))) * s[t * 2560 + c]; // ASSIGN: first writer of moe_out (no memset, graph-capturable)
}

// ---------------- GDN (p6/p11/p13-verified) ----------------
// causal conv k=4 over [C][T] rows, with the persistent [C][3] state feeding
// positions t<3 on chunks after the first (chunk 0 state is zeros = left pad)
extern "C" __global__ void conv_silu(const float* __restrict__ in_, const float* __restrict__ w,
                                     float* __restrict__ out, const int* __restrict__ t_p,
                                     const float* __restrict__ state) {
    int tt = *t_p;
    int ch = blockIdx.x;
    for (int t = threadIdx.x; t < tt; t += blockDim.x) {
        float acc = 0.0f;
        for (int k = 0; k < 4; k++) {
            int src_t = t + k - 3;
            float v;
            if (src_t >= 0) v = in_[(size_t)ch * tt + src_t];
            else v = state[ch * 3 + (3 + src_t)];
            acc += w[ch * 4 + k] * v;
        }
        out[(size_t)ch * tt + t] = acc / (1.0f + expf(-acc));
    }
}
// refresh the [C][3] conv state from the last 3 rows of a chunk ([C][T] layout).
// A chunk shorter than the window (tt < 3: a prompt tail, a warm resume with a
// 1-2 token suffix) keeps the newest 3 - tt OLD rows, shifted down by tt - the
// window conv_step would hold after tt single steps. All reads before any write:
// thread j reads old slot j + tt, which another thread overwrites.
extern "C" __global__ void conv_state_update(const float* __restrict__ in_, float* __restrict__ state,
                                             const int* __restrict__ t_p) {
    int tt = *t_p;
    int ch = blockIdx.x;
    int j = threadIdx.x; // 3 threads
    float v = 0.0f;
    if (j < 3) {
        int src = tt - 3 + j;
        v = (src >= 0) ? in_[(size_t)ch * tt + src] : state[ch * 3 + j + tt];
    }
    __syncthreads();
    if (j < 3) state[ch * 3 + j] = v;
}
// [T][C] -> [C][T] device transpose (replaces the p13 host roundtrip)
extern "C" __global__ void transpose_rt(const float* __restrict__ in_, float* __restrict__ out,
                                        const int* __restrict__ t_p, const int* __restrict__ c_p) {
    int tt = *t_p;
    int c = *c_p;
    int j = blockIdx.x;
    for (int t = threadIdx.x; t < tt; t += blockDim.x)
        out[(size_t)j * tt + t] = in_[(size_t)t * c + j];
}
// split conv output [C][T] (q 2048 | k 2048 | v 6144) into [T][·] rows
extern "C" __global__ void split_qkv(const float* __restrict__ src, float* __restrict__ q,
                                     float* __restrict__ k, float* __restrict__ v,
                                     const int* __restrict__ t_p) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int tt = *t_p;
    if (i < tt * 2048) {
        int t = i / 2048, j = i % 2048;
        q[(size_t)t * 2048 + j] = src[(size_t)j * tt + t];
    } else if (i < tt * 4096) {
        int r = i - tt * 2048;
        int t = r / 2048, j = r % 2048;
        k[(size_t)t * 2048 + j] = src[(size_t)(2048 + j) * tt + t];
    } else if (i < tt * 10240) {
        int r = i - tt * 4096;
        int t = r / 6144, j = r % 6144;
        v[(size_t)t * 6144 + j] = src[(size_t)(4096 + j) * tt + t];
    }
}
extern "C" __global__ void l2norm_repeat(const float* __restrict__ q_in, const float* __restrict__ k_in,
                                         float* __restrict__ q_out, float* __restrict__ k_out) {
    int vhead = blockIdx.x;
    int t = blockIdx.y;
    int d = threadIdx.x;
    int khead = vhead / 3;
    const float* qp = q_in + (t * 16 + khead) * 128;
    const float* kp = k_in + (t * 16 + khead) * 128;
    float qn = 0.0f, kn = 0.0f;
    for (int i = 0; i < 128; i++) { qn += qp[i] * qp[i]; kn += kp[i] * kp[i]; }
    qn = rsqrtf(qn + 1e-6f); kn = rsqrtf(kn + 1e-6f);
    q_out[((size_t)t * 48 + vhead) * 128 + d] = qp[d] * qn * rsqrtf(128.0f);
    k_out[((size_t)t * 48 + vhead) * 128 + d] = kp[d] * kn;
}
extern "C" __global__ void beta_g(const float* __restrict__ b_pr, const float* __restrict__ a_pr,
                                  const float* __restrict__ a_log, const float* __restrict__ dt_bias,
                                  float* __restrict__ beta_out, float* __restrict__ g_out,
                                  const int* __restrict__ t_p) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    // guard: t*48 is not a multiple of the block size - without it the prompt
    // path wrote past gbeta/gg (found 2026-09-04: chunk-size-dependent
    // nondeterminism, C=8 parity runs differed from row 1 on)
    if (i >= *t_p * 48) return;
    float beta = 1.0f / (1.0f + expf(-b_pr[i]));
    float a = a_pr[i] + dt_bias[i % 48];
    float sp = logf(1.0f + expf(a));
    beta_out[i] = beta;
    g_out[i] = -expf(a_log[i % 48]) * sp;
}
// batched prompt recurrence with persistent state; init_p=1 zeroes S first
extern "C" __global__ void delta_rule_persist(const float* __restrict__ q, const float* __restrict__ k,
                                              const float* __restrict__ v, const float* __restrict__ g,
                                              const float* __restrict__ beta, float* __restrict__ out,
                                              float* __restrict__ s_global, const int* __restrict__ steps_p,
                                              const int* __restrict__ init_p) {
    int steps = *steps_p;
    int head = blockIdx.x;
    float* S = s_global + head * 128 * 128;
    int d = threadIdx.x;
    if (*init_p) for (int dk = 0; dk < 128; dk++) S[dk * 128 + d] = 0.0f;
    for (int t = 0; t < steps; t++) {
        float g_t = expf(g[t * 48 + head]);
        float beta_t = beta[t * 48 + head];
        const float* qt = q + (t * 48 + head) * 128;
        const float* kt = k + (t * 48 + head) * 128;
        const float* vt = v + (t * 48 + head) * 128;
        for (int dk = 0; dk < 128; dk++) S[dk * 128 + d] *= g_t;
        float kv = 0.0f;
        for (int dk = 0; dk < 128; dk++) kv += S[dk * 128 + d] * kt[dk];
        float delta = (vt[d] - kv) * beta_t;
        for (int dk = 0; dk < 128; dk++) S[dk * 128 + d] += kt[dk] * delta;
        float o = 0.0f;
        for (int dk = 0; dk < 128; dk++) o += S[dk * 128 + d] * qt[dk];
        out[(t * 48 + head) * 128 + d] = o;
    }
}
extern "C" __global__ void conv_step(const float* __restrict__ mq1, const float* __restrict__ w,
                                     float* __restrict__ cs, float* __restrict__ cout) {
    int ch = blockIdx.x * blockDim.x + threadIdx.x;
    if (ch >= 10240) return;
    const float* wv = w + ch * 4;
    float acc = wv[0] * cs[ch * 3 + 0] + wv[1] * cs[ch * 3 + 1]
              + wv[2] * cs[ch * 3 + 2] + wv[3] * mq1[ch];
    cout[ch] = acc / (1.0f + expf(-acc));
    cs[ch * 3 + 0] = cs[ch * 3 + 1];
    cs[ch * 3 + 1] = cs[ch * 3 + 2];
    cs[ch * 3 + 2] = mq1[ch];
}
extern "C" __global__ void delta_rule_step(float* __restrict__ s_global, const float* __restrict__ q,
                                           const float* __restrict__ k, const float* __restrict__ v,
                                           const float* __restrict__ g, const float* __restrict__ beta,
                                           float* __restrict__ core) {
    int head = blockIdx.x;
    float* S = s_global + head * 128 * 128;
    int d = threadIdx.x;
    float g_t = expf(g[head]);
    float beta_t = beta[head];
    const float* qt = q + head * 128;
    const float* kt = k + head * 128;
    const float* vt = v + head * 128;
    for (int dk = 0; dk < 128; dk++) S[dk * 128 + d] *= g_t;
    float kv = 0.0f;
    for (int dk = 0; dk < 128; dk++) kv += S[dk * 128 + d] * kt[dk];
    float delta = (vt[d] - kv) * beta_t;
    for (int dk = 0; dk < 128; dk++) S[dk * 128 + d] += kt[dk] * delta;
    float o = 0.0f;
    for (int dk = 0; dk < 128; dk++) o += S[dk * 128 + d] * qt[dk];
    core[head * 128 + d] = o;
}
// Register-resident delta rule (2026-09-04): the 128x128 head state lives in
// registers (one column per thread) for the whole chunk instead of being
// streamed through global memory four times per token. Same per-element
// operation order as delta_rule_persist -> bit-identical results; the scale
// and kv passes (and the update and output passes) are fused per element.
// Intrinsics pin the op sequence: without them nvcc contracted the decode
// step's `rn(s*g) + k*delta` as fma(s, g, k*delta) (found 2026-09-04 by PTX
// diff, parity rows 9+ off by 1e0) — the original kernel's global-memory
// round trip had prevented that contraction.
extern "C" __global__ void delta_rule_persist_r(const float* __restrict__ q, const float* __restrict__ k,
                                                const float* __restrict__ v, const float* __restrict__ g,
                                                const float* __restrict__ beta, float* __restrict__ out,
                                                float* __restrict__ s_global, const int* __restrict__ steps_p,
                                                const int* __restrict__ init_p) {
    int steps = *steps_p;
    int head = blockIdx.x;
    float* S = s_global + head * 128 * 128;
    int d = threadIdx.x;
    __shared__ float ks[128];
    __shared__ float qs[128];
    float sr[128];
    if (*init_p) {
#pragma unroll
        for (int dk = 0; dk < 128; dk++) sr[dk] = 0.0f;
    } else {
#pragma unroll
        for (int dk = 0; dk < 128; dk++) sr[dk] = S[dk * 128 + d];
    }
    for (int t = 0; t < steps; t++) {
        float g_t = expf(g[t * 48 + head]);
        float beta_t = beta[t * 48 + head];
        const float* qt = q + (t * 48 + head) * 128;
        const float* kt = k + (t * 48 + head) * 128;
        const float* vt = v + (t * 48 + head) * 128;
        ks[d] = kt[d];
        qs[d] = qt[d];
        __syncthreads();
        float kv = 0.0f;
#pragma unroll
        for (int dk = 0; dk < 128; dk++) { sr[dk] = __fmul_rn(sr[dk], g_t); kv = __fmaf_rn(sr[dk], ks[dk], kv); }
        float delta = (vt[d] - kv) * beta_t;
        float o = 0.0f;
#pragma unroll
        for (int dk = 0; dk < 128; dk++) { sr[dk] = __fmaf_rn(ks[dk], delta, sr[dk]); o = __fmaf_rn(sr[dk], qs[dk], o); }
        out[(t * 48 + head) * 128 + d] = o;
        __syncthreads();
    }
#pragma unroll
    for (int dk = 0; dk < 128; dk++) S[dk * 128 + d] = sr[dk];
}
extern "C" __global__ void delta_rule_step_r(float* __restrict__ s_global, const float* __restrict__ q,
                                             const float* __restrict__ k, const float* __restrict__ v,
                                             const float* __restrict__ g, const float* __restrict__ beta,
                                             float* __restrict__ core) {
    int head = blockIdx.x;
    float* S = s_global + head * 128 * 128;
    int d = threadIdx.x;
    __shared__ float ks[128];
    __shared__ float qs[128];
    float g_t = expf(g[head]);
    float beta_t = beta[head];
    ks[d] = k[head * 128 + d];
    qs[d] = q[head * 128 + d];
    __syncthreads();
    float sr[128];
#pragma unroll
    for (int dk = 0; dk < 128; dk++) sr[dk] = S[dk * 128 + d];
    float kv = 0.0f;
#pragma unroll
    for (int dk = 0; dk < 128; dk++) { sr[dk] = __fmul_rn(sr[dk], g_t); kv = __fmaf_rn(sr[dk], ks[dk], kv); }
    float delta = (v[head * 128 + d] - kv) * beta_t;
    float o = 0.0f;
#pragma unroll
    for (int dk = 0; dk < 128; dk++) { sr[dk] = __fmaf_rn(ks[dk], delta, sr[dk]); o = __fmaf_rn(sr[dk], qs[dk], o); }
#pragma unroll
    for (int dk = 0; dk < 128; dk++) S[dk * 128 + d] = sr[dk];
    core[head * 128 + d] = o;
}
extern "C" __global__ void rmsnorm_gated(const float* __restrict__ x, const float* __restrict__ z,
                                         const float* __restrict__ w, float* __restrict__ out) {
    int vhead = blockIdx.x;
    int t = blockIdx.y;
    int d = threadIdx.x;
    const float* xp = x + ((size_t)t * 48 + vhead) * 128;
    __shared__ float red[128];
    red[d] = xp[d] * xp[d];
    __syncthreads();
    for (int st = 64; st > 0; st >>= 1) {
        if (d < st) red[d] += red[d + st];
        __syncthreads();
    }
    float rms = rsqrtf(red[0] / 128.0f + 1e-6f);
    float gate = 1.0f / (1.0f + expf(-z[((size_t)t * 48 + vhead) * 128 + d]));
    out[((size_t)t * 48 + vhead) * 128 + d] = w[d] * xp[d] * rms * gate;
}

// ---------------- attention (p7/p12/p16-verified + QSA lists) ----------------
extern "C" __global__ void split_qg(const float* __restrict__ qg, float* __restrict__ q,
                                    float* __restrict__ gate) {
    int head = blockIdx.x;
    int t = blockIdx.y;
    int d = threadIdx.x;
    int src = t * 12288 + head * 512 + d;
    q[(t * 24 + head) * 256 + d] = qg[src];
    gate[t * 6144 + head * 256 + d] = qg[src + 256];
}
extern "C" __global__ void rope(const float* __restrict__ x, const float* __restrict__ cos_,
                                const float* __restrict__ sin_, float* __restrict__ out) {
    int head = blockIdx.x;
    int t = blockIdx.y;
    int d = threadIdx.x;
    const float* xp = x + (t * gridDim.x + head) * 256;
    float* op = out + (t * gridDim.x + head) * 256;
    if (d >= 32) {
        if (d >= 64) op[d] = xp[d];
        return;
    }
    float a = xp[d], b = xp[d + 32];
    float c = cos_[t * 32 + d], s = sin_[t * 32 + d];
    op[d] = a * c - b * s;
    op[d + 32] = b * c + a * s;
}
// rope with a DEVICE position scalar (graph-replay safe: no host-computed
// table offset baked into the kernel args). Decode path, t = blockIdx.y.
extern "C" __global__ void rope_p(const float* __restrict__ x, const float* __restrict__ cos_,
                                  const float* __restrict__ sin_, float* __restrict__ out,
                                  const int* __restrict__ pos_base_p) {
    int head = blockIdx.x;
    int t = blockIdx.y;
    int d = threadIdx.x;
    int pt = *pos_base_p + t;
    const float* xp = x + (t * gridDim.x + head) * 256;
    float* op = out + (t * gridDim.x + head) * 256;
    if (d >= 32) {
        if (d >= 64) op[d] = xp[d];
        return;
    }
    float a = xp[d], b = xp[d + 32];
    float c = cos_[pt * 32 + d], s = sin_[pt * 32 + d];
    op[d] = a * c - b * s;
    op[d + 32] = b * c + a * s;
}
// KV cache append with on-store cast (e4m3 default; bf16 keep path).
// grid (4, T): slot = slot_base + blockIdx.y (chunked prompt or single token)
extern "C" __global__ void store_kv(const float* __restrict__ kr, const float* __restrict__ vr,
                                    unsigned char* __restrict__ kcache, unsigned char* __restrict__ vcache,
                                    const int* __restrict__ slot_p, const int* __restrict__ tmax_p,
                                    const int* __restrict__ mode_p) {
    int h = blockIdx.x;
    int d = threadIdx.x;
    if (d >= 256) return;
    int slot = *slot_p + (int)blockIdx.y;
    int tmax = *tmax_p;
    int mode = *mode_p;
    int bh = (h < 2) ? h : h - 2;
    size_t rowb = (size_t)(bh * tmax + slot) * 256 * (mode ? 2 : 1);
    const float* src = (h < 2) ? kr + (size_t)(blockIdx.y * 2 + bh) * 256
                               : vr + (size_t)(blockIdx.y * 2 + bh) * 256;
    unsigned char* dst = (h < 2) ? kcache + rowb : vcache + rowb;
    if (mode == 0) dst[d] = enc_e4m3(src[d]);
    else ((unsigned short*)dst)[d] = f32_bf16_bits(src[d]);
}
// ---------------- #96: the attention softmax scale ----------------
// 1/sqrt(256) = 0.0625f is folded into every attention variant below. Issue
// #96 threads the YaRN mscale into it WITHOUT touching a single default-path
// launch: the RT = 0 instantiation of attn_scale_src returns the compile-time
// literal (the kernels of record compile to the same folded multiply as before
// #96), and only the _y twins read this device global, which Kernels::new sets
// once at boot when the checkpoint config carries a rope_scaling whose mscale
// differs from 1. The value is the same 0.0625f either way (a power of two:
// the multiply is exact), so default-path logits are bit-identical.
extern "C" __device__ float d_attn_scale = 0.0625f; // 1/sqrt(256) unless YaRN rewrites it at boot (extern "C" keeps the plain PTX name the #96 test greps)
template <int RT> __device__ __forceinline__ float attn_scale_src();
template <> __device__ __forceinline__ float attn_scale_src<0>() { return 0.0625f; } // 1/sqrt(256)
template <> __device__ __forceinline__ float attn_scale_src<1>() { return d_attn_scale; } // #96 YaRN mscale path
// one boot-time write of the runtime scale (scalars live in device buffers - the p5 rule)
extern "C" __global__ void set_attn_scale(const float* __restrict__ v) {
    if (threadIdx.x == 0 && blockIdx.x == 0) d_attn_scale = *v;
}
// list-based attention: softmax over the QSA-selected (or dense = all) token
// list read from the persistent KV cache with on-load dequant. grid (24, Tq).
// #96: the body is a template on RT ONLY so the _y twin can read the runtime
// scale; RT = 0 (attn_sel) folds 0.0625f exactly as the pre-#96 kernel did.
template <int RT>
__device__ __forceinline__ void attn_sel_body(const float* __restrict__ q, const unsigned char* __restrict__ kc,
                                    const unsigned char* __restrict__ vc, const int* __restrict__ sel,
                                    const int* __restrict__ sel_n, const int* __restrict__ tmax_p,
                                    const int* __restrict__ mode_p, const int* __restrict__ sel_max_p,
                                    float* __restrict__ out) {
    int head = blockIdx.x;
    int t = blockIdx.y;
    int d = threadIdx.x;
    int kvh = head / 12;
    int n = sel_n[t];
    if (n < 0) n = 0;
    if (n > *sel_max_p) n = *sel_max_p;
    const int* list = sel + (size_t)t * *sel_max_p;
    const float* qt = q + ((size_t)t * 24 + head) * 256;
    __shared__ float p[2051];
    __shared__ float red[256];
    int mode = *mode_p;
    int warp = d >> 5, lane = d & 31;
    const float scale = attn_scale_src<RT>(); // 1/sqrt(256)
    for (int j0 = 0; j0 < n; j0 += 8) {
        int j = j0 + warp;
        if (j < n) {
            int tok = list[j];
            if (tok < 0) tok = 0;
            if (tok >= *tmax_p) tok = *tmax_p - 1;
            const unsigned char* kp = kc + (size_t)(kvh * *tmax_p + tok) * 256 * (mode ? 2 : 1);
            float acc = 0.0f;
            for (int e = lane; e < 256; e += 32) acc += qt[e] * kv_load(kp, e, mode);
            for (int o = 16; o > 0; o >>= 1) acc += __shfl_down_sync(0xffffffffu, acc, o);
            if (lane == 0) p[j] = acc * scale;
        }
    }
    __syncthreads();
    float mx = -3.0e38f;
    for (int j = d; j < n; j += 256) mx = fmaxf(mx, p[j]);
    red[d] = mx;
    __syncthreads();
    for (int st = 128; st > 0; st >>= 1) {
        if (d < st) red[d] = fmaxf(red[d], red[d + st]);
        __syncthreads();
    }
    mx = red[0];
    __syncthreads(); // thread 0 writes red[0] = sum below: without this barrier a late reader takes that sum as mx
    float sum = 0.0f;
    for (int j = d; j < n; j += 256) { float e = expf(p[j] - mx); p[j] = e; sum += e; }
    red[d] = sum;
    __syncthreads();
    for (int st = 128; st > 0; st >>= 1) {
        if (d < st) red[d] += red[d + st];
        __syncthreads();
    }
    sum = red[0];
    float o = 0.0f;
    for (int j = 0; j < n; j++) {
        int tok = list[j];
        if (tok < 0) tok = 0;
        if (tok >= *tmax_p) tok = *tmax_p - 1;
        const unsigned char* vp = vc + (size_t)(kvh * *tmax_p + tok) * 256 * (mode ? 2 : 1);
        o += (p[j] / sum) * kv_load(vp, d, mode);
    }
    out[((size_t)t * 24 + head) * 256 + d] = o;
}
extern "C" __global__ void attn_sel(const float* __restrict__ q, const unsigned char* __restrict__ kc,
                                    const unsigned char* __restrict__ vc, const int* __restrict__ sel,
                                    const int* __restrict__ sel_n, const int* __restrict__ tmax_p,
                                    const int* __restrict__ mode_p, const int* __restrict__ sel_max_p,
                                    float* __restrict__ out) {
    attn_sel_body<0>(q, kc, vc, sel, sel_n, tmax_p, mode_p, sel_max_p, out);
}
// #96: the YaRN runtime-scale twin (same signature, same launch sites — the
// handle swap happens once, in Kernels::new, when an mscale is armed)
extern "C" __global__ void attn_sel_y(const float* __restrict__ q, const unsigned char* __restrict__ kc,
                                      const unsigned char* __restrict__ vc, const int* __restrict__ sel,
                                      const int* __restrict__ sel_n, const int* __restrict__ tmax_p,
                                      const int* __restrict__ mode_p, const int* __restrict__ sel_max_p,
                                      float* __restrict__ out) {
    attn_sel_body<1>(q, kc, vc, sel, sel_n, tmax_p, mode_p, sel_max_p, out);
}
// attn_sel_r (#10 step 3, 2026-09-06): attn_sel with q held in registers (8 floats per lane, same e order),
// the softmax weights normalised once in shared memory (the same IEEE division per element as p[j] / sum inline)
// and the V loop unrolled x4 with the loads hoisted; the accumulation order is unchanged. Each per-element op is
// the same single-product chain as attn_sel, so nvcc contracts identically -> meant bit-identical (gate: parity).
// R = 8 / 9 are DIAGNOSTICS (no K dot / no V loop, wrong output) that measure the two phases' floors.
// #96: RT = the scale source (0 folded, 1 the YaRN runtime global), as attn_sel.
template <int R, int RT>
__device__ __forceinline__ void attn_sel_r_body(const float* __restrict__ q, const unsigned char* __restrict__ kc,
                                                const unsigned char* __restrict__ vc, const int* __restrict__ sel,
                                                const int* __restrict__ sel_n, const int* __restrict__ tmax_p,
                                                const int* __restrict__ mode_p, const int* __restrict__ sel_max_p,
                                                float* __restrict__ out) {
    int head = blockIdx.x;
    int t = blockIdx.y;
    int d = threadIdx.x;
    int kvh = head / 12;
    int n = sel_n[t];
    if (n < 0) n = 0;
    const int sel_max = *sel_max_p;
    if (n > sel_max) n = sel_max;
    const int tmax = *tmax_p;
    const int* list = sel + (size_t)t * sel_max;
    const float* qt = q + ((size_t)t * 24 + head) * 256;
    __shared__ float p[2051];
    __shared__ float red[256];
    const int mode = *mode_p;
    const int esz = mode ? 2 : 1;
    const size_t kvbase = (size_t)kvh * tmax;
    int warp = d >> 5, lane = d & 31;
    const float scale = attn_scale_src<RT>(); // 1/sqrt(256)
    float qr[8];
#pragma unroll
    for (int k = 0; k < 8; k++) qr[k] = qt[lane + 32 * k];
    if (R == 8) {
        for (int j = d; j < n; j += 256) p[j] = (float)(j & 15) * 0.01f; // DIAGNOSTIC: no K dot
    } else {
        for (int j0 = 0; j0 < n; j0 += 8) {
            int j = j0 + warp;
            if (j < n) {
                int tok = list[j];
                if (tok < 0) tok = 0;
                if (tok >= tmax) tok = tmax - 1;
                const unsigned char* kp = kc + (kvbase + tok) * 256 * esz;
                float acc = 0.0f;
#pragma unroll
                for (int k = 0; k < 8; k++) acc += qr[k] * kv_load(kp, lane + 32 * k, mode);
                for (int o = 16; o > 0; o >>= 1) acc += __shfl_down_sync(0xffffffffu, acc, o);
                if (lane == 0) p[j] = acc * scale;
            }
        }
    }
    __syncthreads();
    float mx = -3.0e38f;
    for (int j = d; j < n; j += 256) mx = fmaxf(mx, p[j]);
    red[d] = mx;
    __syncthreads();
    for (int st = 128; st > 0; st >>= 1) {
        if (d < st) red[d] = fmaxf(red[d], red[d + st]);
        __syncthreads();
    }
    mx = red[0];
    __syncthreads(); // thread 0 writes red[0] = sum below: without this barrier a late reader takes that sum as mx
    float sum = 0.0f;
    for (int j = d; j < n; j += 256) { float e = expf(p[j] - mx); p[j] = e; sum += e; }
    red[d] = sum;
    __syncthreads();
    for (int st = 128; st > 0; st >>= 1) {
        if (d < st) red[d] += red[d + st];
        __syncthreads();
    }
    sum = red[0];
    __syncthreads();
    for (int j = d; j < n; j += 256) p[j] = p[j] / sum; // the same division attn_sel does inline
    __syncthreads();
    float o = 0.0f;
    if (R == 9) {
        o = sum; // DIAGNOSTIC: no V loop
    } else {
        const unsigned char* vb = vc + kvbase * 256 * esz;
        int j = 0;
        for (; j + 4 <= n; j += 4) {
            int t0 = list[j], t1 = list[j + 1], t2 = list[j + 2], t3 = list[j + 3];
            t0 = t0 < 0 ? 0 : (t0 >= tmax ? tmax - 1 : t0);
            t1 = t1 < 0 ? 0 : (t1 >= tmax ? tmax - 1 : t1);
            t2 = t2 < 0 ? 0 : (t2 >= tmax ? tmax - 1 : t2);
            t3 = t3 < 0 ? 0 : (t3 >= tmax ? tmax - 1 : t3);
            float v0 = kv_load(vb + (size_t)t0 * 256 * esz, d, mode);
            float v1 = kv_load(vb + (size_t)t1 * 256 * esz, d, mode);
            float v2 = kv_load(vb + (size_t)t2 * 256 * esz, d, mode);
            float v3 = kv_load(vb + (size_t)t3 * 256 * esz, d, mode);
            o += p[j] * v0;
            o += p[j + 1] * v1;
            o += p[j + 2] * v2;
            o += p[j + 3] * v3;
        }
        for (; j < n; j++) {
            int tok = list[j];
            if (tok < 0) tok = 0;
            if (tok >= tmax) tok = tmax - 1;
            o += p[j] * kv_load(vb + (size_t)tok * 256 * esz, d, mode);
        }
    }
    out[((size_t)t * 24 + head) * 256 + d] = o;
}
extern "C" __global__ void attn_sel_r(const float* __restrict__ q, const unsigned char* __restrict__ kc,
                                      const unsigned char* __restrict__ vc, const int* __restrict__ sel,
                                      const int* __restrict__ sel_n, const int* __restrict__ tmax_p,
                                      const int* __restrict__ mode_p, const int* __restrict__ sel_max_p,
                                      float* __restrict__ out) {
    attn_sel_r_body<1, 0>(q, kc, vc, sel, sel_n, tmax_p, mode_p, sel_max_p, out);
}
extern "C" __global__ void attn_sel_r_y(const float* __restrict__ q, const unsigned char* __restrict__ kc,
                                        const unsigned char* __restrict__ vc, const int* __restrict__ sel,
                                        const int* __restrict__ sel_n, const int* __restrict__ tmax_p,
                                        const int* __restrict__ mode_p, const int* __restrict__ sel_max_p,
                                        float* __restrict__ out) {
    attn_sel_r_body<1, 1>(q, kc, vc, sel, sel_n, tmax_p, mode_p, sel_max_p, out); // #96 YaRN runtime scale
}
extern "C" __global__ void attn_sel_d8(const float* __restrict__ q, const unsigned char* __restrict__ kc,
                                       const unsigned char* __restrict__ vc, const int* __restrict__ sel,
                                       const int* __restrict__ sel_n, const int* __restrict__ tmax_p,
                                       const int* __restrict__ mode_p, const int* __restrict__ sel_max_p,
                                       float* __restrict__ out) {
    attn_sel_r_body<8, 0>(q, kc, vc, sel, sel_n, tmax_p, mode_p, sel_max_p, out);
}
extern "C" __global__ void attn_sel_d8_y(const float* __restrict__ q, const unsigned char* __restrict__ kc,
                                         const unsigned char* __restrict__ vc, const int* __restrict__ sel,
                                         const int* __restrict__ sel_n, const int* __restrict__ tmax_p,
                                         const int* __restrict__ mode_p, const int* __restrict__ sel_max_p,
                                         float* __restrict__ out) {
    attn_sel_r_body<8, 1>(q, kc, vc, sel, sel_n, tmax_p, mode_p, sel_max_p, out); // #96 YaRN runtime scale
}
extern "C" __global__ void attn_sel_d9(const float* __restrict__ q, const unsigned char* __restrict__ kc,
                                       const unsigned char* __restrict__ vc, const int* __restrict__ sel,
                                       const int* __restrict__ sel_n, const int* __restrict__ tmax_p,
                                       const int* __restrict__ mode_p, const int* __restrict__ sel_max_p,
                                       float* __restrict__ out) {
    attn_sel_r_body<9, 0>(q, kc, vc, sel, sel_n, tmax_p, mode_p, sel_max_p, out);
}
extern "C" __global__ void attn_sel_d9_y(const float* __restrict__ q, const unsigned char* __restrict__ kc,
                                         const unsigned char* __restrict__ vc, const int* __restrict__ sel,
                                         const int* __restrict__ sel_n, const int* __restrict__ tmax_p,
                                         const int* __restrict__ mode_p, const int* __restrict__ sel_max_p,
                                         float* __restrict__ out) {
    attn_sel_r_body<9, 1>(q, kc, vc, sel, sel_n, tmax_p, mode_p, sel_max_p, out); // #96 YaRN runtime scale
}
// kv_ld: kv_load with an optional shared-memory e4m3 LUT (lut[b] = dec_e4m3(b): the same float, one load instead of
// the branchy decode). LUT = 0 is kv_load itself.
template <int LUT>
__device__ __forceinline__ float kv_ld(const unsigned char* p, int i, int mode, const float* lut) {
    if (LUT) { if (mode == 0) return lut[p[i]]; }
    return kv_load(p, i, mode);
}
// attn_sel_s (#10 step 4, 2026-09-06): attn_sel_r with the K and V rows staged through shared memory. Per chunk of
// CHB bytes (CHB / (256*esz) keys) all 256 threads fetch the rows with 16-byte loads into registers - the next chunk's
// loads are in flight while the current chunk is computed - store them to shared memory, and the UNCHANGED per-key op
// chain (acc += q*k over the same e order, the same warp reduction, the same IEEE division, o += w*v in list order)
// runs from shared memory. Only the loads move; every fma chain and reduction order is that of attn_sel_r -> meant
// bit-identical (gate: parity 8 + 512). Rows are 256*esz bytes at 256-byte-aligned offsets of one cuMemAlloc buffer,
// so the uint4 loads are aligned. CHB 16384 = 64 keys (e4m3) / 32 (bf16) per chunk; 8192 = 32 / 16.
template <int NV>
__device__ __forceinline__ void attn_fetch(uint4* pre, const unsigned char* __restrict__ base, size_t kvbase, int rb,
                                           int sh, const int* __restrict__ list, int j0, int n, int tmax, int d) {
#pragma unroll
    for (int i = 0; i < NV; i++) {
        int idx = d + 256 * i;
        int row = idx >> sh;
        int col = idx & ((1 << sh) - 1);
        int j = j0 + row;
        if (j < n) {
            int tok = list[j];
            if (tok < 0) tok = 0;
            if (tok >= tmax) tok = tmax - 1;
            pre[i] = *(const uint4*)(base + (kvbase + (size_t)tok) * rb + (size_t)col * 16);
        } else {
            pre[i] = make_uint4(0u, 0u, 0u, 0u);
        }
    }
}
template <int NV>
__device__ __forceinline__ void attn_store(const uint4* pre, unsigned char* kb, int d) {
#pragma unroll
    for (int i = 0; i < NV; i++) *(uint4*)(kb + (size_t)(d + 256 * i) * 16) = pre[i];
}
// #96: RT = the scale source (0 folded, 1 the YaRN runtime global), as attn_sel.
template <int CHB, int LUT, int RT>
__device__ __forceinline__ void attn_sel_s_body(const float* __restrict__ q, const unsigned char* __restrict__ kc,
                                                const unsigned char* __restrict__ vc, const int* __restrict__ sel,
                                                const int* __restrict__ sel_n, const int* __restrict__ tmax_p,
                                                const int* __restrict__ mode_p, const int* __restrict__ sel_max_p,
                                                float* __restrict__ out) {
    int head = blockIdx.x;
    int t = blockIdx.y;
    int d = threadIdx.x;
    int kvh = head / 12;
    int n = sel_n[t];
    if (n < 0) n = 0;
    const int sel_max = *sel_max_p;
    if (n > sel_max) n = sel_max;
    const int tmax = *tmax_p;
    const int* list = sel + (size_t)t * sel_max;
    const float* qt = q + ((size_t)t * 24 + head) * 256;
    __shared__ float p[2051];
    __shared__ float red[256];
    __shared__ __align__(16) unsigned char kb[CHB];
    __shared__ float lut[256];
    if (LUT) lut[threadIdx.x] = dec_e4m3((unsigned char)threadIdx.x); // read only after the first __syncthreads
    const int mode = *mode_p;
    const int esz = mode ? 2 : 1;
    const int rb = 256 * esz;           // bytes per K/V row
    const int sh = mode ? 5 : 4;        // log2(uint4 per row)
    const int KB = CHB / rb;            // keys per chunk
    constexpr int NV = CHB / 16 / 256;  // uint4 per thread per chunk
    const size_t kvbase = (size_t)kvh * tmax;
    int warp = d >> 5, lane = d & 31;
    const float scale = attn_scale_src<RT>(); // 1/sqrt(256)
    float qr[8];
#pragma unroll
    for (int k = 0; k < 8; k++) qr[k] = qt[lane + 32 * k];
    uint4 pre[NV];
    const int nch = (n + KB - 1) / KB;
    // K phase: the scores, each key's dot from shared memory (same e order, same warp reduction as attn_sel_r)
    if (nch > 0) attn_fetch<NV>(pre, kc, kvbase, rb, sh, list, 0, n, tmax, d);
    for (int c = 0; c < nch; c++) {
        const int j0 = c * KB;
        attn_store<NV>(pre, kb, d);
        __syncthreads();
        if (c + 1 < nch) attn_fetch<NV>(pre, kc, kvbase, rb, sh, list, j0 + KB, n, tmax, d);
        for (int jj = warp; jj < KB; jj += 8) {
            int j = j0 + jj;
            if (j < n) {
                const unsigned char* kp = kb + jj * rb;
                float acc = 0.0f;
#pragma unroll
                for (int k = 0; k < 8; k++) acc += qr[k] * kv_ld<LUT>(kp, lane + 32 * k, mode, lut);
                for (int o = 16; o > 0; o >>= 1) acc += __shfl_down_sync(0xffffffffu, acc, o);
                if (lane == 0) p[j] = acc * scale;
            }
        }
        __syncthreads();
    }
    __syncthreads();
    float mx = -3.0e38f;
    for (int j = d; j < n; j += 256) mx = fmaxf(mx, p[j]);
    red[d] = mx;
    __syncthreads();
    for (int st = 128; st > 0; st >>= 1) {
        if (d < st) red[d] = fmaxf(red[d], red[d + st]);
        __syncthreads();
    }
    mx = red[0];
    __syncthreads(); // thread 0 writes red[0] = sum below: without this barrier a late reader takes that sum as mx
    float sum = 0.0f;
    for (int j = d; j < n; j += 256) { float e = expf(p[j] - mx); p[j] = e; sum += e; }
    red[d] = sum;
    __syncthreads();
    for (int st = 128; st > 0; st >>= 1) {
        if (d < st) red[d] += red[d + st];
        __syncthreads();
    }
    sum = red[0];
    __syncthreads();
    for (int j = d; j < n; j += 256) p[j] = p[j] / sum; // the same division attn_sel does inline
    __syncthreads();
    // V phase: o accumulated in list order (the serial chain of attn_sel), the rows from shared memory
    float o = 0.0f;
    if (nch > 0) attn_fetch<NV>(pre, vc, kvbase, rb, sh, list, 0, n, tmax, d);
    for (int c = 0; c < nch; c++) {
        const int j0 = c * KB;
        attn_store<NV>(pre, kb, d);
        __syncthreads();
        if (c + 1 < nch) attn_fetch<NV>(pre, vc, kvbase, rb, sh, list, j0 + KB, n, tmax, d);
        const int jend = (j0 + KB < n) ? (j0 + KB) : n;
        for (int j = j0; j < jend; j++) o += p[j] * kv_ld<LUT>(kb + (j - j0) * rb, d, mode, lut);
        __syncthreads();
    }
    out[((size_t)t * 24 + head) * 256 + d] = o;
}
extern "C" __global__ void attn_sel_s(const float* __restrict__ q, const unsigned char* __restrict__ kc,
                                      const unsigned char* __restrict__ vc, const int* __restrict__ sel,
                                      const int* __restrict__ sel_n, const int* __restrict__ tmax_p,
                                      const int* __restrict__ mode_p, const int* __restrict__ sel_max_p,
                                      float* __restrict__ out) {
    attn_sel_s_body<16384, 0, 0>(q, kc, vc, sel, sel_n, tmax_p, mode_p, sel_max_p, out);
}
extern "C" __global__ void attn_sel_s_y(const float* __restrict__ q, const unsigned char* __restrict__ kc,
                                        const unsigned char* __restrict__ vc, const int* __restrict__ sel,
                                        const int* __restrict__ sel_n, const int* __restrict__ tmax_p,
                                        const int* __restrict__ mode_p, const int* __restrict__ sel_max_p,
                                        float* __restrict__ out) {
    attn_sel_s_body<16384, 0, 1>(q, kc, vc, sel, sel_n, tmax_p, mode_p, sel_max_p, out); // #96 YaRN runtime scale
}
extern "C" __global__ void attn_sel_s8(const float* __restrict__ q, const unsigned char* __restrict__ kc,
                                       const unsigned char* __restrict__ vc, const int* __restrict__ sel,
                                       const int* __restrict__ sel_n, const int* __restrict__ tmax_p,
                                       const int* __restrict__ mode_p, const int* __restrict__ sel_max_p,
                                       float* __restrict__ out) {
    attn_sel_s_body<8192, 0, 0>(q, kc, vc, sel, sel_n, tmax_p, mode_p, sel_max_p, out);
}
extern "C" __global__ void attn_sel_s8_y(const float* __restrict__ q, const unsigned char* __restrict__ kc,
                                         const unsigned char* __restrict__ vc, const int* __restrict__ sel,
                                         const int* __restrict__ sel_n, const int* __restrict__ tmax_p,
                                         const int* __restrict__ mode_p, const int* __restrict__ sel_max_p,
                                         float* __restrict__ out) {
    attn_sel_s_body<8192, 0, 1>(q, kc, vc, sel, sel_n, tmax_p, mode_p, sel_max_p, out); // #96 YaRN runtime scale
}
extern "C" __global__ void attn_sel_s8l(const float* __restrict__ q, const unsigned char* __restrict__ kc,
                                        const unsigned char* __restrict__ vc, const int* __restrict__ sel,
                                        const int* __restrict__ sel_n, const int* __restrict__ tmax_p,
                                        const int* __restrict__ mode_p, const int* __restrict__ sel_max_p,
                                        float* __restrict__ out) {
    attn_sel_s_body<8192, 1, 0>(q, kc, vc, sel, sel_n, tmax_p, mode_p, sel_max_p, out);
}
extern "C" __global__ void attn_sel_s8l_y(const float* __restrict__ q, const unsigned char* __restrict__ kc,
                                          const unsigned char* __restrict__ vc, const int* __restrict__ sel,
                                          const int* __restrict__ sel_n, const int* __restrict__ tmax_p,
                                          const int* __restrict__ mode_p, const int* __restrict__ sel_max_p,
                                          float* __restrict__ out) {
    attn_sel_s_body<8192, 1, 1>(q, kc, vc, sel, sel_n, tmax_p, mode_p, sel_max_p, out); // #96 YaRN runtime scale
}
// attn_sel_g (#10 step 5, 2026-09-06): ONE block per (KV head, query) computing the 12 q heads that share the K/V rows
// (grid (2, Tq), 384 threads = 12 warps, warp h = head kvh*12+h). The K and V rows are staged in 4 KB chunks (16 keys
// e4m3 / 8 keys bf16) through shared memory ONCE for the 12 heads, e4m3 decoded through the shared LUT. The softmax
// weights are never stored (12 x 2051 floats would not fit): the scores are recomputed per pass from the staged K rows -
// pass 1 the max (fmaxf is exact, order-free), pass 2 the sum in attn_sel's slot order (attn_sel's thread d summed the
// keys j = d mod 256 in increasing j, then the 8-step tree red[d] += red[d+st]; here slot s = lane + 32*k lives in
// register k of lane s & 31, and the same tree runs in registers and shuffles), pass 3 o += (e / sum) * v in list order.
// Every per-element op is attn_sel's (same fma chains, same shuffle tree, same expf, same IEEE division) -> meant
// bit-identical (gate: parity 8 + 512).
// #96: RT = the scale source (0 folded, 1 the YaRN runtime global), as attn_sel.
template <int RT>
__device__ __forceinline__ float g_score(const float* qr, const unsigned char* kp, int lane, int mode, const float* lut) {
    float acc = 0.0f;
#pragma unroll
    for (int k = 0; k < 8; k++) acc += qr[k] * kv_ld<1>(kp, lane + 32 * k, mode, lut);
    for (int o = 16; o > 0; o >>= 1) acc += __shfl_down_sync(0xffffffffu, acc, o);
    const float scale = attn_scale_src<RT>(); // 1/sqrt(256)
    return __shfl_sync(0xffffffffu, acc * scale, 0);
}
__device__ __forceinline__ uint4 g_fetch(const unsigned char* __restrict__ base, size_t kvbase, int rb, int sh,
                                         const int* __restrict__ list, int j0, int n, int tmax, int tid) {
    int row = tid >> sh;
    int col = tid & ((1 << sh) - 1);
    int j = j0 + row;
    if (j < n) {
        int tok = list[j];
        if (tok < 0) tok = 0;
        if (tok >= tmax) tok = tmax - 1;
        return *(const uint4*)(base + (kvbase + (size_t)tok) * rb + (size_t)col * 16);
    }
    return make_uint4(0u, 0u, 0u, 0u);
}
// #96: the body is a template on RT (0 folded scale, 1 the YaRN runtime
// global), inlined into the two __launch_bounds__ wrappers below.
template <int RT>
__device__ __forceinline__ void attn_sel_g_body(const float* __restrict__ q, const unsigned char* __restrict__ kc,
                                      const unsigned char* __restrict__ vc, const int* __restrict__ sel,
                                      const int* __restrict__ sel_n, const int* __restrict__ tmax_p,
                                      const int* __restrict__ mode_p, const int* __restrict__ sel_max_p,
                                      float* __restrict__ out) {
    const int kvh = blockIdx.x;
    const int t = blockIdx.y;
    const int tid = threadIdx.x;
    const int warp = tid >> 5, lane = tid & 31;
    const int head = kvh * 12 + warp;
    int n = sel_n[t];
    if (n < 0) n = 0;
    const int sel_max = *sel_max_p;
    if (n > sel_max) n = sel_max;
    const int tmax = *tmax_p;
    const int* list = sel + (size_t)t * sel_max;
    const float* qt = q + ((size_t)t * 24 + head) * 256;
    __shared__ float lut[256];
    __shared__ __align__(16) unsigned char kb[4096];
    __shared__ __align__(16) unsigned char vb[4096];
    const int mode = *mode_p;
    const int esz = mode ? 2 : 1;
    const int rb = 256 * esz;      // bytes per K/V row
    const int sh = mode ? 5 : 4;   // log2(uint4 per row)
    const int KG = 4096 / rb;      // keys per chunk
    const size_t kvbase = (size_t)kvh * tmax;
    const bool ld = tid < 256;     // the 256 fetching threads (one uint4 each per chunk)
    if (ld) lut[tid] = dec_e4m3((unsigned char)tid);
    float qr[8];
#pragma unroll
    for (int k = 0; k < 8; k++) qr[k] = qt[lane + 32 * k];
    const int nch = (n + KG - 1) / KG;
    uint4 pk = make_uint4(0u, 0u, 0u, 0u), pv = make_uint4(0u, 0u, 0u, 0u);
    // pass 1: the max of this head's scores
    float mx = -3.0e38f;
    if (nch > 0 && ld) pk = g_fetch(kc, kvbase, rb, sh, list, 0, n, tmax, tid);
    for (int c = 0; c < nch; c++) {
        const int j0 = c * KG;
        if (ld) *(uint4*)(kb + (size_t)tid * 16) = pk;
        __syncthreads();
        if (c + 1 < nch && ld) pk = g_fetch(kc, kvbase, rb, sh, list, j0 + KG, n, tmax, tid);
        const int jend = (j0 + KG < n) ? (j0 + KG) : n;
        for (int j = j0; j < jend; j++) mx = fmaxf(mx, g_score<RT>(qr, kb + (j - j0) * rb, lane, mode, lut));
        __syncthreads();
    }
    // pass 2: the sum, slot s = j mod 256 accumulated in increasing j (register s>>5 of lane s&31), then attn_sel's tree
    float ps[8];
#pragma unroll
    for (int k = 0; k < 8; k++) ps[k] = 0.0f;
    if (nch > 0 && ld) pk = g_fetch(kc, kvbase, rb, sh, list, 0, n, tmax, tid);
    for (int c = 0; c < nch; c++) {
        const int j0 = c * KG;
        if (ld) *(uint4*)(kb + (size_t)tid * 16) = pk;
        __syncthreads();
        if (c + 1 < nch && ld) pk = g_fetch(kc, kvbase, rb, sh, list, j0 + KG, n, tmax, tid);
        const int jend = (j0 + KG < n) ? (j0 + KG) : n;
        for (int j = j0; j < jend; j++) {
            float e = expf(g_score<RT>(qr, kb + (j - j0) * rb, lane, mode, lut) - mx);
            const int slot = j & 255;
            if ((slot & 31) == lane) {
                const int k = slot >> 5;
#pragma unroll
                for (int kk = 0; kk < 8; kk++) if (kk == k) ps[kk] += e;
            }
        }
        __syncthreads();
    }
#pragma unroll
    for (int k = 0; k < 4; k++) ps[k] += ps[k + 4];   // st = 128
    ps[0] += ps[2]; ps[1] += ps[3];                    // st = 64
    ps[0] += ps[1];                                    // st = 32
    for (int st = 16; st > 0; st >>= 1) {              // st = 16 .. 1 across lanes
        float v = __shfl_down_sync(0xffffffffu, ps[0], st);
        if (lane < st) ps[0] += v;
    }
    const float sum = __shfl_sync(0xffffffffu, ps[0], 0);
    // pass 3: o += (e / sum) * v in list order, K and V chunks staged together
    float o[8];
#pragma unroll
    for (int k = 0; k < 8; k++) o[k] = 0.0f;
    if (nch > 0 && ld) { pk = g_fetch(kc, kvbase, rb, sh, list, 0, n, tmax, tid); pv = g_fetch(vc, kvbase, rb, sh, list, 0, n, tmax, tid); }
    for (int c = 0; c < nch; c++) {
        const int j0 = c * KG;
        if (ld) { *(uint4*)(kb + (size_t)tid * 16) = pk; *(uint4*)(vb + (size_t)tid * 16) = pv; }
        __syncthreads();
        if (c + 1 < nch && ld) { pk = g_fetch(kc, kvbase, rb, sh, list, j0 + KG, n, tmax, tid); pv = g_fetch(vc, kvbase, rb, sh, list, j0 + KG, n, tmax, tid); }
        const int jend = (j0 + KG < n) ? (j0 + KG) : n;
        for (int j = j0; j < jend; j++) {
            float w = expf(g_score<RT>(qr, kb + (j - j0) * rb, lane, mode, lut) - mx) / sum; // the same division attn_sel does
            const unsigned char* vp = vb + (j - j0) * rb;
#pragma unroll
            for (int k = 0; k < 8; k++) o[k] += w * kv_ld<1>(vp, lane + 32 * k, mode, lut);
        }
        __syncthreads();
    }
#pragma unroll
    for (int k = 0; k < 8; k++) out[((size_t)t * 24 + head) * 256 + lane + 32 * k] = o[k];
}
extern "C" __global__ void __launch_bounds__(384) attn_sel_g(const float* __restrict__ q, const unsigned char* __restrict__ kc,
                                      const unsigned char* __restrict__ vc, const int* __restrict__ sel,
                                      const int* __restrict__ sel_n, const int* __restrict__ tmax_p,
                                      const int* __restrict__ mode_p, const int* __restrict__ sel_max_p,
                                      float* __restrict__ out) {
    attn_sel_g_body<0>(q, kc, vc, sel, sel_n, tmax_p, mode_p, sel_max_p, out);
}
extern "C" __global__ void __launch_bounds__(384) attn_sel_g_y(const float* __restrict__ q, const unsigned char* __restrict__ kc,
                                      const unsigned char* __restrict__ vc, const int* __restrict__ sel,
                                      const int* __restrict__ sel_n, const int* __restrict__ tmax_p,
                                      const int* __restrict__ mode_p, const int* __restrict__ sel_max_p,
                                      float* __restrict__ out) {
    attn_sel_g_body<1>(q, kc, vc, sel, sel_n, tmax_p, mode_p, sel_max_p, out); // #96 YaRN runtime scale
}
extern "C" __global__ void gate_mul(const float* __restrict__ core, const float* __restrict__ gate,
                                    float* __restrict__ out) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    float g = gate[i];
    out[i] = core[i] / (1.0f + expf(-g));
}
// flat FP8 cast kernels (KV self-check + reuse)
extern "C" __global__ void cast_e4m3_flat(const float* __restrict__ x, unsigned char* __restrict__ out,
                                          const int* __restrict__ n_p) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= *n_p) return;
    out[i] = enc_e4m3(x[i]);
}
extern "C" __global__ void dec_e4m3_flat(const unsigned char* __restrict__ x, float* __restrict__ out,
                                         const int* __restrict__ n_p) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= *n_p) return;
    out[i] = dec_e4m3(x[i]);
}
// full-flat add (PLE adds onto the [T][10240] stream, reference: hidden += ple(...))
extern "C" __global__ void add_flat(const float* __restrict__ in_, float* __restrict__ out,
                                    const int* __restrict__ n_p) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= *n_p) return;
    out[i] += in_[i];
}

// ---------------- QSA indexer (p14-verified math + engine additions) ----------------
extern "C" __global__ void rms128(const float* __restrict__ x, const float* __restrict__ w,
                                  float* __restrict__ out, const int* __restrict__ heads_p,
                                  const int* __restrict__ stride_p) {
    int heads = *heads_p;
    int stride = *stride_p;
    int head = blockIdx.x;
    int t = blockIdx.y;
    int d = threadIdx.x;
    const float* xp = x + (size_t)t * stride + head * 128;
    __shared__ float red[128];
    red[d] = xp[d] * xp[d];
    __syncthreads();
    for (int st = 64; st > 0; st >>= 1) {
        if (d < st) red[d] += red[d + st];
        __syncthreads();
    }
    float rms = rsqrtf(red[0] / 128.0f + 1e-6f);
    out[((size_t)t * heads + head) * 128 + d] = xp[d] * rms * (1.0f + w[d]);
}
// partial rotary 64 dims; p = (pos_base + t) * pos_mul (pos_base for chunked rows)
extern "C" __global__ void rope64(const float* __restrict__ x, const float* __restrict__ cos_,
                                  const float* __restrict__ sin_, float* __restrict__ out,
                                  const int* __restrict__ heads_p, const int* __restrict__ pos_mul_p,
                                  const int* __restrict__ stride_p, const int* __restrict__ pos_base_p) {
    int heads = *heads_p;
    int pos_mul = *pos_mul_p;
    int stride = *stride_p;
    int head = blockIdx.x;
    int t = blockIdx.y;
    int d = threadIdx.x;
    const float* xp = x + (size_t)t * stride + head * 128;
    float* op = out + ((size_t)t * heads + head) * 128;
    int p = (*pos_base_p + t) * pos_mul;
    if (d >= 32) {
        if (d >= 64) op[d] = xp[d];
        return;
    }
    float a = xp[d], b = xp[d + 32];
    float c = cos_[p * 32 + d], s = sin_[p * 32 + d];
    op[d] = a * c - b * s;
    op[d + 32] = b * c + a * s;
}
// pool a block of 4 raw keys from the compact [Tmax][128] indexer cache
extern "C" __global__ void pool4_cache(const float* __restrict__ keys, float* __restrict__ pooled,
                                       const int* __restrict__ stride_p, const int* __restrict__ block_base_p,
                                       const int* __restrict__ n_blocks_p, const int* __restrict__ ring_p) {
    int b = blockIdx.x;
    int d = threadIdx.x;
    if (b >= *n_blocks_p) return;
    // raw-key ring (row = pos % ring, ring % 4 == 0: a block's 4 rows never wrap)
    int row0 = (int)(((long long)(*block_base_p + b) * 4) % *ring_p);
    const float* kp = keys + (size_t)row0 * *stride_p + d;
    pooled[(size_t)b * 128 + d] =
        (kp[0] + kp[*stride_p] + kp[2 * *stride_p] + kp[3 * *stride_p]) * 0.25f;
}
// append the k-part (columns 512..640) of a [T][640] qk matrix to the cache
extern "C" __global__ void qk_k_append(const float* __restrict__ qk, float* __restrict__ keys,
                                       const int* __restrict__ row_base_p, const int* __restrict__ n_rows_p,
                                       const int* __restrict__ ring_p) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= *n_rows_p * 128) return;
    int t = i / 128;
    int d = i % 128;
    int row = (int)(((long long)*row_base_p + t) % *ring_p);
    keys[(size_t)row * 128 + d] = qk[(size_t)t * 640 + 512 + d];
}
// copy hd staging values into pooled[block_base] — graph-safe dynamic offset
// (the offset is read from the device buffer, which is refreshed per token by
// the scalar staging upload that precedes the graph launch)
extern "C" __global__ void d2d_block(float* __restrict__ dst_base,
                                     const float* __restrict__ src,
                                     const int* __restrict__ block_base_p,
                                     const int* __restrict__ hd_p) {
    int d = threadIdx.x;
    int hd = *hd_p;
    int b = *block_base_p;
    dst_base[(size_t)b * hd + d] = src[d];
}
// scores per query row: score[tq][b] = sum_h relu(q·pooled[b]) / sqrt(128)
extern "C" __global__ void qsa_scores(const float* __restrict__ q, const float* __restrict__ pooled,
                                      float* __restrict__ scores, const int* __restrict__ cap_p,
                                      const int* __restrict__ pos_base_p) {
    int tq = blockIdx.x;
    int d = threadIdx.x;
    int pos = *pos_base_p + tq;
    int ncb = (pos + 1) >> 2;
    if (ncb > *cap_p) ncb = *cap_p;
    __shared__ float qs[4 * 128];
    __shared__ float red[128];
    for (int h = 0; h < 4; h++) qs[h * 128 + d] = q[((size_t)tq * 4 + h) * 128 + d];
    __syncthreads();
    for (int b = 0; b < ncb; b++) {
        const float* pk = pooled + (size_t)b * 128;
        float ssum = 0.0f;
        for (int h = 0; h < 4; h++) {
            red[d] = qs[h * 128 + d] * pk[d];
            __syncthreads();
            for (int st = 64; st > 0; st >>= 1) {
                if (d < st) red[d] += red[d + st];
                __syncthreads();
            }
            ssum += fmaxf(red[0], 0.0f);
            __syncthreads();
        }
        if (d == 0) scores[(size_t)tq * *cap_p + b] = ssum * rsqrtf(128.0f);
    }
}
__device__ __forceinline__ unsigned int ordkey(float v) {
    unsigned int b = __float_as_uint(v);
    return (b & 0x80000000u) ? ~b : (b | 0x80000000u);
}
// EXACT per-query top-k block selection (radix refine on ordered keys, ties
// resolved lowest-index — set-equivalent to torch.topk within f32 ties).
// One block (256 threads) per query. Emits the token list (4 per selected
// block, ascending; tail tokens ascending) + count.
extern "C" __global__ void qsa_select(const float* __restrict__ scores, const int* __restrict__ ncb_p,
                                      int* __restrict__ sel_list, int* __restrict__ sel_n,
                                      const int* __restrict__ k_p, const int* __restrict__ cap_p,
                                      const int* __restrict__ sel_max_p, const int* __restrict__ pos_p) {
    int qi = blockIdx.x;
    int ncb = ncb_p[qi];
    int K = *k_p;
    if (K >= ncb) {
        // dense regime (below the selection budget): every complete block is
        // selected, plus the tail -> the list is simply 0..=pos. Identical to
        // the radix path's output, without its fixed 4-pass + bitmap cost.
        // (#97 / F1: the same shortcut qsa_select_fast and qsa_select_par_e
        // always carried — without it the threshold scan's `cum + c >= need`
        // is never satisfiable when K > ncb and the fill degenerates.)
        int pos = pos_p[qi];
        for (int i = threadIdx.x; i <= pos; i += blockDim.x) sel_list[qi * *sel_max_p + i] = i;
        if (threadIdx.x == 0) sel_n[qi] = pos + 1;
        return;
    }
    const float* row = scores + (size_t)qi * *cap_p;
    __shared__ unsigned int hist[256];
    __shared__ unsigned int bitmap[8192]; // 65536 blocks
    __shared__ unsigned int s_thr;
    __shared__ unsigned int s_above;
    __shared__ unsigned int eq_total;
    __shared__ unsigned int scnt[32];
    __shared__ unsigned int sbase[32];
    unsigned int thr = 0, above = 0;
    for (int i = threadIdx.x; i < 8192; i += blockDim.x) bitmap[i] = 0;
    __syncthreads();
    if (ncb > 0) {
        for (int r = 0; r < 4; r++) {
            int shift = 24 - 8 * r;
            __syncthreads();
            if (threadIdx.x < 256) hist[threadIdx.x] = 0;
            __syncthreads();
            for (int i = threadIdx.x; i < ncb; i += blockDim.x) {
                unsigned int key = ordkey(row[i]);
                bool match = (r == 0) || ((key >> (shift + 8)) == thr);
                if (match) atomicAdd(&hist[(key >> shift) & 0xFF], 1u);
            }
            __syncthreads();
            if (threadIdx.x == 0) {
                unsigned int need = K - above;
                unsigned int cum = 0;
                int B = 255;
                for (int b = 255; b >= 0; b--) {
                    unsigned int c = hist[b];
                    if (cum + c >= need) { B = b; break; }
                    cum += c;
                }
                above += cum;
                thr = (thr << 8) | (unsigned int)B;
                s_thr = thr;
                s_above = above;
            }
            __syncthreads();
            thr = s_thr;
            above = s_above;
        }
    }
    if (threadIdx.x == 0) eq_total = 0;
    __syncthreads();
    for (int i = threadIdx.x; i < ncb; i += blockDim.x) {
        unsigned int key = ordkey(row[i]);
        if (key > thr) atomicOr(&bitmap[i >> 5], 1u << (i & 31));
        else if (key == thr) atomicAdd(&eq_total, 1u);
    }
    __syncthreads();
    unsigned int fill = (ncb > 0) ? (K - above) : 0;
    if (eq_total <= fill) {
        for (int i = threadIdx.x; i < ncb; i += blockDim.x)
            if (ordkey(row[i]) == thr) atomicOr(&bitmap[i >> 5], 1u << (i & 31));
        __syncthreads();
    } else if (fill > 0) {
        // lowest-index fill among exactly-equal keys: ordered per-warp stripes
        int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
        int nwarps = blockDim.x >> 5;
        if (threadIdx.x < 32) scnt[threadIdx.x] = 0;
        __syncthreads();
        int stripe = (ncb + nwarps - 1) / nwarps;
        int lo = warp * stripe;
        int hi = min((warp + 1) * stripe, ncb);
        unsigned int cnt = 0;
        for (int i = lo + lane; i < hi; i += 32) if (ordkey(row[i]) == thr) cnt++;
        for (int o = 16; o > 0; o >>= 1) cnt += __shfl_down_sync(0xffffffffu, cnt, o);
        if (lane == 0) scnt[warp] = cnt;
        __syncthreads();
        if (threadIdx.x == 0) {
            unsigned int run = 0;
            for (int w = 0; w < nwarps; w++) { sbase[w] = run; run += scnt[w]; }
        }
        __syncthreads();
        if (lane == 0 && sbase[warp] < fill) {
            unsigned int remaining = fill - sbase[warp];
            for (int i = lo; i < hi && remaining > 0; i++) {
                if (ordkey(row[i]) == thr) {
                    atomicOr(&bitmap[i >> 5], 1u << (i & 31));
                    remaining--;
                }
            }
        }
        __syncthreads();
    }
    // deterministic token list: selected blocks ascending via per-warp stripes
    {
        int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
        int nwarps = blockDim.x >> 5;
        if (threadIdx.x < 32) scnt[threadIdx.x] = 0;
        __syncthreads();
        int wper = 8192 / nwarps; // bitmap words per warp
        int wlo = warp * wper;
        unsigned int cnt = 0;
        for (int w = wlo; w < wlo + wper; w++) cnt += __popc(bitmap[w]);
        // all lanes scanned the SAME words — cnt is already warp-uniform
        if (lane == 0) scnt[warp] = cnt;
        __syncthreads();
        if (threadIdx.x == 0) {
            unsigned int run = 0;
            for (int w = 0; w < nwarps; w++) { sbase[w] = run; run += scnt[w]; }
        }
        __syncthreads();
        unsigned int total = sbase[nwarps - 1] + scnt[nwarps - 1];
        unsigned int rank0 = sbase[warp];
        if (lane == 0) {
            unsigned int r = rank0;
            for (int w = wlo; w < wlo + wper; w++) {
                unsigned int bits = bitmap[w];
                while (bits) {
                    unsigned int lowest = bits & (~bits + 1u);
                    int b = (w << 5) + __ffs(lowest) - 1;
                    int slot = (int)(r * 4);
                    for (int c = 0; c < 4; c++) sel_list[qi * *sel_max_p + slot + c] = b * 4 + c;
                    r++;
                    bits ^= lowest;
                }
            }
            if (warp == nwarps - 1) {
                // this warp saw the last stripes: r == total here
                int pos = pos_p[qi];
                int tailn = (pos + 1) - 4 * ncb;
                int base = 4 * (int)total;
                for (int c = 0; c < tailn; c++) sel_list[qi * *sel_max_p + base + c] = 4 * ncb + c;
                sel_n[qi] = base + tailn;
            }
        }
    }
}

// qsa_select with the dense-regime shortcut (CROW_QSA_FAST, default on)
extern "C" __global__ void qsa_select_fast(const float* __restrict__ scores, const int* __restrict__ ncb_p,
                                      int* __restrict__ sel_list, int* __restrict__ sel_n,
                                      const int* __restrict__ k_p, const int* __restrict__ cap_p,
                                      const int* __restrict__ sel_max_p, const int* __restrict__ pos_p) {
    int qi = blockIdx.x;
    int ncb = ncb_p[qi];
    int K = *k_p;
    if (K >= ncb) {
        // dense regime (below the selection budget): every complete block is
        // selected, plus the tail -> the list is simply 0..=pos. Identical to
        // the radix path's output, without its fixed 4-pass + bitmap cost.
        int pos = pos_p[qi];
        for (int i = threadIdx.x; i <= pos; i += blockDim.x) sel_list[qi * *sel_max_p + i] = i;
        if (threadIdx.x == 0) sel_n[qi] = pos + 1;
        return;
    }
    const float* row = scores + (size_t)qi * *cap_p;
    __shared__ unsigned int hist[256];
    __shared__ unsigned int bitmap[8192]; // 65536 blocks
    __shared__ unsigned int s_thr;
    __shared__ unsigned int s_above;
    __shared__ unsigned int eq_total;
    __shared__ unsigned int scnt[32];
    __shared__ unsigned int sbase[32];
    unsigned int thr = 0, above = 0;
    for (int i = threadIdx.x; i < 8192; i += blockDim.x) bitmap[i] = 0;
    __syncthreads();
    if (ncb > 0) {
        for (int r = 0; r < 4; r++) {
            int shift = 24 - 8 * r;
            __syncthreads();
            if (threadIdx.x < 256) hist[threadIdx.x] = 0;
            __syncthreads();
            for (int i = threadIdx.x; i < ncb; i += blockDim.x) {
                unsigned int key = ordkey(row[i]);
                bool match = (r == 0) || ((key >> (shift + 8)) == thr);
                if (match) atomicAdd(&hist[(key >> shift) & 0xFF], 1u);
            }
            __syncthreads();
            if (threadIdx.x == 0) {
                unsigned int need = K - above;
                unsigned int cum = 0;
                int B = 255;
                for (int b = 255; b >= 0; b--) {
                    unsigned int c = hist[b];
                    if (cum + c >= need) { B = b; break; }
                    cum += c;
                }
                above += cum;
                thr = (thr << 8) | (unsigned int)B;
                s_thr = thr;
                s_above = above;
            }
            __syncthreads();
            thr = s_thr;
            above = s_above;
        }
    }
    if (threadIdx.x == 0) eq_total = 0;
    __syncthreads();
    for (int i = threadIdx.x; i < ncb; i += blockDim.x) {
        unsigned int key = ordkey(row[i]);
        if (key > thr) atomicOr(&bitmap[i >> 5], 1u << (i & 31));
        else if (key == thr) atomicAdd(&eq_total, 1u);
    }
    __syncthreads();
    unsigned int fill = (ncb > 0) ? (K - above) : 0;
    if (eq_total <= fill) {
        for (int i = threadIdx.x; i < ncb; i += blockDim.x)
            if (ordkey(row[i]) == thr) atomicOr(&bitmap[i >> 5], 1u << (i & 31));
        __syncthreads();
    } else if (fill > 0) {
        // lowest-index fill among exactly-equal keys: ordered per-warp stripes
        int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
        int nwarps = blockDim.x >> 5;
        if (threadIdx.x < 32) scnt[threadIdx.x] = 0;
        __syncthreads();
        int stripe = (ncb + nwarps - 1) / nwarps;
        int lo = warp * stripe;
        int hi = min((warp + 1) * stripe, ncb);
        unsigned int cnt = 0;
        for (int i = lo + lane; i < hi; i += 32) if (ordkey(row[i]) == thr) cnt++;
        for (int o = 16; o > 0; o >>= 1) cnt += __shfl_down_sync(0xffffffffu, cnt, o);
        if (lane == 0) scnt[warp] = cnt;
        __syncthreads();
        if (threadIdx.x == 0) {
            unsigned int run = 0;
            for (int w = 0; w < nwarps; w++) { sbase[w] = run; run += scnt[w]; }
        }
        __syncthreads();
        if (lane == 0 && sbase[warp] < fill) {
            unsigned int remaining = fill - sbase[warp];
            for (int i = lo; i < hi && remaining > 0; i++) {
                if (ordkey(row[i]) == thr) {
                    atomicOr(&bitmap[i >> 5], 1u << (i & 31));
                    remaining--;
                }
            }
        }
        __syncthreads();
    }
    // deterministic token list: selected blocks ascending via per-warp stripes
    {
        int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
        int nwarps = blockDim.x >> 5;
        if (threadIdx.x < 32) scnt[threadIdx.x] = 0;
        __syncthreads();
        int wper = 8192 / nwarps; // bitmap words per warp
        int wlo = warp * wper;
        unsigned int cnt = 0;
        for (int w = wlo; w < wlo + wper; w++) cnt += __popc(bitmap[w]);
        // all lanes scanned the SAME words — cnt is already warp-uniform
        if (lane == 0) scnt[warp] = cnt;
        __syncthreads();
        if (threadIdx.x == 0) {
            unsigned int run = 0;
            for (int w = 0; w < nwarps; w++) { sbase[w] = run; run += scnt[w]; }
        }
        __syncthreads();
        unsigned int total = sbase[nwarps - 1] + scnt[nwarps - 1];
        unsigned int rank0 = sbase[warp];
        if (lane == 0) {
            unsigned int r = rank0;
            for (int w = wlo; w < wlo + wper; w++) {
                unsigned int bits = bitmap[w];
                while (bits) {
                    unsigned int lowest = bits & (~bits + 1u);
                    int b = (w << 5) + __ffs(lowest) - 1;
                    int slot = (int)(r * 4);
                    for (int c = 0; c < 4; c++) sel_list[qi * *sel_max_p + slot + c] = b * 4 + c;
                    r++;
                    bits ^= lowest;
                }
            }
            if (warp == nwarps - 1) {
                // this warp saw the last stripes: r == total here
                int pos = pos_p[qi];
                int tailn = (pos + 1) - 4 * ncb;
                int base = 4 * (int)total;
                for (int c = 0; c < tailn; c++) sel_list[qi * *sel_max_p + base + c] = 4 * ncb + c;
                sel_n[qi] = base + tailn;
            }
        }
    }
}


// ------------- qsa_select_par (CROW_QSA_PAR default since 61b, 0 = fallback) -------------
// Same exact top-k selection as qsa_select_fast, spread over many blocks.
// Two launches per call:
//   qsa_select_par_h  grid (G, nq) x 256   12-bit histogram of the ordered keys
//                                          into h1[4096] (global, zero on entry)
//   qsa_select_par_e  grid (nq) x 1024     threshold refine (10 + 10 bits),
//                                          tie fill by lowest index, ascending
//                                          emit, and it zeroes h1 for the next call
// Output rule reproduced byte for byte from qsa_select_fast:
//   thr      = the K-th largest ordered key (radix digits, any digit widths)
//   above    = count of keys strictly greater than thr
//   fill     = K - above
//   selected = {i : key_i > thr} plus the first `fill` indices with key_i == thr
//   list     = for every selected block i ascending: 4i, 4i+1, 4i+2, 4i+3
//              then the tail tokens 4*ncb .. pos ascending
//   sel_n    = 4 * total + tail, total = above + min(eq_total, fill)
// The dense regime (K >= ncb) takes the same shortcut: the list is 0 .. pos.
#define QSA_PAR_BINS 4096
// suffix threshold search over nb bins of shared `sh` with 1024 threads.
// Writes the digit B and cum = sum over bins above B, matching the downward
// scan of qsa_select: the first b from the top whose inclusive suffix reaches
// `need`. nb must be a multiple of 1024.
// inclusive prefix sum over exactly 1024 threads (32 warp scans plus one warp
// scan of the warp totals): two barriers, against 20 for a shared-memory scan.
// *s_tot takes the total. tmp holds 32 words and must not be reused before the
// next barrier.
__device__ __forceinline__ unsigned int qsa_par_scan(unsigned int v, unsigned int* tmp,
                                                     unsigned int* s_tot) {
    int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    unsigned int x = v;
    #pragma unroll
    for (int d = 1; d < 32; d <<= 1) {
        unsigned int y = __shfl_up_sync(0xffffffffu, x, d);
        if (lane >= d) x += y;
    }
    if (lane == 31) tmp[warp] = x;
    __syncthreads();
    if (warp == 0) {
        unsigned int w = tmp[lane];
        #pragma unroll
        for (int d = 1; d < 32; d <<= 1) {
            unsigned int y = __shfl_up_sync(0xffffffffu, w, d);
            if (lane >= d) w += y;
        }
        tmp[lane] = w;
        if (lane == 31) *s_tot = w;
    }
    __syncthreads();
    if (warp > 0) x += tmp[warp - 1];
    return x;
}
__device__ __forceinline__ void qsa_par_thr(unsigned int* sh, int nb, unsigned int need,
                                            unsigned int* tmp, unsigned int* s_tot,
                                            unsigned int* s_b, unsigned int* s_cum) {
    int tid = threadIdx.x;
    int per = nb >> 10;
    unsigned int v = 0;
    for (int j = 0; j < per; j++) v += sh[tid * per + j];
    unsigned int pfx = qsa_par_scan(v, tmp, s_tot);
    unsigned int cum0 = *s_tot - pfx; // bins above this chunk
    if (cum0 < need && cum0 + v >= need) {
        unsigned int cum = cum0;
        int B = tid * per;
        for (int b = tid * per + per - 1; b >= tid * per; b--) {
            unsigned int c = sh[b];
            if (cum + c >= need) { B = b; break; }
            cum += c;
        }
        *s_b = (unsigned int)B;
        *s_cum = cum;
    }
    __syncthreads();
}
extern "C" __global__ void qsa_select_par_h(const float* __restrict__ scores, const int* __restrict__ ncb_p,
                                            unsigned int* __restrict__ h1,
                                            const int* __restrict__ k_p, const int* __restrict__ cap_p) {
    int qi = blockIdx.y;
    int ncb = ncb_p[qi];
    int K = *k_p;
    if (K >= ncb) return; // dense regime: the emit kernel writes 0..pos, h1 stays zero
    const float* row = scores + (size_t)qi * *cap_p;
    unsigned int* hq = h1 + (size_t)qi * QSA_PAR_BINS;
    __shared__ unsigned int sh[QSA_PAR_BINS];
    for (int b = threadIdx.x; b < QSA_PAR_BINS; b += blockDim.x) sh[b] = 0u;
    __syncthreads();
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < ncb; i += gridDim.x * blockDim.x)
        atomicAdd(&sh[ordkey(row[i]) >> 20], 1u);
    __syncthreads();
    for (int b = threadIdx.x; b < QSA_PAR_BINS; b += blockDim.x)
        if (sh[b]) atomicAdd(&hq[b], sh[b]);
}
extern "C" __global__ void qsa_select_par_e(const float* __restrict__ scores, const int* __restrict__ ncb_p,
                                            int* __restrict__ sel_list, int* __restrict__ sel_n,
                                            const int* __restrict__ k_p, const int* __restrict__ cap_p,
                                            const int* __restrict__ sel_max_p, const int* __restrict__ pos_p,
                                            unsigned int* __restrict__ h1) {
    int qi = blockIdx.x;
    int ncb = ncb_p[qi];
    int K = *k_p;
    int tid = threadIdx.x; // blockDim.x == 1024
    int smax = *sel_max_p;
    if (K >= ncb) {
        int pos = pos_p[qi];
        for (int i = tid; i <= pos; i += blockDim.x) sel_list[qi * smax + i] = i;
        if (tid == 0) sel_n[qi] = pos + 1;
        return;
    }
    const float* row = scores + (size_t)qi * *cap_p;
    unsigned int* hq = h1 + (size_t)qi * QSA_PAR_BINS;
    __shared__ unsigned int sh[QSA_PAR_BINS];
    __shared__ unsigned int tmp[32];
    __shared__ unsigned int s_b, s_cum, s_tot, s_gt, s_eq;
    // round A: the 12 top bits, histogram built by qsa_select_par_h
    for (int b = tid; b < QSA_PAR_BINS; b += blockDim.x) sh[b] = hq[b];
    __syncthreads();
    qsa_par_thr(sh, QSA_PAR_BINS, (unsigned int)K, tmp, &s_tot, &s_b, &s_cum);
    unsigned int pref = s_b;
    unsigned int above = s_cum;
    for (int b = tid; b < QSA_PAR_BINS; b += blockDim.x) hq[b] = 0u; // zero for the next call
    // rounds B: 10 bits each over the surviving prefix
    for (int r = 0; r < 2; r++) {
        int shift = 10 - 10 * r;
        __syncthreads();
        sh[tid] = 0u;
        __syncthreads();
        for (int i = tid; i < ncb; i += blockDim.x) {
            unsigned int key = ordkey(row[i]);
            if ((key >> (shift + 10)) == pref) atomicAdd(&sh[(key >> shift) & 0x3FFu], 1u);
        }
        __syncthreads();
        qsa_par_thr(sh, 1024, (unsigned int)K - above, tmp, &s_tot, &s_b, &s_cum);
        above += s_cum;
        pref = (pref << 10) | s_b;
    }
    unsigned int thr = pref;
    // one contiguous chunk per thread: ascending index order is the emit order
    int per = (ncb + 1023) >> 10;
    int lo = tid * per;
    int hi = min(ncb, lo + per);
    unsigned int gt = 0u, eq = 0u;
    for (int i = lo; i < hi; i++) {
        unsigned int key = ordkey(row[i]);
        if (key > thr) gt++;
        else if (key == thr) eq++;
    }
    unsigned int gt_before = qsa_par_scan(gt, tmp, &s_gt) - gt;
    __syncthreads(); // tmp is reused by the second scan
    unsigned int eq_before = qsa_par_scan(eq, tmp, &s_eq) - eq;
    unsigned int gt_total = s_gt;
    unsigned int eq_total = s_eq;
    unsigned int fill = (gt_total < (unsigned int)K) ? ((unsigned int)K - gt_total) : 0u;
    unsigned int total = gt_total + ((eq_total < fill) ? eq_total : fill);
    unsigned int g = gt_before, e = eq_before;
    for (int i = lo; i < hi; i++) {
        unsigned int key = ordkey(row[i]);
        unsigned int r = 0u;
        bool sel = false;
        if (key > thr) { r = g + ((e < fill) ? e : fill); sel = true; g++; }
        else if (key == thr) { if (e < fill) { r = g + e; sel = true; } e++; }
        if (sel) {
            int slot = qi * smax + (int)(r * 4u);
            for (int c = 0; c < 4; c++) sel_list[slot + c] = i * 4 + c;
        }
    }
    __syncthreads();
    if (tid == 0) {
        int pos = pos_p[qi];
        int tailn = (pos + 1) - 4 * ncb;
        int base = 4 * (int)total;
        for (int c = 0; c < tailn; c++) sel_list[qi * smax + base + c] = 4 * ncb + c;
        sel_n[qi] = base + tailn;
    }
}
// ---------------- router (softmax + top-10 + residency + pointers) ----------------
// grid (T), block 512. Emits per token: ids[10], normalized weights[10],
// separate gate_up / down weight pointer pairs (VRAM or pinned UVA — residency
// invisible), cold bitmask. Updates per-layer u64 counters [selections, cold]
// and per-expert u64 selection counts (warm-up bookkeeping, every routed
// choice counts — hit or miss, spec 2.2).
extern "C" __global__ void router_top10(const float* __restrict__ logits,
                                        const unsigned int* __restrict__ bitmap,
                                        int* __restrict__ ids, float* __restrict__ wts,
                                        unsigned long long* __restrict__ gu_ptrs,
                                        unsigned long long* __restrict__ dn_ptrs,
                                        const unsigned long long* __restrict__ table,
                                        unsigned int* __restrict__ cold_mask,
                                        unsigned long long* __restrict__ counters,
                                        unsigned long long* __restrict__ sel_counts) {
    int t = blockIdx.x;
    int e = threadIdx.x;
    __shared__ float p[512];
    __shared__ float red[512];
    __shared__ int s_ids[10];
    __shared__ float s_ws[10];
    __shared__ float s_sum;
    __shared__ unsigned int s_cold;
    float l = logits[(size_t)t * 512 + e];
    red[e] = l;
    __syncthreads();
    for (int st = 256; st > 0; st >>= 1) {
        if (e < st) red[e] = fmaxf(red[e], red[e + st]);
        __syncthreads();
    }
    float mx = red[0];
    // every thread must have read red[0] before it is overwritten below —
    // without this barrier warp 0 could store its `ex` into red[0] while a
    // slower warp still reads it as the max (measured 2026-09-04: the routed
    // top-10 of ~9 of 512 prompt tokens changed run to run)
    __syncthreads();
    float ex = expf(l - mx);
    p[e] = ex;
    red[e] = ex;
    __syncthreads();
    for (int st = 256; st > 0; st >>= 1) {
        if (e < st) red[e] += red[e + st];
        __syncthreads();
    }
    p[e] = ex / red[0];
    if (e == 0) { s_sum = 0.0f; s_cold = 0; }
    for (int j = 0; j < 10; j++) {
        if (e == 0) s_ids[j] = 512;
        __syncthreads();
        red[e] = p[e];
        __syncthreads();
        for (int st = 256; st > 0; st >>= 1) {
            if (e < st) red[e] = fmaxf(red[e], red[e + st]);
            __syncthreads();
        }
        float best = red[0];
        if (p[e] == best) atomicMin(&s_ids[j], (int)e);
        __syncthreads();
        int pick = s_ids[j];
        if (e == 0) { s_ws[j] = p[pick]; s_sum += p[pick]; }
        // thread 0 reads p[pick] above while thread `pick` retires it below
        __syncthreads();
        if (e == pick) p[e] = -1.0f;
        __syncthreads();
    }
    if (e < 10) {
        int id = s_ids[e];
        wts[t * 10 + e] = s_ws[e] / s_sum;
        ids[t * 10 + e] = id;
        gu_ptrs[t * 10 + e] = table[id * 2];
        dn_ptrs[t * 10 + e] = table[id * 2 + 1];
        atomicAdd(&sel_counts[id], 1ull);
        unsigned int bit = (bitmap[id >> 5] >> (id & 31)) & 1u;
        if (!bit) atomicOr(&s_cold, 1u << e);
    }
    __syncthreads();
    if (e == 0) {
        cold_mask[t] = s_cold;
        atomicAdd(&counters[0], 10ull);
        atomicAdd(&counters[1], (unsigned long long)__popc(s_cold));
    }
}

// ---------------- cold-expert staging (CROW_STAGE, decode) ----------------
// The zero-copy MMA read of a cold expert issues 4-byte loads scattered over
// 16 rows per warp instruction - over PCIe every such load is its own
// transaction (measured: gemv_fp4_mma 1.1 ms/call at 354 cold experts/token).
// This kernel first pulls every COLD combo into a VRAM staging slot with
// fully coalesced 16-byte loads (a warp moves 512 contiguous bytes per
// instruction, SPLIT blocks per expert keep enough PCIe requests in flight),
// then rewrites the combo pointer tables so the GEMVs read VRAM. Hot combos
// keep their VRAM slab pointer (no copy). grid (combos, 2 = gate_up|down,
// SPLIT), block 256. Byte counts are multiples of 16 * SPLIT (asserted host-side).
extern "C" __global__ void stage_cold(const unsigned long long* __restrict__ gu_ptrs,
                                      const unsigned long long* __restrict__ dn_ptrs,
                                      const unsigned int* __restrict__ cold_mask,
                                      unsigned char* __restrict__ stage_gu,
                                      unsigned char* __restrict__ stage_dn,
                                      unsigned long long* __restrict__ sgu_ptrs,
                                      unsigned long long* __restrict__ sdn_ptrs,
                                      const int* __restrict__ gu_bytes_p,
                                      const int* __restrict__ dn_bytes_p) {
    int combo = blockIdx.x;
    int which = blockIdx.y;
    int split = gridDim.z;
    int part = blockIdx.z;
    int t = combo / 10, j = combo % 10;
    bool cold = (cold_mask[t] >> j) & 1u;
    size_t bytes = (size_t)(which ? *dn_bytes_p : *gu_bytes_p);
    const unsigned char* src = (const unsigned char*)(which ? dn_ptrs[combo] : gu_ptrs[combo]);
    unsigned char* dst = (which ? stage_dn : stage_gu) + (size_t)combo * bytes;
    if (part == 0 && threadIdx.x == 0) {
        unsigned long long p = cold ? (unsigned long long)dst : (unsigned long long)src;
        if (which) sdn_ptrs[combo] = p; else sgu_ptrs[combo] = p;
    }
    if (!cold) return;
    size_t n16 = bytes >> 4;               // uint4 units
    size_t per = n16 / split;              // per split block (exact: host asserts)
    const uint4* s4 = (const uint4*)src + part * per;
    uint4* d4 = (uint4*)dst + part * per;
    size_t i = threadIdx.x;
    // 4 independent loads in flight per thread before the first store
    for (; i + 3 * blockDim.x < per; i += 4 * blockDim.x) {
        uint4 a = s4[i], b = s4[i + blockDim.x], c = s4[i + 2 * blockDim.x], d = s4[i + 3 * blockDim.x];
        d4[i] = a; d4[i + blockDim.x] = b; d4[i + 2 * blockDim.x] = c; d4[i + 3 * blockDim.x] = d;
    }
    for (; i < per; i += blockDim.x) d4[i] = s4[i];
}

// ------------- #19d: persistent cp.async.cg staging (CROW_STAGE_KERNEL=2) -------------
// Same nine inputs and the same two outputs as stage_cold, plus the combo count
// n_combo_arg BY VALUE: a persistent grid cannot read it from gridDim.x, and the
// host passes the very t * TOPK that it bounds-checked against stage.max at the
// launch site (gen.rs), so the device item count and the host bound can never
// disagree and no device scalar has to be refreshed for this kernel (fix I2).
// 19c measured a device-issued read of pinned host memory at 52.9 GB/s through
// cp.async.cg.shared.global into 4 KB shared tiles on 40 x 256 blocks, against
// 34.1 GB/s for the stage_cold shape in the same process.
// Grid: G x 1 x 1 (CROW_STAGE_BLOCKS, default 40), block 256, one launch per layer.
// Work item = (which, combo) over t * TOPK combos and both matrices:
//   - the pointer-table entry of an item is written by the single owning block
//     (item % gridDim.x), thread 0, so every combo is owned by exactly one block
//     per matrix and every entry is written before the kernel ends, hence before
//     any GEMV of the layer reads the tables;
//   - the TILES of a cold item are split over ALL blocks (block b takes tile b,
//     b + G, b + 2G, ...), because a decode layer has 2.55 cold combos of 10 on
//     average: one item per block would leave 35 of 40 blocks idle and the bytes
//     in flight would follow the cold count instead of G.
// Static partition, no atomics, no shared counters, so the kernel is
// graph-capturable and deterministic.
// Tile 4096 B = 256 threads x 16 B. REQUIREMENT: both staged byte counts are
// exact multiples of 4096, asserted host-side at the launch site (gen.rs); a
// container whose slabs are not 4 KB multiples must set CROW_STAGE_KERNEL=1 and
// run stage_cold (19e made kernel 2 the default). The tail-tile branch of the first 19d draft was removed in fix
// round 1: it was unreachable here and therefore never executed (fix I3).
// This container: gate_up 1843200 = 450 x 4096, down 921600 = 225 x 4096.
// blockDim.x MUST be 256: the shared tile is 256 x 16 B and every thread owns
// smem[buf][threadIdx.x]; the single launch site hardcodes 256.
// Double buffer: two 4 KB shared tiles; the loads of the next tile are committed
// before cp.async.wait_group 1 releases the current tile for the 16 B stores.
extern "C" __global__ void stage_cold_ca(const unsigned long long* __restrict__ gu_ptrs,
                                         const unsigned long long* __restrict__ dn_ptrs,
                                         const unsigned int* __restrict__ cold_mask,
                                         unsigned char* __restrict__ stage_gu,
                                         unsigned char* __restrict__ stage_dn,
                                         unsigned long long* __restrict__ sgu_ptrs,
                                         unsigned long long* __restrict__ sdn_ptrs,
                                         const int* __restrict__ gu_bytes_p,
                                         const int* __restrict__ dn_bytes_p,
                                         const unsigned long long n_combo_arg) {
    __shared__ uint4 smem[2][256];             // 2 x 4096 B double buffer
    const int n_combo = (int)n_combo_arg;
    const unsigned int tid = threadIdx.x;
    const unsigned int G = gridDim.x;
    const size_t soff = (size_t)tid << 4;      // byte offset of this thread in a tile
    for (int which = 0; which < 2; ++which) {
        const size_t bytes = (size_t)(which ? *dn_bytes_p : *gu_bytes_p);
        const size_t ntile = bytes >> 12;      // host asserts bytes % 4096 == 0
        for (int combo = 0; combo < n_combo; ++combo) {
            int tk = combo / 10, j = combo % 10;
            bool cold = (cold_mask[tk] >> j) & 1u;
            const unsigned char* src = (const unsigned char*)(which ? dn_ptrs[combo] : gu_ptrs[combo]);
            unsigned char* dst = (which ? stage_dn : stage_gu) + (size_t)combo * bytes;
            if (tid == 0 && (unsigned int)(which * n_combo + combo) % G == blockIdx.x) {
                unsigned long long p = cold ? (unsigned long long)dst : (unsigned long long)src;
                if (which) sdn_ptrs[combo] = p; else sgu_ptrs[combo] = p;
            }
            if (!cold) continue;
            size_t idx = blockIdx.x;
            if (idx >= ntile) continue;
            int buf = 0;
            size_t base = idx << 12;
            {
                unsigned int sa = (unsigned int)__cvta_generic_to_shared(&smem[0][tid]);
                asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" :: "r"(sa), "l"(src + base + soff) : "memory");
            }
            asm volatile("cp.async.commit_group;" ::: "memory");
            for (;;) {
                size_t nidx = idx + G;
                size_t nbase = nidx << 12;
                if (nidx < ntile) {
                    unsigned int sa = (unsigned int)__cvta_generic_to_shared(&smem[buf ^ 1][tid]);
                    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" :: "r"(sa), "l"(src + nbase + soff) : "memory");
                    asm volatile("cp.async.commit_group;" ::: "memory");
                    asm volatile("cp.async.wait_group 1;" ::: "memory");
                } else {
                    asm volatile("cp.async.wait_group 0;" ::: "memory");
                }
                __syncthreads();
                *(uint4*)(dst + base + soff) = smem[buf][tid];
                if (nidx >= ntile) break;
                __syncthreads();
                idx = nidx; base = nbase; buf ^= 1;
            }
            __syncthreads();
        }
    }
}

// ---------------- prefill MoE: expert-grouped GEMM (CROW_PF_GEMM) ----------------
// The per-combo GEMV read every routed expert once PER TOKEN (t=512: 5120
// expert reads of 2.76 MB, most of them cold over PCIe). The grouped path
// sorts the combos by expert (moe_align principle, on device), cuts the
// selections into tiles of <= 8 tokens of one expert, stages the cold
// experts of a tile GROUP into VRAM once (coalesced copy), and runs ONE
// block-scaled mma per (16 weight rows, 8 tokens, 64 k) - the n dimension of
// m16n8k64 finally carries 8 distinct tokens instead of a broadcast row.
// Per-token math (k order, residual levels, KS split) is identical to
// gemv_fp4_mma, so the outputs are bit-identical to the GEMV path.

// counts[e] += 1 per routed combo (counts zeroed by the previous moe_plan)
extern "C" __global__ void moe_count(const int* __restrict__ ids, unsigned int* __restrict__ counts,
                                     const int* __restrict__ n_p) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= *n_p) return;
    atomicAdd(&counts[ids[i]], 1u);
}
// single block: exclusive scan -> offsets[513]; tile list (e, start, n,
// slot|first-flag) grouped TG tiles at a time; cursor[] and counts[] zeroed
// for the scatter / the next layer. tiles.w = slot (index of the expert
// within its group) | 0x10000 if this tile is the expert's first tile in
// the group (that block performs the staging copy).
extern "C" __global__ void moe_plan(unsigned int* __restrict__ counts, int* __restrict__ offsets,
                                    int4* __restrict__ tiles, int* __restrict__ n_tiles,
                                    unsigned int* __restrict__ cursor, const int* __restrict__ tg_p,
                                    const int* __restrict__ max_tiles_p) {
    __shared__ int s_off[513];
    if (threadIdx.x == 0) {
        int run = 0;
        for (int e = 0; e < 512; e++) { s_off[e] = run; run += (int)counts[e]; }
        s_off[512] = run;
        int tg = *tg_p;
        int nt = 0, slot = 0, group_start = 0;
        int last_e = -1;
        for (int e = 0; e < 512; e++) {
            int cnt = (int)counts[e];
            for (int s = 0; s < cnt; s += 8) {
                if (nt >= *max_tiles_p) break;
                if (nt - group_start >= tg) { group_start = nt; slot = 0; last_e = -1; }
                int first = (e != last_e) ? 1 : 0;
                if (first && last_e >= 0) slot++;
                last_e = e;
                tiles[nt] = make_int4(e, s_off[e] + s, min(8, cnt - s), slot | (first << 16));
                nt++;
            }
        }
        *n_tiles = nt;
    }
    __syncthreads();
    for (int e = threadIdx.x; e < 513; e += blockDim.x) offsets[e] = s_off[e];
    for (int e = threadIdx.x; e < 512; e += blockDim.x) { cursor[e] = 0u; counts[e] = 0u; }
}
// perm[offsets[e] + k] = combo for the k-th combo routed to expert e
extern "C" __global__ void moe_scatter(const int* __restrict__ ids, const int* __restrict__ offsets,
                                       unsigned int* __restrict__ cursor, int* __restrict__ perm,
                                       const int* __restrict__ n_p) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= *n_p) return;
    int e = ids[i];
    unsigned int k = atomicAdd(&cursor[e], 1u);
    perm[offsets[e] + k] = i;
}
// stage the cold experts of tile group `group` (TG tiles) into VRAM slots and
// write per-tile weight pointers (staged slot or hot VRAM slab).
// grid (TG, 2 = gate_up|down, SPLIT), block 256.
extern "C" __global__ void stage_tiles(const int4* __restrict__ tiles, const int* __restrict__ n_tiles_p,
                                       const int* __restrict__ group_p, const int* __restrict__ tg_p,
                                       const unsigned long long* __restrict__ table,
                                       const unsigned int* __restrict__ bitmap,
                                       unsigned char* __restrict__ stage_gu, unsigned char* __restrict__ stage_dn,
                                       unsigned long long* __restrict__ eptr,
                                       const int* __restrict__ gu_bytes_p, const int* __restrict__ dn_bytes_p,
                                       const unsigned char* __restrict__ pin_gu, const unsigned char* __restrict__ pin_dn,
                                       const unsigned char* __restrict__ ring_gu, const unsigned char* __restrict__ ring_dn) {
    int ti = (*group_p) * (*tg_p) + blockIdx.x;
    if (ti >= *n_tiles_p) return;
    int4 tl = tiles[ti];
    int e = tl.x;
    int which = blockIdx.y;
    int slot = (tl.w & 0xFFFF) + ((tg_p[1] > 1) ? ((*group_p) & 1) * (*tg_p) : 0); // CROW_PF_ASYNC: slot set per group parity
    bool first = (tl.w >> 16) & 1;
    bool cold = !((bitmap[e >> 5] >> (e & 31)) & 1u);
    size_t bytes = (size_t)(which ? *dn_bytes_p : *gu_bytes_p);
    const unsigned char* src = (const unsigned char*)table[e * 2 + which];
    unsigned char* dst = (which ? stage_dn : stage_gu) + (size_t)slot * bytes;
    if (blockIdx.z == 0 && threadIdx.x == 0) {
        eptr[ti * 2 + which] = cold ? (unsigned long long)dst : (unsigned long long)src;
    }
    if (!cold || !first) return;
    if (tg_p[2] >= 2) return; // CROW_PF_ASYNC=2: the copy engine stages (host memcpys ahead on the side stream); =3 diagnostic, no copy
    // copy-engine prefetch (A-P3b): the layer's whole pinned cold slab was
    // DMA'd into a VRAM ring on a side stream; read the expert from there
    // (same offset) instead of pulling it over PCIe here
    const unsigned char* ring = which ? ring_dn : ring_gu;
    if (ring) src = ring + (src - (which ? pin_dn : pin_gu));
    size_t n16 = bytes >> 4;
    size_t per = n16 / gridDim.z;
    const uint4* s4 = (const uint4*)src + blockIdx.z * per;
    uint4* d4 = (uint4*)dst + blockIdx.z * per;
    size_t i = threadIdx.x;
    for (; i + 3 * blockDim.x < per; i += 4 * blockDim.x) {
        uint4 a = s4[i], b = s4[i + blockDim.x], c = s4[i + 2 * blockDim.x], d = s4[i + 3 * blockDim.x];
        d4[i] = a; d4[i + blockDim.x] = b; d4[i + 2 * blockDim.x] = c; d4[i + 3 * blockDim.x] = d;
    }
    for (; i < per; i += blockDim.x) d4[i] = s4[i];
}
// tile GEMM: 64 weight rows per block (4 row groups x 16), 8 tokens per tile
// as the n columns, blockDim = 128 * KS (k split, smem reduce, fixed order).
// grid (rows/64, TG); y[combo][row] compact (rows = gridDim.x * 64).
extern "C" __global__ void gemm_fp4_tiles(const int4* __restrict__ tiles, const int* __restrict__ n_tiles_p,
                                          const int* __restrict__ group_p, const int* __restrict__ tg_p,
                                          const unsigned long long* __restrict__ eptr, const int* __restrict__ which_p,
                                          const unsigned char* __restrict__ xq, const int* __restrict__ perm,
                                          float* __restrict__ y, const int* __restrict__ k_dim_p,
                                          const int* __restrict__ xdiv_p, const float* __restrict__ gs_ptr) {
    int ti = (*group_p) * (*tg_p) + blockIdx.y;
    if (ti >= *n_tiles_p) return;
    int4 tl = tiles[ti];
    int k_dim = *k_dim_p;
    int bpr = k_dim >> 6;
    const unsigned char* w = (const unsigned char*)eptr[ti * 2 + *which_p];
    int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    int ks_n = blockDim.x >> 7;
    int rg = warp & 3, ks = warp >> 2;
    int g = lane >> 2, t = lane & 3;
    int w0 = (blockIdx.x << 6) + (rg << 4);
    int rows = gridDim.x << 6;
    // column g = token g of the tile (columns past n replay token 0, stores masked)
    int cg = perm[tl.y + ((g < tl.z) ? g : 0)];
    const unsigned char* xa = xq + (size_t)(cg / *xdiv_p) * bpr * 108;
    const unsigned char* rowg = w + (size_t)(w0 + g) * bpr * 36;
    const unsigned char* rowg8 = rowg + 8 * bpr * 36;
    const unsigned char* sfrow = (t & 1) ? rowg8 : rowg;
    float d0 = 0.0f, d1 = 0.0f, d2 = 0.0f, d3 = 0.0f;
    int bpb = (bpr + ks_n - 1) / ks_n;
    int b_lo = ks * bpb, b_hi = min(bpr, b_lo + bpb);
    for (int b = b_lo; b < b_hi; b++) {
        const unsigned char* blk = rowg + b * 36;
        const unsigned char* blk8 = rowg8 + b * 36;
        const unsigned char* ab = xa + b * 36;
        unsigned int sa = sfrow[b * 36 + 0] | (unsigned int)sfrow[b * 36 + 1] << 8
                        | (unsigned int)sfrow[b * 36 + 2] << 16 | (unsigned int)sfrow[b * 36 + 3] << 24;
        unsigned int a0 = *(const unsigned int*)(blk + 4 + 4 * t);
        unsigned int a1 = *(const unsigned int*)(blk8 + 4 + 4 * t);
        unsigned int a2 = *(const unsigned int*)(blk + 20 + 4 * t);
        unsigned int a3 = *(const unsigned int*)(blk8 + 20 + 4 * t);
        unsigned int sb = ab[0] | (unsigned int)ab[1] << 8
                        | (unsigned int)ab[2] << 16 | (unsigned int)ab[3] << 24;
        unsigned int b0 = *(const unsigned int*)(ab + 4 + 4 * t);
        unsigned int b1 = *(const unsigned int*)(ab + 20 + 4 * t);
        mma_fp4_16n8k64(d0, d1, d2, d3, a0, a1, a2, a3, b0, b1, sa, sb);
        #pragma unroll
        for (int lv = 1; lv < 3; lv++) {
            const unsigned char* ab2 = ab + lv * bpr * 36;
            unsigned int sb2 = ab2[0] | (unsigned int)ab2[1] << 8
                             | (unsigned int)ab2[2] << 16 | (unsigned int)ab2[3] << 24;
            unsigned int c0 = *(const unsigned int*)(ab2 + 4 + 4 * t);
            unsigned int c1 = *(const unsigned int*)(ab2 + 20 + 4 * t);
            mma_fp4_16n8k64(d0, d1, d2, d3, a0, a1, a2, a3, c0, c1, sa, sb2);
        }
    }
    __shared__ float red[4][4][32][4]; // [ks][rg][lane][d0..d3]
    red[ks][rg][lane][0] = d0; red[ks][rg][lane][1] = d1; red[ks][rg][lane][2] = d2; red[ks][rg][lane][3] = d3;
    __syncthreads();
    if (ks == 0) {
        float s0 = 0.0f, s1 = 0.0f, s2 = 0.0f, s3 = 0.0f;
        for (int i = 0; i < ks_n; i++) {
            s0 += red[i][rg][lane][0]; s1 += red[i][rg][lane][1];
            s2 += red[i][rg][lane][2]; s3 += red[i][rg][lane][3];
        }
        float gs = gs_ptr[0];
        int n0 = 2 * t, n1 = 2 * t + 1;
        if (n0 < tl.z) {
            size_t c0 = (size_t)perm[tl.y + n0] * rows;
            y[c0 + w0 + g] = s0 * gs; y[c0 + w0 + g + 8] = s2 * gs;
        }
        if (n1 < tl.z) {
            size_t c1 = (size_t)perm[tl.y + n1] * rows;
            y[c1 + w0 + g] = s1 * gs; y[c1 + w0 + g + 8] = s3 * gs;
        }
    }
}
// silu(gate)*up for the combos of tile group `group`: grid (TG, 8), block 256
extern "C" __global__ void silu_tiles(const float* __restrict__ h1, float* __restrict__ h2,
                                      const int4* __restrict__ tiles, const int* __restrict__ n_tiles_p,
                                      const int* __restrict__ group_p, const int* __restrict__ tg_p,
                                      const int* __restrict__ perm) {
    int ti = (*group_p) * (*tg_p) + blockIdx.x;
    if (ti >= *n_tiles_p) return;
    int4 tl = tiles[ti];
    int j = blockIdx.y;
    if (j >= tl.z) return;
    size_t c = (size_t)perm[tl.y + j];
    for (int i = threadIdx.x; i < 640; i += blockDim.x) {
        float gate = h1[c * 1280 + i];
        h2[c * 640 + i] = (gate / (1.0f + expf(-gate))) * h1[c * 1280 + 640 + i];
    }
}
// per-combo activation quant (k=640) for the combos of tile group `group`:
// grid (TG, 8), block 128 - same 3-level cascade as quant_x_fp4
extern "C" __global__ void quant_tiles(const float* __restrict__ h2, unsigned char* __restrict__ xq,
                                       const int4* __restrict__ tiles, const int* __restrict__ n_tiles_p,
                                       const int* __restrict__ group_p, const int* __restrict__ tg_p,
                                       const int* __restrict__ perm) {
    int ti = (*group_p) * (*tg_p) + blockIdx.x;
    if (ti >= *n_tiles_p) return;
    int4 tl = tiles[ti];
    int j = blockIdx.y;
    if (j >= tl.z) return;
    size_t c = (size_t)perm[tl.y + j];
    const int bpr = 10;
    const float* xp = h2 + c * 640;
    unsigned char* op = xq + c * bpr * 108;
    for (int sb = threadIdx.x; sb < bpr * 4; sb += blockDim.x) {
        int b = sb >> 2, s = sb & 3;
        const float* p = xp + b * 64 + s * 16;
        float r0[16], r1[16];
        quant_level(p, r0, op + b * 36 + s, op + b * 36 + 4 + s * 8);
        quant_level(r0, r1, op + bpr * 36 + b * 36 + s, op + bpr * 36 + b * 36 + 4 + s * 8);
        quant_level(r1, r0, op + 2 * bpr * 36 + b * 36 + s, op + 2 * bpr * 36 + b * 36 + 4 + s * 8);
    }
}

// ---------------- fused activation quant (CROW_QFUSE) ----------------
// 16-lane groups quantize one 16-wide sub-block in place: producer kernels
// (mix_streams, rmsnorm_gated, gate_mul, silu_mul640, silu_mul_combo) emit
// the NVFP4 3-level cascade alongside their f32 output - one launch and one
// HBM round trip less per projection. Bit-identical to quant_x_fp4 (same
// amax -> ue4m3 ceiling -> RNE nibble -> residual math per element).
// Caller guarantees whole warps active (all element counts are multiples
// of 32 and block starts are multiples of 32).
__device__ __forceinline__ void quant16_store(float x, int j, unsigned char* sc_base,
                                              unsigned char* d_base, size_t lv_stride) {
    #pragma unroll
    for (int lv = 0; lv < 3; lv++) {
        float amax = fabsf(x);
        amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, 8));
        amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, 4));
        amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, 2));
        amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, 1));
        unsigned char sc = enc_ue4m3_up(amax * (1.0f / 6.0f));
        float scf = ue4m3(sc);
        float inv = 1.0f / scf;
        unsigned int nib = q_e2m1(x * inv);
        float r = x - e2m1(nib) * scf;
        unsigned int other = __shfl_xor_sync(0xffffffffu, nib, 1);
        if ((j & 1) == 0) d_base[lv * lv_stride + (j >> 1)] = (unsigned char)(nib | (other << 4));
        if (j == 0) sc_base[lv * lv_stride] = sc;
        x = r;
    }
}
// element e of a k_dim row -> quantize+store into that row (level stride bpr*36)
__device__ __forceinline__ void qf_store(float v, unsigned char* xq_row, int bpr, int e) {
    int b = e >> 6, sb = (e & 63) >> 4, jj = e & 15;
    quant16_store(v, jj, xq_row + b * 36 + sb, xq_row + b * 36 + 4 + sb * 8, (size_t)bpr * 36);
}

extern "C" __global__ void mix_streams_q(const float* __restrict__ mixw, const float* __restrict__ normed,
                                         float* __restrict__ out, unsigned char* __restrict__ xq) {
    int c = blockIdx.x * 256 + threadIdx.x;
    int t = blockIdx.y;
    float acc = 0.0f;
    for (int g = 0; g < 4; g++)
        acc += mixw[t * 10240 + g * 2560 + c] * normed[t * 10240 + g * 2560 + c];
    float v = acc * 0.25f;
    out[t * 2560 + c] = v;
    const int bpr = 40;
    unsigned char* row = xq + (size_t)t * bpr * 108;
    qf_store(v, row, bpr, c);
}
extern "C" __global__ void rmsnorm_gated_q(const float* __restrict__ x, const float* __restrict__ z,
                                           const float* __restrict__ w, float* __restrict__ out,
                                           unsigned char* __restrict__ xq) {
    int vhead = blockIdx.x;
    int t = blockIdx.y;
    int d = threadIdx.x;
    const float* xp = x + ((size_t)t * 48 + vhead) * 128;
    __shared__ float red[128];
    red[d] = xp[d] * xp[d];
    __syncthreads();
    for (int st = 64; st > 0; st >>= 1) {
        if (d < st) red[d] += red[d + st];
        __syncthreads();
    }
    float rms = rsqrtf(red[0] / 128.0f + 1e-6f);
    float gate = 1.0f / (1.0f + expf(-z[((size_t)t * 48 + vhead) * 128 + d]));
    float v = w[d] * xp[d] * rms * gate;
    out[((size_t)t * 48 + vhead) * 128 + d] = v;
    const int bpr = 96;
    int e = vhead * 128 + d;
    unsigned char* row = xq + (size_t)t * bpr * 108;
    qf_store(v, row, bpr, e);
}
extern "C" __global__ void gate_mul_q(const float* __restrict__ core, const float* __restrict__ gate,
                                      float* __restrict__ out, unsigned char* __restrict__ xq) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    float g = gate[i];
    float v = core[i] / (1.0f + expf(-g));
    out[i] = v;
    const int bpr = 96;
    int t = i / 6144, e = i % 6144;
    unsigned char* row = xq + (size_t)t * bpr * 108;
    qf_store(v, row, bpr, e);
}
extern "C" __global__ void silu_mul640_q(const float* __restrict__ h1, float* __restrict__ h2,
                                         unsigned char* __restrict__ xq) {
    int j = blockIdx.x * 256 + threadIdx.x;
    int t = blockIdx.y;
    if (j >= 640) return; // whole warps (640 = 20 warps)
    float gate = h1[t * 1280 + j];
    float v = (gate / (1.0f + expf(-gate))) * h1[t * 1280 + 640 + j];
    h2[t * 640 + j] = v;
    const int bpr = 10;
    unsigned char* row = xq + (size_t)t * bpr * 108;
    qf_store(v, row, bpr, j);
}
// ---------------- 19h (CROW_QFUSE=1): fused shared-expert decode launches ----------------
// ONE launch replaces the TWO shared-expert gate|up gemv_fp4_mma_d launches AND
// the silu_mul640_q launch (decode, t < 8): grid (10, T) = (INTER/64, T), block
// mma_bx(), i.e. the exact gemv_fp4_mma_d shape the separate launches use.
//  - pass 0 / pass 1: the gemv_fp4_mma_d body VERBATIM over the gate slab (sg.w)
//    and the up slab (su.w) against the SAME xq_gu row: same lane split, same
//    k-block walk per ks slice, same red[4][2][64] ks-split reduce, so every mma
//    and FP32 add is one the separate launches makes (they differ only in the
//    weight pointer, gs and the store address). The two finished f32 accumulator
//    values (s0 * gs) are exchanged through smem, which is exact.
//  - the silu_mul640_q math folds in warp-wide on the FINISHED pairs:
//    v = (gate / (1 + expf(-gate))) * up, written to sh2[t][j] and quantized
//    with qf_store (bpr = 10) exactly like silu_mul640_q. The epilogue keeps
//    j = tile + rg*32 + lane (ks == 0, rg in {0,1}: exactly 2 warps x 32 lanes
//    = the 64-row tile), so within a warp the low 4 lane bits equal j & 15 and
//    quant16_store's shuffle groups (partners j^1/2/4/8 inside the same
//    16-wide sub-block) see exactly the layout the standalone launch has.
//    The sh12 write is dead in the fused chain and is skipped (the 19f
//    dead-write pattern).
extern "C" __global__ void sh_gate_up_q(const unsigned char* __restrict__ wg,
                                        const unsigned char* __restrict__ wu,
                                        const unsigned char* __restrict__ xq,
                                        const float* __restrict__ gsg_ptr,
                                        const float* __restrict__ gsu_ptr,
                                        float* __restrict__ h2,
                                        unsigned char* __restrict__ xq_s,
                                        const int* __restrict__ k_dim_p,
                                        const int* __restrict__ rows_p) {
    int k_dim = *k_dim_p;
    int bpr = k_dim >> 6;
    int rows = *rows_p;
    int tok = blockIdx.y;
    int tile = blockIdx.x << 6;
    int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    int ks_n = blockDim.x >> 7;
    int rg = warp & 3, ks = warp >> 2;
    int g = lane >> 2, lt = lane & 3;
    int w0 = tile + (rg << 4);
    bool active = w0 < rows;   // whole-warp guard; no early return (smem barrier below)
    int r0 = w0 + g, r1 = w0 + g + 8;
    __shared__ float red[4][2][64];
    __shared__ float sg_sh[64], su_sh[64];
    float sg0 = 0.0f, sg2 = 0.0f, su0 = 0.0f, su2 = 0.0f;
    for (int pass = 0; pass < 2; pass++) {
        const unsigned char* w = (pass == 0) ? wg : wu;
        float d0 = 0.0f, d1 = 0.0f, d2 = 0.0f, d3 = 0.0f;
        if (active) {
            const unsigned char* xa = xq + (size_t)tok * bpr * 108;
            const unsigned char* rowg = w + (size_t)min(r0, rows - 1) * bpr * 36;
            const unsigned char* rowg8 = w + (size_t)min(r1, rows - 1) * bpr * 36;
            const unsigned char* sfrow = (lt & 1) ? rowg8 : rowg;
            int bpb = (bpr + ks_n - 1) / ks_n;
            int b_lo = ks * bpb, b_hi = min(bpr, b_lo + bpb);
            for (int b = b_lo; b < b_hi; b++) {
                const unsigned char* blk = rowg + b * 36;
                const unsigned char* blk8 = rowg8 + b * 36;
                const unsigned char* ab = xa + b * 36;
                unsigned int sa = sfrow[b * 36 + 0] | (unsigned int)sfrow[b * 36 + 1] << 8
                                | (unsigned int)sfrow[b * 36 + 2] << 16 | (unsigned int)sfrow[b * 36 + 3] << 24;
                unsigned int a0 = *(const unsigned int*)(blk + 4 + 4 * lt);
                unsigned int a1 = *(const unsigned int*)(blk8 + 4 + 4 * lt);
                unsigned int a2 = *(const unsigned int*)(blk + 20 + 4 * lt);
                unsigned int a3 = *(const unsigned int*)(blk8 + 20 + 4 * lt);
                unsigned int sb = ab[0] | (unsigned int)ab[1] << 8
                                | (unsigned int)ab[2] << 16 | (unsigned int)ab[3] << 24;
                unsigned int b0 = *(const unsigned int*)(ab + 4 + 4 * lt);
                unsigned int b1 = *(const unsigned int*)(ab + 20 + 4 * lt);
                mma_fp4_16n8k64(d0, d1, d2, d3, a0, a1, a2, a3, b0, b1, sa, sb);
                #pragma unroll
                for (int lv = 1; lv < 3; lv++) {
                    const unsigned char* ab2 = ab + lv * bpr * 36; // residual level
                    unsigned int sb2 = ab2[0] | (unsigned int)ab2[1] << 8
                                     | (unsigned int)ab2[2] << 16 | (unsigned int)ab2[3] << 24;
                    unsigned int c0 = *(const unsigned int*)(ab2 + 4 + 4 * lt);
                    unsigned int c1 = *(const unsigned int*)(ab2 + 20 + 4 * lt);
                    mma_fp4_16n8k64(d0, d1, d2, d3, a0, a1, a2, a3, c0, c1, sa, sb2);
                }
            }
        }
        if (lt == 0) { red[ks][0][(rg << 4) + g] = d0; red[ks][1][(rg << 4) + g] = d2; }
        __syncthreads();
        if (ks == 0 && lt == 0 && active) {
            float s0 = 0.0f, s2 = 0.0f;
            for (int i = 0; i < ks_n; i++) { s0 += red[i][0][(rg << 4) + g]; s2 += red[i][1][(rg << 4) + g]; }
            float gs = ((pass == 0) ? gsg_ptr : gsu_ptr)[0];
            if (pass == 0) { sg0 = s0 * gs; sg2 = s2 * gs; }
            else           { su0 = s0 * gs; su2 = s2 * gs; }
        }
        __syncthreads(); // the red slots are rewritten by the next pass
    }
    if (ks == 0 && lt == 0 && active) {
        sg_sh[r0 - tile] = sg0; sg_sh[r1 - tile] = sg2;
        su_sh[r0 - tile] = su0; su_sh[r1 - tile] = su2;
    }
    __syncthreads();
    if (ks == 0 && rg < 2) {
        int j = tile + (rg << 5) + lane;
        if (j < rows) {
            float gate = sg_sh[j - tile];
            float v = (gate / (1.0f + expf(-gate))) * su_sh[j - tile];
            h2[(size_t)tok * 640 + j] = v;
            const int bprq = 10;
            unsigned char* row = xq_s + (size_t)tok * bprq * 108;
            qf_store(v, row, bprq, j);
        }
    }
}

// 19h: gemv_fp4_mma_d with the gate_shared epilogue folded in (CROW_QFUSE=1
// shared-expert chain, the down projection): gate_shared reads the finished
// down accumulator back and ASSIGNS (1 / (1 + expf(-sgv[t]))) * s[t][c] into
// moe_out as its FIRST WRITER (no memset, graph-capturable), so folding it
// into the store is bit-identical: the stored f32 (s0 * gs) is exactly what
// gate_shared re-reads, and the expression order is kept. y = moe_out; the
// s.sdown write is dead in the fused chain and is skipped.
extern "C" __global__ void gemv_fp4_mma_dg(const unsigned char* __restrict__ w,
                                           const unsigned char* __restrict__ xq,
                                           const float* __restrict__ gs_ptr,
                                           const float* __restrict__ sgv,
                                           float* __restrict__ y,
                                           const int* __restrict__ k_dim_p,
                                           const int* __restrict__ rows_p,
                                           const int* __restrict__ y_stride_p) {
    int k_dim = *k_dim_p;
    int bpr = k_dim >> 6;
    int rows = *rows_p;
    int ys = *y_stride_p;
    int tok = blockIdx.y;
    int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    int ks_n = blockDim.x >> 7;
    int rg = warp & 3, ks = warp >> 2;
    int g = lane >> 2, lt = lane & 3;
    int w0 = (blockIdx.x << 6) + (rg << 4);
    bool active = w0 < rows;   // whole-warp guard; no early return (smem barrier below)
    float d0 = 0.0f, d1 = 0.0f, d2 = 0.0f, d3 = 0.0f;
    int r0 = w0 + g, r1 = w0 + g + 8;
    if (active) {
        const unsigned char* xa = xq + (size_t)tok * bpr * 108;
        const unsigned char* rowg = w + (size_t)min(r0, rows - 1) * bpr * 36;
        const unsigned char* rowg8 = w + (size_t)min(r1, rows - 1) * bpr * 36;
        const unsigned char* sfrow = (lt & 1) ? rowg8 : rowg;
        int bpb = (bpr + ks_n - 1) / ks_n;
        int b_lo = ks * bpb, b_hi = min(bpr, b_lo + bpb);
        for (int b = b_lo; b < b_hi; b++) {
            const unsigned char* blk = rowg + b * 36;
            const unsigned char* blk8 = rowg8 + b * 36;
            const unsigned char* ab = xa + b * 36;
            unsigned int sa = sfrow[b * 36 + 0] | (unsigned int)sfrow[b * 36 + 1] << 8
                            | (unsigned int)sfrow[b * 36 + 2] << 16 | (unsigned int)sfrow[b * 36 + 3] << 24;
            unsigned int a0 = *(const unsigned int*)(blk + 4 + 4 * lt);
            unsigned int a1 = *(const unsigned int*)(blk8 + 4 + 4 * lt);
            unsigned int a2 = *(const unsigned int*)(blk + 20 + 4 * lt);
            unsigned int a3 = *(const unsigned int*)(blk8 + 20 + 4 * lt);
            unsigned int sb = ab[0] | (unsigned int)ab[1] << 8
                            | (unsigned int)ab[2] << 16 | (unsigned int)ab[3] << 24;
            unsigned int b0 = *(const unsigned int*)(ab + 4 + 4 * lt);
            unsigned int b1 = *(const unsigned int*)(ab + 20 + 4 * lt);
            mma_fp4_16n8k64(d0, d1, d2, d3, a0, a1, a2, a3, b0, b1, sa, sb);
            #pragma unroll
            for (int lv = 1; lv < 3; lv++) {
                const unsigned char* ab2 = ab + lv * bpr * 36; // residual level
                unsigned int sb2 = ab2[0] | (unsigned int)ab2[1] << 8
                                 | (unsigned int)ab2[2] << 16 | (unsigned int)ab2[3] << 24;
                unsigned int c0 = *(const unsigned int*)(ab2 + 4 + 4 * lt);
                unsigned int c1 = *(const unsigned int*)(ab2 + 20 + 4 * lt);
                mma_fp4_16n8k64(d0, d1, d2, d3, a0, a1, a2, a3, c0, c1, sa, sb2);
            }
        }
    }
    __shared__ float red[4][2][64];
    if (lt == 0) { red[ks][0][(rg << 4) + g] = d0; red[ks][1][(rg << 4) + g] = d2; }
    __syncthreads();
    float gs = gs_ptr[0];
    if (ks == 0 && lt == 0 && active) {
        float s0 = 0.0f, s2 = 0.0f;
        for (int i = 0; i < ks_n; i++) { s0 += red[i][0][(rg << 4) + g]; s2 += red[i][1][(rg << 4) + g]; }
        float sig = 1.0f / (1.0f + expf(-sgv[tok]));   // gate_shared epilogue, ASSIGN
        if (r0 < rows) y[(size_t)tok * ys + r0] = sig * (s0 * gs);
        if (r1 < rows) y[(size_t)tok * ys + r1] = sig * (s2 * gs);
    }
}

extern "C" __global__ void silu_mul_combo_q(const float* __restrict__ h1, float* __restrict__ h2,
                                            const int* __restrict__ n640_p, unsigned char* __restrict__ xq) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= *n640_p) return; // n640 is a multiple of 640 -> whole warps
    int c = i / 640;
    int j = i % 640;
    float gate = h1[(size_t)c * 1280 + j];
    float v = (gate / (1.0f + expf(-gate))) * h1[(size_t)c * 1280 + 640 + j];
    h2[i] = v;
    const int bpr = 10;
    unsigned char* row = xq + (size_t)c * bpr * 108;
    qf_store(v, row, bpr, j);
}

// ---------------- decode attention that does not scale with the context ----------------
// qsa_scores_par: warp per pooled block, fixed grid (graph-static), strided
// over ncb; 4 heads x 128 dims via 4 elements per lane + shuffle reduce.
extern "C" __global__ void qsa_scores_par(const float* __restrict__ q, const float* __restrict__ pooled,
                                          float* __restrict__ scores, const int* __restrict__ cap_p,
                                          const int* __restrict__ pos_base_p) {
    int tq = blockIdx.y;
    int pos = *pos_base_p + tq;
    int ncb = (pos + 1) >> 2;
    if (ncb > *cap_p) ncb = *cap_p;
    int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    float q0[4], q1[4], q2[4], q3[4];
    const float* qb = q + (size_t)tq * 4 * 128;
    #pragma unroll
    for (int i = 0; i < 4; i++) {
        q0[i] = qb[lane * 4 + i]; q1[i] = qb[128 + lane * 4 + i];
        q2[i] = qb[256 + lane * 4 + i]; q3[i] = qb[384 + lane * 4 + i];
    }
    for (int b = blockIdx.x * 4 + warp; b < ncb; b += gridDim.x * 4) {
        const float4 pk = *(const float4*)(pooled + (size_t)b * 128 + lane * 4);
        float s0 = q0[0] * pk.x + q0[1] * pk.y + q0[2] * pk.z + q0[3] * pk.w;
        float s1 = q1[0] * pk.x + q1[1] * pk.y + q1[2] * pk.z + q1[3] * pk.w;
        float s2 = q2[0] * pk.x + q2[1] * pk.y + q2[2] * pk.z + q2[3] * pk.w;
        float s3 = q3[0] * pk.x + q3[1] * pk.y + q3[2] * pk.z + q3[3] * pk.w;
        #pragma unroll
        for (int o = 16; o > 0; o >>= 1) {
            s0 += __shfl_xor_sync(0xffffffffu, s0, o);
            s1 += __shfl_xor_sync(0xffffffffu, s1, o);
            s2 += __shfl_xor_sync(0xffffffffu, s2, o);
            s3 += __shfl_xor_sync(0xffffffffu, s3, o);
        }
        if (lane == 0)
            scores[(size_t)tq * *cap_p + b] = (fmaxf(s0, 0.0f) + fmaxf(s1, 0.0f) + fmaxf(s2, 0.0f) + fmaxf(s3, 0.0f)) * rsqrtf(128.0f);
    }
}
// attn_sel_split: the selected list (<= 2051 tokens) split over gridDim.z
// blocks per head; each writes an unnormalized partial (m, l, o[256]) in
// flash-decoding form, attn_merge combines. grid (24, T, S), block 256.
// LUT (#61f, 2026-09-18): 1 decodes the e4m3 KV bytes through a shared 256-entry
// table (`kv_ld<1>`, lut[b] = dec_e4m3(b) - the same float, one shared load
// instead of the branchy ldexpf/divide decode), exactly as attn_sel_s8l does
// against attn_sel_s8. Only the LOAD changes: every fma chain, the e order, the
// shuffle tree, the expf and the j order are those of LUT = 0, so the two
// instantiations are bit-identical by construction.
// #96: RT = the scale source (0 folded, 1 the YaRN runtime global), as attn_sel.
template <int LUT, int RT>
__device__ __forceinline__ void attn_sel_split_body(const float* __restrict__ q, const unsigned char* __restrict__ kc,
                                          const unsigned char* __restrict__ vc, const int* __restrict__ sel,
                                          const int* __restrict__ sel_n, const int* __restrict__ tmax_p,
                                          const int* __restrict__ mode_p, const int* __restrict__ sel_max_p,
                                          float* __restrict__ part_o, float* __restrict__ part_ml) {
    int head = blockIdx.x;
    int t = blockIdx.y;
    int split = blockIdx.z, S = gridDim.z;
    int d = threadIdx.x;
    int kvh = head / 12;
    int n = sel_n[t];
    if (n < 0) n = 0;
    if (n > *sel_max_p) n = *sel_max_p;
    int per = (n + S - 1) / S;
    int j_lo = split * per, j_hi = min(n, j_lo + per);
    int cnt = max(0, j_hi - j_lo);
    const int* list = sel + (size_t)t * *sel_max_p + j_lo;
    const float* qt = q + ((size_t)t * 24 + head) * 256;
    __shared__ float p[2051];
    __shared__ float red[256];
    __shared__ float lut[LUT ? 256 : 1];
    int mode = *mode_p;
    int warp = d >> 5, lane = d & 31;
    const float scale = attn_scale_src<RT>(); // 1/sqrt(256)
    if (LUT) {
        // block is 256 threads (AHD), so one entry per thread; the barrier is the
        // only instruction the LUT adds outside the loops
        lut[d] = dec_e4m3((unsigned char)d);
        __syncthreads();
    }
    for (int j0 = 0; j0 < cnt; j0 += 8) {
        int j = j0 + warp;
        if (j < cnt) {
            int tok = list[j];
            if (tok < 0) tok = 0;
            if (tok >= *tmax_p) tok = *tmax_p - 1;
            const unsigned char* kp = kc + (size_t)(kvh * *tmax_p + tok) * 256 * (mode ? 2 : 1);
            float acc = 0.0f;
            for (int e = lane; e < 256; e += 32) acc += qt[e] * kv_ld<LUT>(kp, e, mode, lut);
            for (int o = 16; o > 0; o >>= 1) acc += __shfl_down_sync(0xffffffffu, acc, o);
            if (lane == 0) p[j] = acc * scale;
        }
    }
    __syncthreads();
    float mx = -3.0e38f;
    for (int j = d; j < cnt; j += 256) mx = fmaxf(mx, p[j]);
    red[d] = mx;
    __syncthreads();
    for (int st = 128; st > 0; st >>= 1) {
        if (d < st) red[d] = fmaxf(red[d], red[d + st]);
        __syncthreads();
    }
    mx = red[0];
    __syncthreads(); // thread 0 writes red[0] = sum below: without this barrier a late reader takes that sum as mx
    float sum = 0.0f;
    for (int j = d; j < cnt; j += 256) { float e = expf(p[j] - mx); p[j] = e; sum += e; }
    red[d] = sum;
    __syncthreads();
    for (int st = 128; st > 0; st >>= 1) {
        if (d < st) red[d] += red[d + st];
        __syncthreads();
    }
    sum = red[0];
    float o = 0.0f;
    for (int j = 0; j < cnt; j++) {
        int tok = list[j];
        if (tok < 0) tok = 0;
        if (tok >= *tmax_p) tok = *tmax_p - 1;
        const unsigned char* vp = vc + (size_t)(kvh * *tmax_p + tok) * 256 * (mode ? 2 : 1);
        o += p[j] * kv_ld<LUT>(vp, d, mode, lut);
    }
    size_t pi = ((size_t)t * 24 + head) * S + split;
    part_o[pi * 256 + d] = o;
    if (d == 0) { part_ml[pi * 2] = (cnt > 0) ? mx : -3.0e38f; part_ml[pi * 2 + 1] = (cnt > 0) ? sum : 0.0f; }
}
extern "C" __global__ void attn_sel_split(const float* __restrict__ q, const unsigned char* __restrict__ kc,
                                          const unsigned char* __restrict__ vc, const int* __restrict__ sel,
                                          const int* __restrict__ sel_n, const int* __restrict__ tmax_p,
                                          const int* __restrict__ mode_p, const int* __restrict__ sel_max_p,
                                          float* __restrict__ part_o, float* __restrict__ part_ml) {
    attn_sel_split_body<0, 0>(q, kc, vc, sel, sel_n, tmax_p, mode_p, sel_max_p, part_o, part_ml);
}
extern "C" __global__ void attn_sel_split_y(const float* __restrict__ q, const unsigned char* __restrict__ kc,
                                            const unsigned char* __restrict__ vc, const int* __restrict__ sel,
                                            const int* __restrict__ sel_n, const int* __restrict__ tmax_p,
                                            const int* __restrict__ mode_p, const int* __restrict__ sel_max_p,
                                            float* __restrict__ part_o, float* __restrict__ part_ml) {
    attn_sel_split_body<0, 1>(q, kc, vc, sel, sel_n, tmax_p, mode_p, sel_max_p, part_o, part_ml); // #96 YaRN runtime scale
}
extern "C" __global__ void attn_sel_split_l(const float* __restrict__ q, const unsigned char* __restrict__ kc,
                                            const unsigned char* __restrict__ vc, const int* __restrict__ sel,
                                            const int* __restrict__ sel_n, const int* __restrict__ tmax_p,
                                            const int* __restrict__ mode_p, const int* __restrict__ sel_max_p,
                                            float* __restrict__ part_o, float* __restrict__ part_ml) {
    attn_sel_split_body<1, 0>(q, kc, vc, sel, sel_n, tmax_p, mode_p, sel_max_p, part_o, part_ml);
}
extern "C" __global__ void attn_sel_split_l_y(const float* __restrict__ q, const unsigned char* __restrict__ kc,
                                              const unsigned char* __restrict__ vc, const int* __restrict__ sel,
                                              const int* __restrict__ sel_n, const int* __restrict__ tmax_p,
                                              const int* __restrict__ mode_p, const int* __restrict__ sel_max_p,
                                              float* __restrict__ part_o, float* __restrict__ part_ml) {
    attn_sel_split_body<1, 1>(q, kc, vc, sel, sel_n, tmax_p, mode_p, sel_max_p, part_o, part_ml); // #96 YaRN runtime scale
}
extern "C" __global__ void attn_merge(const float* __restrict__ part_o, const float* __restrict__ part_ml,
                                      float* __restrict__ out, const int* __restrict__ s_p) {
    int head = blockIdx.x;
    int t = blockIdx.y;
    int d = threadIdx.x;
    int S = *s_p;
    size_t base = ((size_t)t * 24 + head) * S;
    float m = -3.0e38f;
    for (int s = 0; s < S; s++) m = fmaxf(m, part_ml[(base + s) * 2]);
    float L = 0.0f, o = 0.0f;
    for (int s = 0; s < S; s++) {
        float w = expf(part_ml[(base + s) * 2] - m);
        L += w * part_ml[(base + s) * 2 + 1];
        o += w * part_o[(base + s) * 256 + d];
    }
    out[((size_t)t * 24 + head) * 256 + d] = o / L;
}

// ---------------- low-bit cold tier expanders (CROW_COLD_TIER, A-P2) ----------------
// Cold experts are pinned as compact records (4 scale bytes + 64 codes of
// `bits` bits, LSB-first); the staging copy expands each record into the
// 36-byte NVFP4 block layout (codes -> e2m1 nibbles via the codebook LUT).
// One thread per block: consecutive threads read consecutive records
// (coalesced) and write consecutive 36-byte blocks. PCIe bytes x0.56 (2-bit).
__device__ __forceinline__ void expand_record(const unsigned char* __restrict__ rec, unsigned char* __restrict__ dst,
                                              int bits, const unsigned char* __restrict__ lut) {
    dst[0] = rec[0]; dst[1] = rec[1]; dst[2] = rec[2]; dst[3] = rec[3];
    unsigned int mask = (1u << bits) - 1u;
    unsigned long long acc = 0; int nb = 0; int ip = 4;
    #pragma unroll 4
    for (int j = 0; j < 32; j++) {
        while (nb < 2 * bits) { acc |= (unsigned long long)rec[ip++] << nb; nb += 8; }
        unsigned int c0 = (unsigned int)(acc & mask); acc >>= bits;
        unsigned int c1 = (unsigned int)(acc & mask); acc >>= bits;
        nb -= 2 * bits;
        dst[4 + j] = (unsigned char)(lut[c0] | (lut[c1] << 4));
    }
}
// expand a whole expert (records -> NVFP4 blocks) into a HOT slab slot:
// prompt-adaptive residency (A-P3) promotes cold experts after the prefill
extern "C" __global__ void expand_slab(const unsigned char* __restrict__ rec, unsigned char* __restrict__ dst,
                                       const int* __restrict__ nblk_p, const int* __restrict__ bits_p,
                                       const unsigned char* __restrict__ lut) {
    int nblk = *nblk_p;
    int bits = *bits_p;
    int rb = 4 + 8 * bits;
    for (int b = blockIdx.x * blockDim.x + threadIdx.x; b < nblk; b += gridDim.x * blockDim.x)
        expand_record(rec + (size_t)b * rb, dst + (size_t)b * 36, bits, lut);
}
// decode staging with expansion: grid (combos, 2, SPLIT), block 256
extern "C" __global__ void stage_cold_lb(const unsigned long long* __restrict__ gu_ptrs,
                                         const unsigned long long* __restrict__ dn_ptrs,
                                         const unsigned int* __restrict__ cold_mask,
                                         unsigned char* __restrict__ stage_gu,
                                         unsigned char* __restrict__ stage_dn,
                                         unsigned long long* __restrict__ sgu_ptrs,
                                         unsigned long long* __restrict__ sdn_ptrs,
                                         const int* __restrict__ gu_bytes_p,
                                         const int* __restrict__ dn_bytes_p,
                                         const int* __restrict__ bits_p,
                                         const unsigned char* __restrict__ lut) {
    int combo = blockIdx.x;
    int which = blockIdx.y;
    int split = gridDim.z;
    int part = blockIdx.z;
    int t = combo / 10, j = combo % 10;
    bool cold = (cold_mask[t] >> j) & 1u;
    size_t bytes = (size_t)(which ? *dn_bytes_p : *gu_bytes_p); // NVFP4 bytes of the slab
    const unsigned char* src = (const unsigned char*)(which ? dn_ptrs[combo] : gu_ptrs[combo]);
    unsigned char* dst = (which ? stage_dn : stage_gu) + (size_t)combo * bytes;
    if (part == 0 && threadIdx.x == 0) {
        unsigned long long p = cold ? (unsigned long long)dst : (unsigned long long)src;
        if (which) sdn_ptrs[combo] = p; else sgu_ptrs[combo] = p;
    }
    if (!cold) return;
    int bits = *bits_p;
    int rec = 4 + 8 * bits;
    size_t nblk = bytes / 36;
    size_t per = (nblk + split - 1) / split;
    size_t b_lo = part * per, b_hi = min(nblk, b_lo + per);
    for (size_t b = b_lo + threadIdx.x; b < b_hi; b += blockDim.x)
        expand_record(src + b * rec, dst + b * 36, bits, lut);
}
// prefill tile staging with expansion: grid (TG, 2, SPLIT), block 256
extern "C" __global__ void stage_tiles_lb(const int4* __restrict__ tiles, const int* __restrict__ n_tiles_p,
                                          const int* __restrict__ group_p, const int* __restrict__ tg_p,
                                          const unsigned long long* __restrict__ table,
                                          const unsigned int* __restrict__ bitmap,
                                          unsigned char* __restrict__ stage_gu, unsigned char* __restrict__ stage_dn,
                                          unsigned long long* __restrict__ eptr,
                                          const int* __restrict__ gu_bytes_p, const int* __restrict__ dn_bytes_p,
                                          const int* __restrict__ bits_p, const unsigned char* __restrict__ lut,
                                          const unsigned char* __restrict__ pin_gu, const unsigned char* __restrict__ pin_dn,
                                          const unsigned char* __restrict__ ring_gu, const unsigned char* __restrict__ ring_dn) {
    int ti = (*group_p) * (*tg_p) + blockIdx.x;
    if (ti >= *n_tiles_p) return;
    int4 tl = tiles[ti];
    int e = tl.x;
    int which = blockIdx.y;
    int slot = (tl.w & 0xFFFF) + ((tg_p[1] > 1) ? ((*group_p) & 1) * (*tg_p) : 0); // CROW_PF_ASYNC: slot set per group parity
    bool first = (tl.w >> 16) & 1;
    bool cold = !((bitmap[e >> 5] >> (e & 31)) & 1u);
    size_t bytes = (size_t)(which ? *dn_bytes_p : *gu_bytes_p);
    const unsigned char* src = (const unsigned char*)table[e * 2 + which];
    unsigned char* dst = (which ? stage_dn : stage_gu) + (size_t)slot * bytes;
    if (blockIdx.z == 0 && threadIdx.x == 0) {
        eptr[ti * 2 + which] = cold ? (unsigned long long)dst : (unsigned long long)src;
    }
    if (!cold || !first) return;
    if (tg_p[2] >= 2) return; // CROW_PF_ASYNC=2: the copy engine stages (host memcpys ahead on the side stream); =3 diagnostic, no copy
    const unsigned char* ring = which ? ring_dn : ring_gu;
    if (ring) src = ring + (src - (which ? pin_dn : pin_gu));
    int bits = *bits_p;
    int rec = 4 + 8 * bits;
    size_t nblk = bytes / 36;
    size_t per = (nblk + gridDim.z - 1) / gridDim.z;
    size_t b_lo = blockIdx.z * per, b_hi = min(nblk, b_lo + per);
    for (size_t b = b_lo + threadIdx.x; b < b_hi; b += blockDim.x)
        expand_record(src + b * rec, dst + b * 36, bits, lut);
}

// ---------------- PLE (p15-verified + NVFP4 row cache + conv state) ----------------
extern "C" __global__ void gather_ple_fp4(const unsigned char* __restrict__ rows,
                                          const float* __restrict__ gs, const int* __restrict__ slot,
                                          float* __restrict__ out) {
    int h = blockIdx.x;
    int t = blockIdx.y;
    int d = threadIdx.x;
    int s = slot[t * 16 + h];
    int i = d; // 0..159
    int b = i >> 6;
    int idx = i & 63;
    const unsigned char* blk = rows + (size_t)s * 108 + b * 36;
    float val = e2m1((idx & 1) ? (((blk[4 + (idx >> 1)] >> 4) & 0xF)) : (blk[4 + (idx >> 1)] & 0xF))
              * ue4m3(blk[idx >> 4]) * gs[s];
    out[(size_t)t * 2560 + h * 160 + d] = val;
}
extern "C" __global__ void gate_dot(const float* __restrict__ key, const float* __restrict__ query,
                                    float* __restrict__ gate) {
    int s = blockIdx.x;
    int t = blockIdx.y;
    int d = threadIdx.x;
    const float* kp = key + ((size_t)t * 4 + s) * 2560;
    const float* qp = query + ((size_t)t * 4 + s) * 2560;
    float acc = 0.0f;
    for (int i = d; i < 2560; i += 256) acc += kp[i] * qp[i];
    __shared__ float red[256];
    red[d] = acc;
    __syncthreads();
    for (int st = 128; st > 0; st >>= 1) {
        if (d < st) red[d] += red[d + st];
        __syncthreads();
    }
    if (d == 0) gate[t * 4 + s] = red[0] * rsqrtf(2560.0f);
}
extern "C" __global__ void gate_apply(const float* __restrict__ gate, const float* __restrict__ value,
                                      float* __restrict__ gate_signed, float* __restrict__ gated) {
    int s = blockIdx.x;
    int t = blockIdx.y;
    int d = threadIdx.x;
    if (d == 0) {
        float g = gate[t * 4 + s];
        gate_signed[t * 4 + s] = sqrtf(fmaxf(fabsf(g), 1e-6f)) * ((g > 0.0f) - (g < 0.0f));
    }
    __syncthreads();
    float sg = 1.0f / (1.0f + expf(-gate_signed[t * 4 + s]));
    for (int i = d; i < 2560; i += 256)
        gated[((size_t)t * 4 + s) * 2560 + i] = sg * value[(size_t)t * 2560 + i];
}
// dilated conv (k=4, dil=3, left state 9) with cross-chunk state + silu + add
extern "C" __global__ void ple_conv(const float* __restrict__ gn, const float* __restrict__ w,
                                    const float* __restrict__ gated, float* __restrict__ out,
                                    const int* __restrict__ t_p, const float* __restrict__ state) {
    int tt = *t_p;
    int c = blockIdx.x;
    for (int t = threadIdx.x; t < tt; t += blockDim.x) {
        float acc = 0.0f;
        for (int k = 0; k < 4; k++) {
            int src = t + k * 3 - 9;
            float v;
            if (src >= 0) v = gn[(size_t)src * 10240 + c];
            else v = state[(size_t)c * 9 + (9 + src)];
            acc += w[c * 4 + k] * v;
        }
        out[(size_t)t * 10240 + c] =
            gated[(size_t)t * 10240 + c] + acc / (1.0f + expf(-acc));
    }
}
extern "C" __global__ void ple_state_update(const float* __restrict__ gn, float* __restrict__ state,
                                            const int* __restrict__ t_p) {
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= 10240) return;
    int tt = *t_p;
    for (int j = 0; j < 9; j++) {
        int src = tt - 9 + j;
        // src < 0: a chunk shorter than the 9-row window (a prompt tail, a warm
        // resume with a short suffix): the old row j + tt shifts down to j, what
        // ple_conv_step does after tt single steps. Ascending j reads slot j + tt
        // before it is overwritten.
        state[c * 9 + j] = (src >= 0) ? gn[(size_t)src * 10240 + c] : state[c * 9 + j + tt];
    }
}
extern "C" __global__ void ple_conv_step(const float* __restrict__ gn_row,
                                         const float* __restrict__ gated_row,
                                         const float* __restrict__ w, float* __restrict__ state,
                                         float* __restrict__ out_row) {
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= 10240) return;
    float acc = w[c * 4 + 0] * state[c * 9 + 0] + w[c * 4 + 1] * state[c * 9 + 3]
              + w[c * 4 + 2] * state[c * 9 + 6] + w[c * 4 + 3] * gn_row[c];
    out_row[c] = gated_row[c] + acc / (1.0f + expf(-acc));
    for (int j = 0; j < 8; j++) state[c * 9 + j] = state[c * 9 + j + 1];
    state[c * 9 + 8] = gn_row[c];
}

// ---------------- #17: bundled hot-set swaps ----------------
// Exchange the bytes of n pairs (hot VRAM slab <-> pinned cold slab, UVA) in
// ONE launch: grid (pairs, SPLIT), block 256, 16-byte units, no bounce slot.
// Replaces 6 stream memcpys per swap (2026-09-05: 8 swaps x 48 layers x 6 =
// 2304 copies per adaptation tick, mean 29.2 ms vs p50 22.8 ms at 16k).
extern "C" __global__ void swap_pairs(unsigned long long* __restrict__ pa,
                                      unsigned long long* __restrict__ pb,
                                      const int* __restrict__ nbytes_p) {
    int pair = blockIdx.x;
    int split = gridDim.y;
    int part = blockIdx.y;
    size_t n16 = ((size_t)*nbytes_p) >> 4;
    size_t per = (n16 + split - 1) / split;
    size_t lo = (size_t)part * per;
    size_t hi = lo + per; if (hi > n16) hi = n16;
    uint4* a = (uint4*)pa[pair];
    uint4* b = (uint4*)pb[pair];
    for (size_t i = lo + threadIdx.x; i < hi; i += blockDim.x) {
        uint4 x = a[i];
        uint4 y = b[i];
        a[i] = y;
        b[i] = x;
    }
}

// ---------------- head: lm_head argmax ----------------
extern "C" __global__ void argmax_k(const float* __restrict__ logits, int* __restrict__ out,
                                    const int* __restrict__ n_p) {
    int n = *n_p;
    float best = -3.0e38f;
    int bi = 0;
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        float v = logits[i];
        if (v > best) { best = v; bi = i; }
    }
    __shared__ float rb[1024];
    __shared__ int ri[1024];
    rb[threadIdx.x] = best;
    ri[threadIdx.x] = bi;
    __syncthreads();
    for (int st = 512; st > 0; st >>= 1) {
        if (threadIdx.x < st && rb[threadIdx.x + st] > rb[threadIdx.x]) {
            rb[threadIdx.x] = rb[threadIdx.x + st];
            ri[threadIdx.x] = ri[threadIdx.x + st];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) out[0] = ri[0];
}
// ---------------- head: device sampler (#20) ----------------
// Bit-for-bit the host sampler of sample.rs: presence penalty on the raw
// logits, top-k in (value desc, index asc) order, f32 (v-m)/temp cast to
// double, exp / softmax / nucleus / draw in double, xorshift64* state in
// `rng`. `params` (36 B) = {temp, top_p, presence, min_p, ln_min_p,
// repeat, freq} f32 + {top_k, last_n} i32. `mask[v]` = 1 once v was
// sampled in this answer. `min_p > 0` (#83, llama.cpp PR #3841) drops
// candidates below cv[0] + ln_min_p AFTER top-k and BEFORE the softmax -
// the SAME host-computed ln constant, so the boundary is bit-equal.
// #84: `last_n > 0 && (repeat != 1.0f || freq > 0.0f)` arms the windowed
// llama.cpp penalties on the RAW logits: `l > 0 ? l/repeat : l*repeat`
// (asymmetric - dividing a negative logit would raise it), then
// `l -= c*freq + (c>0)*presence` with c = counts[i]; `counts[V]` is u16
// and `ring` is i32 {head, fill, ids[last_n]} in global memory, seeded
// from the PROMPT TAIL per request and advanced by one accept per draw
// (the evicted tail id decrements exactly, llama.cpp ring semantics).
// While the window is NOT armed the HF presence mask (#68) applies alone,
// byte-identical to the pre-#84 sampler.
// Writes the token into out[0] (the argmax slot), so the readback behind the
// graph is unchanged.
// Two stages (v2): sample_topk_part - SAMPLE_PARTS blocks each pick the k
// largest keys of their slice into cand; sample_k - one block runs the same
// key-ordered rounds over the SAMPLE_PARTS*k candidates and draws. The union
// of the slice top-ks contains the global top-k and the order is one total
// order, so the draw equals the single-block v1 (1.8 ms/token on one SM).
#define SAMPLE_MAXK 64
#define SAMPLE_PARTS 64
#define SAMPLE_THREADS 256
#define SAMPLE_RING_MAX 1024

__device__ __forceinline__ void sample_rounds(const float* __restrict__ vals, const int* __restrict__ idx, int n,
                                              const unsigned char* __restrict__ mask, float pres, int k,
                                              float* __restrict__ rb, int* __restrict__ ri,
                                              float* __restrict__ cv, int* __restrict__ ci, int nthreads) {
    // k rounds of "largest key below the previous pick" over (vals, idx); when
    // idx == 0 the element index is its position and the mask penalty applies
    __shared__ float pv;
    __shared__ int pi;
    const int tid = threadIdx.x;
    if (tid == 0) { pv = __int_as_float(0x7f800000); pi = -1; }
    __syncthreads();
    for (int r = 0; r < k; r++) {
        const float lim_v = pv;
        const int lim_i = pi;
        float best = -__int_as_float(0x7f800000);
        int bi = 0x7fffffff;
        for (int j = tid; j < n; j += nthreads) {
            float v = vals[j];
            int i;
            if (idx == 0) { i = j; if (mask[i]) v -= pres; } else { i = idx[j]; }
            if (v < lim_v || (v == lim_v && i > lim_i)) {
                if (v > best || (v == best && i < bi)) { best = v; bi = i; }
            }
        }
        rb[tid] = best;
        ri[tid] = bi;
        __syncthreads();
        for (int st = nthreads >> 1; st > 0; st >>= 1) {
            if (tid < st) {
                const float ov = rb[tid + st];
                const int oi = ri[tid + st];
                if (ov > rb[tid] || (ov == rb[tid] && oi < ri[tid])) { rb[tid] = ov; ri[tid] = oi; }
            }
            __syncthreads();
        }
        if (tid == 0) { cv[r] = rb[0]; ci[r] = ri[0]; pv = rb[0]; pi = ri[0]; }
        __syncthreads();
    }
}

// stage 1: block b owns logits [b*slice, min(n, (b+1)*slice)); writes k keys
// (value with the #68/#84 penalties applied, index) into cand_v/cand_i[b*k ..]
extern "C" __global__ void __launch_bounds__(SAMPLE_THREADS) sample_topk_part(
        const float* __restrict__ logits, const int* __restrict__ n_p, const unsigned char* __restrict__ mask,
        const unsigned short* __restrict__ counts, const float* __restrict__ params,
        float* __restrict__ cand_v, int* __restrict__ cand_i) {
    const int n = *n_p;
    const float pres = params[2];
    // #84: the windowed penalties arm on the new knobs; presence joins them
    // as llama.cpp's penalty_present while armed (defaults stay #68's HF form)
    const float rep = params[6], fq = params[7];
    int lastn = ((const int*)params)[8];
    if (lastn < 0) lastn = 0;
    if (lastn > SAMPLE_RING_MAX) lastn = SAMPLE_RING_MAX;
    const int win = lastn > 0 && (rep != 1.0f || fq > 0.0f);
    int k = ((const int*)params)[3];
    if (k < 1) k = 1;
    if (k > SAMPLE_MAXK) k = SAMPLE_MAXK;
    const int slice = (n + SAMPLE_PARTS - 1) / SAMPLE_PARTS;
    const int lo = blockIdx.x * slice;
    const int hi = min(n, lo + slice);
    __shared__ float rb[SAMPLE_THREADS];
    __shared__ int ri[SAMPLE_THREADS];
    __shared__ float cv[SAMPLE_MAXK];
    __shared__ int ci[SAMPLE_MAXK];
    // rounds over the slice: the helper sees position j, we offset by lo
    __shared__ float pv;
    __shared__ int pi;
    const int tid = threadIdx.x;
    if (tid == 0) { pv = __int_as_float(0x7f800000); pi = -1; }
    __syncthreads();
    for (int r = 0; r < k; r++) {
        const float lim_v = pv;
        const int lim_i = pi;
        float best = -__int_as_float(0x7f800000);
        int bi = 0x7fffffff;
        for (int i = lo + tid; i < hi; i += SAMPLE_THREADS) {
            float v = logits[i];
            if (win) {
                // #84: llama.cpp penalties - the WHOLE per-candidate block
                // sits behind a count > 0 hit (token_count.find); the op
                // order is the host's win_pen, bit for bit
                const float c = (float)counts[i];
                if (c > 0.0f) {
                    if (rep != 1.0f) {
                        if (v > 0.0f) v /= rep; else v *= rep;
                    }
                    v -= c * fq + pres;
                }
            } else if (mask[i]) {
                v -= pres;
            }
            if (v < lim_v || (v == lim_v && i > lim_i)) {
                if (v > best || (v == best && i < bi)) { best = v; bi = i; }
            }
        }
        rb[tid] = best;
        ri[tid] = bi;
        __syncthreads();
        for (int st = SAMPLE_THREADS >> 1; st > 0; st >>= 1) {
            if (tid < st) {
                const float ov = rb[tid + st];
                const int oi = ri[tid + st];
                if (ov > rb[tid] || (ov == rb[tid] && oi < ri[tid])) { rb[tid] = ov; ri[tid] = oi; }
            }
            __syncthreads();
        }
        if (tid == 0) { cv[r] = rb[0]; ci[r] = ri[0]; pv = rb[0]; pi = ri[0]; }
        __syncthreads();
    }
    if (tid < k) {
        cand_v[blockIdx.x * k + tid] = cv[tid];
        cand_i[blockIdx.x * k + tid] = ci[tid];
    }
}

// stage 2: one block, the same rounds over the SAMPLE_PARTS*k candidates
// (values already penalized), then the draw
extern "C" __global__ void __launch_bounds__(SAMPLE_THREADS) sample_k(
        const float* __restrict__ cand_v, const int* __restrict__ cand_i, int* __restrict__ out,
        const int* __restrict__ n_p, unsigned char* __restrict__ mask,
        unsigned short* __restrict__ counts, int* __restrict__ ring,
        unsigned long long* __restrict__ rng, const float* __restrict__ params) {
    const int n = *n_p;
    const float temp = params[0], top_p = params[1];
    // #84: the same arming rule stage 1 applies (presence stays params[2])
    const float rep = params[6], fq = params[7];
    int lastn = ((const int*)params)[8];
    if (lastn < 0) lastn = 0;
    if (lastn > SAMPLE_RING_MAX) lastn = SAMPLE_RING_MAX;
    const int win = lastn > 0 && (rep != 1.0f || fq > 0.0f);
    int k = ((const int*)params)[3];
    if (k < 1) k = 1;
    if (k > n) k = n;
    if (k > SAMPLE_MAXK) k = SAMPLE_MAXK;
    const int nc = SAMPLE_PARTS * k;
    __shared__ float rb[SAMPLE_THREADS];
    __shared__ int ri[SAMPLE_THREADS];
    __shared__ float cv[SAMPLE_MAXK];
    __shared__ int ci[SAMPLE_MAXK];
    sample_rounds(cand_v, cand_i, nc, mask, 0.0f, k, rb, ri, cv, ci, SAMPLE_THREADS);
    if (threadIdx.x == 0) {
        int tok;
        if (temp <= 0.0f) {
            tok = ci[0];
        } else {
            // #83: min_p, the log-space tail filter (llama.cpp PR #3841) -
            // AFTER the rounds (top-k), BEFORE the softmax. cv is sorted
            // descending, so the survivors are a prefix; k2 starts at 1
            // because the top candidate is min_keep and survives even a
            // degenerate min_p > 1. ln_min_p (params[5]) is computed ONCE on
            // the host, so both samplers add the same constant to the max.
            int k2 = k;
            if (params[4] > 0.0f) {
                const float thr = cv[0] + params[5];
                k2 = 1;
                while (k2 < k && cv[k2] >= thr) k2++;
            }
            double pr[SAMPLE_MAXK];
            const float m = cv[0];
            double z = 0.0;
            for (int i = 0; i < k2; i++) {
                const float a = (cv[i] - m) / temp;
                pr[i] = exp((double)a);
                z += pr[i];
            }
            for (int i = 0; i < k2; i++) pr[i] /= z;
            int keep = k2;
            double acc = 0.0;
            for (int i = 0; i < k2; i++) {
                acc += pr[i];
                if (acc >= (double)top_p) { keep = i + 1; break; }
            }
            double z2 = 0.0;
            for (int i = 0; i < keep; i++) z2 += pr[i];
            unsigned long long x = rng[0];
            x ^= x >> 12;
            x ^= x << 25;
            x ^= x >> 27;
            rng[0] = x;
            const unsigned long long y = x * 0x2545F4914F6CDD1DULL;
            const double rnd = ((double)(y >> 11) / 9007199254740992.0) * z2;
            tok = ci[keep - 1];
            acc = 0.0;
            for (int i = 0; i < keep; i++) {
                acc += pr[i];
                if (rnd < acc) { tok = ci[i]; break; }
            }
        }
        out[0] = tok;
        mask[tok] = 1;
        // #84: accept the drawn token into the ring window (llama.cpp
        // penalties accept): once the window is full the oldest id leaves it
        // and its count drops by EXACTLY one. The update runs after the
        // draw, so this token's own count bites from the NEXT sample on -
        // the host's `observe` sits at the same place in its loop.
        if (win) {
            int head = ring[0], fill = ring[1];
            if (fill >= lastn) {
                counts[ring[2 + head]] -= 1;
            } else {
                fill += 1;
                ring[1] = fill;
            }
            ring[2 + head] = tok;
            counts[tok] += 1;
            head += 1;
            if (head >= lastn) head = 0;
            ring[0] = head;
        }
    }
}

// ===================== ViT (the #VIT visual tower kernels) =====================
// Reference: transformers 5.16.1 models/qwen4_exp/modeling_qwen4_exp.py
// (Qwen4ExpVisionModel). All math f32; the NVFP4 GEMV below keeps the exact
// gemv_fp4_b op order (16-wide sub-blocks ascending, part * scale, one acc).
//
// The vision MLP fc2 has k_dim 4304, which is a multiple of 16 but not of 64,
// so rows do not start on 36-byte block boundaries and the gemv_fp4_b row
// pointer does not apply. gemv_fp4_vit addresses at 16-value sub-block
// granularity (every vit k_dim is a multiple of 16): for row r sub-block b
// the global sub-block is G = r*(k_dim/16) + b, its 64-value block is G>>2,
// its scale byte sits at block*36 + (G&3), and its values at block*36 +
// 4 + (((G&3)*16 + j)>>1) (low nibble even idx, high odd — the layout of
// record per cnq::dequant_block / gemv_fp4_b). Per sub-block:
// part = sum_j e2m1 * x then acc += part * scale, sub-blocks ascending — the
// identical FP sequence gemv_fp4_b produces on a 64-multiple k_dim.
extern "C" __global__ void gemv_fp4_vit(const unsigned char* __restrict__ w, const float* __restrict__ x,
                                        const float* __restrict__ gs_ptr, float* __restrict__ y,
                                        const int* __restrict__ k_dim_p) {
    int k_dim = *k_dim_p;
    int bpr = k_dim >> 4;                     // 16-value sub-blocks per row
    int row = blockIdx.x;
    int t = blockIdx.y;
    const float* xp = x + (size_t)t * k_dim;
    float gs = gs_ptr[0];
    int g0 = row * bpr;                       // global sub-block of this row's start
    float acc = 0.0f;
    for (int b = threadIdx.x; b < bpr; b += blockDim.x) {
        int G = g0 + b;
        const unsigned char* blk = w + (size_t)(G >> 2) * 36;
        int sb = G & 3;
        float s = ue4m3(blk[sb]) * gs;
        float part = 0.0f;
        #pragma unroll
        for (int j = 0; j < 16; j++) {
            int idx = sb * 16 + j;
            unsigned int byte = blk[4 + (idx >> 1)];
            unsigned int nib = (idx & 1) ? ((byte >> 4) & 0xF) : (byte & 0xF);
            part += e2m1(nib) * xp[b * 16 + j];
        }
        acc += part * s;
    }
    __shared__ float red[256];
    red[threadIdx.x] = acc;
    __syncthreads();
    for (int st = blockDim.x / 2; st > 0; st >>= 1) {
        if (threadIdx.x < st) red[threadIdx.x] += red[threadIdx.x + st];
        __syncthreads();
    }
    if (threadIdx.x == 0) y[(size_t)t * gridDim.x + row] = red[0];
}

// LayerNorm (weight + bias, eps 1e-6, biased variance) over `cols` — the
// vision norm1/norm2 (1152) and the merger norm (1152). torch.nn.LayerNorm
// semantics: (x - mean) * rsqrt(var + eps) * w + b.
extern "C" __global__ void vit_ln(const float* __restrict__ x, const float* __restrict__ w,
                                  const float* __restrict__ b, float* __restrict__ out,
                                  const int* __restrict__ cols_p) {
    int cols = *cols_p;
    int row = blockIdx.x;
    int t = threadIdx.x;
    __shared__ float red[256];
    const float* xp = x + (size_t)row * cols;
    float s = 0.0f, ss = 0.0f;
    for (int d = t; d < cols; d += 256) {
        float v = xp[d];
        s += v;
        ss += v * v;
    }
    red[t] = s;
    __syncthreads();
    for (int st = 128; st > 0; st >>= 1) {
        if (t < st) red[t] += red[t + st];
        __syncthreads();
    }
    float mean = red[0] / cols;
    __syncthreads();
    red[t] = ss;
    __syncthreads();
    for (int st = 128; st > 0; st >>= 1) {
        if (t < st) red[t] += red[t + st];
        __syncthreads();
    }
    float var = red[0] / cols - mean * mean;
    float r = rsqrtf(var + 1e-6f);
    for (int d = t; d < cols; d += 256) {
        out[(size_t)row * cols + d] = (xp[d] - mean) * r * w[d] + b[d];
    }
}

// out[row * stride + d] += bias[d] — the every-linear bias of the vision tower.
extern "C" __global__ void vit_add_bias(float* __restrict__ out, const float* __restrict__ bias,
                                        const int* __restrict__ stride_p, const int* __restrict__ cols_p) {
    int cols = *cols_p;
    int row = blockIdx.x;
    int stride = *stride_p;
    float* op = out + (size_t)row * stride;
    for (int d = threadIdx.x; d < cols; d += 256) op[d] += bias[d];
}

// x[i][1152] += sum of 4 bilinear taps into the learned 48x48 position table
// (align_corners=True): idx[i*4+k] = h_tap*48 + w_tap, wts[i*4+k] the outer
// product weights — the host precomputes both in the oracle's tap order.
extern "C" __global__ void vit_pe_add(float* __restrict__ x, const float* __restrict__ pos,
                                      const int* __restrict__ idx, const float* __restrict__ wts,
                                      const int* __restrict__ n_p) {
    int i = blockIdx.x;
    if (i >= *n_p) return;
    int d = threadIdx.x;
    for (int dd = d; dd < 1152; dd += 256) {
        float acc = 0.0f;
        #pragma unroll
        for (int k = 0; k < 4; k++) {
            acc += pos[(size_t)idx[i * 4 + k] * 1152 + dd] * wts[i * 4 + k];
        }
        x[(size_t)i * 1152 + dd] += acc;
    }
}

// Vision rotary on the fused qkv buffer, in place: q at cols [0,1152), k at
// [1152,2304), head h dims [h*72, h*72+72), rotary pairs (d, d+36) for d<36
// with per-token cos/sin [t][36] (host-built from the (h,w) position ids).
extern "C" __global__ void vit_rope(float* __restrict__ qkv, const float* __restrict__ cos_,
                                    const float* __restrict__ sin_) {
    int t = blockIdx.y;
    int h = blockIdx.x;
    int d = threadIdx.x;
    if (d >= 36) return;
    float c = cos_[t * 36 + d];
    float s = sin_[t * 36 + d];
    float* qp = qkv + (size_t)t * 3456 + h * 72 + d;
    float* kp = qkv + (size_t)t * 3456 + 1152 + h * 72 + d;
    float a = qp[0], b = qp[36];
    qp[0] = a * c - b * s;
    qp[36] = b * c + a * s;
    a = kp[0];
    b = kp[36];
    kp[0] = a * c - b * s;
    kp[36] = b * c + a * s;
}

// Non-causal single-image attention, online softmax (flash row form), scale
// 1/sqrt(72), head dim 72, q/k/v read straight from the fused [n][3456] qkv
// buffer thirds. #98 step 2: a block owns a TILE of 16 query rows of one head
// (grid (ceil(n/16), 16 heads), block 256, 47 KiB static smem): K and V are
// staged in 64-key chunks and reused by all 16 rows, so global K/V traffic is
// 1/16 of the one-row-per-block kernel's, and all 256 threads work in both
// the score and the P.V phase.
// BIT-IDENTICAL to the pre-#98 row kernel by construction, and that is the
// contract (`vit::attn_98` checks every output bit against the old kernel,
// kept verbatim in the test): the key tile stays 256 wide; each score is the
// same sequential 72-term fma chain times SCALE; the tile max is fmaxf (exact,
// order-free); the tile sum is the SAME 256-leaf pairwise tree (k + 128, 64,
// 32 in smem, then 16 .. 1 as shfl_down, which pairs lane k with k + off
// exactly as red[k] += red[k + off] did); l = l*r + ln, acc *= r and the
// ascending-jj fma walk per (row, dim) are the old expressions in the old
// order; the final divide is the same. Nothing is reassociated, so no
// tolerance and no oracle re-run is needed. (Pre-#98 every one of 72 threads
// walked all 72 dims: n^2 * 72 * 72 FMAs per head and layer, ~20 s of tower
// at 3,520 patches; step 1 cut that to one dim per thread.)
extern "C" __global__ void vit_attn(const float* __restrict__ qkv, float* __restrict__ out,
                                    const int* __restrict__ n_p) {
    const int BR = 16, KC = 64, KS = 73;   // rows per block, key chunk, padded smem row
    int n = *n_p;
    int q0 = blockIdx.x * BR;
    int h = blockIdx.y;
    int t = threadIdx.x;
    int lane = t & 31, w = t >> 5;
    __shared__ float qs[16 * 72];
    __shared__ float kv[64 * 73];          // one K or V chunk, stride 73: conflict-free
    __shared__ float ps[16 * 256];         // the tile's scores, then its probabilities
    __shared__ float red[16 * 128];        // the sum tree's upper levels
    __shared__ float mrow[16], lrow[16], mnew[16], rsc[16];
    const float NEG_INF = -__int_as_float(0x7f800000);
    for (int i = t; i < BR * 72; i += 256) {
        int r = i / 72, d = i - r * 72;
        qs[i] = (q0 + r < n) ? qkv[(size_t)(q0 + r) * 3456 + h * 72 + d] : 0.0f;
    }
    if (t < BR) { mrow[t] = NEG_INF; lrow[t] = 0.0f; }
    // output accumulator a = t + 256 * i is (row a / 72, dim a % 72): 1152 per block
    float acc[5];
    #pragma unroll
    for (int i = 0; i < 5; i++) acc[i] = 0.0f;
    const float SCALE = 0.11785113019775793f;   // 1/sqrt(72)
    for (int kt = 0; kt < n; kt += 256) {
        int lim = (n - kt) < 256 ? (n - kt) : 256;
        // scores, 64 keys at a time: thread = key (t & 63) x rows (t >> 6) + 4i
        for (int c = 0; c < 4; c++) {
            __syncthreads();   // kv is free: the last chunk / the last tile's V walk is done
            for (int i = t; i < KC * 72; i += 256) {
                int kk = i / 72, d = i - kk * 72;
                int j = kt + c * KC + kk;
                kv[kk * KS + d] = (j < n) ? qkv[(size_t)j * 3456 + 1152 + h * 72 + d] : 0.0f;
            }
            __syncthreads();
            int kk = t & 63, rg = t >> 6;
            const float* kr = kv + kk * KS;
            float dot0 = 0.0f, dot1 = 0.0f, dot2 = 0.0f, dot3 = 0.0f;
            #pragma unroll 8
            for (int d = 0; d < 72; d++) {
                float kd = kr[d];
                dot0 += qs[rg * 72 + d] * kd;
                dot1 += qs[(rg + 4) * 72 + d] * kd;
                dot2 += qs[(rg + 8) * 72 + d] * kd;
                dot3 += qs[(rg + 12) * 72 + d] * kd;
            }
            bool in = kt + c * KC + kk < n;
            float* sp = ps + rg * 256 + c * KC + kk;
            sp[0] = in ? dot0 * SCALE : NEG_INF;
            sp[4 * 256] = in ? dot1 * SCALE : NEG_INF;
            sp[8 * 256] = in ? dot2 * SCALE : NEG_INF;
            sp[12 * 256] = in ? dot3 * SCALE : NEG_INF;
        }
        __syncthreads();
        // the tile max per row: warp w owns rows w and w + 8
        for (int rr = 0; rr < 2; rr++) {
            int r = w + 8 * rr;
            float mx = NEG_INF;
            for (int i = 0; i < 8; i++) mx = fmaxf(mx, ps[r * 256 + lane + 32 * i]);
            for (int o = 16; o > 0; o >>= 1) mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, o));
            if (lane == 0) mnew[r] = fmaxf(mrow[r], mx);
        }
        __syncthreads();
        for (int i = t; i < BR * 256; i += 256) {
            int r = i >> 8, k = i & 255;
            ps[i] = (kt + k < n) ? expf(ps[i] - mnew[r]) : 0.0f;
        }
        __syncthreads();
        // the tile sum per row: the pre-#98 tree, level by level
        for (int i = t; i < BR * 128; i += 256) {
            int r = i >> 7, k = i & 127;
            red[i] = ps[r * 256 + k] + ps[r * 256 + k + 128];
        }
        __syncthreads();
        for (int i = t; i < BR * 64; i += 256) {
            int r = i >> 6, k = i & 63;
            red[r * 128 + k] += red[r * 128 + k + 64];
        }
        __syncthreads();
        for (int i = t; i < BR * 32; i += 256) {
            int r = i >> 5, k = i & 31;
            red[r * 128 + k] += red[r * 128 + k + 32];
        }
        __syncthreads();
        for (int rr = 0; rr < 2; rr++) {
            int r = w + 8 * rr;
            float v = red[r * 128 + lane];
            for (int o = 16; o > 0; o >>= 1) v += __shfl_down_sync(0xffffffffu, v, o);
            if (lane == 0) {
                float mn_new = mnew[r];
                float rs = expf(mrow[r] - mn_new);
                lrow[r] = lrow[r] * rs + v;
                mrow[r] = mn_new;
                rsc[r] = rs;
            }
        }
        __syncthreads();
        #pragma unroll
        for (int i = 0; i < 5; i++) {
            int a = t + 256 * i;
            if (a < BR * 72) acc[i] *= rsc[a / 72];
        }
        // P.V: the tile's keys ascending, one 64-key V chunk at a time
        for (int c = 0; c * KC < lim; c++) {
            __syncthreads();
            for (int i = t; i < KC * 72; i += 256) {
                int kk = i / 72, d = i - kk * 72;
                int j = kt + c * KC + kk;
                kv[kk * KS + d] = (j < n) ? qkv[(size_t)j * 3456 + 2304 + h * 72 + d] : 0.0f;
            }
            __syncthreads();
            int cl = (lim - c * KC) < KC ? (lim - c * KC) : KC;
            #pragma unroll
            for (int i = 0; i < 5; i++) {
                int a = t + 256 * i;
                if (a < BR * 72) {
                    int r = a / 72, d = a - r * 72;
                    const float* pr = ps + r * 256 + c * KC;
                    for (int jj = 0; jj < cl; jj++) acc[i] += pr[jj] * kv[jj * KS + d];
                }
            }
        }
    }
    #pragma unroll
    for (int i = 0; i < 5; i++) {
        int a = t + 256 * i;
        if (a < BR * 72) {
            int r = a / 72, d = a - r * 72;
            if (q0 + r < n) out[((size_t)(q0 + r) * 16 + h) * 72 + d] = acc[i] / lrow[r];
        }
    }
}

// Tiled f32-activation NVFP4 GEMM for the vision tower (#VIT hotfix):
// y[t][rows] = W x^T with RAW f32 activations (the text dense GEMMs consume
// the quantized cascade, which would move the tower off the ~3e-6 oracle
// band). One block = 64 rows x 32 tokens; k walked in 64-value tiles; the
// tile's weight values (e2m1 magnitudes, NO scale) and the activation tile
// are staged in smem, so weight and activation reads amortize over 32 tokens
// and 64 rows instead of the per-(row, token) GEMV re-reads.
// Per output element the product tree is EXACTLY gemv_fp4_vit's: sub-blocks
// ascending, part = sum_j e2m1 * x (j ascending), acc += part * scale, one
// acc — bit-identical results, hundreds of times fewer instructions.
// Requires rows % 64 == 0 and k_dim % 16 == 0 (every vision linear shape).
extern "C" __global__ void gemm_fp4_f32x(const unsigned char* __restrict__ w,
                                         const float* __restrict__ x,
                                         const float* __restrict__ gs_ptr,
                                         float* __restrict__ y,
                                         const int* __restrict__ k_dim_p,
                                         const int* __restrict__ rows_p,
                                         const int* __restrict__ t_p) {
    int k_dim = *k_dim_p;
    int rows = *rows_p;
    int t = *t_p;
    int bpr = k_dim >> 4;                 // 16-value sub-blocks per row
    int r0 = blockIdx.x * 64;
    int t0 = blockIdx.y * 32;
    int ti = threadIdx.x;
    __shared__ float wv[64][64];          // tile weight values (e2m1, unscaled)
    __shared__ float xs[32][64];          // tile activations
    __shared__ float sc[64][4];           // per-row sub-block scales
    float acc[8];
    #pragma unroll
    for (int i = 0; i < 8; i++) acc[i] = 0.0f;
    int rl = ti >> 2;                     // row within the block (0..63)
    int tg = ti & 3;                      // token octant (0..3)
    float gs = gs_ptr[0];
    for (int kt = 0; kt < k_dim; kt += 64) {
        int lim = k_dim - kt; if (lim > 64) lim = 64;
        // decode 64 rows x lim values (unscaled e2m1)
        for (int i = ti; i < 64 * 64; i += 256) {
            int rr = i >> 6, vv = i & 63;
            float outv = 0.0f;
            if (vv < lim) {
                int G = (r0 + rr) * bpr + (kt >> 4) + (vv >> 4);
                const unsigned char* blk = w + (size_t)(G >> 2) * 36;
                int sb = G & 3;
                int idx = sb * 16 + (vv & 15);
                unsigned int byte = blk[4 + (idx >> 1)];
                unsigned int nib = (idx & 1) ? ((byte >> 4) & 0xF) : (byte & 0xF);
                outv = e2m1(nib);
            }
            wv[rr][vv] = outv;
        }
        // 64 rows x 4 sub-block scales
        for (int i = ti; i < 64 * 4; i += 256) {
            int rr = i >> 2, sb = i & 3;
            float s = 0.0f;
            if (kt + sb * 16 < k_dim) {
                int G = (r0 + rr) * bpr + (kt >> 4) + sb;
                const unsigned char* blk = w + (size_t)(G >> 2) * 36;
                s = ue4m3(blk[G & 3]) * gs;
            }
            sc[rr][sb] = s;
        }
        // 32 tokens x lim activations
        for (int i = ti; i < 32 * 64; i += 256) {
            int tt = i >> 6, vv = i & 63;
            float xv = 0.0f;
            if (t0 + tt < t && vv < lim) xv = x[(size_t)(t0 + tt) * k_dim + kt + vv];
            xs[tt][vv] = xv;
        }
        __syncthreads();
        // 8 (row, token) pairs per thread, the gemv_fp4_vit product tree
        for (int j = 0; j < 8; j++) {
            int tok = tg * 8 + j;
            if (t0 + tok >= t) continue;
            float a = 0.0f;
            for (int sb = 0; sb < 4; sb++) {
                if (kt + sb * 16 >= k_dim) break;
                int n = k_dim - kt - sb * 16; if (n > 16) n = 16;
                const float* wr = wv[rl] + sb * 16;
                const float* xr = xs[tok] + sb * 16;
                float part = 0.0f;
                #pragma unroll
                for (int jj = 0; jj < 16; jj++) {
                    if (jj >= n) break;
                    part += wr[jj] * xr[jj];
                }
                a += part * sc[rl][sb];
            }
            acc[j] += a;
        }
        __syncthreads();
    }
    for (int j = 0; j < 8; j++) {
        int tok = t0 + tg * 8 + j;
        if (tok < t) y[(size_t)tok * rows + r0 + rl] = acc[j];
    }
}

// exact-erf GELU (the merger activation): 0.5*x*(1+erff(x/sqrt(2)))
extern "C" __global__ void gelu_erf(float* __restrict__ x, const int* __restrict__ n_p) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= *n_p) return;
    float v = x[i];
    x[i] = 0.5f * v * (1.0f + erff(v * 0.70710678118654752440f));
}

// tanh-approx GELU (the vision MLP activation, config hidden_act
// gelu_pytorch_tanh): 0.5*x*(1+tanh(sqrt(2/pi)*(x+0.044715*x^3)))
extern "C" __global__ void gelu_tanh(float* __restrict__ x, const int* __restrict__ n_p) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= *n_p) return;
    float v = x[i];
    x[i] = 0.5f * v * (1.0f + tanhf(0.7978845608028654f * (v + 0.044715f * v * v * v)));
}
"#;

use crate::cuda;
use cudarc::driver::sys::CUfunction;
use std::collections::HashMap;

/// Read `#define <name> <integer>` out of the FROZEN `KERNEL_SRC`.
/// The CUDA source cannot change, so the Rust twins of its four `#define`s
/// (`QSA_PAR_BINS`, `SAMPLE_MAXK`, `SAMPLE_PARTS`, `SAMPLE_THREADS`) are
/// checked against it at boot instead of being trusted to a comment.
pub fn define_u32(name: &str) -> u32 {
    let pat = format!("#define {name} ");
    let i = KERNEL_SRC
        .find(&pat)
        .unwrap_or_else(|| panic!("{name}: no such #define in KERNEL_SRC"));
    let rest = &KERNEL_SRC[i + pat.len()..];
    let end = rest.find(|c: char| !c.is_ascii_digit()).unwrap_or(rest.len());
    rest[..end]
        .parse()
        .unwrap_or_else(|_| panic!("{name}: #define is not a plain integer"))
}

pub struct Kernels {
    map: HashMap<&'static str, CUfunction>,
}

impl Kernels {
    /// Every kernel the host LAUNCHES, resolved once. The frozen KERNEL_SRC defines
    /// more (six of them have no launch site left); this list is the launched set.
    pub unsafe fn new(module: &crate::cuda::Module) -> Kernels {
        let names: &[&'static str] = &[
            "gemv_b", "gemv_fp4", "gemv_fp4_b", "gemv_fp4_bs", "gemv_fp4_ptrb", "gemv_bf16",
            "rms_group", "rmsnorm_1pw", "silu_div4",
            "sigmoid_el", "sig2_div4", "mix_streams", "inject_residual", "silu_mul640",
            "silu_mul_combo", "acc_combo", "gate_shared", "conv_silu", "transpose_rt",
            "conv_state_update", "split_qkv", "l2norm_repeat", "beta_g", "delta_rule_persist", "conv_step",
            "delta_rule_step", "delta_rule_persist_r", "delta_rule_step_r", "rmsnorm_gated", "split_qg", "rope", "rope_p", "stage_cold", "stage_cold_ca", "stage_cold_lb", "stage_tiles_lb", "swap_pairs", "expand_slab", "moe_count", "moe_plan", "moe_scatter", "stage_tiles", "gemm_fp4_tiles", "silu_tiles", "quant_tiles", "store_kv", "attn_sel",
            "gate_mul", "rms128", "rope64", "pool4_cache", "qk_k_append", "d2d_block", "qsa_scores",
            "qsa_select", "qsa_select_fast", "qsa_select_par_h", "qsa_select_par_e", "router_top10", "gather_ple_fp4", "gate_dot", "gate_apply", "ple_conv",
            "ple_state_update", "ple_conv_step", "argmax_k", "sample_topk_part", "sample_k", "add_flat",
            "gemv_bf16_b", "gemv_bf16_w", "gemv_fp4_b1k", "hc_down_inj", "gemv_bf16_ws", "gemv_bf16_bs",
            "gemm_fp4_dense", "gemm_bf16_dense", "gemm_fp4_dense_b", "gemm_bf16_dense_b", "mix_streams_q", "rmsnorm_gated_q", "gate_mul_q", "silu_mul640_q", "silu_mul_combo_q", "qsa_scores_par", "attn_sel_split", "attn_sel_split_l", "attn_merge", "cast_e4m3_flat", "dec_e4m3_flat",
            // #96: the YaRN runtime-scale twins (RT = 1) + the one-boot-write
            // setter; resolved by name but launched only when an mscale is armed
            "attn_sel_y", "attn_sel_r_y", "attn_sel_d8_y", "attn_sel_d9_y", "attn_sel_s_y", "attn_sel_s8_y", "attn_sel_s8l_y", "attn_sel_g_y", "attn_sel_split_y", "attn_sel_split_l_y", "set_attn_scale",
            "quant_x_fp4", "gemv_fp4_mma", "gemv_fp4_mma_d", "gemv_fp4_mma_dg", "sh_gate_up_q", "gemv_fp4_mma_d32", "gemv_fp4_mma_g32", "attn_sel_r", "attn_sel_d8", "attn_sel_d9", "attn_sel_s", "attn_sel_s8", "attn_sel_s8l", "attn_sel_g",
            "gemm_fp4_f32x", "vit_ln", "vit_add_bias", "vit_pe_add", "vit_rope", "vit_attn", "gelu_erf", "gelu_tanh",
        ];
        let mut map = HashMap::new();
        for n in names {
            map.insert(*n, module.get(n));
        }
        // #96: arm the YaRN runtime attention scale. With no rope_scaling (the
        // checkpoint of record) nothing below runs — no upload, no substitution,
        // every launch resolves the RT = 0 kernel it always resolved, with the
        // same arguments: byte-identical map, launches and logits. Armed (an
        // mscale != 1), the scale 0.0625·mscale is written ONCE into the device
        // global and the ten attention variants swap to their _y twins, which
        // read it — same signatures, same launch sites, one scalar different
        // by design.
        if let Some(s) = crate::meta::boot_rope_scaling().filter(|s| s.mscale() != 1.0) {
            let scale = 0.0625f32 * s.mscale() as f32;
            let mut v = cuda::to_f32_dev(&[scale]);
            launch_sync(*map.get("set_attn_scale").unwrap(), 1, 1, 1, 32, &[v as u64]);
            cuda::free_dev(&mut v);
            for (n, y) in [
                ("attn_sel", "attn_sel_y"),
                ("attn_sel_r", "attn_sel_r_y"),
                ("attn_sel_d8", "attn_sel_d8_y"),
                ("attn_sel_d9", "attn_sel_d9_y"),
                ("attn_sel_s", "attn_sel_s_y"),
                ("attn_sel_s8", "attn_sel_s8_y"),
                ("attn_sel_s8l", "attn_sel_s8l_y"),
                ("attn_sel_g", "attn_sel_g_y"),
                ("attn_sel_split", "attn_sel_split_y"),
                ("attn_sel_split_l", "attn_sel_split_l_y"),
            ] {
                map.insert(n, *map.get(y).unwrap());
            }
            tracing::info!(
                target: "kernels",
                "[kernels] #96 attention scale 0.0625 -> {:.6} (YaRN mscale {:.4}, factor {}) - the _y runtime-scale twins carry every attention launch",
                scale,
                s.mscale(),
                s.factor
            );
        }
        Kernels { map }
    }

    pub fn f(&self, name: &str) -> CUfunction {
        if KPROF_ON.load(std::sync::atomic::Ordering::Relaxed) {
            LAST_NAME.with(|c| *c.borrow_mut() = name.to_string());
        }
        *self.map.get(name).unwrap_or_else(|| panic!("kernel missing: {name}"))
    }
}

/// CROW_KPROF=1: per-kernel GPU-time profile (sync before/after each launch,
/// so the number is kernel time + one launch latency). `f()` records the
/// name of the kernel about to be launched; `launch_v` below attributes the time.
pub static KPROF_ON: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
thread_local! {
    pub static LAST_NAME: std::cell::RefCell<String> = std::cell::RefCell::new(String::new());
}
pub fn kprof_init() {
    KPROF_ON.store(std::env::var("CROW_KPROF").is_ok(), std::sync::atomic::Ordering::Relaxed);
}
pub fn last_name() -> String {
    LAST_NAME.with(|c| c.borrow().clone())

}

// ---- per-kernel profile (CROW_KPROF=1) ----
pub static KPROF: std::sync::Mutex<Option<std::collections::HashMap<String, (u64, u64)>>> =
    std::sync::Mutex::new(None);
pub fn kprof_add(name: String, us: u64) {
    let mut g = KPROF.lock().unwrap();
    let m = g.get_or_insert_with(std::collections::HashMap::new);
    let e = m.entry(name).or_insert((0, 0));
    e.0 += 1;
    e.1 += us;
}
/// per-kernel table (sorted by total time), normalized per `steps`
pub fn kprof_report(steps: u64) {
    let g = KPROF.lock().unwrap();
    let Some(m) = g.as_ref() else { return };
    let mut rows: Vec<(&String, &(u64, u64))> = m.iter().collect();
    rows.sort_by(|a, b| b.1 .1.cmp(&a.1 .1));
    let total: u64 = rows.iter().map(|r| r.1 .1).sum();
    let st = steps.max(1) as f64;
    tracing::info!(target: "kprof", "[kprof] kernel time incl. one launch latency each, over {steps} steps (total {:.1} ms/step)", total as f64 / 1000.0 / st);
    tracing::info!(target: "kprof", "[kprof] {:<22} {:>9} {:>11} {:>9} {:>6}", "kernel", "calls/step", "ms/step", "us/call", "share");
    for (n, (c, us)) in rows {
        tracing::info!(target: "kprof", "[kprof] {:<22} {:>9.1} {:>11.3} {:>9.1} {:>5.1}%", n, *c as f64 / st, *us as f64 / 1000.0 / st, *us as f64 / *c as f64, 100.0 * *us as f64 / total as f64);
    }
}

/// every kernel arg is a device address (scalars live in device buffers — the
/// p5 rule), so the arg list is just u64 values.
pub unsafe fn launch_sync(
    f: cudarc::driver::sys::CUfunction,
    gx: u32,
    gy: u32,
    gz: u32,
    bx: u32,
    vals: &[u64],
) {
    launch_v(f, gx, gy, gz, bx, vals);
    cuda::sync();
}

pub unsafe fn launch_v(
    f: cudarc::driver::sys::CUfunction,
    gx: u32,
    gy: u32,
    gz: u32,
    bx: u32,
    vals: &[u64],
) {
    use cudarc::driver::sys;
    use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
    static DBG: AtomicBool = AtomicBool::new(false);
    static INIT: AtomicBool = AtomicBool::new(false);
    static LAUNCH_N: AtomicU64 = AtomicU64::new(0);
    if !INIT.swap(true, Ordering::Relaxed) {
        DBG.store(std::env::var("ENGINE_DEBUG_SYNC").is_ok(), Ordering::Relaxed);
    }
    let mut ptrs: Vec<*mut std::ffi::c_void> = vals
        .iter()
        .map(|v| v as *const u64 as *mut std::ffi::c_void)
        .collect();
    let stream = cuda::cur_stream();
    let kprof = KPROF_ON.load(Ordering::Relaxed);
    if kprof {
        cuda::sync();
    }
    let t_k = std::time::Instant::now();
    let r = sys::cuLaunchKernel(
        f, gx, gy, gz, bx, 1, 1, 0,
        stream,
        ptrs.as_mut_ptr(),
        std::ptr::null_mut(),
    );
    if kprof {
        cuda::sync();
        kprof_add(format!("{}[{}x{}]", last_name(), gx, gy), t_k.elapsed().as_micros() as u64);
    }
    if std::env::var("ENGINE_DEBUG_LAUNCH").is_ok() {
        tracing::info!(target: "kernels", "[launch] stream={:p} gx={gx} gy={gy} r={:?}", stream as *mut std::ffi::c_void, r);
    }
    cuda::ck(r);
    if DBG.load(Ordering::Relaxed) {
        let n = LAUNCH_N.fetch_add(1, Ordering::Relaxed);
        if std::env::var("ENGINE_DEBUG_TRACE").as_deref() == Ok("1") {
            tracing::info!(target: "kernels",
                "[launch {n}] go gx={gx} gy={gy} gz={gz} bx={bx} a0={:x} a1={:x} a2={:x}",
                vals.get(0).copied().unwrap_or(0),
                vals.get(1).copied().unwrap_or(0),
                vals.get(2).copied().unwrap_or(0)
            );
        }
        cuda::sync();
    }
}

#[cfg(test)]
mod tests_96 {
    //! #96: the kernel-source gate. NVRTC is a HOST-side compiler: this parses
    //! and compiles the frozen KERNEL_SRC to PTX with no CUDA context, no GPU
    //! and no pinned allocation — the earliest possible gate on a CUDA error in
    //! the runtime-scale twins. The PTX text then carries the two byte-identity
    //! claims of the issue: every kernel of record folds the 0.0625f immediate
    //! and never touches the runtime global; the _y twins load it.
    use super::KERNEL_SRC;

    /// the PTX text of one entry, from its `.entry <name>(` to the closing
    /// brace at column 0 (inner braces are indented in PTX)
    fn body_of<'a>(ptx: &'a str, entry: &str) -> &'a str {
        let start = ptx
            .find(&format!(".entry {entry}("))
            .unwrap_or_else(|| panic!("no .entry {entry} in the PTX"));
        let end = ptx[start..]
            .find("\n}\n")
            .map(|e| start + e + 3)
            .unwrap_or(ptx.len());
        &ptx[start..end]
    }

    #[test]
    fn the_source_compiles_and_the_scale_split_is_visible_in_the_ptx() {
        let opts = cudarc::nvrtc::CompileOptions {
            options: vec!["--gpu-architecture=compute_120a".into()],
            ..Default::default()
        };
        let ptx = cudarc::nvrtc::compile_ptx_with_opts(KERNEL_SRC, opts)
            .expect("nvrtc compile")
            .to_src();
        assert!(ptx.contains(".entry set_attn_scale"), "the #96 boot setter must exist");
        // nvrtc emits `.global .align 4 .f32 d_attn_scale = 0f3D800000;`
        // (0f3D800000 IS 0.0625f = 2^-4; 0f3E000000 would be 0.125f)
        assert!(ptx.contains(".global .align 4 .f32 d_attn_scale"), "the runtime scale global must exist");
        assert!(ptx.contains("= 0f3D800000"), "the runtime global initializes to 0.0625f");
        // the kernels of record: the folded immediate, never the global
        for k in ["attn_sel", "attn_sel_r", "attn_sel_s8l", "attn_sel_g", "attn_sel_split_l"] {
            let b = body_of(&ptx, k);
            assert!(b.contains("0f3D800000"), "{k} must fold the 0.0625f immediate");
            assert!(!b.contains("d_attn_scale"), "{k} must not read the runtime scale");
        }
        // the _y twins: the global load, present in the module whether or not
        // any boot ever arms them
        for k in ["attn_sel_y", "attn_sel_r_y", "attn_sel_s8l_y", "attn_sel_g_y", "attn_sel_split_l_y"] {
            let b = body_of(&ptx, k);
            assert!(b.contains("d_attn_scale"), "{k} must read the runtime scale");
        }
    }
}

#[cfg(test)]
mod tests_ue4m3_enc {
    //! Host twin of the CUDA `enc_ue4m3_up` (the activation-quant scale
    //! encoder of #10), mirrored expression by expression: ldexpf(1, 10 - e)
    //! is the exact power of two 2^(10-e), ceilf is f32::ceil. The source
    //! check pins the twin to the kernel text so the two cannot drift.
    use super::KERNEL_SRC;

    fn enc_ue4m3_up(s: f32) -> u8 {
        if !(s > 0.0) { return 0; }
        for e in 0..16i32 {
            let mul = if e == 0 { 512.0f32 } else { 2.0f32.powi(10 - e) };
            let m = (s * mul).ceil() - if e == 0 { 0.0 } else { 8.0 };
            let m_max = if e == 15 { 6.0 } else { 7.0 };
            if m >= 0.0 && m <= m_max { return ((e << 3) | m as i32) as u8; }
        }
        0x7E
    }

    #[test]
    fn the_twin_matches_the_kernel_text() {
        assert!(KERNEL_SRC.contains("float m_max = (e == 15) ? 6.0f : 7.0f;"));
        assert!(KERNEL_SRC.contains("if (m >= 0.0f && m <= m_max) return (unsigned char)((e << 3) | (int)m);"));
    }

    #[test]
    fn the_encoder_never_emits_the_nan_byte() {
        // geometric sweep 1e-4 .. 2000, ~20k points per decade
        let (lo, hi) = (1e-4f64, 2000f64);
        let n = 150_000usize;
        let step = (hi / lo).ln() / n as f64;
        for i in 0..=n {
            let s = (lo * (step * i as f64).exp()) as f32;
            let b = enc_ue4m3_up(s);
            assert_ne!(b, 0x7F, "s = {s} encodes to the E4M3 NaN byte");
        }
        // every f32 in (448, 480] explicitly (the old failure band)
        let mut s = 448.0f32;
        while s <= 480.0 {
            s = f32::from_bits(s.to_bits() + 1);
            assert_ne!(enc_ue4m3_up(s), 0x7F, "s = {s}");
        }
        assert_eq!(enc_ue4m3_up(460.0), 0x7E);
        assert_eq!(enc_ue4m3_up(480.0), 0x7E);
        assert_eq!(enc_ue4m3_up(448.0), 0x7E);
        assert_eq!(enc_ue4m3_up(2000.0), 0x7E);
        // the regular range is unchanged: 416 < s <= 448 still hits m = 6
        assert_eq!(enc_ue4m3_up(420.0), (15 << 3) | 6);
        assert_eq!(enc_ue4m3_up(1.0), 7 << 3);
        assert_eq!(enc_ue4m3_up(0.0), 0);
    }
}
