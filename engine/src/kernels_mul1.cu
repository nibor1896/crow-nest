// crow-nest #180: MUL1 trellis expert kernels (NVRTC, compute_120a), the GPU half of
// `cpu_mul1.rs`. Its own module (`kernels::mul1::Kernels::new` compiles `MUL1_SRC` alone):
// one more `.entry` in KERNEL_SRC would break the PTX of record (`kernels::tests_300_c4`).
//
// Format: the CNQ MUL1 record of #181 (`converter/src/mul1.rs`), one per expert:
// [gate.trellis][up.trellis][down.trellis][gate.suh][gate.svh][up.suh][up.svh][down.suh][down.svh]
// [zeros]. A linear [k = in, n = out] is a trellis of (k/16)(n/16) tiles of `n32` u32 words
// (tile kb * n/16 + nb holds the 16x16 block of rows 16 kb.., columns 16 nb..), plus fp16 suh
// [k] and svh [n]. y = x W with W = diag(suh) H W_hat H diag(svh) / 128 (H the 128-wide
// Sylvester Hadamard, entries +-1): exllamav3 `LinearEXL3.get_weight_tensor` and the
// `had_r_128` / `hgemm` forward path (`exllamav3/modules/quant/exl3.py:193-249` at commit
// 151539c7). So every GEMV is three steps, none of which writes a dequantized weight:
//   mul1_had_in   xh = H (x * suh)                    (activations only)
//   mul1_gemv     y' = xh W_hat, W_hat decoded from the trellis in the inner loop
//   mul1_had_out  y  = (H y') / 128 * svh             (also sums the k-split partials)
// mul1_act_had_in is mul1_had_out for gate and up, silu(g) * u, and mul1_had_in for down.
//
// Decode, after exllamav3 (MIT, turboderp-org/exllamav3 @ 151539c7):
// `exllamav3_ext/quant/codebook.cuh:38-52` (decode_mul1_product_2: state * 0x83DCD12D,
// __dp4a byte sum + 0x6400, one fp16 fma with k_inv = 0x1eee, k_bias = 0xc931) and the
// in-warp window extraction of `exllamav3_ext/quant/exl3_gemv_kernel.cuh:1-21` (tile words
// staged through shared memory, the SMEM_STAGE form). The words reach shared memory as whole
// 128-byte lines loaded by the whole block, two stages deep (#183, mul1_gemv): from pinned RAM
// the first kernel's 96-byte warp loads (mul1_gemv_warp, kept as the reference) read ~8 GB/s,
// the block runs ~36 GB/s on the RTX 5090. Unlike exllamav3 (fp16
// MMA, fp16 accumulation) every product and sum here is f32. `mul1_w` computes the exact
// value (1024 + s) * k_inv + k_bias in f32 (11-bit by 10-bit product, sum on a 2^-18 grid
// below 2^4: exact) and rounds it once to fp16 (cvt.rn), which is __hfma's one rounding, so
// the weight is the codec's fp16 value bit for bit (`mul1_decode_states`, tested).
//
// The f32 order (the rounding count n of docs/mul1-gemv.md section 3, `mul1::rounding_steps`):
// x * suh (1), FWHT 7 stages (7), per lane an fmaf chain over its k-split: 4 terms per tile,
// k / (16 S) tiles (k / (4 S)), two xor-shuffle adds (2), S partials summed left to right
// (S - 1), FWHT (7), * 2^-7 (exact), * svh (1): n = k / (4 S) + S + 17.
// Every rounded add and multiply outside the fmaf chain is an __fadd_rn / __fmul_rn
// intrinsic, which NVRTC never contracts into an fma (-fmad=true is its default), so the
// order above is the compiled order. expf is the accurate one (no --use_fast_math).
// Scalar arguments come in one device int buffer per launch (the KERNEL_SRC rule).

#define MUL1_MAXT 8
#define MUL1_XROWS 512
#define MUL1_U 4
// u32 words of the largest tile (K = 8: 16 * 8 / 2), and 16-byte loads per thread per stage
#define MUL1_N32MAX 64
#define MUL1_PIECES ((MUL1_U * 8 * MUL1_N32MAX / 4 + 255) / 256)

__device__ __forceinline__ float mul1_h2f(unsigned short h) {
    float f;
    asm("cvt.f32.f16 %0, %1;" : "=f"(f) : "h"(h));
    return f;
}

// the codec weight of one 16-bit trellis state (exact fp16 value as f32)
__device__ __forceinline__ float mul1_w(unsigned int state) {
    unsigned int x = state * 0x83DCD12Du;
    unsigned int s;
    asm("dp4a.u32.u32 %0, %1, %2, %3;" : "=r"(s) : "r"(x), "r"(0x01010101u), "r"(0u));
    // 0x1eee = 1774 * 2^-18, 0xc931 = -10.3828125: both exact in f32, so is the result
    float v = (float)(1024u + s) * 6.76727294921875e-3f + (-10.3828125f);
    unsigned short h;
    asm("cvt.rn.f16.f32 %0, %1;" : "=h"(h) : "f"(v));
    return mul1_h2f(h);
}

// exclusive end bit of trellis position t in a tile's circular stream (#181 `Bitrate::end_bit`)
__device__ __forceinline__ int mul1_end_bit(int t, int bits, int half) {
    if (!half) return (t + 1) * bits;
    return (t & 1) ? (t / 2 + 1) * (2 * bits + 1) : (t / 2) * (2 * bits + 1) + bits;
}

// In-place natural-order Walsh-Hadamard transform of 128 floats over one warp, element
// 4 * lane + j in v[j]: stages on bits 0, 1 in-thread, bits 2..6 by xor shuffles, each
// stage (a, b) -> (a + b, a - b). The same order as cpu_mul1::fwht128.
__device__ __forceinline__ void mul1_fwht128(float v[4], int lane) {
    float a0 = __fadd_rn(v[0], v[1]), a1 = __fsub_rn(v[0], v[1]), a2 = __fadd_rn(v[2], v[3]), a3 = __fsub_rn(v[2], v[3]);
    v[0] = __fadd_rn(a0, a2);
    v[1] = __fadd_rn(a1, a3);
    v[2] = __fsub_rn(a0, a2);
    v[3] = __fsub_rn(a1, a3);
    #pragma unroll
    for (int m = 1; m < 32; m <<= 1) {
        #pragma unroll
        for (int j = 0; j < 4; j++) {
            float o = __shfl_xor_sync(0xffffffffu, v[j], m);
            v[j] = (lane & m) ? __fsub_rn(o, v[j]) : __fadd_rn(v[j], o);
        }
    }
}

// xh = H (x * suh) for one 128-block of one token row; v in, v out
__device__ __forceinline__ void mul1_in_block(float v[4], const unsigned short* __restrict__ suh, int c0, int lane) {
    #pragma unroll
    for (int j = 0; j < 4; j++) v[j] = __fmul_rn(v[j], mul1_h2f(suh[c0 + j]));
    mul1_fwht128(v, lane);
}

// y = (H sum_s part[s]) / 128 * svh for outputs blk*128 .. +128 of token t, slot e
__device__ __forceinline__ void mul1_out_block(float v[4], const float* __restrict__ part, const unsigned short* __restrict__ svh,
                                               int e, int S, int T, int t, int n, int blk, int lane) {
    int c0 = blk * 128 + 4 * lane;
    #pragma unroll
    for (int j = 0; j < 4; j++) {
        float s = part[(((size_t)e * S + 0) * T + t) * n + c0 + j];
        for (int sp = 1; sp < S; sp++) s = __fadd_rn(s, part[(((size_t)e * S + sp) * T + t) * n + c0 + j]);
        v[j] = s;
    }
    mul1_fwht128(v, lane);
    #pragma unroll
    for (int j = 0; j < 4; j++) v[j] = __fmul_rn(__fmul_rn(v[j], 0.0078125f), mul1_h2f(svh[c0 + j]));
}

// grid (k/128, T, E), block 32. p: [0] k, [1] T, [2] suh_off (bytes into the record).
// x, xh: [E][T][k]; ptrs[e]: record base (VRAM slot or pinned UVA address)
extern "C" __global__ void mul1_had_in(const unsigned long long* __restrict__ ptrs, const float* __restrict__ x,
                                       float* __restrict__ xh, const int* __restrict__ p) {
    const int k = p[0], T = p[1];
    const int e = blockIdx.z, t = blockIdx.y, blk = blockIdx.x, lane = threadIdx.x;
    const unsigned short* suh = (const unsigned short*)(ptrs[e] + (unsigned long long)p[2]);
    const size_t row = ((size_t)e * T + t) * k;
    const int c0 = blk * 128 + 4 * lane;
    float v[4];
    #pragma unroll
    for (int j = 0; j < 4; j++) v[j] = x[row + c0 + j];
    mul1_in_block(v, suh, c0, lane);
    #pragma unroll
    for (int j = 0; j < 4; j++) xh[row + c0 + j] = v[j];
}

// One tile of the GEMV inner loop for one warp: lane g decodes its 8 weights from the tile's
// words w (n32 u32, shared memory) and adds them into its fmaf chains, token by token. Both GEMV
// kernels call exactly this, so their f32 operation order is one and the same.
__device__ __forceinline__ void mul1_tile_fma(const unsigned int* w, const int lo[8], int n32, const float* xs, int rows,
                                              int r0, int T, float acc0[MUL1_MAXT], float acc1[MUL1_MAXT]) {
    float wv[8];
    #pragma unroll
    for (int j = 0; j < 8; j++) {
        int i = lo[j] >> 5, o = lo[j] & 31;
        int i1 = (i + 1 == n32) ? 0 : i + 1;
        unsigned long long pair = ((unsigned long long)w[i] << 32) | w[i1];
        wv[j] = mul1_w((unsigned int)(pair >> (48 - o)) & 0xffffu);
    }
    #pragma unroll
    for (int t = 0; t < MUL1_MAXT; t++) {
        if (t < T) {
            const float* xr = xs + t * rows + r0;
            float x0 = xr[0], x1 = xr[1], x8 = xr[8], x9 = xr[9];
            float a = acc0[t], b = acc1[t];
            a = fmaf(wv[0], x0, a);
            a = fmaf(wv[1], x1, a);
            a = fmaf(wv[2], x8, a);
            a = fmaf(wv[3], x9, a);
            b = fmaf(wv[4], x0, b);
            b = fmaf(wv[5], x1, b);
            b = fmaf(wv[6], x8, b);
            b = fmaf(wv[7], x9, b);
            acc0[t] = a;
            acc1[t] = b;
        }
    }
}

// the lane chains of one warp into its k-split partials (two xor-shuffle adds per chain)
__device__ __forceinline__ void mul1_store_part(float* __restrict__ part, const float acc0[MUL1_MAXT], const float acc1[MUL1_MAXT],
                                                int e, int S, int sp, int T, int n, int nb, int lane) {
    #pragma unroll
    for (int t = 0; t < MUL1_MAXT; t++) {
        if (t < T) {
            float a = acc0[t], b = acc1[t];
            a = __fadd_rn(a, __shfl_xor_sync(0xffffffffu, a, 1));
            b = __fadd_rn(b, __shfl_xor_sync(0xffffffffu, b, 1));
            a = __fadd_rn(a, __shfl_xor_sync(0xffffffffu, a, 2));
            b = __fadd_rn(b, __shfl_xor_sync(0xffffffffu, b, 2));
            if ((lane & 3) == 0) {
                float* pp = part + (((size_t)e * S + sp) * T + t) * n + nb * 16 + (lane >> 2);
                pp[0] = a;
                pp[8] = b;
            }
        }
    }
}

// the activation rows of k-split sp into shared memory, xs[t * rows + r]
__device__ __forceinline__ void mul1_stage_x(float* xs, const float* __restrict__ xh, int e, int T, int k, int kb0, int rows) {
    for (int i = threadIdx.x; i < T * rows; i += blockDim.x) {
        int t = i / rows, r = i - t * rows;
        xs[t * rows + r] = xh[((size_t)e * T + t) * k + (size_t)kb0 * 16 + r];
    }
}

// window start bit of each of lane g's 8 trellis positions
__device__ __forceinline__ void mul1_lane_lo(int lo[8], int lane, int bits, int half, int n32) {
    const int sbits = n32 * 32;
    #pragma unroll
    for (int j = 0; j < 8; j++) lo[j] = (mul1_end_bit(8 * lane + j, bits, half) - 16 + sbits) % sbits;
}

// grid (n/128, S, E), block 256 = 8 warps. Block bx owns the 8 adjacent tile columns
// nb = 8 bx .. 8 bx + 7 over the tile rows kb = blockIdx.y * tps .. + tps (tps = k / 16 / S); warp
// w computes tile column 8 bx + w. Lane g is trellis lane g: positions 8 g + j land at rows
// 2 (g % 4) + {0, 1, 8, 9}[j % 4], column g / 4 + 8 (j / 4) of the tile (#181 `tile_index`).
// Loads: the 8 tiles of one tile row are one contiguous run of 32 n32 bytes, a multiple of 128
// for every bitrate (n32 = 8 bits, or 8 bits + 4), so the runs are 128-byte aligned whenever the
// trellis base is. The whole block reads MUL1_U runs per stage with 16-byte loads, whole 128-byte
// lines, so a zero-copy read from pinned RAM leaves the GPU as full-line PCIe requests (EMOGI,
// arXiv 2006.06890 section 3.3), into a two-deep shared ring: the loads of stage s + 1 are in
// flight while the warps compute stage s. A trellis base that is not 16-byte aligned falls back
// to 4-byte loads of the same words. The f32 order is that of mul1_gemv_warp (both call
// mul1_tile_fma per tile, kb ascending, then mul1_store_part), so the bits are the same.
// p: [0] k, [1] n, [2] S, [3] tr_off, [4] n32 (u32 words per tile), [5] bits, [6] half, [7] T.
// xh: [E][T][k]; part: [E][S][T][n] (raw y' partials, one per k-split)
extern "C" __global__ void __launch_bounds__(256) mul1_gemv(const unsigned long long* __restrict__ ptrs, const float* __restrict__ xh,
                                                            float* __restrict__ part, const int* __restrict__ p) {
    const int k = p[0], n = p[1], S = p[2], n32 = p[4], bits = p[5], half = p[6], T = p[7];
    const int e = blockIdx.z, sp = blockIdx.y;
    const unsigned long long tbase = ptrs[e] + (unsigned long long)p[3];
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int tiles_n = n >> 4, tps = (k >> 4) / S;
    const int nb0 = blockIdx.x * 8, nb = nb0 + warp;
    const int kb0 = sp * tps, rows = tps * 16;
    __shared__ float xs[MUL1_MAXT * MUL1_XROWS];
    __shared__ __align__(16) unsigned int ring[2][MUL1_U * 8 * MUL1_N32MAX];
    mul1_stage_x(xs, xh, e, T, k, kb0, rows);
    const int run4 = 2 * n32;          // 16-byte pieces of one tile row's run (8 n32 words)
    const int stage4 = MUL1_U * run4;  // pieces per stage, <= MUL1_PIECES * 256
    const bool vec = (tbase & 15ull) == 0;
    uint4 rg[MUL1_PIECES];
    int lo[8];
    mul1_lane_lo(lo, lane, bits, half, n32);
    float acc0[MUL1_MAXT], acc1[MUL1_MAXT];
    #pragma unroll
    for (int t = 0; t < MUL1_MAXT; t++) {
        acc0[t] = 0.0f;
        acc1[t] = 0.0f;
    }
    const int rbase = 2 * (lane & 3);
    for (int kt = 0, buf = 0; kt < tps + MUL1_U; kt += MUL1_U, buf ^= 1) {
        // issue the loads of the stage at tile row kt (if any) into registers
        if (kt < tps) {
            #pragma unroll
            for (int q = 0; q < MUL1_PIECES; q++) {
                const int i = threadIdx.x + q * 256;
                const int u = i / run4, c = i - u * run4;
                if (i < stage4 && kt + u < tps) {
                    const size_t w0 = ((size_t)(kb0 + kt + u) * tiles_n + nb0) * n32 + 4 * c;
                    if (vec) {
                        rg[q] = ((const uint4*)tbase)[w0 >> 2];
                    } else {
                        const unsigned int* s = (const unsigned int*)tbase + w0;
                        rg[q] = make_uint4(s[0], s[1], s[2], s[3]);
                    }
                }
            }
        }
        // compute the previous stage (rows kt - MUL1_U ..) from the other ring slot
        if (kt > 0) {
            const int kp = kt - MUL1_U;
            #pragma unroll
            for (int u = 0; u < MUL1_U; u++) {
                if (kp + u < tps)
                    mul1_tile_fma(ring[buf ^ 1] + u * (8 * MUL1_N32MAX) + warp * n32, lo, n32, xs, rows, (kp + u) * 16 + rbase, T, acc0, acc1);
            }
        }
        // the stage at kt into its ring slot
        if (kt < tps) {
            #pragma unroll
            for (int q = 0; q < MUL1_PIECES; q++) {
                const int i = threadIdx.x + q * 256;
                const int u = i / run4, c = i - u * run4;
                if (i < stage4 && kt + u < tps) ((uint4*)ring[buf])[u * (2 * MUL1_N32MAX) + c] = rg[q];
            }
        }
        __syncthreads();
    }
    mul1_store_part(part, acc0, acc1, e, S, sp, T, n, nb, lane);
}

// The first GEMV kernel of #180, kept as the reference arm of the bit-identity test and the
// benchmark: grid and block as mul1_gemv, but warp w loads its own tile (lane i < n32 loads word
// i, MUL1_U tiles, through warp-private shared memory, no prefetch). From pinned RAM that load
// shape caps at about 8 GB/s on the RTX 5090 (rd_warp96, #183).
extern "C" __global__ void mul1_gemv_warp(const unsigned long long* __restrict__ ptrs, const float* __restrict__ xh,
                                          float* __restrict__ part, const int* __restrict__ p) {
    const int k = p[0], n = p[1], S = p[2], n32 = p[4], bits = p[5], half = p[6], T = p[7];
    const int e = blockIdx.z, sp = blockIdx.y;
    const unsigned int* tr = (const unsigned int*)(ptrs[e] + (unsigned long long)p[3]);
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int tiles_n = n >> 4, tps = (k >> 4) / S;
    const int nb = blockIdx.x * 8 + warp;
    const int kb0 = sp * tps, rows = tps * 16;
    __shared__ float xs[MUL1_MAXT * MUL1_XROWS];
    __shared__ unsigned int tw[8][MUL1_U][64];
    mul1_stage_x(xs, xh, e, T, k, kb0, rows);
    __syncthreads();
    const int rbase = 2 * (lane & 3);
    int lo[8];
    mul1_lane_lo(lo, lane, bits, half, n32);
    float acc0[MUL1_MAXT], acc1[MUL1_MAXT];
    #pragma unroll
    for (int t = 0; t < MUL1_MAXT; t++) {
        acc0[t] = 0.0f;
        acc1[t] = 0.0f;
    }
    for (int kt = 0; kt < tps; kt += MUL1_U) {
        #pragma unroll
        for (int u = 0; u < MUL1_U; u++) {
            if (kt + u < tps) {
                const unsigned int* tp = tr + ((size_t)(kb0 + kt + u) * tiles_n + nb) * n32;
                unsigned int v0 = lane < n32 ? tp[lane] : 0u;
                unsigned int v1 = lane + 32 < n32 ? tp[lane + 32] : 0u;
                tw[warp][u][lane] = v0;
                tw[warp][u][lane + 32] = v1;
            }
        }
        __syncwarp();
        #pragma unroll
        for (int u = 0; u < MUL1_U; u++) {
            if (kt + u < tps) mul1_tile_fma(tw[warp][u], lo, n32, xs, rows, (kt + u) * 16 + rbase, T, acc0, acc1);
        }
        __syncwarp();
    }
    mul1_store_part(part, acc0, acc1, e, S, sp, T, n, nb, lane);
}

// grid (n/128, T, E), block 32. p: [0] n, [1] S, [2] T, [3] svh_off. y: [E][T][n]
extern "C" __global__ void mul1_had_out(const unsigned long long* __restrict__ ptrs, const float* __restrict__ part,
                                        float* __restrict__ y, const int* __restrict__ p) {
    const int n = p[0], S = p[1], T = p[2];
    const int e = blockIdx.z, t = blockIdx.y, blk = blockIdx.x, lane = threadIdx.x;
    const unsigned short* svh = (const unsigned short*)(ptrs[e] + (unsigned long long)p[3]);
    float v[4];
    mul1_out_block(v, part, svh, e, S, T, t, n, blk, lane);
    const size_t o = ((size_t)e * T + t) * n + blk * 128 + 4 * lane;
    #pragma unroll
    for (int j = 0; j < 4; j++) y[o + j] = v[j];
}

// The FFN middle: gate and up finished (mul1_had_out), h = silu(g) * u with
// silu(g) * u = g / (1 + exp(-g)) * u (the cpu_mul1 / silu_mul640 formula), h stored, then
// xh_d = H (h * suh_down) (mul1_had_in). grid (inter/128, T, E), block 32.
// p: [0] inter, [1] S (gate and up), [2] T, [3] svh_g_off, [4] svh_u_off, [5] suh_d_off.
// part_g, part_u: [E][S][T][inter]; h, xh_d: [E][T][inter]
extern "C" __global__ void mul1_act_had_in(const unsigned long long* __restrict__ ptrs, const float* __restrict__ part_g,
                                           const float* __restrict__ part_u, float* __restrict__ h,
                                           float* __restrict__ xh_d, const int* __restrict__ p) {
    const int inter = p[0], S = p[1], T = p[2];
    const int e = blockIdx.z, t = blockIdx.y, blk = blockIdx.x, lane = threadIdx.x;
    const unsigned long long base = ptrs[e];
    float g[4], u[4], v[4];
    mul1_out_block(g, part_g, (const unsigned short*)(base + (unsigned long long)p[3]), e, S, T, t, inter, blk, lane);
    mul1_out_block(u, part_u, (const unsigned short*)(base + (unsigned long long)p[4]), e, S, T, t, inter, blk, lane);
    const size_t o = ((size_t)e * T + t) * inter + blk * 128 + 4 * lane;
    #pragma unroll
    for (int j = 0; j < 4; j++) {
        v[j] = __fmul_rn(__fdiv_rn(g[j], __fadd_rn(1.0f, expf(-g[j]))), u[j]);
        h[o + j] = v[j];
    }
    mul1_in_block(v, (const unsigned short*)(base + (unsigned long long)p[5]), blk * 128 + 4 * lane, lane);
    #pragma unroll
    for (int j = 0; j < 4; j++) xh_d[o + j] = v[j];
}

// test hook: the weight of every 16-bit state, out[65536]
extern "C" __global__ void mul1_decode_states(float* __restrict__ out) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < 65536u) out[i] = mul1_w(i);
}
