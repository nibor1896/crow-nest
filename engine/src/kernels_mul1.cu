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

// #187 (CROW_GLM_GEMV2=1): the T = 1 GEMV for decode. Grid, block, k-split and the f32 order
// are those of mul1_gemv (mul1_tile_fma2 is mul1_tile_fma at T = 1, mul1_store_part as there), so
// the bits are the same. Two changes: (1) the block loads its whole k-split at once (tps tile
// rows x 8 tiles, at most MUL1_G2_WORDS words, 16-byte loads from every thread issued before
// any store) and syncs once, instead of MUL1_U rows per stage with a sync each: more bytes in
// flight per SM and no stage bubbles; small shared memory (T = 1 activations only), so more
// blocks fit on an SM. (2) The weights decode two at a time the way the codec defines them
// (exllamav3 decode_mul1_product_2): dp4a byte sum + 0x6400 as fp16 bits, one fma.rn.f16x2
// with k_inv, k_bias; that single fp16 rounding is the one mul1_w emulates in f32, so the
// weights are the same, without mul1_w's int-to-float and float-to-half conversions.
#define MUL1_G2_WORDS 4096

__device__ __forceinline__ void mul1_w2(unsigned int st0, unsigned int st1, float& w0, float& w1) {
    unsigned int x0 = st0 * 0x83DCD12Du, x1 = st1 * 0x83DCD12Du, s0, s1, r;
    asm("dp4a.u32.u32 %0, %1, %2, %3;" : "=r"(s0) : "r"(x0), "r"(0x01010101u), "r"(0x6400u));
    asm("dp4a.u32.u32 %0, %1, %2, %3;" : "=r"(s1) : "r"(x1), "r"(0x01010101u), "r"(0x6400u));
    unsigned int h = __byte_perm(s0, s1, 0x5410);
    asm("fma.rn.f16x2 %0, %1, %2, %3;" : "=r"(r) : "r"(h), "r"(0x1eee1eeeu), "r"(0xc931c931u));
    unsigned short lo, hi;
    asm("mov.b32 {%0, %1}, %2;" : "=h"(lo), "=h"(hi) : "r"(r));
    w0 = mul1_h2f(lo);
    w1 = mul1_h2f(hi);
}

__device__ __forceinline__ unsigned int mul1_state(const unsigned int* w, int i, int i1, int o) {
    unsigned long long pair = ((unsigned long long)w[i] << 32) | w[i1];
    return (unsigned int)(pair >> (48 - o)) & 0xffffu;
}

// p as mul1_gemv (p[7] = T must be 1). xh: [E][1][k]; part: [E][S][1][n]
extern "C" __global__ void __launch_bounds__(256) mul1_gemv2(const unsigned long long* __restrict__ ptrs, const float* __restrict__ xh,
                                                             float* __restrict__ part, const int* __restrict__ p) {
    const int k = p[0], n = p[1], S = p[2], n32 = p[4], bits = p[5], half = p[6];
    const int e = blockIdx.z, sp = blockIdx.y;
    const unsigned long long tbase = ptrs[e] + (unsigned long long)p[3];
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int tiles_n = n >> 4, tps = (k >> 4) / S;
    const int nb0 = blockIdx.x * 8, nb = nb0 + warp;
    const int kb0 = sp * tps, rows = tps * 16;
    __shared__ float xs[MUL1_XROWS];
    __shared__ __align__(16) unsigned int tw[MUL1_G2_WORDS];
    const int run4 = 2 * n32, tot4 = tps * run4;
    uint4 rg[MUL1_G2_WORDS / 4 / 256];
    if ((tbase & 15ull) == 0) {
        #pragma unroll
        for (int q = 0; q < MUL1_G2_WORDS / 4 / 256; q++) {
            const int i = threadIdx.x + q * 256;
            if (i < tot4) {
                const int u = i / run4, c = i - u * run4;
                rg[q] = ((const uint4*)tbase)[((((size_t)(kb0 + u) * tiles_n + nb0) * n32) >> 2) + c];
            }
        }
    } else {
        #pragma unroll
        for (int q = 0; q < MUL1_G2_WORDS / 4 / 256; q++) {
            const int i = threadIdx.x + q * 256;
            if (i < tot4) {
                const int u = i / run4, c = i - u * run4;
                const unsigned int* s = (const unsigned int*)tbase + ((size_t)(kb0 + u) * tiles_n + nb0) * n32 + 4 * c;
                rg[q] = make_uint4(s[0], s[1], s[2], s[3]);
            }
        }
    }
    for (int i = threadIdx.x; i < rows; i += blockDim.x) xs[i] = xh[(size_t)e * k + (size_t)kb0 * 16 + i];
    #pragma unroll
    for (int q = 0; q < MUL1_G2_WORDS / 4 / 256; q++) {
        const int i = threadIdx.x + q * 256;
        if (i < tot4) ((uint4*)tw)[i] = rg[q];
    }
    int lo[8], wi[8], wi1[8], wo[8];
    mul1_lane_lo(lo, lane, bits, half, n32);
    #pragma unroll
    for (int j = 0; j < 8; j++) {
        wi[j] = lo[j] >> 5;
        wo[j] = lo[j] & 31;
        wi1[j] = (wi[j] + 1 == n32) ? 0 : wi[j] + 1;
    }
    __syncthreads();
    const int rbase = 2 * (lane & 3);
    float a = 0.0f, b = 0.0f;
    for (int u = 0; u < tps; u++) {
        const unsigned int* w = tw + u * 8 * n32 + warp * n32;
        float wv[8];
        #pragma unroll
        for (int j = 0; j < 8; j += 2) mul1_w2(mul1_state(w, wi[j], wi1[j], wo[j]), mul1_state(w, wi[j + 1], wi1[j + 1], wo[j + 1]), wv[j], wv[j + 1]);
        const float* xr = xs + u * 16 + rbase;
        float x0 = xr[0], x1 = xr[1], x8 = xr[8], x9 = xr[9];
        a = fmaf(wv[0], x0, a);
        a = fmaf(wv[1], x1, a);
        a = fmaf(wv[2], x8, a);
        a = fmaf(wv[3], x9, a);
        b = fmaf(wv[4], x0, b);
        b = fmaf(wv[5], x1, b);
        b = fmaf(wv[6], x8, b);
        b = fmaf(wv[7], x9, b);
    }
    float acc0[MUL1_MAXT], acc1[MUL1_MAXT];
    acc0[0] = a;
    acc1[0] = b;
    mul1_store_part(part, acc0, acc1, e, S, sp, 1, n, nb, lane);
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

// ---------------- prefill: the expert-major grouped GEMM (GLM-5.3-Flash prompt calls) ----------------
// One launch runs every (expert, row tile) work item of a prompt call's sub-batch: the expert's
// trellis is decoded once per tile of up to MUL1_GT rows routed to it, instead of once per
// (row, pick) combo as the T = 1 slots of GemvPlan do. Each block is one work item x one 128-wide
// output block and runs the three steps of the GemvPlan path in-block, per row in the same f32
// order, so every output row has the bits of the T = 1 slot path:
//   had_in   the k-split's 128-blocks of H (x * suh) staged straight into shared memory (the
//            mul1_had_in arithmetic), x read through the row list (no gathered copy)
//   gemv     the k-splits one after the other, each the tile loop of mul1_gemv (the fmaf chains
//            of mul1_tile_fma from 0, the two xor-shuffle adds of mul1_store_part), the split's
//            partial added to a running sum left to right: s = p_0, s = s + p_1, ... (the order
//            mul1_out_block sums the partials in)
//   had_out  the 128 sums of a row through mul1_fwht128, * 2^-7, * svh (mul1_out_block)
// No [E][S][T][n] partial buffer exists: the scratch of a prompt call is its rows only.
// mode 1 reads the input as glm5_swiglu_clamp(g, u) = silu(min(g, L)) * clamp(u, -L, L) (the
// kernels_glm5_moe.cu formula and operation order), so the down GEMM needs no h buffer.
// blockIdx.z picks one of two matrices of the record (gate and up share k, n and the input).
//
// grid (n / 128, W, Z), block 256. work [W][3] i32: expert id, first entry in list, rows (1..GT).
// list: combo indices c, grouped by expert; the input row of entry i is list[i] / p[6] (row
// stride k), its output row list[i] (row stride n). table [E] u64: record bases (VRAM, pinned
// UVA or staging). p: [0] k, [1] n, [2] S, [3] n32, [4] bits, [5] half, [6] in_div, [7] mode,
// [8] limit (f32 bits), [9 + 3 z ..] tr_off, suh_off, svh_off of matrix z, [15] loc. k / S <= MUL1_GXROWS.
// #196 loc = 1: the gate / up activations live at the entry's position in this launch's piece of
// the list (entry i -> row i - work[1], work[1] = the launch's first entry), so they hold the
// combos of one piece only: mode 0 writes its output row there, mode 1 reads its input row there
// (the output row stays list[i]). Positions only: every row's arithmetic is unchanged.
#define MUL1_GT 16
#define MUL1_GXROWS 256

__device__ __forceinline__ float mul1_clamp_swiglu(float gv, float uv, float L) {
    gv = gv > L ? L : gv;
    uv = uv > L ? L : (uv < -L ? -L : uv);
    const float silu = __fdiv_rn(gv, __fadd_rn(1.0f, expf(-gv)));
    return __fmul_rn(silu, uv);
}

// mul1_tile_fma for up to MUL1_GT rows (the same operations per row)
__device__ __forceinline__ void mul1_tile_fma_g(const unsigned int* w, const int lo[8], int n32, const float* xs, int rows,
                                                int r0, int T, float acc0[MUL1_GT], float acc1[MUL1_GT]) {
    float wv[8];
    #pragma unroll
    for (int j = 0; j < 8; j++) {
        int i = lo[j] >> 5, o = lo[j] & 31;
        int i1 = (i + 1 == n32) ? 0 : i + 1;
        unsigned long long pair = ((unsigned long long)w[i] << 32) | w[i1];
        wv[j] = mul1_w((unsigned int)(pair >> (48 - o)) & 0xffffu);
    }
    #pragma unroll
    for (int t = 0; t < MUL1_GT; t++) {
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

extern "C" __global__ void __launch_bounds__(256) mul1_gemm_grp(const unsigned long long* __restrict__ table, const int* __restrict__ work,
                                                                const int* __restrict__ list, const float* __restrict__ xa,
                                                                const float* __restrict__ xb, float* __restrict__ y0,
                                                                float* __restrict__ y1, const int* __restrict__ p) {
    const int k = p[0], n = p[1], S = p[2], n32 = p[3], bits = p[4], half = p[5], in_div = p[6], mode = p[7];
    const float L = __int_as_float(p[8]);
    const int z = blockIdx.z;
    const int tr_off = p[9 + 3 * z], suh_off = p[10 + 3 * z], svh_off = p[11 + 3 * z];
    float* __restrict__ y = z ? y1 : y0;
    const int wi = blockIdx.y;
    const int e = work[3 * wi], f = work[3 * wi + 1], T = work[3 * wi + 2];
    const unsigned long long base = table[e];
    const unsigned long long tbase = base + (unsigned long long)tr_off;
    const unsigned short* suh = (const unsigned short*)(base + (unsigned long long)suh_off);
    const unsigned short* svh = (const unsigned short*)(base + (unsigned long long)svh_off);
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int tiles_n = n >> 4, tps = (k >> 4) / S, rows = tps * 16, nblk = rows >> 7;
    const int nb0 = blockIdx.x * 8;
    __shared__ float xs[MUL1_GT * MUL1_GXROWS];
    __shared__ __align__(16) unsigned int ring[2][MUL1_U * 8 * MUL1_N32MAX];
    __shared__ int in_row[MUL1_GT], out_row[MUL1_GT];
    if (threadIdx.x < MUL1_GT) {
        const int c = (int)threadIdx.x < T ? list[f + threadIdx.x] : 0;
        const int lp = (int)threadIdx.x < T ? f + (int)threadIdx.x - work[1] : 0;
        const bool loc = p[15] != 0;
        in_row[threadIdx.x] = (loc && mode) ? lp : c / in_div;
        out_row[threadIdx.x] = (loc && !mode) ? lp : c;
    }
    const int run4 = 2 * n32;
    const int stage4 = MUL1_U * run4;
    const bool vec = (tbase & 15ull) == 0;
    uint4 rg[MUL1_PIECES];
    int lo[8];
    mul1_lane_lo(lo, lane, bits, half, n32);
    const int rbase = 2 * (lane & 3);
    float s0[MUL1_GT], s1[MUL1_GT];
    #pragma unroll
    for (int t = 0; t < MUL1_GT; t++) {
        s0[t] = 0.0f;
        s1[t] = 0.0f;
    }
    for (int sp = 0; sp < S; sp++) {
        const int kb0 = sp * tps;
        // every read of the previous split's xs and ring is done (its loop ended in a barrier);
        // the first split waits for in_row
        __syncthreads();
        for (int task = warp; task < T * nblk; task += 8) {
            const int t = task / nblk, b = task - t * nblk;
            const int c0 = kb0 * 16 + b * 128 + 4 * lane;
            const size_t ro = (size_t)in_row[t] * k + c0;
            float v[4];
            #pragma unroll
            for (int j = 0; j < 4; j++) v[j] = mode ? mul1_clamp_swiglu(xa[ro + j], xb[ro + j], L) : xa[ro + j];
            mul1_in_block(v, suh, c0, lane);
            #pragma unroll
            for (int j = 0; j < 4; j++) xs[t * rows + b * 128 + 4 * lane + j] = v[j];
        }
        float acc0[MUL1_GT], acc1[MUL1_GT];
        #pragma unroll
        for (int t = 0; t < MUL1_GT; t++) {
            acc0[t] = 0.0f;
            acc1[t] = 0.0f;
        }
        for (int kt = 0, buf = 0; kt < tps + MUL1_U; kt += MUL1_U, buf ^= 1) {
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
            if (kt > 0) {
                const int kp = kt - MUL1_U;
                #pragma unroll
                for (int u = 0; u < MUL1_U; u++) {
                    if (kp + u < tps)
                        mul1_tile_fma_g(ring[buf ^ 1] + u * (8 * MUL1_N32MAX) + warp * n32, lo, n32, xs, rows, (kp + u) * 16 + rbase, T, acc0, acc1);
                }
            }
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
        // mul1_store_part's reduction, then the left-to-right sum of the partials
        #pragma unroll
        for (int t = 0; t < MUL1_GT; t++) {
            if (t < T) {
                float a = acc0[t], b = acc1[t];
                a = __fadd_rn(a, __shfl_xor_sync(0xffffffffu, a, 1));
                b = __fadd_rn(b, __shfl_xor_sync(0xffffffffu, b, 1));
                a = __fadd_rn(a, __shfl_xor_sync(0xffffffffu, a, 2));
                b = __fadd_rn(b, __shfl_xor_sync(0xffffffffu, b, 2));
                s0[t] = sp == 0 ? a : __fadd_rn(s0[t], a);
                s1[t] = sp == 0 ? b : __fadd_rn(s1[t], b);
            }
        }
    }
    // the sums of this block's 128 outputs into shared memory (xs is free: the last split ended
    // in a barrier), then mul1_out_block's FWHT and scaling, one warp per row
    if ((lane & 3) == 0) {
        #pragma unroll
        for (int t = 0; t < MUL1_GT; t++) {
            if (t < T) {
                xs[t * 128 + warp * 16 + (lane >> 2)] = s0[t];
                xs[t * 128 + warp * 16 + (lane >> 2) + 8] = s1[t];
            }
        }
    }
    __syncthreads();
    const int c0 = blockIdx.x * 128 + 4 * lane;
    for (int t = warp; t < T; t += 8) {
        float v[4];
        #pragma unroll
        for (int j = 0; j < 4; j++) v[j] = xs[t * 128 + 4 * lane + j];
        mul1_fwht128(v, lane);
        const size_t o = (size_t)out_row[t] * n + c0;
        #pragma unroll
        for (int j = 0; j < 4; j++) y[o + j] = __fmul_rn(__fmul_rn(v[j], 0.0078125f), mul1_h2f(svh[c0 + j]));
    }
}
