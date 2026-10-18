// crow-nest #241 stage B: the pre-scaled fused producers (NVRTC, compute_120a). Its own
// module (`kernels::qp::source` compiles it alone): one more `.entry` in KERNEL_SRC would
// break the PTX of record (`kernels::tests_300_c4`), the MUL1_SRC precedent.
//
// What this module does NOT carry: the quantizer. `kernels::qp::source` puts the C4 prelude
// (the `CN_*` geometry) and three spans of KERNEL_SRC in front of this text, cut out of it
// VERBATIM: `e2m1` / `ue4m3`, the NVFP4 activation-cascade helpers (`q_e2m1`, `enc_ue4m3_up`,
// `XQ_ROW`, `xq_rs`, `xq_set_rs`, `row_prescale_k`, `quant_level`, `quant_row_prescaled`) and
// `mma_fp4_16n8k64`. So every cascade below is `quant_x_fp4`'s, byte for byte.
//
// Why: the fused producers of KERNEL_SRC (`mix_streams_q`, `rmsnorm_gated_q`, `gate_mul_q`,
// `silu_mul_combo_q`, `sh_gate_up_q`) split each row over several blocks, so none of them sees
// the row amax that the per-row pre-scale needs; they store the factor 1.0 and keep the ue4m3
// floor (CROW_QFUSE=1). The kernels here OWN their rows (decode, t < 8): each block computes
// the producer's f32 values for whole rows in the producer's own expression order, writes
// them to global memory and to shared memory, and runs `quant_row_prescaled` over the shared
// row. `sh_gate_up_qp` keeps the 10-block split of the GEMV and lets the LAST block of each
// token row quantize it (the memory-fence pattern of the CUDA Programming Guide, C++ Language
// Extensions, Memory Fence Functions: store, device fence, `atomicInc` count, the last block
// reads the others' values and resets the counter for the next launch, i.e. the next replay
// of the decode graph). The f32 outputs and the cascades are the stage-A sequence's (the
// plain producer, then `quant_x_fp4`) bit for bit: `kernels::tests_241_qp_gpu`.
//
// Launch shapes (gen.rs, decode t < 8 only; every kernel strides over its row with blockDim,
// which must be a multiple of 32 and at most 1024):
//   mix_streams_qp     grid (T),          block qp::BX_ROW
//   rmsnorm_gated_qp   grid (T),          block qp::BX_NORM (a multiple of CN_GDV)
//   gate_mul_qp        grid (T),          block qp::BX_ROW
//   silu_mul_combo_qp  grid (T * TOPK),   block qp::BX_COMBO
//   sh_gate_up_qp      grid (INTER/64, T), block mma_bx() (the sh_gate_up_q shape)

// hc mix: out[t][c] = mean_g mixw * normed (mix_streams), plus its pre-scaled cascade
extern "C" __global__ void mix_streams_qp(const float* __restrict__ mixw, const float* __restrict__ normed,
                                          float* __restrict__ out, unsigned char* __restrict__ xq) {
    int t = blockIdx.x;
    __shared__ float row[CN_H];
    for (int c = threadIdx.x; c < CN_H; c += blockDim.x) {
        float acc = 0.0f;
        for (int g = 0; g < CN_HCN; g++)
            acc += mixw[t * CN_HCT + g * CN_H + c] * normed[t * CN_HCT + g * CN_H + c];
        float v = acc * (1.0f / CN_HCN);
        out[t * CN_H + c] = v;
        row[c] = v;
    }
    __syncthreads();
    quant_row_prescaled(row, xq + (size_t)t * XQ_ROW(CN_H / 64), CN_H / 64);
}

// GDN gated RMSNorm (rmsnorm_gated) over the whole [VHEADS][GDV] row of token t: the block
// is blockDim / CN_GDV groups of CN_GDV threads, group `grp` normalizes the value heads
// grp, grp + groups, ...; each head keeps rmsnorm_gated's own shared tree (the same pairs in
// the same order) in its group's slice of `red`
extern "C" __global__ void rmsnorm_gated_qp(const float* __restrict__ x, const float* __restrict__ z,
                                            const float* __restrict__ w, float* __restrict__ out,
                                            unsigned char* __restrict__ xq) {
    int t = blockIdx.x;
    int groups = blockDim.x / CN_GDV;
    int grp = threadIdx.x / CN_GDV, d = threadIdx.x % CN_GDV;
    __shared__ float red[1024];
    __shared__ float row[CN_GDN_VAL];
    float* r = red + grp * CN_GDV;
    for (int h0 = 0; h0 < CN_GDN_VHEADS; h0 += groups) {
        int vhead = h0 + grp;
        bool act = grp < groups && vhead < CN_GDN_VHEADS;   // uniform per group; every thread meets every barrier
        const float* xp = x + ((size_t)t * CN_GDN_VHEADS + vhead) * CN_GDV;
        if (act) r[d] = xp[d] * xp[d];
        __syncthreads();
        for (int st = CN_GDV / 2; st > 0; st >>= 1) {
            if (act && d < st) r[d] += r[d + st];
            __syncthreads();
        }
        if (act) {
            float rms = rsqrtf(r[0] / (float)CN_GDV + CN_EPS);
#if CN_GATE_ACT == 0 // sigmoid (Flash-Next `output_gate_type` sigmoid)
            float gate = 1.0f / (1.0f + expf(-z[((size_t)t * CN_GDN_VHEADS + vhead) * CN_GDV + d]));
#else // swish = x * sigmoid(x) (the dense Qwen3.5 family: `swish` / `silu`)
            float zg = z[((size_t)t * CN_GDN_VHEADS + vhead) * CN_GDV + d];
            float gate = zg / (1.0f + expf(-zg));
#endif
            float v = w[d] * xp[d] * rms * gate;
            out[((size_t)t * CN_GDN_VHEADS + vhead) * CN_GDV + d] = v;
            row[vhead * CN_GDV + d] = v;
        }
        __syncthreads(); // r is rewritten by the next round
    }
    quant_row_prescaled(row, xq + (size_t)t * XQ_ROW(CN_GDN_VAL / 64), CN_GDN_VAL / 64);
}

// attention output gate (gate_mul): out = core * sigmoid(gate) over the [CORE] row of token t
extern "C" __global__ void gate_mul_qp(const float* __restrict__ core, const float* __restrict__ gate,
                                       float* __restrict__ out, unsigned char* __restrict__ xq) {
    int t = blockIdx.x;
    __shared__ float row[CN_CORE];
    for (int e = threadIdx.x; e < CN_CORE; e += blockDim.x) {
        int i = t * CN_CORE + e;
        float g = gate[i];
        float v = core[i] / (1.0f + expf(-g));
        out[i] = v;
        row[e] = v;
    }
    __syncthreads();
    quant_row_prescaled(row, xq + (size_t)t * XQ_ROW(CN_CORE / 64), CN_CORE / 64);
}

// routed SwiGLU (silu_mul_combo): h2[c] = silu(gate) * up over combo row c of [C][2*INTER]
extern "C" __global__ void silu_mul_combo_qp(const float* __restrict__ h1, float* __restrict__ h2,
                                             unsigned char* __restrict__ xq) {
    int c = blockIdx.x;
    __shared__ float row[CN_INTER];
    for (int j = threadIdx.x; j < CN_INTER; j += blockDim.x) {
        float gate = h1[(size_t)c * (2 * CN_INTER) + j];
        float v = (gate / (1.0f + expf(-gate))) * h1[(size_t)c * (2 * CN_INTER) + CN_INTER + j];
        h2[(size_t)c * CN_INTER + j] = v;
        row[j] = v;
    }
    __syncthreads();
    quant_row_prescaled(row, xq + (size_t)c * XQ_ROW(CN_INTER / 64), CN_INTER / 64);
}

// The shared-expert gate|up of decode (sh_gate_up_q) with the pre-scaled cascade: the
// sh_gate_up_q body VERBATIM up to the h2 store (both gemv_fp4_mma_d passes, the ks-split
// reduce, the SwiGLU epilogue on the finished pairs), minus its unscaled qf_store. Then the
// last-block-done step per token row: every block fences its h2 stores (device scope) and
// counts itself in cnt[tok]; the block that sees gridDim.x - 1 is the last, reads the whole
// 640-wide row back (L2, `__ldcg`), quantizes it with quant_row_prescaled and resets
// cnt[tok] to 0, so the counters are 0 again before the next launch (graph replay).
// cnt: one unsigned int per token row (t < 8), zeroed once at allocation.
extern "C" __global__ void sh_gate_up_qp(const unsigned char* __restrict__ wg,
                                         const unsigned char* __restrict__ wu,
                                         const unsigned char* __restrict__ xq,
                                         const float* __restrict__ gsg_ptr,
                                         const float* __restrict__ gsu_ptr,
                                         float* __restrict__ h2,
                                         unsigned char* __restrict__ xq_s,
                                         const int* __restrict__ k_dim_p,
                                         const int* __restrict__ rows_p,
                                         unsigned int* __restrict__ cnt) {
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
            const unsigned char* xa = xq + (size_t)tok * XQ_ROW(bpr);
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
            float rs = xq_rs(xq, bpr, (size_t)tok); // the input row's pre-scale 2^-k (exact)
            if (pass == 0) { sg0 = (s0 * gs) * rs; sg2 = (s2 * gs) * rs; }
            else           { su0 = (s0 * gs) * rs; su2 = (s2 * gs) * rs; }
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
            h2[(size_t)tok * CN_INTER + j] = v;
        }
    }
    // the last-block-done step: this block's h2 stores are ordered before its count
    __shared__ bool last;
    __threadfence();
    __syncthreads();
    if (threadIdx.x == 0) {
        unsigned int seen = atomicInc(&cnt[tok], gridDim.x);
        last = (seen == gridDim.x - 1);
    }
    __syncthreads();
    if (last) {
        __shared__ float row[CN_INTER];
        __threadfence(); // the other blocks' h2 stores are read after their counts
        for (int i = threadIdx.x; i < CN_INTER; i += blockDim.x) row[i] = __ldcg(h2 + (size_t)tok * CN_INTER + i);
        __syncthreads();
        quant_row_prescaled(row, xq_s + (size_t)tok * XQ_ROW(CN_INTER / 64), CN_INTER / 64);
        if (threadIdx.x == 0) cnt[tok] = 0; // every block of this launch has counted: ready for the next one
    }
}
