// #162 (GLM-5.3-Flash plan step 13b): the three KDA (Kimi delta attention) kernels the GDN set of
// KERNEL_SRC does not have. Its own NVRTC module (`kernels::glm5_kda::Kernels`), compiled alone:
// one more `.entry` in KERNEL_SRC would break the PTX of record (`tests_300_c4`). The rest of a KDA
// layer (q|k|v GEMM/GEMV, causal conv + state, split, l2norm, gated RMSNorm) runs on the KERNEL_SRC
// kernels compiled with the KDA geometry (`glm5_kda::kernel_geo`). docs/glm5-kda.md.
//
// Every scalar argument is a device pointer (the p5 rule). Head count = gridDim.x of the launch;
// the head dim is fixed at KDA_D (GLM-5.3-Flash `linear_attn_config.head_dim` 128, asserted by the
// host). Layouts: q, k [T][heads][KDA_D] (l2-normalised, q scaled), v and out [T][heads][KDA_D],
// g [T][heads][KDA_D] (log-decay per key channel), beta [T][heads], state S [heads][key][value].
#define KDA_D 128

// forget gate and beta (modeling_glm5_next.py Glm5NextTextForgetGate.forward, `linear_lower_bound` path,
// and `beta = sigmoid(b_proj x)`):
//   g[t][h][c] = lb * sigmoid(exp(A_log[h]) * (f[t][h][c] + dt_bias[h][c])),   lb = gate_lower_bound (-5)
//   beta[t][h] = sigmoid(b[t][h])
// f = f_b(f_a(x)) [T][heads*KDA_D], b = b_proj(x) [T][heads]. grid ceil(T*heads*KDA_D / 256), block 256.
extern "C" __global__ void kda_gate(const float* __restrict__ f, const float* __restrict__ dt_bias,
                                    const float* __restrict__ a_log, const float* __restrict__ b,
                                    float* __restrict__ g_out, float* __restrict__ beta_out,
                                    const int* __restrict__ t_p, const int* __restrict__ heads_p,
                                    const float* __restrict__ lb_p) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int heads = *heads_p;
    int tt = *t_p;
    int w = heads * KDA_D;
    if (i < tt * heads) beta_out[i] = 1.0f / (1.0f + expf(-b[i]));
    if (i >= tt * w) return;
    int c = i % w;
    float a = expf(a_log[c / KDA_D]) * (f[i] + dt_bias[c]);
    g_out[i] = *lb_p * (1.0f / (1.0f + expf(-a)));
}

// The KDA recurrence, register-resident (the delta_rule_persist_r form of KERNEL_SRC, same operation
// order and the same intrinsics so nvcc cannot contract `s*g + k*delta`, kernels.rs "Register-resident
// delta rule"), with ONE change: the decay is per key channel, S[dk][:] *= exp(g[dk]), where GDN
// multiplies the whole head by exp(g[head]). Per token (recurrent_kimi_delta_attention):
//   S <- diag(exp g_t) S;  kv = S^T k_t;  delta = beta_t (v_t - kv);  S <- S + k_t delta^T;  o_t = S^T q_t
// grid (heads), block KDA_D (thread d owns value column d). init_p = 1 starts from S = 0.
extern "C" __global__ void kda_persist_r(const float* __restrict__ q, const float* __restrict__ k,
                                         const float* __restrict__ v, const float* __restrict__ g,
                                         const float* __restrict__ beta, float* __restrict__ out,
                                         float* __restrict__ s_global, const int* __restrict__ steps_p,
                                         const int* __restrict__ init_p) {
    int steps = *steps_p;
    int head = blockIdx.x;
    int heads = gridDim.x;
    float* S = s_global + (size_t)head * KDA_D * KDA_D;
    int d = threadIdx.x;
    __shared__ float ks[KDA_D];
    __shared__ float qs[KDA_D];
    __shared__ float gs[KDA_D];
    float sr[KDA_D];
    if (*init_p) {
#pragma unroll
        for (int dk = 0; dk < KDA_D; dk++) sr[dk] = 0.0f;
    } else {
#pragma unroll
        for (int dk = 0; dk < KDA_D; dk++) sr[dk] = S[dk * KDA_D + d];
    }
    for (int t = 0; t < steps; t++) {
        size_t row = ((size_t)t * heads + head) * KDA_D;
        float beta_t = beta[t * heads + head];
        ks[d] = k[row + d];
        qs[d] = q[row + d];
        gs[d] = expf(g[row + d]);
        __syncthreads();
        float kv = 0.0f;
#pragma unroll
        for (int dk = 0; dk < KDA_D; dk++) { sr[dk] = __fmul_rn(sr[dk], gs[dk]); kv = __fmaf_rn(sr[dk], ks[dk], kv); }
        float delta = (v[row + d] - kv) * beta_t;
        float o = 0.0f;
#pragma unroll
        for (int dk = 0; dk < KDA_D; dk++) { sr[dk] = __fmaf_rn(ks[dk], delta, sr[dk]); o = __fmaf_rn(sr[dk], qs[dk], o); }
        out[row + d] = o;
        __syncthreads();
    }
#pragma unroll
    for (int dk = 0; dk < KDA_D; dk++) S[dk * KDA_D + d] = sr[dk];
}

// one decode token: the kda_persist_r body for steps = 1, init 0 (bit-identical to it)
extern "C" __global__ void kda_step_r(float* __restrict__ s_global, const float* __restrict__ q,
                                      const float* __restrict__ k, const float* __restrict__ v,
                                      const float* __restrict__ g, const float* __restrict__ beta,
                                      float* __restrict__ out) {
    int head = blockIdx.x;
    float* S = s_global + (size_t)head * KDA_D * KDA_D;
    int d = threadIdx.x;
    size_t row = (size_t)head * KDA_D;
    __shared__ float ks[KDA_D];
    __shared__ float qs[KDA_D];
    __shared__ float gs[KDA_D];
    float beta_t = beta[head];
    ks[d] = k[row + d];
    qs[d] = q[row + d];
    gs[d] = expf(g[row + d]);
    __syncthreads();
    float sr[KDA_D];
#pragma unroll
    for (int dk = 0; dk < KDA_D; dk++) sr[dk] = S[dk * KDA_D + d];
    float kv = 0.0f;
#pragma unroll
    for (int dk = 0; dk < KDA_D; dk++) { sr[dk] = __fmul_rn(sr[dk], gs[dk]); kv = __fmaf_rn(sr[dk], ks[dk], kv); }
    float delta = (v[row + d] - kv) * beta_t;
    float o = 0.0f;
#pragma unroll
    for (int dk = 0; dk < KDA_D; dk++) { sr[dk] = __fmaf_rn(ks[dk], delta, sr[dk]); o = __fmaf_rn(sr[dk], qs[dk], o); }
#pragma unroll
    for (int dk = 0; dk < KDA_D; dk++) S[dk * KDA_D + d] = sr[dk];
    out[row + d] = o;
}

// ---------------- #202 A (CROW_GLM_ATTN_FUSE): the KDA decode row in 6 launches instead of 12 ----------------
// `glm5_kda::step_fused_with`. Each kernel below carries the statements of the kernels it replaces,
// expression for expression (the KERNEL_SRC kernels compiled with the KDA geometry: CN_GD = CN_GDV =
// KDA_D, key heads = value heads = heads, conv kernel 4, sigmoid gate), so every value keeps its
// bits; only where an operand comes from changes (a register or shared memory instead of the global
// copy the next launch read back). Scalars: one device int buffer fp (the p5 rule):
// fp[0] hidden, [1] heads, [2] width = heads * KDA_D, [3] eps (f32 bits), [4] lower bound (f32 bits).

// gemv_bf16_w (KERNEL_SRC) for one output row by one warp (row 0 of x): the lane sums and the
// shuffle tree; lane 0 holds the row's value
__device__ __forceinline__ float kda_gemv_row(const unsigned short* __restrict__ w, const float* __restrict__ x, int k_dim, int row, int lane) {
    const unsigned short* wp = w + (size_t)row * k_dim;
    const float* xp = x;
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
    return acc;
}

// conv_step's statements for channel ch (the short conv over the window, silu, the window shift)
__device__ __forceinline__ float kda_conv_ch(const float* __restrict__ mq1, const float* __restrict__ w, float* __restrict__ cs,
                                             float* __restrict__ cout, int ch) {
    const float* wv = w + ch * 4;
    float acc = wv[0] * cs[ch * 3 + 0] + wv[1] * cs[ch * 3 + 1]
              + wv[2] * cs[ch * 3 + 2] + wv[3] * mq1[ch];
    const float o = acc / (1.0f + expf(-acc));
    cout[ch] = o;
    cs[ch * 3 + 0] = cs[ch * 3 + 1];
    cs[ch * 3 + 1] = cs[ch * 3 + 2];
    cs[ch * 3 + 2] = mq1[ch];
    return o;
}

// conv_step + l2norm_repeat of one decode row. grid (heads), block KDA_D. Thread d of head h runs
// conv_step for the q, k and v channels h * KDA_D + d (+ W, + 2 W), then l2norm_repeat's
// statements for q and k of head h over the conv outputs (from shared memory).
// mq1 [3W] (the q|k|v projection), w [3W][4], cs [3W][3] (window), cout [3W], q_out / k_out [heads][KDA_D].
extern "C" __global__ void kda_conv_l2(const float* __restrict__ mq1, const float* __restrict__ w, float* __restrict__ cs,
                                       float* __restrict__ cout, float* __restrict__ q_out, float* __restrict__ k_out,
                                       const int* __restrict__ fp) {
    const int W = fp[2];
    const int head = blockIdx.x, d = threadIdx.x;
    const int c = head * KDA_D + d;
    __shared__ float qs[KDA_D];
    __shared__ float ks[KDA_D];
    qs[d] = kda_conv_ch(mq1, w, cs, cout, c);
    ks[d] = kda_conv_ch(mq1, w, cs, cout, W + c);
    kda_conv_ch(mq1, w, cs, cout, 2 * W + c);
    __syncthreads();
    const float* qp = qs;
    const float* kp = ks;
    float qn = 0.0f, kn = 0.0f;
    for (int i = 0; i < KDA_D; i++) { qn += qp[i] * qp[i]; kn += kp[i] * kp[i]; }
    qn = rsqrtf(qn + 1e-6f); kn = rsqrtf(kn + 1e-6f);
    q_out[(size_t)head * KDA_D + d] = qp[d] * qn * rsqrtf((float)KDA_D);
    k_out[(size_t)head * KDA_D + d] = kp[d] * kn;
}

// the three hidden -> small projections of one decode row (f_a [KDA_D][hidden], b [heads][hidden],
// g_a [KDA_D][hidden], gemv_bf16_w each) in one launch: grid (ceil(max(KDA_D, heads) / 8), 3),
// block 256; blockIdx.y = 0 f_a -> fa, 1 b -> b, 2 g_a -> ga; warp = output row. x [hidden].
extern "C" __global__ void kda_proj_a3(const unsigned short* __restrict__ w_fa, const unsigned short* __restrict__ w_b,
                                       const unsigned short* __restrict__ w_ga, const float* __restrict__ x,
                                       float* __restrict__ fa, float* __restrict__ b, float* __restrict__ ga,
                                       const int* __restrict__ fp) {
    const int k_dim = fp[0], heads = fp[1];
    const int m = blockIdx.y;
    const int rows = m == 1 ? heads : KDA_D;
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int row = blockIdx.x * 8 + warp;
    if (row >= rows) return;
    const float acc = kda_gemv_row(m == 0 ? w_fa : m == 1 ? w_b : w_ga, x, k_dim, row, lane);
    if (lane == 0) (m == 0 ? fa : m == 1 ? b : ga)[row] = acc;
}

// the two small -> width projections (f_b [W][KDA_D] on fa, g_b [W][KDA_D] on ga, gemv_bf16_w)
// with kda_gate folded into the f_b rows: grid (W / 8, 2), block 256; blockIdx.y = 0 f_b -> f and
// the forget gate g (kda_gate's expression on the row's value), 1 g_b -> gate. Block (0, 0) also
// writes beta = sigmoid(b) of the heads (kda_gate's beta). dt_bias [W], a_log [heads].
extern "C" __global__ void kda_proj_b2_gate(const unsigned short* __restrict__ w_fb, const unsigned short* __restrict__ w_gb,
                                            const float* __restrict__ fa, const float* __restrict__ ga,
                                            const float* __restrict__ b, const float* __restrict__ dt_bias,
                                            const float* __restrict__ a_log, float* __restrict__ f, float* __restrict__ g_out,
                                            float* __restrict__ beta_out, float* __restrict__ gate, const int* __restrict__ fp) {
    const int heads = fp[1], W = fp[2];
    const float lb = __int_as_float(fp[4]);
    const int m = blockIdx.y;
    const int i = threadIdx.x;
    if (m == 0 && blockIdx.x == 0 && i < heads) beta_out[i] = 1.0f / (1.0f + expf(-b[i]));
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int row = blockIdx.x * 8 + warp;
    if (row >= W) return;
    const float acc = kda_gemv_row(m == 0 ? w_fb : w_gb, m == 0 ? fa : ga, KDA_D, row, lane);
    if (lane != 0) return;
    if (m == 1) {
        gate[row] = acc;
        return;
    }
    f[row] = acc;
    const int c = row;
    float a = expf(a_log[c / KDA_D]) * (acc + dt_bias[c]);
    g_out[row] = lb * (1.0f / (1.0f + expf(-a)));
}

// kda_step_r + rmsnorm_gated (sigmoid gate) of one decode row: grid (heads), block KDA_D.
// kda_step_r's statements, then rmsnorm_gated's for head `head` over the outputs (its tree sum over
// KDA_D values, eps fp[3]). z: the g_b output [heads][KDA_D]; nw: the o_norm weight [KDA_D]; out
// (kda_step_r's output) is stored too.
extern "C" __global__ void kda_step_norm(float* __restrict__ s_global, const float* __restrict__ q,
                                         const float* __restrict__ k, const float* __restrict__ v,
                                         const float* __restrict__ g, const float* __restrict__ beta,
                                         float* __restrict__ out, const float* __restrict__ z, const float* __restrict__ nw,
                                         float* __restrict__ normed, const int* __restrict__ fp) {
    int head = blockIdx.x;
    float* S = s_global + (size_t)head * KDA_D * KDA_D;
    int d = threadIdx.x;
    size_t row = (size_t)head * KDA_D;
    __shared__ float ks[KDA_D];
    __shared__ float qs[KDA_D];
    __shared__ float gs[KDA_D];
    float beta_t = beta[head];
    ks[d] = k[row + d];
    qs[d] = q[row + d];
    gs[d] = expf(g[row + d]);
    __syncthreads();
    float sr[KDA_D];
#pragma unroll
    for (int dk = 0; dk < KDA_D; dk++) sr[dk] = S[dk * KDA_D + d];
    float kv = 0.0f;
#pragma unroll
    for (int dk = 0; dk < KDA_D; dk++) { sr[dk] = __fmul_rn(sr[dk], gs[dk]); kv = __fmaf_rn(sr[dk], ks[dk], kv); }
    float delta = (v[row + d] - kv) * beta_t;
    float o = 0.0f;
#pragma unroll
    for (int dk = 0; dk < KDA_D; dk++) { sr[dk] = __fmaf_rn(ks[dk], delta, sr[dk]); o = __fmaf_rn(sr[dk], qs[dk], o); }
#pragma unroll
    for (int dk = 0; dk < KDA_D; dk++) S[dk * KDA_D + d] = sr[dk];
    out[row + d] = o;
    // rmsnorm_gated of head `head` (t = 0)
    const float eps = __int_as_float(fp[3]);
    __shared__ float red[KDA_D];
    red[d] = o * o;
    __syncthreads();
    for (int st = KDA_D / 2; st > 0; st >>= 1) {
        if (d < st) red[d] += red[d + st];
        __syncthreads();
    }
    float rms = rsqrtf(red[0] / (float)KDA_D + eps);
    float gt = 1.0f / (1.0f + expf(-z[row + d]));
    normed[row + d] = nw[d] * o * rms * gt;
}
