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
