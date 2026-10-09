// crow-nest #165 (plan step 13e): the glm5_next head before the lm_head (NVRTC, compute_120a).
// Its own module (`kernels::glm5_head::Kernels::new` compiles `GLM5_HEAD_SRC` alone), the
// `kernels_mul1.cu` pattern: one more `.entry` in KERNEL_SRC would break the PTX of record
// (`kernels::tests_300_c4`), and the kernel uses none of its helpers.
//
// GLM-5.3-Flash collapses its hc_mult residual streams by a plain mean and applies one RMSNorm
// (transformers 5.16.1 modeling_glm5_next.py: `self.norm(self.hc_head(hidden_states))` :1493,
// `Glm5NextTextHyperHead.forward` = `hidden_streams.mean(dim=2)` :298-302, "unlike DeepSeek-V4,
// an unweighted mean"; `Glm5NextTextRMSNorm` :66-80 = weight * (x * rsqrt(mean(x^2) + eps)),
// f32, NOT (1 + weight)). docs/glm5-next-recipe.md section 3 rows 4-5, docs/glm5-head.md.
//
// glm5_stream_mean_rms: grid (rows), block 256.
//   x   [rows][S][H] f32     the last decoder layer's output (S streams per token row)
//   w   [H] f32              model.language_model.norm.weight (BF16 in the checkpoint, widened)
//   out [rows][H] f32        the lm_head input
//   prm i32 {H, S, one_plus_w, eps as f32 bits}  (scalars in one device int buffer, the KERNEL_SRC rule)
//   m[d] = (x[0][d] + ... + x[S-1][d]) / S;  r = rsqrt(sum_d m[d]^2 / H + eps);  out[d] = w'[d] * (m[d] * r)
// with w' = 1 + w when one_plus_w (a zero-centred gamma), else w (GLM). The mean goes through
// `out` (each thread writes and re-reads only its own d), so no shared row buffer bounds H.

#define GLM5_HEAD_THREADS 256

extern "C" __global__ void glm5_stream_mean_rms(const float* __restrict__ x, const float* __restrict__ w,
                                                float* __restrict__ out, const int* __restrict__ prm) {
    const int H = prm[0], S = prm[1], one_plus_w = prm[2];
    const float eps = __int_as_float(prm[3]);
    const size_t row = blockIdx.x;
    const float* xr = x + row * (size_t)S * H;
    float* o = out + row * (size_t)H;
    float ss = 0.0f;
    for (int d = threadIdx.x; d < H; d += GLM5_HEAD_THREADS) {
        float m = 0.0f;
        for (int s = 0; s < S; s++) m += xr[(size_t)s * H + d];
        m = m / (float)S;  // HF's mean: sum, then one IEEE division (NVRTC default -prec-div=true)
        o[d] = m;
        ss += m * m;
    }
    __shared__ float red[GLM5_HEAD_THREADS];
    red[threadIdx.x] = ss;
    __syncthreads();
    for (int st = GLM5_HEAD_THREADS / 2; st > 0; st >>= 1) {
        if (threadIdx.x < st) red[threadIdx.x] += red[threadIdx.x + st];
        __syncthreads();
    }
    const float r = rsqrtf(red[0] / (float)H + eps);
    for (int d = threadIdx.x; d < H; d += GLM5_HEAD_THREADS) {
        const float g = one_plus_w ? 1.0f + w[d] : w[d];
        o[d] = g * (o[d] * r);
    }
}
