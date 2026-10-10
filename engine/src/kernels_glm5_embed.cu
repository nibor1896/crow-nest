// #186 CROW_GLM_EMBED_GATHER: the trunk input of a prompt call gathered and expanded on the device
// (glm5_embed.rs). One block per prompt row: the row's BF16 embedding (the `slot[r]`-th of the
// uploaded distinct rows) widened to f32 by the bit shift `bf16 << 16` (cnq::bf16_bytes_to_f32,
// no arithmetic) and written into every residual stream (glm5_model::trunk_input). The stores are
// u32 bit patterns, so NaN payloads, -0 and denormals land exactly as the host path writes them.
extern "C" __global__ void glm5_embed_expand(const unsigned short* __restrict__ rows, const int* __restrict__ slot,
                                             unsigned int* __restrict__ x, unsigned long long hidden,
                                             unsigned long long streams)
{
    const unsigned long long r = blockIdx.x;
    const unsigned short* src = rows + (unsigned long long)slot[r] * hidden;
    unsigned int* dst = x + r * streams * hidden;
    for (unsigned long long j = threadIdx.x; j < hidden; j += blockDim.x) {
        const unsigned int v = ((unsigned int)src[j]) << 16;
        for (unsigned long long s = 0; s < streams; ++s) {
            dst[s * hidden + j] = v;
        }
    }
}
