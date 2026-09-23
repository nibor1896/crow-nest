// layerdiff: decode ids[0..N-1] without callback, then the last id alone with a cb_eval
// that saves whitelisted per-layer tensors (1 row = the site position) + the logits row.
#include "arg.h"
#include "common.h"
#include "log.h"
#include "llama.h"
#include "ggml.h"
#include "ggml-backend.h"
#include <clocale>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <map>
#include <string>
#include <vector>

static bool g_on = false;
static std::string g_dir;
static std::map<std::string,int> g_cnt;
static FILE * g_idx = nullptr;
static const char * PFX[] = {"l_last-","hc_mixed-","hc_inject-","hc_combine-","hc_gate-","attn_output-","linear_attn_out-",
  "ffn_out-","ffn_moe_out-","ffn_shexp_gated-","ffn_shexp-","shared_expert_gate","ffn_moe_topk-","ffn_moe_weights_norm-",
  "ffn_moe_weights_scaled-","ffn_moe_logits-","ple_gate-","ple_gated_value-","ple_conv_out-","result_norm","result_output",
  "hc_init","attn_pregate-","attn_gated-","final_output-","indexer_top_k-","ple_embd","model.input_embed", "conv_state_at-1", "conv_input", nullptr};
static bool want(const char * n) {
  for (int i = 0; PFX[i]; ++i) if (strncmp(n, PFX[i], strlen(PFX[i])) == 0) return true;
  return false;
}
static bool cb(struct ggml_tensor * t, bool ask, void *) {
  if (ask) return g_on && want(t->name);
  if (!g_on || !want(t->name)) return true;
  if (!ggml_is_contiguous(t)) { fprintf(g_idx, "SKIP noncontig %s\n", t->name); return true; }
  size_t nb = ggml_nbytes(t);
  if (nb > (64u<<20)) { fprintf(g_idx, "SKIP big %s %zu\n", t->name, nb); return true; }
  std::vector<char> buf(nb);
  ggml_backend_tensor_get(t, buf.data(), 0, nb);
  int k = g_cnt[t->name]++;
  std::string fn = g_dir + "/" + t->name + "#" + std::to_string(k) + "." + ggml_type_name(t->type);
  FILE * f = fopen(fn.c_str(), "wb"); fwrite(buf.data(), 1, nb, f); fclose(f);
  fprintf(g_idx, "%s#%d %s ne %lld %lld %lld %lld\n", t->name, k, ggml_type_name(t->type),
      (long long)t->ne[0], (long long)t->ne[1], (long long)t->ne[2], (long long)t->ne[3]);
  return true;
}
int main(int argc, char ** argv) {
  std::setlocale(LC_NUMERIC, "C");
  const char * idsf = getenv("LD_IDS"); const char * dir = getenv("LD_DIR");
  if (!idsf || !dir) { fprintf(stderr, "need LD_IDS LD_DIR\n"); return 1; }
  g_dir = dir;
  std::vector<llama_token> ids; { std::ifstream in(idsf); long v; while (in >> v) ids.push_back((llama_token)v); }
  common_params params; common_init();
  if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_COMMON)) return 1;
  llama_backend_init(); llama_numa_init(params.numa);
  params.cb_eval = cb; params.cb_eval_user_data = nullptr; params.warmup = false;
  auto init = common_init_from_params(params);
  auto * model = init->model(); auto * ctx = init->context();
  if (!model || !ctx) { fprintf(stderr, "init failed\n"); return 1; }
  const int nb = llama_n_batch(ctx);
  const int N = (int) ids.size();
  fprintf(stderr, "ldump: %d ids, n_batch %d\n", N, nb);
  const int K = getenv("LD_K") ? atoi(getenv("LD_K")) : 1;
  const std::string base = g_dir;
  for (int i = 0; i < N - K; i += nb) {
    int n = std::min(nb, N - K - i);
    if (llama_decode(ctx, llama_batch_get_one(ids.data() + i, n))) { fprintf(stderr, "decode fail at %d\n", i); return 1; }
    if ((i / nb) % 8 == 0) fprintf(stderr, "ldump: %d/%d\n", i + n, N);
  }
  int nv = llama_vocab_n_tokens(llama_model_get_vocab(model));
  for (int p = N - K; p < N; ++p) {
    g_dir = base + "/p" + std::to_string(p);
    std::string cmd = "mkdir -p " + g_dir; if (system(cmd.c_str())) return 1;
    g_cnt.clear();
    g_idx = fopen((g_dir + "/index.txt").c_str(), "w");
    g_on = true;
    if (llama_decode(ctx, llama_batch_get_one(ids.data() + p, 1))) { fprintf(stderr, "decode fail at %d\n", p); return 1; }
    g_on = false;
    fclose(g_idx);
    const float * lg = llama_get_logits_ith(ctx, -1);
    FILE * f = fopen((g_dir + "/logits.f32").c_str(), "wb"); fwrite(lg, sizeof(float), nv, f); fclose(f);
  }
  fprintf(stderr, "ldump: done, n_vocab %d\n", nv);
  llama_backend_free();
  return 0;
}
