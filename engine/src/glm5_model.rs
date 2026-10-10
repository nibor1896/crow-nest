//! crow-nest #161-#165 wiring: the glm5_next decoder layers as one runnable path on [`Glm5Geo`].
//!
//! The five block modules (`glm5_mhc` #161, `glm5_kda` #162, `glm5_mla` #163, `glm5_moe` #164,
//! `glm5_head` #165) each passed G3 against HF goldens on synthetic weights; this module calls
//! them in the decoder order of `docs/glm5-next-recipe.md` sections 3-4 (HF
//! `modeling_glm5_next.py:1291-1327`, `:1477-1493`), per call of `t` rows:
//!
//! ```text
//! attn_hc coeffs -> input_layernorm -> KDA or MLA + DSA -> expand
//! ffn_hc coeffs  -> post_attention_layernorm -> dense FFN or MoE -> expand
//! ```
//!
//! The trunk input is the embedding copied into the 4 streams ([`trunk_input`], HF `:1477`); the
//! final collapse, norm and lm_head are `glm5_head`. What this module adds besides the order:
//!
//! - **Codec gap.** The container stores KDA q/k/v/o and the KDA short conv, and MLA
//!   q_a/q_b/kv_a/kv_b/o as NVFP4 (`converter/src/recipe.rs`, PREREG "Recipe as committed").
//!   The projections run on the glm5 NVFP4 GEMV (`glm5_gemv_fp4` in `kernels_glm5_moe.cu`, #191:
//!   bit-identical to the engine's `gemv_fp4_b` / `gemv_fp4_bs`, several rows per block and no
//!   local-memory decode table; a row stride for the strided q|k|v rows) through the modules'
//!   projection hooks (`glm5_kda::prompt_with` / `step_with`, `glm5_mla::MlaScratch::forward_with`), with every scale byte 0x7F rewritten to
//!   0x7E first (`residency::sanitize_sf_slab`, the rule of `gen::load_pw_x`, #177). The conv
//!   weight is decoded once to f32 (HF holds it in f32), `kv_b` once to BF16 (the form
//!   `gm_absorb` / `gm_out_v` read), its inexact-value count reported ([`LoadReport`], the
//!   `dense_overlay` control); the #159 planner books the BF16 bytes
//!   (`Glm5Geo::kv_b_decode_bytes`).
//! - **One shared glm5 `KernelGeo` compile** ([`Glm5Kernels`]): `KERNEL_SRC` once at
//!   `glm5_kda::kernel_geo`; the KDA kernels, the engine kernel table of the router and the head
//!   (`gemv_bf16_b`, `gemv_bf16_w`, `argmax_k`) and MLA's `qsa_select_fast` all come from it.
//! - **Layer-at-a-time weights** ([`load_layer`]): one layer's tensors in VRAM, its 288 MUL1
//!   expert records in one VRAM buffer behind the `GpuMoePlan` record table, all freed after the
//!   layer. No cache, no NVMe tier.
//! - **Experts from the three tiers** (#175, #149, plan steps 16-17): [`load_layer_without_experts`]
//!   loads a layer without its records and [`Glm5Pass::call_with_experts`] asks a hook for the
//!   record table after the router ran; `glm5_tiers` keeps every layer resident that way and
//!   serves the experts from VRAM, pinned RAM or the container (NVMe). The layer math is the same.
//!
//! The `Engine` / `Geo` path of Flash-Next and the 27B does not see this module (gate R, #174).
//! Its callers are the `decode glmgolden` harness (`bin/decode.rs`, [`golden`]) and `glm5_tiers`
//! (`bin/glm5_run.rs`).

use crate::cnq::{self, Cnq, TensorInfo};
use crate::cuda;
use crate::geo::Glm5Geo;
use crate::glm5_head::{Head, HeadGeo};
use crate::glm5_kda::{self, KdaDims, KdaKernels, KdaProj, KdaScratch, KdaState, KdaWeights};
use crate::glm5_mhc::{self, SiteDev, HC, MIX};
use crate::glm5_mla::{MlaCache, MlaDims, MlaKernels, MlaProj, MlaScratch, MlaWeights};
use crate::glm5_moe::{ExpertMajor, GpuFfnPlan, GpuFfnWeights, GpuMoeGroupedPlan, GpuMoePlan, GpuMoeWeights, GpuNvfp4, MoeGeo, Routing, GROUP_ROWS};
use crate::kernels::{self, launch_v, mul1, KernelGeo};
use cudarc::driver::sys::CUdeviceptr;

pub type Dev = CUdeviceptr;

/// the 3-bit GLM-5.3-Flash container (MUL1 K=3 experts, NVFP4 + BF16 dense part), relative to
/// the repo root (`geo::from_engine_dir` from `engine/`)
pub const GLM5_MUL1K3_CNQ: &str = "converter/GLM-5.3-Flash-MUL1K3.cnq";

/// the checkpoint prefix of a text layer
pub fn layer_prefix(l: usize) -> String {
    format!("model.language_model.layers.{l}.")
}

// ---------------------------------------------------------------- the layer schedule

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AttnKind {
    /// Kimi delta attention, `layer_types` `linear_attention`
    Kda,
    /// MLA + DSA indexer, `deepseek_sparse_attention`
    Mla,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FfnKind {
    /// the clamped SwiGLU of layers 0-2
    Dense,
    /// router + 288 routed experts + shared expert
    Moe,
}

pub fn attn_kind(g: &Glm5Geo, l: usize) -> AttnKind {
    if g.is_dsa(l) { AttnKind::Mla } else { AttnKind::Kda }
}

pub fn ffn_kind(g: &Glm5Geo, l: usize) -> FfnKind {
    if l < g.dense_prefix { FfnKind::Dense } else { FfnKind::Moe }
}

/// the runner's `layer_kinds` label (`kda+dense`, `dsa+moe`, ...)
pub fn kind_label(g: &Glm5Geo, l: usize) -> &'static str {
    match (attn_kind(g, l), ffn_kind(g, l)) {
        (AttnKind::Kda, FfnKind::Dense) => "kda+dense",
        (AttnKind::Kda, FfnKind::Moe) => "kda+moe",
        (AttnKind::Mla, FfnKind::Dense) => "dsa+dense",
        (AttnKind::Mla, FfnKind::Moe) => "dsa+moe",
    }
}

// ---------------------------------------------------------------- the tensor plan

/// How the loader takes one container tensor.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Take {
    /// BF16 bytes as stored (gm_gemm / gemm_bf16_dense / gemv_bf16 matrices, mHC `fn`)
    Bf16,
    /// f32 on the device: an F32 carry as stored, a BF16 keep widened exactly (norms, biases)
    F32,
    /// NVFP4 on `gemv_fp4_b`, scale bytes sanitized
    Fp4,
    /// NVFP4 decoded once to f32 (the KDA short conv)
    Fp4ToF32,
    /// NVFP4 decoded once to BF16 (MLA `kv_b`)
    Fp4ToBf16,
}

impl Take {
    /// the index dtypes this take accepts
    pub fn dtypes(&self) -> &'static [&'static str] {
        match self {
            Take::Bf16 => &["bf16"],
            Take::F32 => &["f32", "bf16"],
            Take::Fp4 | Take::Fp4ToF32 | Take::Fp4ToBf16 => &["nvfp4"],
        }
    }
}

/// one tensor the loader reads: its checkpoint name, shape and take
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Planned {
    pub name: String,
    pub shape: Vec<u64>,
    pub take: Take,
}

fn planned(name: String, shape: &[usize], take: Take) -> Planned {
    Planned { name, shape: shape.iter().map(|&v| v as u64).collect(), take }
}

/// Every non-expert tensor of text layer `l`, in load order (`docs/glm5-next-recipe.md` sections
/// 5-10; the codec of each is the converter's `cnq4.5-glm5-next` row). The routed experts are
/// the record table's ([`load_layer`]).
pub fn layer_tensors(g: &Glm5Geo, l: usize) -> Vec<Planned> {
    let p = layer_prefix(l);
    let n = |s: &str| format!("{p}{s}");
    let h = g.hidden;
    let mut v = Vec::new();
    for site in ["attn", "ffn"] {
        v.push(planned(n(&format!("hc_{site}_fn")), &[MIX, HC * h], Take::Bf16));
        v.push(planned(n(&format!("hc_{site}_base")), &[MIX], Take::F32));
        v.push(planned(n(&format!("hc_{site}_scale")), &[3], Take::F32));
    }
    v.push(planned(n("input_layernorm.weight"), &[h], Take::F32));
    v.push(planned(n("post_attention_layernorm.weight"), &[h], Take::F32));
    match attn_kind(g, l) {
        AttnKind::Kda => {
            let (w, hd, heads) = (g.kda_heads * g.kda_head_dim, g.kda_head_dim, g.kda_heads);
            for q in ["q", "k", "v"] {
                v.push(planned(n(&format!("self_attn.{q}_proj.weight")), &[w, h], Take::Fp4));
            }
            for q in ["q", "k", "v"] {
                v.push(planned(n(&format!("self_attn.{q}_conv1d.weight")), &[w, 1, g.kda_conv], Take::Fp4ToF32));
            }
            v.push(planned(n("self_attn.f_a_proj.weight"), &[hd, h], Take::Bf16));
            v.push(planned(n("self_attn.f_b_proj.weight"), &[w, hd], Take::Bf16));
            v.push(planned(n("self_attn.dt_bias"), &[w], Take::F32));
            v.push(planned(n("self_attn.A_log"), &[heads], Take::F32));
            v.push(planned(n("self_attn.b_proj.weight"), &[heads, h], Take::Bf16));
            v.push(planned(n("self_attn.g_a_proj.weight"), &[hd, h], Take::Bf16));
            v.push(planned(n("self_attn.g_b_proj.weight"), &[w, hd], Take::Bf16));
            v.push(planned(n("self_attn.o_norm.weight"), &[hd], Take::F32));
            v.push(planned(n("self_attn.o_proj.weight"), &[h, w], Take::Fp4));
        }
        AttnKind::Mla => {
            let d = MlaDims::of(g);
            v.push(planned(n("self_attn.q_a_proj.weight"), &[d.q_lora, h], Take::Fp4));
            v.push(planned(n("self_attn.q_a_layernorm.weight"), &[d.q_lora], Take::F32));
            v.push(planned(n("self_attn.q_b_proj.weight"), &[d.heads * d.nope, d.q_lora], Take::Fp4));
            v.push(planned(n("self_attn.kv_a_proj_with_mqa.weight"), &[d.kv_lora, h], Take::Fp4));
            v.push(planned(n("self_attn.kv_a_layernorm.weight"), &[d.kv_lora], Take::F32));
            v.push(planned(n("self_attn.kv_b_proj.weight"), &[d.heads * (d.nope + d.v), d.kv_lora], Take::Fp4ToBf16));
            v.push(planned(n("self_attn.o_proj.weight"), &[h, d.heads * d.v], Take::Fp4));
            v.push(planned(n("self_attn.indexer.wq_b.weight"), &[d.idx_heads * d.idx_dim, d.q_lora], Take::Bf16));
            v.push(planned(n("self_attn.indexer.wk.weight"), &[d.idx_dim, h], Take::Bf16));
            v.push(planned(n("self_attn.indexer.index_kpool_compress_gate"), &[d.idx_dim, h], Take::Bf16));
            v.push(planned(n("self_attn.indexer.weights_proj.weight"), &[d.idx_heads, h], Take::Bf16));
            v.push(planned(n("self_attn.indexer.k_norm.weight"), &[d.idx_dim], Take::F32));
            v.push(planned(n("self_attn.indexer.k_norm.bias"), &[d.idx_dim], Take::F32));
            v.push(planned(n("self_attn.indexer.index_kpool_compress_ape"), &[d.kpool, d.idx_dim], Take::F32));
        }
    }
    match ffn_kind(g, l) {
        FfnKind::Dense => {
            v.push(planned(n("mlp.gate_proj.weight"), &[g.dense_inter, h], Take::Fp4));
            v.push(planned(n("mlp.up_proj.weight"), &[g.dense_inter, h], Take::Fp4));
            v.push(planned(n("mlp.down_proj.weight"), &[h, g.dense_inter], Take::Fp4));
        }
        FfnKind::Moe => {
            let si = g.expert_inter * g.shared_experts;
            v.push(planned(n("mlp.gate.weight"), &[g.experts, h], Take::Bf16));
            v.push(planned(n("mlp.gate.e_score_correction_bias"), &[g.experts], Take::F32));
            v.push(planned(n("mlp.shared_experts.gate_proj.weight"), &[si, h], Take::Fp4));
            v.push(planned(n("mlp.shared_experts.up_proj.weight"), &[si, h], Take::Fp4));
            v.push(planned(n("mlp.shared_experts.down_proj.weight"), &[h, si], Take::Fp4));
        }
    }
    v
}

/// the model-level tensors: the embedding (read by rows), the final norm, the untied lm_head
pub fn model_tensors(g: &Glm5Geo) -> Vec<Planned> {
    vec![
        planned("model.language_model.embed_tokens.weight".into(), &[g.vocab, g.hidden], Take::Bf16),
        planned("model.language_model.norm.weight".into(), &[g.hidden], Take::F32),
        planned("lm_head.weight".into(), &[g.vocab, g.hidden], Take::Bf16),
    ]
}

/// the names of routed expert `e` of layer `l`: gate, up, down (the record starts at gate)
pub fn expert_names(l: usize, e: usize) -> [String; 3] {
    let p = layer_prefix(l);
    ["gate", "up", "down"].map(|q| format!("{p}mlp.experts.{e}.{q}_proj.weight"))
}

/// Every planned tensor against the index (text section): present, a dtype its take accepts,
/// the planned shape. `Err` lists every miss by name, before any byte is read.
pub fn check_plan(plan: &[Planned], index: &[TensorInfo]) -> Result<(), String> {
    let mut bad = Vec::new();
    for p in plan {
        match index.iter().find(|t| t.name == p.name && t.section == "text") {
            None => bad.push(format!("{}: not in the index", p.name)),
            Some(t) => {
                if !p.take.dtypes().contains(&t.dtype.as_str()) {
                    bad.push(format!("{}: dtype {}, the glm5_next path takes {:?} as {:?}", p.name, t.dtype, p.take.dtypes(), p.take));
                }
                if t.shape != p.shape {
                    bad.push(format!("{}: shape {:?}, want {:?}", p.name, t.shape, p.shape));
                }
            }
        }
    }
    if bad.is_empty() {
        Ok(())
    } else {
        Err(format!("refusing the container for the glm5_next path (#161): {} tensor(s) do not fit\n  {}", bad.len(), bad.join("\n  ")))
    }
}

// ---------------------------------------------------------------- host decodes (load time)

/// f32 -> BF16 bits, round to nearest even (NaN kept a NaN): the converter's `f32_to_bf16_rne`
pub fn f32_to_bf16_rne(v: f32) -> u16 {
    let bits = v.to_bits();
    if (bits & 0x7F80_0000) == 0x7F80_0000 && (bits & 0x007F_FFFF) != 0 {
        return ((bits >> 16) | 0x0040) as u16;
    }
    let rounding = 0x7FFF + ((bits >> 16) & 1);
    (bits.wrapping_add(rounding) >> 16) as u16
}

/// NVFP4 bytes (36 B per 64 values, scale bytes as given) -> `n` f32 values: `e2m1 * (ue4m3 *
/// global)`, `cnq::dequant_block`, the arithmetic of `gemv_fp4_b`
pub fn nvfp4_to_f32(raw: &[u8], global: f32, n: usize) -> Vec<f32> {
    assert!(raw.len() >= n.div_ceil(64) * 36, "nvfp4: {} B hold fewer than {n} values", raw.len());
    let mut out = Vec::with_capacity(n.div_ceil(64) * 64);
    let mut blk = [0f32; 64];
    for b in raw.chunks_exact(36).take(n.div_ceil(64)) {
        cnq::dequant_block(b, global, &mut blk);
        out.extend_from_slice(&blk);
    }
    out.truncate(n);
    out
}

/// [`nvfp4_to_f32`] rounded to BF16 (RNE) and the count of values BF16 does not hold exactly
pub fn nvfp4_to_bf16(raw: &[u8], global: f32, n: usize) -> (Vec<u16>, u64) {
    let mut inexact = 0u64;
    let out = nvfp4_to_f32(raw, global, n)
        .into_iter()
        .map(|v| {
            let b = f32_to_bf16_rne(v);
            if f32::from_bits((b as u32) << 16) != v {
                inexact += 1;
            }
            b
        })
        .collect();
    (out, inexact)
}

/// `[n][h]` embedding rows -> the trunk input `[n][4][h]`: each row copied into every stream
/// (HF `modeling_glm5_next.py:1477`)
pub fn trunk_input(embed: &[f32], h: usize, streams: usize) -> Vec<f32> {
    assert_eq!(embed.len() % h, 0, "trunk_input: {} values are not [n][{h}]", embed.len());
    let mut x = Vec::with_capacity(embed.len() * streams);
    for row in embed.chunks_exact(h) {
        for _ in 0..streams {
            x.extend_from_slice(row);
        }
    }
    x
}

// ---------------------------------------------------------------- kernels

/// the one shared glm5 compile of `KERNEL_SRC` (the KDA geometry, `glm5_kda::kernel_geo`)
pub fn kernel_geo(g: &Glm5Geo) -> KernelGeo {
    glm5_kda::kernel_geo(g)
}

/// every `KERNEL_SRC` entry the glm5 path launches from the shared module
pub const MAIN_NAMES: &[&str] = &["gemv_bf16_b", "gemv_bf16_w", "argmax_k", "qsa_select_fast"];

/// The modules of the glm5_next path: `KERNEL_SRC` compiled ONCE at [`kernel_geo`] (KDA's GDN
/// kernels, the engine kernel table, `qsa_select_fast`), plus the blocks' own NVRTC modules
/// (mHC, KDA's three kernels, MLA + DSA, router/MUL1, head).
pub struct Glm5Kernels {
    pub main: cuda::Module,
    pub k: kernels::Kernels,
    pub kda: KdaKernels,
    pub mla: MlaKernels,
    pub mhc: glm5_mhc::Kernels,
    pub moe: kernels::glm5_moe::Kernels,
    pub mul1: mul1::Kernels,
    /// #186 `CROW_GLM_EMBED_GATHER=1`: the prompt calls' device gather (`glm5_embed`); `None` off
    pub embed: Option<crate::glm5_embed::Gather>,
}

impl Glm5Kernels {
    /// # Safety
    /// A CUDA context is current.
    pub unsafe fn new(g: &Glm5Geo) -> Glm5Kernels {
        let main = cuda::compile(&kernel_geo(g).source());
        let k = kernels::Kernels::new(&main, false);
        // a second handle to the same module: `main` owns it
        let kda = KdaKernels::with_base(g, cuda::Module(main.0));
        let mla = MlaKernels::new(MlaDims::of(g), main.get("qsa_select_fast"));
        Glm5Kernels { k, kda, mla, mhc: glm5_mhc::Kernels::new(), moe: kernels::glm5_moe::Kernels::new(), mul1: mul1::Kernels::new(), embed: crate::glm5_embed::Gather::from_env(), main }
    }
}

// ---------------------------------------------------------------- one layer's weights

/// What a load did, for the load line: bytes uploaded, scale bytes rewritten 0x7F -> 0x7E, and
/// the `kv_b` BF16 decode's inexact values
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct LoadReport {
    pub bytes: u64,
    pub sanitized: u64,
    pub kv_b_values: u64,
    pub kv_b_inexact: u64,
}

pub struct KdaLayer {
    /// the BF16 gates, conv (f32), biases; `qkv` and `o_proj` are 0 (the NVFP4 hook runs them)
    pub w: KdaWeights,
    pub q: GpuNvfp4,
    pub k: GpuNvfp4,
    pub v: GpuNvfp4,
    pub o: GpuNvfp4,
}

pub struct MlaLayer {
    /// norms, `kv_b` (BF16, decoded at load), the indexer; `q_a`, `q_b`, `kv_a`, `o_proj` are 0
    pub w: MlaWeights,
    pub q_a: GpuNvfp4,
    pub q_b: GpuNvfp4,
    pub kv_a: GpuNvfp4,
    pub o: GpuNvfp4,
}

pub enum AttnW {
    Kda(KdaLayer),
    Mla(MlaLayer),
}

pub enum FfnW {
    Dense(GpuFfnWeights),
    /// `records`: the layer's routed experts back to back (`[E][record bytes]`), `table` the
    /// device `[E]` u64 of their bases (`GpuMoePlan::run`)
    Moe { w: GpuMoeWeights, records: Dev, table: Dev },
}

pub struct LayerW {
    pub layer: usize,
    pub attn_hc: SiteDev,
    pub ffn_hc: SiteDev,
    pub input_norm: Dev,
    pub post_norm: Dev,
    pub attn: AttnW,
    pub ffn: FfnW,
}

unsafe fn free_fp4(m: &mut GpuNvfp4) {
    cuda::free_dev(&mut m.w);
    cuda::free_dev(&mut m.gs);
}

impl LayerW {
    /// # Safety
    /// No launch reading these weights is pending.
    pub unsafe fn free(&mut self) {
        self.attn_hc.free();
        self.ffn_hc.free();
        cuda::free_dev(&mut self.input_norm);
        cuda::free_dev(&mut self.post_norm);
        match &mut self.attn {
            AttnW::Kda(a) => {
                a.w.free();
                for m in [&mut a.q, &mut a.k, &mut a.v, &mut a.o] {
                    free_fp4(m);
                }
            }
            AttnW::Mla(a) => {
                let w = &mut a.w;
                for p in [&mut w.q_a_norm, &mut w.kv_a_norm, &mut w.kv_b, &mut w.idx_wq_b, &mut w.idx_x, &mut w.idx_k_norm_w, &mut w.idx_k_norm_b, &mut w.idx_ape] {
                    cuda::free_dev(p);
                }
                for m in [&mut a.q_a, &mut a.q_b, &mut a.kv_a, &mut a.o] {
                    free_fp4(m);
                }
            }
        }
        match &mut self.ffn {
            FfnW::Dense(f) => {
                for m in [&mut f.gate, &mut f.up, &mut f.down] {
                    free_fp4(m);
                }
            }
            FfnW::Moe { w, records, table } => {
                cuda::free_dev(&mut w.router);
                cuda::free_dev(&mut w.bias);
                for m in [&mut w.shared.gate, &mut w.shared.up, &mut w.shared.down] {
                    free_fp4(m);
                }
                cuda::free_dev(records);
                cuda::free_dev(table);
            }
        }
    }
}

/// the container reads of one load, every tensor checked against its plan row first
struct Loader<'a> {
    cnq: &'a mut Cnq,
    rep: &'a mut LoadReport,
    plan: Vec<Planned>,
}

impl Loader<'_> {
    /// the plan row and index entry of the tensor whose name ends in `suffix`
    fn get(&self, suffix: &str) -> (Planned, TensorInfo) {
        let p = self.plan.iter().find(|p| p.name.ends_with(suffix)).unwrap_or_else(|| panic!("glm5_model: no plan row ends in {suffix}")).clone();
        let t = self.cnq.find(&p.name, "text").clone();
        assert!(p.take.dtypes().contains(&t.dtype.as_str()) && t.shape == p.shape, "glm5_model: {} is {} {:?}, the plan says {:?} {:?}", p.name, t.dtype, t.shape, p.take, p.shape);
        (p, t)
    }

    fn bytes(&mut self, suffix: &str) -> (Planned, TensorInfo, Vec<u8>) {
        let (p, t) = self.get(suffix);
        let raw = self.cnq.read_bytes(&t);
        (p, t, raw)
    }

    /// BF16 bytes as stored, host side (for stacking)
    fn bf16_host(&mut self, suffix: &str) -> Vec<u8> {
        let (p, _, raw) = self.bytes(suffix);
        assert_eq!(p.take, Take::Bf16);
        raw
    }

    unsafe fn bf16(&mut self, suffix: &str) -> Dev {
        let raw = self.bf16_host(suffix);
        self.rep.bytes += raw.len() as u64;
        cuda::upload_dev(&raw)
    }

    fn f32_host(&mut self, suffix: &str) -> Vec<f32> {
        let (p, t, raw) = self.bytes(suffix);
        assert_eq!(p.take, Take::F32);
        match t.dtype.as_str() {
            "f32" => cnq::f32_bytes_to_f32(&raw),
            _ => cnq::bf16_bytes_to_f32(&raw),
        }
    }

    unsafe fn f32(&mut self, suffix: &str) -> Dev {
        let v = self.f32_host(suffix);
        self.rep.bytes += v.len() as u64 * 4;
        cuda::to_f32_dev(&v)
    }

    /// NVFP4 bytes with every scale byte 0x7F rewritten to 0x7E (#177), and the global scale
    fn fp4_host(&mut self, suffix: &str) -> (Planned, TensorInfo, Vec<u8>) {
        let (p, t, mut raw) = self.bytes(suffix);
        self.rep.sanitized += crate::residency::sanitize_sf_slab(&mut raw);
        (p, t, raw)
    }

    unsafe fn fp4(&mut self, suffix: &str) -> GpuNvfp4 {
        let (p, t, raw) = self.fp4_host(suffix);
        assert_eq!(p.take, Take::Fp4);
        self.rep.bytes += raw.len() as u64 + 4;
        GpuNvfp4 { w: cuda::upload_dev(&raw), gs: cuda::to_f32_dev(&[t.global_scale]), rows: p.shape[0] as usize, cols: p.shape[1..].iter().product::<u64>() as usize }
    }

    fn fp4_f32_host(&mut self, suffix: &str) -> Vec<f32> {
        let (p, t, raw) = self.fp4_host(suffix);
        assert_eq!(p.take, Take::Fp4ToF32);
        nvfp4_to_f32(&raw, t.global_scale, t.n_values as usize)
    }

    unsafe fn fp4_bf16(&mut self, suffix: &str) -> Dev {
        let (p, t, raw) = self.fp4_host(suffix);
        assert_eq!(p.take, Take::Fp4ToBf16);
        let (v, inexact) = nvfp4_to_bf16(&raw, t.global_scale, t.n_values as usize);
        self.rep.kv_b_values += v.len() as u64;
        self.rep.kv_b_inexact += inexact;
        self.rep.bytes += v.len() as u64 * 2;
        cuda::to_dev(&v)
    }
}

/// Load text layer `l` into VRAM: the planned tensors ([`layer_tensors`]) and, for a MoE layer,
/// its routed-expert records (`moe.record.bytes` each, from the gate tensor's offset, the #181
/// layout `nvme_source::glm5_record_from_index` checked) into one buffer behind a `[E]` table.
///
/// # Safety
/// A CUDA context is current.
pub unsafe fn load_layer(cnq: &mut Cnq, g: &Glm5Geo, moe: &MoeGeo, l: usize, rep: &mut LoadReport) -> LayerW {
    load_layer_with(cnq, g, moe, l, rep, true)
}

/// #175: [`load_layer`] without the routed-expert records: a MoE layer's `FfnW::Moe` has
/// `records` and `table` 0, and its experts come from the three tiers
/// (`glm5_tiers::ExpertTiers`, through [`Glm5Pass::call_with_experts`]).
///
/// # Safety
/// A CUDA context is current.
pub unsafe fn load_layer_without_experts(cnq: &mut Cnq, g: &Glm5Geo, moe: &MoeGeo, l: usize, rep: &mut LoadReport) -> LayerW {
    load_layer_with(cnq, g, moe, l, rep, false)
}

unsafe fn load_layer_with(cnq: &mut Cnq, g: &Glm5Geo, moe: &MoeGeo, l: usize, rep: &mut LoadReport, with_records: bool) -> LayerW {
    let plan = layer_tensors(g, l);
    if let Err(why) = check_plan(&plan, &cnq.tensors) {
        panic!("{why}");
    }
    let mut ld = Loader { cnq, rep, plan };
    let site = |ld: &mut Loader, s: &str| -> SiteDev {
        SiteDev { fn_: ld.bf16(&format!("hc_{s}_fn")), base: ld.f32(&format!("hc_{s}_base")), scale: ld.f32(&format!("hc_{s}_scale")) }
    };
    let attn_hc = site(&mut ld, "attn");
    let ffn_hc = site(&mut ld, "ffn");
    let input_norm = ld.f32("input_layernorm.weight");
    let post_norm = ld.f32("post_attention_layernorm.weight");
    let attn = match attn_kind(g, l) {
        AttnKind::Kda => {
            let mut conv = Vec::with_capacity(3 * g.kda_heads * g.kda_head_dim * g.kda_conv);
            for q in ["q", "k", "v"] {
                conv.extend(ld.fp4_f32_host(&format!("self_attn.{q}_conv1d.weight")));
            }
            ld.rep.bytes += conv.len() as u64 * 4;
            let w = KdaWeights {
                qkv: 0,
                conv: cuda::to_f32_dev(&conv),
                f_a: ld.bf16("self_attn.f_a_proj.weight"),
                f_b: ld.bf16("self_attn.f_b_proj.weight"),
                dt_bias: ld.f32("self_attn.dt_bias"),
                a_log: ld.f32("self_attn.A_log"),
                b: ld.bf16("self_attn.b_proj.weight"),
                g_a: ld.bf16("self_attn.g_a_proj.weight"),
                g_b: ld.bf16("self_attn.g_b_proj.weight"),
                o_norm: ld.f32("self_attn.o_norm.weight"),
                o_proj: 0,
            };
            AttnW::Kda(KdaLayer {
                w,
                q: ld.fp4("self_attn.q_proj.weight"),
                k: ld.fp4("self_attn.k_proj.weight"),
                v: ld.fp4("self_attn.v_proj.weight"),
                o: ld.fp4("self_attn.o_proj.weight"),
            })
        }
        AttnKind::Mla => {
            let mut idx_x = ld.bf16_host("self_attn.indexer.wk.weight");
            idx_x.extend(ld.bf16_host("self_attn.indexer.index_kpool_compress_gate"));
            idx_x.extend(ld.bf16_host("self_attn.indexer.weights_proj.weight"));
            ld.rep.bytes += idx_x.len() as u64;
            let w = MlaWeights {
                q_a: 0,
                q_a_norm: ld.f32("self_attn.q_a_layernorm.weight"),
                q_b: 0,
                kv_a: 0,
                kv_a_norm: ld.f32("self_attn.kv_a_layernorm.weight"),
                kv_b: ld.fp4_bf16("self_attn.kv_b_proj.weight"),
                o_proj: 0,
                idx_wq_b: ld.bf16("self_attn.indexer.wq_b.weight"),
                idx_x: cuda::upload_dev(&idx_x),
                idx_k_norm_w: ld.f32("self_attn.indexer.k_norm.weight"),
                idx_k_norm_b: ld.f32("self_attn.indexer.k_norm.bias"),
                idx_ape: ld.f32("self_attn.indexer.index_kpool_compress_ape"),
            };
            AttnW::Mla(MlaLayer {
                w,
                q_a: ld.fp4("self_attn.q_a_proj.weight"),
                q_b: ld.fp4("self_attn.q_b_proj.weight"),
                kv_a: ld.fp4("self_attn.kv_a_proj_with_mqa.weight"),
                o: ld.fp4("self_attn.o_proj.weight"),
            })
        }
    };
    let ffn = match ffn_kind(g, l) {
        FfnKind::Dense => FfnW::Dense(GpuFfnWeights { gate: ld.fp4("mlp.gate_proj.weight"), up: ld.fp4("mlp.up_proj.weight"), down: ld.fp4("mlp.down_proj.weight") }),
        FfnKind::Moe => {
            let w = GpuMoeWeights {
                router: ld.bf16("mlp.gate.weight"),
                bias: ld.f32("mlp.gate.e_score_correction_bias"),
                shared: GpuFfnWeights {
                    gate: ld.fp4("mlp.shared_experts.gate_proj.weight"),
                    up: ld.fp4("mlp.shared_experts.up_proj.weight"),
                    down: ld.fp4("mlp.shared_experts.down_proj.weight"),
                },
            };
            if !with_records {
                return LayerW { layer: l, attn_hc, ffn_hc, input_norm, post_norm, attn, ffn: FfnW::Moe { w, records: 0, table: 0 } };
            }
            let rb = moe.record.bytes;
            let records = cuda::alloc_named("glm5 layer expert records", (g.experts as u64 * rb) as usize);
            let mut bases = Vec::with_capacity(g.experts);
            for e in 0..g.experts {
                let [gate, _, _] = expert_names(l, e);
                let t = ld.cnq.find(&gate, "text").clone();
                assert_eq!(t.dtype, "mul1", "glm5_model: {gate} is {}, the record table takes MUL1 records (#181)", t.dtype);
                let rec = ld.cnq.read_range(&t, 0, rb as usize);
                let base = records + e as u64 * rb;
                cuda::into_dev(base, rec.as_slice());
                bases.push(base);
            }
            ld.rep.bytes += g.experts as u64 * rb;
            FfnW::Moe { w, records, table: cuda::to_u64_dev(&bases) }
        }
    };
    LayerW { layer: l, attn_hc, ffn_hc, input_norm, post_norm, attn, ffn }
}

/// the final norm (f32) and the untied BF16 lm_head in VRAM
pub struct HeadW {
    pub norm: Dev,
    pub lm: Dev,
}

/// # Safety
/// A CUDA context is current.
pub unsafe fn load_head(cnq: &mut Cnq, g: &Glm5Geo, rep: &mut LoadReport) -> HeadW {
    let plan = model_tensors(g);
    if let Err(why) = check_plan(&plan, &cnq.tensors) {
        panic!("{why}");
    }
    let mut ld = Loader { cnq, rep, plan };
    HeadW { norm: ld.f32("model.language_model.norm.weight"), lm: ld.bf16("lm_head.weight") }
}

/// the embedding rows of `ids`, widened to f32 `[n][hidden]` (the token embedding is a BF16 keep)
pub fn embed_rows(cnq: &mut Cnq, g: &Glm5Geo, ids: &[i64]) -> Vec<f32> {
    let t = cnq.find("model.language_model.embed_tokens.weight", "text").clone();
    assert_eq!((t.dtype.as_str(), t.shape.as_slice()), ("bf16", &[g.vocab as u64, g.hidden as u64][..]), "glm5_model: embed_tokens");
    let row = g.hidden * 2;
    let mut out = Vec::with_capacity(ids.len() * g.hidden);
    for &id in ids {
        assert!((0..g.vocab as i64).contains(&id), "glm5_model: token id {id} outside the vocab of {}", g.vocab);
        out.extend(cnq::bf16_bytes_to_f32(&cnq.read_range(&t, id as u64 * row as u64, row)));
    }
    out
}

// ---------------------------------------------------------------- the layer driver

/// device i32 scalars the GEMV launches read (the p5 rule: every scalar is a device pointer)
struct Ints {
    dev: Dev,
    vals: Vec<usize>,
}

impl Ints {
    unsafe fn new(vals: &[usize]) -> Ints {
        let v: Vec<i32> = vals.iter().map(|&x| i32::try_from(x).expect("glm5_model: scalar beyond i32")).collect();
        Ints { dev: cuda::to_i32_dev(&v), vals: vals.to_vec() }
    }
    fn p(&self, v: usize) -> Dev {
        let i = self.vals.iter().position(|&x| x == v).unwrap_or_else(|| panic!("glm5_model: no device scalar {v}"));
        self.dev + (i * 4) as u64
    }
}

/// `y [t][rows] = W x [t][cols]` on the glm5 NVFP4 GEMV (`glm5_gemv_fp4`, #191: bit-identical
/// to the record `gemv_fp4_b`); with `ldy` the output row stride is `ldy` (the record's
/// `gemv_fp4_bs`), else `rows`. #186: with `CROW_GLM_DENSE_GEMM=1` a call of at least
/// `TC_MIN_ROWS` rows runs on the FP16 tensor-core GEMM instead (`glm5_gemm_fp4_tc`).
unsafe fn fp4_gemv(kn: &Glm5Kernels, ints: &Ints, m: &GpuNvfp4, x: Dev, y: Dev, t: usize, ldy: Option<usize>) {
    fp4_gemv_of(kn, ints, m, x, y, t, ldy, t)
}

/// #196: [`fp4_gemv`] for `t` rows of a call of `call_t` rows run in sub-blocks: the kernel is
/// picked by `call_t` (`CROW_GLM_DENSE_GEMM`), so every row of the call takes the kernel the
/// whole call gives it (as `GpuFfnPlan::run_rows_of`)
#[allow(clippy::too_many_arguments)]
unsafe fn fp4_gemv_of(kn: &Glm5Kernels, ints: &Ints, m: &GpuNvfp4, x: Dev, y: Dev, t: usize, ldy: Option<usize>, call_t: usize) {
    let ld = ldy.unwrap_or(m.rows);
    if kn.moe.dense_tc(call_t) {
        let small = kernels::glm5_moe::tc_small(call_t);
        return kn.moe.gemm_tc_on(small, [m.w; 3], [m.gs; 3], 1, x, y, ints.p(m.cols), ints.p(ld), ints.p(m.rows), m.rows, t);
    }
    let (blocks, threads) = kernels::glm5_moe::fp4_launch(m.rows, m.cols);
    launch_v(kn.moe.fp4, blocks, t as u32, 1, threads, &[m.w, x, m.gs, y, ints.p(m.cols), ints.p(ld), ints.p(m.rows)]);
}

/// [`fp4_gemv`] of three `[rows, cols]` matrices on one `x` in ONE launch (`glm5_gemv_fp4_x3`):
/// matrix `m` writes the columns `m * rows ..` of `y` (row stride `ldy`), each output bit-identical
/// to its own `fp4_gemv` (#186: on the tensor-core GEMM as `fp4_gemv`). The KDA q|k|v projections of a call.
#[cfg(test)]
unsafe fn fp4_gemv_x3(kn: &Glm5Kernels, ints: &Ints, ms: [&GpuNvfp4; 3], x: Dev, y: Dev, t: usize, ldy: usize) {
    fp4_gemv_x3_of(kn, ints, ms, x, y, t, ldy, t)
}

/// #196: [`fp4_gemv_x3`] for `t` rows of a call of `call_t` rows (the kernel picked by `call_t`,
/// as [`fp4_gemv_of`])
#[allow(clippy::too_many_arguments)]
unsafe fn fp4_gemv_x3_of(kn: &Glm5Kernels, ints: &Ints, ms: [&GpuNvfp4; 3], x: Dev, y: Dev, t: usize, ldy: usize, call_t: usize) {
    let (rows, cols) = (ms[0].rows, ms[0].cols);
    assert!(ms.iter().all(|m| m.rows == rows && m.cols == cols), "glm5_model: q|k|v of different shapes");
    let [a, b, c] = ms;
    if kn.moe.dense_tc(call_t) {
        let small = kernels::glm5_moe::tc_small(call_t);
        return kn.moe.gemm_tc_on(small, [a.w, b.w, c.w], [a.gs, b.gs, c.gs], 3, x, y, ints.p(cols), ints.p(ldy), ints.p(rows), rows, t);
    }
    let (blocks, threads) = kernels::glm5_moe::fp4_launch(rows, cols);
    launch_v(kn.moe.fp4_x3, 3 * blocks, t as u32, 1, threads, &[a.w, b.w, c.w, a.gs, b.gs, c.gs, x, y, ints.p(cols), ints.p(ldy), ints.p(rows)]);
}

/// What one mHC site of a call produced, appended call by call (`[rows][..]` in row order): the
/// harness's taps on the sub-block boundaries the runner captures (`--capture-subblocks`).
#[derive(Clone, Debug, Default, PartialEq)]
pub struct SiteCap {
    /// `[rows][4]`: not in the runner's capture (HF's hyper-connection does not return it)
    pub pre: Vec<f32>,
    /// `[rows][4]`
    pub post: Vec<f32>,
    /// `[rows][4][4]`, `[j][i]` = source j, destination i (HF's layout)
    pub comb: Vec<f32>,
    /// `[rows][hidden]`: the collapse, before the layernorm
    pub collapsed: Vec<f32>,
    /// `[rows][hidden]`: the sub-layer output
    pub out: Vec<f32>,
    /// `[rows][4][hidden]`: the streams after the expand
    pub expanded: Vec<f32>,
}

/// The harness's hooks into [`Glm5Pass::call`]: the two sites' taps, and optionally the FFN
/// site's input for this call (`[t][4][hidden]`, e.g. the golden `l<k>-ffn_hc-in.f32` rows), which
/// replaces the attention site's expanded streams so the FFN site is judged on the golden input.
#[derive(Clone, Debug, Default)]
pub struct Taps {
    pub attn: SiteCap,
    pub ffn: SiteCap,
    pub ffn_in: Option<Vec<f32>>,
}

/// #175: the expert hook of [`Glm5Pass::call_with_experts`]: `(layer, selected ids [t][topk]
/// i32)` -> the device `[E]` u64 record table the MUL1 kernels read for this call
pub type ExpertHook<'a> = dyn FnMut(usize, &[i32]) -> Result<Dev, String> + 'a;

/// #186: the expert hook of [`Glm5Pass::call_with_expert_batches`]: `(layer, selection
/// [rows][topk] i32, run)`. The hook puts the records of consecutive row sub-batches in place and
/// calls `run(row0, rows, table)` for each, in row order, every row exactly once; `run` queues
/// that sub-batch's experts through the device `[E]` u64 `table`. The prompt call hands it the
/// pseudo-rows of its expert-major schedule ([`ExpertMajor::sel`]), not its token rows.
pub type ExpertBatchHook<'a> = dyn FnMut(usize, &[i32], &mut dyn FnMut(usize, usize, Dev) -> Result<(), String>) -> Result<(), String> + 'a;

/// #196: rows of one KDA sub-block of a prompt call (the KDA scratch's rows; the recurrent state
/// carries from one sub-block to the next, `glm5_kda::prompt_rows_with`)
pub const KDA_SUB_ROWS: usize = 1024;
/// #196: rows of one MLA sub-call of a prompt call (the MLA scratch's rows, so the indexer's pool
/// scores are a fixed block of 256 rows x `cap / 4`, exllamav3's `mla_attn.py` block); the
/// sub-calls run in position order, each storing its cache rows before it selects and attends
pub const MLA_SUB_ROWS: usize = 256;

/// #196: the (KDA, MLA) scratch rows of a pass of calls of up to `max_t` rows
pub fn attn_rows(max_t: usize) -> (usize, usize) {
    (max_t.min(KDA_SUB_ROWS), max_t.min(MLA_SUB_ROWS))
}

/// #196: the bytes of a pass's attention region at (KDA, MLA) scratch rows `rows` over caches of
/// `cap` rows: the KDA and the MLA activations share it (a layer is one or the other), so it is
/// the larger of the two (exllamav3 reserves the max over layers, not the sum)
pub fn attn_region_bytes(g: &Glm5Geo, rows: (usize, usize), cap: usize) -> usize {
    KdaScratch::region_bytes(&KdaDims::of(g), rows.0).max(MlaScratch::region_bytes(&MlaDims::of(g), rows.1, cap))
}

/// The per-sequence state and scratch of a layer-at-a-time pass over up to `cap` rows in calls
/// of up to `max_t` rows: one mHC plan, one KDA state, one MLA cache, the KDA and MLA scratch in
/// one shared region at their sub-block rows ([`attn_rows`], #196), the FFN plans per call size
/// and the prompt call's expert-major plan with the dense FFN over its region. Every launch
/// queues on the current stream.
pub struct Glm5Pass {
    pub g: Glm5Geo,
    pub kn: Glm5Kernels,
    pub moe: MoeGeo,
    pub max_t: usize,
    pub cap: usize,
    mhc: glm5_mhc::Plan,
    kda_st: KdaState,
    kda_sc: KdaScratch,
    mla_sc: MlaScratch,
    mla_c: MlaCache,
    /// #196: the region `kda_sc` and `mla_sc` are views into ([`attn_region_bytes`])
    attn: Dev,
    /// #196: whole calls (test reference): the attention scratch at `max_t` rows and one gate /
    /// up piece for every combo, the pre-#196 scratch
    whole: bool,
    /// #196: rows of the last MLA call (its selection, [`Glm5Pass::selection`])
    last_mla_t: usize,
    ints: Ints,
    /// `[2]` i32: the slot `gm_rmsnorm` takes (reads none of it)
    st2: Dev,
    /// `[max_t][hidden]`: the collapsed, normed sublayer input
    pub collapsed: Dev,
    /// `[max_t][hidden]`: the sublayer output
    pub sub: Dev,
    moe_plans: Vec<GpuMoePlan>,
    /// the expert-major MoE plan of prompt calls (`call_with_expert_batches`), `max_t` rows
    grouped: Option<GpuMoeGroupedPlan>,
    dense_plans: Vec<GpuFfnPlan>,
    last_ffn_t: usize,
    /// #149 path B (`CROW_GLM_FLAGS=1`, set by `Glm5Run`): the router ids of a MoE call reach the
    /// host through mapped memory behind a flag instead of a stream sync + blocking copy; `None`
    /// (every other user of the pass) is the sync
    pub routed: Option<crate::glm5_flags::Routed>,
    /// `CROW_GLM_SHARED_OVERLAP=1` (set by `Glm5Run`): a decode call with [`Glm5Pass::routed`]
    /// queues the shared expert between the publish and the host's wait (outside graph captures)
    pub overlap: bool,
    /// `CROW_GLM_CONTROLLER=1` (set by `Glm5Run`): while `active`, a decode call's MoE layer hands
    /// its routing to the controller's ring and waits for the reply on the device
    pub ctl: Option<crate::glm5_flags::Ctl>,
}

impl Glm5Pass {
    /// # Safety
    /// A CUDA context is current.
    pub unsafe fn new(g: &Glm5Geo, moe: MoeGeo, max_t: usize, cap: usize) -> Glm5Pass {
        Glm5Pass::new_with(g, moe, max_t, cap, false)
    }

    /// #196 test reference: [`Glm5Pass::new`] with the pre-#196 scratch (whole calls: the KDA
    /// and MLA scratch at `max_t` rows, one gate / up piece for every combo), the same algorithm
    /// in one piece per call
    ///
    /// # Safety
    /// As [`Glm5Pass::new`].
    #[cfg(test)]
    pub(crate) unsafe fn new_whole(g: &Glm5Geo, moe: MoeGeo, max_t: usize, cap: usize) -> Glm5Pass {
        Glm5Pass::new_with(g, moe, max_t, cap, true)
    }

    unsafe fn new_with(g: &Glm5Geo, moe: MoeGeo, max_t: usize, cap: usize, whole: bool) -> Glm5Pass {
        assert!(max_t >= 1 && cap >= max_t, "glm5_model: calls of {max_t} rows over {cap}");
        let mut kn = Glm5Kernels::new(g);
        // #186: the tensor-core path of prompt calls (`CROW_GLM_DENSE_GEMM`)
        kn.moe.set_dense_tc(kernels::glm5_moe::dense_tc_for(max_t));
        let kd = KdaDims::of(g);
        let md = MlaDims::of(g);
        let h = g.hidden;
        let kda_st = KdaState::alloc(&kd);
        kda_st.reset();
        let (attn, kda_sc, mla_sc) = Glm5Pass::attn_scratch(g, max_t, cap, whole);
        Glm5Pass {
            g: *g,
            moe,
            max_t,
            cap,
            mhc: glm5_mhc::Plan::new(h, max_t),
            kda_sc,
            kda_st,
            mla_sc,
            mla_c: MlaCache::new(&md, cap),
            attn,
            whole,
            last_mla_t: 0,
            // every K, output stride and row count `fp4_gemv` passes (#191: rows too)
            ints: Ints::new(&[h, kd.width(), kd.conv_ch(), md.q_lora, md.heads * md.v, md.heads * md.nope, md.kv_lora]),
            st2: cuda::to_i32_dev(&[0i32, 0]),
            collapsed: cuda::alloc_named("glm5 collapsed", max_t * h * 4),
            sub: cuda::alloc_named("glm5 sublayer out", max_t * h * 4),
            moe_plans: Vec::new(),
            grouped: None,
            dense_plans: Vec::new(),
            last_ffn_t: 0,
            routed: None,
            overlap: false,
            ctl: None,
            kn,
        }
    }

    /// #196: the attention region and the KDA / MLA scratch views into it for calls of up to
    /// `max_t` rows (`whole`: at `max_t` rows, else [`attn_rows`]); the MLA selection holds
    /// `max_t` rows
    unsafe fn attn_scratch(g: &Glm5Geo, max_t: usize, cap: usize, whole: bool) -> (Dev, KdaScratch, MlaScratch) {
        let rows = if whole { (max_t, max_t) } else { attn_rows(max_t) };
        let attn = cuda::alloc_named("glm5 attention scratch (KDA | MLA)", attn_region_bytes(g, rows, cap));
        (attn, KdaScratch::alloc_in(&KdaDims::of(g), rows.0, attn), MlaScratch::new_in(&MlaDims::of(g), rows.1, max_t, cap, attn))
    }

    /// #196: the prompt call's expert-major plan of `max_t` rows (made on first use), whose region
    /// also holds the dense FFN of a dense layer's call
    ///
    /// # Safety
    /// A CUDA context is current; no capture is open.
    unsafe fn grouped_plan(&mut self) -> &GpuMoeGroupedPlan {
        if self.grouped.is_none() {
            let piece = if self.whole { usize::MAX } else { crate::glm5_moe::GROUP_PIECE_COMBOS };
            self.grouped = Some(GpuMoeGroupedPlan::with_piece(&self.moe, self.max_t, &self.kn.mul1, piece));
        }
        self.grouped.as_ref().unwrap()
    }

    /// #196: the attention sub-block of a call of layer `lw`: rows `pos0 .. pos0 + t` of
    /// `collapsed` into `sub`. KDA: the decode step (`decode`, one row) or the prompt path in
    /// sub-blocks of the KDA scratch's rows with the state carried (`glm5_kda::prompt_rows_with`);
    /// MLA: sub-calls of the MLA scratch's rows in position order, each appending its cache rows
    /// before it selects and attends, its selection at its rows of the call's and the call's
    /// split count (`MlaScratch::forward_rows_with`). The projections pick their kernel by the
    /// call's `t` (`CROW_GLM_DENSE_GEMM`), so every row has the bits of one call of `t` rows on
    /// a scratch of `t` rows.
    ///
    /// # Safety
    /// As [`Glm5Pass::call`].
    unsafe fn attention(&mut self, lw: &LayerW, pos0: usize, t: usize, decode: bool) {
        let (kn, ints) = (&self.kn, &self.ints);
        match &lw.attn {
            AttnW::Kda(a) => {
                let mut proj = |p: KdaProj, xi: Dev, yo: Dev, tt: usize| match p {
                    KdaProj::Qkv => {
                        let cc = kn.kda.d.conv_ch();
                        fp4_gemv_x3_of(kn, ints, [&a.q, &a.k, &a.v], xi, yo, tt, cc, t);
                    }
                    KdaProj::O => fp4_gemv_of(kn, ints, &a.o, xi, yo, tt, None, t),
                };
                if decode {
                    glm5_kda::step_with(&kn.kda, &a.w, &self.kda_st, &self.kda_sc, self.collapsed, self.sub, &mut proj);
                } else {
                    glm5_kda::prompt_rows_with(&kn.kda, &a.w, &self.kda_st, &self.kda_sc, self.collapsed, t, self.sub, &mut proj);
                }
            }
            AttnW::Mla(a) => {
                let mut proj = |s: &MlaScratch, p: MlaProj, xi: Dev, yo: Dev| {
                    let m = match p {
                        MlaProj::QA => &a.q_a,
                        MlaProj::QB => &a.q_b,
                        MlaProj::KVA => &a.kv_a,
                        MlaProj::O => &a.o,
                    };
                    fp4_gemv_of(kn, ints, m, xi, yo, s.t(), None, t);
                };
                let (rb, step) = ((self.g.hidden * 4) as u64, self.mla_sc.max_t);
                let mut r = 0;
                while r < t {
                    let n = step.min(t - r);
                    let (xi, yo) = (self.collapsed + r as u64 * rb, self.sub + r as u64 * rb);
                    self.mla_sc.forward_rows_with(&kn.mla, &a.w, &self.mla_c, xi, yo, pos0 + r, n, r, t, &mut proj);
                    r += n;
                }
                self.last_mla_t = t;
            }
        }
    }

    /// Calls of up to `max_t` rows from now on: the row-sized scratch (mHC, KDA chunk, MLA call,
    /// `collapsed` / `sub`) freed and allocated again at `max_t` rows, the FFN plans of larger
    /// calls and the expert-major plan dropped (made again on first use). The KDA state, the MLA
    /// cache and every weight stay. `CROW_GLM_ARENA` elastic: the prompt phase borrows its scratch
    /// this way (`Glm5Run::prefill_with`) and gives it back after the prompt.
    ///
    /// # Safety
    /// A CUDA context is current; no launch reading the scratch is pending (it synchronizes);
    /// no capture is open.
    pub unsafe fn set_max_t(&mut self, max_t: usize) {
        assert!(max_t >= 1 && self.cap >= max_t, "glm5_model: calls of {max_t} rows over {}", self.cap);
        if max_t == self.max_t {
            return;
        }
        cuda::sync();
        let h = self.g.hidden;
        self.mhc.free();
        self.kda_sc.free();
        self.mla_sc.free();
        cuda::free_dev(&mut self.attn);
        cuda::free_dev(&mut self.collapsed);
        cuda::free_dev(&mut self.sub);
        if let Some(mut p) = self.grouped.take() {
            p.free();
        }
        for p in self.moe_plans.iter_mut().filter(|p| p.tokens > max_t) {
            p.free();
        }
        self.moe_plans.retain(|p| p.tokens <= max_t);
        for p in self.dense_plans.iter_mut().filter(|p| p.tokens > max_t) {
            p.free();
        }
        self.dense_plans.retain(|p| p.tokens <= max_t);
        self.mhc = glm5_mhc::Plan::new(h, max_t);
        (self.attn, self.kda_sc, self.mla_sc) = Glm5Pass::attn_scratch(&self.g, max_t, self.cap, self.whole);
        self.collapsed = cuda::alloc_named("glm5 collapsed", max_t * h * 4);
        self.sub = cuda::alloc_named("glm5 sublayer out", max_t * h * 4);
        self.kn.moe.set_dense_tc(kernels::glm5_moe::dense_tc_for(max_t));
        self.max_t = max_t;
    }

    /// a new layer of the same sequence: the KDA state and conv window back to zero (each layer
    /// has its own; the MLA cache rows are rewritten by the layer's calls before they are read)
    ///
    /// # Safety
    /// No launch of the previous layer is pending on another stream.
    pub unsafe fn begin_layer(&mut self) {
        self.kda_st.reset();
    }

    /// #190: make the FFN plans of `t`-row calls now (`call_inner` makes them on first use; a
    /// `CROW_GLM_GRAPH` capture may not allocate)
    ///
    /// # Safety
    /// A CUDA context is current; no capture is open.
    pub unsafe fn ensure_plans(&mut self, t: usize) {
        let g = self.g;
        let has = |k: FfnKind| (0..g.layers).any(|l| ffn_kind(&g, l) == k);
        if has(FfnKind::Dense) && !self.dense_plans.iter().any(|p| p.tokens == t) {
            self.dense_plans.push(GpuFfnPlan::new(g.hidden, g.dense_inter, t, g.swiglu_limit as f32));
        }
        if has(FfnKind::Moe) && !self.moe_plans.iter().any(|p| p.tokens == t) {
            self.moe_plans.push(GpuMoePlan::new(&self.moe, t));
        }
    }

    /// #190: the device `[pos0, t]` every MLA call of this pass reads (`CROW_GLM_GRAPH` stages it)
    pub fn mla_st(&self) -> Dev {
        self.mla_sc.st_dev()
    }

    /// One call of layer `lw`: rows `pos0 .. pos0 + t` of the sequence, `x` `[t][4][hidden]` f32
    /// updated in place. `decode` takes KDA's recurrent single-row step (t = 1); a prompt call
    /// takes its chunk path. Queued on the current stream, no host sync.
    ///
    /// # Safety
    /// `x` holds `t` rows; the calls of a layer come in position order from 0.
    pub unsafe fn call(&mut self, lw: &LayerW, x: Dev, pos0: usize, t: usize, decode: bool) {
        self.call_tapped(lw, x, pos0, t, decode, None);
    }

    /// the coefficients and the collapse of the site just computed (synchronizes)
    unsafe fn tap_coeffs(&self, c: &mut SiteCap, t: usize) {
        cuda::sync();
        c.pre.extend(cuda::dtoh(self.mhc.pre, t * HC));
        c.post.extend(cuda::dtoh(self.mhc.post, t * HC));
        c.comb.extend(cuda::dtoh(self.mhc.comb, t * HC * HC));
        c.collapsed.extend(cuda::dtoh(self.collapsed, t * self.g.hidden));
    }

    /// [`Glm5Pass::call`] with the harness's [`Taps`]: every tap synchronizes, so a tapped call is
    /// a measurement path, not the decode path.
    ///
    /// # Safety
    /// As [`Glm5Pass::call`]; `taps.ffn_in`, when set, holds `t x 4 x hidden` values.
    pub unsafe fn call_tapped(&mut self, lw: &LayerW, x: Dev, pos0: usize, t: usize, decode: bool, taps: Option<&mut Taps>) {
        self.call_inner(lw, x, pos0, t, decode, taps, None).expect("glm5_model: a call without an expert hook cannot fail");
    }

    /// #175: [`Glm5Pass::call`] for a layer loaded without its expert records
    /// ([`load_layer_without_experts`]). In a MoE layer the router runs first; the host reads the
    /// selected ids (`[t][topk]` i32, pick order; synchronizes, or with [`Glm5Pass::routed`] waits
    /// for their flag in mapped memory) and hands them with the layer
    /// index to `experts`, which puts those records where the MUL1 kernels can read them and
    /// returns the device `[E]` u64 table of record bases; the experts then run through that
    /// table. The launches are those of `call`, in the same order: only the table differs.
    ///
    /// # Safety
    /// As [`Glm5Pass::call`]; every table entry of a selected id is a readable record base until
    /// the next call.
    pub unsafe fn call_with_experts(&mut self, lw: &LayerW, x: Dev, pos0: usize, t: usize, decode: bool, experts: &mut ExpertHook) -> Result<(), String> {
        self.call_inner(lw, x, pos0, t, decode, None, Some(experts))
    }

    /// #175: swap this pass's KDA state with `s` (the token-by-token path keeps one per KDA layer
    /// and swaps it in around the layer's call instead of [`Glm5Pass::begin_layer`])
    pub fn swap_kda_state(&mut self, s: &mut KdaState) {
        std::mem::swap(&mut self.kda_st, s);
    }

    /// #175: swap this pass's MLA cache with `c` (one per DSA layer in the token-by-token path);
    /// `c` must have the pass's capacity
    pub fn swap_mla_cache(&mut self, c: &mut MlaCache) {
        assert_eq!(c.cap, self.cap, "glm5_model: an MLA cache of {} tokens on a pass of {}", c.cap, self.cap);
        std::mem::swap(&mut self.mla_c, c);
    }

    #[allow(clippy::too_many_arguments)]
    unsafe fn call_inner(&mut self, lw: &LayerW, x: Dev, pos0: usize, t: usize, decode: bool, mut taps: Option<&mut Taps>, experts: Option<&mut ExpertHook>) -> Result<(), String> {
        assert!((1..=self.max_t).contains(&t) && pos0 + t <= self.cap, "glm5_model: call rows {pos0}..{} (max_t {}, cap {})", pos0 + t, self.max_t, self.cap);
        assert!(!decode || t == 1, "glm5_model: a decode call is one row, got {t}");
        let h = self.g.hidden;
        let kn = &self.kn;
        // attention site; CROW_GLM_HCFUSE folds the norm into the site (untapped: taps read collapsed before it)
        let fuse = taps.is_none() && glm5_mhc::hcfuse();
        if fuse { self.mhc.coeffs_norm(&kn.mhc, &lw.attn_hc, x, lw.input_norm, self.collapsed, t) } else { self.mhc.coeffs(&kn.mhc, &lw.attn_hc, x, self.collapsed, t) }
        if let Some(tp) = taps.as_deref_mut() {
            self.tap_coeffs(&mut tp.attn, t);
        }
        if !fuse { kn.mla.rmsnorm_rows(self.collapsed, lw.input_norm, h, t, self.st2) }
        self.attention(lw, pos0, t, decode);
        let kn = &self.kn;
        if let Some(tp) = taps.as_deref_mut() {
            cuda::sync();
            tp.attn.out.extend(cuda::dtoh(self.sub, t * h));
        }
        self.mhc.expand(&kn.mhc, x, self.sub, x, t);
        if let Some(tp) = taps.as_deref_mut() {
            cuda::sync();
            tp.attn.expanded.extend(cuda::dtoh(x, t * HC * h));
            if let Some(g) = tp.ffn_in.take() {
                assert_eq!(g.len(), t * HC * h, "glm5_model: ffn_in holds {} values, the call {t} rows", g.len());
                cuda::to_f32_into(x, &g);
            }
        }
        // FFN site
        if fuse { self.mhc.coeffs_norm(&kn.mhc, &lw.ffn_hc, x, lw.post_norm, self.collapsed, t) } else { self.mhc.coeffs(&kn.mhc, &lw.ffn_hc, x, self.collapsed, t) }
        if let Some(tp) = taps.as_deref_mut() {
            self.tap_coeffs(&mut tp.ffn, t);
        }
        if !fuse { kn.mla.rmsnorm_rows(self.collapsed, lw.post_norm, h, t, self.st2) }
        match &lw.ffn {
            FfnW::Dense(w) => {
                if !self.dense_plans.iter().any(|p| p.tokens == t) {
                    self.dense_plans.push(GpuFfnPlan::new(h, self.g.dense_inter, t, self.g.swiglu_limit as f32));
                }
                let p = self.dense_plans.iter().find(|p| p.tokens == t).unwrap();
                p.run(&self.kn.k, &self.kn.moe, w, self.collapsed, self.sub);
            }
            FfnW::Moe { w, table, .. } => {
                if !self.moe_plans.iter().any(|p| p.tokens == t) {
                    self.moe_plans.push(GpuMoePlan::new(&self.moe, t));
                }
                let p = self.moe_plans.iter().find(|p| p.tokens == t).unwrap();
                match experts {
                    None => {
                        assert!(*table != 0, "glm5_model: layer {} was loaded without its expert records: call it through call_with_experts", lw.layer);
                        p.run(&self.kn.k, &self.kn.mul1, &self.kn.moe, w, *table, self.collapsed, self.sub);
                    }
                    Some(hook) => {
                        // CROW_GLM_PREFETCH (one row): the next layer's guess around the router
                        let guess = if t == 1 { self.routed.as_mut() } else { None };
                        if let Some(r) = guess {
                            r.predict_early(&self.kn.k, &self.kn.moe, lw.layer, self.collapsed);
                            p.route(&self.kn.k, &self.kn.moe, w, self.collapsed);
                            r.predict(&self.kn.k, &self.kn.moe, lw.layer, self.collapsed);
                        } else {
                            p.route(&self.kn.k, &self.kn.moe, w, self.collapsed);
                        }
                        // CROW_GLM_CONTROLLER: the request into the ring, the reply waited for on the
                        // device, the experts through the layer's fixed table; no host hand-off
                        if let Some(c) = self.ctl.as_mut().filter(|c| c.active && t == 1) {
                            let tb = c.tables.get(lw.layer).copied().unwrap_or(0);
                            if tb == 0 {
                                return Err(format!("layer {}: {}: no record table for this layer", lw.layer, crate::glm5_flags::ENV_CONTROLLER));
                            }
                            // at most CTL_AHEAD layers queued behind a device wait: a whole row
                            // queued at once filled the launch queue and starved the controller
                            c.pace().map_err(|e| format!("layer {}: {e}", lw.layer))?;
                            // the CPU lane: the MoE input row to the host before the request
                            let lane = c.lane;
                            if let Some(dl) = lane.as_ref() {
                                dl.queue_x(self.collapsed, h);
                            }
                            // the routing weights travel with the ids: the CPU lane sums its experts weighted
                            c.publish(lw.layer, p.ids, t * self.moe.topk, self.routed.as_ref().and_then(|r| r.guess_bufs()), if lane.is_some() { p.wts } else { 0 });
                            if self.overlap {
                                // the shared expert while the controller serves the layer (bit-identical)
                                p.shared_early(&self.kn.k, &self.kn.moe, w, self.collapsed);
                            }
                            c.wait_reply();
                            if c.early {
                                // #202: the reply came at once; each late expert waits for its own landing
                                c.experts_early(p, &self.kn.k, &self.kn.mul1, &self.kn.moe, w, tb, self.collapsed, self.sub);
                            } else {
                                p.experts(&self.kn.k, &self.kn.mul1, &self.kn.moe, w, tb, self.collapsed, self.sub);
                            }
                            // the CPU lane: its one summed row added behind the combine (the GPU
                            // computed the CPU's combos from a zeroed record: they added 0)
                            if let Some(dl) = lane.as_ref() {
                                c.combine_lane(dl, self.sub, h);
                            }
                            self.last_ffn_t = t;
                            self.mhc.expand(&self.kn.mhc, x, self.sub, x, t);
                            return Ok(());
                        }
                        // CROW_GLM_SHARED_OVERLAP: decided before the seam (a capture is open up to it)
                        let overlap = self.overlap && t == 1 && self.routed.is_some() && !crate::glm5_graph::capturing();
                        // #202 CROW_GLM_RT2: x to the CPU lane's host buffer ahead of the publish,
                        // so the host has it once it saw the router's flag
                        if t == 1 && crate::glm5_moe::lane::X_EARLY.load(std::sync::atomic::Ordering::Relaxed) && !crate::glm5_graph::capturing() {
                            p.lane_x_early(self.collapsed);
                        }
                        crate::glm5_graph::seam_end(lw.layer, p.ids, t * self.moe.topk)?;
                        let ids = match self.routed.as_mut().filter(|_| overlap) {
                            Some(r) => {
                                r.publish(p.ids, t * self.moe.topk);
                                p.shared_early(&self.kn.k, &self.kn.moe, w, self.collapsed);
                                cuda::stream_query(cuda::cur_stream());
                                r.wait_layer(lw.layer).map_err(|e| {
                                    p.shared_forget();
                                    format!("layer {}: {e}", lw.layer)
                                })?
                            }
                            None => router_ids(self.routed.as_mut(), p.ids, t * self.moe.topk, lw.layer)?,
                        };
                        let tb = hook(lw.layer, &ids).inspect_err(|_| p.shared_forget())?;
                        crate::glm5_graph::seam_begin(tb);
                        p.experts(&self.kn.k, &self.kn.mul1, &self.kn.moe, w, tb, self.collapsed, self.sub);
                    }
                }
            }
        }
        self.last_ffn_t = t;
        if let Some(tp) = taps.as_deref_mut() {
            cuda::sync();
            tp.ffn.out.extend(cuda::dtoh(self.sub, t * h));
        }
        self.mhc.expand(&self.kn.mhc, x, self.sub, x, t);
        if let Some(tp) = taps {
            cuda::sync();
            tp.ffn.expanded.extend(cuda::dtoh(x, t * HC * h));
        }
        Ok(())
    }

    /// the DSA selection of the last MLA call, one ascending token list per row (synchronizes)
    ///
    /// # Safety
    /// The last call was an MLA call.
    pub unsafe fn selection(&self) -> Vec<Vec<usize>> {
        cuda::sync();
        let (t, w) = (self.last_mla_t, MlaDims::of(&self.g).sel_max());
        let n = cuda::dtoh_i32(self.mla_sc.sel_n, t);
        let s = cuda::dtoh_i32(self.mla_sc.sel, t * w);
        (0..t)
            .map(|r| {
                let mut v: Vec<usize> = s[r * w..r * w + n[r] as usize].iter().map(|&e| e as usize).collect();
                v.sort_unstable();
                v
            })
            .collect()
    }

    /// the routing of the last MoE call (synchronizes)
    ///
    /// # Safety
    /// The last call ran a MoE layer.
    pub unsafe fn routing(&self) -> Routing {
        if let Some(p) = self.moe_plans.iter().find(|p| p.tokens == self.last_ffn_t) {
            return p.read_routing();
        }
        // a prompt call (`call_with_expert_batches`) routed its `last_ffn_t` rows on the grouped plan
        let p = self.grouped.as_ref().expect("glm5_model: no MoE call yet");
        cuda::sync();
        let c = self.last_ffn_t * self.moe.topk;
        Routing { topk: self.moe.topk, ids: cuda::dtoh_i32(p.ids, c).into_iter().map(|v| v as u32).collect(), weights: cuda::dtoh(p.wts, c) }
    }

    /// # Safety
    /// No launch of this pass is pending.
    pub unsafe fn free(&mut self) {
        self.mhc.free();
        self.kda_st.free();
        self.kda_sc.free();
        self.mla_sc.free();
        self.mla_c.free();
        for p in self.moe_plans.iter_mut() {
            p.free();
        }
        if let Some(p) = self.grouped.as_mut() {
            p.free();
        }
        for p in self.dense_plans.iter_mut() {
            p.free();
        }
        if let Some(r) = self.routed.as_mut() {
            r.free();
        }
        for d in [&mut self.ints.dev, &mut self.st2, &mut self.collapsed, &mut self.sub, &mut self.attn] {
            cuda::free_dev(d);
        }
    }
}

/// #190: a MoE call's `n` router ids (`[t][topk]` i32 at `ids`, pick order) to the host: a stream
/// sync + blocking copy, or with `routed` (`CROW_GLM_FLAGS`) the publish and its flag. The one
/// hand-off of `Glm5Pass::call_with_experts` and of the `CROW_GLM_GRAPH` replay.
///
/// # Safety
/// A CUDA context is current; `ids` holds `n` i32 written by work queued before; no capture is open.
pub unsafe fn router_ids(routed: Option<&mut crate::glm5_flags::Routed>, ids: Dev, n: usize, layer: usize) -> Result<Vec<i32>, String> {
    match routed {
        None => {
            cuda::sync();
            Ok(cuda::dtoh_i32(ids, n))
        }
        Some(r) => {
            r.publish(ids, n);
            r.wait_layer(layer).map_err(|e| format!("layer {layer}: {e}"))
        }
    }
}

/// #186: the prompt call with its experts served in sub-batches, expert-major.
impl Glm5Pass {
    /// #186: one prompt call (`decode` false) of a layer loaded without its expert records. The
    /// launches of [`Glm5Pass::call_with_experts`] for a prompt call, in the same order, up to the
    /// FFN (#196: the attention in sub-blocks, [`attn_rows`]). A dense FFN runs on the one dense
    /// plan of `max_t` rows over the expert-major plan's region ([`GpuMoeGroupedPlan::dense`]).
    /// The MoE runs
    /// expert-major on the pass's [`GpuMoeGroupedPlan`] of `max_t` rows: the router on the `t`
    /// rows, the host reads their ids (one stream sync for the call) and builds the
    /// [`ExpertMajor`] schedule, `experts` serves its pseudo-rows (every selected expert in
    /// exactly one sub-batch, fetched once for the call) and each sub-batch runs every row routed
    /// to its experts (`mul1_gemm_grp`, the expert decoded once per tile of 16 rows); the shared
    /// expert and the combine follow once. Every FFN row has the bits of the per-combo path
    /// ([`GpuMoePlan`], [`moe_rows`]), so each row gets the bits of one `t`-row call.
    ///
    /// # Safety
    /// As [`Glm5Pass::call`]; every table entry of a sub-batch's selected id is a readable record
    /// base until `run` for the next sub-batch is called or the call returns.
    pub unsafe fn call_with_expert_batches(&mut self, lw: &LayerW, x: Dev, pos0: usize, t: usize, experts: &mut ExpertBatchHook) -> Result<(), String> {
        assert!((1..=self.max_t).contains(&t) && pos0 + t <= self.cap, "glm5_model: call rows {pos0}..{} (max_t {}, cap {})", pos0 + t, self.max_t, self.cap);
        let h = self.g.hidden;
        // attention site: `call_inner`'s launches for a prompt call, without taps
        let fuse = glm5_mhc::hcfuse();
        let kn = &self.kn;
        if fuse { self.mhc.coeffs_norm(&kn.mhc, &lw.attn_hc, x, lw.input_norm, self.collapsed, t) } else { self.mhc.coeffs(&kn.mhc, &lw.attn_hc, x, self.collapsed, t) }
        if !fuse { kn.mla.rmsnorm_rows(self.collapsed, lw.input_norm, h, t, self.st2) }
        self.attention(lw, pos0, t, false);
        let kn = &self.kn;
        self.mhc.expand(&kn.mhc, x, self.sub, x, t);
        // FFN site
        if fuse { self.mhc.coeffs_norm(&kn.mhc, &lw.ffn_hc, x, lw.post_norm, self.collapsed, t) } else { self.mhc.coeffs(&kn.mhc, &lw.ffn_hc, x, self.collapsed, t) }
        if !fuse { kn.mla.rmsnorm_rows(self.collapsed, lw.post_norm, h, t, self.st2) }
        let (collapsed, sub) = (self.collapsed, self.sub);
        let w = match &lw.ffn {
            FfnW::Dense(w) => {
                // #196: one dense FFN plan of `max_t` rows over the expert-major plan's region;
                // every launch is per row, so each row has the bits of one `t`-row plan
                self.grouped_plan();
                self.grouped.as_ref().unwrap().dense.run_rows_of(&self.kn.moe, w, collapsed, sub, t, t);
                self.last_ffn_t = t;
                self.mhc.expand(&self.kn.mhc, x, self.sub, x, t);
                return Ok(());
            }
            FfnW::Moe { w, .. } => w,
        };
        // expert-major (0xSero's glm53-flash-offload prefill): the router on the call's rows, one
        // routing sync, the call's schedule; the tiers serve its pseudo-rows (each selected expert
        // in exactly one), and each sub-batch runs every row routed to its experts
        self.grouped_plan();
        let gp = self.grouped.as_ref().unwrap();
        gp.route(&self.kn.k, &self.kn.moe, w, self.collapsed, t);
        let ids = router_ids(None, gp.ids, t * self.moe.topk, lw.layer)?;
        let s = ExpertMajor::new(&ids, self.moe.topk, self.moe.experts, GROUP_ROWS).map_err(|e| format!("layer {}: {e}", lw.layer))?;
        gp.upload(&s);
        let (sel, pr, collapsed) = (s.sel(), s.pseudo_rows(), self.collapsed);
        let mut covered = 0usize;
        let mut run = |r0: usize, rows: usize, table: Dev| -> Result<(), String> {
            if r0 != covered || rows == 0 || r0 + rows > pr {
                return Err(format!("layer {}: a sub-batch of pseudo-rows {r0}..{} after {covered} of {pr}", lw.layer, r0 + rows));
            }
            gp.experts_items(table, collapsed, s.work_of_rows(r0, rows));
            covered += rows;
            Ok(())
        };
        experts(lw.layer, &sel, &mut run)?;
        if covered != pr {
            return Err(format!("layer {}: the expert hook served {covered} of {pr} pseudo-rows", lw.layer));
        }
        gp.finish(&self.kn.moe, w, self.collapsed, self.sub, t);
        self.last_ffn_t = t;
        self.mhc.expand(&self.kn.mhc, x, self.sub, x, t);
        Ok(())
    }
}

/// #186: the routed experts of rows `r0 .. r0 + rows` of the MoE call routed in `plans[full]`
/// (`GpuMoePlan::route` ran on its `tokens` rows of `x`), through `table`, into those rows of `y`.
/// The whole call is `plans[full].experts` itself; any other row range runs in descending
/// power-of-two pieces, each on the plan of its size (made here on first use, the sizes the plan
/// books, `manager::glm5_prompt_call_sizes`) with its rows' ids and weights copied from
/// `plans[full]` (D2D). Every launch is per row or per (row, expert) combo, so each row gets the
/// bits of the whole call.
///
/// # Safety
/// A CUDA context is current; `x` and `y` hold `plans[full].tokens` rows; every table entry of
/// a selected id of these rows is a readable record base until the launches finished.
#[allow(clippy::too_many_arguments)]
pub unsafe fn moe_rows(
    kn: &kernels::Kernels,
    mk: &mul1::Kernels,
    gk: &kernels::glm5_moe::Kernels,
    moe: &MoeGeo,
    plans: &mut Vec<GpuMoePlan>,
    full: usize,
    w: &GpuMoeWeights,
    table: Dev,
    x: Dev,
    y: Dev,
    r0: usize,
    rows: usize,
) {
    let t = plans[full].tokens;
    assert!(rows >= 1 && r0 + rows <= t, "glm5_model: expert rows {r0}..{} of a {t}-row call", r0 + rows);
    if r0 == 0 && rows == t {
        plans[full].experts(kn, mk, gk, w, table, x, y);
        return;
    }
    let (h, k) = (moe.hidden, moe.topk);
    let (ids, wts) = (plans[full].ids, plans[full].wts);
    let mut r = r0;
    while r < r0 + rows {
        let left = r0 + rows - r;
        let s = 1usize << (usize::BITS - 1 - left.leading_zeros());
        if !plans.iter().any(|p| p.tokens == s) {
            plans.push(GpuMoePlan::new(moe, s));
        }
        let p = plans.iter().find(|p| p.tokens == s).unwrap();
        cuda::d2d_async(p.ids, ids + (r * k * 4) as u64, s * k * 4);
        cuda::d2d_async(p.wts, wts + (r * k * 4) as u64, s * k * 4);
        p.experts(kn, mk, gk, w, table, x + (r * h * 4) as u64, y + (r * h * 4) as u64);
        r += s;
    }
}

/// The head over `rows` rows of the residual `x` `[rows][4][hidden]`: stream mean + final norm +
/// lm_head into `logits` `[rows][vocab]`, greedy ids into `ids` `[rows]` i32.
///
/// # Safety
/// A CUDA context is current; the buffers hold those shapes.
pub unsafe fn run_head(kn: &Glm5Kernels, head: &Head, w: &HeadW, x: Dev, normed: Dev, logits: Dev, ids: Dev, rows: usize) {
    head.run(&kn.k, x, w.norm, w.lm, normed, logits, rows);
    head.argmax(&kn.k, logits, ids, rows);
}

/// the head geometry of the path (re-exported for the harness)
pub fn head_geo(g: &Glm5Geo) -> HeadGeo {
    HeadGeo::of(g)
}

// ---------------------------------------------------------------- #192: the MTP verify call

impl Glm5Pass {
    /// #192 (`CROW_GLM_MTP`): one layer over the `t` verify rows `pos0 .. pos0 + t` of a
    /// speculative step, `x` `[t][4][hidden]` updated in place, each row's result bit for bit the
    /// one-row decode call's ([`Glm5Pass::call_with_experts`] with `t = 1`, `decode`):
    ///
    /// - mHC coefficients and expand, both norms, the dense FFN, the router, the routed experts,
    ///   the shared expert and the combine run over all `t` rows (their kernels compute every row
    ///   on its own: `glm5_mhc_coeffs` / `gm_rmsnorm` a block per row, `gemv_fp4_b` and
    ///   `gemv_bf16_b` grid y = row, MUL1 slots that never interact);
    /// - the router ids of all rows reach `experts` in ONE hook call (`[t][topk]`), so the tiers
    ///   stage the union of the rows' experts once;
    /// - KDA runs the decode step row after row (`glm5_kda::step_with`; the prompt path is another
    ///   operation order), and after row `r < t - 1` the state (S and conv window) is copied into
    ///   `kda_snaps[r]` (queued, D2D): the caller restores the state of the last accepted row;
    /// - MLA runs one one-row call per row (`attn_splits(t)` would change the split-K partition);
    ///   rows of rejected positions stay in the cache and are rewritten before any later row
    ///   reads them (absolute positions, store before select).
    ///
    /// Any change to the launches of `call_inner` must be mirrored here (the GPU test
    /// `glm5_tiers::spec_tests::glm5_mtp_spec_gpu_is_lossless` compares the bits).
    ///
    /// # Safety
    /// As [`Glm5Pass::call_with_experts`]; `t <= max_t`; a KDA layer needs `t - 1` snapshot
    /// states of this pass's dims.
    pub unsafe fn call_verify_with_experts(&mut self, lw: &LayerW, x: Dev, pos0: usize, t: usize, kda_snaps: &[KdaState], experts: &mut ExpertHook) -> Result<(), String> {
        assert!((1..=self.max_t).contains(&t) && pos0 + t <= self.cap, "glm5_model: verify rows {pos0}..{} (max_t {}, cap {})", pos0 + t, self.max_t, self.cap);
        let h = self.g.hidden;
        let row = |b: Dev, r: usize| b + (r * h * 4) as u64;
        let fuse = glm5_mhc::hcfuse();
        if fuse { self.mhc.coeffs_norm(&self.kn.mhc, &lw.attn_hc, x, lw.input_norm, self.collapsed, t) } else { self.mhc.coeffs(&self.kn.mhc, &lw.attn_hc, x, self.collapsed, t) }
        if !fuse { self.kn.mla.rmsnorm_rows(self.collapsed, lw.input_norm, h, t, self.st2) }
        match &lw.attn {
            AttnW::Kda(a) => {
                assert!(kda_snaps.len() + 1 >= t, "glm5_model: {t} verify rows, {} KDA snapshot slots", kda_snaps.len());
                let (kn, ints) = (&self.kn, &self.ints);
                let w = kn.kda.d.width();
                let cc = kn.kda.d.conv_ch();
                let mut proj = |p: KdaProj, xi: Dev, yo: Dev, tt: usize| match p {
                    KdaProj::Qkv => {
                        fp4_gemv(kn, ints, &a.q, xi, yo, tt, Some(cc));
                        fp4_gemv(kn, ints, &a.k, xi, yo + (w * 4) as u64, tt, Some(cc));
                        fp4_gemv(kn, ints, &a.v, xi, yo + (2 * w * 4) as u64, tt, Some(cc));
                    }
                    KdaProj::O => fp4_gemv(kn, ints, &a.o, xi, yo, tt, None),
                };
                for r in 0..t {
                    glm5_kda::step_with(&kn.kda, &a.w, &self.kda_st, &self.kda_sc, row(self.collapsed, r), row(self.sub, r), &mut proj);
                    if r + 1 < t {
                        kda_snaps[r].copy_from(&self.kda_st);
                    }
                }
            }
            AttnW::Mla(a) => {
                let (kn, ints) = (&self.kn, &self.ints);
                let mut proj = |s: &MlaScratch, p: MlaProj, xi: Dev, yo: Dev| {
                    let m = match p {
                        MlaProj::QA => &a.q_a,
                        MlaProj::QB => &a.q_b,
                        MlaProj::KVA => &a.kv_a,
                        MlaProj::O => &a.o,
                    };
                    fp4_gemv(kn, ints, m, xi, yo, s.t(), None);
                };
                for r in 0..t {
                    self.mla_sc.forward_with(&kn.mla, &a.w, &self.mla_c, row(self.collapsed, r), row(self.sub, r), pos0 + r, 1, &mut proj);
                }
            }
        }
        self.mhc.expand(&self.kn.mhc, x, self.sub, x, t);
        if fuse { self.mhc.coeffs_norm(&self.kn.mhc, &lw.ffn_hc, x, lw.post_norm, self.collapsed, t) } else { self.mhc.coeffs(&self.kn.mhc, &lw.ffn_hc, x, self.collapsed, t) }
        if !fuse { self.kn.mla.rmsnorm_rows(self.collapsed, lw.post_norm, h, t, self.st2) }
        match &lw.ffn {
            FfnW::Dense(w) => {
                if !self.dense_plans.iter().any(|p| p.tokens == t) {
                    self.dense_plans.push(GpuFfnPlan::new(h, self.g.dense_inter, t, self.g.swiglu_limit as f32));
                }
                let p = self.dense_plans.iter().find(|p| p.tokens == t).unwrap();
                p.run(&self.kn.k, &self.kn.moe, w, self.collapsed, self.sub);
            }
            FfnW::Moe { w, .. } => {
                if !self.moe_plans.iter().any(|p| p.tokens == t) {
                    self.moe_plans.push(GpuMoePlan::new(&self.moe, t));
                }
                let p = self.moe_plans.iter().find(|p| p.tokens == t).unwrap();
                p.route(&self.kn.k, &self.kn.moe, w, self.collapsed);
                // `CROW_GLM_FLAGS` publishes one row's ids; more rows read theirs by a sync
                let routed = if t == 1 { self.routed.as_mut() } else { None };
                let ids = router_ids(routed, p.ids, t * self.moe.topk, lw.layer)?;
                let tb = experts(lw.layer, &ids)?;
                p.experts(&self.kn.k, &self.kn.mul1, &self.kn.moe, w, tb, self.collapsed, self.sub);
            }
        }
        self.last_ffn_t = t;
        self.mhc.expand(&self.kn.mhc, x, self.sub, x, t);
        Ok(())
    }
}

// ---------------------------------------------------------------- CROW_GLM_MAX_BATCH: one row per sequence

impl Glm5Pass {
    /// `CROW_GLM_MAX_BATCH`: one layer over `t = pos.len()` decode rows of `t` DIFFERENT
    /// sequences, row `r` at position `pos[r]` of its own sequence, `x` `[t][4][hidden]` updated
    /// in place, each row's result bit for bit the one-row decode call's of its sequence
    /// ([`Glm5Pass::call_with_experts`] with `t = 1`, `decode`). The launches are those of
    /// [`Glm5Pass::call_verify_with_experts`]; only the recurrent state and the cache differ:
    ///
    /// - mHC, both norms, the dense FFN, the router, the routed and the shared experts and the
    ///   combine run over all `t` rows (every row on its own, as in the verify call);
    /// - the router ids of all rows reach `experts` in ONE hook call (`[t][topk]`), so the tiers
    ///   stage the union of the rows' experts once;
    /// - KDA: row `r` takes the decode step on `kda[r]`, its sequence's state of this layer;
    /// - MLA: row `r` is a one-row call on `mla[r]`, its sequence's cache of this layer, at
    ///   `pos[r]`.
    ///
    /// Any change to the launches of `call_inner` must be mirrored here, as in the verify call
    /// (the GPU test `glm5_engine::tests_batch_gpu` compares the bits with the solo rows).
    ///
    /// # Safety
    /// As [`Glm5Pass::call_with_experts`]; `t <= max_t`; a KDA layer gets `t` states, a DSA
    /// layer `t` caches of this pass's capacity, every one of a different sequence.
    pub unsafe fn call_slots_with_experts(&mut self, lw: &LayerW, x: Dev, pos: &[usize], kda: &[&KdaState], mla: &[&MlaCache], experts: &mut ExpertHook) -> Result<(), String> {
        let t = pos.len();
        assert!((1..=self.max_t).contains(&t), "glm5_model: {t} sequence rows (max_t {})", self.max_t);
        assert!(pos.iter().all(|&p| p < self.cap), "glm5_model: sequence rows at {pos:?} (cap {})", self.cap);
        let h = self.g.hidden;
        let row = |b: Dev, r: usize| b + (r * h * 4) as u64;
        self.mhc.coeffs(&self.kn.mhc, &lw.attn_hc, x, self.collapsed, t);
        self.kn.mla.rmsnorm_rows(self.collapsed, lw.input_norm, h, t, self.st2);
        match &lw.attn {
            AttnW::Kda(a) => {
                assert_eq!(kda.len(), t, "glm5_model: {t} sequence rows, {} KDA states", kda.len());
                let (kn, ints) = (&self.kn, &self.ints);
                let w = kn.kda.d.width();
                let cc = kn.kda.d.conv_ch();
                let mut proj = |p: KdaProj, xi: Dev, yo: Dev, tt: usize| match p {
                    KdaProj::Qkv => {
                        fp4_gemv(kn, ints, &a.q, xi, yo, tt, Some(cc));
                        fp4_gemv(kn, ints, &a.k, xi, yo + (w * 4) as u64, tt, Some(cc));
                        fp4_gemv(kn, ints, &a.v, xi, yo + (2 * w * 4) as u64, tt, Some(cc));
                    }
                    KdaProj::O => fp4_gemv(kn, ints, &a.o, xi, yo, tt, None),
                };
                for (r, st) in kda.iter().enumerate() {
                    glm5_kda::step_with(&kn.kda, &a.w, st, &self.kda_sc, row(self.collapsed, r), row(self.sub, r), &mut proj);
                }
            }
            AttnW::Mla(a) => {
                assert_eq!(mla.len(), t, "glm5_model: {t} sequence rows, {} MLA caches", mla.len());
                let (kn, ints) = (&self.kn, &self.ints);
                let mut proj = |s: &MlaScratch, p: MlaProj, xi: Dev, yo: Dev| {
                    let m = match p {
                        MlaProj::QA => &a.q_a,
                        MlaProj::QB => &a.q_b,
                        MlaProj::KVA => &a.kv_a,
                        MlaProj::O => &a.o,
                    };
                    fp4_gemv(kn, ints, m, xi, yo, s.t(), None);
                };
                for (r, c) in mla.iter().enumerate() {
                    assert_eq!(c.cap, self.cap, "glm5_model: an MLA cache of {} tokens on a pass of {}", c.cap, self.cap);
                    self.mla_sc.forward_with(&kn.mla, &a.w, c, row(self.collapsed, r), row(self.sub, r), pos[r], 1, &mut proj);
                }
            }
        }
        self.mhc.expand(&self.kn.mhc, x, self.sub, x, t);
        self.mhc.coeffs(&self.kn.mhc, &lw.ffn_hc, x, self.collapsed, t);
        self.kn.mla.rmsnorm_rows(self.collapsed, lw.post_norm, h, t, self.st2);
        match &lw.ffn {
            FfnW::Dense(w) => {
                if !self.dense_plans.iter().any(|p| p.tokens == t) {
                    self.dense_plans.push(GpuFfnPlan::new(h, self.g.dense_inter, t, self.g.swiglu_limit as f32));
                }
                let p = self.dense_plans.iter().find(|p| p.tokens == t).unwrap();
                p.run(&self.kn.k, &self.kn.moe, w, self.collapsed, self.sub);
            }
            FfnW::Moe { w, .. } => {
                if !self.moe_plans.iter().any(|p| p.tokens == t) {
                    self.moe_plans.push(GpuMoePlan::new(&self.moe, t));
                }
                let p = self.moe_plans.iter().find(|p| p.tokens == t).unwrap();
                p.route(&self.kn.k, &self.kn.moe, w, self.collapsed);
                // `CROW_GLM_FLAGS` publishes one row's ids; more rows read theirs by a sync
                let routed = if t == 1 { self.routed.as_mut() } else { None };
                let ids = router_ids(routed, p.ids, t * self.moe.topk, lw.layer)?;
                let tb = experts(lw.layer, &ids)?;
                p.experts(&self.kn.k, &self.kn.mul1, &self.kn.moe, w, tb, self.collapsed, self.sub);
            }
        }
        self.last_ffn_t = t;
        self.mhc.expand(&self.kn.mhc, x, self.sub, x, t);
        Ok(())
    }
}

// ---------------------------------------------------------------- the golden harness's host side

/// The host side of `decode glmgolden` (#161): the layerwise runner's manifest
/// (`docs/glm5-reference-runner.md` section 5), the call split, the metrics and the overlaps.
pub mod golden {
    use std::path::Path;

    /// gate G3 (PREREG, plan "Tore"): cosine per layer per row group against the golden
    pub const G3: f64 = 0.9999;

    /// what the harness reads from `manifest.json`
    #[derive(Clone, Debug, PartialEq)]
    pub struct Manifest {
        pub ids: Vec<i64>,
        /// prompt rows, decode rows, all rows
        pub t: usize,
        pub d: usize,
        pub n: usize,
        pub anchors: Vec<usize>,
        /// the runner's `--prompt-chunk` (0 or absent: the prompt in one call)
        pub prompt_chunk: usize,
        pub hidden: usize,
        pub hc: usize,
        /// the layers whose `l<k>-output.f32` the manifest lists, ascending
        pub layers: Vec<usize>,
        pub files: serde_json::Value,
        /// `subblocks[layer][site][role]` -> file (`--capture-subblocks`, #156); `Null` without
        pub subblocks: serde_json::Value,
        /// `head[role]` -> file, role `mean` / `norm` / `logits` (`--capture-head`, #165); `Null` without
        pub head: serde_json::Value,
        pub complete: bool,
    }

    /// the mHC site roles the runner captures, in the harness's print order (`pre` is not one:
    /// HF's hyper-connection does not return it)
    pub const SITE_ROLES: &[&str] = &["post", "comb", "collapsed", "out", "expanded"];

    /// roles judged by G3 (vectors of a row); `post` / `comb` are coefficients, reported by max_abs
    pub fn role_is_judged(role: &str) -> bool {
        matches!(role, "collapsed" | "out" | "expanded")
    }

    impl Manifest {
        pub fn parse(text: &str) -> Result<Manifest, String> {
            let m: serde_json::Value = serde_json::from_str(text).map_err(|e| format!("manifest.json: {e}"))?;
            let u = |k: &str| m[k].as_u64().map(|v| v as usize).ok_or_else(|| format!("manifest.json: no {k}"));
            let ids: Vec<i64> = m["ids"].as_array().ok_or("manifest.json: no ids")?.iter().map(|v| v.as_i64().ok_or("manifest.json: an id is not an integer")).collect::<Result<_, _>>()?;
            let (t, d) = (u("T")?, u("D")?);
            if ids.len() != t + d {
                return Err(format!("manifest.json: {} ids, T {t} + D {d}", ids.len()));
            }
            let files = m["files"].clone();
            let mut layers: Vec<usize> = files
                .as_object()
                .ok_or("manifest.json: no files")?
                .keys()
                .filter_map(|k| k.strip_prefix('l').and_then(|r| r.strip_suffix("-output.f32")).and_then(|v| v.parse().ok()))
                .collect();
            layers.sort_unstable();
            Ok(Manifest {
                ids,
                t,
                d,
                n: t + d,
                anchors: m["anchors"].as_array().map(|a| a.iter().filter_map(|v| v.as_u64().map(|v| v as usize)).collect()).unwrap_or_default(),
                prompt_chunk: m["prompt_chunk"].as_u64().unwrap_or(0) as usize,
                hidden: u("hidden_size")?,
                hc: u("hc_mult")?,
                layers,
                files,
                subblocks: m["subblocks"].clone(),
                head: m["head"].clone(),
                complete: m["complete"].as_bool().unwrap_or(false),
            })
        }

        /// the file of `role` at `site` (`attn` / `ffn`) of layer `l`, `None` without a capture
        pub fn subblock(&self, l: usize, site: &str, role: &str) -> Option<String> {
            self.subblocks[l.to_string()][site][role].as_str().map(str::to_string)
        }

        /// the file of the head's `role` (`mean`, `norm`, `logits`) at every row, `None` without a capture
        pub fn head_file(&self, role: &str) -> Option<String> {
            self.head[role].as_str().map(str::to_string)
        }

        /// the declared shape of a file, `None` when the manifest does not list it
        pub fn shape(&self, file: &str) -> Option<Vec<usize>> {
            self.files[file]["shape"].as_array().map(|a| a.iter().filter_map(|v| v.as_u64().map(|v| v as usize)).collect())
        }

        /// the runner's calls: the prompt in calls of `prompt_chunk` rows (0 = one call), then
        /// each decode row alone: `(first row, rows, decode)`
        pub fn calls(&self) -> Vec<(usize, usize, bool)> {
            calls(self.t, self.d, self.prompt_chunk)
        }
    }

    pub fn calls(t: usize, d: usize, chunk: usize) -> Vec<(usize, usize, bool)> {
        let c = if chunk == 0 { t.max(1) } else { chunk };
        let mut v: Vec<(usize, usize, bool)> = (0..t).step_by(c).map(|r| (r, c.min(t - r), false)).collect();
        v.extend((t..t + d).map(|r| (r, 1, true)));
        v
    }

    /// raw little-endian f32 / i32 of exactly `n` values
    pub fn read_f32(path: &Path, n: usize) -> Result<Vec<f32>, String> {
        let b = std::fs::read(path).map_err(|e| format!("{}: {e}", path.display()))?;
        if b.len() != n * 4 {
            return Err(format!("{}: {} B on disk, the manifest declares {n} values = {} B", path.display(), b.len(), n * 4));
        }
        Ok(b.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect())
    }

    pub fn read_i32(path: &Path, n: usize) -> Result<Vec<i32>, String> {
        let b = std::fs::read(path).map_err(|e| format!("{}: {e}", path.display()))?;
        if b.len() != n * 4 {
            return Err(format!("{}: {} B on disk, the manifest declares {n} values = {} B", path.display(), b.len(), n * 4));
        }
        Ok(b.chunks_exact(4).map(|c| i32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect())
    }

    /// one comparison of engine vs golden over rows of `row` values
    #[derive(Clone, Copy, Debug, PartialEq)]
    pub struct Cmp {
        pub cosine: f64,
        pub max_abs: f64,
        /// `sqrt(sum (a - b)^2 / sum b^2)`
        pub rel_rms: f64,
        pub worst_row: usize,
        pub worst_row_cosine: f64,
        /// NaN or inf values in the engine output
        pub non_finite: usize,
    }

    fn cos(a: &[f32], b: &[f32]) -> f64 {
        let (mut ab, mut aa, mut bb) = (0f64, 0f64, 0f64);
        for (&x, &y) in a.iter().zip(b) {
            ab += x as f64 * y as f64;
            aa += x as f64 * x as f64;
            bb += y as f64 * y as f64;
        }
        if aa == 0.0 || bb == 0.0 {
            return if aa == bb { 1.0 } else { 0.0 };
        }
        ab / (aa.sqrt() * bb.sqrt())
    }

    /// `a` engine, `b` golden, both `[rows][row]`; row indices are relative to the slice
    pub fn compare(a: &[f32], b: &[f32], row: usize) -> Cmp {
        assert!(a.len() == b.len() && row > 0 && a.len() % row == 0, "compare: {} vs {} values in rows of {row}", a.len(), b.len());
        let (mut e2, mut r2, mut mx) = (0f64, 0f64, 0f64);
        let mut non_finite = 0;
        for (&x, &y) in a.iter().zip(b) {
            if !x.is_finite() {
                non_finite += 1;
            }
            let d = x as f64 - y as f64;
            e2 += d * d;
            r2 += y as f64 * y as f64;
            mx = f64::max(mx, d.abs());
        }
        let (mut worst_row, mut worst_row_cosine) = (0, f64::INFINITY);
        for (r, (ra, rb)) in a.chunks_exact(row).zip(b.chunks_exact(row)).enumerate() {
            let c = cos(ra, rb);
            if !(c >= worst_row_cosine) {
                worst_row = r;
                worst_row_cosine = c;
            }
        }
        let cosine = if non_finite > 0 { f64::NAN } else { cos(a, b) };
        Cmp { cosine, max_abs: mx, rel_rms: (e2 / r2.max(1e-30)).sqrt(), worst_row, worst_row_cosine, non_finite }
    }

    /// G3 on one row group: cosine >= 0.9999 and every value finite (NaN never passes)
    pub fn passes(c: &Cmp) -> bool {
        c.non_finite == 0 && c.cosine >= G3
    }

    /// routing top-k overlap: `engine` one id list per row (any order), `golden` `[rows][k]`
    /// ascending; returns (mean |engine ∩ golden| / k, rows with the identical set)
    pub fn routing_overlap(engine: &[Vec<u32>], golden: &[i32], k: usize) -> (f64, usize) {
        assert_eq!(golden.len(), engine.len() * k, "routing_overlap: {} rows vs {} golden ids at top-{k}", engine.len(), golden.len());
        let (mut sum, mut same) = (0f64, 0usize);
        for (r, e) in engine.iter().enumerate() {
            let mut g: Vec<u32> = golden[r * k..(r + 1) * k].iter().map(|&v| v as u32).collect();
            g.sort_unstable();
            let mut s = e.clone();
            s.sort_unstable();
            let hit = s.iter().filter(|v| g.binary_search(v).is_ok()).count();
            sum += hit as f64 / k as f64;
            if s == g {
                same += 1;
            }
        }
        (if engine.is_empty() { 1.0 } else { sum / engine.len() as f64 }, same)
    }

    /// DSA selection overlap: `engine` one ascending token list per row, `golden` `[rows][w]`
    /// with `-1` = empty (any order); returns (mean |engine ∩ golden| / |golden|, rows with the
    /// identical set)
    pub fn dsa_overlap(engine: &[Vec<usize>], golden: &[i32], w: usize) -> (f64, usize) {
        assert_eq!(golden.len(), engine.len() * w, "dsa_overlap: {} rows vs {} golden values at width {w}", engine.len(), golden.len());
        let (mut sum, mut same) = (0f64, 0usize);
        for (r, e) in engine.iter().enumerate() {
            let mut g: Vec<usize> = golden[r * w..(r + 1) * w].iter().filter(|&&v| v >= 0).map(|&v| v as usize).collect();
            g.sort_unstable();
            g.dedup();
            let hit = e.iter().filter(|v| g.binary_search(v).is_ok()).count();
            sum += if g.is_empty() { 1.0 } else { hit as f64 / g.len() as f64 };
            if *e == g {
                same += 1;
            }
        }
        (if engine.is_empty() { 1.0 } else { sum / engine.len() as f64 }, same)
    }

    /// the golden's greedy id of one logits row: the first index of the maximum (NaN never wins)
    pub fn argmax(row: &[f32]) -> usize {
        row.iter().enumerate().fold((0usize, f32::NEG_INFINITY), |m, (j, &x)| if x > m.1 { (j, x) } else { m }).0
    }

    /// #165 top-1 agreement: the engine's greedy ids against the golden logits `[rows][v]`; returns
    /// the rows (relative to the slice) whose ids differ, with (engine id, golden id, golden logit of
    /// the engine's id minus the golden maximum) — 0 at an exact golden tie
    pub fn top1_disagreements(engine: &[i32], golden: &[f32], v: usize) -> Vec<(usize, usize, usize, f32)> {
        assert_eq!(golden.len(), engine.len() * v, "top1: {} rows vs {} golden logits at vocab {v}", engine.len(), golden.len());
        engine
            .iter()
            .enumerate()
            .filter_map(|(r, &e)| {
                let row = &golden[r * v..(r + 1) * v];
                let g = argmax(row);
                let e = e as usize;
                (e != g).then(|| (r, e, g, row.get(e).map_or(f32::NAN, |&x| x - row[g])))
            })
            .collect()
    }

    /// #161 failure mode: `1 - cosine` rises monotonically over the layers in depth order
    /// (reported as a finding even when every layer passes); fewer than three layers say nothing
    pub fn monotone_rise(one_minus_cos: &[f64]) -> bool {
        one_minus_cos.len() >= 3 && one_minus_cos.windows(2).all(|w| w[1] > w[0])
    }

    /// `--layers A:B` (B exclusive; either side may be empty) against the layers on disk
    pub fn layer_range(spec: Option<&str>, on_disk: &[usize], total: usize) -> Result<Vec<usize>, String> {
        let Some(s) = spec else { return Ok(on_disk.to_vec()) };
        let (a, b) = s.split_once(':').ok_or_else(|| format!("--layers {s:?}: want A:B"))?;
        let num = |v: &str, dflt: usize| if v.is_empty() { Ok(dflt) } else { v.parse::<usize>().map_err(|_| format!("--layers {s:?}: {v:?} is not a layer")) };
        let (a, b) = (num(a, 0)?, num(b, total)?);
        if a >= b || b > total {
            return Err(format!("--layers {s:?}: want 0 <= A < B <= {total}"));
        }
        let want: Vec<usize> = (a..b).collect();
        if let Some(m) = want.iter().find(|l| !on_disk.contains(l)) {
            return Err(format!("--layers {s:?}: the golden has no l{m}-output.f32 (it holds {on_disk:?})"));
        }
        Ok(want)
    }
}

#[cfg(test)]
mod tests {
    //! Host tests (no GPU): the schedule, the tensor plan against the real container's index,
    //! the load-time decodes, the shared compile, the harness's host side.
    use super::golden::*;
    use super::*;
    use crate::geo::{ExpertCodec, ExpertRecordSpec};

    const G: Glm5Geo = Glm5Geo::GLM_5_3_FLASH;

    fn fixture() -> (serde_json::Value, Vec<TensorInfo>) {
        let p = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures/glm5/model/index-l0-4.json");
        let v: serde_json::Value = serde_json::from_slice(&std::fs::read(p).unwrap()).unwrap();
        let idx = serde_json::json!({ "tensors": v["tensors"] });
        let t = cnq::parse_tensor_index(&idx, false);
        (v, t)
    }

    #[test]
    fn the_schedule_is_the_recipe_layer_types() {
        let kinds: Vec<&str> = (0..G.layers).map(|l| kind_label(&G, l)).collect();
        assert_eq!(&kinds[..5], &["kda+dense", "kda+dense", "kda+dense", "dsa+moe", "kda+moe"]);
        assert_eq!(kinds.iter().filter(|k| k.starts_with("dsa")).count(), G.dsa_layers);
        assert_eq!(kinds.iter().filter(|k| k.ends_with("dense")).count(), G.dense_prefix);
        assert_eq!(kinds[43], "dsa+moe");
        assert_eq!(kinds[44], "kda+moe");
    }

    /// the plan resolves against the real 3-bit container's index (layers 0-4: dense KDA, DSA +
    /// MoE, KDA + MoE): every name present, every dtype one its take accepts, every shape planned;
    /// the NVFP4 projections of the codec gap are NVFP4 there
    #[test]
    fn the_tensor_plan_fits_the_real_container_index() {
        let (v, t) = fixture();
        assert_eq!(v["container_bytes"].as_u64(), Some(124_591_634_567));
        let mut plan = model_tensors(&G);
        for l in 0..5 {
            plan.extend(layer_tensors(&G, l));
        }
        check_plan(&plan, &t).unwrap();
        let take = |s: &str| plan.iter().find(|p| p.name.ends_with(s)).unwrap().take;
        for s in ["layers.0.self_attn.q_proj.weight", "layers.0.self_attn.o_proj.weight", "layers.3.self_attn.q_a_proj.weight", "layers.3.self_attn.o_proj.weight"] {
            assert_eq!(take(s), Take::Fp4, "{s}");
        }
        assert_eq!(take("layers.0.self_attn.k_conv1d.weight"), Take::Fp4ToF32);
        assert_eq!(take("layers.3.self_attn.kv_b_proj.weight"), Take::Fp4ToBf16);
        // every non-expert tensor of layers 0-4 is planned exactly once: nothing left unread
        let planned: std::collections::HashSet<&str> = plan.iter().map(|p| p.name.as_str()).collect();
        assert_eq!(planned.len(), plan.len());
        let unplanned: Vec<&str> = t.iter().map(|t| t.name.as_str()).filter(|n| !n.contains(".mlp.experts.") && !planned.contains(n)).collect();
        assert!(unplanned.is_empty(), "{unplanned:?}");
        // the expert record of the container: MUL1, 9,474,048 B, gate first
        // the expert records (the fixture keeps experts 0 and 287 of layers 3 and 4): MUL1,
        // 9,474,048 B, gate, up, down inside the record, the record on a 4096-B file boundary
        let blob = v["blob_offset"].as_u64().unwrap();
        let mut n = 0;
        for x in v["tensors"].as_array().unwrap().iter().filter(|x| x["dtype"] == "mul1") {
            let m = &x["mul1"];
            assert_eq!((m["record_bytes"].as_u64(), m["k"].as_f64()), (Some(9_474_048), Some(3.0)), "{}", x["name"]);
            let (r0, off) = (m["record_offset"].as_u64().unwrap(), x["offset"].as_u64().unwrap());
            assert!(off >= r0 && off + x["len"].as_u64().unwrap() <= r0 + 9_474_048 && (blob + r0) % 4096 == 0, "{}", x["name"]);
            n += 1;
        }
        assert_eq!(n, 12);
        let rec = ExpertRecordSpec::new(ExpertCodec::Mul1, 9_474_048).unwrap();
        for (l, e) in [(3, 0), (3, 287), (4, 0), (4, 287)] {
            let [gate, up, down] = expert_names(l, e);
            let off = |name: &str| t.iter().find(|x| x.name == name && x.dtype == "mul1").unwrap_or_else(|| panic!("{name}")).offset;
            assert!(off(&gate) < off(&up) && off(&up) < off(&down), "layer {l} expert {e}: the record starts at gate");
        }
        MoeGeo::new(&G, rec).unwrap();
    }

    /// a container whose codec or shapes differ is refused by name before a byte is read
    #[test]
    fn a_tensor_of_another_codec_or_shape_is_refused_by_name() {
        let (_, mut t) = fixture();
        let q = t.iter_mut().find(|x| x.name.ends_with("layers.0.self_attn.q_proj.weight")).unwrap();
        q.dtype = "bf16".into();
        let ln = t.iter_mut().find(|x| x.name.ends_with("layers.3.input_layernorm.weight")).unwrap();
        ln.shape = vec![2048];
        t.retain(|x| !x.name.ends_with("layers.4.self_attn.A_log"));
        let plan: Vec<Planned> = (0..5).flat_map(|l| layer_tensors(&G, l)).collect();
        let why = check_plan(&plan, &t).unwrap_err();
        assert!(why.contains("3 tensor(s) do not fit"), "{why}");
        assert!(why.contains("layers.0.self_attn.q_proj.weight: dtype bf16"), "{why}");
        assert!(why.contains("layers.3.input_layernorm.weight: shape [2048], want [4096]"), "{why}");
        assert!(why.contains("layers.4.self_attn.A_log: not in the index"), "{why}");
    }

    fn block(scales: [u8; 4], nibble: u8) -> Vec<u8> {
        let mut b = scales.to_vec();
        b.extend(std::iter::repeat_n(nibble | (nibble << 4), 32));
        b
    }

    /// the load decodes: NVFP4 -> f32 is `e2m1 * ue4m3 * global`; -> BF16 counts the values BF16
    /// does not hold; the 0x7F scale byte is rewritten to 0x7E (448) before either
    #[test]
    fn the_load_decodes_count_inexact_bf16_and_sanitize_0x7f() {
        // scale 0x38 = 1.0, code 2 = 1.0: exact at global 1 and 0.5; at global 0.1 inexact
        let raw = block([0x38; 4], 2);
        assert_eq!(nvfp4_to_f32(&raw, 1.0, 64), vec![1.0; 64]);
        assert_eq!(nvfp4_to_bf16(&raw, 0.5, 64), (vec![0x3F00; 64], 0));
        let (b, inexact) = nvfp4_to_bf16(&raw, 0.1, 64);
        assert_eq!(inexact, 64);
        assert_eq!(b[0], f32_to_bf16_rne(0.1));
        // one 0x7F sub-block: read raw it is 480, sanitized 448
        let mut raw = block([0x7F, 0x38, 0x38, 0x38], 2);
        assert_eq!(nvfp4_to_f32(&raw, 1.0, 64)[0], 480.0);
        assert_eq!(crate::residency::sanitize_sf_slab(&mut raw), 1);
        let v = nvfp4_to_f32(&raw, 1.0, 64);
        assert_eq!((v[0], v[15], v[16]), (448.0, 448.0, 1.0));
        // a partial last block is cut at n
        assert_eq!(nvfp4_to_f32(&block([0x38; 4], 2), 1.0, 40).len(), 40);
        // RNE: the midpoint 1 + 2^-8 goes to the even 1.0, NaN stays NaN
        assert_eq!(f32_to_bf16_rne(1.0 + 1.0 / 256.0), 0x3F80);
        assert_eq!(f32_to_bf16_rne(1.0 + 3.0 / 256.0), 0x3F82);
        assert!(f32::from_bits((f32_to_bf16_rne(f32::NAN) as u32) << 16).is_nan());
    }

    #[test]
    fn the_trunk_input_copies_each_embedding_row_into_the_four_streams() {
        let e = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        let x = trunk_input(&e, 3, 4);
        assert_eq!(x.len(), 24);
        assert_eq!(&x[..12], &[1.0, 2.0, 3.0, 1.0, 2.0, 3.0, 1.0, 2.0, 3.0, 1.0, 2.0, 3.0]);
        assert_eq!(&x[12..15], &[4.0, 5.0, 6.0]);
        assert_eq!(&x[21..], &[4.0, 5.0, 6.0]);
    }

    /// one `KERNEL_SRC` compile at the glm5 geometry serves every module: it has every entry the
    /// glm5 path launches, and the same entry set as the Flash-Next compile (so the engine kernel
    /// table `Kernels::new` resolves on it); NVRTC on the host, no GPU
    #[test]
    fn the_shared_glm5_compile_has_every_entry_the_path_launches() {
        use crate::kernels::tests_300_c4::{entries, ptx};
        let names = |src: &str| -> std::collections::BTreeSet<String> { entries(&ptx(src)).into_iter().map(|(n, _)| n).collect() };
        let glm = names(&kernel_geo(&G).source());
        for n in MAIN_NAMES.iter().chain(glm5_kda::BASE_NAMES) {
            assert!(glm.contains(*n), "{n} missing from the glm5 compile");
        }
        assert_eq!(glm, names(&KernelGeo::flash_next().source()), "the glm5 compile's entries differ from Flash-Next's");
        assert!(!kernel_geo(&G).p2, "the glm5 compile appends no P2_SRC");
    }

    #[test]
    fn the_harness_reads_the_runner_manifest_and_splits_its_calls() {
        let text = r#"{"ids": [1,2,3,4,5,6,7], "T": 5, "D": 2, "anchors": [4,5,6], "prompt_chunk": 2,
            "hidden_size": 4096, "hc_mult": 4, "complete": true,
            "files": {"embed.f32": {"shape": [7, 4096]}, "l0-output.f32": {"shape": [7, 4, 4096]},
                      "l3-output.f32": {"shape": [7, 4, 4096]}, "l3-routing-ids.i32": {"shape": [7, 8]}},
            "subblocks": {"3": {"attn": {"in": "l3-attn_hc-in.f32", "collapsed": "l3-attn_hc-collapsed.f32", "expanded": "l3-ffn_hc-in.f32"}}},
            "head": {"mean": "head-mean.f32", "norm": "head-norm.f32", "logits": "head-logits.f32"}}"#;
        let m = Manifest::parse(text).unwrap();
        assert_eq!((m.t, m.d, m.n, m.prompt_chunk, m.layers.clone()), (5, 2, 7, 2, vec![0, 3]));
        assert_eq!(m.calls(), vec![(0, 2, false), (2, 2, false), (4, 1, false), (5, 1, true), (6, 1, true)]);
        assert_eq!(m.shape("l3-routing-ids.i32"), Some(vec![7, 8]));
        assert_eq!(m.subblock(3, "attn", "expanded").as_deref(), Some("l3-ffn_hc-in.f32"));
        assert_eq!((m.subblock(3, "attn", "pre"), m.subblock(0, "attn", "in")), (None, None));
        // #165: the head capture of every row; a manifest without it has none
        assert_eq!((m.head_file("norm").as_deref(), m.head_file("logits").as_deref()), (Some("head-norm.f32"), Some("head-logits.f32")));
        assert_eq!(Manifest::parse(r#"{"ids": [1], "T": 1, "D": 0, "hidden_size": 1, "hc_mult": 4, "files": {}}"#).unwrap().head_file("logits"), None);
        assert!(role_is_judged("out") && !role_is_judged("comb") && !SITE_ROLES.contains(&"pre"));
        // a pre-#147 manifest has no prompt_chunk: the prompt in one call
        assert_eq!(calls(86, 4, 0)[0], (0, 86, false));
        assert_eq!(calls(86, 4, 0).len(), 5);
        assert!(Manifest::parse(r#"{"ids": [1], "T": 2, "D": 0, "hidden_size": 1, "hc_mult": 4, "files": {}}"#).unwrap_err().contains("1 ids, T 2 + D 0"));
        assert_eq!(layer_range(Some("0:1"), &m.layers, 45), Ok(vec![0]));
        assert!(layer_range(Some("0:4"), &m.layers, 45).unwrap_err().contains("no l1-output.f32"));
        assert_eq!(layer_range(None, &m.layers, 45), Ok(vec![0, 3]));
    }

    #[test]
    fn the_harness_metrics_and_overlaps() {
        let g = [1.0f32, 0.0, 0.0, 2.0];
        let c = compare(&g, &g, 2);
        assert!((c.cosine - 1.0).abs() < 1e-15 && c.max_abs == 0.0 && c.rel_rms == 0.0 && passes(&c));
        let e = [1.0f32, 0.0, 0.0, -2.0];
        let c = compare(&e, &g, 2);
        assert_eq!((c.worst_row, c.max_abs), (1, 4.0));
        assert!((c.worst_row_cosine + 1.0).abs() < 1e-15 && !passes(&c));
        assert!((c.rel_rms - (16.0f64 / 5.0).sqrt()).abs() < 1e-12);
        let c = compare(&[f32::NAN, 0.0, 0.0, 2.0], &g, 2);
        assert_eq!(c.non_finite, 1);
        assert!(!passes(&c), "NaN never passes G3");
        // routing: row 0 identical in another order, row 1 shares 6 of 8
        let gold = [0, 1, 2, 3, 4, 5, 6, 7, 10, 11, 12, 13, 14, 15, 16, 17];
        let eng = vec![vec![7, 6, 5, 4, 3, 2, 1, 0], vec![10, 11, 12, 13, 14, 15, 98, 99]];
        assert_eq!(routing_overlap(&eng, &gold, 8), ((1.0 + 0.75) / 2.0, 1));
        // DSA: -1 is empty; row 1 misses one of four
        let gold = [0, 1, 2, -1, 3, 0, 1, 2];
        assert_eq!(dsa_overlap(&[vec![0, 1, 2], vec![0, 1, 2]], &gold, 4), ((1.0 + 0.75) / 2.0, 1));
        assert!(monotone_rise(&[1e-6, 2e-6, 5e-6]));
        // #165 top-1: row 0 agrees, row 1 picks id 0 (0.5 below the golden max at id 2), row 2 an exact tie
        let gold = [3.0f32, 1.0, 2.0, 1.0, 0.5, 1.5, 7.0, 7.0, 0.0];
        assert_eq!(argmax(&gold[6..9]), 0, "the first index of the maximum");
        assert_eq!(top1_disagreements(&[0, 0, 1], &gold, 3), vec![(1, 0, 2, -0.5), (2, 1, 0, 0.0)]);
        assert!(top1_disagreements(&[0, 2, 0], &gold, 3).is_empty());
        assert!(!monotone_rise(&[1e-6, 2e-6, 2e-6]) && !monotone_rise(&[1e-6, 2e-6]));
    }
}

#[cfg(test)]
mod tests_dense_gpu {
    //! #191 on the GPU (RTX 5090, sm_120): the glm5_next dense decode launches at the
    //! GLM-5.3-Flash shapes on synthetic weights: the pass's NVFP4 GEMV dispatch ([`fp4_gemv`],
    //! the one the KDA / MLA hooks call) and the FFN plan bit-identical to the record kernels of
    //! `KERNEL_SRC` (`gemv_fp4_b` / `gemv_fp4_bs`), and their time per decode row. Each timed
    //! launch reads a different copy of its matrix (>= 256 MiB per shape), so the weights come
    //! from VRAM as in a decode row, not from the L2. `#[ignore]`: CI has no GPU. Run with
    //! `cargo test --release --lib glm5_dense_gpu -- --ignored --nocapture --test-threads 1`.
    use super::*;
    use crate::cpu_mul1::testkit::Rng;
    use cudarc::driver::sys;

    const G: Glm5Geo = Glm5Geo::GLM_5_3_FLASH;

    /// one dense NVFP4 launch of a decode row: what, rows, cols, output row stride, calls per row
    struct Shape {
        what: &'static str,
        rows: usize,
        cols: usize,
        ldy: Option<usize>,
        per_row: usize,
    }

    /// the dense NVFP4 GEMVs of one GLM-5.3-Flash decode row (34 KDA, 11 MLA, 3 dense, 42 MoE
    /// layers), from the geometry
    fn shapes() -> Vec<Shape> {
        let (kd, md, h) = (KdaDims::of(&G), MlaDims::of(&G), G.hidden);
        let (kda, mla, dense, moe) = (G.kda_layers, G.dsa_layers, G.dense_prefix, G.moe_layers());
        let si = G.expert_inter * G.shared_experts;
        let s = |what, rows, cols, ldy, per_row| Shape { what, rows, cols, ldy, per_row };
        vec![
            s("kda q|k|v", kd.width(), h, Some(kd.conv_ch()), 3 * kda),
            s("kda o", h, kd.width(), None, kda),
            s("mla q_a", md.q_lora, h, None, mla),
            s("mla q_b", md.heads * md.nope, md.q_lora, None, mla),
            s("mla kv_a", md.kv_lora, h, None, mla),
            s("mla o", h, md.heads * md.v, None, mla),
            s("dense gate|up", G.dense_inter, h, None, 2 * dense),
            s("dense down", h, G.dense_inter, None, dense),
            s("shared gate|up", si, h, None, 2 * moe),
            s("shared down", h, si, None, moe),
        ]
    }

    /// random NVFP4 bytes: scale bytes in [0x30, 0x3F] (no 0x7F, the sanitized rule), codes
    /// uniform over all 16 nibbles; with `zeros` every 7th block all +0/-0 codes (the signed-zero
    /// paths of the reductions)
    fn nvfp4(rows: usize, cols: usize, rng: &mut Rng, zeros: bool) -> Vec<u8> {
        let mut b = vec![0u8; rows * cols / 64 * 36];
        for (k, blk) in b.chunks_exact_mut(36).enumerate() {
            for (i, v) in blk.iter_mut().enumerate() {
                *v = if i < 4 {
                    0x30 + (rng.next() % 16) as u8
                } else if zeros && k % 7 == 3 {
                    0x88
                } else {
                    rng.next() as u8
                };
            }
        }
        b
    }

    fn xs(n: usize, rng: &mut Rng) -> Vec<f32> {
        (0..n).map(|i| if i % 97 == 5 { -0.0 } else { rng.f(2.0) }).collect()
    }

    fn bits(v: &[f32]) -> Vec<u32> {
        v.iter().map(|x| x.to_bits()).collect()
    }

    /// the record launch: `gemv_fp4_b`, or `gemv_fp4_bs` with a stride (the path before #191)
    unsafe fn record(kn: &Glm5Kernels, ints: &Ints, m: &GpuNvfp4, x: Dev, y: Dev, t: usize, ldy: Option<usize>) {
        match ldy {
            None => launch_v(kn.k.f("gemv_fp4_b"), m.rows as u32, t as u32, 1, 256, &[m.w, x, m.gs, y, ints.p(m.cols)]),
            Some(s) => launch_v(kn.k.f("gemv_fp4_bs"), m.rows as u32, t as u32, 1, 256, &[m.w, x, m.gs, y, ints.p(m.cols), ints.p(s)]),
        }
    }

    unsafe fn ints_for(sh: &[Shape]) -> Ints {
        let mut v: Vec<usize> = sh.iter().flat_map(|s| [s.rows, s.cols, s.ldy.unwrap_or(s.rows)]).collect();
        v.sort_unstable();
        v.dedup();
        Ints::new(&v)
    }

    /// mean GPU time of one `f(i)` over `n` calls, events around the whole queue (us); when the
    /// host queues slower than the GPU runs, this is the host's launch rate (`launch_us`)
    unsafe fn time_us(n: usize, f: impl FnMut(usize)) -> f64 {
        time2_us(n, f).0
    }

    /// (`time_us`, mean host time to queue one `f(i)`) in us
    unsafe fn time2_us(n: usize, mut f: impl FnMut(usize)) -> (f64, f64) {
        f(0);
        cuda::sync();
        let mk = || {
            let mut e: sys::CUevent = std::ptr::null_mut();
            cuda::ck(sys::cuEventCreate(&mut e, 0));
            e
        };
        let (a, b) = (mk(), mk());
        let s = cuda::cur_stream();
        cuda::event_record(a, s);
        let t0 = std::time::Instant::now();
        for i in 0..n {
            f(i);
        }
        let host = t0.elapsed().as_secs_f64() * 1e6 / n as f64;
        cuda::event_record(b, s);
        cuda::sync();
        let mut ms = 0f32;
        cuda::ck(sys::cuEventElapsedTime_v2(&mut ms, a, b));
        cuda::event_destroy(a);
        cuda::event_destroy(b);
        (ms as f64 * 1e3 / n as f64, host)
    }

    #[test]
    #[ignore = "needs the GPU: cargo test --release --lib glm5_dense_gpu -- --ignored --nocapture --test-threads 1"]
    fn glm5_dense_gpu_fp4_gemv_is_bit_identical_to_the_record_kernels() {
        unsafe {
            let _ctx = cuda::Ctx::init();
            let kn = Glm5Kernels::new(&G);
            let sh = shapes();
            let ints = ints_for(&sh);
            let mut rng = Rng(0x5eed_d3a5e);
            for s in &sh {
                for t in [1usize, 3] {
                    let wb = nvfp4(s.rows, s.cols, &mut rng, true);
                    let m = GpuNvfp4 { w: cuda::upload_dev(&wb), gs: cuda::to_f32_dev(&[0.37]), rows: s.rows, cols: s.cols };
                    let x = cuda::to_f32_dev(&xs(t * s.cols, &mut rng));
                    let ld = s.ldy.unwrap_or(s.rows);
                    // NaN-filled outputs: an element one path leaves unwritten is a mismatch
                    let nan = vec![f32::NAN; t * ld];
                    let (ya, yb) = (cuda::to_f32_dev(&nan), cuda::to_f32_dev(&nan));
                    record(&kn, &ints, &m, x, ya, t, s.ldy);
                    fp4_gemv(&kn, &ints, &m, x, yb, t, s.ldy);
                    cuda::sync();
                    let (a, b) = (bits(&cuda::dtoh(ya, t * ld)), bits(&cuda::dtoh(yb, t * ld)));
                    let differ = a.iter().zip(&b).filter(|(p, q)| p != q).count();
                    assert_eq!(differ, 0, "{} [{} x {}] t {t}: {differ} of {} outputs differ from the record kernel", s.what, s.rows, s.cols, t * ld);
                    for mut d in [m.w, m.gs, x, ya, yb] {
                        cuda::free_dev(&mut d);
                    }
                }
            }
            // the FFN plan (shared expert and dense layers) against record launches of the same math
            for (inter, t) in [(G.expert_inter, 1usize), (G.dense_inter, 2)] {
                let h = G.hidden;
                let up = |rng: &mut Rng, r: usize, c: usize| GpuNvfp4 { w: cuda::upload_dev(&nvfp4(r, c, rng, true)), gs: cuda::to_f32_dev(&[0.21]), rows: r, cols: c };
                let w = GpuFfnWeights { gate: up(&mut rng, inter, h), up: up(&mut rng, inter, h), down: up(&mut rng, h, inter) };
                let x = cuda::to_f32_dev(&xs(t * h, &mut rng));
                let mut plan = GpuFfnPlan::new(h, inter, t, G.swiglu_limit as f32);
                let y = cuda::alloc_zeroed(t * h * 4);
                plan.run(&kn.k, &kn.moe, &w, x, y);
                cuda::sync();
                let got = bits(&cuda::dtoh(y, t * h));
                let fints = Ints::new(&[h, inter]);
                let (g, u, a, r) = (cuda::alloc_zeroed(t * inter * 4), cuda::alloc_zeroed(t * inter * 4), cuda::alloc_zeroed(t * inter * 4), cuda::alloc_zeroed(t * h * 4));
                record(&kn, &fints, &w.gate, x, g, t, None);
                record(&kn, &fints, &w.up, x, u, t, None);
                let prm_n = cuda::to_i32_dev(&[(t * inter) as i32]);
                let prm_f = cuda::to_f32_dev(&[0.0, G.swiglu_limit as f32]);
                launch_v(kn.moe.act, (t * inter).div_ceil(256) as u32, 1, 1, 256, &[g, u, a, prm_n, prm_f]);
                record(&kn, &fints, &w.down, a, r, t, None);
                cuda::sync();
                let want = bits(&cuda::dtoh(r, t * h));
                assert_eq!(got, want, "FFN [{inter}] t {t}: the plan differs from the record launches");
                plan.free();
                for mut d in [w.gate.w, w.gate.gs, w.up.w, w.up.gs, w.down.w, w.down.gs, x, y, g, u, a, r, prm_n, prm_f] {
                    cuda::free_dev(&mut d);
                }
            }
        }
    }

    /// #191 acceptance bound, fixed in the ticket before the after-measurement: the dense
    /// NVFP4 GEMVs of one decode row on synthetic VRAM-cold weights take at most this (us)
    const DENSE_ROW_US: f64 = 6000.0;

    #[test]
    #[ignore = "needs the GPU: cargo test --release --lib glm5_dense_gpu -- --ignored --nocapture --test-threads 1"]
    fn glm5_dense_gpu_fp4_gemv_reads_the_weights_at_vram_rate() {
        unsafe {
            let _ctx = cuda::Ctx::init();
            let kn = Glm5Kernels::new(&G);
            let sh = shapes();
            let ints = ints_for(&sh);
            let mut rng = Rng(0xbe4c_0001);
            // the reference rate: a device-to-device copy of 512 MiB (reads + writes)
            let cb = 512usize << 20;
            let (mut c0, mut c1) = (cuda::alloc_zeroed(cb), cuda::alloc_zeroed(cb));
            let cp = time_us(8, |_| cuda::memcpy_async(c1, c0, cb));
            eprintln!("dense reference: device copy {cb} B in {cp:.0} us = {:.0} GB/s read + write", 2.0 * cb as f64 / cp / 1e3);
            cuda::free_dev(&mut c0);
            cuda::free_dev(&mut c1);
            let (mut rec_row, mut new_row, mut bytes_row) = (0f64, 0f64, 0f64);
            for s in &sh {
                let wb = nvfp4(s.rows, s.cols, &mut rng, false);
                let bytes = wb.len();
                let copies = (256usize << 20).div_ceil(bytes).max(2);
                let first = cuda::upload_dev(&wb);
                let mut ws = vec![first];
                for _ in 1..copies {
                    let d = cuda::alloc_zeroed(bytes);
                    cuda::memcpy_async(d, first, bytes);
                    ws.push(d);
                }
                let gs = cuda::to_f32_dev(&[0.37]);
                let x = cuda::to_f32_dev(&xs(s.cols, &mut rng));
                let y = cuda::alloc_zeroed(s.ldy.unwrap_or(s.rows) * 4);
                let m = |i: usize| GpuNvfp4 { w: ws[i % copies], gs, rows: s.rows, cols: s.cols };
                let n = (4 * copies).max(64);
                let rec = time_us(n, |i| record(&kn, &ints, &m(i), x, y, 1, s.ldy));
                let (new, host) = time2_us(n, |i| fp4_gemv(&kn, &ints, &m(i), x, y, 1, s.ldy));
                let gbs = |us: f64| bytes as f64 / us / 1e3;
                eprintln!(
                    "dense {:<15} [{:>5} x {:>5}] {:>9} B x {:>3}/row: record {:>7.1} us {:>6.0} GB/s | pass {:>7.1} us {:>6.0} GB/s (host queues one in {:.1} us)",
                    s.what, s.rows, s.cols, bytes, s.per_row, rec, gbs(rec), new, gbs(new), host
                );
                rec_row += rec * s.per_row as f64;
                new_row += new * s.per_row as f64;
                bytes_row += (bytes * s.per_row) as f64;
                for mut d in ws.into_iter().chain([gs, x, y]) {
                    cuda::free_dev(&mut d);
                }
            }
            eprintln!(
                "dense NVFP4 GEMVs per decode row: {:.0} MB; record {:.2} ms ({:.0} GB/s), pass {:.2} ms ({:.0} GB/s)",
                bytes_row / 1e6,
                rec_row / 1e3,
                bytes_row / rec_row / 1e3,
                new_row / 1e3,
                bytes_row / new_row / 1e3
            );
            assert!(new_row <= DENSE_ROW_US, "the pass's dense NVFP4 GEMVs take {:.2} ms per decode row (bound {:.2} ms)", new_row / 1e3, DENSE_ROW_US / 1e3);
        }
    }

    /// the KDA q, k, v matrices of one layer (`[width, hidden]` each) from `wb`, uploaded
    unsafe fn qkv(wb: [&[u8]; 3], gs: [f32; 3]) -> [GpuNvfp4; 3] {
        let (kd, h) = (KdaDims::of(&G), G.hidden);
        [0, 1, 2].map(|i| GpuNvfp4 { w: cuda::upload_dev(wb[i]), gs: cuda::to_f32_dev(&[gs[i]]), rows: kd.width(), cols: h })
    }

    #[test]
    #[ignore = "needs the GPU: cargo test --release --lib glm5_dense_gpu -- --ignored --nocapture --test-threads 1"]
    fn glm5_dense_gpu_kda_qkv_in_one_launch_is_bit_identical_to_three() {
        unsafe {
            let _ctx = cuda::Ctx::init();
            let kn = Glm5Kernels::new(&G);
            let (kd, h) = (KdaDims::of(&G), G.hidden);
            let (w, cc) = (kd.width(), kd.conv_ch());
            let ints = Ints::new(&[h, w, cc]);
            let mut rng = Rng(0x05ee_d0e3);
            for t in [1usize, 3] {
                let wb: Vec<Vec<u8>> = (0..3).map(|_| nvfp4(w, h, &mut rng, true)).collect();
                let m = qkv([&wb[0], &wb[1], &wb[2]], [0.37, 0.21, 0.53]);
                let x = cuda::to_f32_dev(&xs(t * h, &mut rng));
                let nan = vec![f32::NAN; t * cc];
                let (ya, yb) = (cuda::to_f32_dev(&nan), cuda::to_f32_dev(&nan));
                for (i, mi) in m.iter().enumerate() {
                    fp4_gemv(&kn, &ints, mi, x, ya + (i * w * 4) as u64, t, Some(cc));
                }
                fp4_gemv_x3(&kn, &ints, [&m[0], &m[1], &m[2]], x, yb, t, cc);
                cuda::sync();
                let (a, b) = (bits(&cuda::dtoh(ya, t * cc)), bits(&cuda::dtoh(yb, t * cc)));
                let differ = a.iter().zip(&b).filter(|(p, q)| p != q).count();
                assert_eq!(differ, 0, "kda q|k|v t {t}: {differ} of {} outputs differ from three fp4_gemv launches", t * cc);
                for mut d in m.iter().flat_map(|mi| [mi.w, mi.gs]).chain([x, ya, yb]) {
                    cuda::free_dev(&mut d);
                }
            }
        }
    }

    /// acceptance bound, fixed before the after-measurement (ticket comment): the one q|k|v launch
    /// of a KDA layer takes at most this share of the three `fp4_gemv` launches it replaces
    const QKV_ONE_VS_THREE: f64 = 1.02;

    #[test]
    #[ignore = "needs the GPU: cargo test --release --lib glm5_dense_gpu -- --ignored --nocapture --test-threads 1"]
    fn glm5_dense_gpu_kda_qkv_one_launch_is_not_slower_than_three() {
        unsafe {
            let _ctx = cuda::Ctx::init();
            let kn = Glm5Kernels::new(&G);
            let (kd, h) = (KdaDims::of(&G), G.hidden);
            let (w, cc) = (kd.width(), kd.conv_ch());
            let ints = Ints::new(&[h, w, cc]);
            let mut rng = Rng(0x0be4_c0e3);
            let wb = nvfp4(w, h, &mut rng, false);
            let bytes = wb.len();
            // >= 256 MiB of copies, three matrices per launch
            let sets = (256usize << 20).div_ceil(3 * bytes).max(2);
            let ms: Vec<[GpuNvfp4; 3]> = (0..sets).map(|_| qkv([&wb, &wb, &wb], [0.37, 0.21, 0.53])).collect();
            let x = cuda::to_f32_dev(&xs(h, &mut rng));
            let y = cuda::alloc_zeroed(cc * 4);
            let n = (4 * sets).max(64);
            let (three, host3) = time2_us(n, |i| {
                for (j, m) in ms[i % sets].iter().enumerate() {
                    fp4_gemv(&kn, &ints, m, x, y + (j * w * 4) as u64, 1, Some(cc));
                }
            });
            let (one, host1) = time2_us(n, |i| {
                let m = &ms[i % sets];
                fp4_gemv_x3(&kn, &ints, [&m[0], &m[1], &m[2]], x, y, 1, cc);
            });
            let gbs = |us: f64| 3.0 * bytes as f64 / us / 1e3;
            eprintln!(
                "kda q|k|v [3 x {w} x {h}] {} B per layer, x {} per row: three launches {three:.1} us {:.0} GB/s (host {host3:.1} us) | one launch {one:.1} us {:.0} GB/s (host {host1:.1} us)",
                3 * bytes,
                G.kda_layers,
                gbs(three),
                gbs(one)
            );
            for m in ms {
                for mut d in m.iter().flat_map(|mi| [mi.w, mi.gs]) {
                    cuda::free_dev(&mut d);
                }
            }
            for mut d in [x, y] {
                cuda::free_dev(&mut d);
            }
            assert!(one <= QKV_ONE_VS_THREE * three, "one q|k|v launch {one:.1} us vs three {three:.1} us (bound x{QKV_ONE_VS_THREE})");
        }
    }

    /// the kernel launches `f` queues: captured into a graph on a non-blocking stream, nodes counted
    unsafe fn kernel_launches(f: impl FnOnce()) -> usize {
        type FnBegin = unsafe extern "system" fn(sys::CUstream, u32) -> sys::CUresult;
        type FnEnd = unsafe extern "system" fn(sys::CUstream, *mut sys::CUgraph) -> sys::CUresult;
        type FnNodes = unsafe extern "system" fn(sys::CUgraph, *mut sys::CUgraphNode, *mut usize) -> sys::CUresult;
        type FnGraph = unsafe extern "system" fn(sys::CUgraph) -> sys::CUresult;
        type FnStream = unsafe extern "system" fn(sys::CUstream) -> sys::CUresult;
        let s = cuda::stream_create_non_blocking();
        cuda::set_stream(s as u64);
        let begin: FnBegin = cuda::graph_sym(b"cuStreamBeginCapture_v2\0");
        cuda::ck(begin(s, 1)); // CU_STREAM_CAPTURE_MODE_THREAD_LOCAL
        f();
        let end: FnEnd = cuda::graph_sym(b"cuStreamEndCapture\0");
        let mut g: sys::CUgraph = std::ptr::null_mut();
        let r = end(s, &mut g);
        cuda::set_stream(0);
        cuda::ck(r);
        let nodes: FnNodes = cuda::graph_sym(b"cuGraphGetNodes\0");
        let mut n = 0usize;
        cuda::ck(nodes(g, std::ptr::null_mut(), &mut n));
        let destroy: FnGraph = cuda::graph_sym(b"cuGraphDestroy\0");
        cuda::ck(destroy(g));
        let sdestroy: FnStream = cuda::graph_sym(b"cuStreamDestroy_v2\0");
        cuda::ck(sdestroy(s));
        n
    }

    /// the KDA q|k|v projections of a call are ONE kernel launch (34 per decode row, not 102)
    #[test]
    #[ignore = "needs the GPU: cargo test --release --lib glm5_dense_gpu -- --ignored --nocapture --test-threads 1"]
    fn glm5_dense_gpu_kda_qkv_is_one_launch() {
        unsafe {
            let _ctx = cuda::Ctx::init();
            let kn = Glm5Kernels::new(&G);
            let (kd, h) = (KdaDims::of(&G), G.hidden);
            let (w, cc) = (kd.width(), kd.conv_ch());
            let ints = Ints::new(&[h, w, cc]);
            let mut rng = Rng(0x00e3_01aa);
            let wb = nvfp4(w, h, &mut rng, false);
            let m = qkv([&wb, &wb, &wb], [0.37, 0.21, 0.53]);
            let (x, y) = (cuda::alloc_zeroed(h * 4), cuda::alloc_zeroed(cc * 4));
            let n = kernel_launches(|| fp4_gemv_x3(&kn, &ints, [&m[0], &m[1], &m[2]], x, y, 1, cc));
            for mut d in m.iter().flat_map(|mi| [mi.w, mi.gs]).chain([x, y]) {
                cuda::free_dev(&mut d);
            }
            assert_eq!(n, 1, "the KDA q|k|v projections of a call take {n} kernel launches");
        }
    }

    // ------------------------------------------------ #186: the tensor-core GEMM of prompt calls

    /// (cosine, max |a - b| / max |b|) of `got` against `want` (f64)
    fn accuracy(got: &[f32], want: &[f32]) -> (f64, f64) {
        let (mut ab, mut aa, mut bb, mut d, mut m) = (0f64, 0f64, 0f64, 0f64, 0f64);
        for (&a, &b) in got.iter().zip(want) {
            let (a, b) = (a as f64, b as f64);
            ab += a * b;
            aa += a * a;
            bb += b * b;
            d = d.max((a - b).abs());
            m = m.max(b.abs());
        }
        (ab / (aa.sqrt() * bb.sqrt()), d / m)
    }

    /// `m` on `t` rows of `x`, the GEMV and then the tensor-core GEMM (the pass's dispatch,
    /// `kn.moe` switched), into NaN-filled outputs: (gemv, tc) `[t][ldy]`
    unsafe fn both(kn: &mut Glm5Kernels, ints: &Ints, m: &GpuNvfp4, x: Dev, t: usize, ldy: Option<usize>) -> (Vec<f32>, Vec<f32>) {
        let ld = ldy.unwrap_or(m.rows);
        let nan = vec![f32::NAN; t * ld];
        let (ya, yb) = (cuda::to_f32_dev(&nan), cuda::to_f32_dev(&nan));
        kn.moe.set_dense_tc(false);
        fp4_gemv(kn, ints, m, x, ya, t, ldy);
        kn.moe.set_dense_tc(true);
        assert!(kn.moe.dense_tc(t));
        fp4_gemv(kn, ints, m, x, yb, t, ldy);
        cuda::sync();
        kn.moe.set_dense_tc(false);
        let out = (cuda::dtoh(ya, t * ld), cuda::dtoh(yb, t * ld));
        for mut d in [ya, yb] {
            cuda::free_dev(&mut d);
        }
        out
    }

    /// `m` on `t` rows of `x` on one tensor-core kernel (`small`: `glm5_gemm_fp4_tcs`, else
    /// `glm5_gemm_fp4_tc`) whatever the row count, into a NaN-filled `[t][ldy]`
    unsafe fn tc_kernel(kn: &mut Glm5Kernels, ints: &Ints, m: &GpuNvfp4, x: Dev, t: usize, ldy: Option<usize>, small: bool) -> Vec<f32> {
        let ld = ldy.unwrap_or(m.rows);
        let mut y = cuda::to_f32_dev(&vec![f32::NAN; t * ld]);
        kn.moe.set_dense_tc(true);
        kn.moe.gemm_tc_on(small, [m.w; 3], [m.gs; 3], 1, x, y, ints.p(m.cols), ints.p(ld), ints.p(m.rows), m.rows, t);
        cuda::sync();
        kn.moe.set_dense_tc(false);
        let out = cuda::dtoh(y, t * ld);
        cuda::free_dev(&mut y);
        out
    }

    /// the outputs of `[t][ld]` a `[rows]` projection writes: finite, the columns past `rows` untouched
    fn written(y: &[f32], t: usize, ld: usize, rows: usize, what: &str) -> Vec<f32> {
        let mut v = Vec::with_capacity(t * rows);
        for r in 0..t {
            let row = &y[r * ld..(r + 1) * ld];
            assert!(row[..rows].iter().all(|x| x.is_finite()), "{what}: row {r} has an unwritten or non-finite output");
            assert!(row[rows..].iter().all(|x| x.is_nan()), "{what}: row {r} wrote past its {rows} columns");
            v.extend_from_slice(&row[..rows]);
        }
        v
    }

    /// G3's per-layer bar, the acceptance of the tensor-core path (robin, 2026-10-10)
    const TC_COSINE: f64 = 0.9999;

    /// #186 `CROW_GLM_DENSE_GEMM=1` on synthetic weights: every GLM-5.3-Flash dense shape plus a
    /// shape with a tail in both tile directions, at row counts on and off the 128-row tile, the
    /// tensor-core GEMM against the GEMV: every output written once (NaN-filled outputs, the
    /// stride columns past `rows` untouched), cosine >= 0.9999, the dispatch and both kernels
    /// (32 x 32 split-K tiles, 128 x 128 tiles) at every row count; and the q|k|v launch of three.
    #[test]
    #[ignore = "needs the GPU: cargo test --release --lib glm5_dense_gpu -- --ignored --nocapture --test-threads 1"]
    fn glm5_dense_gpu_tc_matches_the_gemv_on_synthetic_shapes() {
        unsafe {
            let _ctx = cuda::Ctx::init();
            let mut kn = Glm5Kernels::new(&G);
            let mut sh = shapes();
            sh.push(Shape { what: "tail 200 x 192", rows: 200, cols: 192, ldy: Some(203), per_row: 0 });
            let mut ints_v: Vec<usize> = sh.iter().flat_map(|s| [s.rows, s.cols, s.ldy.unwrap_or(s.rows)]).collect();
            ints_v.sort_unstable();
            ints_v.dedup();
            let ints = Ints::new(&ints_v);
            let mut rng = Rng(0x1860_7c01);
            for s in &sh {
                let wb = nvfp4(s.rows, s.cols, &mut rng, true);
                let m = GpuNvfp4 { w: cuda::upload_dev(&wb), gs: cuda::to_f32_dev(&[0.37]), rows: s.rows, cols: s.cols };
                for t in [2, 5, kernels::glm5_moe::TC_MIN_ROWS, 33, 128, 129, 300] {
                    let mut x = cuda::to_f32_dev(&xs(t * s.cols, &mut rng));
                    let ld = s.ldy.unwrap_or(s.rows);
                    let what = format!("{} [{} x {}] t {t}", s.what, s.rows, s.cols);
                    let a = if t >= kernels::glm5_moe::TC_MIN_ROWS {
                        let (a, b) = both(&mut kn, &ints, &m, x, t, s.ldy);
                        let (a, b) = (written(&a, t, ld, s.rows, &what), written(&b, t, ld, s.rows, &what));
                        let (cos, rel) = accuracy(&b, &a);
                        eprintln!("tc {what:<34} dispatch: cosine {cos:.9}, max |d| / max |ref| {rel:.2e}");
                        assert!(cos >= TC_COSINE, "{what} dispatch: cosine {cos} below {TC_COSINE}");
                        a
                    } else {
                        kn.moe.set_dense_tc(false);
                        let mut ya = cuda::to_f32_dev(&vec![f32::NAN; t * ld]);
                        fp4_gemv(&kn, &ints, &m, x, ya, t, s.ldy);
                        cuda::sync();
                        let a = written(&cuda::dtoh(ya, t * ld), t, ld, s.rows, &what);
                        cuda::free_dev(&mut ya);
                        a
                    };
                    for small in [true, false] {
                        let b = written(&tc_kernel(&mut kn, &ints, &m, x, t, s.ldy, small), t, ld, s.rows, &what);
                        let (cos, rel) = accuracy(&b, &a);
                        let k = if small { "32x32" } else { "128x128" };
                        eprintln!("tc {what:<34} {k:>8}: cosine {cos:.9}, max |d| / max |ref| {rel:.2e}");
                        assert!(cos >= TC_COSINE, "{what} {k}: cosine {cos} below {TC_COSINE}");
                    }
                    cuda::free_dev(&mut x);
                }
                for mut d in [m.w, m.gs] {
                    cuda::free_dev(&mut d);
                }
            }
            // q|k|v in one launch: the three single launches' outputs, bit for bit
            let (kd, h) = (KdaDims::of(&G), G.hidden);
            let (w, cc) = (kd.width(), kd.conv_ch());
            let t = 300;
            let wb: Vec<Vec<u8>> = (0..3).map(|_| nvfp4(w, h, &mut rng, true)).collect();
            let m = qkv([&wb[0], &wb[1], &wb[2]], [0.37, 0.21, 0.53]);
            let x = cuda::to_f32_dev(&xs(t * h, &mut rng));
            let (ya, yb) = (cuda::to_f32_dev(&vec![f32::NAN; t * cc]), cuda::to_f32_dev(&vec![f32::NAN; t * cc]));
            kn.moe.set_dense_tc(true);
            for (i, mi) in m.iter().enumerate() {
                fp4_gemv(&kn, &ints, mi, x, ya + (i * w * 4) as u64, t, Some(cc));
            }
            fp4_gemv_x3(&kn, &ints, [&m[0], &m[1], &m[2]], x, yb, t, cc);
            cuda::sync();
            kn.moe.set_dense_tc(false);
            let (a, b) = (bits(&cuda::dtoh(ya, t * cc)), bits(&cuda::dtoh(yb, t * cc)));
            assert_eq!(a.iter().zip(&b).filter(|(p, q)| p != q).count(), 0, "tc q|k|v in one launch differs from three");
            for mut d in m.iter().flat_map(|mi| [mi.w, mi.gs]).chain([x, ya, yb]) {
                cuda::free_dev(&mut d);
            }
        }
    }

    /// #186 `CROW_GLM_DENSE_GEMM=1` on the real 3-bit container's weights: every dense NVFP4
    /// projection of one layer of each kind (KDA, MLA, dense FFN, MoE shared expert) at 300 rows
    /// of activations of RMS 1 (the normed inputs), the tensor-core GEMM against the GEMV:
    /// cosine >= 0.9999 per projection.
    #[test]
    #[ignore = "needs the GPU and the real 3-bit container (CROW_CNQ; reads one layer's dense tensors per kind): cargo test --release --lib glm5_dense_gpu_tc_accuracy -- --ignored --nocapture --test-threads 1"]
    fn glm5_dense_gpu_tc_accuracy_on_real_weights() {
        let path = std::env::var("CROW_CNQ").unwrap_or_else(|_| crate::geo::from_engine_dir(GLM5_MUL1K3_CNQ));
        let mut cnq = Cnq::open(&path);
        let kda = (0..G.layers).find(|&l| attn_kind(&G, l) == AttnKind::Kda).unwrap();
        let mla = (0..G.layers).find(|&l| attn_kind(&G, l) == AttnKind::Mla).unwrap();
        let moe = G.dense_prefix;
        let mut picks: Vec<Planned> = Vec::new();
        for (l, want) in [(kda, "self_attn"), (mla, "self_attn"), (0, "mlp"), (moe, "shared_expert")] {
            picks.extend(layer_tensors(&G, l).into_iter().filter(|p| p.take == Take::Fp4 && p.name.contains(want)));
        }
        unsafe {
            let _ctx = cuda::Ctx::init();
            let mut kn = Glm5Kernels::new(&G);
            let mut v: Vec<usize> = picks.iter().flat_map(|p| [p.shape[0] as usize, p.shape[1..].iter().product::<u64>() as usize]).collect();
            v.sort_unstable();
            v.dedup();
            let ints = Ints::new(&v);
            let mut rng = Rng(0x1860_7c02);
            let t = 300;
            let mut worst = 1f64;
            for p in &picks {
                let ti = cnq.find(&p.name, "text").clone();
                let mut raw = cnq.read_bytes(&ti);
                crate::residency::sanitize_sf_slab(&mut raw);
                let (rows, cols) = (p.shape[0] as usize, p.shape[1..].iter().product::<u64>() as usize);
                let m = GpuNvfp4 { w: cuda::upload_dev(&raw), gs: cuda::to_f32_dev(&[ti.global_scale]), rows, cols };
                // uniform in +-sqrt(3): RMS 1 per row
                let xh: Vec<f32> = (0..t * cols).map(|_| rng.f(3f32.sqrt())).collect();
                let x = cuda::to_f32_dev(&xh);
                let (a, b) = both(&mut kn, &ints, &m, x, t, None);
                let (a, b) = (written(&a, t, rows, rows, &p.name), written(&b, t, rows, rows, &p.name));
                let (cos, rel) = accuracy(&b, &a);
                // the 32 x 32 split-K kernel on the same rows (the dispatch takes it up to TC_SMALL_MAX_ROWS)
                let bs = written(&tc_kernel(&mut kn, &ints, &m, x, t, None, true), t, rows, rows, &p.name);
                let (cs, rs) = accuracy(&bs, &a);
                eprintln!("tc real {:<62} [{rows:>5} x {cols:>5}]: cosine {cos:.9} / {cs:.9} (32x32), max |d| / max |ref| {rel:.2e} / {rs:.2e}", p.name);
                worst = worst.min(cos).min(cs);
                assert!(cos >= TC_COSINE && cs >= TC_COSINE, "{}: cosine {cos} / {cs} below {TC_COSINE}", p.name);
                for mut d in [m.w, m.gs, x] {
                    cuda::free_dev(&mut d);
                }
            }
            eprintln!("tc real: {} projections, worst cosine {worst:.9}", picks.len());
            assert!(picks.len() >= 10, "{} projections picked", picks.len());
        }
    }

    /// KL(p || q) in nats of two logit rows (f64 softmax)
    fn kl(p: &[f32], q: &[f32]) -> f64 {
        let ls = |v: &[f32]| -> Vec<f64> {
            let m = v.iter().fold(f64::NEG_INFINITY, |a, &x| a.max(x as f64));
            let z: f64 = v.iter().map(|&x| (x as f64 - m).exp()).sum();
            v.iter().map(|&x| x as f64 - m - z.ln()).collect()
        };
        let (a, b) = (ls(p), ls(q));
        a.iter().zip(&b).map(|(x, y)| x.exp() * (x - y)).sum()
    }

    fn argmax(v: &[f32]) -> usize {
        v.iter().enumerate().fold(0, |b, (i, &x)| if x > v[b] { i } else { b })
    }

    /// #186 `CROW_GLM_DENSE_GEMM=1` on a synthetic prompt call (the 8-layer model of the stager
    /// tests: 3 dense layers, 5 MoE): (1) layer by layer, one 256-row prompt call per layer on two
    /// passes (GEMV / tensor cores) from the same input: the residual's cosine after every layer
    /// >= 0.9999, and the head's logits of every prompt row: KL and top-1; (2)
    /// `Glm5Run::generate` with `CROW_CHUNK=256` (prompt calls 256 + 44 through the tiers, the
    /// switch read at load) off and on: the generated ids and the KL of their logits. Reported.
    #[test]
    #[ignore = "needs the GPU (about 4 GB VRAM, a 2.3 GB synthetic container in the temp dir): cargo test --release --lib glm5_dense_gpu_tc_synthetic -- --ignored --nocapture --test-threads 1"]
    fn glm5_dense_gpu_tc_synthetic_prompt_call_layers_and_logits() {
        use crate::glm5_flags::tests::{geo8, synth_model};
        use crate::glm5_tiers::{ExpertTiers, Glm5Run, TierSizes};
        let g = geo8();
        let s = synth_model(&g, 9_474_048);
        let (spec, _) = crate::nvme_source::glm5_record_of_container(&s.path).unwrap();
        let moe = MoeGeo::new(&g, spec).unwrap();
        let mut cnq = Cnq::open_checked(&s.path).unwrap();
        let rows = 256usize;
        let prompt: Vec<i64> = (0..rows as i64).map(|i| (i * 131 + 7) % g.vocab as i64).collect();
        unsafe {
            let _ctx = cuda::Ctx::init();
            // (1) the chain
            let row = g.hc_streams * g.hidden;
            let mut pr = Glm5Pass::new(&g, moe, rows, rows);
            let mut pt = Glm5Pass::new(&g, moe, rows, rows);
            pr.kn.moe.set_dense_tc(false);
            pt.kn.moe.set_dense_tc(true);
            let x0 = trunk_input(&embed_rows(&mut cnq, &g, &prompt), g.hidden, g.hc_streams);
            let (mut xr, mut xt) = (cuda::alloc_named("tc chain ref", rows * row * 4), cuda::alloc_named("tc chain tc", rows * row * 4));
            cuda::to_f32_into(xr, &x0);
            cuda::to_f32_into(xt, &x0);
            let mut rep = LoadReport::default();
            let mut cos_l = Vec::new();
            for l in 0..g.layers {
                let mut lw = load_layer(&mut cnq, &g, &moe, l, &mut rep);
                pr.begin_layer();
                pt.begin_layer();
                pr.call(&lw, xr, 0, rows, false);
                pt.call(&lw, xt, 0, rows, false);
                cuda::sync();
                let (cos, rel) = accuracy(&cuda::dtoh(xt, rows * row), &cuda::dtoh(xr, rows * row));
                eprintln!("tc chain layer {l} ({}): residual cosine {cos:.9}, max |d| / max |ref| {rel:.2e}", kind_label(&g, l));
                cos_l.push(cos);
                lw.free();
            }
            let mut head = Head::new(head_geo(&g));
            let mut hw = load_head(&mut cnq, &g, &mut rep);
            let (mut normed, mut logits, mut ids) = (cuda::alloc_zeroed(rows * g.hidden * 4), cuda::alloc_zeroed(rows * g.vocab * 4), cuda::alloc_zeroed(rows * 4));
            let mut lg = Vec::new();
            for x in [xr, xt] {
                run_head(&pr.kn, &head, &hw, x, normed, logits, ids, rows);
                cuda::sync();
                lg.push(cuda::dtoh(logits, rows * g.vocab));
            }
            let v = g.vocab;
            let kls: Vec<f64> = (0..rows).map(|r| kl(&lg[0][r * v..(r + 1) * v], &lg[1][r * v..(r + 1) * v])).collect();
            let top1 = (0..rows).filter(|&r| argmax(&lg[0][r * v..(r + 1) * v]) == argmax(&lg[1][r * v..(r + 1) * v])).count();
            let finite = lg[0].iter().filter(|x| x.is_finite()).count();
            eprintln!(
                "tc chain head over {rows} prompt rows: KL mean {:.3e} max {:.3e}, top-1 agree {top1} / {rows}",
                kls.iter().sum::<f64>() / rows as f64,
                kls.iter().cloned().fold(0f64, f64::max)
            );
            for d in [&mut xr, &mut xt, &mut normed, &mut logits, &mut ids, &mut hw.norm, &mut hw.lm] {
                cuda::free_dev(d);
            }
            head.free();
            pr.free();
            pt.free();
            assert_eq!(finite, rows * v, "the synthetic model must stay finite for the comparison to mean something");
            // (2) generate with prompt calls through the tiers, the switch read at load
            let prompt2: Vec<i64> = (0..300i64).map(|i| (i * 77 + 3) % g.vocab as i64).collect();
            let n = 6;
            let keys = ["CROW_CHUNK", "CROW_GLM_DENSE_GEMM"];
            let old: Vec<Option<String>> = keys.iter().map(|k| std::env::var(k).ok()).collect();
            let mut gens = Vec::new();
            for on in [false, true] {
                std::env::set_var("CROW_CHUNK", "256");
                if on {
                    std::env::set_var("CROW_GLM_DENSE_GEMM", "1")
                } else {
                    std::env::remove_var("CROW_GLM_DENSE_GEMM")
                }
                let mut run = Glm5Run::load(&mut cnq, &g, &moe, prompt2.len() + n + 1, &mut |_| {});
                for (k, o) in keys.iter().zip(&old) {
                    match o {
                        Some(o) => std::env::set_var(k, o),
                        None => std::env::remove_var(k),
                    }
                }
                let mut tiers = ExpertTiers::new(&cnq, &s.path, &g, &moe, TierSizes { vram: 3, pinned: 4 }, 1, g.topk).unwrap();
                gens.push(run.generate(&mut cnq, &mut tiers, &prompt2, n, true, &mut |_| {}).unwrap());
                tiers.free();
                run.free();
            }
            let gkl: Vec<f64> = gens[0].logits.iter().zip(&gens[1].logits).map(|(p, q)| kl(p, q)).collect();
            eprintln!("tc generate (chunk 256, prompt 300): ids off {:?} on {:?}, KL per generated row {:?}", gens[0].ids, gens[1].ids, gkl.iter().map(|k| format!("{k:.3e}")).collect::<Vec<_>>());
            let worst = cos_l.iter().cloned().fold(1f64, f64::min);
            assert!(worst >= TC_COSINE, "per-layer residual cosine {worst} below {TC_COSINE}: {cos_l:?}");
        }
    }

    /// #186 micro-bench: the dense NVFP4 projections of one 8192-row prompt chunk per layer kind
    /// (KDA q|k|v + o, MLA q_a + q_b + kv_a + o, dense FFN and MoE shared expert gate + up + down;
    /// the SwiGLU between them not timed), the GEMV path vs the tensor-core GEMM, in ms and rows/s;
    /// then the row count from which the tensor cores win (every dense launch of one call,
    /// VRAM-cold weights), the base of `TC_MIN_ROWS`. Prints only.
    #[test]
    #[ignore = "needs the GPU (about 3 GB VRAM, a few minutes): cargo test --release --lib glm5_dense_gpu_tc_bench -- --ignored --nocapture --test-threads 1"]
    fn glm5_dense_gpu_tc_bench_8192_row_chunk() {
        unsafe {
            let _ctx = cuda::Ctx::init();
            let mut kn = Glm5Kernels::new(&G);
            let sh = shapes();
            let ints = ints_for(&sh);
            let mut rng = Rng(0x1860_be0c);
            let kinds: [(&str, &[&str], usize); 4] = [
                ("kda", &["kda q|k|v", "kda o"], G.kda_layers),
                ("mla", &["mla q_a", "mla q_b", "mla kv_a", "mla o"], G.dsa_layers),
                ("dense ffn", &["dense gate|up", "dense down"], G.dense_prefix),
                ("moe shared", &["shared gate|up", "shared down"], G.moe_layers()),
            ];
            // launches of a layer per shape: q|k|v is one x3 launch, gate|up two
            let launches = |what: &str| if what.ends_with("gate|up") { 2.0 } else { 1.0 };
            let t = 8192;
            let mut tot = [0f64; 2];
            for (kind, whats, per) in kinds {
                let mut ms = [0f64; 2];
                for what in whats {
                    let s = sh.iter().find(|s| s.what == *what).unwrap();
                    let n3 = if *what == "kda q|k|v" { 3 } else { 1 };
                    let ws: Vec<GpuNvfp4> = (0..n3).map(|_| GpuNvfp4 { w: cuda::upload_dev(&nvfp4(s.rows, s.cols, &mut rng, false)), gs: cuda::to_f32_dev(&[0.37]), rows: s.rows, cols: s.cols }).collect();
                    let ld = s.ldy.unwrap_or(s.rows);
                    let x = cuda::to_f32_dev(&xs(t * s.cols, &mut rng));
                    let y = cuda::alloc_zeroed(t * ld * 4);
                    for (k, on) in [false, true].into_iter().enumerate() {
                        kn.moe.set_dense_tc(on);
                        let reps = if on { 5 } else { 1 };
                        let us = time_us(reps, |_| {
                            if n3 == 3 {
                                fp4_gemv_x3(&kn, &ints, [&ws[0], &ws[1], &ws[2]], x, y, t, ld)
                            } else {
                                fp4_gemv(&kn, &ints, &ws[0], x, y, t, s.ldy)
                            }
                        });
                        ms[k] += us * launches(what) / 1e3;
                    }
                    kn.moe.set_dense_tc(false);
                    for mut d in ws.iter().flat_map(|m| [m.w, m.gs]).chain([x, y]) {
                        cuda::free_dev(&mut d);
                    }
                }
                eprintln!(
                    "tc bench {kind:<10} 8192 rows: GEMV {:>9.1} ms ({:>7.0} rows/s) | tensor cores {:>7.2} ms ({:>9.0} rows/s), x{:.0}",
                    ms[0],
                    t as f64 / ms[0] * 1e3,
                    ms[1],
                    t as f64 / ms[1] * 1e3,
                    ms[0] / ms[1]
                );
                tot[0] += ms[0] * per as f64;
                tot[1] += ms[1] * per as f64;
            }
            eprintln!(
                "tc bench all {} layers, one 8192-row chunk, dense projections: GEMV {:.2} s ({:.0} rows/s) | tensor cores {:.3} s ({:.0} rows/s)",
                G.layers,
                tot[0] / 1e3,
                t as f64 / tot[0] * 1e3,
                tot[1] / 1e3,
                t as f64 / tot[1] * 1e3
            );
            // the crossover: every dense launch of one call of t rows, VRAM-cold weights; the GEMM
            // launched directly (below TC_MIN_ROWS the dispatch would take the GEMV)
            for t in [1usize, 2, 4, 8, 16, 32, 64, 128, 256, 512] {
                let mut us = [0f64; 3];
                for s in &sh {
                    let wb = nvfp4(s.rows, s.cols, &mut rng, false);
                    let copies = (256usize << 20).div_ceil(wb.len()).max(2);
                    let first = cuda::upload_dev(&wb);
                    let mut wsv = vec![first];
                    for _ in 1..copies {
                        let d = cuda::alloc_zeroed(wb.len());
                        cuda::memcpy_async(d, first, wb.len());
                        wsv.push(d);
                    }
                    let gs = cuda::to_f32_dev(&[0.37]);
                    let x = cuda::to_f32_dev(&xs(t * s.cols, &mut rng));
                    let ld = s.ldy.unwrap_or(s.rows);
                    let y = cuda::alloc_zeroed(t * ld * 4);
                    kn.moe.set_dense_tc(false);
                    us[0] += s.per_row as f64 * time_us(2 * copies, |i| fp4_gemv(&kn, &ints, &GpuNvfp4 { w: wsv[i % copies], gs, rows: s.rows, cols: s.cols }, x, y, t, s.ldy));
                    kn.moe.set_dense_tc(true);
                    for (k, small) in [(1, false), (2, true)] {
                        us[k] += s.per_row as f64 * time_us(2 * copies, |i| kn.moe.gemm_tc_on(small, [wsv[i % copies]; 3], [gs; 3], 1, x, y, ints.p(s.cols), ints.p(ld), ints.p(s.rows), s.rows, t));
                    }
                    kn.moe.set_dense_tc(false);
                    for mut d in wsv.into_iter().chain([gs, x, y]) {
                        cuda::free_dev(&mut d);
                    }
                }
                eprintln!(
                    "tc bench crossover t {t:>3}: dense launches of one call GEMV {:>8.2} ms | tensor cores 128x128 {:>8.2} ms | 32x32 split-K {:>8.2} ms",
                    us[0] / 1e3,
                    us[1] / 1e3,
                    us[2] / 1e3
                );
            }
        }
    }
}

#[cfg(test)]
mod tests_196_gpu {
    //! #196: the prompt call in sub-blocks (KDA 1024 rows with the state carried, MLA sub-calls
    //! of 256 rows, gate / up in pieces of 4096 combos, the dense FFN over the MoE plan's region)
    //! against the pre-#196 scratch (`Glm5Pass::new_whole`: whole calls, one piece). `#[ignore]`:
    //! CI has no GPU.
    use super::*;
    use crate::glm5_flags::tests::{geo8, synth_model};
    use sha2::{Digest, Sha256};

    fn hash(v: &[f32]) -> String {
        let mut h = Sha256::new();
        for x in v {
            h.update(x.to_bits().to_le_bytes());
        }
        h.finalize().iter().take(8).map(|b| format!("{b:02x}")).collect()
    }

    fn cosine(a: &[f32], b: &[f32]) -> f64 {
        let (mut ab, mut aa, mut bb) = (0f64, 0f64, 0f64);
        for (&x, &y) in a.iter().zip(b) {
            let (x, y) = (x as f64, y as f64);
            ab += x * y;
            aa += x * x;
            bb += y * y;
        }
        ab / (aa.sqrt() * bb.sqrt())
    }

    /// KL(p || q) in nats of two logit rows (f64 softmax)
    fn kl(p: &[f32], q: &[f32]) -> f64 {
        let ls = |v: &[f32]| -> Vec<f64> {
            let m = v.iter().fold(f64::NEG_INFINITY, |a, &x| a.max(x as f64));
            let z: f64 = v.iter().map(|&x| (x as f64 - m).exp()).sum();
            v.iter().map(|&x| x as f64 - m - z.ln()).collect()
        };
        let (a, b) = (ls(p), ls(q));
        a.iter().zip(&b).map(|(x, y)| x.exp() * (x - y)).sum()
    }

    /// The synthetic 8-layer model (3 KDA + dense, MLA + MoE, 3 KDA + MoE, MLA + MoE; real layer
    /// shapes, 16 MUL1 experts top-8), a prompt of 9,221 rows (8192 + 1029: every chunk ends on a
    /// call whose MLA sub-calls end on 5 rows) in prompt calls (`call_with_expert_batches`, every
    /// record in VRAM) of 256 / 1024 / 8192 rows, layer by layer, the sub-blocked pass against
    /// the whole-call reference from the same input: after every layer the residual of all rows
    /// has cosine >= 0.9999 (reported: values that differ in bits and the residual's hash), and
    /// the head over every row gives the same greedy ids (reported: KL mean / max, top-1).
    #[test]
    #[ignore = "needs the GPU (about 16 GB VRAM, a 2.3 GB synthetic container in the temp dir, a few minutes): cargo test --release --lib glm5_scratch_gpu_196 -- --ignored --nocapture --test-threads 1"]
    fn glm5_scratch_gpu_196_sub_blocks_match_whole_calls() {
        let g = geo8();
        let s = synth_model(&g, 9_474_048);
        let (spec, _) = crate::nvme_source::glm5_record_of_container(&s.path).unwrap();
        let moe = MoeGeo::new(&g, spec).unwrap();
        let mut cnq = Cnq::open_checked(&s.path).unwrap();
        let n = 8192 + 1029;
        let ids: Vec<i64> = (0..n as i64).map(|i| (i * 131 + 7) % g.vocab as i64).collect();
        let (row, v) = (g.hc_streams * g.hidden, g.vocab);
        unsafe {
            let _ctx = cuda::Ctx::init();
            let x0 = trunk_input(&embed_rows(&mut cnq, &g, &ids), g.hidden, g.hc_streams);
            let mut rep = LoadReport::default();
            let mut layers: Vec<LayerW> = (0..g.layers).map(|l| load_layer(&mut cnq, &g, &moe, l, &mut rep)).collect();
            let mut head = Head::new(head_geo(&g));
            let mut hw = load_head(&mut cnq, &g, &mut rep);
            let (mut normed, mut logits, mut next) = (cuda::alloc_zeroed(n * g.hidden * 4), cuda::alloc_zeroed(n * v * 4), cuda::alloc_zeroed(n * 4));
            let mut worst = 1f64;
            for chunk in [256usize, 1024, 8192] {
                let t0 = std::time::Instant::now();
                let mut pn = Glm5Pass::new(&g, moe, chunk, n);
                let mut pw = Glm5Pass::new_whole(&g, moe, chunk, n);
                let (mut xn, mut xw) = (cuda::alloc_named("196 residual", n * row * 4), cuda::alloc_named("196 residual ref", n * row * 4));
                cuda::to_f32_into(xn, &x0);
                cuda::to_f32_into(xw, &x0);
                for lw in &layers {
                    let tb = match &lw.ffn {
                        FfnW::Moe { table, .. } => *table,
                        FfnW::Dense(_) => 0,
                    };
                    for (p, x) in [(&mut pn, xn), (&mut pw, xw)] {
                        p.begin_layer();
                        for (r0, t) in crate::glm5_tiers::prompt_calls(n, chunk) {
                            let mut hook = |_l: usize, sel: &[i32], run: &mut dyn FnMut(usize, usize, Dev) -> Result<(), String>| run(0, sel.len() / g.topk, tb);
                            p.call_with_expert_batches(lw, x + (r0 * row * 4) as u64, r0, t, &mut hook).unwrap();
                        }
                    }
                    cuda::sync();
                    let (a, b) = (cuda::dtoh(xn, n * row), cuda::dtoh(xw, n * row));
                    let cos = cosine(&a, &b);
                    let differ = a.iter().zip(&b).filter(|(x, y)| x.to_bits() != y.to_bits()).count();
                    eprintln!("#196 chunk {chunk} layer {} ({}): residual cosine {cos:.12}, {differ} of {} values differ in bits, hash {} (whole calls {})", lw.layer, kind_label(&g, lw.layer), a.len(), hash(&a), hash(&b));
                    assert!(a.iter().all(|x| x.is_finite()), "chunk {chunk} layer {}: the synthetic model must stay finite", lw.layer);
                    assert!(cos >= 0.9999, "chunk {chunk} layer {}: residual cosine {cos} below 0.9999", lw.layer);
                    worst = worst.min(cos);
                }
                let mut lg = Vec::new();
                let mut top = Vec::new();
                for x in [xn, xw] {
                    run_head(&pn.kn, &head, &hw, x, normed, logits, next, n);
                    cuda::sync();
                    lg.push(cuda::dtoh(logits, n * v));
                    top.push(cuda::dtoh_i32(next, n));
                }
                let kls: Vec<f64> = (0..n).map(|r| kl(&lg[1][r * v..(r + 1) * v], &lg[0][r * v..(r + 1) * v])).collect();
                let agree = (0..n).filter(|&r| top[0][r] == top[1][r]).count();
                eprintln!(
                    "#196 chunk {chunk}: head over {n} rows: ids {} / {n} identical, KL mean {:.3e} max {:.3e}, logits hash {} (whole calls {}), {:.1} s",
                    agree,
                    kls.iter().sum::<f64>() / n as f64,
                    kls.iter().cloned().fold(0f64, f64::max),
                    hash(&lg[0]),
                    hash(&lg[1]),
                    t0.elapsed().as_secs_f64()
                );
                assert_eq!(top[0], top[1], "chunk {chunk}: greedy ids of the prompt rows");
                cuda::free_dev(&mut xn);
                cuda::free_dev(&mut xw);
                pn.free();
                pw.free();
            }
            eprintln!("#196: worst per-layer residual cosine {worst:.12}");
            for lw in layers.iter_mut() {
                lw.free();
            }
            for d in [&mut normed, &mut logits, &mut next, &mut hw.norm, &mut hw.lm] {
                cuda::free_dev(d);
            }
            head.free();
        }
    }
}
