//! The glm-flash-lite integration: the decode switches of the rebuild branches (global arena,
//! controller + LA, router prefetch, CPU lane split, NVMe piece pool, prefill scratch, batch)
//! together on the synthetic 8-layer model of the stager tests. Every test here needs the GPU:
//! `cargo test --release --lib glm5_int_gpu -- --ignored --nocapture --test-threads 1`.

use crate::cnq::Cnq;
use crate::cuda;
use crate::glm5_flags::tests::{geo8, synth_model};
use crate::glm5_flags::Switches;
use crate::glm5_moe::MoeGeo;
use crate::glm5_tiers::{ExpertTiers, Generated, Glm5Run, TierSizes};

const REC: u64 = 9_474_048;

/// The switches the integration tests set; every one is cleared at the start of an arm and the
/// process values come back on drop (the tests run with `--test-threads 1`).
pub(crate) const KEYS: &[&str] = &[
    "CROW_GLM_ARENA",
    "CROW_GLM_ARENA_ADMIT_MAX",
    "CROW_GLM_ARENA_NOADMIT",
    "CROW_GLM_ARENA_WARM",
    "CROW_GLM_ARENA_VRING",
    "CROW_GLM_ARENA_ELASTIC_GB",
    "CROW_GLM_ARENA_STAGE_GB",
    "CROW_GLM_ARENA_STAGE_MIN",
    "CROW_GLM_ARENA_FREQ",
    "CROW_GLM_ARENA_REGROW",
    "CROW_GLM_ARENA_LAZY_REFILL",
    "CROW_GLM_ARENA_STAGE_LEND",
    "CROW_GLM_STAGE_OVERLAP",
    "CROW_GLM_PREFILL_NVPF",
    "CROW_GLM_PREFILL_NVPF_MIN_ROWS",
    "CROW_GLM_PREFILL_NVPF_EARLY",
    "CROW_GLM_CPU_LANE",
    "CROW_GLM_LANE_THREADS",
    "CROW_GLM_PINNED",
    "CROW_PINNED_ALLOC",
    "CROW_NVME_POOL",
    "CROW_NVME_POOL_THREADS",
    "CROW_NVME_POOL_PIECE_KB",
    "CROW_GLM_HCFUSE",
    "CROW_GLM_DENSE_GEMM",
    "CROW_GLM_MOE_TC",
    "CROW_CHUNK",
    "CROW_GLM_FLAGS",
    "CROW_GLM_STAGER",
    "CROW_GLM_CONTROLLER",
    "CROW_GLM_LA",
    "CROW_GLM_PREFETCH",
    "CROW_GLM_PREFETCH_SIDE",
    "CROW_GLM_SHARED_OVERLAP",
    "CROW_GLM_MAX_BATCH",
    "CROW_GLM_LANES2",
    "CROW_GLM_RT2",
    "CROW_GLM_RT2_TABLE_SM",
    "CROW_GLM_ATTN2",
];

pub(crate) struct Env(Vec<(String, Option<String>)>);

impl Env {
    pub(crate) fn set(kv: &[(&str, String)]) -> Env {
        let old = KEYS.iter().map(|n| (n.to_string(), std::env::var(n).ok())).collect();
        for n in KEYS {
            std::env::remove_var(n);
        }
        for (k, v) in kv {
            assert!(KEYS.contains(k), "{k} is not restored by Env");
            std::env::set_var(k, v);
        }
        Env(old)
    }
}

impl Drop for Env {
    fn drop(&mut self) {
        for (k, v) in &self.0 {
            match v {
                Some(v) => std::env::set_var(k, v),
                None => std::env::remove_var(k),
            }
        }
    }
}

/// one arm: environment, decode switches, the stager
pub(crate) struct Arm {
    pub name: &'static str,
    pub env: Vec<(&'static str, String)>,
    pub sw: Switches,
    pub stager: bool,
}

pub(crate) fn arm(name: &'static str, env: &[(&'static str, &str)], sw: Switches, stager: bool) -> Arm {
    Arm { name, env: env.iter().map(|(k, v)| (*k, v.to_string())).collect(), sw, stager }
}

/// the bits in which two logit rows differ, per generated position
pub(crate) fn bit_diff(a: &[Vec<f32>], b: &[Vec<f32>]) -> Vec<usize> {
    a.iter().zip(b).map(|(x, y)| x.iter().zip(y).filter(|(p, q)| p.to_bits() != q.to_bits()).count()).collect()
}

/// what one arm produced: the run, the tier store's NVMe reads (the cache's own count) and the
/// experts the CPU lane computed
pub(crate) struct Out {
    pub gen: Generated,
    pub nvme_reads: u64,
    pub lane_experts: u64,
    /// #202: the stager's early answers and late experts (0 without the stager)
    pub early: (u64, u64),
}

/// `generate` of the 5-id prompt and `n` greedy ids under every arm, a fresh tier store per arm
/// (VRAM 3 + pinned 4 slots per layer)
pub(crate) fn run_arms(arms: &[Arm], n: usize) -> Vec<Out> {
    run_arms_sized(arms, n, TierSizes { vram: 3, pinned: 4 })
}

/// [`run_arms`] with the tier sizes per layer
pub(crate) fn run_arms_sized(arms: &[Arm], n: usize, sizes: TierSizes) -> Vec<Out> {
    let g = geo8();
    let s = synth_model(&g, REC);
    let (spec, _) = crate::nvme_source::glm5_record_of_container(&s.path).unwrap();
    let moe = MoeGeo::new(&g, spec).unwrap();
    let mut cnq = Cnq::open_checked(&s.path).unwrap();
    let prompt = [3i64, 17, 101, 999, 5];
    let mut outs = Vec::new();
    unsafe {
        let _ctx = cuda::Ctx::init();
        let mut run = Glm5Run::load(&mut cnq, &g, &moe, prompt.len() + n + 1, &mut |s| eprintln!("{s}"));
        for a in arms {
            let _env = Env::set(&a.env);
            run.set_switches(&mut cnq, a.sw);
            let mut tiers = ExpertTiers::new(&cnq, &s.path, &g, &moe, sizes, 1, g.topk).unwrap_or_else(|e| panic!("{}: {e}", a.name));
            tiers.set_stager(a.stager).unwrap_or_else(|e| panic!("{}: {e}", a.name));
            tiers.set_prefetch(a.sw.prefetch);
            let gen = run.generate(&mut cnq, &mut tiers, &prompt, n, true, &mut |_| {}).unwrap_or_else(|e| panic!("{}: {e}", a.name));
            eprintln!("glm5 int {}: ids {:?}, NVMe reads {}, prefetch {:?}", a.name, gen.ids, tiers.nvme_reads, tiers.prefetch_stats());
            let nvme_reads = tiers.nvme_reads;
            let lane_experts = tiers.cpu_lane_clock().read().1;
            // #202: the controller's early answers and the late experts they named
            let early = tiers.stager_stats().map_or((0, 0), |st| (st.early, st.late_items));
            eprintln!("glm5 int {}: early answers {}, late experts {}", a.name, early.0, early.1);
            tiers.free();
            outs.push(Out { gen, nvme_reads, lane_experts, early });
        }
        run.free();
    }
    let finite = outs[0].gen.logits.iter().flatten().filter(|v| v.is_finite()).count();
    assert_eq!(finite, n * g.vocab, "the synthetic model must stay finite for the comparison to mean something");
    outs
}

/// G3 on the logits of one generated position: cosine, KL(p || q) of the softmaxes, top-1 agreement
pub(crate) fn g3(p: &[f32], q: &[f32]) -> (f64, f64, bool) {
    let dot: f64 = p.iter().zip(q).map(|(a, b)| *a as f64 * *b as f64).sum();
    let n = |v: &[f32]| v.iter().map(|a| (*a as f64).powi(2)).sum::<f64>().sqrt();
    let sm = |v: &[f32]| {
        let m = v.iter().fold(f32::NEG_INFINITY, |a, b| a.max(*b)) as f64;
        let e: Vec<f64> = v.iter().map(|a| (*a as f64 - m).exp()).collect();
        let z: f64 = e.iter().sum();
        e.into_iter().map(|x| x / z).collect::<Vec<f64>>()
    };
    let (a, b) = (sm(p), sm(q));
    let kl = a.iter().zip(&b).filter(|(x, _)| **x > 0.0).map(|(x, y)| x * (x / y.max(1e-300)).ln()).sum();
    let arg = |v: &[f32]| v.iter().enumerate().fold(0, |m, (i, x)| if *x > v[m] { i } else { m });
    (dot / (n(p) * n(q)), kl, arg(p) == arg(q))
}

/// robin 2026-10-10: an arm whose placement differs (the RAM-tier prefetch, #203) is held to
/// accuracy, not bits: the reference's ids, and per generated position logits cosine >= 0.9999
/// (G3's bound), KL and top-1 reported
pub(crate) fn assert_g3(name: &str, o: &Out, base: &Out) {
    assert_eq!(o.gen.ids, base.gen.ids, "{name}: ids");
    let r: Vec<(f64, f64, bool)> = o.gen.logits.iter().zip(&base.gen.logits).map(|(p, q)| g3(q, p)).collect();
    let cos = r.iter().map(|x| x.0).fold(1.0, f64::min);
    let kl = r.iter().map(|x| x.1).fold(0.0, f64::max);
    let top1 = r.iter().filter(|x| x.2).count();
    eprintln!("glm5 int G3 {name}: logits cosine min {cos:.7}, KL max {kl:.3e}, top-1 {top1}/{}, bits differ {:?}", r.len(), bit_diff(&o.gen.logits, &base.gen.logits));
    assert!(cos >= 0.9999, "{name}: logits cosine {cos} under G3's 0.9999");
}

/// every arm gives the first arm's ids and logits bit for bit
pub(crate) fn assert_bit_identical(arms: &[Arm], outs: &[Out]) {
    for (a, o) in arms.iter().zip(outs) {
        assert_eq!(o.gen.ids, outs[0].gen.ids, "{}: ids", a.name);
        let d = bit_diff(&o.gen.logits, &outs[0].gen.logits);
        assert!(d.iter().all(|&x| x == 0), "{}: logits differ in bits per generated position {d:?}", a.name);
    }
}

/// Merge fix of glm-router-prefetch into the global arena: the controller's `table_reply`, the
/// prefetch store and its hint read go through the global arena (`table_global`) when
/// `CROW_GLM_ARENA=global`, not through the per-layer cache the arena does not use. Every arm
/// gives the switch-off ids and logits bit for bit (the MUL1 kernels read VRAM and pinned
/// records alike), every global arm without the stager's prefetch the NVMe reads of the global
/// arm without switches (the same arena moves; the per-layer cache reads 236, the arena 376 on
/// this model, so an arm that bypassed the arena shows here). #203: with the stager the guesses
/// go into the arena's pinned tier, so a guessed expert is a pinned hit at its layer: those arms
/// read fewer demand records from the NVMe than the global arm.
#[test]
#[ignore = "needs the GPU (about 2 GB VRAM, a 2.3 GB synthetic container in the temp dir)"]
fn glm5_int_gpu_the_global_arena_serves_the_controller_and_the_prefetch() {
    let ctl = Switches { flags: true, controller: true, ..Switches::default() };
    let gl: &[(&str, &str)] = &[("CROW_GLM_ARENA", "global"), ("CROW_GLM_ARENA_VRING", "2")];
    let arms = [
        arm("off", &[], Switches::default(), false),
        arm("global", gl, Switches::default(), false),
        arm("global flags+prefetch", gl, Switches { flags: true, prefetch: true, ..Switches::default() }, false),
        arm("global flags+stager+prefetch", gl, Switches { flags: true, prefetch: true, ..Switches::default() }, true),
        arm("global ctl", gl, ctl, true),
        arm("global ctl+la+prefetch+side+overlap", gl, Switches { la: true, prefetch: true, pf_side: true, overlap: true, ..ctl }, true),
    ];
    let outs = run_arms(&arms, 6);
    assert_bit_identical(&arms, &outs);
    assert_ne!(outs[0].nvme_reads, outs[1].nvme_reads, "the arena must move differently from the per-layer cache for the check to mean something");
    // #202: under the controller the global arena answers early, with experts that waited for
    // their own landing (so the bit-identity above covers the late pass)
    for (a, o) in arms.iter().zip(&outs).filter(|(a, _)| a.sw.controller) {
        assert!(o.early.0 > 0 && o.early.1 > 0, "{}: early answers {}, late experts {}: the early reply did not run with a late expert", a.name, o.early.0, o.early.1);
    }
    for (a, o) in arms.iter().zip(&outs).skip(2) {
        if a.stager && a.sw.prefetch {
            assert!(o.nvme_reads < outs[1].nvme_reads, "{}: {} demand NVMe reads, the global arm {}: the guesses did not become pinned hits", a.name, o.nvme_reads, outs[1].nvme_reads);
        } else {
            assert_eq!(o.nvme_reads, outs[1].nvme_reads, "{}: NVMe reads of the global arena", a.name);
        }
    }
}

/// Cross-wiring 1: under the NVMe piece pool (`CROW_NVME_POOL=1`) the prefetch store's reads
/// (`Prefetch::issue`) join the pool's `Prefetch` queue, so a demand read overtakes them. One
/// pool worker in 4 KiB pieces, a store of 8 records of 1 MiB issued first, then one 8 KiB
/// demand read: it completes while the store's last record has not landed (the store's reads
/// queued as `Demand` would be one FIFO with it). Without the pool the store reads as before
/// (the per-reader backends have one FIFO each and ignore the queue).
#[test]
#[ignore = "needs the GPU (pinned store): cargo test --release --lib glm5_int_gpu -- --ignored --test-threads 1"]
fn glm5_int_gpu_prefetch_reads_wait_behind_demand_in_the_piece_pool() {
    use crate::geo::ExpertCodec;
    use crate::nvme_source::{ColdSource, ExpertRecord, NvmeConfig, NvmeSource, PoolAsk, PoolConfig, ReadPriority, RecordDst, RecordLayout, Span};
    const LEN: usize = 16 << 20;
    let dir = std::env::temp_dir().join(format!("crow-int-prio-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("raw.bin");
    let bytes: Vec<u8> = (0..LEN).map(|i| (i as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15).rotate_left(17) as u8).collect();
    std::fs::write(&path, &bytes).unwrap();
    let rec = |id: u32, off: u64, len: usize| ExpertRecord { layer: 0, id, gu: Span { off, len }, dn: Span { off: 0, len: 0 }, codec: ExpertCodec::Mul1, layout: RecordLayout::OneUnit };
    let recs: Vec<ExpertRecord> = (0..8).map(|k| rec(k, (k as u64) << 20, 1 << 20)).collect();
    let mut cfg = NvmeConfig::new(&path);
    cfg.pool = PoolAsk::On(PoolConfig { threads: 1, piece: 4096 });
    let src = NvmeSource::open(&cfg).unwrap();
    assert_eq!(crate::glm5_flags::prefetch_priority(&src), ReadPriority::Prefetch);
    unsafe {
        let _ctx = cuda::Ctx::init();
        let mut pf = crate::glm5_flags::Prefetch::new(8, 1 << 20);
        let want: Vec<u32> = (0..8).collect();
        assert_eq!(pf.issue(&src, &recs, 0, &[], &want).unwrap(), 8);
        let layout = std::alloc::Layout::from_size_align(8192, 4096).unwrap();
        let b = std::alloc::alloc(layout);
        let t = src.fetch_prio(&[(rec(99, 12 << 20, 8192), RecordDst { gu: b, dn: std::ptr::null_mut() })], None, ReadPriority::Demand).unwrap();
        src.wait(t).unwrap();
        let last = pf.landed_value(7);
        assert!(std::slice::from_raw_parts(b, 8192) == &bytes[12 << 20..(12 << 20) + 8192]);
        std::alloc::dealloc(b, layout);
        pf.free(&src).unwrap();
        assert!(!last, "the store's last record had landed when the demand read completed: the prefetch reads did not queue as Prefetch");
    }
    drop(src);
    // without the pool the store keeps its former read (Demand; one FIFO per reader)
    let mut cfg = NvmeConfig::new(&path);
    cfg.pool = PoolAsk::Off;
    let off = NvmeSource::open(&cfg).unwrap();
    assert_eq!(crate::glm5_flags::prefetch_priority(&off), ReadPriority::Demand);
    drop(off);
    let _ = std::fs::remove_dir_all(&dir);
}

/// Cross-wiring 4: `MlaScratch::begin` stages the call's `[pos0, t]` without a host sync, so a
/// row enqueued ahead (`CROW_GLM_LA`) does not block the host at its first DSA layer. A kernel
/// holds the stream for 300 ms, then `begin` 48 times (fewer than the ring's entries; a real row
/// stages 11 DSA calls at most, two rows in flight with LA): the host returns long before the
/// kernel ends, and the device holds the last call's values once the stream passed. Then 200
/// more calls wrap the ring three times and the last one is on the device.
#[test]
#[ignore = "needs the GPU: cargo test --release --lib glm5_int_gpu -- --ignored --test-threads 1"]
fn glm5_int_gpu_the_mla_call_scalars_upload_without_a_host_sync() {
    const SLOW: &str = r#"
extern "C" __global__ void hold(long long ns)
{
    unsigned long long t0, t;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t0));
    do { asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t)); } while ((long long) (t - t0) < ns);
}
"#;
    let g = crate::geo::Glm5Geo::GLM_5_3_FLASH;
    let d = crate::glm5_mla::MlaDims::of(&g);
    unsafe {
        let _ctx = cuda::Ctx::init();
        let mut m = cuda::compile(SLOW);
        let hold = m.get("hold");
        let mut sc = crate::glm5_mla::MlaScratch::new(&d, 4, 1024);
        crate::kernels::launch_v(hold, 1, 1, 1, 32, &[300_000_000u64]);
        let t0 = std::time::Instant::now();
        for i in 0..48usize {
            sc.begin(i % 900, 1 + i % 4);
        }
        let host = t0.elapsed();
        cuda::sync();
        let waited = t0.elapsed();
        let st: Vec<i32> = cuda::dtoh_t(sc.st_dev(), 2);
        eprintln!("glm5 int mla begin: 48 calls {host:?} on the host, stream done after {waited:?}, st {st:?}");
        assert_eq!(st, vec![47, 4], "the device holds the last call");
        assert!(waited.as_millis() >= 250, "the kernel must hold the stream for the check to mean something");
        assert!(host.as_millis() < 100, "begin waited for the stream: {host:?}");
        for i in 0..200usize {
            sc.begin(i, 1 + i % 3);
        }
        cuda::sync();
        assert_eq!(cuda::dtoh_t::<i32>(sc.st_dev(), 2), vec![199, 2], "after the ring wrapped");
        sc.free();
        m.unload();
    }
}

/// Cross-wiring 3: with `CROW_GLM_ARENA=global` + `CROW_GLM_ARENA_ELASTIC_GB` the prompt phase
/// borrows its scratch. `Glm5Run::load` holds the decode rows only (1); `prefill` hands back the
/// elastic chunks its scratch takes, runs its prompt calls at `CROW_CHUNK` rows and gives the
/// scratch back (the pass holds 1 row again); the chunks grow back and the elastic part takes the
/// borrowed scratch on top (`elastic_lift`). Ids
/// and logits of a prompt + greedy run equal, bit for bit, the same chunk without the borrow on
/// the per-layer path and on the global arena without an elastic part.
#[test]
#[ignore = "needs the GPU (about 2 GB VRAM, a 2.3 GB synthetic container in the temp dir)"]
fn glm5_int_gpu_the_prompt_borrows_its_scratch_from_the_elastic_arena() {
    let g = geo8();
    let s = synth_model(&g, REC);
    let (spec, _) = crate::nvme_source::glm5_record_of_container(&s.path).unwrap();
    let moe = MoeGeo::new(&g, spec).unwrap();
    let mut cnq = Cnq::open_checked(&s.path).unwrap();
    let prompt: Vec<i64> = (0..11).map(|i| (i * 37 + 5) % 2048).collect();
    let n = 6;
    let sizes = TierSizes { vram: 3, pinned: 4 };
    let elastic = format!("{}", 4.0 * 3.0 * REC as f64 / (1u64 << 30) as f64);
    // #196 NVPF: the overlap path with the stage engine's ring reading the next layers ahead
    // (staging on for its engine, never staging a call: the minimum is above every call's picks)
    let stage = format!("{}", g.experts as f64 * REC as f64 / (1u64 << 30) as f64);
    let nvpf = |elastic: Option<&String>| -> Vec<(&str, String)> {
        let mut v = vec![
            ("CROW_CHUNK", "4".to_string()),
            ("CROW_GLM_ARENA", "global".into()),
            ("CROW_GLM_STAGE_OVERLAP", "1".into()),
            ("CROW_GLM_ARENA_STAGE_GB", stage.clone()),
            ("CROW_GLM_ARENA_STAGE_MIN", "1000000".into()),
            ("CROW_GLM_PREFILL_NVPF", "1".into()),
            ("CROW_GLM_PREFILL_NVPF_MIN_ROWS", "1".into()),
        ];
        if let Some(e) = elastic {
            v.push(("CROW_GLM_ARENA_ELASTIC_GB", e.clone()));
        }
        v
    };
    // #196 NVPF early: the plan from each prompt call's embedding on, every NVMe-tier expert
    let early = |elastic: Option<&String>| -> Vec<(&str, String)> {
        let mut v = nvpf(elastic);
        v.push(("CROW_GLM_PREFILL_NVPF_EARLY", "1".into()));
        v
    };
    let arms: [(&str, Vec<(&str, String)>, bool); 9] = [
        ("chunk 4", vec![("CROW_CHUNK", "4".into())], false),
        ("chunk 4 global", vec![("CROW_CHUNK", "4".into()), ("CROW_GLM_ARENA", "global".into())], false),
        ("chunk 4 global elastic", vec![("CROW_CHUNK", "4".into()), ("CROW_GLM_ARENA", "global".into()), ("CROW_GLM_ARENA_ELASTIC_GB", elastic.clone())], true),
        // #196: the prompt sub-batches on the copy stream (CROW_GLM_STAGE_OVERLAP=1)
        ("chunk 4 global overlap", vec![("CROW_CHUNK", "4".into()), ("CROW_GLM_ARENA", "global".into()), ("CROW_GLM_STAGE_OVERLAP", "1".into())], false),
        (
            "chunk 4 global elastic overlap",
            vec![("CROW_CHUNK", "4".into()), ("CROW_GLM_ARENA", "global".into()), ("CROW_GLM_ARENA_ELASTIC_GB", elastic.clone()), ("CROW_GLM_STAGE_OVERLAP", "1".into())],
            true,
        ),
        ("chunk 4 global overlap nvpf", nvpf(None), false),
        ("chunk 4 global elastic overlap nvpf", nvpf(Some(&elastic)), true),
        ("chunk 4 global overlap nvpf early", early(None), false),
        ("chunk 4 global elastic overlap nvpf early", early(Some(&elastic)), true),
    ];
    let mut outs: Vec<Generated> = Vec::new();
    unsafe {
        let _ctx = cuda::Ctx::init();
        for (name, env, borrow) in &arms {
            let _env = Env::set(env);
            let mut run = Glm5Run::load(&mut cnq, &g, &moe, prompt.len() + n + 1, &mut |s| eprintln!("{s}"));
            assert_eq!(run.rows_held(), if *borrow { 1 } else { 4 }, "{name}: rows held at load");
            let mut tiers = ExpertTiers::new(&cnq, &s.path, &g, &moe, sizes, 1, g.topk).unwrap();
            if *borrow {
                let (live, all, v) = tiers.elastic_live().unwrap();
                assert!(all >= 2 && live == all && v == 3, "{name}: the elastic part at construction ({live} of {all} chunks)");
                // the prompt alone: chunks handed back for it, scratch back after it, and the
                // elastic part grown back on the host thread at its end (`decode_ready`)
                let e0 = tiers.arena_elastic_stats().unwrap();
                let id = run.prefill(&mut cnq, &mut tiers, &prompt, 0, &mut |_| {}).unwrap();
                let e1 = tiers.arena_elastic_stats().unwrap();
                assert_eq!(run.rows_held(), 1, "{name}: the prompt's scratch went back");
                // only the chunks the prompt's scratch took went back (none when free VRAM held it)
                // and each hand-back grew back; the borrowed scratch then lifted the part
                assert_eq!(e1.enter - e0.enter, e1.exit - e0.exit, "{name}: every hand-back grew back ({e0:?} -> {e1:?})");
                assert!(e1.lifted >= 1 && e1.lift_bytes == run.borrow_bytes(4), "{name}: the elastic part took the borrowed scratch ({e1:?})");
                let (live1, all1, _) = tiers.elastic_live().unwrap();
                assert!(live1 == all1 && all1 == all + e1.lifted, "{name}: every elastic chunk live after the prompt ({live1} of {all1})");
                run.row(&mut cnq, &mut tiers, id, prompt.len(), true).unwrap();
                assert_eq!(tiers.elastic_live().unwrap().0, all1, "{name}: every elastic chunk live in decode");
                eprintln!("glm5 int borrow: elastic {all} chunks x {v} slots handed back for the prompt and grown back after it ({e0:?} -> {e1:?})");
                tiers.reset_cache().unwrap();
                for k in run.kda_states() {
                    k.reset();
                }
            }
            let gen = run.generate(&mut cnq, &mut tiers, &prompt, n, true, &mut |_| {}).unwrap();
            eprintln!("glm5 int {name}: ids {:?}, NVMe reads {}", gen.ids, tiers.nvme_reads);
            if name.ends_with("nvpf") || name.ends_with("nvpf early") {
                let st = tiers.arena_stage_stats().unwrap().0;
                eprintln!("glm5 int {name}: {st:?}");
                assert!(st.nvpf_calls > 0 && st.nvpf_copied > 0 && st.calls == 0, "{name}: the prompt calls ran on the read-ahead plan ({st:?})");
                if name.ends_with("early") {
                    // one plan per prompt call, opened at its embedding and kept by its MoE layers;
                    // the expert-major calls read nothing on the host path
                    assert_eq!(st.nvpf_host, 0, "{name}: every NVMe record out of the ring ({st:?})");
                    let prefills = 1 + u64::from(*borrow);
                    assert_eq!(st.nvpf_plans, prefills * prompt.len().div_ceil(4) as u64, "{name}: one plan per prompt call ({st:?})");
                }
            }
            assert_eq!(run.rows_held(), if *borrow { 1 } else { 4 }, "{name}: rows held after the run");
            tiers.free();
            run.free();
            outs.push(gen);
        }
    }
    for (i, (name, _, _)) in arms.iter().enumerate() {
        assert_eq!(outs[i].ids, outs[0].ids, "{name}: ids");
        let d = bit_diff(&outs[i].logits, &outs[0].logits);
        assert!(d.iter().all(|&x| x == 0), "{name}: logits differ in bits per generated position {d:?}");
    }
}

/// Cross-wiring 3, the plan: the VRAM hot set per MoE layer during decode on the RTX 5090 at
/// 200,000 rows (the #186 planner test's card), chunk 1 / 2048 / 8192, with the prompt scratch
/// booked (the plan of record: chunk 8192 leaves 42 slots per layer, #196) and borrowed from the elastic
/// part (`decode_hot_per_layer`, on the plan's numbers: the scratch under the plan's ceiling turns
/// into elastic chunks above the 2.5 GiB reserve), elastic part unbounded and at 10 GiB (the
/// template's serving setting).
#[test]
fn glm5_int_decode_hot_set_with_the_borrowed_prompt_scratch() {
    use crate::glm5_tiers::decode_hot_per_layer;
    use crate::geo::HOST_PINNED_CAP;
    use crate::manager::{glm5_chunk_scratch_bytes, plan_glm5_next_chunk};
    const CARD: u64 = 32_607 << 20;
    const MUL1: u64 = 9_474_048;
    let g = crate::geo::Glm5Geo::GLM_5_3_FLASH;
    let ml = g.moe_layers();
    let mut rows = Vec::new();
    for chunk in [1usize, 2048, 8192] {
        let (_, _, p) = plan_glm5_next_chunk(&g, 200_000, CARD, HOST_PINNED_CAP, crate::geo::GLM5_NEXT_DENSE_BYTES, MUL1, 64, true, chunk).unwrap();
        let sc = glm5_chunk_scratch_bytes(&g, chunk, 200_000);
        let booked = decode_hot_per_layer(&p, sc, false, MUL1, ml, u64::MAX);
        let borrowed = decode_hot_per_layer(&p, sc, true, MUL1, ml, u64::MAX);
        let borrowed10 = decode_hot_per_layer(&p, sc, true, MUL1, ml, 10 << 30);
        eprintln!(
            "glm5 decode hot set per layer, chunk {chunk}: plan {} ; scratch {:.2} GiB booked: {:.1} ({} elastic chunks) ; borrowed: {:.1} ({} chunks), at 10 GiB elastic {:.1} ({} chunks)",
            p.hot, sc as f64 / (1u64 << 30) as f64, booked.1, booked.0, borrowed.1, borrowed.0, borrowed10.1, borrowed10.0
        );
        rows.push((chunk, p.hot, booked.1, borrowed.1));
    }
    let (c1, c8) = (rows[0], rows[2]);
    assert_eq!(c1.2, c1.3, "chunk 1 books no prompt scratch");
    // #196: the prompt scratch at chunk 8192 is 2.79 GiB (15.87 GiB before: 7 slots, +35 borrowed)
    assert_eq!(c8.1, 42, "the plan of record at chunk 8192 (#186, #196)");
    assert!(c8.3 >= c8.2 && c8.3 <= c1.1 as f64, "chunk 8192: the borrowed scratch holds experts during decode ({:.1} vs {:.1})", c8.3, c8.2);
}

/// Cross-wiring 2: the CPU lane (`1` and `split`) with the global arena, the stager, the
/// controller and its lookahead. The lane's CPU combos have other bits than the GPU's, so a
/// comparison holds when the same experts go to the CPU: with every expert in pinned (V 0 + P 16
/// of 16 per layer) `1` gives every pick to the CPU on every path (under the controller every resident one, #202), and `split` plans from the same
/// heat on every path (every call counts it). Every arm without the controller gives the bits of
/// the synchronous per-layer lane of its mode, and the CPU computed experts in every arm. Then
/// V 3 + P 4: the controller (with LA and the prefetch) against flags + stager with the lane on,
/// per layer and on the global arena (same placement, same plan). #202 D-C: under the controller
/// the CPU sums its experts weighted into one row that one kernel adds behind the combine, another
/// f32 order than the combine over every combo, so the controller's arms are held to G3 (robin
/// 2026-10-10: accuracy, not bits): the same ids, logits cosine >= 0.9999.
#[test]
#[ignore = "needs the GPU (about 2 GB VRAM, a 2.3 GB synthetic container in the temp dir)"]
fn glm5_int_gpu_the_cpu_lane_runs_with_the_arena_the_stager_and_the_controller() {
    let fs = Switches { flags: true, ..Switches::default() };
    let ctl = Switches { flags: true, controller: true, ..Switches::default() };
    let la = Switches { la: true, prefetch: true, overlap: true, ..ctl };
    for mode in ["1", "split"] {
        let lane: &[(&str, &str)] = &[("CROW_GLM_CPU_LANE", mode), ("CROW_PINNED_ALLOC", "host")];
        let gl = [lane, &[("CROW_GLM_ARENA", "global"), ("CROW_GLM_ARENA_VRING", "0")]].concat();
        let arms = [
            arm("lane", lane, Switches::default(), false),
            arm("lane global", &gl, Switches::default(), false),
            arm("lane flags+stager", lane, fs, true),
            arm("lane global flags+stager", &gl, fs, true),
            arm("lane ctl", lane, ctl, true),
            arm("lane ctl+la+prefetch+overlap", lane, la, true),
            arm("lane global ctl+la+prefetch+overlap", &gl, la, true),
        ];
        let outs = run_arms_sized(&arms, 6, TierSizes { vram: 0, pinned: 16 });
        for (a, o) in arms.iter().zip(&outs) {
            eprintln!("glm5 int lane {mode} {}: CPU experts {}", a.name, o.lane_experts);
            assert!(o.lane_experts > 0, "{mode} {}: the CPU lane computed nothing", a.name);
        }
        assert_bit_identical(&arms[..4], &outs[..4]);
        for (a, o) in arms.iter().zip(&outs).skip(4) {
            assert_g3(&format!("lane {mode} {}", a.name), o, &outs[0]);
        }
        if mode == "1" {
            assert!(outs[..4].iter().all(|o| o.lane_experts == outs[0].lane_experts), "every pick on the CPU on every path without the controller");
            // #202: under the controller a pick still landing from the NVMe goes to the GPU
            assert!(outs[4..].iter().all(|o| o.lane_experts < outs[0].lane_experts), "the controller's lane took picks still landing");
        }
    }
    // mixed tiers: the controller with the lane against flags + stager with the lane
    let lane: &[(&str, &str)] = &[("CROW_GLM_CPU_LANE", "split"), ("CROW_PINNED_ALLOC", "host")];
    let gl = [lane, &[("CROW_GLM_ARENA", "global"), ("CROW_GLM_ARENA_VRING", "2")]].concat();
    for (what, env) in [("per layer", lane.to_vec()), ("global", gl)] {
        let arms = [
            arm("lane flags+stager", &env, fs, true),
            arm("lane ctl", &env, ctl, true),
            arm("lane ctl+la+prefetch+overlap", &env, la, true),
        ];
        let outs = run_arms(&arms, 6);
        for (a, o) in arms.iter().zip(&outs) {
            eprintln!("glm5 int lane split V3 P4 {what} {}: CPU experts {}, ids {:?}", a.name, o.lane_experts, o.gen.ids);
        }
        assert!(outs[0].lane_experts > 0, "{what}: the CPU lane computed nothing");
        // #202 D-C: the controller's lane row (and #203 on the global arena: the guesses become
        // pinned hits, so the split hands the CPU other experts): held to G3
        for (a, o) in arms.iter().zip(&outs).skip(1) {
            assert_g3(&format!("lane split V3 P4 {what} {}", a.name), o, &outs[0]);
        }
    }
}

/// The "all template switches on" arm on the synthetic model, sized for it (the real container's
/// values are in the integration report): NVMe piece pool, fused mHC, the global arena with warm
/// start, write-back ring, elastic part and staged prompt calls, prompt calls of 12 rows (the
/// borrowed scratch), flags + stager + router prefetch (side stream) + shared overlap +
/// controller + LA, `CROW_GLM_MAX_BATCH=8`. `warm` is the synthetic warm-start file.
pub(crate) fn full_arm_env(warm: &str) -> Vec<(&'static str, String)> {
    let gib = |records: f64| format!("{}", records * REC as f64 / (1u64 << 30) as f64);
    vec![
        ("CROW_NVME_POOL", "1".into()),
        ("CROW_GLM_HCFUSE", "1".into()),
        ("CROW_GLM_ARENA", "global".into()),
        ("CROW_GLM_ARENA_WARM", warm.into()),
        ("CROW_GLM_ARENA_VRING", "2".into()),
        ("CROW_GLM_ARENA_ELASTIC_GB", gib(2.0 * 3.0)),
        ("CROW_GLM_ARENA_STAGE_GB", gib(12.0)),
        ("CROW_GLM_ARENA_STAGE_MIN", "16".into()),
        ("CROW_CHUNK", "12".into()),
        ("CROW_GLM_FLAGS", "1".into()),
        ("CROW_GLM_STAGER", "1".into()),
        ("CROW_GLM_PREFETCH", "1".into()),
        ("CROW_GLM_PREFETCH_SIDE", "1".into()),
        ("CROW_GLM_SHARED_OVERLAP", "1".into()),
        ("CROW_GLM_CONTROLLER", "1".into()),
        ("CROW_GLM_LA", "1".into()),
        ("CROW_GLM_MAX_BATCH", "8".into()),
    ]
}

/// a warm-start file for the synthetic model's 5 MoE layers x 16 experts
pub(crate) fn synth_warm(dir: &std::path::Path) -> String {
    let w = dir.join("int-warm.json");
    std::fs::write(&w, serde_json::to_string(&(0..5).map(|l| (0..16).map(|e| ((e * 7 + l) % 16) as f64).collect::<Vec<_>>()).collect::<Vec<_>>()).unwrap()).unwrap();
    w.to_str().unwrap().to_string()
}

/// `generate` of `prompt` and `n` greedy ids per arm, each arm with its own `Glm5Run::load` (the
/// prompt chunk, the batch slots and the decode switches are read at load) and tier store
pub(crate) fn run_loaded_arms(arms: &[(&str, Vec<(&'static str, String)>)], prompt: &[i64], n: usize, sizes: TierSizes) -> Vec<Out> {
    let g = geo8();
    let s = synth_model(&g, REC);
    let (spec, _) = crate::nvme_source::glm5_record_of_container(&s.path).unwrap();
    let moe = MoeGeo::new(&g, spec).unwrap();
    let mut cnq = Cnq::open_checked(&s.path).unwrap();
    let mut outs = Vec::new();
    unsafe {
        let _ctx = cuda::Ctx::init();
        for (name, env) in arms {
            let env: Vec<(&str, String)> = env.iter().map(|(k, v)| (*k, v.clone())).collect();
            let _env = Env::set(&env);
            let mut run = Glm5Run::load(&mut cnq, &g, &moe, prompt.len() + n + 1, &mut |s| eprintln!("{s}"));
            let mut tiers = ExpertTiers::new(&cnq, &s.path, &g, &moe, sizes, 1, g.topk).unwrap_or_else(|e| panic!("{name}: {e}"));
            let gen = run.generate(&mut cnq, &mut tiers, prompt, n, true, &mut |_| {}).unwrap_or_else(|e| panic!("{name}: {e}"));
            let lane_experts = tiers.cpu_lane_clock().read().1;
            eprintln!(
                "glm5 int full {name}: ids {:?}, NVMe reads {}, CPU experts {lane_experts}, prefetch {:?}, elastic {:?}, rows held {}",
                gen.ids,
                tiers.nvme_reads,
                tiers.prefetch_stats(),
                tiers.elastic_live(),
                run.rows_held()
            );
            let nvme_reads = tiers.nvme_reads;
            let early = tiers.stager_stats().map_or((0, 0), |st| (st.early, st.late_items));
            eprintln!("glm5 int full {name}: early answers {}, late experts {}", early.0, early.1);
            tiers.free();
            run.free();
            outs.push(Out { gen, nvme_reads, lane_experts, early });
        }
    }
    let finite = outs[0].gen.logits.iter().flatten().filter(|v| v.is_finite()).count();
    assert_eq!(finite, n * g.vocab, "the synthetic model must stay finite for the comparison to mean something");
    outs
}

fn assert_same(names: &[&str], outs: &[Out]) {
    for (name, o) in names.iter().zip(outs) {
        assert_eq!(o.gen.ids, outs[0].gen.ids, "{name}: ids against {}", names[0]);
        let d = bit_diff(&o.gen.logits, &outs[0].gen.logits);
        assert!(d.iter().all(|&x| x == 0), "{name}: logits differ in bits from {} per generated position {d:?}", names[0]);
    }
}

/// The full arm against the default path on the synthetic model, a 20-id prompt and 6 greedy ids.
/// A prompt in calls of 12 rows (`CROW_CHUNK`, #186) has the ids of the row-by-row prompt but not
/// its bits (documented with #186: bit-identical to a layer-at-a-time chain), so every arm is held
/// against the default path at the same prompt chunk:
/// - V 3 + P 4: the full arm without the lane = the default at chunk 12, and without the chunk =
///   the default row by row; ids equal across all four.
/// - V 0 + P 16 (every expert pinned) with `CROW_GLM_CPU_LANE=split` (`CROW_PINNED_ALLOC=host`):
///   the lane's CPU combos have other bits than the GPU's by design (#188), so the full arm with
///   the lane is held against the default path with the same lane (same CPU set: same heat, no
///   VRAM hit), at chunk 12 and row by row; to G3, since under the controller the CPU sums its
///   experts into one row added behind the combine (#202 D-C, another f32 order).
#[test]
#[ignore = "needs the GPU (about 3 GB VRAM, a 2.3 GB synthetic container in the temp dir)"]
fn glm5_int_gpu_the_full_template_arm_is_the_default_path() {
    let dir = std::env::temp_dir().join(format!("crow-int-full-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let warm = synth_warm(&dir);
    let full = full_arm_env(&warm);
    let no_chunk = |v: &[(&'static str, String)]| -> Vec<(&'static str, String)> { v.iter().filter(|(k, _)| *k != "CROW_CHUNK").cloned().collect() };
    let chunk: Vec<(&'static str, String)> = vec![("CROW_CHUNK", "12".into())];
    let lane: Vec<(&'static str, String)> = vec![("CROW_GLM_CPU_LANE", "split".into()), ("CROW_PINNED_ALLOC", "host".into())];
    let plus = |a: &[(&'static str, String)], b: &[(&'static str, String)]| -> Vec<(&'static str, String)> { a.iter().chain(b).cloned().collect() };
    let prompt: Vec<i64> = (0..20).map(|i| (i * 53 + 11) % 2048).collect();
    let n = 6;
    let a = [("default", Vec::new()), ("full without the lane and the chunk", no_chunk(&full)), ("default chunk 12", chunk.clone()), ("full without the lane", full.clone())];
    let outs = run_loaded_arms(&a, &prompt, n, TierSizes { vram: 3, pinned: 4 });
    assert_same(&["default V3 P4", "full without the lane and the chunk V3 P4"], &outs[..2]);
    assert_same(&["default chunk 12 V3 P4", "full without the lane V3 P4"], &outs[2..]);
    assert!(outs.iter().all(|o| o.gen.ids == outs[0].gen.ids), "ids across the chunk");
    let b = [("lane split", lane.clone()), ("full with the lane, without the chunk", plus(&no_chunk(&full), &lane)), ("lane split chunk 12", plus(&chunk, &lane)), ("full", plus(&full, &lane))];
    let outs = run_loaded_arms(&b, &prompt, n, TierSizes { vram: 0, pinned: 16 });
    assert!(outs.iter().all(|o| o.lane_experts > 0), "the lane ran in every arm");
    assert_g3("full with the lane, without the chunk V0 P16", &outs[1], &outs[0]);
    assert_g3("full V0 P16", &outs[3], &outs[2]);
    let _ = std::fs::remove_dir_all(&dir);
}

/// The measurement arm of the GLM-5.3-Flash rows (global arena with warm start, elastic and stage
/// budgets, NVMe piece pool of 8 workers, HC fuse, #186 dense tensor-core GEMM, prompt chunk 8192,
/// flags + stager + prefetch on the side stream + shared overlap + controller + LA, CPU lane split
/// on host pinned memory) against the default path (every switch off) on the synthetic model, a
/// 300-id prompt (one prompt call) and 6 greedy ids. The tensor-core GEMM and the lane change bits
/// by design (#186, #188), so the bar is accuracy, not bits: the same ids, and per generated row a
/// logit cosine >= 0.9999 (KL reported). Two sizings: the arm's own budgets (10 GB elastic, 2.6 GB
/// stage: the whole synthetic model fits) at V 3 + P 4, and budgets of a few records (the arena
/// spills to the tiers, the piece pool reads) at V 3 + P 4 and V 0 + P 16 (every expert pinned,
/// the lane computes). The warm file is the synthetic model's (the real one has the real geometry).
#[test]
#[ignore = "needs the GPU (about 12 GB VRAM, a 2.3 GB synthetic container in the temp dir)"]
fn glm5_int_gpu_the_measurement_arm_is_the_default_path_within_the_accuracy_bar() {
    let dir = std::env::temp_dir().join(format!("crow-int-arm-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let warm = synth_warm(&dir);
    let s = |v: &str| v.to_string();
    let arm_env = |elastic: String, stage: String| -> Vec<(&'static str, String)> {
        vec![
            ("CROW_NVME_POOL", s("1")),
            ("CROW_NVME_POOL_THREADS", s("8")),
            ("CROW_GLM_HCFUSE", s("1")),
            ("CROW_GLM_DENSE_GEMM", s("1")),
            ("CROW_GLM_CPU_LANE", s("split")),
            ("CROW_PINNED_ALLOC", s("host")),
            ("CROW_GLM_ARENA", s("global")),
            ("CROW_GLM_ARENA_WARM", warm.clone()),
            ("CROW_GLM_ARENA_ELASTIC_GB", elastic),
            ("CROW_GLM_ARENA_STAGE_GB", stage),
            ("CROW_CHUNK", s("8192")),
            ("CROW_GLM_FLAGS", s("1")),
            ("CROW_GLM_STAGER", s("1")),
            ("CROW_GLM_PREFETCH", s("1")),
            ("CROW_GLM_PREFETCH_SIDE", s("1")),
            ("CROW_GLM_SHARED_OVERLAP", s("1")),
            ("CROW_GLM_CONTROLLER", s("1")),
            ("CROW_GLM_LA", s("1")),
        ]
    };
    let gib = |records: f64| format!("{}", records * REC as f64 / (1u64 << 30) as f64);
    let literal = arm_env(s("10"), s("2.6"));
    let small = arm_env(gib(2.0 * 3.0), gib(12.0));
    let prompt: Vec<i64> = (0..300).map(|i| (i * 77 + 3) % 2048).collect();
    let n = 6;
    let cos = |a: &[f32], b: &[f32]| -> f64 {
        let (mut ab, mut aa, mut bb) = (0f64, 0f64, 0f64);
        for (x, y) in a.iter().zip(b) {
            let (x, y) = (*x as f64, *y as f64);
            ab += x * y;
            aa += x * x;
            bb += y * y;
        }
        ab / (aa.sqrt() * bb.sqrt())
    };
    let kl = |p: &[f32], q: &[f32]| -> f64 {
        let sm = |v: &[f32]| -> Vec<f64> {
            let m = v.iter().cloned().fold(f32::MIN, f32::max) as f64;
            let e: Vec<f64> = v.iter().map(|x| (*x as f64 - m).exp()).collect();
            let z: f64 = e.iter().sum();
            e.into_iter().map(|x| x / z).collect()
        };
        let (p, q) = (sm(p), sm(q));
        p.iter().zip(&q).map(|(a, b)| if *a > 0.0 { a * (a / b.max(1e-300)).ln() } else { 0.0 }).sum()
    };
    let check = |what: &str, outs: &[Out]| {
        let (d, a) = (&outs[0].gen, &outs[1].gen);
        let c: Vec<f64> = d.logits.iter().zip(&a.logits).map(|(x, y)| cos(x, y)).collect();
        let k: Vec<f64> = d.logits.iter().zip(&a.logits).map(|(x, y)| kl(x, y)).collect();
        eprintln!("glm5 int arm {what}: ids default {:?} arm {:?}, logit cosine {c:.9?}, KL {:?}, arm NVMe reads {}, CPU experts {}", d.ids, a.ids, k.iter().map(|x| format!("{x:.3e}")).collect::<Vec<_>>(), outs[1].nvme_reads, outs[1].lane_experts);
        assert_eq!(a.ids, d.ids, "{what}: the arm's ids against the default path");
        assert!(c.iter().all(|&x| x >= 0.9999), "{what}: logit cosine below 0.9999: {c:?}");
    };
    let outs = run_loaded_arms(&[("default", Vec::new()), ("arm 10 / 2.6 GB", literal)], &prompt, n, TierSizes { vram: 3, pinned: 4 });
    check("10 / 2.6 GB V3 P4", &outs);
    let outs = run_loaded_arms(&[("default", Vec::new()), ("arm small budgets", small.clone())], &prompt, n, TierSizes { vram: 3, pinned: 4 });
    check("small budgets V3 P4", &outs);
    let outs = run_loaded_arms(&[("default", Vec::new()), ("arm small budgets", small)], &prompt, n, TierSizes { vram: 0, pinned: 16 });
    assert!(outs[1].lane_experts > 0, "the lane ran in the arm");
    check("small budgets V0 P16", &outs);
    let _ = std::fs::remove_dir_all(&dir);
}

/// #202 lanes: the measurement arm with `CROW_GLM_LANES2=1` (the CPU lane takes only resident
/// records, top-k late slots with null spares, no moves-word wait without a copy the experts read,
/// guessed reads in the pool's prefetch queue and moved up when joined) against the default path
/// and against the measurement arm without it, on the synthetic model (300-id prompt, 6 greedy
/// ids), small budgets (the arena spills, the pool reads) at V 3 + P 4 and V 0 + P 16 (every
/// expert pinned, the lane computes): the same ids and a logit cosine >= 0.9999 per generated
/// row against both (the lane's picks may differ, so accuracy, not bits).
#[test]
#[ignore = "needs the GPU (about 12 GB VRAM, a 2.3 GB synthetic container in the temp dir)"]
fn glm5_int_gpu_lanes2_is_the_measurement_arm_within_the_accuracy_bar() {
    let dir = std::env::temp_dir().join(format!("crow-int-lanes2-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let warm = synth_warm(&dir);
    let s = |v: &str| v.to_string();
    let gib = |records: f64| format!("{}", records * REC as f64 / (1u64 << 30) as f64);
    let arm: Vec<(&'static str, String)> = vec![
        ("CROW_NVME_POOL", s("1")),
        ("CROW_NVME_POOL_THREADS", s("8")),
        ("CROW_GLM_HCFUSE", s("1")),
        ("CROW_GLM_DENSE_GEMM", s("1")),
        ("CROW_GLM_CPU_LANE", s("split")),
        ("CROW_PINNED_ALLOC", s("host")),
        ("CROW_GLM_ARENA", s("global")),
        ("CROW_GLM_ARENA_WARM", warm.clone()),
        ("CROW_GLM_ARENA_ELASTIC_GB", gib(2.0 * 3.0)),
        ("CROW_GLM_ARENA_STAGE_GB", gib(12.0)),
        ("CROW_CHUNK", s("8192")),
        ("CROW_GLM_FLAGS", s("1")),
        ("CROW_GLM_STAGER", s("1")),
        ("CROW_GLM_PREFETCH", s("1")),
        ("CROW_GLM_PREFETCH_SIDE", s("1")),
        ("CROW_GLM_SHARED_OVERLAP", s("1")),
        ("CROW_GLM_CONTROLLER", s("1")),
        ("CROW_GLM_LA", s("1")),
    ];
    let mut lanes2 = arm.clone();
    lanes2.push(("CROW_GLM_LANES2", s("1")));
    let prompt: Vec<i64> = (0..300).map(|i| (i * 77 + 3) % 2048).collect();
    let n = 6;
    for (what, sizes) in [("V3 P4", TierSizes { vram: 3, pinned: 4 }), ("V0 P16", TierSizes { vram: 0, pinned: 16 })] {
        let outs = run_loaded_arms(&[("default", Vec::new()), ("arm", arm.clone()), ("arm lanes2", lanes2.clone())], &prompt, n, sizes);
        eprintln!(
            "glm5 int lanes2 {what}: NVMe reads arm {} lanes2 {}, CPU experts arm {} lanes2 {}, early answers / late experts arm {:?} lanes2 {:?}",
            outs[1].nvme_reads, outs[2].nvme_reads, outs[1].lane_experts, outs[2].lane_experts, outs[1].early, outs[2].early
        );
        assert_g3(&format!("lanes2 {what} vs default"), &outs[2], &outs[0]);
        assert_g3(&format!("lanes2 {what} vs arm"), &outs[2], &outs[1]);
        if sizes.vram == 0 {
            assert!(outs[2].lane_experts > 0, "{what}: the lane ran under lanes2");
        }
        assert!(outs[2].early.0 > 0, "{what}: the controller answered early under lanes2");
    }
    let _ = std::fs::remove_dir_all(&dir);
}

/// #202 RT2: the decode arm of record without the controller ("ARM2": global arena with warm
/// start, small elastic and stage budgets so the arena spills and the NVMe reads, HC fuse, dense
/// GEMM, prompt chunk 8192, flags + stager + prefetch on the side stream + shared overlap,
/// `CROW_GLM_PINNED=zerocopy`, CPU lane split on host pinned memory) with and without
/// `CROW_GLM_RT2=1`, on the synthetic model (300-id prompt, 6 greedy ids) at V 3 + P 4 and
/// V 0 + P 16 (every expert pinned, the lane computes). Without the lane the RT2 arm has the
/// arm's ids and logits bit for bit (the same experts on the GPU, the same combine sum); with the
/// lane the same ids and a logit cosine >= 0.9999 per generated row (RT2's lane leaves a pick still
/// landing or written back to the GPU, so the CPU set may differ). Every RT2 arm ran its decode
/// layers through `experts_rt2`. #202 `CROW_GLM_RT2_TABLE_SM=1` (the tables through the SMs, not
/// the copy engine): without the lane the RT2 arm's ids and logits bit for bit, with the lane the
/// arm's ids within the same bar.
#[test]
#[ignore = "needs the GPU (about 12 GB VRAM, a 2.3 GB synthetic container in the temp dir)"]
fn glm5_int_gpu_rt2_is_the_arm_without_the_controller_within_the_accuracy_bar() {
    let dir = std::env::temp_dir().join(format!("crow-int-rt2-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let warm = synth_warm(&dir);
    let s = |v: &str| v.to_string();
    let gib = |records: f64| format!("{}", records * REC as f64 / (1u64 << 30) as f64);
    let base: Vec<(&'static str, String)> = vec![
        ("CROW_GLM_HCFUSE", s("1")),
        ("CROW_GLM_DENSE_GEMM", s("1")),
        ("CROW_GLM_PINNED", s("zerocopy")),
        ("CROW_PINNED_ALLOC", s("host")),
        ("CROW_GLM_ARENA", s("global")),
        ("CROW_GLM_ARENA_WARM", warm.clone()),
        ("CROW_GLM_ARENA_ELASTIC_GB", gib(2.0 * 3.0)),
        ("CROW_GLM_ARENA_STAGE_GB", gib(12.0)),
        ("CROW_CHUNK", s("8192")),
        ("CROW_GLM_FLAGS", s("1")),
        ("CROW_GLM_STAGER", s("1")),
        ("CROW_GLM_PREFETCH", s("1")),
        ("CROW_GLM_PREFETCH_SIDE", s("1")),
        ("CROW_GLM_SHARED_OVERLAP", s("1")),
    ];
    let plus = |a: &[(&'static str, String)], b: &[(&'static str, &str)]| -> Vec<(&'static str, String)> { a.iter().cloned().chain(b.iter().map(|(k, v)| (*k, v.to_string()))).collect() };
    let lane = plus(&base, &[("CROW_GLM_CPU_LANE", "split")]);
    let prompt: Vec<i64> = (0..300).map(|i| (i * 77 + 3) % 2048).collect();
    let n = 6;
    let calls = || crate::glm5_moe::lane::RT2_CALLS.load(std::sync::atomic::Ordering::Relaxed);
    for (what, sizes) in [("V3 P4", TierSizes { vram: 3, pinned: 4 }), ("V0 P16", TierSizes { vram: 0, pinned: 16 })] {
        let c0 = calls();
        let sm = [("CROW_GLM_RT2", "1"), ("CROW_GLM_RT2_TABLE_SM", "1")];
        let outs = run_loaded_arms(&[("arm no lane", base.clone()), ("arm no lane rt2", plus(&base, &[("CROW_GLM_RT2", "1")])), ("arm no lane rt2 sm", plus(&base, &sm))], &prompt, n, sizes);
        let c1 = calls();
        eprintln!("glm5 int rt2 {what} no lane: NVMe reads arm {} rt2 {} rt2 sm {}, RT2 calls {}", outs[0].nvme_reads, outs[1].nvme_reads, outs[2].nvme_reads, c1 - c0);
        assert_same(&["arm no lane", "arm no lane rt2", "arm no lane rt2 sm"], &outs);
        assert_eq!(outs[0].nvme_reads, outs[1].nvme_reads, "{what}: RT2 moves no record differently");
        assert_eq!(outs[0].nvme_reads, outs[2].nvme_reads, "{what}: RT2 with the SM tables moves no record differently");
        assert!(c1 > c0, "{what}: no decode layer ran through experts_rt2");
        let outs = run_loaded_arms(&[("arm", lane.clone()), ("arm rt2", plus(&lane, &[("CROW_GLM_RT2", "1")])), ("arm rt2 sm", plus(&lane, &sm))], &prompt, n, sizes);
        let c2 = calls();
        eprintln!(
            "glm5 int rt2 {what} lane split: NVMe reads arm {} rt2 {}, CPU experts arm {} rt2 {}, RT2 calls {}",
            outs[0].nvme_reads,
            outs[1].nvme_reads,
            outs[0].lane_experts,
            outs[1].lane_experts,
            c2 - c1
        );
        assert_g3(&format!("rt2 {what} lane split vs arm"), &outs[1], &outs[0]);
        assert_g3(&format!("rt2 sm {what} lane split vs arm"), &outs[2], &outs[0]);
        assert!(c2 > c1, "{what}: no decode layer ran through experts_rt2");
        if sizes.vram == 0 {
            assert!(outs[1].lane_experts > 0, "{what}: the lane ran under RT2");
        }
    }
    let _ = std::fs::remove_dir_all(&dir);
}

/// #202 / #203 / #209, the synthetic counters: the template arm (global arena, write-back ring,
/// NVMe piece pool of 8 workers, flags + stager + controller + LA + prefetch on the side stream +
/// shared overlap) and flags + stager + prefetch without the controller, 5-id prompt + 10 ids,
/// V 3 + P 4 per layer. Per row: NVMe records read from the drive (demand and prefetch), records
/// in flight when a layer is answered, the prefetch's issued / used / wasted / joins. A
/// measurement: it prints; it holds the counters consistent and each arm to G3 against the
/// default path (the joins' device waits: a kernel that read a pinned slot before its record
/// landed would show here).
#[test]
#[ignore = "needs the GPU (about 2 GB VRAM, a 2.3 GB synthetic container in the temp dir)"]
fn glm5_int_gpu_nvme_overlap_counters() {
    let g = geo8();
    let s = synth_model(&g, REC);
    let (spec, _) = crate::nvme_source::glm5_record_of_container(&s.path).unwrap();
    let moe = MoeGeo::new(&g, spec).unwrap();
    let mut cnq = Cnq::open_checked(&s.path).unwrap();
    let prompt = [3i64, 17, 101, 999, 5];
    let n = 10;
    let ctl = Switches { flags: true, controller: true, la: true, prefetch: true, pf_side: true, overlap: true, ..Switches::default() };
    let fsp = Switches { flags: true, prefetch: true, ..Switches::default() };
    let env: &[(&str, &str)] = &[("CROW_GLM_ARENA", "global"), ("CROW_GLM_ARENA_VRING", "2"), ("CROW_NVME_POOL", "1"), ("CROW_NVME_POOL_THREADS", "8")];
    let arms = [arm("default", &[], Switches::default(), false), arm("template ctl", env, ctl, true), arm("flags+stager+prefetch", env, fsp, true)];
    let mut outs: Vec<Out> = Vec::new();
    unsafe {
        let _ctx = cuda::Ctx::init();
        let mut run = Glm5Run::load(&mut cnq, &g, &moe, prompt.len() + n + 1, &mut |s| eprintln!("{s}"));
        for a in &arms {
            let _env = Env::set(&a.env);
            run.set_switches(&mut cnq, a.sw);
            let mut tiers = ExpertTiers::new(&cnq, &s.path, &g, &moe, TierSizes { vram: 3, pinned: 4 }, 1, g.topk).unwrap();
            tiers.set_stager(a.stager).unwrap();
            tiers.set_prefetch(a.sw.prefetch);
            let gen = run.generate(&mut cnq, &mut tiers, &prompt, n, true, &mut |_| {}).unwrap();
            let rows = (prompt.len() + n - 1) as f64;
            let (drive, left) = tiers.nvme_io();
            let Some(st) = tiers.stager_stats() else {
                outs.push(Out { gen, nvme_reads: tiers.nvme_reads, lane_experts: 0, early: (0, 0) });
                tiers.free();
                continue;
            };
            let pf = tiers.prefetch_stats().unwrap();
            eprintln!(
                "glm5 int counters {}: ids {:?}; per row: demand NVMe reads {:.2}, drive records {:.2}; records in flight per answered layer {:.3} ({} answers); prefetch issued {} used {} wasted {} joins {}; in flight at the end {left}",
                a.name,
                gen.ids,
                tiers.nvme_reads as f64 / rows,
                drive as f64 / rows,
                st.inflight_at_answer as f64 / st.answers.max(1) as f64,
                st.answers,
                pf.issued,
                pf.used,
                pf.wasted,
                pf.joins
            );
            assert!(st.answers > 0 && drive >= pf.issued, "{}: the counters", a.name);
            outs.push(Out { gen, nvme_reads: tiers.nvme_reads, lane_experts: 0, early: (0, 0) });
            tiers.free();
        }
        run.free();
    }
    for (a, o) in arms.iter().zip(&outs).skip(1) {
        assert_g3(a.name, o, &outs[0]);
    }
}

/// #202 D4: the prefetch store never waits for its own reads. A half of 8 store records of
/// 1 MiB read through one pool worker in 4 KiB pieces, then the next guess of the same parity:
/// its `issue` returns while those reads still run (the slots still in flight are skipped, not
/// waited for), and `forget` then drains them.
#[test]
#[ignore = "needs the GPU (pinned store): cargo test --release --lib glm5_int_gpu -- --ignored --test-threads 1"]
fn glm5_int_gpu_the_prefetch_store_does_not_wait_for_its_reads() {
    use crate::geo::ExpertCodec;
    use crate::nvme_source::{ExpertRecord, NvmeConfig, NvmeSource, PoolAsk, PoolConfig, RecordLayout, Span};
    const LEN: usize = 16 << 20;
    let dir = std::env::temp_dir().join(format!("crow-int-retire-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("raw.bin");
    std::fs::write(&path, vec![7u8; LEN]).unwrap();
    let rec = |id: u32, off: u64| ExpertRecord { layer: 0, id, gu: Span { off, len: 1 << 20 }, dn: Span { off: 0, len: 0 }, codec: ExpertCodec::Mul1, layout: RecordLayout::OneUnit };
    let recs: Vec<ExpertRecord> = (0..16).map(|k| rec(k, (k as u64) << 20)).collect();
    let mut cfg = NvmeConfig::new(&path);
    cfg.pool = PoolAsk::On(PoolConfig { threads: 1, piece: 4096 });
    let src = NvmeSource::open(&cfg).unwrap();
    unsafe {
        let _ctx = cuda::Ctx::init();
        let mut pf = crate::glm5_flags::Prefetch::new(8, 1 << 20);
        assert_eq!(pf.issue(&src, &recs, 0, &[], &(0..8).collect::<Vec<u32>>()).unwrap(), 8);
        let t0 = std::time::Instant::now();
        let n = pf.issue(&src, &recs, 2, &[], &(8..16).collect::<Vec<u32>>()).unwrap();
        let dt = t0.elapsed();
        let first_left = !pf.landed_value(7);
        eprintln!("glm5 int store: the second issue took {dt:?}, issued {n}, the first half's last read still running {first_left}");
        assert_eq!(n, 0, "the second issue rewrote slots whose reads it had waited for (issued {n})");
        assert!(first_left, "the first half's reads landed before the second issue returned");
        pf.forget(&src).unwrap();
        assert!(pf.landed_value(7), "forget drains the reads");
        pf.free(&src).unwrap();
    }
    drop(src);
    let _ = std::fs::remove_dir_all(&dir);
}

/// #202: the ARM2 measurement env of the real container (`runs/glm53-flash/quick/arm2.env`:
/// global arena with warm start, 10 GB elastic / 2.6 GB stage, CPU lane split on host pinned
/// memory, zero-copy pinned tier, flags + stager + router prefetch on the side stream + shared
/// overlap, HC fuse, dense GEMM, MoE TC 2, ATTN2, prompt chunk 8192 with stage overlap and NVPF)
/// on the synthetic model, a 40-id prompt and 8 greedy ids, two generations in one run (as
/// `glm5_run --reps 2`): after the first decode row of a generation (its warm-up) no decode row
/// allocates, frees, clears synchronously or reads the free VRAM (`cuda::mem_api_counts`). Then
/// the real card's case after a long prompt: the elastic part handed back and the free VRAM
/// held under the reserve by a ballast, so no chunk can grow back; a third generation's decode
/// rows still read no free VRAM after the first one (before #202 every row read it once per
/// chunk left down). Prints a digest of every generation's ids and logit bits (the before/after
/// check of #202).
#[test]
#[ignore = "needs the GPU (about 3 GB VRAM, a 2.3 GB synthetic container in the temp dir)"]
fn glm5_int_gpu_decode_rows_allocate_nothing_after_warm_up() {
    use crate::cuda::MemApiCounts;
    use crate::glm5_tiers::ARENA_RESERVE_BYTES;
    let dir = std::env::temp_dir().join(format!("crow-int-alloc-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let warm = synth_warm(&dir);
    let s = |v: &str| v.to_string();
    let env: Vec<(&'static str, String)> = vec![
        ("CROW_GLM_PINNED", s("zerocopy")),
        ("CROW_GLM_FLAGS", s("1")),
        ("CROW_GLM_STAGER", s("1")),
        ("CROW_GLM_ARENA", s("global")),
        ("CROW_GLM_ARENA_WARM", warm),
        ("CROW_GLM_ARENA_ELASTIC_GB", s("10")),
        ("CROW_GLM_ARENA_STAGE_GB", s("2.6")),
        ("CROW_GLM_CPU_LANE", s("split")),
        ("CROW_PINNED_ALLOC", s("host")),
        ("CROW_GLM_PREFETCH", s("1")),
        ("CROW_GLM_PREFETCH_SIDE", s("1")),
        ("CROW_GLM_SHARED_OVERLAP", s("1")),
        ("CROW_GLM_HCFUSE", s("1")),
        ("CROW_GLM_DENSE_GEMM", s("1")),
        ("CROW_CHUNK", s("8192")),
        ("CROW_GLM_STAGE_OVERLAP", s("1")),
        ("CROW_GLM_MOE_TC", s("2")),
        ("CROW_GLM_ATTN2", s("1")),
        ("CROW_GLM_PREFILL_NVPF", s("1")),
    ];
    let g = geo8();
    let sy = synth_model(&g, REC);
    let (spec, _) = crate::nvme_source::glm5_record_of_container(&sy.path).unwrap();
    let moe = MoeGeo::new(&g, spec).unwrap();
    let mut cnq = Cnq::open_checked(&sy.path).unwrap();
    let prompt: Vec<i64> = (0..40).map(|i| (i * 61 + 7) % 2048).collect();
    let n = 8;
    let digest = |gen: &Generated| -> u64 {
        let mut h = 0xcbf2_9ce4_8422_2325u64;
        for v in gen.ids.iter().map(|&i| i as u64).chain(gen.logits.iter().flatten().map(|x| x.to_bits() as u64)) {
            h = (h ^ v).wrapping_mul(0x0100_0000_01b3);
        }
        h
    };
    // the calls of every decode row (the report after it, against the report before it)
    let decode_rows = |run: &mut Glm5Run, cnq: &mut Cnq, tiers: &mut ExpertTiers| -> (Generated, Vec<MemApiCounts>) {
        let mut last = cuda::mem_api_counts();
        let mut rows = Vec::new();
        let mut report = |r: &crate::glm5_tiers::TokenReport| {
            let now = cuda::mem_api_counts();
            if !r.prompt {
                rows.push(now.since(&last));
            }
            last = now;
        };
        // SAFETY: the context is current for the whole test (`cuda::Ctx::init` below)
        let gen = unsafe { run.generate(cnq, tiers, &prompt, n, true, &mut report) }.unwrap();
        (gen, rows)
    };
    unsafe {
        let _ctx = cuda::Ctx::init();
        let _env = Env::set(&env);
        let mut run = Glm5Run::load(&mut cnq, &g, &moe, prompt.len() + n + 1, &mut |s| eprintln!("{s}"));
        let mut tiers = ExpertTiers::new(&cnq, &sy.path, &g, &moe, TierSizes { vram: 3, pinned: 4 }, 1, g.topk).unwrap();
        let mut gens = Vec::new();
        for rep in 0..2 {
            let (gen, rows) = decode_rows(&mut run, &mut cnq, &mut tiers);
            eprintln!("glm5 int alloc rep {rep}: ids {:?}, digest {:016x}, decode rows {}, calls per decode row {rows:?}", gen.ids, digest(&gen), rows.len());
            assert_eq!(rows.len(), n - 1, "rep {rep}: one report per decode row");
            for (i, c) in rows.iter().enumerate().skip(1) {
                assert_eq!(*c, MemApiCounts::default(), "rep {rep}: decode row {i} called the driver's memory API");
            }
            gens.push(gen);
        }
        assert_eq!(gens[1].ids, gens[0].ids, "the second generation's ids");
        // every elastic chunk handed back, the free VRAM held under the reserve: none grows back
        tiers.elastic_hand_back(1 << 40).unwrap();
        let (live, all, vpl) = tiers.elastic_live().unwrap();
        assert!(live == 0 && all > 0, "every elastic chunk handed back ({live} of {all} live)");
        let leave = ARENA_RESERVE_BYTES + vpl as u64 * REC / 2;
        let free = cuda::free_vram_bytes();
        assert!(free > leave + (1 << 30), "{free} B free VRAM");
        let mut ballast = cuda::try_alloc_zeroed("test ballast", (free - leave) as usize).unwrap();
        let fail0 = tiers.arena_elastic_stats().unwrap().realloc_fail;
        let (gen, rows) = decode_rows(&mut run, &mut cnq, &mut tiers);
        let e = tiers.arena_elastic_stats().unwrap();
        eprintln!("glm5 int alloc tight: ids {:?}, digest {:016x}, calls per decode row {rows:?}, elastic {e:?}", gen.ids, digest(&gen));
        assert_eq!(tiers.elastic_live().unwrap().0, 0, "no elastic chunk grew back under the ballast");
        assert!(e.realloc_fail > fail0, "the decode rows tried the chunks");
        for (i, c) in rows.iter().enumerate().skip(1) {
            assert_eq!(*c, MemApiCounts::default(), "tight: decode row {i} called the driver's memory API");
        }
        cuda::free_dev(&mut ballast);
        tiers.free();
        run.free();
    }
    let _ = std::fs::remove_dir_all(&dir);
}

/// #188 `CROW_GLM_ARENA_REGROW`: the real card's case after setup on the synthetic model. The
/// ARM2 measurement env + RT2 + `CROW_GLM_ARENA_FREQ=1` (prompt chunk 8192: the prompt phase
/// borrows its scratch from the elastic part), an elastic part of 12 chunks (the arena below the
/// model's 80 experts, so a hand-back writes experts back) and the prefill staging set at setup
/// (as glm5_run); right after construction a ballast leaves the free VRAM 2.5 chunks under the
/// reserve (the RTX 5090 has 1.15-1.44 GiB free after setup, the reserve is 2.5 GiB), so the
/// prompt's hand-back pays the reserve back. Off: those chunks stay down through the decode
/// (today). On: they grow back once at the prompt's end, as far as the free VRAM read there allows
/// down to the free VRAM of before the prompt, and are refilled before the first decode row. From
/// the prompt's last call to the end of the first decode row the switch reads the free VRAM no
/// more often than today's tries, and no decode row after the first calls the driver's memory API
/// (#202). Only where records lie changes: ids equal off/on, logits to G3 (cosine >= 0.9999; the
/// CPU lane's combos have other bits than the GPU's, so a record served from VRAM instead of
/// pinned changes bits by design, as in `glm5_int_gpu_the_full_template_arm_is_the_default_path`).
#[test]
#[ignore = "needs the GPU (most of its free VRAM as a ballast, a 2.3 GB synthetic container in the temp dir)"]
fn glm5_int_gpu_the_prompts_chunks_grow_back_at_its_end() {
    use crate::cuda::MemApiCounts;
    use crate::glm5_tiers::ARENA_RESERVE_BYTES;
    let dir = std::env::temp_dir().join(format!("crow-int-regrow-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let warm = synth_warm(&dir);
    let s = |v: &str| v.to_string();
    let base: Vec<(&'static str, String)> = vec![
        ("CROW_GLM_PINNED", s("zerocopy")),
        ("CROW_GLM_FLAGS", s("1")),
        ("CROW_GLM_STAGER", s("1")),
        ("CROW_GLM_ARENA", s("global")),
        ("CROW_GLM_ARENA_WARM", warm),
        ("CROW_GLM_ARENA_ELASTIC_GB", format!("{}", 12.0 * 3.0 * REC as f64 / (1u64 << 30) as f64)),
        ("CROW_GLM_ARENA_STAGE_GB", s("2.6")),
        ("CROW_GLM_ARENA_FREQ", s("1")),
        ("CROW_GLM_CPU_LANE", s("split")),
        ("CROW_PINNED_ALLOC", s("host")),
        ("CROW_GLM_PREFETCH", s("1")),
        ("CROW_GLM_PREFETCH_SIDE", s("1")),
        ("CROW_GLM_SHARED_OVERLAP", s("1")),
        ("CROW_GLM_HCFUSE", s("1")),
        ("CROW_GLM_DENSE_GEMM", s("1")),
        ("CROW_CHUNK", s("8192")),
        ("CROW_GLM_STAGE_OVERLAP", s("1")),
        ("CROW_GLM_MOE_TC", s("2")),
        ("CROW_GLM_ATTN2", s("1")),
        ("CROW_GLM_PREFILL_NVPF", s("1")),
        ("CROW_GLM_RT2", s("1")),
    ];
    let g = geo8();
    let sy = synth_model(&g, REC);
    let (spec, _) = crate::nvme_source::glm5_record_of_container(&sy.path).unwrap();
    let moe = MoeGeo::new(&g, spec).unwrap();
    let mut cnq = Cnq::open_checked(&sy.path).unwrap();
    let prompt: Vec<i64> = (0..40).map(|i| (i * 61 + 7) % 2048).collect();
    let n = 8;
    // (generation, live elastic chunks after it, of all, the calls of every decode row: the first
    // one's from the prompt's last report on, the prompt's end included)
    let mut outs: Vec<(Generated, usize, usize, Vec<MemApiCounts>)> = Vec::new();
    unsafe {
        let _ctx = cuda::Ctx::init();
        for regrow in [false, true] {
            let mut env = base.clone();
            env.push(("CROW_GLM_ARENA_REGROW", s(if regrow { "1" } else { "0" })));
            let _env = Env::set(&env);
            let mut run = Glm5Run::load(&mut cnq, &g, &moe, prompt.len() + n + 1, &mut |s| eprintln!("{s}"));
            let mut tiers = ExpertTiers::new(&cnq, &sy.path, &g, &moe, TierSizes { vram: 3, pinned: 4 }, 1, g.topk).unwrap();
            assert_eq!(tiers.arena_config().unwrap().regrow, regrow);
            // the prefill staging set at setup, as glm5_run allocates it (not lazily in the prompt)
            tiers.alloc_prefill_stage(crate::glm5_tiers::prefill_stage_slots(g.topk)).unwrap();
            let (live0, all, vpl) = tiers.elastic_live().unwrap();
            assert!(live0 == all && all == 12, "the elastic part at construction ({live0} of {all} chunks)");
            let cb = vpl as u64 * REC;
            let leave = ARENA_RESERVE_BYTES - 5 * cb / 2;
            let free = cuda::free_vram_bytes();
            assert!(free > leave + (1 << 30), "{free} B free VRAM");
            let mut ballast = cuda::try_alloc_zeroed("test ballast", (free - leave) as usize).unwrap();
            let mut last = cuda::mem_api_counts();
            let mut rows = Vec::new();
            let mut report = |r: &crate::glm5_tiers::TokenReport| {
                let now = cuda::mem_api_counts();
                if !r.prompt {
                    rows.push(now.since(&last));
                }
                last = now;
            };
            let gen = run.generate(&mut cnq, &mut tiers, &prompt, n, true, &mut report).unwrap();
            let e = tiers.arena_elastic_stats().unwrap();
            let (live, _, _) = tiers.elastic_live().unwrap();
            let left = tiers.arena().unwrap().enabled_vram();
            eprintln!("glm5 int regrow {regrow}: ids {:?}, elastic {live} of {all} chunks live, {left} VRAM slots, calls per decode row {rows:?}, {e:?}", gen.ids);
            assert!(e.enter >= 1 && e.write_backs > 0, "the prompt handed chunks back ({e:?})");
            if regrow {
                assert_eq!(e.regrows, 1, "one regrowth at the prompt's end ({e:?})");
                assert_eq!(e.regrow_floor_bytes, e.enter_free_bytes.min(ARENA_RESERVE_BYTES), "the floor: the free VRAM before the hand-back");
                let down = all - live;
                assert!(e.regrown >= 1 && e.exit_free_bytes - e.regrown * cb >= e.regrow_floor_bytes, "never below the floor ({e:?})");
                assert!(down == 0 || e.exit_free_bytes - e.regrown * cb < e.regrow_floor_bytes + cb, "every chunk that fits above the floor ({e:?})");
                assert!(e.refilled > 0, "the brought-back slots refilled before the decode ({e:?})");
            } else {
                assert_eq!(e.regrows, 0);
                assert!(live < all, "today: the chunks that paid the reserve back stay down ({live} of {all})");
            }
            for (i, c) in rows.iter().enumerate().skip(1) {
                assert_eq!(*c, MemApiCounts::default(), "regrow {regrow}: decode row {i} called the driver's memory API");
            }
            cuda::free_dev(&mut ballast);
            tiers.free();
            run.free();
            outs.push((gen, live, all, rows));
        }
    }
    let (off, on) = (&outs[0], &outs[1]);
    assert!(on.1 > off.1, "more elastic chunks live in decode with the regrowth ({} vs {})", on.1, off.1);
    assert!(on.3[0].infos <= off.3[0].infos, "the regrowth reads the free VRAM no more often than today's tries ({:?} vs {:?})", on.3[0], off.3[0]);
    assert_eq!(on.0.ids, off.0.ids, "ids");
    let r: Vec<(f64, f64, bool)> = on.0.logits.iter().zip(&off.0.logits).map(|(p, q)| g3(q, p)).collect();
    let cos = r.iter().map(|x| x.0).fold(1.0, f64::min);
    eprintln!("glm5 int regrow on vs off: logits cosine min {cos:.7}, KL max {:.3e}, bits differ {:?}", r.iter().map(|x| x.1).fold(0.0, f64::max), bit_diff(&on.0.logits, &off.0.logits));
    assert!(cos >= 0.9999, "logits cosine {cos} under G3's 0.9999");
    let _ = std::fs::remove_dir_all(&dir);
}

/// #188 T (`CROW_GLM_ARENA_LAZY_REFILL`) and the lent prefill staging set
/// (`CROW_GLM_ARENA_STAGE_LEND`) on the operating set of `glm5_int_gpu_the_prompts_chunks_grow_back_at_its_end`
/// (ARM2 + RT2 + FREQ + REGROW, the ballast that makes the prompt hand chunks back), two
/// generations per arm (the second prompt takes the lent set back, its end lends it again). Only
/// where records lie changes: every arm gives the off arm's ids, logits to G3 (cosine >= 0.9999).
/// Lazy: the regrown (and lent) slots are not refilled; lend: the set's slots join the decode.
#[test]
#[ignore = "needs the GPU (most of its free VRAM as a ballast, a 2.3 GB synthetic container in the temp dir)"]
fn glm5_int_gpu_lazy_refill_and_the_lent_staging_set_keep_the_ids() {
    use crate::glm5_tiers::ARENA_RESERVE_BYTES;
    let dir = std::env::temp_dir().join(format!("crow-int-lend-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let warm = synth_warm(&dir);
    let s = |v: &str| v.to_string();
    let base: Vec<(&'static str, String)> = vec![
        ("CROW_GLM_PINNED", s("zerocopy")),
        ("CROW_GLM_FLAGS", s("1")),
        ("CROW_GLM_STAGER", s("1")),
        ("CROW_GLM_ARENA", s("global")),
        ("CROW_GLM_ARENA_WARM", warm),
        ("CROW_GLM_ARENA_ELASTIC_GB", format!("{}", 12.0 * 3.0 * REC as f64 / (1u64 << 30) as f64)),
        ("CROW_GLM_ARENA_STAGE_GB", s("2.6")),
        ("CROW_GLM_ARENA_FREQ", s("1")),
        ("CROW_GLM_ARENA_REGROW", s("1")),
        ("CROW_GLM_CPU_LANE", s("split")),
        ("CROW_PINNED_ALLOC", s("host")),
        ("CROW_GLM_PREFETCH", s("1")),
        ("CROW_GLM_PREFETCH_SIDE", s("1")),
        ("CROW_GLM_SHARED_OVERLAP", s("1")),
        ("CROW_GLM_HCFUSE", s("1")),
        ("CROW_GLM_DENSE_GEMM", s("1")),
        ("CROW_CHUNK", s("8192")),
        ("CROW_GLM_STAGE_OVERLAP", s("1")),
        ("CROW_GLM_MOE_TC", s("2")),
        ("CROW_GLM_ATTN2", s("1")),
        ("CROW_GLM_PREFILL_NVPF", s("1")),
        ("CROW_GLM_RT2", s("1")),
    ];
    let g = geo8();
    let sy = synth_model(&g, REC);
    let (spec, _) = crate::nvme_source::glm5_record_of_container(&sy.path).unwrap();
    let moe = MoeGeo::new(&g, spec).unwrap();
    let mut cnq = Cnq::open_checked(&sy.path).unwrap();
    let prompt: Vec<i64> = (0..40).map(|i| (i * 61 + 7) % 2048).collect();
    let n = 8;
    let pf = crate::glm5_tiers::prefill_stage_slots(g.topk);
    // per arm: both generations, the VRAM slots in the second decode, the elastic part's counters
    let mut outs: Vec<(&str, Vec<Generated>, usize)> = Vec::new();
    unsafe {
        let _ctx = cuda::Ctx::init();
        for (name, lazy, lend) in [("off", false, false), ("lazy", true, false), ("lend", false, true), ("lend+lazy", true, true)] {
            let mut env = base.clone();
            env.push(("CROW_GLM_ARENA_LAZY_REFILL", s(if lazy { "1" } else { "0" })));
            env.push(("CROW_GLM_ARENA_STAGE_LEND", s(if lend { "1" } else { "0" })));
            let _env = Env::set(&env);
            let mut run = Glm5Run::load(&mut cnq, &g, &moe, prompt.len() + n + 1, &mut |s| eprintln!("{s}"));
            let mut tiers = ExpertTiers::new(&cnq, &sy.path, &g, &moe, TierSizes { vram: 3, pinned: 4 }, 1, g.topk).unwrap();
            let c = tiers.arena_config().unwrap();
            assert_eq!((c.lazy_refill, c.stage_lend), (lazy, lend), "{name}: the switches");
            tiers.alloc_prefill_stage(pf).unwrap();
            let (_, _, vpl) = tiers.elastic_live().unwrap();
            let leave = ARENA_RESERVE_BYTES - 5 * (vpl as u64 * REC) / 2;
            let free = cuda::free_vram_bytes();
            assert!(free > leave + (1 << 30), "{free} B free VRAM");
            let mut ballast = cuda::try_alloc_zeroed("test ballast", (free - leave) as usize).unwrap();
            let mut gens = Vec::new();
            for _ in 0..2 {
                gens.push(run.generate(&mut cnq, &mut tiers, &prompt, n, true, &mut |_| {}).unwrap());
            }
            let e = tiers.arena_elastic_stats().unwrap();
            // the VRAM slots in decode beside the live elastic chunks (how many regrow depends on the
            // free VRAM each arm reads)
            let (live, _, _) = tiers.elastic_live().unwrap();
            let all = tiers.arena().unwrap().enabled_vram();
            let slots = all - live * vpl;
            eprintln!("glm5 int lend {name}: ids {:?} / {:?}, {all} VRAM slots in decode ({live} elastic chunks live), lent {}, {e:?}", gens[0].ids, gens[1].ids, tiers.stage_lent());
            assert!(e.regrows >= 2 && e.regrown > 0, "{name}: the prompts' chunks regrown ({e:?})");
            if lazy {
                assert!(e.refilled == 0 && e.lazy_refills >= 2 && e.lazy_slots > 0, "{name}: the slots left to admission ({e:?})");
            } else {
                assert!(e.refilled > 0 && e.lazy_refills == 0, "{name}: refilled ({e:?})");
            }
            if lend {
                assert!(tiers.stage_lent(), "{name}: lent in the decode");
                assert_eq!((e.lend_slots, e.lends, e.take_backs), (pf as u64, 2, 1), "{name}: lent after each prompt, taken back by the second ({e:?})");
            } else {
                assert!(!tiers.stage_lent() && e.lends == 0, "{name}: never lent");
            }
            cuda::free_dev(&mut ballast);
            tiers.free();
            run.free();
            outs.push((name, gens, slots));
        }
    }
    let off = &outs[0];
    for (name, gens, slots) in &outs {
        let lend = name.starts_with("lend");
        assert_eq!(*slots, off.2 + if lend { pf } else { 0 }, "{name}: the VRAM slots in decode beside the elastic chunks");
        for (i, (gen, base)) in gens.iter().zip(&off.1).enumerate() {
            assert_eq!(gen.ids, base.ids, "{name}: ids of generation {i}");
            let r: Vec<(f64, f64, bool)> = gen.logits.iter().zip(&base.logits).map(|(p, q)| g3(q, p)).collect();
            let cos = r.iter().map(|x| x.0).fold(1.0, f64::min);
            eprintln!("glm5 int lend {name} vs off, generation {i}: logits cosine min {cos:.7}, bits differ {:?}", bit_diff(&gen.logits, &base.logits));
            assert!(cos >= 0.9999, "{name}: logits cosine {cos} under G3's 0.9999");
        }
    }
    let _ = std::fs::remove_dir_all(&dir);
}
