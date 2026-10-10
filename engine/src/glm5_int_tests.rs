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
    "CROW_GLM_CPU_LANE",
    "CROW_GLM_LANE_THREADS",
    "CROW_GLM_PINNED",
    "CROW_PINNED_ALLOC",
    "CROW_NVME_POOL",
    "CROW_NVME_POOL_THREADS",
    "CROW_NVME_POOL_PIECE_KB",
    "CROW_GLM_HCFUSE",
    "CROW_CHUNK",
    "CROW_GLM_FLAGS",
    "CROW_GLM_STAGER",
    "CROW_GLM_CONTROLLER",
    "CROW_GLM_LA",
    "CROW_GLM_PREFETCH",
    "CROW_GLM_PREFETCH_SIDE",
    "CROW_GLM_SHARED_OVERLAP",
    "CROW_GLM_MAX_BATCH",
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
            tiers.free();
            outs.push(Out { gen, nvme_reads, lane_experts });
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
    let arms: [(&str, Vec<(&str, String)>, bool); 3] = [
        ("chunk 4", vec![("CROW_CHUNK", "4".into())], false),
        ("chunk 4 global", vec![("CROW_CHUNK", "4".into()), ("CROW_GLM_ARENA", "global".into())], false),
        ("chunk 4 global elastic", vec![("CROW_CHUNK", "4".into()), ("CROW_GLM_ARENA", "global".into()), ("CROW_GLM_ARENA_ELASTIC_GB", elastic.clone())], true),
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
/// booked (the plan of record: chunk 8192 leaves 7 slots per layer) and borrowed from the elastic
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
    assert_eq!(c8.1, 7, "the plan of record at chunk 8192 (#186)");
    assert!(c8.3 > c8.2 + 20.0, "chunk 8192: the borrowed scratch holds experts during decode ({:.1} vs {:.1})", c8.3, c8.2);
}

/// Cross-wiring 2: the CPU lane (`1` and `split`) with the global arena, the stager, the
/// controller and its lookahead. The lane's CPU combos have other bits than the GPU's, so a
/// comparison holds when the same experts go to the CPU: with every expert in pinned (V 0 + P 16
/// of 16 per layer) `1` gives every pick to the CPU on every path, and `split` plans from the same
/// heat on every path (every call counts it). Every arm gives the bits of the synchronous
/// per-layer lane of its mode, and the CPU computed experts in every arm. Then V 3 + P 4: the
/// controller (with LA and the prefetch) gives the bits of flags + stager with the lane on, per
/// layer and on the global arena (same placement, same plan).
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
        assert_bit_identical(&arms, &outs);
        if mode == "1" {
            assert!(outs.iter().all(|o| o.lane_experts == outs[0].lane_experts), "every pick on the CPU on every path");
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
        // #203: on the global arena the guesses become pinned hits, so the split hands the CPU
        // other experts (other bits by design, #188): held to G3 there
        if what == "global" {
            assert_bit_identical(&arms[..2], &outs[..2]);
            assert_g3("lane split V3 P4 global ctl+la+prefetch+overlap", &outs[2], &outs[0]);
        } else {
            assert_bit_identical(&arms, &outs);
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
            tiers.free();
            run.free();
            outs.push(Out { gen, nvme_reads, lane_experts });
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
///   VRAM hit), at chunk 12 and row by row.
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
    assert_same(&["lane split V0 P16", "full with the lane, without the chunk V0 P16"], &outs[..2]);
    assert_same(&["lane split chunk 12 V0 P16", "full V0 P16"], &outs[2..]);
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
                outs.push(Out { gen, nvme_reads: tiers.nvme_reads, lane_experts: 0 });
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
            outs.push(Out { gen, nvme_reads: tiers.nvme_reads, lane_experts: 0 });
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
