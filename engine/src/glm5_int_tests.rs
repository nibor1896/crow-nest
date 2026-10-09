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

/// what one arm produced: the run and the tier store's NVMe reads (the cache's own count)
pub(crate) struct Out {
    pub gen: Generated,
    pub nvme_reads: u64,
}

/// `generate` of the 5-id prompt and `n` greedy ids under every arm, a fresh tier store per arm
/// (VRAM 3 + pinned 4 slots per layer)
pub(crate) fn run_arms(arms: &[Arm], n: usize) -> Vec<Out> {
    let g = geo8();
    let s = synth_model(&g, REC);
    let (spec, _) = crate::nvme_source::glm5_record_of_container(&s.path).unwrap();
    let moe = MoeGeo::new(&g, spec).unwrap();
    let mut cnq = Cnq::open_checked(&s.path).unwrap();
    let prompt = [3i64, 17, 101, 999, 5];
    let sizes = TierSizes { vram: 3, pinned: 4 };
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
            tiers.free();
            outs.push(Out { gen, nvme_reads });
        }
        run.free();
    }
    let finite = outs[0].gen.logits.iter().flatten().filter(|v| v.is_finite()).count();
    assert_eq!(finite, n * g.vocab, "the synthetic model must stay finite for the comparison to mean something");
    outs
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
/// gives the switch-off ids and logits bit for bit, and every global arm the NVMe reads of the
/// global arm without switches (the same arena moves; the per-layer cache reads 236, the arena
/// 376 on this model, so an arm that bypassed the arena shows here).
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
        assert_eq!(o.nvme_reads, outs[1].nvme_reads, "{}: NVMe reads of the global arena", a.name);
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
