//! #190 `CROW_GLM_GRAPH=1`: the glm5_next decode row (`glm5_tiers::Glm5Run`) as piecewise CUDA
//! graphs. glm5_next only; nothing on the Flash-Next / 27B path constructs or calls anything here
//! (gate R). Default off; off, every function here is a no-op and the row is the one before it,
//! call for call.
//!
//! **Segments.** The only host hand-off inside a row is a MoE layer's router: its ids go to the
//! host (`glm5_model::router_ids`: a stream sync + copy, or the `CROW_GLM_FLAGS` flag), the host
//! puts the selected records in place (`ExpertTiers::table_for`) and writes the layer's record
//! table, then the experts run through that table. A row is cut there:
//!
//! - segment 0: every layer before the first MoE layer, and that layer up to its `route`;
//! - segment m: MoE layer m-1's `experts` + expand, then every layer up to MoE layer m's `route`;
//! - the last segment: the last MoE layer's `experts` + expand (+ any later dense layer);
//! - the head (`glm5_model::run_head`) is a graph of its own.
//!
//! **Capture.** `Glm5Run` runs a row's eager body with a capture open ([`begin_row`]): every
//! launch of the segment goes to a non-blocking capture stream (`cuda::set_stream`, thread-local
//! capture mode). At a seam ([`seam_end`], called by `Glm5Pass::call_inner` right after `route`)
//! the segment is closed, checked (kernel nodes only, see below), instantiated and launched on the
//! legacy stream; the hand-off runs eagerly on the legacy stream as without the switch; then
//! [`seam_begin`] opens the next segment. [`end_row`] closes the last one. The row's GPU work runs
//! once, from the graphs, in the same order as the eager launches.
//!
//! **Replay.** Later rows launch segment 0, then per seam: the router ids, `table_for`, the next
//! segment. The graphs run on the legacy stream, so every eager copy and sync of `table_for`, the
//! embedding upload and the readbacks stay ordered with them exactly as with eager launches.
//!
//! **What changes per row, and how it reaches the graphs:**
//! - the position: every kernel reads its scalars from device buffers (the p5 rule); the only
//!   position scalars of a decode row are the MLA pass's `[pos0, t]` (`MlaScratch::st`, shared by
//!   every DSA layer). With the switch on, `Glm5Run` writes them once per row from pinned memory
//!   ([`RowGraphs::stage`]) and `MlaScratch::begin` skips its own upload while a capture is open
//!   ([`capturing`]). Chosen over `cuGraphExecKernelNodeSetParams`: no node needs a new parameter,
//!   and nothing here depends on which kernels the segments hold.
//! - the one launch shape that depends on the position: `MlaScratch::select`'s `idx_scores` grid
//!   (`glm5_mla::score_grid`, 0 = not launched). The graphs are keyed on it and recaptured when it
//!   changes (pos 3, then every 128 positions).
//! - the record tables: one device buffer per MoE layer (`ExpertTiers::tables`), rewritten by
//!   `table_for` before the segment that reads it runs; the key holds their addresses (another
//!   store recaptures). A replayed seam's table must be the captured one, or the row is refused.
//!
//! **Refused by name:** a segment that captured anything but kernel launches (a host upload
//! inside a segment would be replayed from a stale host address); `CROW_GLM_CPU_LANE=1` (its host
//! work sits inside the post-router segment). Allocations: the t = 1 FFN plans are made before
//! the first capture (`Glm5Pass::ensure_plans`).

use crate::cuda;
use cudarc::driver::sys::{self, CUresult, CUstream};
use std::cell::RefCell;

pub type Dev = sys::CUdeviceptr;

pub const ENV: &str = "CROW_GLM_GRAPH";

/// captured rows kept per run (LRU by key): one per `score_grid` value seen (a new value every
/// 128 positions) and table set. A miss recaptures, past `KEYS` the least recently used goes.
/// Cost (2026-10-09, RTX 5090, `glm5_graph_gpu_seen_keys_replay_without_recapture`): 4 kept rows
/// of 266 kernel nodes dropped free VRAM by 4,194,304 B against 0 B with one kept row (the
/// driver's 2 MiB granularity); a GLM-5.3-Flash row has about 6x the nodes. Host memory: not measured.
pub const KEYS: usize = 8;

/// `1` turns the switch on; unset or any other value leaves it off (the repo's `CROW_*` rule)
pub fn parse(v: Option<&str>) -> bool {
    v == Some("1")
}

pub fn on_from_env() -> bool {
    parse(std::env::var(ENV).ok().as_deref())
}

/// `CUgraphNodeType` (cudarc 0.19.9 `driver/sys`): KERNEL 0, MEMCPY 1, MEMSET 2, HOST 3, ...
const NODE_KERNEL: u32 = 0;
/// `CU_STREAM_CAPTURE_MODE_THREAD_LOCAL`: unsafe calls are refused on this thread only
const CAPTURE_THREAD_LOCAL: u32 = 1;

type FnBegin = unsafe extern "system" fn(CUstream, u32) -> CUresult;
type FnEnd = unsafe extern "system" fn(CUstream, *mut sys::CUgraph) -> CUresult;
type FnNodes = unsafe extern "system" fn(sys::CUgraph, *mut sys::CUgraphNode, *mut usize) -> CUresult;
type FnNodeType = unsafe extern "system" fn(sys::CUgraphNode, *mut u32) -> CUresult;
type FnInstantiate = unsafe extern "system" fn(*mut sys::CUgraphExec, sys::CUgraph, u64) -> CUresult;
type FnDestroy = unsafe extern "system" fn(sys::CUgraph) -> CUresult;
type FnProfiler = unsafe extern "system" fn() -> CUresult;

/// one captured, instantiated segment
#[derive(Clone, Copy, Debug)]
pub struct Seg {
    exec: u64,
    /// its kernel nodes (= the eager launches it replaces)
    pub kernels: usize,
}

/// one host hand-off: MoE layer `layer`'s router ids (`n` i32 at `ids`) and the record table the
/// segment after it reads
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Seam {
    pub layer: usize,
    pub ids: Dev,
    pub n: usize,
    pub table: Dev,
}

/// what the graphs of a row were captured for
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Key {
    /// `glm5_mla::score_grid(pos, 1)`
    pub score_grid: usize,
    /// `ExpertTiers::tables()`
    pub tables: Vec<Dev>,
}

/// a captured row (or the head): `segs.len() == seams.len() + 1`
#[derive(Debug)]
pub struct Captured {
    pub segs: Vec<Seg>,
    pub seams: Vec<Seam>,
}

impl Captured {
    /// kernel nodes over all segments (= eager launches per replay)
    pub fn kernels(&self) -> usize {
        self.segs.iter().map(|s| s.kernels).sum()
    }

    /// # Safety
    /// A CUDA context is current.
    unsafe fn destroy(&mut self) {
        for s in self.segs.drain(..) {
            cuda::graph_exec_destroy(s.exec as sys::CUgraphExec);
        }
        self.seams.clear();
    }
}

struct Capture {
    stream: CUstream,
    /// a segment capture is open on `stream`
    open: bool,
    done: Captured,
}

thread_local! {
    static CAP: RefCell<Option<Capture>> = const { RefCell::new(None) };
}

/// a segment capture is open on this thread (`MlaScratch::begin` skips its scalar upload then)
pub fn capturing() -> bool {
    CAP.with(|c| c.borrow().as_ref().is_some_and(|c| c.open))
}

unsafe fn open(stream: CUstream) {
    let f: FnBegin = cuda::graph_sym(b"cuStreamBeginCapture_v2\0");
    cuda::set_stream(stream as u64);
    cuda::ck(f(stream, CAPTURE_THREAD_LOCAL));
}

/// end the open capture on `stream` and go back to the legacy stream; the template graph
unsafe fn end(stream: CUstream) -> Result<sys::CUgraph, String> {
    let f: FnEnd = cuda::graph_sym(b"cuStreamEndCapture\0");
    let mut g: sys::CUgraph = std::ptr::null_mut();
    let r = f(stream, &mut g);
    cuda::set_stream(0);
    if r != CUresult::CUDA_SUCCESS {
        return Err(format!("{ENV}=1: cuStreamEndCapture -> {r:?} (a launch inside the segment broke the capture)"));
    }
    Ok(g)
}

/// the kernel-node count of `g`; `Err` by name if it holds any other node
unsafe fn kernel_nodes(g: sys::CUgraph, seg: usize) -> Result<usize, String> {
    let f_nodes: FnNodes = cuda::graph_sym(b"cuGraphGetNodes\0");
    let f_type: FnNodeType = cuda::graph_sym(b"cuGraphNodeGetType\0");
    let mut n = 0usize;
    cuda::ck(f_nodes(g, std::ptr::null_mut(), &mut n));
    let mut nodes: Vec<sys::CUgraphNode> = vec![std::ptr::null_mut(); n];
    cuda::ck(f_nodes(g, nodes.as_mut_ptr(), &mut n));
    for &nd in &nodes[..n] {
        let mut t = u32::MAX;
        cuda::ck(f_type(nd, &mut t));
        if t != NODE_KERNEL {
            return Err(format!(
                "{ENV}=1: segment {seg} captured a graph node of type {t} (1 memcpy, 2 memset, 3 host); only kernel launches may be captured, a host upload inside a segment would replay stale bytes"
            ));
        }
    }
    Ok(n)
}

/// close the open segment: check, instantiate, launch it on the legacy stream
unsafe fn close(c: &mut Capture) -> Result<(), String> {
    c.open = false;
    let g = end(c.stream)?;
    let f_destroy: FnDestroy = cuda::graph_sym(b"cuGraphDestroy\0");
    let made = kernel_nodes(g, c.done.segs.len()).and_then(|kernels| {
        let f_inst: FnInstantiate = cuda::graph_sym(b"cuGraphInstantiateWithFlags\0");
        let mut e: sys::CUgraphExec = std::ptr::null_mut();
        let r = f_inst(&mut e, g, 0);
        (r == CUresult::CUDA_SUCCESS).then_some((e, kernels)).ok_or_else(|| format!("{ENV}=1: cuGraphInstantiateWithFlags -> {r:?}"))
    });
    // #18: the exec is independent of its template
    cuda::ck(f_destroy(g));
    let (exec, kernels) = made?;
    c.done.segs.push(Seg { exec: exec as u64, kernels });
    cuda::launch_graph(exec, std::ptr::null_mut());
    Ok(())
}

/// Open a row capture on `stream` (a non-blocking stream; launches go there until the matching
/// [`end_row`] / [`abort_row`]).
///
/// # Safety
/// A CUDA context is current; no capture is open on this thread.
pub unsafe fn begin_row(stream: CUstream) {
    CAP.with(|c| {
        let mut c = c.borrow_mut();
        assert!(c.is_none(), "{ENV}: a row capture is already open");
        open(stream);
        *c = Some(Capture { stream, open: true, done: Captured { segs: Vec::new(), seams: Vec::new() } });
    });
}

/// The router seam of MoE layer `layer` (`n` ids at `ids`): with a row capture open, the segment
/// up to here is closed, instantiated and launched; a no-op otherwise. Called by
/// `Glm5Pass::call_inner` right after `route`, before the ids go to the host.
///
/// # Safety
/// A CUDA context is current.
pub unsafe fn seam_end(layer: usize, ids: Dev, n: usize) -> Result<(), String> {
    CAP.with(|c| {
        let mut c = c.borrow_mut();
        let Some(cap) = c.as_mut() else { return Ok(()) };
        close(cap)?;
        cap.done.seams.push(Seam { layer, ids, n, table: 0 });
        Ok(())
    })
}

/// After the hand-off of the last [`seam_end`]: the record table its experts read, and the next
/// segment opens. A no-op without a row capture.
///
/// # Safety
/// A CUDA context is current.
pub unsafe fn seam_begin(table: Dev) {
    CAP.with(|c| {
        let mut c = c.borrow_mut();
        let Some(cap) = c.as_mut() else { return };
        cap.done.seams.last_mut().expect("glm5_graph: seam_begin without seam_end").table = table;
        open(cap.stream);
        cap.open = true;
    });
}

/// Close the last segment (launched) and take the row's graphs.
///
/// # Safety
/// A CUDA context is current; [`begin_row`] opened the capture.
pub unsafe fn end_row() -> Result<Captured, String> {
    let mut cap = CAP.with(|c| c.borrow_mut().take()).expect("glm5_graph: end_row without begin_row");
    if let Err(e) = close(&mut cap) {
        cap.done.destroy();
        return Err(e);
    }
    Ok(cap.done)
}

/// Drop an open row capture after an error (the open segment is ended and discarded, the
/// segments already launched are destroyed after they ran).
///
/// # Safety
/// A CUDA context is current.
pub unsafe fn abort_row() {
    if let Some(mut cap) = CAP.with(|c| c.borrow_mut().take()) {
        if cap.open {
            if let Ok(g) = end(cap.stream) {
                let f_destroy: FnDestroy = cuda::graph_sym(b"cuGraphDestroy\0");
                let _ = f_destroy(g);
            }
        }
        cuda::set_stream(0);
        cap.done.destroy();
    }
}

/// What `Glm5Run` holds with the switch on: the capture stream, the pinned position scalars, the
/// row's graphs and the head's.
pub struct RowGraphs {
    pub stream: CUstream,
    /// `[pos, 1]` i32 for `MlaScratch::st`
    pos: cuda::Pinned,
    /// the captured rows, most recently used first, at most [`KEYS`]: a key seen before (a new
    /// sequence back at row 0, a later 128-position step) replays instead of recapturing
    pub rows: Vec<(Key, Captured)>,
    pub head: Option<Captured>,
    /// row captures, row replays, head captures since construction
    pub captures: u64,
    pub replays: u64,
    pub head_captures: u64,
}

impl RowGraphs {
    /// # Safety
    /// A CUDA context is current.
    pub unsafe fn new() -> RowGraphs {
        let pos = cuda::Pinned::alloc(4096);
        RowGraphs { stream: cuda::stream_create_non_blocking(), pos, rows: Vec::new(), head: None, captures: 0, replays: 0, head_captures: 0 }
    }

    /// Queue `[pos, 1]` into the MLA scalars `st` (async from pinned memory on the legacy stream).
    /// The host rewrites the pinned slot only at the next row's stage, after that row's host
    /// waited at least once on the stream past this copy (a router sync or flag; `Glm5Run` syncs
    /// itself when the model has no MoE layer).
    ///
    /// # Safety
    /// A CUDA context is current; `st` holds two i32; no capture is open.
    pub unsafe fn stage(&mut self, st: Dev, pos: usize) {
        let p = self.pos.host as *mut i32;
        std::ptr::write_volatile(p, i32::try_from(pos).expect("glm5_graph: position beyond i32"));
        std::ptr::write_volatile(p.add(1), 1);
        cuda::upload_from_pinned(st, self.pos.host, 8);
    }

    /// a row captured for `key` moves to the front (the one [`RowGraphs::current`] replays); false
    /// on a miss
    pub fn promote(&mut self, key: &Key) -> bool {
        let Some(i) = self.rows.iter().position(|(k, _)| k == key) else { return false };
        let hit = self.rows.remove(i);
        self.rows.insert(0, hit);
        true
    }

    /// the most recently used captured row
    pub fn current(&self) -> Option<&Captured> {
        self.rows.first().map(|(_, c)| c)
    }

    /// keep `c` as the row's graphs for `key`, in front; past [`KEYS`] the least recently used
    /// is destroyed (freed by the driver after its last launch)
    ///
    /// # Safety
    /// A CUDA context is current.
    pub unsafe fn keep(&mut self, key: Key, c: Captured) {
        self.rows.insert(0, (key, c));
        while self.rows.len() > KEYS {
            let (_, mut old) = self.rows.pop().expect("glm5_graph: non-empty");
            old.destroy();
        }
        self.captures += 1;
    }

    /// # Safety
    /// No launch of these graphs is pending.
    pub unsafe fn free(&mut self) {
        cuda::sync();
        for (_, mut c) in self.rows.drain(..) {
            c.destroy();
        }
        if let Some(mut c) = self.head.take() {
            c.destroy();
        }
        cuda::stream_destroy(self.stream);
        self.stream = std::ptr::null_mut();
        self.pos.free();
    }
}

/// launch segment `i` of `c` on the legacy stream
///
/// # Safety
/// A CUDA context is current; the buffers the segment reads are live.
pub unsafe fn launch(c: &Captured, i: usize) {
    cuda::launch_graph(c.segs[i].exec as sys::CUgraphExec, std::ptr::null_mut());
}

/// `cuProfilerStart` / `cuProfilerStop` (the measurement test's nsys capture range; no effect
/// without a profiler)
///
/// # Safety
/// A CUDA context is current.
pub unsafe fn profiler(on: bool) {
    let f: FnProfiler = cuda::graph_sym(if on { b"cuProfilerStart\0" } else { b"cuProfilerStop\0" });
    cuda::ck(f());
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cnq::Cnq;
    use crate::geo::Glm5Geo;
    use crate::glm5_flags::tests::{synth_model, unclocked};
    use crate::glm5_flags::Switches;
    use crate::glm5_moe::MoeGeo;
    use crate::glm5_tiers::{ExpertTiers, Generated, Glm5Run, TierSizes, TokenReport};

    #[test]
    fn only_1_turns_the_graph_switch_on() {
        assert!(parse(Some("1")));
        for v in [None, Some("0"), Some(""), Some("on"), Some("true"), Some(" 1"), Some("2")] {
            assert!(!parse(v), "{v:?}");
        }
    }

    #[test]
    fn the_score_grid_is_select_s_launch_shape() {
        use crate::glm5_mla::{score_grid, KPOOL};
        assert_eq!(KPOOL, 4);
        // rows 0..=pos: no full pool of 4 before position 3, one block of 32 pools up to 130
        let want = [(0, 0), (2, 0), (3, 1), (130, 1), (131, 2), (258, 2), (259, 3)];
        for (pos, g) in want {
            assert_eq!(score_grid(pos, 1), g, "pos {pos}");
        }
        assert_eq!(score_grid(0, 4), 1);
    }

    /// the synthetic glm5_next model of this module's GPU tests: 8 layers, 0-2 KDA + dense,
    /// 3-7 MoE (16 MUL1 experts, top-8) with DSA at 3 and 7, vocab 2048
    fn geo() -> Glm5Geo {
        let mut g = Glm5Geo::GLM_5_3_FLASH;
        (g.layers, g.dense_prefix, g.experts, g.topk, g.vocab) = (8, 3, 16, 8, 2048);
        g
    }

    const REC: u64 = 9_474_048;

    /// The graphs are invisible in the output. The synthetic 8-layer model (5 MoE layers: 6 row
    /// segments + the head; 2 DSA layers sharing the MLA scalars), a 5-id prompt and 6 greedy ids
    /// (rows 0-9: the score grid changes at row 3, so the row graphs are captured twice), V 3 + P 4,
    /// a fresh store per arm: `CROW_GLM_GRAPH` alone, with `CROW_GLM_FLAGS`, with both decode
    /// switches, gives the switch-off ids, every logit's bits and every row report except its
    /// clock. Also serve's door: `row` with the graphs gives the switch-off ids and logits.
    #[test]
    #[ignore = "needs the GPU (about 2 GB VRAM, a 2.3 GB synthetic container in the temp dir): cargo test --release --lib glm5_graph_gpu -- --ignored --nocapture --test-threads 1"]
    fn glm5_graph_gpu_the_graphs_are_invisible_in_ids_logits_and_reports() {
        let g = geo();
        let t0 = std::time::Instant::now();
        let s = synth_model(&g, REC);
        eprintln!("glm5 graph: synthetic model written in {:.1} s", t0.elapsed().as_secs_f64());
        let (spec, records) = crate::nvme_source::glm5_record_of_container(&s.path).unwrap();
        assert_eq!((spec.bytes, records), (REC, 16 * 5));
        let moe = MoeGeo::new(&g, spec).unwrap();
        let mut cnq = Cnq::open_checked(&s.path).unwrap();
        let prompt = [3i64, 17, 101, 999, 5];
        let n = 6;
        let sizes = TierSizes { vram: 3, pinned: 4 };
        let arms = [
            (false, Switches::default()),
            (true, Switches::default()),
            (true, Switches { flags: true, lookahead: false, ..Switches::default() }),
            (true, Switches { flags: true, lookahead: true, ..Switches::default() }),
        ];
        let mut outs: Vec<(String, Generated, Vec<TokenReport>)> = Vec::new();
        let mut door: (Vec<i64>, Vec<Vec<f32>>) = (Vec::new(), Vec::new());
        unsafe {
            let _ctx = cuda::Ctx::init();
            let mut run = Glm5Run::load(&mut cnq, &g, &moe, prompt.len() + n, &mut |s| eprintln!("{s}"));
            for (i, (graph, sw)) in arms.into_iter().enumerate() {
                run.set_graph(graph);
                run.set_switches(&mut cnq, sw);
                let mut tiers = ExpertTiers::new(&cnq, &s.path, &g, &moe, sizes, 1, g.topk).unwrap();
                let mut reps: Vec<TokenReport> = Vec::new();
                let gen = run.generate(&mut cnq, &mut tiers, &prompt, n, true, &mut |r| reps.push(r.clone())).unwrap();
                let name = format!("{ENV} {}, {}", if graph { "on" } else { "off" }, sw.label());
                if let Some(gr) = run.graphs() {
                    let c = gr.current().unwrap();
                    eprintln!(
                        "glm5 graph {name}: ids {:?}; row captures {} replays {}; {} segments, {} seams, {} kernel nodes per row; head {} kernel nodes",
                        gen.ids, gr.captures, gr.replays, c.segs.len(), c.seams.len(), c.kernels(), gr.head.as_ref().unwrap().kernels()
                    );
                    assert_eq!((c.segs.len(), c.seams.len()), (6, 5), "{name}: one segment per MoE router + 1");
                    if i == 1 {
                        // rows 0-2 on the first capture, rows 3-9 on the second (score grid 0 -> 1)
                        assert_eq!((gr.captures, gr.replays, gr.head_captures), (2, 8, 1), "{name}: captures / replays");
                    }
                } else {
                    eprintln!("glm5 graph {name}: ids {:?}", gen.ids);
                }
                tiers.free();
                outs.push((name, gen, reps));
            }
            // serve's door with the graphs: the prompt rows, then one decode row per id
            run.set_switches(&mut cnq, Switches::default());
            run.set_graph(true);
            let mut tiers = ExpertTiers::new(&cnq, &s.path, &g, &moe, sizes, 1, g.topk).unwrap();
            run.kda_states().for_each(|k| k.reset());
            let mut tok = 0i64;
            for (pos, &p) in prompt.iter().enumerate() {
                if let Some(id) = run.row(&mut cnq, &mut tiers, p, pos, pos + 1 == prompt.len()).unwrap() {
                    tok = id;
                }
            }
            for i in 0..n {
                door.0.push(tok);
                door.1.push(cuda::dtoh(run.logits_dev(), g.vocab));
                if i + 1 < n {
                    tok = run.row(&mut cnq, &mut tiers, tok, prompt.len() + i, true).unwrap().unwrap();
                }
            }
            tiers.free();
            run.free();
        }
        drop(cnq);
        let (_, g0, r0) = &outs[0];
        assert_eq!((g0.ids.len(), g0.logits.len(), r0.len()), (n, n, prompt.len() + n - 1));
        let finite = g0.logits.iter().flatten().filter(|v| v.is_finite()).count();
        assert_eq!(finite, n * g.vocab, "the synthetic model must stay finite for the comparison to mean something");
        assert!(r0.iter().map(|r| r.moves.iter().map(|m| m.nvme_reads()).sum::<u64>()).sum::<u64>() > 0, "the run must read records from NVMe");
        let bits = |a: &[Vec<f32>], b: &[Vec<f32>]| a.iter().zip(b).map(|(x, y)| x.iter().zip(y).filter(|(p, q)| p.to_bits() != q.to_bits()).count()).collect::<Vec<_>>();
        for (name, gx, rx) in &outs[1..] {
            assert_eq!(gx.ids, g0.ids, "{name}: ids");
            assert_eq!(gx.logits.len(), g0.logits.len(), "{name}: logit rows");
            let diff = bits(&gx.logits, &g0.logits);
            assert!(diff.iter().all(|&d| d == 0), "{name}: logits differ in bits per generated position {diff:?}");
            assert_eq!(rx.iter().map(unclocked).collect::<Vec<_>>(), r0.iter().map(unclocked).collect::<Vec<_>>(), "{name}: row reports");
        }
        assert_eq!(door.0, g0.ids, "row() with the graphs: ids");
        let diff = bits(&door.1, &g0.logits);
        assert!(diff.iter().all(|&d| d == 0), "row() with the graphs: logits differ in bits {diff:?}");
    }

    /// A key seen before replays: the synthetic 8-layer model, two sequences of 5 prompt + 266
    /// generated ids on one store (rows 0-269 cross the score grid 0 -> 1 -> 2 -> 3 at rows 3,
    /// 131, 259): the first sequence captures 4 rows, the second none (8 without the per-key
    /// cache: every sequence starts again at grid 0), and both give the switch-off ids and logits
    /// bit for bit. Also prints the free-VRAM drop of the first sequence's captures (the cost of
    /// the cached rows).
    #[test]
    #[ignore = "needs the GPU (about 2 GB VRAM, a 2.3 GB synthetic container in the temp dir): cargo test --release --lib glm5_graph_gpu -- --ignored --nocapture --test-threads 1"]
    fn glm5_graph_gpu_seen_keys_replay_without_recapture() {
        let g = geo();
        let s = synth_model(&g, REC);
        let (spec, _) = crate::nvme_source::glm5_record_of_container(&s.path).unwrap();
        let moe = MoeGeo::new(&g, spec).unwrap();
        let mut cnq = Cnq::open_checked(&s.path).unwrap();
        let prompt = [3i64, 17, 101, 999, 5];
        let n = 266;
        let sizes = TierSizes { vram: 3, pinned: 4 };
        let mut outs: Vec<Vec<Generated>> = Vec::new();
        unsafe {
            let _ctx = cuda::Ctx::init();
            let mut run = Glm5Run::load(&mut cnq, &g, &moe, prompt.len() + n, &mut |s| eprintln!("{s}"));
            for graph in [false, true] {
                run.set_graph(graph);
                let mut tiers = ExpertTiers::new(&cnq, &s.path, &g, &moe, sizes, 1, g.topk).unwrap();
                let mut seqs = Vec::new();
                for seq in 0..2 {
                    cuda::sync();
                    let free0 = cuda::free_vram_bytes();
                    let gen = run.generate(&mut cnq, &mut tiers, &prompt, n, true, &mut |_| {}).unwrap();
                    cuda::sync();
                    let drop = free0 as i64 - cuda::free_vram_bytes() as i64;
                    if let Some(gr) = run.graphs() {
                        eprintln!(
                            "glm5 graph keys: sequence {}: row captures {} replays {}, {} rows kept ({} kernel nodes each), head captures {}; free VRAM dropped {drop} B",
                            seq + 1, gr.captures, gr.replays, gr.rows.len(), gr.current().unwrap().kernels(), gr.head_captures
                        );
                        let rows = (prompt.len() + n - 1) as u64;
                        assert_eq!((gr.captures, gr.replays), (4, (seq as u64 + 1) * rows - 4), "sequence {}: captures / replays", seq + 1);
                    }
                    seqs.push(gen);
                }
                tiers.free();
                outs.push(seqs);
            }
            run.free();
        }
        drop(cnq);
        let finite = outs[0][0].logits.iter().flatten().filter(|v| v.is_finite()).count();
        assert_eq!(finite, n * g.vocab, "the synthetic model must stay finite for the comparison to mean something");
        for seq in 0..2 {
            let (a, b) = (&outs[0][seq], &outs[1][seq]);
            assert_eq!(b.ids, a.ids, "sequence {}: ids", seq + 1);
            let diff: usize = a.logits.iter().zip(&b.logits).map(|(x, y)| x.iter().zip(y).filter(|(p, q)| p.to_bits() != q.to_bits()).count()).sum();
            assert_eq!(diff, 0, "sequence {}: {diff} logits differ in bits", seq + 1);
        }
    }

    /// The measurement harness of #190 (no assertion beyond the run): the synthetic 8-layer model,
    /// `GLM_GRAPH_PROFILE_ARM` = `off` | `graph` | `graph+flags`, 5 prompt rows and 3 warm decode
    /// rows through `row`, then `cuProfilerStart`, 8 decode rows, `cuProfilerStop`. Under
    /// `nsys profile --capture-range=cudaProfilerApi` the API counts divided by 8 are the
    /// per-row counts of steady decode rows (positions 8-15: one score grid, no capture inside).
    #[test]
    #[ignore = "measurement: nsys profile -t cuda --capture-range=cudaProfilerApi <test exe> glm5_graph_gpu_profile_rows --ignored --nocapture --test-threads 1"]
    fn glm5_graph_gpu_profile_rows() {
        let arm = std::env::var("GLM_GRAPH_PROFILE_ARM").unwrap_or_else(|_| "off".into());
        let (graph, flags) = match arm.as_str() {
            "off" => (false, false),
            "graph" => (true, false),
            "graph+flags" => (true, true),
            a => panic!("GLM_GRAPH_PROFILE_ARM={a:?}: off | graph | graph+flags"),
        };
        let g = geo();
        let s = synth_model(&g, REC);
        let (spec, _) = crate::nvme_source::glm5_record_of_container(&s.path).unwrap();
        let moe = MoeGeo::new(&g, spec).unwrap();
        let mut cnq = Cnq::open_checked(&s.path).unwrap();
        let prompt = [3i64, 17, 101, 999, 5];
        let (warm, rows) = (3usize, 8usize);
        unsafe {
            let _ctx = cuda::Ctx::init();
            let mut run = Glm5Run::load(&mut cnq, &g, &moe, prompt.len() + warm + rows, &mut |s| eprintln!("{s}"));
            run.set_graph(graph);
            run.set_switches(&mut cnq, Switches { flags, lookahead: false, ..Switches::default() });
            // every record in VRAM: the rows measured move nothing, only the decode path is counted
            let mut tiers = ExpertTiers::new(&cnq, &s.path, &g, &moe, TierSizes { vram: 16, pinned: 0 }, 1, g.topk).unwrap();
            run.kda_states().for_each(|k| k.reset());
            let mut tok = 0i64;
            for (pos, &p) in prompt.iter().enumerate() {
                if let Some(id) = run.row(&mut cnq, &mut tiers, p, pos, pos + 1 == prompt.len()).unwrap() {
                    tok = id;
                }
            }
            let mut pos = prompt.len();
            for _ in 0..warm {
                tok = run.row(&mut cnq, &mut tiers, tok, pos, true).unwrap().unwrap();
                pos += 1;
            }
            cuda::sync();
            profiler(true);
            let t0 = std::time::Instant::now();
            for _ in 0..rows {
                tok = run.row(&mut cnq, &mut tiers, tok, pos, true).unwrap().unwrap();
                pos += 1;
            }
            profiler(false);
            let gr = run.graphs().map(|gr| (gr.captures, gr.replays, gr.current().map(|c| (c.segs.len(), c.kernels())), gr.head.as_ref().map(|c| c.kernels())));
            eprintln!("glm5 graph profile arm {arm}: {rows} decode rows in {:.4} s (synthetic model, all records in VRAM); graphs {gr:?}", t0.elapsed().as_secs_f64());
            tiers.free();
            run.free();
        }
        drop(cnq);
    }
}
