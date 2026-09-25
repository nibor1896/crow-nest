//! #117: VRAM lending to a co-resident GPU client (Crow's `render_page`).
//!
//! While Crow renders, the engine is idle: the model waits for the tool result.
//! serve lends the PHYSICAL memory of the buffers that hold no state between
//! requests and takes it back afterwards. The buffers are allocated through CUDA
//! VMM (`cuda::alloc_lendable`), so the return maps new memory at the SAME virtual
//! address and the captured decode graph and every stored pointer stay valid (the
//! torch_memory_saver design behind SGLang's `release_memory_occupation`).
//!
//! This module is the pure half, testable without a GPU:
//! - [`LendGate`]: the state machine. Idle or lent; no double lend; the ttl; a
//!   failed return retried every second; engine requests parked while lent.
//! - [`Parked`]: the FIFO of engine requests that arrived while the memory was
//!   lent. They are served, in order, right after the return (vLLM #28714 is what
//!   happens when a request runs on released memory).
//! - [`parse_lend`]: the `POST /v1/crow/vram/lend` body.
//! - [`tier1_plan`]: which buffers are lendable, with their bytes, from the same
//!   size formulas the allocators use.
//!
//! The idle gate itself is serve's accept loop: it is single threaded and
//! blocking, so a lend request is read only after the request in flight ended.

use std::collections::VecDeque;
use std::time::{Duration, Instant};

/// `ttl_s` when the body names none
pub const TTL_DEFAULT_S: u64 = 120;
/// the longest lend serve accepts: past this every parked request would wait too long
pub const TTL_MAX_S: u64 = 600;
/// how long a failed return waits before serve tries again
pub const RETRY: Duration = Duration::from_secs(1);

/// why a lend was refused
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Refusal {
    /// 409: the memory is already lent
    AlreadyLent,
    /// 400: the body is not `{"mib": N>0, "ttl_s": 1..=600}`
    BadRequest(String),
}

/// a parsed lend request
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LendParams {
    pub bytes: u64,
    pub ttl: Duration,
}

/// `{"mib": N, "ttl_s": T}`: `mib` a whole number > 0, `ttl_s` 1..=600 (default 120)
pub fn parse_lend(body: &[u8]) -> Result<LendParams, Refusal> {
    let bad = |m: &str| Refusal::BadRequest(m.to_string());
    let v: serde_json::Value = serde_json::from_slice(body).map_err(|e| bad(&format!("body is not JSON: {e}")))?;
    let mib = v.get("mib").and_then(|m| m.as_u64()).ok_or_else(|| bad("\"mib\" must be a whole number of MiB > 0"))?;
    if mib == 0 {
        return Err(bad("\"mib\" must be a whole number of MiB > 0"));
    }
    let ttl = match v.get("ttl_s") {
        None | Some(serde_json::Value::Null) => TTL_DEFAULT_S,
        Some(t) => t.as_u64().filter(|&t| (1..=TTL_MAX_S).contains(&t))
            .ok_or_else(|| bad(&format!("\"ttl_s\" must be a whole number of seconds in 1..={TTL_MAX_S}")))?,
    };
    Ok(LendParams { bytes: mib << 20, ttl: Duration::from_secs(ttl) })
}

/// what is due while the memory is lent
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Due {
    /// the ttl expired: return automatically
    Ttl,
    /// a return failed earlier (another process held the VRAM): try again
    Retry,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Phase {
    Idle,
    Lent { bytes: u64, since: Instant, deadline: Instant, retry_at: Option<Instant> },
}

/// the lend state of one serve process
#[derive(Debug, Clone)]
pub struct LendGate {
    phase: Phase,
}

impl Default for LendGate {
    fn default() -> Self {
        Self::new()
    }
}

impl LendGate {
    pub fn new() -> LendGate {
        LendGate { phase: Phase::Idle }
    }

    pub fn is_lent(&self) -> bool {
        matches!(self.phase, Phase::Lent { .. })
    }

    /// bytes lent right now (0 when idle)
    pub fn lent_bytes(&self) -> u64 {
        match self.phase {
            Phase::Lent { bytes, .. } => bytes,
            Phase::Idle => 0,
        }
    }

    /// may a lend start now? No double lend.
    pub fn check_lend(&self) -> Result<(), Refusal> {
        if self.is_lent() { Err(Refusal::AlreadyLent) } else { Ok(()) }
    }

    /// the memory was released: lent until a return, at the latest `ttl` from now
    pub fn lent(&mut self, now: Instant, bytes: u64, ttl: Duration) {
        assert!(!self.is_lent(), "double lend: check_lend first");
        self.phase = Phase::Lent { bytes, since: now, deadline: now + ttl, retry_at: None };
    }

    /// may an ENGINE request (chat, slot save/restore) run now? Never while lent.
    pub fn admits_engine(&self) -> bool {
        !self.is_lent()
    }

    /// the automatic return that is due at `now`, if any
    pub fn due(&self, now: Instant) -> Option<Due> {
        match self.phase {
            Phase::Idle => None,
            Phase::Lent { retry_at: Some(r), .. } => (now >= r).then_some(Due::Retry),
            Phase::Lent { deadline, .. } => (now >= deadline).then_some(Due::Ttl),
        }
    }

    /// how long the accept loop may block before something is due (`None` when idle)
    pub fn wait(&self, now: Instant) -> Option<Duration> {
        match self.phase {
            Phase::Idle => None,
            Phase::Lent { retry_at: Some(r), .. } => Some(r.saturating_duration_since(now)),
            Phase::Lent { deadline, .. } => Some(deadline.saturating_duration_since(now)),
        }
    }

    /// seconds of ttl left (`None` when idle)
    pub fn ttl_remaining(&self, now: Instant) -> Option<Duration> {
        match self.phase {
            Phase::Lent { deadline, .. } => Some(deadline.saturating_duration_since(now)),
            Phase::Idle => None,
        }
    }

    /// the memory is back: idle again. Returns (bytes, how long it was lent).
    pub fn returned(&mut self, now: Instant) -> Option<(u64, Duration)> {
        match std::mem::replace(&mut self.phase, Phase::Idle) {
            Phase::Lent { bytes, since, .. } => Some((bytes, now.saturating_duration_since(since))),
            Phase::Idle => None,
        }
    }

    /// the return failed (the VRAM is still held elsewhere): still lent, retry in `RETRY`
    pub fn return_failed(&mut self, now: Instant) {
        if let Phase::Lent { retry_at, .. } = &mut self.phase {
            *retry_at = Some(now + RETRY);
        }
    }
}

/// the engine requests that arrived while the memory was lent, first come first served
#[derive(Debug)]
pub struct Parked<T> {
    q: VecDeque<T>,
}

impl<T> Default for Parked<T> {
    fn default() -> Self {
        Parked { q: VecDeque::new() }
    }
}

impl<T> Parked<T> {
    pub fn park(&mut self, t: T) {
        self.q.push_back(t);
    }
    pub fn len(&self) -> usize {
        self.q.len()
    }
    pub fn is_empty(&self) -> bool {
        self.q.is_empty()
    }
    /// the next parked request, but only once the gate admits engine requests again
    pub fn next_admitted(&mut self, gate: &LendGate) -> Option<T> {
        if gate.admits_engine() { self.q.pop_front() } else { None }
    }
    /// every parked request, dropped (shutdown)
    pub fn clear(&mut self) {
        self.q.clear();
    }
}

/// VMM granularity the plan rounds to (the RTX 5090's minimum; the allocator
/// reads the real value from the driver)
pub const PLAN_GRANULARITY: u64 = 2 << 20;

/// #117 tier 1: every buffer the engine allocates as LENDABLE, `(name, bytes)`,
/// at chunk `chunk`, vit patch cap `vit_cap` (`None` = tower not loaded) and
/// context `context` (the mrope pair, held at boot). The sizes come from the
/// allocators' own formulas. Buffers below `cuda::LEND_MIN_BYTES` are left out:
/// they stay on `cuMemAlloc`.
///
/// NOT lendable, by construction (state that the next request reads): the KV
/// cache, the QSA key ring and pooled cache, the GDN S and conv state, the rope
/// table, the hot experts and their pointer table, the PLE row cache, the dense
/// weights, `logits`/`argmax`, the device sampler, the scalar parameter slots
/// (`Params`, the vit scalar slots) and every small buffer of `Stage`.
pub fn tier1_plan(chunk: usize, vit_cap: Option<usize>, context: usize, stage_slots: usize, gu_bytes: usize, dn_bytes: usize) -> Vec<(&'static str, u64)> {
    let (persist, union) = crate::gen::Scratch::diet_region_bytes(chunk);
    let cap_blocks = 65536usize;
    let mut v: Vec<(&'static str, usize)> = vec![
        ("scratch persist region", persist),
        ("scratch union region", union),
        ("qsa pool_raw", cap_blocks * crate::gen::QSA_HID),
        ("qsa pool_nrm", cap_blocks * crate::gen::QSA_HID),
        ("qsa pool_rot", cap_blocks * crate::gen::QSA_HID),
        ("qsa scores", chunk.clamp(1, crate::gen::ATTN_SB) * cap_blocks * 4),
        ("stage gate_up", stage_slots * gu_bytes),
        ("stage down", stage_slots * dn_bytes),
    ];
    if let Some(cap) = vit_cap {
        v.extend(crate::vit::scratch_buffer_bytes(cap));
        let half = crate::vit::mrope_bytes(context) / 2;
        v.push(("mrope cos", half));
        v.push(("mrope sin", half));
    }
    v.into_iter()
        .filter(|&(_, b)| b >= crate::cuda::LEND_MIN_BYTES)
        .map(|(n, b)| (n, crate::cuda::round_up(b, PLAN_GRANULARITY as usize) as u64))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    const MIB: u64 = 1 << 20;

    #[test]
    fn a_lend_body_is_mib_and_an_optional_ttl() {
        assert_eq!(parse_lend(br#"{"mib": 1700, "ttl_s": 90}"#), Ok(LendParams { bytes: 1700 * MIB, ttl: Duration::from_secs(90) }));
        assert_eq!(parse_lend(br#"{"mib": 512}"#).unwrap().ttl, Duration::from_secs(TTL_DEFAULT_S));
        for bad in [&br#"{"mib": 0}"#[..], br#"{"mib": -5}"#, br#"{"mib": 1.5}"#, br#"{}"#, b"nope",
                    br#"{"mib": 100, "ttl_s": 0}"#, br#"{"mib": 100, "ttl_s": 601}"#, br#"{"mib": 100, "ttl_s": "x"}"#] {
            assert!(matches!(parse_lend(bad), Err(Refusal::BadRequest(_))), "{}", String::from_utf8_lossy(bad));
        }
    }

    #[test]
    fn a_second_lend_is_refused_until_the_return() {
        let t0 = Instant::now();
        let mut g = LendGate::new();
        assert_eq!(g.check_lend(), Ok(()));
        g.lent(t0, 1700 * MIB, Duration::from_secs(120));
        assert_eq!(g.check_lend(), Err(Refusal::AlreadyLent));
        assert_eq!(g.lent_bytes(), 1700 * MIB);
        let (bytes, dur) = g.returned(t0 + Duration::from_secs(7)).unwrap();
        assert_eq!((bytes, dur), (1700 * MIB, Duration::from_secs(7)));
        assert_eq!(g.check_lend(), Ok(()));
        // a return with nothing lent is a no-op, not an error
        assert_eq!(g.returned(t0), None);
    }

    #[test]
    fn engine_requests_are_gated_while_lent_and_run_in_order_after_the_return() {
        let t0 = Instant::now();
        let mut g = LendGate::new();
        let mut p: Parked<&str> = Parked::default();
        assert!(g.admits_engine());
        g.lent(t0, MIB, Duration::from_secs(60));
        assert!(!g.admits_engine());
        p.park("chat 1");
        p.park("slot save");
        p.park("chat 2");
        // nothing runs while the memory is lent
        assert_eq!(p.next_admitted(&g), None);
        assert_eq!(p.len(), 3);
        g.returned(t0 + Duration::from_secs(3));
        assert_eq!(p.next_admitted(&g), Some("chat 1"));
        assert_eq!(p.next_admitted(&g), Some("slot save"));
        assert_eq!(p.next_admitted(&g), Some("chat 2"));
        assert!(p.is_empty());
    }

    #[test]
    fn the_ttl_returns_automatically_and_a_failed_return_retries_every_second() {
        let t0 = Instant::now();
        let mut g = LendGate::new();
        assert_eq!((g.due(t0), g.wait(t0)), (None, None));
        g.lent(t0, MIB, Duration::from_secs(120));
        assert_eq!(g.due(t0 + Duration::from_secs(119)), None);
        assert_eq!(g.wait(t0 + Duration::from_secs(100)), Some(Duration::from_secs(20)));
        assert_eq!(g.ttl_remaining(t0 + Duration::from_secs(100)), Some(Duration::from_secs(20)));
        assert_eq!(g.due(t0 + Duration::from_secs(120)), Some(Due::Ttl));
        // the ttl return found the VRAM still taken: still lent, requests still wait
        let t1 = t0 + Duration::from_secs(120);
        g.return_failed(t1);
        assert!(g.is_lent() && !g.admits_engine());
        assert_eq!(g.due(t1), None);
        assert_eq!(g.wait(t1), Some(RETRY));
        assert_eq!(g.due(t1 + RETRY), Some(Due::Retry));
        assert!(g.returned(t1 + RETRY).is_some());
        assert!(g.admits_engine());
    }

    /// the serve operating point of 2026-09-25: chunk 2048, vit cap 5120 patches,
    /// n_ctx 200,000, 2 x 64 stage slots of 1,843,200 + 921,600 B
    fn serve_plan() -> Vec<(&'static str, u64)> {
        tier1_plan(2048, Some(5120), 200_000, 128, 1_843_200, 921_600)
    }

    #[test]
    fn tier_1_frees_at_least_1_6_gib_at_the_serve_operating_point() {
        let plan = serve_plan();
        let total: u64 = plan.iter().map(|&(_, b)| b).sum();
        assert!(total >= 1638 * MIB, "tier 1 lends only {:.1} MiB: {plan:?}", total as f64 / MIB as f64);
        // the two diet regions are the #10b log's 277 + 582 MiB
        let get = |n: &str| plan.iter().find(|&&(k, _)| k == n).map(|&(_, b)| b).unwrap();
        assert_eq!(get("scratch persist region") / MIB, 278, "{plan:?}");
        assert_eq!(get("scratch union region") / MIB, 582, "{plan:?}");
        assert_eq!(get("qsa scores"), 128 * MIB);
        // every lendable size is a whole number of granules
        assert!(plan.iter().all(|&(_, b)| b % PLAN_GRANULARITY == 0));
    }

    #[test]
    fn only_stateless_scratch_is_lendable() {
        let plan = serve_plan();
        let names: Vec<&str> = plan.iter().map(|&(n, _)| n).collect();
        // what the ticket classified as scratch (written before read in every request)
        let scratch = ["scratch persist region", "scratch union region", "qsa pool_raw", "qsa pool_nrm", "qsa pool_rot",
            "qsa scores", "stage gate_up", "stage down", "vit x", "vit normed", "vit qkv", "vit attn", "vit mlp",
            "vit m1", "vit out", "vit patches", "mrope cos", "mrope sin"];
        for n in &names {
            assert!(scratch.contains(n), "{n} is lendable but not classified as scratch");
        }
        // state, parameters and the small per-image tables never are
        for n in ["kv", "qsa keys", "qsa pooled", "gdn s", "gdn conv", "rope", "hot experts", "ple cache",
                  "logits", "argmax", "sampler", "vit scalars", "vit pe_idx", "vit pe_w", "vit cs", "vit sn"] {
            assert!(!names.contains(&n), "{n} must not be lendable");
        }
        // text-only boot: no vit, no mrope
        let text = tier1_plan(2048, None, 200_000, 128, 1_843_200, 921_600);
        assert!(text.iter().all(|&(n, _)| !n.starts_with("vit") && !n.starts_with("mrope")));
    }
}
