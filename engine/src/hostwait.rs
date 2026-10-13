//! #227: which host wait a glm5_next decode row is in, for serve's decode watchdog.
//!
//! - The wait sites (router flag, device / stream / event syncs, NVMe tickets, the deferred-read
//!   channel, the controller's ring, reply and job, the CPU lane's landings) call [`open`]; the
//!   guard stamps the wait's kind, layer and start into one static slot per kind and clears it on
//!   drop. The device's decode (`Glm5Device`'s `Rows::decode` / `decode_batch`) holds a [`row`]
//!   guard.
//! - Off (the default of every process; serve arms it on glm5_next, `CROW_GLM_WATCHDOG_S`) a site
//!   costs one relaxed load; armed, a clock read and three relaxed stores. No lock, no allocation.
//! - serve's watchdog thread reads a [`Snapshot`] once a second; [`Watch::check`] says when a row
//!   passed the limit (once per row) and when that row ended. Log only: nothing here waits, aborts
//!   or changes an id.
//! - One slot per kind: two threads in the same kind at once share it; the later one shows, and
//!   the first to leave clears it only while it still holds it (a wait may be missed, never shown
//!   after it ended).
//! - Depends on `std` only (`cuda` calls it).

use std::sync::atomic::{AtomicBool, AtomicU32, AtomicU64, Ordering::Relaxed};
use std::sync::OnceLock;
use std::time::Instant;

/// `CROW_GLM_WATCHDOG_S`: seconds a decode row may run before serve logs its open host waits
pub const WATCHDOG_ENV: &str = "CROW_GLM_WATCHDOG_S";
/// the default limit: the router-flag wait's own (`glm5_flags::ROUTED_TIMEOUT`, 30 s)
pub const WATCHDOG_DEFAULT_S: u64 = 30;

/// `CROW_GLM_WATCHDOG_S`: unset or empty = [`WATCHDOG_DEFAULT_S`]; `0` = off; a whole number of
/// seconds; anything else refused by name
pub fn watchdog_secs(v: Option<&str>) -> Result<u64, String> {
    match v.map(str::trim) {
        None | Some("") => Ok(WATCHDOG_DEFAULT_S),
        Some(s) => s.parse::<u64>().map_err(|_| format!("{WATCHDOG_ENV}={s:?}: want a whole number of seconds (0 = off, unset = {WATCHDOG_DEFAULT_S})")),
    }
}

/// a host wait the watchdog names
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Wait {
    /// `CROW_GLM_FLAGS`: the spin on the router's flag (`Routed::wait_layer`)
    RouterFlag,
    /// `cuda::sync`: the active stream drained (the router hand-off without flags, a row's end)
    DeviceSync,
    /// `cuda::stream_sync`: one explicit stream (stager, write-back ring, side streams)
    StreamSync,
    /// `cuEventSynchronize`: a write-back into a pinned slot the CPU lane reads, the LA readback
    EventSync,
    /// an NVMe ticket drained (`nvme_source::Ticket`; `n` = parts still out)
    NvmeBatch,
    /// the stager's deferred NVMe reads (`n` = tickets owed)
    DeferredRead,
    /// `CROW_GLM_CONTROLLER`: the controller thread waits for the device's request `n`
    CtlRequest,
    /// `CROW_GLM_CONTROLLER`: the host's pace on the reply to request `n`
    CtlReply,
    /// `CROW_GLM_CONTROLLER`: the row's job result from the controller thread
    CtlJob,
    /// the CPU lane waits for the records of request `n` to land
    LaneLanding,
}

const KINDS: usize = 10;

impl Wait {
    pub const ALL: [Wait; KINDS] = [
        Wait::RouterFlag,
        Wait::DeviceSync,
        Wait::StreamSync,
        Wait::EventSync,
        Wait::NvmeBatch,
        Wait::DeferredRead,
        Wait::CtlRequest,
        Wait::CtlReply,
        Wait::CtlJob,
        Wait::LaneLanding,
    ];

    pub fn name(self) -> &'static str {
        match self {
            Wait::RouterFlag => "router flag",
            Wait::DeviceSync => "device sync",
            Wait::StreamSync => "stream sync",
            Wait::EventSync => "event sync",
            Wait::NvmeBatch => "NVMe batch",
            Wait::DeferredRead => "deferred NVMe read",
            Wait::CtlRequest => "controller request",
            Wait::CtlReply => "controller reply",
            Wait::CtlJob => "controller job",
            Wait::LaneLanding => "CPU lane landing",
        }
    }

    /// what `n` counts, for the line
    fn n_label(self) -> Option<&'static str> {
        match self {
            Wait::NvmeBatch => Some("parts"),
            Wait::DeferredRead => Some("tickets"),
            Wait::CtlRequest | Wait::CtlReply | Wait::LaneLanding => Some("request"),
            _ => None,
        }
    }
}

const NO_LAYER: u32 = u32::MAX;

struct Slot {
    /// [`now_ns`] at the open; 0 = closed
    since: AtomicU64,
    layer: AtomicU32,
    n: AtomicU64,
}

static ARMED: AtomicBool = AtomicBool::new(false);
static EPOCH: OnceLock<Instant> = OnceLock::new();
static SLOTS: [Slot; KINDS] = [const { Slot { since: AtomicU64::new(0), layer: AtomicU32::new(NO_LAYER), n: AtomicU64::new(0) } }; KINDS];
/// the decoder layer whose host hand-off ran last in this row ([`at_layer`])
static LAYER: AtomicU32 = AtomicU32::new(NO_LAYER);
static ROW_SEQ: AtomicU64 = AtomicU64::new(0);
static ROW_SINCE: AtomicU64 = AtomicU64::new(0);
static ROW_POS: AtomicU64 = AtomicU64::new(0);
static ROW_N: AtomicU64 = AtomicU64::new(0);

/// serve: record the waits from here on (`false`: the sites go back to one load)
pub fn arm(on: bool) {
    if on {
        EPOCH.get_or_init(Instant::now);
    }
    ARMED.store(on, Relaxed);
}

#[inline]
pub fn armed() -> bool {
    ARMED.load(Relaxed)
}

/// ns since the epoch, + 1 (0 marks a closed slot)
fn now_ns() -> u64 {
    EPOCH.get_or_init(Instant::now).elapsed().as_nanos() as u64 + 1
}

/// the decoder layer the host hands off now (router ids, a controller request)
#[inline]
pub fn at_layer(l: usize) {
    if armed() {
        LAYER.store(l as u32, Relaxed);
    }
}

/// an open wait; dropping it closes the slot (while it still holds it)
#[must_use]
pub struct Open(Option<(usize, u64)>);

/// a wait of `kind` starts; `layer` `None` = the row's current layer ([`at_layer`])
#[inline]
pub fn open(kind: Wait, layer: Option<usize>, n: u64) -> Open {
    if !armed() {
        return Open(None);
    }
    let k = kind as usize;
    let s = &SLOTS[k];
    let t = now_ns();
    s.layer.store(layer.map_or_else(|| LAYER.load(Relaxed), |l| l as u32), Relaxed);
    s.n.store(n, Relaxed);
    s.since.store(t, Relaxed);
    Open(Some((k, t)))
}

impl Drop for Open {
    #[inline]
    fn drop(&mut self) {
        if let Some((k, t)) = self.0 {
            let _ = SLOTS[k].since.compare_exchange(t, 0, Relaxed, Relaxed);
        }
    }
}

/// an open decode row; dropping it closes the row
#[must_use]
pub struct Row(bool);

/// a decode row (or batched step of `rows` rows) at `pos` starts
#[inline]
pub fn row(pos: usize, rows: usize) -> Row {
    if !armed() {
        return Row(false);
    }
    LAYER.store(NO_LAYER, Relaxed);
    ROW_POS.store(pos as u64, Relaxed);
    ROW_N.store(rows as u64, Relaxed);
    ROW_SEQ.fetch_add(1, Relaxed);
    ROW_SINCE.store(now_ns(), Relaxed);
    Row(true)
}

impl Drop for Row {
    #[inline]
    fn drop(&mut self) {
        if self.0 {
            ROW_SINCE.store(0, Relaxed);
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RowSnap {
    pub seq: u64,
    pub pos: u64,
    pub rows: u64,
    pub since: u64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct WaitSnap {
    pub kind: Wait,
    pub layer: Option<u32>,
    pub n: u64,
    pub since: u64,
}

/// what the watchdog reads at `now` (ns on the epoch of the slots)
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Snapshot {
    pub now: u64,
    pub row: Option<RowSnap>,
    pub layer: Option<u32>,
    /// the open waits, oldest first
    pub waits: Vec<WaitSnap>,
}

fn some_layer(l: u32) -> Option<u32> {
    (l != NO_LAYER).then_some(l)
}

/// the slots as they are now (a row that changed while they were read reads as none)
pub fn snapshot() -> Snapshot {
    let now = now_ns();
    let seq = ROW_SEQ.load(Relaxed);
    let since = ROW_SINCE.load(Relaxed);
    let (pos, rows) = (ROW_POS.load(Relaxed), ROW_N.load(Relaxed));
    let layer = some_layer(LAYER.load(Relaxed));
    let mut waits: Vec<WaitSnap> = Wait::ALL
        .iter()
        .filter_map(|&kind| {
            let s = &SLOTS[kind as usize];
            let since = s.since.load(Relaxed);
            (since != 0).then(|| WaitSnap { kind, layer: some_layer(s.layer.load(Relaxed)), n: s.n.load(Relaxed), since })
        })
        .collect();
    waits.sort_by_key(|w| w.since);
    let row = (since != 0 && ROW_SEQ.load(Relaxed) == seq).then_some(RowSnap { seq, pos, rows, since });
    Snapshot { now, row, layer, waits }
}

/// what [`Watch::check`] saw
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Event {
    /// the open row passed the limit (once per row)
    Stall,
    /// the row of the last [`Event::Stall`] is no longer open; it ended within `upper_ns` of its
    /// start (the tick before this one saw it open)
    Ended { seq: u64, pos: u64, upper_ns: u64 },
}

/// the watchdog's state: the limit and the row it logged
#[derive(Clone, Debug)]
pub struct Watch {
    limit_ns: u64,
    /// `(seq, pos, since)` of the row logged as stalled
    logged: Option<(u64, u64, u64)>,
}

impl Watch {
    /// `secs` 0 = off ([`Watch::check`] never fires)
    pub fn new(secs: u64) -> Watch {
        Watch { limit_ns: secs.saturating_mul(1_000_000_000), logged: None }
    }

    pub fn check(&mut self, s: &Snapshot) -> Option<Event> {
        if self.limit_ns == 0 {
            return None;
        }
        if let Some((seq, pos, since)) = self.logged {
            if s.row.is_none_or(|r| r.seq != seq) {
                self.logged = None;
                return Some(Event::Ended { seq, pos, upper_ns: s.now.saturating_sub(since) });
            }
            return None;
        }
        let r = s.row?;
        (s.now.saturating_sub(r.since) >= self.limit_ns).then(|| {
            self.logged = Some((r.seq, r.pos, r.since));
            Event::Stall
        })
    }
}

fn secs(ns: u64) -> f64 {
    ns as f64 / 1e9
}

/// the stall line of `s` (`utc_start`, `utc_now`: the row's start and now, ISO-8601 UTC)
pub fn stall_line(s: &Snapshot, limit_s: u64, utc_start: &str, utc_now: &str) -> String {
    let Some(r) = s.row else { return format!("[watchdog] no decode row open at {utc_now} (#227)") };
    let rows = if r.rows > 1 { format!(", a batched step of {} rows", r.rows) } else { String::new() };
    let layer = s.layer.map_or("none yet".to_string(), |l| l.to_string());
    let waits = if s.waits.is_empty() {
        "none (the host is in no recorded wait: compute, a lock, or a wait not instrumented)".to_string()
    } else {
        s.waits
            .iter()
            .map(|w| {
                let l = w.layer.map_or(String::new(), |l| format!(" layer {l}"));
                let n = w.kind.n_label().map_or(String::new(), |lbl| format!(" ({lbl} {})", w.n));
                format!("{}{l}{n} for {:.1} s", w.kind.name(), secs(s.now.saturating_sub(w.since)))
            })
            .collect::<Vec<_>>()
            .join("; ")
    };
    format!(
        "[watchdog] decode row at pos {}{rows} (row {}) still running after {:.1} s (limit {limit_s} s, {WATCHDOG_ENV}): started {utc_start}, now {utc_now}; last layer handed off {layer}; open host waits, oldest first: {waits} (#227, log only)",
        r.pos,
        r.seq,
        secs(s.now.saturating_sub(r.since))
    )
}

/// the line when the stalled row is no longer open
pub fn ended_line(seq: u64, pos: u64, upper_ns: u64, utc_now: &str) -> String {
    format!("[watchdog] the stalled decode row at pos {pos} (row {seq}) ended within {:.1} s of its start, seen at {utc_now} (#227)", secs(upper_ns))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn snap(now_s: u64, row: Option<(u64, u64)>, waits: Vec<WaitSnap>) -> Snapshot {
        Snapshot { now: now_s * 1_000_000_000, row: row.map(|(seq, since_s)| RowSnap { seq, pos: 75, rows: 1, since: since_s * 1_000_000_000 + 1 }), layer: Some(12), waits }
    }

    #[test]
    fn watchdog_secs_parses_default_off_and_refuses_by_name() {
        assert_eq!(watchdog_secs(None), Ok(30));
        assert_eq!(watchdog_secs(Some("")), Ok(30));
        assert_eq!(watchdog_secs(Some("0")), Ok(0));
        assert_eq!(watchdog_secs(Some(" 45 ")), Ok(45));
        for bad in ["-1", "1.5", "on", "30s"] {
            let e = watchdog_secs(Some(bad)).unwrap_err();
            assert!(e.starts_with("CROW_GLM_WATCHDOG_S="), "{e}");
        }
    }

    /// #227: below the limit nothing, at the limit once per row, the end once, the next row again
    #[test]
    fn the_watch_fires_once_per_stalled_row_and_once_at_its_end() {
        let mut w = Watch::new(30);
        assert_eq!(w.check(&snap(5, None, vec![])), None);
        assert_eq!(w.check(&snap(29, Some((1, 0)), vec![])), None);
        assert_eq!(w.check(&snap(31, Some((1, 0)), vec![])), Some(Event::Stall));
        assert_eq!(w.check(&snap(32, Some((1, 0)), vec![])), None);
        assert_eq!(w.check(&snap(90, Some((1, 0)), vec![])), None);
        // the row ended (a new row is open, or none)
        assert_eq!(w.check(&snap(91, Some((2, 91)), vec![])), Some(Event::Ended { seq: 1, pos: 75, upper_ns: 91_000_000_000 - 1 }));
        assert_eq!(w.check(&snap(100, Some((2, 91)), vec![])), None);
        // the row's start is 1 ns past 91 s: 30 s later minus that ns is not yet the limit
        assert_eq!(w.check(&snap(121, Some((2, 91)), vec![])), None);
        assert_eq!(w.check(&snap(122, Some((2, 91)), vec![])), Some(Event::Stall));
        assert!(matches!(w.check(&snap(123, None, vec![])), Some(Event::Ended { seq: 2, .. })));
        assert_eq!(w.check(&snap(124, None, vec![])), None);
        // 0 = off
        let mut off = Watch::new(0);
        assert_eq!(off.check(&snap(10_000, Some((3, 0)), vec![])), None);
    }

    #[test]
    fn the_stall_line_names_the_row_and_every_open_wait_oldest_first() {
        let w = |kind, layer, n, since_s: u64| WaitSnap { kind, layer, n, since: since_s * 1_000_000_000 };
        let s = snap(31, Some((7, 0)), vec![w(Wait::DeviceSync, Some(12), 0, 1), w(Wait::NvmeBatch, Some(12), 3, 20)]);
        let l = stall_line(&s, 30, "2026-10-13T10:00:00.000Z", "2026-10-13T10:00:31.000Z");
        assert_eq!(
            l,
            "[watchdog] decode row at pos 75 (row 7) still running after 31.0 s (limit 30 s, CROW_GLM_WATCHDOG_S): started 2026-10-13T10:00:00.000Z, now 2026-10-13T10:00:31.000Z; last layer handed off 12; open host waits, oldest first: device sync layer 12 for 30.0 s; NVMe batch layer 12 (parts 3) for 11.0 s (#227, log only)"
        );
        let none = stall_line(&snap(31, Some((7, 0)), vec![]), 30, "a", "b");
        assert!(none.contains("open host waits, oldest first: none (the host is in no recorded wait"), "{none}");
        assert_eq!(ended_line(7, 75, 42_500_000_000, "c"), "[watchdog] the stalled decode row at pos 75 (row 7) ended within 42.5 s of its start, seen at c (#227)");
    }

    /// the slots through the guards (the one test that arms the statics; it uses two kinds no
    /// host test reaches, the controller's job and the CPU lane, so tests in parallel do not mix)
    #[test]
    fn the_guards_stamp_and_clear_the_slots() {
        let mine = |s: &Snapshot| -> Vec<(Wait, Option<u32>, u64)> {
            s.waits.iter().filter(|w| matches!(w.kind, Wait::CtlJob | Wait::LaneLanding)).map(|w| (w.kind, w.layer, w.n)).collect()
        };
        arm(true);
        {
            let _r = row(75, 1);
            at_layer(12);
            let a = open(Wait::CtlJob, None, 0);
            let b = open(Wait::LaneLanding, Some(13), 4);
            let s = snapshot();
            assert_eq!(s.row.map(|r| (r.pos, r.rows)), Some((75, 1)));
            assert_eq!(s.layer, Some(12));
            assert_eq!(mine(&s), vec![(Wait::CtlJob, Some(12), 0), (Wait::LaneLanding, Some(13), 4)]);
            drop(a);
            assert_eq!(mine(&snapshot()), vec![(Wait::LaneLanding, Some(13), 4)]);
            // a later open of the same kind holds the slot; the earlier guard leaves it alone
            let c = open(Wait::LaneLanding, Some(14), 1);
            drop(b);
            assert_eq!(mine(&snapshot()), vec![(Wait::LaneLanding, Some(14), 1)]);
            drop(c);
            assert_eq!(mine(&snapshot()), vec![]);
        }
        assert_eq!(snapshot().row, None);
        arm(false);
        let _r = row(1, 1);
        let _o = open(Wait::CtlJob, None, 0);
        assert_eq!(snapshot().row, None);
        assert_eq!(mine(&snapshot()), vec![]);
    }
}
