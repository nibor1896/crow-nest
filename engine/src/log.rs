//! The engine's logging: `tracing` as the single facade, one rotating file plus
//! the stderr mirror the tools parse, levels routed by `CROW_LOG` without a
//! rebuild (#13, 2026-09-18).
//!
//! # What this module is, and what it deliberately is not
//!
//! Before this module every line the engine said was an `eprintln!`: always on,
//! never levelled, never in a file, and synchronous on the calling thread. The
//! 153 sites in `engine/src` are `tracing` events now — 152 of them, with the
//! SAME message text byte for byte and a per-component **target** (119 `info`,
//! 18 `warn`, 14 `error`, 1 `debug` — the `[chat] ids` line); the one left alone
//! is a test helper inside `#[cfg(test)]` in `toolcall.rs`, where no subscriber
//! is installed and an event would swallow a failing test's diagnosis. The text
//! had to survive byte for byte because `tools/` and
//! `docs/` grep those lines (`[chat] prompt `, `[chat] the client is gone at
//! step`, the `[serve]`/`[load]`/`[budget]` boot lines). The stderr mirror is
//! therefore formatted MESSAGE ONLY — no timestamp, no level, no target — and a
//! default-level run of `serve`, `decode` or `parity` writes the same stderr it
//! wrote at `5a58e0b`. The file gets the machine form: ISO-8601 UTC, level,
//! target, message.
//!
//! # The two sinks, and why neither is in the hot path
//!
//! Both sinks sit behind `tracing_appender::non_blocking`: one bounded channel
//! and one worker thread each, `lossy(true)`, so a call site never waits on a
//! write and never waits on a full queue — it drops instead (the drop count is
//! reported by the worker). The rotation, the gzip and the pruning all run ON
//! that worker thread, never on the thread that emitted the event.
//!
//! # Crates, and the one thing written here by hand
//!
//! - `tracing` + `tracing-subscriber` (`std`, `fmt`, `env-filter`, `registry`,
//!   no `ansi`, no `time`): the facade, the `RUST_LOG`-style filter and the two
//!   formatting layers.
//! - `tracing-appender`: `non_blocking` only. Its `rolling` module rolls by TIME
//!   alone, never by size and never with compression, so it is not used.
//! - `flate2`: the gzip of a rotated file. It costs **zero** new crates in this
//!   tree — `image`'s `png` feature already pulls `flate2`, `miniz_oxide`,
//!   `crc32fast` and `adler2` (pure Rust, no C toolchain, the same code path on
//!   Windows and Linux).
//! - [`RotatingFile`] is ~150 lines here instead of a crate, because the
//!   maintained alternatives each pay more than they give: `rolling-file` and
//!   `tracing-rolling-file` roll by size but never compress and never prune with
//!   a retention count, and `file-rotate` does all three but brings `chrono` for
//!   its timestamp suffixes. The suffix this file needs is nine lines of
//!   Hinnant's civil-from-days ([`civil_from_days`]), which the daily rotation
//!   boundary needs anyway, so the hand-written writer adds no dependency at all
//!   while giving all four properties the ticket asks for: size limit, day
//!   boundary, retention N, gzip.
//!
//! # The targets
//!
//! One target per component, taken from the bracket prefix the message already
//! carried, so `CROW_LOG=info,chat=debug` is a per-component switch:
//!
//! | target | what it covers |
//! |---|---|
//! | `boot` | the one structured JSON line of the operating point ([`boot`]) |
//! | `routing` | the one structured JSON line per request ([`routing`]) |
//! | `log` | this module's own line: where the file is, its limit, its retention |
//! | `serve` | the HTTP front door: boot lines, per-request status, bind and I/O errors |
//! | `chat` | one request's summary, the sampling provenance, the normaliser, `ids` (DEBUG) |
//! | `tokenize` | the `serve tokenize` subcommand |
//! | `slot` | `POST /slots/0` |
//! | `prefill` | `gen::prefill`: the START line and the per-chunk tok/s |
//! | `decode` | `gen::decode_step`: the per-step forensics, TRACE only |
//! | `load` | the loader's progress callback |
//! | `budget` | the host-pinned and VRAM budget lines of the planner |
//! | `policy` | `geo::apply_adapt_policy` |
//! | `adapt` | the hot-set re-cut and the stream trickle |
//! | `residency` | the hot set, the cold tier, the sidecar adapt lines |
//! | `manager` | the two-sided planner clamp |
//! | `ple` | the PLE layer (all of it behind the existing debug flags) |
//! | `vit` | the visual tower and the image request path |
//! | `cnq` | the container reader |
//! | `cuda` | streams, graphs, allocation failures, the drop-time diagnostics |
//! | `kernels` | launches and the `[kprof]` table |
//! | `toolcall` | the streaming `<tool_call>` parser |
//! | `nanwatch` | the NaN watch of `ENGINE_DEBUG_NAN` |
//! | `attn` | the attention debug dumps |
//!
//! The spec's component names (`scheduler`, `ring`, `stager`, `kv`, `gdn`,
//! `qsa`, `converter`, `loader`) are covered by the targets that actually own
//! code in this tree: the scheduler and the stager live in `residency` and
//! `adapt`, the ring and the KV in `gen` (target `prefill` / `decode`), the
//! loader is `load` + `budget`, and the converter is a separate crate with no
//! engine log site.
//!
//! # Environment
//!
//! | variable | default | effect |
//! |---|---|---|
//! | `CROW_LOG` | `info` | `RUST_LOG`-style filter, e.g. `info,routing=debug,decode=trace` |
//! | `CROW_LOG_DIR` | per-OS (see [`default_log_dir`]) | the directory of `engine.log` |
//! | `CROW_LOG_ROTATE_MB` | `64` | size limit in MiB, decimals accepted (`0.001` = 1 KiB) |
//! | `CROW_LOG_KEEP` | `8` | how many `engine-*.log.gz` are kept |

use std::fs;
use std::io::{self, Write};
use std::path::{Path, PathBuf};
use std::sync::{Mutex, OnceLock};
use std::time::{SystemTime, UNIX_EPOCH};

use tracing_appender::non_blocking::{NonBlocking, NonBlockingBuilder, WorkerGuard};
use tracing_subscriber::fmt::format::Writer;
use tracing_subscriber::fmt::time::FormatTime;
use tracing_subscriber::fmt::{FmtContext, FormatEvent, FormatFields};
use tracing_subscriber::layer::SubscriberExt;
use tracing_subscriber::registry::LookupSpan;
use tracing_subscriber::util::SubscriberInitExt;
use tracing_subscriber::EnvFilter;

/// the filter in force when `CROW_LOG` is unset: INFO for operators
pub const DEFAULT_FILTER: &str = "info";
/// the size limit in force when `CROW_LOG_ROTATE_MB` is unset
pub const DEFAULT_ROTATE_MIB: f64 = 64.0;
/// how many gzipped files are kept when `CROW_LOG_KEEP` is unset
pub const DEFAULT_KEEP: usize = 8;
/// the smallest limit a caller can ask for: below this a single long line would
/// rotate on every write
pub const MIN_ROTATE_BYTES: u64 = 1024;
/// the file name stem: `engine.log` live, `engine-YYYYMMDD-HHMMSS.log.gz` rotated
pub const STEM: &str = "engine";
/// how many lines each sink may hold before it starts dropping instead of
/// blocking the call site. 131,072 lines is four orders of magnitude above the
/// ~10 lines a request writes at INFO.
pub const BUFFERED_LINES: usize = 131_072;

// ---------------------------------------------------------------------------
// configuration (pure halves first, so every rule is unit tested without env)
// ---------------------------------------------------------------------------

/// The resolved logging configuration of this process.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Cfg {
    /// the `RUST_LOG`-style filter string actually installed
    pub filter: String,
    /// the directory `engine.log` lives in
    pub dir: PathBuf,
    /// rotate when the live file would pass this many bytes
    pub rotate_bytes: u64,
    /// how many `engine-*.log.gz` survive a rotation
    pub keep: usize,
    /// set when `CROW_LOG` held something this build could not parse; the
    /// default filter is installed instead and this is said out loud
    pub filter_note: Option<String>,
}

/// The filter to install, and the complaint if the caller's string was not one.
///
/// Pure: the value comes in as an argument, never from the environment, so the
/// whole rule is unit tested. An unset, empty or all-whitespace value is the
/// operator default (`info`). Anything else is handed to `EnvFilter`, and a
/// string `EnvFilter` refuses falls back to the default WITH a note, because a
/// typo in an operating switch may never silence a running server.
pub fn filter_spec(raw: Option<&str>) -> (String, Option<String>) {
    let want = match raw {
        Some(s) if !s.trim().is_empty() => s.trim(),
        _ => return (DEFAULT_FILTER.to_string(), None),
    };
    match EnvFilter::builder().parse(want) {
        Ok(_) => (want.to_string(), None),
        Err(e) => (
            DEFAULT_FILTER.to_string(),
            Some(format!(
                "CROW_LOG={want:?} is not a filter ({e}) - falling back to {DEFAULT_FILTER}"
            )),
        ),
    }
}

/// MiB (decimals accepted) to a byte limit, clamped to something a writer can
/// honour. Only a positive finite number is a limit; anything else takes the
/// default. `0.001` is 1,048 B and is how the forced-rotation proof was produced.
pub fn rotate_bytes(raw: Option<&str>) -> u64 {
    let mib = raw
        .and_then(|s| s.trim().parse::<f64>().ok())
        .filter(|v| v.is_finite() && *v > 0.0)
        .unwrap_or(DEFAULT_ROTATE_MIB);
    let bytes = (mib * 1024.0 * 1024.0) as u64;
    bytes.clamp(MIN_ROTATE_BYTES, 1 << 32)
}

/// How many rotated files survive. 0 is not a retention policy, it is "delete
/// the evidence", so the floor is 1.
pub fn keep_count(raw: Option<&str>) -> usize {
    raw.and_then(|s| s.trim().parse::<usize>().ok())
        .unwrap_or(DEFAULT_KEEP)
        .clamp(1, 10_000)
}

/// The per-OS default log directory, as a pure function of the four variables
/// that decide it — so the Windows rule is tested on Linux and the reverse.
///
/// Windows: `%LOCALAPPDATA%\crow\logs`, the convention for per-user state.
/// Unix: `$XDG_STATE_HOME/crow/logs`, else `$HOME/.local/state/crow/logs` (the
/// XDG default), else the system temp directory, which always exists.
pub fn log_dir_from(
    windows: bool,
    localappdata: Option<&str>,
    userprofile: Option<&str>,
    xdg_state: Option<&str>,
    home: Option<&str>,
    temp: &Path,
) -> PathBuf {
    let ok = |v: Option<&str>| v.filter(|s| !s.trim().is_empty()).map(PathBuf::from);
    let base = if windows {
        ok(localappdata)
            .or_else(|| ok(userprofile).map(|p| p.join("AppData").join("Local")))
            .unwrap_or_else(|| temp.to_path_buf())
    } else {
        ok(xdg_state)
            .or_else(|| ok(home).map(|p| p.join(".local").join("state")))
            .unwrap_or_else(|| temp.to_path_buf())
    };
    base.join("crow").join("logs")
}

/// [`log_dir_from`] on this process's environment and this OS.
pub fn default_log_dir() -> PathBuf {
    let g = |k: &str| std::env::var(k).ok();
    let (la, up, xs, ho) = (
        g("LOCALAPPDATA"),
        g("USERPROFILE"),
        g("XDG_STATE_HOME"),
        g("HOME"),
    );
    log_dir_from(
        cfg!(windows),
        la.as_deref(),
        up.as_deref(),
        xs.as_deref(),
        ho.as_deref(),
        &std::env::temp_dir(),
    )
}

/// The whole configuration off the environment: `CROW_LOG`, `CROW_LOG_DIR`,
/// `CROW_LOG_ROTATE_MB`, `CROW_LOG_KEEP`.
pub fn cfg_from_env() -> Cfg {
    let g = |k: &str| std::env::var(k).ok();
    let (filter, filter_note) = filter_spec(g("CROW_LOG").as_deref());
    let dir = match g("CROW_LOG_DIR") {
        Some(d) if !d.trim().is_empty() => PathBuf::from(d.trim()),
        _ => default_log_dir(),
    };
    Cfg {
        filter,
        dir,
        rotate_bytes: rotate_bytes(g("CROW_LOG_ROTATE_MB").as_deref()),
        keep: keep_count(g("CROW_LOG_KEEP").as_deref()),
        filter_note,
    }
}

// ---------------------------------------------------------------------------
// the calendar, nine lines instead of a dependency
// ---------------------------------------------------------------------------

/// Hinnant's `civil_from_days`: days since 1970-01-01 to (year, month, day),
/// proleptic Gregorian, exact for every day this process can see.
pub fn civil_from_days(z: i64) -> (i64, u32, u32) {
    let z = z + 719_468;
    let era = if z >= 0 { z } else { z - 146_096 } / 146_097;
    let doe = (z - era * 146_097) as u64; // [0, 146096]
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365; // [0, 399]
    let y = yoe as i64 + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100); // [0, 365]
    let mp = (5 * doy + 2) / 153; // [0, 11]
    let d = (doy - (153 * mp + 2) / 5 + 1) as u32; // [1, 31]
    let m = if mp < 10 { mp + 3 } else { mp - 9 } as u32; // [1, 12]
    (if m <= 2 { y + 1 } else { y }, m, d)
}

/// The UTC day a unix second belongs to (floor division, negative safe).
pub fn day_of(secs: i64) -> i64 {
    secs.div_euclid(86_400)
}

/// `YYYYMMDD-HHMMSS` in UTC: the suffix of a rotated file. Sorts
/// chronologically as a plain string, which is what the retention prune uses
/// instead of asking the filesystem for timestamps.
pub fn stamp(secs: i64) -> String {
    let (y, m, d) = civil_from_days(day_of(secs));
    let sod = secs.rem_euclid(86_400);
    format!(
        "{y:04}{m:02}{d:02}-{:02}{:02}{:02}",
        sod / 3600,
        (sod % 3600) / 60,
        sod % 60
    )
}

/// `YYYY-MM-DDTHH:MM:SS.mmmZ`: the timestamp of a line in the FILE. The stderr
/// mirror has no timestamp, because the lines the tools parse never had one.
pub fn iso8601(secs: i64, millis: u32) -> String {
    let (y, m, d) = civil_from_days(day_of(secs));
    let sod = secs.rem_euclid(86_400);
    format!(
        "{y:04}-{m:02}-{d:02}T{:02}:{:02}:{:02}.{millis:03}Z",
        sod / 3600,
        (sod % 3600) / 60,
        sod % 60
    )
}

fn now_parts() -> (i64, u32) {
    match SystemTime::now().duration_since(UNIX_EPOCH) {
        Ok(d) => (d.as_secs() as i64, d.subsec_millis()),
        Err(e) => (-(e.duration().as_secs() as i64), 0),
    }
}

// ---------------------------------------------------------------------------
// the rotating writer
// ---------------------------------------------------------------------------

/// `<dir>/engine.log`, rotated when it would pass `limit` bytes or when the UTC
/// day changes, gzipped on rotation, `keep` of the gzipped files retained.
///
/// Every one of those four steps runs on the `non_blocking` worker thread, never
/// on the thread that emitted the event.
pub struct RotatingFile {
    dir: PathBuf,
    stem: String,
    limit: u64,
    keep: usize,
    file: Option<fs::File>,
    len: u64,
    day: i64,
    /// a write error is reported ONCE; a log that cannot be written may not turn
    /// into a log that says so on every line
    complained: bool,
    /// the second the last rotation named its archive after, and the counter
    /// inside it. The counter only ever RISES inside one second, because
    /// scanning for the first free `-NNN` would reuse a number the prune has
    /// just freed and hand the NEWEST archive the OLDEST name (measured
    /// 2026-09-18: with 20 rotations and `keep` 3 the kept set had a hole).
    last_secs: i64,
    seq: u32,
}

impl RotatingFile {
    /// Opens (or appends to) `<dir>/<stem>.log`, creating `dir`.
    pub fn new(dir: impl Into<PathBuf>, stem: &str, limit: u64, keep: usize) -> io::Result<Self> {
        let dir = dir.into();
        fs::create_dir_all(&dir)?;
        let mut f = RotatingFile {
            dir,
            stem: stem.to_string(),
            limit: limit.max(MIN_ROTATE_BYTES),
            keep: keep.max(1),
            file: None,
            len: 0,
            day: 0,
            complained: false,
            last_secs: i64::MIN,
            seq: 0,
        };
        f.open_live()?;
        Ok(f)
    }

    /// the live file's path
    pub fn path(&self) -> PathBuf {
        self.dir.join(format!("{}.log", self.stem))
    }

    fn open_live(&mut self) -> io::Result<()> {
        let path = self.path();
        let f = fs::OpenOptions::new().create(true).append(true).open(&path)?;
        self.len = f.metadata().map(|m| m.len()).unwrap_or(0);
        // an APPENDED file keeps the day of its own last write, not of this
        // process's start, or a server restarted at 00:01 would carry yesterday's
        // lines into today's file for a whole day
        self.day = f
            .metadata()
            .and_then(|m| m.modified())
            .ok()
            .and_then(|t| t.duration_since(UNIX_EPOCH).ok())
            .map(|d| day_of(d.as_secs() as i64))
            .unwrap_or_else(|| day_of(now_parts().0));
        self.file = Some(f);
        Ok(())
    }

    /// The rotation decision, pure: the live length, the incoming bytes, the day
    /// the file belongs to and today. An EMPTY file never rotates, so a process
    /// that starts on a new day does not produce a zero-byte `.gz`.
    pub fn should_rotate(len: u64, incoming: u64, limit: u64, file_day: i64, today: i64) -> bool {
        len > 0 && (len + incoming > limit || today != file_day)
    }

    /// The name a rotation taken at `secs` gives the retired file, with `n`
    /// distinguishing two rotations inside the same second.
    ///
    /// The `-NNN` counter is ALWAYS there, zero padded, because the retention
    /// prune sorts by name: with the suffix only on a collision, `-1` sorted
    /// BEFORE the unsuffixed name of its own second (`-` is 0x2D, `.` is 0x2E)
    /// and the prune then deleted the SECOND archive of a second instead of the
    /// oldest one. Measured on a live `decode run 32` at
    /// `CROW_LOG_ROTATE_MB=0.001 CROW_LOG_KEEP=3` (2026-09-18): the three kept
    /// archives had a 450 ms hole in the middle. With the padded counter every
    /// name sorts chronologically, which the test below pins.
    pub fn rotated_name(stem: &str, secs: i64, n: u32) -> String {
        format!("{stem}-{}-{n:03}.log", stamp(secs))
    }

    fn rotate(&mut self) -> io::Result<()> {
        drop(self.file.take());
        let live = self.path();
        let secs = now_parts().0;
        // monotone inside one second; a fresh second starts at 000
        if secs == self.last_secs {
            self.seq = self.seq.saturating_add(1);
        } else {
            self.last_secs = secs;
            self.seq = 0;
        }
        let mut target = PathBuf::new();
        for n in self.seq..self.seq.saturating_add(1000) {
            let cand = self.dir.join(Self::rotated_name(&self.stem, secs, n));
            let gz = self.dir.join(format!(
                "{}.gz",
                Self::rotated_name(&self.stem, secs, n)
            ));
            if !cand.exists() && !gz.exists() {
                self.seq = n;
                target = cand;
                break;
            }
        }
        if target.as_os_str().is_empty() {
            // 1000 rotations in one second is not a log, it is a loop: keep
            // writing to the live file rather than losing it
            self.open_live()?;
            return Ok(());
        }
        fs::rename(&live, &target)?;
        // gzip, then drop the plain file. A failed gzip leaves the plain
        // rotated file in place: the bytes are what matter, the compression is
        // not worth losing them for.
        if let Err(e) = gzip_file(&target) {
            self.open_live()?;
            return Err(e);
        }
        let _ = fs::remove_file(&target);
        self.prune();
        self.open_live()
    }

    /// Deletes the oldest `engine-*.log.gz` until `keep` are left. The names
    /// sort chronologically, so this needs no filesystem timestamps and gives
    /// the same answer on both platforms.
    pub fn prune(&self) {
        let mut found: Vec<PathBuf> = Vec::new();
        let head = format!("{}-", self.stem);
        if let Ok(rd) = fs::read_dir(&self.dir) {
            for e in rd.flatten() {
                let name = e.file_name().to_string_lossy().to_string();
                if name.starts_with(&head) && name.ends_with(".log.gz") {
                    found.push(e.path());
                }
            }
        }
        found.sort();
        let excess = found.len().saturating_sub(self.keep);
        for p in found.into_iter().take(excess) {
            let _ = fs::remove_file(p);
        }
    }
}

impl Write for RotatingFile {
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        let today = day_of(now_parts().0);
        if Self::should_rotate(self.len, buf.len() as u64, self.limit, self.day, today) {
            if let Err(e) = self.rotate() {
                if !self.complained {
                    self.complained = true;
                    let _ = writeln!(io::stderr(), "[log] rotation failed: {e}");
                }
            }
        }
        match self.file.as_mut() {
            Some(f) => {
                f.write_all(buf)?;
                self.len += buf.len() as u64;
                Ok(buf.len())
            }
            None => {
                // the live file is gone (a rotation that could not reopen): say
                // so once and keep the process alive
                if !self.complained {
                    self.complained = true;
                    let _ = writeln!(io::stderr(), "[log] no log file open - lines are lost");
                }
                Ok(buf.len())
            }
        }
    }

    fn flush(&mut self) -> io::Result<()> {
        match self.file.as_mut() {
            Some(f) => f.flush(),
            None => Ok(()),
        }
    }
}

/// `path` -> `path.gz`, then the caller deletes `path`.
fn gzip_file(path: &Path) -> io::Result<()> {
    let mut src = fs::File::open(path)?;
    let dst = fs::File::create(format!("{}.gz", path.display()))?;
    let mut enc = flate2::write::GzEncoder::new(dst, flate2::Compression::default());
    io::copy(&mut src, &mut enc)?;
    enc.finish()?.sync_all()
}

// ---------------------------------------------------------------------------
// the two formats
// ---------------------------------------------------------------------------

/// The stderr mirror: the message and nothing else, so every line `tools/` and
/// `docs/` grep for is byte-identical to the `eprintln!` it replaced.
struct MessageOnly;

impl<S, N> FormatEvent<S, N> for MessageOnly
where
    S: tracing::Subscriber + for<'a> LookupSpan<'a>,
    N: for<'a> FormatFields<'a> + 'static,
{
    fn format_event(
        &self,
        ctx: &FmtContext<'_, S, N>,
        mut w: Writer<'_>,
        event: &tracing::Event<'_>,
    ) -> std::fmt::Result {
        ctx.field_format().format_fields(w.by_ref(), event)?;
        writeln!(w)
    }
}

/// ISO-8601 UTC to the millisecond, from [`iso8601`]: no `time`, no `chrono`.
struct Utc;

impl FormatTime for Utc {
    fn format_time(&self, w: &mut Writer<'_>) -> std::fmt::Result {
        let (s, ms) = now_parts();
        write!(w, "{}", iso8601(s, ms))
    }
}

// ---------------------------------------------------------------------------
// init
// ---------------------------------------------------------------------------

static STARTED: OnceLock<()> = OnceLock::new();
static GUARDS: OnceLock<Mutex<Vec<WorkerGuard>>> = OnceLock::new();

/// Keeps the two writer threads alive; dropping it drains both queues.
///
/// Bind it in `main` (`let _log = log::init();`) so the last lines of a process
/// that returns from `main` are on disk. Before a `std::process::exit` call
/// [`shutdown`] by hand — `exit` runs no destructor.
pub struct Guard {
    live: bool,
}

impl Drop for Guard {
    fn drop(&mut self) {
        if self.live {
            shutdown();
        }
    }
}

/// Drains and stops both writer threads. Idempotent; safe before `exit`.
pub fn shutdown() {
    if let Some(m) = GUARDS.get() {
        if let Ok(mut g) = m.lock() {
            g.clear();
        }
    }
}

fn non_blocking<W: Write + Send + 'static>(w: W, name: &str) -> (NonBlocking, WorkerGuard) {
    NonBlockingBuilder::default()
        .lossy(true)
        .buffered_lines_limit(BUFFERED_LINES)
        .thread_name(name)
        .finish(w)
}

/// Installs the subscriber for this process: the file layer, the stderr mirror
/// and the `CROW_LOG` filter. Idempotent — the second call is a no-op and
/// returns an inert guard, so a bin may call it from `main` without knowing
/// whether a test harness got there first.
pub fn init() -> Guard {
    if STARTED.set(()).is_err() {
        return Guard { live: false };
    }
    let cfg = cfg_from_env();
    let (stderr_nb, stderr_guard) = non_blocking(io::stderr(), "crow-log-stderr");
    let mirror = tracing_subscriber::fmt::layer()
        .with_ansi(false)
        .event_format(MessageOnly)
        .with_writer(stderr_nb);

    let mut guards = vec![stderr_guard];
    let mut file_err = None;
    let mut file_layer = None;
    let mut file_path = PathBuf::new();
    match RotatingFile::new(&cfg.dir, STEM, cfg.rotate_bytes, cfg.keep) {
        Ok(rf) => {
            file_path = rf.path();
            let (nb, g) = non_blocking(rf, "crow-log-file");
            guards.push(g);
            file_layer = Some(
                tracing_subscriber::fmt::layer()
                    .with_ansi(false)
                    .with_timer(Utc)
                    .with_target(true)
                    .with_level(true)
                    .with_writer(nb),
            );
        }
        Err(e) => file_err = Some(format!("{} : {e}", cfg.dir.display())),
    }

    let filter = EnvFilter::builder()
        .parse(&cfg.filter)
        .unwrap_or_else(|_| EnvFilter::new(DEFAULT_FILTER));
    let _ = tracing_subscriber::registry()
        .with(filter)
        .with(file_layer)
        .with(mirror)
        .try_init();
    let _ = GUARDS.set(Mutex::new(guards));

    if let Some(note) = &cfg.filter_note {
        tracing::warn!(target: "log", "[log] {note}");
    }
    match &file_err {
        None => tracing::info!(
            target: "log",
            "[log] file {} (rotate at {:.3} MiB, keep {}, gzip on rotation), stderr mirror on, filter {}",
            file_path.display(),
            cfg.rotate_bytes as f64 / (1024.0 * 1024.0),
            cfg.keep,
            cfg.filter
        ),
        Some(e) => tracing::warn!(
            target: "log",
            "[log] no file log ({e}) - stderr only, filter {}",
            cfg.filter
        ),
    }
    Guard { live: true }
}

// ---------------------------------------------------------------------------
// the two structured lines
// ---------------------------------------------------------------------------

/// The operating point of one process: everything a later number references.
#[derive(Clone, Debug)]
pub struct BootPoint<'a> {
    /// the binary that is reporting (`serve`, `decode`, `parity`, ...)
    pub bin: &'a str,
    /// the container this process opened
    pub container: &'a str,
    /// the hot-set sidecar, and where the hot set came from
    pub hotsets: &'a str,
    pub hotset_source: &'a str,
    /// context length in tokens, and the prefill chunk
    pub n_ctx: usize,
    pub prompt_chunk: usize,
    /// hot experts per layer after the planner's clamp, and the slots allocated
    pub residency_n: usize,
    pub residency_stride: usize,
    /// experts per layer in the model
    pub experts: usize,
    /// hot experts per layer, per layer — the cold-path policy of every layer,
    /// because `experts - hot[l]` is exactly what layer `l` reads from the tier
    pub hot_per_layer: Vec<usize>,
    /// the KV cache dtype (`fp8_e4m3` / `bf16`)
    pub kv_dtype: &'a str,
    /// the kernel path in force (`mma`/`scalar`, graph on or off)
    pub kernel_path: &'a str,
    /// what a cold expert costs and where it comes from
    pub cold_tier: &'a str,
    pub expert_bytes: u64,
    pub pinned_bytes: u64,
    /// the trickle / re-cut policy this process runs
    pub cold_policy: &'a str,
    /// the rest of the operating point a later number may reference
    pub layers: usize,
    pub qsa_ring_rows: usize,
    pub ple_cache_bytes: u64,
    pub vit: bool,
    pub prefix_cache: bool,
}

/// The boot report as ONE line of JSON. Pure, so its validity is a unit test.
pub fn boot_json(p: &BootPoint) -> String {
    let per_layer: Vec<serde_json::Value> =
        p.hot_per_layer.iter().map(|n| (*n).into()).collect();
    let v = serde_json::json!({
        "ts": iso8601(now_parts().0, now_parts().1),
        "event": "operating_point",
        "bin": p.bin,
        "container": p.container,
        "hotsets": p.hotsets,
        "hotset_source": p.hotset_source,
        "n_ctx": p.n_ctx,
        "prompt_chunk": p.prompt_chunk,
        "layers": p.layers,
        "experts_per_layer": p.experts,
        "residency_n": p.residency_n,
        "residency_stride": p.residency_stride,
        "kv_dtype": p.kv_dtype,
        "kernel_path": p.kernel_path,
        "cold_path": {
            "tier": p.cold_tier,
            "policy": p.cold_policy,
            "expert_bytes": p.expert_bytes,
            "pinned_bytes": p.pinned_bytes,
            "hot_per_layer": per_layer,
        },
        "qsa_ring_rows": p.qsa_ring_rows,
        "ple_cache_bytes": p.ple_cache_bytes,
        "vit": p.vit,
        "prefix_cache": p.prefix_cache,
    });
    v.to_string()
}

/// Emits [`boot_json`] at INFO on target `boot`.
pub fn boot(p: &BootPoint) {
    tracing::info!(target: "boot", "{}", boot_json(p));
}

/// The counters of ONE request, drained from the blocks that already exist
/// (`residency::counters` per layer, `Ple::req`/`miss`, the trickle's swaps) and
/// differenced against the previous request, so every number is request-local.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Routing {
    /// which request this is in this process, and what it did
    pub seq: u64,
    pub route: String,
    pub finish: String,
    /// the id counts the wire's `timings` block carries
    pub prompt_n: usize,
    pub cached_n: usize,
    pub predicted_n: usize,
    pub prompt_ms: f64,
    pub predicted_ms: f64,
    pub tok_s: f64,
    /// expert selections and how many of them were cold, this request
    pub selections: u64,
    pub cold: u64,
    /// how many of the 48 layers had cold work at all
    pub layers_cold: usize,
    /// bytes the cold path streamed for this request: `cold * expert_bytes`
    pub bytes_streamed: u64,
    /// PLE rows asked for, and the fills (misses) those rows cost
    pub ple_rows: u64,
    pub ple_fills: u64,
    /// hot-set swaps the stream trickle started for this request
    pub trickle_swaps: usize,
    /// what reading the device counter block cost
    pub counters_ms: f64,
    /// #68 (2026-09-18): the cross-turn repeat counter of THIS answer, as `serve` keeps
    /// it (`RepeatRing`, `bin/serve.rs`). `repeat_of` is how many answers back the most
    /// recent identical answer is, 0 = none in the ring; `repeat_run` is how many
    /// identical answers in a row ended with this one, 1 = none; `single_token` is true
    /// when the answer is exactly one generated id and the model ended it itself. Pure
    /// observability: nothing here is read by the sampler or by any decision. A
    /// `prefill_chunk` line has no answer, so all three stay at their default.
    pub repeat_of: usize,
    pub repeat_run: usize,
    pub single_token: bool,
}

impl Routing {
    /// The residency hit rate: the share of expert selections that found their
    /// expert in VRAM. 1.0 when nothing was selected, because no selection
    /// missed.
    pub fn hit_rate(&self) -> f64 {
        if self.selections == 0 {
            return 1.0;
        }
        1.0 - (self.cold as f64 / self.selections as f64)
    }

    /// The PLE miss rate over the rows this request asked for.
    pub fn ple_miss_rate(&self) -> f64 {
        if self.ple_rows == 0 {
            return 0.0;
        }
        self.ple_fills as f64 / self.ple_rows as f64
    }
}

/// One request's telemetry as ONE line of JSON. Pure, so its fields are a unit
/// test.
pub fn routing_json(r: &Routing) -> String {
    let round = |v: f64| (v * 1000.0).round() / 1000.0;
    let v = serde_json::json!({
        "ts": iso8601(now_parts().0, now_parts().1),
        "event": "routing",
        "seq": r.seq,
        "route": r.route,
        "finish": r.finish,
        "prompt_n": r.prompt_n,
        "cached_n": r.cached_n,
        "predicted_n": r.predicted_n,
        "prompt_ms": round(r.prompt_ms),
        "predicted_ms": round(r.predicted_ms),
        "tok_s": round(r.tok_s),
        "selections": r.selections,
        "cold": r.cold,
        "hit_rate": round(r.hit_rate()),
        "layers_cold": r.layers_cold,
        "bytes_streamed": r.bytes_streamed,
        "ple_rows": r.ple_rows,
        "ple_fills": r.ple_fills,
        "ple_miss_rate": round(r.ple_miss_rate()),
        "trickle_swaps": r.trickle_swaps,
        "counters_ms": round(r.counters_ms),
        "repeat_of": r.repeat_of,
        "repeat_run": r.repeat_run,
        "single_token": r.single_token,
    });
    v.to_string()
}

/// Emits [`routing_json`] at INFO on target `routing` — one line per request.
pub fn routing(r: &Routing) {
    tracing::info!(target: "routing", "{}", routing_json(r));
}

/// The same line for one prefill CHUNK, at DEBUG: `CROW_LOG=routing=debug` asks
/// for it, an operator's INFO file never carries it.
pub fn routing_chunk(r: &Routing) {
    tracing::debug!(target: "routing", "{}", routing_json(r));
}

// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Read;

    fn scratch(tag: &str) -> PathBuf {
        let n = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0);
        let p = std::env::temp_dir().join(format!("crow-log-{tag}-{}-{n}", std::process::id()));
        fs::create_dir_all(&p).unwrap();
        p
    }

    /// The `CROW_LOG` rule, without touching the environment: unset is INFO, a
    /// per-target string is passed through, and a string `EnvFilter` refuses
    /// falls back to INFO with a complaint instead of silencing the process.
    #[test]
    fn the_filter_rule_defaults_to_info_and_never_silences_a_typo() {
        assert_eq!(filter_spec(None), ("info".to_string(), None));
        assert_eq!(filter_spec(Some("")), ("info".to_string(), None));
        assert_eq!(filter_spec(Some("   ")), ("info".to_string(), None));
        assert_eq!(filter_spec(Some("trace")), ("trace".to_string(), None));
        // the two shapes the ticket names: an operator default with one
        // component raised, and the decode-path forensics on request
        assert_eq!(
            filter_spec(Some("info,routing=debug")),
            ("info,routing=debug".to_string(), None)
        );
        assert_eq!(
            filter_spec(Some(" info,decode=trace ")),
            ("info,decode=trace".to_string(), None)
        );
        let (spec, note) = filter_spec(Some("chat=nonsense"));
        assert_eq!(spec, "info", "a filter this build cannot parse must not silence it");
        assert!(note.is_some(), "and it must be said out loud: {note:?}");
        assert!(note.unwrap().contains("chat=nonsense"));
    }

    /// The two numeric knobs, including the decimal MiB the forced-rotation
    /// proof uses and the 0 that is not a retention policy.
    #[test]
    fn the_rotation_knobs_are_clamped_to_values_a_writer_can_honour() {
        assert_eq!(rotate_bytes(None), 64 * 1024 * 1024);
        assert_eq!(rotate_bytes(Some("1")), 1024 * 1024);
        assert_eq!(rotate_bytes(Some("0.001")), 1048); // ~1 KiB, the forced-rotation proof's limit
        // only a POSITIVE FINITE number is a limit; everything else is not a
        // value and the default applies, because a 0 MiB or -4 MiB log file is
        // not a thing a writer can honour
        assert_eq!(rotate_bytes(Some("0")), 64 * 1024 * 1024);
        assert_eq!(rotate_bytes(Some("-4")), 64 * 1024 * 1024);
        assert_eq!(rotate_bytes(Some("nonsense")), 64 * 1024 * 1024);
        assert_eq!(rotate_bytes(Some("nan")), 64 * 1024 * 1024);
        assert_eq!(rotate_bytes(Some("inf")), 64 * 1024 * 1024);
        // and the floor is MIN_ROTATE_BYTES, so a silly-small ask still works
        assert_eq!(rotate_bytes(Some("0.0000001")), MIN_ROTATE_BYTES);
        assert_eq!(rotate_bytes(Some("1000000")), 1 << 32, "and the ceiling is 4 GiB");
        assert_eq!(keep_count(None), DEFAULT_KEEP);
        assert_eq!(keep_count(Some("3")), 3);
        assert_eq!(keep_count(Some("0")), 1, "0 kept files is not a retention policy");
        assert_eq!(keep_count(Some("x")), DEFAULT_KEEP);
    }

    /// Windows and Linux are the same code path with different inputs, so both
    /// rules are tested on whichever host runs the suite.
    #[test]
    fn the_default_log_directory_follows_the_convention_of_each_os() {
        let tmp = Path::new("/tmp");
        assert_eq!(
            log_dir_from(true, Some("C:\\Users\\r\\AppData\\Local"), None, None, None, tmp),
            PathBuf::from("C:\\Users\\r\\AppData\\Local").join("crow").join("logs")
        );
        assert_eq!(
            log_dir_from(true, None, Some("C:\\Users\\r"), None, None, tmp),
            PathBuf::from("C:\\Users\\r/AppData/Local/crow/logs")
        );
        assert_eq!(
            log_dir_from(false, None, None, Some("/home/r/.local/state"), None, tmp),
            PathBuf::from("/home/r/.local/state/crow/logs")
        );
        assert_eq!(
            log_dir_from(false, None, None, None, Some("/home/r"), tmp),
            PathBuf::from("/home/r/.local/state/crow/logs")
        );
        // an empty value is not a value, and with nothing at all the temp
        // directory is the one place that always exists
        assert_eq!(
            log_dir_from(false, None, None, Some("  "), None, tmp),
            PathBuf::from("/tmp/crow/logs")
        );
        assert_eq!(
            log_dir_from(true, None, None, None, None, tmp),
            PathBuf::from("/tmp/crow/logs")
        );
    }

    /// The calendar the rotated names and the file timestamps are built from.
    #[test]
    fn the_calendar_is_the_gregorian_one() {
        assert_eq!(civil_from_days(0), (1970, 1, 1));
        assert_eq!(civil_from_days(-1), (1969, 12, 31));
        assert_eq!(civil_from_days(11_016), (2000, 2, 29)); // a leap day
        assert_eq!(civil_from_days(20_714), (2026, 9, 18)); // the day of #13
        assert_eq!(stamp(0), "19700101-000000");
        assert_eq!(stamp(20_714 * 86_400 + 5 * 3600 + 11 * 60 + 22), "20260918-051122");
        assert_eq!(iso8601(20_714 * 86_400 + 5 * 3600 + 11 * 60 + 22, 7), "2026-09-18T05:11:22.007Z");
        assert_eq!(day_of(-1), -1, "a second before the epoch is the day before it");
    }

    /// The rotation decision, pure. An empty file never rotates - not on size,
    /// not on the day boundary - or a restart just after midnight would write a
    /// zero-byte archive.
    #[test]
    fn the_rotation_decision_fires_on_size_and_on_the_day_boundary() {
        assert!(!RotatingFile::should_rotate(0, 4096, 1024, 5, 9));
        assert!(!RotatingFile::should_rotate(500, 100, 1024, 5, 5));
        assert!(RotatingFile::should_rotate(1000, 100, 1024, 5, 5), "size");
        assert!(RotatingFile::should_rotate(1, 1, 1 << 30, 5, 6), "the day changed");
        assert_eq!(RotatingFile::rotated_name("engine", 0, 0), "engine-19700101-000000-000.log");
        assert_eq!(RotatingFile::rotated_name("engine", 0, 2), "engine-19700101-000000-002.log");
        // the retention prune sorts by NAME, so the name must sort the way the
        // rotations happened - this is the bug of 2026-09-18 (see `rotated_name`)
        let (a, b, c) = (
            RotatingFile::rotated_name("engine", 100, 0),
            RotatingFile::rotated_name("engine", 100, 1),
            RotatingFile::rotated_name("engine", 101, 0),
        );
        assert!(a < b && b < c, "a name sorts out of order: {a} {b} {c}");
    }

    /// The writer itself, against a real temp directory with a 1 KiB limit: the
    /// live file, k of N gzipped archives kept, and the gzip really is one -
    /// its bytes decompress to the lines that were written.
    #[test]
    fn a_tiny_limit_rotates_gzips_and_keeps_exactly_n_files() {
        let dir = scratch("rotate");
        let keep = 3;
        {
            let mut w = RotatingFile::new(&dir, "engine", 1024, keep).unwrap();
            assert_eq!(w.path(), dir.join("engine.log"));
            for i in 0..400 {
                writeln!(w, "line {i:04} {}", "x".repeat(40)).unwrap();
            }
            w.flush().unwrap();
        }
        let mut gz: Vec<PathBuf> = fs::read_dir(&dir)
            .unwrap()
            .flatten()
            .map(|e| e.path())
            .filter(|p| p.to_string_lossy().ends_with(".log.gz"))
            .collect();
        gz.sort();
        assert_eq!(gz.len(), keep, "retention: exactly {keep} archives, found {gz:?}");
        assert!(dir.join("engine.log").is_file(), "the live file is still there");
        // no plain rotated file survives a successful gzip
        let plain: Vec<PathBuf> = fs::read_dir(&dir)
            .unwrap()
            .flatten()
            .map(|e| e.path())
            .filter(|p| {
                let n = p.file_name().unwrap().to_string_lossy().to_string();
                n.starts_with("engine-") && n.ends_with(".log")
            })
            .collect();
        assert!(plain.is_empty(), "a rotated plain file was left behind: {plain:?}");
        // the newest archive is a real gzip of real lines
        let f = fs::File::open(gz.last().unwrap()).unwrap();
        let mut txt = String::new();
        flate2::read::GzDecoder::new(f).read_to_string(&mut txt).unwrap();
        assert!(txt.starts_with("line "), "not the lines that were written: {:?}", &txt[..40.min(txt.len())]);
        assert!(txt.lines().count() >= 10, "an archive of {} lines", txt.lines().count());
        for l in txt.lines() {
            assert_eq!(l.len(), 50, "a line was cut across the rotation: {l:?}");
        }
        // and the kept archives are the NEWEST ones, in order, contiguous with the
        // live file: read the three archives by name and then `engine.log`, and the
        // line counter must rise by one from the first kept line to line 0399.
        // This is what the 2026-09-18 naming bug broke - the prune kept three
        // archives with a hole in the middle (see `rotated_name`).
        let mut kept = String::new();
        for g in &gz {
            let mut t = String::new();
            flate2::read::GzDecoder::new(fs::File::open(g).unwrap())
                .read_to_string(&mut t)
                .unwrap();
            kept.push_str(&t);
        }
        kept.push_str(&fs::read_to_string(dir.join("engine.log")).unwrap());
        let nums: Vec<usize> = kept
            .lines()
            .map(|l| l[5..9].parse::<usize>().expect("a line number"))
            .collect();
        assert_eq!(*nums.last().unwrap(), 399, "the live file must end at the last line written");
        assert!(nums[0] > 0, "nothing was pruned, so the retention was never exercised");
        for w in nums.windows(2) {
            assert_eq!(w[1], w[0] + 1, "a hole between two kept files at line {}", w[0]);
        }
        fs::remove_dir_all(&dir).ok();
    }

    /// An existing `engine.log` is appended to, and its length counts towards
    /// the limit - a restarted server may not start a fresh 64 MiB.
    #[test]
    fn a_restart_appends_to_the_live_file_and_keeps_its_length() {
        let dir = scratch("append");
        {
            let mut w = RotatingFile::new(&dir, "engine", 1 << 20, 2).unwrap();
            writeln!(w, "first process").unwrap();
        }
        {
            let mut w = RotatingFile::new(&dir, "engine", 1 << 20, 2).unwrap();
            writeln!(w, "second process").unwrap();
        }
        let txt = fs::read_to_string(dir.join("engine.log")).unwrap();
        assert_eq!(txt, "first process\nsecond process\n");
        fs::remove_dir_all(&dir).ok();
    }

    /// What ONE line costs at the CALL SITE, the three ways this engine can pay
    /// for it - the before/after pair of the ticket, measured with no GPU and no
    /// model, so it is reproducible anywhere:
    ///
    /// 1. `writeln!` to a real file: what today's `eprintln!` cost, minus the
    ///    stderr lock (the same `write` syscall shape).
    /// 2. an ENABLED `tracing` event through the real pipeline (rotating file
    ///    behind `non_blocking`): the call site formats and hands over, the write
    ///    happens on the worker thread.
    /// 3. a DISABLED event (`trace!` under an `info` filter): the cost the
    ///    decode-loop forensics of `decode_step` pays on every operator run.
    ///
    /// `cargo test --release -- --nocapture what_one_line_costs` prints them.
    /// The ceilings are loose on purpose: this is a regression guard against a
    /// call site that starts BLOCKING, not a benchmark with a target.
    #[test]
    fn what_one_line_costs_at_the_call_site() {
        use std::time::Instant;
        const N: usize = 100_000;
        const LINE: &str = "[dec] pos 41 in 13 -> out 248046, graph 1, captured 0, ple rows 0 misses 0, step 23.800 ms";
        let dir = scratch("cost");

        let mut plain = fs::File::create(dir.join("eprintln.txt")).unwrap();
        let t = Instant::now();
        for _ in 0..N {
            writeln!(plain, "{LINE}").unwrap();
        }
        plain.flush().unwrap();
        let sync_ns = t.elapsed().as_nanos() as f64 / N as f64;

        let rf = RotatingFile::new(&dir, "bench", 1 << 30, 2).unwrap();
        let (nb, guard) = non_blocking(rf, "crow-log-bench");
        let sub = tracing_subscriber::registry()
            .with(EnvFilter::new(DEFAULT_FILTER))
            .with(
                tracing_subscriber::fmt::layer()
                    .with_ansi(false)
                    .event_format(MessageOnly)
                    .with_writer(nb),
            );
        // `#[inline(never)]`, because a call site in a loop lets the optimiser
        // hoist the callsite-interest load out of it and the disabled event then
        // measures 0.2 ns, which is not what `decode_step` pays. One call, one
        // interest load, as in the real loop.
        #[inline(never)]
        fn enabled_once(i: usize) {
            tracing::info!(target: "decode", "[dec] pos {i} in 13 -> out 248046, graph 1, captured 0, ple rows 0 misses 0, step 23.800 ms");
        }
        #[inline(never)]
        fn disabled_once(i: usize) {
            tracing::trace!(target: "decode", "[dec] pos {i} in 13 -> out 248046, graph 1, captured 0, ple rows 0 misses 0, step 23.800 ms");
        }
        let (on_ns, off_ns) = tracing::subscriber::with_default(sub, || {
            let t = Instant::now();
            for i in 0..N {
                enabled_once(std::hint::black_box(i));
            }
            let on = t.elapsed().as_nanos() as f64 / N as f64;
            let t = Instant::now();
            for i in 0..N {
                disabled_once(std::hint::black_box(i));
            }
            (on, t.elapsed().as_nanos() as f64 / N as f64)
        });
        drop(guard);

        println!(
            "one {} B line, {N} calls each: synchronous writeln! to a file {sync_ns:.0} ns/call, \
             ENABLED tracing event through the non-blocking rotating file {on_ns:.0} ns/call, \
             DISABLED event (trace! under the `info` filter) {off_ns:.1} ns/call",
            LINE.len() + 1
        );
        // the numbers of record on the RTX 5090 / Arch host (2026-09-18, #13):
        // 333-354 ns synchronous, 347-373 ns enabled, 0.6 ns disabled (three runs)
        assert!(off_ns < 200.0, "a DISABLED event may not cost {off_ns:.1} ns - the filter is not being checked first");
        assert!(off_ns < sync_ns, "a disabled event must be cheaper than the write it replaced");
        assert!(on_ns < 50_000.0, "an ENABLED event at {on_ns:.0} ns/call is BLOCKING on the write");
        fs::remove_dir_all(&dir).ok();
    }

    /// The boot report is ONE line, it is valid JSON, and it carries every
    /// number of the operating point the ticket names.
    #[test]
    fn the_boot_line_is_one_valid_json_line_with_the_operating_point() {
        let p = BootPoint {
            bin: "serve",
            container: "converter/Qwen3.8-Flash-Next-CNQ4.5-M.cnq",
            hotsets: "decode_out/hotsets-M-longctx2100-n160.json",
            hotset_source: "sidecar",
            n_ctx: 200_000,
            prompt_chunk: 2048,
            residency_n: 155,
            residency_stride: 162,
            experts: 512,
            hot_per_layer: vec![155; 48],
            kv_dtype: "fp8_e4m3",
            kernel_path: "mma, cuda graph",
            cold_tier: "nvfp4 pinned host",
            expert_bytes: 1_179_648,
            pinned_bytes: 46_000_000_000,
            cold_policy: "zero-copy read, stream trickle every 8, max 7/layer",
            layers: 48,
            qsa_ring_rows: 2100,
            ple_cache_bytes: 134_217_728,
            vit: true,
            prefix_cache: true,
        };
        let line = boot_json(&p);
        assert!(!line.contains('\n'), "the boot report must be ONE line");
        let v: serde_json::Value = serde_json::from_str(&line).expect("valid JSON");
        assert_eq!(v["event"], "operating_point");
        assert_eq!(v["n_ctx"], 200_000);
        assert_eq!(v["residency_n"], 155);
        assert_eq!(v["kv_dtype"], "fp8_e4m3");
        assert_eq!(v["kernel_path"], "mma, cuda graph");
        assert_eq!(v["cold_path"]["tier"], "nvfp4 pinned host");
        assert_eq!(
            v["cold_path"]["hot_per_layer"].as_array().unwrap().len(),
            48,
            "the cold-path policy is reported PER LAYER"
        );
        assert!(v["ts"].as_str().unwrap().ends_with('Z'));
    }

    /// The routing line: one JSON line, every counter the spec promises, and the
    /// two derived rates.
    #[test]
    fn the_routing_line_carries_every_counter_of_one_request() {
        let r = Routing {
            seq: 7,
            route: "POST /v1/chat/completions".into(),
            finish: "stop".into(),
            prompt_n: 2100,
            cached_n: 1024,
            predicted_n: 64,
            prompt_ms: 812.5,
            predicted_ms: 870.4,
            tok_s: 72.4,
            selections: 48 * 10 * 64,
            cold: 3072,
            layers_cold: 41,
            bytes_streamed: 3072 * 1_179_648,
            ple_rows: 8192,
            ple_fills: 41,
            trickle_swaps: 56,
            counters_ms: 0.312,
            repeat_of: 2,
            repeat_run: 3,
            single_token: true,
        };
        let line = routing_json(&r);
        assert!(!line.contains('\n'), "one line per request");
        let v: serde_json::Value = serde_json::from_str(&line).expect("valid JSON");
        for k in [
            "ts", "seq", "prompt_n", "cached_n", "predicted_n", "tok_s", "cold", "hit_rate",
            "layers_cold", "bytes_streamed", "ple_rows", "ple_fills", "trickle_swaps",
            // #68: the three cross-turn fields a log reader plots per request
            "repeat_of", "repeat_run", "single_token",
        ] {
            assert!(!v[k].is_null(), "the routing line has no {k}");
        }
        assert_eq!(v["repeat_of"], 2);
        assert_eq!(v["repeat_run"], 3);
        assert_eq!(v["single_token"], true);
        // a line that carries no answer (the prefill chunk) says so with the defaults
        assert_eq!(serde_json::from_str::<serde_json::Value>(
            &routing_json(&Routing::default()))
            .expect("valid JSON")["repeat_run"], 0);
        assert_eq!(v["bytes_streamed"], 3_623_878_656u64);
        assert_eq!(v["hit_rate"], 0.9, "1 - 3072/30720");
        assert_eq!(v["ple_miss_rate"], 0.005);
        // nothing selected is not a miss
        let empty = Routing::default();
        assert_eq!(empty.hit_rate(), 1.0);
        assert_eq!(empty.ple_miss_rate(), 0.0);
    }
}
