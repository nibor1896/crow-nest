//! #149 (plan step 17b, measurement book F lever 4): the NVMe expert tier's read backend.
//!
//! The stager's source is an interface (`docs/architecture.md` 2.3, decision 2026-09-02): RAM
//! tier now, NVMe tier (variant C) a second backend of the same interface. This module is that
//! interface ([`ColdSource`]) and the NVMe backend ([`NvmeSource`]). It is **opt-in and wired
//! nowhere**: no decode path, no planner and no default constructs it yet. What exists is the
//! read itself, so the hand-off (job ring, `p9_job_ring` stage C) can be built on top of a
//! backend that is already proven byte-identical to the load path.
//!
//! What one fetch does:
//!
//! - takes at most [`MAX_IN_FLIGHT`] (8) expert records of one layer — a record is the expert's
//!   two NVFP4 slabs, gate_up and down, at their absolute container offsets — and the caller's
//!   destination for each slab (pinned host memory in the engine);
//! - refuses, by name, any offset, length or destination that is not a multiple of [`ALIGN`]
//!   (4096 B, this drive's physical sector; `FILE_FLAG_NO_BUFFERING` / `O_DIRECT` need it). It does
//!   NOT round a span out: the record goes straight into the caller's buffer, which has no head
//!   room for the bytes around it;
//! - deals the records round-robin to the reader threads. **Each reader owns its own file
//!   handle** (robin's llama.cpp fork `66f40bc`, 2026-08-03: one handle per thread 2.22x, one
//!   shared handle at queue depth 8 1.01x). On Windows a reader opens the container with
//!   `FILE_FLAG_NO_BUFFERING | FILE_FLAG_OVERLAPPED`, binds it to its own I/O completion port,
//!   issues EVERY read of its share before it drains the first completion, so all slabs of a
//!   fetch are in flight at once. On Linux a reader opens it `O_DIRECT` and reads with `pread`
//!   (synchronous per reader; no io_uring here);
//! - runs `residency::sanitize_sf_slab` (scale byte 0x7F -> 0x7E) on both slabs in the
//!   destination before [`ColdSource::wait`] returns, exactly as the load path does
//!   (`residency.rs`, every slab), so no record is ever published unsanitized.
//!
//! Not built here: IoRing (optional per plan step 17), the three-tier split, the RAM tier behind
//! the trait, and the boot wiring beyond the refusal in `boot.rs` (`CROW_NVME_TIER` together
//! with `CROW_COLD_TIER`).

use crate::cnq::{Cnq, TensorInfo};
use crate::residency::sanitize_sf_slab;
use std::path::{Path, PathBuf};
use std::sync::mpsc;

/// Alignment of every offset, length and destination address the backend accepts: 4096 B, the
/// physical sector of this machine's NVMe (`docs/nvme-read-rate.md`: 512 B logical / 4096 B
/// physical) and the page size `VirtualAlloc` / `cuMemHostAlloc` hand out.
pub const ALIGN: u64 = 4096;

/// The most expert records one fetch carries: the misses of one layer, all in flight at once
/// (the llama.cpp fork measurement saturated at queue depth 8; depth 64 gave 2.15x vs 2.22x).
pub const MAX_IN_FLIGHT: usize = 8;

/// One contiguous byte range of the container, in absolute file offsets.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Span {
    pub off: u64,
    pub len: usize,
}

/// One routed expert of one layer: its gate_up slab and its down slab.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ExpertRecord {
    pub layer: u32,
    pub id: u32,
    pub gu: Span,
    pub dn: Span,
}

impl ExpertRecord {
    /// The record of expert `id` in the layer whose expert tensors are `gu` and `dn`, with the
    /// per-expert slab sizes the residency uses (`id * slab` into the tensor, the same rows
    /// `Cnq::read_range` reads at load). Refuses an overlay tensor (it lives in another file), a
    /// tensor that is not `nvfp4`, and an id past the tensor's end.
    pub fn locate(cnq: &Cnq, gu: &TensorInfo, dn: &TensorInfo, layer: u32, id: u32, gu_bytes: u64, dn_bytes: u64) -> Result<Self, String> {
        let mut spans = [Span { off: 0, len: 0 }; 2];
        for (k, (t, slab)) in [(gu, gu_bytes), (dn, dn_bytes)].into_iter().enumerate() {
            if t.overlay {
                return Err(format!("{}: an overlay tensor is not read by the NVMe tier (it lives in the overlay file)", t.name));
            }
            if t.dtype != "nvfp4" {
                return Err(format!("{}: dtype {} - the NVMe tier reads nvfp4 expert slabs only", t.name, t.dtype));
            }
            let rel = id as u64 * slab;
            if rel + slab > Cnq::byte_len(t) {
                return Err(format!("{}: expert {id} x {slab} B runs past the tensor's {} B", t.name, Cnq::byte_len(t)));
            }
            spans[k] = Span { off: cnq.abs_offset(t, rel), len: slab as usize };
        }
        Ok(ExpertRecord { layer, id, gu: spans[0], dn: spans[1] })
    }
}

/// Where one record goes: the caller's buffers for the gate_up and the down slab. Each must be
/// [`ALIGN`]-aligned and hold at least the slab's length.
#[derive(Clone, Copy, Debug)]
pub struct RecordDst {
    pub gu: *mut u8,
    pub dn: *mut u8,
}

/// The refusal of a span or destination the unbuffered read cannot take. `None` = accepted.
pub fn alignment_refusal(rec: &ExpertRecord, dst: &RecordDst) -> Option<String> {
    for (what, s, p) in [("gate_up", rec.gu, dst.gu), ("down", rec.dn, dst.dn)] {
        let tag = format!("layer {} expert {} {what}", rec.layer, rec.id);
        if s.len == 0 {
            return Some(format!("{tag}: empty span"));
        }
        if !s.off.is_multiple_of(ALIGN) {
            return Some(format!(
                "{tag}: file offset {} is not a multiple of {ALIGN} B (unbuffered NVMe reads need sector-aligned offsets; the container's expert slabs must start on {ALIGN} B)",
                s.off
            ));
        }
        if !(s.len as u64).is_multiple_of(ALIGN) {
            return Some(format!("{tag}: length {} B is not a multiple of {ALIGN} B", s.len));
        }
        if s.len as u64 > u32::MAX as u64 {
            return Some(format!("{tag}: length {} B exceeds one read (u32)", s.len));
        }
        if p.is_null() || !(p as usize).is_multiple_of(ALIGN as usize) {
            return Some(format!("{tag}: destination {p:p} is not {ALIGN} B aligned"));
        }
    }
    None
}

/// What a finished fetch delivered.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct FetchReport {
    pub records: usize,
    pub bytes: u64,
    /// scale bytes `sanitize_sf_slab` rewrote 0x7F -> 0x7E across all slabs of the fetch
    pub clamped: u64,
}

/// One outstanding fetch. [`ColdSource::wait`] consumes it; dropping it unwaited blocks until
/// the readers are done, because they write into the caller's buffers until then.
pub struct Ticket {
    parts: Vec<mpsc::Receiver<Result<FetchReport, String>>>,
}

impl Ticket {
    fn drain(&mut self) -> Result<FetchReport, String> {
        let mut sum = FetchReport::default();
        let mut err = None;
        for rx in self.parts.drain(..) {
            match rx.recv() {
                Ok(Ok(r)) => {
                    sum.records += r.records;
                    sum.bytes += r.bytes;
                    sum.clamped += r.clamped;
                }
                Ok(Err(e)) => {
                    err.get_or_insert(e);
                }
                Err(_) => {
                    err.get_or_insert("an NVMe reader thread died".to_string());
                }
            }
        }
        match err {
            Some(e) => Err(e),
            None => Ok(sum),
        }
    }
}

impl Drop for Ticket {
    fn drop(&mut self) {
        let _ = self.drain();
    }
}

/// The stager's source (`docs/architecture.md` 2.3): RAM tier now, NVMe tier a second backend.
pub trait ColdSource {
    /// Start reading `jobs` (at most [`MAX_IN_FLIGHT`] records of one layer) into their
    /// destinations. Refuses the whole fetch, before any read, if one span or destination is
    /// misaligned or there are too many records.
    ///
    /// # Safety
    ///
    /// Every destination must be valid for writes of its slab's length and must not be read or
    /// written by anyone else until the returned ticket has been waited on (or dropped).
    unsafe fn fetch(&self, jobs: &[(ExpertRecord, RecordDst)]) -> Result<Ticket, String>;

    /// Block until the fetch is complete and every slab sanitized.
    fn wait(&self, t: Ticket) -> Result<FetchReport, String>;
}

/// The NVMe backend's knobs. Opt-in: nothing builds one by default.
#[derive(Clone, Debug)]
pub struct NvmeConfig {
    /// the container file
    pub path: PathBuf,
    /// reader threads, each with its own handle; 1..=MAX_IN_FLIGHT. Default 1: PREREG amendment 5
    /// (robin, 2026-10-08) makes 1 reader binding for step 14 (`docs/nvme-read-rate.md`).
    pub readers: usize,
    /// CPU index per reader (reader i pins to `affinity[i % len]`); `None` leaves placement to
    /// the OS scheduler
    pub affinity: Option<Vec<usize>>,
}

impl NvmeConfig {
    pub fn new(path: impl AsRef<Path>) -> Self {
        NvmeConfig { path: path.as_ref().to_path_buf(), readers: 1, affinity: None }
    }
}

struct Job {
    rec: ExpertRecord,
    dst: RecordDst,
}
// the destinations are the caller's, valid until the ticket is waited on (`fetch`'s contract)
unsafe impl Send for Job {}

struct Batch {
    jobs: Vec<Job>,
    reply: mpsc::Sender<Result<FetchReport, String>>,
}

/// The NVMe backend: `readers` threads, one container handle each.
pub struct NvmeSource {
    tx: Vec<mpsc::Sender<Batch>>,
    threads: Vec<std::thread::JoinHandle<()>>,
}

impl NvmeSource {
    /// Open the container once per reader (unbuffered, overlapped on Windows; `O_DIRECT` on
    /// Linux) and start the readers. Any open or affinity failure is returned by name.
    pub fn open(cfg: &NvmeConfig) -> Result<NvmeSource, String> {
        if cfg.readers == 0 || cfg.readers > MAX_IN_FLIGHT {
            return Err(format!("NVMe tier: readers {} outside 1..={MAX_IN_FLIGHT}", cfg.readers));
        }
        if let Some(a) = &cfg.affinity {
            if a.is_empty() {
                return Err("NVMe tier: empty affinity list".into());
            }
        }
        let mut tx = Vec::new();
        let mut threads = Vec::new();
        for i in 0..cfg.readers {
            let cpu = cfg.affinity.as_ref().map(|a| a[i % a.len()]);
            let reader = Reader::open(&cfg.path)?;
            let (btx, brx) = mpsc::channel::<Batch>();
            let (ready_tx, ready_rx) = mpsc::channel::<Result<(), String>>();
            let h = std::thread::Builder::new()
                .name(format!("nvme-reader-{i}"))
                .spawn(move || {
                    if let Some(c) = cpu {
                        if let Err(e) = pin_current_thread(c) {
                            let _ = ready_tx.send(Err(e));
                            return;
                        }
                    }
                    let _ = ready_tx.send(Ok(()));
                    while let Ok(b) = brx.recv() {
                        let _ = b.reply.send(reader.run(&b.jobs));
                    }
                })
                .map_err(|e| format!("NVMe tier: spawning reader {i}: {e}"))?;
            match ready_rx.recv() {
                Ok(Ok(())) => {}
                Ok(Err(e)) => return Err(e),
                Err(_) => return Err(format!("NVMe tier: reader {i} died at start")),
            }
            tx.push(btx);
            threads.push(h);
        }
        Ok(NvmeSource { tx, threads })
    }

    pub fn readers(&self) -> usize {
        self.tx.len()
    }
}

impl Drop for NvmeSource {
    fn drop(&mut self) {
        self.tx.clear(); // closes every channel: the readers fall out of `recv`
        for h in self.threads.drain(..) {
            let _ = h.join();
        }
    }
}

impl ColdSource for NvmeSource {
    unsafe fn fetch(&self, jobs: &[(ExpertRecord, RecordDst)]) -> Result<Ticket, String> {
        if jobs.len() > MAX_IN_FLIGHT {
            return Err(format!("NVMe tier: {} records in one fetch, at most {MAX_IN_FLIGHT} (the misses of one layer)", jobs.len()));
        }
        for (rec, dst) in jobs {
            if let Some(why) = alignment_refusal(rec, dst) {
                return Err(format!("NVMe tier refused: {why}"));
            }
        }
        let n = self.tx.len().min(jobs.len());
        let mut share: Vec<Vec<Job>> = (0..n).map(|_| Vec::new()).collect();
        for (k, (rec, dst)) in jobs.iter().enumerate() {
            share[k % n].push(Job { rec: *rec, dst: *dst });
        }
        let mut parts = Vec::new();
        for (i, jobs) in share.into_iter().enumerate() {
            let (reply, rx) = mpsc::channel();
            self.tx[i].send(Batch { jobs, reply }).map_err(|_| format!("NVMe tier: reader {i} is gone"))?;
            parts.push(rx);
        }
        Ok(Ticket { parts })
    }

    fn wait(&self, mut t: Ticket) -> Result<FetchReport, String> {
        t.drain()
    }
}

/// Sanitize both slabs of every job in place, in the destination (the load path's rule).
///
/// # Safety
///
/// The reads into the destinations have completed with the full length.
unsafe fn sanitize_jobs(jobs: &[Job]) -> u64 {
    let mut n = 0;
    for j in jobs {
        for (s, p) in [(j.rec.gu, j.dst.gu), (j.rec.dn, j.dst.dn)] {
            n += sanitize_sf_slab(std::slice::from_raw_parts_mut(p, s.len));
        }
    }
    n
}

#[cfg(windows)]
mod win {
    use std::ffi::c_void;

    pub type Handle = *mut c_void;

    /// `OVERLAPPED` (minwinbase.h), x64 layout: the Offset/OffsetHigh arm of the union
    #[repr(C)]
    pub struct Overlapped {
        pub internal: usize,
        pub internal_high: usize,
        pub offset: u32,
        pub offset_high: u32,
        pub h_event: Handle,
    }

    pub const FILE_FLAG_NO_BUFFERING: u32 = 0x2000_0000;
    pub const FILE_FLAG_OVERLAPPED: u32 = 0x4000_0000;
    pub const ERROR_IO_PENDING: i32 = 997;
    pub const INFINITE: u32 = 0xFFFF_FFFF;

    type FnCreateIoCompletionPort = unsafe extern "system" fn(Handle, Handle, usize, u32) -> Handle;
    type FnReadFile = unsafe extern "system" fn(Handle, *mut c_void, u32, *mut u32, *mut Overlapped) -> i32;
    type FnGetQueuedCompletionStatus = unsafe extern "system" fn(Handle, *mut u32, *mut usize, *mut *mut Overlapped, u32) -> i32;
    type FnCloseHandle = unsafe extern "system" fn(Handle) -> i32;
    type FnSetThreadAffinityMask = unsafe extern "system" fn(Handle, usize) -> usize;
    type FnGetCurrentThread = unsafe extern "system" fn() -> Handle;

    /// kernel32 entry points through libloading, the crate's convention (`cnq.rs`, `cuda.rs`)
    pub struct K32 {
        pub create_iocp: FnCreateIoCompletionPort,
        pub read_file: FnReadFile,
        pub gqcs: FnGetQueuedCompletionStatus,
        pub close: FnCloseHandle,
        pub set_affinity: FnSetThreadAffinityMask,
        pub current_thread: FnGetCurrentThread,
        _lib: libloading::Library,
    }

    pub fn k32() -> Result<&'static K32, String> {
        static K: std::sync::OnceLock<Result<K32, String>> = std::sync::OnceLock::new();
        K.get_or_init(|| unsafe {
            let lib = libloading::Library::new("kernel32.dll").map_err(|e| format!("kernel32.dll: {e}"))?;
            fn sym<T: Copy>(lib: &libloading::Library, n: &[u8]) -> Result<T, String> {
                unsafe { lib.get::<T>(n).map(|s| *s).map_err(|e| format!("kernel32 {}: {e}", String::from_utf8_lossy(n))) }
            }
            Ok(K32 {
                create_iocp: sym(&lib, b"CreateIoCompletionPort\0")?,
                read_file: sym(&lib, b"ReadFile\0")?,
                gqcs: sym(&lib, b"GetQueuedCompletionStatus\0")?,
                close: sym(&lib, b"CloseHandle\0")?,
                set_affinity: sym(&lib, b"SetThreadAffinityMask\0")?,
                current_thread: sym(&lib, b"GetCurrentThread\0")?,
                _lib: lib,
            })
        })
        .as_ref()
        .map_err(|e| e.clone())
    }
}

/// One reader's own handle to the container (and, on Windows, its own completion port).
struct Reader {
    file: std::fs::File,
    #[cfg(windows)]
    port: usize,
}

#[cfg(windows)]
impl Reader {
    fn open(path: &Path) -> Result<Reader, String> {
        use std::os::windows::fs::OpenOptionsExt;
        use std::os::windows::io::AsRawHandle;
        let k = win::k32()?;
        let file = std::fs::OpenOptions::new()
            .read(true)
            .custom_flags(win::FILE_FLAG_NO_BUFFERING | win::FILE_FLAG_OVERLAPPED)
            .open(path)
            .map_err(|e| format!("NVMe tier: {}: {e}", path.display()))?;
        let port = unsafe { (k.create_iocp)(file.as_raw_handle() as win::Handle, std::ptr::null_mut(), 1, 1) };
        if port.is_null() {
            return Err(format!("NVMe tier: CreateIoCompletionPort: {}", std::io::Error::last_os_error()));
        }
        Ok(Reader { file, port: port as usize })
    }

    /// Issue every read of `jobs`, then drain exactly as many completions as were issued (a read
    /// still in flight writes into the caller's buffer, so the function never returns before the
    /// last one is back), then sanitize.
    fn run(&self, jobs: &[Job]) -> Result<FetchReport, String> {
        use std::os::windows::io::AsRawHandle;
        let k = win::k32()?;
        let h = self.file.as_raw_handle() as win::Handle;
        let port = self.port as win::Handle;
        let mut ovs: Vec<Box<win::Overlapped>> = Vec::with_capacity(jobs.len() * 2);
        let mut lens: Vec<usize> = Vec::with_capacity(jobs.len() * 2);
        let mut err: Option<String> = None;
        'issue: for j in jobs {
            for (s, p) in [(j.rec.gu, j.dst.gu), (j.rec.dn, j.dst.dn)] {
                let mut ov = Box::new(win::Overlapped {
                    internal: 0,
                    internal_high: 0,
                    offset: s.off as u32,
                    offset_high: (s.off >> 32) as u32,
                    h_event: std::ptr::null_mut(),
                });
                let ok = unsafe { (k.read_file)(h, p as *mut _, s.len as u32, std::ptr::null_mut(), &mut *ov) };
                if ok == 0 {
                    let e = std::io::Error::last_os_error();
                    if e.raw_os_error() != Some(win::ERROR_IO_PENDING) {
                        err = Some(format!("NVMe tier: ReadFile layer {} expert {} at {}: {e}", j.rec.layer, j.rec.id, s.off));
                        break 'issue;
                    }
                }
                // issued: a completion packet is queued whether it completed now or later
                ovs.push(ov);
                lens.push(s.len);
            }
        }
        let mut bytes = 0u64;
        for _ in 0..ovs.len() {
            let mut n = 0u32;
            let mut key = 0usize;
            let mut pov: *mut win::Overlapped = std::ptr::null_mut();
            let ok = unsafe { (k.gqcs)(port, &mut n, &mut key, &mut pov, win::INFINITE) };
            if pov.is_null() {
                // no packet dequeued with an INFINITE wait: the port itself failed while reads
                // may still write into caller memory - nothing safe to return
                panic!("NVMe tier: GetQueuedCompletionStatus failed with reads in flight: {}", std::io::Error::last_os_error());
            }
            let i = ovs.iter().position(|o| std::ptr::eq(&**o, pov)).expect("completion for an OVERLAPPED this reader did not issue");
            if ok == 0 {
                err.get_or_insert(format!("NVMe tier: read failed: {}", std::io::Error::last_os_error()));
            } else if n as usize != lens[i] {
                err.get_or_insert(format!("NVMe tier: short read, {n} of {} B (past the end of the file?)", lens[i]));
            }
            bytes += n as u64;
        }
        if let Some(e) = err {
            return Err(e);
        }
        let clamped = unsafe { sanitize_jobs(jobs) };
        Ok(FetchReport { records: jobs.len(), bytes, clamped })
    }
}

#[cfg(windows)]
impl Drop for Reader {
    fn drop(&mut self) {
        if let Ok(k) = win::k32() {
            unsafe { (k.close)(self.port as win::Handle) };
        }
    }
}

#[cfg(unix)]
impl Reader {
    fn open(path: &Path) -> Result<Reader, String> {
        use std::os::unix::fs::OpenOptionsExt;
        let file = std::fs::OpenOptions::new()
            .read(true)
            .custom_flags(libc::O_DIRECT)
            .open(path)
            .map_err(|e| format!("NVMe tier: {} (O_DIRECT): {e}", path.display()))?;
        Ok(Reader { file })
    }

    /// `pread` every slab of `jobs` on this reader's own descriptor, then sanitize.
    fn run(&self, jobs: &[Job]) -> Result<FetchReport, String> {
        use std::os::unix::fs::FileExt;
        let mut bytes = 0u64;
        for j in jobs {
            for (s, p) in [(j.rec.gu, j.dst.gu), (j.rec.dn, j.dst.dn)] {
                let buf = unsafe { std::slice::from_raw_parts_mut(p, s.len) };
                self.file
                    .read_exact_at(buf, s.off)
                    .map_err(|e| format!("NVMe tier: pread layer {} expert {} at {}: {e}", j.rec.layer, j.rec.id, s.off))?;
                bytes += s.len as u64;
            }
        }
        let clamped = unsafe { sanitize_jobs(jobs) };
        Ok(FetchReport { records: jobs.len(), bytes, clamped })
    }
}

/// Pin the calling thread to CPU `cpu`.
#[cfg(windows)]
fn pin_current_thread(cpu: usize) -> Result<(), String> {
    if cpu >= usize::BITS as usize {
        return Err(format!("NVMe tier: CPU {cpu} is outside the first processor group (0..{})", usize::BITS));
    }
    let k = win::k32()?;
    let prev = unsafe { (k.set_affinity)((k.current_thread)(), 1usize << cpu) };
    if prev == 0 {
        return Err(format!("NVMe tier: SetThreadAffinityMask(CPU {cpu}): {}", std::io::Error::last_os_error()));
    }
    Ok(())
}

#[cfg(target_os = "linux")]
fn pin_current_thread(cpu: usize) -> Result<(), String> {
    unsafe {
        let mut set: libc::cpu_set_t = std::mem::zeroed();
        if cpu >= 8 * std::mem::size_of::<libc::cpu_set_t>() {
            return Err(format!("NVMe tier: CPU {cpu} is outside cpu_set_t"));
        }
        libc::CPU_SET(cpu, &mut set);
        if libc::sched_setaffinity(0, std::mem::size_of::<libc::cpu_set_t>(), &set) != 0 {
            return Err(format!("NVMe tier: sched_setaffinity(CPU {cpu}): {}", std::io::Error::last_os_error()));
        }
    }
    Ok(())
}

#[cfg(all(unix, not(target_os = "linux")))]
fn pin_current_thread(cpu: usize) -> Result<(), String> {
    Err(format!("NVMe tier: reader affinity (CPU {cpu}) is not supported on this OS"))
}

#[cfg(test)]
mod tests {
    use super::*;

    // GLM-5.3-Flash expert slab sizes (#149 Evidence): gate_up 2 x 2048 x 4096 NVFP4 values,
    // down 4096 x 2048, 36 B per 64 values. Both are multiples of 4096 and of 36.
    const GU: u64 = 9_437_184;
    const DN: u64 = 4_718_592;
    const EXPERTS: u64 = 10;

    /// A synthetic CNQ v2 container in its own temp dir, removed on drop (also on a failed
    /// assert). 10 experts at GLM slab size: 141.6 MB.
    struct Synth {
        dir: PathBuf,
        path: PathBuf,
    }

    impl Drop for Synth {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.dir);
        }
    }

    /// `gu_rel` = the gate_up tensor's offset relative to the blob start (12). 4084 puts every
    /// expert slab on a 4096 B boundary; 0 is the format's default and leaves them all at 12 mod 4096.
    fn synth(tag: &str, gu_rel: u64) -> Synth {
        use std::io::Write;
        let dir = std::env::temp_dir().join(format!("crow-nvme-{tag}-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("synth.cnq");
        let dn_rel = gu_rel + EXPERTS * GU;
        let blob_len = dn_rel + EXPERTS * DN;
        let sha = |s: &str| crate::cnq::sha256_hex(s.as_bytes());
        let index = serde_json::json!({
            "format": "crow-nest-quant",
            "format_version": 2,
            "blob_offset": 12,
            "recipe": "synthetic-nvme-149",
            "model": {
                "family": "GlmSynthetic",
                "model_type": "synthetic",
                "config_json": "{}",
                "config_json_sha256": sha("{}"),
                "generation_config_json": "{}",
                "generation_config_json_sha256": sha("{}"),
                "source": { "repo": "crow-nest/synthetic-nvme", "revision": "149", "shards": [] },
                "geo": {}
            },
            "tensors": [
                { "name": "layers.0.mlp.experts.gate_up_proj", "section": "text", "dtype": "nvfp4",
                  "offset": gu_rel, "n_values": EXPERTS * GU / 36 * 64, "shape": [EXPERTS, 4096, 4096] },
                { "name": "layers.0.mlp.experts.down_proj", "section": "text", "dtype": "nvfp4",
                  "offset": dn_rel, "n_values": EXPERTS * DN / 36 * 64, "shape": [EXPERTS, 4096, 2048] }
            ]
        });
        let ib = serde_json::to_vec(&index).unwrap();
        let mut f = std::io::BufWriter::with_capacity(1 << 20, std::fs::File::create(&path).unwrap());
        f.write_all(b"CNQ1\0\0\0\0\0\0\0\0").unwrap();
        // xorshift bytes: every byte value occurs, so the scale bytes include 0x7F
        let mut x = 0x9E37_79B9_7F4A_7C15u64;
        let mut chunk = vec![0u8; 1 << 20];
        let mut left = blob_len;
        while left > 0 {
            for w in chunk.chunks_exact_mut(8) {
                x ^= x << 13;
                x ^= x >> 7;
                x ^= x << 17;
                w.copy_from_slice(&x.to_le_bytes());
            }
            let n = left.min(chunk.len() as u64) as usize;
            f.write_all(&chunk[..n]).unwrap();
            left -= n as u64;
        }
        f.write_all(&ib).unwrap();
        f.write_all(&(ib.len() as u64).to_le_bytes()).unwrap();
        f.flush().unwrap();
        Synth { dir, path }
    }

    /// a 4096-aligned heap buffer standing in for the engine's pinned slot (no GPU here)
    struct Aligned {
        p: *mut u8,
        layout: std::alloc::Layout,
    }

    impl Aligned {
        fn new(len: usize) -> Aligned {
            let layout = std::alloc::Layout::from_size_align(len, ALIGN as usize).unwrap();
            let p = unsafe { std::alloc::alloc_zeroed(layout) };
            assert!(!p.is_null());
            Aligned { p, layout }
        }
        fn bytes(&self) -> &[u8] {
            unsafe { std::slice::from_raw_parts(self.p, self.layout.size()) }
        }
    }

    impl Drop for Aligned {
        fn drop(&mut self) {
            unsafe { std::alloc::dealloc(self.p, self.layout) };
        }
    }

    fn tensors(cnq: &Cnq) -> (TensorInfo, TensorInfo) {
        (
            cnq.find("layers.0.mlp.experts.gate_up_proj", "text").clone(),
            cnq.find("layers.0.mlp.experts.down_proj", "text").clone(),
        )
    }

    /// Eight records of one layer, out of order, through two readers with their own handles:
    /// each destination is byte-identical to `Cnq::read_range` + `sanitize_sf_slab` (the load
    /// path's bytes), the raw container bytes did carry 0x7F scale bytes (so sanitize had work),
    /// and the report counts exactly the bytes the load path clamps.
    #[test]
    fn an_expert_record_read_through_the_nvme_backend_is_the_load_paths_bytes() {
        let s = synth("ident", 4084);
        let mut cnq = Cnq::open_checked(s.path.to_str().unwrap()).unwrap();
        let (gt, dt) = tensors(&cnq);
        let ids = [9u32, 0, 3, 7, 1, 8, 5, 2];
        let recs: Vec<ExpertRecord> = ids.iter().map(|&id| ExpertRecord::locate(&cnq, &gt, &dt, 0, id, GU, DN).unwrap()).collect();
        let bufs: Vec<(Aligned, Aligned)> = ids.iter().map(|_| (Aligned::new(GU as usize), Aligned::new(DN as usize))).collect();
        let jobs: Vec<(ExpertRecord, RecordDst)> =
            recs.iter().zip(&bufs).map(|(r, (g, d))| (*r, RecordDst { gu: g.p, dn: d.p })).collect();
        let mut cfg = NvmeConfig::new(&s.path);
        cfg.readers = 2;
        let src = NvmeSource::open(&cfg).unwrap();
        assert_eq!(src.readers(), 2);
        let t = unsafe { src.fetch(&jobs) }.unwrap();
        let rep = src.wait(t).unwrap();
        assert_eq!((rep.records, rep.bytes), (8, 8 * (GU + DN)));
        let mut want_clamped = 0;
        let mut raw_had_7f = false;
        for (k, &id) in ids.iter().enumerate() {
            for (t, slab, buf) in [(&gt, GU, &bufs[k].0), (&dt, DN, &bufs[k].1)] {
                let raw = cnq.read_range(t, id as u64 * slab, slab as usize);
                let mut want = raw.clone();
                let n = sanitize_sf_slab(&mut want);
                want_clamped += n;
                raw_had_7f |= n > 0;
                assert!(buf.bytes() == want.as_slice(), "expert {id} {}: NVMe bytes differ from read_range + sanitize", t.name);
            }
        }
        assert!(raw_had_7f, "the synthetic slabs carried no 0x7F scale byte - the sanitize check proves nothing");
        assert_eq!(rep.clamped, want_clamped);
        drop(src);
        drop(cnq);
    }

    /// Sanitize is applied before the record is published: the destination holds no 0x7F scale
    /// byte although the container does, and it differs from the raw container bytes.
    #[test]
    fn sanitize_is_applied_before_wait_returns() {
        let s = synth("sanitize", 4084);
        let mut cnq = Cnq::open_checked(s.path.to_str().unwrap()).unwrap();
        let (gt, dt) = tensors(&cnq);
        let rec = ExpertRecord::locate(&cnq, &gt, &dt, 0, 4, GU, DN).unwrap();
        let (g, d) = (Aligned::new(GU as usize), Aligned::new(DN as usize));
        let src = NvmeSource::open(&NvmeConfig::new(&s.path)).unwrap();
        let rep = src.wait(unsafe { src.fetch(&[(rec, RecordDst { gu: g.p, dn: d.p })]) }.unwrap()).unwrap();
        assert!(rep.clamped > 0);
        let raw = cnq.read_range(&gt, 4 * GU, GU as usize);
        assert!(raw.chunks_exact(36).any(|b| b[..4].contains(&0x7F)), "container slab has a 0x7F scale byte");
        assert!(g.bytes() != raw.as_slice());
        for buf in [g.bytes(), d.bytes()] {
            assert!(!buf.chunks_exact(36).any(|b| b[..4].contains(&0x7F)), "a 0x7F scale byte reached the destination");
        }
        drop(src);
        drop(cnq);
    }

    /// A misaligned request is refused by name before any read: the format's default layout
    /// (slabs at 12 mod 4096), a misaligned destination, a length off the sector, and a fetch of
    /// more than eight records.
    #[test]
    fn a_misaligned_request_is_refused_by_name() {
        let s = synth("misaligned", 0);
        let cnq = Cnq::open_checked(s.path.to_str().unwrap()).unwrap();
        let (gt, dt) = tensors(&cnq);
        let rec = ExpertRecord::locate(&cnq, &gt, &dt, 0, 1, GU, DN).unwrap();
        assert_eq!(rec.gu.off % ALIGN, 12);
        let (g, d) = (Aligned::new(GU as usize + 4096), Aligned::new(DN as usize + 4096));
        let ok_dst = RecordDst { gu: g.p, dn: d.p };
        let src = NvmeSource::open(&NvmeConfig::new(&s.path)).unwrap();
        let e = unsafe { src.fetch(&[(rec, ok_dst)]) }.err().expect("an offset at 12 mod 4096 was accepted");
        assert!(e.contains("file offset") && e.contains("4096"), "{e}");
        // an aligned span with a destination off the 4096 B grid
        let mut good = rec;
        good.gu.off -= 12;
        good.dn.off -= 12;
        let bad_dst = RecordDst { gu: unsafe { g.p.add(512) }, dn: d.p };
        let e = unsafe { src.fetch(&[(good, bad_dst)]) }.err().expect("a misaligned destination was accepted");
        assert!(e.contains("destination") && e.contains("aligned"), "{e}");
        let mut short = good;
        short.dn.len -= 512;
        let e = unsafe { src.fetch(&[(short, ok_dst)]) }.err().expect("a length off the sector was accepted");
        assert!(e.contains("length"), "{e}");
        let nine = vec![(good, ok_dst); 9];
        let e = unsafe { src.fetch(&nine) }.err().expect("nine records were accepted");
        assert!(e.contains("at most 8"), "{e}");
        // nothing was read into the buffers
        assert!(g.bytes().iter().all(|&b| b == 0) && d.bytes().iter().all(|&b| b == 0));
        drop(src);
        drop(cnq);
    }
}
