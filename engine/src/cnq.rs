//! CNQ container reader (magic `CNQ1`, index-as-trailer). Layout and facts are
//! probe-pinned (p10/p15/p16, converter Sauberlauf): payload blob at [12..),
//! index JSON + u64 LE `index_len` in the last 8 bytes, tensor offsets relative
//! to blob start. The `section` field separates text/vit/ple/mtp — ALWAYS filter
//! by section (mtp carries its own layers.0 copy). dtype "bf16" = keep (raw LE
//! bytes), "nvfp4" = 36 B per 64 values (4 ue4m3 scales + 32 B LSB-first
//! nibbles) + one global f32 scale per tensor, rows contiguous without padding,
//! "i64" = raw metadata tables (never quantized).

use std::io::{Read, Seek, SeekFrom};

#[derive(Clone)]
pub struct TensorInfo {
    pub name: String,
    pub section: String,
    pub dtype: String,
    pub offset: u64, // relative to blob start
    pub n_values: u64,
    pub global_scale: f32,
    pub shape: Vec<u64>,
}

pub struct Cnq {
    pub file: std::fs::File,
    pub blob_offset: u64,
    pub tensors: Vec<TensorInfo>,
    /// read-only view of the whole container (windows: a kernel32 file mapping,
    /// unix: mmap; 0 = not mapped -> seek/read fallback). On WINDOWS this is the
    /// PLE row read path and a row read is a page-cache memcpy (~1 us on a hit
    /// instead of a seek+read syscall pair). On LINUX no read takes it since
    /// TASK H (see `read_at`); what is left of it there is
    /// `CROW_PLE_FETCH=madv`. CROW_MMAP=0 disables.
    pub map: usize,
    pub map_len: u64,
    /// the CreateFileMappingW handle behind `map` (closed in Drop after the
    /// unmap); always 0 on unix, where munmap takes `map_len` instead
    pub map_handle: usize,
    /// container path: Drop re-opens it unbuffered once to purge its cached pages
    pub path: String,
    /// the pending DONTNEED window of `fadvise_consumed`: [lo, hi) of container
    /// bytes already read and no longer needed in the page cache
    fadv: (u64, u64),
    /// [lo, hi) of the `ple` section in container bytes — the one range `Drop`
    /// leaves in the page cache for the next process (TASK H, 2026-09-17).
    /// (0, 0) when the container carries no `ple` section, and then `Drop`
    /// purges the whole file exactly as it did before.
    ple_range: (u64, u64),
}

/// pending bytes that make `Cnq::fadvise_consumed` issue its DONTNEED call.
/// 64 MiB is one syscall per 36 expert slabs on the sequential cold-tier fill;
/// smaller only raises the syscall count, larger only delays the reclaim.
const FADV_BATCH: u64 = 64 << 20;

// ---------------------------------------------------------------------------
// the row fetch (TASK H, 2026-09-17): a chunk's PLE row misses as ONE batch of
// concurrent reads instead of one blocking read per row.
//
// Every row address a prefill chunk (or a decode token) needs is a pure
// function of its ids, so the whole set is known before the forward pass. Read
// one at a time the set costs `misses x latency`; issued together it costs
// `misses / queue_depth x latency`. Measured on this NVMe, 2000 random 4 KiB
// reads over the 26.8 GiB `ple` section: 10,607 IOPS on one thread (0.094 ms
// each), 201,305 at 16 threads (0.005 ms), 280,198 at 32 (0.004 ms), and it
// falls back to ~210,000 at 128. Nothing here changes which bytes the engine
// reads - only how long `Cnq::read_range` waits for them.
// ---------------------------------------------------------------------------

/// page granularity of every container read the kernel serves
const PAGE: u64 = 4096;

/// longest byte run one worker claims in one read. A coalesced run longer than
/// this is split, so a chunk's pages spread over the pool instead of piling up
/// behind one thread.
const WARM_RUN_MAX: u64 = 64 << 10;

/// worker count of the row-fetch pool when `CROW_PLE_FETCH` names none
const WARM_THREADS: usize = 16;

/// How `Warm::rows` turns a set of row offsets into I/O — `CROW_PLE_FETCH`.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum WarmMode {
    /// the pre-TASK-H path: touch every row through the mapping, on this
    /// thread, in the order given (`CROW_PLE_FETCH=0`)
    Serial,
    /// `n` worker threads, each reading whole pages (default, `n` = 16)
    Threads(usize),
    /// one `madvise(MADV_WILLNEED)` per coalesced run and no workers
    /// (`CROW_PLE_FETCH=madv`)
    Willneed,
}

/// read once — the env is fixed per process
pub fn warm_mode() -> WarmMode {
    static M: std::sync::OnceLock<WarmMode> = std::sync::OnceLock::new();
    *M.get_or_init(|| match std::env::var("CROW_PLE_FETCH").as_deref() {
        Ok("0") => WarmMode::Serial,
        Ok("madv") => WarmMode::Willneed,
        Ok(v) => WarmMode::Threads(v.parse::<usize>().ok().filter(|n| *n > 0).unwrap_or(WARM_THREADS)),
        Err(_) => WarmMode::Threads(WARM_THREADS),
    })
}

/// The pages a set of `row_len`-byte rows occupies, as sorted, deduplicated,
/// coalesced byte runs of at most `WARM_RUN_MAX`, clamped to `file_len`.
///
/// Deduplication is the point as much as the sort: a 4 KiB page holds ~37 PLE
/// rows, so a chunk's misses land on far fewer pages than it has rows.
pub fn page_runs(offsets: &[u64], row_len: usize, file_len: u64) -> Vec<(u64, u32)> {
    let mut pages: Vec<u64> = Vec::with_capacity(offsets.len());
    for &off in offsets {
        let end = (off + row_len as u64).min(file_len);
        let mut p = off / PAGE * PAGE;
        while p < end {
            pages.push(p);
            p += PAGE;
        }
    }
    pages.sort_unstable();
    pages.dedup();
    let mut runs: Vec<(u64, u32)> = Vec::new();
    for p in pages {
        match runs.last_mut() {
            Some(run) if run.0 + run.1 as u64 == p && (run.1 as u64) < WARM_RUN_MAX => run.1 += PAGE as u32,
            _ => runs.push((p, PAGE as u32)),
        }
    }
    if let Some(run) = runs.last_mut() {
        if run.0 + run.1 as u64 > file_len {
            run.1 = (file_len - run.0) as u32;
        }
    }
    runs
}

/// one worker's claim on a batch: `runs[lo..hi]`
#[cfg(unix)]
struct WarmJob {
    batch: std::sync::Arc<WarmBatch>,
    lo: usize,
    hi: usize,
}

/// one caller's set of runs, and the count it waits on
#[cfg(unix)]
struct WarmBatch {
    runs: Vec<(u64, u32)>,
    left: std::sync::Mutex<usize>,
    done: std::sync::Condvar,
}

/// The pool behind `WarmMode::Threads`. One per process and per container: the
/// workers outlive every `Cnq`, so a handle costs nothing to clone and a
/// teardown never waits on a read in flight.
#[cfg(unix)]
struct WarmPool {
    /// a second read-only descriptor on the container. It carries
    /// POSIX_FADV_RANDOM on purpose: `Cnq::open`'s descriptor carries
    /// POSIX_FADV_SEQUENTIAL for the load sweep, which DOUBLES the readahead
    /// window around every 4 KiB row page (measured: 0.193 ms against 0.141 ms
    /// per row fault on the same file).
    file: std::fs::File,
    /// urgent queue (the forward pass is waiting) and the background queue (the
    /// next chunk's rows). Workers drain `now` first, or a prefetch of 10k
    /// pages issued one chunk ahead would stand in front of the 2k pages the
    /// current chunk is blocked on.
    now: std::sync::Mutex<(std::collections::VecDeque<WarmJob>, std::collections::VecDeque<WarmJob>)>,
    cv: std::sync::Condvar,
    /// how many workers actually started (see `start`)
    threads: std::sync::atomic::AtomicUsize,
}

#[cfg(unix)]
impl WarmPool {
    fn start(path: &str, threads: usize) -> Option<std::sync::Arc<WarmPool>> {
        use std::os::unix::io::AsRawFd;
        let file = std::fs::File::open(path).ok()?;
        unsafe { libc::posix_fadvise(file.as_raw_fd(), 0, 0, libc::POSIX_FADV_RANDOM) };
        let pool = std::sync::Arc::new(WarmPool {
            file,
            now: std::sync::Mutex::new((std::collections::VecDeque::new(), std::collections::VecDeque::new())),
            cv: std::sync::Condvar::new(),
            threads: threads.into(),
        });
        // A worker that starts and one that does not are both fine; a worker
        // that starts while the caller gives up is NOT, because it would wait on
        // a queue nobody feeds. So count what started and keep it.
        let mut started = 0usize;
        for _ in 0..threads {
            let p = std::sync::Arc::clone(&pool);
            if std::thread::Builder::new().name("cnq-rowfetch".into()).spawn(move || p.work()).is_ok() {
                started += 1;
            }
        }
        if started == 0 {
            return None;
        }
        pool.threads.store(started, std::sync::atomic::Ordering::Relaxed);
        Some(pool)
    }

    /// a lock that survives a panicking caller: a poisoned queue would turn a
    /// row fetch into an abort, and the teardown rule of `f8f75c0` is that no
    /// cleanup path aborts
    fn lock(&self) -> std::sync::MutexGuard<'_, (std::collections::VecDeque<WarmJob>, std::collections::VecDeque<WarmJob>)> {
        self.now.lock().unwrap_or_else(|e| e.into_inner())
    }

    fn work(self: std::sync::Arc<WarmPool>) {
        use std::os::unix::io::AsRawFd;
        let fd = self.file.as_raw_fd();
        let mut buf = vec![0u8; WARM_RUN_MAX as usize];
        loop {
            let job = {
                let mut q = self.lock();
                loop {
                    if let Some(j) = q.0.pop_front().or_else(|| q.1.pop_front()) {
                        break j;
                    }
                    q = self.cv.wait(q).unwrap_or_else(|e| e.into_inner());
                }
            };
            for &(off, len) in &job.batch.runs[job.lo..job.hi] {
                // the bytes go nowhere: this reads the pages INTO the page
                // cache, and `Cnq::read_range` then reads the row from there
                unsafe {
                    libc::pread(fd, buf.as_mut_ptr() as *mut libc::c_void, len as usize, off as libc::off_t);
                }
            }
            let mut left = job.batch.left.lock().unwrap_or_else(|e| e.into_inner());
            *left -= job.hi - job.lo;
            if *left == 0 {
                job.batch.done.notify_all();
            }
        }
    }

    /// queue `runs` and return when every one of them has been read
    fn submit(&self, runs: Vec<(u64, u32)>, urgent: bool) {
        let n = runs.len();
        if n == 0 {
            return;
        }
        // enough jobs to keep every worker busy, without one lock round per run
        let per = n.div_ceil(self.threads.load(std::sync::atomic::Ordering::Relaxed) * 2).clamp(1, 32);
        let batch = std::sync::Arc::new(WarmBatch {
            runs,
            left: std::sync::Mutex::new(n),
            done: std::sync::Condvar::new(),
        });
        {
            let mut q = self.lock();
            let mut lo = 0usize;
            while lo < n {
                let hi = (lo + per).min(n);
                let job = WarmJob { batch: std::sync::Arc::clone(&batch), lo, hi };
                if urgent { q.0.push_back(job) } else { q.1.push_back(job) }
                lo = hi;
            }
        }
        self.cv.notify_all();
        let mut left = batch.left.lock().unwrap_or_else(|e| e.into_inner());
        while *left > 0 {
            left = batch.done.wait(left).unwrap_or_else(|e| e.into_inner());
        }
    }
}

/// the process's one pool, started on the first batch that asks for it
#[cfg(unix)]
fn warm_pool(path: &str, threads: usize) -> Option<&'static std::sync::Arc<WarmPool>> {
    static POOL: std::sync::OnceLock<Option<(String, std::sync::Arc<WarmPool>)>> = std::sync::OnceLock::new();
    match POOL.get_or_init(|| WarmPool::start(path, threads).map(|p| (path.to_string(), p))) {
        Some((p, pool)) if p == path => Some(pool),
        _ => None,
    }
}

/// A `Send` handle to a container's row fetch, from `Cnq::warm`.
#[derive(Clone)]
pub struct Warm {
    path: std::sync::Arc<str>,
    map: usize,
    map_len: u64,
}

impl Warm {
    /// Make the pages of `offsets` (each `row_len` bytes long) readable, and
    /// return when they are. The rows themselves are still read by
    /// `Cnq::read_range`, byte for byte as before.
    pub fn rows(&self, offsets: &[u64], row_len: usize) {
        if offsets.is_empty() {
            return;
        }
        #[cfg(unix)]
        match warm_mode() {
            WarmMode::Threads(n) => {
                if let Some(pool) = warm_pool(&self.path, n) {
                    pool.submit(page_runs(offsets, row_len, self.map_len), true);
                }
            }
            WarmMode::Willneed => {
                if self.map != 0 {
                    for (off, len) in page_runs(offsets, row_len, self.map_len) {
                        unsafe {
                            libc::madvise((self.map + off as usize) as *mut libc::c_void, len as usize, libc::MADV_WILLNEED);
                        }
                    }
                }
            }
            // CROW_PLE_FETCH=0: no batch. `Cnq::read_range` then reads each row
            // on its own, one latency at a time — the arm the batch is measured
            // against.
            WarmMode::Serial => {}
        }
        #[cfg(windows)]
        {
            // the mapping IS the row read path on windows, so the fetch is the
            // touch; the reader pool and `path` behind it are unix-only
            let _ = &self.path;
            self.touch(offsets, row_len);
        }
    }

    /// The same set, queued BEHIND every urgent one: the prefill's prefetch
    /// thread warms the next chunk while the current chunk is computing, and
    /// must never delay the rows that chunk is blocked on.
    pub fn rows_ahead(&self, offsets: &[u64], row_len: usize) {
        #[cfg(unix)]
        if let WarmMode::Threads(n) = warm_mode() {
            if !offsets.is_empty() {
                if let Some(pool) = warm_pool(&self.path, n) {
                    pool.submit(page_runs(offsets, row_len, self.map_len), false);
                    return;
                }
            }
        }
        self.rows(offsets, row_len);
    }

    /// windows: read the first and last byte of every row through the mapping,
    /// on this thread, one fault at a time — there the mapping IS the row read
    /// path, and the Linux measurement that replaced it does not apply. No-op
    /// without a mapping (`CROW_MMAP=0`).
    #[cfg(windows)]
    fn touch(&self, offsets: &[u64], row_len: usize) {
        if self.map == 0 || row_len == 0 {
            return;
        }
        let mut sink = 0u8;
        for &off in offsets {
            if off + row_len as u64 <= self.map_len {
                unsafe {
                    sink ^= std::ptr::read_volatile((self.map as *const u8).add(off as usize));
                    sink ^= std::ptr::read_volatile((self.map as *const u8).add(off as usize + row_len - 1));
                }
            }
        }
        std::hint::black_box(sink);
    }
}

/// drop the file's pages from the system cache: an open with
/// FILE_FLAG_NO_BUFFERING while no cached handle exists makes NTFS purge the
/// cache section (2026-09-06: the cache manager kept 3.3 GB of tier reads in
/// the system cache working set until process exit, so a harness reload saw
/// that much less "available" RAM and residency.rs refused to pin the tier)
///
/// `keep` is the one byte range the purge steps over: the `ple` section, whose
/// rows are the only bytes a NEXT process wants to find warm (TASK H fix A,
/// 2026-09-17). `(0, 0)` purges the whole file, which is what every caller
/// outside `Cnq::drop` gets and what windows does in either case — there the
/// reason for the purge is the standby list counting against the next load's
/// available RAM, which the reclaim-aware Linux RAM gate (`0c9feb5`) does not
/// need. On Linux the kept pages cost the next process nothing: page cache is
/// not part of `free_for_pin` (architecture 8.8) and is reclaimable.
pub fn purge_cache(path: &str, keep: (u64, u64)) {
    // CROW_CNQ_PURGE=0 keeps the cache (control knob, unchanged: no purge at all)
    if std::env::var("CROW_CNQ_PURGE").as_deref() == Ok("0") {
        return;
    }
    #[cfg(windows)]
    {
        let _ = keep;
        use std::os::windows::fs::OpenOptionsExt;
        let _ = std::fs::OpenOptions::new().read(true).custom_flags(0x2000_0000 /* FILE_FLAG_NO_BUFFERING */).open(path);
    }
    // unix twin: POSIX_FADV_DONTNEED over the file (len 0 = to EOF) drops its
    // clean page-cache pages. It only evicts unmapped, unreferenced pages, so
    // the Drop order below - unmap, close the handle, then purge - is what
    // makes them droppable here too.
    #[cfg(unix)]
    {
        use std::os::unix::io::AsRawFd;
        if let Ok(f) = std::fs::File::open(path) {
            let fd = f.as_raw_fd();
            let dontneed = |lo: u64, len: u64| unsafe {
                libc::posix_fadvise(fd, lo as libc::off_t, len as libc::off_t, libc::POSIX_FADV_DONTNEED)
            };
            let (lo, hi) = keep;
            if hi > lo {
                dontneed(0, lo);
                dontneed(hi, 0); // len 0 = to EOF
            } else {
                dontneed(0, 0);
            }
        }
    }
}

/// the process's null device: Drop parks `Cnq.file` on it to close the container
/// handle before the purge
#[cfg(windows)]
const NULL_DEVICE: &str = "NUL";
#[cfg(unix)]
const NULL_DEVICE: &str = "/dev/null";

#[cfg(windows)]
type FnUnmapView = unsafe extern "system" fn(*const std::ffi::c_void) -> i32;
#[cfg(windows)]
type FnCloseHandle = unsafe extern "system" fn(*mut std::ffi::c_void) -> i32;

impl Drop for Cnq {
    /// unmap the view and close the mapping object: pages of the container
    /// touched through the view live in this process's working set until the
    /// unmap, and a harness that opens a Cnq per request otherwise carries
    /// them into its next load (~4.5 GB less "available" RAM on the reload)
    fn drop(&mut self) {
        if self.map != 0 {
        #[cfg(windows)]
        unsafe {
            let Ok(lib) = libloading::Library::new("kernel32.dll") else { return };
            if let Ok(unmap) = lib.get::<FnUnmapView>(b"UnmapViewOfFile\0") {
                unmap(self.map as *const _);
            }
            if self.map_handle != 0 {
                if let Ok(close) = lib.get::<FnCloseHandle>(b"CloseHandle\0") {
                    close(self.map_handle as *mut _);
                }
            }
        }
        // unix: one munmap of the whole mapping, no handle to close after it
        #[cfg(unix)]
        unsafe {
            libc::munmap(self.map as *mut libc::c_void, self.map_len as usize);
        }
        }
        self.map = 0;
        self.map_handle = 0;
        self.fadvise_flush();
        // close the cached handle first (the field would drop after this body),
        // then the unbuffered open / fadvise purges the cache
        if let Ok(nul) = std::fs::File::open(NULL_DEVICE) {
            drop(std::mem::replace(&mut self.file, nul));
            purge_cache(&self.path, self.ple_range);
        }
    }
}

#[cfg(windows)]
type FnCreateMapping = unsafe extern "system" fn(*mut std::ffi::c_void, *mut std::ffi::c_void, u32, u32, u32, *const u16) -> *mut std::ffi::c_void;
#[cfg(windows)]
type FnMapView = unsafe extern "system" fn(*mut std::ffi::c_void, u32, u32, u32, usize) -> *mut std::ffi::c_void;

/// map the file read-only (whole length); (view, mapping handle), (0, 0) on any failure
#[cfg(windows)]
fn map_file(f: &std::fs::File) -> (usize, usize) {
    if std::env::var("CROW_MMAP").as_deref() == Ok("0") {
        return (0, 0);
    }
    use std::os::windows::io::AsRawHandle;
    unsafe {
        let Ok(lib) = libloading::Library::new("kernel32.dll") else { return (0, 0) };
        let Ok(create) = lib.get::<FnCreateMapping>(b"CreateFileMappingW\0") else { return (0, 0) };
        let Ok(view) = lib.get::<FnMapView>(b"MapViewOfFile\0") else { return (0, 0) };
        let h = create(f.as_raw_handle() as *mut _, std::ptr::null_mut(), 2 /* PAGE_READONLY */, 0, 0, std::ptr::null());
        if h.is_null() { return (0, 0) }
        let p = view(h, 4 /* FILE_MAP_READ */, 0, 0, 0);
        // the mapping handle stays with the Cnq: Drop unmaps the view and closes it
        (p as usize, h as usize)
    }
}

/// map the file read-only (whole length); (view, 0), (0, 0) on any failure.
/// MAP_SHARED is the twin of the windows PAGE_READONLY section; for a PROT_READ
/// mapping the delivered bytes are the same either way. There is no mapping
/// handle on unix - Drop munmaps `map_len` bytes, which `Cnq::open` sets to the
/// same file length this maps.
#[cfg(unix)]
fn map_file(f: &std::fs::File) -> (usize, usize) {
    if std::env::var("CROW_MMAP").as_deref() == Ok("0") {
        return (0, 0);
    }
    use std::os::unix::io::AsRawFd;
    let Ok(md) = f.metadata() else { return (0, 0) };
    let len = md.len() as usize;
    if len == 0 { return (0, 0) } // mmap(len = 0) is EINVAL, as CreateFileMapping of an empty file fails
    unsafe {
        let p = libc::mmap(std::ptr::null_mut(), len, libc::PROT_READ, libc::MAP_SHARED, f.as_raw_fd(), 0);
        if p == libc::MAP_FAILED { return (0, 0) }
        (p as usize, 0)
    }
}

/// read-only open with FILE_FLAG_SEQUENTIAL_SCAN (cache pages are not retained)
#[cfg(windows)]
pub fn open_sequential(path: &str) -> std::fs::File {
    use std::os::windows::fs::OpenOptionsExt;
    std::fs::OpenOptions::new().read(true).custom_flags(0x0800_0000 /* FILE_FLAG_SEQUENTIAL_SCAN */).open(path).unwrap()
}

/// read-only open with POSIX_FADV_SEQUENTIAL (len 0 = whole file). That is the
/// readahead half of FILE_FLAG_SEQUENTIAL_SCAN; linux does not drop pages behind
/// the cursor for this hint, so the retention half comes from the DONTNEED purge
/// in `purge_cache`.
#[cfg(unix)]
pub fn open_sequential(path: &str) -> std::fs::File {
    use std::os::unix::io::AsRawFd;
    let f = std::fs::File::open(path).unwrap();
    unsafe { libc::posix_fadvise(f.as_raw_fd(), 0, 0, libc::POSIX_FADV_SEQUENTIAL) };
    f
}

impl Cnq {
    pub fn open(path: &str) -> Cnq {
        // FILE_FLAG_SEQUENTIAL_SCAN: the cache manager drops container pages
        // behind the read cursor instead of keeping them in the system cache
        // working set (2026-09-06: 3.3 GB stayed "in use" after the engine
        // drop until process exit, and the reload's RAM check refused the tier)
        let mut f = open_sequential(path);
        let file_len = f.metadata().unwrap().len();
        f.seek(SeekFrom::Start(file_len - 8)).unwrap();
        let mut b8 = [0u8; 8];
        f.read_exact(&mut b8).unwrap();
        let idx_len = u64::from_le_bytes(b8) as usize;
        f.seek(SeekFrom::Start(file_len - 8 - idx_len as u64)).unwrap();
        let mut ib = vec![0u8; idx_len];
        f.read_exact(&mut ib).unwrap();
        let index: serde_json::Value = serde_json::from_slice(&ib).unwrap();
        let blob_offset = index["blob_offset"].as_u64().unwrap();
        let mut tensors = Vec::new();
        for t in index["tensors"].as_array().unwrap() {
            tensors.push(TensorInfo {
                name: t["name"].as_str().unwrap().to_string(),
                section: t["section"].as_str().unwrap().to_string(),
                dtype: t["dtype"].as_str().unwrap().to_string(),
                offset: t["offset"].as_u64().unwrap(),
                n_values: t["n_values"].as_u64().unwrap(),
                global_scale: t["global_scale"].as_f64().unwrap_or(1.0) as f32,
                shape: t["shape"]
                    .as_array()
                    .map(|a| a.iter().map(|v| v.as_u64().unwrap()).collect())
                    .unwrap_or_default(),
            });
        }
        let map_len = file_len;
        let (map, map_handle) = map_file(&f);
        if map == 0 {
            tracing::warn!(target: "cnq", "[cnq] file mapping unavailable - seek/read fallback");
        }
        // the `ple` section as one byte range: the 128 shard tables are laid out
        // contiguously by the converter (verified 2026-09-17 on the -M container:
        // [3_240_788_100, 32_040_949_636), 26.82 GiB, and the only non-ple tensors
        // inside it are three tables of a few KB). Drop keeps this range warm.
        let mut ple_range = (u64::MAX, 0u64);
        for t in &tensors {
            if t.section == "ple" {
                let lo = blob_offset + t.offset;
                ple_range = (ple_range.0.min(lo), ple_range.1.max(lo + Self::byte_len(t)));
            }
        }
        if ple_range.0 > ple_range.1 {
            ple_range = (0, 0);
        }
        Cnq { file: f, blob_offset, tensors, map, map_len, map_handle, path: path.to_string(), fadv: (0, 0), ple_range }
    }

    /// the `ple` section as one `[lo, hi)` byte range of the container, or
    /// `(0, 0)` when there is none
    pub fn ple_range(&self) -> (u64, u64) { self.ple_range }

    /// A `Send` handle to this container's row fetch (`CROW_PLE_FETCH`): the
    /// prefill's prefetch thread holds one while the engine goes on using the
    /// `Cnq` it came from.
    pub fn warm(&self) -> Warm {
        Warm { path: std::sync::Arc::from(self.path.as_str()), map: self.map, map_len: self.map_len }
    }

    /// absolute file offset of a tensor byte range (for prefetch touches)
    pub fn abs_offset(&self, t: &TensorInfo, rel_off: u64) -> u64 {
        self.blob_offset + t.offset + rel_off
    }

    pub fn find(&self, name: &str, section: &str) -> &TensorInfo {
        self.tensors
            .iter()
            .find(|t| t.name == name && t.section == section)
            .unwrap_or_else(|| panic!("tensor not found: {name} [{section}]"))
    }

    pub fn byte_len(t: &TensorInfo) -> u64 {
        match t.dtype.as_str() {
            "bf16" => t.n_values * 2,
            "i64" => t.n_values * 8,
            _ => (t.n_values + 63) / 64 * 36, // nvfp4
        }
    }

    /// The one container read: `len` bytes at absolute container offset `off`.
    /// Every section but `ple` goes through seek+read and hands its pages
    /// straight back via `fadvise_consumed`, so the model is never held twice
    /// during the load.
    ///
    /// `ple` is the section read one 108 B row at a time, at random offsets, on
    /// the critical path of every token, and it is read with `pread` on unix —
    /// NOT through the mapping. What the mapping costs is the COLD fault: it
    /// runs the FILE's readahead state, and `Cnq::open` sets
    /// POSIX_FADV_SEQUENTIAL on that descriptor, which doubles the window — a
    /// quarter megabyte read per 108-byte row. Measured 2026-09-17 (TASK H),
    /// 1024-token prefill, 10,192 row misses of 16,400 rows, identical bytes on
    /// every arm: cold through the mapping 11.14 s (92 tok/s) and 27.07 ms of
    /// PLE host time per decode step, cold with `pread` 2.04 s (502 tok/s) and
    /// 1.49 ms — 1.09 ms against 0.20 ms per row, 0.094 ms of which is the
    /// device. Warm the mapping is free (0.08 ms/step, 1.41 s prefill on the
    /// unchanged binary), so this is a cold-start cost, and the batch in
    /// `Warm::rows` is what makes the cold start cheap.
    ///
    /// Windows keeps the mapping: the measurement above is a Linux one and the
    /// mapping is why the row read is a page-cache memcpy there.
    fn read_at(&mut self, section: &str, off: u64, len: usize) -> Vec<u8> {
        let mut raw = vec![0u8; len];
        #[cfg(unix)]
        if section == "ple" {
            use std::os::unix::fs::FileExt;
            self.file.read_exact_at(&mut raw, off).unwrap();
            return raw;
        }
        #[cfg(windows)]
        if self.map != 0 && section == "ple" && off + len as u64 <= self.map_len {
            unsafe { std::ptr::copy_nonoverlapping((self.map as *const u8).add(off as usize), raw.as_mut_ptr(), len) };
            return raw;
        }
        self.file.seek(SeekFrom::Start(off)).unwrap();
        self.file.read_exact(&mut raw).unwrap();
        if section != "ple" {
            self.fadvise_consumed(off, len);
        }
        raw
    }

    pub fn read_bytes(&mut self, t: &TensorInfo) -> Vec<u8> {
        self.read_at(&t.section, self.blob_offset + t.offset, Self::byte_len(t) as usize)
    }

    /// read a byte range of a tensor (row-aligned reads for expert slabs, PLE rows)
    pub fn read_range(&mut self, t: &TensorInfo, rel_off: u64, len: usize) -> Vec<u8> {
        self.read_at(&t.section, self.blob_offset + t.offset + rel_off, len)
    }

    /// Drop the page-cache pages of a range this loader has consumed.
    ///
    /// The cold tier and the hot slabs are read once, memcpy'd into pinned or
    /// device memory, and never read again - but the kernel keeps every byte in
    /// the page cache, so the model is held TWICE during the load (measured
    /// 2026-09-17: the page cache grew to 33 GiB while `MemFree` fell to
    /// 1.15 GiB, and that sustained reclaim pressure is what systemd-oomd kills
    /// on). `FILE_FLAG_SEQUENTIAL_SCAN` does this on windows; linux needs the
    /// call. The `ple` section is exempt - it is designed to live in page cache.
    ///
    /// Ranges are coalesced while they stay contiguous and flushed at
    /// `FADV_BATCH`, so a sequential fill pays one syscall per 64 MiB and a
    /// random one pays one per read. Only pages inside a range already read are
    /// ever dropped, so no readahead is thrown away and no byte is re-read.
    fn fadvise_consumed(&mut self, off: u64, len: usize) {
        static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        if len == 0 || !*ON.get_or_init(|| std::env::var("CROW_CNQ_PURGE").as_deref() != Ok("0")) {
            return;
        }
        let end = off + len as u64;
        let (lo, hi) = self.fadv;
        if hi == off && hi > lo {
            self.fadv = (lo, end);
        } else {
            self.fadvise_flush();
            self.fadv = (off, end);
        }
        if self.fadv.1 - self.fadv.0 >= FADV_BATCH {
            self.fadvise_flush();
        }
    }

    /// issue the pending DONTNEED window, if any
    fn fadvise_flush(&mut self) {
        let (lo, hi) = std::mem::replace(&mut self.fadv, (0, 0));
        if hi <= lo {
            return;
        }
        #[cfg(unix)]
        {
            use std::os::unix::io::AsRawFd;
            unsafe {
                libc::posix_fadvise(self.file.as_raw_fd(), lo as libc::off_t,
                    (hi - lo) as libc::off_t, libc::POSIX_FADV_DONTNEED);
            }
        }
        #[cfg(not(unix))]
        let _ = (lo, hi); // windows: FILE_FLAG_SEQUENTIAL_SCAN already does this
    }

    pub fn read_f32(&mut self, name: &str, section: &str) -> Vec<f32> {
        let t = self.find(name, section).clone();
        let raw = self.read_bytes(&t);
        match t.dtype.as_str() {
            "bf16" => bf16_bytes_to_f32(&raw),
            _ => panic!("{name}: read_f32 expects a bf16 keep, got {}", t.dtype),
        }
    }

    pub fn read_i64(&mut self, name: &str, section: &str) -> Vec<i64> {
        let t = self.find(name, section).clone();
        assert_eq!(t.dtype, "i64", "{name}: expected i64 table");
        let raw = self.read_bytes(&t);
        raw.chunks_exact(8)
            .map(|c| i64::from_le_bytes([c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7]]))
            .collect()
    }
}

pub fn bf16_bytes_to_f32(raw: &[u8]) -> Vec<f32> {
    raw.chunks_exact(2)
        .map(|c| f32::from_bits((u16::from_le_bytes([c[0], c[1]]) as u32) << 16))
        .collect()
}

// CPU-side NVFP4 decode — the exact twin of the device decoder (p10 stage A).
pub fn e2m1(nibble: u32) -> f32 {
    const MAG: [f32; 8] = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0];
    let v = MAG[(nibble & 0x7) as usize];
    if nibble & 0x8 != 0 { -v } else { v }
}

/// magnitude index of an E2M1 level - the inverse of `e2m1`'s table, and the
/// only place either direction of that table is written down.
pub fn mag_index(m: f32) -> u32 {
    for (i, v) in [0.0f32, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0].iter().enumerate() {
        if (*v - m).abs() < 1e-6 {
            return i as u32;
        }
    }
    panic!("codebook magnitude {m} is not an e2m1 level");
}

/// Requantise ONE 16-value NVFP4 sub-block against the codebook `cb_val`:
/// search the +/-4 neighbourhood of the original UE4M3 scale byte `orig` and
/// return the (sse, scale byte, 16 codebook indices) with the smallest sum of
/// squared error. The search order - `d` ascending, strict `<` on both the
/// per-value error and the sse - decides ties and IS the container's bit
/// contract, so the offline builders (`coldtier`, `hybrid`) share this copy.
pub fn best_scale_and_codes(vals: &[f32], orig: i32, gs: f32, cb_val: &[f32]) -> (f64, u32, [u32; 16]) {
    let mut best = (f64::INFINITY, orig as u32, [0u32; 16]);
    for d in -4i32..=4 {
        let byte = orig + d;
        if byte < 1 || byte > 0x7E { continue; }
        let s = ue4m3(byte as u32) * gs;
        let mut sse = 0f64;
        let mut cd = [0u32; 16];
        for (j, &v) in vals.iter().enumerate() {
            let mut bk = 0usize;
            let mut be = f32::INFINITY;
            for (k, &c) in cb_val.iter().enumerate() {
                let err = (v - c * s).abs();
                if err < be { be = err; bk = k; }
            }
            cd[j] = bk as u32;
            sse += (be as f64) * (be as f64);
        }
        if sse < best.0 { best = (sse, byte as u32, cd); }
    }
    best
}

pub fn ue4m3(byte: u32) -> f32 {
    let e = (byte >> 3) & 0xF;
    let m = byte & 0x7;
    if e == 0 {
        (m as f32) * 1.953125e-3
    } else {
        (1.0 + (m as f32) / 8.0) * 2.0f32.powi(e as i32 - 7)
    }
}

/// dequantize a 64-value NVFP4 block (36 B) — CPU reference / self-check
pub fn dequant_block(blk: &[u8], gs: f32, out: &mut [f32; 64]) {
    for sb in 0..4 {
        let s = ue4m3(blk[sb] as u32) * gs;
        for j in 0..16 {
            let idx = sb * 16 + j;
            let byte = blk[4 + (idx >> 1)];
            let nib = if idx & 1 == 1 { (byte >> 4) & 0xF } else { byte & 0xF };
            out[idx] = e2m1(nib as u32) * s;
        }
    }
}

/// CPU FP8 E4M3 encode (round-to-nearest-even, saturating) — twin of the
/// device encoder used for the KV cache. 0x7F is the only NaN pattern.
pub fn f32_to_e4m3(v: f32) -> u8 {
    if v.is_nan() {
        return if v.is_sign_negative() { 0xFF } else { 0x7F };
    }
    let s = if v.is_sign_negative() { 0x80u8 } else { 0u8 };
    let f = v.abs();
    if f.is_infinite() {
        return s | 0x7E; // saturate to 448
    }
    if f == 0.0 {
        return s;
    }
    if f >= 464.0 {
        return s | 0x7E; // 448, satfinite boundary (448 + halfstep 16)
    }
    if f < 2.0f32.powi(-6) {
        // subnormal: step 2^-9, RNE (values < half step round to zero naturally)
        let step = 2.0f32.powi(-9);
        let m = (f / step).round_ties_even();
        if m >= 8.0 {
            // carried into the normal range: 2^-6 with m=0
            return s | (1u8 << 3);
        }
        return s | (m as u8 & 0x7);
    }
    let e_f = f.log2().floor();
    let mut e_i = e_f as i32;
    // guard against log2 edge (e.g. exact powers of two near boundaries)
    while 2.0f32.powi(e_i) > f {
        e_i -= 1;
    }
    while 2.0f32.powi(e_i + 1) <= f {
        e_i += 1;
    }
    let exp_field = e_i + 7;
    if exp_field > 15 {
        return s | 0x7E; // 448 satfinite
    }
    let step = 2.0f32.powi(e_i - 3);
    let q = (f / step).round_ties_even();
    if q >= 16.0 {
        // mantissa overflow: bump exponent
        let exp_field2 = e_i + 8;
        if exp_field2 > 15 {
            return s | 0x7E;
        }
        return s | ((exp_field2 as u8) << 3);
    }
    s | ((exp_field as u8) << 3) | (q as u8 & 0x7)
}

/// CPU FP8 E4M3 decode
pub fn e4m3_to_f32(b: u8) -> f32 {
    let s = b & 0x80 != 0;
    let e = ((b >> 3) & 0xF) as i32;
    let m = (b & 0x7) as f32;
    let v = if e == 15 && m == 7.0 {
        f32::NAN
    } else if e == 0 {
        m * 2.0f32.powi(-9)
    } else {
        (1.0 + m / 8.0) * 2.0f32.powi(e - 7)
    };
    if s { -v } else { v }
}

#[cfg(test)]
mod tests {
    use super::{page_runs, PAGE, WARM_RUN_MAX};

    /// 37 PLE rows share one 4 KiB page, and a chunk's misses arrive in id
    /// order, not in offset order: the fetch has to sort and deduplicate or it
    /// reads the same page dozens of times.
    #[test]
    fn page_runs_sorts_dedups_and_coalesces() {
        let base = 10 * PAGE;
        // out of order, three of them on the same page, one page further on
        let offsets = [base + 20000, base + 3000, base + 12, base + 3000, base + 5000, base + 108];
        let runs = page_runs(&offsets, 108, 1 << 30);
        // pages base and base+PAGE coalesce; base+4*PAGE (20000 / 4096 = 4) is
        // its own run
        assert_eq!(runs, vec![(base, 2 * PAGE as u32), (base + 4 * PAGE, PAGE as u32)]);
    }

    /// a row that straddles a page boundary needs both pages
    #[test]
    fn page_runs_covers_a_straddling_row() {
        let runs = page_runs(&[PAGE - 8], 108, 1 << 30);
        assert_eq!(runs, vec![(0, 2 * PAGE as u32)]);
    }

    /// one contiguous stretch is split at WARM_RUN_MAX so it spreads over the
    /// pool instead of piling up behind one thread, and the last run never
    /// reaches past the end of the container
    #[test]
    fn page_runs_splits_and_clamps() {
        let n = (WARM_RUN_MAX / PAGE) as usize + 2;
        let offsets: Vec<u64> = (0..n as u64).map(|i| i * PAGE).collect();
        let file_len = (n as u64 - 1) * PAGE + 100;
        let runs = page_runs(&offsets, 108, file_len);
        assert_eq!(runs.len(), 2);
        assert_eq!(runs[0], (0, WARM_RUN_MAX as u32));
        assert_eq!(runs[1].0, WARM_RUN_MAX);
        assert_eq!(runs[1].0 + runs[1].1 as u64, file_len);
    }
}
