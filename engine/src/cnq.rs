//! CNQ container reader (magic `CNQ1`, index-as-trailer). Layout and facts are
//! probe-pinned (p10/p15/p16, converter Sauberlauf): payload blob at [12..),
//! index JSON + u64 LE `index_len` in the last 8 bytes, tensor offsets relative
//! to blob start. The `section` field separates text/vit/ple/mtp — ALWAYS filter
//! by section (mtp carries its own layers.0 copy). dtype "bf16" = keep (raw LE
//! bytes), "nvfp4" = 36 B per 64 values (4 ue4m3 scales + 32 B LSB-first
//! nibbles) + one global f32 scale per tensor, rows contiguous without padding,
//! "i64" = raw metadata tables (never quantized), "f32" = raw LE f32 (Crow #300 C6: the dense
//! recipe's `A_log`).
//!
//! **Index versions (Crow #300 C6).** An index v2 (`format_version: 2`) carries a `model` block:
//! the checkpoint's `config.json` and `generation_config.json` VERBATIM (as JSON strings, each
//! with its sha256, checked here), the family, the recipe, the converter's derived geometry and
//! the source repo, revision and per-shard sha256. A v1 index (no `format_version`, `version: 1`)
//! carries no model block, and is accepted for exactly ONE container: the Flash-Next CNQ4.5-M
//! container of record, recognised by the sha256 of its index trailer
//! ([`CNQ45M_INDEX_SHA256`]). robin, 2026-09-25: the old container stays unchanged. Any other v1
//! index is refused by name ([`V1_REFUSAL`]).

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
    /// #77: this tensor's bytes live in the OVERLAY container, not in the base one.
    /// `read_bytes` / `read_range` switch file on it and nothing else has to know.
    pub overlay: bool,
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
    /// #77: the dense-BF16 overlay, when `CROW_CNQ_OVERLAY` named one. `None` is
    /// today's behaviour exactly - `find` never looks, `read_bytes` never switches.
    pub ov: Option<Overlay>,
    /// Crow #300 C6: which index this container carries - the v1 container of record, or a v2
    /// with its `model` block. Read through `model()`, `config_json()`, `family()`.
    pub index_kind: IndexKind,
}

/// sha256 of the index trailer of `Qwen3.8-Flash-Next-CNQ4.5-M.cnq`, the one CNQ index v1 the
/// engine accepts (Crow #300 C6). "Index trailer" = the `index_len` JSON bytes in front of the
/// last 8 bytes, without the length word.
///
/// Provenance: computed 2026-09-26 from the container of record,
/// `converter/Qwen3.8-Flash-Next-CNQ4.5-M.cnq` (104,727,179,972 B, 2026-09-03; the file the
/// package ships as `nibor1896/Qwen3.8-Flash-Next-CNQ4.5-M`), reading only its trailer: by
/// `Cnq::open` itself (the `#[ignore]` test `tests_300_c6::the_container_of_record_opens_as_v1`)
/// and independently with `tail -c 473536 | head -c 473528 | sha256sum`. Both gave this value.
pub const CNQ45M_INDEX_SHA256: &str = "a21afc43203d983e46be5dd378ac5fb34581ab31fb5a0f6e8093d4730078c4ba";

/// the byte length of that trailer, for the refusal text and the test
pub const CNQ45M_INDEX_LEN: u64 = 473_528;

/// The refusal of any CNQ index v1 other than the container of record. The words are the
/// contract; the message that carries them adds the two hashes.
pub const V1_REFUSAL: &str = "CNQ index v1 is only accepted for the Flash-Next CNQ4.5-M container of record";

/// The `model` block of an index v2 (Crow #300 C6), as the converter wrote it.
#[derive(Clone, Debug, PartialEq)]
pub struct ModelBlock {
    /// the engine's family name (`meta::Family` `Debug`): `FlashNext`, `Qwen35Dense`
    pub family: String,
    /// `text_config.model_type` of the config
    pub model_type: String,
    /// the converter's recipe row (`cnq4.5-flash-next`, `cnq4.5-qwen35-dense`); from the index top level
    pub recipe: String,
    /// `config.json`, byte for byte as the converter read it
    pub config_json: String,
    /// `generation_config.json`, byte for byte
    pub generation_config_json: String,
    pub source_repo: String,
    pub source_revision: String,
    /// the converter's derived geometry - informational; the engine derives its own `Geo`
    /// from `config_json` (`meta.rs`)
    pub geo: serde_json::Value,
    /// the source shards as recorded (`file`, `size`, `sha256`, `sha256_from`)
    pub shards: serde_json::Value,
}

/// Which index a container carries (Crow #300 C6).
#[derive(Clone, Debug, PartialEq)]
pub enum IndexKind {
    /// the Flash-Next CNQ4.5-M container of record: index v1, no model block
    V1OfRecord,
    V2(Box<ModelBlock>),
}

/// What `Cnq::peek_index` reads without mapping the container (Crow #300 C7): the index kind
/// and the tensor table.
#[derive(Clone)]
pub struct IndexPeek {
    pub kind: IndexKind,
    pub tensors: Vec<TensorInfo>,
}

impl IndexPeek {
    /// the index v2 `model` block; `None` for the v1 container of record
    pub fn model(&self) -> Option<&ModelBlock> {
        match &self.kind {
            IndexKind::V2(m) => Some(m),
            IndexKind::V1OfRecord => None,
        }
    }

    /// the family the container was converted as (`Cnq::family`'s rule)
    pub fn family(&self) -> &str {
        self.model().map(|m| m.family.as_str()).unwrap_or("FlashNext")
    }
}

/// The index trailer of an open CNQ1 file: `(file length, the raw index bytes, the parsed
/// index)`. Checks the magic and that the recorded index length fits the file. One reader for
/// `Cnq::open_checked` and `Cnq::peek_index`, so the two cannot classify differently.
fn read_index_trailer(f: &mut std::fs::File) -> Result<(u64, Vec<u8>, serde_json::Value), String> {
    let file_len = f.metadata().map_err(|e| e.to_string())?.len();
    if file_len < 20 {
        return Err(format!("{file_len} B is too small to be a CNQ1 container"));
    }
    let mut magic = [0u8; 4];
    f.seek(SeekFrom::Start(0)).map_err(|e| e.to_string())?;
    f.read_exact(&mut magic).map_err(|e| e.to_string())?;
    if &magic != b"CNQ1" {
        return Err(format!("magic is {magic:?}, not CNQ1"));
    }
    f.seek(SeekFrom::Start(file_len - 8)).map_err(|e| e.to_string())?;
    let mut b8 = [0u8; 8];
    f.read_exact(&mut b8).map_err(|e| e.to_string())?;
    let idx_len = u64::from_le_bytes(b8) as usize;
    if idx_len == 0 || idx_len as u64 + 20 > file_len {
        return Err(format!("index length {idx_len} does not fit a {file_len} B file"));
    }
    f.seek(SeekFrom::Start(file_len - 8 - idx_len as u64)).map_err(|e| e.to_string())?;
    let mut ib = vec![0u8; idx_len];
    f.read_exact(&mut ib).map_err(|e| e.to_string())?;
    let index: serde_json::Value = serde_json::from_slice(&ib).map_err(|e| format!("index json: {e}"))?;
    Ok((file_len, ib, index))
}

/// hex sha256
pub fn sha256_hex(bytes: &[u8]) -> String {
    use sha2::Digest;
    sha2::Sha256::digest(bytes).iter().map(|b| format!("{b:02x}")).collect()
}

/// Classify an index trailer: its raw bytes (for the v1 hash) and its parsed JSON.
pub fn classify_index(index_bytes: &[u8], index: &serde_json::Value) -> Result<IndexKind, String> {
    classify_index_against(index_bytes, index, CNQ45M_INDEX_SHA256)
}

/// `classify_index` with the v1 hash of record as a parameter, so the accepting v1 path can be
/// tested on a synthetic trailer (the real one is 473,528 B inside a 104.7 GB file).
pub fn classify_index_against(index_bytes: &[u8], index: &serde_json::Value, v1_of_record: &str) -> Result<IndexKind, String> {
    match index.get("format_version") {
        None => {
            // v1: `version: 1` (or, before the converter wrote one, nothing)
            if let Some(v) = index.get("version") {
                if v.as_u64() != Some(1) {
                    return Err(format!("CNQ index `version` {v} without a `format_version` - not an index this engine reads"));
                }
            }
            let got = sha256_hex(index_bytes);
            if got == v1_of_record {
                Ok(IndexKind::V1OfRecord)
            } else {
                Err(format!(
                    "{V1_REFUSAL} (index sha256 {got}, {} B; of record {v1_of_record}, {CNQ45M_INDEX_LEN} B) - \
re-convert the checkpoint: the converter writes index v2, which carries the model's config",
                    index_bytes.len()
                ))
            }
        }
        Some(v) if v.as_u64() == Some(2) => parse_model_block(index).map(|m| IndexKind::V2(Box::new(m))),
        Some(v) => Err(format!("CNQ index format_version {v} is not one this engine reads (it reads 2, and v1 only for the container of record)")),
    }
}

/// The index v2 `model` block, every field required, and both config strings checked against
/// the sha256 the converter recorded beside them.
fn parse_model_block(index: &serde_json::Value) -> Result<ModelBlock, String> {
    let m = &index["model"];
    if !m.is_object() {
        return Err("CNQ index v2 without a `model` block".into());
    }
    let s = |v: &serde_json::Value, what: &str| -> Result<String, String> {
        v.as_str().map(str::to_string).ok_or_else(|| format!("CNQ index v2: `{what}` missing or not a string"))
    };
    let config_json = s(&m["config_json"], "model.config_json")?;
    let generation_config_json = s(&m["generation_config_json"], "model.generation_config_json")?;
    for (what, text, sha) in [
        ("config_json", &config_json, &m["config_json_sha256"]),
        ("generation_config_json", &generation_config_json, &m["generation_config_json_sha256"]),
    ] {
        let want = s(sha, &format!("model.{what}_sha256"))?;
        let got = sha256_hex(text.as_bytes());
        if got != want {
            return Err(format!("CNQ index v2: model.{what} hashes to {got}, the index records {want} - the stored config was altered"));
        }
    }
    Ok(ModelBlock {
        family: s(&m["family"], "model.family")?,
        model_type: s(&m["model_type"], "model.model_type")?,
        recipe: s(&index["recipe"], "recipe")?,
        config_json,
        generation_config_json,
        source_repo: s(&m["source"]["repo"], "model.source.repo")?,
        source_revision: s(&m["source"]["revision"], "model.source.revision")?,
        geo: m["geo"].clone(),
        shards: m["source"]["shards"].clone(),
    })
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

/// The `tensors` array of a CNQ1 index trailer, as `TensorInfo`. One parse for the base
/// container and for the #77 overlay - the overlay IS a CNQ1 container, so it must be read
/// by the same code or the two could drift apart.
pub fn parse_tensor_index(index: &serde_json::Value, overlay: bool) -> Vec<TensorInfo> {
    let mut tensors = Vec::new();
    for t in index["tensors"].as_array().into_iter().flatten() {
        tensors.push(TensorInfo {
            name: t["name"].as_str().unwrap().to_string(),
            section: t["section"].as_str().unwrap().to_string(),
            dtype: t["dtype"].as_str().unwrap().to_string(),
            offset: t["offset"].as_u64().unwrap(),
            n_values: t["n_values"].as_u64().unwrap(),
            global_scale: t["global_scale"].as_f64().unwrap_or(1.0) as f32,
            shape: t["shape"].as_array().map(|a| a.iter().map(|v| v.as_u64().unwrap()).collect()).unwrap_or_default(),
            overlay,
        });
    }
    tensors
}

/// #77: the dense-BF16 overlay opened beside the base container. A tensor named here shadows
/// the base tensor of the same name and section; nothing else about the base changes, and
/// without `CROW_CNQ_OVERLAY` this is `None` and not one byte of behaviour moves.
pub struct Overlay {
    pub path: String,
    pub file: std::fs::File,
    pub blob_offset: u64,
    pub tensors: Vec<TensorInfo>,
    /// the `overlay` block of the index trailer: base name and size, source, kinds, date
    pub header: serde_json::Value,
}

/// What `Cnq::attach_overlay` found, for the boot log: the shadowed tensors per kind and the
/// bytes they will hold on the card.
pub struct OverlayReport {
    pub path: String,
    pub source: String,
    pub built: String,
    /// the `overlay.kind` of the trailer: `dense-bf16` (#77) or `expert-nvfp4` (#79)
    pub kind: String,
    pub tensors: usize,
    pub values: u64,
    pub bytes: u64,
    /// the bytes the BASE container holds for the same tensors, so the boot line can say what
    /// the overlay costs - or, for an nvfp4 expert overlay, that it costs nothing
    pub base_bytes: u64,
    /// (kind, tensors, values), kind-sorted
    pub per_kind: Vec<(String, usize, u64)>,
}

/// The KIND of a dense tensor: the name without the `…layers.<N>.` prefix and without
/// `.weight`. The twin of `dense_overlay::kind_of` in the converter (#77).
pub fn kind_of(name: &str) -> String {
    let mut s = name;
    if let Some(rest) = s.strip_prefix("model.language_model.layers.") {
        if let Some(dot) = rest.find('.') {
            if rest[..dot].bytes().all(|b| b.is_ascii_digit()) {
                s = &rest[dot + 1..];
            }
        }
    }
    s.strip_suffix(".weight").unwrap_or(s).to_string()
}

/// Every way an overlay can fail to belong to this base container, as ONE named error each.
/// The point of the list is that none of them may reach a kernel: a shape that does not match
/// would be a silent wrong-size GEMV, an unknown name a weight nobody ever reads.
pub fn overlay_refusal(
    ov_index: &serde_json::Value,
    ov_tensors: &[TensorInfo],
    base_name: &str,
    base_bytes: u64,
    base: &[TensorInfo],
) -> Option<String> {
    let h = &ov_index["overlay"];
    if !h.is_object() {
        return Some("the index trailer carries no `overlay` block - this is not an overlay container".into());
    }
    let want_name = h["base_name"].as_str().unwrap_or("");
    if want_name != base_name {
        return Some(format!("built over base `{want_name}`, but this engine opened `{base_name}`"));
    }
    let want_bytes = h["base_bytes"].as_u64().unwrap_or(0);
    if want_bytes != base_bytes {
        return Some(format!(
            "built over a base of {want_bytes} B, but `{base_name}` is {base_bytes} B - the base container changed"
        ));
    }
    if ov_tensors.is_empty() {
        return Some("carries no tensors".into());
    }
    for t in ov_tensors {
        let Some(b) = base.iter().find(|b| b.name == t.name && b.section == t.section) else {
            return Some(format!("{} [{}]: not in the base container", t.name, t.section));
        };
        if b.n_values != t.n_values {
            return Some(format!(
                "{}: {} values in the overlay, {} in the base container",
                t.name, t.n_values, b.n_values
            ));
        }
        if !t.shape.is_empty() && !b.shape.is_empty() && t.shape != b.shape {
            return Some(format!("{}: shape {:?} in the overlay, {:?} in the base container", t.name, t.shape, b.shape));
        }
        match t.dtype.as_str() {
            // #77: a DENSE tensor carried at full precision. Its byte length changes (2 B per
            // value against 36 B per 64) and every reader takes that from `byte_len`. A routed
            // expert may NOT come this way: `residency` cuts per-expert slabs out of
            // `byte_len / 512` and hands them to kernels that index 36 B per 64 values, so a
            // bf16 expert would be a silently wrong-size slab, not a refusal.
            "bf16" => {
                if t.name.contains(".mlp.experts.") {
                    return Some(format!(
                        "{}: bf16 - a routed expert may only be shadowed as nvfp4 of the same byte length, \
the expert slabs and the kernels are cut from it",
                        t.name
                    ));
                }
            }
            // #79: routed experts re-quantized onto the SAME grid. Here not one byte of length
            // may move: the residency planner, the hot VRAM slabs, the pinned cold tier and
            // `gs_dev` are all sized from it, and the kernels are the base container's kernels.
            "nvfp4" => {
                if b.dtype != "nvfp4" {
                    return Some(format!("{}: nvfp4 in the overlay but {} in the base container", t.name, b.dtype));
                }
                let (have, want) = (Cnq::byte_len(t), Cnq::byte_len(b));
                if have != want {
                    return Some(format!("{}: {have} B in the overlay, {want} B in the base container", t.name));
                }
                if !(t.global_scale.is_finite() && t.global_scale > 0.0) {
                    return Some(format!(
                        "{}: global scale {} - an nvfp4 overlay carries its own and every block is read through it",
                        t.name, t.global_scale
                    ));
                }
            }
            other => {
                return Some(format!(
                    "{}: dtype {other} - an overlay stores bf16 (the dense path, #77) or nvfp4 (the routed experts, #79)",
                    t.name
                ))
            }
        }
    }
    None
}

impl Cnq {
    /// Open an overlay container beside this one and let its tensors shadow the base tensors
    /// of the same name and section (#77, `CROW_CNQ_OVERLAY`). Returns the refusal as an
    /// `Err(String)` - the caller names it at boot, so no mismatch can reach a kernel.
    pub fn attach_overlay(&mut self, path: &str) -> Result<OverlayReport, String> {
        let mut f = std::fs::File::open(path).map_err(|e| format!("{path}: {e}"))?;
        let file_len = f.metadata().map_err(|e| format!("{path}: {e}"))?.len();
        if file_len < 20 {
            return Err(format!("{path}: {file_len} B is too small to be a CNQ1 container"));
        }
        let mut magic = [0u8; 4];
        f.read_exact(&mut magic).map_err(|e| format!("{path}: {e}"))?;
        if &magic != b"CNQ1" {
            return Err(format!("{path}: magic is {magic:?}, not CNQ1"));
        }
        f.seek(SeekFrom::Start(file_len - 8)).map_err(|e| format!("{path}: {e}"))?;
        let mut b8 = [0u8; 8];
        f.read_exact(&mut b8).map_err(|e| format!("{path}: {e}"))?;
        let idx_len = u64::from_le_bytes(b8);
        if idx_len == 0 || idx_len + 8 > file_len {
            return Err(format!("{path}: index length {idx_len} does not fit a {file_len} B file"));
        }
        f.seek(SeekFrom::Start(file_len - 8 - idx_len)).map_err(|e| format!("{path}: {e}"))?;
        let mut ib = vec![0u8; idx_len as usize];
        f.read_exact(&mut ib).map_err(|e| format!("{path}: {e}"))?;
        let index: serde_json::Value = serde_json::from_slice(&ib).map_err(|e| format!("{path}: index json: {e}"))?;
        let blob_offset = index["blob_offset"].as_u64().unwrap_or(12);
        let tensors = parse_tensor_index(&index, true);
        let base_name = std::path::Path::new(&self.path)
            .file_name()
            .map(|s| s.to_string_lossy().to_string())
            .unwrap_or_default();
        if let Some(why) = overlay_refusal(&index, &tensors, &base_name, self.map_len, &self.tensors) {
            return Err(format!("{path}: {why}"));
        }
        // the payload has to be inside the file, or a read would run off the end
        for t in &tensors {
            let end = blob_offset + t.offset + Self::byte_len(t);
            if end > file_len - 8 - idx_len {
                return Err(format!("{path}: {} runs to byte {end}, past the payload", t.name));
            }
        }
        let mut per_kind: std::collections::BTreeMap<String, (usize, u64)> = std::collections::BTreeMap::new();
        let mut values = 0u64;
        let mut bytes = 0u64;
        let mut base_bytes = 0u64;
        for t in &tensors {
            let e = per_kind.entry(kind_of(&t.name)).or_insert((0, 0));
            e.0 += 1;
            e.1 += t.n_values;
            values += t.n_values;
            bytes += Self::byte_len(t);
            if let Some(b) = self.tensors.iter().find(|b| b.name == t.name && b.section == t.section) {
                base_bytes += Self::byte_len(b);
            }
        }
        let report = OverlayReport {
            path: path.to_string(),
            source: index["overlay"]["source"].as_str().unwrap_or("?").to_string(),
            built: index["overlay"]["built"].as_str().unwrap_or("?").to_string(),
            kind: index["overlay"]["kind"].as_str().unwrap_or("?").to_string(),
            tensors: tensors.len(),
            values,
            bytes,
            base_bytes,
            per_kind: per_kind.into_iter().map(|(k, (c, n))| (k, c, n)).collect(),
        };
        #[cfg(unix)]
        {
            use std::os::unix::io::AsRawFd;
            unsafe { libc::posix_fadvise(f.as_raw_fd(), 0, 0, libc::POSIX_FADV_SEQUENTIAL) };
        }
        self.ov = Some(Overlay { path: path.to_string(), file: f, blob_offset, tensors, header: index["overlay"].clone() });
        Ok(report)
    }

    /// the overlay's own tensor list, or an empty slice when none is attached
    pub fn overlay_tensors(&self) -> &[TensorInfo] {
        self.ov.as_ref().map(|o| o.tensors.as_slice()).unwrap_or(&[])
    }

    /// Open the container, or die naming why (Crow #300 C6: an index v1 that is not the
    /// container of record, an unknown `format_version`, an altered model block).
    pub fn open(path: &str) -> Cnq {
        Cnq::open_checked(path).unwrap_or_else(|why| panic!("[cnq] refused: {path}: {why}"))
    }

    /// `open` with the refusal as an `Err`. The reads and the mapping are `open`'s of `64c242b`;
    /// what C6 adds is the index classification before anything is mapped.
    pub fn open_checked(path: &str) -> Result<Cnq, String> {
        // FILE_FLAG_SEQUENTIAL_SCAN: the cache manager drops container pages
        // behind the read cursor instead of keeping them in the system cache
        // working set (2026-09-06: 3.3 GB stayed "in use" after the engine
        // drop until process exit, and the reload's RAM check refused the tier)
        let mut f = open_sequential(path);
        let (file_len, ib, index) = read_index_trailer(&mut f)?;
        let index_kind = classify_index(&ib, &index)?;
        let blob_offset = index["blob_offset"].as_u64().ok_or("index has no blob_offset")?;
        let tensors = parse_tensor_index(&index, false);
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
        Ok(Cnq { file: f, blob_offset, tensors, map, map_len, map_handle, path: path.to_string(), fadv: (0, 0), ple_range, ov: None, index_kind })
    }

    /// Crow #300 C7: the index trailer alone - classified exactly as `open_checked` classifies
    /// it, plus the tensor table - without mapping the file. The boot door reads it FIRST: an
    /// index v2 carries the config the metadata gate judges, and the tensor table is what the
    /// gate holds that config against (`meta::container_mismatch`), all before `Cnq::open` maps
    /// the container and before any CUDA call. The trailer is read twice per boot (here and in
    /// `open`); on the container of record that is 473,528 B.
    pub fn peek_index(path: &str) -> Result<IndexPeek, String> {
        let mut f = std::fs::File::open(path).map_err(|e| e.to_string())?;
        let (_, ib, index) = read_index_trailer(&mut f)?;
        let kind = classify_index(&ib, &index)?;
        Ok(IndexPeek { kind, tensors: parse_tensor_index(&index, false) })
    }

    /// The index v2 `model` block; `None` for the v1 container of record.
    pub fn model(&self) -> Option<&ModelBlock> {
        match &self.index_kind {
            IndexKind::V2(m) => Some(m),
            IndexKind::V1OfRecord => None,
        }
    }

    /// The checkpoint's `config.json`, byte for byte, from an index v2. `None` for the v1
    /// container of record, which carries none: its boot keeps reading `CROW_MODEL_DIR`.
    pub fn config_json(&self) -> Option<&str> {
        self.model().map(|m| m.config_json.as_str())
    }

    /// The checkpoint's `generation_config.json`, byte for byte, from an index v2.
    pub fn generation_config_json(&self) -> Option<&str> {
        self.model().map(|m| m.generation_config_json.as_str())
    }

    /// The model family the container was converted as: the v2 `model.family`, or `FlashNext`
    /// for the v1 container of record (the only v1 container `open` accepts).
    pub fn family(&self) -> &str {
        self.model().map(|m| m.family.as_str()).unwrap_or("FlashNext")
    }

    /// The index version: 1 (the container of record) or 2.
    pub fn index_version(&self) -> u32 {
        match self.index_kind {
            IndexKind::V1OfRecord => 1,
            IndexKind::V2(_) => 2,
        }
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

    /// absolute file offset of a tensor byte range (for prefetch touches). Base container
    /// only: the prefetch is the PLE row path and #77 never shadows a `ple`-section table.
    pub fn abs_offset(&self, t: &TensorInfo, rel_off: u64) -> u64 {
        debug_assert!(!t.overlay, "{}: abs_offset is a base-container offset", t.name);
        self.blob_offset + t.offset + rel_off
    }

    /// The tensor of record for `(name, section)`. #77: an attached overlay is asked FIRST,
    /// so a tensor it carries shadows the base one for every reader in the engine - there is
    /// no second lookup anywhere and no loader has to know an overlay exists.
    pub fn find(&self, name: &str, section: &str) -> &TensorInfo {
        if let Some(ov) = &self.ov {
            if let Some(t) = ov.tensors.iter().find(|t| t.name == name && t.section == section) {
                return t;
            }
        }
        self.tensors
            .iter()
            .find(|t| t.name == name && t.section == section)
            .unwrap_or_else(|| panic!("tensor not found: {name} [{section}]"))
    }

    /// the BASE tensor, even when the overlay shadows it (the boot log's before/after)
    pub fn find_base(&self, name: &str, section: &str) -> &TensorInfo {
        self.tensors
            .iter()
            .find(|t| t.name == name && t.section == section)
            .unwrap_or_else(|| panic!("tensor not found: {name} [{section}]"))
    }

    pub fn byte_len(t: &TensorInfo) -> u64 {
        match t.dtype.as_str() {
            "bf16" => t.n_values * 2,
            "i64" => t.n_values * 8,
            "f32" => t.n_values * 4, // Crow #300 C6: the dense recipe's A_log
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
        if t.overlay {
            return self.read_overlay(t.offset, Self::byte_len(t) as usize);
        }
        self.read_at(&t.section, self.blob_offset + t.offset, Self::byte_len(t) as usize)
    }

    /// read a byte range of a tensor (row-aligned reads for expert slabs, PLE rows)
    pub fn read_range(&mut self, t: &TensorInfo, rel_off: u64, len: usize) -> Vec<u8> {
        if t.overlay {
            return self.read_overlay(t.offset + rel_off, len);
        }
        self.read_at(&t.section, self.blob_offset + t.offset + rel_off, len)
    }

    /// #77: the one read that goes to the overlay file. `rel_off` is relative to the
    /// overlay's blob start. Every byte is read once at load and never again, so the pages
    /// are handed straight back - the same DONTNEED discipline `fadvise_consumed` keeps for
    /// the base container, and for the same reason: 5.2 GB held twice during the load is
    /// exactly the reclaim pressure issue #15 was about.
    fn read_overlay(&mut self, rel_off: u64, len: usize) -> Vec<u8> {
        let ov = self.ov.as_mut().expect("overlay tensor without an attached overlay");
        let off = ov.blob_offset + rel_off;
        let mut raw = vec![0u8; len];
        ov.file.seek(SeekFrom::Start(off)).unwrap();
        ov.file.read_exact(&mut raw).unwrap();
        #[cfg(unix)]
        if std::env::var("CROW_CNQ_PURGE").as_deref() != Ok("0") {
            use std::os::unix::io::AsRawFd;
            unsafe {
                libc::posix_fadvise(ov.file.as_raw_fd(), off as libc::off_t, len as libc::off_t, libc::POSIX_FADV_DONTNEED);
            }
        }
        raw
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
            "f32" => f32_bytes_to_f32(&raw),
            _ => panic!("{name}: read_f32 expects a bf16 or f32 keep, got {}", t.dtype),
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

/// raw little-endian f32 (the `f32` dtype, Crow #300 C6)
pub fn f32_bytes_to_f32(raw: &[u8]) -> Vec<f32> {
    let (words, _) = raw.as_chunks::<4>();
    words.iter().map(|w| f32::from_le_bytes(*w)).collect()
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
    use super::{kind_of, overlay_refusal, page_runs, TensorInfo, PAGE, WARM_RUN_MAX};

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
    // ---- #77: the dense-BF16 overlay, all of it host logic with no GPU and no container ----

    fn ti(name: &str, section: &str, dtype: &str, n: u64, shape: &[u64], overlay: bool) -> TensorInfo {
        TensorInfo {
            name: name.into(),
            section: section.into(),
            dtype: dtype.into(),
            offset: 0,
            n_values: n,
            global_scale: 1.0,
            shape: shape.to_vec(),
            overlay,
        }
    }

    fn head(base_name: &str, base_bytes: u64) -> serde_json::Value {
        serde_json::json!({ "overlay": { "base_name": base_name, "base_bytes": base_bytes } })
    }

    /// The kind is what `--kinds` names and what the boot log counts per line. It has to be
    /// the SAME rule the converter applies, or an ablation would select nothing and read as
    /// "the originals change nothing".
    #[test]
    fn the_overlay_kind_is_the_name_without_the_layer_prefix_and_the_weight_suffix() {
        assert_eq!(kind_of("model.language_model.layers.7.linear_attn.in_proj_qkv.weight"), "linear_attn.in_proj_qkv");
        assert_eq!(kind_of("model.language_model.layers.0.self_attn.indexer.index_qk_proj.weight"), "self_attn.indexer.index_qk_proj");
        assert_eq!(kind_of("model.language_model.layers.1.ple.key_proj.weight"), "ple.key_proj");
        assert_eq!(kind_of("model.language_model.layers.47.mlp_hyper_connection.block_inject_weight.weight"), "mlp_hyper_connection.block_inject_weight");
        // not a per-layer name: nothing is stripped but the suffix
        assert_eq!(kind_of("model.language_model.layers.x.foo.weight"), "model.language_model.layers.x.foo");
    }

    /// An overlay that matches its base is accepted, and every way it can FAIL to match is
    /// one named refusal. This is the list that keeps a mismatch out of a kernel: a wrong
    /// shape would be a silently wrong-size GEMV, an unknown name a weight nobody reads.
    #[test]
    fn an_overlay_that_does_not_belong_to_this_base_is_refused_by_name() {
        let base = vec![
            ti("a.weight", "text", "nvfp4", 128, &[2, 64], false),
            ti("b.weight", "text", "nvfp4", 64, &[1, 64], false),
        ];
        let ov = vec![ti("a.weight", "text", "bf16", 128, &[2, 64], true)];
        // the accepting case first, so the refusals below are not vacuous
        assert!(overlay_refusal(&head("base.cnq", 999), &ov, "base.cnq", 999, &base).is_none());

        let no_block = serde_json::json!({ "format": "crow-nest-quant" });
        let m = overlay_refusal(&no_block, &ov, "base.cnq", 999, &base).unwrap();
        assert!(m.contains("no `overlay` block"), "{m}");

        let m = overlay_refusal(&head("other.cnq", 999), &ov, "base.cnq", 999, &base).unwrap();
        assert!(m.contains("other.cnq") && m.contains("base.cnq"), "{m}");

        let m = overlay_refusal(&head("base.cnq", 1000), &ov, "base.cnq", 999, &base).unwrap();
        assert!(m.contains("the base container changed"), "{m}");

        let m = overlay_refusal(&head("base.cnq", 999), &[], "base.cnq", 999, &base).unwrap();
        assert!(m.contains("no tensors"), "{m}");

        // #79 widened the dtype rule from "bf16 only" to "bf16 or nvfp4", so the refusal is
        // now a dtype that is NEITHER
        let wrong_dtype = vec![ti("a.weight", "text", "f32", 128, &[2, 64], true)];
        let m = overlay_refusal(&head("base.cnq", 999), &wrong_dtype, "base.cnq", 999, &base).unwrap();
        assert!(m.contains("dtype f32"), "{m}");

        let unknown = vec![ti("zzz.weight", "text", "bf16", 128, &[2, 64], true)];
        let m = overlay_refusal(&head("base.cnq", 999), &unknown, "base.cnq", 999, &base).unwrap();
        assert!(m.contains("not in the base container"), "{m}");

        // the same name in a DIFFERENT section is a different tensor (mtp carries its own
        // layers.0 copy - the reason every reader filters by section)
        let wrong_section = vec![ti("a.weight", "mtp", "bf16", 128, &[2, 64], true)];
        let m = overlay_refusal(&head("base.cnq", 999), &wrong_section, "base.cnq", 999, &base).unwrap();
        assert!(m.contains("not in the base container"), "{m}");

        let wrong_n = vec![ti("a.weight", "text", "bf16", 64, &[1, 64], true)];
        let m = overlay_refusal(&head("base.cnq", 999), &wrong_n, "base.cnq", 999, &base).unwrap();
        assert!(m.contains("64 values in the overlay, 128 in the base container"), "{m}");

        let wrong_shape = vec![ti("a.weight", "text", "bf16", 128, &[64, 2], true)];
        let m = overlay_refusal(&head("base.cnq", 999), &wrong_shape, "base.cnq", 999, &base).unwrap();
        assert!(m.contains("shape"), "{m}");
    }

    /// The overlay is bf16, the base is nvfp4, and `byte_len` is what every read length and
    /// the VRAM figure are derived from. 2 B per value against 36 B per 64 - the +3.7 GB the
    /// residency planner has to see.
    #[test]
    fn a_shadowed_tensor_is_two_bytes_per_value_where_the_base_is_four_and_a_half_bits() {
        let base = ti("a.weight", "text", "nvfp4", 64 * 100, &[100, 64], false);
        let ov = ti("a.weight", "text", "bf16", 64 * 100, &[100, 64], true);
        assert_eq!(super::Cnq::byte_len(&base), 100 * 36);
        assert_eq!(super::Cnq::byte_len(&ov), 64 * 100 * 2);
        // 3.555x the bytes, which is 16 / 4.5
        assert_eq!(super::Cnq::byte_len(&ov) * 45, super::Cnq::byte_len(&base) * 160);
    }

    // ---- #79: the routed-expert overlay, nvfp4 over nvfp4 ----

    /// An expert overlay carries the SAME format at the SAME byte length, so the accepting
    /// case is the one where nothing about the geometry moved - and every refusal below is a
    /// way that could stop being true without anybody noticing until a kernel read garbage.
    #[test]
    fn an_expert_overlay_is_accepted_as_nvfp4_of_the_same_byte_length_and_refused_otherwise() {
        let name = "model.language_model.layers.7.mlp.experts.gate_up_proj";
        let mut base_t = ti(name, "text", "nvfp4", 64 * 100, &[2, 50, 64], false);
        base_t.global_scale = 2.5e-4;
        let base = vec![base_t, ti("d.weight", "text", "nvfp4", 64 * 10, &[10, 64], false)];
        let mut ov_t = ti(name, "text", "nvfp4", 64 * 100, &[2, 50, 64], true);
        ov_t.global_scale = 3.1e-4; // its own, and different: that is the point of the overlay
        let ov = vec![ov_t];
        assert!(overlay_refusal(&head("base.cnq", 999), &ov, "base.cnq", 999, &base).is_none());
        assert_eq!(super::Cnq::byte_len(&ov[0]), super::Cnq::byte_len(&base[0]));

        // a routed expert as bf16 would be a slab of 2 B per value handed to a kernel that
        // indexes 36 B per 64 - the residency planner cuts `byte_len / 512` out of it
        let as_bf16 = vec![ti(name, "text", "bf16", 64 * 100, &[2, 50, 64], true)];
        let m = overlay_refusal(&head("base.cnq", 999), &as_bf16, "base.cnq", 999, &base).unwrap();
        assert!(m.contains("only be shadowed as nvfp4"), "{m}");

        // an nvfp4 overlay over a bf16 base tensor: the base's own keeps are raw bytes
        let bf16_base = vec![ti("k.weight", "text", "bf16", 64, &[1, 64], false)];
        let over_keep = vec![ti("k.weight", "text", "nvfp4", 64, &[1, 64], true)];
        let m = overlay_refusal(&head("base.cnq", 999), &over_keep, "base.cnq", 999, &bf16_base).unwrap();
        assert!(m.contains("but bf16 in the base container"), "{m}");

        // a global scale of zero or a NaN would make every block of the tensor zero or NaN,
        // and nothing downstream looks at it again
        for bad in [0.0f32, -1.0, f32::NAN] {
            let mut t = ti(name, "text", "nvfp4", 64 * 100, &[2, 50, 64], true);
            t.global_scale = bad;
            let m = overlay_refusal(&head("base.cnq", 999), &[t], "base.cnq", 999, &base).unwrap();
            assert!(m.contains("global scale"), "{bad}: {m}");
        }
    }
}

/// Crow #300 C6: the index v2 reader and the v1-of-record rule.
#[cfg(test)]
mod tests_300_c6 {
    use super::{classify_index, classify_index_against, sha256_hex, Cnq, IndexKind, CNQ45M_INDEX_LEN, CNQ45M_INDEX_SHA256, V1_REFUSAL};

    fn fixture(f: &str) -> String {
        format!("{}/tests/fixtures/synthetic-v2/{f}", env!("CARGO_MANIFEST_DIR"))
    }

    /// The round trip across the two crates: `synthetic-v2.cnq` is what the converter writes
    /// for the synthetic dense model (its test `a_synthetic_v2_container_carries_its_config_
    /// verbatim_and_is_the_engine_fixture` proves it byte for byte), and this reader hands back
    /// both config files byte-identical to the files the converter read.
    #[test]
    fn a_v2_container_hands_back_its_config_files_byte_for_byte() {
        let cnq = Cnq::open_checked(&fixture("synthetic-v2.cnq")).unwrap();
        assert_eq!(cnq.index_version(), 2);
        assert_eq!(cnq.config_json().unwrap().as_bytes(), std::fs::read(fixture("config.json")).unwrap().as_slice());
        assert_eq!(cnq.generation_config_json().unwrap().as_bytes(), std::fs::read(fixture("generation_config.json")).unwrap().as_slice());
        assert_eq!(cnq.family(), "Qwen35Dense");
        let m = cnq.model().unwrap();
        assert_eq!((m.model_type.as_str(), m.recipe.as_str()), ("qwen3_5_text", "cnq4.5-qwen35-dense"));
        assert_eq!((m.source_repo.as_str(), m.source_revision.as_str()), ("crow-nest/synthetic-v2", "c6"));
        assert_eq!(m.geo["hidden"], 64);
        assert_eq!(m.shards.as_array().unwrap().len(), 1);
        // the config parses as the JSON meta.rs will read
        let cfg: serde_json::Value = serde_json::from_str(cnq.config_json().unwrap()).unwrap();
        assert_eq!(cfg["text_config"]["model_type"], "qwen3_5_text");
    }

    /// The f32 dtype (the dense recipe's A_log) is 4 B per value and `read_f32` reads it; every
    /// tensor of the fixture lies inside the payload.
    #[test]
    fn the_f32_dtype_is_four_bytes_per_value_and_read_f32_reads_it() {
        let mut cnq = Cnq::open_checked(&fixture("synthetic-v2.cnq")).unwrap();
        let t = cnq.find("model.language_model.layers.0.linear_attn.A_log", "text").clone();
        assert_eq!((t.dtype.as_str(), Cnq::byte_len(&t)), ("f32", 8));
        let v = cnq.read_f32("model.language_model.layers.0.linear_attn.A_log", "text");
        assert_eq!(v.len(), 2);
        // f32 widened from bf16: the low 16 bits are zero, and the value is in the LCG's range
        assert!(v.iter().all(|x| x.to_bits() & 0xFFFF == 0 && x.abs() <= 0.25), "{v:?}");
        let len = std::fs::metadata(fixture("synthetic-v2.cnq")).unwrap().len();
        for t in &cnq.tensors {
            assert!(cnq.blob_offset + t.offset + Cnq::byte_len(t) <= len, "{}", t.name);
        }
    }

    /// A v1 index is accepted when, and only when, its bytes hash to the container of record.
    /// The accepting branch runs on a synthetic trailer with its own hash as "of record"; the
    /// real constant is checked on the real file by the `#[ignore]` test below.
    #[test]
    fn a_v1_index_is_accepted_only_as_the_container_of_record() {
        let bytes = br#"{"format":"crow-nest-quant","version":1,"blob_offset":12,"tensors":[]}"#;
        let v: serde_json::Value = serde_json::from_slice(bytes).unwrap();
        let own = sha256_hex(bytes);
        assert_eq!(classify_index_against(bytes, &v, &own).unwrap(), IndexKind::V1OfRecord);
        // the same bytes against the real hash of record: refused, by name, with both hashes
        let m = classify_index(bytes, &v).unwrap_err();
        assert!(m.starts_with(V1_REFUSAL), "{m}");
        assert!(m.contains(&own) && m.contains(CNQ45M_INDEX_SHA256), "{m}");
        // one byte different from the record is a different container
        let mut other = bytes.to_vec();
        other[2] = b'F';
        assert!(classify_index_against(&other, &v, &own).unwrap_err().starts_with(V1_REFUSAL));
        // and the refusal is what `open` says too, on a real file
        let dir = std::env::temp_dir().join(format!("cnq-c6-v1-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let p = dir.join("v1.cnq");
        let mut f = b"CNQ1\0\0\0\0\0\0\0\0".to_vec();
        f.extend_from_slice(bytes);
        f.extend_from_slice(&(bytes.len() as u64).to_le_bytes());
        std::fs::write(&p, &f).unwrap();
        let m = Cnq::open_checked(p.to_str().unwrap()).err().unwrap();
        assert!(m.starts_with(V1_REFUSAL), "{m}");
        std::fs::remove_dir_all(&dir).ok();
    }

    /// Everything else is refused by name: an unknown format_version, a v2 without its model
    /// block, a v2 whose stored config no longer hashes to what the converter recorded.
    #[test]
    fn an_unknown_version_a_missing_model_block_and_an_altered_config_are_refused() {
        let b = std::fs::read(fixture("synthetic-v2.cnq")).unwrap();
        let n = u64::from_le_bytes(b[b.len() - 8..].try_into().unwrap()) as usize;
        let ib = &b[b.len() - 8 - n..b.len() - 8];
        let good: serde_json::Value = serde_json::from_slice(ib).unwrap();
        assert!(matches!(classify_index(ib, &good).unwrap(), IndexKind::V2(_)));

        let mut v3 = good.clone();
        v3["format_version"] = 3.into();
        assert!(classify_index(ib, &v3).unwrap_err().contains("format_version 3 is not one this engine reads"));

        let mut no_model = good.clone();
        no_model.as_object_mut().unwrap().remove("model");
        assert!(classify_index(ib, &no_model).unwrap_err().contains("without a `model` block"));

        let mut altered = good.clone();
        let c = altered["model"]["config_json"].as_str().unwrap().replace("\"hidden_size\": 64", "\"hidden_size\": 65");
        altered["model"]["config_json"] = c.into();
        let m = classify_index(ib, &altered).unwrap_err();
        assert!(m.contains("model.config_json hashes to") && m.contains("altered"), "{m}");

        let mut no_repo = good.clone();
        no_repo["model"]["source"].as_object_mut().unwrap().remove("repo");
        assert!(classify_index(ib, &no_repo).unwrap_err().contains("model.source.repo"));

        let mut bad_v1 = good.clone();
        bad_v1.as_object_mut().unwrap().remove("format_version");
        bad_v1["version"] = 2.into();
        assert!(classify_index(ib, &bad_v1).unwrap_err().contains("`version` 2 without a `format_version`"));
    }

    /// The real container of record opens through the v1 path. It reads only the trailer (the
    /// mapping is lazy), but it needs the 104.7 GB file: `CROW_CNQ` names it, else the default
    /// `converter/Qwen3.8-Flash-Next-CNQ4.5-M.cnq` beside the engine.
    #[test]
    #[ignore = "needs the 104.7 GB container of record: CROW_CNQ=<path> cargo test --release tests_300_c6 -- --ignored"]
    fn the_container_of_record_opens_as_v1() {
        let path = std::env::var("CROW_CNQ")
            .unwrap_or_else(|_| format!("{}/../converter/Qwen3.8-Flash-Next-CNQ4.5-M.cnq", env!("CARGO_MANIFEST_DIR")));
        let cnq = Cnq::open_checked(&path).unwrap();
        assert_eq!(cnq.index_version(), 1);
        assert_eq!(cnq.index_kind, IndexKind::V1OfRecord);
        assert!(cnq.config_json().is_none() && cnq.model().is_none());
        assert_eq!(cnq.family(), "FlashNext");
        assert_eq!(cnq.tensors.len(), 1658);
        let len = std::fs::metadata(&path).unwrap().len();
        assert_eq!(len, 104_727_179_972);
        // the pinned hash, recomputed from the trailer bytes here
        use std::io::{Read, Seek, SeekFrom};
        let mut f = std::fs::File::open(&path).unwrap();
        f.seek(SeekFrom::Start(len - 8 - CNQ45M_INDEX_LEN)).unwrap();
        let mut ib = vec![0u8; CNQ45M_INDEX_LEN as usize];
        f.read_exact(&mut ib).unwrap();
        assert_eq!(sha256_hex(&ib), CNQ45M_INDEX_SHA256);
    }
}
