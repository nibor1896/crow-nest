//! CUDA driver-API helpers shared by the whole engine. Every pattern here is
//! probe-pinned (p1–p16): raw sys API, scalar args ONLY via device buffers,
//! HtoD always async + explicit sync (WDDM), 64-v2 memops only, device
//! pointers never dereferenced on the host.

use cudarc::driver::sys::{self, CUfunction, CUmodule, CUresult, CUstream};
pub use cudarc::driver::sys::CUdeviceptr;
use std::ffi::CString;
use std::sync::atomic::{AtomicU64, Ordering};
use std::collections::HashMap;
use std::sync::{Mutex, OnceLock};

// ---- CUDA Graphs (CROW_GRAPH=1) ---------------------------------------
// Kernel launches route through this stream; 0 = legacy default stream
// (normal operation). During capture and replay it holds the capture stream.
static CUR_STREAM: AtomicU64 = AtomicU64::new(0);

pub fn set_stream(h: u64) {
    CUR_STREAM.store(h, Ordering::Relaxed);
}

pub fn cur_stream() -> CUstream {
    CUR_STREAM.load(Ordering::Relaxed) as CUstream
}

// cudarc 0.19.9 binds the graph entry points only for CUDA 11.4-11.8 - on
// cuda-13030 we load them directly out of the driver library (same loader
// pattern as the 32-bit memops were handled with in p9).
type FnStreamCreate = unsafe extern "system" fn(*mut CUstream, u32) -> CUresult;
type FnBeginCapture = unsafe extern "system" fn(CUstream, sys::CUstreamCaptureMode) -> CUresult;
type FnEndCapture = unsafe extern "system" fn(CUstream, *mut sys::CUgraph) -> CUresult;
type FnInstantiate = unsafe extern "system" fn(*mut sys::CUgraphExec, sys::CUgraph, u64) -> CUresult;
type FnGraphLaunch = unsafe extern "system" fn(sys::CUgraphExec, CUstream) -> CUresult;

/// the CUDA driver library the graph entry points come out of (cudarc has it
/// open already; this is a second dlopen/LoadLibrary of the same soname)
#[cfg(windows)]
const DRIVER_LIB: &str = "nvcuda.dll";
#[cfg(unix)]
const DRIVER_LIB: &str = "libcuda.so.1";

fn graph_lib() -> &'static libloading::Library {
    static LIB: std::sync::OnceLock<libloading::Library> = std::sync::OnceLock::new();
    LIB.get_or_init(|| unsafe {
        libloading::Library::new(DRIVER_LIB).unwrap_or_else(|e| panic!("{DRIVER_LIB} (graph api): {e}"))
    })
}

unsafe fn graph_sym<T: Copy>(name: &[u8]) -> T {
    unsafe {
        let lib = graph_lib();
        *lib.get(name).unwrap_or_else(|e| panic!("graph symbol missing in {DRIVER_LIB}: {e}"))
    }
}

/// create a non-blocking stream (required: capture is illegal on the legacy
/// default stream)
pub unsafe fn stream_create_non_blocking() -> CUstream {
    let f: FnStreamCreate = graph_sym(b"cuStreamCreate\0");
    let mut s: CUstream = std::ptr::null_mut();
    let r = f(&mut s, 1); // CU_STREAM_NON_BLOCKING = 0x1
    eprintln!("[graph] cuStreamCreate -> {r:?}, handle {:p}", s);
    ck(r);
    s
}

/// begin capture on the given stream (GLOBAL mode - single-threaded engine)
pub unsafe fn begin_capture(s: CUstream) {
    let f: FnBeginCapture = graph_sym(b"cuStreamBeginCapture_v2\0"); // v2 = (stream, mode); the v1 entry has no mode arg
    ck(f(s, sys::CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_GLOBAL));
}

type FnIsCapturing = unsafe extern "system" fn(CUstream, *mut u32) -> CUresult;

/// capture status of a stream: 0 = NONE, 1 = ACTIVE, 2 = INVALIDATED (debug aid)
pub unsafe fn capture_status(s: CUstream) -> u32 {
    let f: FnIsCapturing = graph_sym(b"cuStreamIsCapturing\0");
    let mut st: u32 = 99;
    let r = f(s, &mut st);
    if r != sys::CUresult::CUDA_SUCCESS {
        eprintln!("[graph] cuStreamIsCapturing -> {r:?}");
    }
    st
}

/// end capture + instantiate; returns the executable graph handle
pub unsafe fn end_capture_instantiate(s: CUstream) -> sys::CUgraphExec {
    let f_end: FnEndCapture = graph_sym(b"cuStreamEndCapture\0");
    let f_inst: FnInstantiate = graph_sym(b"cuGraphInstantiateWithFlags\0");
    let mut graph: sys::CUgraph = std::ptr::null_mut();
    let r_end = f_end(s, &mut graph);
    eprintln!("[graph] EndCapture -> {r_end:?} graph={:p}", graph);
    if r_end != sys::CUresult::CUDA_SUCCESS {
        panic!("EndCapture failed: {r_end:?}");
    }
    let mut exec: sys::CUgraphExec = std::ptr::null_mut();
    let r_inst = f_inst(&mut exec, graph, 0);
    eprintln!("[graph] Instantiate -> {r_inst:?} exec={:p}", exec);
    if r_inst != sys::CUresult::CUDA_SUCCESS {
        panic!("Instantiate failed: {r_inst:?}");
    }
    // #18: instantiation takes a snapshot; the template graph held ~6 MB of
    // VRAM per engine load (measured 2026-09-05: GRAPH=1 214 MB vs GRAPH=0 208 MB
    // per reload cycle) and was never destroyed. The exec is independent of it.
    let f_destroy: FnGraphDestroy = graph_sym(b"cuGraphDestroy\0");
    let r_del = f_destroy(graph);
    if r_del != sys::CUresult::CUDA_SUCCESS {
        eprintln!("[graph] cuGraphDestroy -> {r_del:?}");
    }
    exec
}

type FnGraphDestroy = unsafe extern "system" fn(sys::CUgraph) -> CUresult;

type FnGraphExecDestroy = unsafe extern "system" fn(sys::CUgraphExec) -> CUresult;

/// destroy an instantiated graph (#18: the exec held VRAM across engine reloads)
pub unsafe fn graph_exec_destroy(exec: sys::CUgraphExec) {
    if !exec.is_null() {
        let f: FnGraphExecDestroy = graph_sym(b"cuGraphExecDestroy\0");
        ck(f(exec));
    }
}

/// destroy a stream created by stream_create_non_blocking (#18)
pub unsafe fn stream_destroy(s: CUstream) {
    if !s.is_null() {
        ck(sys::cuStreamDestroy_v2(s));
    }
}

/// destroy an event created by event_create (#18)
pub unsafe fn event_destroy(e: sys::CUevent) {
    if !e.is_null() {
        ck(sys::cuEventDestroy_v2(e));
    }
}

/// launch an instantiated graph on the given stream
pub unsafe fn launch_graph(exec: sys::CUgraphExec, s: CUstream) {
    let f: FnGraphLaunch = graph_sym(b"cuGraphLaunch\0");
    ck(f(exec, s));
}

/// async HtoD from PINNED host memory - safe without sync (pinned staging;
/// stack-sourced async HtoD dies with the frame on a non-blocking stream)
pub unsafe fn upload_from_pinned(dst: CUdeviceptr, host: *const std::ffi::c_void, bytes: usize) {
    let s = cur_stream();
    let r = sys::cuMemcpyHtoDAsync_v2(dst, host, bytes, s);
    if r != sys::CUresult::CUDA_SUCCESS {
        eprintln!("[graph-dbg] HtoD dst={:p} host={:p} bytes={} stream={:p} -> {:?}",
            dst as *const std::ffi::c_void, host, bytes, s, r);
    }
    ck(r);
}

/// device-to-device async copy on the current stream (graph-capturable) -
/// used for the dynamic pool-offset write in the captured QSA sequence
pub unsafe fn d2d_async(dst: CUdeviceptr, src: CUdeviceptr, bytes: usize) {
    let s = cur_stream();
    ck(sys::cuMemcpyDtoDAsync_v2(
        dst,
        src,
        bytes,
        s,
    ));
}

/// generic async copy on the current stream (unified addressing: pinned
/// host <-> device in either direction, device <-> device)
pub unsafe fn memcpy_async(dst: CUdeviceptr, src: CUdeviceptr, bytes: usize) {
    ck(sys::cuMemcpyAsync(dst, src, bytes, cur_stream()));
}

pub unsafe fn event_create() -> sys::CUevent {
    let mut e: sys::CUevent = std::ptr::null_mut();
    ck(sys::cuEventCreate(&mut e, 2)); // CU_EVENT_DISABLE_TIMING
    e
}
pub unsafe fn event_record(e: sys::CUevent, s: CUstream) {
    ck(sys::cuEventRecord(e, s));
}
/// waiting on a never-recorded event is a no-op (CUDA semantics)
pub unsafe fn stream_wait_event(s: CUstream, e: sys::CUevent) {
    ck(sys::cuStreamWaitEvent(s, e, 0));
}
pub unsafe fn memcpy_async_on(dst: CUdeviceptr, src: CUdeviceptr, bytes: usize, s: CUstream) {
    ck(sys::cuMemcpyAsync(dst, src, bytes, s));
}
/// synchronize one explicit stream (side streams; `sync()` covers the active one)
pub unsafe fn stream_sync(s: CUstream) {
    ck(sys::cuStreamSynchronize(s));
}

/// non-blocking stream at the highest scheduling priority: its pending blocks
/// are preferred by the block scheduler whenever an SM frees up (CROW_PF_ASYNC
/// staging next to the tile GEMMs)
pub unsafe fn stream_create_priority() -> CUstream {
    let (mut least, mut greatest) = (0i32, 0i32);
    ck(sys::cuCtxGetStreamPriorityRange(&mut least, &mut greatest));
    let mut s: CUstream = std::ptr::null_mut();
    ck(sys::cuStreamCreateWithPriority(&mut s, 1, greatest)); // CU_STREAM_NON_BLOCKING
    eprintln!("[stream] priority side stream {:p} (priority range {least}..{greatest})", s);
    s
}

/// non-blocking status query; on WDDM it also submits the stream's pending
/// command buffer to the GPU (CROW_PF_ASYNC side stream)
pub unsafe fn stream_query(s: CUstream) -> bool {
    sys::cuStreamQuery(s) == sys::CUresult::CUDA_SUCCESS
}

/// NVRTC compile + module load (compute_120a — probe 2 requirement)
pub unsafe fn compile(src: &str) -> Module {
    let opts = cudarc::nvrtc::CompileOptions {
        options: vec!["--gpu-architecture=compute_120a".into()],
        ..Default::default()
    };
    let ptx = cudarc::nvrtc::compile_ptx_with_opts(src, opts).expect("nvrtc compile");
    let c = CString::new(ptx.to_src()).unwrap();
    let mut m: CUmodule = std::ptr::null_mut();
    ck(sys::cuModuleLoadData(&mut m, c.as_ptr() as *const _));
    Module(m)
}

pub fn ck(r: CUresult) {
    if r != CUresult::CUDA_SUCCESS {
        panic!("CUDA error: {r:?}");
    }
}

pub struct Ctx {
    pub ctx: *mut cudarc::driver::sys::CUctx_st,
    pub dev: std::os::raw::c_int,
}

impl Ctx {
    /// primary context on device 0 (probe convention)
    pub unsafe fn init() -> Ctx {
        ck(sys::cuInit(0));
        let mut dev = 0;
        ck(sys::cuDeviceGet(&mut dev, 0));
        let mut ctx = std::ptr::null_mut();
        ck(sys::cuDevicePrimaryCtxRetain(&mut ctx, dev));
        ck(sys::cuCtxSetCurrent(ctx));
        Ctx { ctx, dev }
    }

}

impl Drop for Ctx {
    /// release the primary context (paired with the retain in `init`). The
    /// driver's host-side memory for the context is not in the process
    /// working set and outlives every Engine drop; a harness that opens a Ctx
    /// per request measured ~3.3 GB less available RAM on its second load
    /// (2026-09-06) and residency.rs refused to pin the tier.
    fn drop(&mut self) {
        unsafe {
            let _ = sys::cuCtxSynchronize();
            let _ = sys::cuDevicePrimaryCtxRelease_v2(self.dev);
        }
    }
}

/// CROW_DROP_DBG=1: print free VRAM at a named point (#18 leak hunt)
pub unsafe fn drop_dbg(tag: &str) {
    if std::env::var("CROW_DROP_DBG").as_deref() == Ok("1") {
        let (n, bytes) = live_dev();
        eprintln!("[drop-dbg] {tag}: free VRAM {:.1} MB, engine live allocs {n} = {:.1} MB", free_vram_bytes() as f64 / 1e6, bytes as f64 / 1e6);
        if n > 0 && n <= 64 {
            eprintln!("[drop-dbg]   surviving sizes (bytes, largest first): {:?}", live_dev_top(64));
        }
    }
}

/// one `cuMemGetInfo_v2` -> (free, total) device bytes; the two named readers below derive
pub unsafe fn vram_info() -> (u64, u64) {
    let mut free: usize = 0;
    let mut total: usize = 0;
    ck(sys::cuMemGetInfo_v2(&mut free, &mut total));
    (free as u64, total as u64)
}

pub unsafe fn free_vram_bytes() -> u64 { vram_info().0 }

pub unsafe fn total_vram_bytes() -> u64 { vram_info().1 }

pub struct Module(pub CUmodule);

impl Module {
    /// #18: unload the NVRTC module (one per Engine::load; never unloaded until 2026-09-05
    /// -> ~220 MB of VRAM per reload, measured by `decode reloadcheck`)
    pub unsafe fn unload(&mut self) {
        if !self.0.is_null() {
            ck(sys::cuModuleUnload(self.0));
            self.0 = std::ptr::null_mut();
        }
    }
}

impl Module {
    pub unsafe fn get(&self, name: &str) -> CUfunction {
        let mut f: CUfunction = std::ptr::null_mut();
        ck(sys::cuModuleGetFunction(
            &mut f,
            self.0,
            CString::new(name).unwrap().as_ptr(),
        ));
        f
    }
}

/// #18: every cuMemAlloc of the engine goes through here; the live table lets
/// `drop_dbg` say whether VRAM that did not come back is an engine buffer
/// (listed here) or driver-owned (not listed).
fn live_allocs() -> &'static Mutex<HashMap<u64, usize>> {
    static M: OnceLock<Mutex<HashMap<u64, usize>>> = OnceLock::new();
    M.get_or_init(|| Mutex::new(HashMap::new()))
}

/// (count, bytes) of engine device allocations not yet freed
pub fn live_dev() -> (usize, u64) {
    let m = live_allocs().lock().unwrap();
    (m.len(), m.values().map(|&v| v as u64).sum())
}

/// the surviving allocations, largest first (size in bytes), at most `n`
pub fn live_dev_top(n: usize) -> Vec<usize> {
    let m = live_allocs().lock().unwrap();
    let mut v: Vec<usize> = m.values().copied().collect();
    v.sort_unstable_by(|a, b| b.cmp(a));
    v.truncate(n);
    v
}

pub unsafe fn alloc_zeroed(bytes: usize) -> CUdeviceptr {
    assert!(bytes > 0, "alloc of 0 bytes");
    let mut d: CUdeviceptr = 0;
    ck(sys::cuMemAlloc_v2(&mut d, bytes));
    ck(sys::cuMemsetD8_v2(d, 0, bytes));
    live_allocs().lock().unwrap().insert(d, bytes);
    d
}

pub unsafe fn free_dev(d: &mut CUdeviceptr) {
    if *d != 0 {
        ck(sys::cuMemFree_v2(*d));
        live_allocs().lock().unwrap().remove(&(*d as u64));
        *d = 0;
    }
}

pub unsafe fn upload_dev(v: &[u8]) -> CUdeviceptr {
    let d = alloc_zeroed(v.len());
    ck(sys::cuMemcpyHtoDAsync_v2(
        d,
        v.as_ptr() as *const std::ffi::c_void,
        v.len(),
        std::ptr::null_mut(),
    ));
    ck(sys::cuStreamSynchronize(std::ptr::null_mut()));
    d
}

/// upload a host slice into a fresh device buffer of exactly `size_of_val(v)` bytes.
/// The three typed names below stay: the element type is what fixes the byte count,
/// so spelling it at the call site is load-bearing, not decoration.
pub unsafe fn to_dev<T: Copy>(v: &[T]) -> CUdeviceptr {
    upload_dev(std::slice::from_raw_parts(v.as_ptr() as *const u8, std::mem::size_of_val(v)))
}

pub unsafe fn to_f32_dev(v: &[f32]) -> CUdeviceptr { to_dev(v) }

pub unsafe fn to_i32_dev(v: &[i32]) -> CUdeviceptr { to_dev(v) }

pub unsafe fn to_u64_dev(v: &[u64]) -> CUdeviceptr { to_dev(v) }

/// overwrite the first `v.len()` bytes of an existing device buffer.
/// Legacy stream (0): async + explicit sync (the WDDM rule for one-shot
/// uploads). Graph stream active: async WITHOUT sync — ordering is the
/// stream's, and a sync here would break graph capture.
pub unsafe fn upload_into(dst: CUdeviceptr, v: &[u8]) {
    let s = cur_stream();
    ck(sys::cuMemcpyHtoDAsync_v2(
        dst,
        v.as_ptr() as *const std::ffi::c_void,
        v.len(),
        s,
    ));
    if s as u64 == 0 {
        ck(sys::cuStreamSynchronize(std::ptr::null_mut()));
    }
}

/// `upload_into` for a typed host slice (same stream rule as `upload_into`)
pub unsafe fn into_dev<T: Copy>(dst: CUdeviceptr, v: &[T]) {
    upload_into(
        dst,
        std::slice::from_raw_parts(v.as_ptr() as *const u8, std::mem::size_of_val(v)),
    );
}

pub unsafe fn to_f32_into(dst: CUdeviceptr, v: &[f32]) { into_dev(dst, v) }

pub unsafe fn to_i32_into(dst: CUdeviceptr, v: &[i32]) { into_dev(dst, v) }

pub unsafe fn to_u64_into(dst: CUdeviceptr, v: &[u64]) { into_dev(dst, v) }

/// blocking D2H of `n` elements. `cuMemcpyDtoH_v2` on the legacy stream is itself
/// a sync point, so only `dtoh_u32` adds an explicit stream sync (see below).
pub unsafe fn dtoh_t<T: Copy + Default>(src: CUdeviceptr, n: usize) -> Vec<T> {
    let mut out = vec![T::default(); n];
    ck(sys::cuMemcpyDtoH_v2(
        out.as_mut_ptr() as *mut std::ffi::c_void,
        src,
        std::mem::size_of_val(&out[..]),
    ));
    out
}

pub unsafe fn dtoh(src: CUdeviceptr, n_f32: usize) -> Vec<f32> { dtoh_t(src, n_f32) }

pub unsafe fn dtoh_i32(src: CUdeviceptr, n: usize) -> Vec<i32> { dtoh_t(src, n) }

/// DIFFERS from the other three on purpose: the u32 readers (route log, block
/// counts) run while the ACTIVE stream may still be in flight, so sync it first.
pub unsafe fn dtoh_u32(src: CUdeviceptr, n: usize) -> Vec<u32> {
    ck(sys::cuStreamSynchronize(cur_stream()));
    dtoh_t(src, n)
}

pub unsafe fn dtoh_u64(src: CUdeviceptr, n: usize) -> Vec<u64> { dtoh_t(src, n) }

/// async launch WITHOUT sync — the decode hot path batches submissions
pub unsafe fn launch_async(
    f: CUfunction,
    gx: u32,
    gy: u32,
    bx: u32,
    smem: u32,
    args: &mut [*mut std::ffi::c_void],
) {
    ck(sys::cuLaunchKernel(
        f, gx, gy, 1, bx, 1, 1, smem,
        std::ptr::null_mut(),
        args.as_mut_ptr(),
        std::ptr::null_mut(),
    ));
}

/// launch + sync (probe pattern; correctness stages and one-off work)
pub unsafe fn launch(
    f: CUfunction,
    gx: u32,
    gy: u32,
    bx: u32,
    smem: u32,
    args: &mut [*mut std::ffi::c_void],
) {
    launch_async(f, gx, gy, bx, smem, args);
    ck(sys::cuStreamSynchronize(std::ptr::null_mut()));
}

pub unsafe fn sync() {
    // synchronize the ACTIVE stream: legacy default in normal operation,
    // the graph/capture stream when CROW_GRAPH is active (0 = legacy)
    let s = CUR_STREAM.load(Ordering::Relaxed);
    ck(sys::cuStreamSynchronize(s as CUstream));
}

/// free physical host RAM in bytes (kernel32 GlobalMemoryStatusEx); 0 if the
/// query fails. Pinned allocations cannot be paged, so the loader refuses to
/// pin more than what is physically free minus a margin (2026-09-04 freeze).
#[cfg(windows)]
pub fn free_physical_ram_parts() -> (u64, u64) {
    #[repr(C)]
    struct MemStatusEx {
        length: u32,
        memory_load: u32,
        total_phys: u64,
        avail_phys: u64,
        total_page: u64,
        avail_page: u64,
        total_virtual: u64,
        avail_virtual: u64,
        avail_ext_virtual: u64,
    }
    type FnGms = unsafe extern "system" fn(*mut MemStatusEx) -> i32;
    unsafe {
        let Ok(lib) = libloading::Library::new("kernel32.dll") else { return (0, 0) };
        let Ok(f) = lib.get::<FnGms>(b"GlobalMemoryStatusEx\0") else { return (0, 0) };
        let mut st = MemStatusEx {
            length: std::mem::size_of::<MemStatusEx>() as u32,
            memory_load: 0, total_phys: 0, avail_phys: 0, total_page: 0,
            avail_page: 0, total_virtual: 0, avail_virtual: 0, avail_ext_virtual: 0,
        };
        if f(&mut st) == 0 { return (0, 0) }
        // one number on windows: avail_phys already excludes what cannot be paged
        (st.avail_phys, st.avail_phys)
    }
}

/// unix twin. Returns `(free_for_pin, mem_available)`; `(0, 0)` without /proc.
///
/// `MemAvailable` is the WRONG input for the pinned tier on this host, low by
/// tens of GiB, for two reasons (both measured 2026-09-17, issue #15):
///
/// - The NVIDIA driver keeps its pinned-page pool after a process exits (about
///   45 GiB after one engine run). Those pages belong to no process and land in
///   no /proc/meminfo class, so `MemAvailable` does not see them - yet the next
///   `cuMemHostAlloc` is served out of that pool, and the pool is handed back
///   under pressure (`pin_leak` took 8 GiB of WC pinned memory with `MemFree`
///   unmoved; a cgroup-capped balloon pushed the whole pool back). Without this
///   every second engine start refused with "only 10.26 GiB physical RAM free".
/// - The page cache is reclaimable by definition, and the cold-tier fill is
///   what fills it.
///
/// So `free_for_pin` counts what canNOT be reclaimed and subtracts it from
/// `MemTotal` (all /proc/meminfo field names, kB):
///
/// ```text
/// free_for_pin = MemTotal
///              - AnonPages   process anonymous memory (swap is not counted on)
///              - Shmem       tmpfs + shared anon, incl. ShmemHugePages
///              - SUnreclaim  kernel slab no shrinker can free
///              - KernelStack - PageTables - Percpu
/// ```
///
/// `Unevictable` and `Mlocked` are deliberately NOT subtracted: an mlocked page
/// is an anonymous or a shmem page, so it is already inside `AnonPages` /
/// `Shmem` and subtracting it again would double count (ramfs is the only
/// unevictable class outside both, and there is none on this host).
/// `MemFree`, `Buffers`, `Cached` and `SReclaimable` are reclaimable and stay
/// counted as free. A field the kernel does not expose counts as 0, so the
/// estimate errs LARGE; the `CROW_RAM_MARGIN_GB` margin, the 46 GiB budget cap
/// and the pre-pin gate in `residency::build` are what bound it.
#[cfg(unix)]
pub fn free_physical_ram_parts() -> (u64, u64) {
    let Ok(txt) = std::fs::read_to_string("/proc/meminfo") else { return (0, 0) };
    let f = |name: &str| -> u64 {
        for line in txt.lines() {
            if let Some(rest) = line.strip_prefix(name) {
                if rest.starts_with(':') {
                    return rest[1..].split_whitespace().next()
                        .and_then(|kb| kb.parse::<u64>().ok()).unwrap_or(0) * 1024;
                }
            }
        }
        0
    };
    let total = f("MemTotal");
    let unreclaimable = f("AnonPages") + f("Shmem") + f("SUnreclaim")
        + f("KernelStack") + f("PageTables") + f("Percpu");
    (total.saturating_sub(unreclaimable), f("MemAvailable"))
}

/// the RAM the cold tier may be pinned into (see `free_physical_ram_parts`)
pub fn free_physical_ram() -> u64 {
    free_physical_ram_parts().0
}

// ---------- pinned host memory (zero-copy cold tier, probe 3/4/5 pattern) ----------

pub struct Pinned {
    pub host: *mut std::ffi::c_void,
    pub dev: CUdeviceptr, // UVA device pointer for zero-copy reads
    pub bytes: usize,
}

impl Pinned {
    /// the one `cuMemHostAlloc`; `alloc` and `alloc_wc` differ only by the flag word
    unsafe fn alloc_flags(bytes: usize, flags: u32) -> Pinned {
        let mut host: *mut std::ffi::c_void = std::ptr::null_mut();
        ck(sys::cuMemHostAlloc(&mut host, bytes, flags));
        let mut dev: CUdeviceptr = 0;
        ck(sys::cuMemHostGetDevicePointer_v2(&mut dev, host, 0));
        Pinned { host, dev, bytes }
    }

    /// mapped pinned allocation; the returned `dev` pointer is what kernels read
    pub unsafe fn alloc(bytes: usize) -> Pinned {
        Pinned::alloc_flags(bytes, sys::CU_MEMHOSTALLOC_PORTABLE | sys::CU_MEMHOSTALLOC_DEVICEMAP)
    }

    pub unsafe fn write_bytes(&mut self, offset: usize, v: &[u8]) {
        assert!(offset + v.len() <= self.bytes);
        std::ptr::copy_nonoverlapping(v.as_ptr(), (self.host as *mut u8).add(offset), v.len());
    }

    pub unsafe fn free(&mut self) {
        if !self.host.is_null() {
            ck(sys::cuMemFreeHost(self.host));
            self.host = std::ptr::null_mut();
            self.dev = 0;
        }
    }

    /// write combined variant for host->device flag buffers (p9 lesson)
    pub unsafe fn alloc_wc(bytes: usize) -> Pinned {
        Pinned::alloc_flags(
            bytes,
            sys::CU_MEMHOSTALLOC_PORTABLE | sys::CU_MEMHOSTALLOC_DEVICEMAP | sys::CU_MEMHOSTALLOC_WRITECOMBINED,
        )
    }
}

// unsafe impl Send: the raw pointers are process-global resources; the engine
// only touches them from the loader thread / control thread with ctx current.
unsafe impl Send for Pinned {}
unsafe impl Send for Module {}
unsafe impl Send for Ctx {}

// ---------- raw little-endian dumps (CROW_DUMP_H / CROW_VIT_DUMP) ----------

// The dump files are the host's own byte image; on a big-endian host that would
// stop being the little-endian format the readers (tools/*.py, cmp) expect.
const _: () = assert!(cfg!(target_endian = "little"), "dump files are little-endian");

/// write a POD slice (f32 / i32 / u64 ...) to `path` as raw little-endian bytes.
/// The one dump writer: 10 hand-rolled `to_le_bytes` loops used to spell this out.
pub fn write_le<T: Copy>(path: &str, v: &[T]) -> std::io::Result<()> {
    use std::io::Write;
    let bytes = unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, std::mem::size_of_val(v)) };
    std::fs::File::create(path)?.write_all(bytes)
}
