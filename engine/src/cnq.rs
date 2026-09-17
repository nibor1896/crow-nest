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
    /// unix: mmap; 0 = not mapped -> seek/read fallback). Row reads become
    /// page-cache memcpys (~1 us on a hit instead of a seek+read syscall pair);
    /// CROW_MMAP=0 disables.
    pub map: usize,
    pub map_len: u64,
    /// the CreateFileMappingW handle behind `map` (closed in Drop after the
    /// unmap); always 0 on unix, where munmap takes `map_len` instead
    pub map_handle: usize,
    /// container path: Drop re-opens it unbuffered once to purge its cached pages
    pub path: String,
}

/// drop the file's pages from the system cache: an open with
/// FILE_FLAG_NO_BUFFERING while no cached handle exists makes NTFS purge the
/// cache section (2026-09-06: the cache manager kept 3.3 GB of tier reads in
/// the system cache working set until process exit, so a harness reload saw
/// that much less "available" RAM and residency.rs refused to pin the tier)
pub fn purge_cache(path: &str) {
    // CROW_CNQ_PURGE=0 keeps the cache (control knob; the purge also cools the
    // PLE rows the next process would have found in standby)
    if std::env::var("CROW_CNQ_PURGE").as_deref() == Ok("0") {
        return;
    }
    #[cfg(windows)]
    {
        use std::os::windows::fs::OpenOptionsExt;
        let _ = std::fs::OpenOptions::new().read(true).custom_flags(0x2000_0000 /* FILE_FLAG_NO_BUFFERING */).open(path);
    }
    // unix twin: POSIX_FADV_DONTNEED over the whole file (len 0 = to EOF) drops
    // its clean page-cache pages. It only evicts unmapped, unreferenced pages,
    // so the Drop order below - unmap, close the handle, then purge - is what
    // makes them droppable here too.
    #[cfg(unix)]
    {
        use std::os::unix::io::AsRawFd;
        if let Ok(f) = std::fs::File::open(path) {
            unsafe { libc::posix_fadvise(f.as_raw_fd(), 0, 0, libc::POSIX_FADV_DONTNEED) };
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
        // close the cached handle first (the field would drop after this body),
        // then the unbuffered open / fadvise purges the cache
        if let Ok(nul) = std::fs::File::open(NULL_DEVICE) {
            drop(std::mem::replace(&mut self.file, nul));
            purge_cache(&self.path);
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
            eprintln!("[cnq] file mapping unavailable - seek/read fallback");
        }
        Cnq { file: f, blob_offset, tensors, map, map_len, map_handle, path: path.to_string() }
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

    pub fn read_bytes(&mut self, t: &TensorInfo) -> Vec<u8> {
        let off = self.blob_offset + t.offset;
        let len = Self::byte_len(t) as usize;
        let mut raw = vec![0u8; len];
        if self.map != 0 && t.section == "ple" && off + len as u64 <= self.map_len {
            unsafe { std::ptr::copy_nonoverlapping((self.map as *const u8).add(off as usize), raw.as_mut_ptr(), len) };
            return raw;
        }
        self.file.seek(SeekFrom::Start(off)).unwrap();
        self.read_exact_into(&mut raw);
        raw
    }

    /// read a byte range of a tensor (row-aligned reads for expert slabs, PLE rows)
    pub fn read_range(&mut self, t: &TensorInfo, rel_off: u64, len: usize) -> Vec<u8> {
        let off = self.blob_offset + t.offset + rel_off;
        let mut raw = vec![0u8; len];
        if self.map != 0 && t.section == "ple" && off + len as u64 <= self.map_len {
            unsafe { std::ptr::copy_nonoverlapping((self.map as *const u8).add(off as usize), raw.as_mut_ptr(), len) };
            return raw;
        }
        self.file.seek(SeekFrom::Start(off)).unwrap();
        self.read_exact_into(&mut raw);
        raw
    }

    fn read_exact_into(&mut self, buf: &mut [u8]) {
        self.file.read_exact(buf).unwrap();
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
