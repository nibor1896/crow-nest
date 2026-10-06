//! #32 A10 - the slot file behind `POST /slots/0?action=save|restore` (spec section 7).
//!
//! Purpose:
//!
//! - Crow saves the KV at exit and restores it at the next start (`crow_core.py:2458`, `:2688`).
//! - Without it every start pays a full prefill: 16k = 21.6 to 24.1 s (spec 7.1, A9 measured).
//! - Spec 7.6 says snapshots are in-process state, not a file format.
//! - A10 is the deliberate exception, so the FILE carries the shape and a restore refuses a
//!   mismatch instead of loading state that is only shaped like the right one.
//!
//! What the file holds, and why exactly that:
//!
//! | part | source | why |
//! |---|---|---|
//! | header | this module | shape, position and payload size, checked before anything is touched |
//! | GDN S, GDN conv, PLE conv, QSA ring | `cache::PrefixCache` slot `SLOT_PROMPT` | the recurrent state, host side already |
//! | KV rows `0..pos` | `ThreeStates::kv_buf` (`manager.rs:305-311`) | absolutely addressed, so a prefix of them is a state |
//! | pooled QSA blocks `0..done_blocks` | `ThreeStates::qsa_pooled` (`gen.rs:1647-1661`) | same argument, block index = `pos/4` |
//! | `history[..pos]` | `Engine::history` | the id list the A9 detection rule compares against |
//!
//! Why the PROMPT slot and nothing else (A9, measured 2026-09-10):
//!
//! - Only a PREFILL CLEAN position reproduces a cold run bit for bit (`cache.rs` module doc).
//! - `decode_step` rows are not bit equal to `prefill` rows at the same position.
//! - So the saved position is the end of the last PROMPT, never the end of the last answer.
//! - `n_saved` is that position; the answer after it is re-prefilled by the next turn.
//!
//! The KV prefix is untouched by everything that ran after the snapshot:
//!
//! | writer | rows it writes | evidence |
//! |---|---|---|
//! | `prefill` | `pos_base .. pos_base + t` | `gen.rs:1644`, `store_kv` at `pos_base` |
//! | `decode_step` | one row at the current `pos`, always `>= snapshot pos` | `kernels.rs:1190-1207` |
//! | pooled blocks | `done_blocks .. (pos_base + t) / 4`, appended only | `gen.rs:1647-1661` |
//!
//! Header layout (little endian, fixed 136 bytes, x86_64 native f32 and i64):
//!
//! ```text
//! offset  size  field
//!      0     8  magic "CROWSLT\x01"
//!      8     8  format_version   (2 since Crow #300 C3; 1 had no model fingerprint)
//!     16     8  model_family     (geo::Family::code of the model that wrote it)
//!     24     8  geo_hash         (geo::Geo::fingerprint of the model that wrote it)
//!     32     8  load_id      (fnv1a-64 of the container path, mixed with n_hot)
//!     40     8  n_ctx
//!     48     8  prompt_chunk
//!     56     8  qsa_ring_rows
//!     64     8  gdn_layers
//!     72     8  attn_layers
//!     80     8  kv_groups        (attn_layers * 2 * Geo::kv_heads)
//!     88     8  kv_row_bytes     (KvDtype::row_bytes(Geo::head_dim): head_dim x 1 fp8, x 2 bf16, #88 q8 + head_dim / 16)
//!     96     8  pooled_row_bytes (the QSA raw key width * 4)
//!    104     8  state_bytes      (Shape::snapshot_bytes, the four recurrent buffers)
//!    112     8  pos
//!    120     8  done_blocks
//!    128     8  history_len
//! ```
//!
//! Crow #300 C3, the model fingerprint (format 2): `model_family` and `geo_hash` name
//! the MODEL, not only the buffer shape. A slot file of another model family, or of the
//! same family with any other geometry, is refused by name before a byte is read past
//! the header - even where every buffer size happens to agree. A format 1 file (no
//! fingerprint) is refused by name too, never guessed at: save the slot again.
//!
//! Payload, in this order, immediately after the header:
//!
//! ```text
//! gdn_s      gdn_layers  x 48*128*128 f32
//! gdn_conv   gdn_layers  x 10240*3    f32
//! ple_state              1 x 10240*9  f32
//! qsa_ring   attn_layers x ring*128   f32
//! kv         kv_groups   x pos        rows of kv_row_bytes
//! pooled     attn_layers x done_blocks rows of pooled_row_bytes
//! history                  history_len i64
//! ```
//!
//! What a restore refuses, and with which answer:
//!
//! | case | answer |
//! |---|---|
//! | no prefill clean position held (fresh process, cache off) | 409, `save` only |
//! | no `--slot-save-path`, or it is no directory | 400 on the request, refused at boot |
//! | `filename` with a path separator, `..`, or a character outside `[A-Za-z0-9._-]` | 400 |
//! | `filename` naming a Windows device (`NUL`, `CON`, `COM1`, `LPT1`, any extension) | 400 |
//! | file absent or unreadable | 400 |
//! | magic, format version or any shape field differs from this load | 400, engine untouched |
//! | a format 1 file (written before the model fingerprint, Crow #300 C3) | 400, engine untouched |
//! | model family or geometry fingerprint differs from the loaded model | 400, engine untouched |
//! | file length not exactly header + payload (truncated or padded) | 400, engine untouched |
//! | `pos` 0, `pos > n_ctx`, or `history_len != pos` | 400, engine untouched |
//! | `done_blocks != pos / 4` | 400, engine untouched |
//! | header payload size over u64 | 400, engine untouched |
//!
//! - Every check runs BEFORE the first device upload, so a refusal cannot leave a half state.
//! - The content checks are `Header::check_content`, pure, and unit tested branch by branch.
//! - `done_blocks` is an UPLOAD LENGTH into `qsa_pooled`, which holds `ceil(n_ctx/4)` blocks
//!   (`manager.rs:216-219`), so an unbounded value from a file would write past the buffer.
//! - `payload_bytes` saturates instead of wrapping: a release build has no overflow checks,
//!   and a wrapped size would name a SMALL payload for a file that asks for a huge upload.
//! - The payload length is checked against the size on disk before the first read, so an
//!   oversized file is refused without a byte of it entering host RAM, and the payload is
//!   then STREAMED into the destinations through one reusable buffer, never held whole.
//!
//! Ordering, and what a failure in the middle can leave (`cuda::ck` ends the process):
//!
//! | step | order | why |
//! |---|---|---|
//! | save: write a sibling `.part-<pid>`, fsync, then `fs::rename` over the target | last | a full disk cannot truncate the previous good file |
//! | save: `n_written` counted from the writes, then checked against the size on disk | last | the number on the wire is measured, not the header formula |
//! | restore: `cache.set_prompt_slot` | AFTER every upload | a `ck` failure mid upload must not leave a cache claiming a position the device never got |
//!
//! What a restore then does, so the NEXT request takes the tested A9 warm path:
//!
//! | step | effect |
//! |---|---|
//! | `cuda::sync`, `Engine::drop_decode_graph` | the A4 ordering, as `cache::PrefixCache::rollback` |
//! | host slot `SLOT_PROMPT` filled, `prefill_clean = true` | the file becomes a reuse candidate |
//! | KV rows and pooled blocks uploaded | the absolutely addressed part of the state |
//! | `Engine::pos`, `done_blocks`, `history`, `route_log` | the host triple `rollback` also sets |
//!
//! - The next chat request runs `decide` with `L >= pos` and lands on `P = pos` (spec 7.4).
//! - So the warm path is the A9 rollback path, not a second one.
//!
//! `load_id`, and what it is NOT:
//!
//! - It is a 64 bit fnv1a of the container PATH mixed with `Config::n_hot`.
//! - It catches a slot file written by a serve started on a different container.
//! - It is NOT a content hash: the same path with different bytes is not caught.
//! - The geometry fields catch every shape difference; this catches the obvious model swap.

use crate::cache::PrefixCache;
use crate::cuda;
use crate::gen::Engine;
use crate::geo::Family;
use cudarc::driver::sys;
use std::io::{Read, Write};

/// first eight bytes of every slot file
pub const MAGIC: [u8; 8] = *b"CROWSLT\x01";
/// bumped whenever the payload order or the header layout changes
/// (2: Crow #300 C3 added `model_family` and `geo_hash`)
pub const FORMAT_VERSION: u64 = 2;
/// the last format without a model fingerprint; refused by name
pub const FORMAT_VERSION_NO_FINGERPRINT: u64 = 1;
/// magic plus sixteen little endian u64
pub const HEADER_BYTES: usize = 8 + 16 * 8;

/// the shape and the position one slot file carries (see the module doc for the layout)
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Header {
    pub format_version: u64,
    /// `geo::Family::code` of the model that wrote the file
    pub model_family: u64,
    /// `geo::Geo::fingerprint` of the model that wrote the file
    pub geo_hash: u64,
    pub load_id: u64,
    pub n_ctx: u64,
    pub prompt_chunk: u64,
    pub qsa_ring_rows: u64,
    pub gdn_layers: u64,
    pub attn_layers: u64,
    pub kv_groups: u64,
    pub kv_row_bytes: u64,
    pub pooled_row_bytes: u64,
    pub state_bytes: u64,
    pub pos: u64,
    pub done_blocks: u64,
    pub history_len: u64,
}

impl Header {
    /// the sixteen u64 of the header, in file order
    fn words(&self) -> [u64; 16] {
        [
            self.format_version,
            self.model_family,
            self.geo_hash,
            self.load_id,
            self.n_ctx,
            self.prompt_chunk,
            self.qsa_ring_rows,
            self.gdn_layers,
            self.attn_layers,
            self.kv_groups,
            self.kv_row_bytes,
            self.pooled_row_bytes,
            self.state_bytes,
            self.pos,
            self.done_blocks,
            self.history_len,
        ]
    }

    /// the fixed 136 byte head of the file
    pub fn encode(&self) -> Vec<u8> {
        let mut b = Vec::with_capacity(HEADER_BYTES);
        b.extend_from_slice(&MAGIC);
        for w in self.words() {
            b.extend_from_slice(&w.to_le_bytes());
        }
        b
    }

    /// - `Err` carries the operator message; every case is a refusal, never a panic
    pub fn decode(bytes: &[u8]) -> Result<Header, String> {
        if bytes.len() < HEADER_BYTES {
            return Err(format!(
                "slot file head is {} bytes, too short for the {HEADER_BYTES} byte header",
                bytes.len()
            ));
        }
        if bytes[..8] != MAGIC {
            return Err(format!("bad magic {:?}, this is no slot file", &bytes[..8]));
        }
        let w = |i: usize| {
            let a = 8 + i * 8;
            u64::from_le_bytes(bytes[a..a + 8].try_into().unwrap())
        };
        // the version decides the layout of everything after it, so it is read and
        // judged first; a format 1 file is refused by NAME (Crow #300 C3), not parsed
        let version = w(0);
        if version == FORMAT_VERSION_NO_FINGERPRINT {
            return Err(format!(
                "slot file format version {version} carries no model fingerprint (written before Crow #300 C3); \
                 this build writes {FORMAT_VERSION} and refuses to guess which model wrote it - save the slot again"
            ));
        }
        let h = Header {
            format_version: version,
            model_family: w(1),
            geo_hash: w(2),
            load_id: w(3),
            n_ctx: w(4),
            prompt_chunk: w(5),
            qsa_ring_rows: w(6),
            gdn_layers: w(7),
            attn_layers: w(8),
            kv_groups: w(9),
            kv_row_bytes: w(10),
            pooled_row_bytes: w(11),
            state_bytes: w(12),
            pos: w(13),
            done_blocks: w(14),
            history_len: w(15),
        };
        if h.format_version != FORMAT_VERSION {
            return Err(format!(
                "slot file format version {}, this build writes {FORMAT_VERSION}",
                h.format_version
            ));
        }
        Ok(h)
    }

    /// - bytes after the header, from the shape alone
    /// - `None` when the header's own numbers do not fit a u64 (a hostile header)
    pub fn payload_bytes_checked(&self) -> Option<u64> {
        let kv = self.kv_groups.checked_mul(self.pos)?.checked_mul(self.kv_row_bytes)?;
        let pooled =
            self.attn_layers.checked_mul(self.done_blocks)?.checked_mul(self.pooled_row_bytes)?;
        let ids = self.history_len.checked_mul(8)?;
        self.state_bytes.checked_add(kv)?.checked_add(pooled)?.checked_add(ids)
    }

    /// - bytes after the header; saturates instead of wrapping, so a bad header can never
    ///   name a SMALL payload by overflowing (release builds have no overflow checks)
    pub fn payload_bytes(&self) -> u64 {
        self.payload_bytes_checked().unwrap_or(u64::MAX)
    }

    /// header plus payload, the exact length a good file has; `None` on the same overflow
    pub fn file_bytes_checked(&self) -> Option<u64> {
        self.payload_bytes_checked()?.checked_add(HEADER_BYTES as u64)
    }

    /// header plus payload, the exact length a good file has; saturating, as `payload_bytes`
    pub fn file_bytes(&self) -> u64 {
        self.file_bytes_checked().unwrap_or(u64::MAX)
    }

    /// - the MODEL first (Crow #300 C3): family, then the geometry fingerprint, each
    ///   refusal naming both sides
    /// - then every field that describes the LOAD must agree; `pos` and its two friends may not
    /// - `Err` names the first field that differs, with both values
    pub fn shape_matches(&self, live: &Header) -> Result<(), String> {
        let family = |code: u64| match Family::from_code(code) {
            Some(f) => format!("{f:?} ({})", f.model_type()),
            None => format!("unknown family code {code}"),
        };
        if self.model_family != live.model_family {
            return Err(format!(
                "slot file model family {}, this engine runs {}; a slot file is only valid for the \
                 model that wrote it (Crow #300)",
                family(self.model_family),
                family(live.model_family)
            ));
        }
        if self.geo_hash != live.geo_hash {
            return Err(format!(
                "slot file model geometry fingerprint {:#018x}, this engine's {} geometry is {:#018x}; \
                 a slot file is only valid for the model that wrote it (Crow #300)",
                self.geo_hash,
                family(live.model_family),
                live.geo_hash
            ));
        }
        let rows: [(&str, u64, u64); 10] = [
            ("load id", self.load_id, live.load_id),
            ("n_ctx", self.n_ctx, live.n_ctx),
            ("prompt chunk", self.prompt_chunk, live.prompt_chunk),
            ("QSA ring rows", self.qsa_ring_rows, live.qsa_ring_rows),
            ("GDN layers", self.gdn_layers, live.gdn_layers),
            ("attention layers", self.attn_layers, live.attn_layers),
            ("KV groups", self.kv_groups, live.kv_groups),
            ("KV row bytes", self.kv_row_bytes, live.kv_row_bytes),
            ("pooled row bytes", self.pooled_row_bytes, live.pooled_row_bytes),
            ("state bytes", self.state_bytes, live.state_bytes),
        ];
        for (name, file, here) in rows {
            if file != here {
                return Err(format!(
                    "slot file {name} {file}, this engine load has {here}; a slot file is only \
                     valid for the shape that wrote it (spec 7.6)"
                ));
            }
        }
        Ok(())
    }

    /// - every CONTENT check a restore owes, pure so it runs without a device
    /// - `self` is the file's header, `live` the geometry of this load
    /// - run AFTER `shape_matches` and BEFORE the first byte is uploaded
    ///
    /// | case | why it is refused |
    /// |---|---|
    /// | `pos == 0` | there is nothing to restore |
    /// | `pos > n_ctx` | the KV upload would run past the end of the KV buffer |
    /// | `done_blocks != pos / 4` | it is an UPLOAD LENGTH for a buffer of `ceil(n_ctx/4)` blocks |
    /// | `history_len != pos` | the id list would not describe the restored position |
    /// | payload size over u64 | a hostile header must not name a small payload by wrapping |
    pub fn check_content(&self, live: &Header) -> Result<(), String> {
        if self.pos == 0 {
            return Err("the file holds position 0, so there is nothing to restore".to_string());
        }
        if self.pos > live.n_ctx {
            return Err(format!("file position {} over n_ctx {}", self.pos, live.n_ctx));
        }
        // gen.rs:1648 and gen.rs:2696 both set done_blocks = floor(pos / 4), so this is an
        // equality, not a bound. It is the ONLY thing between a crafted header and a device
        // write past `qsa_pooled`, which holds ceil(n_ctx / 4) blocks (manager.rs:216-219).
        // Crow #300 phase 2: a model without pooled blocks (full attention, pooled_row_bytes 0)
        // keeps done_blocks at 0
        let want = if live.pooled_row_bytes == 0 { 0 } else { self.pos / 4 };
        if self.done_blocks != want {
            return Err(format!(
                "file done blocks {}, position {} needs exactly {want} (floor of pos/4, gen.rs:1648; 0 without pooled blocks)",
                self.done_blocks,
                self.pos,
            ));
        }
        if self.history_len != self.pos {
            return Err(format!(
                "file history {} does not match position {}",
                self.history_len, self.pos
            ));
        }
        if self.file_bytes_checked().is_none() {
            return Err("the header names a payload size that does not fit a u64".to_string());
        }
        Ok(())
    }
}

/// bytes a `filename` may consist of; everything else is refused
fn plain_byte(b: u8) -> bool {
    b.is_ascii_alphanumeric() || b == b'.' || b == b'_' || b == b'-'
}

/// - `true` for `NUL`, `con.bin`, `COM1`, `lpt9.slot.bin` and every case of them
/// - `NUL` opens the null device: a save would write NOTHING and still answer 200
/// - Windows resolves a device name whatever the extension, so the BASE name decides
fn reserved_device_name(name: &str) -> bool {
    let base = name.split('.').next().unwrap_or(name).to_ascii_uppercase();
    match base.as_str() {
        "CON" | "PRN" | "AUX" | "NUL" => true,
        // COM0..COM9 and LPT0..LPT9, and nothing longer
        _ => {
            base.len() == 4
                && (base.starts_with("COM") || base.starts_with("LPT"))
                && base.as_bytes()[3].is_ascii_digit()
        }
    }
}

/// - a bare file name: no separator, no `..`, only `[A-Za-z0-9._-]`
/// - `Err` carries the operator message, so the 400 body says which rule bit
pub fn sanitize_filename(name: &str) -> Result<&str, String> {
    if name.is_empty() {
        return Err("filename is empty".to_string());
    }
    if name.len() > 255 {
        return Err(format!("filename is {} bytes, over the 255 byte limit", name.len()));
    }
    if name == "." || name == ".." {
        return Err(format!("filename {name:?} is a directory name, not a file name"));
    }
    if let Some(b) = name.bytes().find(|b| !plain_byte(*b)) {
        return Err(format!(
            "filename {name:?} holds byte {b:#04x}; only [A-Za-z0-9._-] is allowed, so no path \
             separator and no directory can be named"
        ));
    }
    if reserved_device_name(name) {
        return Err(format!(
            "filename {name:?} names a reserved Windows device (CON, PRN, AUX, NUL, COM0-9, \
             LPT0-9, with or without an extension); it would open the device, not a file"
        ));
    }
    Ok(name)
}

/// - fnv1a-64 of the container path, mixed with the hot set target
/// - see the module doc: an identity of the LOAD, not a content hash
pub fn load_id(model_path: &str, n_hot: usize) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for b in model_path.bytes().chain((n_hot as u64).to_le_bytes()) {
        h ^= b as u64;
        h = h.wrapping_mul(0x0000_0100_0000_01b3);
    }
    h
}

// --------------------------------------------------------------- device side

/// what one save wrote
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Saved {
    /// `n_saved` on the wire: the prefill clean position the file holds
    pub n_saved: usize,
    /// `n_written` on the wire: the bytes the writer actually took, counted
    pub n_written: u64,
    /// wall of the whole save in ms
    pub ms: f64,
}

/// what one restore read
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Restored {
    /// `n_restored` on the wire: the position now held, equal to the file's `pos`
    pub n_restored: usize,
    /// `n_read` on the wire: the header plus the payload actually read, counted
    pub n_read: u64,
    /// wall of the whole restore in ms
    pub ms: f64,
}

/// the geometry of THIS load, with `pos`, `done_blocks` and `history_len` of the held slot
pub fn live_header(eng: &Engine, cache: &PrefixCache, model_path: &str) -> Header {
    let shape = cache.shape();
    let geo = &eng.geo;
    Header {
        format_version: FORMAT_VERSION,
        model_family: geo.family.code(),
        geo_hash: geo.fingerprint(),
        load_id: load_id(model_path, eng.cfg.n_hot),
        n_ctx: eng.st.context as u64,
        prompt_chunk: eng.cfg.prompt_chunk as u64,
        qsa_ring_rows: eng.st.qsa_ring_rows as u64,
        gdn_layers: shape.gdn_layers as u64,
        attn_layers: shape.attn_layers as u64,
        kv_groups: (shape.attn_layers * 2 * geo.kv_heads) as u64,
        kv_row_bytes: eng.st.kv.row_bytes(geo.head_dim) as u64,
        pooled_row_bytes: crate::cache::pooled_row_bytes(geo) as u64,
        state_bytes: shape.snapshot_bytes() as u64,
        pos: 0,
        done_blocks: 0,
        history_len: 0,
    }
}

/// f32 slice as the bytes the file holds (x86_64 native little endian)
fn f32_bytes(v: &[f32]) -> &[u8] {
    unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, v.len() * 4) }
}

/// the reverse of `f32_bytes`; `src` may be unaligned, `dst` is a `Vec<f32>` and is not
fn bytes_into_f32(src: &[u8], dst: &mut [f32]) {
    assert_eq!(src.len(), dst.len() * 4);
    unsafe { std::ptr::copy_nonoverlapping(src.as_ptr(), dst.as_mut_ptr() as *mut u8, src.len()) };
}

/// - device to host copy of `dst.len()` bytes
/// - `cuMemcpyDtoH_v2` blocks the host, so `dst` is complete when this returns
///
/// # Safety
///
/// - a CUDA context must be current and `src` must hold at least `dst.len()` bytes
/// The KV row groups in file order - layer, then K before V, then kv head:
/// this sequence IS the payload layout, so save and restore both walk it here.
pub(crate) fn kv_row_order(attn_layers: usize, kv_heads: usize) -> impl Iterator<Item = (usize, bool, usize)> {
    // C3: `kv_heads` is the model's `Geo::kv_heads` (2 on Flash-Next)
    (0..attn_layers).flat_map(move |layer| {
        [true, false]
            .into_iter()
            .flat_map(move |is_k| (0..kv_heads).map(move |kvh| (layer, is_k, kvh)))
    })
}

pub(crate) unsafe fn dtoh_bytes(dst: &mut [u8], src: cuda::CUdeviceptr) {
    cuda::ck(sys::cuMemcpyDtoH_v2(
        dst.as_mut_ptr() as *mut std::ffi::c_void,
        src,
        dst.len(),
    ));
}

/// - the sibling name a save writes before it renames over `path`
/// - the pid keeps two serve processes on one directory out of each other's way
fn temp_path(path: &std::path::Path) -> Result<std::path::PathBuf, String> {
    let name = path.file_name().ok_or_else(|| format!("{path:?} names no file"))?;
    let mut tmp = name.to_os_string();
    tmp.push(format!(".part-{}", std::process::id()));
    Ok(path.with_file_name(tmp))
}

/// removes the half written temp file when a save leaves early; `mem::forget` keeps it
struct TempFile<'a>(&'a std::path::Path);

impl Drop for TempFile<'_> {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(self.0);
    }
}

/// - one write, and the byte count that follows from it
/// - `n` is what the file HOLDS, not what the header formula predicts
fn put(
    w: &mut impl Write,
    bytes: &[u8],
    n: &mut u64,
    path: &std::path::Path,
) -> Result<(), String> {
    w.write_all(bytes).map_err(|e| format!("write to {path:?} failed: {e}"))?;
    *n += bytes.len() as u64;
    Ok(())
}

/// - write the held prefill clean state to `path` (see the module doc for the layout)
/// - reads the device, never writes it; the engine state is not touched
///
/// # Safety
///
/// - a CUDA context must be current, as for every other engine call
/// - no kernel of this engine may be in flight on another thread
pub unsafe fn save(
    eng: &Engine,
    cache: &PrefixCache,
    model_path: &str,
    path: &std::path::Path,
) -> Result<Saved, String> {
    let t0 = std::time::Instant::now();
    if !cache.enabled() {
        return Err("the prefix cache is off (CROW_PREFIX_CACHE=0), so no state is held".to_string());
    }
    let (pos, done_blocks) = cache
        .prompt_slot()
        .ok_or_else(|| "no prefill clean state is held; run one chat request first".to_string())?;
    if pos == 0 {
        return Err("the held state is empty (position 0)".to_string());
    }
    if eng.history.len() < pos {
        return Err(format!(
            "engine history {} shorter than the held position {pos}",
            eng.history.len()
        ));
    }
    let mut h = live_header(eng, cache, model_path);
    h.pos = pos as u64;
    h.done_blocks = done_blocks as u64;
    h.history_len = pos as u64;

    let blocks = cache.prompt_state_blocks();
    let state: u64 = blocks.iter().map(|b| (b.len() * 4) as u64).sum();
    if state != h.state_bytes {
        return Err(format!("held state {state} B, the shape says {} B", h.state_bytes));
    }

    // #32 review: the previous good file is not touched until the new one is whole. A
    // disk-full or a device error used to leave a truncated file where a good one had been,
    // and every later restore refused it. `fs::rename` is atomic on one volume, and the temp
    // name is a sibling, so the rename never crosses a volume.
    let tmp = temp_path(path)?;
    let guard = TempFile(&tmp);
    let f = std::fs::File::create(&tmp).map_err(|e| format!("cannot write {tmp:?}: {e}"))?;
    let mut w = std::io::BufWriter::with_capacity(1 << 20, f);
    // MEASURED, never computed: what the writer actually took, byte for byte
    let mut n_written: u64 = 0;
    put(&mut w, &h.encode(), &mut n_written, &tmp)?;
    for block in blocks {
        put(&mut w, f32_bytes(block), &mut n_written, &tmp)?;
    }

    // whatever the last request left in flight must land before the copies read it
    cuda::sync();
    let mut kv_row = vec![0u8; pos * h.kv_row_bytes as usize];
    for (layer, is_k, kvh) in kv_row_order(h.attn_layers as usize, eng.geo.kv_heads) {
        dtoh_bytes(&mut kv_row, eng.st.kv_row_ptr(layer, is_k, kvh, 0));
        put(&mut w, &kv_row, &mut n_written, &tmp)?;
    }
    let mut pooled = vec![0u8; done_blocks * h.pooled_row_bytes as usize];
    // full attention pools nothing (pooled_row_bytes 0, no `qsa_pooled` buffers)
    for layer in 0..if h.pooled_row_bytes == 0 { 0 } else { h.attn_layers as usize } {
        dtoh_bytes(&mut pooled, eng.st.qsa_pooled[layer]);
        put(&mut w, &pooled, &mut n_written, &tmp)?;
    }
    for id in &eng.history[..pos] {
        put(&mut w, &id.to_le_bytes(), &mut n_written, &tmp)?;
    }
    let f = w.into_inner().map_err(|e| format!("write to {tmp:?} failed: {e}"))?;
    // flush AND fsync: a device error must surface here, before the rename, not after it
    f.sync_all().map_err(|e| format!("flush of {tmp:?} failed: {e}"))?;
    drop(f);
    if n_written != h.file_bytes() {
        return Err(format!("wrote {n_written} B, the header says {} B", h.file_bytes()));
    }
    // the writer is closed, so Windows lets the rename replace the previous good file
    std::fs::rename(&tmp, path).map_err(|e| format!("cannot move {tmp:?} to {path:?}: {e}"))?;
    std::mem::forget(guard);
    let on_disk = std::fs::metadata(path).map_err(|e| format!("cannot stat {path:?}: {e}"))?.len();
    if on_disk != n_written {
        return Err(format!("{path:?} holds {on_disk} B, {n_written} B were written"));
    }
    Ok(Saved { n_saved: pos, n_written, ms: t0.elapsed().as_secs_f64() * 1e3 })
}

/// - load `path` into the engine and into the `SLOT_PROMPT` slot of `cache`
/// - every check runs before the first upload, so a refusal leaves the engine untouched
///
/// # Safety
///
/// - a CUDA context must be current, as for every other engine call
/// - no kernel of this engine may be in flight on another thread
pub unsafe fn restore(
    eng: &mut Engine,
    cache: &mut PrefixCache,
    model_path: &str,
    path: &std::path::Path,
) -> Result<Restored, String> {
    let t0 = std::time::Instant::now();
    if !cache.enabled() {
        return Err(
            "the prefix cache is off (CROW_PREFIX_CACHE=0), so there is no slot to restore into"
                .to_string(),
        );
    }
    let live = live_header(eng, cache, model_path);
    let mut f = std::fs::File::open(path).map_err(|e| format!("cannot read {path:?}: {e}"))?;
    let mut head = [0u8; HEADER_BYTES];
    f.read_exact(&mut head)
        .map_err(|e| format!("{path:?} is shorter than the {HEADER_BYTES} byte header: {e}"))?;
    let h = Header::decode(&head)?;
    h.shape_matches(&live)?;
    h.check_content(&live)?;
    let want = h.payload_bytes();
    // #32 review: BOUNDED, and STREAMED since the Linux port (issue #15). The length
    // on disk is the whole check, so a padded or truncated file is refused here, before
    // the first upload, and the payload never needs to live in host RAM at once: a
    // full-context restore used to be one 2.70 GiB anonymous allocation on top of the
    // 47.7 GiB steady state, the worst instantaneous host-RAM event in the engine.
    let on_disk = std::fs::metadata(path).map_err(|e| format!("cannot stat {path:?}: {e}"))?.len();
    let held_bytes = on_disk.saturating_sub(HEADER_BYTES as u64);
    if held_bytes != want {
        let held = if held_bytes > want { "more than" } else { "only" };
        return Err(format!(
            "{path:?} holds {held} {held_bytes} payload bytes, the header asks for {want}"
        ));
    }
    let state: u64 = cache.prompt_state_blocks().iter().map(|b| (b.len() * 4) as u64).sum();
    if state != h.state_bytes {
        return Err(format!("this load holds {state} B of state, the file says {}", h.state_bytes));
    }

    // from here the engine is written; every refusal above left it untouched
    cuda::sync();
    // a restore REPLACES the held conversation: the other snapshots now name a different
    // history and must not stay reuse candidates (they were only valid for the state the
    // file overwrites).
    cache.invalidate();
    // the A4 ordering, exactly as `cache::PrefixCache::rollback`: the uploads below and the
    // prefill of the next request need the legacy stream (`reset.rs`, "The active stream")
    eng.drop_decode_graph();
    // one reusable buffer, sized by the largest single consumer (a KV row group at
    // full context, about 51 MB), instead of the whole payload
    let mut buf: Vec<u8> = Vec::new();
    let mut n_payload: u64 = 0;
    let fill = |f: &mut std::fs::File, buf: &mut Vec<u8>, n: usize, n_payload: &mut u64| -> Result<(), String> {
        buf.resize(n, 0);
        f.read_exact(buf).map_err(|e| format!("read {path:?} failed: {e}"))?;
        *n_payload += n as u64;
        Ok(())
    };
    for block in cache.prompt_state_blocks_mut() {
        let n = block.len() * 4;
        fill(&mut f, &mut buf, n, &mut n_payload)?;
        bytes_into_f32(&buf, block);
    }

    let kv_bytes = h.pos as usize * h.kv_row_bytes as usize;
    for (layer, is_k, kvh) in kv_row_order(h.attn_layers as usize, eng.geo.kv_heads) {
        fill(&mut f, &mut buf, kv_bytes, &mut n_payload)?;
        cuda::upload_into(eng.st.kv_row_ptr(layer, is_k, kvh, 0), &buf);
    }
    let pooled_bytes = h.done_blocks as usize * h.pooled_row_bytes as usize;
    // full attention pools nothing (pooled_row_bytes 0, no `qsa_pooled` buffers)
    for layer in 0..if h.pooled_row_bytes == 0 { 0 } else { h.attn_layers as usize } {
        fill(&mut f, &mut buf, pooled_bytes, &mut n_payload)?;
        cuda::upload_into(eng.st.qsa_pooled[layer], &buf);
    }
    fill(&mut f, &mut buf, h.history_len as usize * 8, &mut n_payload)?;
    let ids: Vec<i64> = buf
        .chunks_exact(8)
        .map(|c| i64::from_le_bytes(c.try_into().unwrap()))
        .collect();
    eng.pos = h.pos as usize;
    eng.done_blocks = h.done_blocks as usize;
    eng.history = ids;
    // #114: a slot file carries ids, not image identities: an image in the restored
    // history can never match a request's image (`cache::common_prefix_len_mm`)
    eng.history_images.clear();
    eng.route_log.clear();
    // #32 review: LAST. Every upload above is a `cuda::ck`, which ends the process on a
    // device error (serve has no catch_unwind), so naming the slot before them could leave
    // a cache that claims a position the device never received. It cannot now.
    cache.set_prompt_slot(h.pos as usize, h.done_blocks as usize);

    // the uploads read `buf`, which dies with this frame
    cuda::sync();
    Ok(Restored {
        n_restored: h.pos as usize,
        // MEASURED: the header this reader consumed plus the payload it actually read
        n_read: HEADER_BYTES as u64 + n_payload,
        ms: t0.elapsed().as_secs_f64() * 1e3,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::geo::{Geo, NKV};

    /// Crow #300 C3: a format 1 slot file (120 bytes of header, no model fingerprint)
    /// is refused by NAME - it is never parsed with the format 2 layout
    #[test]
    fn a_format_one_slot_file_is_refused_by_name() {
        // the pre-C3 layout: magic, then fourteen u64 starting with version 1
        let mut b = MAGIC.to_vec();
        for w in [1u64, 0x0123_4567_89ab_cdef, 200_000, 2048, 2052, 36, 12, 48, 256, 512, 130_646_016, 16_064, 4016, 16_064] {
            b.extend_from_slice(&w.to_le_bytes());
        }
        b.resize(HEADER_BYTES, 0); // the restore reads a full format 2 head
        let e = Header::decode(&b).unwrap_err();
        assert!(e.contains("format version 1") && e.contains("no model fingerprint"), "{e}");
    }

    /// Crow #300 C3: a slot file written by another model is refused by name even
    /// when every buffer size agrees - the family first, then the geometry
    #[test]
    fn a_slot_file_of_another_model_is_refused_by_name() {
        let mut h = live();
        h.model_family = Family::Qwen35Dense.code();
        let e = h.shape_matches(&live()).unwrap_err();
        assert!(e.contains("Qwen35Dense (qwen3_5_text)") && e.contains("FlashNext (qwen4_exp_text)"), "{e}");
        // same family, any other geometry: the fingerprint catches it
        let mut other = Geo::FLASH_NEXT;
        other.rope_theta = 1e6;
        assert_ne!(other.fingerprint(), Geo::FLASH_NEXT.fingerprint());
        let mut h = live();
        h.geo_hash = other.fingerprint();
        let e = h.shape_matches(&live()).unwrap_err();
        assert!(e.contains("geometry fingerprint") && e.contains("FlashNext"), "{e}");
        // and a header that encodes a family no build knows names the code
        let mut h = live();
        h.model_family = 99;
        let e = h.shape_matches(&live()).unwrap_err();
        assert!(e.contains("unknown family code 99"), "{e}");
    }

    /// the file layout: kv_groups groups, layer-major, K before V, kv head ascending.
    /// A save and the restore that reads it back walk this same sequence, so a
    /// reordering here would silently transpose every restored KV row.
    #[test]
    fn kv_row_order_is_the_file_layout() {
        let h = live();
        let got: Vec<(usize, bool, usize)> = kv_row_order(h.attn_layers as usize, NKV).collect();
        assert_eq!(got.len() as u64, h.kv_groups, "one group per (layer, k/v, kv head)");
        assert_eq!(&got[..4], &[(0, true, 0), (0, true, 1), (0, false, 0), (0, false, 1)]);
        let last = h.attn_layers as usize - 1;
        assert_eq!(got[got.len() - 1], (last, false, NKV - 1));
        // and the payload the header sizes is exactly what that walk writes
        assert_eq!(
            got.len() as u64 * h.pos * h.kv_row_bytes,
            h.kv_groups * h.pos * h.kv_row_bytes
        );
    }

    /// the shape of the gate's operating point: chunk 2048, ring 2052, FP8 KV
    fn live() -> Header {
        Header {
            format_version: FORMAT_VERSION,
            model_family: Family::FlashNext.code(),
            geo_hash: Geo::FLASH_NEXT.fingerprint(),
            load_id: 0x0123_4567_89ab_cdef,
            n_ctx: 200_000,
            prompt_chunk: 2048,
            qsa_ring_rows: 2052,
            gdn_layers: 36,
            attn_layers: 12,
            kv_groups: 48,
            kv_row_bytes: 256,
            pooled_row_bytes: 512,
            state_bytes: 130_646_016,
            pos: 16_064,
            done_blocks: 4016,
            history_len: 16_064,
        }
    }

    #[test]
    fn a_header_is_exactly_one_hundred_and_thirty_six_bytes() {
        assert_eq!(HEADER_BYTES, 136);
        assert_eq!(live().encode().len(), HEADER_BYTES);
    }

    #[test]
    fn a_header_survives_encode_and_decode() {
        let h = live();
        assert_eq!(Header::decode(&h.encode()), Ok(h));
    }

    #[test]
    fn the_magic_is_the_first_eight_bytes() {
        let b = live().encode();
        assert_eq!(&b[..8], &MAGIC);
    }

    #[test]
    fn a_foreign_magic_is_refused() {
        let mut b = live().encode();
        b[0] = b'X';
        let e = Header::decode(&b).unwrap_err();
        assert!(e.contains("magic"), "{e}");
    }

    #[test]
    fn a_head_shorter_than_the_header_is_refused() {
        let b = live().encode();
        let e = Header::decode(&b[..HEADER_BYTES - 1]).unwrap_err();
        assert!(e.contains("119") || e.contains("short"), "{e}");
    }

    #[test]
    fn a_foreign_format_version_is_refused() {
        let mut h = live();
        h.format_version = FORMAT_VERSION + 1;
        let e = Header::decode(&h.encode()).unwrap_err();
        assert!(e.contains("format version"), "{e}");
    }

    #[test]
    fn the_payload_size_is_the_sum_of_the_seven_parts() {
        let h = live();
        // 130,646,016 state + 48 * 16064 * 256 KV + 12 * 4016 * 512 pooled + 16064 * 8 history
        let want = 130_646_016u64 + 48 * 16_064 * 256 + 12 * 4016 * 512 + 16_064 * 8;
        assert_eq!(h.payload_bytes(), want);
        assert_eq!(h.file_bytes(), want + HEADER_BYTES as u64);
    }

    #[test]
    fn an_empty_slot_file_still_has_a_payload_of_the_recurrent_state() {
        let mut h = live();
        h.pos = 0;
        h.done_blocks = 0;
        h.history_len = 0;
        assert_eq!(h.payload_bytes(), h.state_bytes);
    }

    #[test]
    fn the_same_shape_matches_itself() {
        assert_eq!(live().shape_matches(&live()), Ok(()));
    }

    /// `pos`, `done_blocks` and `history_len` describe the CONTENT, not the load
    #[test]
    fn a_different_position_is_not_a_shape_difference() {
        let mut h = live();
        h.pos = 12;
        h.done_blocks = 3;
        h.history_len = 12;
        assert_eq!(h.shape_matches(&live()), Ok(()));
    }

    #[test]
    fn every_shape_field_is_checked() {
        let fields: Vec<(&str, fn(&mut Header))> = vec![
            ("model family", |h| h.model_family = Family::Qwen35Dense.code()),
            ("geometry fingerprint", |h| h.geo_hash ^= 1),
            ("load", |h| h.load_id ^= 1),
            ("n_ctx", |h| h.n_ctx += 1),
            ("prompt chunk", |h| h.prompt_chunk = 512),
            ("QSA ring rows", |h| h.qsa_ring_rows = 516),
            ("GDN layers", |h| h.gdn_layers = 35),
            ("attention layers", |h| h.attn_layers = 11),
            ("KV groups", |h| h.kv_groups = 44),
            ("KV row bytes", |h| h.kv_row_bytes = 512),
            ("pooled row bytes", |h| h.pooled_row_bytes = 256),
            ("state bytes", |h| h.state_bytes = 1),
        ];
        for (name, bend) in fields {
            let mut h = live();
            bend(&mut h);
            let e = h.shape_matches(&live()).unwrap_err();
            assert!(e.contains(name), "field {name}: message was {e:?}");
        }
    }

    #[test]
    fn a_good_header_passes_the_content_checks() {
        assert_eq!(live().check_content(&live()), Ok(()));
        let mut h = live();
        h.pos = 12;
        h.done_blocks = 3;
        h.history_len = 12;
        assert_eq!(h.check_content(&live()), Ok(()));
        // floor, not ceil: gen.rs:1647 sets new_done = (pos_base + t) / 4
        h.pos = 13;
        h.history_len = 13;
        assert_eq!(h.check_content(&live()), Ok(()));
    }

    /// every content branch, one bend at a time, and the message names the field
    #[test]
    fn every_content_check_refuses_and_names_the_case() {
        let cases: Vec<(&str, fn(&mut Header))> = vec![
            ("position 0", |h| {
                h.pos = 0;
                h.done_blocks = 0;
                h.history_len = 0;
            }),
            ("n_ctx", |h| {
                h.pos = 200_001;
                h.done_blocks = 50_000;
                h.history_len = 200_001;
            }),
            ("done blocks", |h| h.done_blocks += 1),
            ("done blocks", |h| h.done_blocks -= 1),
            ("history", |h| h.history_len += 1),
        ];
        for (name, bend) in cases {
            let mut h = live();
            bend(&mut h);
            let e = h.check_content(&live()).unwrap_err();
            assert!(e.contains(name), "case {name}: message was {e:?}");
        }
    }

    /// Crow #300 phase 2: a model without pooled blocks (full attention: pooled_row_bytes 0)
    /// keeps done_blocks at 0; the file's pooled payload is empty and the KV groups count
    /// the Geo's attention layers (`Shape::of`), not the QSA buffers (of which it has none)
    #[test]
    fn a_full_attention_slot_has_no_pooled_blocks_and_keeps_its_kv_rows() {
        let g = crate::meta::dense_fixture_geo();
        assert_eq!(crate::cache::pooled_row_bytes(&g), 0);
        assert_eq!(crate::cache::pooled_row_bytes(&Geo::FLASH_NEXT), 512, "the QSA raw key width x 4");
        let mut live = live();
        live.pooled_row_bytes = 0;
        live.attn_layers = g.attn_layers as u64;
        live.kv_groups = (g.attn_layers * 2 * g.kv_heads) as u64;
        let mut h = live;
        h.pos = 12;
        h.history_len = 12;
        h.done_blocks = 0;
        assert_eq!(h.check_content(&live), Ok(()));
        h.done_blocks = 3;
        assert!(h.check_content(&live).unwrap_err().contains("0 without pooled blocks"));
        h.done_blocks = 0;
        // KV rows of every attention layer are in the payload; pooled rows are not
        let kv = h.pos * h.kv_row_bytes * h.kv_groups;
        assert_eq!(h.payload_bytes(), h.state_bytes + kv + h.history_len * 8);
    }

    /// the review finding: `done_blocks` is a device upload length for a buffer sized
    /// `ceil(n_ctx/4)` blocks, so a header naming more than `pos / 4` writes out of bounds
    #[test]
    fn a_done_blocks_count_that_would_overrun_the_pooled_buffer_is_refused() {
        let mut h = live();
        h.pos = 1;
        h.history_len = 1;
        h.done_blocks = 200_000;
        let e = h.check_content(&live()).unwrap_err();
        assert!(e.contains("done blocks"), "{e}");
        // and the same header with a file of exactly the length it names is still refused
        assert_eq!(h.payload_bytes_checked(), Some(h.payload_bytes()));
    }

    #[test]
    fn a_payload_size_that_does_not_fit_a_u64_is_refused_not_wrapped() {
        let mut h = live();
        h.pos = u64::MAX / 8;
        h.history_len = h.pos;
        h.done_blocks = h.pos / 4;
        assert_eq!(h.payload_bytes_checked(), None);
        assert_eq!(h.file_bytes_checked(), None);
        // the saturating views never name a SMALL payload for a hostile header
        assert_eq!(h.payload_bytes(), u64::MAX);
        assert_eq!(h.file_bytes(), u64::MAX);
    }

    /// `{"filename":"NUL"}` opened the null device, so a save wrote nothing and answered 200
    #[test]
    fn a_windows_reserved_device_name_is_refused() {
        for bad in [
            "NUL", "nul", "Nul", "CON", "con.bin", "PRN", "AUX", "COM1", "com9.bin", "LPT1",
            "lpt9", "NUL.tmp", "aux.slot.bin",
        ] {
            let e = sanitize_filename(bad).unwrap_err();
            assert!(e.contains("reserved"), "{bad:?}: {e}");
        }
        // names that only LOOK reserved stay allowed
        for good in ["nulls.bin", "console.bin", "com.bin", "com10.bin", "lpt.bin", "auxiliary"] {
            assert_eq!(sanitize_filename(good), Ok(good), "{good:?} was refused");
        }
    }

    #[test]
    fn a_plain_file_name_passes() {
        assert_eq!(sanitize_filename("crow-slot.bin"), Ok("crow-slot.bin"));
        assert_eq!(sanitize_filename("crow-session.bin"), Ok("crow-session.bin"));
        assert_eq!(sanitize_filename("a_1.BIN"), Ok("a_1.BIN"));
    }

    #[test]
    fn a_path_separator_is_refused() {
        for bad in ["../x.bin", "..\\x.bin", "a/b.bin", "a\\b.bin", "/etc/passwd", "C:\\x.bin"] {
            assert!(sanitize_filename(bad).is_err(), "{bad:?} passed");
        }
    }

    #[test]
    fn the_two_dot_names_are_refused() {
        assert!(sanitize_filename(".").is_err());
        assert!(sanitize_filename("..").is_err());
    }

    #[test]
    fn an_empty_or_wild_name_is_refused() {
        for bad in ["", "a b.bin", "a\0b", "a\nb", "x*.bin", "\u{00e4}.bin", "a?b"] {
            assert!(sanitize_filename(bad).is_err(), "{bad:?} passed");
        }
    }

    #[test]
    fn a_name_over_two_hundred_and_fifty_five_bytes_is_refused() {
        let long = "a".repeat(256);
        assert!(sanitize_filename(&long).is_err());
        let ok = "a".repeat(255);
        assert_eq!(sanitize_filename(&ok), Ok(ok.as_str()));
    }

    #[test]
    fn the_load_id_separates_two_containers() {
        let a = load_id("converter/M.cnq", 160);
        let b = load_id("converter/L.cnq", 160);
        let c = load_id("converter/M.cnq", 128);
        assert_ne!(a, b);
        assert_ne!(a, c);
        assert_eq!(a, load_id("converter/M.cnq", 160));
        assert_ne!(a, 0);
    }
}
