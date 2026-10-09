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
//!   `FILE_FLAG_NO_BUFFERING | FILE_FLAG_OVERLAPPED` and issues EVERY read of its share before it
//!   drains the first completion, so all slabs of a fetch are in flight at once. The completion
//!   mechanism is the reader's backend ([`NvmeBackend`], `NvmeConfig::backend` or
//!   `CROW_NVME_BACKEND`): `iocp` (default) binds the handle to the reader's own I/O completion
//!   port; `ioring` gives the reader its own Windows 11 I/O ring (`CreateIoRing`,
//!   `BuildIoRingReadFile`, `SubmitIoRing`, `PopIoRingCompletion`; the API use of
//!   `tools/nvme_read_rate.py` `IoRingReader`, #171). On Linux a reader opens it `O_DIRECT` and
//!   reads with `pread` (synchronous per reader; no io_uring here; `ioring` is refused by name);
//! - runs `residency::sanitize_sf_slab` (scale byte 0x7F -> 0x7E) on both slabs of an NVFP4
//!   record in the destination before [`ColdSource::wait`] returns, exactly as the load path does
//!   (`residency.rs`, every slab), so no NVFP4 record is ever published unsanitized.
//!
//! - #149 path B (`CROW_GLM_STAGER`, [`NvmeSource::fetch_landed`]): a fetch may carry one
//!   **landed flag** per record (a u64 in mapped pinned memory). The reader raises it after its
//!   share is read and sanitized, before the ticket completes, so a device stream can wait on the
//!   flag (`cuStreamWaitValue64`) while the host goes on; a failed read raises it too (no device
//!   wait hangs on it) and its error comes from the ticket.
//!
//! Not built here: the RAM tier behind the trait, and the boot wiring beyond the refusal in
//! `boot.rs` (`CROW_NVME_TIER` together with `CROW_COLD_TIER`). The three-tier split that uses
//! this backend is `glm5_tiers`.
//!
//! The record is a parameter of the container (#159 / #176 / #149, 2026-10-08): a glm5_next
//! container stores each routed expert of each layer as ONE unit (gate, up, down back to back,
//! on a 4096-B file offset), whose size and codec its index gives
//! ([`glm5_record_of_container`]: the codec from the expert tensors' `dtype`, the record from
//! the units' extent in the index offsets, cross-checked against the NVFP4 byte rule where the
//! codec is NVFP4). Such a record is read as one span ([`ExpertRecord::locate_glm5`],
//! [`RecordLayout::OneUnit`]); `sanitize_sf_slab` runs only on NVFP4 records - a MUL1 record
//! has no ue4m3 scale bytes and is delivered byte for byte as stored.

use crate::cnq::{Cnq, TensorInfo};
use crate::geo::{expert_record_refusal, ExpertCodec, ExpertRecordSpec, EXPERT_RECORD_ALIGN, GLM5_NEXT_MODEL_TYPE};
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

/// How one record lies in the container.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RecordLayout {
    /// two slabs of fused expert tensors, gate_up and down (`ExpertRecord::locate`)
    TwoSlabs,
    /// one contiguous unit, gate + up + down back to back (glm5_next, `ExpertRecord::locate_glm5`):
    /// `gu` is the whole record, `dn` and `RecordDst::dn` are unused
    OneUnit,
}

/// One routed expert of one layer: its gate_up slab and its down slab, or (glm5_next) its one
/// record in `gu`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ExpertRecord {
    pub layer: u32,
    pub id: u32,
    pub gu: Span,
    pub dn: Span,
    /// the codec of the stored bytes: NVFP4 is sanitized before publication, MUL1 is not
    pub codec: ExpertCodec,
    pub layout: RecordLayout,
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
        Ok(ExpertRecord { layer, id, gu: spans[0], dn: spans[1], codec: ExpertCodec::Nvfp4, layout: RecordLayout::TwoSlabs })
    }

    /// The glm5_next record of expert `id` in layer `layer`: one span of `spec.bytes` starting at
    /// the expert's gate projection (`model.language_model.layers.{layer}.mlp.experts.{id}.gate_proj.weight`).
    /// Refuses a missing tensor, an overlay tensor, a dtype other than the spec's codec, and a
    /// down projection that does not start inside the record.
    pub fn locate_glm5(cnq: &Cnq, spec: &ExpertRecordSpec, layer: u32, id: u32) -> Result<Self, String> {
        Self::locate_glm5_by(cnq, spec, layer, id, &|name| cnq.tensors.iter().find(|t| t.name == name && t.section == "text"))
    }

    /// #149 / #175: the records of `experts` routed experts of every layer in `layers`, each
    /// [`ExpertRecord::locate_glm5`] with its checks, through one name map of the index (the
    /// linear `find` of `locate_glm5` per record would scan the index 12,096 x 2 times at
    /// GLM-5.3-Flash). `[layer position][expert]`.
    pub fn glm5_table(cnq: &Cnq, spec: &ExpertRecordSpec, layers: &[u32], experts: u32) -> Result<Vec<Vec<Self>>, String> {
        let by_name: std::collections::HashMap<&str, &TensorInfo> =
            cnq.tensors.iter().filter(|t| t.section == "text").map(|t| (t.name.as_str(), t)).collect();
        let find = |name: &str| by_name.get(name).copied();
        layers.iter().map(|&l| (0..experts).map(|e| Self::locate_glm5_by(cnq, spec, l, e, &find)).collect()).collect()
    }

    fn locate_glm5_by<'a>(cnq: &Cnq, spec: &ExpertRecordSpec, layer: u32, id: u32, lookup: &dyn Fn(&str) -> Option<&'a TensorInfo>) -> Result<Self, String> {
        let find = |p: &str| -> Result<&TensorInfo, String> {
            let name = glm5_expert_tensor_name(layer, id, p);
            lookup(&name).ok_or_else(|| format!("{name}: not in the container index"))
        };
        let (gate, down) = (find("gate")?, find("down")?);
        for t in [gate, down] {
            if t.overlay {
                return Err(format!("{}: an overlay tensor is not read by the NVMe tier (it lives in the overlay file)", t.name));
            }
            if t.dtype != spec.codec.dtype() {
                return Err(format!(
                    "{}: dtype {}, the expert record says codec {} - the record spec is not this container's",
                    t.name,
                    t.dtype,
                    spec.codec.dtype()
                ));
            }
        }
        if down.offset < gate.offset || down.offset >= gate.offset + spec.bytes {
            return Err(format!(
                "layer {layer} expert {id}: the down projection at {} lies outside the {} B record at {}",
                down.offset, spec.bytes, gate.offset
            ));
        }
        Ok(ExpertRecord {
            layer,
            id,
            gu: Span { off: cnq.abs_offset(gate, 0), len: spec.bytes as usize },
            dn: Span { off: 0, len: 0 },
            codec: spec.codec,
            layout: RecordLayout::OneUnit,
        })
    }

    /// the spans of the record with their destinations: two slabs, or the one glm5_next unit
    pub fn parts(&self, dst: &RecordDst) -> Vec<(&'static str, Span, *mut u8)> {
        match self.layout {
            RecordLayout::TwoSlabs => vec![("gate_up", self.gu, dst.gu), ("down", self.dn, dst.dn)],
            RecordLayout::OneUnit => vec![("record", self.gu, dst.gu)],
        }
    }
}

/// `model.language_model.layers.{layer}.mlp.experts.{id}.{proj}_proj.weight`, the converter's
/// name of a glm5_next routed-expert projection (`recipe::glm_expert`)
pub fn glm5_expert_tensor_name(layer: u32, id: u32, proj: &str) -> String {
    format!("model.language_model.layers.{layer}.mlp.experts.{id}.{proj}_proj.weight")
}

/// `(layer, expert, projection index 0 gate / 1 up / 2 down)` of a glm5_next routed-expert name
fn glm5_expert_parts(name: &str) -> Option<(u32, u32, usize)> {
    let r = name.strip_prefix("model.language_model.layers.")?;
    let (l, r) = r.split_once('.')?;
    let r = r.strip_prefix("mlp.experts.")?;
    let (e, p) = r.split_once('.')?;
    let k = match p {
        "gate_proj.weight" => 0,
        "up_proj.weight" => 1,
        "down_proj.weight" => 2,
        _ => return None,
    };
    Some((l.parse().ok()?, e.parse().ok()?, k))
}

/// #159 / #149: the routed-expert record of a glm5_next container, from its index tensors.
///
/// - codec: the `dtype` of every routed-expert tensor (one dtype for all; a mix or an unknown
///   dtype is refused by name);
/// - each record is its expert's gate, up and down, in that order, starting on a 4096-B file
///   offset (`blob_offset + offset`);
/// - its size is the unit's extent: from the gate projection to the next tensor of the index
///   after the down projection (or the blob end, `blob_len`). The converter puts zeros only in
///   front of a record, so the extent is exact wherever a record is followed by an unaligned
///   tensor or by the blob end, and a record that is not a whole number of 4096-B sectors shows
///   there and is refused by name;
/// - NVFP4: the record is also the format's own byte rule (`Cnq::byte_len` of the three
///   projections, back to back), refused when the layout disagrees;
/// - every record has the same size, else refused naming two that differ.
///
/// Returns the record and the number of records. Pure: no file is read.
pub fn glm5_record_from_index(tensors: &[TensorInfo], blob_offset: u64, blob_len: u64) -> Result<(ExpertRecordSpec, usize), String> {
    use std::collections::BTreeMap;
    let mut recs: BTreeMap<(u32, u32), [Option<&TensorInfo>; 3]> = BTreeMap::new();
    for t in tensors.iter().filter(|t| t.section == "text" && !t.overlay) {
        if let Some((l, e, k)) = glm5_expert_parts(&t.name) {
            recs.entry((l, e)).or_default()[k] = Some(t);
        }
    }
    let Some(first) = recs.values().flat_map(|r| r.iter().flatten()).next().copied() else {
        return Err("the index carries no glm5_next routed-expert tensor (model.language_model.layers.L.mlp.experts.E.{gate,up,down}_proj.weight) - no expert record to plan or read (#159)".into());
    };
    let dtype = first.dtype.as_str();
    if let Some(other) = recs.values().flat_map(|r| r.iter().flatten()).find(|t| t.dtype != dtype) {
        return Err(format!(
            "refusing the expert record: the routed-expert tensors mix dtypes ({} is {dtype}, {} is {}) - one codec per container (#149)",
            first.name, other.name, other.dtype
        ));
    }
    let codec = ExpertCodec::from_dtype(dtype)?;
    let mut offsets: Vec<u64> = tensors.iter().filter(|t| !t.overlay).map(|t| t.offset).collect();
    offsets.sort_unstable();
    offsets.dedup();
    const PROJ: [&str; 3] = ["gate", "up", "down"];
    let mut sizes: Vec<((u32, u32), u64)> = Vec::with_capacity(recs.len());
    for (&(l, e), r) in &recs {
        let p = r
            .iter()
            .zip(PROJ)
            .map(|(t, name)| t.ok_or_else(|| format!("refusing the expert record: layer {l} expert {e} lacks its {name} projection in the index")))
            .collect::<Result<Vec<&TensorInfo>, String>>()?;
        let (gate, up, down) = (p[0], p[1], p[2]);
        if !(gate.offset < up.offset && up.offset < down.offset) {
            return Err(format!(
                "refusing the expert record: layer {l} expert {e} is not stored gate, up, down back to back (offsets {}, {}, {})",
                gate.offset, up.offset, down.offset
            ));
        }
        let abs = blob_offset + gate.offset;
        if !abs.is_multiple_of(EXPERT_RECORD_ALIGN) {
            return Err(format!(
                "refusing the expert record: layer {l} expert {e} starts at file offset {abs}, not on a {EXPERT_RECORD_ALIGN} B boundary (unbuffered NVMe reads need sector-aligned records, #149)"
            ));
        }
        let next = offsets.iter().copied().find(|&o| o > down.offset).unwrap_or(blob_len);
        let bytes = if codec == ExpertCodec::Nvfp4 {
            let (lg, lu, ld) = (Cnq::byte_len(gate), Cnq::byte_len(up), Cnq::byte_len(down));
            if up.offset != gate.offset + lg || down.offset != up.offset + lu || down.offset + ld > next {
                return Err(format!(
                    "refusing the expert record: layer {l} expert {e} at nvfp4 is {lg} + {lu} + {ld} B, the index layout disagrees (offsets {}, {}, {}, next tensor {next})",
                    gate.offset, up.offset, down.offset
                ));
            }
            lg + lu + ld
        } else {
            next - gate.offset
        };
        if let Some(why) = expert_record_refusal(bytes) {
            return Err(format!("layer {l} expert {e} ({}): {why}", codec.dtype()));
        }
        sizes.push(((l, e), bytes));
    }
    let (k0, b0) = sizes[0];
    if let Some(&(k1, b1)) = sizes.iter().find(|(_, b)| *b != b0) {
        return Err(format!(
            "refusing the expert record: records differ in size (layer {} expert {} is {b0} B, layer {} expert {} is {b1} B) - one record size per container",
            k0.0, k0.1, k1.0, k1.1
        ));
    }
    Ok((ExpertRecordSpec::new(codec, b0)?, sizes.len()))
}

/// #159 / #149: [`glm5_record_from_index`] on a container file. Reads only the index trailer
/// (magic, index length, index JSON), never a weight; refuses a container that is not an index
/// v2 of model type `glm5_next_text`.
pub fn glm5_record_of_container(path: &str) -> Result<(ExpertRecordSpec, usize), String> {
    use std::io::{Read, Seek, SeekFrom};
    let at = |e: std::io::Error| format!("{path}: {e}");
    let mut f = std::fs::File::open(path).map_err(at)?;
    let file_len = f.metadata().map_err(at)?.len();
    let mut magic = [0u8; 4];
    f.read_exact(&mut magic).map_err(at)?;
    if &magic != b"CNQ1" || file_len < 20 {
        return Err(format!("{path}: not a CNQ1 container"));
    }
    f.seek(SeekFrom::Start(file_len - 8)).map_err(at)?;
    let mut b8 = [0u8; 8];
    f.read_exact(&mut b8).map_err(at)?;
    let idx_len = u64::from_le_bytes(b8);
    if idx_len == 0 || idx_len + 20 > file_len {
        return Err(format!("{path}: index length {idx_len} does not fit a {file_len} B file"));
    }
    f.seek(SeekFrom::Start(file_len - 8 - idx_len)).map_err(at)?;
    let mut ib = vec![0u8; idx_len as usize];
    f.read_exact(&mut ib).map_err(at)?;
    let index: serde_json::Value = serde_json::from_slice(&ib).map_err(|e| format!("{path}: index json: {e}"))?;
    match crate::cnq::classify_index(&ib, &index).map_err(|e| format!("{path}: {e}"))? {
        crate::cnq::IndexKind::V2(m) if m.model_type == GLM5_NEXT_MODEL_TYPE => {}
        crate::cnq::IndexKind::V2(m) => {
            return Err(format!("{path}: model_type {}, not {GLM5_NEXT_MODEL_TYPE} - the expert record is the glm5_next family's (#159)", m.model_type))
        }
        crate::cnq::IndexKind::V1OfRecord => return Err(format!("{path}: the Flash-Next container of record, not a glm5_next container (#159)")),
    }
    let blob_offset = index["blob_offset"].as_u64().ok_or_else(|| format!("{path}: index has no blob_offset"))?;
    let blob_len = (file_len - 8 - idx_len).checked_sub(blob_offset).ok_or_else(|| format!("{path}: blob_offset {blob_offset} past the index"))?;
    glm5_record_from_index(&crate::cnq::parse_tensor_index(&index, false), blob_offset, blob_len).map_err(|e| format!("{path}: {e}"))
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
    for (what, s, p) in rec.parts(dst) {
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

/// The environment variable that picks the reader backend when [`NvmeConfig::backend`] is `None`.
pub const BACKEND_ENV: &str = "CROW_NVME_BACKEND";

/// How a reader learns that its reads completed (#149). Both read the same spans into the same
/// destinations with the same checks; only the completion mechanism differs.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum NvmeBackend {
    /// Windows: one I/O completion port per reader. The default. Off Windows this is the
    /// platform's own path (Linux: `O_DIRECT` + `pread`).
    #[default]
    Iocp,
    /// Windows 11 (build 22000+): one I/O ring per reader. Refused by name where the system has
    /// none; never a silent fall-back to `Iocp`.
    IoRing,
}

impl NvmeBackend {
    /// `iocp` or `ioring`; anything else is refused by name.
    pub fn parse(s: &str) -> Result<NvmeBackend, String> {
        match s {
            "iocp" => Ok(NvmeBackend::Iocp),
            "ioring" => Ok(NvmeBackend::IoRing),
            other => Err(format!("{BACKEND_ENV}={other:?}: unknown NVMe reader backend, one of iocp (default), ioring")),
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            NvmeBackend::Iocp => "iocp",
            NvmeBackend::IoRing => "ioring",
        }
    }
}

/// The backend a source opens with: the config's field if set, else `env` (the value of
/// [`BACKEND_ENV`]; unset or empty = [`NvmeBackend::Iocp`]). An unknown value is refused by name.
pub fn resolve_backend(field: Option<NvmeBackend>, env: Option<&str>) -> Result<NvmeBackend, String> {
    match (field, env.map(str::trim)) {
        (Some(b), _) => Ok(b),
        (None, None) | (None, Some("")) => Ok(NvmeBackend::default()),
        (None, Some(v)) => NvmeBackend::parse(v),
    }
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
    /// the reader backend; `None` (default) = [`BACKEND_ENV`], unset = [`NvmeBackend::Iocp`]
    pub backend: Option<NvmeBackend>,
}

impl NvmeConfig {
    pub fn new(path: impl AsRef<Path>) -> Self {
        NvmeConfig { path: path.as_ref().to_path_buf(), readers: 1, affinity: None, backend: None }
    }
}

struct Job {
    rec: ExpertRecord,
    dst: RecordDst,
    /// #149 path B: raised by the reader after this job's share ran ([`NvmeSource::fetch_landed`])
    landed: Option<Landed>,
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
    backend: NvmeBackend,
}

impl NvmeSource {
    /// Open the container once per reader (unbuffered, overlapped on Windows; `O_DIRECT` on
    /// Linux) with the backend of [`resolve_backend`] and start the readers. Any open, backend or
    /// affinity failure is returned by name.
    pub fn open(cfg: &NvmeConfig) -> Result<NvmeSource, String> {
        let backend = resolve_backend(cfg.backend, std::env::var(BACKEND_ENV).ok().as_deref())?;
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
            let mut reader = Reader::open(&cfg.path, backend)?;
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
                        let r = reader.run(&b.jobs);
                        // SAFETY: `fetch_landed`'s contract: every flag is a live u64 until the
                        // ticket is waited on, which cannot happen before the reply below
                        unsafe { raise_landed(&b.jobs) };
                        let _ = b.reply.send(r);
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
        Ok(NvmeSource { tx, threads, backend })
    }

    pub fn readers(&self) -> usize {
        self.tx.len()
    }

    /// the backend every reader of this source runs
    pub fn backend(&self) -> NvmeBackend {
        self.backend
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

/// #149 path B (`CROW_GLM_STAGER`): a u64 in host memory the reader raises to `value` once a
/// record has landed. `flag` is the host address of a mapped pinned word; the device waits on its
/// device alias (`cuStreamWaitValue64`, cyclic greater-or-equal).
#[derive(Clone, Copy, Debug)]
pub struct Landed {
    pub flag: *mut u64,
    pub value: u64,
}

/// Raise the landed flag of every job of a share. Called by the reader after [`Reader::run`]
/// returned, so after the last byte of the share and its sanitize; also when the run failed, so a
/// device waiting on a flag never waits for a read that will not come (the error reaches the host
/// through the ticket).
///
/// # Safety
///
/// Every flag of `jobs` is a live, 8-B aligned u64 nobody else writes.
unsafe fn raise_landed(jobs: &[Job]) {
    if jobs.iter().all(|j| j.landed.is_none()) {
        return;
    }
    // the record bytes (written by the device's DMA, then sanitized on this thread) before the flag
    std::sync::atomic::fence(std::sync::atomic::Ordering::Release);
    for l in jobs.iter().filter_map(|j| j.landed) {
        std::ptr::write_volatile(l.flag, l.value);
    }
    std::sync::atomic::fence(std::sync::atomic::Ordering::SeqCst);
}

impl NvmeSource {
    /// #149 path B: [`ColdSource::fetch`] with one landed flag per job (`landed[i]` for
    /// `jobs[i]`): the reader that reads job i raises `landed[i]` after its share is read and
    /// sanitized, before the ticket completes. The host need not wait: a device stream waits on
    /// the flag instead. A failed read raises the flag too; its error comes from the ticket.
    ///
    /// # Safety
    ///
    /// [`ColdSource::fetch`]'s, and every flag is a live, 8-B aligned u64 that only this fetch
    /// writes until the ticket has been waited on (or dropped).
    pub unsafe fn fetch_landed(&self, jobs: &[(ExpertRecord, RecordDst)], landed: &[Landed]) -> Result<Ticket, String> {
        if landed.len() != jobs.len() {
            return Err(format!("NVMe tier: {} landed flags for {} records", landed.len(), jobs.len()));
        }
        self.fetch_with(jobs, Some(landed))
    }

    unsafe fn fetch_with(&self, jobs: &[(ExpertRecord, RecordDst)], landed: Option<&[Landed]>) -> Result<Ticket, String> {
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
            share[k % n].push(Job { rec: *rec, dst: *dst, landed: landed.map(|l| l[k]) });
        }
        let mut parts = Vec::new();
        for (i, jobs) in share.into_iter().enumerate() {
            let (reply, rx) = mpsc::channel();
            self.tx[i].send(Batch { jobs, reply }).map_err(|_| format!("NVMe tier: reader {i} is gone"))?;
            parts.push(rx);
        }
        Ok(Ticket { parts })
    }
}

impl ColdSource for NvmeSource {
    unsafe fn fetch(&self, jobs: &[(ExpertRecord, RecordDst)]) -> Result<Ticket, String> {
        self.fetch_with(jobs, None)
    }

    fn wait(&self, mut t: Ticket) -> Result<FetchReport, String> {
        t.drain()
    }
}

/// Sanitize every span of every NVFP4 job in place, in the destination (the load path's rule).
/// A record of another codec (MUL1) has no ue4m3 scale bytes and stays as stored.
///
/// # Safety
///
/// The reads into the destinations have completed with the full length.
unsafe fn sanitize_jobs(jobs: &[Job]) -> u64 {
    let mut n = 0;
    for j in jobs.iter().filter(|j| j.rec.codec.sanitizes_scales()) {
        for (_, s, p) in j.rec.parts(&j.dst) {
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

    fn sym<T: Copy>(lib: &libloading::Library, n: &[u8]) -> Result<T, String> {
        unsafe { lib.get::<T>(n).map(|s| *s).map_err(|e| format!("kernel32 {}: {e}", String::from_utf8_lossy(&n[..n.len() - 1]))) }
    }

    pub fn k32() -> Result<&'static K32, String> {
        static K: std::sync::OnceLock<Result<K32, String>> = std::sync::OnceLock::new();
        K.get_or_init(|| unsafe {
            let lib = libloading::Library::new("kernel32.dll").map_err(|e| format!("kernel32.dll: {e}"))?;
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

    // ---- ioringapi.h (Windows 11, build 22000+), learn.microsoft.com/windows/win32/api/ioringapi,
    // read 2026-10-09; the same layouts `tools/nvme_read_rate.py` passes through ctypes (#171) ----

    pub type HIoRing = *mut c_void;
    pub const IORING_VERSION_1: i32 = 1;
    /// `IORING_REF_RAW`: the handle / buffer is a raw `HANDLE` / address, not a registered index
    pub const IORING_REF_RAW: i32 = 0;
    pub const S_OK: i32 = 0;
    /// `PopIoRingCompletion`: the completion queue is empty
    pub const S_FALSE: i32 = 1;
    /// `HRESULT_FROM_NT(STATUS_END_OF_FILE)`, a read at or past the end of the file
    pub const HRESULT_EOF: i32 = 0xD000_0011u32 as i32;

    /// `IORING_CREATE_FLAGS { Required, Advisory }`, both 0 = none
    #[repr(C)]
    #[derive(Clone, Copy)]
    pub struct IoRingCreateFlags {
        pub required: i32,
        pub advisory: i32,
    }

    /// `IORING_HANDLE_REF { Kind; union { HANDLE Handle; UINT32 Index; } }`, the raw-handle arm
    #[repr(C)]
    #[derive(Clone, Copy)]
    pub struct IoRingHandleRef {
        pub kind: i32,
        pub handle: Handle,
    }

    /// `IORING_BUFFER_REF { Kind; union { void *Address; IORING_REGISTERED_BUFFER } }`, the address arm
    #[repr(C)]
    #[derive(Clone, Copy)]
    pub struct IoRingBufferRef {
        pub kind: i32,
        pub address: *mut c_void,
    }

    /// `IORING_CQE { UINT_PTR UserData; HRESULT ResultCode; ULONG_PTR Information; }`
    #[repr(C)]
    #[derive(Clone, Copy, Default)]
    pub struct IoRingCqe {
        pub user_data: usize,
        pub result_code: i32,
        pub information: usize,
    }

    type FnCreateIoRing = unsafe extern "system" fn(i32, IoRingCreateFlags, u32, u32, *mut HIoRing) -> i32;
    type FnBuildIoRingReadFile = unsafe extern "system" fn(HIoRing, IoRingHandleRef, IoRingBufferRef, u32, u64, usize, i32) -> i32;
    type FnSubmitIoRing = unsafe extern "system" fn(HIoRing, u32, u32, *mut u32) -> i32;
    type FnPopIoRingCompletion = unsafe extern "system" fn(HIoRing, *mut IoRingCqe) -> i32;
    type FnCloseIoRing = unsafe extern "system" fn(HIoRing) -> i32;

    /// the IoRing entry points, loaded apart from [`K32`] so the IOCP backend keeps running on a
    /// Windows that has none
    pub struct IoRingApi {
        pub create: FnCreateIoRing,
        pub build_read: FnBuildIoRingReadFile,
        pub submit: FnSubmitIoRing,
        pub pop: FnPopIoRingCompletion,
        pub close: FnCloseIoRing,
        _lib: libloading::Library,
    }

    /// The IoRing API, or the refusal naming why this system has none.
    pub fn ioring() -> Result<&'static IoRingApi, String> {
        static R: std::sync::OnceLock<Result<IoRingApi, String>> = std::sync::OnceLock::new();
        R.get_or_init(|| unsafe {
            let lib = libloading::Library::new("kernel32.dll").map_err(|e| format!("kernel32.dll: {e}"))?;
            let un = |e: String| format!("IoRing unavailable ({e}; Windows 11, build 22000+, has it)");
            Ok(IoRingApi {
                create: sym(&lib, b"CreateIoRing\0").map_err(un)?,
                build_read: sym(&lib, b"BuildIoRingReadFile\0").map_err(un)?,
                submit: sym(&lib, b"SubmitIoRing\0").map_err(un)?,
                pop: sym(&lib, b"PopIoRingCompletion\0").map_err(un)?,
                close: sym(&lib, b"CloseIoRing\0").map_err(un)?,
                _lib: lib,
            })
        })
        .as_ref()
        .map_err(|e| e.clone())
    }

    /// an HRESULT as Windows prints it
    pub fn hr(h: i32) -> String {
        format!("HRESULT {:#010x}", h as u32)
    }
}

/// The completion side of one Windows reader.
#[cfg(windows)]
enum WinIo {
    /// the reader's own I/O completion port, bound to its handle
    Iocp { port: usize },
    /// the reader's own I/O ring; `ring` 0 = closed after a failed submission (`broken` says why)
    IoRing { ring: usize, broken: Option<String> },
}

/// One reader's own handle to the container (and, on Windows, its own completion port or ring).
struct Reader {
    file: std::fs::File,
    #[cfg(windows)]
    io: WinIo,
}

/// Submission-queue entries per ring: every slab of the largest share one reader can get (all
/// [`MAX_IN_FLIGHT`] records of a fetch, two slabs each), so a fetch never waits for a free entry.
#[cfg(windows)]
const RING_SQ: u32 = 2 * MAX_IN_FLIGHT as u32;

#[cfg(windows)]
impl Reader {
    fn open(path: &Path, backend: NvmeBackend) -> Result<Reader, String> {
        use std::os::windows::fs::OpenOptionsExt;
        use std::os::windows::io::AsRawHandle;
        let k = win::k32()?;
        // the ring first: on a system without IoRing the refusal names that, not the file
        let ring = match backend {
            NvmeBackend::Iocp => None,
            NvmeBackend::IoRing => {
                let api = win::ioring().map_err(|e| format!("NVMe tier: {BACKEND_ENV}=ioring refused: {e}"))?;
                let mut ring: win::HIoRing = std::ptr::null_mut();
                let flags = win::IoRingCreateFlags { required: 0, advisory: 0 };
                let h = unsafe { (api.create)(win::IORING_VERSION_1, flags, RING_SQ, 2 * RING_SQ, &mut ring) };
                if h < 0 || ring.is_null() {
                    return Err(format!("NVMe tier: {BACKEND_ENV}=ioring refused: IoRing unavailable, CreateIoRing returned {}", win::hr(h)));
                }
                Some(ring as usize)
            }
        };
        let file = match std::fs::OpenOptions::new()
            .read(true)
            .custom_flags(win::FILE_FLAG_NO_BUFFERING | win::FILE_FLAG_OVERLAPPED)
            .open(path)
        {
            Ok(f) => f,
            Err(e) => {
                if let (Some(r), Ok(api)) = (ring, win::ioring()) {
                    unsafe { (api.close)(r as win::HIoRing) };
                }
                return Err(format!("NVMe tier: {}: {e}", path.display()));
            }
        };
        let io = match ring {
            Some(ring) => WinIo::IoRing { ring, broken: None },
            None => {
                let port = unsafe { (k.create_iocp)(file.as_raw_handle() as win::Handle, std::ptr::null_mut(), 1, 1) };
                if port.is_null() {
                    return Err(format!("NVMe tier: CreateIoCompletionPort: {}", std::io::Error::last_os_error()));
                }
                WinIo::Iocp { port: port as usize }
            }
        };
        Ok(Reader { file, io })
    }

    fn run(&mut self, jobs: &[Job]) -> Result<FetchReport, String> {
        match self.io {
            WinIo::Iocp { port } => self.run_iocp(port, jobs),
            WinIo::IoRing { .. } => self.run_ioring(jobs),
        }
    }

    /// Issue every read of `jobs`, then drain exactly as many completions as were issued (a read
    /// still in flight writes into the caller's buffer, so the function never returns before the
    /// last one is back), then sanitize.
    fn run_iocp(&self, port: usize, jobs: &[Job]) -> Result<FetchReport, String> {
        use std::os::windows::io::AsRawHandle;
        let k = win::k32()?;
        let h = self.file.as_raw_handle() as win::Handle;
        let port = port as win::Handle;
        let mut ovs: Vec<Box<win::Overlapped>> = Vec::with_capacity(jobs.len() * 2);
        let mut lens: Vec<usize> = Vec::with_capacity(jobs.len() * 2);
        let mut err: Option<String> = None;
        'issue: for j in jobs {
            for (_, s, p) in j.rec.parts(&j.dst) {
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

    /// The IoRing twin of [`Reader::run_iocp`]: build one read entry per span (user data = the
    /// span's index), submit them all in one `SubmitIoRing`, then pop exactly as many completions
    /// as were submitted, waiting in `SubmitIoRing` whenever the completion queue is empty (a read
    /// still in flight writes into the caller's buffer, so the function never returns before the
    /// last one is back), then sanitize. A failed read and a short read are errors by name.
    fn run_ioring(&mut self, jobs: &[Job]) -> Result<FetchReport, String> {
        use std::os::windows::io::AsRawHandle;
        let h = self.file.as_raw_handle() as win::Handle;
        let WinIo::IoRing { ring, broken } = &mut self.io else { unreachable!("run_ioring on an IOCP reader") };
        if let Some(why) = broken {
            return Err(why.clone());
        }
        let api = win::ioring()?;
        let r = *ring as win::HIoRing;
        let mut spans: Vec<(u32, u32, Span)> = Vec::with_capacity(jobs.len() * 2);
        let mut err: Option<String> = None;
        'build: for j in jobs {
            for (_, s, p) in j.rec.parts(&j.dst) {
                let file = win::IoRingHandleRef { kind: win::IORING_REF_RAW, handle: h };
                let buf = win::IoRingBufferRef { kind: win::IORING_REF_RAW, address: p as *mut _ };
                let e = unsafe { (api.build_read)(r, file, buf, s.len as u32, s.off, spans.len(), 0) };
                if e < 0 {
                    err = Some(format!("NVMe tier (ioring): BuildIoRingReadFile layer {} expert {} at {}: {}", j.rec.layer, j.rec.id, s.off, win::hr(e)));
                    break 'build;
                }
                spans.push((j.rec.layer, j.rec.id, s));
            }
        }
        if spans.is_empty() {
            return match err {
                Some(e) => Err(e),
                None => Ok(FetchReport::default()),
            };
        }
        let mut submitted = 0u32;
        let e = unsafe { (api.submit)(r, 0, 0, &mut submitted) };
        if e < 0 {
            // learn.microsoft.com SubmitIoRing, Remarks: on an error other than a wait timeout
            // every entry stays in the submission queue - nothing is in flight, but the next
            // submission would send these entries (into this fetch's buffers) again. The ring is
            // closed and the reader refuses every later fetch by name.
            unsafe { (api.close)(r) };
            let why = format!("NVMe tier (ioring): SubmitIoRing of {} reads failed: {}; this reader's ring is closed", spans.len(), win::hr(e));
            *ring = 0;
            *broken = Some(why.clone());
            return Err(why);
        }
        let in_flight = (submitted as usize).min(spans.len());
        let mut done = vec![false; spans.len()];
        let mut left = in_flight;
        let mut bytes = 0u64;
        while left > 0 {
            let mut cqe = win::IoRingCqe::default();
            let p = unsafe { (api.pop)(r, &mut cqe) };
            if p == win::S_FALSE {
                let w = unsafe { (api.submit)(r, 1, win::INFINITE, std::ptr::null_mut()) };
                if w < 0 {
                    // reads are in flight into caller memory and the ring cannot be waited on:
                    // nothing safe to return (the IOCP twin panics on the same condition)
                    panic!("NVMe tier (ioring): SubmitIoRing wait failed with {left} reads in flight: {}", win::hr(w));
                }
                continue;
            }
            if p != win::S_OK {
                panic!("NVMe tier (ioring): PopIoRingCompletion failed with {left} reads in flight: {}", win::hr(p));
            }
            let i = cqe.user_data;
            assert!(i < spans.len() && !done[i], "NVMe tier (ioring): completion {i} this reader did not issue");
            done[i] = true;
            left -= 1;
            let (layer, id, s) = spans[i];
            if cqe.result_code < 0 && cqe.result_code != win::HRESULT_EOF {
                err.get_or_insert(format!("NVMe tier (ioring): read layer {layer} expert {id} at {} failed: {}", s.off, win::hr(cqe.result_code)));
            } else {
                let n = if cqe.result_code < 0 { 0 } else { cqe.information };
                if n != s.len {
                    err.get_or_insert(format!("NVMe tier (ioring): short read layer {layer} expert {id} at {}, {n} of {} B (past the end of the file?)", s.off, s.len));
                }
                bytes += n as u64;
            }
        }
        if in_flight < spans.len() {
            // S_OK promises every entry submitted; should it not hold, the rest still sits in the
            // submission queue - what was in flight is drained, the ring is not trusted again
            unsafe { (api.close)(r) };
            let why = format!("NVMe tier (ioring): SubmitIoRing sent {in_flight} of {} reads; this reader's ring is closed", spans.len());
            *ring = 0;
            *broken = Some(why.clone());
            err.get_or_insert(why);
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
        // `run` never returns with a read in flight, so nothing is pending here
        match self.io {
            WinIo::Iocp { port } => {
                if let Ok(k) = win::k32() {
                    unsafe { (k.close)(port as win::Handle) };
                }
            }
            WinIo::IoRing { ring, .. } => {
                if let (true, Ok(api)) = (ring != 0, win::ioring()) {
                    unsafe { (api.close)(ring as win::HIoRing) };
                }
            }
        }
    }
}

#[cfg(unix)]
impl Reader {
    fn open(path: &Path, backend: NvmeBackend) -> Result<Reader, String> {
        use std::os::unix::fs::OpenOptionsExt;
        if backend == NvmeBackend::IoRing {
            return Err(format!("NVMe tier: {BACKEND_ENV}=ioring refused: IoRing is a Windows 11 API; this platform reads with O_DIRECT + pread (leave {BACKEND_ENV} unset)"));
        }
        let file = std::fs::OpenOptions::new()
            .read(true)
            .custom_flags(libc::O_DIRECT)
            .open(path)
            .map_err(|e| format!("NVMe tier: {} (O_DIRECT): {e}", path.display()))?;
        Ok(Reader { file })
    }

    /// `pread` every slab of `jobs` on this reader's own descriptor, then sanitize.
    fn run(&mut self, jobs: &[Job]) -> Result<FetchReport, String> {
        use std::os::unix::fs::FileExt;
        let mut bytes = 0u64;
        for j in jobs {
            for (_, s, p) in j.rec.parts(&j.dst) {
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

    // ---- #149: the reader backend, IOCP (default) or IoRing ----

    /// `iocp` unless asked otherwise: the config's field wins over the variable, unset or empty
    /// is the default, and a name that is not a backend is refused naming the variable and the
    /// values it takes (never a silent default).
    #[test]
    fn the_backend_is_iocp_unless_asked_and_an_unknown_name_is_refused() {
        assert_eq!(NvmeBackend::default(), NvmeBackend::Iocp);
        assert_eq!(NvmeConfig::new("x").backend, None);
        assert_eq!(resolve_backend(None, None), Ok(NvmeBackend::Iocp));
        assert_eq!(resolve_backend(None, Some("")), Ok(NvmeBackend::Iocp));
        assert_eq!(resolve_backend(None, Some("iocp")), Ok(NvmeBackend::Iocp));
        assert_eq!(resolve_backend(None, Some("ioring")), Ok(NvmeBackend::IoRing));
        assert_eq!(resolve_backend(Some(NvmeBackend::IoRing), Some("iocp")), Ok(NvmeBackend::IoRing));
        assert_eq!(resolve_backend(Some(NvmeBackend::Iocp), Some("bogus")), Ok(NvmeBackend::Iocp));
        for bad in ["IoRing", "io_ring", "uring", "1"] {
            let e = resolve_backend(None, Some(bad)).unwrap_err();
            assert!(e.contains("CROW_NVME_BACKEND") && e.contains("unknown NVMe reader backend") && e.contains("iocp (default), ioring"), "{e}");
        }
        for b in [NvmeBackend::Iocp, NvmeBackend::IoRing] {
            assert_eq!(NvmeBackend::parse(b.name()), Ok(b));
        }
    }

    /// Off Windows an explicit `ioring` is refused by name at open, not run as `pread`.
    #[cfg(not(windows))]
    #[test]
    fn ioring_off_windows_is_refused_by_name() {
        let dir = std::env::temp_dir().join(format!("crow-nvme-ioring-off-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("f.bin");
        std::fs::write(&path, vec![0u8; 8192]).unwrap();
        let mut cfg = NvmeConfig::new(&path);
        cfg.backend = Some(NvmeBackend::IoRing);
        let e = NvmeSource::open(&cfg).err().expect("ioring opened off Windows");
        let _ = std::fs::remove_dir_all(&dir);
        assert!(e.contains("CROW_NVME_BACKEND=ioring refused"), "{e}");
    }

    /// A plain file of `len` xorshift bytes in its own temp dir, removed on drop.
    #[cfg(windows)]
    fn raw_file(tag: &str, len: usize) -> (Synth, Vec<u8>) {
        let dir = std::env::temp_dir().join(format!("crow-nvme-raw-{tag}-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("raw.bin");
        let mut x = 0x5851_F42D_4C95_7F2Du64;
        let mut bytes = vec![0u8; len];
        for w in bytes.chunks_exact_mut(8) {
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            w.copy_from_slice(&x.to_le_bytes());
        }
        std::fs::write(&path, &bytes).unwrap();
        (Synth { dir, path }, bytes)
    }

    /// a record of `len` B at `off`, one unit, a codec that is not sanitized (bytes as stored)
    #[cfg(windows)]
    fn raw_rec(id: u32, off: u64, len: usize) -> ExpertRecord {
        ExpertRecord { layer: 0, id, gu: Span { off, len }, dn: Span { off: 0, len: 0 }, codec: ExpertCodec::Mul1, layout: RecordLayout::OneUnit }
    }

    /// Eight records (one 3-bit record of 9,474,048 B, the rest of other sector-whole sizes, one
    /// of them two slabs), out of file order, read from a 48 MiB temp file by every backend at 1
    /// and 2 readers, twice per source (the ring is reused): each destination is the plain read
    /// of its span, byte for byte, and the source reports the backend it was asked for.
    #[cfg(windows)]
    #[test]
    fn both_backends_read_records_byte_identical_to_a_plain_read() {
        const LEN: usize = 48 << 20;
        let (f, want) = raw_file("ident", LEN);
        let lens = [MUL1_REC as usize, 4096, 1 << 20, 2_813_952, 409_600, 3 << 20, 8192, 5_005_312];
        let offs = [36u64 << 20, 0, 12 << 20, 20 << 20, 4096, 28 << 20, (48 << 20) - 8192, 16 << 20];
        let mut recs: Vec<ExpertRecord> = (0..8).map(|k| raw_rec(k as u32, offs[k], lens[k])).collect();
        // record 3 as two slabs: its first 1 MiB and the 1 MiB at 44 MiB
        recs[3].layout = RecordLayout::TwoSlabs;
        recs[3].gu.len = 1 << 20;
        recs[3].dn = Span { off: 44 << 20, len: 1 << 20 };
        let total: u64 = recs.iter().map(|r| if r.layout == RecordLayout::TwoSlabs { r.gu.len + r.dn.len } else { r.gu.len } as u64).sum();
        for backend in [NvmeBackend::Iocp, NvmeBackend::IoRing] {
            for readers in [1, 2] {
                let mut cfg = NvmeConfig::new(&f.path);
                cfg.readers = readers;
                cfg.backend = Some(backend);
                let src = NvmeSource::open(&cfg).unwrap_or_else(|e| panic!("{} x {readers}: {e}", backend.name()));
                assert_eq!((src.backend(), src.readers()), (backend, readers));
                for round in 0..2 {
                    let bufs: Vec<(Aligned, Aligned)> = recs.iter().map(|r| (Aligned::new(r.gu.len), Aligned::new(r.dn.len.max(4096)))).collect();
                    let jobs: Vec<(ExpertRecord, RecordDst)> = recs.iter().zip(&bufs).map(|(r, (g, d))| (*r, RecordDst { gu: g.p, dn: d.p })).collect();
                    let rep = src.wait(unsafe { src.fetch(&jobs) }.unwrap()).unwrap_or_else(|e| panic!("{} x {readers} round {round}: {e}", backend.name()));
                    assert_eq!((rep.records, rep.bytes, rep.clamped), (8, total, 0), "{} x {readers} round {round}", backend.name());
                    for (r, (g, d)) in recs.iter().zip(&bufs) {
                        for (what, s, buf) in [("gu", r.gu, g), ("dn", r.dn, d)] {
                            let at = s.off as usize;
                            assert!(buf.bytes()[..s.len] == want[at..at + s.len], "{} x {readers} round {round}: record {} {what} differs from the plain read", backend.name(), r.id);
                        }
                    }
                }
            }
        }
    }

    /// A span that runs past the end of the file is a short read, an error by name on both
    /// backends, and the source keeps working: the next fetch on the same reader (the same ring)
    /// reads its record byte-identical.
    #[cfg(windows)]
    #[test]
    fn a_short_read_is_an_error_by_name_on_both_backends() {
        const LEN: usize = 1 << 20;
        let (f, want) = raw_file("short", LEN);
        for backend in [NvmeBackend::Iocp, NvmeBackend::IoRing] {
            let mut cfg = NvmeConfig::new(&f.path);
            cfg.backend = Some(backend);
            let src = NvmeSource::open(&cfg).unwrap();
            let b = Aligned::new(8192);
            let past = raw_rec(7, (LEN - 4096) as u64, 8192);
            let e = src.wait(unsafe { src.fetch(&[(past, RecordDst { gu: b.p, dn: std::ptr::null_mut() })]) }.unwrap()).unwrap_err();
            assert!(e.contains("short read") && e.contains("4096 of 8192 B"), "{}: {e}", backend.name());
            let ok = raw_rec(8, 8192, 8192);
            let rep = src.wait(unsafe { src.fetch(&[(ok, RecordDst { gu: b.p, dn: std::ptr::null_mut() })]) }.unwrap()).unwrap();
            assert_eq!(rep.bytes, 8192, "{}", backend.name());
            assert!(b.bytes() == &want[8192..16384], "{}: the read after the short read differs", backend.name());
        }
    }

    /// #149 path B: a landed flag is raised only once its record is in the destination. Eight
    /// records read with [`NvmeSource::fetch_landed`] (1 and 2 readers, both backends): the host
    /// spins on each flag WITHOUT waiting on the ticket, and the moment a flag shows its value the
    /// record must already equal the plain read. A short read raises its flag too (a device
    /// waiting on it must not hang) and the ticket carries the error by name. A count mismatch
    /// between jobs and flags is refused.
    #[cfg(windows)]
    #[test]
    fn a_landed_flag_rises_after_its_record_and_also_on_a_failed_read() {
        const LEN: usize = 48 << 20;
        let (f, want) = raw_file("landed", LEN);
        let lens = [MUL1_REC as usize, 4096, 1 << 20, 2_813_952, 409_600, 3 << 20, 8192, 5_005_312];
        let offs = [36u64 << 20, 0, 12 << 20, 20 << 20, 4096, 28 << 20, (48 << 20) - 8192, 16 << 20];
        let recs: Vec<ExpertRecord> = (0..8).map(|k| raw_rec(k as u32, offs[k], lens[k])).collect();
        for backend in [NvmeBackend::Iocp, NvmeBackend::IoRing] {
            for readers in [1, 2] {
                let mut cfg = NvmeConfig::new(&f.path);
                cfg.readers = readers;
                cfg.backend = Some(backend);
                let src = NvmeSource::open(&cfg).unwrap();
                let flags: Vec<u64> = vec![0; 8];
                for round in 1..=3u64 {
                    let bufs: Vec<Aligned> = recs.iter().map(|r| Aligned::new(r.gu.len)).collect();
                    let jobs: Vec<(ExpertRecord, RecordDst)> = recs.iter().zip(&bufs).map(|(r, b)| (*r, RecordDst { gu: b.p, dn: std::ptr::null_mut() })).collect();
                    let landed: Vec<Landed> = (0..8).map(|k| Landed { flag: &flags[k] as *const u64 as *mut u64, value: round * 100 + k as u64 }).collect();
                    let t = unsafe { src.fetch_landed(&jobs, &landed) }.unwrap();
                    let mut seen = [false; 8];
                    let t0 = std::time::Instant::now();
                    while seen.iter().any(|s| !s) {
                        for k in 0..8 {
                            if !seen[k] && unsafe { std::ptr::read_volatile(&flags[k]) } == landed[k].value {
                                std::sync::atomic::fence(std::sync::atomic::Ordering::Acquire);
                                let at = offs[k] as usize;
                                assert!(bufs[k].bytes()[..lens[k]] == want[at..at + lens[k]], "{} x {readers} round {round}: record {k}'s flag rose before its bytes", backend.name());
                                seen[k] = true;
                            }
                        }
                        assert!(t0.elapsed().as_secs() < 30, "{} x {readers} round {round}: flags {seen:?} never rose", backend.name());
                        std::hint::spin_loop();
                    }
                    src.wait(t).unwrap();
                }
                let b = Aligned::new(8192);
                let flag = 0u64;
                let past = raw_rec(7, (LEN - 4096) as u64, 8192);
                let t = unsafe { src.fetch_landed(&[(past, RecordDst { gu: b.p, dn: std::ptr::null_mut() })], &[Landed { flag: &flag as *const u64 as *mut u64, value: 9 }]) }.unwrap();
                let e = src.wait(t).unwrap_err();
                assert!(e.contains("short read"), "{}: {e}", backend.name());
                assert_eq!(unsafe { std::ptr::read_volatile(&flag) }, 9, "{}: a failed read must still raise its flag", backend.name());
                let e = unsafe { src.fetch_landed(&[(past, RecordDst { gu: b.p, dn: std::ptr::null_mut() })], &[]) }.err().unwrap();
                assert!(e.contains("0 landed flags for 1 records"), "{e}");
            }
        }
    }

    // ---- #159 / #176 / #149: the glm5_next record is the container's, not 14,155,776 B ----

    /// the plan's 3.05-bpw MUL1 record (2313 x 4096), a third per projection (771 x 4096)
    const MUL1_REC: u64 = 9_474_048;
    /// the CNQ4.5 NVFP4 record (3 x 4,718,592)
    const NVFP4_REC: u64 = 14_155_776;
    /// logical values of one GLM projection (2048 x 4096)
    const PROJ_VALUES: u64 = 2048 * 4096;

    /// A synthetic glm5_next container: a 4084-B bf16 tensor, then `experts` records of layer 3
    /// (gate, up, down back to back, a third of `rec` each, every record on a 4096-B file
    /// offset), then a 4096-B bf16 tensor right after the last record; index v2 of model type
    /// `glm5_next_text`. Removed on drop.
    fn synth_glm(tag: &str, codec: ExpertCodec, rec: u64, experts: u32) -> Synth {
        use std::io::Write;
        let dir = std::env::temp_dir().join(format!("crow-nvme-glm-{tag}-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("synth-glm.cnq");
        let proj = rec / 3;
        let lead = 4084u64; // 12 + 4084 = 4096: the first record sits on a sector
        let mut tensors = vec![serde_json::json!({ "name": "model.language_model.layers.3.mlp.gate.weight", "section": "text",
            "dtype": "bf16", "offset": 0, "n_values": lead / 2, "shape": [2, lead / 4] })];
        for e in 0..experts {
            for (k, p) in ["gate", "up", "down"].into_iter().enumerate() {
                tensors.push(serde_json::json!({ "name": glm5_expert_tensor_name(3, e, p), "section": "text", "dtype": codec.dtype(),
                    "offset": lead + e as u64 * rec + k as u64 * proj, "n_values": PROJ_VALUES, "shape": [2048, 4096] }));
            }
        }
        let tail_off = lead + experts as u64 * rec;
        tensors.push(serde_json::json!({ "name": "model.language_model.norm.weight", "section": "text", "dtype": "bf16",
            "offset": tail_off, "n_values": 2048, "shape": [2048] }));
        let blob_len = tail_off + 4096;
        let sha = |s: &str| crate::cnq::sha256_hex(s.as_bytes());
        let index = serde_json::json!({
            "format": "crow-nest-quant", "format_version": 2, "blob_offset": 12, "recipe": "synthetic-glm-record",
            "model": {
                "family": "Glm5Next", "model_type": "glm5_next_text",
                "config_json": "{}", "config_json_sha256": sha("{}"),
                "generation_config_json": "{}", "generation_config_json_sha256": sha("{}"),
                "source": { "repo": "crow-nest/synthetic-glm-record", "revision": "159", "shards": [] },
                "geo": {}
            },
            "tensors": tensors
        });
        let ib = serde_json::to_vec(&index).unwrap();
        let mut f = std::io::BufWriter::with_capacity(1 << 20, std::fs::File::create(&path).unwrap());
        f.write_all(b"CNQ1\0\0\0\0\0\0\0\0").unwrap();
        let mut x = 0x2545_F491_4F6C_DD1Du64 ^ rec;
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

    /// A 3-bit (MUL1, 9,474,048 B) record read through the NVMe backend is `Cnq::read_range` of
    /// the record, byte for byte: the record size and codec come from the container's index,
    /// the read is one span per record, and no NVFP4 sanitize touches it (the raw bytes carry
    /// 0x7F at NVFP4 scale positions, so a sanitize would have changed them). 4 records, 38 MB.
    #[test]
    fn a_3bit_record_read_through_the_nvme_backend_is_read_range_byte_for_byte() {
        let s = synth_glm("mul1", ExpertCodec::Mul1, MUL1_REC, 4);
        let path = s.path.to_str().unwrap();
        let (spec, n) = glm5_record_of_container(path).unwrap();
        assert_eq!((spec, n), (ExpertRecordSpec { codec: ExpertCodec::Mul1, bytes: MUL1_REC }, 4));
        let mut cnq = Cnq::open_checked(path).unwrap();
        let ids = [2u32, 0, 3];
        let recs: Vec<ExpertRecord> = ids.iter().map(|&id| ExpertRecord::locate_glm5(&cnq, &spec, 3, id).unwrap()).collect();
        assert!(recs.iter().all(|r| r.layout == RecordLayout::OneUnit && r.gu.len as u64 == MUL1_REC && r.gu.off % ALIGN == 0));
        let bufs: Vec<Aligned> = ids.iter().map(|_| Aligned::new(MUL1_REC as usize)).collect();
        let jobs: Vec<(ExpertRecord, RecordDst)> =
            recs.iter().zip(&bufs).map(|(r, b)| (*r, RecordDst { gu: b.p, dn: std::ptr::null_mut() })).collect();
        let mut cfg = NvmeConfig::new(&s.path);
        cfg.readers = 2;
        let src = NvmeSource::open(&cfg).unwrap();
        let rep = src.wait(unsafe { src.fetch(&jobs) }.unwrap()).unwrap();
        assert_eq!((rep.records, rep.bytes, rep.clamped), (3, 3 * MUL1_REC, 0));
        for (k, &id) in ids.iter().enumerate() {
            let gate = cnq.find(&glm5_expert_tensor_name(3, id, "gate"), "text").clone();
            let raw = cnq.read_range(&gate, 0, MUL1_REC as usize);
            assert!(bufs[k].bytes() == raw.as_slice(), "expert {id}: NVMe bytes differ from read_range of the 9,474,048-B record");
            let mut nv = raw.clone();
            assert!(sanitize_sf_slab(&mut nv) > 0, "the raw record has no 0x7F at an NVFP4 scale position - the no-sanitize check proves nothing");
        }
        // the 4.5-bit spec on this 3-bit container is refused by name, not read
        let wrong = ExpertRecordSpec { codec: ExpertCodec::Nvfp4, bytes: NVFP4_REC };
        let e = ExpertRecord::locate_glm5(&cnq, &wrong, 3, 0).unwrap_err();
        assert!(e.contains("dtype mul1, the expert record says codec nvfp4"), "{e}");
        drop(src);
        drop(cnq);
    }

    /// #175: the table of every record (one index name map) is `locate_glm5` record by record,
    /// and a layer the container does not hold is refused by name, not left empty. 4 records, 38 MB.
    #[test]
    fn the_record_table_is_locate_glm5_record_by_record() {
        let s = synth_glm("table", ExpertCodec::Mul1, MUL1_REC, 4);
        let path = s.path.to_str().unwrap();
        let (spec, _) = glm5_record_of_container(path).unwrap();
        let cnq = Cnq::open_checked(path).unwrap();
        let t = ExpertRecord::glm5_table(&cnq, &spec, &[3], 4).unwrap();
        assert_eq!(t.len(), 1);
        let want: Vec<ExpertRecord> = (0..4).map(|e| ExpertRecord::locate_glm5(&cnq, &spec, 3, e).unwrap()).collect();
        assert_eq!(t[0], want);
        assert!(t[0].windows(2).all(|w| w[1].gu.off == w[0].gu.off + MUL1_REC), "records back to back");
        let e = ExpertRecord::glm5_table(&cnq, &spec, &[3, 4], 4).unwrap_err();
        assert!(e.contains("layers.4.mlp.experts.0.gate_proj.weight: not in the container index"), "{e}");
        drop(cnq);
    }

    /// The NVFP4 record (14,155,776 B) of a glm5_next container is read as one unit and
    /// sanitized: the destination is `read_range` + `sanitize_sf_slab` of the record. 3 records, 42 MB.
    #[test]
    fn an_nvfp4_glm_record_is_read_as_one_unit_and_sanitized() {
        let s = synth_glm("nvfp4", ExpertCodec::Nvfp4, NVFP4_REC, 3);
        let path = s.path.to_str().unwrap();
        let (spec, n) = glm5_record_of_container(path).unwrap();
        assert_eq!((spec, n), (ExpertRecordSpec { codec: ExpertCodec::Nvfp4, bytes: NVFP4_REC }, 3));
        let mut cnq = Cnq::open_checked(path).unwrap();
        let rec = ExpertRecord::locate_glm5(&cnq, &spec, 3, 1).unwrap();
        let b = Aligned::new(NVFP4_REC as usize);
        let src = NvmeSource::open(&NvmeConfig::new(&s.path)).unwrap();
        let rep = src.wait(unsafe { src.fetch(&[(rec, RecordDst { gu: b.p, dn: std::ptr::null_mut() })]) }.unwrap()).unwrap();
        let gate = cnq.find(&glm5_expert_tensor_name(3, 1, "gate"), "text").clone();
        let mut want = cnq.read_range(&gate, 0, NVFP4_REC as usize);
        let clamped = sanitize_sf_slab(&mut want);
        assert!(clamped > 0);
        assert_eq!((rep.bytes, rep.clamped), (NVFP4_REC, clamped));
        assert!(b.bytes() == want.as_slice(), "the NVFP4 record differs from read_range + sanitize");
        drop(src);
        drop(cnq);
    }

    /// one index entry, base container
    fn ti(name: &str, dtype: &str, offset: u64, n_values: u64) -> TensorInfo {
        TensorInfo { name: name.into(), section: "text".into(), dtype: dtype.into(), offset, n_values, global_scale: 1.0, shape: vec![], overlay: false }
    }

    /// `experts` records of layer 3 at relative `start + e * stride`, a third of `rec` per projection,
    /// then an unaligned bf16 tensor right after the last record; returns (tensors, blob_len)
    fn glm_index(dtype: &str, rec: u64, stride: u64, start: u64, experts: u32) -> (Vec<TensorInfo>, u64) {
        let proj = rec / 3;
        let mut v = vec![];
        for e in 0..experts {
            for (k, p) in ["gate", "up", "down"].into_iter().enumerate() {
                v.push(ti(&glm5_expert_tensor_name(3, e, p), dtype, start + e as u64 * stride + k as u64 * proj, PROJ_VALUES));
            }
        }
        let tail = start + (experts as u64 - 1) * stride + rec;
        v.push(ti("model.language_model.norm.weight", "bf16", tail, 2048));
        (v, tail + 4096)
    }

    /// The record comes from the index (codec from the dtype, size from the 4096-B aligned
    /// units), and every layout that does not give one sector-whole record is refused by name.
    #[test]
    fn the_glm_record_comes_from_the_index_and_bad_layouts_are_refused_by_name() {
        let (t, end) = glm_index("nvfp4", NVFP4_REC, NVFP4_REC, 4084, 5);
        assert_eq!(glm5_record_from_index(&t, 12, end), Ok((ExpertRecordSpec { codec: ExpertCodec::Nvfp4, bytes: NVFP4_REC }, 5)));
        let (t, end) = glm_index("mul1", MUL1_REC, MUL1_REC, 4084, 5);
        assert_eq!(glm5_record_from_index(&t, 12, end), Ok((ExpertRecordSpec { codec: ExpertCodec::Mul1, bytes: MUL1_REC }, 5)));
        // a 9,474,000-B record: the converter pads 48 B before each next record, the last one
        // is followed by the unaligned tensor - refused as not a whole number of sectors
        let (t, end) = glm_index("mul1", MUL1_REC - 48, MUL1_REC, 4084, 5);
        let e = glm5_record_from_index(&t, 12, end).unwrap_err();
        assert!(e.contains("refusing expert record of 9474000 B: not a multiple of 4096 B"), "{e}");
        // records on 12 mod 4096 (the format's default, no alignment)
        let (t, end) = glm_index("mul1", MUL1_REC, MUL1_REC, 0, 2);
        let e = glm5_record_from_index(&t, 12, end).unwrap_err();
        assert!(e.contains("starts at file offset 12, not on a 4096 B boundary"), "{e}");
        // an unknown codec, a mix of codecs, no experts, a missing projection
        let (t, end) = glm_index("q3k", MUL1_REC, MUL1_REC, 4084, 2);
        assert!(glm5_record_from_index(&t, 12, end).unwrap_err().starts_with("refusing expert codec \"q3k\""));
        let (mut t, end) = glm_index("mul1", MUL1_REC, MUL1_REC, 4084, 2);
        t[4].dtype = "nvfp4".into();
        assert!(glm5_record_from_index(&t, 12, end).unwrap_err().contains("mix dtypes"));
        let e = glm5_record_from_index(&[ti("model.language_model.norm.weight", "bf16", 0, 2048)], 12, 4096).unwrap_err();
        assert!(e.contains("no glm5_next routed-expert tensor"), "{e}");
        let (mut t, end) = glm_index("mul1", MUL1_REC, MUL1_REC, 4084, 2);
        t.remove(5);
        assert!(glm5_record_from_index(&t, 12, end).unwrap_err().contains("layer 3 expert 1 lacks its down projection"));
        // nvfp4 whose layout is not the format's byte rule (the down projection 4096 B late)
        let (mut t, end) = glm_index("nvfp4", NVFP4_REC, NVFP4_REC + 4096, 4084, 2);
        t[2].offset += 4096;
        assert!(glm5_record_from_index(&t, 12, end).unwrap_err().contains("at nvfp4 is 4718592 + 4718592 + 4718592 B, the index layout disagrees"));
    }
}
