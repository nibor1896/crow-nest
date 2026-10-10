//! #186 prefill item 2: the trunk input of a glm5_next prompt call gathered on the device.
//!
//! **`CROW_GLM_EMBED_GATHER=1`** (default off; off, every call is `embed_rows` + `trunk_input` +
//! `to_f32_into` exactly as before). Today a prompt call of `t` rows reads `t` embedding rows of
//! 8 KiB one by one from the container (`glm5_model::embed_rows`, a seek + read pair each), widens
//! them on the host, copies each into the four residual streams (`glm5_model::trunk_input`, a
//! 512 MiB host fill at 8,192 rows) and uploads that from pageable memory; the GPU idles all along
//! (Nsight 2026-10-10, 0.349 s before the first layer of a cold 8,192-row prefill). With the
//! switch on, [`trunk_into`] instead
//!
//! 1. dedups the call's ids ([`dedup`]: the distinct ids in ascending order and each row's index
//!    into them; 1,759 distinct of 8,192 on the real-text prompt of `quick.sh`),
//! 2. reads each distinct row once, in id order, neighbours at most [`MAX_GAP`] rows apart in one
//!    read ([`spans`]), the spans split over up to [`READERS`] threads with their own handles and
//!    positional reads ([`read_rows`]; an overlay tensor keeps `Cnq::read_range`),
//! 3. uploads those BF16 rows and the row -> distinct index map once, and
//! 4. widens and expands them into `x` on the device (`glm5_embed_expand`,
//!    `kernels_glm5_embed.cu`): the bit shift `bf16 << 16` stored as u32, the arithmetic of
//!    `cnq::bf16_bytes_to_f32`, so `x` is bit-identical to the host path (tests below).
//!
//! The staging buffers are allocated per call and freed after a stream sync (the next work of the
//! call, its layers, waits for `x` anyway). Only the prompt calls take this path
//! (`glm5_tiers::Glm5Run::prompt_calls_with`); decode rows and every other caller keep the host path.

use crate::cnq::Cnq;
use crate::cuda;
use crate::geo::Glm5Geo;
use crate::kernels::launch_v;
use cudarc::driver::sys::{CUdeviceptr, CUfunction};

pub type Dev = CUdeviceptr;

/// the switch
pub const ENV: &str = "CROW_GLM_EMBED_GATHER";

/// `1` turns the switch on; unset or any other value leaves it off (the repo's `CROW_*` rule)
pub fn parse(v: Option<&str>) -> bool {
    v == Some("1")
}

/// the switch of this process
pub fn on() -> bool {
    parse(std::env::var(ENV).ok().as_deref())
}

/// distinct ids closer than this many rows are read as one range (8 rows = 64 KiB of gap at
/// hidden 4096; on the `quick.sh` prompt 1,759 distinct ids -> 993 reads, 27.8 MiB)
pub const MAX_GAP: u64 = 8;

/// reader threads at most
pub const READERS: usize = 8;

/// spans per reader thread at least (a short call reads on one thread)
const SPANS_PER_READER: usize = 32;

const SRC: &str = include_str!("kernels_glm5_embed.cu");

const EMBED: &str = "model.language_model.embed_tokens.weight";

/// The distinct ids of `ids` in ascending order, and for each row of `ids` its index into them
pub fn dedup(ids: &[i64]) -> (Vec<i64>, Vec<i32>) {
    let mut uniq = ids.to_vec();
    uniq.sort_unstable();
    uniq.dedup();
    let slot = ids.iter().map(|id| i32::try_from(uniq.binary_search(id).expect("glm5_embed: an id of the call")).expect("glm5_embed: more than 2^31 distinct ids")).collect();
    (uniq, slot)
}

/// The reads of ascending distinct ids: index ranges `[a, b)` of `uniq` whose neighbours are at
/// most `max_gap` ids apart, each read as the one row range `uniq[a] ..= uniq[b - 1]`
pub fn spans(uniq: &[i64], max_gap: u64) -> Vec<(usize, usize)> {
    let mut out = Vec::new();
    let mut a = 0;
    for i in 1..=uniq.len() {
        if i == uniq.len() || (uniq[i] - uniq[i - 1]) as u64 > max_gap {
            out.push((a, i));
            a = i;
        }
    }
    out
}

/// read exactly `buf.len()` bytes at `off` without the file's cursor (a handle per thread)
fn read_exact_at(f: &std::fs::File, buf: &mut [u8], off: u64) -> std::io::Result<()> {
    #[cfg(unix)]
    {
        use std::os::unix::fs::FileExt;
        f.read_exact_at(buf, off)
    }
    #[cfg(windows)]
    {
        use std::os::windows::fs::FileExt;
        let mut done = 0;
        while done < buf.len() {
            match f.seek_read(&mut buf[done..], off + done as u64)? {
                0 => return Err(std::io::ErrorKind::UnexpectedEof.into()),
                n => done += n,
            }
        }
        Ok(())
    }
}

/// The BF16 embedding rows of the ascending distinct ids `uniq`, packed `[uniq.len()][hidden]`
/// as stored (`embed_rows`' tensor and checks; each row the bytes `Cnq::read_range` returns)
pub fn read_rows(cnq: &mut Cnq, g: &Glm5Geo, uniq: &[i64]) -> Vec<u8> {
    let t = cnq.find(EMBED, "text").clone();
    assert_eq!((t.dtype.as_str(), t.shape.as_slice()), ("bf16", &[g.vocab as u64, g.hidden as u64][..]), "glm5_model: embed_tokens");
    for &id in uniq {
        assert!((0..g.vocab as i64).contains(&id), "glm5_model: token id {id} outside the vocab of {}", g.vocab);
    }
    assert!(uniq.windows(2).all(|w| w[0] < w[1]), "glm5_embed: the ids are not distinct and ascending");
    let row = g.hidden * 2;
    let mut packed = vec![0u8; uniq.len() * row];
    let sp = spans(uniq, MAX_GAP);
    // one span: its row range read into `buf`, the wanted rows copied out in order
    let take = |buf: &[u8], ids: &[i64], out: &mut [u8]| {
        for (k, &id) in ids.iter().enumerate() {
            let at = (id - ids[0]) as usize * row;
            out[k * row..(k + 1) * row].copy_from_slice(&buf[at..at + row]);
        }
    };
    if t.overlay {
        for &(a, b) in &sp {
            let len = (uniq[b - 1] - uniq[a] + 1) as usize * row;
            let buf = cnq.read_range(&t, uniq[a] as u64 * row as u64, len);
            take(&buf, &uniq[a..b], &mut packed[a * row..b * row]);
        }
        return packed;
    }
    let base = cnq.blob_offset + t.offset;
    let path = cnq.path.clone();
    let readers = READERS.min(sp.len().div_ceil(SPANS_PER_READER)).max(1);
    let per = sp.len().div_ceil(readers).max(1);
    std::thread::scope(|s| {
        let mut rest: &mut [u8] = &mut packed;
        let mut done = 0;
        for group in sp.chunks(per) {
            let (a, b) = (group[0].0, group[group.len() - 1].1);
            assert_eq!(a, done);
            let (mine, tail) = std::mem::take(&mut rest).split_at_mut((b - a) * row);
            rest = tail;
            done = b;
            let path = &path;
            s.spawn(move || {
                let f = std::fs::File::open(path).unwrap_or_else(|e| panic!("glm5_embed: {path}: {e}"));
                let mut buf = Vec::new();
                for &(sa, sb) in group {
                    let len = (uniq[sb - 1] - uniq[sa] + 1) as usize * row;
                    buf.resize(len, 0);
                    let off = base + uniq[sa] as u64 * row as u64;
                    read_exact_at(&f, &mut buf, off).unwrap_or_else(|e| panic!("glm5_embed: {path}: {len} B at {off}: {e}"));
                    take(&buf, &uniq[sa..sb], &mut mine[(sa - a) * row..(sb - a) * row]);
                }
            });
        }
    });
    packed
}

/// The host twin of `glm5_embed_expand`: `[slot.len()][streams][hidden]` f32 bits of the packed
/// BF16 rows (what the device writes, for the tests)
pub fn expand_host(packed: &[u8], slot: &[i32], hidden: usize, streams: usize) -> Vec<u32> {
    let row = hidden * 2;
    let mut x = Vec::with_capacity(slot.len() * streams * hidden);
    for &s in slot {
        let r = &packed[s as usize * row..(s as usize + 1) * row];
        for _ in 0..streams {
            x.extend(r.chunks_exact(2).map(|c| (u16::from_le_bytes([c[0], c[1]]) as u32) << 16));
        }
    }
    x
}

/// the compiled `glm5_embed_expand`
pub struct Gather {
    module: cuda::Module,
    expand: CUfunction,
}

impl Gather {
    /// # Safety
    /// A CUDA context is current.
    pub unsafe fn new() -> Gather {
        let module = cuda::compile(SRC);
        let expand = module.get("glm5_embed_expand");
        Gather { module, expand }
    }

    /// the gather when [`ENV`] is on, else `None` (nothing compiled)
    ///
    /// # Safety
    /// As [`Gather::new`].
    pub unsafe fn from_env() -> Option<Gather> {
        on().then(|| Gather::new())
    }

    /// `x [slot.len()][streams][hidden]` = the packed BF16 rows widened, row `r` from row
    /// `slot[r]`, in every stream; returns after a sync of the current stream
    ///
    /// # Safety
    /// A CUDA context is current; `x` holds `slot.len() x streams x hidden` f32.
    pub unsafe fn expand_into(&self, packed: &[u8], slot: &[i32], g: &Glm5Geo, x: Dev) {
        if slot.is_empty() {
            return;
        }
        let mut rows = cuda::alloc_named("glm5 embed gather rows", packed.len());
        let mut sl = cuda::alloc_named("glm5 embed gather slots", slot.len() * 4);
        cuda::upload_into(rows, packed);
        cuda::to_i32_into(sl, slot);
        launch_v(self.expand, slot.len() as u32, 1, 1, 256, &[rows, sl, x, g.hidden as u64, g.hc_streams as u64]);
        cuda::sync();
        cuda::free_dev(&mut rows);
        cuda::free_dev(&mut sl);
    }

    /// the trunk input of `ids` into `x` on this path: [`dedup`], [`read_rows`], [`Gather::expand_into`]
    ///
    /// # Safety
    /// As [`Gather::expand_into`] for `ids.len()` rows.
    pub unsafe fn trunk_into(&self, cnq: &mut Cnq, g: &Glm5Geo, ids: &[i64], x: Dev) {
        let (uniq, slot) = dedup(ids);
        let packed = read_rows(cnq, g, &uniq);
        self.expand_into(&packed, &slot, g, x);
    }

    /// # Safety
    /// No launch of this gather is pending.
    pub unsafe fn free(&mut self) {
        self.module.unload();
    }
}

/// The trunk input of the prompt rows `ids` into `x` (`[ids.len()][streams][hidden]` f32): with a
/// gather (`CROW_GLM_EMBED_GATHER=1`) [`Gather::trunk_into`], else the host path of before,
/// call for call (`embed_rows`, `trunk_input`, `to_f32_into`)
///
/// # Safety
/// A CUDA context is current; `x` holds `ids.len() x streams x hidden` f32.
pub unsafe fn trunk_into(gather: Option<&Gather>, cnq: &mut Cnq, g: &Glm5Geo, ids: &[i64], x: Dev) {
    match gather {
        Some(k) => k.trunk_into(cnq, g, ids, x),
        None => cuda::to_f32_into(x, &crate::glm5_model::trunk_input(&crate::glm5_model::embed_rows(cnq, g, ids), g.hidden, g.hc_streams)),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cnq;
    use crate::glm5_model::{embed_rows, trunk_input};

    /// splitmix64
    struct Rng(u64);

    impl Rng {
        fn u64(&mut self) -> u64 {
            self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut z = self.0;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            z ^ (z >> 31)
        }
    }

    /// a container holding only `embed_tokens` (`[vocab][hidden]` BF16 `raw`), removed on drop
    struct Table {
        dir: std::path::PathBuf,
        path: String,
    }

    impl Drop for Table {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.dir);
        }
    }

    fn table(g: &Glm5Geo, raw: &[u8], tag: &str) -> Table {
        use std::io::Write;
        assert_eq!(raw.len(), g.vocab * g.hidden * 2);
        let dir = std::env::temp_dir().join(format!("crow-glm5-embed-{tag}-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("embed.cnq");
        let mut f = std::io::BufWriter::new(std::fs::File::create(&path).unwrap());
        f.write_all(b"CNQ1\0\0\0\0\0\0\0\0").unwrap();
        // a few bytes in front of the table, so the tensor offset is not 0
        f.write_all(&[0xA5u8; 6]).unwrap();
        f.write_all(raw).unwrap();
        let sha = |s: &str| cnq::sha256_hex(s.as_bytes());
        let index = serde_json::json!({
            "format": "crow-nest-quant", "format_version": 2, "blob_offset": 12, "recipe": "synthetic-glm5-embed",
            "model": { "family": "Glm5Next", "model_type": "glm5_next_text", "config_json": "{}", "config_json_sha256": sha("{}"),
                "generation_config_json": "{}", "generation_config_json_sha256": sha("{}"),
                "source": { "repo": "crow-nest/synthetic-glm5-embed", "revision": "186", "shards": [] }, "geo": {} },
            "tensors": [{ "name": EMBED, "section": "text", "dtype": "bf16", "offset": 6, "n_values": (g.vocab * g.hidden) as u64, "shape": [g.vocab, g.hidden] }]
        });
        let ib = serde_json::to_vec(&index).unwrap();
        f.write_all(&ib).unwrap();
        f.write_all(&(ib.len() as u64).to_le_bytes()).unwrap();
        f.flush().unwrap();
        Table { dir, path: path.to_str().unwrap().to_string() }
    }

    /// random BF16 bits, plus NaN payloads (quiet and signalling, both signs), +-Inf, -0 and
    /// denormals in rows 0, 7 and the last
    fn raw_table(g: &Glm5Geo, seed: u64) -> Vec<u8> {
        let mut rng = Rng(seed);
        let mut raw = vec![0u8; g.vocab * g.hidden * 2];
        for w in raw.chunks_exact_mut(2) {
            w.copy_from_slice(&(rng.u64() as u16).to_le_bytes());
        }
        let specials: [u16; 9] = [0x7FC0, 0x7FC1, 0x7F81, 0xFF81, 0x7F80, 0xFF80, 0x8000, 0x0001, 0x807F];
        for (i, v) in specials.iter().enumerate() {
            for id in [0usize, 7, g.vocab - 1] {
                let at = (id * g.hidden + (i * 7) % g.hidden) * 2;
                raw[at..at + 2].copy_from_slice(&v.to_le_bytes());
            }
        }
        raw
    }

    /// prompt rows with repeats, the first and last id, runs of neighbours and far jumps
    fn prompt(g: &Glm5Geo, n: usize, seed: u64) -> Vec<i64> {
        let mut rng = Rng(seed);
        let v = g.vocab as u64;
        let mut ids: Vec<i64> = (0..n)
            .map(|i| match rng.u64() % 4 {
                0 => (rng.u64() % 16) as i64,
                1 => (rng.u64() % v) as i64,
                2 => ((i as u64 * 3) % v) as i64,
                _ => (v - 1 - rng.u64() % 40) as i64,
            })
            .collect();
        ids[0] = 0;
        ids[n / 2] = g.vocab as i64 - 1;
        ids
    }

    #[test]
    fn only_1_turns_the_embed_gather_on() {
        assert!(parse(Some("1")));
        for v in [None, Some("0"), Some(""), Some("on"), Some("true"), Some(" 1"), Some("2")] {
            assert!(!parse(v), "{v:?}");
        }
    }

    #[test]
    fn dedup_gives_ascending_distinct_ids_and_each_rows_index() {
        let ids = [5i64, 3, 5, 9, 0, 3, 3, 120, 9];
        let (u, s) = dedup(&ids);
        assert_eq!(u, vec![0, 3, 5, 9, 120]);
        assert_eq!(s, vec![2, 1, 2, 3, 0, 1, 1, 4, 3]);
        assert!(ids.iter().zip(&s).all(|(&id, &k)| u[k as usize] == id));
        assert_eq!(dedup(&[]), (vec![], vec![]));
    }

    #[test]
    fn spans_join_neighbours_up_to_the_gap() {
        assert_eq!(spans(&[], 8), vec![]);
        assert_eq!(spans(&[4], 8), vec![(0, 1)]);
        assert_eq!(spans(&[0, 1, 9, 18, 27, 100], 8), vec![(0, 3), (3, 4), (4, 5), (5, 6)]);
        assert_eq!(spans(&[0, 8, 16, 24], 8), vec![(0, 4)]);
        assert_eq!(spans(&[0, 1, 2], 0), vec![(0, 1), (1, 2), (2, 3)]);
    }

    /// The read path against the host path of before on a synthetic container: for prompts of 1 to
    /// 6,000 rows (one reader to [`READERS`]), the packed distinct rows expanded through the
    /// row map equal `trunk_input(embed_rows(..))` in every bit (NaN payloads, Inf, -0, denormals)
    #[test]
    fn the_gathered_rows_expand_to_the_host_trunk_input_bit_for_bit() {
        let mut g = Glm5Geo::GLM_5_3_FLASH;
        (g.vocab, g.hidden) = (20_000, 64);
        let raw = raw_table(&g, 0x186E);
        let tb = table(&g, &raw, "cpu");
        let mut cnq = Cnq::open_checked(&tb.path).unwrap();
        for (n, seed) in [(1usize, 1u64), (2, 2), (37, 3), (512, 4), (6000, 5)] {
            let ids = prompt(&g, n, seed);
            let (uniq, slot) = dedup(&ids);
            let packed = read_rows(&mut cnq, &g, &uniq);
            for (k, &id) in uniq.iter().enumerate() {
                let r = id as usize * g.hidden * 2;
                assert!(packed[k * g.hidden * 2..(k + 1) * g.hidden * 2] == raw[r..r + g.hidden * 2], "n {n}: row of id {id}");
            }
            let want: Vec<u32> = trunk_input(&embed_rows(&mut cnq, &g, &ids), g.hidden, g.hc_streams).iter().map(|v| v.to_bits()).collect();
            let got = expand_host(&packed, &slot, g.hidden, g.hc_streams);
            assert!(got == want, "n {n}: {} of {} values differ in bits", got.iter().zip(&want).filter(|(a, b)| a != b).count(), want.len());
            let readers = READERS.min(spans(&uniq, MAX_GAP).len().div_ceil(SPANS_PER_READER)).max(1);
            eprintln!("glm5 embed: n {n}, {} distinct, {} reads, {readers} readers", uniq.len(), spans(&uniq, MAX_GAP).len());
        }
        // a host widen of the specials keeps their bits (the reference itself is exact)
        let w = cnq::bf16_bytes_to_f32(&[0x81, 0x7F, 0x01, 0x00]);
        assert_eq!((w[0].to_bits(), w[1].to_bits()), (0x7F81_0000, 0x0001_0000));
    }

    /// The device gather against the host path of before, bit for bit: a synthetic container at
    /// the real hidden size (4,096, vocab 3,000, specials in three rows), prompt calls of 1 to
    /// 8,192 rows; `trunk_into` with the gather writes the same `x` as without it.
    #[test]
    #[ignore = "needs the GPU (about 0.6 GB VRAM): cargo test --release --lib glm5_embed_gpu -- --ignored --nocapture --test-threads 1"]
    fn glm5_embed_gpu_the_device_gather_is_the_host_path_bit_for_bit() {
        let mut g = Glm5Geo::GLM_5_3_FLASH;
        g.vocab = 3000;
        let raw = raw_table(&g, 0x186D);
        let tb = table(&g, &raw, "gpu");
        let mut cnq = Cnq::open_checked(&tb.path).unwrap();
        unsafe {
            let _ctx = cuda::Ctx::init();
            let mut k = Gather::new();
            for (n, seed) in [(1usize, 11u64), (3, 12), (300, 13), (8192, 14)] {
                let ids = prompt(&g, n, seed);
                let m = n * g.hc_streams * g.hidden;
                let (mut xo, mut xn) = (cuda::alloc_named("test embed x host", m * 4), cuda::alloc_named("test embed x gather", m * 4));
                cuda::upload_into(xn, &vec![0xEEu8; m * 4]);
                trunk_into(None, &mut cnq, &g, &ids, xo);
                trunk_into(Some(&k), &mut cnq, &g, &ids, xn);
                cuda::sync();
                let (a, b): (Vec<u32>, Vec<u32>) = (cuda::dtoh_t(xo, m), cuda::dtoh_t(xn, m));
                assert!(a == b, "n {n}: {} of {m} values differ in bits", a.iter().zip(&b).filter(|(p, q)| p != q).count());
                cuda::free_dev(&mut xo);
                cuda::free_dev(&mut xn);
                eprintln!("glm5 embed gpu: n {n}: x bit-identical ({} distinct ids)", dedup(&ids).0.len());
            }
            k.free();
        }
    }

    /// As above on the real container's `embed_tokens` (154,880 x 4,096): one prompt call of the
    /// ids in `GLM_EMBED_TEST_IDS` (comma-separated, e.g. `quick.sh`'s `prefill8192.ids`), else
    /// 8,192 ids of a fixed pattern over the whole vocab.
    #[test]
    #[ignore = "needs the GPU (about 0.6 GB VRAM) and the real container (CROW_CNQ): cargo test --release --lib glm5_embed_gpu -- --ignored --nocapture --test-threads 1"]
    fn glm5_embed_gpu_real_container_prompt_is_bit_identical() {
        let g = Glm5Geo::GLM_5_3_FLASH;
        let path = std::env::var("CROW_CNQ").unwrap_or_else(|_| crate::geo::from_engine_dir(crate::glm5_model::GLM5_MUL1K3_CNQ));
        let ids: Vec<i64> = match std::env::var("GLM_EMBED_TEST_IDS") {
            Ok(p) => std::fs::read_to_string(&p).unwrap().split(|c: char| !c.is_ascii_digit()).filter(|s| !s.is_empty()).map(|s| s.parse().unwrap()).collect(),
            Err(_) => prompt(&g, 8192, 0x8192),
        };
        let mut cnq = Cnq::open(&path);
        unsafe {
            let _ctx = cuda::Ctx::init();
            let mut k = Gather::new();
            let m = ids.len() * g.hc_streams * g.hidden;
            let (mut xo, mut xn) = (cuda::alloc_named("test embed x host", m * 4), cuda::alloc_named("test embed x gather", m * 4));
            trunk_into(None, &mut cnq, &g, &ids, xo);
            trunk_into(Some(&k), &mut cnq, &g, &ids, xn);
            cuda::sync();
            let (a, b): (Vec<u32>, Vec<u32>) = (cuda::dtoh_t(xo, m), cuda::dtoh_t(xn, m));
            assert!(a == b, "{} of {m} values differ in bits", a.iter().zip(&b).filter(|(p, q)| p != q).count());
            eprintln!("glm5 embed gpu real: {} rows, {} distinct, {} reads: x bit-identical", ids.len(), dedup(&ids).0.len(), spans(&dedup(&ids).0, MAX_GAP).len());
            cuda::free_dev(&mut xo);
            cuda::free_dev(&mut xn);
            k.free();
        }
    }
}
