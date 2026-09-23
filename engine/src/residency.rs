//! #8 — the expert residency scheduler (Strategy Z, spec 2.2/2.3/3.3/3.4):
//!
//! - Per-layer TOP-N hot sets by selection frequency (warm-up counts, persisted
//!   in a sidecar next to the container, refreshable by rerunning warm-up).
//! - Hot experts live in VRAM slabs, cold experts in a PINNED HOST tier read
//!   ZERO-COPY over PCIe inside the GEMM (probe 3: 21.6–25.1 GB/s) — the
//!   pointer table makes residency invisible to the kernel (p5 pattern).
//! - Routing-gated skip ON GPU: `router_top10` tests the per-layer bitmap —
//!   a layer whose 10 routed experts are all resident produces no cold work
//!   at all; the counters respond to exactly that (§3.3/§3.6 reporting).
//! - No host in the decode hot loop: pointers/ids/weights never leave the GPU;
//!   per-layer u64 counters [selections, cold] are drained by the control
//!   plane BETWEEN tokens (the job ring's control-plane role, spec 3.4).
//!
//! The expert FFN math itself lives in gen.rs (gate_up → silu·up → down), the
//! p5-verified chain over per-combo device pointer tables.

use crate::cuda::{self, Pinned};
use crate::geo::{E, GIB, LAYERS, MIB};
use crate::cnq::Cnq;
use cudarc::driver::sys::CUdeviceptr;
use std::collections::HashMap;

/// marker for an unoccupied hot slot (spare of the stream-side trickle)
pub const EMPTY: u32 = u32::MAX;

/// one in-flight swap of the stream-side trickle (A-P3c): phase A copies the
/// incoming expert's pinned bytes into the layer's spare hot slot, commit A
/// makes it resident (the evicted one stays reachable in its old slot),
/// phase B copies the evicted expert's exact bytes into the pinned slot the
/// incoming one left, commit B retires the old slot as the new spare.
pub struct PendingSwap {
    pub l: usize,
    pub evict_slot: usize,
    pub evict: u32,
    pub new_id: u32,
    pub cs: usize,
    pub spare: usize,
}

pub struct Residency {
    pub n: usize, // hot experts per layer (after clamp; logical, excludes spares)
    pub stride: usize, // hot slots allocated per layer (n + spares)
    pub gu_bytes: u64,
    pub dn_bytes: u64,
    pub sets: Vec<Vec<u32>>, // [48][stride] expert id per hot slot (EMPTY = spare)
    pub spare_free: Vec<Vec<usize>>, // [48] unoccupied hot slots
    pub hot_gu: CUdeviceptr, // [48][stride][gu_bytes]
    pub hot_dn: CUdeviceptr,
    pub cold_gu: Vec<Pinned>, // [48] compact cold-only slabs
    pub cold_dn: Vec<Pinned>,
    pub tables: CUdeviceptr,  // [48][512][2] u64 device pointers
    pub bitmaps: CUdeviceptr, // [48][16] u32
    pub counters: CUdeviceptr, // [48][2] u64
    pub cold_index: Vec<HashMap<u32, usize>>, // expert id -> cold slot per layer
    pub source: String,       // "sidecar" | "warmup"
    pub gs_dev: CUdeviceptr,  // [LAYERS][2] f32: gate_up gs, down gs (per-tensor global scales, per layer)
    /// low-bit cold tier (CROW_COLD_TIER): bits, device LUT (codes -> e2m1
    /// nibbles), device i32 bits; cold slabs then hold compact records
    pub lb: Option<LowBit>,
    /// VRAM bounce slot (one expert) for the exact three-way swap of the
    /// prompt-adaptive hot set on the NVFP4 tier (A-P3)
    pub bounce_gu: CUdeviceptr,
    pub bounce_dn: CUdeviceptr,
    /// every expert has a pinned record (low-bit full tier)
    pub full: bool,
}

pub struct LowBit {
    pub bits: u32,
    pub gu_rec: u64,
    pub dn_rec: u64,
    pub lut_dev: CUdeviceptr,
    pub bits_dev: CUdeviceptr,
}

/// header of a `coldtier` file (JSON trailer + u64 length, like CNQ)
pub fn read_coldtier_header(path: &str) -> (serde_json::Value, u64) {
    use std::io::{Read, Seek, SeekFrom};
    let mut f = std::fs::File::open(path).expect("cold tier file");
    let len = f.metadata().unwrap().len();
    f.seek(SeekFrom::Start(len - 8)).unwrap();
    let mut b8 = [0u8; 8];
    f.read_exact(&mut b8).unwrap();
    let hl = u64::from_le_bytes(b8);
    f.seek(SeekFrom::Start(len - 8 - hl)).unwrap();
    let mut hb = vec![0u8; hl as usize];
    f.read_exact(&mut hb).unwrap();
    (serde_json::from_slice(&hb).unwrap(), len)
}

pub struct ExpertSlabs {
    pub gu_bytes: u64,
    pub dn_bytes: u64,
    pub gu_gs: f32,
    pub dn_gs: f32,
}

pub fn expert_slab_info(cnq: &Cnq, layer: usize, section: &str) -> ExpertSlabs {
    let gu = cnq.find(&format!("model.language_model.layers.{layer}.mlp.experts.gate_up_proj"), section);
    let dn = cnq.find(&format!("model.language_model.layers.{layer}.mlp.experts.down_proj"), section);
    ExpertSlabs {
        gu_bytes: Cnq::byte_len(gu) / E as u64,
        dn_bytes: Cnq::byte_len(dn) / E as u64,
        gu_gs: gu.global_scale,
        dn_gs: dn.global_scale,
    }
}

/// Clamp ue4m3 scale byte 0x7F (the E4M3 NaN encoding) to 0x7E (448, the
/// hardware-representable max) in a raw expert slab. The container format's
/// CPU decode (kernels.rs ue4m3) reads 0x7F as 480 — fine for the naive
/// scalar GEMVs, but the mxf4nvf4 tensor-core instruction produces NaN for
/// that byte (mma_probe2), so slabs served to the MMA path must not carry it.
/// Rare (1 byte in ~5e7 at layer 9); error is a 6.7% scale reduction on the
/// affected 16-element subblocks only.
fn sanitize_sf_slab(raw: &mut [u8]) -> u64 {
    let mut n = 0u64;
    for blk in raw.chunks_exact_mut(36) {
        for b in blk.iter_mut().take(4) {
            if *b == 0x7F {
                *b = 0x7E;
                n += 1;
            }
        }
    }
    n
}

impl Residency {
    /// Load hot sets from the sidecar if present, else build from warm-up
    /// counts (per-layer selection frequencies) and persist. Then materialize
    /// slabs + tables: hot → VRAM, cold → pinned UVA.
    pub unsafe fn build(
        cnq: &mut Cnq,
        section: &str,
        n: usize,
        warmup_counts: Option<&[[u64; E]; LAYERS]>,
        sidecar_path: &str,
        persist: bool,
        full_tier_req: bool,
        spare: usize,
        progress: &mut dyn FnMut(&str),
    ) -> Residency {
        let slabs = expert_slab_info(cnq, 0, section);
        // `n` is the planned slot count; `spare` of them stay unoccupied for
        // the stream-side trickle (A-P3c) - the logical hot set is n - spare
        assert!(spare < n, "spare hot slots {spare} must leave a hot set (N={n})");
        let n_slots = n;
        let n = n - spare;
        if spare > 0 {
            progress(&format!("stream trickle: {spare} spare hot slot(s) per layer, logical hot set N={n} of {n_slots} slots"));
        }
        // all layers share the expert geometry (checkpoint fact, asserted once)
        for l in (1..LAYERS).step_by(16) {
            let s = expert_slab_info(cnq, l, section);
            assert_eq!((s.gu_bytes, s.dn_bytes), (slabs.gu_bytes, slabs.dn_bytes));
        }
        progress(&format!(
            "expert slabs per layer: gate_up {:.2} MB + down {:.2} MB per expert ({} experts, {} layers)",
            slabs.gu_bytes as f64 / MIB,
            slabs.dn_bytes as f64 / MIB,
            E,
            LAYERS
        ));

        // ---- sets: sidecar or warm-up ----
        // the parse, the per-row length rule (#49) and every refusal live in
        // `sidecar_sets` below, which is a pure function and unit tested
        let (sets, source) = if std::path::Path::new(sidecar_path).exists() {
            let txt = std::fs::read_to_string(sidecar_path)
                .unwrap_or_else(|e| panic!("hot-set sidecar {sidecar_path}: {e}"));
            let (sets, notes) = sidecar_sets(&txt, n)
                .unwrap_or_else(|e| panic!("hot-set sidecar {sidecar_path}: {e}"));
            for m in &notes {
                progress(m);
            }
            progress(&format!("hot sets loaded from sidecar {sidecar_path}"));
            (sets, "sidecar".to_string())
        } else {
            let counts = warmup_counts
                .expect("no sidecar and no warm-up counts — run the warm-up first");
            let mut sets = Vec::with_capacity(LAYERS);
            for l in 0..LAYERS {
                let mut ord: Vec<u32> = (0..E as u32).collect();
                ord.sort_by(|&a, &b| counts[l][b as usize].cmp(&counts[l][a as usize]).then(a.cmp(&b)));
                ord.truncate(n);
                // keep FREQUENCY order — the sidecar's ordering is what makes
                // a later N-truncate (two-sided budget clamp) keep the hottest
                // experts instead of the lowest ids (measured bug, 2026-09-03)
                sets.push(ord);
            }
            if persist {
                persist_sidecar(sidecar_path, n, &sets, &slabs, "engine warm-up run");
                progress(&format!("warm-up promoted top-{n} per layer → sidecar {sidecar_path}"));
            } else {
                progress(&format!("warm-up counts applied (top-{n} per layer, not persisted)"));
            }
            (sets, "warmup".to_string())
        };

        // ---- RAM guard: the cold tier is PINNED (unpageable). Refuse when it
        // would not leave a margin of physical RAM for the process heap and
        // the OS - running at the edge froze the machine on 2026-09-04.
        // low-bit cold tier: records instead of NVFP4 blocks in the pinned slabs
        let tier_path = std::env::var("CROW_COLD_TIER").ok();
        let lb_hdr = tier_path.as_ref().map(|p| read_coldtier_header(p));
        let (cold_gu_bytes, cold_dn_bytes) = match &lb_hdr {
            Some((h, _)) => (h["gu_record_bytes"].as_u64().unwrap(), h["dn_record_bytes"].as_u64().unwrap()),
            None => (slabs.gu_bytes, slabs.dn_bytes),
        };
        if let Some((h, _)) = &lb_hdr {
            progress(&format!("low-bit cold tier: {}-bit records {} + {} B per expert (NVFP4: {} B) from {}",
                h["bits"], cold_gu_bytes, cold_dn_bytes, slabs.gu_bytes + slabs.dn_bytes, tier_path.as_ref().unwrap()));
        }
        // #79: an expert overlay shadows the CONTAINER's expert bytes. A low-bit cold tier is a
        // SECOND file the pinned slabs are filled from, so under both at once only the hot
        // experts would carry the overlay and the arm would be half an arm. Named at boot
        // rather than measured.
        if lb_hdr.is_some() && cnq.overlay_tensors().iter().any(|t| t.name.contains(".mlp.experts.")) {
            panic!(
                "CROW_COLD_TIER={} and an expert overlay together: the pinned cold tier is filled from that \
file, so the overlay would reach the hot experts only",
                tier_path.as_deref().unwrap_or("?")
            );
        }
        // with the low-bit tier every expert is pinned (2.5 bpw: 37.7 GB for all
        // 512 x 48) so the hot set can be re-cut per prompt without host data
        // movement (A-P3); the NVFP4 tier stays cold-only (67.8 GB would not fit)
        let full_tier = lb_hdr.is_some() && full_tier_req;
        let cold_total: u64 = (0..LAYERS)
            .map(|l| (if full_tier { E } else { E - sets[l].len() }) as u64 * (cold_gu_bytes + cold_dn_bytes))
            .sum();
        // free_phys is the RAM that can be PINNED, not MemAvailable: the driver's
        // pinned-page pool and the page cache are both reclaimable and both
        // invisible to MemAvailable (cuda::free_physical_ram_parts). The planner
        // already sized the tier against `budget = min(cap, free_phys - margin)`
        // (manager::derive_host_pinned_budget), so this is the backstop for the
        // RAM that other processes took between the two readings.
        let free_phys = cuda::free_physical_ram();
        let margin = crate::manager::ram_margin_bytes();
        progress(&format!(
            "pinned cold tier {:.2} GiB vs free physical RAM {:.2} GiB (margin {:.0} GiB), allocation {:?} (CROW_PINNED_ALLOC)",
            cold_total as f64 / GIB, free_phys as f64 / GIB, margin as f64 / GIB, cuda::pin_alloc_mode()
        ));
        if free_phys > 0 && cold_total + margin > free_phys {
            panic!(
                "refusing to pin {:.2} GiB with only {:.2} GiB physical RAM free (margin {:.0} GiB): raise N (free VRAM), shrink the keep-set, or set CROW_RAM_MARGIN_GB",
                cold_total as f64 / GIB, free_phys as f64 / GIB, margin as f64 / GIB
            );
        }
        // ---- VRAM hot slabs + pinned cold slabs ----
        let hot_gu_bytes = (LAYERS * n_slots) as u64 * slabs.gu_bytes;
        let hot_dn_bytes = (LAYERS * n_slots) as u64 * slabs.dn_bytes;
        let hot_gu = cuda::alloc_zeroed(hot_gu_bytes as usize);
        let hot_dn = cuda::alloc_zeroed(hot_dn_bytes as usize);
        let mut cold_gu = Vec::with_capacity(LAYERS);
        let mut cold_dn = Vec::with_capacity(LAYERS);
        let mut cold_index = Vec::with_capacity(LAYERS);
        // hot experts: container -> staging -> VRAM slab (async + sync, WDDM rule)
        let mut stage = vec![0u8; slabs.gu_bytes.max(slabs.dn_bytes) as usize];
        for l in 0..LAYERS {
            let ncold = if full_tier { E } else { E - sets[l].len() };
            // #103 (2026-09-23): the cold slabs are anonymous memory
            // registered with the driver (`Pinned::alloc_cold`, CROW_PINNED_ALLOC,
            // default `register` on unix), so the kernel owns the pages and gets
            // them back on free, exit and SIGKILL. The old write-combined
            // `cuMemHostAlloc` (`CROW_PINNED_ALLOC=wc`, still the windows default)
            // left every freed page in the NVIDIA driver's sysmem pool. The 2026-09-04
            // WC argument (47.6 vs 24 GB/s) was a Windows measurement: on this Linux
            // box WC, cacheable and registered all read at 51.6 GB/s in the stage
            // pattern (`pin_return_probe bw`), and CROW_KPROF measured 244.5 vs
            // 244.7 us per stage call (architecture 4.8).
            let (mut pg, mut pd) = (
                Pinned::alloc_cold((ncold as u64 * cold_gu_bytes) as usize),
                Pinned::alloc_cold((ncold as u64 * cold_dn_bytes) as usize),
            );
            let mut tier_file = tier_path.as_ref().map(|p| crate::cnq::open_sequential(p)); // no cache retention (see cnq.rs)
            let mut idx = HashMap::with_capacity(ncold);
            let mut slot = 0usize;
            for id in 0..E as u32 {
                if full_tier || !sets[l].contains(&id) {
                    idx.insert(id, slot);
                    slot += 1;
                }
            }
            // cold rows stream straight from the container into pinned RAM
            let gname = format!("model.language_model.layers.{l}.mlp.experts.gate_up_proj");
            let dname = format!("model.language_model.layers.{l}.mlp.experts.down_proj");
            let gt = cnq.find(&gname, section).clone();
            let dt = cnq.find(&dname, section).clone();
            let mut done = 0usize;
            let mut clamped = 0u64;
            // the low-bit cold tier lives in its own file: fill the pinned slabs from
            // it here, and let the container sweep below serve the hot slabs only
            if let (Some(tf), Some((h, _))) = (tier_file.as_mut(), &lb_hdr) {
                use std::io::{Read, Seek, SeekFrom};
                let layer_bytes = h["layer_bytes"].as_u64().unwrap();
                let mut order: Vec<(u32, usize)> = idx.iter().map(|(&id, &cs)| (id, cs)).collect();
                order.sort_unstable();
                for &(id, cs) in &order {
                    let base = l as u64 * layer_bytes + id as u64 * (cold_gu_bytes + cold_dn_bytes);
                    let mut raw = vec![0u8; cold_gu_bytes as usize];
                    tf.seek(SeekFrom::Start(base)).unwrap();
                    tf.read_exact(&mut raw).unwrap();
                    pg.write_bytes((cs as u64 * cold_gu_bytes) as usize, &raw);
                    let mut raw = vec![0u8; cold_dn_bytes as usize];
                    tf.seek(SeekFrom::Start(base + cold_gu_bytes)).unwrap();
                    tf.read_exact(&mut raw).unwrap();
                    pd.write_bytes((cs as u64 * cold_dn_bytes) as usize, &raw);
                    done += 1;
                }
            }
            // ONE ascending sweep per expert tensor over every id: a hot id goes to
            // its VRAM slot, a cold id to its pinned slot. Same bytes into the same
            // destinations as the two separate passes this replaces - what changes is
            // that the tensor is now read WHOLE and IN ORDER, which is what lets
            // `cnq::fadvise_consumed` drop every byte behind the cursor. Skipping the
            // hot ids left their pages stranded: the kernel's readahead window sails
            // over a 1.76 MB gap, and nothing ever dropped what it pulled in (measured
            // 2026-09-17: the page cache still held 14 GiB at the end of the fill).
            let mut hot_slot = vec![usize::MAX; E];
            for (slot, &id) in sets[l].iter().enumerate() {
                hot_slot[id as usize] = slot;
            }
            let cold_here = lb_hdr.is_none(); // else the tier file above filled `pg`/`pd`
            for id in 0..E as u32 {
                let slot = hot_slot[id as usize];
                let cs = if cold_here { idx.get(&id).copied() } else { None };
                if slot == usize::MAX && cs.is_none() {
                    continue;
                }
                let mut raw = cnq.read_range(&gt, id as u64 * slabs.gu_bytes, slabs.gu_bytes as usize);
                let n = sanitize_sf_slab(&mut raw);
                if let Some(cs) = cs {
                    clamped += n;
                    pg.write_bytes((cs as u64 * slabs.gu_bytes) as usize, &raw);
                }
                if slot != usize::MAX {
                    stage[..raw.len()].copy_from_slice(&raw);
                    let dst = hot_gu as u64 + ((l * n_slots + slot) as u64) * slabs.gu_bytes;
                    cuda::upload_into(dst, &stage[..raw.len()]);
                }
            }
            for id in 0..E as u32 {
                let slot = hot_slot[id as usize];
                let cs = if cold_here { idx.get(&id).copied() } else { None };
                if slot == usize::MAX && cs.is_none() {
                    continue;
                }
                let mut raw = cnq.read_range(&dt, id as u64 * slabs.dn_bytes, slabs.dn_bytes as usize);
                let n = sanitize_sf_slab(&mut raw);
                if let Some(cs) = cs {
                    clamped += n;
                    pd.write_bytes((cs as u64 * slabs.dn_bytes) as usize, &raw);
                    done += 1;
                    if done % 128 == 0 {
                        progress(&format!(
                            "  layer {l}: cold experts {done}/{ncold} pinned ({:.0} MB)",
                            (done as u64 * (slabs.gu_bytes + slabs.dn_bytes)) as f64 / MIB
                        ));
                    }
                }
                if slot != usize::MAX {
                    stage[..raw.len()].copy_from_slice(&raw);
                    let dst = hot_dn as u64 + ((l * n_slots + slot) as u64) * slabs.dn_bytes;
                    cuda::upload_into(dst, &stage[..raw.len()]);
                }
            }
            if clamped > 0 {
                progress(&format!("  layer {l}: clamped {clamped} scale bytes 0x7F->0x7E (hardware NaN encoding)"));
            }
            if l % 8 == 0 {
                progress(&format!("  layer {l}: hot set on VRAM"));
            }
            cold_gu.push(pg);
            cold_dn.push(pd);
            cold_index.push(idx);
        }
        drop(stage);

        // ---- pointer tables + bitmaps + counters ----
        let tables = cuda::alloc_zeroed(LAYERS * E * 2 * 8);
        let bitmaps = cuda::alloc_zeroed(LAYERS * 16 * 4);
        let counters = cuda::alloc_zeroed(LAYERS * 2 * 8);
        let bounce_gu = cuda::alloc_zeroed(slabs.gu_bytes as usize);
        let bounce_dn = cuda::alloc_zeroed(slabs.dn_bytes as usize);
        let mut tbl = vec![0u64; LAYERS * E * 2];
        let mut bmp = vec![0u32; LAYERS * 16];
        for l in 0..LAYERS {
            for (&id, &cs) in &cold_index[l] {
                tbl[(l * E + id as usize) * 2] = cold_gu[l].dev as u64 + (cs as u64) * cold_gu_bytes;
                tbl[(l * E + id as usize) * 2 + 1] =
                    cold_dn[l].dev as u64 + (cs as u64) * cold_dn_bytes;
            }
            for (slot, &id) in sets[l].iter().enumerate() {
                tbl[(l * E + id as usize) * 2] =
                    hot_gu as u64 + ((l * n_slots + slot) as u64) * slabs.gu_bytes;
                tbl[(l * E + id as usize) * 2 + 1] =
                    hot_dn as u64 + ((l * n_slots + slot) as u64) * slabs.dn_bytes;
            }
            for &id in &sets[l] {
                bmp[l * 16 + (id >> 5) as usize] |= 1 << (id & 31);
            }
        }
        cuda::to_u64_into(tables, &tbl);
        upload_u32(bitmaps, &bmp);

        // per-tensor global scales DIFFER per layer (sidecar ratio range was
        // 0.42..1.77 against layer 0) — [LAYERS][2] f32, indexed by moe_run
        let mut gs_all = vec![0f32; LAYERS * 2];
        for (li, g) in gs_all.chunks_exact_mut(2).enumerate() {
            let s = expert_slab_info(cnq, li, section);
            g[0] = s.gu_gs;
            g[1] = s.dn_gs;
        }
        let lb = lb_hdr.as_ref().map(|(h, _)| {
            let bits = h["bits"].as_u64().unwrap() as u32;
            let nibs: Vec<u8> = h["codebook_nibbles"].as_array().unwrap().iter().map(|v| v.as_u64().unwrap() as u8).collect();
            let mut lut = vec![0u8; 16];
            lut[..nibs.len()].copy_from_slice(&nibs);
            let lut_dev = cuda::upload_dev(&lut);
            LowBit { bits, gu_rec: cold_gu_bytes, dn_rec: cold_dn_bytes, lut_dev, bits_dev: cuda::to_i32_dev(&[bits as i32]) }
        });
        let gs_dev = cuda::alloc_zeroed((LAYERS * 8) as usize);
        cuda::to_f32_into(gs_dev, &gs_all);

        // spare slots: unoccupied, marked EMPTY in the slot map
        let mut sets = sets;
        let mut spare_free = Vec::with_capacity(LAYERS);
        for (l, s) in sets.iter_mut().enumerate() {
            // an internal invariant since #49: a sidecar row is normalized to
            // exactly `n` by `sidecar_sets`, a warm-up row is a top-n of E
            assert_eq!(s.len(), n, "layer {l} hot set holds {} ids, not N={n}", s.len());
            s.resize(n_slots, EMPTY);
            spare_free.push((n..n_slots).collect::<Vec<usize>>());
        }
        Residency {
            n,
            stride: n_slots,
            gu_bytes: slabs.gu_bytes,
            dn_bytes: slabs.dn_bytes,
            sets,
            spare_free,
            hot_gu,
            hot_dn,
            cold_gu,
            cold_dn,
            tables,
            bitmaps,
            counters,
            cold_index,
            source,
            gs_dev,
            lb,
            bounce_gu,
            bounce_dn,
            full: full_tier,
        }
    }

    /// #17 bundled variant of `swap_in` (exact NVFP4 tier only): does the
    /// bookkeeping (cold index, sets) and appends the (hot, cold) slab pointer
    /// pairs for gate_up and down; the caller exchanges the bytes of ALL pairs
    /// of a tick with one `swap_pairs` launch per size class. Bit-identical to
    /// the bounce path: the same bytes end in the same places.
    pub fn swap_in_bundled(&mut self, l: usize, slot: usize, evict: u32, new_id: u32,
                           gu_pairs: &mut Vec<(u64, u64)>, dn_pairs: &mut Vec<(u64, u64)>) {
        assert!(self.lb.is_none(), "swap_in_bundled: exact NVFP4 tier only");
        let (dst_gu, dst_dn) = self.hot_ptrs(l, slot);
        let cs = self.cold_index[l].remove(&new_id).expect("incoming expert must be cold");
        let (src_gu, src_dn) = self.cold_ptrs(l, cs);
        gu_pairs.push((dst_gu, src_gu));
        dn_pairs.push((dst_dn, src_dn));
        self.cold_index[l].insert(evict, cs);
        self.sets[l][slot] = new_id;
    }

    /// #17: exchange the bytes of the collected pairs (one launch per size class)
    pub unsafe fn swap_pairs_launch(&self, k: &crate::kernels::Kernels, gu_pairs: &[(u64, u64)], dn_pairs: &[(u64, u64)]) {
        for (pairs, bytes) in [(gu_pairs, self.gu_bytes), (dn_pairs, self.dn_bytes)] {
            if pairs.is_empty() { continue; }
            assert!(bytes % 16 == 0);
            let a: Vec<u64> = pairs.iter().map(|p| p.0).collect();
            let b: Vec<u64> = pairs.iter().map(|p| p.1).collect();
            let mut da = cuda::to_u64_dev(&a);
            let mut db = cuda::to_u64_dev(&b);
            let mut nb = cuda::to_i32_dev(&[bytes as i32]);
            let split: u32 = 8;
            crate::kernels::launch_v(k.f("swap_pairs"), pairs.len() as u32, split, 1, 256, &[da, db, nb]);
            cuda::sync();
            cuda::free_dev(&mut da);
            cuda::free_dev(&mut db);
            cuda::free_dev(&mut nb);
        }
    }

    /// Prompt-adaptive residency (A-P3): replace hot slot `slot` of layer `l`
    /// (currently expert `evict`) by expert `new_id`, expanded from its pinned
    /// low-bit record on the GPU. Updates pointer table, bitmap and `sets`.
    /// Requires the full low-bit tier (every expert pinned). Stream-ordered.
    pub unsafe fn swap_in(&mut self, k: &crate::kernels::Kernels, l: usize, slot: usize, evict: u32, new_id: u32,
                          nblk_gu: CUdeviceptr, nblk_dn: CUdeviceptr) {
        let (dst_gu, dst_dn) = self.hot_ptrs(l, slot);
        if self.lb.is_none() {
            // exact NVFP4 tier (cold-only pinned): three-way exchange through the
            // VRAM bounce slot — the incoming expert's pinned slot receives the
            // evicted expert's exact bytes, so host RAM never grows and every
            // expert stays reachable. Stream-ordered copies (UVA).
            let cs = self.cold_index[l].remove(&new_id).expect("incoming expert must be cold");
            let (src_gu, src_dn) = self.cold_ptrs(l, cs);
            cuda::memcpy_async(self.bounce_gu, src_gu, self.gu_bytes as usize);
            cuda::memcpy_async(self.bounce_dn, src_dn, self.dn_bytes as usize);
            cuda::memcpy_async(src_gu, dst_gu, self.gu_bytes as usize);
            cuda::memcpy_async(src_dn, dst_dn, self.dn_bytes as usize);
            cuda::memcpy_async(dst_gu, self.bounce_gu, self.gu_bytes as usize);
            cuda::memcpy_async(dst_dn, self.bounce_dn, self.dn_bytes as usize);
            self.cold_index[l].insert(evict, cs);
            self.sets[l][slot] = new_id;
            return;
        }
        let lb = self.lb.as_ref().unwrap();
        let cs = *self.cold_index[l].get(&new_id).expect("full tier: every expert has a record");
        let rec_gu = self.cold_gu[l].dev as u64 + cs as u64 * lb.gu_rec;
        let rec_dn = self.cold_dn[l].dev as u64 + cs as u64 * lb.dn_rec;
        crate::kernels::launch_v(k.f("expand_slab"), 200, 1, 1, 256, &[rec_gu, dst_gu, nblk_gu as u64, lb.bits_dev as u64, lb.lut_dev as u64]);
        crate::kernels::launch_v(k.f("expand_slab"), 100, 1, 1, 256, &[rec_dn, dst_dn, nblk_dn as u64, lb.bits_dev as u64, lb.lut_dev as u64]);
        self.sets[l][slot] = new_id;
    }

    /// Build layer `l`'s pointer table (512 x 2 u64) and its 16-word hot
    /// bitmap from the host bookkeeping: every expert's cold record first,
    /// then the hot residents written over the top. The ONE place that order
    /// lives - both flushes fill their buffer through it.
    fn fill_layer_table(&self, l: usize, tbl: &mut [u64], words: &mut [u32]) {
        let (cgu, cdn) = match &self.lb { Some(lb) => (lb.gu_rec, lb.dn_rec), None => (self.gu_bytes, self.dn_bytes) };
        for (&id, &cs) in &self.cold_index[l] {
            tbl[id as usize * 2] = self.cold_gu[l].dev as u64 + cs as u64 * cgu;
            tbl[id as usize * 2 + 1] = self.cold_dn[l].dev as u64 + cs as u64 * cdn;
        }
        for (slot, &id) in self.sets[l].iter().enumerate() {
            if id == EMPTY { continue; }
            let (gu, dn) = self.hot_ptrs(l, slot);
            tbl[id as usize * 2] = gu;
            tbl[id as usize * 2 + 1] = dn;
            words[(id >> 5) as usize] |= 1u32 << (id & 31);
        }
    }

    /// one layer's table + bitmap, uploaded once — called after a layer's
    /// swaps (two small HtoD copies per layer instead of four syncs per swap)
    pub unsafe fn flush_layer_tables(&self, l: usize) {
        let mut tbl = vec![0u64; E * 2];
        let mut words = [0u32; 16];
        self.fill_layer_table(l, &mut tbl, &mut words);
        cuda::to_u64_into(self.tables + (l * E * 2 * 8) as u64, &tbl);
        upload_u32(self.bitmaps + (l * 16 * 4) as u64, &words);
    }

    /// rebuild EVERY layer's pointer table and bitmap from the host
    /// bookkeeping and upload them in two copies (the trickle tick touches
    /// up to 48 layers per token; 96 small uploads cost ~2 ms of host time)
    pub unsafe fn flush_all_tables(&self) {
        let mut tbl = vec![0u64; LAYERS * E * 2];
        let mut words = vec![0u32; LAYERS * 16];
        for l in 0..LAYERS {
            self.fill_layer_table(l, &mut tbl[l * E * 2..(l + 1) * E * 2], &mut words[l * 16..(l + 1) * 16]);
        }
        cuda::to_u64_into(self.tables, &tbl);
        upload_u32(self.bitmaps, &words);
    }

    /// device pointers of hot slot `slot` of layer `l`
    fn hot_ptrs(&self, l: usize, slot: usize) -> (u64, u64) {
        (self.hot_gu as u64 + ((l * self.stride + slot) as u64) * self.gu_bytes,
         self.hot_dn as u64 + ((l * self.stride + slot) as u64) * self.dn_bytes)
    }

    /// UVA pointers of cold slot `cs` of layer `l` on the EXACT NVFP4 tier
    /// (record stride == slab size). The low-bit tier strides by `lb.gu_rec`
    /// / `lb.dn_rec` instead and computes its own - see `swap_in`.
    fn cold_ptrs(&self, l: usize, cs: usize) -> (u64, u64) {
        (self.cold_gu[l].dev as u64 + cs as u64 * self.gu_bytes,
         self.cold_dn[l].dev as u64 + cs as u64 * self.dn_bytes)
    }

    /// rank layer `l` by the cumulative routing counts `c`: up to `k` (0 = all)
    /// (slot, evicted id, incoming id) pairs - the hottest absent experts
    /// against the coldest residents. Residents win ties (a short prompt must
    /// not evict the static prior for zero-count experts in id order); slots
    /// in `excl` (swaps in flight) are left alone.
    pub fn plan_swaps(&self, l: usize, c: &[u64], k: usize, excl: &std::collections::HashSet<usize>) -> Vec<(usize, u32, u32)> {
        let n = self.n;
        // resident flag per expert (array, not a hash set: the comparator runs
        // ~5k times per layer and 48 layers per tick)
        let mut have = [false; E];
        for &e in &self.sets[l] { if e != EMPTY { have[e as usize] = true; } }
        let mut want: Vec<u32> = (0..E as u32).collect();
        want.sort_unstable_by(|&a, &b| c[b as usize].cmp(&c[a as usize])
            .then(have[b as usize].cmp(&have[a as usize])).then(a.cmp(&b)));
        want.truncate(n);
        let mut want_set = [false; E];
        for &e in &want { want_set[e as usize] = true; }
        // candidates in: hottest first; slots out: coldest resident first
        let incoming: Vec<u32> = want.iter().copied().filter(|&e| !have[e as usize]).collect();
        let mut out_slots: Vec<(usize, u32)> = self.sets[l].iter().enumerate()
            .filter(|(s, e)| **e != EMPTY && !want_set[**e as usize] && !excl.contains(s))
            .map(|(s, &e)| (s, e)).collect();
        out_slots.sort_by(|a, b| c[a.1 as usize].cmp(&c[b.1 as usize]).then(a.1.cmp(&b.1)));
        let k = if k == 0 { incoming.len() } else { incoming.len().min(k) };
        let k = k.min(out_slots.len());
        (0..k).map(|i| (out_slots[i].0, out_slots[i].1, incoming[i])).collect()
    }

    /// stream trickle phase A on `s`: the incoming expert's pinned bytes ->
    /// the layer's spare hot slot (nothing references that slot yet)
    pub unsafe fn swap_stream_a(&mut self, l: usize, evict_slot: usize, new_id: u32, s: cudarc::driver::sys::CUstream) -> PendingSwap {
        assert!(self.lb.is_none(), "stream trickle: exact NVFP4 tier only");
        let spare = self.spare_free[l].pop().expect("stream trickle: no spare hot slot");
        let cs = *self.cold_index[l].get(&new_id).expect("incoming expert must be cold");
        let evict = self.sets[l][evict_slot];
        assert!(evict != EMPTY, "stream trickle: evicting an empty slot");
        let (src_gu, src_dn) = self.cold_ptrs(l, cs);
        let (dst_gu, dst_dn) = self.hot_ptrs(l, spare);
        cuda::memcpy_async_on(dst_gu, src_gu, self.gu_bytes as usize, s);
        cuda::memcpy_async_on(dst_dn, src_dn, self.dn_bytes as usize, s);
        PendingSwap { l, evict_slot, evict, new_id, cs, spare }
    }

    /// commit A (host bookkeeping; the caller flushes the tables after the
    /// phase-A copies landed): the incoming expert is resident in the spare
    /// slot, the evicted one stays resident in its old slot until commit B
    pub fn swap_commit_a(&mut self, p: &PendingSwap) {
        self.cold_index[p.l].remove(&p.new_id);
        self.sets[p.l][p.spare] = p.new_id;
    }

    /// stream trickle phase B on `s`: the evicted expert's exact hot bytes ->
    /// the pinned slot the incoming expert left (no table points there since
    /// commit A; the hot slot is only READ concurrently)
    pub unsafe fn swap_stream_b(&self, p: &PendingSwap, s: cudarc::driver::sys::CUstream) {
        let (src_gu, src_dn) = self.hot_ptrs(p.l, p.evict_slot);
        let (dst_gu, dst_dn) = self.cold_ptrs(p.l, p.cs);
        cuda::memcpy_async_on(dst_gu, src_gu, self.gu_bytes as usize, s);
        cuda::memcpy_async_on(dst_dn, src_dn, self.dn_bytes as usize, s);
    }

    /// commit B (host bookkeeping; tables flushed by the caller): the evicted
    /// expert is cold again, its old hot slot becomes the spare
    pub fn swap_commit_b(&mut self, p: &PendingSwap) {
        self.cold_index[p.l].insert(p.evict, p.cs);
        self.sets[p.l][p.evict_slot] = EMPTY;
        self.spare_free[p.l].push(p.evict_slot);
    }

    /// control-plane drain (between tokens / chunks, never per layer)
    pub unsafe fn drain_counters(&self) -> Vec<[u64; 2]> {
        let raw = cuda::dtoh_u64(self.counters, LAYERS * 2);
        (0..LAYERS).map(|l| [raw[l * 2], raw[l * 2 + 1]]).collect()
    }

    pub fn layer_ptrs(&self, layer: usize) -> (u64, u64, u64, u64, u64) {
        (
            self.tables as u64 + (layer * E * 2 * 8) as u64,
            self.bitmaps as u64 + (layer * 16 * 4) as u64,
            self.counters as u64 + (layer * 2 * 8) as u64,
            self.gu_bytes,
            self.dn_bytes,
        )
    }

    pub fn pinned_bytes(&self) -> u64 {
        self.cold_gu.iter().map(|p| p.bytes as u64).sum::<u64>()
            + self.cold_dn.iter().map(|p| p.bytes as u64).sum::<u64>()
    }
}

fn upload_u32(dst: CUdeviceptr, v: &[u32]) {
    unsafe {
        let bytes = std::slice::from_raw_parts(v.as_ptr() as *const u8, v.len() * 4);
        cuda::upload_into(dst, bytes);
    }
}

/// The `sets` of a hot-set sidecar, parsed and normalized to `LAYERS` rows of
/// exactly `n` expert ids, plus the lines the loader logs about what it did;
/// `Err(msg)` names a file that cannot be read as a hot set at all.
///
/// `#49` (found 2026-09-10, fixed 2026-09-18): a RAGGED sidecar - rows of
/// unequal length, which is what the pre-`#52` warm-up left next to the
/// container - survived the load here and died two hundred lines later in
/// `assert_eq!(s.len(), n)` with `47 != 155`, naming neither the file, nor the
/// row, nor its length. Worse, whether it died at all depended on ROW 0: the
/// old code took the sidecar's N from `sets[0].len()` and adapted every row
/// only when that one differed from the config N.
///
/// The rule now is that same adapt rule applied to EVERY row on its own: a
/// short row is PADDED with the lowest unused expert ids, a long one is
/// TRUNCATED, and each adapted row is named with its own length.
///
/// Why padding is the meaning, and not a refusal: the planner gives every
/// layer the same `n` hot slots whatever the file says, so a short row does
/// not describe a smaller layer - it leaves slots the run has already paid for
/// unspecified. Filling them is a placement choice and not a numeric one,
/// because residency is numerically invisible on the default tier (an expert's
/// sanitized bytes are identical in a VRAM slot and in the pinned cold tier;
/// the pointer table only decides which of the two the kernel reads -
/// architecture 7.5 condition 2). Leaving them EMPTY instead would strand that
/// VRAM and put `spare_free` out of step with the occupied slots, and a row
/// LONGER than the stride would hand the stream trickle a slot that holds a
/// live expert. The padding is deterministic - the lowest unused ids in
/// ascending order, appended AFTER the file's own ids, so the sidecar's
/// frequency order stays first and a later truncate still keeps the hottest
/// experts - so one sidecar at one N always yields one hot set, which is what
/// the lossy `CROW_COLD_TIER` tier needs, where the hot set IS numeric.
///
/// REFUSED by name, because no padding gives it a meaning: a file that is not
/// one JSON object with a `sets` array of `LAYERS` rows (the converter's
/// `*.cnq.sidecar.jsonl` is a per-tensor quantization report with one JSON
/// object per LINE, not a hot set, and lands here), an entry that is not an
/// expert id, an id outside `0..E`, and an id named twice in one row - a
/// duplicate would leave the layer's cold tier, sized `E - sets[l].len()`, one
/// slot short of the ids indexed into it, and the fill would die in
/// `Pinned::write_bytes`'s own bounds assert instead.
pub fn sidecar_sets(txt: &str, n: usize) -> Result<(Vec<Vec<u32>>, Vec<String>), String> {
    if n == 0 || n > E {
        return Err(format!("N={n} is not a hot-set size (1..={E} experts per layer)"));
    }
    let v: serde_json::Value = serde_json::from_str(txt).map_err(|e| {
        format!("not one JSON object ({e}); a hot-set sidecar is one JSON object with a \"sets\" array of {LAYERS} rows of expert ids")
    })?;
    let rows = v.get("sets").and_then(|s| s.as_array()).ok_or_else(|| {
        format!("no \"sets\" array; a hot-set sidecar is one JSON object with a \"sets\" array of {LAYERS} rows of expert ids")
    })?;
    if rows.len() != LAYERS {
        return Err(format!("\"sets\" has {} rows, the model has {LAYERS} layers", rows.len()));
    }
    let mut sets: Vec<Vec<u32>> = Vec::with_capacity(LAYERS);
    for (l, r) in rows.iter().enumerate() {
        let a = r.as_array().ok_or_else(|| format!("row {l} is not an array of expert ids"))?;
        let mut row: Vec<u32> = Vec::with_capacity(a.len());
        for (i, x) in a.iter().enumerate() {
            let id = x.as_u64().ok_or_else(|| format!("row {l} entry {i} is not an expert id: {x}"))?;
            if id >= E as u64 {
                return Err(format!("row {l} entry {i} names expert {id}, outside 0..{E}"));
            }
            let id = id as u32;
            if row.contains(&id) {
                return Err(format!("row {l} names expert {id} twice (entry {i}): a hot set is a set"));
            }
            row.push(id);
        }
        sets.push(row);
    }
    // ---- the length rule (#49): per row, never from row 0 ----
    let lens: Vec<usize> = sets.iter().map(|s| s.len()).collect();
    let (lo, hi) = (*lens.iter().min().unwrap(), *lens.iter().max().unwrap());
    let mut notes = Vec::new();
    if lo == hi && lo != n {
        // one N for the whole file: the two-sided budget clamp moved N between
        // the warm-up that wrote it and this run - one line, as before #49
        notes.push(format!("sidecar N={lo} adapted to config N={n} (deterministic truncate/extend)"));
    } else if lo != hi {
        notes.push(format!(
            "ragged sidecar (#49): the {LAYERS} rows are {lo}..{hi} ids long, not all N={n} - every row is adapted on its own"
        ));
    }
    let mut named = 0usize;
    let mut unnamed = 0usize;
    for (l, s) in sets.iter_mut().enumerate() {
        let had = s.len();
        if had == n {
            continue;
        }
        s.truncate(n);
        if s.len() < n {
            let mut extra = (0..E as u32).filter(|id| !s.contains(id)).collect::<Vec<_>>();
            extra.truncate(n - s.len());
            s.extend(extra);
        }
        if lo == hi {
            continue; // already said once, above
        }
        // name the row and its length - the message #49 asked for. The first
        // eight say it in full; a file where every row is ragged gets a count
        // instead of 48 boot-log lines.
        if named < 8 {
            named += 1;
            notes.push(if had > n {
                format!("  row {l} has {had} ids: truncated to the first {n} (the file's own order)")
            } else {
                format!("  row {l} has {had} ids: padded with the {} lowest unused expert ids", n - had)
            });
        } else {
            unnamed += 1;
        }
    }
    if unnamed > 0 {
        notes.push(format!("  ... and {unnamed} further rows adapted the same way"));
    }
    Ok((sets, notes))
}

pub fn persist_sidecar(
    path: &str,
    n: usize,
    sets: &[Vec<u32>],
    slabs: &ExpertSlabs,
    provenance: &str,
) {
    let v = serde_json::json!({
        "version": 1,
        "n_per_layer": n,
        "strategy": "Z zero-copy direct read (spec 3.4 amended)",
        "expert_bytes": { "gate_up": slabs.gu_bytes, "down": slabs.dn_bytes },
        "provenance": provenance,
        "sets": sets,
    });
    std::fs::write(path, serde_json::to_string_pretty(&v).unwrap()).unwrap();
}

impl Drop for Residency {
    fn drop(&mut self) {
        unsafe { cuda::drop_dbg("before Residency"); }
        unsafe {
            cuda::free_dev(&mut self.hot_gu);
            cuda::free_dev(&mut self.hot_dn);
            cuda::free_dev(&mut self.tables);
            cuda::free_dev(&mut self.bitmaps);
            cuda::free_dev(&mut self.counters);
            cuda::free_dev(&mut self.gs_dev);
            cuda::free_dev(&mut self.bounce_gu); // #18
            cuda::free_dev(&mut self.bounce_dn);
            for p in self.cold_gu.iter_mut() { p.free(); }
            for p in self.cold_dn.iter_mut() { p.free(); }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `lens` for a file whose rows are all `k` long
    fn uniform(k: usize, rows: usize) -> Vec<usize> {
        std::iter::repeat_n(k, rows).collect()
    }

    /// a sidecar in the shape the loader writes, with `lens` per row
    fn sidecar(lens: &[usize]) -> String {
        let sets: Vec<Vec<u32>> = lens.iter().map(|&k| (0..k as u32).map(|i| (i * 3 + 7) % E as u32).collect()).collect();
        // the ids above are distinct for every k used here (k <= 170, stride 3
        // against 512), which is what a real row is
        serde_json::json!({ "version": 1, "n_per_layer": lens[0], "sets": sets }).to_string()
    }

    /// `#49`: rows of unequal length are adapted row by row and every adapted
    /// row is NAMED with its own length - the old loader asserted `47 != 155`
    /// two hundred lines further down, and only when row 0 happened to be N.
    #[test]
    fn a_ragged_sidecar_is_padded_row_by_row_and_names_every_row() {
        let n = 160;
        let mut lens = uniform(n, LAYERS);
        lens[0] = n; // row 0 is exactly N: the case the old code could not see
        lens[3] = 47; // the length of the finding
        lens[12] = 0;
        lens[17] = 170; // longer than N
        let (sets, notes) = sidecar_sets(&sidecar(&lens), n).expect("a ragged sidecar loads");
        assert_eq!(sets.len(), LAYERS);
        for (l, s) in sets.iter().enumerate() {
            assert_eq!(s.len(), n, "row {l}");
            let mut u = s.clone();
            u.sort_unstable();
            u.dedup();
            assert_eq!(u.len(), n, "row {l} holds an expert twice");
            assert!(u.iter().all(|&id| id < E as u32), "row {l} holds an id outside 0..{E}");
        }
        let all = notes.join("\n");
        assert!(all.contains("ragged sidecar (#49)"), "{all}");
        assert!(all.contains("0..170 ids long"), "{all}");
        assert!(all.contains("row 3 has 47 ids: padded with the 113 lowest unused expert ids"), "{all}");
        assert!(all.contains("row 12 has 0 ids: padded with the 160 lowest unused expert ids"), "{all}");
        assert!(all.contains("row 17 has 170 ids: truncated to the first 160"), "{all}");
        // the file's own ids come first, the padding after (frequency order)
        assert_eq!(&sets[3][..47], &sidecar_sets(&sidecar(&uniform(47, LAYERS)), 47).unwrap().0[3][..]);
    }

    /// one N for the whole file is the pre-#49 case and keeps its one line
    #[test]
    fn a_sidecar_cut_at_another_n_keeps_its_single_adapted_line() {
        let (sets, notes) = sidecar_sets(&sidecar(&uniform(160, LAYERS)), 155).unwrap();
        assert!(sets.iter().all(|s| s.len() == 155));
        assert_eq!(notes.len(), 1, "{notes:?}");
        assert_eq!(notes[0], "sidecar N=160 adapted to config N=155 (deterministic truncate/extend)");
        let (sets, notes) = sidecar_sets(&sidecar(&uniform(160, LAYERS)), 160).unwrap();
        assert!(sets.iter().all(|s| s.len() == 160));
        assert!(notes.is_empty(), "{notes:?}");
    }

    /// a file that is not a hot set is refused with a message that names what
    /// it is - `CROW_HOTSETS` pointed at the converter's per-tensor sidecar
    /// (one JSON object per LINE) is the case from the ticket
    #[test]
    fn a_file_that_is_not_a_hot_set_is_refused_by_name() {
        let jsonl = "{\"dtype\":\"bf16\",\"n\":10240,\"name\":\"a.weight\",\"section\":\"text\"}\n\
                     {\"dtype\":\"nvfp4\",\"n\":3276800,\"name\":\"b.weight\",\"section\":\"text\"}\n";
        let e = sidecar_sets(jsonl, 160).unwrap_err();
        assert_eq!(
            e,
            "not one JSON object (trailing characters at line 2 column 1); a hot-set sidecar is \
             one JSON object with a \"sets\" array of 48 rows of expert ids"
        );

        let e = sidecar_sets("{\"version\":1}", 160).unwrap_err();
        assert!(e.starts_with("no \"sets\" array"), "{e}");

        let e = sidecar_sets(&sidecar(&uniform(160, LAYERS - 1)), 160).unwrap_err();
        assert_eq!(e, "\"sets\" has 47 rows, the model has 48 layers");

        let mut v: serde_json::Value = serde_json::from_str(&sidecar(&uniform(4, LAYERS))).unwrap();
        v["sets"][5][2] = serde_json::json!(E);
        let e = sidecar_sets(&v.to_string(), 4).unwrap_err();
        assert_eq!(e, format!("row 5 entry 2 names expert {E}, outside 0..{E}"));

        v["sets"][5][2] = v["sets"][5][0].clone();
        let e = sidecar_sets(&v.to_string(), 4).unwrap_err();
        assert_eq!(e, "row 5 names expert 7 twice (entry 2): a hot set is a set");

        v["sets"][5][2] = serde_json::json!("7");
        let e = sidecar_sets(&v.to_string(), 4).unwrap_err();
        assert_eq!(e, "row 5 entry 2 is not an expert id: \"7\"");

        v["sets"][5] = serde_json::json!(160);
        let e = sidecar_sets(&v.to_string(), 4).unwrap_err();
        assert_eq!(e, "row 5 is not an array of expert ids");
    }
}
