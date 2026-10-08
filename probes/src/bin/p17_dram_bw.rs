//! Probe 17 (crow-nest #170, GLM-5.3-Flash plan step 4): DRAM read bandwidth per core class.
//!
//! Replaces the nominal 89.6 GB/s (2 channels x 5600 MT/s x 8 B) with a measured number.
//! Streaming AVX2 reads over a buffer of >= 8 GiB (STREAM's rule is 4x the last-level cache;
//! the 285K has 36 MB), threads pinned one per core, in three configurations detected at run
//! time from `GetSystemCpuSetInformation` `EfficiencyClass`: all cores, the highest class
//! (P cores), the lowest class (E cores).
//!
//! Method: the buffer holds `w[i] = i` (u64), so the expected sum of one pass is the closed
//! form n(n-1)/2 mod 2^64, independent of any read. Every run sums what it read and compares
//! against that; a run whose sum differs is void, so a read loop that is optimised away or
//! skips memory cannot produce a number. Work is claimed in 4 MiB chunks from one atomic
//! counter, so fast and slow cores both stay busy (a static equal split would let the E cores
//! set the wall time of the `all` configuration). Per configuration: one unmeasured warm-up
//! run, then `--reps` (>= 3) measured runs interleaved across configurations; median GB/s
//! (10^9 B/s), min, max, spread = max/min. Spread > 1.15 -> VOID. Any run above the nominal
//! 89.6 GB/s -> INVALID (that is cache or a wrong buffer, not DRAM).
//!
//! Output: JSON on stdout (and `--out FILE`). Exit 0 valid, 1 void/invalid/checksum mismatch,
//! 2 refusal. `--selftest`: 256 MiB, 1 pass, 3 reps, checksum only, no bandwidth verdict.
//!
//! Full run (idle machine only, never next to a download or another long job):
//!   cargo run -p crow-nest-probes --release --bin p17_dram_bw -- --gib 8 --reps 5 --out dram-bw.json
//! Windows x86_64 only; elsewhere the binary exits 2.

#[cfg(not(all(windows, target_arch = "x86_64")))]
fn main() {
    eprintln!("p17_dram_bw: Windows x86_64 only");
    std::process::exit(2);
}

#[cfg(all(windows, target_arch = "x86_64"))]
fn main() {
    imp::main();
}

#[cfg(all(windows, target_arch = "x86_64"))]
mod imp {
    use serde_json::{json, Value};
    use std::collections::BTreeMap;
    use std::ffi::c_void;
    use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
    use std::sync::Barrier;
    use std::time::Instant;

    const NOMINAL_GBS: f64 = 89.6;
    const SPREAD_LIMIT: f64 = 1.15;
    const CHUNK: usize = 4 << 20; // bytes claimed per atomic fetch; multiple of 128
    const MIN_FULL_MIB: usize = 8 << 10; // 8 GiB
    const RAM_MARGIN: u64 = 1 << 30; // robin's 1 GiB margin

    // ---- kernel32, declared by hand (no windows crate in this workspace) -------------------

    #[repr(C)]
    struct GroupAffinity {
        mask: usize,
        group: u16,
        reserved: [u16; 3],
    }

    #[repr(C)]
    struct MemoryStatusEx {
        length: u32,
        memory_load: u32,
        total_phys: u64,
        avail_phys: u64,
        total_page_file: u64,
        avail_page_file: u64,
        total_virtual: u64,
        avail_virtual: u64,
        avail_extended_virtual: u64,
    }

    #[link(name = "kernel32")]
    extern "system" {
        fn VirtualAlloc(addr: *mut c_void, size: usize, alloc_type: u32, protect: u32) -> *mut c_void;
        fn VirtualFree(addr: *mut c_void, size: usize, free_type: u32) -> i32;
        fn GetSystemCpuSetInformation(
            info: *mut u8,
            len: u32,
            returned: *mut u32,
            process: *mut c_void,
            flags: u32,
        ) -> i32;
        fn GetCurrentProcess() -> *mut c_void;
        fn GetCurrentThread() -> *mut c_void;
        fn SetThreadGroupAffinity(
            thread: *mut c_void,
            affinity: *const GroupAffinity,
            previous: *mut GroupAffinity,
        ) -> i32;
        fn GlobalMemoryStatusEx(status: *mut MemoryStatusEx) -> i32;
    }

    // ---- core classes -----------------------------------------------------------------------

    #[derive(Clone, Copy, Debug)]
    struct Cpu {
        group: u16,
        index: u8,
        core: u8,
        class: u8,
    }

    /// One logical processor per physical core (SMT siblings share `core`), from the CPU sets.
    /// Layout of SYSTEM_CPU_SET_INFORMATION (Microsoft Learn): Size u32 @0, Type u32 @4
    /// (0 = CpuSetInformation), Id u32 @8, Group u16 @12, LogicalProcessorIndex u8 @14,
    /// CoreIndex u8 @15, LastLevelCacheIndex u8 @16, NumaNodeIndex u8 @17, EfficiencyClass u8 @18.
    fn detect_cpus() -> Result<(Vec<Cpu>, usize), String> {
        let mut need = 0u32;
        unsafe {
            GetSystemCpuSetInformation(std::ptr::null_mut(), 0, &mut need, GetCurrentProcess(), 0);
        }
        if need == 0 {
            return Err("GetSystemCpuSetInformation reports no CPU sets".into());
        }
        let mut buf = vec![0u64; (need as usize).div_ceil(8)];
        let mut got = 0u32;
        let ok = unsafe {
            GetSystemCpuSetInformation(buf.as_mut_ptr() as *mut u8, need, &mut got, GetCurrentProcess(), 0)
        };
        if ok == 0 {
            return Err("GetSystemCpuSetInformation failed".into());
        }
        let bytes: &[u8] = unsafe { std::slice::from_raw_parts(buf.as_ptr() as *const u8, got as usize) };
        let mut logical = 0usize;
        let mut by_core: BTreeMap<(u16, u8), Cpu> = BTreeMap::new();
        let mut off = 0usize;
        while off + 24 <= bytes.len() {
            let size = u32::from_le_bytes(bytes[off..off + 4].try_into().unwrap()) as usize;
            let ty = u32::from_le_bytes(bytes[off + 4..off + 8].try_into().unwrap());
            if size < 24 || off + size > bytes.len() {
                return Err(format!("malformed CPU set record at offset {off} (size {size})"));
            }
            if ty == 0 {
                let group = u16::from_le_bytes(bytes[off + 12..off + 14].try_into().unwrap());
                let cpu = Cpu {
                    group,
                    index: bytes[off + 14],
                    core: bytes[off + 15],
                    class: bytes[off + 18],
                };
                logical += 1;
                by_core.entry((group, cpu.core)).or_insert(cpu);
            }
            off += size;
        }
        if by_core.is_empty() {
            return Err("no CpuSetInformation records".into());
        }
        Ok((by_core.into_values().collect(), logical))
    }

    struct Config {
        name: String,
        class: Option<u8>,
        cpus: Vec<Cpu>,
    }

    fn build_configs(cpus: &[Cpu]) -> Vec<Config> {
        let mut classes: Vec<u8> = cpus.iter().map(|c| c.class).collect();
        classes.sort_unstable();
        classes.dedup();
        let mut out = vec![Config { name: "all".into(), class: None, cpus: cpus.to_vec() }];
        if classes.len() >= 2 {
            let (lo, hi) = (classes[0], *classes.last().unwrap());
            for &cl in classes.iter().rev() {
                let name = if cl == hi {
                    "p_cores".to_string()
                } else if cl == lo {
                    "e_cores".to_string()
                } else {
                    format!("class_{cl}")
                };
                out.push(Config {
                    name,
                    class: Some(cl),
                    cpus: cpus.iter().copied().filter(|c| c.class == cl).collect(),
                });
            }
        }
        out
    }

    // ---- buffer -----------------------------------------------------------------------------

    struct Buf {
        ptr: *mut u64,
        bytes: usize,
    }
    unsafe impl Send for Buf {}
    unsafe impl Sync for Buf {}
    impl Drop for Buf {
        fn drop(&mut self) {
            unsafe {
                VirtualFree(self.ptr as *mut c_void, 0, 0x8000); // MEM_RELEASE
            }
        }
    }

    fn alloc_filled(bytes: usize) -> Result<Buf, String> {
        let p = unsafe { VirtualAlloc(std::ptr::null_mut(), bytes, 0x3000, 0x04) }; // COMMIT|RESERVE, RW
        if p.is_null() {
            return Err(format!("VirtualAlloc({bytes} B) failed"));
        }
        let buf = Buf { ptr: p as *mut u64, bytes };
        let words = bytes / 8;
        let slice = unsafe { std::slice::from_raw_parts_mut(buf.ptr, words) };
        let threads = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(4);
        let per = words.div_ceil(threads);
        std::thread::scope(|s| {
            for (t, part) in slice.chunks_mut(per).enumerate() {
                s.spawn(move || {
                    let start = (t * per) as u64;
                    for (k, w) in part.iter_mut().enumerate() {
                        *w = start + k as u64;
                    }
                });
            }
        });
        Ok(buf)
    }

    /// Sum of w[i] = i over n words, i.e. n(n-1)/2 mod 2^64, times the number of passes.
    pub(crate) fn expected_sum(words: u64, passes: u64) -> u64 {
        let n = words as u128;
        ((n * (n - 1) / 2) as u64).wrapping_mul(passes)
    }

    // ---- kernel -----------------------------------------------------------------------------

    /// Wrapping u64-lane sum of `bytes` bytes at `p` (32-byte aligned, `bytes` a multiple of 128).
    #[target_feature(enable = "avx2")]
    pub(crate) unsafe fn sum_block(p: *const u8, bytes: usize) -> u64 {
        use std::arch::x86_64::*;
        let mut a0 = _mm256_setzero_si256();
        let mut a1 = _mm256_setzero_si256();
        let mut a2 = _mm256_setzero_si256();
        let mut a3 = _mm256_setzero_si256();
        let mut q = p as *const __m256i;
        let end = q.add(bytes / 32);
        while q < end {
            a0 = _mm256_add_epi64(a0, _mm256_load_si256(q));
            a1 = _mm256_add_epi64(a1, _mm256_load_si256(q.add(1)));
            a2 = _mm256_add_epi64(a2, _mm256_load_si256(q.add(2)));
            a3 = _mm256_add_epi64(a3, _mm256_load_si256(q.add(3)));
            q = q.add(4);
        }
        let s = _mm256_add_epi64(_mm256_add_epi64(a0, a1), _mm256_add_epi64(a2, a3));
        let mut lanes = [0u64; 4];
        _mm256_storeu_si256(lanes.as_mut_ptr() as *mut __m256i, s);
        lanes.iter().fold(0u64, |x, y| x.wrapping_add(*y))
    }

    fn pin(cpu: &Cpu) -> Result<(), String> {
        let ga = GroupAffinity { mask: 1usize << cpu.index, group: cpu.group, reserved: [0; 3] };
        let ok = unsafe { SetThreadGroupAffinity(GetCurrentThread(), &ga, std::ptr::null_mut()) };
        if ok == 0 {
            Err(format!("SetThreadGroupAffinity(group {}, cpu {}) failed", cpu.group, cpu.index))
        } else {
            Ok(())
        }
    }

    struct RunResult {
        gbs: f64,
        checksum_ok: bool,
    }

    fn run_once(buf: &Buf, cpus: &[Cpu], passes: usize) -> Result<RunResult, String> {
        let chunks_per_pass = buf.bytes / CHUNK;
        let total_items = chunks_per_pass * passes;
        let next = AtomicUsize::new(0);
        let sum = AtomicU64::new(0);
        let barrier = Barrier::new(cpus.len());
        let base = buf.ptr as usize;
        let spans: Vec<Result<(Instant, Instant), String>> = std::thread::scope(|s| {
            let handles: Vec<_> = cpus
                .iter()
                .map(|cpu| {
                    let (next, sum, barrier) = (&next, &sum, &barrier);
                    s.spawn(move || {
                        let pinned = pin(cpu);
                        barrier.wait();
                        pinned?;
                        let t0 = Instant::now();
                        let mut local = 0u64;
                        loop {
                            let i = next.fetch_add(1, Ordering::Relaxed);
                            if i >= total_items {
                                break;
                            }
                            let off = (i % chunks_per_pass) * CHUNK;
                            local = local.wrapping_add(unsafe { sum_block((base + off) as *const u8, CHUNK) });
                        }
                        let t1 = Instant::now();
                        sum.fetch_add(local, Ordering::Relaxed); // wrapping on atomics
                        Ok((t0, t1))
                    })
                })
                .collect();
            handles.into_iter().map(|h| h.join().expect("worker panicked")).collect()
        });
        let mut t_start: Option<Instant> = None;
        let mut t_end: Option<Instant> = None;
        for r in spans {
            let (a, b) = r?;
            t_start = Some(t_start.map_or(a, |x| x.min(a)));
            t_end = Some(t_end.map_or(b, |x| x.max(b)));
        }
        let wall = t_end.unwrap().duration_since(t_start.unwrap()).as_secs_f64();
        let bytes = (total_items * CHUNK) as f64;
        let ok = sum.load(Ordering::Relaxed) == expected_sum((buf.bytes / 8) as u64, passes as u64);
        Ok(RunResult { gbs: bytes / wall / 1e9, checksum_ok: ok })
    }

    // ---- statistics -------------------------------------------------------------------------

    pub(crate) fn median(v: &[f64]) -> f64 {
        let mut s = v.to_vec();
        s.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let n = s.len();
        if n % 2 == 1 {
            s[n / 2]
        } else {
            (s[n / 2 - 1] + s[n / 2]) / 2.0
        }
    }

    pub(crate) fn verdict(rates: &[f64], checksum_ok: bool, nominal: f64) -> &'static str {
        let min = rates.iter().copied().fold(f64::INFINITY, f64::min);
        let max = rates.iter().copied().fold(0.0, f64::max);
        if !checksum_ok {
            "invalid_checksum"
        } else if max > nominal {
            "invalid_above_nominal"
        } else if max / min > SPREAD_LIMIT {
            "void_spread"
        } else {
            "valid"
        }
    }

    // ---- main -------------------------------------------------------------------------------

    #[derive(Debug)]
    struct Args {
        mib: usize,
        reps: usize,
        passes: usize,
        nominal: f64,
        out: Option<String>,
        selftest: bool,
    }

    fn refuse(msg: &str) -> ! {
        eprintln!("p17_dram_bw: {msg}");
        std::process::exit(2);
    }

    fn parse_args() -> Args {
        let raw: Vec<String> = std::env::args().skip(1).collect();
        parse_args_from(&raw).unwrap_or_else(|e| refuse(&e))
    }

    /// Pure argument parsing and validation; `Err` carries the refusal message (exit 2).
    fn parse_args_from(raw: &[String]) -> Result<Args, String> {
        let selftest = raw.iter().any(|a| a == "--selftest");
        let mut a = Args {
            mib: if selftest { 256 } else { MIN_FULL_MIB },
            reps: if selftest { 3 } else { 5 },
            passes: if selftest { 1 } else { 8 },
            nominal: NOMINAL_GBS,
            out: None,
            selftest,
        };
        let mut it = raw.iter();
        while let Some(k) = it.next() {
            let mut val = || it.next().cloned().ok_or_else(|| format!("{k} needs a value"));
            match k.as_str() {
                "--selftest" => {}
                "--gib" => a.mib = val()?.parse::<usize>().map_err(|_| "--gib: integer".to_string())? << 10,
                "--mib" => a.mib = val()?.parse().map_err(|_| "--mib: integer".to_string())?,
                "--reps" => a.reps = val()?.parse().map_err(|_| "--reps: integer".to_string())?,
                "--passes" => a.passes = val()?.parse().map_err(|_| "--passes: integer".to_string())?,
                "--nominal-gbs" => {
                    a.nominal = val()?.parse().map_err(|_| "--nominal-gbs: number".to_string())?
                }
                "--out" => a.out = Some(val()?),
                other => return Err(format!("unknown argument {other}")),
            }
        }
        if a.mib == 0 || a.mib % 4 != 0 {
            return Err("buffer must be a non-zero multiple of 4 MiB".into());
        }
        if a.passes == 0 || a.reps == 0 {
            return Err("--passes and --reps must be >= 1".into());
        }
        if !a.selftest {
            if a.mib < MIN_FULL_MIB {
                return Err("buffer below 8 GiB would measure cache; use --selftest for a small buffer".into());
            }
            if a.reps < 3 {
                return Err("--reps must be >= 3 (median and spread need three runs)".into());
            }
        }
        Ok(a)
    }

    /// Pure RAM guard: the buffer plus the 1 GiB margin must fit in available physical RAM.
    fn check_ram(avail_phys: u64, buffer_bytes: u64) -> Result<(), String> {
        if avail_phys < buffer_bytes + RAM_MARGIN {
            Err(format!(
                "available physical RAM {} MiB is below buffer {} MiB + 1 GiB margin",
                avail_phys >> 20,
                buffer_bytes >> 20
            ))
        } else {
            Ok(())
        }
    }

    pub fn main() {
        let args = parse_args();
        if !is_x86_feature_detected!("avx2") {
            refuse("this CPU has no AVX2");
        }
        let bytes = args.mib << 20;

        let mut ms = MemoryStatusEx {
            length: std::mem::size_of::<MemoryStatusEx>() as u32,
            memory_load: 0,
            total_phys: 0,
            avail_phys: 0,
            total_page_file: 0,
            avail_page_file: 0,
            total_virtual: 0,
            avail_virtual: 0,
            avail_extended_virtual: 0,
        };
        if unsafe { GlobalMemoryStatusEx(&mut ms) } == 0 {
            refuse("GlobalMemoryStatusEx failed");
        }
        check_ram(ms.avail_phys, bytes as u64).unwrap_or_else(|e| refuse(&e));

        let (cpus, logical) = detect_cpus().unwrap_or_else(|e| refuse(&e));
        let configs = build_configs(&cpus);
        eprintln!(
            "p17_dram_bw: {} cores ({} logical), configs: {}, buffer {} MiB, {} passes, {} reps{}",
            cpus.len(),
            logical,
            configs.iter().map(|c| format!("{}={}", c.name, c.cpus.len())).collect::<Vec<_>>().join(" "),
            args.mib,
            args.passes,
            args.reps,
            if args.selftest { " [selftest]" } else { "" }
        );
        let buf = alloc_filled(bytes).unwrap_or_else(|e| refuse(&e));

        let mut rates: Vec<Vec<f64>> = vec![Vec::new(); configs.len()];
        let mut checks: Vec<bool> = vec![true; configs.len()];
        // unmeasured warm-up, one pass per configuration (still checksum-verified)
        for (ci, cfg) in configs.iter().enumerate() {
            let r = run_once(&buf, &cfg.cpus, 1).unwrap_or_else(|e| refuse(&e));
            checks[ci] &= r.checksum_ok;
        }
        for rep in 0..args.reps {
            for (ci, cfg) in configs.iter().enumerate() {
                let r = run_once(&buf, &cfg.cpus, args.passes).unwrap_or_else(|e| refuse(&e));
                eprintln!(
                    "  rep {} {:8} {:7.2} GB/s checksum {}",
                    rep + 1,
                    cfg.name,
                    r.gbs,
                    if r.checksum_ok { "ok" } else { "MISMATCH" }
                );
                checks[ci] &= r.checksum_ok;
                rates[ci].push(r.gbs);
            }
        }

        let mut cfg_json: Vec<Value> = Vec::new();
        let mut all_good = true;
        for (ci, cfg) in configs.iter().enumerate() {
            let r = &rates[ci];
            let min = r.iter().copied().fold(f64::INFINITY, f64::min);
            let max = r.iter().copied().fold(0.0, f64::max);
            let v = if args.selftest {
                if checks[ci] { "selftest_checksum_ok" } else { "invalid_checksum" }
            } else {
                verdict(r, checks[ci], args.nominal)
            };
            all_good &= if args.selftest { checks[ci] } else { v == "valid" };
            cfg_json.push(json!({
                "name": cfg.name,
                "efficiency_class": cfg.class,
                "threads": cfg.cpus.len(),
                "rates_gbs": r,
                "median_gbs": median(r),
                "min_gbs": min,
                "max_gbs": max,
                "spread": max / min,
                "checksum_ok": checks[ci],
                "verdict": v,
            }));
        }
        let mut class_counts: BTreeMap<u8, usize> = BTreeMap::new();
        for c in &cpus {
            *class_counts.entry(c.class).or_default() += 1;
        }
        let report = json!({
            "tool": "p17_dram_bw",
            "ticket": "nibor1896/crow-nest#170",
            "unix_time": std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH).map(|d| d.as_secs()).unwrap_or(0),
            "selftest": args.selftest,
            "machine": {
                "cores": cpus.len(),
                "logical_processors": logical,
                "classes": class_counts.iter()
                    .map(|(k, n)| json!({"efficiency_class": k, "cores": n})).collect::<Vec<_>>(),
                "available_phys_mib_before": ms.avail_phys >> 20,
            },
            "buffer_bytes": bytes,
            "chunk_bytes": CHUNK,
            "passes": args.passes,
            "reps": args.reps,
            "unit": "GB/s = 1e9 B/s",
            "nominal_gbs": args.nominal,
            "spread_limit": SPREAD_LIMIT,
            "configs": cfg_json,
            "verdict": if all_good { "valid" } else { "void_or_invalid" },
        });
        let text = serde_json::to_string_pretty(&report).unwrap();
        println!("{text}");
        if let Some(path) = &args.out {
            std::fs::write(path, &text).unwrap_or_else(|e| refuse(&format!("cannot write {path}: {e}")));
        }
        if args.selftest {
            eprintln!("selftest: {}", if all_good { "PASS" } else { "FAIL" });
        }
        std::process::exit(if all_good { 0 } else { 1 });
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        #[test]
        fn closed_form_matches_naive_sum() {
            for n in [1u64, 16, 1024, 4096] {
                let naive: u64 = (0..n).fold(0u64, |a, i| a.wrapping_add(i));
                assert_eq!(expected_sum(n, 1), naive);
                assert_eq!(expected_sum(n, 3), naive.wrapping_mul(3));
            }
        }

        #[test]
        fn kernel_sums_what_it_reads() {
            // 32-byte aligned: Vec<__m256i>-like via u64 vec with padding
            let mut v = vec![0u64; 1024 + 4];
            let off = ((32 - (v.as_ptr() as usize % 32)) % 32) / 8;
            for (i, w) in v[off..off + 1024].iter_mut().enumerate() {
                *w = i as u64;
            }
            let got = unsafe { sum_block(v[off..].as_ptr() as *const u8, 1024 * 8) };
            assert_eq!(got, expected_sum(1024, 1));
        }

        #[test]
        fn median_and_verdicts() {
            assert_eq!(median(&[3.0, 1.0, 2.0]), 2.0);
            assert_eq!(median(&[1.0, 2.0, 3.0, 4.0]), 2.5);
            assert_eq!(verdict(&[60.0, 61.0, 62.0], true, 89.6), "valid");
            assert_eq!(verdict(&[60.0, 70.0, 62.0], true, 89.6), "void_spread");
            assert_eq!(verdict(&[60.0, 90.0, 61.0], true, 89.6), "invalid_above_nominal");
            assert_eq!(verdict(&[60.0, 61.0, 62.0], false, 89.6), "invalid_checksum");
        }

        fn args(v: &[&str]) -> Result<Args, String> {
            parse_args_from(&v.iter().map(|s| s.to_string()).collect::<Vec<_>>())
        }

        #[test]
        fn refuses_buffer_below_8_gib_outside_selftest() {
            assert!(args(&["--gib", "1"]).unwrap_err().contains("8 GiB"));
            assert!(args(&["--mib", "8188"]).unwrap_err().contains("8 GiB")); // 8 GiB - 4 MiB
            assert_eq!(args(&["--gib", "8"]).unwrap().mib, 8192);
            assert_eq!(args(&[]).unwrap().mib, 8192); // default is the full size
            assert_eq!(args(&["--selftest"]).unwrap().mib, 256); // selftest may be small
        }

        #[test]
        fn refuses_fewer_than_3_reps_outside_selftest() {
            assert!(args(&["--gib", "8", "--reps", "2"]).unwrap_err().contains("--reps must be >= 3"));
            assert!(args(&["--gib", "8", "--reps", "1"]).is_err());
            assert_eq!(args(&["--gib", "8", "--reps", "3"]).unwrap().reps, 3);
            assert_eq!(args(&["--selftest", "--reps", "1"]).unwrap().reps, 1);
        }

        #[test]
        fn refuses_malformed_arguments() {
            assert!(args(&["--mib", "10"]).is_err()); // not a multiple of 4 MiB
            assert!(args(&["--mib", "0"]).is_err());
            assert!(args(&["--gib", "8", "--passes", "0"]).is_err());
            assert!(args(&["--gib"]).unwrap_err().contains("needs a value"));
            assert!(args(&["--bogus"]).unwrap_err().contains("unknown argument"));
        }

        #[test]
        fn ram_margin_is_one_gib_on_top_of_the_buffer() {
            let buf = 8u64 << 30;
            assert!(check_ram(buf + (1 << 30) - 1, buf).unwrap_err().contains("below buffer"));
            assert!(check_ram(buf, buf).is_err());
            assert!(check_ram(buf + (1 << 30), buf).is_ok());
        }

        #[test]
        fn run_above_nominal_is_invalid_even_when_stable() {
            // all runs within spread, one just above 89.6 -> invalid, not valid
            assert_eq!(verdict(&[89.7, 89.7, 89.7], true, 89.6), "invalid_above_nominal");
            assert_eq!(verdict(&[89.6, 89.6, 89.6], true, 89.6), "valid"); // at the ceiling is allowed
            // above-nominal wins over a void spread
            assert_eq!(verdict(&[40.0, 95.0, 41.0], true, 89.6), "invalid_above_nominal");
        }

        #[test]
        fn spread_above_1_15_is_void() {
            assert_eq!(verdict(&[60.0, 69.0, 61.0], true, 89.6), "valid"); // 69/60 = 1.15 exactly
            assert_eq!(verdict(&[60.0, 69.1, 61.0], true, 89.6), "void_spread");
        }
    }
}
