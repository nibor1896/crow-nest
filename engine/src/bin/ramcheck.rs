//! #103, the #82 follow-up (2026-09-23): the engine's own host-RAM view for scripts, no CUDA.
//!
//!   ramcheck                 print one line: free_for_pin, MemAvailable, the NVIDIA
//!                            driver pages (live + reclaimable pool), FOLL_PIN pins
//!   ramcheck --need <GiB>    the same, exit 0 when free_for_pin >= <GiB>, else exit 1
//!                            and name the biggest resident processes
//!
//! Measurement scripts gate a boot on THIS instead of `MemAvailable`: the NVIDIA
//! driver's sysmem page pool (freed pages of driver allocations, e.g. the old
//! write-combined cold tier) is invisible to `MemAvailable` but reclaimable, so a
//! `MemAvailable` gate says "reboot" where the engine boots fine
//! (`cuda::free_physical_ram_parts` has the derivation). A refusal here is a real
//! shortage - a live process holds the RAM - never a case for a reboot.
use crow_nest_engine::cuda;
use crow_nest_engine::geo::GIB;

fn foll_pin_outstanding() -> i64 {
    let t = std::fs::read_to_string("/proc/vmstat").unwrap_or_default();
    let g = |k: &str| t.lines().find_map(|l| l.strip_prefix(k).and_then(|v| v.trim().parse::<i64>().ok())).unwrap_or(0);
    g("nr_foll_pin_acquired ") - g("nr_foll_pin_released ")
}

/// the `n` largest resident processes (VmRSS), readable ones only
fn top_rss(n: usize) -> Vec<(u64, u32, String)> {
    let mut v = Vec::new();
    let Ok(rd) = std::fs::read_dir("/proc") else { return v };
    for e in rd.flatten() {
        let Some(pid) = e.file_name().to_str().and_then(|s| s.parse::<u32>().ok()) else { continue };
        let Ok(st) = std::fs::read_to_string(e.path().join("status")) else { continue };
        let rss = st
            .lines()
            .find_map(|l| l.strip_prefix("VmRSS:"))
            .and_then(|r| r.split_whitespace().next().and_then(|k| k.parse::<u64>().ok()))
            .unwrap_or(0)
            * 1024;
        let name = st.lines().find_map(|l| l.strip_prefix("Name:")).unwrap_or("").trim().to_string();
        v.push((rss, pid, name));
    }
    v.sort_by(|a, b| b.0.cmp(&a.0));
    v.truncate(n);
    v
}

fn main() {
    let a: Vec<String> = std::env::args().skip(1).collect();
    let need: Option<f64> = a.iter().position(|s| s == "--need").and_then(|i| a.get(i + 1)).and_then(|v| v.parse().ok());
    let r = cuda::free_physical_ram_parts();
    let g = |b: u64| b as f64 / GIB;
    let pool = r.driver_held.saturating_sub(r.driver_live);
    println!(
        "ramcheck: free_for_pin {:.2} GiB, MemAvailable {:.2} GiB, NVIDIA driver pages {:.2} GiB (mapped by live processes {:.2}, pool + unmapped rest {:.2}), FOLL_PIN outstanding {} pages{}",
        g(r.free_for_pin),
        g(r.mem_available),
        g(r.driver_held),
        g(r.driver_live),
        g(pool),
        foll_pin_outstanding(),
        if r.other_cuda { ", a CUDA process is alive" } else { "" }
    );
    if let Some(n) = need {
        if g(r.free_for_pin) >= n {
            println!("ramcheck: OK, need {n:.1} GiB");
        } else {
            let total = g(cuda::meminfo().get("MemTotal"));
            if n > total {
                println!("ramcheck: SHORT - need {n:.1} GiB is more than the {total:.2} GiB of physical RAM");
                std::process::exit(1);
            }
            println!("ramcheck: SHORT by {:.2} GiB (need {n:.1}) - live processes hold the RAM; a reboot is not the fix:", n - g(r.free_for_pin));
            for (rss, pid, name) in top_rss(6) {
                println!("ramcheck:   {:>7} {name:<24} VmRSS {:.2} GiB", pid, g(rss));
            }
            std::process::exit(1);
        }
    }
}
