//! isolated Ple::load hang repro (scratch)
use crow_nest_engine::geo::{from_engine_dir, DEFAULT_CNQ};

fn main() {
    let t0 = std::time::Instant::now();
    let step = |m: &str| eprintln!("[+{:.2}s] {m}", t0.elapsed().as_secs_f64());
    // #60 (2026-09-18): the repro defaults to the production -M container, like
    // `decode` and `parity` (#51) and the two generator bins (#52), and reads
    // CROW_CNQ like they do. It used to hard-code the pre-#51 container, which
    // on this machine is a file the tree no longer ships.
    let cnq_path = std::env::var("CROW_CNQ").unwrap_or_else(|_| from_engine_dir(DEFAULT_CNQ));
    let mut cnq = crow_nest_engine::cnq::Cnq::open(&cnq_path);
    step("container open");
    unsafe {
        let _ctx = crow_nest_engine::cuda::Ctx::init();
        step("ctx");
        let ple = crow_nest_engine::gen::Ple::load(&mut cnq, 1 << 30);
        step("ple loaded");
        let _ = ple;
    }
}
