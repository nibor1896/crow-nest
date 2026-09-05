//! isolated Ple::load hang repro (scratch)
fn main() {
    let t0 = std::time::Instant::now();
    let step = |m: &str| eprintln!("[+{:.2}s] {m}", t0.elapsed().as_secs_f64());
    let mut cnq = crow_nest_engine::cnq::Cnq::open("../converter/Qwen3.8-Flash-Next-CNQ4.5.cnq");
    step("container open");
    unsafe {
        let _ctx = crow_nest_engine::cuda::Ctx::init();
        step("ctx");
        let ple = crow_nest_engine::gen::Ple::load(&mut cnq, 1 << 30);
        step("ple loaded");
        let _ = ple;
    }
}
