//! Measurement B-0 (2026-09-04): CUDA-graph node cost on THIS machine (WDDM,
//! Blackwell). No published number exists. Captures a graph of N trivial
//! kernels, replays it R times, reports us per node; also the eager launch
//! cost for comparison. No model, no large allocations.
use crow_nest_engine::cuda;
use crow_nest_engine::gen::launch_v;

const SRC: &str = r#"
extern "C" __global__ void nop_k(float* p) {
    if (threadIdx.x == 0 && blockIdx.x == 0) p[0] += 1.0f;
}
// a "typical small" engine kernel shape: 4 blocks x 256 threads touching 10 KB
extern "C" __global__ void small_k(float* p) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    p[i] = p[i] * 0.999f + 0.001f;
}
"#;

fn main() {
    unsafe {
        let _ctx = cuda::Ctx::init();
        let module = cuda::compile(SRC);
        let nop = module.get("nop_k");
        let small = module.get("small_k");
        let buf = cuda::alloc_zeroed(4096 * 4);
        let stream = cuda::stream_create_non_blocking();
        cuda::set_stream(stream as u64);
        let reps = 200u32;
        println!("graph node cost probe (WDDM): {reps} replays per point");
        for &(name, f, gx, bx) in &[("nop 1x32", nop, 1u32, 32u32), ("small 4x256", small, 4, 256)] {
            for &n in &[1usize, 64, 256, 1024, 2048] {
                // eager
                let t0 = std::time::Instant::now();
                for _ in 0..n {
                    launch_v(f, gx, 1, 1, bx, &[buf as u64]);
                }
                cuda::sync();
                let eager_us = t0.elapsed().as_secs_f64() * 1e6 / n as f64;
                // captured graph
                cuda::begin_capture(stream);
                for _ in 0..n {
                    launch_v(f, gx, 1, 1, bx, &[buf as u64]);
                }
                let exec = cuda::end_capture_instantiate(stream);
                for _ in 0..10 {
                    cuda::launch_graph(exec, stream);
                }
                cuda::sync();
                let t1 = std::time::Instant::now();
                for _ in 0..reps {
                    cuda::launch_graph(exec, stream);
                }
                cuda::sync();
                let per_node_us = t1.elapsed().as_secs_f64() * 1e6 / (reps as f64 * n as f64);
                let per_graph_us = t1.elapsed().as_secs_f64() * 1e6 / reps as f64;
                println!(
                    "[graph-probe] {name:12} N={n:5}: eager {eager_us:7.2} us/launch | graph {per_node_us:6.2} us/node ({per_graph_us:8.1} us/replay)"
                );
            }
        }
    }
}
