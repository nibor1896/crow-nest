//! Probe 1 (Crow #188, next step a): cudarc context on Windows with CUDA 13.x.
//!
//! Answers: does `CudaContext::new(0)` come up on this machine (RTX 5090, sm_120,
//! driver 616.56, CUDA 13.3 toolkit, cudarc 0.19.9 dynamic-loading), does the
//! host<->device path work, and does the full NVRTC -> module -> launch chain work?
//! This is the driver/toolkit layer the engine would sit on.

use cudarc::driver::result::DriverError;
use cudarc::driver::safe::{CudaContext, LaunchConfig, PushKernelArg};
use cudarc::driver::sys::CUdevice_attribute;
use cudarc::nvrtc::{compile_ptx_with_opts, CompileOptions};

const ADD_SRC: &str = r#"
extern "C" __global__ void vec_add(const float* a, const float* b, float* c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) c[i] = a[i] + b[i];
}
"#;

fn main() -> Result<(), DriverError> {
    println!("p1: CudaContext::new(0) on dynamic-loading cudarc 0.19.9, feature cuda-13030 ...");
    let ctx = CudaContext::new(0)?;

    let cc_major = ctx.attribute(CUdevice_attribute::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR)?;
    let cc_minor = ctx.attribute(CUdevice_attribute::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR)?;
    let sm_count = ctx.attribute(CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT)?;
    println!("p1: context OK, compute capability {cc_major}.{cc_minor}, {sm_count} SMs");

    let stream = ctx.default_stream();

    let host: Vec<f32> = (0..1024).map(|i| i as f32).collect();
    let dev = stream.clone_htod(&host)?;
    let back: Vec<f32> = stream.clone_dtoh(&dev)?;
    assert_eq!(host, back, "htod/dtoh round trip mismatch");
    println!("p1: host<->device round trip OK (1024 x f32)");

    let opts = CompileOptions {
        arch: Some("compute_120"),
        ..Default::default()
    };
    let ptx = compile_ptx_with_opts(ADD_SRC, opts).expect("nvrtc compile failed");
    println!("p1: nvrtc compile OK ({} bytes PTX)", ptx.to_src().len());
    let module = ctx.load_module(ptx)?;
    let func = module.load_function("vec_add")?;

    let b_dev = stream.clone_htod(&host)?;
    let mut c_dev = stream.alloc_zeros::<f32>(1024)?;
    unsafe {
        stream
            .launch_builder(&func)
            .arg(&dev)
            .arg(&b_dev)
            .arg(&mut c_dev)
            .arg(&(1024_i32))
            .launch(LaunchConfig::for_num_elems(1024))
    }?;

    let c_host: Vec<f32> = stream.clone_dtoh(&c_dev)?;
    let ok = c_host.iter().zip(host.iter()).all(|(c, a)| *c == 2.0 * a);
    println!("p1: NVRTC kernel launch + result: {}", if ok { "OK" } else { "WRONG" });
    if ok {
        println!("p1: PASS");
        Ok(())
    } else {
        std::process::exit(1);
    }
}
