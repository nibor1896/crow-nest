//! Weight bytes: the container tensor -> device buffer loaders and the NVFP4
//! weight pair they produce. No launch policy lives here (that is `gen`), so
//! `vit` and `gen` can both load without depending on each other.

use crate::cnq::{self, Cnq};
use crate::cuda::{self, CUdeviceptr as Dev};

#[derive(Clone, Copy)]
pub struct Fp4 {
    pub w: Dev,
    pub gs: Dev,
}

/// bf16 device copy of a BF16-keep tensor (exact: the keep is bf16)
pub unsafe fn load_bf16_twin(cnq: &mut Cnq, name: &str, sec: &str) -> Dev {
    let v = cnq.read_f32(name, sec);
    let b: Vec<u16> = v.iter().map(|x| (x.to_bits() >> 16) as u16).collect();
    cuda::upload_dev(std::slice::from_raw_parts(b.as_ptr() as *const u8, b.len() * 2))
}

pub unsafe fn load_f32(cnq: &mut Cnq, name: &str, sec: &str) -> Dev {
    let v = cnq.read_f32(name, sec);
    cuda::to_f32_dev(&v)
}

pub unsafe fn load_fp4(cnq: &mut Cnq, name: &str, sec: &str) -> Fp4 {
    let t = cnq.find(name, sec).clone();
    assert!(!matches!(t.dtype.as_str(), "bf16" | "f32"), "{name}: expected NVFP4, got {}", t.dtype);
    let raw = cnq.read_bytes(&t);
    let w = cuda::upload_dev(&raw);
    let gs = cuda::to_f32_dev(&[t.global_scale]);
    Fp4 { w, gs }
}

/// NVFP4 -> host f32: the 36-byte-block walk, block order preserved.
/// `n` is the value count; the tail beyond `raw.len()/36*64` stays zero.
fn dequant_fp4_host(raw: &[u8], gs: f32, n: usize) -> Vec<f32> {
    let mut out = vec![0f32; n];
    let mut blk = [0f32; 64];
    for (b, chunk) in raw.chunks_exact(36).enumerate() {
        cnq::dequant_block(chunk, gs, &mut blk);
        out[b * 64..(b + 1) * 64].copy_from_slice(&blk);
    }
    out
}

/// f32 device copy of a weight the kernels read as f32 whatever the file holds (the two
/// conv1d kinds). #77: a `bf16` tensor - a dense overlay shadowing the base NVFP4 one - is
/// widened instead of dequantized, which is the WHOLE change those two kinds need: the kernel
/// that consumes them (`conv_silu` / `ple_conv`) always read f32 and never saw the quant.
pub unsafe fn dequant_fp4_dev(cnq: &mut Cnq, name: &str, sec: &str, n: usize) -> Dev {
    let t = cnq.find(name, sec).clone();
    let raw = cnq.read_bytes(&t);
    let v = host_f32(name, &t, &raw, n);
    assert_eq!(v.len(), n, "{name}: {} values, expected {n}", v.len());
    cuda::to_f32_dev(&v)
}

/// tiny tensors (A_log, dt_bias): any dtype → host f32 → device
pub unsafe fn load_small_f32(cnq: &mut Cnq, name: &str, sec: &str, n: usize) -> Dev {
    let t = cnq.find(name, sec).clone();
    let raw = cnq.read_bytes(&t);
    let v = host_f32(name, &t, &raw, n);
    assert_eq!(v.len(), n, "{name}: size mismatch");
    cuda::to_f32_dev(&v)
}

/// The host f32 values of a tensor's container bytes, by its dtype: `bf16` widened, `f32`
/// as stored (Crow #300 C6/C7: the dense recipe's `A_log`), `nvfp4` dequantized. Without the
/// `f32` arm an f32 tensor fell into the NVFP4 walk, which reads 36-byte blocks: 8 bytes of
/// `A_log` are no block at all, so it came back as zeros and the size assert passed.
fn host_f32(name: &str, t: &cnq::TensorInfo, raw: &[u8], n: usize) -> Vec<f32> {
    match t.dtype.as_str() {
        "bf16" => cnq::bf16_bytes_to_f32(raw),
        "f32" => cnq::f32_bytes_to_f32(raw),
        "i64" => panic!("{name}: i64 unexpected here"),
        _ => dequant_fp4_host(raw, t.global_scale, n),
    }
}

#[cfg(test)]
mod tests_300_c7 {
    use super::host_f32;
    use crate::cnq::Cnq;

    /// The f32 dtype reaches the loaders as f32 (`load_small_f32` / `dequant_fp4_dev` share
    /// `host_f32`): on the synthetic v2 fixture's `A_log` it gives exactly what `read_f32`
    /// gives, which is not all zeros (the NVFP4 walk's answer for 8 bytes).
    #[test]
    fn an_f32_tensor_loads_as_f32_not_as_an_nvfp4_dequant() {
        let path = format!("{}/tests/fixtures/synthetic-v2/synthetic-v2.cnq", env!("CARGO_MANIFEST_DIR"));
        let mut cnq = Cnq::open_checked(&path).unwrap();
        let name = "model.language_model.layers.0.linear_attn.A_log";
        let t = cnq.find(name, "text").clone();
        assert_eq!(t.dtype, "f32");
        let raw = cnq.read_bytes(&t);
        let want = cnq.read_f32(name, "text");
        assert!(want.iter().any(|x| *x != 0.0), "{want:?}");
        assert_eq!(host_f32(name, &t, &raw, 2), want);
    }
}
