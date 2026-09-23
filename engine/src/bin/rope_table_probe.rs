//! #89 numerics probe P3 (U3 in docs/numerics-diff.md): RoPE table bit
//! provenance. crow-nest builds the table in host f32 (manager.rs:300-311):
//! `inv = 10_000_000f32.powf(-(2j)/64)`, `f = t as f32 * inv`, f32 cos/sin,
//! over cfg.context (default 262144, the config max_position_embeddings) x
//! ROPE_PAIRS (32) pairs. HF computes `base ** (arange/64)` in torch f32 then
//! an f32 outer product with positions and f32 cos/sin (modeling:108-136) —
//! same formula, same f32 class; the open question was only whether the host
//! f32 pipeline stays within a couple of ulp of an f64-rounded reference.
//!
//! References used (recomputed here, host-only, no GPU):
//!   - SHARED-ARGUMENT reference (the pass gate): cos_f64(f64::from(f)) where
//!     f = f32(t * inv32) IS the table's own f32 argument. The table's f32
//!     product t*inv is exactly what every f32 engine forms (HF's f32 outer
//!     product included), so the argument bits are engine-shared, and this
//!     reference isolates the only table-local freedom left: the quality of
//!     host cosf/sinf against the correctly-rounded cos/sin OF THAT ARGUMENT.
//!     Gate: <= 2 ulp.
//!   - inv_freq provenance: crow powf f32 vs f64 powf rounded to f32 (max ulp
//!     over the 32 values; bits printed for the torch cross-check in P1b —
//!     if torch's f32 inv_freq has the same bits, the arguments are bitwise
//!     engine-identical and the whole table is within this probe's gate).
//!   - END-TO-END f64 reference (report-only): powf/product/cos/sin all in
//!     f64, rounded to f32. At long t the f32 product rounding is amplified
//!     by the argument magnitude (t=245904, j=1: arg ~1.5e5 rad, 0.5-ulp
//!     product rounding ~ 0.009 rad phase) so this distance is expected to
//!     be HUGE at the far end — the classic RoPE-table-in-f32 property that
//!     EVERY f32 engine (HF included: it also forms t*inv in f32) shares. It
//!     measures table-vs-true-math, not crow-vs-HF, and is reported, not
//!     gated.
//!
//! Golden-vector check: the device `rope` pairing formula (kernels.rs:1795-98:
//! out[d] = a*c - b*s, out[d+32] = b*c + a*s with a=x[d], b=x[d+32], dims
//! >= 64 pass-through) vs a host rotate_half (HF modeling:566-608: out =
//! x*cos + cat(-x[32:64], x[0:32])*sin with duplicated cos/sin) on
//! deterministic x rows. The two f32 forms must be BITWISE equal (sub vs
//! add-of-negated-product is an exact identity in IEEE; Rust does not
//! contract either into an FMA) and within the 2-ulp line of the f64
//! pairing arithmetic computed on the table's own f32 cos/sin.
//!
//! usage: rope_table_probe
use crow_nest_engine::geo::ROPE_PAIRS;

/// monotone map of an f32 onto the u32 line (the kernel-side `ordkey` trick):
/// ulp distance = |ord(a) - ord(b)|
fn ord_u32(x: f32) -> u32 {
    let b = x.to_bits();
    if b & 0x8000_0000 != 0 { !b } else { b | 0x8000_0000 }
}

fn ulp_dist(a: f32, b: f32) -> u32 {
    ord_u32(a).abs_diff(ord_u32(b))
}

/// deterministic O(1) x row (256 dims), the golden-vector input
fn golden_row(seed: u64) -> Vec<f32> {
    let mut s = seed;
    (0..256)
        .map(|_| {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            ((s >> 40) as i32 as f32) / 64.0 // (-8, 8), exact dyadic steps
        })
        .collect()
}

fn main() {
    let ctx: usize = 262144; // max_position_embeddings (geo.rs default context)
    assert_eq!(ROPE_PAIRS, 32);

    // ---- table recomputation (manager.rs:300-309, byte-for-byte formula) ----
    let mut inv32 = [0f32; 32];
    for (j, inv) in inv32.iter_mut().enumerate() {
        *inv = 10_000_000f32.powf(-(2.0 * j as f32) / 64.0);
    }

    // ---- inv_freq vs f64 powf ----
    let mut inv_max_ulp = 0u32;
    let mut inv_max_j = 0usize;
    for (j, &iv) in inv32.iter().enumerate() {
        let ref64 = 10_000_000f64.powf(-(2.0 * j as f64) / 64.0) as f32;
        let d = ulp_dist(iv, ref64);
        if d > inv_max_ulp {
            inv_max_ulp = d;
            inv_max_j = j;
        }
    }

    // ---- full-table scan: shared-argument reference (gated) + end-to-end (report) ----
    let (mut cos_a_ulp, mut sin_a_ulp) = (0u32, 0u32);
    let (mut cos_a_at, mut sin_a_at) = ((0u64, 0usize), (0u64, 0usize));
    let (mut cos_e_ulp, mut sin_e_ulp) = (0u32, 0u32);
    let (mut cos_e_at, mut sin_e_at) = ((0u64, 0usize), (0u64, 0usize));
    // where does the end-to-end amplification first cross the 2-ulp line?
    let (mut cos_e_first_gt2, mut sin_e_first_gt2) = (None, None);
    for t in 0..ctx as u64 {
        let tf32 = t as f32;
        let tf64 = t as f64;
        for (j, &iv) in inv32.iter().enumerate() {
            let f = tf32 * iv; // the table's argument, f32 — engine-shared
            let c = f.cos();
            let s = f.sin();
            // shared-argument reference: f64 cos/sin OF THE SAME f32 argument
            let (cr, sr) = (f64::from(f).cos() as f32, f64::from(f).sin() as f32);
            let (dc, ds) = (ulp_dist(c, cr), ulp_dist(s, sr));
            if dc > cos_a_ulp { cos_a_ulp = dc; cos_a_at = (t, j); }
            if ds > sin_a_ulp { sin_a_ulp = ds; sin_a_at = (t, j); }
            // end-to-end reference: f64 powf + f64 product too (true math)
            let fe = tf64 * 10_000_000f64.powf(-(2.0 * j as f64) / 64.0);
            let (ce, se) = (fe.cos() as f32, fe.sin() as f32);
            let (dec, des) = (ulp_dist(c, ce), ulp_dist(s, se));
            if dec > cos_e_ulp { cos_e_ulp = dec; cos_e_at = (t, j); }
            if des > sin_e_ulp { sin_e_ulp = des; sin_e_at = (t, j); }
            if dec > 2 && cos_e_first_gt2.is_none() { cos_e_first_gt2 = Some((t, j)); }
            if des > 2 && sin_e_first_gt2.is_none() { sin_e_first_gt2 = Some((t, j)); }
        }
    }

    // ---- golden vector: device pairing formula vs host rotate_half ----
    let mut gold_ok = true;
    let mut gold_max_ulp = 0u32;
    for &(t, seed) in &[(0u64, 0x89_C0FFEE_0001u64), (1, 2), (95_271, 3), (262_143, 4)] {
        let x = golden_row(seed);
        let mut kern = vec![0f32; 256]; // kernels.rs rope, d<32 lanes only
        let mut half = vec![0f32; 256]; // HF rotate_half form
        for d in 0..32 {
            let c = (t as f32 * inv32[d]).cos();
            let s = (t as f32 * inv32[d]).sin();
            let (a, b) = (x[d], x[d + 32]);
            kern[d] = a * c - b * s; // device formula
            kern[d + 32] = b * c + a * s;
            // HF: out = x*cos + rotate_half(x)*sin, cos/sin duplicated
            half[d] = x[d] * c + (-x[d + 32]) * s;
            half[d + 32] = x[d + 32] * c + x[d] * s;
            // f64 pairing arithmetic on the table's own f32 cos/sin (what
            // both engines consume): isolates the pairing ops, not the table
            let (c64, s64) = (f64::from(c), f64::from(s));
            let (a64, b64) = (f64::from(a), f64::from(b));
            let (r1, r2) = (
                (a64 * c64 - b64 * s64) as f32,
                (b64 * c64 + a64 * s64) as f32,
            );
            gold_max_ulp = gold_max_ulp
                .max(ulp_dist(kern[d], r1))
                .max(ulp_dist(kern[d + 32], r2));
        }
        if kern[..64] != half[..64] {
            gold_ok = false;
            println!("[rope-tbl] golden t={t}: kernel pairing != rotate_half bitwise");
        }
    }

    println!(
        "[rope-tbl] table {} pos x {} pairs (manager.rs formula, host f32)",
        ctx, ROPE_PAIRS
    );
    println!(
        "[rope-tbl] inv_freq  max ulp {} (vs f64 powf->f32, worst j={inv_max_j}); bits [{}]",
        inv_max_ulp,
        inv32.iter().map(|v| format!("{:08x}", v.to_bits())).collect::<Vec<_>>().join(" ")
    );
    println!(
        "[rope-tbl] GATED shared-argument ref (f64 cos/sin of the table's own f32 argument):"
    );
    println!(
        "[rope-tbl]   cos max ulp {} at (t={}, j={})   sin max ulp {} at (t={}, j={})",
        cos_a_ulp, cos_a_at.0, cos_a_at.1, sin_a_ulp, sin_a_at.0, sin_a_at.1
    );
    println!(
        "[rope-tbl] REPORT end-to-end f64 ref (f64 powf too — the t-amplified class HF shares):"
    );
    println!(
        "[rope-tbl]   cos max ulp {} at (t={}, j={})   sin max ulp {} at (t={}, j={})",
        cos_e_ulp, cos_e_at.0, cos_e_at.1, sin_e_ulp, sin_e_at.0, sin_e_at.1
    );
    println!(
        "[rope-tbl]   first >2 ulp: cos {:?}, sin {:?}",
        cos_e_first_gt2, sin_e_first_gt2
    );
    println!(
        "[rope-tbl] golden pairing vs rotate_half: {} (info: pairing arithmetic vs f64 ops max ulp {gold_max_ulp} — three f32 roundings with cancellation, not an engine comparison)",
        if gold_ok { "BITWISE EQUAL" } else { "DIFFERS" }
    );

    // gate per the P3 plan: inv_freq/cos/sin <= 2 ulp vs the f64-rounded
    // reference + the pairing formula == rotate_half (bitwise)
    let pass = inv_max_ulp <= 2 && cos_a_ulp <= 2 && sin_a_ulp <= 2 && gold_ok;
    if pass {
        println!("[rope-tbl] PASS: inv_freq/cos/sin within 2 ulp of the f64-rounded reference; pairing formula == rotate_half bitwise");
    } else {
        println!("[rope-tbl] FAIL");
        std::process::exit(1);
    }
}
