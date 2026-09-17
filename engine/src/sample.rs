//! #20 — host-side sampling for production decode (opt-in, default greedy).
//!
//! The engine's kernels end in `argmax_k`; with `CROW_SAMPLE=1` the caller
//! reads the logits row back (1 MB over PCIe, ~0.3 ms) and samples on the host
//! with the data-sheet profile of the checkpoint (`generation_config.json`,
//! instruct mode): temperature 0.7, top_p 0.8, top_k 20, presence_penalty 1.5.
//! Every knob is an env var so a series can pin its profile:
//!   CROW_SAMPLE=1  CROW_TEMP  CROW_TOP_P  CROW_TOP_K  CROW_PRESENCE  CROW_SEED
//! Greedy stays the gate discipline: with CROW_SAMPLE unset nothing here runs
//! and the traces are unchanged.
//!
//! Since the GPU sampler (kernels.rs `sample_k`) the draw runs on the device as
//! the node behind `argmax_k`; this host code is its bit-for-bit reference and
//! stays reachable with CROW_SAMPLE_HOST=1 (logits readback + host top-k).

/// xorshift64* — seeded, dependency-free, good enough for token sampling.
pub struct Rng(u64);

impl Rng {
    pub fn new(seed: u64) -> Self {
        Rng(seed.wrapping_mul(0x9E37_79B9_7F4A_7C15).wrapping_add(0xD1B5_4A32_D192_ED03) | 1)
    }
    /// the raw state used AS IS — no `new` mixing. The probes seed with a
    /// literal state so their inputs are the same bytes on every machine.
    pub fn from_state(state: u64) -> Self {
        Rng(state)
    }
    pub fn next_u64(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.0 = x;
        x.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }
    /// uniform in [0, 1), f32 — the probe draw (24 mantissa bits)
    pub fn f01(&mut self) -> f32 {
        ((self.next_u64() >> 40) as f32) / (1u32 << 24) as f32
    }
    /// uniform in [0, 1)
    pub fn next_f64(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }
    /// raw state (uploaded to the device sampler, which advances it the same way)
    pub fn state(&self) -> u64 {
        self.0
    }
}

/// CROW_SAMPLE_HOST=1: keep sampling on the host (reference path); otherwise
/// CROW_SAMPLE=1 samples on the device
pub fn host_forced() -> bool {
    std::env::var("CROW_SAMPLE_HOST").as_deref() == Ok("1")
}

#[derive(Clone, Debug)]
pub struct Sampler {
    pub temperature: f32,
    pub top_p: f32,
    pub top_k: usize,
    pub presence_penalty: f32,
    /// CROW_SEED as given (the record names it, so a sampled answer is reproducible)
    pub seed: u64,
    pub rng: Rng,
    /// tokens generated so far in this answer (presence penalty applies to them)
    seen: std::collections::HashSet<usize>,
}

impl Clone for Rng {
    fn clone(&self) -> Self { Rng(self.0) }
}
impl std::fmt::Debug for Rng {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result { write!(f, "Rng({})", self.0) }
}

impl Sampler {
    /// data-sheet instruct profile; every field overridable by env
    pub fn from_env() -> Option<Self> {
        if std::env::var("CROW_SAMPLE").as_deref() != Ok("1") {
            return None;
        }
        let f = |k: &str, d: f32| std::env::var(k).ok().and_then(|v| v.parse().ok()).unwrap_or(d);
        let u = |k: &str, d: usize| std::env::var(k).ok().and_then(|v| v.parse().ok()).unwrap_or(d);
        Some(Sampler {
            temperature: f("CROW_TEMP", 0.7),
            top_p: f("CROW_TOP_P", 0.8),
            top_k: u("CROW_TOP_K", 20),
            presence_penalty: f("CROW_PRESENCE", 1.5),
            seed: u("CROW_SEED", 0) as u64,
            rng: Rng::new(u("CROW_SEED", 0) as u64),
            seen: Default::default(),
        })
    }

    /// #28: a sampler with the data-sheet instruct profile and an explicit seed.
    ///
    /// - The server path builds one PER REQUEST, so a warm process draws what a cold one draws.
    /// - Env is NOT read here; `from_env` stays the harness path and its override.
    /// - The caller overwrites `temperature`, `top_p`, `top_k`, `presence_penalty` (all pub).
    /// - `rng` starts at `Rng::new(seed)`, the state `enable_dev_sampler` uploads.
    pub fn new(seed: u64) -> Self {
        Sampler {
            temperature: 0.7,
            top_p: 0.8,
            top_k: 20,
            presence_penalty: 1.5,
            seed,
            rng: Rng::new(seed),
            seen: Default::default(),
        }
    }

    pub fn describe(&self) -> String {
        format!("sample: temp {} top_p {} top_k {} presence {} seed {} {}", self.temperature, self.top_p, self.top_k, self.presence_penalty, self.seed,
            if host_forced() { "host" } else { "gpu" })
    }

    /// register a token that is part of the answer (for the presence penalty)
    pub fn observe(&mut self, tok: usize) {
        self.seen.insert(tok);
    }

    /// one token from a logits row; greedy when temperature <= 0
    pub fn sample(&mut self, logits: &[f32]) -> usize {
        if self.temperature <= 0.0 {
            // greedy with the presence penalty applied (HF order: penalties, then argmax)
            let mut best = 0usize;
            let mut bv = f32::NEG_INFINITY;
            for (i, &l) in logits.iter().enumerate() {
                let v = if self.seen.contains(&i) { l - self.presence_penalty } else { l };
                if v > bv { bv = v; best = i; }
            }
            return best;
        }
        // presence penalty (HF semantics: subtract from every token already present)
        // + top_k on the raw scores: keep the k largest candidates
        let k = self.top_k.max(1).min(logits.len());
        let mut cand: Vec<(usize, f32)> = Vec::with_capacity(k + 1);
        for (i, &l) in logits.iter().enumerate() {
            let v = if self.seen.contains(&i) { l - self.presence_penalty } else { l };
            if cand.len() < k {
                cand.push((i, v));
                if cand.len() == k {
                    cand.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
                }
            } else if v > cand[k - 1].1 {
                // insert sorted, drop the smallest
                let pos = cand.iter().position(|c| v > c.1).unwrap();
                cand.insert(pos, (i, v));
                cand.pop();
            }
        }
        if cand.len() < k {
            cand.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        }
        // softmax with temperature over the candidates, then top_p nucleus
        let m = cand[0].1;
        let mut probs: Vec<f64> = cand.iter().map(|c| (((c.1 - m) / self.temperature) as f64).exp()).collect();
        let z: f64 = probs.iter().sum();
        for p in probs.iter_mut() {
            *p /= z;
        }
        let mut keep = probs.len();
        let mut acc = 0.0;
        for (i, p) in probs.iter().enumerate() {
            acc += p;
            if acc >= self.top_p as f64 {
                keep = i + 1;
                break;
            }
        }
        let z2: f64 = probs[..keep].iter().sum();
        let r = self.rng.next_f64() * z2;
        let mut acc = 0.0;
        for i in 0..keep {
            acc += probs[i];
            if r < acc {
                return cand[i].0;
            }
        }
        cand[keep - 1].0
    }
}

pub fn argmax(logits: &[f32]) -> usize {
    let mut best = 0usize;
    let mut bv = f32::NEG_INFINITY;
    for (i, &v) in logits.iter().enumerate() {
        if v > bv {
            bv = v;
            best = i;
        }
    }
    best
}

/// EOS ids of the checkpoint (generation_config): `<|im_end|>` and
/// `<|endoftext|>`. The second one is `geo::PLE_EOS` - the PLE shard reader's
/// end marker and the sampler's stop id are the SAME token, written once.
pub const EOS_IDS: [usize; 2] = [248046, crate::geo::PLE_EOS as usize];

/// the same two ids as i64, for the callers that compare a signed id
pub const EOS_IDS_I64: [i64; 2] = [EOS_IDS[0] as i64, EOS_IDS[1] as i64];

pub fn stop_on_eos() -> bool {
    std::env::var("CROW_STOP_EOS").as_deref() == Ok("1")
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn greedy_when_cold() {
        let mut s = Sampler { temperature: 0.0, top_p: 1.0, top_k: 5, presence_penalty: 0.0, seed: 0, rng: Rng::new(1), seen: Default::default() };
        assert_eq!(s.sample(&[0.1, 3.0, 2.0]), 1);
    }
    #[test]
    fn nucleus_never_picks_outside_top_k() {
        let mut s = Sampler { temperature: 1.0, top_p: 1.0, top_k: 2, presence_penalty: 0.0, seed: 0, rng: Rng::new(7), seen: Default::default() };
        let logits = [5.0, 4.0, -50.0, -50.0];
        for _ in 0..200 {
            assert!(s.sample(&logits) < 2);
        }
    }
    #[test]
    fn presence_penalty_moves_mass() {
        let mut s = Sampler { temperature: 0.0, top_p: 1.0, top_k: 3, presence_penalty: 5.0, seed: 0, rng: Rng::new(1), seen: Default::default() };
        s.observe(1);
        assert_eq!(s.sample(&[2.0, 3.0, 1.0]), 0);
    }
    #[test]
    fn seeded_is_reproducible() {
        let logits: Vec<f32> = (0..50).map(|i| (i as f32 * 0.37).sin() * 3.0).collect();
        let mk = || Sampler { temperature: 0.7, top_p: 0.8, top_k: 20, presence_penalty: 1.5, seed: 42, rng: Rng::new(42), seen: Default::default() };
        let (mut a, mut b) = (mk(), mk());
        let ta: Vec<usize> = (0..20).map(|_| { let t = a.sample(&logits); a.observe(t); t }).collect();
        let tb: Vec<usize> = (0..20).map(|_| { let t = b.sample(&logits); b.observe(t); t }).collect();
        assert_eq!(ta, tb);
    }
}
