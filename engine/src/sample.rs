//! #20 — host-side sampling for production decode (opt-in, default greedy).
//!
//! The engine's kernels end in `argmax_k`; with `CROW_SAMPLE=1` the caller
//! reads the logits row back (1 MB over PCIe, ~0.3 ms) and samples on the host
//! with the data-sheet profile of the checkpoint (`generation_config.json`,
//! instruct mode): temperature 0.7, top_p 0.8, top_k 20, presence_penalty 1.5.
//! Every knob is an env var so a series can pin its profile:
//!   CROW_SAMPLE=1  CROW_TEMP  CROW_TOP_P  CROW_TOP_K  CROW_PRESENCE  CROW_SEED
//!   CROW_MIN_P (#83: log-space tail filter, 0 = off)
//!   CROW_REPEAT CROW_FREQ CROW_LASTN (#84: windowed llama.cpp penalties,
//!   defaults neutral - repeat 1.0 / freq 0 / last_n 64)
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
    /// #83: the min_p tail filter, in log space exactly as llama.cpp has it
    /// (PR ggml-org/llama.cpp#3841): AFTER the top-k candidate selection and
    /// BEFORE the temperature softmax, candidates below
    /// `max_logit + ln(min_p)` drop out (min_keep 1 - the top token always
    /// survives). `<= 0` (and absent) disables the filter; the operating
    /// point of the module doc is min_p 0.01.
    pub min_p: f32,
    /// #84: llama.cpp windowed repetition penalty. Asymmetric, on the raw
    /// logits: `l > 0 -> l /= repeat`, else `l *= repeat` (the CTRL paper
    /// only divides; dividing a negative logit would RAISE it). `1.0` is
    /// neutral and skips the division/multiplication entirely, as llama.cpp
    /// does. Default 1.0.
    pub repeat_penalty: f32,
    /// #84: llama.cpp frequency penalty - `l -= c * freq` where c is the
    /// token's count in the window. Default 0 (neutral).
    pub frequency_penalty: f32,
    /// #84: the window depth in tokens, PROMPT TAIL + GENERATED (llama.cpp
    /// `penalty_last_n`, default 64). `0` disables the windowed pass. Clamped
    /// to `gen::SAMPLE_RING_MAX` at construction - the device twin's ring is
    /// that deep.
    pub penalty_last_n: usize,
    /// CROW_SEED as given (the record names it, so a sampled answer is reproducible)
    pub seed: u64,
    pub rng: Rng,
    /// tokens generated so far in THIS answer - the whole presence-penalty set (#68).
    ///
    /// Scope, pinned by `the_penalty_set_is_this_answers_tokens_only` below and by the device
    /// twin (`gen::enable_dev_sampler` uploads a zeroed mask per request, `kernels.rs sample_k`
    /// sets `mask[tok] = 1` for the token it just drew): the PROMPT is never in it, an earlier
    /// turn of the same session is never in it, and there is no last-n window. That is the
    /// semantics the model card's `presence_penalty` is written in (HF / vLLM, the penalty over
    /// the tokens generated in this response), not llama.cpp's windowed `repeat_penalty` over
    /// prompt plus generation.
    ///
    /// #84: that is STILL the presence scope, one door down. The llama.cpp window (`win`,
    /// `win_counts`, prompt tail + generated, ring of `penalty_last_n`) is a SECOND state that
    /// only the `repeat_penalty`/`frequency_penalty` pass reads. While that pass is armed
    /// (`win_armed`), `presence_penalty` joins IT - llama.cpp's `l -= c*freq + (c>0)*presence`
    /// over the window - instead of subtracting HF-style over this set; while it is not armed
    /// (the defaults), presence is exactly what it was, and no existing row moves.
    seen: std::collections::HashSet<usize>,
    /// #84: the last `penalty_last_n` accepted ids - prompt tail first, then
    /// every drawn id - oldest first. The device twin keeps the same window
    /// as `[head, fill, ids...]` in its ring buffer.
    win: std::collections::VecDeque<u32>,
    /// #84: count of every id currently in `win`; a token evicted from the
    /// ring decrements by exactly one (llama.cpp `token_count`, incremental).
    win_counts: std::collections::HashMap<usize, u16>,
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
        let f = |k: &str, d: f32| crate::geo::env_parse::<f32>(k).unwrap_or(d);
        let u = |k: &str, d: usize| crate::geo::env_parse::<usize>(k).unwrap_or(d);
        Some(Sampler {
            temperature: f("CROW_TEMP", 0.7),
            top_p: f("CROW_TOP_P", 0.8),
            top_k: u("CROW_TOP_K", 20),
            presence_penalty: f("CROW_PRESENCE", 1.5),
            min_p: f("CROW_MIN_P", 0.0),
            repeat_penalty: f("CROW_REPEAT", 1.0),
            frequency_penalty: f("CROW_FREQ", 0.0),
            penalty_last_n: u("CROW_LASTN", 64).min(crate::gen::SAMPLE_RING_MAX),
            seed: u("CROW_SEED", 0) as u64,
            rng: Rng::new(u("CROW_SEED", 0) as u64),
            seen: Default::default(),
            win: Default::default(),
            win_counts: Default::default(),
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
            min_p: 0.0,
            repeat_penalty: 1.0,
            frequency_penalty: 0.0,
            penalty_last_n: 64,
            seed,
            rng: Rng::new(seed),
            seen: Default::default(),
            win: Default::default(),
            win_counts: Default::default(),
        }
    }

    pub fn describe(&self) -> String {
        format!("sample: temp {} top_p {} top_k {} presence {} min_p {} repeat {} freq {} last_n {} seed {} {}",
            self.temperature, self.top_p, self.top_k, self.presence_penalty, self.min_p,
            self.repeat_penalty, self.frequency_penalty, self.penalty_last_n, self.seed,
            if host_forced() { "host" } else { "gpu" })
    }

    /// #83: ln(min_p) as ONE host-computed f32. The device sampler receives
    /// this value in its params block and adds it to ITS max candidate, so
    /// both sides threshold against the same constant and the filter is
    /// bit-equal; computing `logf` on the device instead would leave the
    /// boundary tied to two libm implementations.
    pub fn ln_min_p(&self) -> f32 {
        if self.min_p > 0.0 { self.min_p.ln() } else { 0.0 }
    }

    /// #84: is the llama.cpp windowed penalty pass armed? It takes
    /// `penalty_last_n > 0` (a window of 0 disables the whole pass, as in
    /// llama.cpp) plus one of the new knobs off neutral. `presence_penalty`
    /// deliberately does NOT arm it: its default is the data sheet's 1.5, and
    /// an armed-by-default window would silently move every existing row.
    pub fn win_armed(&self) -> bool {
        self.penalty_last_n > 0 && (self.repeat_penalty != 1.0 || self.frequency_penalty > 0.0)
    }

    /// #84: the windowed per-candidate penalty, the exact llama.cpp form -
    /// and exactly its SCOPE: only a candidate that is IN the window (c > 0,
    /// `token_count.find` hit) is touched at all. For such a candidate:
    /// asymmetric div/mul first (`if l > 0 { l / repeat } else { l * repeat }`
    /// - dividing a negative logit would raise it), then
    /// `l -= c*freq + presence`. While the window is armed,
    /// `presence_penalty` joins THIS form (llama.cpp `penalty_present`);
    /// while it is not, it keeps its HF subtraction over `seen` (#68).
    fn win_pen(&self, l: f32, c: u16) -> f32 {
        if c == 0 {
            return l; // not in the window: llama.cpp leaves the logit untouched
        }
        let mut v = l;
        if self.repeat_penalty != 1.0 {
            if v > 0.0 { v /= self.repeat_penalty; } else { v *= self.repeat_penalty; }
        }
        let c = c as f32;
        v - (c * self.frequency_penalty + self.presence_penalty)
    }

    /// the count of `tok` inside the window (0 = never seen in it)
    fn win_count(&self, tok: usize) -> u16 {
        self.win_counts.get(&tok).copied().unwrap_or(0)
    }

    /// push one id into the window, evicting (and decrementing) the oldest
    /// once `penalty_last_n` is full - llama.cpp's ring semantics, exactly
    fn win_push(&mut self, tok: u32) {
        let n = self.penalty_last_n;
        if n == 0 {
            return;
        }
        self.win.push_back(tok);
        *self.win_counts.entry(tok as usize).or_insert(0) += 1;
        if self.win.len() > n {
            if let Some(old) = self.win.pop_front() {
                if let Some(c) = self.win_counts.get_mut(&(old as usize)) {
                    *c -= 1;
                }
            }
        }
    }

    /// register a token that is part of the answer: the HF presence set (#68)
    /// AND the #84 window (the drawn id is an accepted id, llama.cpp `accept`)
    pub fn observe(&mut self, tok: usize) {
        self.seen.insert(tok);
        self.win_push(tok as u32);
    }

    /// #84: seed the window with the PROMPT - its last `penalty_last_n` ids,
    /// exactly what llama.cpp feeds its ring before the first sampled token.
    /// The HF presence set stays EMPTY here: it is this answer's tokens only
    /// (the #68 pin two fields up).
    pub fn observe_prompt(&mut self, ids: &[u32]) {
        self.win.clear();
        self.win_counts.clear();
        let n = self.penalty_last_n;
        let tail: &[u32] = if ids.len() > n { &ids[ids.len() - n..] } else { ids };
        for &t in tail {
            self.win_push(t);
        }
    }

    /// #84: the window contents oldest -> newest, for the device ring upload
    pub fn win_ids(&self) -> impl Iterator<Item = u32> + '_ {
        self.win.iter().copied()
    }

    /// #84: the nonzero window counts, for the device counts upload
    pub fn win_counts_nonzero(&self) -> impl Iterator<Item = (usize, u16)> + '_ {
        self.win_counts.iter().map(|(&k, &v)| (k, v))
    }

    /// one token from a logits row; greedy when temperature <= 0
    pub fn sample(&mut self, logits: &[f32]) -> usize {
        // #84: penalties FIRST, on the raw logits (llama.cpp chain order).
        // Window armed -> the llama.cpp form: asymmetric repeat, then
        // c*freq + (c>0)*presence over the prompt+generation window. Not
        // armed (the defaults) -> the HF presence subtraction over this
        // answer's tokens (#68), byte-identical to the pre-#84 sampler.
        let armed = self.win_armed();
        let pen = |s: &Self, i: usize, l: f32| -> f32 {
            if armed {
                s.win_pen(l, s.win_count(i))
            } else if s.seen.contains(&i) {
                l - s.presence_penalty
            } else {
                l
            }
        };
        if self.temperature <= 0.0 {
            // greedy: the penalized argmax. #84 put the windowed penalties
            // here too - llama.cpp runs its penalties sampler in greedy and
            // sampled alike; a not-armed sampler is the plain argmax of
            // record (greedy never had a penalty before #84).
            let mut best = 0usize;
            let mut bv = f32::NEG_INFINITY;
            for (i, &l) in logits.iter().enumerate() {
                let v = pen(self, i, l);
                if v > bv { bv = v; best = i; }
            }
            return best;
        }
        // top_k on the penalized scores: keep the k largest candidates
        let k = self.top_k.max(1).min(logits.len());
        let mut cand: Vec<(usize, f32)> = Vec::with_capacity(k + 1);
        for (i, &l) in logits.iter().enumerate() {
            let v = pen(self, i, l);
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
        // #83: min_p, the log-space tail filter of llama.cpp (PR #3841, paper
        // arXiv:2407.01082): AFTER top-k, BEFORE the temperature softmax, drop
        // every candidate below `max_logit + ln(min_p)` - no softmax needed.
        // The list is sorted descending, so the survivors are a prefix;
        // `max(1)` is min_keep: the top candidate always survives, which also
        // covers a degenerate min_p > 1 (ln > 0, threshold above the max).
        if self.min_p > 0.0 {
            let thr = cand[0].1 + self.ln_min_p();
            let mut keep = 0;
            while keep < cand.len() && cand[keep].1 >= thr {
                keep += 1;
            }
            cand.truncate(keep.max(1));
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
        let mut s = Sampler { temperature: 0.0, top_p: 1.0, top_k: 5, presence_penalty: 0.0, rng: Rng::new(1), ..Sampler::new(0) };
        assert_eq!(s.sample(&[0.1, 3.0, 2.0]), 1);
    }
    #[test]
    fn nucleus_never_picks_outside_top_k() {
        let mut s = Sampler { temperature: 1.0, top_p: 1.0, top_k: 2, presence_penalty: 0.0, rng: Rng::new(7), ..Sampler::new(0) };
        let logits = [5.0, 4.0, -50.0, -50.0];
        for _ in 0..200 {
            assert!(s.sample(&logits) < 2);
        }
    }
    #[test]
    fn presence_penalty_moves_mass() {
        let mut s = Sampler { temperature: 0.0, top_p: 1.0, top_k: 3, presence_penalty: 5.0, rng: Rng::new(1), ..Sampler::new(0) };
        s.observe(1);
        assert_eq!(s.sample(&[2.0, 3.0, 1.0]), 0);
    }
    /// #68: presence, not frequency - a token that was drawn twice is penalized ONCE.
    ///
    /// The two readings differ by more than a constant: with logits [0.0, 2.0] and a penalty of
    /// 1.5, presence leaves token 1 at 0.5 and it still wins; frequency would take 3.0 off it
    /// and token 0 would win. The device twin stores the set as a `u8` mask, which cannot count.
    #[test]
    fn the_presence_penalty_is_applied_once_per_distinct_token() {
        let mut s = Sampler { temperature: 0.0, top_p: 1.0, top_k: 3, presence_penalty: 1.5, rng: Rng::new(1), ..Sampler::new(0) };
        s.observe(1);
        s.observe(1);
        s.observe(1);
        assert_eq!(s.sample(&[0.0, 2.0]), 1);
    }

    /// #68: the set is this answer's own tokens - nothing else is in it.
    ///
    /// A fresh sampler penalizes nothing, however large the penalty is, and `Sampler::new` is
    /// what the server builds per request (`bin/serve.rs` `sampler_from`), so a 300-turn session
    /// on a warm process starts every turn with an empty set. Only `observe` fills it.
    #[test]
    fn the_penalty_set_is_this_answers_tokens_only() {
        let logits = [3.0f32, 1.0, 2.0];
        let mut fresh = Sampler::new(0);
        fresh.temperature = 0.0;
        fresh.presence_penalty = 100.0;
        assert_eq!(fresh.sample(&logits), 0, "a fresh sampler penalizes nothing");
        fresh.observe(0);
        assert_eq!(fresh.sample(&logits), 2, "only an observed token is penalized");
        // a second request builds a second sampler: the set of the first one is gone
        let mut next = Sampler::new(0);
        next.temperature = 0.0;
        next.presence_penalty = 100.0;
        assert_eq!(next.sample(&logits), 0);
    }

    #[test]
    fn seeded_is_reproducible() {
        let logits: Vec<f32> = (0..50).map(|i| (i as f32 * 0.37).sin() * 3.0).collect();
        let mk = || Sampler::new(42);
        let (mut a, mut b) = (mk(), mk());
        let ta: Vec<usize> = (0..20).map(|_| { let t = a.sample(&logits); a.observe(t); t }).collect();
        let tb: Vec<usize> = (0..20).map(|_| { let t = b.sample(&logits); b.observe(t); t }).collect();
        assert_eq!(ta, tb);
    }

    /// #83: min_p off (0.0, and absent) is the pre-#83 sampler byte for byte.
    /// The sequence is the golden the old code drew at this seed on this exact
    /// vector (captured 2026-09-20, before the filter existed) - defaults must
    /// never move a golden row implicitly.
    #[test]
    fn min_p_disabled_is_the_old_sampler_exactly() {
        let logits: Vec<f32> = (0..50).map(|i| (i as f32 * 0.37).sin() * 3.0).collect();
        let mut s = Sampler::new(42);
        let got: Vec<usize> = (0..20).map(|_| { let t = s.sample(&logits); s.observe(t); t }).collect();
        assert_eq!(
            got,
            vec![21, 39, 6, 37, 20, 5, 3, 22, 38, 24, 23, 4, 41, 40, 2, 19, 7, 21, 3, 1]
        );
    }

    /// #83: the filter keeps EXACTLY the log-space set - candidates with
    /// `logit >= max_logit + ln(min_p)`, llama.cpp PR #3841 - and every
    /// survivor stays reachable (the tail is cut, not the head).
    #[test]
    fn min_p_filters_exactly_the_log_space_set() {
        // max 4.0, ln(0.05) ~ -2.9957: threshold ~1.0043 keeps 4, 3, 2 and drops 0, -1
        let mut s = Sampler { temperature: 1.0, top_p: 1.0, top_k: 5, presence_penalty: 0.0, min_p: 0.05, ..Sampler::new(7) };
        let logits = [4.0f32, 3.0, 2.0, 0.0, -1.0];
        let mut seen = std::collections::HashSet::new();
        for _ in 0..400 {
            let t = s.sample(&logits);
            assert!(t < 3, "token {t} is below the min_p threshold");
            seen.insert(t);
        }
        assert_eq!(seen, [0, 1, 2].into_iter().collect(), "all three survivors stay reachable");
        // the comparison is >=, against the host's own f32 threshold constant
        // (the same bytes the device twin receives): a candidate exactly AT the
        // threshold survives, one 0.001 under it drops
        let thr = 4.0f32 + 0.05f32.ln();
        let mut b = Sampler { temperature: 1.0, top_p: 1.0, top_k: 5, presence_penalty: 0.0, min_p: 0.05, ..Sampler::new(7) };
        let logits = [4.0f32, thr, thr - 0.001, thr - 0.002];
        let mut bseen = std::collections::HashSet::new();
        for _ in 0..400 {
            let t = b.sample(&logits);
            assert!(t < 2, "token {t} is below the min_p threshold");
            bseen.insert(t);
        }
        assert_eq!(bseen, [0, 1].into_iter().collect(), "the at-threshold candidate survives");
    }

    /// #83: min_p 1 keeps only the maximum (p_i >= p_max); a degenerate
    /// min_p > 1 still keeps one candidate (min_keep, as llama.cpp).
    #[test]
    fn min_p_one_keeps_only_the_top_and_a_degenerate_still_keeps_one() {
        let logits = [3.0f32, 3.0, 2.999, 1.0];
        let mut one = Sampler { temperature: 1.0, top_p: 1.0, top_k: 4, presence_penalty: 0.0, min_p: 1.0, ..Sampler::new(5) };
        for _ in 0..100 {
            assert!(one.sample(&logits) < 2, "min_p 1 keeps only the maxima");
        }
        let mut big = Sampler { temperature: 1.0, top_p: 1.0, top_k: 4, presence_penalty: 0.0, min_p: 2.0, ..Sampler::new(5) };
        for _ in 0..100 {
            assert_eq!(big.sample(&logits), 0, "min_keep: the top candidate survives a threshold above the max");
        }
    }

    /// #83: greedy never sees min_p - there is no candidate list to filter and
    /// the effect of the filter on an argmax is identity anyway.
    #[test]
    fn greedy_ignores_min_p() {
        let mut s = Sampler { temperature: 0.0, top_p: 1.0, top_k: 3, presence_penalty: 0.0, min_p: 0.9, rng: Rng::new(1), ..Sampler::new(0) };
        assert_eq!(s.sample(&[0.0, 5.0, 4.9]), 1);
    }

    /// #83: `ln_min_p` is the ONE threshold constant - computed on the host,
    /// shipped to the device - so no libm disagreement can move the boundary.
    #[test]
    fn ln_min_p_is_the_host_computed_threshold_constant() {
        let off = Sampler::new(0);
        assert_eq!(off.min_p, 0.0);
        assert_eq!(off.ln_min_p(), 0.0, "disabled is 0, never a NaN from ln(0)");
        let mut on = Sampler::new(0);
        on.min_p = 0.01;
        assert_eq!(on.ln_min_p(), 0.01f32.ln());
    }

    /// #83: the device sampler of `kernels.rs` (`sample_topk_part` +
    /// `sample_k`, v2) as a host mirror, written from the CUDA source and not
    /// from `Sampler::sample`, so agreement between the two is evidence and
    /// not tautology. Same slice top-k, same key rounds in the (value desc,
    /// index asc) order, same f32 penalty arithmetic (#68 HF presence and the
    /// #84 windowed form), same f64 softmax/nucleus/draw, same xorshift64*
    /// state advance, same ring/count accept after the draw. The GPU itself
    /// stays robin's parity gate (CROW_SAMPLE_HOST=1 on the golden prompts).
    mod dev_mirror {
        pub const SAMPLE_MAXK: usize = 64;
        pub const SAMPLE_PARTS: usize = 64;
        pub const SAMPLE_RING_MAX: usize = 1024;

        /// one round of "largest key strictly below the previous pick", the
        /// exact conditions of `sample_rounds` / `sample_topk_part`
        fn rounds(items: &[(f32, usize)], k: usize) -> Vec<(f32, usize)> {
            let mut out = Vec::with_capacity(k);
            let (mut lv, mut li) = (f32::INFINITY, -1i64);
            for _ in 0..k {
                let mut best = f32::NEG_INFINITY;
                let mut bi = i64::MAX; // the kernel's 0x7fffffff empty-round sentinel
                for &(v, i) in items {
                    let ii = i as i64;
                    if v < lv || (v == lv && ii > li) {
                        if v > best || (v == best && ii < bi) {
                            best = v;
                            bi = ii;
                        }
                    }
                }
                out.push((best, bi as usize));
                lv = best;
                li = bi;
            }
            out
        }

        /// the #84 device-side window state: counts[V] u16 plus the ring
        /// {head, fill, ids} as `enable_dev_sampler` uploads them
        pub struct Window {
            pub counts: Vec<u16>,
            head: i32,
            fill: i32,
            ids: Vec<i32>,
        }

        impl Window {
            /// seed from a prompt tail, exactly the upload `enable_dev_sampler`
            /// builds: fill = min(len, lastn), head = fill % lastn, oldest first
            pub fn from_prompt(v: usize, tail: &[u32], lastn: usize) -> Self {
                let mut w = Window { counts: vec![0; v], head: 0, fill: 0, ids: vec![0; SAMPLE_RING_MAX] };
                let fill = tail.len().min(lastn);
                for (j, &t) in tail[tail.len() - fill..].iter().enumerate() {
                    w.ids[j] = t as i32;
                    w.counts[t as usize] += 1;
                }
                w.fill = fill as i32;
                w.head = if lastn > 0 { (fill % lastn) as i32 } else { 0 };
                w
            }

            /// the post-draw accept of `sample_k` thread 0, verbatim
            fn accept(&mut self, tok: usize, lastn: usize) {
                let mut head = self.head;
                let mut fill = self.fill;
                if fill >= lastn as i32 {
                    self.counts[self.ids[head as usize] as usize] -= 1;
                } else {
                    fill += 1;
                    self.fill = fill;
                }
                self.ids[head as usize] = tok as i32;
                self.counts[tok] += 1;
                head += 1;
                if head >= lastn as i32 {
                    head = 0;
                }
                self.head = head;
            }
        }

        /// the two stages and the draw of one `sample_k` launch; `state` is
        /// the device rng word, advanced exactly as the kernel advances it
        #[allow(clippy::too_many_arguments)]
        pub fn sample_k(
            logits: &[f32],
            mask: &[u8],
            win: &mut Window,
            temp: f32,
            top_p: f32,
            top_k: usize,
            pres: f32,
            min_p: f32,
            ln_min_p: f32,
            rep: f32,
            fq: f32,
            lastn: usize,
            state: &mut u64,
        ) -> usize {
            let n = logits.len();
            let k = top_k.max(1).min(n).min(SAMPLE_MAXK);
            let lastn = lastn.min(SAMPLE_RING_MAX);
            let armed = lastn > 0 && (rep != 1.0 || fq > 0.0);
            // stage 1: SAMPLE_PARTS blocks, k keys each; the #68/#84 penalties
            // on the raw scores, exactly the kernel's branch
            let slice = (n + SAMPLE_PARTS - 1) / SAMPLE_PARTS;
            let mut cand: Vec<(f32, usize)> = Vec::with_capacity(SAMPLE_PARTS * k);
            for b in 0..SAMPLE_PARTS {
                let lo = b * slice;
                let hi = (lo + slice).min(n);
                let items: Vec<(f32, usize)> = (lo..hi)
                    .map(|i| {
                        let mut v = logits[i];
                        if armed {
                            // #84: llama.cpp penalties - the whole block sits
                            // behind a count > 0 hit, tokens outside the
                            // window are untouched
                            let c = win.counts[i] as f32;
                            if c > 0.0 {
                                if rep != 1.0 {
                                    if v > 0.0 { v /= rep; } else { v *= rep; }
                                }
                                v -= c * fq + pres;
                            }
                        } else if mask[i] != 0 {
                            v -= pres;
                        }
                        (v, i)
                    })
                    .collect();
                cand.extend(rounds(&items, k));
            }
            // stage 2: the same rounds over the union (values already penalized)
            let cc = rounds(&cand, k);
            let tok = if temp <= 0.0 {
                cc[0].1
            } else {
                // #83: the min_p log-space filter, before the softmax, min_keep 1
                let mut k2 = k;
                if min_p > 0.0 {
                    let thr = cc[0].0 + ln_min_p;
                    k2 = 1;
                    while k2 < k && cc[k2].0 >= thr {
                        k2 += 1;
                    }
                }
                let m = cc[0].0;
                let mut pr = [0.0f64; SAMPLE_MAXK];
                let mut z = 0.0f64;
                for i in 0..k2 {
                    let a = (cc[i].0 - m) / temp;
                    pr[i] = (a as f64).exp();
                    z += pr[i];
                }
                for p in pr.iter_mut().take(k2) {
                    *p /= z;
                }
                let mut keep = k2;
                let mut acc = 0.0f64;
                for i in 0..k2 {
                    acc += pr[i];
                    if acc >= top_p as f64 {
                        keep = i + 1;
                        break;
                    }
                }
                let mut z2 = 0.0f64;
                for i in 0..keep {
                    z2 += pr[i];
                }
                let mut x = *state;
                x ^= x >> 12;
                x ^= x << 25;
                x ^= x >> 27;
                *state = x;
                let y = x.wrapping_mul(0x2545_F491_4F6C_DD1D);
                let rnd = ((y >> 11) as f64 / 9007199254740992.0) * z2;
                let mut tok = cc[keep - 1].1;
                acc = 0.0;
                for i in 0..keep {
                    acc += pr[i];
                    if rnd < acc {
                        tok = cc[i].1;
                        break;
                    }
                }
                tok
            };
            // the post-draw accepts, both device-side states
            if armed {
                win.accept(tok, lastn);
            }
            tok
        }
    }

    /// #83: host and device-mirror draw the same token for token on random
    /// vectors, min_p on and off - the unit-level half of the bit-parity gate.
    #[test]
    fn the_device_mirror_draws_what_the_host_draws_min_p() {
        let mut gen = Rng::new(1234);
        for &(temp, top_p, top_k, pres, min_p) in &[
            (0.7f32, 0.8f32, 20usize, 1.5f32, 0.0f32),
            (1.0, 0.95, 40, 0.0, 0.01),
            (1.0, 0.95, 40, 0.0, 0.05),
            (0.7, 0.8, 64, 1.5, 0.2),
            (1.0, 1.0, 5, 0.0, 1.0),
            (0.5, 0.9, 13, 0.7, 0.003),
        ] {
            let n = 300; // several of the device's 64 slices, slices shorter than k
            let logits: Vec<f32> = (0..n).map(|_| gen.next_f64() as f32 * 20.0 - 10.0).collect();
            let ln = if min_p > 0.0 { min_p.ln() } else { 0.0 };
            let mut host = Sampler {
                temperature: temp, top_p, top_k, presence_penalty: pres, min_p, ..Sampler::new(77)
            };
            let mut mask = vec![0u8; n];
            let mut state = Rng::new(77).state();
            let mut win = dev_mirror::Window::from_prompt(n, &[], 64);
            for step in 0..12 {
                let h = host.sample(&logits);
                let d = dev_mirror::sample_k(&logits, &mask, &mut win, temp, top_p, top_k, pres, min_p, ln, 1.0, 0.0, 64, &mut state);
                assert_eq!(h, d, "step {step} of temp {temp} top_p {top_p} k {top_k} pres {pres} min_p {min_p}");
                host.observe(h);
                mask[d] = 1;
            }
        }
    }

    // ---- #84: the windowed llama.cpp penalties ----

    /// #84: the defaults are NEUTRAL and the goldens are byte-identical. The
    /// two sequences are the goldens the pre-#83/#84 code drew (captured
    /// 2026-09-20): the sampled default profile and greedy with HF presence
    /// over this answer's tokens. A default sampler must never move them.
    #[test]
    fn windowed_defaults_are_neutral_and_goldens_stay_byte_identical() {
        let mut d = Sampler::new(0);
        assert_eq!((d.repeat_penalty, d.frequency_penalty, d.penalty_last_n), (1.0, 0.0, 64));
        assert!(!d.win_armed(), "the defaults must not arm the window");
        d.penalty_last_n = 0;
        assert!(!d.win_armed(), "a zero window disables the whole pass, as in llama.cpp");
        d.penalty_last_n = 64;
        d.repeat_penalty = 1.0;
        d.frequency_penalty = 0.0;
        assert!(!d.win_armed());

        let logits: Vec<f32> = (0..50).map(|i| (i as f32 * 0.37).sin() * 3.0).collect();
        let mut s = Sampler::new(42);
        let got: Vec<usize> = (0..20).map(|_| { let t = s.sample(&logits); s.observe(t); t }).collect();
        assert_eq!(
            got,
            vec![21, 39, 6, 37, 20, 5, 3, 22, 38, 24, 23, 4, 41, 40, 2, 19, 7, 21, 3, 1],
            "neutral penalties keep the sampled golden byte-identical"
        );
        let mut g = Sampler::new(9);
        g.temperature = 0.0;
        let gg: Vec<usize> = (0..10).map(|_| { let t = g.sample(&logits); g.observe(t); t }).collect();
        assert_eq!(gg, vec![38, 21, 4, 5, 22, 39, 37, 20, 3, 6],
            "greedy with HF presence (the not-armed form) keeps its golden");
    }

    /// #84: the repeat penalty is ASYMMETRIC, exactly llama.cpp's form: a
    /// positive logit is DIVIDED, a negative one is MULTIPLIED (dividing a
    /// negative would raise it - the CTRL-paper fix).
    #[test]
    fn repeat_penalty_is_asymmetric_div_for_positive_mul_for_negative() {
        let mk = |rep: f32| {
            let mut s = Sampler { repeat_penalty: rep, temperature: 0.0, presence_penalty: 0.0, ..Sampler::new(1) };
            s.observe_prompt(&[0, 1]); // both in the window, count 1 each
            s
        };
        // logits [1.0, -1.0, x]: rep 2 -> [0.5, -2.0, x] and the neutral
        // token 2 at 0.6 wins (0.6 > 0.5); at rep 1 (not armed) token 0's
        // 1.0 wins - and no token outside the window is ever divided
        assert_eq!(mk(2.0).sample(&[1.0, -1.0, 0.6]), 2);
        assert_eq!(mk(2.0).sample(&[1.0, -1.0, 0.4]), 0, "an uncounted 0.4 stays 0.4 and loses to 0.5");
        assert_eq!(mk(1.0).sample(&[1.0, -1.0, 0.6]), 0);
        // and the asymmetric direction itself: mul on the negative makes the
        // -1.0 token LOSE to an equal-magnitude neutral -1.0
        let mut s = Sampler { repeat_penalty: 2.0, temperature: 0.0, presence_penalty: 0.0, ..Sampler::new(1) };
        s.observe_prompt(&[1]);
        assert_eq!(s.sample(&[-1.0, -1.0]), 0, "the counted negative is multiplied, not divided");
    }

    /// #84: the frequency penalty scales with the WINDOW COUNT - once per
    /// occurrence in the window, unlike presence (#68's pin).
    #[test]
    fn frequency_penalty_scales_with_the_window_count() {
        let mk = |window: &[u32]| {
            let mut s = Sampler { frequency_penalty: 1.0, temperature: 0.0, presence_penalty: 0.0, ..Sampler::new(1) };
            s.observe_prompt(window);
            s
        };
        // token 1 at 3.0: count 2 -> 3.0 - 2 = 1.0, still beats 0.5
        assert_eq!(mk(&[1, 1]).sample(&[0.5, 3.0, 0.0]), 1);
        // count 4 -> 3.0 - 4 = -1.0, token 0's 0.5 wins
        assert_eq!(mk(&[1, 1, 1, 1]).sample(&[0.5, 3.0, 0.0]), 0);
    }

    /// #84: while the window is ARMED, `presence_penalty` joins the llama.cpp
    /// form - `(c > 0) * presence` over the WINDOW - and the HF per-answer
    /// subtraction is not read. The arm here is a numerically invisible
    /// frequency penalty (1e-30 * c is 0.0 at any logit scale), so the test
    /// sees the presence term alone.
    #[test]
    fn presence_joins_the_window_form_when_armed() {
        let mk = |freq: f32| {
            let mut s = Sampler { frequency_penalty: freq, presence_penalty: 1.5, temperature: 0.0, ..Sampler::new(1) };
            s.observe_prompt(&[1]);
            s
        };
        // armed: the PROMPT token 1 (count 1) is pushed to 3.0 - 1.5 = 1.5
        // and loses to token 0's 2.0; token 2 (count 0) keeps its 2.6 and wins
        assert_eq!(mk(1e-30).sample(&[2.0, 3.0, 2.6]), 2, "count 0 keeps its logit");
        let mut tight = mk(1e-30);
        assert_eq!(tight.sample(&[2.0, 3.0]), 0, "a counted prompt token is penalized");
        // not armed: presence stays HF - the prompt is NOT in the set, so the
        // same window changes nothing and token 1's 3.0 wins
        assert_eq!(mk(0.0).sample(&[2.0, 3.0]), 1);
    }

    /// #84: the window spans the PROMPT TAIL and evicts exactly - a token
    /// that fell out of the last-n window loses its count, and a drawn token
    /// enters it (llama.cpp ring semantics, incremental counts).
    #[test]
    fn the_window_spans_the_prompt_tail_and_evicts_exactly() {
        let mk = |last_n: usize| {
            let mut s = Sampler { frequency_penalty: 10.0, temperature: 0.0, presence_penalty: 0.0, penalty_last_n: last_n, ..Sampler::new(1) };
            s.observe_prompt(&[5, 6, 7]);
            s
        };
        let logits: Vec<f32> = (0..8).map(|i| [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 2.9, 2.8][i]).collect();
        // last_n 2 -> window {6, 7}: token 5 EVICTED, its 1.0 stands and wins
        // over the counted 2.9 (2.9 - 10 < 0)
        assert_eq!(mk(2).sample(&logits), 5);
        // last_n 3 -> window {5, 6, 7}: all three counted, token 0's 0.0 wins
        assert_eq!(mk(3).sample(&logits), 0);
        // a drawn token enters, the oldest leaves: observe(8) on last_n 2 ->
        // window {7, 8}: token 6 is free again and its 2.9 wins
        let mut s = mk(2);
        s.observe(8);
        assert_eq!(s.sample(&logits), 6);
    }

    /// #84: greedy applies the windowed penalties too - llama.cpp runs its
    /// penalties sampler in greedy and sampled alike. Not armed, the same
    /// request is the plain argmax of record.
    #[test]
    fn greedy_applies_the_windowed_penalties_too() {
        let mk = |rep: f32| {
            let mut s = Sampler { repeat_penalty: rep, temperature: 0.0, presence_penalty: 0.0, ..Sampler::new(1) };
            s.observe_prompt(&[1]);
            s
        };
        // token 1 at 4.0, rep 3: 4/3 ~ 1.33 < 1.9 -> token 2 wins
        assert_eq!(mk(3.0).sample(&[0.0, 4.0, 1.9]), 2);
        // rep 1.0 (neutral): plain argmax, token 1 wins
        assert_eq!(mk(1.0).sample(&[0.0, 4.0, 1.9]), 1);
    }

    /// #84: the device ring's depth is a frozen define pair - the
    /// `#define SAMPLE_RING_MAX` of `KERNEL_SRC` and `gen::SAMPLE_RING_MAX` -
    /// asserted at boot by `assert_kernel_defines`; pinned here without a GPU,
    /// together with the mirror's own copy, so the three cannot drift.
    #[test]
    fn the_ring_depth_define_matches_its_rust_twin() {
        assert_eq!(crate::kernels::define_u32("SAMPLE_RING_MAX"), crate::gen::SAMPLE_RING_MAX as u32);
        assert_eq!(crate::gen::SAMPLE_RING_MAX, 1024);
        assert_eq!(dev_mirror::SAMPLE_RING_MAX, crate::gen::SAMPLE_RING_MAX);
    }

    /// #84: host and device-mirror draw the same token for token with ARMED
    /// windows - prompt-seeded on both sides, host `observe` vs the kernel's
    /// post-draw ring accept - and in the armed-greedy form too.
    #[test]
    fn the_device_mirror_draws_what_the_host_draws_windowed() {
        let mut gen = Rng::new(4321);
        for &(temp, top_p, top_k, pres, min_p, rep, fq, last_n) in &[
            (1.0f32, 0.95f32, 40usize, 1.5f32, 0.01f32, 1.05f32, 0.0f32, 64usize),
            (1.0, 1.0, 20, 0.0, 0.0, 1.0, 0.3, 8),
            (0.8, 0.9, 64, 1.5, 0.05, 1.1, 0.2, 16),
            (0.0, 1.0, 5, 1.5, 0.0, 1.3, 0.1, 12), // armed GREEDY
        ] {
            let n = 300;
            let logits: Vec<f32> = (0..n).map(|_| gen.next_f64() as f32 * 20.0 - 10.0).collect();
            let ln = if min_p > 0.0 { min_p.ln() } else { 0.0 };
            let prompt: Vec<u32> = (0..40).map(|i| (gen.next_u64() % n as u64) as u32).collect();
            let mut host = Sampler {
                temperature: temp, top_p, top_k, presence_penalty: pres, min_p,
                repeat_penalty: rep, frequency_penalty: fq, penalty_last_n: last_n,
                ..Sampler::new(77)
            };
            host.observe_prompt(&prompt);
            let mut mask = vec![0u8; n];
            let mut state = Rng::new(77).state();
            let mut win = dev_mirror::Window::from_prompt(n, &prompt, last_n);
            for step in 0..12 {
                let h = host.sample(&logits);
                let d = dev_mirror::sample_k(&logits, &mask, &mut win, temp, top_p, top_k, pres, min_p, ln, rep, fq, last_n, &mut state);
                assert_eq!(
                    h, d,
                    "step {step} of temp {temp} rep {rep} fq {fq} last_n {last_n} pres {pres}"
                );
                host.observe(h);
                mask[d] = 1;
            }
        }
    }
}
