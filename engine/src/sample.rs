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
//!   CROW_DRY_MULTIPLIER CROW_DRY_BASE CROW_DRY_ALLOWED_LENGTH CROW_DRY_LASTN
//!   (#85: DRY, multiplier 0 = off) and the #92 tier, ALL off by default:
//!   CROW_TOP_N_SIGMA CROW_TYPICAL_P CROW_XTC_PROB CROW_XTC_THR
//!   CROW_MIROSTAT CROW_MIROSTAT_TAU CROW_MIROSTAT_ETA
//! Greedy stays the gate discipline: with CROW_SAMPLE unset nothing here runs
//! and the traces are unchanged.
//!
//! #85/#92 run on the HOST ONLY (the #85 issue's recommendation (a), and #92's
//! "host-sampler path is the supported route"): `Sampler::host_route` says
//! whether one of them is armed, and the caller (serve, decode, parity) then
//! reads the logits row back and draws here instead of arming `sample_k`. The
//! device sampler of #83/#84 is untouched and keeps its bit-parity role.
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
    /// #92: the seed word of XTC's OWN stream - the request seed through a
    /// different constant than the draw stream's, so the two never correlate
    /// and an armed XTC never shifts the main xorshift64* state.
    pub fn new_xtc(seed: u64) -> Self {
        Rng::new(seed ^ 0x5854_4353_5F4F_574E) // "XTCS_OWN"
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
    /// #85: DRY (Don't Repeat Yourself) multiplier, ported from KoboldCpp PR
    /// #982 into llama.cpp PR #9702. Over the last `dry_last_n` tokens
    /// (prompt tail + generated, its OWN window beside #84's) a reverse
    /// Z-algorithm finds, per candidate, the maximal repeat length L the
    /// candidate would extend; `logit -= multiplier * base^(L - allowed)`.
    /// `0.0` (the default) disables the whole pass, exactly llama.cpp's rule
    /// (`multiplier != 0 && base >= 1 && last_n != 0`).
    pub dry_multiplier: f32,
    /// #85: DRY base of the exponential, llama.cpp default 1.75. `>= 1.0`
    /// required or the pass is disabled; at exactly 1.0 the penalty is flat
    /// (`multiplier * 1`), as `powf(1.0, x) = 1.0` in the reference too.
    pub dry_base: f32,
    /// #85: DRY allowed length, llama.cpp default 2 - repeats up to this
    /// length cost nothing (`L - allowed` is the exponent).
    pub dry_allowed_length: i32,
    /// #85: DRY window depth (`dry_penalty_last_n`), llama.cpp default 64.
    /// HOST-only state (a `VecDeque`, no device twin), so no ring clamp.
    pub dry_last_n: usize,
    /// #92: top-nσ (llama.cpp PR #13264, arXiv:2411.07641): mean and std of
    /// the (penalized) non-inf logits, mask `logit < max - n*std`. BEFORE
    /// top-k, as in the chain. `<= 0` (the default) disables.
    pub top_n_sigma: f32,
    /// #92: typical_p (arXiv:2202.00666): softmax, entropy H, keep the
    /// smallest `|-ln p - H|` until the kept mass exceeds the threshold.
    /// AFTER top-k, BEFORE top_p/min_p, its llama.cpp chain position. `>= 1.0`
    /// (the default) disables.
    pub typical_p: f32,
    /// #92: XTC "exclude top choices" (llama.cpp PR #9742): per token, with
    /// probability `xtc_probability`, remove the leading run of tokens with
    /// `p >= xtc_threshold` (renormalized over the current candidate set),
    /// keeping `xtc_min_keep`. `<= 0` (the default) disables; a threshold
    /// `> 0.5` disables too, exactly llama.cpp's rule.
    pub xtc_probability: f32,
    /// #92: XTC threshold, llama.cpp default 0.1; values above 0.5 disable.
    pub xtc_threshold: f32,
    /// #92: XTC min_keep, llama.cpp chain default 1.
    pub xtc_min_keep: usize,
    /// #92: mirostat v2 (arXiv:2007.14966): `2` arms it, `0` (the default)
    /// is off. When armed it REPLACES the top-k/typical/top_p/min_p/xtc tail
    /// exactly as llama.cpp's chain build does (the mirostat branch adds only
    /// temp + mirostat): truncate `surprise = -log2(p) > mu`, renormalize,
    /// draw, `mu -= eta * (surprise - tau)`.
    pub mirostat: u8,
    /// #92: mirostat v2 target surprise, llama.cpp default 5.0.
    pub mirostat_tau: f32,
    /// #92: mirostat v2 learning rate, llama.cpp default 0.1.
    pub mirostat_eta: f32,
    /// #92: mirostat v2's mu - `2*tau` at request start (llama.cpp reset),
    /// then updated by every draw. Reset per request like the seed.
    pub miro_mu: f32,
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
    /// #85: the DRY window - the last `dry_last_n` accepted ids, prompt tail
    /// first, oldest first. A SECOND state beside #84's `win` because DRY's
    /// depth is its own knob; only `dry_armed` samplers push into it (llama.cpp
    /// `llama_sampler_dry_accept` early-returns when disabled).
    dry_win: std::collections::VecDeque<u32>,
    /// #85: the sequence breakers as llama.cpp processes them: head id -> the
    /// tokenizations of every tail that completes a breaker begun by the head
    /// (an EMPTY tail is a single-token breaker - `\n`, `:`, `"`, `*` are all
    /// that kind). Shared via `Arc` so a per-request sampler clones a
    /// reference, not the map.
    dry_breakers: std::sync::Arc<std::collections::HashMap<u32, Vec<Vec<u32>>>>,
    /// #92: XTC's OWN rng stream. llama.cpp seeds XTC's `std::mt19937`
    /// independently of the dist rng "for reproducibility"; the twin here is
    /// a second xorshift64* derived from the request seed through a different
    /// word, so enabling XTC never perturbs the main draw stream.
    xtc_rng: Rng,
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
        let i = |k: &str, d: i32| crate::geo::env_parse::<i32>(k).unwrap_or(d);
        let seed = u("CROW_SEED", 0) as u64;
        // #85: the DRY knobs (multiplier 0 = off, the default). The breaker
        // map needs the tokenizer; the harness path derives it from the
        // process-global one when that is reachable, and runs without
        // breakers otherwise (documented in docs/env.md).
        let dry_multiplier = f("CROW_DRY_MULTIPLIER", 0.0);
        let dry_base = f("CROW_DRY_BASE", 1.75);
        let dry_last_n = u("CROW_DRY_LASTN", 64);
        let dry_armed = dry_multiplier != 0.0 && dry_base >= 1.0 && dry_last_n > 0;
        let dry_breakers = if dry_armed {
            match crate::tokenizer::global() {
                Ok(tk) => dry_breaker_map(crate::geo::V, |id| tk.decode(&[id]).ok(), |s| tk.encode_raw(s).ok()),
                Err(_) => std::sync::Arc::new(std::collections::HashMap::new()),
            }
        } else {
            std::sync::Arc::new(std::collections::HashMap::new())
        };
        let mirostat = u("CROW_MIROSTAT", 0).min(2) as u8;
        let mirostat_tau = f("CROW_MIROSTAT_TAU", 5.0);
        Some(Sampler {
            temperature: f("CROW_TEMP", 0.7),
            top_p: f("CROW_TOP_P", 0.8),
            top_k: u("CROW_TOP_K", 20),
            presence_penalty: f("CROW_PRESENCE", 1.5),
            min_p: f("CROW_MIN_P", 0.0),
            repeat_penalty: f("CROW_REPEAT", 1.0),
            frequency_penalty: f("CROW_FREQ", 0.0),
            penalty_last_n: u("CROW_LASTN", 64).min(crate::gen::SAMPLE_RING_MAX),
            dry_multiplier,
            dry_base,
            dry_allowed_length: i("CROW_DRY_ALLOWED_LENGTH", 2),
            dry_last_n,
            top_n_sigma: f("CROW_TOP_N_SIGMA", 0.0),
            typical_p: f("CROW_TYPICAL_P", 1.0),
            xtc_probability: f("CROW_XTC_PROB", 0.0),
            xtc_threshold: f("CROW_XTC_THR", 0.1),
            xtc_min_keep: 1,
            mirostat,
            mirostat_tau,
            mirostat_eta: f("CROW_MIROSTAT_ETA", 0.1),
            miro_mu: 2.0 * mirostat_tau,
            seed,
            rng: Rng::new(seed),
            seen: Default::default(),
            win: Default::default(),
            win_counts: Default::default(),
            dry_win: Default::default(),
            dry_breakers,
            xtc_rng: Rng::new_xtc(seed),
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
            dry_multiplier: 0.0,
            dry_base: 1.75,
            dry_allowed_length: 2,
            dry_last_n: 64,
            top_n_sigma: 0.0,
            typical_p: 1.0,
            xtc_probability: 0.0,
            xtc_threshold: 0.1,
            xtc_min_keep: 1,
            mirostat: 0,
            mirostat_tau: 5.0,
            mirostat_eta: 0.1,
            miro_mu: 10.0,
            seed,
            rng: Rng::new(seed),
            seen: Default::default(),
            win: Default::default(),
            win_counts: Default::default(),
            dry_win: Default::default(),
            dry_breakers: std::sync::Arc::new(std::collections::HashMap::new()),
            xtc_rng: Rng::new_xtc(seed),
        }
    }

    pub fn describe(&self) -> String {
        format!("sample: temp {} top_p {} top_k {} presence {} min_p {} repeat {} freq {} last_n {} dry mul {} base {} allowed {} win {} nsigma {} typical {} xtc {}/{} miro {} tau {} eta {} seed {} {}",
            self.temperature, self.top_p, self.top_k, self.presence_penalty, self.min_p,
            self.repeat_penalty, self.frequency_penalty, self.penalty_last_n,
            self.dry_multiplier, self.dry_base, self.dry_allowed_length, self.dry_last_n,
            self.top_n_sigma, self.typical_p, self.xtc_probability, self.xtc_threshold,
            self.mirostat, self.mirostat_tau, self.mirostat_eta,
            self.seed,
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

    /// #85: llama.cpp's exact DRY disable rule
    /// (`dry_multiplier == 0 || dry_base < 1 || dry_penalty_last_n == 0`).
    pub fn dry_armed(&self) -> bool {
        self.dry_multiplier != 0.0 && self.dry_base >= 1.0 && self.dry_last_n > 0
    }

    /// #85/#92: does THIS request need the host sampler? DRY and the whole
    /// #92 tier are host-only (the #85 issue's recommendation (a), #92's "the
    /// host-sampler path is the supported route"); a sampler that answers
    /// false arms the device twin of #83/#84 as before. Neutral defaults
    /// answer false, so no existing row moves.
    pub fn host_route(&self) -> bool {
        self.dry_armed()
            || self.top_n_sigma > 0.0
            || self.typical_p < 1.0
            || (self.xtc_probability > 0.0 && self.xtc_threshold <= 0.5)
            || self.mirostat == 2
    }

    /// #92: mirostat v2's mu back to `2*tau`, the per-request reset llama.cpp
    /// performs (`llama_sampler_mirostat_v2_reset`). serve calls it after
    /// overwriting tau; `Sampler::new` starts there.
    pub fn reset_mirostat(&mut self) {
        self.miro_mu = 2.0 * self.mirostat_tau;
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
    /// AND the #85 DRY window (llama.cpp dry `accept`, armed samplers only)
    pub fn observe(&mut self, tok: usize) {
        self.seen.insert(tok);
        self.win_push(tok as u32);
        self.dry_push(tok as u32);
    }

    /// #84: seed the window with the PROMPT - its last `penalty_last_n` ids,
    /// exactly what llama.cpp feeds its ring before the first sampled token.
    /// The HF presence set stays EMPTY here: it is this answer's tokens only
    /// (the #68 pin two fields up).
    /// #85: the DRY window seeds from the prompt tail too (the server feeds
    /// the chain the prompt before the first draw), at its own depth.
    pub fn observe_prompt(&mut self, ids: &[u32]) {
        self.win.clear();
        self.win_counts.clear();
        let n = self.penalty_last_n;
        let tail: &[u32] = if ids.len() > n { &ids[ids.len() - n..] } else { ids };
        for &t in tail {
            self.win_push(t);
        }
        self.dry_win.clear();
        if self.dry_armed() {
            let dn = self.dry_last_n;
            let dtail: &[u32] = if ids.len() > dn { &ids[ids.len() - dn..] } else { ids };
            for &t in dtail {
                self.dry_push(t);
            }
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

    // ---------------------------------------------------------- #85: DRY

    /// #85: push one id into the DRY window (own depth `dry_last_n`). Only an
    /// ARMED sampler pushes - llama.cpp's dry `accept` early-returns while
    /// disabled, so a neutral sampler never touches this state.
    fn dry_push(&mut self, tok: u32) {
        if !self.dry_armed() {
            return;
        }
        self.dry_win.push_back(tok);
        while self.dry_win.len() > self.dry_last_n {
            self.dry_win.pop_front();
        }
    }

    /// #85: `dry_max_token_repeat` of llama.cpp `llama_sampler_dry_apply`,
    /// steps 1-3 verbatim: the breaker-limited rep_limit (step 1), the reverse
    /// Z-algorithm over the window (step 2, the ivanyu public-domain form the
    /// reference adapted), and the per-token maximum repeat a candidate would
    /// EXTEND (step 3). Returns `token -> L`; step 4 (`dry_logit_delta`)
    /// consumes it.
    fn dry_max_repeats(&self) -> std::collections::HashMap<u32, i32> {
        let mut out = std::collections::HashMap::new();
        let n = self.dry_win.len().min(self.dry_last_n);
        if n == 0 || n <= self.dry_allowed_length as usize {
            return out; // llama.cpp: last_n_repeat <= allowed -> nothing to do
        }
        let w = &self.dry_win;
        // rat(i): the i-th id from the END of the window (llama.cpp `rat`)
        let rat = |i: usize| w[w.len() - 1 - i];
        // step 1: the most recent sequence breaker bounds the scan; the head
        // token plus the longest matching tail behind it wins
        let mut rep_limit = n as i32;
        'scan: for i in 0..n {
            let Some(tails) = self.dry_breakers.get(&rat(i)) else { continue };
            let mut longest = -1i32;
            for tail in tails {
                let seq_len = tail.len() as i32;
                // the head is already matched; the tail must fit at rat(i-1..)
                if seq_len > longest && seq_len <= i as i32 {
                    if (0..tail.len()).all(|off| rat(i - off - 1) == tail[off]) {
                        longest = seq_len;
                    }
                }
            }
            if longest >= 0 {
                rep_limit = i as i32 - longest;
                break 'scan;
            }
        }
        if rep_limit < self.dry_allowed_length {
            return out;
        }
        // step 2: the reverse Z-algorithm. dry_repeat_count[last - k] is the
        // longest common prefix of rat(0..) and rat(k..), clamped to rep_limit.
        let last = n as i32 - 1;
        let mut rc = vec![0i32; n];
        let (mut lt, mut rt) = (0i32, 0i32);
        for k in 1..n as i32 {
            if k > rt {
                // outside the current Z-box: naive match
                let mut m = 0i32;
                while m + k < n as i32 && rat(m as usize) == rat((m + k) as usize) {
                    m += 1;
                }
                rc[(last - k) as usize] = m.min(rep_limit);
                if m > 0 {
                    lt = k;
                    rt = k + m - 1;
                }
            } else {
                let p = k - lt; // pair index
                let right_part_len = rt - k + 1;
                if rc[(last - p) as usize] < right_part_len {
                    rc[(last - k) as usize] = rc[(last - p) as usize].min(rep_limit);
                } else {
                    let mut j = rt + 1;
                    while j < n as i32 && rat(j as usize) == rat((j - k) as usize) {
                        j += 1;
                    }
                    rc[(last - k) as usize] = (j - k).min(rep_limit);
                    lt = k;
                    rt = j - 1;
                }
            }
        }
        // step 3: a token ends a repeat of length rc[i] -> drawing it would
        // extend one; keep the maximum per token
        for i in 0..n - 1 {
            let rl = rc[i];
            if rl >= self.dry_allowed_length {
                let tok = rat(n - 2 - i);
                let e = out.entry(tok).or_insert(i32::MIN);
                if *e < rl {
                    *e = rl;
                }
            }
        }
        out
    }

    /// #85: step 4, the logit penalty of a maximal repeat length L:
    /// `multiplier * base^(L - allowed)`, the exponent clamped at
    /// `FLOAT_MAX_LOG / ln(base)` = 88.7228391/ln(base) so `powf` cannot
    /// overflow (llama.cpp's own guard; the clamp only arms above
    /// base 1.000001, and a flat base of exactly 1 is powf's 1).
    fn dry_logit_delta(&self, repeat_len: i32) -> f32 {
        const FLOAT_MAX_LOG: f32 = 88.7228391;
        let mut max_exponent = 0i32;
        if self.dry_base > 1.000001 {
            max_exponent = (FLOAT_MAX_LOG / self.dry_base.ln()) as i32;
        }
        let mut e = repeat_len - self.dry_allowed_length;
        if max_exponent > 0 && e > max_exponent {
            e = max_exponent;
        }
        self.dry_multiplier * self.dry_base.powf(e as f32)
    }

    /// #85: does `tok` BEGIN a single-token sequence breaker (empty tail)?
    /// llama.cpp exempts exactly those from the penalty.
    fn dry_is_single_breaker(&self, tok: u32) -> bool {
        self.dry_breakers.get(&tok).map(|t| t.iter().any(|x| x.is_empty())).unwrap_or(false)
    }

    /// one token from a logits row; greedy when temperature <= 0
    ///
    /// The chain, in llama.cpp's order, with the crow-nest positions of #83
    /// (min_p after top-k, before the temperature softmax) kept as pinned:
    /// penalties (#68/#84) -> DRY (#85) -> top-nσ (#92, full row) -> top-k ->
    /// typical (#92) -> min_p (#83) -> temperature softmax -> top_p -> XTC
    /// (#92) -> draw; mirostat v2 (#92) REPLACES everything after top-nσ, as
    /// llama.cpp's own chain build does. Neutral defaults take none of the new
    /// branches, so the draws are the pre-#85/#92 draws byte for byte.
    pub fn sample(&mut self, logits: &[f32]) -> usize {
        // #84: penalties FIRST, on the raw logits (llama.cpp chain order).
        // Window armed -> the llama.cpp form: asymmetric repeat, then
        // c*freq + (c>0)*presence over the prompt+generation window. Not
        // armed (the defaults) -> the HF presence subtraction over this
        // answer's tokens (#68), byte-identical to the pre-#84 sampler.
        let armed = self.win_armed();
        // #85: DRY sits right behind the penalties (llama.cpp chain:
        // penalties -> DRY), subtracting from the SAME penalized logit.
        let dry = if self.dry_armed() { self.dry_max_repeats() } else { std::collections::HashMap::new() };
        let pen = |s: &Self, i: usize, l: f32| -> f32 {
            let v = if armed {
                s.win_pen(l, s.win_count(i))
            } else if s.seen.contains(&i) {
                l - s.presence_penalty
            } else {
                l
            };
            if let Some(&rl) = dry.get(&(i as u32)) {
                if !s.dry_is_single_breaker(i as u32) {
                    return v - s.dry_logit_delta(rl);
                }
            }
            v
        };
        // #92: mirostat v2 takes over the tail of the chain (temp-softmax,
        // surprise truncation, draw, mu update) over the FULL row - llama.cpp
        // builds [.. penalties/dry, temp, mirostat_v2] and nothing else.
        if self.mirostat == 2 {
            let scored: Vec<f32> = logits.iter().enumerate().map(|(i, &l)| pen(self, i, l)).collect();
            return self.mirostat_v2_draw(&scored);
        }
        if self.temperature <= 0.0 {
            // greedy: the penalized argmax. #84 put the windowed penalties
            // here too - llama.cpp runs its penalties sampler in greedy and
            // sampled alike; a not-armed sampler is the plain argmax of
            // record (greedy never had a penalty before #84). #85's DRY joins
            // the same branch, and #92's top-nσ is an identity on an argmax
            // (the maximum never sits below max - n*std), so greedy never
            // needs it.
            let mut best = 0usize;
            let mut bv = f32::NEG_INFINITY;
            for (i, &l) in logits.iter().enumerate() {
                let v = pen(self, i, l);
                if v > bv { bv = v; best = i; }
            }
            return best;
        }
        // #92: top-nσ BEFORE top-k (llama.cpp PR #13264, arXiv:2411.07641):
        // mean and std of the non-inf penalized logits, mask
        // `logit < max - n*std`. No softmax, no sort. f32 accumulators in
        // index order, as the reference accumulates.
        let sigma_cut: Option<f32> = if self.top_n_sigma > 0.0 && logits.len() > 1 {
            let mut maxv = f32::NEG_INFINITY;
            let mut sum = 0f32;
            let mut cnt = 0usize;
            for (i, &l) in logits.iter().enumerate() {
                let v = pen(self, i, l);
                if v != f32::NEG_INFINITY {
                    if v > maxv { maxv = v; }
                    sum += v;
                    cnt += 1;
                }
            }
            let mean = if cnt > 0 { sum / cnt as f32 } else { 0.0 };
            let mut acc = 0f32;
            for (i, &l) in logits.iter().enumerate() {
                let v = pen(self, i, l);
                if v != f32::NEG_INFINITY {
                    let d = v - mean;
                    acc += d * d;
                }
            }
            let std = if cnt > 0 { (acc / cnt as f32).sqrt() } else { 0.0 };
            Some(maxv - self.top_n_sigma * std)
        } else {
            None
        };
        // top_k on the penalized scores: keep the k largest candidates
        let k = self.top_k.max(1).min(logits.len());
        let mut cand: Vec<(usize, f32)> = Vec::with_capacity(k + 1);
        for (i, &l) in logits.iter().enumerate() {
            let mut v = pen(self, i, l);
            if let Some(t) = sigma_cut {
                if v < t {
                    v = f32::NEG_INFINITY; // llama.cpp masks to -INFINITY
                }
            }
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
        // #92: typical_p, its llama.cpp position (after top-k, before
        // top-p/min-p; llama.cpp runs it BEFORE the temperature stage, so the
        // filter's softmax is at temperature 1.0): entropy H, keep the
        // smallest |-ln p - H| until the kept mass EXCEEDS the threshold
        // (strict >, llama.cpp's own comparison), min_keep 1 (the chain
        // default). The kept set is re-sorted by logit, as the next llama.cpp
        // stage re-sorts; crow-nest's candidate list never left that order.
        if self.typical_p < 1.0 && cand.len() > 1 {
            let m = cand[0].1;
            let pr: Vec<f64> = cand.iter().map(|c| ((c.1 - m) as f64).exp()).collect();
            let z: f64 = pr.iter().sum();
            let pn: Vec<f64> = pr.iter().map(|p| p / z).collect();
            let h: f64 = pn.iter().map(|p| -p * p.ln()).sum();
            let shifted: Vec<f64> = pn.iter().map(|p| (-p.ln() - h).abs()).collect();
            let mut idx: Vec<usize> = (0..cand.len()).collect();
            idx.sort_by(|&a, &b| shifted[a].partial_cmp(&shifted[b]).unwrap());
            let mut cum = 0.0f64;
            let mut last = idx.len();
            for (j, &ix) in idx.iter().enumerate() {
                cum += pn[ix];
                // min_keep 1 (the llama.cpp chain default) makes the
                // `i >= min_keep - 1` half of the reference's condition
                // always true, so only the strict `>` on the mass remains
                if cum > self.typical_p as f64 {
                    last = j + 1;
                    break;
                }
            }
            let keep: std::collections::HashSet<usize> = idx[..last].iter().copied().collect();
            cand.retain(|c| keep.contains(&c.0));
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
        // #92: XTC, its llama.cpp position (after min-p, before the dist).
        // Its OWN rng stream draws the chance first - armed, threshold <= 0.5
        // and at least two candidates, exactly the reference's gate - and the
        // threshold compares the probs RENORMALIZED over the current kept
        // set, because llama.cpp's xtc re-softmaxes the array it receives.
        // With probability xtc_probability the leading run of `p >= threshold`
        // is dropped, keeping its LAST member and at least min_keep entries.
        let mut start = 0usize;
        if self.xtc_probability > 0.0 && self.xtc_threshold <= 0.5 && keep >= 2 {
            let chance = self.xtc_rng.next_f64();
            if chance <= self.xtc_probability as f64 {
                let zk: f64 = probs[..keep].iter().sum();
                let thr = self.xtc_threshold as f64;
                let mut pos_last = 0usize;
                for i in 0..keep {
                    if probs[i] / zk >= thr {
                        pos_last = i;
                    } else {
                        break;
                    }
                }
                if keep - pos_last >= self.xtc_min_keep && pos_last > 0 {
                    start = pos_last;
                }
            }
        }
        let z2: f64 = probs[start..keep].iter().sum();
        let r = self.rng.next_f64() * z2;
        let mut acc = 0.0;
        for i in start..keep {
            acc += probs[i];
            if r < acc {
                return cand[i].0;
            }
        }
        cand[keep - 1].0
    }

    /// #92: mirostat v2 over the full penalized row (llama.cpp
    /// `llama_sampler_mirostat_v2_apply`): temperature softmax (llama.cpp runs
    /// its temp sampler BEFORE mirostat), truncate the sorted candidates at
    /// the first with `surprise = -log2(p) > mu` (min 1), renormalize, draw,
    /// `mu -= eta * (surprise - tau)` on the RENORMALIZED drawn probability.
    fn mirostat_v2_draw(&mut self, scored: &[f32]) -> usize {
        let mut idx: Vec<u32> = (0..scored.len() as u32).collect();
        // value desc, index asc - the total order the rest of this file uses
        idx.sort_by(|&a, &b| {
            scored[b as usize].partial_cmp(&scored[a as usize]).unwrap().then(a.cmp(&b))
        });
        let t = if self.temperature > 0.0 { self.temperature } else { 1.0 };
        let mx = scored[idx[0] as usize];
        let mut pr: Vec<f64> = idx.iter().map(|&i| (((scored[i as usize] - mx) / t) as f64).exp()).collect();
        let z: f64 = pr.iter().sum();
        for p in pr.iter_mut() {
            *p /= z;
        }
        let mut keep = pr.len();
        for (i, p) in pr.iter().enumerate() {
            if -p.log2() > self.miro_mu as f64 {
                keep = i;
                break;
            }
        }
        let keep = keep.max(1);
        let zk: f64 = pr[..keep].iter().sum();
        let r = self.rng.next_f64() * zk;
        let mut acc = 0.0f64;
        let mut pick = keep - 1;
        for i in 0..keep {
            acc += pr[i];
            if r < acc {
                pick = i;
                break;
            }
        }
        let tok = idx[pick] as usize;
        let observed = -(pr[pick] / zk).log2();
        self.miro_mu =
            (self.miro_mu as f64 - self.mirostat_eta as f64 * (observed - self.mirostat_tau as f64)) as f32;
        tok
    }
}

/// #85: the four DRY sequence breakers of the issue - llama.cpp's own
/// default set. Repeats are not penalized across them, and a token that
/// BEGINS one is exempt from the penalty itself.
pub const DRY_DEFAULT_BREAKERS: [&str; 4] = ["\n", ":", "\"", "*"];

/// #85: llama.cpp `get_overlapping_token_sequences` (from KoboldCpp PR #982),
/// for the four default breakers: every vocab token whose text CONTAINS a
/// breaker becomes a single-token breaker (empty tail); every token whose
/// text merely BEGINS one maps to the tokenization of the breaker's
/// unmatched rest (its tail, clamped to 20 ids). For the single-character
/// defaults only the containment branch can fire - a token either contains
/// the character or does not begin it - so the tails are all empty and the
/// map is exactly "the ids whose text carries one of the four characters".
/// The full head->tails shape is kept because the scan and the exemption read
/// it (a multi-token breaker is exercisable through this door and by tests).
pub fn dry_breaker_map(
    n_vocab: usize,
    text_of: impl Fn(u32) -> Option<String>,
    encode: impl Fn(&str) -> Option<Vec<u32>>,
) -> std::sync::Arc<std::collections::HashMap<u32, Vec<Vec<u32>>>> {
    const MAX_CHAR_LEN: usize = 40;
    const MAX_SEQ_LEN: usize = 20;
    let mut map: std::collections::HashMap<u32, Vec<Vec<u32>>> = Default::default();
    for br in DRY_DEFAULT_BREAKERS {
        let s: &str = if br.len() > MAX_CHAR_LEN { &br[..MAX_CHAR_LEN] } else { br };
        let sb = s.as_bytes();
        for id in 0..n_vocab as u32 {
            let Some(word) = text_of(id) else { continue };
            let wb = word.as_bytes();
            if word.contains(s) {
                map.entry(id).or_default().push(Vec::new());
            } else {
                for pos in 0..wb.len() {
                    if wb[pos] != sb[0] {
                        continue;
                    }
                    let mut i = 1usize;
                    let mut matched = true;
                    while i < sb.len() && i + pos < wb.len() {
                        if wb[pos + i] != sb[i] {
                            matched = false;
                            break;
                        }
                        i += 1;
                    }
                    if matched {
                        let mut toks = encode(&s[i..]).unwrap_or_default();
                        if toks.len() > MAX_SEQ_LEN {
                            toks.truncate(MAX_SEQ_LEN);
                        }
                        let tails = map.entry(id).or_default();
                        if !tails.iter().any(|t| t == &toks) {
                            tails.push(toks);
                        }
                    }
                }
            }
        }
    }
    std::sync::Arc::new(map)
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

/// #91: the most `top_logprobs` a request may ask for - OpenAI's own ceiling
/// (`CreateChatCompletionRequest.top_logprobs`, 0..=20).
pub const MAX_TOP_LOGPROBS: usize = 20;

/// what the log-probability of a JSON-unrepresentable value (a `-inf` logit, or
/// a NaN) is reported as: OpenAI's own floor for "vanishingly unlikely"
pub const LOGPROB_FLOOR: f64 = -9999.0;

/// #91: the log-probabilities of ONE position, read off the RAW logits row
#[derive(Debug, Clone, PartialEq)]
pub struct PosLogprobs {
    /// the id this position generated (drawn, argmax, or #81-injected)
    pub chosen: usize,
    /// `ln p(chosen)` under the raw distribution
    pub chosen_lp: f64,
    /// the `n` most likely ids, value descending, index ascending on a tie
    pub top: Vec<(usize, f64)>,
}

/// - #91: `log_softmax(row)` at `chosen` and at the `n` largest entries, in f64
/// - the RAW model distribution: the lm_head row as it left the device, before
///   `logit_bias`, the penalties, DRY, temperature and every truncation of the
///   chain (`Sampler::sample`) - llama-server's default (`n_probs` without
///   `post_sampling_probs`: "a simple softmax of the logits without considering
///   any other sampler settings"). The margin a near-tie question needs is a
///   property of the MODEL, not of the sampler profile that happened to run.
/// - order: value descending, index ascending on an exact tie - the total order
///   the rest of this file uses; so `top[0]` is the lowest id among the maxima.
///   The device `argmax_k` reduction may pick another id of an exact tie, and
///   then `chosen != top[0].0` with a margin of exactly 0.
/// - `ln Z` = max + ln(sum exp(l - max)), accumulated in f64 in index order.
///   NaN entries are skipped (never top, not in Z); a `-inf` entry is probability
///   0. A non-finite result is reported as `LOGPROB_FLOOR` (JSON has no -inf).
/// - one pass for the max and the top-n (a bounded insertion list, n <= 20), one
///   for Z: 2 x V reads of an f32 row the caller already holds on the host.
/// - pure: the test drives it on synthetic rows.
pub fn pos_logprobs(row: &[f32], chosen: usize, n: usize) -> PosLogprobs {
    let n = n.min(MAX_TOP_LOGPROBS).min(row.len());
    let mut maxv = f32::NEG_INFINITY;
    // descending by value; a later index never displaces an equal earlier one
    let mut top: Vec<(usize, f32)> = Vec::with_capacity(n + 1);
    for (i, &l) in row.iter().enumerate() {
        if l.is_nan() {
            continue;
        }
        if l > maxv {
            maxv = l;
        }
        if n > 0 && (top.len() < n || l > top[top.len() - 1].1) {
            let pos = top.iter().position(|c| l > c.1).unwrap_or(top.len());
            top.insert(pos, (i, l));
            if top.len() > n {
                top.pop();
            }
        }
    }
    let mut z = 0f64;
    if maxv.is_finite() {
        for &l in row {
            if !l.is_nan() {
                z += ((l - maxv) as f64).exp();
            }
        }
    }
    let lnz = maxv as f64 + z.ln();
    let lp = |l: f32| -> f64 {
        let v = l as f64 - lnz;
        if v.is_finite() { v } else { LOGPROB_FLOOR }
    };
    PosLogprobs {
        chosen,
        chosen_lp: row.get(chosen).map(|&l| lp(l)).unwrap_or(LOGPROB_FLOOR),
        top: top.into_iter().map(|(i, l)| (i, lp(l))).collect(),
    }
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

/// #93: the device buffer a re-booking write goes to
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RebookBuf {
    /// the presence mask, `[V]` u8
    Mask,
    /// the #84 window counts, `[V]` u16
    Counts,
    /// the #84 ring, i32 `{head, fill, ids[..]}`
    Ring,
}

/// - #93: the device `sample_k` node ACCEPTS the id it drew (presence
///   bit, and with the window armed: ring slot + count, `kernels.rs` after the draw).
///   When the host REPLACES that id (a tool-grammar redraw), the booking must follow,
///   or the refused id is penalized for the rest of the answer (an EOS the grammar
///   refused mid-call would carry the presence penalty to the end) and the kept one
///   is not.
/// - pure: the writes, as `(buffer, byte offset, bytes)`, that turn the state after
///   `accept(drawn)` into the state after `accept(kept)`. Inputs read off the device
///   AFTER the draw: `head_after` = ring[0], `count_drawn` / `count_kept` = counts.
/// - `drawn_seen`: `drawn` was in this answer before this draw, so its presence bit
///   stays; `lastn`: the device window depth, 0 when the window is not armed.
pub fn rebook_plan(
    drawn: usize,
    kept: usize,
    drawn_seen: bool,
    lastn: usize,
    head_after: i32,
    count_drawn: u16,
    count_kept: u16,
) -> Vec<(RebookBuf, usize, Vec<u8>)> {
    let mut w = Vec::new();
    if drawn == kept {
        return w;
    }
    if !drawn_seen {
        w.push((RebookBuf::Mask, drawn, vec![0u8]));
    }
    w.push((RebookBuf::Mask, kept, vec![1u8]));
    if lastn > 0 {
        let slot = (head_after.max(0) as usize + lastn - 1) % lastn;
        w.push((RebookBuf::Ring, (2 + slot) * 4, (kept as i32).to_le_bytes().to_vec()));
        w.push((RebookBuf::Counts, drawn * 2, count_drawn.saturating_sub(1).to_le_bytes().to_vec()));
        w.push((RebookBuf::Counts, kept * 2, count_kept.saturating_add(1).to_le_bytes().to_vec()));
    }
    w
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

    // ---- #85: DRY (Don't Repeat Yourself) ----

    /// a DRY sampler with hand-set breaker ids (empty tail = single-token
    /// breaker, a Vec<u32> tail = the tokens that complete a multi-token one)
    fn dry_with(breakers: &[(u32, Vec<Vec<u32>>)]) -> Sampler {
        let mut s = Sampler::new(1);
        s.dry_multiplier = 0.8;
        s.dry_base = 1.75;
        s.dry_allowed_length = 2;
        s.dry_last_n = 64;
        s.dry_breakers = std::sync::Arc::new(breakers.iter().cloned().collect());
        s
    }

    /// #85: the defaults are OFF (multiplier 0) and the goldens are
    /// byte-identical - the pre-#85/#92 draws, pinned. Also llama.cpp's exact
    /// disable rule: base < 1 or window 0 or multiplier 0 all leave the
    /// sampler inert, whatever the other knobs say.
    #[test]
    fn dry_and_tier_defaults_are_neutral_and_goldens_stay_byte_identical() {
        let d = Sampler::new(0);
        assert_eq!((d.dry_multiplier, d.dry_base, d.dry_allowed_length, d.dry_last_n), (0.0, 1.75, 2, 64));
        assert!(!d.dry_armed(), "multiplier 0 disables, the default");
        assert!(!d.host_route(), "a neutral sampler arms the device twin as before");
        assert_eq!((d.top_n_sigma, d.typical_p, d.xtc_probability, d.xtc_threshold, d.mirostat), (0.0, 1.0, 0.0, 0.1, 0));
        assert!(!d.host_route(), "the #92 tier is off by default");
        // the disable rule, one knob at a time
        let mut s = dry_with(&[]);
        assert!(s.dry_armed());
        s.dry_base = 0.999;
        assert!(!s.dry_armed(), "base < 1 disables, llama.cpp's own rule");
        s.dry_base = 1.0;
        assert!(s.dry_armed(), "base exactly 1 is armed (the penalty is then flat)");
        s.dry_last_n = 0;
        assert!(!s.dry_armed(), "window 0 disables");
        s.dry_last_n = 64;
        s.dry_multiplier = 0.0;
        assert!(!s.dry_armed());

        // the goldens of the pre-#83/#84/#85 code, byte for byte
        let logits: Vec<f32> = (0..50).map(|i| (i as f32 * 0.37).sin() * 3.0).collect();
        let mut a = Sampler::new(42);
        let got: Vec<usize> = (0..20).map(|_| { let t = a.sample(&logits); a.observe(t); t }).collect();
        assert_eq!(
            got,
            vec![21, 39, 6, 37, 20, 5, 3, 22, 38, 24, 23, 4, 41, 40, 2, 19, 7, 21, 3, 1],
            "neutral DRY/#92 keeps the sampled golden byte-identical"
        );
        let mut g = Sampler::new(9);
        g.temperature = 0.0;
        let gg: Vec<usize> = (0..10).map(|_| { let t = g.sample(&logits); g.observe(t); t }).collect();
        assert_eq!(gg, vec![38, 21, 4, 5, 22, 39, 37, 20, 3, 6], "and the greedy golden");
    }

    /// #85: a candidate that would EXTEND an existing repeat pays exactly
    /// `multiplier * base^(L - allowed)`. Window [5,6,5,6], allowed 2: the
    /// Z-scan gives token 5 L=2, so the penalty is the flat `multiplier`
    /// (exponent 0); token 6 extends nothing and pays nothing.
    #[test]
    fn dry_penalizes_the_token_that_would_extend_a_repeat() {
        let mk = |window: &[u32], logits: &[f32]| {
            let mut s = dry_with(&[]);
            s.temperature = 0.0;
            s.presence_penalty = 0.0;
            s.observe_prompt(window);
            s.sample(logits)
        };
        // token 5 at 0.9 pays 0.8 -> 0.1, still beats token 7's 0.05
        assert_eq!(mk(&[5, 6, 5, 6], &[0.05, -10.0, -10.0, -10.0, -10.0, 0.9, -10.0, 0.0]), 5);
        // at 0.7 the same candidate falls to -0.1 and the neutral 0.05 wins
        assert_eq!(mk(&[5, 6, 5, 6], &[0.05, -10.0, -10.0, -10.0, -10.0, 0.7, -10.0, 0.0]), 0);
        // token 6 is NOT a repeat extender here: it keeps its 0.7 and wins
        assert_eq!(mk(&[5, 6, 5, 6], &[0.05, -10.0, -10.0, -10.0, -10.0, -10.0, 0.7, 0.0]), 6);
        // multiplier 0 (disabled): the same 0.7 for token 5 wins untouched
        let mut off = dry_with(&[]);
        off.dry_multiplier = 0.0;
        off.temperature = 0.0;
        off.presence_penalty = 0.0;
        off.observe_prompt(&[5, 6, 5, 6]);
        assert_eq!(off.sample(&[0.05, -10.0, -10.0, -10.0, -10.0, 0.7, -10.0, 0.0]), 5);
        // the exponent is live: allowed 0 on window [5,6,5,6] gives token 5
        // L=2 -> penalty 0.8 * 1.75^2 = 2.45, so even 2.0 loses to 0.05
        let mut exp = dry_with(&[]);
        exp.dry_allowed_length = 0;
        exp.temperature = 0.0;
        exp.presence_penalty = 0.0;
        exp.observe_prompt(&[5, 6, 5, 6]);
        assert_eq!(exp.sample(&[0.05, -10.0, -10.0, -10.0, -10.0, 2.0, -10.0, 0.0]), 0);
        assert_eq!(exp.sample(&[0.05, -10.0, -10.0, -10.0, -10.0, 2.6, -10.0, 0.0]), 5, "2.6 - 2.45 wins again");
    }

    /// #85: breaker semantics. A single-token breaker as the LAST token
    /// zeroes rep_limit and nothing is penalized (repeats are not penalized
    /// across natural boundaries); a breaker further back bounds the scan;
    /// and the breaker token that would itself extend a repeat is EXEMPT
    /// (llama.cpp skips candidates that begin a single-token breaker).
    #[test]
    fn dry_sequence_breakers_bound_the_scan_and_exempt_their_own_token() {
        // (a) breaker 9 as the last window id: rep_limit = 0 -> nothing pays
        let mut s = dry_with(&[(9, vec![vec![]])]);
        s.temperature = 0.0;
        s.presence_penalty = 0.0;
        s.observe_prompt(&[5, 6, 5, 6, 9]);
        assert_eq!(s.sample(&[0.05, -10.0, -10.0, -10.0, -10.0, 0.7, -10.0, 0.0]), 5,
            "the boundary behind the breaker hides the repeat from the scan");
        // (b) window [3,4,9,3,4], breaker 9: rep_limit 2, and the extender the
        // scan finds IS token 9 - which is exempt, so 9 keeps its logit while
        // without the breaker it would pay 0.8
        let mut b = dry_with(&[(9, vec![vec![]])]);
        b.temperature = 0.0;
        b.presence_penalty = 0.0;
        b.observe_prompt(&[3, 4, 9, 3, 4]);
        assert_eq!(b.sample(&[0.05, -10.0, -10.0, -10.0, -10.0, -10.0, -10.0, -10.0, -10.0, 0.7]), 9,
            "the single-token breaker is exempt from its own repeat penalty");
        let mut nob = dry_with(&[]);
        nob.temperature = 0.0;
        nob.presence_penalty = 0.0;
        nob.observe_prompt(&[3, 4, 9, 3, 4]);
        assert_eq!(nob.sample(&[0.05, -10.0, -10.0, -10.0, -10.0, -10.0, -10.0, -10.0, -10.0, 0.7]), 0,
            "the same window without the breaker pays the flat 0.8 (0.7 - 0.8 < 0.05)");
        // (c) a MULTI-token breaker (head 7, tail [8]): the full pair at the
        // end bounds the scan exactly like a single-token one, and a head
        // that only BEGINS one (its tail does not follow) is NOT exempt
        let mut c = dry_with(&[(7, vec![vec![8]])]);
        c.temperature = 0.0;
        c.presence_penalty = 0.0;
        c.observe_prompt(&[5, 6, 5, 6, 7, 8]);
        assert!(c.dry_max_repeats().is_empty(), "the 7,8 pair at the end hides the repeat");
        // window [5,6,7,8,5,6]: rep_limit = 2 (breaker at i=3, tail len 1),
        // the scan's extender is token 7 - the HEAD of a multi-token breaker,
        // which llama.cpp does NOT exempt
        let mut d = dry_with(&[(7, vec![vec![8]])]);
        d.temperature = 0.0;
        d.presence_penalty = 0.0;
        d.observe_prompt(&[5, 6, 7, 8, 5, 6]);
        assert_eq!(d.sample(&[0.05, -10.0, -10.0, -10.0, -10.0, -10.0, -10.0, 0.7]), 0,
            "a head-with-tail breaker is not a single-token breaker: 0.7 - 0.8 < 0.05");
    }

    /// #85: the exponent clamp of llama.cpp step 4 - `88.7228391/ln(base)`,
    /// armed only above base 1.000001. At base 1.75 the cap is 158: an
    /// exponent past it freezes the penalty at `multiplier * 1.75^158`
    /// (finite, ~2.6e38) instead of overflowing, and a base at or under
    /// 1.000001 clamps nothing (powf's own `powf(1.0, x) = 1`).
    #[test]
    fn dry_exponent_clamps_at_float_max_log_over_ln_base() {
        let mut s = dry_with(&[]); // base 1.75, allowed 2
        let cap = (88.722_84_f32 / 1.75f32.ln()) as i32; // 158
        assert_eq!(cap, 158);
        let at_cap = 0.8f32 * 1.75f32.powf(cap as f32);
        assert_eq!(s.dry_logit_delta(cap + 2), at_cap, "exponent exactly at the cap");
        assert_eq!(s.dry_logit_delta(cap + 100), at_cap, "far past the cap: frozen at the cap value");
        // one under the cap (repeat_len cap+1 -> exponent cap-1, allowed 2) is unclamped
        assert_eq!(s.dry_logit_delta(cap + 1), 0.8f32 * 1.75f32.powf((cap - 1) as f32));
        // the behavioral half: an all-one-token window cannot produce a NaN
        // or an inf - the penalized candidate just loses, finitely
        let mut b = dry_with(&[]);
        b.dry_multiplier = 1.0;
        b.temperature = 0.0;
        b.presence_penalty = 0.0;
        let all7: Vec<u32> = vec![7; 300];
        b.observe_prompt(&all7);
        let t = b.sample(&[0.05, -10.0, -10.0, -10.0, -10.0, -10.0, -10.0, 1.0]);
        assert!(t < 8, "a valid index came back (no NaN): {t}");
        assert_eq!(t, 0, "the clamped penalty (~2.6e38) pushes any logit under 0.05");
        // a base under 1.000001 clamps nothing: flat or near-flat penalties
        let mut c = dry_with(&[]);
        c.dry_base = 1.0000005;
        assert_eq!(c.dry_logit_delta(1_000_000), 0.8f32 * c.dry_base.powf(999_998.0),
            "no clamp below the 1.000001 line (the exponent is L - allowed 2)");
        let mut one = dry_with(&[]);
        one.dry_base = 1.0;
        assert_eq!(one.dry_logit_delta(999), 0.8, "base exactly 1: the penalty is the flat multiplier");
    }

    /// #85: the window spans the PROMPT TAIL at its own depth and pushes
    /// every drawn id, evicting past `dry_last_n` (llama.cpp dry accept).
    #[test]
    fn dry_window_spans_the_prompt_tail_and_evicts_at_its_own_depth() {
        let mut s = dry_with(&[]);
        s.dry_last_n = 2;
        s.temperature = 0.0;
        s.presence_penalty = 0.0;
        s.observe_prompt(&[5, 6, 5, 6, 9]);
        // only the last two ids are in the window ([6, 9]) - a window that
        // shallow cannot hold a repeat of the allowed length
        assert!(s.dry_max_repeats().is_empty(), "nothing reaches allowed 2 at depth 2");
        // the repeat has to END at the end of the window: [9,5,6,5,6] ends in
        // it and token 5 extends L=2; the same ids at depth 2 see only [5,6]
        let mut w = dry_with(&[]);
        w.temperature = 0.0;
        w.presence_penalty = 0.0;
        w.observe_prompt(&[9, 5, 6, 5, 6]);
        assert_eq!(w.dry_max_repeats().get(&5), Some(&2), "full depth: token 5 extends L=2");
        w.dry_last_n = 2;
        w.observe_prompt(&[9, 5, 6, 5, 6]);
        assert!(w.dry_max_repeats().is_empty(), "depth 2: the window is [5,6]");
        // a drawn id enters: after observe(5) on depth 2 the window is [6, 5]
        w.observe(5);
        assert!(w.dry_max_repeats().is_empty(), "the window is [6,5]: no repeat reaches allowed 2");
        // a disabled sampler never touches its window
        let mut off = dry_with(&[]);
        off.dry_multiplier = 0.0;
        off.observe_prompt(&[5, 6, 5, 6]);
        off.observe(5);
        assert!(off.dry_win.is_empty(), "llama.cpp dry accept early-returns while disabled");
    }

    /// #85: the HOST MIRROR. The Z-algorithm's Z-box bookkeeping is where a
    /// port can silently diverge, so this test recomputes llama.cpp's
    /// `dry_max_token_repeat` the naive way - a direct O(n^2) LCP per shift,
    /// no lt/rt reuse - over randomized windows, knobs and breaker maps, and
    /// demands the same map the sampler computed. Agreement is evidence, not
    /// tautology: the two share only the definition.
    #[test]
    fn dry_matches_the_naive_llama_reference_over_random_windows() {
        let mut gen = Rng::new(20260920);
        let mut singles = std::collections::HashMap::<u32, Vec<Vec<u32>>>::new();
        singles.insert(3, vec![vec![]]);
        singles.insert(9, vec![vec![]]);
        let mut multi = singles.clone();
        multi.insert(5, vec![vec![6, 7]]); // "5,6,7" is a breaker too
        for case in 0..300 {
            let n = 4 + (gen.next_u64() % 30) as usize;
            let win: Vec<u32> = (0..n).map(|_| (gen.next_u64() % 6) as u32).collect();
            let allowed = 1 + (gen.next_u64() % 3) as i32;
            let base = 1.0 + gen.next_f64() as f32 * 1.5;
            let mut s = Sampler {
                dry_multiplier: 0.1 + gen.next_f64() as f32,
                dry_base: base,
                dry_allowed_length: allowed,
                dry_last_n: n + 5,
                dry_breakers: std::sync::Arc::new(match case % 3 {
                    0 => std::collections::HashMap::new(),
                    1 => singles.clone(),
                    _ => multi.clone(),
                }),
                ..Sampler::new(1)
            };
            s.observe_prompt(&win);
            // the naive reference: reversed window, plain LCP per shift,
            // breaker scan by hand, step 3 by hand
            let rev: Vec<u32> = win.iter().rev().copied().collect();
            let m = rev.len();
            let mut rep_limit = m as i32;
            'brk: for i in 0..m {
                if let Some(tails) = s.dry_breakers.get(&rev[i]) {
                    let mut longest = -1i32;
                    for tail in tails {
                        let sl = tail.len() as i32;
                        if sl > longest && sl <= i as i32
                            && (0..tail.len()).all(|o| rev[i - o - 1] == tail[o])
                        {
                            longest = sl;
                        }
                    }
                    if longest >= 0 {
                        rep_limit = i as i32 - longest;
                        break 'brk;
                    }
                }
            }
            let mut expect: std::collections::HashMap<u32, i32> = Default::default();
            if rep_limit >= allowed {
                let mut rc = vec![0i32; m];
                for k in 1..m {
                    let mut z = 0usize;
                    while z + k < m && rev[z] == rev[z + k] {
                        z += 1;
                    }
                    rc[m - 1 - k] = (z as i32).min(rep_limit);
                }
                for i in 0..m - 1 {
                    if rc[i] >= allowed {
                        let tok = rev[m - 2 - i];
                        let e = expect.entry(tok).or_insert(i32::MIN);
                        if *e < rc[i] {
                            *e = rc[i];
                        }
                    }
                }
            }
            assert_eq!(
                s.dry_max_repeats(),
                expect,
                "case {case}: window {win:?} allowed {allowed} base {base:.3}"
            );
        }
    }

    /// #85: the breaker map derivation is llama.cpp's
    /// `get_overlapping_token_sequences` over a tiny fake vocab - tokens that
    /// CONTAIN a breaker become single-token breakers, tokens that BEGIN one
    /// map to the tokenization of the rest, deduped per (head, tail).
    #[test]
    fn dry_breaker_map_follows_get_overlapping_token_sequences() {
        // fake vocab: id -> text. The map is built over ALL FOUR default
        // breakers at once, as the reference accumulates one multimap.
        let vocab = [
            "",        // 0: nothing
            "\n",      // 1: contains the breaker -> single-token
            "a\nb",    // 2: contains it inside -> single-token too
            ":",       // 3
            "xx\"yy",  // 4
            "zz*",     // 5
            "abc",     // 6: no breaker char at all
            "a",       // 7: begins nothing
            "\rx",     // 8: contains none of the four
        ];
        let text_of = |id: u32| vocab.get(id as usize).map(|s| s.to_string());
        // "encode" the unmatched rest: map each char to its byte as an id, so
        // tests can predict the tails
        let encode = |s: &str| Some(s.bytes().map(|b| b as u32 + 1000).collect::<Vec<u32>>());
        let m = dry_breaker_map(vocab.len(), text_of, encode);
        // single-token breakers, empty tails
        for id in [1u32, 2, 3, 4, 5] {
            assert!(m.get(&id).map(|t| t.iter().any(|x| x.is_empty())).unwrap_or(false), "id {id}");
        }
        assert!(!m.contains_key(&0) && !m.contains_key(&6) && !m.contains_key(&7) && !m.contains_key(&8));
        // the prefix branch never fires for a single-character breaker: a
        // token either contains the character or does not begin it, so every
        // tail the four defaults produce is empty
        for tails in m.values() {
            assert!(tails.iter().all(|t| t.is_empty()), "single-char breakers have empty tails only");
        }
    }

    // ---- #92: the optional tier ----

    /// #92: top-nσ keeps exactly `{logit >= max - n*std}` over the penalized
    /// row - no softmax, no sort - and the comparison is strict on the LOW
    /// side only, so a candidate exactly at the threshold survives.
    #[test]
    fn top_n_sigma_keeps_exactly_max_minus_n_std() {
        // [3.0, 1.0, -1.0, -3.0]: mean 0, var 5, std sqrt(5) ~ 2.2360680
        let mk = |n: f32| {
            let mut s = Sampler { top_n_sigma: n, temperature: 1.0, top_p: 1.0, top_k: 4,
                presence_penalty: 0.0, ..Sampler::new(7) };
            s.observe_prompt(&[]);
            let logits = [3.0f32, 1.0, -1.0, -3.0];
            let mut seen = std::collections::HashSet::new();
            for _ in 0..600 {
                let t = s.sample(&logits);
                seen.insert(t);
            }
            seen
        };
        assert_eq!(mk(1.0), [0, 1].into_iter().collect(), "thr = 3 - 2.236 ~ 0.764 keeps 3 and 1");
        assert_eq!(mk(2.0), [0, 1, 2].into_iter().collect(), "thr ~ -1.472 keeps three");
        assert_eq!(mk(10.0), [0, 1, 2, 3].into_iter().collect(), "a wide n keeps the row");
        // the boundary: with all logits equal, std = 0 and thr = max, and
        // `v < thr` is false AT the max - nothing is masked
        let mut b = Sampler { top_n_sigma: 5.0, temperature: 1.0, top_p: 1.0, top_k: 3, presence_penalty: 0.0, ..Sampler::new(7) };
        let mut seen = std::collections::HashSet::new();
        for _ in 0..300 {
            seen.insert(b.sample(&[2.0f32, 2.0, 2.0]));
        }
        assert_eq!(seen, [0, 1, 2].into_iter().collect(), "std 0: the at-threshold maximum survives");
        // and it composes with the penalties: mean and std are computed over
        // the PENALIZED row (llama.cpp chain order), so a windowed repeat
        // penalty that drags the top token under the line costs it its place.
        // repeat 40 -> 3.0/40 = 0.075; mean -0.73125, std ~ 1.4889,
        // thr = 1.0 - 0.5*std ~ 0.2556 > 0.075 -> only token 1 survives
        let mut c = Sampler { top_n_sigma: 0.5, temperature: 1.0, top_p: 1.0, top_k: 4,
            repeat_penalty: 40.0, presence_penalty: 0.0, ..Sampler::new(7) };
        c.observe_prompt(&[0]); // token 0 is in the window
        let mut seen = std::collections::HashSet::new();
        for _ in 0..600 {
            seen.insert(c.sample(&[3.0f32, 1.0, -1.0, -3.0]));
        }
        assert_eq!(seen, [1].into_iter().collect(),
            "the mean/std are computed over the PENALIZED row, llama.cpp's chain order");
    }

    /// #92: typical_p keeps the LOCALLY typical set - smallest `|-ln p - H|`
    /// until the kept mass exceeds the threshold - which can rank the most
    /// probable token BEHIND a more typical one. p ~ [0.4, 0.35, 0.25]:
    /// H ~ 1.0805, |diffs| ~ [0.164, 0.031, 0.306] -> order 1, 0, 2.
    #[test]
    fn typical_p_keeps_the_locally_typical_set() {
        let logits = [(-0.9163f32), (-1.0498), (-1.3863)]; // ln of .4/.35/.25
        let mk = |p: f32| {
            let mut s = Sampler { typical_p: p, temperature: 1.0, top_p: 1.0, top_k: 3,
                presence_penalty: 0.0, ..Sampler::new(11) };
            let mut seen = std::collections::HashSet::new();
            for _ in 0..600 {
                seen.insert(s.sample(&logits));
            }
            seen
        };
        // mass 0.5: the typical order is 1 (0.35), then 0 (0.4) -> cum 0.75:
        // keep {1, 0}, token 2 is out
        assert_eq!(mk(0.5), [0, 1].into_iter().collect(), "the atypical tail drops");
        // a tight 0.1 keeps ONLY the most typical token - not the argmax:
        // typical sampling's signature
        assert_eq!(mk(0.1), [1].into_iter().collect(), "0.35 alone crosses 0.1, and it is token 1");
        // disabled (>= 1.0) is the old sampler exactly
        assert_eq!(mk(1.0), [0, 1, 2].into_iter().collect());
        // greedy never sees it: no candidate list exists at temperature 0
        let mut g = Sampler { typical_p: 0.1, temperature: 0.0, top_p: 1.0, top_k: 3,
            presence_penalty: 0.0, ..Sampler::new(11) };
        assert_eq!(g.sample(&logits), 0, "greedy is the plain penalized argmax, typical included");
    }

    /// #92: XTC removes the leading run of `p >= threshold` (renormalized
    /// over the kept set) with probability `xtc_probability`, keeping the
    /// run's LAST member and at least `min_keep` entries - and it draws its
    /// chance from its OWN rng stream, so the main xorshift64* state after a
    /// token is exactly what a no-XTC sampler's is.
    #[test]
    fn xtc_removes_the_leading_run_with_its_own_rng() {
        let logits = [3.0f32, 2.9, 0.0, 0.0]; // p ~ [0.499, 0.451, 0.025, 0.025]
        let mk = |prob: f32, thr: f32, min_keep: usize| {
            let mut s = Sampler { xtc_probability: prob, xtc_threshold: thr, xtc_min_keep: min_keep,
                temperature: 1.0, top_p: 1.0, top_k: 4, presence_penalty: 0.0, ..Sampler::new(13) };
            let mut seen = std::collections::HashSet::new();
            for _ in 0..600 {
                seen.insert(s.sample(&logits));
            }
            seen
        };
        // probability 1, threshold 0.4: the leading run is {0, 1}, the cut
        // keeps index 1 onward - the consensus token 0 is NEVER drawn
        assert_eq!(mk(1.0, 0.4, 1), [1, 2, 3].into_iter().collect(),
            "the top choice is hard-removed, the second leads");
        // min_keep 4 blocks the cut (3 would remain)
        assert_eq!(mk(1.0, 0.4, 4), [0, 1, 2, 3].into_iter().collect(), "min_keep holds the set");
        // a threshold above 0.5 disables XTC outright, llama.cpp's own rule
        assert_eq!(mk(1.0, 0.6, 1), [0, 1, 2, 3].into_iter().collect(), "threshold > 0.5 is off");
        // probability 0 is the old sampler exactly
        assert_eq!(mk(0.0, 0.4, 1), [0, 1, 2, 3].into_iter().collect());
        // THE STREAM PIN: an armed, firing XTC and a disabled one consume the
        // SAME main-rng state per token - XTC's chance came from its own
        let mut armed = Sampler { xtc_probability: 1.0, xtc_threshold: 0.4, xtc_min_keep: 1,
            temperature: 1.0, top_p: 1.0, top_k: 4, presence_penalty: 0.0, ..Sampler::new(13) };
        let mut plain = Sampler::new(13);
        plain.temperature = 1.0;
        plain.top_p = 1.0;
        plain.top_k = 4;
        plain.presence_penalty = 0.0;
        for _ in 0..50 {
            let _ = armed.sample(&logits);
            let _ = plain.sample(&logits);
            assert_eq!(armed.rng.state(), plain.rng.state(),
                "XTC's chance must never advance the main draw stream");
        }
        // and the disabled-by-rule forms do not even advance XTC's own stream
        let mut thr_off = Sampler { xtc_probability: 1.0, xtc_threshold: 0.6, xtc_min_keep: 1,
            temperature: 1.0, top_p: 1.0, top_k: 4, presence_penalty: 0.0, ..Sampler::new(13) };
        let x0 = thr_off.xtc_rng.state();
        let _ = thr_off.sample(&logits);
        assert_eq!(thr_off.xtc_rng.state(), x0, "llama.cpp returns before the chance draw");
        // reproducibility: same seed, same everything -> same sequence
        let seq = |seed: u64| {
            let mut s = Sampler { xtc_probability: 1.0, xtc_threshold: 0.4, xtc_min_keep: 1,
                temperature: 1.0, top_p: 1.0, top_k: 4, presence_penalty: 0.0, ..Sampler::new(seed) };
            (0..20).map(|_| s.sample(&logits)).collect::<Vec<_>>()
        };
        assert_eq!(seq(99), seq(99));
    }

    /// #92: mirostat v2 truncates the temperature softmax at
    /// `surprise = -log2(p) > mu`, draws from the renormalized remainder,
    /// and moves `mu` by `-eta*(surprise - tau)` - the state that makes it
    /// per-request (reset with the seed). It also REPLACES the usual tail:
    /// top_k does not bound it, exactly as llama.cpp's chain build drops
    /// top-k/top-p in the mirostat branch.
    #[test]
    fn mirostat_v2_truncates_by_surprise_and_updates_mu() {
        let logits = [3.0f32, 2.0, 1.0, 0.0]; // p ~ [.644, .237, .087, .032], -log2 ~ [.64, 2.08, 3.52, 4.96]
        // tau 0.5 -> mu 1.0: every token but the first has surprise > mu, so
        // the truncation keeps ONE candidate and the draw is deterministic
        let mut s = Sampler { mirostat: 2, mirostat_tau: 0.5, mirostat_eta: 0.1,
            temperature: 1.0, top_k: 1, top_p: 1.0, presence_penalty: 0.0, ..Sampler::new(3) };
        s.reset_mirostat();
        assert_eq!(s.miro_mu, 1.0);
        assert_eq!(s.sample(&logits), 0, "surprise 0.64 <= mu 1.0 survives alone; top_k 1 did not matter");
        // mu moved by -eta*(surprise - tau) = -0.1*(0 - 0.5) = +0.05
        assert!((s.miro_mu - 1.05f32).abs() < 1e-6, "mu after one draw: {}", s.miro_mu);
        // tau 5 -> mu 10: nothing truncated, ALL FOUR stay reachable - over
        // the FULL row, not the top_k slice (top_k is 1 above and was ignored)
        let mut w = Sampler { mirostat: 2, mirostat_tau: 5.0, mirostat_eta: 0.1,
            temperature: 1.0, top_k: 1, top_p: 1.0, presence_penalty: 0.0, ..Sampler::new(3) };
        let mut seen = std::collections::HashSet::new();
        for _ in 0..600 {
            seen.insert(w.sample(&logits));
        }
        assert_eq!(seen, [0, 1, 2, 3].into_iter().collect(),
            "mirostat sees the full row; mu 10 truncates nothing here");
        // the state carries: after many low-surprise draws mu drifts UP toward
        // 2*tau and above, and the reachability set only grows
        let mu_now = w.miro_mu;
        assert!(mu_now > 5.0, "surprises under tau 5 raise mu, got {mu_now}");
        // reset is the per-request door: tau changed, mu follows 2*tau
        let mut r = Sampler::new(0);
        r.mirostat_tau = 3.0;
        r.reset_mirostat();
        assert_eq!(r.miro_mu, 6.0);
        // and mirostat 0 (the default) never takes the branch: the golden of
        // the neutral sampler is already pinned above
    }

    /// #85/#92: `host_route` is the whole routing truth table - DRY or any
    /// armed #92 knob says host, everything else (the #83/#84 profile) says
    /// device.
    #[test]
    fn host_route_is_dry_or_the_tier_and_nothing_else() {
        let mut s = Sampler::new(0);
        assert!(!s.host_route());
        s.min_p = 0.01;
        s.repeat_penalty = 1.05;
        s.frequency_penalty = 0.2;
        assert!(!s.host_route(), "#83/#84 stay on the device sampler");
        s.dry_multiplier = 0.8;
        assert!(s.host_route());
        s.dry_multiplier = 0.0;
        assert!(!s.host_route());
        s.top_n_sigma = 1.5;
        assert!(s.host_route());
        s.top_n_sigma = 0.0;
        s.typical_p = 0.9;
        assert!(s.host_route());
        s.typical_p = 1.0;
        s.xtc_probability = 0.3;
        assert!(s.host_route());
        s.xtc_probability = 0.0;
        s.mirostat = 2;
        assert!(s.host_route());
    }

    /// #93: `rebook_plan` against a host model of the device accept,
    /// over random answers, with and without the window, eviction included.
    #[test]
    fn rebook_turns_the_device_accept_of_the_drawn_id_into_the_accept_of_the_kept_one() {
        // a host model of `sample_k`'s accept (kernels.rs, after the draw): presence bit,
        // then, window armed, evict-when-full + ring write + count
        #[derive(Clone, PartialEq, Debug)]
        struct Dev {
            mask: Vec<u8>,
            counts: Vec<u16>,
            ring: Vec<i32>,
        }
        fn accept(d: &mut Dev, tok: usize, lastn: usize) {
            d.mask[tok] = 1;
            if lastn > 0 {
                let (head, mut fill) = (d.ring[0] as usize, d.ring[1] as usize);
                if fill >= lastn {
                    let old = d.ring[2 + head] as usize;
                    d.counts[old] -= 1;
                } else {
                    fill += 1;
                    d.ring[1] = fill as i32;
                }
                d.ring[2 + head] = tok as i32;
                d.counts[tok] += 1;
                d.ring[0] = ((head + 1) % lastn) as i32;
            }
        }
        fn apply(d: &mut Dev, w: &[(RebookBuf, usize, Vec<u8>)]) {
            for (buf, off, b) in w {
                match buf {
                    RebookBuf::Mask => d.mask[*off] = b[0],
                    RebookBuf::Counts => d.counts[off / 2] = u16::from_le_bytes([b[0], b[1]]),
                    RebookBuf::Ring => d.ring[off / 4] = i32::from_le_bytes([b[0], b[1], b[2], b[3]]),
                }
            }
        }
        let v = 16;
        let mut rng = Rng::new(5);
        for lastn in [0usize, 1, 3, 8] {
            for trial in 0..200 {
                let mut d = Dev { mask: vec![0; v], counts: vec![0; v], ring: vec![0; 2 + 8] };
                let mut out = Vec::new();
                for _ in 0..(rng.next_u64() % 12) {
                    let t = (rng.next_u64() % v as u64) as usize;
                    accept(&mut d, t, lastn);
                    out.push(t);
                }
                let drawn = (rng.next_u64() % v as u64) as usize;
                let kept = (rng.next_u64() % v as u64) as usize;
                let mut want = d.clone();
                accept(&mut want, kept, lastn);
                let seen = out.contains(&drawn);
                accept(&mut d, drawn, lastn);
                let w = rebook_plan(drawn, kept, seen, lastn, d.ring[0], d.counts[drawn], d.counts[kept]);
                apply(&mut d, &w);
                assert_eq!(d, want, "lastn {lastn} trial {trial} drawn {drawn} kept {kept} out {out:?}");
            }
        }
    }

    /// #91: `pos_logprobs` is log_softmax of the RAW row - checked against an
    /// f64 reference on a synthetic row, the chosen id may sit outside the top
    /// n, the probabilities of the full row sum to 1, and n is capped at 20.
    #[test]
    fn pos_logprobs_is_the_log_softmax_of_the_raw_row() {
        let row: Vec<f32> = (0..1000).map(|i| ((i * 7919) % 1000) as f32 * 0.013 - 6.0).collect();
        let m = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max) as f64;
        let lnz = m + row.iter().map(|&l| (l as f64 - m).exp()).sum::<f64>().ln();
        let mut order: Vec<usize> = (0..row.len()).collect();
        order.sort_by(|&a, &b| row[b].partial_cmp(&row[a]).unwrap().then(a.cmp(&b)));
        let p = pos_logprobs(&row, 17, 5);
        assert_eq!(p.chosen, 17);
        assert!((p.chosen_lp - (row[17] as f64 - lnz)).abs() < 1e-9);
        assert_eq!(p.top.iter().map(|t| t.0).collect::<Vec<_>>(), order[..5].to_vec());
        for (i, lp) in &p.top {
            assert!((lp - (row[*i] as f64 - lnz)).abs() < 1e-9);
        }
        let all = pos_logprobs(&row, 0, 20);
        assert_eq!(all.top.len(), 20, "n is capped at MAX_TOP_LOGPROBS");
        let total: f64 = (0..row.len()).map(|i| pos_logprobs(&row, i, 0).chosen_lp.exp()).sum();
        assert!((total - 1.0).abs() < 1e-9, "the row's probabilities sum to {total}");
        assert!(pos_logprobs(&row, 3, 0).top.is_empty(), "top_logprobs 0 is an empty list");
    }

    /// #91: the near-tie the logprobs exist to measure. An EXACT tie orders by
    /// index ascending (margin 0, both entries equal), NaN never enters the
    /// list or Z, `-inf` is probability 0 and reports as the OpenAI floor.
    #[test]
    fn pos_logprobs_orders_ties_by_index_and_survives_non_finite_logits() {
        let mut row = vec![0.0f32; 64];
        row[40] = 5.0;
        row[9] = 5.0; // exact tie with 40: 9 comes first
        row[30] = 4.999;
        row[2] = f32::NAN;
        row[3] = f32::NEG_INFINITY;
        let p = pos_logprobs(&row, 40, 4);
        assert_eq!(p.top.iter().map(|t| t.0).collect::<Vec<_>>(), vec![9, 40, 30, 0]);
        assert_eq!(p.top[0].1, p.top[1].1, "a tie is a tie");
        assert_eq!(p.chosen_lp, p.top[1].1);
        assert!(p.top[1].1 - p.top[2].1 > 0.0 && p.top[1].1 - p.top[2].1 < 2e-3);
        // Z over the finite entries only: 2 e^5 + e^4.999 + 59 e^0 (NaN skipped, -inf = 0)
        let z = 2.0 * 5f64.exp() + (4.999f32 as f64).exp() + 59.0;
        assert!((p.chosen_lp - (5.0 - z.ln())).abs() < 1e-9);
        assert_eq!(pos_logprobs(&row, 3, 0).chosen_lp, LOGPROB_FLOOR);
        assert_eq!(pos_logprobs(&row, 2, 0).chosen_lp, LOGPROB_FLOOR);
        assert_eq!(pos_logprobs(&row, 999, 0).chosen_lp, LOGPROB_FLOOR, "an id off the row");
    }
}
