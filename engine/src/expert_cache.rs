//! #175 - the dynamic expert cache policy (root #169, plan step 16, measurement book F lever 3).
//!
//! An EXCLUSIVE two-tier cache per MoE layer (optionally one global pool over all layers):
//! every expert is in exactly one of
//!
//! - **VRAM** (capacity `vram` per layer): the hot slots of `Residency`,
//! - **pinned** (capacity `pinned` per layer): the pinned host tier,
//! - **NVMe**: the miss tier - everything the two cached tiers do not hold.
//!
//! A pinned hit is promoted into VRAM and the VRAM victim is demoted into the pinned slot the
//! promoted expert left - the exact three-way exchange `Residency::swap_stream_a` / `swap_in`
//! already execute. #188: [`ExpertCache::set_pinned_stays`] turns that promotion off (a pinned
//! hit is read in place, zero-copy or by the CPU lane); NVMe misses are unaffected. An NVMe miss is admitted into VRAM (or into pinned when `vram` is 0); the
//! demoted VRAM victim goes to pinned and the pinned victim drops to NVMe.
//!
//! Policies:
//! - `Lru`: recency (Eliseev & Mazur, arXiv:2312.17238 section 3.1). Exclusive two-tier LRU is
//!   ONE LRU stack split at `vram` (Mattson et al. 1970): its NVMe misses are those of a single
//!   LRU of `vram + pinned` slots and its VRAM misses those of a single LRU of `vram` slots, which
//!   is what the cross-check against `tools/glm_tier_sim.py` `lru_reads` tests.
//! - `Clock { admit }`: second chance (Corbato 1968) per tier ring; reference bit set on use and
//!   on insert, cleared by the hand; a demoted expert enters the pinned ring with a clear bit.
//!   `admit` (1..=64) bounds the NVMe admissions per layer per tick; misses beyond it are served
//!   from NVMe without entering the cache (bypass).
//! - `Lfu { decay }`: frequency with exponential decay (LRFU, Lee et al. 2001): every tick
//!   multiplies the layer's scores by `decay` (1.0 = plain LFU), an access adds its weight;
//!   VRAM holds the highest scores, pinned the next; ties go to the more recent access.
//!
//! The cache only decides. `Residency::plan_swaps` turns its VRAM set into the existing
//! `(slot, evicted, incoming)` triples; NVMe reads are not executed here (no NVMe tier in
//! `Residency` yet, #149). Counters `[vram, pinned, nvme]` per layer count the tier each access
//! was served from. Pure host code: no CUDA, unit tested without a GPU.

use std::cmp::Ordering;

/// the container family the cache is on for by default (`Cnq::family`)
pub const GLM5_NEXT_FAMILY: &str = "Glm5Next";
/// the opt-in / override variable (`off`, `lru`, `clock`, `clock:N`, `lfu`, `lfu:D`)
pub const ENV: &str = "CROW_EXPERT_CACHE";
/// the largest CLOCK admission limit per layer per tick
pub const ADMIT_MAX: usize = 64;
/// LFU decay of a bare `lfu`: the engine's existing selection-window decay (`CROW_ADAPT_DECAY`
/// default); not measured for this policy
pub const LFU_DECAY_DEFAULT: f64 = 0.5;

const EMPTY: u32 = u32::MAX;

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Policy {
    Lru,
    Clock { admit: Option<usize> },
    Lfu { decay: f64 },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Scope {
    /// one pool per layer, `vram` + `pinned` slots each (what `Residency` can execute)
    PerLayer,
    /// one pool over every layer with `layers x vram` + `layers x pinned` slots
    /// (simulation and counters only)
    Global,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Tier {
    Vram = 0,
    Pinned = 1,
    Nvme = 2,
}

/// one exclusive pool (a layer, or every layer under `Scope::Global`)
struct Pool {
    tier: Vec<Tier>,
    stamp: Vec<u64>,
    score: Vec<f64>,
    refb: Vec<bool>,
    /// CLOCK: the slots of the VRAM [0] and pinned [1] rings (`EMPTY` = free)
    ring: [Vec<u32>; 2],
    hand: [usize; 2],
    /// CLOCK: a cached key's slot in its tier's ring
    slot: Vec<usize>,
    cap: [usize; 2],
    len: [usize; 2],
    /// #188: a pinned hit stays in pinned (no promotion into VRAM)
    pin_stay: bool,
}

impl Pool {
    fn new(keys: usize, vram: usize, pinned: usize) -> Pool {
        Pool {
            tier: vec![Tier::Nvme; keys],
            stamp: vec![0; keys],
            score: vec![0.0; keys],
            refb: vec![false; keys],
            ring: [vec![EMPTY; vram], vec![EMPTY; pinned]],
            hand: [0, 0],
            slot: vec![usize::MAX; keys],
            cap: [vram, pinned],
            len: [0, 0],
            pin_stay: false,
        }
    }

    fn set(&mut self, k: usize, t: Tier) {
        let old = self.tier[k];
        if old != Tier::Nvme {
            self.len[old as usize] -= 1;
        }
        if t != Tier::Nvme {
            self.len[t as usize] += 1;
        }
        self.tier[k] = t;
    }

    /// what the policy keeps longer is larger
    fn prio(&self, p: Policy, k: usize) -> (f64, u64) {
        match p {
            Policy::Lru => (0.0, self.stamp[k]),
            Policy::Lfu { .. } => (self.score[k], self.stamp[k]),
            Policy::Clock { .. } => {
                let t = self.tier[k];
                if t == Tier::Nvme {
                    return (0.0, 0);
                }
                let (i, cap) = (t as usize, self.cap[t as usize]);
                // ref-clear experts in hand order go first: distance from the hand
                let dist = (self.slot[k] + cap - self.hand[i]) % cap;
                (self.refb[k] as u8 as f64, dist as u64)
            }
        }
    }

    /// `Greater` = `a` is kept longer than `b`; equal priorities keep the lower id
    fn keep(&self, p: Policy, a: usize, b: usize) -> Ordering {
        let (pa, pb) = (self.prio(p, a), self.prio(p, b));
        pa.0.partial_cmp(&pb.0).unwrap_or(Ordering::Equal).then(pa.1.cmp(&pb.1)).then(b.cmp(&a))
    }

    fn min_in(&self, p: Policy, t: Tier) -> Option<usize> {
        (0..self.tier.len()).filter(|&k| self.tier[k] == t).min_by(|&a, &b| self.keep(p, a, b))
    }

    /// LRU / LFU placement of the just-accessed key `k`
    fn ranked(&mut self, p: Policy, k: usize) {
        let t = self.tier[k];
        // #188 no-promote: a pinned hit keeps its slot (its recency / score is already updated)
        if t == Tier::Vram || (t == Tier::Pinned && self.pin_stay) {
            return;
        }
        if self.cap[0] > 0 {
            if self.len[0] < self.cap[0] {
                self.set(k, Tier::Vram);
                return;
            }
            let v = self.min_in(p, Tier::Vram).expect("a full VRAM tier has a victim");
            if self.keep(p, k, v) == Ordering::Greater {
                self.set(k, Tier::Vram);
                self.offer_pinned(p, v);
                return;
            }
        }
        if t == Tier::Nvme {
            self.offer_pinned(p, k);
        }
    }

    /// `k` (an NVMe key, or a VRAM victim) competes for a pinned slot; the loser is on NVMe
    fn offer_pinned(&mut self, p: Policy, k: usize) {
        if self.cap[1] == 0 {
            self.set(k, Tier::Nvme);
        } else if self.len[1] < self.cap[1] {
            self.set(k, Tier::Pinned);
        } else {
            let u = self.min_in(p, Tier::Pinned).expect("a full pinned tier has a victim");
            if self.keep(p, k, u) == Ordering::Greater {
                self.set(u, Tier::Nvme);
                self.set(k, Tier::Pinned);
            } else {
                self.set(k, Tier::Nvme);
            }
        }
    }

    /// CLOCK: the slot of ring `i` the hand stops at (a free slot or a clear reference bit)
    fn sweep(&mut self, i: usize) -> usize {
        let cap = self.cap[i];
        loop {
            let h = self.hand[i];
            let k = self.ring[i][h];
            if k == EMPTY || !self.refb[k as usize] {
                return h;
            }
            self.refb[k as usize] = false;
            self.hand[i] = (h + 1) % cap;
        }
    }

    /// CLOCK: `k` into slot `s` of ring `i`; the hand moves past it; returns the occupant
    fn put(&mut self, i: usize, s: usize, k: usize, refb: bool) -> Option<usize> {
        let old = self.ring[i][s];
        self.ring[i][s] = k as u32;
        self.slot[k] = s;
        self.refb[k] = refb;
        self.hand[i] = (s + 1) % self.cap[i];
        (old != EMPTY).then_some(old as usize)
    }

    /// CLOCK access of `k`; `admit` false = an NVMe miss bypasses the cache
    fn clock(&mut self, k: usize, admit: bool) {
        match self.tier[k] {
            Tier::Vram => self.refb[k] = true,
            Tier::Pinned => {
                if self.cap[0] == 0 || self.pin_stay {
                    self.refb[k] = true;
                    return;
                }
                let ps = self.slot[k];
                let s = self.sweep(0);
                let v = self.put(0, s, k, true);
                self.set(k, Tier::Vram);
                match v {
                    // the exact three-way exchange: the victim takes the promoted one's pinned slot
                    Some(v) => {
                        self.ring[1][ps] = v as u32;
                        self.slot[v] = ps;
                        self.refb[v] = false;
                        self.set(v, Tier::Pinned);
                    }
                    None => self.ring[1][ps] = EMPTY,
                }
            }
            Tier::Nvme => {
                if !admit {
                    return;
                }
                if self.cap[0] > 0 {
                    let s = self.sweep(0);
                    let v = self.put(0, s, k, true);
                    self.set(k, Tier::Vram);
                    if let Some(v) = v {
                        self.clock_demote(v);
                    }
                } else if self.cap[1] > 0 {
                    let q = self.sweep(1);
                    let u = self.put(1, q, k, true);
                    self.set(k, Tier::Pinned);
                    if let Some(u) = u {
                        self.set(u, Tier::Nvme);
                    }
                }
            }
        }
    }

    fn clock_demote(&mut self, v: usize) {
        if self.cap[1] == 0 {
            self.set(v, Tier::Nvme);
            return;
        }
        let q = self.sweep(1);
        let u = self.put(1, q, v, false);
        self.set(v, Tier::Pinned);
        if let Some(u) = u {
            self.set(u, Tier::Nvme);
        }
    }
}

/// The cache: policy, scope, per-layer capacities, the pools and the counters.
pub struct ExpertCache {
    pub policy: Policy,
    pub scope: Scope,
    pub layers: usize,
    pub experts: usize,
    /// VRAM slots per layer
    pub vram: usize,
    /// pinned slots per layer
    pub pinned: usize,
    pools: Vec<Pool>,
    now: u64,
    counters: Vec<[u64; 3]>,
    admitted: Vec<usize>,
    /// `observe_counts`: the cumulative counts of the previous call, per layer
    last: Vec<Vec<u64>>,
}

impl ExpertCache {
    pub fn new(policy: Policy, scope: Scope, layers: usize, experts: usize, vram: usize, pinned: usize) -> Result<ExpertCache, String> {
        if layers == 0 || experts == 0 {
            return Err(format!("expert cache: {layers} layers x {experts} experts is no MoE shape"));
        }
        match policy {
            Policy::Clock { admit: Some(a) } if a == 0 || a > ADMIT_MAX => {
                return Err(format!("expert cache: CLOCK admission limit {a} outside 1..={ADMIT_MAX}"));
            }
            Policy::Lfu { decay } if !(decay > 0.0 && decay <= 1.0) => {
                return Err(format!("expert cache: LFU decay {decay} outside (0, 1]"));
            }
            _ => {}
        }
        let pools = match scope {
            Scope::PerLayer => (0..layers).map(|_| Pool::new(experts, vram, pinned)).collect(),
            Scope::Global => vec![Pool::new(layers * experts, layers * vram, layers * pinned)],
        };
        Ok(ExpertCache {
            policy,
            scope,
            layers,
            experts,
            vram,
            pinned,
            pools,
            now: 0,
            counters: vec![[0; 3]; layers],
            admitted: vec![0; layers],
            last: vec![Vec::new(); layers],
        })
    }

    fn loc(&self, l: usize, e: u32) -> (usize, usize) {
        assert!(l < self.layers && (e as usize) < self.experts, "expert cache: layer {l} expert {e} outside the shape");
        match self.scope {
            Scope::PerLayer => (l, e as usize),
            Scope::Global => (0, l * self.experts + e as usize),
        }
    }

    /// #188 (`CROW_GLM_PINNED=zerocopy`, `CROW_GLM_CPU_LANE=1`): with `on`, a pinned hit stays
    /// in pinned (LRU / LFU: its recency and score are updated in place; CLOCK: its reference bit
    /// is set) instead of being promoted into VRAM; an NVMe miss keeps the policy's rule (into
    /// VRAM, the VRAM victim to pinned, the pinned victim to NVMe). Off (the default) is the
    /// exchange rule of the module doc, unchanged.
    pub fn set_pinned_stays(&mut self, on: bool) {
        for p in &mut self.pools {
            p.pin_stay = on;
        }
    }

    /// whether a pinned hit stays in pinned ([`ExpertCache::set_pinned_stays`])
    pub fn pinned_stays(&self) -> bool {
        self.pools[0].pin_stay
    }

    pub fn tier(&self, l: usize, e: u32) -> Tier {
        let (p, k) = self.loc(l, e);
        self.pools[p].tier[k]
    }

    /// the experts of layer `l` the cache wants in VRAM, ascending id
    pub fn vram_set(&self, l: usize) -> Vec<u32> {
        (0..self.experts as u32).filter(|&e| self.tier(l, e) == Tier::Vram).collect()
    }

    /// `Greater` = expert `a` of layer `l` is kept longer than `b`
    pub fn keep_order(&self, l: usize, a: u32, b: u32) -> Ordering {
        let ((p, ka), (_, kb)) = (self.loc(l, a), self.loc(l, b));
        self.pools[p].keep(self.policy, ka, kb)
    }

    /// `[vram, pinned, nvme]` per layer: the tier each access was served from
    pub fn counters(&self) -> &[[u64; 3]] {
        &self.counters
    }

    /// Place a starting state into layer `l` (empty so far): `vram` ids hottest first, then
    /// `pinned`. Recency follows the order given (the first id is the most recent).
    pub fn seed(&mut self, l: usize, vram: &[u32], pinned: &[u32]) -> Result<(), String> {
        let total = (vram.len() + pinned.len()) as u64;
        let base = self.now;
        for (t, ids) in [(Tier::Vram, vram), (Tier::Pinned, pinned)] {
            for (i, &e) in ids.iter().enumerate() {
                let (p, k) = self.loc(l, e);
                let pool = &mut self.pools[p];
                if pool.tier[k] != Tier::Nvme {
                    return Err(format!("expert cache seed: layer {l} expert {e} placed twice"));
                }
                if pool.len[t as usize] >= pool.cap[t as usize] {
                    return Err(format!("expert cache seed: layer {l} has more {t:?} ids than {} slots", pool.cap[t as usize]));
                }
                let rank = if t == Tier::Vram { i } else { vram.len() + i } as u64;
                pool.stamp[k] = base + total - rank;
                if let Policy::Clock { .. } = self.policy {
                    let s = pool.ring[t as usize].iter().position(|&x| x == EMPTY).expect("a free slot under the cap");
                    pool.ring[t as usize][s] = k as u32;
                    pool.slot[k] = s;
                    pool.refb[k] = false;
                }
                pool.set(k, t);
            }
        }
        self.now = base + total;
        Ok(())
    }

    /// one tick of layer `l`: decay (LFU) and a fresh admission budget (CLOCK)
    fn tick(&mut self, l: usize) {
        self.admitted[l] = 0;
        if let Policy::Lfu { decay } = self.policy {
            let (p, r) = match self.scope {
                Scope::PerLayer => (l, 0..self.experts),
                Scope::Global => (0, l * self.experts..(l + 1) * self.experts),
            };
            for s in &mut self.pools[p].score[r] {
                *s *= decay;
            }
        }
    }

    /// one access; returns the tier it was served from
    fn access(&mut self, l: usize, e: u32, weight: u64) -> Tier {
        let (p, k) = self.loc(l, e);
        self.now += 1;
        let now = self.now;
        let policy = self.policy;
        let t = self.pools[p].tier[k];
        self.counters[l][t as usize] += 1;
        let pool = &mut self.pools[p];
        pool.stamp[k] = now;
        match policy {
            Policy::Lru => pool.ranked(policy, k),
            Policy::Lfu { .. } => {
                pool.score[k] += weight as f64;
                pool.ranked(policy, k);
            }
            Policy::Clock { admit } => {
                let mut ok = true;
                if t == Tier::Nvme {
                    ok = admit.is_none_or(|a| self.admitted[l] < a);
                    if ok && pool.cap[0] + pool.cap[1] > 0 {
                        self.admitted[l] += 1;
                    }
                }
                pool.clock(k, ok);
            }
        }
        t
    }

    /// one token of layer `l`: the routed ids in the order given (the routing file's
    /// ascending order in `glm_tier_sim`)
    pub fn observe_token(&mut self, l: usize, ids: &[u32]) {
        self.tick(l);
        for &e in ids {
            self.access(l, e, 1);
        }
    }

    /// One tick of layer `l` from CUMULATIVE per-expert counts (`Gen::drain_sel_counts`): every
    /// expert whose count rose since the previous call is one access, weighted by the rise (LFU),
    /// in ascending order of the rise and then of the id - so the most selected ends most recent,
    /// and a one-token decode tick is the ascending-id order. A count below the previous one is a
    /// counter reset and counts from zero.
    pub fn observe_counts(&mut self, l: usize, c: &[u64]) {
        assert_eq!(c.len(), self.experts, "expert cache: one count per expert");
        let last = &mut self.last[l];
        if last.len() != c.len() {
            *last = vec![0; c.len()];
        }
        let mut touched: Vec<(u32, u64)> = c
            .iter()
            .zip(last.iter())
            .enumerate()
            .filter_map(|(e, (&now, &was))| {
                let d = if now >= was { now - was } else { now };
                (d > 0).then_some((e as u32, d))
            })
            .collect();
        last.copy_from_slice(c);
        touched.sort_by(|a, b| a.1.cmp(&b.1).then(a.0.cmp(&b.0)));
        self.tick(l);
        for (e, w) in touched {
            self.access(l, e, w);
        }
    }

    /// the exclusivity invariant: each expert has ONE tier field (so one tier by construction);
    /// this checks that no tier is over its capacity, that the occupancy counts agree with the
    /// tier fields, and that every CLOCK ring slot holds exactly the experts of its tier
    pub fn check_exclusive(&self) -> Result<(), String> {
        for (pi, p) in self.pools.iter().enumerate() {
            let mut n = [0usize; 3];
            for &t in &p.tier {
                n[t as usize] += 1;
            }
            for (i, &ni) in n.iter().enumerate().take(2) {
                if ni != p.len[i] || ni > p.cap[i] {
                    return Err(format!("pool {pi}: tier {i} holds {ni} (len {}, cap {})", p.len[i], p.cap[i]));
                }
            }
            if let Policy::Clock { .. } = self.policy {
                for (i, (ring, &ni)) in p.ring.iter().zip(n.iter()).enumerate() {
                    let occ = ring.iter().filter(|&&k| k != EMPTY).count();
                    if occ != ni {
                        return Err(format!("pool {pi}: ring {i} holds {occ} keys, tier {i} {ni}"));
                    }
                    for (s, &k) in ring.iter().enumerate() {
                        if k != EMPTY && (p.tier[k as usize] as usize != i || p.slot[k as usize] != s) {
                            return Err(format!("pool {pi}: ring {i} slot {s} holds key {k} of tier {:?}", p.tier[k as usize]));
                        }
                    }
                }
            }
        }
        Ok(())
    }
}

/// The policy a boot takes: `CROW_EXPERT_CACHE` unset (or empty) = LRU for the family
/// `Glm5Next` and off for every other family; `off` / `0` = off; `lru`, `clock`, `clock:N`
/// (admission limit 1..=64), `lfu` (decay 0.5), `lfu:D` (decay in (0, 1]) = that policy on any
/// family (the explicit opt-in). Refused by name with `CROW_ADAPT_WINDOW=1`: the planner then
/// gets a decayed window, not cumulative counts, so a rise no longer marks an access.
pub fn policy_for(family: &str, var: Option<&str>, window: bool) -> Result<Option<Policy>, String> {
    let p = match var.map(str::trim) {
        None | Some("") => (family == GLM5_NEXT_FAMILY).then_some(Policy::Lru),
        Some("off") | Some("0") => None,
        Some(s) => Some(parse_policy(s)?),
    };
    if p.is_some() && window {
        return Err(format!(
            "the expert cache ({ENV}, on for family {family}) reads cumulative selection counts, \
and CROW_ADAPT_WINDOW=1 passes a decayed window: set CROW_ADAPT_WINDOW=0 or {ENV}=off"
        ));
    }
    Ok(p)
}

fn parse_policy(s: &str) -> Result<Policy, String> {
    let bad = || format!("{ENV}={s:?} is not a cache policy; accepted: off, lru, clock, clock:N (N 1..={ADMIT_MAX}), lfu, lfu:D (D in (0, 1])");
    let (name, arg) = match s.split_once(':') {
        Some((n, a)) => (n, Some(a)),
        None => (s, None),
    };
    match (name, arg) {
        ("lru", None) => Ok(Policy::Lru),
        ("clock", None) => Ok(Policy::Clock { admit: None }),
        ("clock", Some(a)) => match a.parse::<usize>() {
            Ok(n) if (1..=ADMIT_MAX).contains(&n) => Ok(Policy::Clock { admit: Some(n) }),
            _ => Err(bad()),
        },
        ("lfu", None) => Ok(Policy::Lfu { decay: LFU_DECAY_DEFAULT }),
        ("lfu", Some(a)) => match a.parse::<f64>() {
            Ok(d) if d > 0.0 && d <= 1.0 => Ok(Policy::Lfu { decay: d }),
            _ => Err(bad()),
        },
        _ => Err(bad()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn one_layer(p: Policy, vram: usize, pinned: usize) -> ExpertCache {
        ExpertCache::new(p, Scope::PerLayer, 1, 16, vram, pinned).unwrap()
    }

    fn tiers(c: &ExpertCache, t: Tier) -> Vec<u32> {
        (0..c.experts as u32).filter(|&e| c.tier(0, e) == t).collect()
    }

    /// LRU, V 2 + P 1, the sequence 1 2 1 3 2 4 1 (one access per token). By hand on the
    /// stack split at V: 1 N, 2 N, 1 V (depth 1), 3 N, 2 P (depth 2), 4 N (1 falls out), 1 N
    #[test]
    fn lru_hand_sequence() {
        let mut c = one_layer(Policy::Lru, 2, 1);
        for e in [1, 2, 1, 3, 2, 4, 1] {
            c.observe_token(0, &[e]);
        }
        assert_eq!(c.counters()[0], [1, 1, 5]);
        assert_eq!((tiers(&c, Tier::Vram), tiers(&c, Tier::Pinned)), (vec![1, 4], vec![2]));
    }

    /// CLOCK, V 2 + P 1, the same sequence. By hand: 1 N and 2 N fill the VRAM ring (bits set);
    /// 1 V; 3 N: the hand clears both bits and evicts 1 to pinned; 2 V (bit set again, where LRU
    /// said pinned); 4 N evicts 2, whose demotion drops 1 to NVMe; 1 N evicts 3, 3 drops 2
    #[test]
    fn clock_hand_sequence() {
        let mut c = one_layer(Policy::Clock { admit: None }, 2, 1);
        for e in [1, 2, 1, 3, 2, 4, 1] {
            c.observe_token(0, &[e]);
        }
        assert_eq!(c.counters()[0], [2, 0, 5]);
        assert_eq!((tiers(&c, Tier::Vram), tiers(&c, Tier::Pinned)), (vec![1, 4], vec![3]));
    }

    /// CLOCK admission limit 1 per tick: tokens [1 2], [1 3], [2]. 2 bypasses in token 1
    /// (budget spent on 1); 3 takes the free VRAM slot; 2 then evicts 1 into pinned
    #[test]
    fn clock_admission_limit_bypasses_the_rest_of_the_tick() {
        let mut c = one_layer(Policy::Clock { admit: Some(1) }, 2, 1);
        c.observe_token(0, &[1, 2]);
        assert_eq!(c.tier(0, 2), Tier::Nvme, "the second miss of the tick bypasses");
        c.observe_token(0, &[1, 3]);
        c.observe_token(0, &[2]);
        assert_eq!(c.counters()[0], [1, 0, 4]);
        assert_eq!((tiers(&c, Tier::Vram), tiers(&c, Tier::Pinned)), (vec![2, 3], vec![1]));
        assert!(ExpertCache::new(Policy::Clock { admit: Some(ADMIT_MAX + 1) }, Scope::PerLayer, 1, 16, 2, 1).is_err());
        assert!(ExpertCache::new(Policy::Clock { admit: Some(0) }, Scope::PerLayer, 1, 16, 2, 1).is_err());
    }

    /// LFU decay 0.5, V 1 + P 1, 1 1 2 3 2 2 (decay at the start of every token). Scores by
    /// hand: t3 2 (1.0) beats 1 (0.75); t4 3 (1.0) beats 2 (0.5), 2 demoted beats 1 (0.375) for
    /// pinned; t5 2 is a pinned hit at 1.25 and beats 3 (0.5); t6 a VRAM hit
    #[test]
    fn lfu_with_decay_hand_sequence() {
        let mut c = one_layer(Policy::Lfu { decay: 0.5 }, 1, 1);
        for e in [1, 1, 2, 3, 2, 2] {
            c.observe_token(0, &[e]);
        }
        assert_eq!(c.counters()[0], [2, 1, 3]);
        assert_eq!((tiers(&c, Tier::Vram), tiers(&c, Tier::Pinned)), (vec![2], vec![3]));
        // decay 1.0 = plain LFU: 1 (count 2) holds VRAM until 2 reaches 2 and is newer
        let mut c = one_layer(Policy::Lfu { decay: 1.0 }, 1, 1);
        for e in [1, 1, 2, 3, 2, 2] {
            c.observe_token(0, &[e]);
        }
        assert_eq!(c.counters()[0], [2, 0, 4]);
        assert_eq!((tiers(&c, Tier::Vram), tiers(&c, Tier::Pinned)), (vec![2], vec![1]));
        assert!(ExpertCache::new(Policy::Lfu { decay: 0.0 }, Scope::PerLayer, 1, 16, 1, 1).is_err());
        assert!(ExpertCache::new(Policy::Lfu { decay: 1.5 }, Scope::PerLayer, 1, 16, 1, 1).is_err());
    }

    /// the xorshift64 trace of `docs/expert-cache.md` (the same generator in Python)
    fn trace(tokens: usize, layers: usize, experts: u64, k: usize) -> Vec<Vec<Vec<u32>>> {
        let mut x: u64 = 0x9E37_79B9_7F4A_7C15;
        let mut out = Vec::with_capacity(tokens);
        for t in 0..tokens {
            let base = (t as u64 / 200) * 16;
            let mut tok = Vec::with_capacity(layers);
            for l in 0..layers as u64 {
                let mut got: Vec<u32> = Vec::with_capacity(k);
                while got.len() < k {
                    x ^= x << 13;
                    x ^= x >> 7;
                    x ^= x << 17;
                    let e = if (x >> 32) % 10 < 7 { (base + l * 7 + x % 48) % experts } else { (x >> 8) % experts } as u32;
                    if !got.contains(&e) {
                        got.push(e);
                    }
                }
                got.sort_unstable();
                tok.push(got);
            }
            out.push(tok);
        }
        out
    }

    fn run(c: &mut ExpertCache, tr: &[Vec<Vec<u32>>]) {
        for tok in tr {
            for (l, ids) in tok.iter().enumerate() {
                c.observe_token(l, ids);
            }
        }
    }

    fn total(c: &ExpertCache) -> [u64; 3] {
        c.counters().iter().fold([0; 3], |a, x| [a[0] + x[0], a[1] + x[1], a[2] + x[2]])
    }

    /// The engine LRU against `tools/glm_tier_sim.py` `lru_reads` on the same synthetic trace
    /// (600 tokens x 42 layers x top 8 of 288; script and command in `docs/expert-cache.md`,
    /// run 2026-10-08 under `.venv-oracle` at `e6f901a`): every visit NVMe-tier, `lru_reads`
    /// summed over positions = W 0 201600, W 25 151180, W 37 125281, W 108 44507, W 145 35092.
    /// Exclusive two-tier LRU: nvme = lru_reads(V + P), pinned + nvme = lru_reads(V).
    #[test]
    fn lru_counters_equal_glm_tier_sim_lru_reads() {
        let tr = trace(600, 42, 288, 8);
        assert_eq!(&tr[0][0], &[6, 9, 31, 36, 42, 44, 45, 66], "the generator drifted from the Python one");
        for (v, p, lru_v, lru_vp) in [(25, 83, 151_180u64, 44_507u64), (37, 108, 125_281, 35_092)] {
            let mut c = ExpertCache::new(Policy::Lru, Scope::PerLayer, 42, 288, v, p).unwrap();
            run(&mut c, &tr);
            let [vr, pi, nv] = total(&c);
            assert_eq!(vr + pi + nv, 201_600, "V {v} P {p}");
            assert_eq!(nv, lru_vp, "V {v} P {p}: NVMe misses = lru_reads(W = V + P)");
            assert_eq!(pi + nv, lru_v, "V {v} P {p}: VRAM misses = lru_reads(W = V)");
        }
    }

    /// every policy, both scopes: after a trace every expert sits in exactly one tier and no
    /// tier is over its capacity (checked after every token)
    #[test]
    fn tiers_stay_exclusive_under_every_policy_and_scope() {
        let tr = trace(120, 6, 64, 8);
        let policies = [Policy::Lru, Policy::Clock { admit: None }, Policy::Clock { admit: Some(3) }, Policy::Lfu { decay: 0.7 }];
        for p in policies {
            for scope in [Scope::PerLayer, Scope::Global] {
                for (v, pin) in [(4, 6), (0, 5), (5, 0), (16, 40)] {
                    let mut c = ExpertCache::new(p, scope, 6, 64, v, pin).unwrap();
                    for tok in &tr {
                        for (l, ids) in tok.iter().enumerate() {
                            c.observe_token(l, ids);
                        }
                        c.check_exclusive().unwrap_or_else(|e| panic!("{p:?} {scope:?} V {v} P {pin}: {e}"));
                    }
                    let n: u64 = total(&c).iter().sum();
                    assert_eq!(n, 120 * 6 * 8);
                    if scope == Scope::PerLayer {
                        for l in 0..6 {
                            assert!(c.vram_set(l).len() <= v);
                        }
                    }
                }
            }
            // a partial seed (VRAM 2 of 4, pinned 3 of 4): pinned hits find a FREE VRAM slot,
            // the one path a trace from an empty cache never takes
            let mut c = one_layer(p, 4, 4);
            c.seed(0, &[1, 2], &[3, 4, 5]).unwrap();
            for ids in [[3u32].as_slice(), &[4], &[5, 6], &[7, 8, 9], &[1, 3]] {
                c.observe_token(0, ids);
                c.check_exclusive().unwrap_or_else(|e| panic!("{p:?} partial seed: {e}"));
            }
        }
    }

    /// capacity 0 in both tiers: every access is an NVMe miss, nothing is ever cached
    #[test]
    fn capacity_zero_is_all_misses() {
        let tr = trace(50, 4, 64, 8);
        for p in [Policy::Lru, Policy::Clock { admit: None }, Policy::Clock { admit: Some(2) }, Policy::Lfu { decay: 0.9 }] {
            for scope in [Scope::PerLayer, Scope::Global] {
                let mut c = ExpertCache::new(p, scope, 4, 64, 0, 0).unwrap();
                run(&mut c, &tr);
                assert_eq!(total(&c), [0, 0, 50 * 4 * 8], "{p:?} {scope:?}");
                assert!((0..4).all(|l| (0..64).all(|e| c.tier(l, e) == Tier::Nvme)));
            }
        }
    }

    /// the global pool over one layer is the per-layer pool
    #[test]
    fn a_global_pool_over_one_layer_is_the_per_layer_pool() {
        let tr = trace(200, 1, 64, 8);
        for p in [Policy::Lru, Policy::Clock { admit: Some(4) }, Policy::Lfu { decay: 0.8 }] {
            let mut a = ExpertCache::new(p, Scope::PerLayer, 1, 64, 6, 10).unwrap();
            let mut b = ExpertCache::new(p, Scope::Global, 1, 64, 6, 10).unwrap();
            run(&mut a, &tr);
            run(&mut b, &tr);
            assert_eq!(a.counters(), b.counters(), "{p:?}");
            assert_eq!(a.vram_set(0), b.vram_set(0), "{p:?}");
        }
    }

    /// cumulative counts: a rise is one access, ordered by the rise, a reset counts from zero
    #[test]
    fn observe_counts_reads_rises_of_cumulative_counts() {
        let mut c = one_layer(Policy::Lru, 2, 2);
        let mut cum = vec![0u64; 16];
        cum[3] = 5;
        cum[7] = 1;
        cum[9] = 2;
        c.observe_counts(0, &cum);
        // ascending rise: 7 (1), 9 (2), 3 (5) - 3 and 9 are the two most recent
        assert_eq!(c.vram_set(0), vec![3, 9]);
        assert_eq!(c.tier(0, 7), Tier::Pinned);
        assert_eq!(c.counters()[0], [0, 0, 3]);
        // no rise: nothing touched
        c.observe_counts(0, &cum);
        assert_eq!(c.counters()[0], [0, 0, 3]);
        // 7 rises again: a pinned hit, promoted
        cum[7] = 2;
        c.observe_counts(0, &cum);
        assert_eq!(c.counters()[0], [0, 1, 3]);
        assert!(c.vram_set(0).contains(&7));
        // a reset (counts below the previous ones) counts from zero
        let mut reset = vec![0u64; 16];
        reset[1] = 1;
        c.observe_counts(0, &reset);
        assert_eq!(c.counters()[0], [0, 1, 4]);
    }

    #[test]
    fn seed_places_the_given_tiers_most_recent_first() {
        let mut c = one_layer(Policy::Lru, 2, 2);
        c.seed(0, &[5, 6], &[1, 2]).unwrap();
        c.check_exclusive().unwrap();
        assert_eq!(c.keep_order(0, 5, 6), Ordering::Greater);
        assert_eq!(c.keep_order(0, 6, 1), Ordering::Greater);
        // a miss demotes the oldest VRAM id (6) and drops the oldest pinned id (2)
        c.observe_token(0, &[9]);
        assert_eq!((tiers(&c, Tier::Vram), tiers(&c, Tier::Pinned)), (vec![5, 9], vec![1, 6]));
        assert!(c.seed(0, &[5], &[]).is_err(), "placed twice");
        let mut c = one_layer(Policy::Clock { admit: None }, 1, 1);
        assert!(c.seed(0, &[1, 2], &[]).is_err(), "over the VRAM capacity");
    }

    #[test]
    fn the_policy_is_on_for_glm5_next_or_by_opt_in_only() {
        assert_eq!(policy_for("Glm5Next", None, false), Ok(Some(Policy::Lru)));
        assert_eq!(policy_for("FlashNext", None, false), Ok(None));
        assert_eq!(policy_for("Qwen35Dense", None, false), Ok(None));
        assert_eq!(policy_for("FlashNext", Some(""), false), Ok(None));
        assert_eq!(policy_for("Glm5Next", Some("off"), false), Ok(None));
        assert_eq!(policy_for("Glm5Next", Some("off"), true), Ok(None));
        assert_eq!(policy_for("FlashNext", Some("lru"), false), Ok(Some(Policy::Lru)));
        assert_eq!(policy_for("FlashNext", Some("clock"), false), Ok(Some(Policy::Clock { admit: None })));
        assert_eq!(policy_for("FlashNext", Some("clock:64"), false), Ok(Some(Policy::Clock { admit: Some(64) })));
        assert_eq!(policy_for("FlashNext", Some("lfu"), false), Ok(Some(Policy::Lfu { decay: 0.5 })));
        assert_eq!(policy_for("FlashNext", Some("lfu:0.9"), false), Ok(Some(Policy::Lfu { decay: 0.9 })));
        for bad in ["clock:65", "clock:0", "lfu:0", "lfu:2", "lru:3", "belady"] {
            let e = policy_for("FlashNext", Some(bad), false).unwrap_err();
            assert!(e.starts_with(&format!("{ENV}={bad:?} is not a cache policy")), "{e}");
        }
        let e = policy_for("Glm5Next", None, true).unwrap_err();
        assert!(e.contains("CROW_ADAPT_WINDOW=1"), "{e}");
        assert_eq!(policy_for("FlashNext", None, true), Ok(None), "off stays off under the window");
    }
}
