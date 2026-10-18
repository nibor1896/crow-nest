#!/usr/bin/env python3
"""#245: tests of tools/residency_replay.py on small synthetic traces with known answers.

  python -I tools/test_residency_replay.py

No GPU, no model, no dump: the routing is written in the engine's CROW_ROUTE_DUMP_PREFILL layout where the reader is
tested, and passed as arrays [positions, layers, top-k] everywhere else. Every policy has a hand-computed trace; LRU and
LFU are also checked against a literal port of engine/src/expert_cache.rs (per-tick multiply, linear victim search),
the frequency tiers against tools/glm_tier_sim.py FreqTiers, and the trickle's plan against a literal port of
Residency::plan_swaps.
"""
import contextlib
import heapq
import io
import os
import random
import shutil
import struct
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import glm_tier_sim as ts  # noqa: E402
import residency_replay as rr  # noqa: E402


def tr(*rows):
    """routes [P, 1, K] of one layer from per-position id lists"""
    return np.array([[r] for r in rows], dtype=np.uint16)


def hits(res):
    return [round(float(x), 6) for x in res["hit"]]


# ---------------------------------------------------------------- literal reference ports

class RefCache:
    """engine/src/expert_cache.rs Pool + ExpertCache (Scope::PerLayer, Lru / Lfu) line by line: tiers VRAM 0,
    pinned 1, NVMe 2; per-tick multiply of the scores; linear victim search by keep()."""

    def __init__(self, lfu, decay, layers, experts, vram, pinned):
        self.lfu, self.decay, self.E = lfu, decay, experts
        self.tier = [[2] * experts for _ in range(layers)]
        self.stamp = [[0] * experts for _ in range(layers)]
        self.score = [[0.0] * experts for _ in range(layers)]
        self.len = [[0, 0] for _ in range(layers)]
        self.cap = [vram, pinned]
        self.now = 0
        self.last = [None] * layers

    def set(self, l, k, t):
        old = self.tier[l][k]
        if old != 2:
            self.len[l][old] -= 1
        if t != 2:
            self.len[l][t] += 1
        self.tier[l][k] = t

    def prio(self, l, k):
        return (self.score[l][k] if self.lfu else 0.0, self.stamp[l][k])

    def keep(self, l, a, b):
        pa, pb = self.prio(l, a), self.prio(l, b)
        if pa != pb:
            return 1 if pa > pb else -1
        return 1 if a < b else (-1 if a > b else 0)

    def min_in(self, l, t):
        best = None
        for k in range(self.E):
            if self.tier[l][k] == t and (best is None or self.keep(l, k, best) < 0):
                best = k
        return best

    def offer_pinned(self, l, k):
        if self.cap[1] == 0:
            self.set(l, k, 2)
        elif self.len[l][1] < self.cap[1]:
            self.set(l, k, 1)
        else:
            u = self.min_in(l, 1)
            if self.keep(l, k, u) > 0:
                self.set(l, u, 2)
                self.set(l, k, 1)
            else:
                self.set(l, k, 2)

    def ranked(self, l, k):
        t = self.tier[l][k]
        if t == 0:
            return
        if self.cap[0] > 0:
            if self.len[l][0] < self.cap[0]:
                self.set(l, k, 0)
                return
            v = self.min_in(l, 0)
            if self.keep(l, k, v) > 0:
                self.set(l, k, 0)
                self.offer_pinned(l, v)
                return
        if t == 2:
            self.offer_pinned(l, k)

    def seed(self, l, vram, pinned):
        total, base = len(vram) + len(pinned), self.now
        for t, ids in ((0, vram), (1, pinned)):
            for i, e in enumerate(ids):
                rank = i if t == 0 else len(vram) + i
                self.stamp[l][e] = base + total - rank
                self.set(l, e, t)
        self.now = base + total

    def tick(self, l):
        if self.lfu:
            self.score[l] = [s * self.decay for s in self.score[l]]

    def access(self, l, e, w):
        self.now += 1
        t = self.tier[l][e]
        self.stamp[l][e] = self.now
        if self.lfu:
            self.score[l][e] += float(w)
        self.ranked(l, e)
        return t

    def observe_token(self, l, ids):
        self.tick(l)
        return [self.access(l, e, 1) for e in ids]

    def observe_counts(self, l, c):
        last = self.last[l] or [0] * self.E
        touched = []
        for e in range(self.E):
            d = c[e] - last[e] if c[e] >= last[e] else c[e]
            if d > 0:
                touched.append((d, e))
        self.last[l] = list(c)
        touched.sort()
        self.tick(l)
        return [self.access(l, e, d) for d, e in touched]

    def vram(self, l):
        return {e for e in range(self.E) if self.tier[l][e] == 0}


def ref_seeded(lfu, decay, rows, experts):
    c = RefCache(lfu, decay, len(rows), experts, len(rows[0]), experts - len(rows[0]))
    for l, r in enumerate(rows):
        hot = set(r)
        c.seed(l, list(r), [e for e in range(experts) if e not in hot])
    return c


def plan_swaps_ref(slots, n, c, k, excl):
    """Residency::plan_swaps (residency.rs, cache off) on one layer's slot list (None = EMPTY), line by line"""
    E = len(c)
    have = [False] * E
    for e in slots:
        if e is not None:
            have[e] = True
    want = sorted(range(E), key=lambda e: (-c[e], not have[e], e))[:n]
    want_set = set(want)
    incoming = [e for e in want if not have[e]]
    out = [(s, e) for s, e in enumerate(slots) if e is not None and e not in want_set and s not in excl]
    out.sort(key=lambda x: (c[x[1]], x[1]))
    kk = len(incoming) if k == 0 else min(len(incoming), k)
    kk = min(kk, len(out))
    return [(out[i][1], incoming[i]) for i in range(kk)]


# ---------------------------------------------------------------- the reader

def write_routes(path, routes, chunks):
    """the CROW_ROUTE_DUMP_PREFILL layout (gen.rs route_dump_prefill): per prefill chunk, per layer in order,
    u32 layer, u32 t, t x TOPK u16 ids"""
    with open(path, "wb") as f:
        p = 0
        for t in chunks:
            for l in range(routes.shape[1]):
                f.write(struct.pack("<II", l, t))
                f.write(routes[p:p + t, l, :].astype("<u2").tobytes())
            p += t
    assert p == len(routes)


class TestRoutesFormat(unittest.TestCase):
    def test_round_trip_in_the_engine_layout(self):
        rng = np.random.default_rng(1)
        routes = rng.integers(0, rr.E, size=(10, rr.LAYERS, rr.TOPK)).astype(np.uint16)
        d = tempfile.mkdtemp()
        try:
            p = os.path.join(d, "x-routes.bin")
            write_routes(p, routes, [5, 3, 2])
            self.assertEqual(os.path.getsize(p), 10 * rr.LAYERS * rr.TOPK * 2 + 3 * rr.LAYERS * 8)
            back = rr.load_routes(p)
            self.assertEqual(back.shape, (10, rr.LAYERS, rr.TOPK))
            np.testing.assert_array_equal(back, routes)
            # the dump self-test: hotset-eval's counts equal the replay's flat counts
            np.testing.assert_array_equal(rr.counts(back), rr.flat_counts(back, rr.E).reshape(rr.LAYERS, rr.E))
            # a dump cut inside a chunk is refused
            with open(p, "rb") as f:
                raw = f.read()
            with open(p, "wb") as f:
                f.write(raw[:len(raw) - (2 * rr.TOPK * 2 + 8)])
            with self.assertRaises(AssertionError):
                rr.load_routes(p)
        finally:
            shutil.rmtree(d)

    def test_spans_and_requests(self):
        d = tempfile.mkdtemp()
        try:
            p = os.path.join(d, "m.json")
            rr.jdump({"spans": [[2, 5], [7, 9]], "skipped": 0, "tokens": 10}, p)
            self.assertEqual(rr.load_spans(p, 10), [(2, 5), (7, 9)])
            with self.assertRaises(AssertionError):
                rr.load_spans(p, 11)
            rr.jdump({"spans": [[2, 5], [4, 9]], "skipped": 0, "tokens": 10}, p)
            with self.assertRaises(AssertionError):
                rr.load_spans(p, 10)
        finally:
            shutil.rmtree(d)
        # serve re-prefills the previous answer: request k's prompt starts at span k-1's start
        self.assertEqual(rr.requests([(2, 5), (7, 9)]), [(0, 2, 5), (2, 7, 9)])
        self.assertEqual(rr.requests([(2, 5), (7, 9)], reprefill=False), [(0, 2, 5), (5, 7, 9)])


# ---------------------------------------------------------------- static

class TestStatic(unittest.TestCase):
    ROUTES = tr([3, 2], [1, 4], [0, 4])
    REQS = [(0, 0, 3)]

    def test_rows_follow_the_sidecar_length_rule(self):
        self.assertEqual(rr.adapt_rows([[2, 0, 1]], 2, 6), [[2, 0]])
        self.assertEqual(rr.adapt_rows([[2, 0, 1]], 3, 6), [[2, 0, 1]])
        self.assertEqual(rr.adapt_rows([[2, 0, 1]], 4, 6), [[2, 0, 1, 3]])
        self.assertEqual(rr.adapt_rows([[5, 0]], 5, 6), [[5, 0, 1, 2, 3]])
        with self.assertRaises(ValueError):
            rr.adapt_rows([[0]], 7, 6)

    def test_hit_rate_at_the_budget_boundary(self):
        # budget 2 truncates to {2, 0}; 3 is the row; 4 pads with the lowest unused id, 3
        for budget, want in ((2, [0.5, 0.0, 0.5]), (3, [0.5, 0.5, 0.5]), (4, [1.0, 0.5, 0.5])):
            rows = rr.adapt_rows([[2, 0, 1]], budget, 6)
            self.assertEqual(hits(rr.run_static(self.ROUTES, self.REQS, rows, 6)), want, budget)

    def test_only_generated_positions_are_scored(self):
        routes = tr([2, 0], [5, 4], [3, 4])
        r = rr.run_static(routes, [(0, 0, 1), (1, 2, 3)], [[2, 0]], 6)
        self.assertEqual(hits(r), [1.0, 0.0])


# ---------------------------------------------------------------- LRU

class TestLru(unittest.TestCase):
    def run_seq(self, cache, seq, k=1):
        return [cache.token([sorted(ids)]) for ids in seq]

    def test_hand_trace(self):
        # VRAM 2 seeded [0, 1] (0 most recent): 2 0 1 2 3 3 -> miss hit miss miss miss hit
        c = rr.Lru([[0, 1]], 6)
        out = self.run_seq(c, [[2], [0], [1], [2], [3], [3]])
        self.assertEqual([h for h, _ in out], [0, 1, 0, 0, 0, 1])
        self.assertEqual(sum(a for _, a in out), 4)
        self.assertEqual(c.vram(0), {2, 3})

    def test_a_token_can_evict_its_own_later_expert(self):
        # observe_token takes the ids one at a time, ascending: 2 evicts the LRU 3 before 3 is reached
        c = rr.Lru([[0, 3]], 6)
        self.assertEqual(c.token([[2, 3]]), (0, 2))
        self.assertEqual(c.vram(0), {2, 3})
        self.assertEqual(c.token([[2, 3]]), (2, 0))

    def test_equals_expert_cache_rs(self):
        rng = random.Random(245)
        for case in range(30):
            L, E, V, K = 2, 12, 3 + case % 4, 1 + case % 3
            rows = [rng.sample(range(E), V) for _ in range(L)]
            fast, ref = rr.Lru(rows, E), ref_seeded(False, 1.0, rows, E)
            cum = np.zeros((L, E), dtype=np.int64)
            last = np.zeros((L, E), dtype=np.int64)
            for step in range(60):
                if step % 7 == 6:     # a batch of counts (observe_counts): a prompt run or a trickle tick
                    cum += np.array([[rng.randrange(3) for _ in range(E)] for _ in range(L)])
                    rr.observe_counts(fast, cum - last)
                    last = cum.copy()
                    for l in range(L):
                        ref.observe_counts(l, cum[l].tolist())
                else:
                    row = [sorted(rng.sample(range(E), K)) for _ in range(L)]
                    h, _ = fast.token(row)
                    want = sum(t == 0 for l in range(L) for t in ref.observe_token(l, row[l]))
                    self.assertEqual(h, want, (case, step))
                for l in range(L):
                    self.assertEqual(fast.vram(l), ref.vram(l), (case, step, l))


# ---------------------------------------------------------------- LFU

class TestLfu(unittest.TestCase):
    def test_hand_trace_decay_half(self):
        # VRAM 2 seeded [0, 1]: 2 2 0 3 2 -> scores decay by half per token; 3 outscores 2 at t3, 2 outscores 0 at t4
        c = rr.Lfu([[0, 1]], 4, 0.5)
        out = [c.token([[e]]) for e in (2, 2, 0, 3, 2)]
        self.assertEqual([h for h, _ in out], [0, 1, 1, 0, 0])
        self.assertEqual(sum(a for _, a in out), 3)
        self.assertEqual(c.vram(0), {2, 3})

    def test_hand_trace_plain_lfu_ties_go_to_the_recent_access(self):
        # decay 1.0: t3 ties 3 with 0 (score 1) and admits it (more recent); t6 refuses 1 (1 < 2); t7 ties and admits
        c = rr.Lfu([[0, 1]], 4, 1.0)
        out = [c.token([[e]]) for e in (2, 2, 0, 3, 2, 3, 1, 1)]
        self.assertEqual([h for h, _ in out], [0, 1, 1, 0, 1, 1, 0, 0])
        self.assertEqual([a for _, a in out], [1, 0, 0, 1, 0, 0, 0, 1])
        self.assertEqual(c.vram(0), {1, 2})

    def test_equals_expert_cache_rs(self):
        rng = random.Random(1245)
        for case in range(24):
            decay = (0.5, 1.0)[case % 2]
            L, E, V, K = 2, 10, 2 + case % 4, 1 + case % 3
            rows = [rng.sample(range(E), V) for _ in range(L)]
            fast, ref = rr.Lfu(rows, E, decay), ref_seeded(True, decay, rows, E)
            cum = np.zeros((L, E), dtype=np.int64)
            last = np.zeros((L, E), dtype=np.int64)
            for step in range(50):
                if step % 5 == 4:
                    cum += np.array([[rng.randrange(4) for _ in range(E)] for _ in range(L)])
                    rr.observe_counts(fast, cum - last)
                    last = cum.copy()
                    for l in range(L):
                        ref.observe_counts(l, cum[l].tolist())
                else:
                    row = [sorted(rng.sample(range(E), K)) for _ in range(L)]
                    h, _ = fast.token(row)
                    want = sum(t == 0 for l in range(L) for t in ref.observe_token(l, row[l]))
                    self.assertEqual(h, want, (case, step))
                for l in range(L):
                    self.assertEqual(fast.vram(l), ref.vram(l), (case, step, l))

    def test_renormalization_keeps_the_order(self):
        # decay 0.5 doubles the increment per tick: 700 ticks renormalize twice (1e100 ~ 2^332)
        rng = random.Random(7)
        rows = [[0, 1, 2]]
        fast, ref = rr.Lfu(rows, 8, 0.5), ref_seeded(True, 0.5, rows, 8)
        for step in range(700):
            ids = sorted(rng.sample(range(8), 2))
            h, _ = fast.token([ids])
            self.assertEqual(h, sum(t == 0 for t in ref.observe_token(0, ids)), step)
        self.assertLess(fast.inc, rr.RENORM)
        self.assertEqual(fast.vram(0), ref.vram(0))


# ---------------------------------------------------------------- frequency tiers

class TestFreq(unittest.TestCase):
    def test_hand_trace_margin(self):
        # VRAM 1 seeded [0], half-life 1 token (increment doubles per token), margin 1: 1 1 0 1 0 0
        c = rr.Freq([[0]], 4, halflife=1.0, margin=1.0)
        out = [c.token([[e]]) for e in (1, 1, 0, 1, 0, 0)]
        self.assertEqual([h for h, _ in out], [0, 1, 0, 1, 0, 0])
        self.assertEqual([a for _, a in out], [1, 0, 0, 0, 0, 1])
        c = rr.Freq([[0]], 4, halflife=1.0, margin=0.0)
        out = [c.token([[e]]) for e in (1, 1, 0, 1, 0, 0)]
        self.assertEqual([h for h, _ in out], [0, 1, 0, 0, 0, 1])

    def test_the_call_s_own_experts_are_no_victims(self):
        # VRAM [0, 1], priors 0 / 2 / 1.5: after the call {0, 2} the scores are 1 / 2 / 2.5; 0 is the lowest but
        # routed in this call, so 2 replaces 1
        c = rr.Freq([[0, 1]], 4, halflife=1.0, margin=0.0, prior=[[0.0, 2.0, 1.5, 0.0]])
        self.assertEqual(c.token([[0, 2]]), (1, 1))
        self.assertEqual({e for e in range(4) if c.inv[0][e]}, {0, 2})

    def test_equals_glm_tier_sim_freq_tiers(self):
        rng = random.Random(2245)
        for case in range(12):
            E, V, K = 16, 3 + case % 4, 1 + case % 3
            halflife, margin = (4.0, 64.0)[case % 2], (0.0, 1.0, 0.5)[case % 3]
            row = rng.sample(range(E), V)
            prior = [rng.random() * 3 for _ in range(E)]
            mine = rr.Freq([row], E, halflife, margin, [prior])
            ft = ts.FreqTiers(V, E - V, halflife, margin, prior, layers=1, experts=E)
            ft.vset, ft.rset = set(row), set(range(E)) - set(row)
            ft.vheap = [(ft.score[k], k) for k in ft.vset]
            ft.rheap = [(ft.score[k], k) for k in ft.rset]
            heapq.heapify(ft.vheap)
            heapq.heapify(ft.rheap)
            for step in range(120):
                ids = sorted(rng.sample(range(E), K))
                h, _ = mine.token([ids])
                t = ft.step(0, ids, True)
                ft.tick()
                self.assertEqual(h, t.count(ts.VRAM), (case, step))
                self.assertEqual({e for e in range(E) if mine.inv[0][e]}, ft.vset, (case, step))


# ---------------------------------------------------------------- the trickle (today)

class TestToday(unittest.TestCase):
    # one layer, 6 experts, top-2, N 2 + 1 spare, a tick every 2 tokens, 1 swap per layer per tick
    KW = dict(spare=1, every=2, max_swaps=1)
    ROUTES = tr([2, 3], [2, 3],                                          # prompt
                [0, 2], [2, 3], [2, 4], [2, 3], [3, 4], [0, 3], [2, 3], [1, 3])

    def test_hand_trace_tick_timing(self):
        # i=2: the first plan sets the window base and plans nothing; i=4: window {2:2, 3:1, 4:1} -> 0 out, 2 in,
        # resident from i=5, 0 gone from i=6; i=6: window {0:1, 2:1, 3:2.5, 4:1.5} -> 1 out, 3 in (i=7)
        spans = []
        r = rr.run_trickle(self.ROUTES, [(0, 2, 10)], [[0, 1]], 6, rr.TodayPlan(), per_span=spans, **self.KW)
        self.assertEqual(hits(r), [0.5, 0.0, 0.0, 0.0, 0.0, 0.5, 0.5, 1.0])
        self.assertEqual(r["moves"], 2)
        self.assertEqual(spans[0]["swaps"], 2)
        self.assertEqual(spans[0]["cold_prompt"], 4)
        self.assertEqual(spans[0]["cold_generated"], 11)

    def test_the_window_sees_the_prompt_runs(self):
        # N 1 + 1 spare, top-1: the plan at request 2's i=2 ranks the window since request 1's plan; the prompt run
        # [3, 7) of 5s outweighs the 0s unless the re-prefill [0, 3) of 0s is in it too
        routes = tr([0], [0], [0], [5], [5], [5], [5], [0], [0], [5], [5])
        spans = [(0, 3), (7, 10), (10, 11)]
        kw = dict(spare=1, every=2, max_swaps=1)
        r = rr.run_trickle(routes, rr.requests(spans, reprefill=False), [[0]], 6, rr.TodayPlan(), **kw)
        self.assertEqual(hits(r), [1, 1, 1, 1, 1, 0, 1])
        self.assertEqual(r["moves"], 1)
        r = rr.run_trickle(routes, rr.requests(spans, reprefill=True), [[0]], 6, rr.TodayPlan(), **kw)
        self.assertEqual(hits(r), [1, 1, 1, 1, 1, 0, 0])
        self.assertEqual(r["moves"], 0)

    def test_plan_equals_residency_plan_swaps(self):
        rng = random.Random(3245)
        for case in range(200):
            E, n, spare = 16, 3 + case % 6, case % 3
            slots = rng.sample(range(E), n) + [None] * spare
            # in-flight swaps: an extra resident in a spare slot (phase A committed)
            if spare and case % 4 == 0:
                slots[n] = rng.choice([e for e in range(E) if e not in slots])
            c = [rng.randrange(5) for _ in range(E)]
            k = (0, 1, 3, 100)[case % 4]
            excl_slots = {s for s in range(len(slots)) if slots[s] is not None and rng.random() < 0.2}
            want = plan_swaps_ref(slots, n, c, k, excl_slots)
            res = np.zeros((1, E), dtype=bool)
            res[0, [e for e in slots if e is not None]] = True
            got = rr.plan_freq(res, np.array([c]), n, [k], [{slots[s] for s in excl_slots}])[0]
            self.assertEqual(got, want, case)


class TestTrickleCache(unittest.TestCase):
    def test_lru_through_the_trickle(self):
        # N 2 + 1 spare, LRU VRAM 2 seeded [0, 1]: at i=2 the cache holds {2, 3} after the rises 2, 3; the most
        # recent (3) comes first; at i=4 the rises 3, 4 bring 4 in for 0 (the LRU resident outside the cache)
        routes = tr([2], [3], [4], [3], [0])
        plan = rr.CachePlan(rr.Lru([[0, 1]], 5))
        r = rr.run_trickle(routes, [(0, 0, 5)], [[0, 1]], 5, plan, spare=1, every=2, max_swaps=1)
        self.assertEqual(hits(r), [0, 0, 0, 1, 1])
        self.assertEqual(r["moves"], 2)

    def test_lru_trickle_differs_from_today(self):
        routes = tr([1], [1], [2], [2], [1])
        kw = dict(spare=1, every=2, max_swaps=1)
        r = rr.run_trickle(routes, [(0, 0, 5)], [[0]], 4, rr.CachePlan(rr.Lru([[0]], 4)), **kw)
        self.assertEqual((hits(r), r["moves"]), ([0, 0, 0, 0, 1], 2))
        r = rr.run_trickle(routes, [(0, 0, 5)], [[0]], 4, rr.TodayPlan(), **kw)
        self.assertEqual((hits(r), r["moves"]), ([0, 0, 0, 0, 0], 1))


# ---------------------------------------------------------------- prompt runs of the per-token caches

class TestPrefillModes(unittest.TestCase):
    # VRAM 1 seeded [0], top-1: the prompt run [0, 2) routes 3; the request decodes 3 then 0
    ROUTES = tr([3], [3], [3], [0])
    REQS = [(0, 2, 4)]

    def test_skip_batch_token(self):
        self.assertEqual(hits(rr.run_cache(self.ROUTES, self.REQS, rr.Lru([[0]], 4), "skip")), [0, 0])
        self.assertEqual(hits(rr.run_cache(self.ROUTES, self.REQS, rr.Lru([[0]], 4), "batch")), [1, 0])
        self.assertEqual(hits(rr.run_cache(self.ROUTES, self.REQS, rr.Lru([[0]], 4), "token")), [1, 0])
        # LFU decay 1.0: one batch access of weight 2 holds 3 against the single later 0
        r = rr.run_cache(self.ROUTES, self.REQS, rr.Lfu([[0]], 4, 1.0), "batch")
        self.assertEqual((hits(r), r["moves"]), ([1, 0], 0))
        with self.assertRaises(ValueError):
            rr.run_cache(self.ROUTES, self.REQS, rr.Freq([[0]], 4), "batch")


# ---------------------------------------------------------------- the command line

def write_case(d, name, routes, spans):
    write_routes(os.path.join(d, name + "-routes.bin"), routes, [len(routes)])
    rr.jdump({"prefill_counts": rr.counts(routes).tolist()}, os.path.join(d, name + "-counts.json"))
    rr.jdump({"spans": [list(s) for s in spans], "skipped": 0, "tokens": len(routes)},
             os.path.join(d, name + "-mask.json"))


class TestCli(unittest.TestCase):
    def setUp(self):
        self.d = tempfile.mkdtemp()
        self.sets = os.path.join(self.d, "sets.json")
        rr.jdump({"sets": [list(range(160)) for _ in range(rr.LAYERS)]}, self.sets)

    def tearDown(self):
        shutil.rmtree(self.d)

    def run_cli(self, *extra):
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            doc = rr.main([self.d, self.d, "held", "--sets", self.sets, "--budgets", "20", *extra])
        return doc, buf.getvalue()

    def test_a_working_set_outside_the_static_set_is_go(self):
        # every generated token routes the same ten experts 200..209 in every layer, none in the static set
        rng = np.random.default_rng(5)
        routes = rng.integers(0, rr.E, size=(240, rr.LAYERS, rr.TOPK)).astype(np.uint16)
        for s, e in ((20, 100), (120, 240)):
            routes[s:e] = np.arange(200, 210, dtype=np.uint16)
        write_case(self.d, "held", routes, [(20, 100), (120, 240)])
        doc, out = self.run_cli("--policies", "static,today,lru,lfu:0.5,freq,trickle-lru", "--json",
                                os.path.join(self.d, "o.json"))
        self.assertIn("dump counts == engine counters: True", out)
        self.assertIn("freq prior: none", out)
        rows = {(r["policy"], r["prefill"]): r for r in doc["budgets"]["20"]["rows"]}
        self.assertEqual(rows[("static", "-")]["hit"], 0.0)
        self.assertEqual(rows[("static", "-")]["cold_per_token"], 480.0)
        self.assertGreater(rows[("lru", "skip")]["hit"], 0.95)
        self.assertEqual(rows[("lru", "skip")]["slots"], "N 20")
        self.assertEqual(rows[("today", "counts")]["slots"], "N 13 + 7 spare")
        self.assertEqual(rows[("today", "counts")]["diff_today"], 0.0)
        self.assertAlmostEqual(rows[("lru", "skip")]["diff_today"],
                               rows[("lru", "skip")]["hit"] - rows[("today", "counts")]["hit"])
        self.assertTrue(doc["budgets"]["20"]["verdict"]["go"])
        self.assertIn("verdict at 20: GO", out)
        self.assertIn("; against today ", out)
        self.assertIn("#245 go rule (a candidate beats static crow0924 by >= 0.10): GO at budget 20", out)
        self.assertEqual(rr.jload(os.path.join(self.d, "o.json"))["generated"], 200)

    def test_uniform_routing_is_no_go(self):
        rng = np.random.default_rng(6)
        routes = np.stack([np.stack([rng.choice(rr.E, rr.TOPK, replace=False) for _ in range(rr.LAYERS)])
                           for _ in range(160)]).astype(np.uint16)
        write_case(self.d, "held", routes, [(10, 80), (90, 160)])
        doc, out = self.run_cli("--policies", "static,lru,lfu:0.5")
        self.assertFalse(doc["budgets"]["20"]["verdict"]["go"])
        self.assertIn("NO GO at every budget replayed", out)

    def test_a_dump_that_disagrees_with_the_counters_is_refused(self):
        routes = np.zeros((4, rr.LAYERS, rr.TOPK), dtype=np.uint16) + np.arange(rr.TOPK, dtype=np.uint16)
        write_case(self.d, "held", routes, [(1, 4)])
        rr.jdump({"prefill_counts": np.zeros((rr.LAYERS, rr.E), int).tolist()},
                 os.path.join(self.d, "held-counts.json"))
        with self.assertRaises(AssertionError), contextlib.redirect_stdout(io.StringIO()):
            rr.main([self.d, self.d, "held", "--sets", self.sets, "--budgets", "20"])


if __name__ == "__main__":
    unittest.main()
