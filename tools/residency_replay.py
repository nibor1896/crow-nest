"""#245 step 1: replay per-position Flash-Next routing through candidate residency policies, offline.

  python -I tools/residency_replay.py <dump-dir> <mask-dir> <held-out> [<cal>...] [--budgets 144,151,170]
      [--policies static,today,lru,lfu:0.5,lfu:0.99,freq,trickle-lru,trickle-lfu:0.5] [--prefill skip,batch]
      [--sets decode_out/hotsets-M-crow0924-n160.json] [--halflife 64] [--margin 1] [--no-reprefill]
      [--per-span] [--json <out.json>]

The inputs are those of tools/hotset-eval.py: <dump-dir> holds <name>-routes.bin (CROW_ROUTE_DUMP_PREFILL) and
<name>-counts.json (`decode routestats`), <mask-dir> holds <name>-mask.json (tools/session_ids.py). The routing
reader, the dump self-test and the block bootstrap are hotset-eval.py's own functions. No GPU, no model.

The held-out file is replayed request by request: an assistant span of the mask is one request's decode, and the
prompt run before it is that request's prefill. Serve re-prefills the previous answer: request k prefills from the
start of span k-1 (2026-10-14 engine.log, request 2: 501 prefilled = 434 tokens of answer 1 + 67 new; the
`--no-reprefill` form starts at the end of span k-1 instead). Every policy starts from the same hot set, the rows of
`--sets` brought to its N by residency::sidecar_sets' rule (truncate, or pad with the lowest unused ids).

Policies, per layer (one budget = VRAM expert slots per layer):
- static          the hot set, N = budget, never changes.
- today           the engine's default: N = budget - 7 plus 7 spare slots and the stream trickle of serve
                  (engine/src/gen.rs trickle_tick, geo.rs apply_adapt_policy: every 8 decode tokens, at most 7
                  swaps per layer, ranked by plan_swaps on the decayed window of CROW_ADAPT_WINDOW=1, decay 0.5,
                  counts x1024; a swap is resident one tick after it is planned and the evicted expert leaves one
                  tick later; the trickle drains at the end of each request).
- lru, lfu:D      engine/src/expert_cache.rs Policy::Lru / Policy::Lfu{decay D} per token (observe_token: the
                  token's ids in ascending order, a miss is admitted when the policy says so), VRAM = budget.
- freq            the frequency tiers of tools/glm_tier_sim.py FreqTiers (#188) per layer: decayed counts with
                  half-life --halflife decode tokens, a miss enters VRAM only above (1 + --margin) x the lowest
                  VRAM score not routed in the call; prior from the calibration files' generated counts.
- trickle-lru, trickle-lfu:D
                  the same caches driven through the trickle as Residency::plan_swaps_cached would drive them
                  (cumulative counts every 8 tokens, at most 7 swaps per layer, N = budget - 7): #245 step 2 as
                  the issue words it.
`--prefill` says what a prompt run does to lru / lfu / freq: skip (nothing; GLM's freq rule), batch (one
observe_counts per layer over the run, expert_cache.rs' rule for cumulative counts; lru / lfu only), token (every
prompt position as a decode token, not scored). today and trickle-* see the prompt runs through the engine's
cumulative selection counts, as the engine does.

Printed per budget: hit rate on the generated positions, cold experts and moves (admissions or swaps started) per
generated token, the difference against static and its 95 % block-bootstrap CI (hotset-eval.py's boot_ci, seed
20260924), the difference against today, and the #245 go line: GO when a candidate (not static, not today) beats
static by at least 0.10 (the issue's rule; the line also gives the candidate's margin over today).
"""
import argparse
import heapq
import importlib.util
import json
import math
import os
from collections import OrderedDict

import numpy as np

TOOLS = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(TOOLS)


def _load_hotset_eval():
    spec = importlib.util.spec_from_file_location("hotset_eval", os.path.join(TOOLS, "hotset-eval.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


HE = _load_hotset_eval()
LAYERS, E, TOPK = HE.LAYERS, HE.E, HE.TOPK
load_routes, counts, boot_ci = HE.load_routes, HE.counts, HE.boot_ci

SETS = os.path.join(REPO, "decode_out", "hotsets-M-crow0924-n160.json")
BUDGETS = (144, 151, 170)
POLICIES = ("static", "today", "lru", "lfu:0.5", "lfu:0.99", "freq", "trickle-lru", "trickle-lfu:0.5")
MODES = ("skip", "batch")
# the stream trickle of serve (geo.rs apply_adapt_policy, chunk >= TRICKLE_CHUNK_THRESHOLD: stream / spare 7 /
# every TRICKLE_EVERY 8 / max 7) and its ranking window (gen.rs decay_window, CROW_ADAPT_DECAY default 0.5;
# window_layer scales by 1024 and truncates to u64)
SPARE, EVERY, MAX_SWAPS, ADAPT_DECAY, WINDOW_SCALE = 7, 8, 7, 0.5, 1024.0
# glm_tier_sim.py arena defaults: half-life 64 decode tokens, margin 1, prior tokens 0.5 x half-life / ln 2
HALFLIFE, MARGIN = 64.0, 1.0
# #245 go rule: a policy beats crow0924 (0.72 on the 2026-09-24 held-out) by at least 0.10
GO_DIFF, REF_HIT = 0.10, 0.72
SEED = 20260924
RENORM = 1e100


# ---------------------------------------------------------------- inputs

def jload(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def jdump(obj, path, **kw):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, **kw)


def load_spans(path, n):
    """the generated spans of a session_ids.py mask, one per assistant message (= one request)"""
    m = jload(path)
    assert m["tokens"] == n, "mask for %d tokens, dump has %d" % (m["tokens"], n)
    spans, prev = [], 0
    for a, b in m["spans"]:
        a, b = int(a), int(b)
        assert prev <= a < b <= n, "spans must be sorted, disjoint and inside the dump: %r" % ([a, b],)
        spans.append((a, b))
        prev = b
    return spans


def requests(spans, reprefill=True):
    """(prompt_lo, span_start, span_end) per request; the prompt run is [prompt_lo, span_start)"""
    out, ps, pe = [], 0, 0
    for s, e in spans:
        out.append((ps if reprefill else pe, s, e))
        ps, pe = s, e
    return out


def adapt_rows(sets, n, experts):
    """residency::sidecar_sets' length rule (#49): a row is truncated to its first n ids or padded with the lowest
    unused ids, appended after the file's own ids"""
    if not 1 <= n <= experts:
        raise ValueError("N=%d is not a hot-set size (1..=%d)" % (n, experts))
    rows = []
    for s in sets:
        r = [int(x) for x in s[:n]]
        if len(r) < n:
            have = set(r)
            r += [e for e in range(experts) if e not in have][:n - len(r)]
        rows.append(r)
    return rows


def member(rows, experts):
    m = np.zeros((len(rows), experts), dtype=bool)
    for l, r in enumerate(rows):
        m[l, r] = True
    return m


def pos_hit(routes, m):
    """per position: share of the routed choices that land in the membership m [L, E]"""
    hit = m[np.arange(m.shape[0])[None, :, None], routes]
    return hit.reshape(len(routes), -1).mean(axis=1)


def flat_counts(routes, experts):
    """per-layer selection counts of routes [P, L, K] as one [L * E] vector"""
    if len(routes) == 0:
        return np.zeros(routes.shape[1] * experts, dtype=np.int64)
    off = (np.arange(routes.shape[1], dtype=np.int64) * experts)[None, :, None]
    return np.bincount((routes.astype(np.int64) + off).ravel(), minlength=routes.shape[1] * experts)


def gen_counts(routes, spans, experts):
    g = np.zeros(routes.shape[1] * experts, dtype=np.int64)
    for s, e in spans:
        g += flat_counts(routes[s:e], experts)
    return g.reshape(routes.shape[1], experts)


def freq_prior(gcounts, halflife, k):
    """glm_tier_sim.arena_prior on generated counts: 0.5 x half-life / ln 2 tokens of each expert's routing rate"""
    c = np.asarray(gcounts, dtype=float)
    rate = c / np.maximum(c.sum(axis=1, keepdims=True), 1.0) * k
    return (0.5 * halflife / math.log(2) * rate).tolist()


# ---------------------------------------------------------------- caches (VRAM tier; the pinned tier is the rest)

class Lru:
    """expert_cache.rs Policy::Lru, one pool per layer, VRAM = len(row), pinned = the rest of the layer: an
    exclusive two-tier LRU over the whole layer, so the VRAM set is the `cap` most recent experts"""

    def __init__(self, rows, experts):
        self.E, self.cap, self.now = experts, len(rows[0]), 0
        self.od, self.stamp = [], []
        for r in rows:
            # ExpertCache::seed: VRAM ids hottest first (rank 0 most recent), then the pinned rest by ascending id
            hot = set(r)
            order = list(r) + [e for e in range(experts) if e not in hot]
            st = [0] * experts
            for rank, e in enumerate(order):
                st[e] = self.now + experts - rank
            self.now += experts
            od = OrderedDict()
            for e in reversed(r):
                od[e] = None
            self.od.append(od)
            self.stamp.append(st)

    def tick(self):
        pass

    def access(self, l, e, w=1):
        """-> (hit, admitted)"""
        self.now += 1
        self.stamp[l][e] = self.now
        od = self.od[l]
        if e in od:
            od.move_to_end(e)
            return True, False
        if len(od) >= self.cap:
            od.popitem(last=False)
        od[e] = None
        return False, True

    def token(self, row):
        """observe_token for every layer: -> (hits, admitted)"""
        h = a = 0
        now, cap = self.now, self.cap
        for l, ids in enumerate(row):
            od, st = self.od[l], self.stamp[l]
            for e in ids:
                now += 1
                st[e] = now
                if e in od:
                    od.move_to_end(e)
                    h += 1
                else:
                    if len(od) >= cap:
                        od.popitem(last=False)
                    od[e] = None
                    a += 1
        self.now = now
        return h, a

    def vram(self, l):
        return set(self.od[l])

    def key(self, l, e):
        return self.stamp[l][e]


class Lfu:
    """expert_cache.rs Policy::Lfu{decay}: a tick multiplies every score by `decay`, an access adds its weight;
    VRAM keeps the highest scores, ties to the more recent access; a miss is admitted when it is not below the
    lowest VRAM score. The scores are held in the growing-increment form of glm_tier_sim's FreqTiers (an access adds
    inc, a tick multiplies inc by 1 / decay): the same order as the engine's per-tick multiply in exact arithmetic,
    and the same floats for decay 0.5 and 1.0 (powers of two)."""

    def __init__(self, rows, experts, decay):
        if not 0.0 < decay <= 1.0:
            raise ValueError("LFU decay %r outside (0, 1]" % decay)
        self.E, self.cap, self.now, self.g, self.inc = experts, len(rows[0]), 0, 1.0 / decay, 1.0
        self.N, self.stamp, self.inv, self.size, self.heap = [], [], [], [], []
        for r in rows:
            hot = set(r)
            order = list(r) + [e for e in range(experts) if e not in hot]
            st = [0] * experts
            for rank, e in enumerate(order):
                st[e] = self.now + experts - rank
            self.now += experts
            iv = [False] * experts
            for e in r:
                iv[e] = True
            self.N.append([0.0] * experts)
            self.stamp.append(st)
            self.inv.append(iv)
            self.size.append(len(r))
            self.heap.append([])
            self._rebuild(len(self.heap) - 1)

    def _rebuild(self, l):
        N, st, iv = self.N[l], self.stamp[l], self.inv[l]
        h = [(N[e], st[e], e) for e in range(self.E) if iv[e]]
        heapq.heapify(h)
        self.heap[l] = h

    def tick(self):
        self.inc *= self.g
        if self.inc > RENORM:
            inc = self.inc
            for l in range(len(self.N)):
                self.N[l] = [x / inc for x in self.N[l]]
                self._rebuild(l)
            self.inc = 1.0

    def _min(self, l):
        h, N, st, iv = self.heap[l], self.N[l], self.stamp[l], self.inv[l]
        while h:
            n_, s_, v = h[0]
            if iv[v] and N[v] == n_ and st[v] == s_:
                return v
            heapq.heappop(h)
        return None

    def access(self, l, e, w=1):
        self.now += 1
        N, st, iv = self.N[l], self.stamp[l], self.inv[l]
        N[e] += w * self.inc
        st[e] = self.now
        h = self.heap[l]
        if iv[e]:
            heapq.heappush(h, (N[e], st[e], e))
            if len(h) > 4 * self.cap + 64:
                self._rebuild(l)
            return True, False
        if self.size[l] < self.cap:
            iv[e] = True
            self.size[l] += 1
            heapq.heappush(h, (N[e], st[e], e))
            return False, True
        v = self._min(l)
        if (N[e], st[e]) > (N[v], st[v]):
            heapq.heappop(h)
            iv[v] = False
            iv[e] = True
            heapq.heappush(h, (N[e], st[e], e))
            return False, True
        return False, False

    def token(self, row):
        self.tick()
        h = a = 0
        for l, ids in enumerate(row):
            for e in ids:
                hit, adm = self.access(l, e)
                h += hit
                a += adm
        return h, a

    def vram(self, l):
        iv = self.inv[l]
        return {e for e in range(self.E) if iv[e]}

    def key(self, l, e):
        return (self.N[l][e], self.stamp[l][e])


def observe_counts(cache, rises):
    """ExpertCache::observe_counts for every layer: rises [L, E] since the previous call; one tick, then every
    expert that rose is one access weighted by its rise, ascending rise then id. -> (hits, admitted)"""
    cache.tick()
    h = a = 0
    for l in range(rises.shape[0]):
        r = rises[l]
        nz = np.flatnonzero(r)
        for e in nz[np.lexsort((nz, r[nz]))].tolist():
            hit, adm = cache.access(l, e, int(r[e]))
            h += hit
            a += adm
    return h, a


class Freq:
    """glm_tier_sim.py FreqTiers (#188, CROW_GLM_ARENA_FREQ=1) per layer, VRAM = len(row), pinned = the rest:
    every routed expert gains the current increment (half-life `halflife` tokens, started at `prior`); a routed expert
    outside VRAM enters only while VRAM has a free slot or when its score is above (1 + margin) x the lowest VRAM
    score not routed in the call, which is written back. Ids in the call's order (ascending)."""

    def __init__(self, rows, experts, halflife=HALFLIFE, margin=MARGIN, prior=None):
        self.E, self.cap, self.margin = experts, len(rows[0]), margin
        self.grow, self.inc, self.ep = 2.0 ** (1.0 / halflife), 1.0, 0
        self.score, self.inv, self.size, self.heap, self.prot = [], [], [], [], []
        for l, r in enumerate(rows):
            self.score.append(list(prior[l]) if prior is not None else [0.0] * experts)
            iv = [False] * experts
            for e in r:
                iv[e] = True
            self.inv.append(iv)
            self.size.append(len(r))
            self.prot.append([0] * experts)
            self.heap.append([])
            self._rebuild(l)

    def _rebuild(self, l):
        sc, iv = self.score[l], self.inv[l]
        h = [(sc[e], e) for e in range(self.E) if iv[e]]
        heapq.heapify(h)
        self.heap[l] = h

    def _push(self, l, e):
        h = self.heap[l]
        heapq.heappush(h, (self.score[l][e], e))
        if len(h) > 4 * self.cap + 64:
            self._rebuild(l)

    def _low(self, l):
        h, sc, iv, pr, ep = self.heap[l], self.score[l], self.inv[l], self.prot[l], self.ep
        held, v = [], None
        while h:
            s, x = heapq.heappop(h)
            if not iv[x] or sc[x] != s:
                continue
            held.append((s, x))
            if pr[x] != ep:
                v = x
                break
        for t in held:
            heapq.heappush(h, t)
        return v

    def step(self, l, ids):
        """one decode call of layer l -> (hits, admitted)"""
        self.ep += 1
        sc, iv, pr, inc = self.score[l], self.inv[l], self.prot[l], self.inc
        misses = []
        for e in ids:
            pr[e] = self.ep
            sc[e] += inc
        for e in ids:
            if iv[e]:
                self._push(l, e)
            else:
                misses.append(e)
        a = 0
        for e in misses:
            if self.size[l] < self.cap:
                self.size[l] += 1
            else:
                v = self._low(l)
                if v is None or not sc[e] > (1.0 + self.margin) * sc[v]:
                    continue
                iv[v] = False
            iv[e] = True
            self._push(l, e)
            a += 1
        return len(ids) - len(misses), a

    def tick(self):
        self.inc *= self.grow
        if self.inc > RENORM:
            inc = self.inc
            for l in range(len(self.score)):
                self.score[l] = [s / inc for s in self.score[l]]
                self._rebuild(l)
            self.inc = 1.0

    def token(self, row):
        h = a = 0
        for l, ids in enumerate(row):
            dh, da = self.step(l, ids)
            h += dh
            a += da
        self.tick()
        return h, a


# ---------------------------------------------------------------- drivers

def _rows_of(routes, lo, hi, block=4096):
    """the positions [lo, hi) as nested lists, ids ascending per layer, in blocks (memory)"""
    for b in range(lo, hi, block):
        yield from np.sort(routes[b:min(hi, b + block)], axis=2).tolist()


def run_static(routes, reqs, rows, experts):
    m = member(rows, experts)
    hit = np.concatenate([pos_hit(routes[s:e], m) for _, s, e in reqs])
    return {"hit": hit, "moves": 0}


def run_cache(routes, reqs, cache, mode="skip"):
    """a per-token cache over the requests: the prompt run per `mode`, then every generated position scored"""
    if mode == "batch" and isinstance(cache, Freq):
        raise ValueError("freq has no batch rule (prompt calls admit nothing): use skip or token")
    L, K = routes.shape[1], routes.shape[2]
    hit, moves = [], 0
    for lo, s, e in reqs:
        if lo < s and mode == "batch":
            observe_counts(cache, flat_counts(routes[lo:s], cache.E).reshape(L, cache.E))
        elif lo < s and mode == "token":
            for row in _rows_of(routes, lo, s):
                cache.token(row)
        for row in _rows_of(routes, s, e):
            h, a = cache.token(row)
            hit.append(h / (L * K))
            moves += a
    return {"hit": np.asarray(hit), "moves": moves}


def plan_freq(res, c, n, k, excl):
    """Residency::plan_swaps without a cache, every layer at once: want = the top n by (count desc, resident first,
    id asc); incoming = want's absent experts in that order; out = residents outside want, not in `excl`, by (count
    asc, id asc); up to k[l] pairs (0 = all). -> per layer [(evicted, incoming)]"""
    L, Ex = res.shape
    ids = np.broadcast_to(np.arange(Ex), (L, Ex))
    want = np.lexsort((ids, ~res, -c), axis=-1)[:, :n]
    wset = np.zeros_like(res)
    wset[np.arange(L)[:, None], want] = True
    asc = np.lexsort((ids, c), axis=-1)
    out = []
    for l in range(L):
        w = want[l]
        inc = w[~res[l, w]].tolist()
        o = asc[l]
        o = [x for x in o[res[l, o] & ~wset[l, o]].tolist() if x not in excl[l]]
        kk = len(inc) if k[l] == 0 else min(len(inc), k[l])
        kk = min(kk, len(o))
        out.append(list(zip(o[:kk], inc[:kk])))
    return out


class TodayPlan:
    """the plan of serve's trickle tick: CROW_ADAPT_WINDOW=1 (gen.rs decay_window / window_layer). The first call
    sets the base (window 0, so it plans nothing); later calls fold the selections since the previous call into the
    decayed window and rank by it x1024, truncated."""

    def __init__(self, decay=ADAPT_DECAY):
        self.decay, self.base, self.ema = decay, None, None

    def __call__(self, now, res, n, k, excl):
        if self.base is None:
            self.base, self.ema = now.copy(), np.zeros(now.shape)
        d = (now - self.base).astype(np.float64)
        self.ema = self.ema * self.decay + d
        self.base = now.copy()
        c = np.floor(self.ema * WINDOW_SCALE).astype(np.int64)
        return plan_freq(res, c, n, k, excl)


class CachePlan:
    """Residency::plan_swaps_cached: the cache observes the rises of the cumulative counts since its previous call
    (expert_cache.rs observe_counts), then incoming = its VRAM set's absent experts, most valuable first, and out =
    the residents it does not hold, least valuable first, outside `excl`; up to k[l] pairs"""

    def __init__(self, cache):
        self.cache, self.last = cache, None

    def __call__(self, now, res, n, k, excl):
        if self.last is None:
            self.last = np.zeros_like(now)
        rises = np.where(now >= self.last, now - self.last, now)
        self.last = now.copy()
        c = self.cache
        observe_counts(c, rises)
        out = []
        for l in range(res.shape[0]):
            v, r = c.vram(l), set(np.flatnonzero(res[l]).tolist())
            inc = sorted(v - r, key=lambda e: c.key(l, e), reverse=True)
            o = sorted(r - v - excl[l], key=lambda e: c.key(l, e))
            kk = len(inc) if k[l] == 0 else min(len(inc), k[l])
            kk = min(kk, len(o))
            out.append(list(zip(o[:kk], inc[:kk])))
        return out


def run_trickle(routes, reqs, rows, experts, plan, spare=SPARE, every=EVERY, max_swaps=MAX_SWAPS, per_span=None):
    """serve's decode loop with the stream trickle (serve.rs: tick before decode_step i for i > 0, plan when
    i % every == 0; gen.rs trickle_tick: commit B of the swaps whose phase B is in flight, commit A of those whose
    phase A is, then plan into the free spares; trickle_drain after the last step). `rows` hold the logical hot set
    N; a swap's incoming expert is resident from the tick after the plan, its evicted one until the tick after
    that. Prompt runs and decode steps add to the cumulative selection counts the plan reads."""
    if isinstance(plan, CachePlan):
        assert every >= 2, "the cache plan observes every layer per tick only when no swap is in flight then"
    L, K = routes.shape[1], routes.shape[2]
    res = member(rows, experts)
    lidx = np.arange(L)[:, None]
    cnt = np.zeros(L * experts, dtype=np.int64)
    hit, swaps = [], 0
    in_a, in_b = [], []
    for q, (lo, s, e) in enumerate(reqs):
        cnt += flat_counts(routes[lo:s], experts)
        cold_pf = int((~res[lidx[None], routes[lo:s]]).sum()) if per_span is not None else 0
        dec, sw0, h0 = s, swaps, len(hit)
        for i in range(e - s):
            p = s + i
            if i > 0:
                for l, ev, _ in in_b:
                    res[l, ev] = False
                to_b, in_a = in_a, []
                for l, _, nw in to_b:
                    res[l, nw] = True
                in_b = to_b
                if i % every == 0:
                    cnt += flat_counts(routes[dec:p], experts)
                    dec = p
                    excl = [set() for _ in range(L)]
                    busy = [0] * L
                    for l, ev, nw in to_b:
                        excl[l].update((ev, nw))
                        busy[l] += 1
                    free = [spare - b for b in busy]
                    k = [f if max_swaps == 0 else min(max_swaps, f) for f in free]
                    pairs = plan(cnt.reshape(L, experts), res, len(rows[0]), k, excl)
                    for l, pl in enumerate(pairs):
                        if free[l] <= 0:
                            continue
                        for ev, nw in pl:
                            in_a.append((l, ev, nw))
                    swaps += len(in_a)
            hit.append(res[lidx, routes[p]].mean())
        cnt += flat_counts(routes[dec:e], experts)
        # trickle_drain: the in-flight swaps complete before the next request prefills
        for l, ev, _ in in_b:
            res[l, ev] = False
        for l, ev, nw in in_a:
            res[l, nw] = True
            res[l, ev] = False
        in_a, in_b = [], []
        if per_span is not None:
            hs = np.asarray(hit[h0:])
            per_span.append({"request": q + 1, "prompt_positions": s - lo, "generated": e - s,
                             "swaps": swaps - sw0, "cold_prompt": cold_pf,
                             "cold_generated": int(round(float(((1.0 - hs) * L * K).sum()))),
                             "hit_generated": float(hs.mean())})
    return {"hit": np.asarray(hit), "moves": swaps}


# ---------------------------------------------------------------- policies and the table

def parse_policy(name):
    """-> (kind, decay or None)"""
    kind, _, arg = name.partition(":")
    if kind in ("lfu", "trickle-lfu"):
        d = float(arg) if arg else 0.5
        if not 0.0 < d <= 1.0:
            raise ValueError("%s: decay %r outside (0, 1]" % (name, d))
        return kind, d
    if arg or kind not in ("static", "today", "lru", "freq", "trickle-lru"):
        raise ValueError("unknown policy %r (static, today, lru, lfu[:D], freq, trickle-lru, trickle-lfu[:D])" % name)
    return kind, None


def make_cache(kind, decay, rows, experts, halflife, margin, prior):
    if kind in ("lru", "trickle-lru"):
        return Lru(rows, experts)
    if kind in ("lfu", "trickle-lfu"):
        return Lfu(rows, experts, decay)
    return Freq(rows, experts, halflife, margin, prior)


def replay_budget(routes, reqs, sets, budget, policies, modes, experts, halflife=HALFLIFE, margin=MARGIN,
                  prior=None, rng=None, per_span=False, spare=SPARE, every=EVERY, max_swaps=MAX_SWAPS):
    """every policy at one budget -> list of rows (static first)"""
    L, K = routes.shape[1], routes.shape[2]
    rng = rng if rng is not None else np.random.default_rng(SEED)
    rows_b = adapt_rows(sets, budget, experts)
    runs = [("static", "-", "N %d" % budget, run_static(routes, reqs, rows_b, experts), None)]
    for name in policies:
        kind, decay = parse_policy(name)
        if kind == "static":
            continue
        if kind in ("today", "trickle-lru", "trickle-lfu"):
            n = budget - spare
            rows_n = adapt_rows(sets, n, experts)
            spans = [] if per_span and kind == "today" else None
            plan = TodayPlan() if kind == "today" else CachePlan(make_cache(kind, decay, rows_n, experts, halflife,
                                                                            margin, prior))
            r = run_trickle(routes, reqs, rows_n, experts, plan, spare, every, max_swaps, spans)
            runs.append((name, "counts", "N %d + %d spare" % (n, spare), r, spans))
            continue
        for mode in modes:
            if kind == "freq" and mode == "batch":
                continue
            cache = make_cache(kind, decay, rows_b, experts, halflife, margin, prior)
            runs.append((name, mode, "N %d" % budget, run_cache(routes, reqs, cache, mode), None))
    base = runs[0][3]["hit"]
    today = next((r["hit"] for name, _, _, r, _ in runs if name == "today"), None)
    out = []
    for name, mode, slots, r, spans in runs:
        g = len(r["hit"])
        d = r["hit"] - base
        nb = g // HE.BLOCK
        lo, hi = boot_ci(d, rng) if nb >= 2 and name != "static" else (float("nan"), float("nan"))
        row = {"policy": name, "prefill": mode, "slots": slots, "hit": float(r["hit"].mean()),
               "cold_per_token": float((1.0 - r["hit"]).mean() * L * K), "moves_per_token": r["moves"] / max(g, 1),
               "diff": float(d.mean()), "ci": [float(lo), float(hi)], "blocks": nb,
               "diff_today": None if today is None else float((r["hit"] - today).mean())}
        if spans is not None:
            row["per_span"] = spans
        out.append(row)
    return out


def verdict(rows):
    cands = [r for r in rows if r["policy"] not in ("static", "today")]
    if not cands:
        return {"go": None, "line": "no candidate policy replayed"}
    best = max(cands, key=lambda r: r["diff"])
    go = best["diff"] >= GO_DIFF
    ci = "n/a" if math.isnan(best["ci"][0]) else "[%+.3f, %+.3f]" % tuple(best["ci"])
    line = ("%s: best candidate %s (prefill %s) against static %+.3f (95 %% CI %s, %d blocks of %d); rule: >= "
            "%+.2f. Absolute: %.3f against %.2f (crow0924's %.2f of 2026-09-24 + %.2f)"
            % ("GO" if go else "NO GO", best["policy"], best["prefill"], best["diff"], ci, best["blocks"], HE.BLOCK,
               GO_DIFF, best["hit"], REF_HIT + GO_DIFF, REF_HIT, GO_DIFF))
    if best["diff_today"] is not None:
        line += "; against today %+.3f" % best["diff_today"]
    return {"go": go, "best": best["policy"], "prefill": best["prefill"], "diff": best["diff"], "line": line}


def print_budget(budget, rows, out=print):
    out("budget %d slots per layer" % budget)
    out("  %-18s %-7s %-17s %8s %10s %11s %9s  %-18s %9s" % ("policy", "prefill", "slots", "gen hit", "cold/tok",
                                                            "moves/tok", "vs static", "95% CI", "vs today"))
    for r in rows:
        ci = "" if math.isnan(r["ci"][0]) else "[%+.3f, %+.3f]" % tuple(r["ci"])
        vt = "" if r["diff_today"] is None else "%+9.3f" % r["diff_today"]
        out("  %-18s %-7s %-17s %8.3f %10.1f %11.2f %+9.3f  %-18s %9s" % (r["policy"], r["prefill"], r["slots"],
                                                                        r["hit"], r["cold_per_token"],
                                                                        r["moves_per_token"], r["diff"], ci, vt))
        for sp in r.get("per_span", []):
            out("      request %3d  prompt %6d  generated %5d  swaps %6d  cold prompt %8d  cold generated %7d  "
                "hit %.3f" % (sp["request"], sp["prompt_positions"], sp["generated"], sp["swaps"], sp["cold_prompt"],
                              sp["cold_generated"], sp["hit_generated"]))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("dump")
    ap.add_argument("maskdir")
    ap.add_argument("held")
    ap.add_argument("cal", nargs="*")
    ap.add_argument("--budgets", default=",".join(map(str, BUDGETS)))
    ap.add_argument("--policies", default=",".join(POLICIES))
    ap.add_argument("--prefill", default=",".join(MODES), help="skip, batch, token (lru / lfu / freq)")
    ap.add_argument("--sets", default=SETS, help="the hot set every policy starts from (a hot-set sidecar)")
    ap.add_argument("--halflife", type=float, default=HALFLIFE)
    ap.add_argument("--margin", type=float, default=MARGIN)
    ap.add_argument("--no-reprefill", action="store_true", help="a request prefills from the end of the last span")
    ap.add_argument("--per-span", action="store_true", help="per-request trickle figures of `today`")
    ap.add_argument("--json")
    a = ap.parse_args(argv)
    budgets = [int(x) for x in a.budgets.split(",") if x]
    policies = [p for p in a.policies.split(",") if p]
    for p in policies:
        parse_policy(p)
    modes = [m for m in a.prefill.split(",") if m]
    for m in modes:
        if m not in ("skip", "batch", "token"):
            ap.error("--prefill: %r is not skip, batch or token" % m)
    for b in budgets:
        if not SPARE < b <= E:
            ap.error("budget %d outside %d..%d" % (b, SPARE + 1, E))

    data = {}
    for nm in [a.held] + a.cal:
        r = load_routes(os.path.join(a.dump, nm + "-routes.bin"))
        eng = np.array(jload(os.path.join(a.dump, nm + "-counts.json"))["prefill_counts"])
        same = np.array_equal(counts(r), eng)
        spans = load_spans(os.path.join(a.maskdir, nm + "-mask.json"), len(r))
        print("self-test %-28s positions %7d generated %6d in %4d spans  dump counts == engine counters: %s"
              % (nm, len(r), sum(e - s for s, e in spans), len(spans), same))
        assert same, "the dump disagrees with the engine's own counters"
        data[nm] = (r, spans)
    sets = jload(a.sets)["sets"]
    assert len(sets) == LAYERS, "%s: %d rows, the model has %d layers" % (a.sets, len(sets), LAYERS)
    prior = None
    if a.cal:
        g = sum(gen_counts(r, sp, E) for r, sp in (data[c] for c in a.cal))
        prior = freq_prior(g, a.halflife, TOPK)
        print("freq prior: generated counts of %s" % ", ".join(a.cal))
    else:
        print("freq prior: none (no calibration files given), every score starts at 0")
    routes, spans = data[a.held]
    reqs = requests(spans, not a.no_reprefill)
    print("\nheld-out %s: %d positions, %d generated in %d requests; start set %s; prompt runs %s"
          % (a.held, len(routes), sum(e - s for s, e in spans), len(spans), os.path.basename(a.sets),
             "from the previous span's start (serve re-prefills the answer)" if not a.no_reprefill
             else "from the previous span's end"))
    rng = np.random.default_rng(SEED)
    doc = {"held": a.held, "cal": a.cal, "positions": int(len(routes)), "generated": sum(e - s for s, e in spans),
           "requests": len(spans), "reprefill": not a.no_reprefill, "sets": a.sets, "halflife": a.halflife,
           "margin": a.margin, "trickle": {"spare": SPARE, "every": EVERY, "max": MAX_SWAPS, "decay": ADAPT_DECAY},
           "budgets": {}}
    gos = []
    for b in budgets:
        rows = replay_budget(routes, reqs, sets, b, policies, modes, E, a.halflife, a.margin, prior, rng,
                             a.per_span)
        print()
        print_budget(b, rows)
        v = verdict(rows)
        print("  verdict at %d: %s" % (b, v["line"]))
        doc["budgets"][str(b)] = {"rows": rows, "verdict": v}
        if v["go"]:
            gos.append(b)
    print("\n#245 go rule (a candidate beats static crow0924 by >= %.2f): %s"
          % (GO_DIFF, "GO at budget %s" % ", ".join(map(str, gos)) if gos else "NO GO at every budget replayed"))
    if a.json:
        jdump(doc, a.json, indent=1)
        print("wrote %s" % a.json)
    return doc


if __name__ == "__main__":
    main()
