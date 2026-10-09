# Dynamic expert cache (#175)

`engine/src/expert_cache.rs` is the cache policy of plan step 16 (code part) and lever 3 of
measurement book F under root #169. It replaces the static hot-set cut with an exclusive two-tier
cache over the VRAM hot slots and the pinned tier, with NVMe as the miss tier. G1 had failed on the
static cut: simulated m 0.5217 (`runs/glm53-flash/step08/20261008-g1.md`). The same routing under a
dynamic LRU simulated m 0.278 / 0.185 / 0.060 at 108 / 145 / 216 slots per layer (#169, #172). The
policy only decides where experts live. The existing swap machinery in `engine/src/residency.rs`
carries the decisions out.

## 1. When it is on

| Container family (`Cnq::family`) | `CROW_EXPERT_CACHE` | Cache |
|---|---|---|
| `Glm5Next` | unset or empty | LRU |
| any | `lru`, `clock`, `clock:N` (N 1..=64), `lfu`, `lfu:D` (D in (0, 1]) | that policy (explicit opt-in) |
| any | `off` or `0` | off |
| `FlashNext`, `Qwen35Dense` | unset | off: `plan_swaps` runs its `e6f901a` body unchanged |

- A bare `lfu` takes decay 0.5, the default of the engine's existing selection window
  (`CROW_ADAPT_DECAY`). That decay is not measured for this policy.
- An active cache together with `CROW_ADAPT_WINDOW=1` is refused at boot, by name. The window
  passes decayed counts, while the cache reads the rises of the cumulative counts. `serve` sets the
  window itself when it is unset, so a `serve` boot with the cache on needs `CROW_ADAPT_WINDOW=0`.
- The engine still refuses `glm5_next_text` by name (`docs/architecture.md`, GLM row). Until a GLM
  container boots, the family default does nothing; only the opt-in reaches a running engine.

## 2. The policy

Every expert of a layer is in exactly one tier: VRAM (capacity V), pinned (capacity P) or NVMe.
When the cache is attached to a `Residency`, V is the logical hot set `n` (spares excluded) and P is
`experts - n`. The cache is seeded from the current hot sets in slot order, so the sidecar's
frequency order becomes the recency order.

| Policy | Rule |
|---|---|
| `Lru` | VRAM holds the V most recent experts, pinned the next P. A pinned hit is promoted and the VRAM victim takes its pinned slot. A miss enters VRAM, its victim moves to pinned, and the pinned victim drops to NVMe. |
| `Clock { admit }` | One second-chance ring per tier. The reference bit is set on use and on insert; a demoted expert enters the pinned ring with a clear bit. `admit` caps the NVMe admissions per layer per tick; later misses in the same tick bypass the cache. |
| `Lfu { decay }` | Each tick multiplies the layer's scores by `decay`, and an access adds its weight. VRAM holds the highest scores, pinned the next. Equal scores go to the more recent access. |

- **Scope.** `PerLayer` is the only scope `Residency` can execute, since its slots are per layer.
  `Global` is one pool of `layers x V` + `layers x P` slots, for simulation and counters only.
- **Counters.** `[vram, pinned, nvme]` per layer: the tier each access was served from.
- **One tick.** `ExpertCache::observe_counts` takes the cumulative per-expert counts
  (`Gen::drain_sel_counts`). Every expert whose count rose since the previous call for that layer
  is one access. Accesses are ordered by the size of the rise, then by id, and an LFU access is
  weighted by its rise. In a one-token decode tick every rise is 1, so the order is ascending id:
  the order of the routing files that `tools/glm_tier_sim.py` reads.
- **A skipped tick.** `trickle_tick` plans only layers that have a free spare slot. A layer it
  skips loses no accesses: they arrive in the next tick, with coarser recency.

## 3. The plan and who executes it

`Residency::plan_swaps` is now `&mut self`. The three callers in `gen.rs` (`adapt_hot_set`,
`adapt_tick`, `trickle_tick`) already hold `&mut self` and compile unchanged.

- **Without a cache** it runs the old body.
- **With a cache** it calls `plan_swaps_cached`, which feeds the counts to the cache and returns the
  same `(slot, evicted, incoming)` triples:
  - incoming: the cache's VRAM members that are not resident and sit in `cold_index`, most valuable
    first;
  - out: the residents the cache demoted, least valuable first;
  - `k` and `excl` work as before.

`swap_stream_a` and the two commits, `swap_in` and `swap_in_bundled` execute the triples
unchanged: the policy's promotion is their exact three-way exchange.

## 4. Cross-check against the simulator

Exclusive two-tier LRU is one LRU stack split at V (Mattson et al. 1970). Two identities follow:

- NVMe misses = `lru_reads(W = V + P)`;
- VRAM misses (pinned + NVMe) = `lru_reads(W = V)`.

`lru_reads` is in `tools/glm_tier_sim.py`. The test `lru_counters_equal_glm_tier_sim_lru_reads`
checks both identities on a synthetic trace. The fixture numbers came from the script below, run on
2026-10-08 under `.venv-oracle` at `e6f901a` (`python -I cache_fixture.py <repo>/tools`):

```python
import sys
import numpy as np
sys.path.insert(0, sys.argv[1])
import glm_tier_sim as ts

M = (1 << 64) - 1
L, E, K, T = 42, 288, 8, 600

def trace():
    x = 0x9E3779B97F4A7C15
    out = np.empty((T, L, K), np.uint16)
    for t in range(T):
        base = (t // 200) * 16
        for l in range(L):
            got = []
            while len(got) < K:
                x ^= (x << 13) & M
                x ^= x >> 7
                x ^= (x << 17) & M
                e = (base + l * 7 + x % 48) % E if (x >> 32) % 10 < 7 else (x >> 8) % E
                if e not in got:
                    got.append(e)
            out[t, l] = sorted(got)
    return out

r = trace()
nv = np.ones(r.shape, bool)          # every visit NVMe-tier: lru_reads is a pure W-slot LRU
for w in (0, 25, 37, 108, 145):
    print("W", w, int(ts.lru_reads(r, nv, w).sum()))
```

| W | 0 | 25 | 37 | 108 | 145 |
|---|---|---|---|---|---|
| `lru_reads`, summed | 201,600 | 151,180 | 125,281 | 44,507 | 35,092 |

The engine counters match at (V, P) = (25, 83) and (37, 108). The test also pins the trace's first
row, `[6, 9, 31, 36, 42, 44, 45, 66]`, so the Rust and Python generators cannot drift apart without
a red test.

## 5. Tests

Run without a GPU:

```
cd engine
CARGO_BUILD_JOBS=4 cargo test --lib -- expert_cache residency::
```

The run gives 21 passed: 11 in `expert_cache`, 3 new and 7 existing in `residency`.

| Test | Checks |
|---|---|
| `lru_hand_sequence`, `clock_hand_sequence`, `clock_admission_limit_bypasses_the_rest_of_the_tick`, `lfu_with_decay_hand_sequence` | counts and final tiers worked out by hand in each doc comment |
| `tiers_stay_exclusive_under_every_policy_and_scope` | capacities, occupancy and CLOCK rings after every token; all policies, both scopes, a partial seed |
| `capacity_zero_is_all_misses` | V = P = 0: every access is NVMe, nothing is cached |
| `lru_counters_equal_glm_tier_sim_lru_reads` | section 4 |
| `with_the_policy_off_plan_swaps_is_unchanged` | 400 randomized layers against a frozen copy of the `e6f901a` body |
| `with_the_cache_on_the_swaps_execute_to_the_cache_vram_set` | after the triples are executed, residents = the cache's VRAM set, and `cold_index` is the complement |
| `with_the_cache_on_k_and_excl_still_bound_the_plan`, `observe_counts_reads_rises_of_cumulative_counts`, `seed_places_the_given_tiers_most_recent_first`, `a_global_pool_over_one_layer_is_the_per_layer_pool`, `the_policy_is_on_for_glm5_next_or_by_opt_in_only` | the interfaces around the policy |

**Mutation check (2026-10-08).** Each hunk was broken in turn and turned its tests red; restoring
it turned them green again:

| Broken hunk | Red tests |
|---|---|
| the `plan_swaps` dispatch | `with_the_cache_on_the_swaps_execute_...` |
| the old body's tie rule | `with_the_policy_off_...` |
| LRU demotion | `lru_hand_sequence`, the cross-check |
| CLOCK free-slot promotion | `tiers_stay_exclusive_...` |
| LFU decay | `lfu_with_decay_hand_sequence` |
| NVMe counting | `capacity_zero_is_all_misses` |

## 6. Limits

- **No NVMe reads.** `Residency` has no NVMe tier (#149). With the cache attached today, P is
  `experts - n`, so no miss ever reaches NVMe.
  The glm5_next path is the exception (2026-10-09): `engine/src/glm5_tiers.rs` (`glm5_run`) uses
  this cache per MoE layer with the #159 plan's V and P and executes its decisions, NVMe misses
  included (`docs/glm5-model.md` section 6).
- **Not replayed in the engine.** That the engine reaches the simulated m on a GLM replay is a
  later GPU step of #169. Nothing here has run on a GPU or booted a model.
- **Feed granularity.** The feed is the counter rise per tick. Ticks span one token at decode and
  the whole prompt after prefill. A per-token device feed of the routed ids would touch `gen.rs`
  and the kernels.
