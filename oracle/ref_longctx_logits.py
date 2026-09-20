#!/usr/bin/env python3
"""#90: the f32 oracle for LONG sequences - the same reference `oracle/ref_engine_logits.py`
is (transformers 5.16.1, eager, f32, CPU, the PLE layer at index 1, the manual PLE
row-gather math), rewritten to run in CHUNKS with caches so 178,553 tokens of context
cost one sweep instead of a T x T mask, and so a run survives the night: resumable at
chunk granularity, logits written only at the planned rows.

WHY A SECOND REFERENCE SCRIPT. `ref_engine_logits.py` builds a [1,1,T,T] additive mask
and calls every layer over the whole sequence at once - honest, exact, and impossible at
six-digit T (a 178k x 178k f32 mask alone is 127 GB). This script feeds the SAME layer
implementations in chunks, carrying

  * a DynamicCache for the 12 full-attention layers (k/v) and their QSA indexer keys -
    the indexer still selects per query against the WHOLE prefix, f32, exactly as the
    architecture defines it; only the queries are batched by chunk;
  * the linear-attention layers' conv and recurrent states (transformers' own chunked
    gated-delta path, initial_state handed over per chunk);
  * the PLE layer's dilated-conv carry (the last 9 gated rows), with the n-gram uid
    table computed ONCE for the whole sequence - vectorized, bit-identical to
    ref_engine_logits.py's Python loops - and the embedding rows gathered per chunk
    from the shards on demand; the 102 GB table still never loads;
  * rotary over the full prefix positions each chunk, the way Qwen4ExpTextModel
    recomputes them from the cache's position history.

`self-test` proves the chunking against the UNCHUNKED path on a tiny random checkpoint
through the same loader (max logit delta printed), plus a resume-equivalence check.
`run` writes a SPARSE dump - `out.f32` + `out.rows.json` - that `tools/oracle-kld.py`
reads natively since #90. `verify` replays an existing dense reference dump row by row
for the day the checkpoint returns. `check-weights` reports what the machine's models/
dir can and cannot load - on this machine, 2026-09-20, it is the honest refusal: the
original checkpoint is gone (131 shard files referenced, 1 on disk).

USAGE
    .venv-oracle/bin/python oracle/ref_longctx_logits.py check-weights
    .venv-oracle/bin/python oracle/ref_longctx_logits.py run \
        --ids decode_out/oracle-longctx/longctx-170k-p1-ids.json \
        --plan decode_out/oracle-longctx/row-plan.json \
        --out decode_out/oracle-longctx/ref/gpu-logits.f32 --chunk 2048
    .venv-oracle/bin/python oracle/ref_longctx_logits.py run --ids CORPUS-ids.json \
        --all-rows --out OUT.f32            # dense dump, the corpus form
    .venv-oracle/bin/python oracle/ref_longctx_logits.py verify \
        --ids decode_out/oracle-tf298/tf298-ids.json \
        --against decode_out/oracle-tf298/ref-logits.f32 --rows 40
    .venv-oracle/bin/python oracle/ref_longctx_logits.py self-test
"""

import argparse
import copy
import json
import os
import sys
import time

import numpy as np
import torch

from transformers import DynamicCache
from transformers.models.qwen4_exp.configuration_qwen4_exp import Qwen4ExpTextConfig
from transformers.models.qwen4_exp.modeling_qwen4_exp import (
    Qwen4ExpTextDecoderLayer,
    Qwen4ExpTextGatedResidual,
    Qwen4ExpTextRotaryEmbedding,
)

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
DEFAULT_MODELS = os.path.join(REPO, "models", "Qwen3.8-Flash-Next-original")

PLE_LAYER = 1
EOS = 248044
NGRAM = 3
H_PER = 8
NHEADS = 16
SHARD_ROWS = 2_500_012          # uid // SHARD_ROWS -> shard, uid % -> row (p15)

# ---------------------------------------------------------------------- checkpoint


class Checkpoint:
    """The weight-loading half of ref_engine_logits.py: fetch f32 tensors from the
    shards named by the index, load into the transformers modules, never instantiate
    the full PLE table."""

    def __init__(self, models_dir):
        self.dir = models_dir
        index_path = os.path.join(models_dir, "model.safetensors.index.json")
        if not os.path.exists(index_path):
            raise SystemExit("no model.safetensors.index.json in %s" % models_dir)
        with open(index_path, encoding="utf-8") as fh:
            index = json.load(fh)
        self.weight_map = index["weight_map"]
        self.total_size = index.get("metadata", {}).get("total_size")
        self.shards = sorted(set(self.weight_map.values()))
        self._handles = {}

    def shard_path(self, shard):
        return os.path.join(self.dir, shard)

    def shard_exists(self, shard):
        return os.path.exists(self.shard_path(shard))

    def fetch(self, full):
        from safetensors import safe_open
        if full not in self.weight_map:
            raise SystemExit("the checkpoint index has no tensor %s" % full)
        path = self.shard_path(self.weight_map[full])
        if not os.path.exists(path):
            raise SystemExit("%s (tensor %s) is not on disk - run check-weights for the "
                             "full gap" % (self.weight_map[full], full))
        with safe_open(path, framework="pt", device="cpu") as f:
            return f.get_tensor(full).to(torch.float32)

    def load_into(self, module, prefix, skip=()):
        state = {}
        for name in module.state_dict().keys():
            if any(s in name for s in skip):
                continue
            full = prefix + name
            if full not in self.weight_map:
                raise SystemExit("missing in checkpoint: %s" % full)
            state[name] = self.fetch(full)
        module.load_state_dict(state, strict=False)
        return module.float().eval()

    def open_shard(self, shard):
        from safetensors import safe_open
        if shard not in self._handles:
            if not self.shard_exists(shard):
                raise SystemExit("PLE shard %s is not on disk - see check-weights" % shard)
            self._handles[shard] = safe_open(self.shard_path(shard), framework="pt",
                                             device="cpu")
        return self._handles[shard]


REQUIRED_PREFIXES = ("model.language_model.embed_tokens.",
                     "lm_head.",
                     "model.language_model.hyper_connection_mixer.",
                     "model.language_model.layers.")
VISUAL_PREFIX = "model.visual."


def required_keys(ckpt):
    return sorted(k for k in ckpt.weight_map
                  if not k.startswith(VISUAL_PREFIX) and k.startswith(REQUIRED_PREFIXES))


def cmd_check_weights(args):
    ckpt = Checkpoint(args.models)
    keys = required_keys(ckpt)
    present = {s for s in ckpt.shards if ckpt.shard_exists(s)}
    missing_shards = [s for s in ckpt.shards if s not in present]
    missing_keys = [k for k in keys if ckpt.weight_map[k] not in present]
    ple_shards = sorted({ckpt.weight_map[k] for k in keys
                         if ".ple_embedding.ngram_embedding.shard_" in k})

    print("index            %s" % os.path.join(args.models, "model.safetensors.index.json"))
    print("index total_size %s B" % format(int(ckpt.total_size or 0), ","))
    print("shard files      %d referenced, %d on disk, %d MISSING"
          % (len(ckpt.shards), len(present), len(missing_shards)))
    if ckpt.total_size and missing_shards:
        approx = int(ckpt.total_size) * len(missing_shards) // len(ckpt.shards)
        print("                 ~%s B of shard payload missing (proportional estimate)"
              % format(approx, ","))
    print("text tensors     %d required by the oracle, %d unloadable"
          % (len(keys), len(missing_keys)))
    print("PLE shards       %d referenced, %d on disk"
          % (len(ple_shards), sum(1 for s in ple_shards if ckpt.shard_exists(s))))
    print("visual tensors   %d in the index (not needed by the text oracle)"
          % sum(1 for k in ckpt.weight_map if k.startswith(VISUAL_PREFIX)))
    if missing_keys:
        print("")
        print("the f32 oracle CANNOT run here: the original checkpoint shards are absent.")
        print("first missing tensors:")
        for k in missing_keys[:8]:
            print("  %s  (in %s)" % (k, ckpt.weight_map[k]))
        if len(missing_keys) > 8:
            print("  ... and %d more" % (len(missing_keys) - 8))
        return 1
    print("the checkpoint looks complete for a text forward")
    return 0


# -------------------------------------------------------------------------- PLE

def ple_uid_table(ids, multipliers, vocab_sizes, offsets):
    """uids[T, 16] for the whole sequence - a vectorized, bit-identical port of
    ref_engine_logits.py's ple_ids_for: EOS-respecting n-gram shifts over
    [EOS]*(NGRAM-1) + ids, the 64-bit multiply/xor mix, per-head modulo."""
    n_ids = len(ids)
    hist = np.concatenate([np.full(NGRAM - 1, EOS, dtype=np.int64),
                           np.asarray(ids, dtype=np.int64)])
    n = len(hist)
    idx = np.arange(n)
    last_eos = np.where(hist == EOS, idx, -1)
    prev_incl = np.maximum.accumulate(last_eos)          # last EOS at or before i
    prev_seg = np.concatenate([[-1], prev_incl[:-1]])    # ... at or before i-1
    seg_start = prev_seg + 1
    pos_in_seg = idx - seg_start

    m = [int(x) & 0xFFFFFFFFFFFFFFFF for x in multipliers.tolist()]
    vs = [int(x) for x in vocab_sizes.tolist()]
    off = [int(x) for x in offsets.tolist()]
    mask64 = np.uint64(0xFFFFFFFFFFFFFFFF)

    # shifted[p][i] = history[i-p] when the position is >= p tokens into its segment
    shifted = []
    for p in range(NGRAM):
        src = idx - p
        valid = (pos_in_seg >= p) & (src >= 0)
        shifted.append(np.where(valid, hist[np.maximum(src, 0)], EOS))

    uids = np.empty((n_ids, NHEADS), dtype=np.int64)
    lo = NGRAM - 1                                    # out row j == history index lo+j
    for ngram in range(2, NGRAM + 1):
        mixed = np.zeros(n, dtype=np.uint64)
        for p in range(ngram):
            col = shifted[p].astype(np.uint64)
            part = (col * np.uint64(m[p])) & mask64
            mixed = part if p == 0 else (mixed ^ part)
        # uint64 -> int64 reinterprets the bits: the two's-complement wrap numpy
        # performs is exactly the original's `mixed - 2**64 if mixed >= 2**63`
        signed = mixed.astype(np.int64)
        start = (ngram - 2) * H_PER
        for h in range(H_PER):
            j = start + h
            uids[:, j] = signed[lo:] % vs[j] + off[j]
    return uids


class PleRef(torch.nn.Module):
    """ref_engine_logits.py's PleRef with sizes inferred from the checkpoint (so the
    tiny self-test checkpoint works), a batched shard-row gather, and a dilated-conv
    carry so it runs chunk by chunk."""

    def __init__(self, ckpt, prefix):
        super().__init__()
        self._ckpt = ckpt
        self.prefix = prefix
        self.key_proj = ckpt.fetch(prefix + "key_proj.weight").float()
        self.value_proj = ckpt.fetch(prefix + "value_proj.weight").float()
        self.norm_key = ckpt.fetch(prefix + "norm_key.weight").float()
        self.norm_query = ckpt.fetch(prefix + "norm_query.weight").float()
        self.norm_conv = ckpt.fetch(prefix + "norm_conv.weight").float()
        self.conv1d = ckpt.fetch(prefix + "conv1d.weight").float()
        self.multipliers = ckpt.fetch(prefix + "ple_embedding.layer_multipliers").to(torch.int64)
        self.vocab_sizes = ckpt.fetch(prefix + "ple_embedding.ngram_heads_vocab_sizes").to(torch.int64)
        self.offsets = ckpt.fetch(prefix + "ple_embedding.ngram_heads_offsets").to(torch.int64)
        self.emb_dim = self.value_proj.shape[1] // NHEADS     # value: [hidden, NHEADS*emb]
        self.hidden = self.value_proj.shape[0]
        self.streams = self.key_proj.shape[0] // self.hidden  # key: [streams*hidden, ...]

    def rms(self, x, w):
        return torch.nn.functional.rms_norm(x, (x.shape[-1],), (1 + w), eps=1e-6)

    def gather_rows(self, uids_chunk):
        """[chunk, 16] uids -> [chunk, 16, emb] rows, read from the shards on demand."""
        uniq, inv = np.unique(uids_chunk, return_inverse=True)   # uniq is ascending
        out = torch.empty((len(uniq), self.emb_dim), dtype=torch.float32)
        by_shard = {}
        for k in range(len(uniq)):
            u = int(uniq[k])
            by_shard.setdefault(u // SHARD_ROWS, []).append((k, u % SHARD_ROWS))
        for shard, items in by_shard.items():
            name = "%sple_embedding.ngram_embedding.shard_%d.weight" % (self.prefix, shard)
            fh = self._ckpt.open_shard(self._ckpt.weight_map[name])
            sl = fh.get_slice(name)
            rows_total = sl.get_shape()[0]
            for k, r in items:
                if r >= rows_total:
                    raise SystemExit("PLE row %d is beyond shard %s (%d rows)"
                                     % (r, name, rows_total))
                out[k] = sl[r:r + 1, :].to(torch.float32)[0]
        sel = out[torch.from_numpy(inv)]
        return sel.view(uids_chunk.shape[0], NHEADS, self.emb_dim)

    def forward(self, hidden, uid_rows, conv_carry=None):
        """hidden [1, chunk, streams*hidden]; uid_rows [chunk, 16, emb]; conv_carry is
        the previous chunk's last 9 gated rows (or None at the sequence start)."""
        T = hidden.shape[1]
        emb = uid_rows.reshape(T, NHEADS * self.emb_dim)
        key = self.rms(emb @ self.key_proj.t(), self.norm_key)
        val = emb @ self.value_proj.t()
        q = self.rms(hidden[0], self.norm_query)
        g = (key.view(T, self.streams, self.hidden) * q.view(T, self.streams, self.hidden)) \
            .sum(-1) / self.hidden ** 0.5
        gs = g.sign() * g.abs().clamp(min=1e-6).sqrt()
        sg = torch.sigmoid(gs)
        gated = (sg.view(T, self.streams, 1) * val.view(T, 1, self.hidden)) \
            .reshape(T, self.streams * self.hidden)
        gn = self.rms(gated, self.norm_conv)
        gnv = gn.view(T, self.streams, self.hidden)
        w = self.conv1d.reshape(self.streams, self.hidden, 4)
        pad = 9
        gnp = torch.zeros(T + pad, self.streams, self.hidden)
        gnp[pad:] = gnv
        if conv_carry is not None:
            gnp[:pad] = conv_carry
        acc = torch.zeros(T, self.streams, self.hidden)
        for k in range(4):
            acc += w[..., k].unsqueeze(0) * gnp[k * 3:k * 3 + T]
        acc = acc.reshape(T, self.streams * self.hidden)
        out = gated + torch.nn.functional.silu(acc)
        return out.unsqueeze(0), gnv[-pad:].clone()


# ------------------------------------------------------------------ the reference


class Ref:
    """The f32 reference model: the module graph of ref_engine_logits.py, loadable
    from any checkpoint directory (the real one or the self-test's tiny one)."""

    def __init__(self, models_dir):
        self.ckpt = Checkpoint(models_dir)
        with open(os.path.join(models_dir, "config.json"), encoding="utf-8") as fh:
            cfg = json.load(fh)
        text = cfg["text_config"] if "text_config" in cfg else cfg
        self.cfg = Qwen4ExpTextConfig.from_dict(text)
        self.cfg._attn_implementation = "eager"
        self.embed_w = self.ckpt.fetch("model.language_model.embed_tokens.weight")
        self.lm_head = self.ckpt.fetch("lm_head.weight")
        self.mixer = self.ckpt.load_into(
            Qwen4ExpTextGatedResidual(self.cfg, use_combine=False),
            "model.language_model.hyper_connection_mixer.")
        self.rotary = Qwen4ExpTextRotaryEmbedding(config=self.cfg).float().eval()
        self.ple = PleRef(self.ckpt,
                          "model.language_model.layers.%d.ple." % PLE_LAYER)
        # the PLE layer's constructor would allocate the full n-gram table only to be
        # dropped below - build THAT layer with a tiny n-gram vocab (ref_engine_logits)
        tiny_cfg = copy.deepcopy(self.cfg)
        tiny_cfg.ngram_vocab_size_base = 64
        self.layers = []
        for i in range(self.cfg.num_hidden_layers):
            layer = Qwen4ExpTextDecoderLayer(tiny_cfg if i == PLE_LAYER else self.cfg, i)
            layer = self.ckpt.load_into(layer, "model.language_model.layers.%d." % i,
                                        skip=("ple",))
            layer.ple = None
            layer.float().eval()
            self.layers.append(layer)

    def uid_table(self, ids):
        return ple_uid_table(ids, self.ple.multipliers, self.ple.vocab_sizes,
                             self.ple.offsets)

    @torch.no_grad()
    def logits_at(self, hidden_rows):
        """hidden [k, streams*hidden] -> [k, vocab] logits (the head, sparse rows)."""
        h = self.mixer(hidden_rows.unsqueeze(0))          # [1, k, hidden]
        return torch.nn.functional.linear(h[0], self.lm_head)

    def forward_full(self, ids):
        """The UNCHUNKED path - ref_engine_logits.py's loop, kept for self-test and
        for short sequences. Returns hidden states [1, T, streams*hidden]."""
        with torch.no_grad():
            T = len(ids)
            cfg = self.cfg
            device = self.embed_w.device
            input_ids = torch.tensor([ids], dtype=torch.long, device=device)
            emb = self.embed_w[input_ids]
            position_ids = torch.arange(T, device=device).view(1, 1, -1).expand(3, 1, -1)
            cos, sin = self.rotary(emb, position_ids)
            min_dtype = torch.finfo(torch.float32).min
            mask = torch.zeros(1, 1, T, T, dtype=torch.float32)
            mask = mask.masked_fill(
                torch.tril(torch.ones(T, T, dtype=torch.bool, device=device)) == 0,
                min_dtype)
            h = emb.repeat(1, 1, cfg.hc_count)
            for i, layer in enumerate(self.layers):
                if i == PLE_LAYER:
                    rows = self.ple.gather_rows(self.uid_table(ids))
                    h = h + self.ple(h, rows, None)[0]
                mixed_a, hyper_a, injw_a = layer.attn_hyper_connection(h)
                if layer.layer_type == "linear_attention":
                    o = layer.linear_attn(mixed_a, cache_params=None, attention_mask=None)
                else:
                    o, _ = layer.self_attn(mixed_a, (cos, sin), attention_mask=mask,
                                           past_key_values=None)
                h1 = hyper_a + (o.unsqueeze(-2) * injw_a.unsqueeze(-1)).flatten(-2)
                mixed_m, hyper_m, injw_m = layer.mlp_hyper_connection(h1)
                o2 = layer.mlp(mixed_m)
                h = hyper_m + (o2.unsqueeze(-2) * injw_m.unsqueeze(-1)).flatten(-2)
            return h


class Runner:
    """The CHUNKED path over the same weights: one DynamicCache for k/v, indexer keys
    and the linear-attention states, the PLE carry, and logits computed only at the
    planned rows."""

    def __init__(self, ref, chunk=2048):
        self.ref = ref
        self.chunk = chunk

    def run(self, ids, want_rows, sink, on_rows=None, cache=None, ple_carry=None,
            start_token=0, max_chunks=None):
        """Teacher-force `ids`; after each chunk, compute logits for the `want_rows`
        (absolute row ids, ascending) that fall inside it and hand them to `on_rows`.
        `cache`/`ple_carry`/`start_token` resume a previous run."""
        with torch.no_grad():
            cfg = self.ref.cfg
            device = self.ref.embed_w.device
            vocab = self.ref.lm_head.shape[0]
            if cache is None:
                cache = DynamicCache(config=cfg)
            uids = self.ref.uid_table(ids)
            want = sorted(want_rows)
            # rows below start_token belong to the resumed-from prefix: they are
            # either already written or lost to a crash between flush and state
            # save - the caller checks both cases
            skipped = [r for r in want if r < start_token]
            wi = len(skipped)
            done = start_token
            chunks_done = 0
            t0 = time.time()
            while done < len(ids):
                end = min(done + self.chunk, len(ids))
                seg = ids[done:end]
                emb = self.ref.embed_w[torch.tensor([seg], dtype=torch.long, device=device)]
                # rotary over the FULL prefix positions, the way Qwen4ExpTextModel
                # rebuilds them from the cache - the indexer needs cos/sin at every
                # key position, not just this chunk's queries
                pos_full = torch.arange(end, device=device).view(1, 1, -1).expand(3, 1, -1)
                cos, sin = self.ref.rotary(emb, pos_full)
                h = emb.repeat(1, 1, cfg.hc_count)

                ple_rows = self.ref.ple.gather_rows(uids[done:end])

                # additive causal mask [1, 1, chunk, T_so_far]: everything up to and
                # including each query's own position is visible (0), the rest min
                T_now = end
                c = end - done
                min_dtype = torch.finfo(torch.float32).min
                mask = torch.full((1, 1, c, T_now), min_dtype, dtype=torch.float32)
                mask[..., :done] = 0.0
                inner = torch.tril(torch.ones(c, c, dtype=torch.bool))  # j_local <= i_local
                mask[..., done:] = torch.where(inner, 0.0, min_dtype)

                for li, layer in enumerate(self.ref.layers):
                    if li == PLE_LAYER:
                        # the PLE layer's injection point: after layer 0, before
                        # layer 1 - exactly where ref_engine_logits.py adds it
                        ple_out, ple_carry = self.ref.ple(h, ple_rows, ple_carry)
                        h = h + ple_out
                    mixed_a, hyper_a, injw_a = layer.attn_hyper_connection(h)
                    if layer.layer_type == "linear_attention":
                        o = layer.linear_attn(mixed_a, cache_params=cache,
                                              attention_mask=None)
                    else:
                        o, _ = layer.self_attn(mixed_a, (cos, sin), attention_mask=mask,
                                               past_key_values=cache)
                    h1 = hyper_a + (o.unsqueeze(-2) * injw_a.unsqueeze(-1)).flatten(-2)
                    mixed_m, hyper_m, injw_m = layer.mlp_hyper_connection(h1)
                    o2 = layer.mlp(mixed_m)
                    h = hyper_m + (o2.unsqueeze(-2) * injw_m.unsqueeze(-1)).flatten(-2)

                rows_here = []
                while wi < len(want) and want[wi] < end:
                    rows_here.append(want[wi] - done)
                    wi += 1
                if rows_here:
                    logits = self.ref.logits_at(h[0, rows_here])     # [k, vocab] f32
                    if on_rows:
                        on_rows([r + done for r in rows_here], logits.numpy())
                done = end
                chunks_done += 1
                if sink:
                    # hand the CURRENT cache/carry so a checkpoint saved mid-run is
                    # the state this chunk actually produced
                    sink(done, chunks_done, time.time() - t0, cache, ple_carry)
                if max_chunks is not None and chunks_done >= max_chunks:
                    break
            return cache, ple_carry, done, skipped


# ------------------------------------------------------------------ output / resume


def load_plan_rows(plan_path):
    """row-plan.json (the #90 longctx form) or a row-groups file -> sorted row ids."""
    with open(plan_path, encoding="utf-8") as fh:
        doc = json.load(fh)
    rows = []
    for g in doc["groups"]:
        rows += [int(r) for r in g["rows"]]
    return sorted(set(rows))


def cmd_run(args):
    ids = json.load(open(args.ids, encoding="utf-8"))
    if not isinstance(ids, list) or not ids:
        raise SystemExit("%s is not a non-empty id array" % args.ids)
    if args.plan:
        want_rows = load_plan_rows(args.plan)
        bad = [r for r in want_rows if r < 0 or r > len(ids) - 2]
        if bad:
            raise SystemExit("plan rows %s are outside 0..%d for %d ids"
                             % (bad[:3], len(ids) - 2, len(ids)))
    elif args.all_rows:
        want_rows = list(range(len(ids) - 1))
    else:
        raise SystemExit("give --plan PLAN.json or --all-rows")

    ref = Ref(args.models)
    vocab = ref.lm_head.shape[0]
    out = args.out
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    sparse = args.plan is not None

    done_rows = []
    if sparse and os.path.exists(out + ".rows.json"):
        with open(out + ".rows.json", encoding="utf-8") as fh:
            done_rows = json.load(fh)["rows"]
        missing = [r for r in done_rows if r not in set(want_rows)]
        if missing:
            raise SystemExit("%s carries rows %s that the plan does not - wrong plan?"
                             % (out, missing[:3]))
    want_left = [r for r in want_rows if r not in set(done_rows)]
    if os.path.exists(out) and not os.path.exists(out + ".state.pt") and (
            sparse or not os.path.exists(out + ".rows.json")):
        # a partial dump with no checkpoint to line it up: appending would corrupt it
        raise SystemExit("%s exists without a state file - delete it (and its "
                         ".rows.json) to start again" % out)
    state = None
    start_token = 0
    if os.path.exists(out + ".state.pt"):
        state = torch.load(out + ".state.pt", weights_only=False)
        start_token = state["tokens_done"]
        if start_token > len(ids):
            raise SystemExit("the saved state is past the end of this id sequence")
        print("resuming from token %d (%d rows already written)"
              % (start_token, len(done_rows)))
    elif done_rows:
        raise SystemExit("%s has rows but no state file - delete both and start again"
                         % out)

    if sparse:
        # rows written so far must be a prefix of the plan in ascending order
        if done_rows != [r for r in want_rows if r in set(done_rows)]:
            raise SystemExit("the rows file is not an ascending prefix of the plan")

    def on_rows(rows, logits):
        with open(out, "ab") as fh:
            fh.write(np.ascontiguousarray(logits, dtype="<f4").tobytes())
        done_rows.extend(rows)
        with open(out + ".rows.json", "w", encoding="utf-8") as fh:
            json.dump({"rows": done_rows, "vocab": vocab, "ids_file": args.ids,
                       "chunk": args.chunk, "note": "sparse f32 oracle rows written by "
                       "oracle/ref_longctx_logits.py (#90)"}, fh)

    runner = Runner(ref, chunk=args.chunk)
    cache = state["cache"] if state else None
    ple_carry = state["ple_carry"] if state else None

    last_save = [state["chunks_done"] if state else 0]

    def sink(tokens_done, chunks_done, wall, cache_now, carry_now):
        print("tokens %7d/%d  rows %5d/%d  chunks %4d  %6.1f s"
              % (tokens_done, len(ids), len(done_rows), len(want_rows), chunks_done,
                 wall), flush=True)
        if chunks_done - last_save[0] >= args.save_every:
            torch.save({"cache": cache_now, "ple_carry": carry_now,
                        "tokens_done": tokens_done, "chunks_done": chunks_done,
                        "rows": done_rows}, out + ".state.pt")
            last_save[0] = chunks_done

    cache, ple_carry, tokens, skipped = runner.run(
        ids, want_left, sink, on_rows=on_rows, cache=cache, ple_carry=ple_carry,
        start_token=start_token, max_chunks=args.max_chunks)
    lost = [r for r in skipped if r not in set(done_rows)]
    if lost:
        raise SystemExit("the resume state is %d tokens in but rows %s were never "
                         "written - delete %s* and start again"
                         % (start_token, lost[:3], out))
    torch.save({"cache": cache, "ple_carry": ple_carry, "tokens_done": tokens,
                "chunks_done": last_save[0] + 1, "rows": done_rows}, out + ".state.pt")
    if not sparse:
        # dense form: the rows file names every row, so the dump reads like the old ones
        with open(out + ".rows.json", "w", encoding="utf-8") as fh:
            json.dump({"rows": done_rows, "vocab": vocab, "ids_file": args.ids,
                       "chunk": args.chunk, "dense": True,
                       "note": "dense f32 oracle rows (all rows) by "
                               "oracle/ref_longctx_logits.py (#90)"}, fh)
    print("done: %d rows x %d vocab -> %s" % (len(done_rows), vocab, out))
    return 0


def cmd_verify(args):
    """Replay an existing dense reference dump with the CHUNKED path and report the
    per-row deltas - the check that today's code and machine reproduce the 2026-09-05
    references (or how close they come)."""
    ids = json.load(open(args.ids, encoding="utf-8"))
    ref = Ref(args.models)
    vocab = ref.lm_head.shape[0]
    total = os.path.getsize(args.against) // (vocab * 4)
    n = min(args.rows, total, len(ids) - 1)
    got = {}
    runner = Runner(ref, chunk=args.chunk)
    runner.run(ids, list(range(n)), sink=None,
               on_rows=lambda rows, lg: got.update(dict(zip(rows, lg))))
    raw = np.fromfile(args.against, dtype="<f4", count=n * vocab).reshape(n, vocab)
    worst = 0.0
    deltas = []
    for r in range(n):
        d = float(np.abs(got[r].astype(np.float64) - raw[r].astype(np.float64)).max())
        deltas.append(d)
        worst = max(worst, d)
    top1 = sum(1 for r in range(n)
               if int(np.argmax(got[r])) == int(np.argmax(raw[r])))
    print("rows %d  max_abs_logit_delta %.3e  mean_max_delta %.3e  same top-1 %d/%d"
          % (n, worst, sum(deltas) / len(deltas), top1, n))
    if worst > args.tol:
        print("EXCEEDS tolerance %.1e - investigate before trusting new rows against "
              "this machine's dumps" % args.tol)
        return 1
    print("within tolerance %.1e: this machine's chunked path reproduces the stored "
          "reference" % args.tol)
    return 0


# ---------------------------------------------------------------------- self-test

TINY_CFG = {
    "attention_bias": False, "attention_dropout": 0.0,
    "bos_token_id": 0, "eos_token_id": 0,
    "full_attention_interval": 4, "hc_count": 2, "hc_lowrank": 8,
    "head_dim": 32, "hidden_act": "silu", "hidden_size": 64,
    "indexer_budget": 32, "indexer_compress_ratio": 4, "indexer_head_dim": 16,
    "indexer_kv_heads": 1, "indexer_n_heads": 2,
    "layer_types": ["linear_attention", "linear_attention", "linear_attention",
                    "full_attention", "linear_attention", "linear_attention"],
    "linear_conv_kernel_dim": 4, "linear_key_head_dim": 32, "linear_num_key_heads": 2,
    "linear_num_value_heads": 4, "linear_value_head_dim": 32,
    "make_ngram_vocab_size_divisible_by": 128,
    "mamba_ssm_dtype": "float32", "max_position_embeddings": 4096,
    "moe_intermediate_size": 16, "ngram_size": 3, "ngram_vocab_size_base": 4096,
    "norm_topk_prob": True, "num_attention_heads": 4, "num_experts": 4,
    "num_experts_per_tok": 2, "num_hidden_layers": 6, "num_key_value_heads": 2,
    "output_gate_type": "sigmoid", "output_router_logits": False,
    "partial_rotary_factor": 0.5, "ple_conv_kernel_size": 4, "ple_embed_dim": 64,
    "ple_layer_ids": [2], "rms_norm_eps": 1e-06, "rope_theta": 10000.0,
    "rope_parameters": {"mrope_interleaved": True, "mrope_section": [2, 2, 1],
                        "partial_rotary_factor": 0.5, "rope_theta": 10000.0,
                        "rope_type": "default"},
    "shared_expert_intermediate_size": 16, "vocab_size": 97,
}


def build_tiny_checkpoint(models_dir, seed=7):
    """A complete random text-only checkpoint in the layout `Ref` loads - every
    tensor name the real model has (minus the visual tower), at toy sizes."""
    import safetensors.torch as stt
    os.makedirs(models_dir, exist_ok=True)
    g = torch.Generator().manual_seed(seed)
    cfg = Qwen4ExpTextConfig.from_dict(TINY_CFG)

    def rand(*shape, scale=0.05):
        return (torch.randn(*shape, generator=g) * scale).to(torch.float32)

    W = {}
    H = cfg.hidden_size
    E = cfg.num_experts
    MI = cfg.moe_intermediate_size
    SI = cfg.shared_expert_intermediate_size
    W["model.language_model.embed_tokens.weight"] = rand(cfg.vocab_size, H)
    W["lm_head.weight"] = rand(cfg.vocab_size, H)
    P = "model.language_model.hyper_connection_mixer."
    HC = cfg.hc_count * H
    W[P + "hc_norm.weight"] = rand(HC)
    W[P + "input_mix_weight_down.weight"] = rand(cfg.hc_lowrank, HC)
    W[P + "input_mix_weight_up.weight"] = rand(HC, cfg.hc_lowrank)
    for i in range(cfg.num_hidden_layers):
        L = "model.language_model.layers.%d." % i
        for part in ("attn_hyper_connection.", "mlp_hyper_connection."):
            W[L + part + "hc_norm.weight"] = rand(HC)
            W[L + part + "input_mix_weight_down.weight"] = rand(cfg.hc_lowrank, HC)
            W[L + part + "input_mix_weight_up.weight"] = rand(HC, cfg.hc_lowrank)
            W[L + part + "block_inject_weight.weight"] = rand(cfg.hc_count, HC)
        if cfg.layer_types[i] == "linear_attention":
            kd = cfg.linear_num_key_heads * cfg.linear_key_head_dim
            vd = cfg.linear_num_value_heads * cfg.linear_value_head_dim
            W[L + "linear_attn.in_proj_qkv.weight"] = rand(2 * kd + vd, H)
            W[L + "linear_attn.in_proj_z.weight"] = rand(vd, H)
            W[L + "linear_attn.in_proj_b.weight"] = rand(cfg.linear_num_value_heads, H)
            W[L + "linear_attn.in_proj_a.weight"] = rand(cfg.linear_num_value_heads, H)
            W[L + "linear_attn.conv1d.weight"] = rand(2 * kd + vd, 1,
                                                      cfg.linear_conv_kernel_dim)
            W[L + "linear_attn.dt_bias"] = rand(cfg.linear_num_value_heads, scale=0.5) + 1.0
            W[L + "linear_attn.A_log"] = rand(cfg.linear_num_value_heads, scale=0.5) + 1.0
            W[L + "linear_attn.norm.weight"] = rand(cfg.linear_value_head_dim)
            W[L + "linear_attn.out_proj.weight"] = rand(H, vd)
        else:
            AH = cfg.num_attention_heads * cfg.head_dim
            KH = cfg.num_key_value_heads * cfg.head_dim
            W[L + "self_attn.q_proj.weight"] = rand(AH * 2, H)
            W[L + "self_attn.k_proj.weight"] = rand(KH, H)
            W[L + "self_attn.v_proj.weight"] = rand(KH, H)
            W[L + "self_attn.o_proj.weight"] = rand(H, AH)
            W[L + "self_attn.q_norm.weight"] = rand(cfg.head_dim)
            W[L + "self_attn.k_norm.weight"] = rand(cfg.head_dim)
            IH = (cfg.indexer_n_heads + cfg.indexer_kv_heads) * cfg.indexer_head_dim
            W[L + "self_attn.indexer.index_qk_proj.weight"] = rand(IH, H)
            W[L + "self_attn.indexer.q_layernorm.weight"] = rand(cfg.indexer_head_dim)
            W[L + "self_attn.indexer.k_layernorm.weight"] = rand(cfg.indexer_head_dim)
        W[L + "mlp.gate.weight"] = rand(E, H)
        W[L + "mlp.experts.gate_up_proj"] = rand(E, 2 * MI, H)
        W[L + "mlp.experts.down_proj"] = rand(E, H, MI)
        W[L + "mlp.shared_expert.gate_proj.weight"] = rand(SI, H)
        W[L + "mlp.shared_expert.up_proj.weight"] = rand(SI, H)
        W[L + "mlp.shared_expert.down_proj.weight"] = rand(H, SI)
        W[L + "mlp.shared_expert_gate.weight"] = rand(1, H)
    # the PLE layer at index 1 (sizes PleRef infers: value [H, 16*emb], key [HC, 16*emb])
    EMB = 4
    PL = "model.language_model.layers.1.ple."
    W[PL + "key_proj.weight"] = rand(HC, NHEADS * EMB)
    W[PL + "value_proj.weight"] = rand(H, NHEADS * EMB)
    W[PL + "norm_key.weight"] = rand(HC)
    W[PL + "norm_query.weight"] = rand(HC)
    W[PL + "norm_conv.weight"] = rand(HC)
    W[PL + "conv1d.weight"] = rand(HC, 1, 4)
    W[PL + "ple_embedding.layer_multipliers"] = torch.from_numpy(
        np.array([0x9E3779B97F4A7C15, 0xBF58476D1CE4E5B9, 0x94D049BB133111EB][:NGRAM],
                 dtype=np.uint64).astype(np.int64))
    W[PL + "ple_embedding.ngram_heads_vocab_sizes"] = torch.full((NHEADS,), 4096,
                                                                 dtype=torch.int64)
    W[PL + "ple_embedding.ngram_heads_offsets"] = torch.arange(NHEADS, dtype=torch.int64) * 4096
    W[PL + "ple_embedding.ngram_embedding.shard_0.weight"] = rand(4096 * NHEADS, EMB)

    shard = os.path.join(models_dir, "tiny-00001-of-00001.safetensors")
    stt.save_file(W, shard)
    with open(os.path.join(models_dir, "model.safetensors.index.json"), "w",
              encoding="utf-8") as fh:
        json.dump({"metadata": {"total_size": sum(t.numel() * 4 for t in W.values())},
                   "weight_map": {k: os.path.basename(shard) for k in W}}, fh)
    with open(os.path.join(models_dir, "config.json"), "w", encoding="utf-8") as fh:
        json.dump({"architectures": ["Qwen4ExpForConditionalGeneration"],
                   "model_type": "qwen4_exp", "text_config": TINY_CFG}, fh)
    return models_dir


def cmd_self_test(args):
    import tempfile
    keep = args.keep or tempfile.mkdtemp(prefix="oracle90-selftest-")
    models = build_tiny_checkpoint(os.path.join(keep, "model"))
    print("tiny checkpoint in %s" % models)
    torch.manual_seed(11)
    ids = [int(x) for x in torch.randint(3, 90, (257,), generator=torch.Generator()
                                         .manual_seed(11))]
    ref = Ref(models)

    # 1. the uid table: vectorized vs the ORIGINAL Python loops, must be EXACT.
    #    ref_engine_logits.py cannot be imported (it executes on import), so its
    #    ple_ids_for is copied VERBATIM here and diffed against.
    def ple_ids_for_original(ids_in, multipliers, vocab_sizes, offsets):
        history = [EOS] * (NGRAM - 1) + list(ids_in)
        n = len(history)
        shifted = []
        for shift in range(NGRAM):
            eos_pos = [-1] * n
            for i, v in enumerate(history):
                if v == EOS:
                    eos_pos[i] = i
            prev_incl = []
            run = -(2 ** 63)
            for i in range(n):
                run = max(run, eos_pos[i])
                prev_incl.append(run)
            row = []
            for i in range(n):
                prev = -1 if i == 0 else prev_incl[i - 1]
                seg_start = prev + 1
                pos_in_seg = i - seg_start
                src = i - shift
                valid = pos_in_seg >= shift and src >= 0
                row.append(history[max(src, 0)] if valid else EOS)
            shifted.append(row)
        out = []
        for i in range(NGRAM - 1, n):
            r = []
            for ngram in range(2, NGRAM + 1):
                start = (ngram - 2) * H_PER
                mixed = (shifted[0][i] & 0xFFFFFFFFFFFFFFFF)
                mixed = (mixed * (multipliers[0].item() & 0xFFFFFFFFFFFFFFFF)) & 0xFFFFFFFFFFFFFFFF
                for p in range(1, ngram):
                    mixed ^= (shifted[p][i] * (multipliers[p].item() & 0xFFFFFFFFFFFFFFFF)) & 0xFFFFFFFFFFFFFFFF
                m = mixed if mixed < 2 ** 63 else mixed - 2 ** 64
                for h in range(H_PER):
                    vs = vocab_sizes[start + h].item()
                    o = offsets[start + h].item()
                    r.append(m % vs + o)
            out.append(r)
        return out

    ids_with_eos = ids[:100] + [EOS] + ids[100:150] + [EOS, EOS] + ids[150:]
    want = ple_ids_for_original(ids_with_eos, ref.ple.multipliers,
                                ref.ple.vocab_sizes, ref.ple.offsets)
    got = ple_uid_table(ids_with_eos, ref.ple.multipliers, ref.ple.vocab_sizes,
                        ref.ple.offsets)
    want_a = np.array(want, dtype=np.int64)
    if want_a.shape != got.shape or not (want_a == got).all():
        d = np.argwhere(want_a != got)[:3]
        raise SystemExit("uid table MISMATCH at %s" % d.tolist())
    print("uid table: identical to ref_engine_logits.py's ple_ids_for (%d x %d)"
          % got.shape)

    # 2. chunked vs the unchunked path over all rows
    full_hidden = ref.forward_full(ids)                        # [1, T, HC]
    full_logits = ref.logits_at(full_hidden[0]).numpy()
    got = {}
    Runner(ref, chunk=64).run(ids, list(range(len(ids) - 1)), sink=None,
                              on_rows=lambda rows, lg: got.update(dict(zip(rows, lg))))
    worst = max(float(np.abs(got[r] - full_logits[r]).max()) for r in got)
    same_top1 = sum(1 for r in got
                    if int(np.argmax(got[r])) == int(np.argmax(full_logits[r])))
    print("chunked vs full: %d rows, max |delta logit| %.3e, same top-1 %d/%d"
          % (len(got), worst, same_top1, len(got)))
    if worst > args.tol:
        print("EXCEEDS tolerance %.1e" % args.tol)
        return 1

    # 3. resume: two chunks, checkpoint, a fresh Runner finishes; must equal (2)
    out = os.path.join(keep, "resume.f32")
    runner = Runner(ref, chunk=64)
    cache, carry, tokens, _ = runner.run(ids, [], sink=None, max_chunks=2)
    torch.save({"cache": cache, "ple_carry": carry, "tokens_done": tokens,
                "chunks_done": 2, "rows": []}, out + ".state.pt")
    state = torch.load(out + ".state.pt", weights_only=False)
    got2 = {}
    Runner(ref, chunk=64).run(ids, list(range(len(ids) - 1)), sink=None,
                              on_rows=lambda rows, lg: got2.update(dict(zip(rows, lg))),
                              cache=state["cache"], ple_carry=state["ple_carry"],
                              start_token=state["tokens_done"])
    worst2 = max(float(np.abs(got2[r] - full_logits[r]).max()) for r in got2)
    print("resumed run:   %d rows, max |delta logit| vs full %.3e" % (len(got2), worst2))
    if worst2 > args.tol:
        print("EXCEEDS tolerance %.1e" % args.tol)
        return 1

    # 4. the PRODUCTION path: cmd_run with a plan, interrupted at chunk 2, resumed,
    #    must byte-match an uninterrupted run
    plan = os.path.join(keep, "plan.json")
    with open(plan, "w", encoding="utf-8") as fh:
        json.dump({"groups": [{"name": "g0", "rows": list(range(0, 130, 2))},
                              {"name": "g1", "rows": list(range(130, 256, 3))}]}, fh)
    whole = os.path.join(keep, "whole.f32")
    # the ids go through a file so cmd_run reads them the way production does
    ids_file = os.path.join(keep, "ids.json")
    with open(ids_file, "w", encoding="utf-8") as fh:
        json.dump(ids, fh)
    whole = os.path.join(keep, "whole.f32")
    cmd_run(argparse.Namespace(ids=ids_file, plan=plan, all_rows=False, out=whole,
                               models=models, chunk=48, save_every=1,
                               max_chunks=None))
    part = os.path.join(keep, "part.f32")
    cmd_run(argparse.Namespace(ids=ids_file, plan=plan, all_rows=False, out=part,
                               models=models, chunk=48, save_every=1, max_chunks=2))
    cmd_run(argparse.Namespace(ids=ids_file, plan=plan, all_rows=False, out=part,
                               models=models, chunk=48, save_every=1,
                               max_chunks=None))
    a = open(whole, "rb").read()
    b = open(part, "rb").read()
    if a != b:
        n = min(len(a), len(b))
        i = next((k for k in range(n) if a[k] != b[k]), n)
        raise SystemExit("production resume MISMATCH: %d vs %d bytes, first diff at %d"
                         % (len(a), len(b), i))
    with open(whole + ".rows.json", encoding="utf-8") as fh:
        rows_w = json.load(fh)["rows"]
    with open(part + ".rows.json", encoding="utf-8") as fh:
        rows_p = json.load(fh)["rows"]
    if rows_w != rows_p:
        raise SystemExit("production resume rows.json MISMATCH")
    print("production resume: interrupted + resumed dump is byte-identical (%d rows)"
          % len(rows_w))
    print("SELF-TEST PASS (tolerance %.1e)" % args.tol)
    return 0


# --------------------------------------------------------------------------- main

def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("check-weights", help="report what this machine's checkpoint "
                                             "directory can actually load")
    p.add_argument("--models", default=DEFAULT_MODELS)
    p.set_defaults(fn=cmd_check_weights)

    p = sub.add_parser("run", help="the chunked f32 oracle over an id sequence")
    p.add_argument("--ids", required=True)
    p.add_argument("--plan", default=None, help="row-plan/row-groups JSON: the rows to "
                                                "write (sparse output)")
    p.add_argument("--all-rows", action="store_true",
                   help="write every teacher-forced row (dense output; corpus form)")
    p.add_argument("--out", required=True, help="the output dump path (.f32)")
    p.add_argument("--models", default=DEFAULT_MODELS)
    p.add_argument("--chunk", type=int, default=2048)
    p.add_argument("--save-every", type=int, default=4,
                   help="save the resume state every N chunks")
    p.add_argument("--max-chunks", type=int, default=None,
                   help="stop after N chunks (testing: exercise the resume path)")
    p.set_defaults(fn=cmd_run)

    p = sub.add_parser("verify", help="replay a stored dense reference and compare")
    p.add_argument("--ids", required=True)
    p.add_argument("--against", required=True, help="a stored ref-logits.f32")
    p.add_argument("--rows", type=int, default=32)
    p.add_argument("--chunk", type=int, default=2048)
    p.add_argument("--models", default=DEFAULT_MODELS)
    p.add_argument("--tol", type=float, default=2e-3)
    p.set_defaults(fn=cmd_verify)

    p = sub.add_parser("self-test", help="tiny random checkpoint: chunked == full, "
                                         "resume == uninterrupted")
    p.add_argument("--keep", default=None, help="keep the tiny checkpoint in this dir")
    p.add_argument("--tol", type=float, default=5e-4)
    p.set_defaults(fn=cmd_self_test)

    args = ap.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
