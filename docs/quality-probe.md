# The quality probe

`tools/quality-probe.py` is the characterization harness of issue #75 and step 1 of the requant
series: a re-runnable reading of ANSWER QUALITY at the real operating row, against any
OpenAI-compatible `/v1/chat/completions` endpoint, so the same command measures `serve` and
llama-server. It changes nothing in the engine, the converter or the quant. Its job is to pin
today's behaviour so that a later change to the quant has something to be compared against.

The symptom it was built around is robin's: German misspelling and dialect-like non-words that
grow with answer length - "Erschd kuck ich mir alles an", "Kunstgewerkds", "Jahrundert" - and a
hex code that comes back one digit wrong. #74 ruled the sampler and the thinking switch out, so
what is left to measure is the text itself.

## 1. What it measures

Six reference-free metrics, each a pure function of one answer, each under unit test in
`tools/test_quality_probe.py` (65 tests, no server, no GPU):

| metric | what it counts |
|---|---|
| non-word rate | words the hunspell dictionary does not know, per 1000 checked words, German and English separately, with the loanword share reported apart |
| exact-literal fidelity | share of literals demanded by the prompt that came back byte for byte, plus every near-miss at case-folded edit distance 1 or 2 |
| JSON | whether the whole answer is a valid document, and whether it has the shape the prompt demanded |
| repetition | longest immediately repeated n-gram run, and the distinct-word ratio |
| foreign script | CJK, Cyrillic, Arabic, Hebrew, Devanagari characters in a German or English answer (Greek is counted apart: an alpha in a technical text is ordinary) |
| length | words, characters, lines and the finish reason |

What never reaches the dictionary: anything inside a fenced code block or an inline code span,
any URL, and any whitespace-delimited token that carries a digit or one of `_ / \ # @ < > { } = |`.
Hex codes, versions, paths and identifiers are therefore not spelling. Hyphenated compounds are
split and each part is checked; single letters and ALL-CAPS tokens are dropped.

## 2. The prompt set and the row

`tools/quality-probe-prompts.json`, version 1, twelve tasks: four long German prose tasks that
ask for 900 to 1100 words of continuous specialist text with no list to hide in, two English
ones as the language control, two tasks that must reproduce literals given in the prompt many
times, two that must return strict JSON of a given shape, and two short German plans in
numbered steps - the agentic shape without tools. The file is evidence: a prompt edited between
two runs makes the two runs incomparable, so it carries a version and changes with a bump.

The row is the client's, not a measurement row: temperature 1.0, top_p 0.95, top_k 20,
presence_penalty 0.0, min_p 0.0, `max_tokens` 2600, `stream` false, a fixed seed list, and
thinking OFF through the door each server owns - the field ABSENT for `serve` (#74: an absent
`reasoning_effort` is the render of record) and a top-level `"reasoning_effort": "none"` for
llama-server. `--reasoning-effort` repeats a subset with thinking on, and only `content` is ever
scored; `reasoning_content` is stored in the record and never measured.

## 3. The dictionaries

`hunspell` is installed on this machine and no dictionary is. The probe fetches `de_DE_frami`
and `en_US` from the LibreOffice dictionaries repository into `~/.cache/crow-nest/dict/` and
records the URL, the byte count and the sha256 of every file it used in `run.json`. They are
never committed. What the baseline below used, fetched 2026-09-18, against
`Hunspell 1.7.3`:

| file | sha256 | bytes |
|---|---|---|
| `de/de_DE_frami.aff` | `646bf3333ac69c23e9d794533ee5241d6f755c359e8fe10a648f87613743d594` | 19,067 |
| `de/de_DE_frami.dic` | `4ca3c958b0e5545910999bc246f668840bf8ede3df8e5e6790d05edd5a586c38` | 4,356,903 |
| `en/en_US.aff` | `e746c882dd6f303c2c46e7452804b9201115a6942cfeb15f18f8edf774d2e24e` | 3,205 |
| `en/en_US.dic` | `f0b1a234bd178bdd01875b2a392a9647f888b8fe879f79c52aae62c2759b3647` | 551,762 |

All four from `https://raw.githubusercontent.com/LibreOffice/dictionaries/master/`.

## 4. How to run both arms

ONE engine on the card at a time; this machine freezes with two. Before every start,
`ps -C serve,decode,parity,llama-server` must be empty and the card near 700 MiB.

Arm A, this engine:

```
tools/serve-linux.sh --port 8099 > decode_out/quality-probe/serve-A.log 2>&1 &
# up when curl -s http://127.0.0.1:8099/health answers, 30 to 60 s
tools/quality-probe.py --label A1-crow --base-url http://127.0.0.1:8099
kill -INT $(ps -C serve -o pid=)          # the real process, not only the wrapper
```

Arm B, llama-server with the Unsloth UD-Q2_K_XL quant (the operating point of `crow`,
port 8083):

```
~/.local/share/crow/venv/bin/python ~/.local/share/crow/tools/start-server.py flash-next-q2-k-xl \
    > decode_out/quality-probe/llama-B.log 2>&1 &
tools/quality-probe.py --label B1-llama --base-url http://127.0.0.1:8083
kill -INT $(ps -C llama-server -o pid=)
```

The engine is auto-detected from `/props` (`build` for `serve`, `build_info` for llama-server);
`--engine` overrides it. Then:

```
tools/quality-probe.py --compare A1-crow B1-llama
tools/quality-probe.py --rescore A1-crow      # metrics recomputed from the stored texts
```

`--rescore` exists because the generations are the expensive half and they are kept verbatim:
a metric corrected afterwards does not cost a re-run, and every arm can be brought to the same
scorer, which is what makes two arms comparable at all. All four runs below were GENERATED at
commit `c1cf95c` and SCORED by the scorer of record at commit `2bc92f2`; each `run.json` names
both.

## 5. The baseline, 2026-09-18

Measured 2026-09-18 on RTX 5090 / Arch Linux, generated at repo commit `c1cf95c` and scored
at `2bc92f2`, prompt set version 1, 36 generations per run (12 prompts x 3 seeds), thinking
off, one engine on the card at a time.

| | A1 `serve`, CNQ4.5-M | B1 llama-server, UD-Q2_K_XL |
|---|---|---|
| model | `Qwen3.8-Flash-Next-CNQ4.5-M` | `Qwen3.8-Flash-Next-UD-Q2_K_XL` |
| seeds | 1201, 1202, 1203 | 1201, 1202, 1203 |
| non-word rate DE per 1000, pooled | 20.49 | 14.37 |
| non-word rate DE per 1000, pooled, loanwords removed | 16.05 | 8.49 |
| **non-word rate DE, the four LONG PROSE tasks only, loanwords removed** | **16.40** | **8.21** |
| non-word rate DE per generation, mean and spread | 20.90 (9.01 to 54.88) | 16.46 (3.37 to 54.19) |
| non-word rate EN per 1000, pooled | 7.74 | 8.72 |
| non-word rate EN, the two long prose tasks only | 6.83 | 9.16 |
| exact literals, share of demanded occurrences | 0.996 (0.966 to 1.000) | 1.000 (1.000 to 1.000) |
| near-miss literals | 3 kinds, incl. `v3.201` and `v3.211` for `v3.2.1` | none |
| JSON: whole answer a valid document / shape ok | 5 / 4 of 6 | 6 / 6 of 6 |
| distinct-word ratio | 0.535 (0.400 to 0.748) | 0.546 (0.455 to 0.900) |
| longest immediate repeat run, max over the arm | 5 | 2 |
| generations with foreign-script characters | 1 of 24 (2 chars) | 2 of 24 (5 chars) |
| words per answer, mean | 703 | 729 |
| wall clock per generation, mean | 19.4 s | 25.8 s |
| whole run | 699 s | 929 s |

The pooled rate is flags over words, not a mean of per-generation means, so a long answer
weighs what it is worth. Per-generation spread is given beside every mean.

### The non-word rate split by task class, loanwords removed

| task class | A1 `serve` | A3 `serve`, other seeds | B1 llama-server |
|---|---|---|---|
| DE long prose, 4 tasks | 16.40 | 15.76 | 8.21 |
| DE literal task | 14.66 | 20.52 | 6.89 |
| DE agentic plans, 2 tasks | 14.75 | 20.72 | 11.61 |
| EN long prose, 2 tasks | 6.58 | 10.24 | 9.16 |
| EN literal task | 12.23 | 8.92 | 5.84 |

## 6. The noise floor

Two questions, and they have different answers.

**Same arm, same seeds.** `serve` reseeds per request, and it reproduces BYTE FOR BYTE: A2 is
A1 repeated with the same seed list and all 36 generations are identical strings, so every
aggregate agrees to the last digit. Run-to-run noise at a fixed seed is therefore ZERO on this
engine, and no difference between two runs of it can be explained by noise.

**Same arm, different seeds.** A3 is the same arm on seeds 4401, 4402, 4403 - this is the only
noise a run of this probe actually carries, and it is seed and prompt variance, not machine
noise:

| reading | A1 | A3 | seed noise |
|---|---|---|---|
| DE long prose, loanwords removed | 16.40 | 15.76 | 0.64, about 4 % |
| DE pooled over all German tasks, loanwords removed | 16.05 | 16.79 | 0.74 |
| EN long prose | 6.58 | 10.24 | 3.66, about 56 % |
| literal occurrence share | 0.996 | 0.959 | 0.037 |
| JSON valid document | 5 of 6 | 4 of 6 | 1 of 6 |

So: **a difference on the German long-prose non-word rate has to be larger than about 1 per 1000
to mean anything; the arms differ by 8.2, which is thirteen times that.** The English rate,
the literal share and the six JSON generations are all too noisy at three seeds to decide
anything on their own - the English gap between the arms (1 to 2.5 per 1000) is smaller than
English's own seed noise, and the JSON and literal readings rest on six and nine generations.

## 7. What hunspell costs, hand-checked

Hunspell's weakness on German is compounds and proper names, and it is large. 30 flagged
German prose words were drawn at random from each arm (seed 175, loanwords already excluded)
and classified by hand:

| arm | real errors | borderline | hunspell false positives |
|---|---|---|---|
| A1 `serve` | 18 of 30 | 1 | 11 of 30, about 37 % |
| B1 llama-server | 4 of 30 | 1 | 25 of 30, about 83 % |

The false positives are of one kind on both arms - legitimate German compounds and names the
dictionary does not carry: `Kriechverhalten`, `Verteilnetz`, `haustechnischen`, `Sammlungsname`
on arm A, `Uferfiltratquellen`, `Cachespeicherkapazität`, `Überzugsmaterialien`,
`jugendstilistischen`, `Legrain`, `Didone` on arm B. The real errors are not of one kind at all.
Arm A's, verbatim with their context:

- `...die es über **Jahrunderte** innehatte.` (`de-prose-buchbinderei`, seed 1202)
- `...ist die Frage der lesenden Distanz zu **erör**. Das Plakat muss **zweifichtig** funktionieren.` (`de-prose-plakat`, 1203)
- `...eine Kaskade von weiteren Anfragen in die nächsttieferen **Hierarchieeb** auslöst` (`de-prose-speicher`, 1203)
- `...ein zu hartes Wasser zu **unkontrolliertenierenden** Kalkablagerungen` (`de-prose-wasserwerk`, 1202)
- `...der auf Wiederholung und Standardisierung **auswar**, während der verblieben, wertschätzende Anteil` (`de-prose-buchbinderei`, 1202)
- `...die durch den Farbwahl im Druck **nachah** dem Betrachter unmittelbar nahe gebracht werden muss` (`de-prose-plakat`, 1203)
- `...Die Symptome sind **unspektifisch** und langsam` (`de-prose-wasserwerk`, 1203)
- `...vom **Aufbereitenwerk** bis zum letzten Entnahmeventil` (`de-prose-wasserwerk`, 1202)
- `...bevor es in die Stufen des **Hochbehältters**` (`de-prose-wasserwerk`, 1203)
- `...muss das Plakat eine zweite, **differenziierte** Lesestufe bieten` (`de-prose-plakat`, 1203)

Arm B's four, for comparison, are ordinary slips and not broken words: `Korretheit` for
Korrektheit, `Nachfluszahlen` for Nachflusszahlen, and two lowercased nouns
(`...Grafikprozessor operationen ausführen kann`, `...das Zentrum der gravität`).

The consequence is that the measured gap UNDERSTATES the real one. Carrying the hand-checked
shares through: arm A's 16.40 per 1000 is about 9.8 real errors per 1000 words, arm B's 8.21 is
about 1.4 - a factor near seven where the raw metric says two. The probe is therefore a
conservative instrument on this axis, and the number it reports is a floor on the difference,
not the difference.

## 8. What this probe can and cannot decide

It CAN say, on this prompt set and these seeds:

- that on long GERMAN prose the two arms differ far beyond the seed noise: the CNQ4.5-M arm is
  higher on 12 of 12 matched (prompt, seed) pairs on the raw rate and on 11 of 12 once loanwords
  are removed (the one exception is `de-prose-speicher` seed 1201, where arm B is 1.86 per 1000
  higher), carrying about twice the flagged-word rate overall and, after the hand check, several
  times the rate of genuinely broken words;
- that arm A's broken words are of a kind arm B does not produce at all: truncated stems
  (`erör`, `nachah`, `Hierarchieeb`), doubled letters (`Hochbehältters`, `überschaubbaren`) and
  invented compounds (`Detailverwoblung`, `atmosphärdichten`);
- that exact-literal fidelity separates the arms in the same direction: arm B reproduced every
  demanded literal in all nine literal generations with not one near-miss, while arm A wrote
  `v3.201` and `v3.211` where the prompt said `v3.2.1`, four times in one answer;
- that `serve` is byte-reproducible at a fixed seed, which makes every future re-run of this
  probe against it an exact comparison.

It CANNOT decide:

- **why.** It is reference-free. It reads answers, not distributions, so it cannot attribute a
  difference to the dense path, to the router, to the sampler or to the chat template. Two
  engines differ here in quant, kernels, tokenizer path and chat template at once.
- **anything on English.** The English gap is inside English's own seed noise, and the `en_US`
  dictionary flags British spellings (`moulding`, `mould`) as errors, which one of the two
  English prompts invites.
- **the JSON, repetition, foreign-script and literal readings on their own.** Six JSON
  generations, nine literal generations and two foreign-script sightings per arm are too few
  to carry a conclusion; they are recorded so that a regression in them would be visible, not
  so that today's values can be quoted.
- **the quality of a thinking answer.** `--reasoning-effort` is wired on both doors and unit
  tested, and it has NOT been run against either server; the baseline above is thinking off.
- **how far this quant is from the original.** The literature's reading of a quant is
  reference-based: mean KL divergence and top-1 agreement against the BF16 model, teacher-forced
  on chat and agentic text (arXiv 2407.09141, llama.cpp discussion 4110), and the
  greedy-continuation agreement over 32 tokens Unsloth publishes. All three need the BF16
  originals, which are not on this machine. That is a later step of this series and nothing in
  this document is a substitute for it.

## 9. Where the records are

`decode_out/quality-probe/<label>/`, tracked for these runs because they are the evidence
the requant series is measured against: `records.jsonl` (one JSON object per generation with the
full text, every metric, the sampling that was sent and the timings), `run.json` (the run header
with the dictionary provenance and the aggregate) and `summary.md` (the table a human reads).
The server logs beside them are not tracked. Labels: `A1-crow`, `A2-crow` (the same-seed
repeat), `A3-crow` (the other seed list) and `B1-llama`.

#77 (2026-09-18) added two more on the same prompt set, the same seeds and the same row, both
generated at commit `22ec6b4`: `C0-crow-control` (the dense-BF16 overlay built from the
container's own dequantized NVFP4 — the control, which reads 15.81 on the German long-prose row
against A1's 16.40, inside the seed noise) and `C1-crow-dense-bf16` (the same overlay built from
the BF16 originals of #76, 14.61). `docs/dense-overlay.md` section 5.4 is the table and the
reading; nothing in section 5 above moves, because those runs are a different arm and not a
re-measurement of this one.
