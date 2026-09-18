#!/usr/bin/env python3
"""#75: the unit tests of the quality probe's metric functions.

  python tools/test_quality_probe.py          # no server, no GPU, no network

Every metric of `tools/quality-probe.py` is a pure function of the answer text, and this is
where each of them is pinned. The one impure metric, `hunspell_flag`, is tested through a
fake binary rather than the real dictionaries: what has to hold is the contract (one word per
line in, flagged words out, nothing that was not sent), not the German dictionary's content.

The module is loaded by path because the tool's file name carries a hyphen.
"""

import importlib.util
import json
import stat
import tempfile
import unittest
from pathlib import Path

TOOLS = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location("quality_probe", TOOLS / "quality-probe.py")
qp = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(qp)


class StripAndTokenize(unittest.TestCase):
    def test_a_fenced_code_block_is_not_prose(self):
        text = "Vorher.\n```rust\nlet Kunstgewerkds = 1;\n```\nNachher."
        self.assertNotIn("Kunstgewerkds", qp.checkable_words(text))
        self.assertIn("Vorher", qp.checkable_words(text))
        self.assertIn("Nachher", qp.checkable_words(text))

    def test_an_unclosed_fence_swallows_the_rest(self):
        # a truncated answer ends mid-block; what follows the opener is still code
        self.assertEqual(qp.checkable_words("Text.\n```\nfoo bar baz"), ["Text"])

    def test_an_inline_code_span_is_code(self):
        self.assertEqual(qp.checkable_words("Die Datei `vorlage.svg` ist da."),
                         ["Die", "Datei", "ist", "da"])

    def test_urls_hex_codes_versions_paths_and_numbers_never_reach_the_dictionary(self):
        text = ("Siehe https://example.org/Kunstgewerbe und #D7FFE0 sowie v3.2.1 "
                "unter /srv/plakat/vorlage.svg mit 4800 Stueck und ssm_conv1d_alpha.")
        words = qp.checkable_words(text)
        for gone in ("https", "example", "org", "Kunstgewerbe", "srv", "plakat",
                     "ssm", "conv", "alpha", "vorlage", "svg"):
            self.assertNotIn(gone, words, gone)
        self.assertEqual(words, ["Siehe", "und", "sowie", "unter", "mit", "Stueck", "und"])

    def test_hyphenated_compounds_are_split_into_their_parts(self):
        self.assertEqual(qp.checkable_words("Jugendstil-Ornamentik"), ["Jugendstil", "Ornamentik"])
        self.assertEqual(qp.checkable_words("Farb\u2013raum"), ["Farb", "raum"])

    def test_single_letters_and_acronyms_are_dropped(self):
        self.assertEqual(qp.checkable_words("Das ist z. B. eine GPU und ein Plakat."),
                         ["Das", "ist", "eine", "und", "ein", "Plakat"])

    def test_umlauts_and_sharp_s_are_letters(self):
        self.assertEqual(qp.checkable_words("Größe schön Maß"), ["Größe", "schön", "Maß"])

    def test_an_english_contraction_stays_one_word(self):
        self.assertEqual(qp.checkable_words("It doesn't matter"), ["It", "doesn't", "matter"])
        self.assertEqual(qp.checkable_words("It doesn\u2019t matter"), ["It", "doesn't", "matter"])

    def test_the_empty_answer_has_no_words(self):
        self.assertEqual(qp.checkable_words(""), [])


class EditDistance(unittest.TestCase):
    def test_the_hex_near_miss_of_the_ticket_is_distance_one(self):
        self.assertEqual(qp.edit_distance("#DFFFE0", "#D7FFE0"), 1)

    def test_identity_and_the_usual_three_edits(self):
        self.assertEqual(qp.edit_distance("abc", "abc"), 0)
        self.assertEqual(qp.edit_distance("abc", "abd"), 1)      # substitute
        self.assertEqual(qp.edit_distance("abc", "ab"), 1)       # delete
        self.assertEqual(qp.edit_distance("abc", "abcd"), 1)     # insert
        self.assertEqual(qp.edit_distance("kitten", "sitting"), 3)

    def test_the_cap_short_circuits_without_lying_below_it(self):
        self.assertEqual(qp.edit_distance("abcdefgh", "zz", cap=2), 3)
        self.assertEqual(qp.edit_distance("abc", "abd", cap=2), 1)
        self.assertGreater(qp.edit_distance("aaaa", "bbbb", cap=2), 2)


class Literals(unittest.TestCase):
    LITS = [{"text": "#D7FFE0", "min": 2}, {"text": "v3.2.1", "min": 1}]

    def test_a_literal_counted_often_enough_is_ok(self):
        r = qp.literal_report("Hier #D7FFE0 und dort #D7FFE0, Fassung v3.2.1.", self.LITS)
        self.assertEqual(r["literals_ok"], 2)
        self.assertEqual(r["literals_share"], 1.0)
        self.assertEqual(r["occurrence_share"], 1.0)
        self.assertEqual(r["near_misses"], [])

    def test_one_occurrence_short_is_not_ok_and_the_share_is_partial(self):
        r = qp.literal_report("Nur einmal #D7FFE0 und v3.2.1.", self.LITS)
        self.assertEqual(r["literals_ok"], 1)
        self.assertAlmostEqual(r["occurrence_share"], 2 / 3)

    def test_the_near_miss_is_found_and_named(self):
        r = qp.literal_report("Wir nehmen #DFFFE0 statt der Vorgabe.", self.LITS)
        miss = [m for m in r["near_misses"] if m["literal"] == "#D7FFE0"]
        self.assertEqual(len(miss), 1)
        self.assertEqual(miss[0]["seen"], "#DFFFE0")
        self.assertEqual(miss[0]["distance"], 1)
        self.assertEqual(miss[0]["count"], 1)

    def test_a_case_slip_is_a_near_miss_at_distance_zero_and_not_an_exact_hit(self):
        r = qp.literal_report("#d7ffe0 #d7ffe0", self.LITS)
        self.assertEqual(r["per_literal"][0]["exact"], 0)
        miss = [m for m in r["near_misses"] if m["seen"] == "#d7ffe0"]
        self.assertEqual(len(miss), 1)
        self.assertEqual((miss[0]["distance"], miss[0]["case_only"], miss[0]["count"]),
                         (0, True, 2))

    def test_a_digit_slip_is_a_near_miss_that_is_not_case_only(self):
        r = qp.literal_report("Wir nehmen #DFFFE0.", self.LITS)
        miss = [m for m in r["near_misses"] if m["seen"] == "#DFFFE0"]
        self.assertEqual((miss[0]["distance"], miss[0]["case_only"]), (1, False))

    def test_a_leading_slash_belongs_to_the_path_and_invents_no_near_miss(self):
        lits = [{"text": "/srv/plakat/vorlage.svg", "min": 1}]
        r = qp.literal_report("Die Vorlage liegt unter /srv/plakat/vorlage.svg.", lits)
        self.assertEqual(r["per_literal"][0]["exact"], 1)
        self.assertEqual(r["near_misses"], [])

    def test_the_same_path_without_its_leading_slash_IS_a_near_miss(self):
        lits = [{"text": "/srv/plakat/vorlage.svg", "min": 1}]
        r = qp.literal_report("Die Vorlage liegt unter srv/plakat/vorlage.svg.", lits)
        self.assertEqual(r["per_literal"][0]["exact"], 0)
        self.assertEqual(r["near_misses"][0]["seen"], "srv/plakat/vorlage.svg")

    def test_trailing_punctuation_does_not_invent_a_near_miss(self):
        r = qp.literal_report("Die Farbe #D7FFE0, dann #D7FFE0.", self.LITS)
        self.assertEqual(r["per_literal"][0]["exact"], 2)
        self.assertEqual([m["seen"] for m in r["near_misses"]], [])


class JsonExtraction(unittest.TestCase):
    def test_a_bare_document_comes_through_the_bare_door(self):
        v, how = qp.extract_json('{"a": 1}')
        self.assertEqual((v, how), ({"a": 1}, "bare"))

    def test_a_fenced_document_is_found(self):
        v, how = qp.extract_json('Hier:\n```json\n{"a": 1}\n```\nFertig.')
        self.assertEqual((v, how), ({"a": 1}, "fenced"))

    def test_a_document_embedded_in_prose_is_found(self):
        v, how = qp.extract_json('Das Ergebnis ist {"a": {"b": [1, 2]}} und fertig.')
        self.assertEqual((v, how), ({"a": {"b": [1, 2]}}, "embedded"))

    def test_a_brace_inside_a_string_does_not_end_the_document(self):
        v, how = qp.extract_json('vorher {"a": "}"} nachher')
        self.assertEqual((v, how), ({"a": "}"}, "embedded"))

    def test_prose_alone_parses_to_nothing(self):
        self.assertEqual(qp.extract_json("Es gibt kein JSON hier."), (None, "no parse"))

    def test_the_empty_answer_is_empty_and_not_a_parse_failure(self):
        self.assertEqual(qp.extract_json("   "), (None, "empty"))


SHAPE = {
    "type": "object",
    "required": {
        "quant": "string",
        "rows": "integer",
        "strict": "boolean",
        "layers": {"type": "array", "length": 2,
                   "items": {"type": "object",
                             "required": {"index": "integer", "bits": "number",
                                          "prev": "integer_or_null"}}},
    },
}


class Shape(unittest.TestCase):
    GOOD = {"quant": "CNQ4.5-M", "rows": 248320, "strict": True,
            "layers": [{"index": 0, "bits": 32, "prev": None},
                       {"index": 4, "bits": 6.5, "prev": 0}]}

    def test_the_shape_of_record_has_no_problems(self):
        self.assertEqual(qp.check_shape(self.GOOD, SHAPE), [])

    def test_a_missing_key_is_named_with_its_path(self):
        bad = dict(self.GOOD)
        del bad["strict"]
        self.assertEqual(qp.check_shape(bad, SHAPE), ["$.strict: missing"])

    def test_a_wrong_type_is_named_with_its_path(self):
        bad = json.loads(json.dumps(self.GOOD))
        bad["rows"] = "248320"
        self.assertEqual(qp.check_shape(bad, SHAPE), ["$.rows: expected integer, got str"])

    def test_a_boolean_is_not_an_integer(self):
        self.assertEqual(qp.check_shape(True, "integer"), ["$: expected integer, got bool"])
        self.assertEqual(qp.check_shape(1, "boolean"), ["$: expected boolean, got int"])

    def test_an_integer_is_a_number_but_a_float_is_not_an_integer(self):
        self.assertEqual(qp.check_shape(6, "number"), [])
        self.assertEqual(qp.check_shape(6.5, "integer"), ["$: expected integer, got float"])

    def test_the_wrong_array_length_is_named(self):
        bad = json.loads(json.dumps(self.GOOD))
        bad["layers"] = bad["layers"][:1]
        self.assertEqual(qp.check_shape(bad, SHAPE), ["$.layers: expected 2 items, got 1"])

    def test_a_problem_inside_an_item_carries_its_index(self):
        bad = json.loads(json.dumps(self.GOOD))
        bad["layers"][1]["bits"] = "6.5"
        self.assertEqual(qp.check_shape(bad, SHAPE),
                         ["$.layers[1].bits: expected number, got str"])

    def test_integer_or_null_takes_both_and_not_a_string(self):
        self.assertEqual(qp.check_shape(None, "integer_or_null"), [])
        self.assertEqual(qp.check_shape(3, "integer_or_null"), [])
        self.assertEqual(len(qp.check_shape("3", "integer_or_null")), 1)


class JsonValidity(unittest.TestCase):
    """The scorer's reading of "valid JSON", which is the whole answer and not a fragment."""

    PROMPT = {"id": "x", "lang": "en", "metrics": ["json"],
              "json_shape": {"type": "object", "required": {"a": "integer"}}}

    def test_a_bare_document_of_the_right_shape_is_valid_and_ok(self):
        m = qp.score('{"a": 1}', self.PROMPT, Path("/nonexistent"))["json"]
        self.assertEqual((m["valid_document"], m["shape_ok"], m["how"]), (True, True, "bare"))

    def test_a_fenced_document_still_counts_as_the_whole_answer(self):
        m = qp.score('```json\n{"a": 1}\n```', self.PROMPT, Path("/nonexistent"))["json"]
        self.assertEqual((m["valid_document"], m["shape_ok"]), (True, True))

    def test_a_broken_document_whose_fragment_parses_is_NOT_valid(self):
        # the shape of the real failure: one stray comma, and the only thing that still
        # parses is an inner object
        broken = '{"a": 1, "b": [\n{"i": 0},\n{"i": 1,\n,\n"j": 2}\n]}'
        m = qp.score(broken, self.PROMPT, Path("/nonexistent"))["json"]
        self.assertEqual(m["how"], "embedded")
        self.assertFalse(m["valid_document"])
        self.assertFalse(m["shape_ok"])

    def test_prose_is_neither_valid_nor_parsed(self):
        m = qp.score("There is no JSON here.", self.PROMPT, Path("/nonexistent"))["json"]
        self.assertEqual((m["parsed"], m["valid_document"], m["shape_ok"]), (False, False, False))


class Repetition(unittest.TestCase):
    def test_an_ordinary_sentence_repeats_nothing(self):
        r = qp.repetition("das Plakat ist grün und das Papier ist rau".split())
        self.assertEqual(r["longest_repeat_count"], 1)

    def test_one_word_hammered_is_found_with_its_run_length(self):
        r = qp.repetition("a b c c c c c d".split())
        self.assertEqual((r["longest_repeat_n"], r["longest_repeat_count"]), (1, 5))
        self.assertEqual(r["longest_repeat_text"], "c")

    def test_a_repeated_pair_is_found_as_a_pair(self):
        r = qp.repetition("x a b a b a b y".split())
        self.assertEqual((r["longest_repeat_n"], r["longest_repeat_count"]), (2, 3))
        self.assertEqual(r["longest_repeat_text"], "a b")

    def test_the_distinct_word_ratio_ignores_case(self):
        r = qp.repetition("Das das DAS x".split())
        self.assertAlmostEqual(r["distinct_word_ratio"], 0.5)

    def test_the_empty_answer_has_no_ratio(self):
        r = qp.repetition([])
        self.assertIsNone(r["distinct_word_ratio"])
        self.assertEqual(r["longest_repeat_count"], 0)


class ForeignScript(unittest.TestCase):
    def test_german_prose_carries_no_foreign_script(self):
        r = qp.foreign_script("Die Größe der Fläche, schön gedruckt auf weißem Papier.")
        self.assertEqual(r["foreign_chars"], 0)
        self.assertEqual(r["counts"], {})

    def test_cjk_and_cyrillic_are_counted_and_sampled(self):
        r = qp.foreign_script("Das Plakat 中文 und кириллица.")
        self.assertEqual(r["counts"]["cjk"], 2)
        self.assertEqual(r["counts"]["cyrillic"], 9)
        self.assertEqual(r["foreign_chars"], 11)
        self.assertTrue(all("context" in s for s in r["samples"]))

    def test_greek_is_counted_apart_because_alpha_is_ordinary_in_a_technical_text(self):
        r = qp.foreign_script("Der Faktor α steht für die Dämpfung.")
        self.assertEqual(r["greek_chars"], 1)
        self.assertEqual(r["foreign_chars"], 0)


class WordContexts(unittest.TestCase):
    def test_the_context_is_verbatim_and_the_word_is_matched_whole(self):
        text = "Erschd kuck ich mir alles an, dann das Kunstgewerkds im Saal."
        ctx = qp.word_contexts(text, ["Kunstgewerkds"])
        self.assertEqual(len(ctx), 1)
        self.assertIn("Kunstgewerkds", ctx[0]["context"])

    def test_a_word_inside_a_longer_word_is_not_matched(self):
        self.assertEqual(qp.word_contexts("Kunstgewerkdsstil", ["Kunstgewerkds"]), [])


class HunspellContract(unittest.TestCase):
    """The impure metric, tested against a fake binary: contract, not dictionary content."""

    def setUp(self):
        self.dir = tempfile.mkdtemp()
        self.fake = Path(self.dir) / "fake-hunspell"
        # echoes back every input word that starts with X, plus one word nobody sent
        self.fake.write_text(
            "#!/usr/bin/env python3\n"
            "import sys\n"
            "sent = [l.strip() for l in sys.stdin if l.strip()]\n"
            "open(%r, 'w').write('\\n'.join(sent))\n"
            "print('\\n'.join([w for w in sent if w.startswith('X')] + ['NEVER_SENT']))\n"
            % str(Path(self.dir) / "seen.txt"), encoding="utf-8")
        self.fake.chmod(self.fake.stat().st_mode | stat.S_IEXEC)

    def test_words_go_in_one_per_line_deduplicated_and_sorted(self):
        qp.hunspell_flag(["b", "a", "b", "Xq"], "stem", str(self.fake))
        seen = (Path(self.dir) / "seen.txt").read_text(encoding="utf-8").splitlines()
        self.assertEqual(seen, ["Xq", "a", "b"])

    def test_only_words_that_were_sent_can_come_back_flagged(self):
        self.assertEqual(qp.hunspell_flag(["a", "Xq"], "stem", str(self.fake)), {"Xq"})

    def test_no_words_means_no_call_and_no_flags(self):
        self.assertEqual(qp.hunspell_flag([], "stem", "/nonexistent-binary"), set())


class NonwordReport(unittest.TestCase):
    """The loanword split, against the REAL dictionaries when they are on this machine."""

    @classmethod
    def setUpClass(cls):
        cls.dict_dir = qp.DEFAULT_DICT_DIR
        if not (cls.dict_dir / "de_DE_frami.dic").exists():
            raise unittest.SkipTest("no dictionaries in %s" % cls.dict_dir)

    def test_a_german_misspelling_is_flagged_and_is_not_a_loanword(self):
        r = qp.nonword_report("Der Betrachers sieht das Plakat.", "de", self.dict_dir)
        self.assertEqual([f["word"] for f in r["flagged"]], ["Betrachers"])
        self.assertFalse(r["flagged"][0]["loanword"])
        self.assertEqual(r["loanword_occurrences"], 0)

    def test_an_english_loanword_in_german_prose_is_flagged_and_marked(self):
        r = qp.nonword_report("Der Thread wartet auf einen Timeout.", "de", self.dict_dir)
        words = {f["word"]: f["loanword"] for f in r["flagged"]}
        self.assertTrue(words.get("Thread"))
        self.assertTrue(words.get("Timeout"))
        self.assertEqual(r["rate_per_1000_excl_loanwords"], 0.0)
        self.assertGreater(r["rate_per_1000"], 0.0)

    def test_clean_german_prose_has_rate_zero(self):
        r = qp.nonword_report("Das Haus ist schön und die Straße ist breit.", "de",
                              self.dict_dir)
        self.assertEqual(r["flagged"], [])
        self.assertEqual(r["rate_per_1000"], 0.0)


class Aggregate(unittest.TestCase):
    def test_a_mean_is_reported_with_its_spread(self):
        sp = qp._spread([1.0, 3.0, 5.0])
        self.assertEqual((sp["n"], sp["mean"], sp["min"], sp["max"]), (3, 3.0, 1.0, 5.0))

    def test_an_empty_series_has_no_mean(self):
        self.assertIsNone(qp._spread([None, None]))

    def test_the_pooled_rate_is_words_over_words_and_not_a_mean_of_means(self):
        recs = [
            {"lang": "de", "finish_reason": "stop", "wall_s": 1.0, "metrics": {
                "nonword": {"lang": "de", "checked_words": 100, "flagged_occurrences": 1,
                            "loanword_occurrences": 0, "rate_per_1000": 10.0,
                            "rate_per_1000_excl_loanwords": 10.0},
                "length": {"words": 100}}},
            {"lang": "de", "finish_reason": "stop", "wall_s": 1.0, "metrics": {
                "nonword": {"lang": "de", "checked_words": 900, "flagged_occurrences": 9,
                            "loanword_occurrences": 9, "rate_per_1000": 10.0,
                            "rate_per_1000_excl_loanwords": 0.0},
                "length": {"words": 900}}},
        ]
        agg = qp.aggregate(recs)
        self.assertEqual(agg["nonword_de"]["checked_words"], 1000)
        self.assertEqual(agg["nonword_de"]["pooled_rate_per_1000"], 10.0)
        self.assertEqual(agg["nonword_de"]["pooled_rate_per_1000_excl_loanwords"], 1.0)
        self.assertEqual(agg["nonword_de"]["per_generation"]["n"], 2)


class TheRequestBody(unittest.TestCase):
    """The row on the wire, and the one difference between the two arms."""

    class Args:
        model = "crow-nest"
        max_tokens = 2600
        reasoning_effort = None

    PROMPT = {"system": "s", "user": "u"}

    def test_the_real_operating_row_is_what_goes_out(self):
        body = qp.build_body(self.PROMPT, 7, self.Args(), "crow")
        self.assertEqual(body["temperature"], 1.0)
        self.assertEqual(body["top_p"], 0.95)
        self.assertEqual(body["top_k"], 20)
        self.assertEqual(body["presence_penalty"], 0.0)
        self.assertEqual(body["min_p"], 0.0)
        self.assertEqual(body["max_tokens"], 2600)
        self.assertEqual(body["seed"], 7)
        self.assertIs(body["stream"], False)

    def test_crow_switches_thinking_off_by_saying_nothing(self):
        self.assertNotIn("reasoning_effort", qp.build_body(self.PROMPT, 7, self.Args(), "crow"))

    def test_llama_switches_thinking_off_through_the_top_level_field(self):
        body = qp.build_body(self.PROMPT, 7, self.Args(), "llama")
        self.assertEqual(body["reasoning_effort"], "none")

    def test_thinking_on_is_the_same_field_on_both_arms(self):
        args = self.Args()
        args.reasoning_effort = "high"
        for engine in ("crow", "llama"):
            self.assertEqual(qp.build_body(self.PROMPT, 7, args, engine)["reasoning_effort"],
                             "high")


class ThePromptSet(unittest.TestCase):
    """The prompt file is evidence: a run is only comparable while it holds its shape."""

    @classmethod
    def setUpClass(cls):
        cls.doc = json.loads((TOOLS / "quality-probe-prompts.json").read_text(encoding="utf-8"))

    def test_every_prompt_has_an_id_a_language_and_metrics(self):
        ids = set()
        for p in self.doc["prompts"]:
            self.assertNotIn(p["id"], ids)
            ids.add(p["id"])
            self.assertIn(p["lang"], ("de", "en"))
            self.assertTrue(p["metrics"])
            self.assertTrue(p["system"] and p["user"])

    def test_the_set_has_the_twelve_tasks_of_the_ticket(self):
        kinds = {}
        for p in self.doc["prompts"]:
            kinds[p["kind"]] = kinds.get(p["kind"], 0) + 1
        self.assertEqual(len(self.doc["prompts"]), 12)
        self.assertEqual(kinds, {"prose": 6, "literal": 2, "json": 2, "agentic": 2})
        prose = [p for p in self.doc["prompts"] if p["kind"] == "prose"]
        self.assertEqual(sum(1 for p in prose if p["lang"] == "de"), 4)
        self.assertEqual(sum(1 for p in prose if p["lang"] == "en"), 2)
        self.assertTrue(all(p["lang"] == "de" for p in self.doc["prompts"]
                            if p["kind"] == "agentic"))

    def test_a_literal_task_declares_its_literals_and_a_json_task_its_shape(self):
        for p in self.doc["prompts"]:
            if "literal" in p["metrics"]:
                self.assertTrue(p["literals"])
                for lit in p["literals"]:
                    self.assertIn(lit["text"], p["user"])
            if "json" in p["metrics"]:
                self.assertTrue(p["json_shape"])

    def test_every_declared_metric_is_one_the_scorer_knows(self):
        known = {"nonword", "literal", "json", "repetition", "foreign", "length"}
        for p in self.doc["prompts"]:
            self.assertTrue(set(p["metrics"]) <= known, p["id"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
