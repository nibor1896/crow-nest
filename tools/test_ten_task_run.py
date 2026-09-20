#!/usr/bin/env python3
"""#80: the unit tests of the ten-task runner's mechanical checks.

  python tools/test_ten_task_run.py        # no server, no GPU, no network

The mechanical checks are the half of the grading that needs no judge, and a check nobody
tested is an opinion with a regex in it. Every one of them is a pure function of the answer
text and is pinned here: the number matcher (separators, false neighbours), the worked-case
extractor of t6-reason, the five exact values of t6b-reason-multi, the language reading of
Rev3 core requirement 0, the 400-word limit of t4-prose, and the compiler harness - which is
the only impure one and is exercised against the real `g++` when it is on PATH.

The module is loaded by path because the tool's file name carries a hyphen.
"""

import importlib.util
import shutil
import unittest
from pathlib import Path

TOOLS = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location("ten_task_run", TOOLS / "ten-task-run.py")
tt = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(tt)

HAVE_GXX = shutil.which("g++") is not None


class Numbers(unittest.TestCase):
    def test_an_integer_is_found_with_any_thousands_separator_or_none(self):
        for written in ("6144", "6,144", "6.144", "6 144", "6_144", "6 144"):
            self.assertTrue(tt.int_present("bytes per token: %s B" % written, 6144), written)

    def test_a_longer_number_is_grouped_from_the_right(self):
        self.assertEqual(tt.grouped_pattern(1610612736).count("|"), 0)
        for written in ("1610612736", "1,610,612,736", "1.610.612.736", "1 610 612 736"):
            self.assertTrue(tt.int_present("total %s bytes" % written, 1610612736), written)

    def test_a_digit_next_door_is_not_the_number(self):
        self.assertFalse(tt.int_present("16144", 6144))
        self.assertFalse(tt.int_present("61440", 6144))
        self.assertTrue(tt.int_present("=6144.", 6144))

    def test_a_decimal_is_read_in_either_notation_and_not_across_a_boundary(self):
        self.assertTrue(tt.decimal_present("margin 728.25 MiB", 728, 25))
        self.assertTrue(tt.decimal_present("Marge 728,25 MiB", 728, 25))
        self.assertFalse(tt.decimal_present("1728.25", 728, 25))
        self.assertFalse(tt.decimal_present("728.251", 728, 25))

    def test_a_percentage_needs_its_sign_or_its_word(self):
        self.assertTrue(tt.percent_present("an overhead of 100 %", 100))
        self.assertTrue(tt.percent_present("100% more", 100))
        self.assertTrue(tt.percent_present("100 percent", 100))
        self.assertFalse(tt.percent_present("100 bytes", 100))


class T6bExactValues(unittest.TestCase):
    GOOD = ("Step 1: 12 x 2 x 256 x 1 B = 6,144 bytes per token.\n"
            "Step 2: 262,144 x 6,144 = 1,610,612,736 bytes = 1.5 GiB exactly.\n"
            "Step 3: 3,221,225,472 / 6,144 = 524,288 full tokens, exact.\n"
            "Step 4: BF16 gives 12,288 B/token, 262,144 x 12,288 = 3,221,225,472 B, "
            "an overhead of 100 % over FP8.\n"
            "Step 5: 200,000 x 12,288 = 2,457,600,000 B <= 3,221,225,472 B -> PASS, "
            "margin 763,625,472 B = 728.25 MiB.")

    def test_a_complete_correct_chain_has_all_five_steps(self):
        got = tt.t6b_numbers(self.GOOD)
        self.assertTrue(got["all_five_steps_exact"], got["missing"])
        self.assertEqual(got["missing"], [])
        self.assertTrue(got["verdict_pass_uppercase"])

    def test_a_wrong_step_one_is_seen(self):
        got = tt.t6b_numbers(self.GOOD.replace("6,144", "25,048"))
        self.assertFalse(got["all_five_steps_exact"])
        self.assertIn("step1_6144", got["missing"])

    def test_a_truncated_chain_misses_the_tail(self):
        got = tt.t6b_numbers(self.GOOD.split("Step 3")[0])
        self.assertFalse(got["all_five_steps_exact"])
        self.assertIn("step3_524288", got["missing"])
        self.assertIn("step5_2457600000", got["missing"])

    def test_the_margin_counts_either_way_round(self):
        only_mib = self.GOOD.replace("763,625,472 B = 728.25 MiB", "728.25 MiB")
        self.assertTrue(tt.t6b_numbers(only_mib)["step5_margin_either"])
        only_b = self.GOOD.replace("763,625,472 B = 728.25 MiB", "763,625,472 B")
        self.assertTrue(tt.t6b_numbers(only_b)["step5_margin_either"])


class T6WorkedCase(unittest.TestCase):
    def test_the_three_stated_answers_are_extracted(self):
        text = ("Query (1,5,2): range [3,1,3,4,1], distinct {1,3,4} -> answer 3\n"
                "Query (2,7,3): distinct {1,3,4,5,9}, the 3rd smallest is 4\n"
                "Query (4,4,2): only one distinct value, so the result is -1\n")
        got = tt.t6_answers(text)
        self.assertTrue(got["all_three_ok"], got)
        self.assertEqual(got["q3_4_4_2"]["stated"], -1)

    def test_a_wrong_answer_is_not_rescued_by_the_right_digit_elsewhere(self):
        text = ("The distinct values are 3 in total.\n"
                "Query (1,5,2): the answer = 4\n")
        got = tt.t6_answers(text)
        self.assertEqual(got["q1_1_5_2"]["stated"], 4)
        self.assertFalse(got["q1_1_5_2"]["ok"])

    def test_a_query_that_is_never_worked_reads_as_missing(self):
        got = tt.t6_answers("A persistent segment tree over last occurrences.")
        self.assertIsNone(got["q1_1_5_2"]["stated"])
        self.assertFalse(got["all_three_ok"])

    def test_the_markers_find_the_intended_algorithm(self):
        got = tt.t6_markers("Sort the queries offline by increasing r and keep the last "
                            "occurrence of every value in a persistent segment tree. "
                            "This is O((n + q) log n).")
        self.assertTrue(got["last_occurrence"])
        self.assertTrue(got["persistent_or_bit"])
        self.assertTrue(got["offline_by_r"])
        self.assertTrue(got["complexity_stated"])


class Language(unittest.TestCase):
    def test_german_prose_reads_as_german(self):
        self.assertEqual(tt.detect_lang(
            "Die wahrscheinlichste falsche Annahme ist, dass der Cache unter der Decke "
            "bleibt und die Kosten pro Slot damit konstant sind.")["lang"], "de")

    def test_english_prose_reads_as_english(self):
        self.assertEqual(tt.detect_lang(
            "The copy does not happen in the code that is shown here; it happens in "
            "load_all_data when the backend buffer has been allocated.")["lang"], "en")

    def test_code_is_not_prose_so_it_cannot_decide_the_language(self):
        got = tt.detect_lang("```cpp\nfor (int i = 0; i < n; ++i) { the(); and(); is(); }\n```\n"
                             "Die Schleife laeuft ueber alle Elemente und ist damit korrekt.")
        self.assertEqual(got["lang"], "de")

    def test_an_answer_without_function_words_is_unknown_not_guessed(self):
        self.assertEqual(tt.detect_lang("6144 12288 524288")["lang"], "unknown")


class WordLimit(unittest.TestCase):
    def test_the_prompts_own_unit_is_whitespace_words(self):
        self.assertEqual(tt.word_count("ein zwei drei"), 3)
        self.assertEqual(tt.word_count(""), 0)

    def test_t4_carries_the_400_word_limit_and_the_german_expectation(self):
        m = tt.mechanical("t4-prose", "Wort " * 401)
        self.assertEqual(m["expected_lang"], "de")
        self.assertEqual(m["word_limit"]["words"], 401)
        self.assertFalse(m["word_limit"]["within_400"])
        self.assertTrue(m["word_limit"]["within_440_tolerance"])

    def test_every_other_task_expects_english(self):
        for task in ("t1-read", "t2-write", "t6b-reason-multi"):
            self.assertEqual(tt.mechanical(task, "x")["expected_lang"], "en")


class FencedBlocks(unittest.TestCase):
    def test_a_closed_block_is_read_with_its_language_tag(self):
        got = tt.fenced_blocks("before\n```cpp\nint main(){}\n```\nafter")
        self.assertEqual(got, [("cpp", "int main(){}\n")])

    def test_an_unclosed_block_runs_to_the_end_of_a_truncated_answer(self):
        got = tt.fenced_blocks("```cpp\nint main(){\n")
        self.assertEqual(got[0][1], "int main(){\n")

    def test_only_c_like_blocks_become_translation_units(self):
        text = "```python\nprint(1)\n```\n```cpp\nint f(){return 0;}\n```"
        names = [n for n, _ in tt.cpp_variants(text)]
        self.assertEqual(names, ["all"])
        self.assertIn("int f()", tt.cpp_variants(text)[0][1])
        self.assertNotIn("print(1)", tt.cpp_variants(text)[0][1])


@unittest.skipUnless(HAVE_GXX, "g++ is not on PATH")
class Compiler(unittest.TestCase):
    def test_a_correct_header_plus_tests_compiles_and_its_own_asserts_run(self):
        text = ("Here it is.\n```cpp\n#pragma once\ntemplate<class K, class V> struct Tiny "
                "{ V v{}; };\n```\nAnd the tests:\n```cpp\nint main(){ Tiny<int,int> t; "
                "assert(t.v == 0); return 0; }\n```\n")
        got = tt.compile_check(text, run_binary=True)
        self.assertTrue(got["compiles"], got["stderr_head"])
        self.assertTrue(got["ran"])
        self.assertEqual(got["exit_code"], 0)

    def test_a_failing_assert_is_a_non_zero_exit_not_a_compile_error(self):
        text = "```cpp\nint main(){ assert(1 == 2); return 0; }\n```"
        got = tt.compile_check(text, run_binary=True)
        self.assertTrue(got["compiles"])
        self.assertTrue(got["ran"])
        self.assertNotEqual(got["exit_code"], 0)

    def test_broken_code_does_not_compile_and_the_error_is_kept(self):
        got = tt.compile_check("```cpp\nint main(){ this is not c++ }\n```")
        self.assertFalse(got["compiles"])
        self.assertTrue(got["stderr_head"])

    def test_a_bare_function_compiles_against_the_prelude(self):
        # t2b-write-refactor answers a function, not a program: the prelude carries <vector>,
        # <cstdint> and <cstring> so a correct answer is not failed for its missing includes
        got = tt.compile_check("```cpp\nbool f(const uint8_t* r, size_t n, "
                               "std::vector<uint32_t>& a) { a.clear(); memcpy(&n, r, 4); "
                               "return n > 0; }\n```")
        self.assertTrue(got["compiles"], got["stderr_head"])

    def test_a_second_block_that_redefines_main_falls_back_to_the_largest_block(self):
        text = ("```cpp\nint main(){ return 0; }\n```\n"
                "```cpp\nint main(){ int a = 1; int b = 2; int c = 3; return a + b + c; }\n```")
        got = tt.compile_check(text)
        self.assertTrue(got["compiles"], got["stderr_head"])
        self.assertEqual(got["variant"], "largest")

    def test_an_answer_with_no_code_at_all_is_a_result_not_a_crash(self):
        got = tt.compile_check("I would use an unordered_map and a list.")
        self.assertFalse(got["compiles"])
        self.assertEqual(got["blocks"], 0)


class Intervals(unittest.TestCase):
    def test_wilson_brackets_the_point_estimate(self):
        lo, hi = tt.wilson(13, 27)
        self.assertLess(lo, 13 / 27)
        self.assertGreater(hi, 13 / 27)

    def test_wilson_does_not_collapse_at_the_ends(self):
        lo, hi = tt.wilson(0, 27)
        self.assertEqual(lo, 0.0)
        self.assertGreater(hi, 0.0)
        lo, hi = tt.wilson(27, 27)
        self.assertLess(lo, 1.0)
        self.assertAlmostEqual(hi, 1.0)


class BlindPlumbing(unittest.TestCase):
    def test_the_opaque_id_is_stable_and_distinguishes_arm_task_and_seed(self):
        a = tt.opaque_id("A-crow", "t1-read", 2101, "salt")
        self.assertEqual(a, tt.opaque_id("A-crow", "t1-read", 2101, "salt"))
        self.assertNotEqual(a, tt.opaque_id("B-llama", "t1-read", 2101, "salt"))
        self.assertNotEqual(a, tt.opaque_id("A-crow", "t1-read", 2102, "salt"))
        self.assertNotEqual(a, tt.opaque_id("A-crow", "t1-read", 2101, "other"))

    def test_seed_diversity_counts_distinct_texts_per_cell(self):
        recs = [{"label": "A", "task": "t1-read", "content": "x", "reasoning_content": "p"},
                {"label": "A", "task": "t1-read", "content": "x", "reasoning_content": "q"},
                {"label": "A", "task": "t1-read", "content": "y", "reasoning_content": "r"}]
        got = tt.seed_diversity(recs)
        self.assertEqual(got["A/t1-read"], {"n": 3, "distinct_answers": 2,
                                            "distinct_thinking": 3, "empty_answers": 0})

    def test_three_empty_answers_are_not_read_as_a_dead_sampler(self):
        # the whole budget went into thinking: the answers are the same string because they
        # are all empty, and only the thinking column can say whether the seed re-sampled
        recs = [{"label": "A", "task": "t6-reason", "content": "", "reasoning_content": t}
                for t in ("a", "b", "c")]
        got = tt.seed_diversity(recs)["A/t6-reason"]
        self.assertEqual(got["distinct_answers"], 1)
        self.assertEqual(got["distinct_thinking"], 3)
        self.assertEqual(got["empty_answers"], 3)


class TheRequestBody(unittest.TestCase):
    class Args:
        model = "crow-nest"
        max_tokens = 16384
        reasoning_effort = "high"

    def test_the_row_is_the_probes_row_and_thinking_is_a_top_level_word(self):
        body = tt.build_body({"id": "t1-read", "text": "hello"}, 2101, self.Args())
        for k, v in tt.ROW.items():
            self.assertEqual(body[k], v)
        self.assertEqual(body["reasoning_effort"], "high")
        self.assertEqual(body["seed"], 2101)
        self.assertEqual(body["max_tokens"], 16384)
        self.assertIs(body["cache_prompt"], False)
        self.assertNotIn("reasoning_budget", body)
        self.assertEqual(body["messages"], [{"role": "user", "content": "hello"}])

    def test_there_is_no_system_message_and_no_tool(self):
        body = tt.build_body({"id": "t2-write", "text": "x"}, 1, self.Args())
        self.assertEqual(len(body["messages"]), 1)
        self.assertNotIn("tools", body)


if __name__ == "__main__":
    unittest.main(verbosity=2)
