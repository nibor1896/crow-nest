#!/usr/bin/env python3
"""#78: the unit tests of the oracle KLD tool.

  python tools/test_oracle_kld.py          # no GPU, no server, no reference file

Every statistic `tools/oracle-kld.py` reports is a pure function, and this is where each of
them is pinned against a value computed by hand or against the formula llama.cpp's
`tools/perplexity/perplexity.cpp` uses. Where a number here looks arbitrary it is a line of
that file, and the comment names it.

The module is loaded by path because the tool's file name carries a hyphen.
"""

import array
import importlib.util
import json
import math
import os
import tempfile
import unittest
from pathlib import Path

TOOLS = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location("oracle_kld", TOOLS / "oracle-kld.py")
ok = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(ok)


def logits(*xs):
    return array.array("f", xs)


class LogSumExpAndArgmax(unittest.TestCase):
    def test_log_sum_exp_of_a_uniform_row_is_log_n_plus_the_value(self):
        self.assertAlmostEqual(ok.log_sum_exp(logits(3.0, 3.0, 3.0, 3.0)), 3.0 + math.log(4), places=6)

    def test_log_sum_exp_survives_a_large_offset(self):
        small = ok.log_sum_exp(logits(0.0, 1.0, 2.0))
        large = ok.log_sum_exp(logits(900.0, 901.0, 902.0))
        self.assertAlmostEqual(large - 900.0, small, places=5)

    def test_the_dropped_tail_is_below_the_resolution_of_the_sum(self):
        # everything more than LSE_CUTOFF below the maximum is skipped; that has to be invisible
        kept = ok.log_sum_exp(logits(0.0, -200.0))
        alone = ok.log_sum_exp(logits(0.0))
        self.assertEqual(kept, alone)

    def test_argmax_keeps_the_first_of_two_equal_maxima(self):
        # perplexity.cpp:194 compares with > , so the first maximum wins there too
        self.assertEqual(ok.argmax(logits(1.0, 5.0, 5.0, 2.0)), 1)

    def test_a_row_of_logits_is_a_probability_distribution_after_the_shift(self):
        row = logits(0.5, -1.0, 2.0, 0.0)
        lse = ok.log_sum_exp(row)
        self.assertAlmostEqual(sum(math.exp(x - lse) for x in row), 1.0, places=9)


class Support(unittest.TestCase):
    def test_the_support_is_the_entries_above_the_cut_in_index_order(self):
        row = logits(0.0, -30.0, -1.0, -100.0, -2.0)
        lse = ok.log_sum_exp(row)
        self.assertEqual(ok.support_indices(row, lse, -16.0), [0, 2, 4])

    def test_the_cut_is_on_the_log_probability_and_not_on_the_logit(self):
        # the same row shifted by +1000 has the same support
        row = logits(0.0, -30.0, -1.0)
        shifted = logits(1000.0, 970.0, 999.0)
        self.assertEqual(ok.support_indices(row, ok.log_sum_exp(row), -16.0),
                         ok.support_indices(shifted, ok.log_sum_exp(shifted), -16.0))

    def test_truncation_keeps_the_most_probable_entries_of_the_support(self):
        row = logits(1.0, 3.0, 2.0, 0.0)
        sup = [0, 1, 2, 3]
        self.assertEqual(ok.truncate_support(row, sup, 2), [1, 2])

    def test_truncation_at_or_above_the_support_size_changes_nothing(self):
        row = logits(1.0, 3.0, 2.0)
        self.assertEqual(ok.truncate_support(row, [0, 1, 2], 3), [0, 1, 2])
        self.assertEqual(ok.truncate_support(row, [0, 1, 2], None), [0, 1, 2])

    def test_nth_largest(self):
        row = logits(5.0, 1.0, 4.0, 2.0)
        self.assertEqual(ok.nth_largest(row, 1), 5.0)
        self.assertEqual(ok.nth_largest(row, 3), 2.0)
        self.assertIsNone(ok.nth_largest(row, 9))
        self.assertIsNone(ok.nth_largest(row, 0))


class Divergence(unittest.TestCase):
    def test_a_row_against_itself_is_zero(self):
        row = logits(2.0, 0.5, -3.0, 1.0)
        lse = ok.log_sum_exp(row)
        sup = ok.support_indices(row, lse)
        kld, missing, mass = ok.row_kld(row, lse, sup, ok.FullRow(row))
        self.assertAlmostEqual(kld, 0.0, places=12)
        self.assertEqual(missing, 0)
        self.assertAlmostEqual(mass, 1.0, places=9)

    def test_a_constant_shift_of_every_logit_is_not_a_difference(self):
        row = logits(2.0, 0.5, -3.0, 1.0)
        shifted = logits(7.0, 5.5, 2.0, 6.0)
        lse = ok.log_sum_exp(row)
        sup = ok.support_indices(row, lse)
        kld, _, _ = ok.row_kld(row, lse, sup, ok.FullRow(shifted))
        self.assertAlmostEqual(kld, 0.0, places=9)

    def test_two_coins_by_hand(self):
        # P = (0.75, 0.25), Q = (0.5, 0.5): KL = .75*ln(1.5) + .25*ln(.5).
        # Seven places, not more: the logits pass through float32 on their way into the
        # file this tool reads, which is worth about 1e-8 on a probability of 0.75.
        p = logits(math.log(0.75), math.log(0.25))
        q = logits(0.0, 0.0)
        want = 0.75 * math.log(0.75 / 0.5) + 0.25 * math.log(0.25 / 0.5)
        lse = ok.log_sum_exp(p)
        kld, _, _ = ok.row_kld(p, lse, ok.support_indices(p, lse), ok.FullRow(q))
        self.assertAlmostEqual(kld, want, places=7)

    def test_the_direction_is_reference_first_and_it_is_not_symmetric(self):
        p = logits(math.log(0.9), math.log(0.1))
        q = logits(math.log(0.5), math.log(0.5))
        lp, lq = ok.log_sum_exp(p), ok.log_sum_exp(q)
        forward, _, _ = ok.row_kld(p, lp, ok.support_indices(p, lp), ok.FullRow(q))
        backward, _, _ = ok.row_kld(q, lq, ok.support_indices(q, lq), ok.FullRow(p))
        self.assertNotAlmostEqual(forward, backward, places=4)
        self.assertGreater(backward, forward)

    def test_the_support_cut_removes_terms_and_not_the_arms_normalisation(self):
        # The second entry is 20 nats down in the reference, below the -16 cut, and the arm
        # is wrong there. That entry contributes NO TERM - but it still sits in the arm's own
        # normalisation, and what survives is exactly the one term the support names.
        p = logits(0.0, -20.0)
        q = logits(0.0, -1.0)
        lse = ok.log_sum_exp(p)
        sup = ok.support_indices(p, lse, -16.0)
        self.assertEqual(sup, [0])
        kld, _, _ = ok.row_kld(p, lse, sup, ok.FullRow(q))
        lse_q = ok.log_sum_exp(q)
        self.assertAlmostEqual(kld, math.exp(p[0] - lse) * ((p[0] - lse) - (q[0] - lse_q)),
                               places=12)


class TruncatedEstimator(unittest.TestCase):
    def setUp(self):
        self.p = logits(math.log(0.5), math.log(0.3), math.log(0.15), math.log(0.05))
        self.lse = ok.log_sum_exp(self.p)
        self.sup = ok.support_indices(self.p, self.lse)

    def test_a_top_n_arm_and_a_full_arm_agree_when_nothing_is_cut(self):
        q = logits(math.log(0.4), math.log(0.4), math.log(0.1), math.log(0.1))
        full, _, _ = ok.row_kld(self.p, self.lse, self.sup, ok.FullRow(q))
        topn, missing, _ = ok.row_kld(self.p, self.lse, self.sup,
                                      ok.TopNRow([0, 1, 2, 3], [0.4, 0.4, 0.1, 0.1]))
        self.assertAlmostEqual(full, topn, places=9)
        self.assertEqual(missing, 0)

    def test_a_missing_token_is_charged_at_the_arms_smallest_known_probability(self):
        arm = ok.TopNRow([0, 1], [0.4, 0.4])
        sup = ok.truncate_support(self.p, self.sup, 3)     # 0, 1, 2 - and 2 is missing
        kld, missing, mass = ok.row_kld(self.p, self.lse, sup, arm)
        self.assertEqual(missing, 1)
        want = (0.5 * math.log(0.5 / 0.4) + 0.3 * math.log(0.3 / 0.4)
                + 0.15 * math.log(0.15 / 0.4))
        self.assertAlmostEqual(kld, want, places=7)
        self.assertAlmostEqual(mass, 0.95, places=7)

    def test_that_substitution_is_a_lower_bound_and_never_an_overestimate(self):
        # the true q of the missing token is smaller than the floor, so the true term is larger
        arm = ok.TopNRow([0, 1], [0.4, 0.4])
        sup = ok.truncate_support(self.p, self.sup, 3)
        bound, _, _ = ok.row_kld(self.p, self.lse, sup, arm)
        truth, _, _ = ok.row_kld(self.p, self.lse, sup,
                                 ok.TopNRow([0, 1, 2], [0.4, 0.4, 0.05]))
        self.assertLess(bound, truth)

    def test_a_full_arm_given_the_same_blindness_reports_the_same_number(self):
        q = logits(math.log(0.4), math.log(0.4), math.log(0.05), math.log(0.15))
        sup = ok.truncate_support(self.p, self.sup, 3)      # reference tokens 0, 1, 2
        blind = ok.FullRow(q, arm_topn=2)                   # its own top 2 are 0 and 1
        kld, missing, _ = ok.row_kld(self.p, self.lse, sup, blind)
        self.assertEqual(missing, 1)                        # token 2 is below its own top 2
        self.assertAlmostEqual(kld, ok.row_kld(self.p, self.lse, sup,
                                               ok.TopNRow([0, 1], [0.4, 0.4]))[0], places=7)

    def test_a_full_arm_without_blindness_can_answer_for_everything(self):
        arm = ok.FullRow(logits(0.0, 0.0, 0.0, 0.0))
        self.assertIsNone(arm.floor_log_q())
        self.assertAlmostEqual(arm.known_mass(), 1.0, places=9)

    def test_a_top_n_row_reports_the_mass_it_covers(self):
        arm = ok.TopNRow([7, 9], [0.6, 0.1])
        self.assertAlmostEqual(arm.known_mass(), 0.7, places=9)
        self.assertEqual(arm.top1, 7)
        self.assertIsNone(arm.log_q(1234))

    def test_a_top_n_row_takes_its_top_1_from_the_probabilities_not_from_the_order(self):
        self.assertEqual(ok.TopNRow([5, 2, 8], [0.1, 0.7, 0.2]).top1, 2)


class LlamaCppStatistics(unittest.TestCase):
    def test_the_percentile_interpolates_at_fraction_times_n_minus_one(self):
        # perplexity.cpp:1954-1960, by hand on 0..10: 0.95*(11-1) = 9.5 -> 9.5
        vals = [float(i) for i in range(11)]
        self.assertAlmostEqual(ok.percentile(vals, 0.95), 9.5, places=9)
        self.assertAlmostEqual(ok.percentile(vals, 0.90), 9.0, places=9)
        self.assertAlmostEqual(ok.percentile(vals, 0.0), 0.0, places=9)
        self.assertAlmostEqual(ok.percentile(vals, 1.0), 10.0, places=9)

    def test_the_percentile_is_not_a_nearest_rank_percentile(self):
        vals = [0.0, 10.0]
        self.assertAlmostEqual(ok.percentile(vals, 0.5), 5.0, places=9)

    def test_the_median_of_an_even_sample_is_the_mean_of_the_middle_pair(self):
        self.assertAlmostEqual(ok.median([1.0, 2.0, 4.0, 8.0]), 3.0, places=9)

    def test_the_median_of_an_odd_sample_is_the_middle_value(self):
        self.assertAlmostEqual(ok.median([1.0, 2.0, 4.0]), 2.0, places=9)

    def test_the_uncertainty_on_a_mean_is_the_standard_error_with_n_minus_one(self):
        vals = [float(i) for i in range(100)]
        mean, unc = ok.mean_and_uncertainty(vals)
        self.assertAlmostEqual(mean, 49.5, places=9)
        var = sum(v * v for v in vals) / 100 - 49.5 ** 2
        self.assertAlmostEqual(unc, math.sqrt(var / 99), places=9)

    def test_a_sample_of_ten_or_fewer_gets_no_uncertainty(self):
        # perplexity.cpp:1776 - `count > 10`
        self.assertEqual(ok.mean_and_uncertainty([1.0] * 10)[1], 0.0)
        self.assertEqual(ok.mean_and_uncertainty([float(i) for i in range(10)])[1], 0.0)
        self.assertGreater(ok.mean_and_uncertainty([float(i) for i in range(11)])[1], 0.0)

    def test_a_sample_with_no_spread_gets_no_uncertainty(self):
        self.assertEqual(ok.mean_and_uncertainty([2.0] * 50), (2.0, 0.0))

    def test_the_share_carries_the_gaussian_binomial_uncertainty(self):
        share, unc = ok.share_and_uncertainty(247, 298)
        self.assertAlmostEqual(share, 247 / 298, places=12)
        self.assertAlmostEqual(unc, math.sqrt((247 / 298) * (1 - 247 / 298) / 297), places=12)

    def test_a_share_of_one_has_no_uncertainty(self):
        self.assertEqual(ok.share_and_uncertainty(50, 50), (1.0, 0.0))

    def test_an_empty_sample_is_not_a_crash(self):
        self.assertEqual(ok.share_and_uncertainty(0, 0), (0.0, 0.0))
        self.assertEqual(ok.mean_and_uncertainty([]), (0.0, 0.0))


class SignTest(unittest.TestCase):
    def test_all_positive_over_ten_rows_is_two_over_a_thousand_and_twenty_four(self):
        pos, neg, tie, p = ok.sign_test([1.0] * 10)
        self.assertEqual((pos, neg, tie), (10, 0, 0))
        self.assertAlmostEqual(p, 2.0 / 1024, places=12)

    def test_an_even_split_cannot_be_distinguished_from_a_coin(self):
        self.assertAlmostEqual(ok.sign_test([1.0] * 5 + [-1.0] * 5)[3], 1.0, places=12)

    def test_ties_leave_the_sample_and_are_reported(self):
        pos, neg, tie, p = ok.sign_test([1.0, -1.0, 0.0, 0.0])
        self.assertEqual((pos, neg, tie), (1, 1, 2))
        self.assertAlmostEqual(p, 1.0, places=12)

    def test_no_non_tied_row_is_not_a_division_by_zero(self):
        self.assertEqual(ok.sign_test([0.0, 0.0]), (0, 0, 2, 1.0))

    def test_a_p_value_never_exceeds_one(self):
        for k in range(0, 13):
            p = ok.sign_test([1.0] * k + [-1.0] * (12 - k))[3]
            self.assertLessEqual(p, 1.0)
            self.assertGreater(p, 0.0)

    def test_a_large_sample_does_not_overflow(self):
        p = ok.sign_test([1.0] * 320 + [-1.0] * 287)[3]
        self.assertLess(p, 1.0)
        self.assertGreater(p, 0.0)


class RowRange(unittest.TestCase):
    def test_a_half_open_range(self):
        self.assertEqual(ok.parse_range("0:298", 302), (0, 298))
        self.assertEqual(ok.parse_range("235:", 302), (235, 302))
        self.assertEqual(ok.parse_range(":8", 302), (0, 8))
        self.assertEqual(ok.parse_range(None, 302), (0, 302))

    def test_a_range_outside_the_file_is_refused(self):
        for bad in ("0:303", "5:5", "-1:8", "298"):
            with self.assertRaises(SystemExit):
                ok.parse_range(bad, 302)

    def test_named_arms(self):
        self.assertEqual(ok.parse_named(["a=x/y.f32", "b=z"], "--arm"),
                         [("a", "x/y.f32"), ("b", "z")])
        with self.assertRaises(SystemExit):
            ok.parse_named(["nopath"], "--arm")


class EndToEnd(unittest.TestCase):
    """A four-row, eight-token reference and two arms, written to real files."""

    VOCAB = 8
    ROWS = 4

    def setUp(self):
        self.dir = tempfile.mkdtemp(prefix="oracle-kld-test-")
        self.ref = self._write("ref.f32", [
            [4.0, 1.0, 0.0, -1.0, -2.0, -3.0, -4.0, -30.0],
            [0.0, 4.0, 1.0, -1.0, -2.0, -3.0, -4.0, -30.0],
            [1.0, 0.0, 4.0, -1.0, -2.0, -3.0, -4.0, -30.0],
            [0.0, 1.0, 0.0, 4.0, -2.0, -3.0, -4.0, -30.0],
        ])
        # `same` is the reference again; `flip` has the top two swapped on every row
        self.same = self._write("same.f32", self._rows_of(self.ref))
        self.flip = self._write("flip.f32", [
            [1.0, 4.0, 0.0, -1.0, -2.0, -3.0, -4.0, -30.0],
            [4.0, 0.0, 1.0, -1.0, -2.0, -3.0, -4.0, -30.0],
            [4.0, 0.0, 1.0, -1.0, -2.0, -3.0, -4.0, -30.0],
            [4.0, 1.0, 0.0, 0.0, -2.0, -3.0, -4.0, -30.0],
        ])

    def tearDown(self):
        for name in os.listdir(self.dir):
            os.unlink(os.path.join(self.dir, name))
        os.rmdir(self.dir)

    def _write(self, name, rows):
        path = os.path.join(self.dir, name)
        with open(path, "wb") as fh:
            for row in rows:
                array.array("f", row).tofile(fh)
        return path

    def _rows_of(self, path):
        f = ok.LogitFile(path, self.VOCAB)
        rows = [list(f.read(r)) for r in range(f.rows)]
        f.close()
        return rows

    def test_the_file_reader_finds_the_rows(self):
        f = ok.LogitFile(self.ref, self.VOCAB)
        self.assertEqual(f.rows, self.ROWS)
        self.assertAlmostEqual(f.read(2)[2], 4.0, places=6)
        f.close()

    def test_a_file_that_is_not_a_whole_number_of_rows_is_refused(self):
        bad = os.path.join(self.dir, "ragged.f32")
        with open(bad, "wb") as fh:
            array.array("f", [0.0] * (self.VOCAB + 3)).tofile(fh)
        with self.assertRaises(SystemExit):
            ok.LogitFile(bad, self.VOCAB)

    def test_an_identical_arm_is_zero_everywhere_and_agrees_on_every_top_1(self):
        acc, per_row = ok.collect(self.ref, self.VOCAB, 0, self.ROWS,
                                  [("same", self.same)], [], ok.MIN_LOG_PROB, None, None)
        s = ok.summarize(acc["same"])
        self.assertEqual(s["same_top1"], 4)
        self.assertAlmostEqual(s["same_top1_share"], 1.0, places=12)
        self.assertLess(abs(s["kld_mean"]), 1e-9)
        self.assertLess(abs(s["dp_mean"]), 1e-9)
        self.assertEqual(len(per_row), 4)

    def test_a_flipped_arm_loses_the_top_1_where_it_flipped(self):
        acc, _ = ok.collect(self.ref, self.VOCAB, 0, self.ROWS,
                            [("flip", self.flip)], [], ok.MIN_LOG_PROB, None, None)
        s = ok.summarize(acc["flip"])
        self.assertEqual(s["same_top1"], 0)
        self.assertGreater(s["kld_mean"], 0.0)
        self.assertLess(s["dp_mean"], 0.0)

    def test_the_prompt_and_answer_split_selects_the_rows_it_names(self):
        acc, _ = ok.collect(self.ref, self.VOCAB, 0, self.ROWS,
                            [("flip", self.flip)], [], ok.MIN_LOG_PROB, None, None)
        rows = acc["flip"].rows
        first_two = ok.summarize(acc["flip"], [rows[i] < 2 for i in range(len(rows))])
        last_two = ok.summarize(acc["flip"], [rows[i] >= 2 for i in range(len(rows))])
        self.assertEqual(first_two["n"], 2)
        self.assertEqual(last_two["n"], 2)
        self.assertAlmostEqual(
            2 * first_two["kld_mean"] + 2 * last_two["kld_mean"],
            4 * ok.summarize(acc["flip"])["kld_mean"], places=9)

    def test_a_row_range_reads_only_the_rows_it_names(self):
        acc, per_row = ok.collect(self.ref, self.VOCAB, 1, 3,
                                  [("flip", self.flip)], [], ok.MIN_LOG_PROB, None, None)
        self.assertEqual([r["row"] for r in per_row], [1, 2])
        self.assertEqual(acc["flip"].rows, [1, 2])

    def test_the_paired_difference_of_an_arm_with_itself_is_exactly_zero(self):
        acc, _ = ok.collect(self.ref, self.VOCAB, 0, self.ROWS,
                            [("a", self.flip), ("b", self.flip)], [], ok.MIN_LOG_PROB, None, None)
        p = ok.paired(acc["a"], acc["b"])
        self.assertEqual(p["mean"], 0.0)
        self.assertEqual((p["a_above"], p["b_above"], p["ties"]), (0, 0, 4))
        self.assertEqual(p["sign_test_p"], 1.0)

    def test_the_paired_difference_is_signed_the_way_the_header_says(self):
        acc, _ = ok.collect(self.ref, self.VOCAB, 0, self.ROWS,
                            [("flip", self.flip), ("same", self.same)], [], ok.MIN_LOG_PROB, None, None)
        p = ok.paired(acc["flip"], acc["same"])
        self.assertGreater(p["mean"], 0.0)          # the worse arm first gives a positive mean
        self.assertEqual(p["a_above"], 4)

    def test_a_top_n_arm_built_from_the_same_numbers_lands_on_the_full_arm(self):
        rows = self._rows_of(self.flip)
        entries = []
        for r, row in enumerate(rows):
            lse = ok.log_sum_exp(array.array("f", row))
            order = sorted(range(self.VOCAB), key=lambda i: row[i], reverse=True)[:4]
            entries.append({"row": r, "ids": order,
                            "probs": [math.exp(row[i] - lse) for i in order]})
        path = os.path.join(self.dir, "topn.json")
        with open(path, "w", encoding="utf-8") as fh:
            json.dump({"n_probs": 4, "rows": entries}, fh)

        acc, _ = ok.collect(self.ref, self.VOCAB, 0, self.ROWS,
                            [("full", self.flip)], [("topn", path)],
                            ok.MIN_LOG_PROB, 3, 4)
        a, b = ok.summarize(acc["full"]), ok.summarize(acc["topn"])
        self.assertAlmostEqual(a["kld_mean"], b["kld_mean"], places=5)
        self.assertEqual(a["same_top1"], b["same_top1"])

    def test_a_missing_row_in_a_top_n_arm_is_refused(self):
        path = os.path.join(self.dir, "short.json")
        with open(path, "w", encoding="utf-8") as fh:
            json.dump({"n_probs": 4, "rows": [{"row": 0, "ids": [0], "probs": [1.0]}]}, fh)
        with self.assertRaises(SystemExit):
            ok.collect(self.ref, self.VOCAB, 0, self.ROWS, [], [("topn", path)],
                       ok.MIN_LOG_PROB, 3, None)

    def test_the_per_row_file_carries_one_record_per_row_and_arm(self):
        out = os.path.join(self.dir, "per-row.json")
        ok.collect(self.ref, self.VOCAB, 0, self.ROWS, [("flip", self.flip)], [],
                   ok.MIN_LOG_PROB, None, None, per_row_path=out)
        doc = json.load(open(out, encoding="utf-8"))
        self.assertEqual(len(doc["rows"]), 4)
        self.assertEqual(doc["first_row"], 0)
        for rec in doc["rows"]:
            self.assertIn("flip", rec["arms"])
            self.assertIn("kld", rec["arms"]["flip"])

    def test_the_cli_refuses_a_top_n_arm_without_a_truncation(self):
        with self.assertRaises(SystemExit):
            ok.main(["--ref", self.ref, "--vocab", str(self.VOCAB),
                     "--topn-arm", "x=/nonexistent.json"])

    def test_the_cli_refuses_a_run_with_no_arm(self):
        with self.assertRaises(SystemExit):
            ok.main(["--ref", self.ref, "--vocab", str(self.VOCAB)])

    def test_the_cli_writes_the_summary_it_printed(self):
        out = os.path.join(self.dir, "summary.json")
        import io
        import contextlib
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            rc = ok.main(["--ref", self.ref, "--vocab", str(self.VOCAB), "--rows", "0:4",
                          "--prompt-rows", "2", "--arm", "flip=" + self.flip,
                          "--arm", "same=" + self.same, "--paired", "flip,same",
                          "--json", out])
        self.assertEqual(rc, 0)
        doc = json.load(open(out, encoding="utf-8"))
        self.assertEqual(set(doc["summaries"]), {"all rows", "prompt rows", "answer rows"})
        self.assertEqual(doc["summaries"]["all rows"]["flip"]["n"], 4)
        self.assertEqual(len(doc["paired"]), 1)
        self.assertIn("same top-1 as ref", buf.getvalue())
        self.assertIn("KL(P_reference || Q_arm)", buf.getvalue())


if __name__ == "__main__":
    unittest.main(verbosity=2)
