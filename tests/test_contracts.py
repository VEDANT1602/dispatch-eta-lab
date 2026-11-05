from copy import deepcopy
import csv
import gzip
import hashlib
import json
from pathlib import Path
import re
import tempfile
import unittest
import xml.etree.ElementTree as ET

import numpy as np
from dispatch_eta.data import FEATURES, DAY, generate, matrix, package, split, targets
from dispatch_eta.evaluate import audit, metrics, promise_policy
from dispatch_eta.intervals import bounds, calibrate


class DataContractTests(unittest.TestCase):
    def setUp(self):
        self.events = generate(events_per_day=10)

    def test_generator_is_reproducible_but_seed_is_an_input(self):
        self.assertEqual(self.events, generate(events_per_day=10))
        self.assertNotEqual(self.events, generate(seed=74, events_per_day=10))

    def test_time_split_includes_embargo_and_original_labels_are_mature(self):
        groups = split(self.events)
        self.assertEqual({k: len(v) for k, v in groups.items()}, {"train": 180, "calibration": 40, "test": 100})
        self.assertLess(max(x["label_available_minute"] for x in groups["train"]), 20 * DAY)
        self.assertLess(max(x["label_available_minute"] for x in groups["calibration"]), 26 * DAY)

    def test_all_partition_pairs_have_disjoint_route_and_courier_ids(self):
        groups = split(self.events)
        for a, b in (("train", "calibration"), ("train", "test"), ("calibration", "test")):
            for field in ("route_id", "courier_id"):
                self.assertFalse({e[field] for e in groups[a]} & {e[field] for e in groups[b]})

    def test_reused_train_test_identity_is_rejected_even_without_feature_use(self):
        for field in ("route_id", "courier_id"):
            events = deepcopy(self.events)
            test = next(e for e in events if e["day"] == 26)
            test[field] = events[0][field]
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, "identity overlap"):
                split(events)

    def test_future_feature_is_rejected(self):
        self.events[0]["feature_available_minute"] = self.events[0]["dispatch_minute"] + 1
        with self.assertRaisesRegex(ValueError, "future feature"):
            split(self.events)

    def test_nonfinite_time_and_duration_cannot_bypass_availability_checks(self):
        for field in ("dispatch_minute", "feature_available_minute", "eta_minutes", "label_available_minute"):
            for value in (float("nan"), float("inf"), -float("inf")):
                events = deepcopy(self.events)
                events[0][field] = value
                with self.subTest(field=field, value=value), self.assertRaisesRegex(ValueError, "finite numeric"):
                    split(events)

    def test_unavailable_training_and_calibration_labels_are_rejected(self):
        for day, boundary in ((17, 20), (23, 26)):
            events = deepcopy(self.events)
            event = next(e for e in events if e["day"] == day)
            event["label_available_minute"] = boundary * DAY + 1
            with self.subTest(day=day), self.assertRaisesRegex(ValueError, "not available"):
                split(events)

    def test_label_cannot_arrive_before_the_delivery(self):
        self.events[0]["label_available_minute"] = self.events[0]["dispatch_minute"]
        with self.assertRaisesRegex(ValueError, "completion"):
            split(self.events)

    def test_duplicate_events_and_false_day_fields_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "unique"):
            split(self.events + [self.events[0]])
        self.events[0]["day"] = 1
        with self.assertRaisesRegex(ValueError, "day disagrees"):
            split(self.events)

    def test_future_outcomes_identity_and_absolute_time_do_not_change_features(self):
        original = matrix(self.events)
        altered = deepcopy(self.events)
        for e in altered:
            e.update(eta_minutes=99999, label_available_minute=999999, route_id="future-route",
                     courier_id="future-courier", day=999, dispatch_minute=999999, shift=999, event_id="future")
        np.testing.assert_array_equal(original, matrix(altered))
        self.assertTrue(set(FEATURES).isdisjoint({"eta_minutes", "label_available_minute", "route_id", "courier_id", "day", "shift", "dispatch_minute"}))

    def test_nonfinite_features_and_invalid_generation_inputs_fail(self):
        self.events[0]["distance_km"] = float("nan")
        with self.assertRaises(ValueError):
            matrix(self.events)
        for args in ({"events_per_day": 0}, {"seed": "bad"}):
            with self.subTest(args=args), self.assertRaises(ValueError):
                generate(**args)

    def test_packaged_dataset_hashes_and_lf_are_verifiable(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "data/events.csv.gz"
            manifest = package(self.events, path)
            raw = gzip.decompress(path.read_bytes())
            self.assertEqual(hashlib.sha256(raw).hexdigest(), manifest["csv_sha256"])
            self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), manifest["gzip_sha256"])
            self.assertNotIn(b"\r", raw)
            self.assertEqual(len(list(csv.DictReader(raw.decode().splitlines()))), 360)


class ConformalContractTests(unittest.TestCase):
    def test_finite_sample_order_statistic_uses_n_plus_one_and_ceiling(self):
        cal = calibrate(np.arange(1, 10), np.zeros(9), alpha=0.2)
        self.assertEqual((cal.n, cal.rank, cal.radius), (9, 8, 8.0))

    def test_tied_residuals_are_conservative_without_interpolation(self):
        cal = calibrate([1] * 9, [0] * 9, 0.1)
        self.assertEqual(cal.radius, 1)
        self.assertEqual(cal.rank, 9)

    def test_too_small_calibration_returns_unbounded_not_fake_finite_radius(self):
        cal = calibrate([2], [1], 0.1)
        self.assertIsNone(cal.radius)
        lower, upper = bounds([1], cal)
        self.assertEqual(lower.tolist(), [0])
        self.assertTrue(np.isposinf(upper).all())
        result = metrics([20], [1], lower, upper)
        self.assertEqual(result["coverage"], 1)
        self.assertIsNone(result["mean_interval_width_minutes"])
        self.assertEqual(result["upper_promise"]["status"], "unbounded_promise")
        json.dumps(result, allow_nan=False)

    def test_nonnegative_support_clip_does_not_exclude_valid_small_duration(self):
        cal = calibrate(np.arange(1, 10), np.zeros(9), 0.1)
        lower, upper = bounds([2], cal)
        self.assertEqual((lower[0], upper[0]), (0, 11))
        self.assertEqual(metrics([0], [2], lower, upper)["coverage"], 1)

    def test_invalid_shapes_values_and_alpha_are_rejected(self):
        for y, p, alpha in [([], [], 0.1), ([1], [1, 2], 0.1), ([float("nan")], [1], 0.1), ([1], [1], 0), ([1], [1], 1)]:
            with self.subTest(y=y, p=p, alpha=alpha), self.assertRaises(ValueError):
                calibrate(y, p, alpha)

    def test_calibration_is_fixed_before_any_test_label_is_inspected(self):
        cal = calibrate(np.arange(1, 10), np.zeros(9), 0.2)
        before = bounds([10, 20], cal)
        # Arbitrarily changed future labels affect audit coverage, never endpoints.
        original = metrics([10, 20], [10, 20], *before)
        shifted = metrics([100, 200], [10, 20], *bounds([10, 20], cal))
        np.testing.assert_array_equal(before[0], bounds([10, 20], cal)[0])
        np.testing.assert_array_equal(before[1], bounds([10, 20], cal)[1])
        self.assertEqual(original["coverage"], 1)
        self.assertEqual(shifted["coverage"], 0)


class MetricContractTests(unittest.TestCase):
    def test_boundary_coverage_and_tail_misses_reconcile(self):
        result = metrics([1, 2, 5, 9], [2, 2, 4, 4], [1, 1, 3, 3], [3, 3, 5, 5])
        self.assertEqual((result["covered"], result["lower_misses"], result["upper_misses"]), (3, 0, 1))
        self.assertEqual(result["coverage"], 0.75)
        self.assertEqual(sum(result[k] for k in ("covered", "lower_misses", "upper_misses")), result["n"])

    def test_policy_denominator_includes_zero_lateness_and_zero_slack(self):
        result = promise_policy([10, 30], [20, 20])
        self.assertEqual(result["missed_promises"], 1)
        self.assertEqual(result["mean_late_minutes"], 5)
        self.assertEqual(result["mean_slack_minutes"], 5)
        self.assertEqual(result["mean_loss_units"], 25)

    def test_higher_promise_trades_misses_for_slack_without_roi_claim(self):
        point = promise_policy([10, 30], [20, 20])
        upper = promise_policy([10, 30], [30, 30])
        self.assertLess(upper["miss_rate"], point["miss_rate"])
        self.assertGreater(upper["mean_slack_minutes"], point["mean_slack_minutes"])

    def test_large_underestimate_is_one_sided_and_strictly_over_ten(self):
        result = metrics([30, 10, 20], [10, 30, 10], [0, 0, 0], [40, 40, 40])
        self.assertEqual(result["underestimate_over_10_minutes_count"], 1)

    def test_empty_slice_is_na_and_invalid_bounds_fail(self):
        result = metrics([], [], [], [])
        self.assertIsNone(result["coverage"])
        self.assertIsNone(result["mae_minutes"])
        self.assertEqual(result["interval_status"], "not_applicable")
        self.assertIsNone(result["point_promise"]["mean_loss_units"])
        with self.assertRaises(ValueError):
            metrics([1], [1], [2], [0])

    def test_group_counts_reconcile_without_adding_overlapping_slices(self):
        events = split(generate(events_per_day=10))["test"]
        y = targets(events)
        rows = audit(events, y, y, y, y)
        self.assertEqual(rows["all"]["n"], rows["before_shift"]["n"] + rows["mechanism_shift"]["n"])
        self.assertEqual(rows["all"]["n"], sum(rows["zone_" + z]["n"] for z in ("core", "suburban", "hills")))


class ExecutedExperimentTests(unittest.TestCase):
    def test_report_tracks_lower_tree_promise_loss_in_an_actual_small_run(self):
        from dispatch_eta.experiment import run
        with tempfile.TemporaryDirectory() as directory:
            result = run(directory, seed=73, events_per_day=10)
            tree, linear = result["test"]["xgboost"]["all"], result["test"]["linear"]["all"]
            self.assertLess(tree["upper_promise"]["mean_loss_units"], linear["upper_promise"]["mean_loss_units"])
            report = (Path(directory) / "case-study.md").read_text(encoding="utf-8")
            self.assertIn("XGBoost-upper promise loss is lower than linear-upper loss", report)
            self.assertNotIn("the better promise under", report)

    def test_report_tracks_higher_tree_mae_in_an_actual_changed_seed(self):
        from dispatch_eta.experiment import run
        with tempfile.TemporaryDirectory() as directory:
            result = run(directory, seed=74, events_per_day=10)
            self.assertGreater(result["test"]["xgboost"]["all"]["mae_minutes"], result["test"]["linear"]["all"]["mae_minutes"])
            report = (Path(directory) / "case-study.md").read_text(encoding="utf-8")
            self.assertIn("XGBoost's test MAE is higher than the linear baseline", report)

    def test_actual_xgboost_run_raw_predictions_report_links_and_hashes(self):
        from dispatch_eta.experiment import run, ROOT
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory) / "outside/reports"
            result = run(out, events_per_day=10)
            self.assertEqual(result["counts"]["test"], 100)
            self.assertEqual(result["runtime"]["device"], "cpu")
            with (out / "predictions.csv").open(encoding="utf-8", newline="") as stream:
                predictions = list(csv.DictReader(stream))
            self.assertEqual(len(predictions), 100)
            y = np.array([float(r["eta_minutes"]) for r in predictions])
            for model in ("linear", "xgboost"):
                point = np.array([float(r[model + "_prediction"]) for r in predictions])
                lo = np.array([float(r[model + "_lower"]) for r in predictions])
                hi = np.array([float(r[model + "_upper"]) for r in predictions])
                recomputed = metrics(y, point, lo, hi)
                self.assertEqual(recomputed, result["test"][model]["all"])
            for rel, expected in result["source_sha256"].items():
                self.assertEqual(hashlib.sha256((ROOT / rel).read_bytes()).hexdigest(), expected)
            report = (out / "case-study.md").read_text(encoding="utf-8")
            for link in re.findall(r'\]\(([^)]+)\)', report):
                self.assertTrue((out / link).resolve().exists(), link)
            ET.parse(out / "coverage.svg")

    def test_models_are_fitted_without_receiving_calibration_or_test_rows(self):
        from dispatch_eta.experiment import fit_models, predict
        groups = split(generate(events_per_day=10))
        models = fit_models(groups["train"])
        before = {name: predict(model, groups["test"]) for name, model in models.items()}
        for e in groups["calibration"] + groups["test"]:
            e["eta_minutes"] += 10000
        for name, model in models.items():
            np.testing.assert_array_equal(before[name], predict(model, groups["test"]))


if __name__ == "__main__":
    unittest.main()
