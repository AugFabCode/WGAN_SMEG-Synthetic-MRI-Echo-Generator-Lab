"""Regression tests for the dependency-free extrapolation benchmark draft."""
import unittest
from echo_gen_extrapolation_benchmark import (
    TASKS, METRICS, WEIGHTS, select_tasks, weighted_score, aggregate,
)


def measures(good=True):
    return {key: float((not good) if key in ("lpips", "mae", "rmse") else good)
            for key in METRICS}


def bounds():
    return {"version": "synthetic-v1", "frozen": True,
            "source_split": "training",
            "bounds": {"magnitude": {k: [0, 1] for k in METRICS}}}


def rows(checkpoint="epoch1"):
    data = []
    for task in TASKS:
        for case, chunk, good in (("A", "1", True), ("A", "2", False),
                                  ("B", "1", True)):
            data.append({"checkpoint": checkpoint, "modality": "magnitude",
                         "case_id": case, "chunk_id": chunk,
                         "combination": task.id, "split": "eval",
                         **measures(good)})
    return data


class BenchmarkTests(unittest.TestCase):
    def test_exact_forward_tasks_and_weights(self):
        self.assertEqual(len(TASKS), 7)
        self.assertEqual(len({task.id for task in TASKS}), 7)
        self.assertAlmostEqual(sum(WEIGHTS.values()), 1.)
        self.assertEqual(len(select_tasks("medium")), 3)
        self.assertEqual(len(select_tasks("hard")), 4)
        for task in TASKS:
            self.assertNotIn(task.target, task.inputs)
            self.assertLess(max(int(te[2:]) for te in task.inputs),
                            int(task.target[2:]))

    def test_task_selection_validation(self):
        pair = TASKS[1].id + "," + TASKS[0].id
        self.assertEqual([t.id for t in select_tasks(pair)],
                         [TASKS[1].id, TASKS[0].id])
        for spec in ("", "no_such_task", TASKS[0].id + "," + TASKS[0].id):
            with self.subTest(spec=spec), self.assertRaises(ValueError):
                select_tasks(spec)

    def test_directionality_and_full_coverage(self):
        b = {k: [0, 1] for k in METRICS}
        self.assertAlmostEqual(weighted_score(measures(True), b), 1.)
        self.assertAlmostEqual(weighted_score(measures(False), b), 0.)
        with self.assertRaises(ValueError):
            weighted_score({"ncc": 1}, b)

    def test_case_balanced_macro_mean(self):
        report = aggregate(rows(), calibration=bounds())
        item = report["results"][0]
        self.assertEqual(item["cases"], 2)
        self.assertEqual(item["chunks_per_combination"], 3)
        # Case A: 0.5 average of 2 chunks. Case B: 1.0 of 1 chunk.
        # Case mean: 0.75, not 2/3 from direct chunk pooling.
        self.assertAlmostEqual(item["score"], .75)
        self.assertAlmostEqual(item["combinations"][0]["score"], .75)
        self.assertEqual(report["ranked_checkpoints_by_modality"]["magnitude"],
                         ["epoch1"])

    def test_without_calibration_no_selection_score(self):
        report = aggregate(rows())
        self.assertIsNone(report["results"][0]["score"])
        self.assertEqual(report["ranked_checkpoints_by_modality"], {})

    def test_missing_case_or_scenario_hard_fails(self):
        for case in (rows()[:-1],
                     [r for r in rows() if r["combination"] != TASKS[0].id]):
            with self.assertRaisesRegex(ValueError, "incomplete"):
                aggregate(case)

    def test_checkpoint_must_use_same_coverage(self):
        dataset = rows("a") + [r for r in rows("b") if r["case_id"] != "B"]
        with self.assertRaisesRegex(ValueError, "different scan/chunk"):
            aggregate(dataset)

    def test_duplicate_test_and_nonfinite_rejected(self):
        original = rows()
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            aggregate(original + [original[0]])
        invalid = [dict(r) for r in original]
        invalid[0]["split"] = "test"
        with self.assertRaisesRegex(ValueError, "split=eval"):
            aggregate(invalid)
        invalid = [dict(r) for r in original]
        invalid[0]["lpips"] = float("nan")
        with self.assertRaisesRegex(ValueError, "finite"):
            aggregate(invalid)

    def test_only_external_or_training_frozen_calibration(self):
        for change in ({"frozen": False}, {"source_split": "eval"},
                       {"version": ""}):
            bad = bounds()
            bad.update(change)
            with self.subTest(change=change), self.assertRaises(ValueError):
                aggregate(rows(), calibration=bad)


if __name__ == "__main__":
    unittest.main()
