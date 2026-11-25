"""Draft ABRIM extrapolation benchmark: fixed tasks and scan-balanced reporting.

Consumes precomputed MRI image metrics; does not run generator inference.
This intentionally does not change existing best-checkpoint selection.
"""
from __future__ import annotations
import argparse
import csv
import json
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from statistics import fmean

METRICS = ("lpips", "ssim", "ms_ssim", "ncc", "psnr", "mae", "rmse")
WEIGHTS = dict(lpips=.25, ssim=.20, ms_ssim=.20, ncc=.15,
               psnr=.10, mae=.05, rmse=.05)
LOWER_BETTER = {"lpips", "mae", "rmse"}


@dataclass(frozen=True)
class Task:
    id: str
    inputs: tuple[str, ...]
    target: str
    difficulty: str


TASKS = (
    Task("TE1_to_TE6", ("TE1",), "TE6", "hard"),
    Task("TE3_to_TE9", ("TE3",), "TE9", "hard"),
    Task("TE1_TE2_to_TE6", ("TE1", "TE2"), "TE6", "medium"),
    Task("TE3_TE4_to_TE9", ("TE3", "TE4"), "TE9", "hard"),
    Task("TE1_TE2_TE3_to_TE6", ("TE1", "TE2", "TE3"), "TE6", "medium"),
    Task("TE3_TE4_TE5_to_TE9", ("TE3", "TE4", "TE5"), "TE9", "medium"),
    Task("TE1_TE2_TE3_to_TE9", ("TE1", "TE2", "TE3"), "TE9", "hard"),
)
BY_ID = {t.id: t for t in TASKS}


def select_tasks(spec="all"):
    """all, medium, hard, or comma-separated exact task IDs."""
    if spec == "all":
        return TASKS
    if spec in ("medium", "hard"):
        return tuple(t for t in TASKS if t.difficulty == spec)
    ids = [part.strip() for part in str(spec).split(",")]
    if not ids or any(not x for x in ids) or len(ids) != len(set(ids)):
        raise ValueError("Task IDs must be nonempty and unique")
    if set(ids) - set(BY_ID):
        raise ValueError("Unknown task IDs: " + repr(sorted(set(ids) - set(BY_ID))))
    return tuple(BY_ID[x] for x in ids)


def finite(value, name):
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(name + " must be finite") from exc
    if not math.isfinite(number):
        raise ValueError(name + " must be finite")
    return number


def frozen_bounds(calibration, modality):
    if (not isinstance(calibration, dict) or not calibration.get("version")
            or calibration.get("source_split") not in ("training", "external")
            or calibration.get("frozen") is not True):
        raise ValueError("Require versioned frozen training/external calibration")
    bounds = calibration.get("bounds", {}).get(modality)
    if not isinstance(bounds, dict) or set(bounds) != set(METRICS):
        raise ValueError(modality + ": missing metric calibration bounds")
    for metric, pair in bounds.items():
        if not isinstance(pair, (list, tuple)) or len(pair) != 2:
            raise ValueError(metric + ": bounds must be [low,high]")
        if finite(pair[0], metric) >= finite(pair[1], metric):
            raise ValueError(metric + ": bounds must increase")
    return bounds


def weighted_score(values, bounds):
    """Score in [0,1], larger is better; every metric must be present."""
    if set(values) != set(METRICS) or set(bounds) != set(METRICS):
        raise ValueError("All seven metrics and frozen bounds are mandatory")
    result = 0.
    for key in METRICS:
        low, high = map(float, bounds[key])
        if not low < high:
            raise ValueError(key + ": invalid calibration range")
        normalized = min(1., max(0., (finite(values[key], key) - low) / (high - low)))
        if key in LOWER_BETTER:
            normalized = 1. - normalized
        result += WEIGHTS[key] * normalized
    return result


def mean_metrics(rows):
    rows = list(rows)
    return {key: float(fmean(row[key] for row in rows)) for key in METRICS}


def aggregate(rows, tasks=TASKS, calibration=None):
    """All selected scenarios require exactly identical scan/chunk coverage.

    First mean chunks within case, then cases within scenario, then scenarios.
    Checkpoints are compared only on identical (case_id, chunk_id) pairs.
    """
    if not tasks or len(set(t.id for t in tasks)) != len(tasks):
        raise ValueError("At least one unique task required")
    if any(t.id not in BY_ID for t in tasks):
        raise ValueError("Unknown task")
    selected = {t.id for t in tasks}
    seen = set()
    per_case = defaultdict(list)
    coverage = defaultdict(set)
    for row in rows:
        combo = str(row.get("combination", ""))
        if combo not in BY_ID:
            raise ValueError("Unrecognized benchmark task: " + combo)
        if combo not in selected:
            continue
        if row.get("split") != "eval":
            raise ValueError("Evaluation requires split=eval, never training/test")
        mod, ckpt, case, chunk = (str(row.get(key, "")).strip() for key in
                                  ("modality", "checkpoint", "case_id", "chunk_id"))
        if mod not in ("magnitude", "phase") or not all((ckpt, case, chunk)):
            raise ValueError("Missing modality/checkpoint/case_id/chunk_id")
        key = (mod, ckpt, combo, case, chunk)
        if key in seen:
            raise ValueError("Duplicate scenario/chunk: " + repr(key))
        seen.add(key)
        per_case[key[:4]].append({k: finite(row.get(k), k) for k in METRICS})
        coverage[(mod, ckpt, combo)].add((case, chunk))
    if not seen:
        raise ValueError("No matching validation metrics")

    scopes = sorted({(m, c) for m, c, _ in coverage})
    per_modality_coverage = {}
    for mod, ckpt in scopes:
        base = coverage.get((mod, ckpt, tasks[0].id), set())
        if not base or any(coverage.get((mod, ckpt, t.id), set()) != base for t in tasks):
            raise ValueError(mod + "/" + ckpt + ": incomplete scenario/scan/chunk coverage")
        if mod in per_modality_coverage and per_modality_coverage[mod] != base:
            raise ValueError(mod + ": checkpoints have different scan/chunk coverage")
        per_modality_coverage[mod] = base

    results = []
    for mod, ckpt in scopes:
        bounds = frozen_bounds(calibration, mod) if calibration is not None else None
        details = []
        for task in tasks:
            cases = sorted({case for case, _ in coverage[(mod, ckpt, task.id)]})
            case_means = [mean_metrics(per_case[(mod, ckpt, task.id, case)])
                          for case in cases]
            scores = [weighted_score(m, bounds) for m in case_means] if bounds else None
            details.append({
                "combination": task.id, "difficulty": task.difficulty,
                "inputs": list(task.inputs), "target": task.target,
                "cases": len(cases),
                "chunks": len(coverage[(mod, ckpt, task.id)]),
                "mean_metrics": mean_metrics(case_means),
                "score": float(fmean(scores)) if scores is not None else None,
            })
        results.append({
            "modality": mod, "checkpoint": ckpt,
            "cases": len({case for case, _ in per_modality_coverage[mod]}),
            "chunks_per_combination": len(per_modality_coverage[mod]),
            "combinations": details,
            "mean_metrics": mean_metrics(d["mean_metrics"] for d in details),
            "score": float(fmean(d["score"] for d in details)) if bounds else None,
        })
    rankings = {}
    if calibration is not None:
        for mod in sorted({record["modality"] for record in results}):
            matches = [r for r in results if r["modality"] == mod]
            matches.sort(key=lambda r: (-r["score"], r["checkpoint"]))
            rankings[mod] = [r["checkpoint"] for r in matches]
    return {
        "protocol": "fixed_extrapolation_v1_draft",
        "selected_combinations": [t.id for t in tasks],
        "split": "eval", "weights": WEIGHTS,
        "calibration_version": calibration.get("version") if calibration else None,
        "results": results, "ranked_checkpoints_by_modality": rankings,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", required=True, type=Path)
    parser.add_argument("--combinations", default="all",
                        help="all, medium, hard, or comma-separated task IDs")
    parser.add_argument("--calibration", type=Path,
                        help="Frozen calibration JSON, optional; required for scores")
    parser.add_argument("--output", type=Path, help="Optional report JSON path")
    args = parser.parse_args(argv)
    calib = json.loads(args.calibration.read_text()) if args.calibration else None
    with args.csv.open(newline="", encoding="utf-8") as handle:
        result = aggregate(csv.DictReader(handle), select_tasks(args.combinations), calib)
    payload = json.dumps(result, indent=2, allow_nan=False) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload, encoding="utf-8")
    else:
        print(payload, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
