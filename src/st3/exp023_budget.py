"""Measured runtime forecasts and conservative admission of one original fit at a time."""

import math
import statistics


def _positive(value, name):
    value = float(value)
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"Invalid measured {name}: {value}")
    return value


def condition_cost(sample, max_epochs, fit_pairs, operations, observations=()):
    """No score-based selection: a full max-epoch fit still anchors admission safety."""
    _positive(sample["pairs"], "inner pairs")
    full_pairs = _positive(sample["full_pairs"], "full pairs")
    inner = _positive(sample["inner_training_epoch_seconds"], "inner epoch") * fit_pairs / full_pairs
    validation = _positive(sample["validation_epoch_seconds"], "validation epoch") * fit_pairs / full_pairs
    refit = _positive(sample["full_refit_epoch_seconds"], "refit epoch") * fit_pairs / full_pairs
    # A slower completed fit raises the admission bound; fast stopping never lowers
    # the configured 150+150 epoch allowance for the next model.
    observed_epoch = max((float(r["fit_seconds"]) / (int(r["selection_epochs"]) + int(r["selected_epoch"]))
                          for r in observations), default=0.)
    training_upper = max(max_epochs * (inner + validation + refit), 2 * max_epochs * observed_epoch)
    io = max(float(operations["minimum_fit_io_seconds"]), float(sample["checkpoint_seconds"]) + float(sample["replay_seconds"]))
    io = max(io, max((float(r["end_to_end_seconds"]) - float(r["fit_seconds"]) for r in observations), default=0.))
    admission = operations["forecast_safety_factor"] * (training_upper + io)
    observed = [float(r["end_to_end_seconds"]) for r in observations]
    expected = statistics.mean(observed) if observed else training_upper + io
    return {"admission_seconds": admission, "expected_fit_seconds": expected,
            "expected_basis": "completed same-condition fits" if observed else "max-epoch planning proxy; stopping distribution unknown",
            "observed_fits": len(observed), "observed_min_seconds": min(observed) if observed else None,
            "observed_max_seconds": max(observed) if observed else None,
            "selection_epochs": [int(r["selection_epochs"]) for r in observations],
            "refit_epochs": [int(r["selected_epoch"]) for r in observations]}


def forecast_work(samples, remaining, completed, config, operations, *, fit_pairs, target_seconds, overhead):
    """Global estimate is diagnostic only; admission is always for one complete fit."""
    by_condition = {row["condition"]: row for row in samples}
    if len(samples) != len(config["conditions"]) or set(by_condition) != set(config["conditions"]):
        raise ValueError("Timing coverage must contain each fixed condition exactly once")
    details, expected, worst = {}, 0., 0.
    observed = 0
    for condition in config["conditions"]:
        rows = [r for r in completed if r["condition"] == condition]
        cost = condition_cost(by_condition[condition], config["max_epochs"], fit_pairs, operations, rows)
        count = sum(item[1] == condition for item in remaining)
        details[condition] = {**cost, "remaining_fits": count}
        expected += count * cost["expected_fit_seconds"]
        worst += count * cost["admission_seconds"]
        observed += len(rows)
    overhead = float(overhead)
    factor = operations["forecast_safety_factor"]
    # Preserve the former estimate as an explicitly named diagnostic as well.
    legacy_epoch = max(row["full_inner_epoch_seconds"] * fit_pairs / row["pairs"] for row in samples)
    legacy = factor * (len(remaining) * config["max_epochs"] * 2 * legacy_epoch + 12 * target_seconds + 300 + 120)
    return {"theoretical_worst_case_seconds": max(worst + factor * overhead, legacy),
            "original_max_epoch_formula_seconds": legacy,
            "measured_expected_remaining_seconds": factor * (expected + overhead) if observed else None,
            "planning_proxy_remaining_seconds": factor * (expected + overhead),
            "estimated_remaining_seconds": factor * (expected + overhead),
            "completed_fit_observations": observed, "remaining_fits": len(remaining), "per_condition": details,
            "safety_factor": factor, "non_fit_overhead_seconds": overhead,
            "limitations": "Engineering heuristics, not confidence intervals. Initially no stopping/refit observations; unknown conditions use a max-epoch proxy. Admission uses a conservative complete-fit bound, never the whole-job forecast."}
