#!/usr/bin/env python3
"""Independent CPU audit of a completed EXP-038 Kaggle run.

Recomputes P2 pooled metrics, the C1/C2/C3 gates (20,000-draw painting bootstrap), the E4 - E3 interval,
Development scores and intervals, and the B1/B2 compositions from the archived predictions, without
reusing the run's metric JSON. Also verifies the export manifest, every seal, the run identity, the
tolerance-replay receipts, optimizer convergence and the CS-CVA seed facts.
"""

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from st3 import data, submission
from st3 import exp038 as science
from st3.competition_stack import compare
from st3.exp036_io import read, sha
from st3.exp037_diagnostics import decompose, pair_diversity, paired_bootstrap
from st3.exp038_io import restore_overlay, unpack_overlay, write_submission, zip_member
from st3.metrics import painting_scores, score

EXPECTED = {"overlay_manifest_sha256": "b3303f035a5765941284d2a8398f0cb6ead2f67f0286c246ea50f2b607a2d202",
            "base_manifest_sha256": "026720f406c02236aeaf5bfdc806853816323b66267f81f0e3e2dbbfe1ce7e0e",
            "source": "8b42a10775f0bfa88c4266855f21bce1fb834e94", "python": "3.12.10", "gpu": "Tesla T4"}
TOLERANCE = 1e-9


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class Audit:
    def __init__(self):
        self.mismatches, self.largest = [], 0.

    def same(self, label, mine, theirs):
        difference = abs(float(mine) - float(theirs))
        self.largest = max(self.largest, difference)
        if difference > TOLERANCE:
            self.mismatches.append({"check": label, "recomputed": mine, "archived": theirs, "difference": difference})
        return float(mine)

    def require(self, label, condition):
        if not condition:
            self.mismatches.append({"check": label, "failed": True})
        return bool(condition)


def verify_archive(results_zip, root, audit):
    with ZipFile(results_zip) as z:
        manifest = json.loads(z.read("export_manifest.json"))
        audit.require("export manifest coverage",
                      sorted(z.namelist()) == sorted([*manifest["files"], "export_manifest.json"]))
        for name, expected in manifest["files"].items():
            audit.require(f"export member {name}", hashlib.sha256(z.read(name)).hexdigest() == expected)
    run = read(root / "run.json")
    seals = {}
    for path in sorted(root.rglob("seal.json")):
        meta = read(path)
        identity = meta["identity"]
        if path.parent.parent.name == "features":
            # Feature caches seal the run identity plus their own arm, encoder spec and chunk size.
            extra = {k: identity.get(k) for k in ("arm", "spec", "chunk_items")}
            identity = {k: v for k, v in identity.items() if k not in extra}
            audit.require(f"seal {path.parent.name} arm", extra["arm"] == path.parent.name and extra["spec"])
        ok = identity == run and all(digest(path.parent / n) == d for n, d in meta["files"].items())
        seals[path.parent.relative_to(root).as_posix()] = ok
        audit.require(f"seal {path.parent.relative_to(root).as_posix()}", ok)
    for key, value in EXPECTED.items():
        audit.require(f"identity {key}", run[key] == value)
    lock = {line.split("==")[0].lower(): line.split("==")[1] for line in
            (ROOT / "requirements-exp035-kaggle.lock.txt").read_text().splitlines() if "==" in line and not line.startswith("#")}
    for name, version in run["packages"].items():
        pinned = lock.get(name.lower()) or {"qwen-vl-utils": "0.0.14"}.get(name)
        audit.require(f"package {name}=={version} pinned", pinned is None or pinned.split()[0] == version)
    return {"export_files": len(manifest["files"]), "export_identity": manifest["identity"], "seals": seals,
            "run": run}


def scoped(joined, excluded):
    return (("full", joined), ("filtered", joined.loc[~joined.language.isin(excluded)]))


def pooled(block, name):
    result = score(block[data.KEYS + ["k", "n"]], block[data.KEYS].assign(p_positive=block[name].to_numpy()))
    return {k: result[k] for k in ("cultural_contrast", "log_loss")}


def contrasts(block, names):
    table = block[["painting", "fold"]].drop_duplicates().set_index("painting")
    for name in names:
        table[name] = painting_scores(block, prediction=name).set_index("painting").contrast
    return table.dropna(subset=list(names))


def p2_audit(root, gold, config, audit):
    oof = pd.read_csv(root / "p2/oof.csv")
    archived = read(root / "p2/metrics.json")
    gates_archived = read(root / "p2/gates.json")
    diagnostics_archived = read(root / "p2/diagnostics.json")
    joined = gold[data.KEYS + ["fold", "k", "n"]].merge(oof.drop(columns="fold"), on=data.KEYS, validate="one_to_one")
    joined["p"] = joined.k / joined.n
    systems = [c for c in oof.columns if c not in (*data.KEYS, "fold")]
    rule, excluded = config["gate"], config["test_like_excluded"]
    out = {"systems": {}, "gates": {}, "standalone": {}}
    for scope, block in scoped(joined, excluded):
        for name in systems:
            mine = pooled(block, name)
            theirs = archived["report"][scope]["systems"][name]["pooled"]
            out["systems"].setdefault(name, {})[scope] = {
                k: audit.same(f"P2 {scope} {name} {k}", mine[k], theirs[k]) for k in mine}
            folds = {int(f): pooled(part, name) for f, part in block.groupby("fold")}
            for f, value in folds.items():
                audit.same(f"P2 {scope} {name} fold {f} Contrast", value["cultural_contrast"],
                           archived["report"][scope]["systems"][name]["folds"][str(f)]["cultural_contrast"])
    reference = rule["reference"]
    for name in config["candidates"]:
        record = {}
        for scope, block in scoped(joined, excluded):
            table = contrasts(block, [name, reference])
            delta = table[name] - table[reference]
            stats = compare(delta, rule)
            folds = {int(f): pooled(part, name)["cultural_contrast"] - pooled(part, reference)["cultural_contrast"]
                     for f, part in block.groupby("fold")}
            a, b = pooled(block, name), pooled(block, reference)
            record[scope] = {"contrast": a["cultural_contrast"], "delta": a["cultural_contrast"] - b["cultural_contrast"],
                             "bootstrap_95": stats["bootstrap_95"], "fold_wins": sum(v > 0 for v in folds.values()),
                             "trimmed_delta": stats["trimmed_contrast_delta"],
                             "log_loss_delta": a["log_loss"] - b["log_loss"],
                             "painting_wins": stats["wins"], "painting_losses": stats["losses"]}
        f, u = record["filtered"], record["full"]
        passed = (f["delta"] >= rule["minimum_filtered_gain"] and f["bootstrap_95"][0] > 0
                  and f["fold_wins"] >= rule["minimum_fold_wins"] and f["trimmed_delta"] > 0 and u["delta"] >= 0
                  and u["log_loss_delta"] <= rule["maximum_log_loss_delta"]
                  and f["log_loss_delta"] <= rule["maximum_log_loss_delta"])
        numbers = gates_archived["gates"][name]["numbers"]
        for key, mine in (("filtered_contrast_delta", f["delta"]), ("filtered_trimmed_delta", f["trimmed_delta"]),
                          ("full_contrast_delta", u["delta"]), ("full_log_loss_delta", u["log_loss_delta"]),
                          ("filtered_log_loss_delta", f["log_loss_delta"]), ("filtered_contrast", f["contrast"])):
            audit.same(f"gate {name} {key}", mine, numbers[key])
        for i in range(2):
            audit.same(f"gate {name} bootstrap bound {i}", f["bootstrap_95"][i], numbers["filtered_bootstrap_95"][i])
        audit.require(f"gate {name} fold wins", f["fold_wins"] == numbers["filtered_fold_wins"])
        audit.require(f"gate {name} decision", passed == gates_archived["gates"][name]["passed"])
        out["gates"][name] = record | {"passed": passed}
    passing = [n for n in config["complexity_order"] if out["gates"][n]["passed"]]
    best = max(out["gates"][n]["filtered"]["contrast"] for n in passing) if passing else None
    selected = next((n for n in passing if best - out["gates"][n]["filtered"]["contrast"] <= rule["tie_tolerance"]), None)
    audit.require("selection rule", selected == gates_archived["selected"])
    out["selected"] = selected
    out["gate_reference"] = archived["gate_reference"]
    audit.require("gate reference is on-platform", archived["gate_reference"].startswith("on-platform R0_corrected"))
    block = joined.loc[~joined.language.isin(excluded)]
    scored = painting_scores(block, prediction="R0_fixed")
    eligible = set(scored.loc[scored.status.eq("scored"), "painting"])
    for name in ("E1", "E2", "E3", "E4"):
        summary = decompose(block, name)[0]
        diversity = pair_diversity(block, ["R0_fixed", name], eligible)[0]
        out["standalone"][name] = {"filtered_contrast": out["systems"][name]["filtered"]["cultural_contrast"],
                                   "filtered_log_loss": out["systems"][name]["filtered"]["log_loss"],
                                   "global_rmse": summary["global_rmse"],
                                   "centered_residual_r_vs_fixed": diversity["centered_residual_r"]}
        audit.same(f"diagnostic {name} centered residual r", diversity["centered_residual_r"],
                   diagnostics_archived["filtered"]["diversity_vs_fixed"][name]["centered_residual_r"])
    out["standalone"]["R0_fixed"] = {"filtered_contrast": out["systems"]["R0_fixed"]["filtered"]["cultural_contrast"],
                                     "filtered_log_loss": out["systems"]["R0_fixed"]["filtered"]["log_loss"],
                                     "global_rmse": decompose(block, "R0_fixed")[0]["global_rmse"]}
    for scope, block in scoped(joined, excluded):
        table = contrasts(block, ["E3", "E4"])
        interval = paired_bootstrap(table.E4, table.E3, seed=config["diagnostic_seed"])
        out.setdefault("E4_minus_E3", {})[scope] = interval
        archived_interval = diagnostics_archived[scope]["E4_minus_E3"]
        audit.same(f"E4-E3 {scope} delta", interval["delta"], archived_interval["delta"])
        for i in range(2):
            audit.same(f"E4-E3 {scope} ci {i}", interval["ci95"][i], archived_interval["ci95"][i])
    convergence = []
    for fold, receipt in archived["fold_receipts"].items():
        fits = [receipt["outer_fits"], *[split["optimizers"] for split in receipt["inner"]["inner_folds"]]]
        for group in fits:
            for arm, fit in group.items():
                convergence.append(fit["converged"] and fit["max_abs_gradient"] <= 5e-7)
    out["fits_converged"] = {"fits": len(convergence), "all": all(convergence)}
    audit.require("P2 fits converged", all(convergence))
    on_platform = science.aligned_values(gold, oof, ["R0_corrected"]).ravel()
    return out, on_platform


def development_audit(root, overlay, base, config, audit):
    dev = data.read_dev_input(base)
    gold = pd.read_csv(base / "scoring/dev_gold.csv")
    scores = read(root / "development/scores.json")
    excluded = config["test_like_excluded"]
    frames = {name: submission.validate_archive(root / item["zip"], dev) for name, item in scores["archives"].items()}
    archived_refs = {"EXP036_ensemble_archived": submission.validate_archive(overlay / "reference/exp036_ensemble_submission.zip", dev),
                     "EXP037_original_blend_archived": submission.validate_archive(overlay / "reference/exp037_best_submission.zip", dev)}
    recomputed = science.development_scores(gold, frames | archived_refs, excluded)
    for name, value in recomputed.items():
        for scope in ("full", "evaluation_like"):
            for key in ("cultural_contrast", "log_loss"):
                audit.same(f"Dev {name} {scope} {key}", value[scope][key], scores["scores"][name][scope][key])
    references = {"EXP036_ensemble": frames["B1_R0_original"], "EXP037_original_blend": frames["B2_R0_original"]}
    intervals = science.development_intervals(gold, {k: v for k, v in frames.items() if k.startswith("B") and "_R0_" not in k},
                                              references, excluded, config["diagnostic_seed"])
    for scope, values in intervals.items():
        for key, value in values.items():
            audit.same(f"Dev interval {scope} {key} delta", value["delta"], scores["intervals"][scope][key]["delta"])
            for i in range(2):
                audit.same(f"Dev interval {scope} {key} ci {i}", value["ci95"][i], scores["intervals"][scope][key]["ci95"][i])
    # Recompose every B1/B2 from its components and compare ZIP member bytes.
    cscva = {"original": submission.validate_archive(overlay / "reference/exp036_standalone_submission.zip", dev)}
    seeds = [frames[f"CSCVA_seed_{s}"] for s in config["cscva"]["new_seeds"]]
    cscva["cscva3"] = science.mean_of([cscva["original"], *seeds], dev)
    c6 = submission.read_predictions(overlay / "reference/exp037_c6_standalone/predictions.csv", dev)
    bases = {"R0": frames["R0_corrected"]} | {n: frames[n] for n in config["candidates"]}
    composed = science.blends(bases, cscva, c6, dev, config)
    scratch = root.parent / "tmp_recompose"
    import shutil

    shutil.rmtree(scratch, ignore_errors=True)
    identical = {}
    for name, frame in sorted(composed.items()):
        archive = write_submission(science.canonical(frame, dev), dev, scratch, name)
        identical[name] = zip_member(archive) == zip_member(root / scores["archives"][name]["zip"])
        audit.require(f"recompose {name}", identical[name])
    cs3 = write_submission(science.canonical(cscva["cscva3"], dev), dev, scratch, "cscva3")
    identical["CSCVA_3"] = zip_member(cs3) == zip_member(root / scores["archives"]["CSCVA_3"]["zip"])
    audit.require("recompose CSCVA_3", identical["CSCVA_3"])
    shutil.rmtree(scratch, ignore_errors=True)
    audit.require("development_used_for_selection is none", scores["development_selection"].startswith("none"))
    fit = scores["fit"]
    groups = [fit["fits"], *[split["optimizers"] for split in fit["crossfit"]["inner_folds"]]]
    converged = [g["converged"] and g["max_abs_gradient"] <= 5e-7 for group in groups for g in group.values()]
    audit.require("Development fits converged", all(converged))
    return {"scores": recomputed, "intervals": intervals, "recomposed_member_identical": identical,
            "fits_converged": {"fits": len(converged), "all": all(converged)},
            "development_selection": scores["development_selection"]}


def replay_audit(root, audit):
    validate = read(root / "validate/validate.json")
    checks = validate["r0_p2"]["checks"] + validate["r0_development"]["checks"]
    ratios = [c["max_abs_difference"] / c["threshold"] for c in checks]
    for c in checks:
        audit.require(f"replay {c['check']} within threshold", c["max_abs_difference"] <= c["threshold"])
    worst = max(checks, key=lambda c: c["max_abs_difference"] / c["threshold"])
    return {"checks": len(checks), "exact_checks": sum(c["exact"] for c in checks),
            "max_ratio_to_threshold": max(ratios), "worst": worst, "platform": validate["platform"],
            "inner_hashes_matched": {k: v["matched"] for k, v in validate["r0_p2"]["inner_hashes"].items()},
            "corrected_csv_sha_matches": validate["r0_p2"]["corrected_csv_sha256_matches_reference"],
            "pooled_onplatform": validate["r0_p2"]["pooled"]}


def cscva_audit(root, config):
    out = {}
    for seed in config["cscva"]["new_seeds"]:
        train = read(root / f"cscva/seed_{seed}/train.json")
        bundle = read(root / f"cscva/seed_{seed}/inference/bundle.json")
        out[str(seed)] = {"seed": train["seed"], "freeze_epoch": train["freeze_epoch"],
                          "schedule_epochs": train["schedule_epochs"], "steps_per_epoch": train["steps_per_epoch"],
                          "epoch_seconds": [h["seconds"] for h in train["history"]],
                          "mean_epoch_seconds": float(np.mean([h["seconds"] for h in train["history"]])),
                          "development_inference_seconds": train["development_inference_seconds"],
                          "bundle_epoch": bundle["epoch"], "bundle_config_is_original": bundle["config"]["seed"] == 203036,
                          "development_epoch_selection": False}
    return out


def budget_audit(root):
    plan = read(root / "budget_plan.json")
    events = (root / "events.log").read_text(encoding="utf-8").splitlines()
    stamps = []
    for line in events:
        match = re.match(r"\[(\d\d):(\d\d):(\d\d)\] (.*?): elapsed (\d+)s", line)
        if match:
            stamps.append((match.group(4), int(match.group(5))))
    seen = {}
    for i, (item, elapsed) in enumerate(stamps):
        nxt = stamps[i + 1][1] if i + 1 < len(stamps) else None
        seen.setdefault(item, (elapsed, nxt))
    actual = {}
    for label, item in (("E1", "Feature extraction / E1"), ("E2", "Feature extraction / E2"),
                        ("E3", "Feature extraction / E3"), ("E4", "Feature extraction / E4"),
                        ("cscva_seed_2", "CS-CVA seeds / cscva_seed_2 (203836)"),
                        ("cscva_seed_3", "CS-CVA seeds / cscva_seed_3 (203936)")):
        if item in seen and seen[item][1] is not None:
            actual[label] = seen[item][1] - seen[item][0]
    status = read(root / "status.json")
    return {"forecast_seconds": plan["forecast_seconds"], "planned_seconds": plan["planned_seconds"],
            "actual_seconds": actual, "elapsed_at_export": status["elapsed_seconds"],
            "remaining_at_export": status["remaining_seconds"], "skipped": plan["skipped"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--closure", type=Path, required=True, help="runs/exp038/closure with the Kaggle outputs")
    parser.add_argument("--overlay-zip", type=Path, required=True)
    parser.add_argument("--base", type=Path, required=True, help="Extracted exp037_inputs root")
    args = parser.parse_args()
    closure = args.closure
    audit = Audit()
    files = {name: {"bytes": (closure / name).stat().st_size, "sha256": digest(closure / name)}
             for name in ("exp038_results.zip", "exp038_best_submission.zip", "exp038_progress.zip")
             if (closure / name).is_file()}
    root = closure / "results"
    config = read(root / "configs_used.json")["exp038"]
    overlay_sha = EXPECTED["overlay_manifest_sha256"]
    mounted = unpack_overlay(args.overlay_zip, closure / "tmp_overlay_zip", overlay_sha, sha(args.overlay_zip))
    overlay = restore_overlay(mounted, overlay_sha, EXPECTED["base_manifest_sha256"], closure / "tmp_overlay")
    gold = pd.read_csv(overlay / "p2/pairs.csv").sort_values(data.KEYS, ignore_index=True)
    archive = verify_archive(closure / "exp038_results.zip", root, audit)
    p2, on_platform = p2_audit(root, gold, config, audit)
    saved = science.aligned_values(gold, pd.read_csv(overlay / "reference/corrected_oof.csv"), ["corrected"]).ravel()
    p2["gate_reference_vs_archived_csv"] = {"rows_differing": int((on_platform != saved).sum()),
                                            "max_abs_difference": float(np.abs(on_platform - saved).max())}
    development = development_audit(root, overlay, args.base, config, audit)
    best = closure / "exp038_best_submission.zip"
    selected_zip = root / "development/submissions/exp038_B2_C2_cscva3_submission.zip"
    audit.require("best ZIP is B2_C2_cscva3", digest(best) == digest(selected_zip) == read(root / "selection.json")["best_sha256"])
    frame = submission.validate_archive(best, data.read_dev_input(args.base))
    result = {"files": files, "archive": archive, "p2": p2, "development": development,
              "replay": replay_audit(root, audit), "cscva": cscva_audit(root, config), "budget": budget_audit(root),
              "selection": read(root / "selection.json"), "decision": read(root / "decision.json")["decision"],
              "best_zip": {"rows": len(frame), "unique_keys": not frame.duplicated(data.KEYS).any(),
                           "min": float(frame.p_positive.min()), "max": float(frame.p_positive.max())},
              "mismatches": audit.mismatches, "largest_metric_difference": audit.largest,
              "metric_tolerance": TOLERANCE, "overlay_zip_sha256": sha(args.overlay_zip)}
    import shutil

    for scratch in ("tmp_overlay_zip", "tmp_overlay"):
        shutil.rmtree(closure / scratch, ignore_errors=True)
    (closure / "audit.json").write_text(json.dumps(result, indent=1, default=float) + "\n", encoding="utf-8")
    print(json.dumps({"mismatches": audit.mismatches, "largest_metric_difference": audit.largest,
                      "selected": p2["selected"], "decision": result["decision"]}, indent=1))


if __name__ == "__main__":
    main()
