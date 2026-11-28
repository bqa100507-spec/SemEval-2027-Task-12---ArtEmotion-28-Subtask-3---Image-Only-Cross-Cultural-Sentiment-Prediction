#!/usr/bin/env python3
"""Independent CPU audit of the official EXP-039 run (run 2), plus its reproducibility against run 1.

Recomputes P2 pooled metrics, the D-candidate gates against on-platform C2 (20,000-draw painting bootstrap),
Development scores and intervals, and every B1/B2 composition from the archived predictions, without
reusing the run's metric JSON. Verifies the export manifest, every seal and the run identity.
"""

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
for _path in (ROOT / "src", ROOT / "scripts"):
    sys.path.insert(0, str(_path))

from audit_exp038 import Audit, contrasts, pooled, scoped  # noqa: E402
from st3 import data, submission  # noqa: E402
from st3 import exp038 as science  # noqa: E402
from st3.competition_stack import compare  # noqa: E402
from st3.exp036_io import read, sha  # noqa: E402
from st3.exp037_diagnostics import decompose, paired_bootstrap  # noqa: E402
from st3.exp038_io import restore_overlay, unpack_overlay, write_submission, zip_member  # noqa: E402

EXPECTED = {"overlay_manifest_sha256": "1ebcfca80a6e0eea5387be22bd4017b4b9807aa012395a67a78fce20fae9cad0",
            "base_manifest_sha256": "026720f406c02236aeaf5bfdc806853816323b66267f81f0e3e2dbbfe1ce7e0e",
            "source": "8f99b9fa74610b4b2f5f470b6a27dcc94d879d41", "python": "3.12.10", "gpu": "Tesla T4"}
GATE_KEYS = ("filtered_contrast_delta", "filtered_bootstrap_95", "filtered_fold_wins", "filtered_trimmed_delta",
             "full_contrast_delta", "full_log_loss_delta", "filtered_log_loss_delta", "filtered_contrast")


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify_archive(results_zip, root, audit):
    with ZipFile(results_zip) as z:
        manifest = json.loads(z.read("export_manifest.json"))
        audit.require("export coverage", sorted(z.namelist()) == sorted([*manifest["files"], "export_manifest.json"]))
        for name, expected in manifest["files"].items():
            audit.require(f"export member {name}", hashlib.sha256(z.read(name)).hexdigest() == expected)
    run = read(root / "run.json")
    extra = read(root / "run39.json")
    identity = run | extra
    seals = {}
    for path in sorted(root.rglob("seal.json")):
        meta = read(path)
        ident = dict(meta["identity"])
        if path.parent.parent.name == "features":
            audit.require(f"seal {path.parent.name} arm", ident.get("arm") == path.parent.name)
            for key in ("arm", "spec", "chunk_items"):
                ident.pop(key, None)
        ok = ident == identity and all(digest(path.parent / n) == d for n, d in meta["files"].items())
        seals[path.parent.relative_to(root).as_posix()] = ok
        audit.require(f"seal {path.parent.relative_to(root).as_posix()}", ok)
    for key, value in EXPECTED.items():
        audit.require(f"identity {key}", run[key] == value)
    return {"export_files": len(manifest["files"]), "export_identity": manifest["identity"], "seals": seals,
            "run": run, "run39": extra}


def p2_audit(root, gold, config, audit):
    oof = pd.read_csv(root / "p2/oof.csv")
    archived = read(root / "p2/metrics.json")
    gates = read(root / "p2/gates.json")
    joined = gold[data.KEYS + ["fold", "k", "n"]].merge(oof.drop(columns="fold"), on=data.KEYS, validate="one_to_one")
    joined["p"] = joined.k / joined.n
    systems = [c for c in oof.columns if c not in (*data.KEYS, "fold")]
    rule, excluded = config["gate"], config["test_like_excluded"]
    out = {"systems": {}, "gates": {}}
    for scope, block in scoped(joined, excluded):
        for name in systems:
            mine = pooled(block, name)
            theirs = archived["report"][scope]["systems"][name]["pooled"]
            out["systems"].setdefault(name, {})[scope] = {
                k: audit.same(f"P2 {scope} {name} {k}", mine[k], theirs[k]) for k in mine}
    reference = rule["reference"]
    for name, gate in gates["gates"].items():
        if gate["status"] == "NOT_AVAILABLE":
            out["gates"][name] = {"status": "NOT_AVAILABLE", "reason": gate["reason"]}
            continue
        record = {}
        for scope, block in scoped(joined, excluded):
            table = contrasts(block, [name, reference])
            stats = compare(table[name] - table[reference], rule)
            folds = {int(f): pooled(part, name)["cultural_contrast"] - pooled(part, reference)["cultural_contrast"]
                     for f, part in block.groupby("fold")}
            a, b = pooled(block, name), pooled(block, reference)
            record[scope] = {"contrast": a["cultural_contrast"], "delta": a["cultural_contrast"] - b["cultural_contrast"],
                             "bootstrap_95": stats["bootstrap_95"], "fold_wins": sum(v > 0 for v in folds.values()),
                             "trimmed_delta": stats["trimmed_contrast_delta"], "log_loss_delta": a["log_loss"] - b["log_loss"],
                             "painting_wins": stats["wins"], "painting_losses": stats["losses"]}
        f, u = record["filtered"], record["full"]
        passed = (f["delta"] >= rule["minimum_filtered_gain"] and f["bootstrap_95"][0] > 0
                  and f["fold_wins"] >= rule["minimum_fold_wins"] and f["trimmed_delta"] > 0 and u["delta"] >= 0
                  and u["log_loss_delta"] <= rule["maximum_log_loss_delta"]
                  and f["log_loss_delta"] <= rule["maximum_log_loss_delta"])
        numbers = gate["numbers"]
        for key, mine in (("filtered_contrast_delta", f["delta"]), ("filtered_trimmed_delta", f["trimmed_delta"]),
                          ("full_contrast_delta", u["delta"]), ("full_log_loss_delta", u["log_loss_delta"]),
                          ("filtered_log_loss_delta", f["log_loss_delta"]), ("filtered_contrast", f["contrast"])):
            audit.same(f"gate {name} {key}", mine, numbers[key])
        for i in range(2):
            audit.same(f"gate {name} bootstrap {i}", f["bootstrap_95"][i], numbers["filtered_bootstrap_95"][i])
        audit.require(f"gate {name} folds", f["fold_wins"] == numbers["filtered_fold_wins"])
        audit.require(f"gate {name} decision", passed == gate["passed"])
        out["gates"][name] = record | {"passed": passed}
    audit.require("NO_GO selection", gates["selected"] is None and not any(g.get("passed") for g in out["gates"].values()))
    audit.require("gate reference is on-platform C2", archived["gate_reference"].startswith("on-platform C2"))
    out["gate_reference"] = archived["gate_reference"]
    out["c2_replay_in_p2"] = {"checks": len(archived["c2_replay"]),
                              "exact": all(c["exact"] for c in archived["c2_replay"]),
                              "max_abs_difference": max(c["max_abs_difference"] for c in archived["c2_replay"])}
    block = joined.loc[~joined.language.isin(excluded)]
    out["standalone_filtered"] = {name: out["systems"][name]["filtered"] | {
        "global_rmse": decompose(block, name)[0]["global_rmse"]} for name in ("R0_fixed", "E3", "E6", "E7")
        if name in out["systems"]}
    for scope, block in scoped(joined, excluded):
        for arm in ("E6", "E7"):
            table = contrasts(block, [arm, "E3"])
            out.setdefault("minus_E3", {}).setdefault(scope, {})[arm] = paired_bootstrap(table[arm], table["E3"],
                                                                                          seed=config["diagnostic_seed"])
    return out, gates


def development_audit(root, overlay, base, config, audit):
    dev = data.read_dev_input(base)
    gold = pd.read_csv(base / "scoring/dev_gold.csv")
    scores = read(root / "development/scores.json")
    excluded = config["test_like_excluded"]
    frames = {name: submission.validate_archive(root / item["zip"], dev) for name, item in scores["archives"].items()}
    recomputed = science.development_scores(gold, frames, excluded)
    for name, value in recomputed.items():
        for scope in ("full", "evaluation_like"):
            for key in ("cultural_contrast", "log_loss"):
                audit.same(f"Dev {name} {scope} {key}", value[scope][key], scores["scores"][name][scope][key])
    cscva3 = submission.validate_archive(overlay / "exp038/CSCVA_3_submission.zip", dev)
    c6 = submission.read_predictions(overlay / "reference/exp037_c6_standalone/predictions.csv", dev)
    bases = {name: frames[name] for name in frames if name in ("C2", "D1", "D2", "D3", "D4")}
    composed = science.blends(bases, {"cscva3": cscva3}, c6, dev, config)
    scratch = root.parent / "tmp_recompose"
    shutil.rmtree(scratch, ignore_errors=True)
    identical = {}
    for name, frame in sorted(composed.items()):
        archive = write_submission(science.canonical(frame, dev), dev, scratch, name)
        identical[name] = zip_member(archive) == zip_member(root / scores["archives"][name]["zip"])
        audit.require(f"recompose {name}", identical[name])
    diagnostic = {f"{n}_plus_C6": science.mix_pair(bases[n], c6, config["blends"]["B2_c6_weight"], dev) for n in bases}
    diagnostic_scores = science.development_scores(gold, {k: science.canonical(v, dev) for k, v in diagnostic.items()},
                                                   excluded)
    for name, value in diagnostic_scores.items():
        for scope in ("full", "evaluation_like"):
            audit.same(f"Dev diagnostic {name} {scope}", value[scope]["cultural_contrast"],
                       scores["diagnostic_not_selectable"][name][scope]["cultural_contrast"])
    shutil.rmtree(scratch, ignore_errors=True)
    incumbent = overlay / "exp038/incumbent_submission.zip"
    b2c2 = root / scores["archives"]["B2_C2_cscva3"]["zip"]
    diff = np.abs(submission.validate_archive(b2c2, dev).p_positive.to_numpy()
                  - submission.validate_archive(incumbent, dev).p_positive.to_numpy())
    audit.require("development_used_for_selection is none", scores["development_selection"].startswith("none"))
    return {"scores": recomputed, "diagnostic_not_selectable": diagnostic_scores, "recomposed_identical": identical,
            "b2_c2_vs_incumbent": {"rows_differing": int((diff > 0).sum()), "max_abs_difference": float(diff.max()),
                                   "member_identical": zip_member(b2c2) == zip_member(incumbent)}}


def run1_comparison(run1_zip, gates2, audit):
    with ZipFile(run1_zip) as z:
        gates1 = json.loads(z.read("p2/gates.json"))["gates"]
    out = {}
    for name in ("D2", "D3", "D4"):
        a, b = gates1[name]["numbers"], gates2["gates"][name]["numbers"]
        equal = all(a[k] == b[k] for k in GATE_KEYS)
        audit.require(f"run1 == run2 gate {name}", equal)
        out[name] = {"identical": equal, "run1": {k: a[k] for k in GATE_KEYS}}
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--closure", type=Path, required=True)
    parser.add_argument("--run1", type=Path, required=True)
    parser.add_argument("--overlay-zip", type=Path, required=True)
    parser.add_argument("--base", type=Path, required=True)
    args = parser.parse_args()
    closure, audit = args.closure, Audit()
    results = closure / "exp039_results.zip"
    root = closure / "results"
    shutil.rmtree(root, ignore_errors=True)
    with ZipFile(results) as z:
        z.extractall(root)
    config = read(root / "configs_used.json")["exp039"]
    sha_manifest = EXPECTED["overlay_manifest_sha256"]
    mounted = unpack_overlay(args.overlay_zip, closure / "tmp_m", sha_manifest, sha(args.overlay_zip),
                             manifest_name="exp039_manifest.json", prefix="exp039_inputs/")
    overlay = restore_overlay(mounted, sha_manifest, EXPECTED["base_manifest_sha256"], closure / "tmp_o",
                              manifest_name="exp039_manifest.json")
    gold = pd.read_csv(overlay / "p2/pairs.csv").sort_values(data.KEYS, ignore_index=True)
    archive = verify_archive(results, root, audit)
    p2, gates = p2_audit(root, gold, config, audit)
    development = development_audit(root, overlay, args.base, config, audit)
    validate = read(root / "validate/validate.json")
    status = read(root / "status.json")
    best = closure / "exp039_best_submission.zip"
    audit.require("best ZIP is the incumbent", digest(best) == config["references"]["exp038_carried_sha256"]["incumbent_zip"]
                  == read(root / "selection.json")["best_sha256"])
    result = {"files": {n: {"bytes": (closure / n).stat().st_size, "sha256": digest(closure / n)}
                        for n in ("exp039_results.zip", "exp039_best_submission.zip", "exp039_progress.zip")
                        if (closure / n).is_file()},
              "run1_results_sha256": digest(args.run1), "archive": archive, "p2": p2, "development": development,
              "run1_vs_run2": run1_comparison(args.run1, gates, audit),
              "validate": {k: v.get("status") for k, v in validate.items() if isinstance(v, dict)},
              "key_audit": validate.get("key_audit", {}).get("result"),
              "pieces": status["pieces"], "decision": read(root / "decision.json")["decision"],
              "mismatches": audit.mismatches, "largest_metric_difference": audit.largest}
    for name in ("tmp_m", "tmp_o"):
        shutil.rmtree(closure / name, ignore_errors=True)
    (closure / "audit.json").write_text(json.dumps(result, indent=1, default=float) + "\n", encoding="utf-8")
    print(json.dumps({"mismatches": audit.mismatches, "largest": audit.largest, "decision": result["decision"]}, indent=1))


if __name__ == "__main__":
    main()
