"""Strict nested EXP-012 evaluation and guarded development scoring."""

from dataclasses import replace
import time

import numpy as np
import pandas as pd

from . import data, submission
from .ablation import predict_lr
from .baseline import file_sha256
from .cli import ROOT, json_output
from .clip_large_study import E50_LARGE
from .contrast_study import new_output, write_development
from .e50_heads import ENCODERS, MANIFEST, inference_config, load_bundle, load_legacy, read_json
from .e50l_heads import CLIP_LARGE, load_clip_large_bundle, load_clip_large_legacy
from .fusion_study import write_cv_analysis
from .low_rank_study import run_metadata, score_result
from .residual import MODELS, Residual, fit_residual
from .residual_crossfit import inner_oof, outer_reference_bytes, residual_inputs, signals


def select_residual(summary, config, folds):
    baseline = summary.loc["M0"]
    candidates = summary.loc[list(MODELS)]
    complete = candidates[["converged", "folds", "scored", "loss_scored"]].eq(folds).all(axis=1)
    complete &= np.isfinite(candidates[["pooled_contrast", "log_loss_mean"]]).all(axis=1)
    if not complete.loc["M0"]:
        raise ValueError("Frozen M0 has incomplete outer results")
    maximum = float(baseline["log_loss_mean"] + config["log_loss_guardrail_bits"])
    eligible = candidates[complete & candidates["pooled_contrast"].gt(baseline["pooled_contrast"])
                          & candidates["log_loss_mean"].le(maximum)]
    leader = eligible["pooled_contrast"].idxmax() if not eligible.empty else "M0"
    cutoff = baseline["pooled_contrast"] + config["minimum_pooled_contrast_gain"]
    winner = leader if summary.loc[leader, "pooled_contrast"] >= cutoff else "M0"
    return {"model": winner, "eligible_candidates": eligible.index.tolist(), "eligible_leader": leader,
            "maximum_mean_log_loss": maximum, "minimum_pooled_contrast_gain": config["minimum_pooled_contrast_gain"]}


def pair_ordering(gold, pred):
    joined = gold[[*data.KEYS, "p"]].merge(pred, on=data.KEYS, validate="one_to_one")
    scores = {}
    for painting, group in joined.groupby("painting", sort=False):
        if len(group) < 3:
            continue
        i, j = np.triu_indices(len(group), k=1)
        target, q = group["p"].to_numpy(), group["p_positive"].to_numpy()
        gold_sign = np.sign(target[i] - target[j])
        distinct = gold_sign != 0
        if distinct.any():
            q_sign = np.sign(q[i] - q[j])[distinct]
            agreement = (q_sign == gold_sign[distinct]).mean() + .5 * (q_sign == 0).mean()
            scores[painting] = float(agreement)
    return scores


def parameter_rows(corrector, info, fold):
    width = len(corrector.support)
    offsets = dict(zip(corrector.support, corrector.parameters[2:2 + width]))
    slopes = dict(zip(corrector.support, corrector.parameters[2 + width:]))
    for language, support in info["language_paintings"].items():
        yield {"model": corrector.model, "fold": fold, "language": language, "training_paintings": support,
               "learned_deviation": language in offsets, "global_offset": corrector.parameters[0],
               "global_slope": corrector.parameters[1], "language_offset": offsets.get(language, 0.),
               "language_slope": slopes.get(language, 0.)}


def corrected_predictions(components, corrector=None):
    q = components["q0"] if corrector is None else corrector.predict(components)
    return components[data.KEYS].assign(p_positive=q)


def run_residual_cv(args, dev):
    start = time.perf_counter()
    with new_output(args.run_dir / "cv") as stage:
        train, features, folds, outer, provenance = residual_inputs(args, dev)
        config = provenance["config"]
        records = {name: [] for name in ("predictions", "paintings", "languages")}
        rows, parameters, fitted, audit = [], [], {}, []
        for fold, (fit_idx, held_idx) in enumerate(folds, 1):
            gold = train.iloc[held_idx]
            inner, inner_manifest = inner_oof(args.run_dir / "inner", train.iloc[fit_idx], gold, fold,
                                             features, provenance)
            audit.append(inner_manifest)
            components = outer.loc[outer["outer_fold"].eq(fold), [*data.KEYS, "clip_b", "siglip2", "clip_l", "q0"]]
            fitted[str(fold)] = {}
            for name in MODELS:
                fit_seconds = 0.
                corrector = None
                info = {"converged": True}
                if name != "M0":
                    tick = time.perf_counter()
                    corrector, info = fit_residual(inner, name, config)
                    fit_seconds = time.perf_counter() - tick
                    zero = replace(corrector, parameters=np.zeros_like(corrector.parameters))
                    if not np.array_equal(zero.predict(components), components["q0"]):
                        raise ValueError(f"{name} zero correction failed in outer fold {fold}")
                    parameters.extend(parameter_rows(corrector, info, fold))
                    fitted[str(fold)][name] = {"fit": info, "correction": corrector.to_dict()}
                tick = time.perf_counter()
                pred = corrected_predictions(components, corrector)
                predict_seconds = time.perf_counter() - tick
                context = {"model": name, "fold": fold}
                contrast, loss = score_result(gold, pred, context, records)
                ordering = pair_ordering(gold, pred)
                paintings = records["paintings"][-1]
                if set(ordering) != set(paintings.loc[paintings["status"].eq("scored"), "painting"]):
                    raise ValueError("Ordering diagnostic and scorer eligibility disagree")
                paintings["ordering_agreement"] = paintings["painting"].map(ordering)
                rows.append({**context, "contrast": contrast, "log_loss": loss, "converged": info["converged"],
                             "fit_seconds": fit_seconds, "predict_seconds": predict_seconds,
                             "seconds": fit_seconds + predict_seconds,
                             "ordering_agreement": paintings["ordering_agreement"].mean()})
                print(f"Outer {fold} {name}: {contrast:.6f} / {loss:.6f}, converged={info['converged']}", flush=True)

        table = pd.DataFrame(rows)
        summary, _ = write_cv_analysis(stage, train, records, table, config, "M0", "M0")
        summary["ordering_mean"] = table.groupby("model")["ordering_agreement"].mean()
        summary["pooled_ordering"] = pd.concat(records["paintings"]).groupby("model")["ordering_agreement"].mean()
        original = pd.read_csv(args.exp010_dir / "cv" / "summary.csv", index_col="model", float_precision="round_trip")
        fields = ["contrast_mean", "log_loss_mean", "pooled_contrast", "pooled_log_loss"]
        np.testing.assert_allclose(summary.loc["M0", fields].astype(float), original.loc[E50_LARGE, fields].astype(float),
                                   rtol=0., atol=1e-14)
        summary.to_csv(stage / "summary.csv", lineterminator="\n")
        by_model = read_json(stage / "pooled_metrics.json")
        base = by_model["M0"]["per_language"]
        changes = [{"model": name, "language": lang, **entry,
                    "delta_vs_m0": entry["log_loss"] - base[lang]["log_loss"]}
                   for name, score in by_model.items() for lang, entry in score["per_language"].items()]
        pd.DataFrame(changes).to_csv(stage / "pooled_language_deltas.csv", index=False, lineterminator="\n")
        outer.to_csv(stage / "outer_components.csv", index=False, lineterminator="\n")
        (stage / "reference_e50l.csv").write_bytes(outer_reference_bytes(outer, provenance["exp002_config"]["fit_seed"]))
        pd.DataFrame(parameters).to_csv(stage / "parameters.csv", index=False, lineterminator="\n")
        json_output(fitted, stage / "corrections.json")
        json_output(audit, stage / "inner_audit.json")
        json_output(run_metadata(provenance, start), stage / "run.json")
        decision = select_residual(summary, config, len(folds))
        decision["sha256"] = {name: file_sha256(stage / name) for name in
                              ("run.json", "summary.csv", "outer_components.csv", "inner_audit.json")}
        json_output(decision, stage / "selection.json")
    print(f"Frozen EXP-012 decision: {decision['model']}", flush=True)


def development_components(args, dev):
    config = inference_config()
    models, manifest, index = load_bundle(args.model_dir, config, ENCODERS)
    large, large_manifest, large_index = load_clip_large_bundle(args.clip_large_model_dir, manifest)
    by_image = {image: file_sha256(data.image_path(args.data_dir, image)) for image in dev["image"].unique()}
    predictions = []
    for name in ENCODERS:
        bank, _ = load_legacy(name, by_image.values(), index, manifest["legacy"][name], args.cache_dir,
                              config["encoders"][name]["width"])
        if set(bank) != set(by_image.values()):
            raise ValueError(f"Missing frozen {name} development features")
        predictions.append(predict_lr(models[name], dev, {image: bank[digest] for image, digest in by_image.items()},
                                      manifest["languages"]))
    bank, _ = load_clip_large_legacy(by_image.values(), large_index, large_manifest["legacy"], args.cache_dir,
                                    large_manifest["encoder"]["width"])
    if set(bank) != set(by_image.values()):
        raise ValueError("Missing frozen CLIP-L development features")
    predictions.append(predict_lr(large, dev, {image: bank[digest] for image, digest in by_image.items()},
                                  manifest["languages"]))
    frame = signals(*predictions)
    reference = args.exp010_dir / "dev" / "predictions" / E50_LARGE / submission.FILENAME
    pins = read_json(ROOT / "configs" / "e50l_inference.json")
    actual = corrected_predictions(frame).to_csv(index=False, lineterminator="\n").encode("utf-8")
    if file_sha256(reference) != pins["reference_csv_sha256"] or actual != reference.read_bytes():
        raise ValueError("Frozen E50+L heads do not reproduce the original development CSV")
    return frame, {name: file_sha256(path / MANIFEST) for name, path in
                   (("e50", args.model_dir), (CLIP_LARGE, args.clip_large_model_dir))}


def run_residual_dev(args, dev):
    start = time.perf_counter()
    cv = args.run_dir / "cv"
    decision = read_json(cv / "selection.json")
    for name in ("run.json", "summary.csv", "outer_components.csv", "inner_audit.json"):
        if decision["sha256"][name] != file_sha256(cv / name):
            raise ValueError(f"Frozen EXP-012 artifact changed: {name}")
    metadata = read_json(cv / "run.json")
    config = metadata["config"]
    summary = pd.read_csv(cv / "summary.csv", index_col="model", float_precision="round_trip")
    expected = select_residual(summary, config, metadata["exp002_config"]["cv_folds"])
    if any(decision[key] != value for key, value in expected.items()):
        raise ValueError("EXP-012 decision disagrees with the frozen selection rule")
    winner = decision["model"]
    if winner == "M0":
        print("EXP-012 NO-GO. Retained E50+L without development scoring or packaging.", flush=True)
        return

    with new_output(args.run_dir / "dev") as stage:
        _, _, _, outer, provenance = residual_inputs(args, dev)
        if any(metadata[key] != value for key, value in provenance.items()):
            raise ValueError("Development inputs differ from the frozen EXP-012 CV run")
        corrector, info = fit_residual(outer, winner, config)
        if not info["converged"]:
            raise ValueError("Full-training OOF residual fit did not converge")
        artifact = stage / "corrector.json"
        json_output(corrector.to_dict(), artifact)
        components, bundles = development_components(args, dev)
        baseline, pred = corrected_predictions(components), corrected_predictions(components, corrector)
        reloaded = Residual.from_dict(read_json(artifact))
        if not np.array_equal(reloaded.predict(components), pred["p_positive"]):
            raise ValueError("Reloaded corrector did not reproduce development predictions")
        gold = data.validation_gold(data.read_annotations(args.data_dir), dev)
        write_development(stage, gold, {"M0": baseline, winner: pred}, winner, config, "M0")
        archive = stage / "submission.zip"
        submission.pack(stage / "predictions" / winner / submission.FILENAME, archive, dev)
        submission.validate_archive(archive, dev)
        json_output(run_metadata({**provenance, "selection": decision, "fit": info, "head_manifests_sha256": bundles,
                                 "corrector_sha256": file_sha256(artifact), "submission_sha256": file_sha256(archive)},
                                 start), stage / "run.json")
    print(f"EXP-012 scored {winner} and validated {len(dev)} development predictions", flush=True)
