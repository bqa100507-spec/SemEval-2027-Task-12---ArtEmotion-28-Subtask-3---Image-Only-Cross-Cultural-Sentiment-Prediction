"""EXP-008 fixed E50 comparison with a train-only promotion gate."""

import json
import time

import pandas as pd

from . import data, submission
from .ablation import paired_bootstrap
from .baseline import file_sha256
from .cli import ROOT, git_head, json_output
from .contrast_study import new_output, summarize, write_development
from .coverage import clip_assets, extract_coverage, image_geometry, load_processor
from .fusion import blend, verify_reference
from .low_rank_study import FOLD_FILE, run_metadata, study_inputs, verify_folds
from .metrics import painting_scores, score
from .siglip2 import load_siglip2_features
from .visual_study import REFERENCE, SIGLIP, fit_head

E50 = "E50"
FULL = "E50-Full"
DUAL = "E50-Dual"
ALTERNATIVES = (FULL, DUAL)


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def run_coverage_audit(args, dev):
    baseline = read_json(ROOT / "configs" / "st3.json")
    config = read_json(ROOT / "configs" / "exp008.json")
    if git_head(args.data_dir) != baseline["upstream_revision"]:
        raise ValueError("EXP-008 requires the pinned organizer checkout")
    keys = ["painting", "image_name", "split"]
    paintings = pd.read_csv(args.data_dir / data.ANNOTATIONS, usecols=keys, dtype=str).drop_duplicates()
    data.check_split_integrity(paintings)
    data.check_dev_isolation(paintings, dev)
    assets = clip_assets(args.cache_dir, baseline, config)
    processor = load_processor(assets["preprocessor_config.json"].parent, config)
    geometry = image_geometry(args.data_dir, paintings, config)
    groups = []
    for field in ("split", "orientation", "style_folder", "coverage_band"):
        summary = geometry.groupby(field).agg(
            paintings=("painting", "size"), mean_retained=("retained_fraction", "mean"),
            median_retained=("retained_fraction", "median"), substantially_cropped=("substantially_cropped", "sum"))
        groups.append(summary.rename_axis("group").reset_index().assign(group_type=field))
    quantiles = (0, .05, .25, .5, .75, .95, 1)
    with new_output(args.run_dir / "audit") as stage:
        geometry.to_csv(stage / "geometry.csv", index=False, lineterminator="\n")
        pd.concat(groups).to_csv(stage / "groups.csv", index=False, lineterminator="\n")
        json_output({"paintings": len(geometry), "mean_retained": float(geometry["retained_fraction"].mean()),
                     "substantially_cropped": int(geometry["substantially_cropped"].sum()),
                     "quantiles": geometry[["aspect_ratio", "retained_fraction"]].quantile(quantiles).to_dict(),
                     "processor": processor.to_dict(), "config": config, "clip_revision": baseline["clip_revision"],
                     "upstream_revision": baseline["upstream_revision"], "project_revision": git_head(ROOT)},
                    stage / "audit.json")
    print(f"EXP-008 label-free audit: {len(geometry)} paintings", flush=True)


def coverage_inputs(args, dev):
    config, annotations, train, clip, provenance = study_inputs(args, dev, "exp008")
    cv = args.exp007_dir / "cv"
    prior = read_json(cv / "run.json")
    decision = read_json(cv / "selection.json")
    for key in ("annotations_sha256", "dev_input_sha256", "embedding", "exp002_folds_sha256", "exp002_config"):
        if provenance[key] != prior[key]:
            raise ValueError(f"EXP-008 input differs from EXP-007: {key}")
    if (decision["model"] != E50 or decision["run_sha256"] != file_sha256(cv / "run.json")
            or decision["summary_sha256"] != file_sha256(cv / "summary.csv")
            or prior["config"]["alphas"][E50] != config["clip_blend_weight"]):
        raise ValueError("EXP-007 reference decision or blend does not match")
    provenance["exp007_sha256"] = {name: file_sha256(cv / name) for name in
                                   ("run.json", "selection.json", "predictions.csv", "summary.csv")}
    return config, annotations, train, clip, provenance, prior


def saved_prediction(frame, model, fold):
    return frame.loc[frame["model"].eq(model) & frame["fold"].eq(fold), [*data.KEYS, "p_positive"]]


def fold_reference(saved, exp006, exp007, held, fold, alpha):
    original = saved.loc[saved["fold"].eq(fold), [*data.KEYS, "p_C"]].rename(columns={"p_C": "p_positive"})
    clip = saved_prediction(exp007, REFERENCE, fold)
    siglip = saved_prediction(exp007, SIGLIP, fold)
    verify_reference(clip, original)
    verify_reference(siglip, saved_prediction(exp006, SIGLIP, fold))
    baseline = blend(clip, siglip, alpha)
    verify_reference(baseline, saved_prediction(exp007, E50, fold))
    data.require_same_pairs(held, baseline)
    return baseline, siglip


def choose_coverage(summary, intervals, config, folds):
    baseline = summary.loc[E50]
    guardrail = config["log_loss_guardrail_bits"]
    gates = {}
    for name in ALTERNATIVES:
        row = summary.loc[name]
        gates[name] = {
            "complete": bool(row[["folds", "scored", "loss_scored", "converged"]].eq(folds).all()),
            "contrast_gain": bool(row["pooled_contrast"] - baseline["pooled_contrast"]
                                  >= config["minimum_pooled_contrast_gain"]),
            "fold_wins": bool(row["fold_wins"] >= config["minimum_fold_wins"]),
            "paired_interval": bool(intervals[name]["ci95"][0] > config["minimum_bootstrap_lower_bound"]),
            "mean_fold_loss": bool(row["log_loss_mean"] <= baseline["log_loss_mean"] + guardrail),
            "pooled_loss": bool(row["pooled_log_loss"] <= baseline["pooled_log_loss"] + guardrail),
        }
    eligible = [name for name in ALTERNATIVES if all(gates[name].values())]
    winner = max(eligible, key=lambda name: summary.loc[name, "pooled_contrast"]) if eligible else E50
    return {"model": winner, "eligible": eligible, "gates": gates}


def write_analysis(stage, train, geometry, predictions, rows, config):
    summary = summarize(rows, E50)
    pooled = {name: score(train, pred[[*data.KEYS, "p_positive"]])
              for name, pred in predictions.groupby("model", sort=False)}
    summary["pooled_contrast"] = pd.Series({name: result["cultural_contrast"] for name, result in pooled.items()})
    summary["pooled_log_loss"] = pd.Series({name: result["log_loss"] for name, result in pooled.items()})
    summary["fit_seconds"] = rows.groupby("model")["fit_seconds"].sum()
    summary["predict_seconds"] = rows.groupby("model")["predict_seconds"].sum()

    paintings, languages = [], []
    for name, pred in predictions.groupby("model", sort=False):
        joined = train[[*data.KEYS, "p"]].merge(pred, on=data.KEYS, validate="one_to_one")
        paintings.append(painting_scores(joined).assign(model=name))
        by_language = pd.DataFrame.from_dict(pooled[name]["per_language"], orient="index")
        languages.append(by_language.rename_axis("language").reset_index().assign(model=name))
    paint = pd.concat(paintings, ignore_index=True)
    lang = pd.concat(languages, ignore_index=True)
    for frame, key, metric in ((paint, "painting", "contrast"), (lang, "language", "log_loss")):
        reference = frame.loc[frame["model"].eq(E50)].set_index(key)[metric]
        frame["delta_vs_e50"] = frame[metric] - frame[key].map(reference)
    paint = paint.merge(geometry.drop(columns=["split", "image_name"]), on="painting", validate="many_to_one")
    if len(paint) != len(summary) * train["painting"].nunique():
        raise ValueError("Missing training geometry")
    scored = paint.loc[paint["status"].eq("scored")]
    summary["painting_wins"] = scored.groupby("model")["delta_vs_e50"].apply(lambda delta: int(delta.gt(0).sum()))
    summary["painting_regressions"] = scored.groupby("model")["delta_vs_e50"].apply(lambda delta: int(delta.lt(0).sum()))
    wide = scored.pivot(index="painting", columns="model", values="delta_vs_e50").reset_index()
    draws, intervals = paired_bootstrap(wide.assign(status="scored"), config["bootstrap_replicates"],
                                       config["bootstrap_seed"], columns=ALTERNATIVES)

    subgroups = []
    for field in ("coverage_band", "orientation"):
        for group, subset in geometry.groupby(field):
            gold = train.loc[train["painting"].isin(subset["painting"])]
            if gold.empty:
                continue
            for name, pred in predictions.groupby("model", sort=False):
                result = score(gold, pred.loc[pred["painting"].isin(gold["painting"]), [*data.KEYS, "p_positive"]])
                subgroups.append({"group_type": field, "group": group, "model": name,
                                  "paintings": gold["painting"].nunique(), "pairs": len(gold),
                                  **{metric: result[metric] for metric in
                                     ("cultural_contrast", "log_loss", "contrast_paintings", "log_loss_languages")}})
    for name, frame in (("fold_metrics", rows), ("predictions", predictions), ("paintings", paint),
                        ("languages", lang), ("bootstrap_distribution", draws), ("subgroups", pd.DataFrame(subgroups))):
        frame.to_csv(stage / f"{name}.csv", index=False, lineterminator="\n")
    summary.to_csv(stage / "summary.csv", lineterminator="\n")
    json_output(intervals, stage / "paired_bootstrap.json")
    return choose_coverage(summary, intervals, config, int(rows["fold"].nunique()))


def run_coverage_cv(args, dev):
    start = time.perf_counter()
    with new_output(args.run_dir / "cv") as stage:
        config, _, train, clip, provenance, prior = coverage_inputs(args, dev)
        saved = pd.read_csv(args.exp002_dir / FOLD_FILE, float_precision="round_trip")
        folds = verify_folds(train, saved, provenance["exp002_config"])
        path = args.exp006_dir / "cv" / "predictions.csv"
        if file_sha256(path) != prior["exp006_predictions_sha256"]:
            raise ValueError("EXP-006 reference prediction hash differs from EXP-007")
        exp006 = pd.read_csv(path, float_precision="round_trip")
        exp007 = pd.read_csv(args.exp007_dir / "cv" / "predictions.csv", float_precision="round_trip")
        baseline = read_json(ROOT / "configs" / "st3.json")
        full, dual, manifest = extract_coverage(args.data_dir, train["image"], clip, args.cache_dir, baseline,
                                               config, provenance["embedding"]["sha256"])
        inputs = train[["painting", "image"]].drop_duplicates().rename(columns={"image": "image_name"})
        geometry = image_geometry(args.data_dir, inputs.assign(split="train"), config)
        seed = provenance["exp002_config"]["fit_seed"]
        rows, predictions = [], []
        for fold, (fit_ids, held_ids) in enumerate(folds, 1):
            print(f"EXP-008 train-only fold {fold}/{len(folds)}", flush=True)
            fit, held = train.iloc[fit_ids], train.iloc[held_ids]
            reference, siglip = fold_reference(saved, exp006, exp007, held, fold, config["clip_blend_weight"])
            for name, vectors in ((E50, None), (FULL, full), (DUAL, dual)):
                if name == E50:
                    pred = reference
                    info = {"converged": True, "fit_seconds": 0., "predict_seconds": 0., "seconds": 0.}
                else:
                    clip_pred, info = fit_head(fit, held, vectors, seed)
                    pred = blend(clip_pred, siglip, config["clip_blend_weight"])
                result = score(held, pred)
                context = {"model": name, "fold": fold, "seed": seed}
                predictions.append(pred.assign(**context))
                rows.append({**context, **info, "contrast": result["cultural_contrast"], "log_loss": result["log_loss"]})
                print(f"  {name}: {result['cultural_contrast']:.6f} / {result['log_loss']:.6f}", flush=True)
        decision = write_analysis(stage, train, geometry, pd.concat(predictions), pd.DataFrame(rows), config)
        metadata = run_metadata({**provenance, "coverage_embedding": manifest,
                                 "config_sha256": file_sha256(ROOT / "configs" / "exp008.json")}, start)
        json_output(metadata, stage / "run.json")
        decision["sha256"] = {name: file_sha256(stage / name) for name in
                              ("summary.csv", "paired_bootstrap.json", "run.json")}
        json_output(decision, stage / "selection.json")
    print(f"Frozen EXP-008 decision: {decision['model']}", flush=True)


def run_coverage_dev(args, dev):
    start = time.perf_counter()
    cv = args.run_dir / "cv"
    decision = read_json(cv / "selection.json")
    for name in ("summary.csv", "paired_bootstrap.json", "run.json"):
        if decision["sha256"][name] != file_sha256(cv / name):
            raise ValueError(f"Frozen EXP-008 artifact changed: {name}")
    metadata = read_json(cv / "run.json")
    summary = pd.read_csv(cv / "summary.csv", index_col="model", float_precision="round_trip")
    config = metadata["config"]
    expected = choose_coverage(summary, read_json(cv / "paired_bootstrap.json"), config,
                               metadata["exp002_config"]["cv_folds"])
    if any(decision[key] != value for key, value in expected.items()):
        raise ValueError("EXP-008 decision disagrees with the frozen gate")
    winner = decision["model"]
    if winner == E50:
        print("EXP-008 retained E50. Development scoring and packaging were skipped.", flush=True)
        return

    with new_output(args.run_dir / "dev") as stage:
        _, annotations, train, clip, provenance, prior = coverage_inputs(args, dev)
        if (any(metadata[key] != value for key, value in provenance.items())
                or metadata["config_sha256"] != file_sha256(ROOT / "configs" / "exp008.json")):
            raise ValueError("Development inputs differ from frozen EXP-008 CV")
        baseline = read_json(ROOT / "configs" / "st3.json")
        caches = {}
        vectors = {}
        for split, names in (("train", train["image"]), ("dev", dev["image"])):
            full, dual, caches[split] = extract_coverage(args.data_dir, names, clip, args.cache_dir, baseline,
                                                        config, provenance["embedding"]["sha256"])
            vectors.update(full if winner == FULL else dual)
        if caches["train"] != metadata["coverage_embedding"]:
            raise ValueError("Training feature cache changed since CV")
        siglip_config = read_json(ROOT / "configs" / "exp006.json")
        siglip, manifest = load_siglip2_features(args, dev, train, siglip_config,
                                                 extraction_dir=args.exp006_dir / "extraction")
        if manifest["features_sha256"] != prior["siglip2_embedding"]["features_sha256"]:
            raise ValueError("SigLIP 2 feature cache differs from EXP-007")
        seed = metadata["exp002_config"]["fit_seed"]
        # EXP-007 saved E50 and CLIP, but its separate full-train SigLIP 2 probabilities were transient.
        # Replay that exact frozen fit once and verify the original blend before changing the CLIP input.
        siglip_pred, siglip_info = fit_head(train, dev, siglip, seed)
        pred, clip_info = fit_head(train, dev, vectors, seed)
        if not siglip_info["converged"] or not clip_info["converged"]:
            raise ValueError("Full-training head did not converge")
        prior_dir = args.exp007_dir / "dev" / "predictions"
        prior_clip = submission.read_predictions(prior_dir / REFERENCE / submission.FILENAME, dev)
        reference = submission.read_predictions(prior_dir / E50 / submission.FILENAME, dev)
        verify_reference(blend(prior_clip, siglip_pred, config["clip_blend_weight"]), reference)
        pred = blend(pred, siglip_pred, config["clip_blend_weight"])
        write_development(stage, data.validation_gold(annotations, dev), {E50: reference, winner: pred},
                          winner, config, E50)
        archive = stage / "submission.zip"
        submission.pack(stage / "predictions" / winner / submission.FILENAME, archive, dev)
        submission.validate_archive(archive, dev)
        json_output({**run_metadata(provenance, start), "selection": decision, "siglip_fit": siglip_info,
                     "clip_fit": clip_info, "coverage_embeddings": caches,
                     "submission_sha256": file_sha256(archive)}, stage / "run.json")
