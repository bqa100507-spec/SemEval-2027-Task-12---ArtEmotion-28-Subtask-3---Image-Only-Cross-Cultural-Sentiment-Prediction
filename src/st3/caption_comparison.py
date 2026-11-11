"""New grouped CV, locked confirmation and guarded development for EXP-017."""

import json
import time

import numpy as np
import pandas as pd
import torch

from . import data, submission
from .ablation import paired_bootstrap
from .baseline import file_sha256
from .caption_data import inner_partition, require_isolation, training_captions
from .caption_student import concatenate_features, fit_student, predict_student
from .caption_study import audited_inputs, checked_decision
from .caption_targets import fit_targets
from .caption_text import load_teacher
from .cli import ROOT, json_output
from .contrast_study import new_output, write_development
from .e50_heads import inference_config, load_bundle, load_legacy
from .e50l_heads import load_clip_large_bundle, load_clip_large_legacy
from .fusion import blend, verify_reference
from .fusion_study import write_cv_analysis
from .low_rank_study import run_metadata, score_result
from .metrics import score
from .visual_study import fit_head

CONDITIONS = {"M1": None, "M2": "clip", "M3": "e5"}


def image_banks(args, pairs):
    config = inference_config()
    _, manifest, index = load_bundle(ROOT / "runs/e50_inference_models", config, ("clip", "siglip2"))
    _, large, large_index = load_clip_large_bundle(ROOT / "runs/e50l_inference_models", manifest)
    by_image = {name: file_sha256(data.image_path(args.data_dir, name)) for name in pairs.image.unique()}
    if "image_sha256" in pairs:
        expected = pairs[["image", "image_sha256"]].dropna().drop_duplicates()
        if not expected.image.map(by_image).eq(expected.image_sha256).all():
            raise ValueError("Artwork bytes differ from the locked EXP-017 audit")
    banks = []
    for name in ("clip", "siglip2"):
        vectors, _ = load_legacy(name, by_image.values(), index, manifest["legacy"][name], args.cache_dir,
                                 config["encoders"][name]["width"])
        if set(vectors) != set(by_image.values()):
            raise ValueError(f"Missing original {name} image features")
        banks.append({image: vectors[digest] for image, digest in by_image.items()})
    vectors, _ = load_clip_large_legacy(by_image.values(), large_index, large["legacy"], args.cache_dir,
                                        large["encoder"]["width"])
    if set(vectors) != set(by_image.values()):
        raise ValueError("Missing original CLIP-L image features")
    banks.append({image: vectors[digest] for image, digest in by_image.items()})
    return banks


def refit_baseline(fit, held, banks):
    require_isolation(fit, held)
    predictions, fits = [], []
    for name, features in zip(("clip", "siglip2", "clip_large"), banks):
        pred, info = fit_head(fit, held[[*data.KEYS, "image"]], features, 2027)
        if not info["converged"]:
            raise ValueError(f"EXP-017 baseline {name} failed to converge")
        predictions.append(pred)
        fits.append({"encoder": name, **info})
    return blend(blend(predictions[0], predictions[1], .5), predictions[2], .5), fits


def paired_order(gold, predictions):
    frame = gold[[*data.KEYS, "p"]].merge(predictions, on=data.KEYS, validate="one_to_one")
    correct = total = 0.
    for _, rows in frame.groupby("painting"):
        gold_values = rows.p.to_numpy()
        predicted = rows.p_positive.to_numpy()
        i, j = np.triu_indices(len(rows), 1)
        truth = np.sign(gold_values[i] - gold_values[j])
        guess = np.sign(predicted[i] - predicted[j])
        valid = truth != 0
        correct += ((truth == guess) & valid).sum() + .5 * ((guess == 0) & valid).sum()
        total += valid.sum()
    return float(correct / total) if total else None


def save_prepared(directory, name, captions, fit, inner, vectors, config):
    prepared = {}
    for phase, rows in (("inner", inner), ("full", fit)):
        result = fit_targets(captions, rows, vectors, config)
        prepared[phase] = result
        _, _, pca, records, metadata = result
        np.savez(directory / f"{name}_{phase}_pca.npz", components=pca.components_, mean=pca.mean_,
                 variance=pca.explained_variance_ratio_)
        records.to_csv(directory / f"{name}_{phase}_targets.csv", index=False, lineterminator="\n")
        json_output(metadata, directory / f"{name}_{phase}_pca.json")
    return prepared


def train_conditions(directory, fit, held, baseline, banks, config, vectors, captions, fold, device, models):
    require_isolation(fit, held)
    inner, stopping = inner_partition(fit, config, fold)
    require_isolation(inner, stopping, held)
    visual = concatenate_features(banks)
    inputs = held[[*data.KEYS, "image"]]
    predictions = {"M0": baseline}
    fits, curves, seeds = [], [], []
    splits = {name: sorted(rows.painting.unique()) for name, rows in (("fit", fit), ("inner", inner),
                                                                   ("stopping", stopping), ("held", held))}
    json_output(splits, directory / "partition.json")
    for name in models:
        teacher = CONDITIONS[name]
        prepared = save_prepared(directory, name, captions, fit, inner, vectors[teacher], config) if teacher else None
        probabilities = []
        for seed in config["seeds"]:
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats()
            model, trace, info = fit_student(fit, inner, stopping, visual, prepared, seed, config, device)
            tick = time.perf_counter()
            values = predict_student(model, inputs, visual, baseline, device, config["batch_size"])
            probabilities.append(values)
            info["predict_seconds"] = time.perf_counter() - tick
            info["peak_cuda_allocated_bytes"] = torch.cuda.max_memory_allocated() if device.type == "cuda" else None
            info["peak_cuda_reserved_bytes"] = torch.cuda.max_memory_reserved() if device.type == "cuda" else None
            checkpoint = directory / f"{name}_{seed}.pt"
            torch.save({"state_dict": {key: value.cpu() for key, value in model.state_dict().items()},
                        "languages": list(model.languages)}, checkpoint)
            info["checkpoint_sha256"] = file_sha256(checkpoint)
            fits.append({"fold": fold, "model": name, "seed": seed, **info})
            curves.extend({"fold": fold, "model": name, "seed": seed, **row} for row in trace)
            seeds.append(inputs[data.KEYS].assign(model=name, seed=seed, p_positive=np.round(values, 6), fold=fold))
            print(f"EXP-017 fold {fold} {name} seed {seed}: epoch {info['selected_epoch']}/{info['selection_epochs']}, "
                  f"fit {info['fit_seconds']:.2f}s", flush=True)
            del model
        predictions[name] = inputs[data.KEYS].assign(p_positive=np.round(np.mean(probabilities, axis=0), 6))
        predictions["B" + name[1:]] = blend(baseline, predictions[name], .5)
    json_output(fits, directory / "student_fits.json")
    pd.DataFrame(curves).to_csv(directory / "curves.csv", index=False, lineterminator="\n")
    pd.concat(seeds, ignore_index=True).to_csv(directory / "seed_predictions.csv", index=False, lineterminator="\n")
    return predictions, fits


def select_caption(summary, config, complete):
    baseline = summary.loc["M0"]
    control = summary.loc["B1"]
    candidates = summary.loc[[name for name in ("B2", "B3") if name in summary.index]]
    qualified = candidates[candidates.pooled_contrast.ge(baseline.pooled_contrast + config["minimum_pooled_contrast_gain"])
                           & candidates.pooled_contrast.ge(control.pooled_contrast + config["minimum_incremental_gain"])
                           & candidates.fold_wins.ge(config["minimum_fold_wins"])
                           & candidates.log_loss_mean.le(baseline.log_loss_mean + config["log_loss_guardrail_bits"])
                           & candidates.folds.eq(config["cv_folds"])] if complete else candidates.iloc[:0]
    winner = qualified.pooled_contrast.idxmax() if len(qualified) else "M0"
    return {"model": winner, "status": "PASS" if winner != "M0" else "NO_GO",
            "qualified": qualified.index.tolist(), "all_seeds_completed": complete,
            "resource_status": "RESEARCH_ONLY" if winner == "B3" else "PRIMARY_EXISTING_RESOURCE"}


def paired_comparisons(paintings, config):
    scored = paintings[paintings.status.eq("scored")].pivot(index="painting", columns="model", values="contrast")
    columns = {}
    for left, right in (("M2", "M1"), ("M3", "M1"), ("B2", "B1"), ("B3", "B1")):
        if left in scored:
            columns[left + "_minus_" + right] = scored[left] - scored[right]
    frame = pd.DataFrame(columns).reset_index().assign(status="scored")
    return paired_bootstrap(frame, config["bootstrap_replicates"], config["bootstrap_seed"], columns=list(columns))


def device017(requested):
    import os

    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(min(4, os.cpu_count() or 1))
    device = "cuda" if requested == "auto" and torch.cuda.is_available() else requested
    return torch.device("cpu" if device == "auto" else device)


def load_vectors(args, train, config, audit, *, full=False, teachers=("clip", "e5")):
    directory = args.run_dir / ("full_text" if full else "text")
    vectors, manifests = {}, {}
    for teacher in teachers:
        vectors[teacher], manifests[teacher] = load_teacher(directory / teacher, config["teachers"][teacher], audit,
                                                           train.painting.unique())
    return vectors, manifests


def run_caption_cv(args):
    start = time.perf_counter()
    config, train, audit = audited_inputs(args)
    partition = train[train.role.eq("development")]
    locked = train[train.role.eq("holdout")]
    require_isolation(partition, locked)
    banks = image_banks(args, partition)
    vectors, manifests = load_vectors(args, partition, config, audit)
    captions = training_captions(args.data_dir)
    captions = captions[captions.painting.isin(partition.painting)]
    device = device017(args.device)
    provenance = {"config": config, "config_sha256": audit["config_sha256"], "protocol_sha256": audit["protocol_sha256"],
                  "annotations_sha256": audit["annotations_sha256"],
                  "teacher_features_sha256": {name: meta["features_sha256"] for name, meta in manifests.items()}}
    records = {name: [] for name in ("predictions", "paintings", "languages")}
    rows, all_fits = [], []
    for fold in range(1, config["cv_folds"] + 1):
        fit, held = partition[partition.fold.ne(fold)], partition[partition.fold.eq(fold)]
        require_isolation(fit, held, locked)
        output = args.run_dir / "cv_work" / f"fold_{fold}"
        if not output.exists():
            with new_output(output) as stage:
                baseline, lr_fits = refit_baseline(fit, held, banks)
                predictions, fits = train_conditions(stage, fit, held, baseline, banks, config, vectors, captions,
                                                     fold, device, CONDITIONS)
                pd.concat([frame.assign(model=name) for name, frame in predictions.items()]).to_csv(
                    stage / "predictions.csv", index=False, lineterminator="\n")
                json_output({**provenance, "lr_fits": lr_fits, "prediction_sha256": file_sha256(stage / "predictions.csv")},
                            stage / "run.json")
        saved = json.loads((output / "run.json").read_text())
        if (any(saved[key] != value for key, value in provenance.items())
                or saved["prediction_sha256"] != file_sha256(output / "predictions.csv")):
            raise ValueError("Completed EXP-017 fold differs from the frozen inputs")
        predictions = pd.read_csv(output / "predictions.csv", float_precision="round_trip")
        fits = json.loads((output / "student_fits.json").read_text())
        for item in fits:
            if file_sha256(output / f"{item['model']}_{item['seed']}.pt") != item["checkpoint_sha256"]:
                raise ValueError("An EXP-017 fold checkpoint changed")
        all_fits.extend(fits)
        for name, pred in predictions.groupby("model", sort=False):
            context = {"model": name, "fold": fold}
            contrast, loss = score_result(held, pred[[*data.KEYS, "p_positive"]], context, records)
            matching = [row for row in fits if row["model"] == name]
            rows.append({**context, "contrast": contrast, "log_loss": loss, "converged": True,
                         "fit_seconds": sum(row["fit_seconds"] for row in matching),
                         "predict_seconds": sum(row["predict_seconds"] for row in matching),
                         "seconds": sum(row["fit_seconds"] + row["predict_seconds"] for row in matching)})
            print(f"EXP-017 fold {fold} {name}: {contrast:.6f} / {loss:.6f}", flush=True)
    expected = {(fold, model, seed) for fold in range(1, config["cv_folds"] + 1)
                for model in CONDITIONS for seed in config["seeds"]}
    complete = {(row["fold"], row["model"], row["seed"]) for row in all_fits} == expected and len(all_fits) == len(expected)
    with new_output(args.run_dir / "cv") as stage:
        summary, predictions = write_cv_analysis(stage, partition, records, pd.DataFrame(rows), config, "M0", "M0")
        bootstrap, comparisons = paired_comparisons(pd.concat(records["paintings"]), config)
        bootstrap.to_csv(stage / "matched_bootstrap_distribution.csv", index=False, lineterminator="\n")
        json_output(comparisons, stage / "matched_bootstrap.json")
        json_output({name: paired_order(partition, pred[[*data.KEYS, "p_positive"]])
                     for name, pred in predictions.groupby("model")}, stage / "pair_order.json")
        json_output(all_fits, stage / "fits.json")
        json_output({**run_metadata(provenance, start), "device": str(device)}, stage / "run.json")
        decision = {**select_caption(summary, config, complete), "config_sha256": audit["config_sha256"],
                    "sha256": {name: file_sha256(stage / name) for name in ("run.json", "fits.json", "summary.csv")}}
        json_output(decision, stage / "selection.json")
    print(f"EXP-017 frozen CV decision: {decision['status']} {decision['model']}", flush=True)


def run_caption_holdout(args):
    decision = checked_decision(args, "cv")
    if decision["status"] != "PASS":
        print("EXP-017 NO-GO. Locked holdout stays sealed.", flush=True)
        return
    config, train, audit = audited_inputs(args)
    fit, held = train[train.role.eq("development")], train[train.role.eq("holdout")]
    winner = decision["model"]
    teacher = "clip" if winner == "B2" else "e5"
    vectors, manifests = load_vectors(args, fit, config, audit, teachers=(teacher,))
    captions = training_captions(args.data_dir)
    captions = captions[captions.painting.isin(fit.painting)]
    banks = image_banks(args, train)
    with new_output(args.run_dir / "holdout") as stage:
        baseline, lr_fits = refit_baseline(fit, held, banks)
        predictions, fits = train_conditions(stage, fit, held, baseline, banks, config, vectors, captions,
                                             6, device017(args.device), ("M1", "M" + winner[1:]))
        write_development(stage, held, {name: predictions[name] for name in ("M0", winner)},
                          winner, config, "M0")
        control = stage / "matched_control"
        control.mkdir()
        write_development(control, held, {name: predictions[name] for name in ("B1", winner)},
                          winner, config, "B1")
        metrics = {name: score(held, predictions[name]) for name in ("M0", "B1", winner)}
        gain = metrics[winner]["cultural_contrast"] - metrics["M0"]["cultural_contrast"]
        incremental = metrics[winner]["cultural_contrast"] - metrics["B1"]["cultural_contrast"]
        loss_delta = metrics[winner]["log_loss"] - metrics["M0"]["log_loss"]
        passed = gain >= config["holdout_minimum_gain"] and incremental > 0 and loss_delta <= config["holdout_loss_guardrail_bits"]
        json_output({"lr_fits": lr_fits, "student_fits": fits, "teacher_manifests": manifests}, stage / "fits.json")
        json_output(metrics, stage / "metrics.json")
        result = {"model": winner, "status": "PASS" if passed else "NO_GO", "contrast_gain": gain,
                  "gain_over_B1": incremental, "log_loss_delta": loss_delta, "config_sha256": audit["config_sha256"],
                  "sha256": {name: file_sha256(stage / name) for name in ("metrics.json", "fits.json")}}
        json_output(result, stage / "selection.json")
    print(f"EXP-017 locked confirmation: {result['status']}", flush=True)


def run_caption_dev(args):
    cv = checked_decision(args, "cv")
    if cv["status"] != "PASS":
        print("EXP-017 NO-GO. Public development stays unscored.", flush=True)
        return
    decision = checked_decision(args, "holdout")
    if decision["status"] != "PASS":
        print("EXP-017 confirmation NO-GO. Public development stays unscored.", flush=True)
        return
    config, train, audit = audited_inputs(args)
    winner = decision["model"]
    teacher = "clip" if winner == "B2" else "e5"
    vectors, _ = load_vectors(args, train, config, audit, full=True, teachers=(teacher,))
    dev = data.read_dev_input(args.data_dir)
    dev["image_sha256"] = dev.image.map(lambda name: file_sha256(data.image_path(args.data_dir, name)))
    banks = image_banks(args, pd.concat([train, dev], ignore_index=True))
    with new_output(args.run_dir / "dev") as stage:
        baseline, lr_fits = refit_baseline(train, dev, banks)
        expected = pd.read_csv(ROOT / "runs/exp010_clip_large/dev/predictions/E50+L/predictions.csv")
        verify_reference(baseline, expected)
        predictions, fits = train_conditions(stage, train, dev, baseline, banks, config, vectors,
                                             training_captions(args.data_dir), 7, device017(args.device), ("M" + winner[1:],))
        gold = data.validation_gold(data.read_annotations(args.data_dir), dev)
        write_development(stage, gold, {name: predictions[name] for name in ("M0", winner)}, winner, config, "M0")
        submission.pack(stage / "predictions" / winner / submission.FILENAME, stage / "submission.zip", dev)
        submission.validate_archive(stage / "submission.zip", dev)
        json_output({"model": winner, "status": "RESEARCH_ONLY" if teacher == "e5" else "PRIMARY_EXISTING_RESOURCE",
                     "lr_fits": lr_fits, "student_fits": fits, "submission_sha256": file_sha256(stage / "submission.zip")},
                    stage / "run.json")
    print(f"EXP-017 scored the fixed {winner} development candidate", flush=True)
