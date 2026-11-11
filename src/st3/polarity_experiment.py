"""Measured pilot, resumable matched fits and frozen EXP-018 CV selection."""

import json
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch

from . import data
from .ablation import paired_bootstrap
from .baseline import file_sha256
from .caption_comparison import paired_order
from .caption_data import inner_partition, require_isolation
from .caption_student import concatenate_features, fit_student, predict_student, validation_bce
from .cli import json_output
from .contrast_study import new_output
from .fusion import blend, verify_reference
from .fusion_study import write_cv_analysis
from .low_rank_study import score_result
from .polarity_student import (PolarityStudent, gradient_diagnostic, polarity_epoch, polarity_rows,
                               seed_polarity_model)
from .polarity_targets import condition_targets
from .polarity_workflow import (ExecutionBudget, config018, context, fingerprint, fit_partition_geometry, geometry,
                                load_geometry, load_raw, read_sealed, seal, stage_budget)

CONDITIONS = ("M1", "M2", "M3")


def run_pilot(args):
    started = time.perf_counter()
    pilot_budget = ExecutionBudget(args.max_seconds if args.max_seconds > 0 else 180.)
    ctx = context(args)
    output = args.run_dir / "pilot"
    if output.exists():
        read_sealed(output, ctx.identity)
        print("Reused verified correctness and runtime pilot", flush=True)
        return
    load_seconds = time.perf_counter() - started
    raw = load_raw(args, ctx.identity)
    fit = ctx.train[ctx.train.fold.ne(1)]
    held = ctx.train[ctx.train.fold.eq(1)]
    inner, stopping = inner_partition(fit, ctx.config, 1)
    require_isolation(inner, stopping, held)
    tick = time.perf_counter()
    inner_bundle = fit_partition_geometry(raw, inner, ctx.config)
    pilot_budget.check("inner PCA")
    full_bundle = fit_partition_geometry(raw, fit, ctx.config)
    pca_seconds = time.perf_counter() - tick
    visual = concatenate_features(ctx.banks)
    device = torch.device(ctx.identity["device"])
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    languages = sorted(inner.language.unique())
    timings, gradients = {}, {}
    for condition in CONDITIONS:
        targets, weights = condition_targets(inner_bundle, condition) if condition != "M1" else (None, None)
        rows = polarity_rows(inner, visual, languages, device, targets, weights)
        validation = polarity_rows(stopping, visual, languages, device)
        model, rng, optimizer = seed_polarity_model(languages, 2027, ctx.config, device)
        if condition != "M1":
            batch = torch.randperm(len(inner), device=device,
                                   generator=torch.Generator(device=device).manual_seed(2027))[:ctx.config["batch_size"]]
            gradients[condition] = gradient_diagnostic(model, rows, batch, ctx.config["text_weight"])
            del model, optimizer
            model, rng, optimizer = seed_polarity_model(languages, 2027, ctx.config, device)
        epoch_times, stop_times, losses = [], [], []
        for epoch in range(ctx.config["pilot_warmup_epochs"] + ctx.config["pilot_timed_epochs"]):
            pilot_budget.check(f"Pilot {condition}", epoch + 1)
            tick = time.perf_counter()
            bce, auxiliary = polarity_epoch(model, rows, rng, optimizer, ctx.config,
                                            0. if condition == "M1" else ctx.config["text_weight"])
            training_seconds = time.perf_counter() - tick
            tick = time.perf_counter()
            stopping_bce = validation_bce(model, validation, ctx.config["batch_size"])
            validation_seconds = time.perf_counter() - tick
            if not np.isfinite([bce, auxiliary, stopping_bce]).all():
                raise ValueError("Nonfinite representative pilot losses")
            if epoch >= ctx.config["pilot_warmup_epochs"]:
                epoch_times.append(training_seconds)
                stop_times.append(validation_seconds)
            losses.append({"epoch": epoch + 1, "train_bce": bce, "auxiliary": auxiliary, "stopping_bce": stopping_bce})
        timings[condition] = {"epoch_seconds": epoch_times, "validation_seconds": stop_times,
                              "mean_epoch_seconds": float(np.mean(epoch_times)),
                              "mean_validation_seconds": float(np.mean(stop_times)), "losses": losses,
                              "parameters": sum(p.numel() for p in model.parameters())}
        print(f"Pilot {condition}: {np.mean(epoch_times):.3f}s per training epoch", flush=True)
        del model, optimizer, rows, validation
    previous = pd.read_json(args.exp017_dir / "cv/fits.json")
    selected = float(previous.selected_epoch.quantile(ctx.config["planning_epoch_quantile"]))
    stopping_epochs = selected + ctx.config["patience"]
    count = ctx.config["cv_folds"] * len(ctx.config["seeds"])
    training_estimate = count * sum((selected + stopping_epochs) * row["mean_epoch_seconds"]
                                    + stopping_epochs * row["mean_validation_seconds"] for row in timings.values())
    pca_estimate = ctx.config["cv_folds"] * pca_seconds
    scoring_estimate = 35.
    cv_estimate = (training_estimate + pca_estimate + load_seconds + scoring_estimate) * ctx.config["planning_overhead_factor"]
    verification_estimate = max(60., load_seconds + pca_seconds + 30.)
    preflight = read_sealed(args.run_dir / "preflight", ctx.identity)
    audit = read_sealed(args.run_dir / "audit", ctx.identity)
    actual = time.perf_counter() - started
    total = preflight["seconds"] + audit["seconds"] + actual + cv_estimate + verification_estimate
    maximum = count * sum(300 * row["mean_epoch_seconds"] + 150 * row["mean_validation_seconds"]
                          for row in timings.values()) + pca_estimate + load_seconds + scoring_estimate
    metadata = {"status": "READY_FOR_LOCAL_RUN", "fit_paintings": fit.painting.nunique(),
                "inner_paintings": inner.painting.nunique(), "inner_pairs": len(inner), "seconds": actual,
                "load_seconds": load_seconds, "two_pca_fits_seconds": pca_seconds,
                "planning_selected_epochs": selected, "planning_stopping_epochs": stopping_epochs,
                "estimated_training_seconds": training_estimate, "estimated_pca_seconds": pca_estimate,
                "estimated_scoring_seconds": scoring_estimate, "estimated_cv_seconds": cv_estimate,
                "estimated_verification_seconds": verification_estimate, "estimated_total_seconds": total,
                "configured_maximum_cv_seconds": maximum, "cv_executed": False,
                "holdout_executed": False, "development_executed": False,
                "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated() if device.type == "cuda" else None,
                "peak_cuda_reserved_bytes": torch.cuda.max_memory_reserved() if device.type == "cuda" else None,
                "automatic_execution_budget_seconds": ctx.config["codex_execution_budget_seconds"]}
    with new_output(output) as stage:
        json_output(timings, stage / "timings.json")
        json_output(gradients, stage / "gradients.json")
        json_output(inner_bundle["metadata"], stage / "inner_pca.json")
        json_output(full_bundle["metadata"], stage / "full_pca.json")
        seal(stage, ctx.identity, metadata)
    print(f"Pilot complete in {actual:.2f}s; estimated complete workflow {total / 60:.2f} minutes. "
          "Full CV remains a separate owner command.", flush=True)


def baseline_for(ctx, held, fold):
    reference = ctx.reference[ctx.reference.fold.eq(fold)][[*data.KEYS, "p_positive"]]
    data.require_same_pairs(held, reference)
    return held[data.KEYS].merge(reference, on=data.KEYS, validate="one_to_one")


def fit_conditions(args, ctx, raw, fit, held, baseline, directory, fold, budget, models=CONDITIONS):
    require_isolation(fit, held)
    inner, stopping = inner_partition(fit, ctx.config, fold)
    require_isolation(inner, stopping, held)
    for temp in directory.glob("**/tmp*") if directory.exists() else ():
        if temp.is_dir():
            print(f"Detected unpublished staging directory {temp}; only sealed seeds are reusable", flush=True)
    bundles = geometry(args, ctx, raw, fit, inner, directory / "geometry")
    visual = concatenate_features(ctx.banks)
    device = torch.device(ctx.identity["device"])
    inputs = held[[*data.KEYS, "image"]]
    predictions, all_fits = {"M0": baseline}, []
    for condition in models:
        prepared = {phase: condition_targets(bundle, condition) for phase, bundle in bundles.items()} if condition != "M1" else None
        probabilities = []
        for seed in ctx.config["seeds"]:
            budget.check(f"fold {fold} {condition} seed {seed}")
            output = directory / "seeds" / f"{condition}_{seed}"
            if not output.exists():
                with new_output(output) as stage:
                    model, curves, info = fit_student(fit, inner, stopping, visual, prepared, seed, ctx.config, device,
                        model_factory=seed_polarity_model, epoch_function=polarity_epoch, training_rows=polarity_rows,
                        progress=lambda phase, epoch: budget.check(f"fold {fold} {condition} {seed} {phase}", epoch))
                    values = predict_student(model, inputs, visual, baseline, device, ctx.config["batch_size"])
                    torch.save({"state_dict": {key: value.cpu() for key, value in model.state_dict().items()},
                                "languages": list(model.languages)}, stage / "student.pt")
                    np.savez(stage / "predictions.npz", painting=inputs.painting.to_numpy(dtype=str),
                             language=inputs.language.to_numpy(dtype=str), probabilities=values)
                    pd.DataFrame(curves).to_csv(stage / "curves.csv", index=False, lineterminator="\n")
                    seal(stage, ctx.identity, {"fold": fold, "model": condition, "seed": seed,
                         "fit_paintings": sorted(fit.painting.unique()), "inner_paintings": sorted(inner.painting.unique()),
                         "stopping_paintings": sorted(stopping.painting.unique()),
                         "held_paintings": sorted(held.painting.unique()), "finite": True, **info})
                    del model
            info = read_sealed(output, ctx.identity)
            if (info["fit_paintings"] != sorted(fit.painting.unique()) or info["held_paintings"] != sorted(held.painting.unique())
                    or info["inner_paintings"] != sorted(inner.painting.unique())
                    or info["stopping_paintings"] != sorted(stopping.painting.unique()) or not info["finite"]):
                raise ValueError("A resumed seed has different training/held painting membership")
            with np.load(output / "predictions.npz", allow_pickle=False) as archive:
                if archive["painting"].tolist() != inputs.painting.tolist() or archive["language"].tolist() != inputs.language.tolist():
                    raise ValueError("Resumed predictions have different row identities")
                values = archive["probabilities"]
            if values.shape != (len(inputs),) or not np.isfinite(values).all() or ((values < 0) | (values > 1)).any():
                raise ValueError("Invalid resumed probability payload")
            probabilities.append(values)
            all_fits.append(info)
            print(f"Completed/reused fold {fold} {condition} seed {seed}, epoch {info['selected_epoch']}", flush=True)
        predictions[condition] = inputs[data.KEYS].assign(p_positive=np.round(np.mean(probabilities, axis=0), 6))
        predictions["B" + condition[1:]] = blend(baseline, predictions[condition], .5)
    return predictions, all_fits


def folded_predictions(args, ctx, fold):
    directory = args.run_dir / "work" / f"fold_{fold}"
    metadata = read_sealed(directory / "result", ctx.identity)
    fits = []
    for condition in CONDITIONS:
        for seed in ctx.config["seeds"]:
            fits.append(read_sealed(directory / "seeds" / f"{condition}_{seed}", ctx.identity))
    if metadata["seed_manifests"] != {f"{condition}_{seed}": file_sha256(directory / "seeds" / f"{condition}_{seed}" / "run.json")
                                      for condition in CONDITIONS for seed in ctx.config["seeds"]}:
        raise ValueError("Fold seed checkpoint provenance changed")
    frame = pd.read_csv(directory / "result/predictions.csv", float_precision="round_trip")
    if set(frame.model) != {"M0", *CONDITIONS, "B1", "B2", "B3"}:
        raise ValueError("Incomplete fixed candidate set in the completed fold")
    return frame, fits


def matched_intervals(paintings, config):
    wide = paintings[paintings.status.eq("scored")].pivot(index="painting", columns="model", values="contrast")
    pairs = (("M2", "M1"), ("M3", "M1"), ("M3", "M2"), ("B2", "B1"), ("B3", "B1"), ("B3", "B2"))
    deltas = {left + "_minus_" + right: wide[left] - wide[right] for left, right in pairs}
    frame = pd.DataFrame(deltas).reset_index().assign(status="scored")
    return paired_bootstrap(frame, config["bootstrap_replicates"], config["bootstrap_seed"], columns=list(deltas))


def run_cv(args):
    start = time.perf_counter()
    ctx = context(args)
    raw = load_raw(args, ctx.identity)
    budget = stage_budget(args, "CV")
    records = {name: [] for name in ("predictions", "paintings", "languages")}
    rows, all_fits = [], []
    for fold in range(1, ctx.config["cv_folds"] + 1):
        budget.check(f"fold {fold}")
        fit, held = ctx.train[ctx.train.fold.ne(fold)], ctx.train[ctx.train.fold.eq(fold)]
        directory = args.run_dir / "work" / f"fold_{fold}"
        baseline = baseline_for(ctx, held, fold)
        if not (directory / "result").exists():
            predictions, fits = fit_conditions(args, ctx, raw, fit, held, baseline, directory, fold, budget)
            with new_output(directory / "result") as stage:
                pd.concat([frame.assign(model=name) for name, frame in predictions.items()]).to_csv(
                    stage / "predictions.csv", index=False, lineterminator="\n")
                seal(stage, ctx.identity, {"seed_manifests": {
                    f"{condition}_{seed}": file_sha256(directory / "seeds" / f"{condition}_{seed}" / "run.json")
                    for condition in CONDITIONS for seed in ctx.config["seeds"]}})
        frame, fits = folded_predictions(args, ctx, fold)
        all_fits.extend(fits)
        for name, pred in frame.groupby("model", sort=False):
            pred = pred[[*data.KEYS, "p_positive"]]
            if name == "M0":
                verify_reference(pred, baseline)
            contrast, loss = score_result(held, pred, {"model": name, "fold": fold}, records)
            matching = [fit for fit in fits if fit["model"] == name]
            seconds = sum(fit["fit_seconds"] for fit in matching)
            rows.append({"model": name, "fold": fold, "contrast": contrast, "log_loss": loss,
                         "converged": all(fit["finite"] for fit in fits),
                         "fit_seconds": seconds, "predict_seconds": 0., "seconds": seconds})
            print(f"EXP-018 fold {fold} {name}: {contrast:.9f} / {loss:.9f}", flush=True)
    if (args.run_dir / "cv").exists():
        read_sealed(args.run_dir / "cv", ctx.identity)
        print("Reused sealed complete CV; Verify remains separately executable", flush=True)
        return
    budget.check("CV scoring")
    with new_output(args.run_dir / "cv") as stage:
        _, predictions = write_cv_analysis(stage, ctx.train, records, pd.DataFrame(rows), ctx.config, "M0", "M0")
        draws, intervals = matched_intervals(pd.concat(records["paintings"]), ctx.config)
        draws.to_csv(stage / "matched_bootstrap_distribution.csv", index=False, lineterminator="\n")
        json_output(intervals, stage / "matched_bootstrap.json")
        json_output({name: paired_order(ctx.train, frame[[*data.KEYS, "p_positive"]])
                     for name, frame in predictions.groupby("model")}, stage / "pair_order.json")
        json_output(all_fits, stage / "fits.json")
        seal(stage, ctx.identity, {"status": "AWAITING_REPLAY_VERIFICATION", "seconds": time.perf_counter() - start,
                                   "pilot_manifest_sha256": file_sha256(args.run_dir / "pilot/run.json"),
                                   "fold_manifests": {str(fold): file_sha256(args.run_dir / "work" / f"fold_{fold}" / "result/run.json")
                                                      for fold in range(1, ctx.config["cv_folds"] + 1)}})
    print("All CV fits and scoring completed. Run Verify before promotion or confirmation.", flush=True)


def choose_candidate(summary, config, complete):
    if not complete:
        return {"model": None, "status": "INCOMPLETE", "qualified": [],
                "all_seeds_finite_and_replayed": False, "eligibility": config["eligibility"]}
    baseline, control = summary.loc["M0"], summary.loc["B1"]
    if not np.isfinite(summary[["pooled_contrast", "log_loss_mean"]].to_numpy(dtype=float)).all():
        raise ValueError("Finite complete CV metrics are required before classification")
    candidates = summary.loc[["B2", "B3"]]
    finite = np.isfinite(candidates[["pooled_contrast", "log_loss_mean"]]).all(axis=1)
    qualifies = candidates[finite & candidates.pooled_contrast.ge(baseline.pooled_contrast + config["minimum_pooled_contrast_gain"])
                          & candidates.pooled_contrast.ge(control.pooled_contrast + config["minimum_incremental_gain"])
                          & candidates.fold_wins.ge(config["minimum_fold_wins"])
                          & candidates.log_loss_mean.le(baseline.log_loss_mean + config["log_loss_guardrail_bits"])
                          & candidates.folds.eq(config["cv_folds"])]
    winner = qualifies.pooled_contrast.idxmax() if len(qualifies) else "M0"
    return {"model": winner, "status": "PASS" if winner != "M0" else "NO_GO", "qualified": qualifies.index.tolist(),
            "all_seeds_finite_and_replayed": complete, "eligibility": config["eligibility"]}


def verified_decision(args):
    identity = fingerprint(args)
    result = read_sealed(args.run_dir / "decision", identity)
    cv = read_sealed(args.run_dir / "cv", identity)
    read_sealed(args.run_dir / "pilot", identity)
    if cv["pilot_manifest_sha256"] != file_sha256(args.run_dir / "pilot/run.json"):
        raise ValueError("The required gradient pilot changed after CV")
    read_sealed(args.run_dir / "verification", identity)
    summary = pd.read_csv(args.run_dir / "cv/summary.csv", index_col="model", float_precision="round_trip")
    expected = choose_candidate(summary, config018(), True)
    if any(result[key] != value for key, value in expected.items()):
        raise ValueError("Decision does not match the frozen promotion rule")
    if (result["cv_manifest_sha256"] != file_sha256(args.run_dir / "cv/run.json")
            or result["verification_manifest_sha256"] != file_sha256(args.run_dir / "verification/run.json")):
        raise ValueError("Verified CV evidence changed after selection")
    ctx = SimpleNamespace(identity=identity, config=config018())
    for fold in range(1, ctx.config["cv_folds"] + 1):
        folded_predictions(args, ctx, fold)
        if file_sha256(args.run_dir / "work" / f"fold_{fold}" / "result/run.json") != cv["fold_manifests"][str(fold)]:
            raise ValueError("Fold evidence changed after CV")
    return result


def run_verify(args):
    start = time.perf_counter()
    ctx = context(args)
    cv = read_sealed(args.run_dir / "cv", ctx.identity)
    raw = load_raw(args, ctx.identity)
    budget = stage_budget(args, "Verify")
    visual = concatenate_features(ctx.banks)
    device = torch.device(ctx.identity["device"])
    seed_rows, seed_metrics, curve_rows, fit_keys = [], [], [], []
    checked = 0
    for fold in range(1, ctx.config["cv_folds"] + 1):
        budget.check(f"Verify fold {fold}")
        directory = args.run_dir / "work" / f"fold_{fold}"
        if file_sha256(directory / "result/run.json") != cv["fold_manifests"][str(fold)]:
            raise ValueError("Fold provenance differs from the completed CV")
        stored, fits = folded_predictions(args, ctx, fold)
        verification_fold = args.run_dir / "verification_work" / (f"fold_{fold}_deep" if args.deep_verify else f"fold_{fold}")
        if verification_fold.exists():
            previous = read_sealed(verification_fold, ctx.identity)
            if previous["fold_manifest_sha256"] != file_sha256(directory / "result/run.json"):
                raise ValueError("A verified fold changed before replay resume")
            seed_rows.append(pd.read_csv(verification_fold / "seed_predictions.csv", float_precision="round_trip"))
            curve_rows.extend(pd.read_csv(verification_fold / "curve_summary.csv").to_dict(orient="records"))
            fit_keys.extend((fold, condition, seed) for condition in CONDITIONS for seed in ctx.config["seeds"])
            checked += len(CONDITIONS) * len(ctx.config["seeds"])
            print(f"Reused sealed replay for fold {fold}", flush=True)
            continue
        first_seed, first_curve = len(seed_rows), len(curve_rows)
        fit, held = ctx.train[ctx.train.fold.ne(fold)], ctx.train[ctx.train.fold.eq(fold)]
        inner, stopping = inner_partition(fit, ctx.config, fold)
        require_isolation(inner, stopping, held)
        baseline = baseline_for(ctx, held, fold)
        verify_reference(stored[stored.model.eq("M0")][[*data.KEYS, "p_positive"]], baseline)
        for phase, rows in (("inner", inner), ("full", fit)):
            bundle = load_geometry(directory / "geometry" / phase, ctx.identity, rows.painting)
            raw_records, raw_means, _ = raw
            indices = pd.MultiIndex.from_frame(raw_records[data.KEYS]).get_indexer(
                pd.MultiIndex.from_frame(bundle["records"][data.KEYS]))
            if (indices < 0).any() or set(bundle["records"].painting) != set(rows.painting):
                raise ValueError("PCA target rows differ from the training partition")
            with np.load(directory / "geometry" / phase / "targets.npz", allow_pickle=False) as archive:
                np.testing.assert_allclose(archive["mean"], raw_means[indices, 0].mean(axis=0), atol=1e-7, rtol=0.)
            if args.deep_verify:
                fresh = fit_partition_geometry(raw, rows, ctx.config)
                with np.load(directory / "geometry" / phase / "targets.npz", allow_pickle=False) as archive:
                    np.testing.assert_allclose(archive["mean"], fresh["pca"].mean_, atol=1e-7, rtol=0.)
                    np.testing.assert_allclose(archive["components"], fresh["pca"].components_, atol=2e-6, rtol=0.)
                np.testing.assert_allclose(bundle["targets"], fresh["targets"], atol=3e-5, rtol=0.)
        for condition in CONDITIONS:
            probabilities = []
            for seed in ctx.config["seeds"]:
                budget.check(f"Replay fold {fold} {condition} {seed}")
                seed_directory = directory / "seeds" / f"{condition}_{seed}"
                info = read_sealed(seed_directory, ctx.identity)
                payload = torch.load(seed_directory / "student.pt", map_location="cpu", weights_only=True)
                if payload["languages"] != sorted(fit.language.unique()):
                    raise ValueError("Student language vocabulary differs from its training partition")
                model = PolarityStudent(payload["languages"], ctx.config).to(device)
                model.load_state_dict(payload["state_dict"], strict=True)
                if not all(bool(torch.isfinite(p).all()) for p in model.parameters()):
                    raise ValueError("Nonfinite saved student parameters")
                values = predict_student(model, held[[*data.KEYS, "image"]], visual, baseline, device, ctx.config["batch_size"])
                with np.load(seed_directory / "predictions.npz", allow_pickle=False) as archive:
                    np.testing.assert_array_equal(values, archive["probabilities"])
                probabilities.append(values)
                prediction = held[data.KEYS].assign(p_positive=np.round(values, 6))
                seed_rows.append(prediction.assign(model=condition, seed=seed, fold=fold))
                curves = pd.read_csv(seed_directory / "curves.csv")
                selection = curves[curves.phase.eq("selection")]
                best = selection.loc[selection.validation_bce.idxmin()]
                if int(best.epoch) != info["selected_epoch"] or len(curves[curves.phase.eq("refit")]) != info["selected_epoch"]:
                    raise ValueError("Saved epoch selection differs from internal BCE curves")
                curve_rows.append({"model": condition, "seed": seed, "fold": fold,
                                   "selected_epoch": info["selected_epoch"], "best_validation_bce": best.validation_bce,
                                   "final_auxiliary": curves.iloc[-1].train_auxiliary})
                fit_keys.append((fold, condition, seed))
                checked += 1
                del model
            prediction = held[data.KEYS].assign(p_positive=np.round(np.mean(probabilities, axis=0), 6))
            verify_reference(prediction, stored[stored.model.eq(condition)][[*data.KEYS, "p_positive"]])
            verify_reference(blend(baseline, prediction, .5), stored[stored.model.eq("B" + condition[1:])][[*data.KEYS, "p_positive"]])
        with new_output(verification_fold) as stage:
            pd.concat(seed_rows[first_seed:], ignore_index=True).to_csv(stage / "seed_predictions.csv", index=False, lineterminator="\n")
            pd.DataFrame(curve_rows[first_curve:]).to_csv(stage / "curve_summary.csv", index=False, lineterminator="\n")
            seal(stage, ctx.identity, {"fold_manifest_sha256": file_sha256(directory / "result/run.json"),
                                       "deep_pca_replay": args.deep_verify})
        print(f"Verified every seed, target boundary and ensemble for fold {fold}", flush=True)
    expected = {(fold, condition, seed) for fold in range(1, ctx.config["cv_folds"] + 1)
                for condition in CONDITIONS for seed in ctx.config["seeds"]}
    if set(fit_keys) != expected or len(fit_keys) != len(expected):
        raise ValueError("Incomplete seed/fold replay coverage")
    predictions = pd.read_csv(args.run_dir / "cv/predictions.csv", float_precision="round_trip")
    from .metrics import score

    summary = pd.read_csv(args.run_dir / "cv/summary.csv", index_col="model", float_precision="round_trip")
    for name, rows in predictions.groupby("model"):
        metric = score(ctx.train, rows[[*data.KEYS, "p_positive"]])
        np.testing.assert_allclose([metric["cultural_contrast"], metric["log_loss"]],
                                   summary.loc[name, ["pooled_contrast", "pooled_log_loss"]].astype(float), atol=1e-14, rtol=0.)
    seeds = pd.concat(seed_rows, ignore_index=True)
    for (condition, seed), rows in seeds.groupby(["model", "seed"]):
        for name, pred in ((condition, rows[[*data.KEYS, "p_positive"]]),
                           ("B" + condition[1:], blend(ctx.reference[[*data.KEYS, "p_positive"]], rows[[*data.KEYS, "p_positive"]], .5))):
            metrics = score(ctx.train, pred)
            seed_metrics.append({"model": name, "seed": seed, "contrast": metrics["cultural_contrast"], "log_loss": metrics["log_loss"]})
    history = json.loads((args.run_dir / "preflight/historical_sha256.json").read_text())
    for path, digest in history.items():
        if file_sha256(Path(path)) != digest:
            raise ValueError(f"Historical artifact changed: {path}")
    if not (args.run_dir / "verification").exists():
        with new_output(args.run_dir / "verification") as stage:
            seeds.to_csv(stage / "seed_predictions.csv", index=False, lineterminator="\n")
            pd.DataFrame(seed_metrics).to_csv(stage / "seed_metrics.csv", index=False, lineterminator="\n")
            pd.DataFrame(curve_rows).to_csv(stage / "curve_summary.csv", index=False, lineterminator="\n")
            seal(stage, ctx.identity, {"checkpoints_replayed": checked, "deep_pca_replay": args.deep_verify,
                                       "historical_files_matched": len(history), "seconds": time.perf_counter() - start})
    else:
        read_sealed(args.run_dir / "verification", ctx.identity)
    decision = choose_candidate(summary, ctx.config, True)
    if not (args.run_dir / "decision").exists():
        with new_output(args.run_dir / "decision") as stage:
            seal(stage, ctx.identity, {**decision, "cv_manifest_sha256": file_sha256(args.run_dir / "cv/run.json"),
                                       "verification_manifest_sha256": file_sha256(args.run_dir / "verification/run.json")})
    verified_decision(args)
    print(f"EXP-018 verified CV decision: {decision['status']} {decision['model']}", flush=True)
