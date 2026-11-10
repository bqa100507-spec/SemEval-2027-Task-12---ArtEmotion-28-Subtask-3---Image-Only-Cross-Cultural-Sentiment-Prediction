"""First-fold resource pilot, five-fold comparison, and gated development."""

import time

import numpy as np
import pandas as pd

from . import data, submission
from .ablation import predict_lr
from .baseline import file_sha256
from .cli import json_output
from .clip_large_study import E50_LARGE, LARGE
from .clip_lora import ClipTail, adapter_parameters, frozen_digest, load_adapter, load_encoder, prefix_cache, save_adapter
from .contrast_study import new_output, write_development
from .e50_heads import head_from_arrays, read_json
from .fusion import blend, verify_reference
from .fusion_study import write_cv_analysis
from .lora_training import adapt_partition, predict_vectors
from .low_rank_study import score_result
from .nonlinear import configure_device, early_split, torch
from .nonlinear_study import device_metadata
from .residual_crossfit import outer_reference_bytes, read_csv, residual_inputs
from .residual_study import development_components, pair_ordering


class ResourceNoGo(RuntimeError):
    pass


def prepare(args, train, original, provenance, dev=None):
    if not torch.cuda.is_available():
        raise ResourceNoGo("CUDA is unavailable for the fixed EXP-014 pilot")
    device = configure_device("cuda")
    config = provenance["config"]
    if config["training_precision"] == "bf16" and not torch.cuda.is_bf16_supported():
        raise ResourceNoGo("The selected CUDA device does not support BF16 training")
    model, processor = load_encoder(args.cache_dir)
    bank, index, prefix = prefix_cache(args, train["image"], model, processor, config, device, "train")
    prefixes = {"train": prefix["features_sha256"]}
    if dev is not None:
        dev_bank, dev_index, manifest = prefix_cache(args, dev["image"], model, processor, config, device, "dev")
        index.update({name: i + len(bank) for name, i in dev_index.items()})
        bank = torch.cat([bank, dev_bank])
        prefixes["dev"] = manifest["features_sha256"]
    tail = ClipTail(model, config).to(device).eval()
    del model
    sample = train.loc[train["image"].isin(sorted(set(train["image"]))[:64])]
    replay = predict_vectors(tail, sample, bank, index, config["image_batch_size"])
    error = max(float(np.max(np.abs(vector - original[name]))) for name, vector in replay.items())
    if error > config["embedding_atol"]:
        raise ValueError(f"Zero adapter differs from original CLIP-L embeddings by {error}")
    parameters = adapter_parameters(tail)
    if len(parameters) != 8 or sum(value.numel() for value in parameters.values()) != 32768:
        raise ValueError("Unexpected trainable adapter count for pinned CLIP-L")
    return tail, bank, index, device, {"prefixes": prefixes, "zero_embedding_max_error": error,
                                       "trainable_parameters": 32768, "targets": tail.targets,
                                       "frozen_tail_sha256": frozen_digest(tail)}


def fit_fold(stage, tail, bank, index, fit, held, original, config, fold, device, audit, pilot=False):
    prediction, model, curves, info = adapt_partition(tail, bank, index, fit, held, original, config, fold, device, pilot)
    if frozen_digest(tail) != audit["frozen_tail_sha256"]:
        raise ValueError("Frozen CLIP-L tail weights changed")
    languages = info["languages"]
    baseline = predict_lr(model, held, original, languages)
    prediction.to_csv(stage / "adapted.csv", index=False, lineterminator="\n")
    baseline.to_csv(stage / "frozen.csv", index=False, lineterminator="\n")
    pd.DataFrame(curves).to_csv(stage / "curves.csv", index=False, lineterminator="\n")
    save_adapter(tail, stage / "adapter.npz")
    np.savez(stage / "head.npz", coef=model.coef_, intercept=model.intercept_)
    tail.reset(config["seed"])
    load_adapter(tail, stage / "adapter.npz")
    with np.load(stage / "head.npz", allow_pickle=False) as saved:
        reloaded = head_from_arrays(saved["coef"], saved["intercept"], 768, languages)
    replay = predict_vectors(tail, held, bank, index, config["image_batch_size"])
    verify_reference(predict_lr(reloaded, held, replay, languages), prediction)
    info["checkpoint_parity"] = True
    info["file_sha256"] = {name: file_sha256(stage / name)
                           for name in ("adapted.csv", "frozen.csv", "curves.csv", "adapter.npz", "head.npz")}
    return prediction, baseline, info


def run_lora_pilot(args, dev):
    destination = args.run_dir / "pilot"
    start = time.perf_counter()
    try:
        train, banks, folds, _, provenance = residual_inputs(args, dev, "exp014")
        original = banks["clip_l"]
        tail, bank, index, device, audit = prepare(args, train, original, provenance)
        fit, held = (train.iloc[rows] for rows in folds[0])
        print("EXP-014 pilot outer fold 1. No Contrast is scored.", flush=True)
        with new_output(destination) as stage:
            _, _, info = fit_fold(stage, tail, bank, index, fit, held, original, provenance["config"], 1,
                                  device, audit, pilot=True)
            json_output(device_metadata({**provenance, "status": "FEASIBLE", "encoder": audit, "fit": info},
                                        start, device), stage / "run.json")
        print("EXP-014 FEASIBLE. Pilot gradients, parity and checkpoint replay passed.", flush=True)
    except (ResourceNoGo, torch.OutOfMemoryError) as error:
        with new_output(destination) as stage:
            json_output({"status": "RESOURCE_NO_GO", "reason": str(error)}, stage / "run.json")
        print(f"EXP-014 RESOURCE_NO_GO. {error}", flush=True)


def choose_lora(summary, config, folds):
    baseline, candidate = summary.loc["M0"], summary.loc["M2"]
    threshold = float(baseline["pooled_contrast"] + config["minimum_pooled_contrast_gain"])
    loss_limit = float(baseline["log_loss_mean"] + config["log_loss_guardrail_bits"])
    passes = (candidate[["folds", "converged", "scored", "loss_scored"]].eq(folds).all()
              and np.isfinite([candidate["pooled_contrast"], candidate["log_loss_mean"]]).all()
              and candidate["pooled_contrast"] >= threshold and candidate["log_loss_mean"] <= loss_limit
              and candidate["fold_wins"] >= config["minimum_fold_wins"])
    return {"model": "M2" if passes else "M0", "minimum_pooled_contrast": threshold,
            "maximum_mean_log_loss": loss_limit, "minimum_fold_wins": config["minimum_fold_wins"]}


def read_verified_pilot(args, provenance, audit, fit, held):
    path = args.run_dir / "pilot"
    saved = read_json(path / "run.json")
    info = saved["fit"]
    if (any(saved[key] != value for key, value in provenance.items()) or saved["encoder"]["prefixes"] != audit["prefixes"]
            or saved["encoder"]["frozen_tail_sha256"] != audit["frozen_tail_sha256"]
            or not info["checkpoint_parity"] or not info["gradient_control"]
            or any(file_sha256(path / name) != digest for name, digest in info["file_sha256"].items())):
        raise ValueError("Pilot artifacts do not match the frozen experiment")
    inner_fit, val = early_split(fit, held, provenance["config"], 1)
    for name, rows in (("inner_fit", inner_fit), ("validation", val), ("held", held)):
        if info["audit"][name] != sorted(rows["painting"].unique()):
            raise ValueError("Pilot split audit differs")
    return read_csv(path / "adapted.csv"), read_csv(path / "frozen.csv"), info


def run_lora_cv(args, dev):
    start = time.perf_counter()
    pilot = read_json(args.run_dir / "pilot" / "run.json")
    if pilot["status"] != "FEASIBLE":
        raise ValueError("EXP-014 requires a successful resource and correctness pilot")
    with new_output(args.run_dir / "cv") as stage:
        train, banks, folds, outer, provenance = residual_inputs(args, dev, "exp014")
        config, original = provenance["config"], banks["clip_l"]
        tail, bank, index, device, audit = prepare(args, train, original, provenance)
        exp010 = read_csv(args.exp010_dir / "cv" / "predictions.csv")
        records = {name: [] for name in ("predictions", "paintings", "languages")}
        rows, fits, diagnostics = [], [], []
        for fold, (fit_idx, held_idx) in enumerate(folds, 1):
            fit, held = train.iloc[fit_idx], train.iloc[held_idx]
            if fold == 1:
                adapted, baseline, info = read_verified_pilot(args, provenance, audit, fit, held)
            else:
                print(f"EXP-014 outer fold {fold}", flush=True)
                folder = stage / f"outer-{fold}"
                folder.mkdir()
                adapted, baseline, info = fit_fold(folder, tail, bank, index, fit, held, original, config,
                                                   fold, device, audit)
            fits.append({"fold": fold, **info})
            expected = exp010.loc[exp010["model"].eq(LARGE) & exp010["fold"].eq(fold), [*data.KEYS, "p_positive"]]
            verify_reference(baseline, expected)
            reference = outer.loc[outer["outer_fold"].eq(fold)]
            e50 = reference[[*data.KEYS, "e50"]].rename(columns={"e50": "p_positive"})
            e50l = reference[[*data.KEYS, "q0"]].rename(columns={"q0": "p_positive"})
            diagnostics.append(baseline.assign(fold=fold))
            for name, pred in (("M0", e50l), ("M1", adapted), ("M2", blend(e50, adapted, .5))):
                context = {"model": name, "fold": fold}
                contrast, loss = score_result(held, pred, context, records)
                paintings = records["paintings"][-1]
                paintings["ordering_agreement"] = paintings["painting"].map(pair_ordering(held, pred))
                fit_seconds = info["fit_seconds"] if name == "M1" else 0.
                predict_seconds = info["predict_seconds"] if name == "M1" else 0.
                rows.append({**context, "contrast": contrast, "log_loss": loss, "converged": True,
                             "fit_seconds": fit_seconds, "predict_seconds": predict_seconds,
                             "seconds": fit_seconds + predict_seconds,
                             "ordering_agreement": paintings["ordering_agreement"].mean()})
                print(f"Outer {fold} {name}: {contrast:.6f} / {loss:.6f}", flush=True)

        summary, _ = write_cv_analysis(stage, train, records, pd.DataFrame(rows), config, "M0", "M0")
        summary["pooled_ordering"] = pd.concat(records["paintings"]).groupby("model")["ordering_agreement"].mean()
        summary.to_csv(stage / "summary.csv", lineterminator="\n")
        pd.concat(diagnostics).to_csv(stage / "frozen_clip_l.csv", index=False, lineterminator="\n")
        (stage / "reference_e50l.csv").write_bytes(outer_reference_bytes(outer, provenance["exp002_config"]["fit_seed"]))
        json_output(fits, stage / "fold_runs.json")
        json_output(device_metadata({**provenance, "encoder": audit}, start, device), stage / "run.json")
        decision = choose_lora(summary, config, len(folds))
        decision["sha256"] = {name: file_sha256(stage / name) for name in ("run.json", "summary.csv", "fold_runs.json")}
        json_output(decision, stage / "selection.json")
    print(f"Frozen EXP-014 decision: {decision['model']}", flush=True)


def run_lora_dev(args, dev):
    start = time.perf_counter()
    cv = args.run_dir / "cv"
    decision = read_json(cv / "selection.json")
    for name in ("run.json", "summary.csv", "fold_runs.json"):
        if decision["sha256"][name] != file_sha256(cv / name):
            raise ValueError(f"Frozen EXP-014 artifact changed: {name}")
    meta = read_json(cv / "run.json")
    config = meta["config"]
    summary = pd.read_csv(cv / "summary.csv", index_col="model", float_precision="round_trip")
    expected = choose_lora(summary, config, meta["exp002_config"]["cv_folds"])
    if any(decision[key] != value for key, value in expected.items()):
        raise ValueError("EXP-014 decision disagrees with the frozen gate")
    if decision["model"] == "M0":
        print("EXP-014 NO-GO. Retained E50+L without development scoring or packaging.", flush=True)
        return
    with new_output(args.run_dir / "dev") as stage:
        train, banks, _, _, provenance = residual_inputs(args, dev, "exp014")
        if any(meta[key] != value for key, value in provenance.items()):
            raise ValueError("Development inputs differ from the frozen EXP-014 run")
        reference, bundles, dev_banks = development_components(args, dev, include_features=True)
        original = {**banks["clip_l"], **dev_banks[2]}
        tail, bank, index, device, audit = prepare(args, train, original, provenance, dev)
        if audit["prefixes"]["train"] != meta["encoder"]["prefixes"]["train"]:
            raise ValueError("Training prefix differs from CV")
        adapted, _, fit = fit_fold(stage, tail, bank, index, train, dev, original, config, 0, device, audit)
        e50 = reference[[*data.KEYS, "e50"]].rename(columns={"e50": "p_positive"})
        baseline = reference[[*data.KEYS, "q0"]].rename(columns={"q0": "p_positive"})
        gold = data.validation_gold(data.read_annotations(args.data_dir), dev)
        write_development(stage, gold, {"M0": baseline, "M2": blend(e50, adapted, .5)}, "M2", config, "M0")
        archive = stage / "submission.zip"
        submission.pack(stage / "predictions" / "M2" / submission.FILENAME, archive, dev)
        submission.validate_archive(archive, dev)
        json_output(device_metadata({**provenance, "selection": decision, "fit": fit, "encoder": audit,
                                     "head_manifests_sha256": bundles, "submission_sha256": file_sha256(archive)},
                                    start, device), stage / "run.json")
    print(f"EXP-014 scored M2 and validated {len(dev)} development predictions", flush=True)
