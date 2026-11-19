"""Read-only recovery of the original sealed EXP-029 pilot."""
import argparse
import hashlib
import json
from pathlib import Path
import tempfile
import zipfile

from run_exp029 import (CONFIG, KEYS, PARENT, ROOT, ZIP_SHA256, Budget, fingerprint, inputs,
                        np, pd, probabilities, reference, score_pilot, sha256, torch,
                        verify_model, weighted_bce, write_json)
from st3.experiment_workflow import export_artifacts

SOURCE = "061f6bf0d3aa4d0b24bfbf1766e9139f307a9d0c"
ATTEMPT = "20261119T103915Z-2c837e97"
MODEL_ID = "3ad45229687ce1fb1315843c29955c49dde33832a74ea5b8f2ccbdeeb17e8e26"
ARMS = ("projector_only", "projector_qlora")
SUCCESS = "PILOT_RECOVERED / EXPLORATORY"


def original_receipt(attempt):
    digest = sha256(attempt / "outputs.zip")
    if digest != (attempt / "outputs.sha256").read_text().split()[0]:
        raise ValueError("Original diagnostic archive checksum failed")
    with zipfile.ZipFile(attempt / "outputs.zip") as archive:
        manifest = json.loads(archive.read("export_manifest.json"))
        files = manifest["files"]
        if (len(archive.namelist()) != len(files) + 1
                or set(archive.namelist()) != {*files, "export_manifest.json"}
                or any(hashlib.sha256(archive.read(n)).hexdigest() != h for n, h in files.items())):
            raise ValueError("Original export member verification failed")
        identity = json.loads(archive.read("identity.json"))
        crossfit = json.loads(archive.read("crossfit.json"))
        shuffle = "projector_qlora_shuffled.csv"
        shuffle_bytes = archive.read(shuffle) if shuffle in files else None
    report = manifest["report"]
    expected = {"source": SOURCE, "config": CONFIG, "runtime": PARENT["runtime"],
                "archive_sha256": ZIP_SHA256, "qwen_revision": PARENT["vlm"]["revision"], "gpu": "NVIDIA L4"}
    if (report["attempt"] != ATTEMPT or report["source"] != SOURCE
            or fingerprint(identity) != MODEL_ID or any(identity.get(k) != v for k, v in expected.items())):
        raise ValueError("Original training attempt or model identity differs")
    return identity, crossfit, digest, shuffle_bytes


def saved_arm(directory, identity, arm, val):
    seal = verify_model(directory, dict(identity, arm=arm))
    if set(seal["files"]) != {"best.pt", "validation_predictions.csv"}:
        raise ValueError(f"Unexpected sealed files for {arm}")
    saved = torch.load(directory / "best.pt", map_location="cpu", weights_only=True)
    predictions = pd.read_csv(directory / "validation_predictions.csv", float_precision="round_trip")
    logits = np.asarray(saved["validation_logits"], dtype=np.float64)
    if logits.shape != (len(val),) or not np.isfinite(logits).all():
        raise ValueError(f"Invalid saved validation logits for {arm}")
    pd.testing.assert_frame_equal(predictions, probabilities(val, logits), check_exact=True)
    curves = saved["curves"]
    if [r["epoch"] for r in curves] != list(range(1, CONFIG["epochs"] + 1)):
        raise ValueError(f"Incomplete epoch curves for {arm}")
    if any(not np.isfinite(r[k]) for r in curves for k in ("train_BCE_nats", "validation_BCE_nats")):
        raise ValueError(f"Nonfinite BCE curve for {arm}")
    best = min(curves, key=lambda r: r["validation_BCE_nats"])
    if saved["epoch"] != best["epoch"] or not np.isclose(
        weighted_bce(logits, val), best["validation_BCE_nats"], atol=1e-12, rtol=0
    ):
        raise ValueError(f"Checkpoint BCE or selected epoch differs for {arm}")
    return predictions, {k: saved[k] for k in ("epoch", "curves", "fit_seconds",
                         "peak_allocated_bytes", "peak_reserved_bytes")} | {"verified_seal": seal}


def replay_shuffle(path, train, val, blocks, cache):
    from run_exp029 import (backend, encode_prompts, new_correction, predict, shuffled_images,
                            visual_vectors)

    budget = Budget(.25, .25, cuda=True, reserve=30)
    tokenizer, decoder, _ = backend(cache, technical=False)
    torch.backends.cuda.matmul.allow_tf32 = False
    module = new_correction()
    saved = torch.load(path, map_location="cpu", weights_only=True)
    module.load_state_dict(saved["model"])
    with torch.no_grad():
        adapters = {name: p for name, p in decoder.named_parameters() if p.requires_grad}
        if adapters.keys() != saved["adapters"].keys():
            raise ValueError("Saved QLoRA parameter names differ from pinned decoder")
        for name, param in adapters.items():
            param.copy_(saved["adapters"][name])
    decoder.requires_grad_(False)
    module.requires_grad_(False)
    languages = sorted(set(train.language) | set(val.language))
    ids, mask = encode_prompts(tokenizer, languages)
    vocab = {lang: row[m.bool()].cpu() for lang, row, m in zip(languages, ids, mask)}
    vectors = visual_vectors(blocks, val)
    logits = predict(module, decoder, val, vocab, vectors, budget)
    np.testing.assert_allclose(logits, saved["validation_logits"], atol=1e-6, rtol=0)
    original = pd.read_csv(path.with_name("validation_predictions.csv"), float_precision="round_trip")
    pd.testing.assert_frame_equal(probabilities(val, logits), original, check_exact=True)
    return probabilities(val, predict(module, decoder, val, vocab, vectors, budget,
                                      shuffled_images(val, CONFIG["seed"])))


def recover(handoff, attempt, models, work, output):
    identity, original_fits, archive_sha, shuffle_bytes = original_receipt(attempt)
    (train, val), blocks = inputs(handoff, work)
    train, val, fits = reference(train, val, blocks, Budget(.25, .25, cuda=False, reserve=0))
    for key, frame in (("train_crossfit", train), ("validation_baseline", val)):
        if fingerprint(frame.p_positive.tolist()) != identity[key]:
            raise ValueError(f"Reconstructed {key} differs from original training receipt")
    for got, original in zip(fits, original_fits, strict=True):
        for key in ("fit_paintings", "predicted_paintings"):
            if got[key] != original[key]:
                raise ValueError("Reconstructed cross-fitting painting membership differs")
    predictions = {"E50+L": val[KEYS + ["p_positive"]]}
    stats = {}
    for arm in ARMS:
        predictions[arm], stats[arm] = saved_arm(models / MODEL_ID / arm, identity, arm, val)
    shuffle = "UNAVAILABLE: not persisted by original run; requires checkpoint-only L4 inference"
    if shuffle_bytes is not None:
        import io
        predictions["projector_qlora_shuffled"] = pd.read_csv(io.BytesIO(shuffle_bytes), float_precision="round_trip")
        shuffle = "VERIFIED_ORIGINAL_EXPORT"
    metrics = score_pilot(val, predictions)
    baseline = metrics["E50+L"]
    if (round(baseline["cultural_contrast"], 6), round(baseline["log_loss"], 6)) != (.417755, .781648):
        raise ValueError("Reconstructed E50+L scores differ from the expected receipt")
    comparisons = [("projector_only", "E50+L"), ("projector_qlora", "E50+L"), ("projector_qlora", "projector_only")]
    if shuffle_bytes is not None:
        comparisons.append(("projector_qlora", "projector_qlora_shuffled"))
    report = {"status": SUCCESS, "training_attempt": ATTEMPT, "training_source": SOURCE,
              "original_archive_sha256": archive_sha, "model_identity": MODEL_ID, "baseline_identity_verified": True,
              "metrics": metrics, "checkpoints": stats, "shuffle": shuffle,
              "comparisons": {f"{a} vs {b}": {
                  key: metrics[a][key] - metrics[b][key] if None not in (metrics[a][key], metrics[b][key]) else None
                  for key in ("cultural_contrast", "log_loss")} for a, b in comparisons}}
    for arm, frame in predictions.items():
        frame.to_csv(output / f"{arm}.csv", index=False, float_format="%.6f")
    write_json(output / "recovered.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("handoff", "original-attempt", "models", "destination"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--authorized-l4-model-cache", type=Path,
                        help="Opt in to checkpoint-only L4 shuffle inference after CPU verification")
    args = parser.parse_args()
    if args.authorized_l4_model_cache is None:
        import os
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    args.destination.mkdir(parents=True, exist_ok=False)
    with tempfile.TemporaryDirectory(prefix="exp029-recovery-") as temp:
        output = Path(temp) / "results"
        output.mkdir()
        try:
            report = recover(args.handoff, args.original_attempt, args.models, Path(temp) / "inputs", output)
        except Exception as exc:
            write_json(args.destination / "status.json", {"status": "BLOCKED", "error": str(exc)})
            raise
        export_artifacts(output, args.destination, report)
        write_json(args.destination / "recovered.json", report)
        if args.authorized_l4_model_cache is not None and report["shuffle"].startswith("UNAVAILABLE"):
            try:
                (train, val), blocks = inputs(args.handoff, Path(temp) / "shuffle_inputs")
                baseline = pd.read_csv(output / "E50+L.csv", float_precision="round_trip")
                from run_exp029 import baseline_frame
                val = baseline_frame(val, baseline)
                preds = replay_shuffle(args.models / MODEL_ID / "projector_qlora/best.pt", train, val, blocks,
                                       args.authorized_l4_model_cache)
                metric = score_pilot(val, {"shuffle": preds})["shuffle"]
                original = report["metrics"]["projector_qlora"]
                report["comparisons"]["projector_qlora vs projector_qlora_shuffled"] = {
                    key: original[key] - metric[key] if None not in (original[key], metric[key]) else None
                    for key in ("cultural_contrast", "log_loss")}
                report["metrics"]["projector_qlora_shuffled"] = metric
                report["shuffle"] = "VERIFIED_CHECKPOINT_ONLY_INFERENCE"
                preds.to_csv(output / "projector_qlora_shuffled.csv", index=False, float_format="%.6f")
                write_json(output / "recovered.json", report)
                export_artifacts(output, args.destination / "shuffle", report)
                write_json(args.destination / "shuffle/recovered.json", report)
            except Exception as exc:
                write_json(args.destination / "shuffle_status.json", {"status": "BLOCKED", "error": str(exc)})
                raise
        print(json.dumps({name: {key: m[key] for key in ("cultural_contrast", "log_loss")}
                          for name, m in report["metrics"].items()}, indent=2))
        print(report["shuffle"])
        print(args.destination)


if __name__ == "__main__":
    main()
