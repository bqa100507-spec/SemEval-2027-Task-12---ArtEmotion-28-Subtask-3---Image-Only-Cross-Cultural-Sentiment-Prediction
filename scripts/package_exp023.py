"""Build a private, self-contained cached-feature handoff. Never parse confirmation labels."""

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import zipfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import numpy as np
import pandas as pd
from st3 import data
from st3.e50_heads import inference_config, load_bundle, load_legacy
from st3.e50l_heads import load_clip_large_bundle, load_clip_large_legacy
from st3.experiment_workflow import restore_handoff, sha256, write_json
from st3.exp023_data import load_inputs

ROOT = Path(__file__).resolve().parents[1]


def build(workspace, output, revision):
    workspace, output = Path(workspace), Path(output)
    output.mkdir(parents=True, exist_ok=True)
    config = json.loads((ROOT / "configs/exp023.json").read_text())
    git_config = subprocess.check_output(["git", "show", revision + ":configs/exp023.json"], cwd=ROOT)
    if hashlib.sha256(git_config).hexdigest() != sha256(ROOT / "configs/exp023.json"):
        raise ValueError("Commit the exact scientific source/config before packaging")
    with tempfile.TemporaryDirectory(prefix="build-", dir=output) as temporary:
        stage = Path(temporary)
        def copy(source, target):
            destination = stage / target
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, destination)
        prepared = workspace / "runs/task023_delivery/v2_portable_restore/inputs"
        prepared_manifest = json.loads((prepared / "run.json").read_text())
        for name, expected in prepared_manifest["files_sha256"].items():
            if sha256(prepared / name) != expected:
                raise ValueError("Original prepared EXP-022 inputs changed: " + name)
        copy(prepared / "targets.csv", "training/pairs.csv")
        pairs = pd.read_csv(prepared / "targets.csv", float_precision="round_trip")
        copy(workspace / "runs/exp017_text_supervision/audit/protocol.csv", "training/protocol.csv")
        protocol = pd.read_csv(stage / "training/protocol.csv")
        text = workspace / "runs/exp017_text_supervision/text/clip"
        if sha256(text / "features.npz") != config["caption_cache_sha256"]:
            raise ValueError("Caption cache differs from preregistration")
        for name in ("features.npz", "run.json", "tokens.csv"):
            copy(text / name, "text/" + name)
        # These are already verified, binary, training-only caption rows on 1335 paintings.
        # Actual caption strings are unnecessary: the exact cached original embeddings
        # and repeated annotation rows completely determine the frozen target protocol.
        captions = pd.read_csv(text / "tokens.csv")[["annotation_id", *data.KEYS, "emotion", "caption_id"]]
        captions = captions.assign(split="train", valid_caption=True, binary=True)
        captions.to_csv(stage / "training/captions.csv", index=False, lineterminator="\n")
        copy(workspace / "runs/exp017_text_supervision/audit/run.json", "history/exp017_audit.json")
        copy(workspace / "runs/exp017_text_supervision/independent_verification.json", "history/independent_verification.json")
        historical = pd.read_csv(workspace / "runs/exp017_text_supervision/cv/predictions.csv", float_precision="round_trip")
        historical = historical[historical.model.isin(["M0", "M1", "M2", "B1", "B2"])]
        historical.to_csv(stage / "history/exp017_predictions.csv", index=False, lineterminator="\n")
        current = pd.read_csv(workspace / "runs/exp022_colab_verified/c30c4ad20092b87c/cv/predictions.csv", float_precision="round_trip")
        current[current.model.eq("R0")][[*data.KEYS, "fold", "p_positive"]].to_csv(stage / "history/exp022_R0.csv", index=False, lineterminator="\n")
        for fold in range(1, 6):
            original = workspace / f"runs/exp017_text_supervision/cv_work/fold_{fold}"
            seeds = pd.read_csv(original / "seed_predictions.csv", float_precision="round_trip")
            seeds[seeds.model.isin(["M1", "M2"])].to_csv(stage / f"history/seeds_fold_{fold}.csv", index=False, lineterminator="\n")
            for name in ("partition.json", "run.json", "student_fits.json"):
                copy(original / name, f"history/fold_{fold}_{name}")
        # Opaque byte copy of the OLD training-only mapping. No parsing, label access,
        # scoring or confirmation caption/target construction occurs during packaging.
        copy(workspace / "runs/exp017_text_supervision/audit/annotation_mapping.csv",
             "sealed/original_training_annotation_mapping.csv")
        official = workspace / "data/upstream"
        dev_path = official / data.DEV_INPUT
        with dev_path.open(encoding="utf-8", newline="") as handle:
            dev = data.read_exact_csv(handle, data.INPUT_COLUMNS)
        copy(dev_path, "evaluation/st3_test.csv")
        names = sorted(set(protocol.image) | set(dev.image))
        by_image = {name: sha256(data.image_path(official, name)) for name in names}
        config_inference = inference_config()
        _, manifest, index = load_bundle(workspace / "runs/e50_inference_models", config_inference, ("clip", "siglip2"))
        _, large, large_index = load_clip_large_bundle(workspace / "runs/e50l_inference_models", manifest)
        banks = []
        for encoder in ("clip", "siglip2"):
            vectors, _ = load_legacy(encoder, by_image.values(), index, manifest["legacy"][encoder], workspace / "cache", config_inference["encoders"][encoder]["width"])
            if set(vectors) != set(by_image.values()):
                raise ValueError("Original visual cache incomplete: " + encoder)
            banks.append(np.stack([vectors[by_image[name]] for name in names]))
        vectors, _ = load_clip_large_legacy(by_image.values(), large_index, large["legacy"], workspace / "cache", large["encoder"]["width"])
        if set(vectors) != set(by_image.values()):
            raise ValueError("Original CLIP-L visual cache incomplete")
        banks.append(np.stack([vectors[by_image[name]] for name in names]))
        (stage / "visual").mkdir()
        np.savez(stage / "visual/features.npz", names=np.array(names), clip=banks[0], siglip=banks[1], large=banks[2])
        write_json(stage / "visual/image_hashes.json", by_image)
        for name in protocol.image.unique():
            copy(data.image_path(official, name), "artwork/images/" + name)
        for name in dev.image.unique():
            copy(data.image_path(official, name), "evaluation/images/" + name)
        write_json(stage / "visual/provenance.json", {"encoders": config_inference["encoders"], "clip_large": large["encoder"],
                                                       "source_feature_hashes": prepared_manifest["files_sha256"],
                                                       "production_heads_used_for_predictions": False, "R0_1335_must_be_fit_after_CV_GO": True})
        copy(ROOT / "requirements-exp023.lock.txt", "environment/requirements-exp023.lock.txt")
        (stage / "source").mkdir()
        subprocess.run(["git", "archive", "--format=zip", "-o", str(stage / "source/source.zip"), revision], cwd=ROOT, check=True)
        source_files = {}
        with zipfile.ZipFile(stage / "source/source.zip") as z:
            for name in z.namelist():
                if not name.endswith("/"):
                    source_files[name] = hashlib.sha256(z.read(name)).hexdigest()
        files = {p.relative_to(stage).as_posix(): sha256(p) for p in stage.rglob("*") if p.is_file()}
        handoff = {"experiment": "EXP-023", "source_revision": revision, "config_sha256": sha256(ROOT / "configs/exp023.json"),
                   "source_files_sha256": source_files, "files": files,
                   "contains_development_gold": False, "confirmation_labels_parsed_during_preparation": False,
                   "confirmation": "opaque historical training-only mapping; parsed solely after verified CV_GO and advance authorization",
                   "caption_text": "Exact permitted caption IDs, binary annotations and original embedding cache; raw strings not required or copied",
                   "cache_coverage": {"train": 1335, "confirmation": 236, "development_pairs": 849}}
        write_json(stage / "input_manifest.json", handoff)
        manifest_hash = sha256(stage / "input_manifest.json")
        temporary_zip = output / "exp023_handoff.partial"
        with zipfile.ZipFile(temporary_zip, "w", zipfile.ZIP_DEFLATED) as z:
            for p in sorted(stage.rglob("*")):
                if p.is_file():
                    z.write(p, p.relative_to(stage).as_posix())
        final = output / "exp023_handoff.zip"
        if final.exists():
            raise FileExistsError("Preserve the existing delivery; choose a new delivery directory")
        temporary_zip.replace(final)
        digest = sha256(final)
        (output / "exp023_handoff.sha256").write_text(digest + "  exp023_handoff.zip\n")
        (output / "input_manifest.sha256").write_text(manifest_hash + "  input_manifest.json\n")
    # Clean restore and independent read-only input reconstruction. No models trained,
    # no targets fit and no sealed confirmation mapping parsed.
    restored = output / "restore_check"
    manifest = restore_handoff(final, restored, digest, manifest_hash)
    train, captions, _, _, _, _, parity = load_inputs(restored, config)
    receipt = {"status": "PASS", "zip": str(final.resolve()), "zip_sha256": digest,
               "manifest_sha256": manifest_hash, "source_revision": revision, "files_verified": len(manifest["files"]),
               "training_pairs": len(train), "caption_rows": len(captions), "historical_parity": parity,
               "confirmation_labels_opened": False, "development_gold_included": False, "GPU_executed": False}
    write_json(output / "delivery.json", receipt)
    print(json.dumps(receipt, indent=2))
    return receipt


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-revision", required=True)
    args = parser.parse_args()
    build(args.workspace_root, args.output, args.source_revision)
