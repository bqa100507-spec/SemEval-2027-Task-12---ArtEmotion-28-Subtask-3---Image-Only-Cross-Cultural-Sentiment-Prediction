"""Prepare and load the frozen EXP-007 coefficients without executable serialization."""

import importlib.metadata
import json
from pathlib import Path, PureWindowsPath

import numpy as np
from sklearn.linear_model import LogisticRegression

from . import data, submission
from .ablation import load_study_embeddings, predict_lr
from .baseline import C, MAX_ITER, design_matrix, file_sha256, fit_lr
from .cli import ROOT, git_head, json_output
from .contrast_study import new_output
from .fusion import blend, verify_reference
from .siglip2 import load_siglip2_features

ENCODERS = ("clip", "siglip2")
HEADS = "heads.npz"
INDEX = "legacy-index.json"
MANIFEST = "manifest.json"


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def inference_config():
    config = read_json(ROOT / "configs" / "e50_inference.json")
    baseline = read_json(ROOT / "configs" / "st3.json")
    siglip = read_json(ROOT / "configs" / "exp006.json")
    config["encoders"]["clip"].update(model=baseline["clip_model"], revision=baseline["clip_revision"],
                                      batch_size=baseline["batch_size"])
    config["encoders"]["siglip2"].update(model=siglip["model"], revision=siglip["revision"],
                                         batch_size=siglip["batch_size"])
    return config


def fit_frozen_head(train, embeddings, languages, seed):
    if "split" not in train or not train["split"].eq("train").all():
        raise ValueError("Frozen head fitting accepts only original training rows")
    matrix = design_matrix(train["image"], train["language"], embeddings, languages)
    model = fit_lr(matrix, train, seed)
    if model.n_iter_[0] >= MAX_ITER:
        raise ValueError("Frozen Model C head did not converge")
    return model


def head_from_arrays(coef, intercept, width, languages):
    columns = width + len(languages) * (1 + width)
    if (coef.shape != (1, columns) or intercept.shape != (1,)
            or any(array.dtype != np.float64 or not np.isfinite(array).all() for array in (coef, intercept))):
        raise ValueError("Invalid frozen Model C coefficients or intercept")
    model = LogisticRegression(C=C, max_iter=MAX_ITER)
    model.coef_, model.intercept_ = coef, intercept
    model.classes_ = np.array([0, 1])
    model.n_features_in_ = columns
    return model


def load_bundle(directory, config, encoders):
    manifest = read_json(directory / MANIFEST)
    if (manifest["version"] != config["bundle_version"] or manifest["encoders"] != config["encoders"]
            or manifest["blend_clip_weight"] != config["blend_clip_weight"]
            or manifest["reference_csv_sha256"] != config["reference_csv_sha256"]):
        raise ValueError("Frozen model bundle does not match the pinned E50 system")
    for name in (HEADS, INDEX):
        if file_sha256(directory / name) != manifest["files_sha256"][name]:
            raise ValueError(f"Frozen model bundle failed its checksum for {name}")
    languages = manifest["languages"]
    expected = sorted(set(config["official_languages"]) - set(config["untrained_official_languages"]))
    if languages != expected:
        raise ValueError("Invalid frozen language vocabulary")
    with np.load(directory / HEADS, allow_pickle=False) as archive:
        models = {name: head_from_arrays(archive[f"{name}_coef"], archive[f"{name}_intercept"],
                                         config["encoders"][name]["width"], languages) for name in encoders}
    return models, manifest, read_json(directory / INDEX)


def prepare_heads(args):
    if args.model_dir.exists():
        raise ValueError(f"Model bundle already exists: {args.model_dir}")
    root = args.data_dir
    baseline = read_json(ROOT / "configs" / "st3.json")
    config = inference_config()
    if git_head(root) != baseline["upstream_revision"]:
        raise ValueError("Head preparation requires the pinned organizer checkout")
    prior_path = args.exp007_dir / "dev" / "run.json"
    prior = read_json(prior_path)
    if (prior["selection"]["model"] != "E50" or prior["config"]["alphas"]["E50"] != config["blend_clip_weight"]
            or prior["upstream_revision"] != baseline["upstream_revision"]):
        raise ValueError("Expected the frozen EXP-007 E50 development run")

    annotations = data.read_annotations(root, training_only=True)
    train = data.training_pairs(annotations).assign(split="train")
    dev = data.read_dev_input(root)
    if set(train["painting"]) & set(dev["painting"]) or set(train["image"]) & set(dev["image"]):
        raise ValueError("Development inputs overlap original training paintings or images")
    clip, provenance = load_study_embeddings(root, train, dev, args.cache_dir, args.embedding_run,
                                             baseline["upstream_revision"], baseline)
    siglip_config = read_json(ROOT / "configs" / "exp006.json")
    siglip, siglip_meta = load_siglip2_features(args, dev, train, siglip_config,
                                               extraction_dir=args.exp006_dir / "extraction")
    if (any(provenance[key] != prior[key] for key in ("annotations_sha256", "dev_input_sha256"))
            or provenance["embedding"]["sha256"] != prior["embedding"]["sha256"]
            or siglip_meta["features_sha256"] != prior["siglip2_embedding"]["features_sha256"]
            or siglip_meta["assets_sha256"] != config["encoders"]["siglip2"]["assets_sha256"]):
        raise ValueError("Head preparation inputs differ from frozen EXP-007")

    legacy = {"clip": {"file": PureWindowsPath(provenance["embedding"]["path"]).name,
                       "sha256": provenance["embedding"]["sha256"], "source": provenance["embedding"]},
              "siglip2": {"file": f"{Path(siglip_meta['path']).name}/features.npz",
                          "sha256": siglip_meta["features_sha256"], "source": siglip_meta}}
    index = {}
    for name in sorted(clip):
        digest = file_sha256(data.image_path(root, name))
        previous = index.setdefault(digest, name)
        if previous != name and any(not np.array_equal(vectors[previous], vectors[name]) for vectors in (clip, siglip)):
            raise ValueError("Byte-identical legacy images have different features")

    languages = sorted(train["language"].unique())
    seed = prior["exp002_config"]["fit_seed"]
    if seed != baseline["seed"]:
        raise ValueError("Frozen EXP-007 fit seed differs from the pinned baseline")
    features = {"clip": clip, "siglip2": siglip}
    with new_output(args.model_dir) as stage:
        models = {name: fit_frozen_head(train, features[name], languages, seed) for name in ENCODERS}
        np.savez(stage / HEADS, **{f"{name}_{key}": getattr(model, f"{key}_")
                                   for name, model in models.items() for key in ("coef", "intercept")})
        json_output(index, stage / INDEX)
        manifest = {
            "version": config["bundle_version"], "system": "EXP-007 E50", "encoders": config["encoders"],
            "blend_clip_weight": config["blend_clip_weight"], "reference_csv_sha256": config["reference_csv_sha256"],
            "languages": languages, "training_pairs": len(train), "training_paintings": train["painting"].nunique(),
            "upstream_revision": baseline["upstream_revision"], "project_revision": git_head(ROOT),
            "exp007_run_sha256": file_sha256(prior_path), "annotations_sha256": provenance["annotations_sha256"],
            "dev_input_sha256": provenance["dev_input_sha256"], "legacy": legacy,
            "lr": {"C": C, "max_iter": MAX_ITER, "solver": "lbfgs", "seed": seed,
                   "iterations": {name: int(model.n_iter_[0]) for name, model in models.items()}},
            "packages": {name: importlib.metadata.version(name)
                         for name in ("scikit-learn", "numpy", "scipy", "pandas", "threadpoolctl")},
            "files_sha256": {name: file_sha256(stage / name) for name in (HEADS, INDEX)},
        }
        json_output(manifest, stage / MANIFEST)
        loaded, _, _ = load_bundle(stage, config, ENCODERS)
        predictions = {}
        for name in ENCODERS:
            predictions[name] = predict_lr(loaded[name], dev, features[name], languages)
            verify_reference(predictions[name], predict_lr(models[name], dev, features[name], languages))
        predictions["e50"] = blend(predictions["clip"], predictions["siglip2"], config["blend_clip_weight"])
        for mode, reference in (("clip", "C-CLIP"), ("e50", "E50")):
            expected = args.exp007_dir / "dev" / "predictions" / reference / submission.FILENAME
            check = stage / submission.FILENAME
            predictions[mode].to_csv(check, index=False, lineterminator="\n")
            if file_sha256(check) != config["reference_csv_sha256"][mode] or check.read_bytes() != expected.read_bytes():
                raise ValueError(f"Reloaded {mode} failed byte-for-byte EXP-007 development parity")
            check.unlink()
        manifest["parity_pairs"] = len(dev)
        json_output(manifest, stage / MANIFEST)
    print(f"Saved frozen E50 heads after exact EXP-007 parity on {len(dev)} input pairs: {args.model_dir}", flush=True)
