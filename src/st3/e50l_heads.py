"""Prepare a separate, parity-verified CLIP-L head for EXP-010 inference."""

import importlib.metadata

import numpy as np

from . import data, submission
from .ablation import predict_lr
from .baseline import C, MAX_ITER, file_sha256
from .cli import ROOT, git_head, json_output
from .contrast_study import new_output
from .e50_heads import (ENCODERS, HEADS, INDEX, MANIFEST, fit_frozen_head, head_from_arrays,
                        image_content_index, inference_config, load_bundle, load_legacy, read_json)
from .frozen_features import cache_location, load_frozen_cache, verify_asset_hashes
from .fusion import blend, verify_reference

CLIP_LARGE = "clip_large"
SPLITS = ("train", "dev")


def clip_large_config():
    config = read_json(ROOT / "configs" / "exp010.json")["encoder"]
    config["transformers"] = config.pop("transformers_version")
    return config


def compose_e50l(e50, clip_large):
    from .clip_large_study import E50_LARGE, compose

    return compose(E50_LARGE, clip_large, e50=e50)


def load_clip_large_bundle(directory, e50_manifest):
    config = read_json(ROOT / "configs" / "e50l_inference.json")
    manifest = read_json(directory / MANIFEST)
    if (manifest["version"] != config["bundle_version"] or manifest["encoder"] != clip_large_config()
            or any(manifest[key] != config[key] for key in ("reference_csv_sha256", "reference_zip_sha256",
                                                           "training_pairs"))):
        raise ValueError("CLIP-L bundle differs from frozen EXP-010")
    if (manifest["e50_heads_sha256"] != e50_manifest["files_sha256"][HEADS]
            or any(manifest[key] != e50_manifest[key] for key in ("languages", "annotations_sha256"))):
        raise ValueError("CLIP-L bundle was verified with different E50 heads or training inputs")
    for name in (HEADS, INDEX):
        if file_sha256(directory / name) != manifest["files_sha256"][name]:
            raise ValueError(f"Frozen CLIP-L bundle failed its checksum for {name}")
    with np.load(directory / HEADS, allow_pickle=False) as archive:
        model = head_from_arrays(archive["coef"], archive["intercept"], manifest["encoder"]["width"],
                                 manifest["languages"])
    index = read_json(directory / INDEX)
    if set(index) != set(SPLITS) or not set(index["train"]).isdisjoint(index["dev"]):
        raise ValueError("CLIP-L legacy index has overlapping or missing splits")
    return model, manifest, index


def load_clip_large_legacy(wanted, index, legacy, cache_dir, width):
    vectors, archives = {}, {}
    for split in SPLITS:
        features, archives[split] = load_legacy(CLIP_LARGE, wanted, index[split], legacy[split], cache_dir, width)
        vectors.update(features)
    return vectors, {"matched": sum(info["matched"] for info in archives.values()), "archives": archives,
                     "status": ("archive_unavailable" if any(info["status"] == "archive_unavailable"
                                                             for info in archives.values())
                                else "verified" if vectors else "no_matching_content")}


def load_experiment_cache(args, names, source, config):
    names = sorted(set(names))
    identity = source["identity"]
    expected = {key: config[key] for key in ("model", "revision", "width", "feature", "batch_size", "assets_sha256")}
    if (any(identity[key] != value for key, value in expected.items())
            or identity["runtime"]["transformers"] != config["transformers"]
            or identity["attention"] != "eager" or identity["processor"] != "slow"):
        raise ValueError("EXP-010 cache has a different CLIP-L encoder profile")
    location = cache_location(args.data_dir, names, args.cache_dir, identity, "clip-large")
    vectors, manifest = load_frozen_cache(location, names, identity)
    verify_asset_hashes(manifest["assets_sha256"], config)
    if manifest["features_sha256"] != source["features_sha256"]:
        raise ValueError("EXP-010 feature cache has changed")
    return vectors, {"file": f"{location.name}/features.npz", "sha256": manifest["features_sha256"], "source": manifest}


def prepare_clip_large_head(args):
    from .clip_large_study import E50_LARGE

    if args.clip_large_model_dir.exists():
        raise ValueError(f"CLIP-L model bundle already exists: {args.clip_large_model_dir}")
    root = args.data_dir
    config = read_json(ROOT / "configs" / "e50l_inference.json")
    encoder = clip_large_config()
    baseline = read_json(ROOT / "configs" / "st3.json")
    e50_config = inference_config()
    models, e50_manifest, e50_index = load_bundle(args.model_dir, e50_config, ENCODERS)
    if git_head(root) != baseline["upstream_revision"] or e50_manifest["upstream_revision"] != baseline["upstream_revision"]:
        raise ValueError("CLIP-L preparation requires the pinned organizer checkout")

    exp_dev = args.exp010_dir / "dev"
    prior_path = exp_dev / "run.json"
    prior = read_json(prior_path)
    reference = exp_dev / "predictions" / E50_LARGE / submission.FILENAME
    if (prior["selection"]["model"] != E50_LARGE or prior["submission_sha256"] != config["reference_zip_sha256"]
            or file_sha256(exp_dev / "submission.zip") != config["reference_zip_sha256"]
            or file_sha256(reference) != config["reference_csv_sha256"]
            or prior["reference_predictions_sha256"] != e50_config["reference_csv_sha256"]["e50"]):
        raise ValueError("EXP-010 reference predictions or ZIP differ from the pinned result")
    for name in ("run.json", "summary.csv"):
        if file_sha256(args.exp010_dir / "cv" / name) != prior["selection"]["sha256"][name]:
            raise ValueError("EXP-010 frozen CV provenance has changed")
    for key, filename in (("annotations_sha256", data.ANNOTATIONS), ("dev_input_sha256", data.DEV_INPUT)):
        if file_sha256(root / filename) != prior[key] or prior[key] != e50_manifest[key]:
            raise ValueError(f"CLIP-L preparation inputs differ from EXP-010: {key}")

    train = data.training_pairs(data.read_annotations(root, training_only=True)).assign(split="train")
    dev = data.read_dev_input(root)
    languages = e50_manifest["languages"]
    seed = prior["exp002_config"]["fit_seed"]
    if (len(train) != config["training_pairs"] or sorted(train["language"].unique()) != languages
            or seed != baseline["seed"] or seed != e50_manifest["lr"]["seed"]):
        raise ValueError("CLIP-L training pairs, vocabulary, or seed differ from the frozen system")
    if set(train["painting"]) & set(dev["painting"]) or set(train["image"]) & set(dev["image"]):
        raise ValueError("Development inputs overlap the original training paintings or images")
    features, legacy = {}, {}
    for split, frame in (("train", train), ("dev", dev)):
        features[split], legacy[split] = load_experiment_cache(args, frame["image"],
                                                               prior[f"{split}_clip_large_embedding"], encoder)
    index = {split: image_content_index(root, vectors) for split, vectors in features.items()}
    if not set(index["train"]).isdisjoint(index["dev"]):
        raise ValueError("Original training and development contain identical artwork bytes")

    components = {}
    dev_images = {name: file_sha256(data.image_path(root, name)) for name in features["dev"]}
    for name in ENCODERS:
        vectors, _ = load_legacy(name, dev_images.values(), e50_index, e50_manifest["legacy"][name], args.cache_dir,
                                 e50_config["encoders"][name]["width"])
        if set(vectors) != set(dev_images.values()):
            raise ValueError(f"Missing verified E50 development features for {name}")
        components[name] = predict_lr(models[name], dev, {image: vectors[digest] for image, digest in dev_images.items()},
                                      languages)
    e50 = blend(components["clip"], components["siglip2"], e50_config["blend_clip_weight"])
    if e50.to_csv(index=False, lineterminator="\n").encode("utf-8") != (
            exp_dev / "predictions" / "E50" / submission.FILENAME).read_bytes():
        raise ValueError("Original E50 head bundle failed EXP-010 development parity")

    with new_output(args.clip_large_model_dir) as stage:
        model = fit_frozen_head(train, features["train"], languages, seed)
        iterations = int(model.n_iter_[0])
        if iterations != prior["fits"]["CLIP-L"]["iterations"]:
            raise ValueError("CLIP-L fit differs from the EXP-010 convergence record")
        np.savez(stage / HEADS, coef=model.coef_, intercept=model.intercept_)
        json_output(index, stage / INDEX)
        manifest = {
            **config, "system": "EXP-010 E50+L", "version": config["bundle_version"], "encoder": encoder,
            "languages": languages, "legacy": legacy, "e50_heads_sha256": e50_manifest["files_sha256"][HEADS],
            "training_paintings": train["painting"].nunique(), "upstream_revision": baseline["upstream_revision"],
            "project_revision": git_head(ROOT), "exp010_run_sha256": file_sha256(prior_path),
            "annotations_sha256": prior["annotations_sha256"], "dev_input_sha256": prior["dev_input_sha256"],
            "lr": {"C": C, "max_iter": MAX_ITER, "solver": "lbfgs", "seed": seed, "iterations": iterations},
            "packages": {name: importlib.metadata.version(name)
                         for name in ("scikit-learn", "numpy", "scipy", "pandas", "threadpoolctl")},
            "files_sha256": {name: file_sha256(stage / name) for name in (HEADS, INDEX)},
        }
        json_output(manifest, stage / MANIFEST)
        loaded, _, _ = load_clip_large_bundle(stage, e50_manifest)
        prediction = predict_lr(loaded, dev, features["dev"], languages)
        verify_reference(prediction, predict_lr(model, dev, features["dev"], languages))
        if compose_e50l(e50, prediction).to_csv(index=False, lineterminator="\n").encode("utf-8") != reference.read_bytes():
            raise ValueError("Reloaded CLIP-L failed exact EXP-010 E50+L development parity")
        manifest["parity_pairs"] = len(dev)
        json_output(manifest, stage / MANIFEST)
    print(f"Saved CLIP-L head after exact EXP-010 parity on {len(dev)} pairs: {args.clip_large_model_dir}", flush=True)
