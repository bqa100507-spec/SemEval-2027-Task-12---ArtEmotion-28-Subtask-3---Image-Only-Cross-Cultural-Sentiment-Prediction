"""Development-only adaptation of the organizers' MIT-licensed CLIP + LR baseline."""

import hashlib
import json
import os
import random
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import sparse
from sklearn.linear_model import LogisticRegression
from threadpoolctl import threadpool_limits

from .data import KEYS, image_path

C = 0.3
MAX_ITER = 3000


def file_sha256(path):
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def feature_cache_path(root, names, cache_dir, identity):
    digest = hashlib.sha256(json.dumps(identity, sort_keys=True).encode())
    for name in names:
        digest.update(name.encode("utf-8") + b"\0" + file_sha256(image_path(root, name)).encode())
    return cache_dir / f"clip-{digest.hexdigest()}.npz"


def load_feature_cache(cache, names):
    with np.load(cache, allow_pickle=False) as saved:
        features = saved["features"]
        if saved["names"].tolist() != names or features.shape != (len(names), 512) or not np.isfinite(features).all():
            raise ValueError(f"Invalid feature cache: {cache}. Remove it to rebuild embeddings.")
    return features


def image_features(root, names, cache_dir, config, requested_device):
    # Set the cache before importing anything that loads Hugging Face Hub.
    os.environ.setdefault("HF_HUB_CACHE", str(cache_dir / "huggingface"))
    os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "120")
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    import torch
    import transformers

    device = requested_device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA requested but torch.cuda.is_available() is false")
    random.seed(config["seed"])
    np.random.seed(config["seed"])
    torch.manual_seed(config["seed"])
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(min(4, os.cpu_count() or 1))
    names = sorted(names)
    identity = {"model": config["clip_model"], "revision": config["clip_revision"],
                "batch_size": config["batch_size"], "torch": str(torch.__version__),
                "transformers": transformers.__version__, "device": device,
                "device_name": torch.cuda.get_device_name() if device == "cuda" else "CPU",
                "feature_format": "normalized-clip-projection-fp32-v1"}
    cache = feature_cache_path(root, names, cache_dir, identity)
    identity["cache"] = str(cache)
    if cache.is_file():
        features = load_feature_cache(cache, names)
        print(f"Reused {len(names)} embeddings from {cache}", flush=True)
    else:
        from transformers import CLIPModel, CLIPProcessor

        kwargs = {"revision": config["clip_revision"]}
        print(f"Loading {config['clip_model']} on {device}", flush=True)
        model = CLIPModel.from_pretrained(config["clip_model"], low_cpu_mem_usage=True, **kwargs).to(device).eval()
        processor = CLIPProcessor.from_pretrained(config["clip_model"], **kwargs)
        batches = []
        batch_size = config["batch_size"]
        with torch.inference_mode():
            for start in range(0, len(names), batch_size):
                images = []
                for name in names[start:start + batch_size]:
                    with Image.open(image_path(root, name)) as image:
                        images.append(image.convert("RGB"))
                output = model.get_image_features(**processor(images=images, return_tensors="pt").to(device))
                batches.append(torch.nn.functional.normalize(output, dim=-1).cpu().numpy())
                done = min(start + batch_size, len(names))
                if start == 0 or (start // batch_size + 1) % 10 == 0 or done == len(names):
                    print(f"Embedded {done}/{len(names)} images", flush=True)
        features = np.concatenate(batches)
        cache_dir.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=cache_dir) as scratch:
            staged = Path(scratch) / cache.name
            np.savez(staged, names=np.array(names), features=features)
            staged.replace(cache)
    return dict(zip(names, features)), identity


def design_matrix(images, langs, embeddings, languages, *, interactions=True):
    """Sparse [image, language one-hot], with the official outer product by default."""
    image = np.stack([embeddings[name] for name in images]).astype(np.float64, copy=False)
    width = image.shape[1]
    language_index = {language: i for i, language in enumerate(languages)}
    indices = np.array([language_index.get(language, -1) for language in langs])
    rows = np.flatnonzero(indices >= 0)
    columns = indices[rows]
    onehot = sparse.csr_matrix((np.ones(len(rows)), (rows, columns)), shape=(len(image), len(languages)))
    blocks = [sparse.csr_matrix(image), onehot]
    if not interactions:
        return sparse.hstack(blocks, format="csr")
    interaction = sparse.csr_matrix(
        (image[rows].ravel(), (np.repeat(rows, width), (columns[:, None] * width + np.arange(width)).ravel())),
        shape=(len(image), len(languages) * width),
    )
    return sparse.hstack([*blocks, interaction], format="csr")


def fit_lr(matrix, train, seed):
    # This is the same positive block followed by negative block as upstream.
    weights = np.r_[train["k"], train["n"] - train["k"]]
    labels = np.repeat([1, 0], len(train))
    # Rows with weight zero contribute nothing to the likelihood. Discard them
    # before stacking to keep peak RAM down on paintings with unanimous votes.
    fit_matrix = sparse.vstack([matrix[train["k"].to_numpy() > 0],
                                matrix[(train["n"] - train["k"]).to_numpy() > 0]], format="csr")
    with threadpool_limits(limits=4):
        return LogisticRegression(C=C, max_iter=MAX_ITER, solver="lbfgs", random_state=seed).fit(
            fit_matrix, labels[weights > 0], sample_weight=weights[weights > 0])


def language_prior(train, dev):
    """Starter-kit baseline: mean eligible pair share per language, fallback 0.7."""
    means = train.groupby("language")["p"].mean()
    return dev[KEYS].assign(p_positive=dev["language"].map(means).fillna(0.7).round(6))
