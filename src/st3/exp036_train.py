"""Training-only caption targets and painting batches for SigLIP2 adaptation."""

import hashlib
import time
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.nn import functional as F

from . import data
from .exp036 import centered_targets
from .exp036_io import check, read, sha, write


def budget(deadline):
    if time.time() >= deadline:
        raise TimeoutError("Session time reached; use the exported resume archive")


def model_assets(cfg, directory, *, text_assets=True):
    weights = {name: value for name, value in cfg["weights"].items()
               if text_assets or name == "model.safetensors"}
    metadata = {name: value for name, value in cfg["metadata_git_blobs"].items()
                if text_assets or name in {"config.json", "preprocessor_config.json"}}
    names = [*weights, *metadata]
    if directory is None:
        from huggingface_hub import snapshot_download
        directory = snapshot_download(cfg["model"], revision=cfg["revision"], allow_patterns=names, token=False)
    directory = Path(directory)
    check(directory, weights)
    for name, expected in metadata.items():
        content = (directory / name).read_bytes()
        header = f"blob {len(content)}\0".encode()
        if hashlib.sha1(header + content, usedforsecurity=False).hexdigest() != expected:
            raise ValueError(f"Pinned SigLIP2 metadata differs: {name}")
    return directory


def backbone(directory, cfg, *, text):
    from transformers import SiglipTextModel, SiglipVisionModel

    cls = SiglipTextModel if text else SiglipVisionModel
    model, info = cls.from_pretrained(directory, dtype=torch.float16, attn_implementation="sdpa",
                                     use_safetensors=True, local_files_only=True, output_loading_info=True)
    if any(info.get(key) for key in ("missing_keys", "mismatched_keys", "error_msgs")):
        raise ValueError(f"SigLIP2 pretrained weight loading failed: {info}")
    if (model.config.hidden_size, model.config.num_hidden_layers) != (cfg["hidden_size"], cfg["layers"]):
        raise ValueError("Unexpected SigLIP2 configuration")
    if not text and (model.config.image_size, model.config.patch_size) != (cfg["image_size"], cfg["patch_size"]):
        raise ValueError("Unexpected vision size")
    return model.eval().requires_grad_(False)


def amp(device):
    return torch.autocast("cuda", dtype=torch.float16) if device.type == "cuda" else nullcontext()


def panels(inputs, cfg):
    columns = [*data.ANNOTATION_COLUMNS, "caption"]
    ann = pd.read_csv(inputs / "training/annotations.csv", dtype=str, keep_default_na=False)
    if list(ann) != columns or set(ann.split) != {"train"}:
        raise ValueError("Caption targets require an exact training-only annotation file")
    pairs = data.training_pairs(ann).sort_values(data.KEYS, ignore_index=True)
    dev = data.read_dev_input(inputs)
    if (len(pairs), pairs.painting.nunique(), len(dev)) != (cfg["train_pairs"], cfg["train_paintings"], cfg["dev_pairs"]):
        raise ValueError("Training or Development coverage changed")
    if set(pairs.painting) & set(dev.painting) or set(pairs.image) & set(dev.image):
        raise ValueError("Training data overlaps Development")
    mapped = ann.emotion.isin(data.POSITIVE | data.NEGATIVE)
    valid = ann.caption.str.strip().ne("") & ~ann.caption.str.contains("\x00", regex=False)
    eligible = ann.loc[mapped & valid, [*data.KEYS, "caption"]]
    grouped = pairs[data.KEYS].assign(pair=np.arange(len(pairs))).merge(eligible, on=data.KEYS, validate="one_to_many")
    return pairs, dev, grouped


def text_targets(root, directory, pairs, caption_rows, cfg, device, deadline):
    folder = root / "caption_cache"
    folder.mkdir(exist_ok=True)
    target, seal = root / "targets.npz", root / "targets.json"
    if seal.exists():
        record = read(seal)
        check(root, {"targets.npz": record["sha256"]})
        with np.load(target, allow_pickle=False) as saved:
            return saved["values"], saved["mask"]

    from transformers import AutoTokenizer
    ids, texts = pd.factorize(caption_rows.caption, sort=True)
    counts = np.bincount(caption_rows.pair, minlength=len(pairs))
    embedding = np.zeros((len(texts), cfg["hidden_size"]), dtype=np.float16)
    teacher = tokenizer = None
    started = time.monotonic()
    for start in range(0, len(texts), cfg["caption_chunk"]):
        stop = min(start + cfg["caption_chunk"], len(texts))
        block, receipt = folder / f"{start}.npy", folder / f"{start}.json"
        if receipt.exists():
            check(folder, {block.name: read(receipt)["sha256"]})
            embedding[start:stop] = np.load(block, allow_pickle=False)
            continue
        if teacher is None:
            teacher = backbone(directory, cfg, text=True).to(device)
            tokenizer = AutoTokenizer.from_pretrained(directory, local_files_only=True, use_fast=True)
        values = []
        with torch.inference_mode():
            for i in range(start, stop, cfg["caption_batch"]):
                budget(deadline)
                tokens = tokenizer(texts[i:min(i+cfg["caption_batch"], stop)].tolist(), padding="max_length",
                                   truncation=True, max_length=teacher.config.max_position_embeddings,
                                   return_tensors="pt").to(device)
                with amp(device):
                    vector = teacher(**tokens).pooler_output
                values.append(F.normalize(vector.float(), dim=-1).half().cpu().numpy())
        vectors = np.concatenate(values)
        if not np.isfinite(vectors).all():
            raise FloatingPointError("Nonfinite caption embedding")
        scratch = block.with_suffix(".tmp.npy")
        np.save(scratch, vectors, allow_pickle=False)
        scratch.replace(block)
        write(receipt, {"sha256": sha(block)})
        embedding[start:stop] = vectors
        print(f"Caption teacher {stop}/{len(texts)} distinct captions", flush=True)
    del teacher, tokenizer
    torch.cuda.empty_cache()
    total = np.zeros((len(pairs), cfg["hidden_size"]), dtype=np.float32)
    pair = caption_rows.pair.to_numpy()
    for start in range(0, len(ids), cfg["caption_chunk"]):
        end = start + cfg["caption_chunk"]
        np.add.at(total, pair[start:end], embedding[ids[start:end]])
    values, mask, summary = centered_targets(total, counts, pd.factorize(pairs.painting)[0],
                                            pd.factorize(pairs.language)[0], cfg["caption_epsilon"])
    if not mask.any():
        raise ValueError("All centered caption targets are degenerate")
    scratch = target.with_suffix(".tmp.npz")
    np.savez(scratch, values=values, mask=mask)
    scratch.replace(target)
    write(seal, summary | {"caption_rows": len(ids), "distinct_captions": len(texts),
                          "elapsed_seconds": time.monotonic() - started, "sha256": sha(target)})
    return values, mask


class Paintings:
    def __init__(self, pairs, languages, *, values=None, mask=None):
        self.pairs = pairs
        self.members = [rows.to_numpy() for rows in pairs.groupby("painting", sort=True).groups.values()]
        self.images = [pairs.image.iloc[ix[0]] for ix in self.members]
        self.languages = pairs.language.map(languages).fillna(-1).to_numpy(dtype=np.int64)
        if values is not None:
            self.values = values
            self.mask = mask
            self.n = pairs.n.to_numpy(dtype=np.float32)
            self.k = pairs.k.to_numpy(dtype=np.float32)

    def groups(self, paintings):
        ids = [self.members[i] for i in paintings]
        return np.concatenate(ids), np.repeat(np.arange(len(ids)), [len(a) for a in ids])

    def batch(self, ids, groups, device):
        arrays = (groups, self.languages[ids], self.k[ids], self.n[ids],
                  self.values[ids], self.mask[ids] * self.n[ids])
        return tuple(torch.as_tensor(values, device=device) for values in arrays)

    def pixels(self, ids, root, processor, device):
        pictures = []
        for i in ids:
            with Image.open(data.image_path(root, self.images[i])) as image:
                pictures.append(image.convert("RGB"))
        return processor(images=pictures, return_tensors="pt").pixel_values.to(device)
