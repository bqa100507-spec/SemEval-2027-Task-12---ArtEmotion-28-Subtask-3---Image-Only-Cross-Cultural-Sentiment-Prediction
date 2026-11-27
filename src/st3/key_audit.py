"""CPU checkpoint key audit (pre-upload checklist item 8): GPU mocks cannot catch checkpoint/class mismatches.

Compare a pinned checkpoint's weight names with the state-dict names of the exact model class the worker
instantiates, built on the ``meta`` device (no weights, no GPU). The only accepted difference is
``lm_head.weight`` missing from the checkpoint: always when the config ties it to the input embeddings
(transformers fills it), and for an untied config only as the documented embedding-model allowance, because
the embedding path reads hidden states and never calls ``lm_head``.
"""

import json
import struct
from pathlib import Path

HEAD = "lm_head.weight"


def tied(config):
    text = config.get("text_config") or {}
    return bool(config.get("tie_word_embeddings")) or bool(text.get("tie_word_embeddings"))


def audit(checkpoint_keys, class_keys, config, tied_keys=(HEAD,)):
    checkpoint, expected = set(checkpoint_keys), set(class_keys)
    missing, unexpected = sorted(expected - checkpoint), sorted(checkpoint - expected)
    is_tied = tied(config)
    tied_missing = [k for k in missing if is_tied and k in tied_keys]
    allowed = [k for k in missing if not is_tied and k == HEAD]
    remaining = sorted(set(missing) - set(tied_missing) - set(allowed))
    status = "PASS" if not remaining and not unexpected else "FAIL"
    return {"status": status, "checkpoint_keys": len(checkpoint), "class_keys": len(expected),
            "tie_word_embeddings": is_tied, "tied_missing": tied_missing, "ignored_missing_keys": allowed,
            "missing": remaining, "unexpected": unexpected,
            "rule": "tied lm_head is filled from the embeddings; an untied lm_head may be absent from an embedding "
                    "checkpoint because the embedding path never calls it"}


def loader_ignored(info, config):
    """from_pretrained loading info: allow only an absent untied lm_head; anything else is a hard failure."""
    missing = list(info.get("missing_keys") or [])
    ignored = [HEAD] if missing == [HEAD] and not tied(config) else []
    if set(missing) - set(ignored) or any(info.get(k) for k in ("unexpected_keys", "mismatched_keys", "error_msgs")):
        raise ValueError(f"Qwen3-VL-Embedding loading failed: {info}")
    return ignored, missing


def forbid_head(model):
    """The embedding path reads hidden states; any call into lm_head is a bug."""
    def forbid(*_):
        raise RuntimeError("lm_head must not be used by the embedding path")

    return model.lm_head.register_forward_pre_hook(forbid)


def require(receipt, label):
    if receipt["status"] != "PASS":
        raise ValueError(f"{label} checkpoint/class key mismatch: missing {receipt['missing'][:5]}, "
                         f"unexpected {receipt['unexpected'][:5]}")
    return receipt


def qwen_class_keys(config):
    """State-dict names of Qwen3VLForConditionalGeneration, built on the meta device (transformers 4.57)."""
    import torch
    from transformers import Qwen3VLConfig, Qwen3VLForConditionalGeneration

    with torch.device("meta"):
        model = Qwen3VLForConditionalGeneration._from_config(Qwen3VLConfig.from_dict(config))
    if getattr(model, "_checkpoint_conversion_mapping", None):
        raise ValueError("Unexpected checkpoint key conversion mapping; extend the audit before use")
    return sorted(model.state_dict()), tuple(getattr(model, "_tied_weights_keys", None) or ())


def snapshot_keys(directory):
    """Weight names of a downloaded snapshot: the shard index, or the single safetensors header."""
    directory = Path(directory)
    index = directory / "model.safetensors.index.json"
    if index.is_file():
        return sorted(json.loads(index.read_text(encoding="utf-8"))["weight_map"])
    with (directory / "model.safetensors").open("rb") as stream:
        size = struct.unpack("<Q", stream.read(8))[0]
        header = json.loads(stream.read(size))
    return sorted(k for k in header if k != "__metadata__")


def remote_keys(model, revision, sharded, opener=None):
    """Weight names from the Hub at a pinned revision, without downloading weights (index or header range)."""
    import urllib.request

    opener = opener or urllib.request.urlopen
    base = f"https://huggingface.co/{model}/resolve/{revision}/"

    def get(name, rng=None):
        request = urllib.request.Request(base + name, headers={"Range": rng} if rng else {})
        with opener(request, timeout=60) as response:
            return response.read()

    if sharded:
        return sorted(json.loads(get("model.safetensors.index.json"))["weight_map"])
    size = struct.unpack("<Q", get("model.safetensors", "bytes=0-7"))[0]
    header = json.loads(get("model.safetensors", f"bytes=8-{7 + size}"))
    return sorted(k for k in header if k != "__metadata__")
