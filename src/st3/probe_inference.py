"""Inference-only conditioning interventions; never constructs an optimizer."""
from contextlib import nullcontext
import json
from pathlib import Path
import time

import pandas as pd
from PIL import Image
import torch

from .interaction_probe import MODELS, validate_rows, analyze
from .vlm_semantics import atomic_json, file_hash, fingerprint
from .vlm_supervised import adapters_disabled, DecoderLoRA, make_head
from .vlm_supervised_training import load_state, FeatureCache

SOURCE_FILES = [
    "configs/task031.json", "configs/task031_manifest.json", "src/st3/interaction_probe.py",
    "src/st3/probe_inference.py", "scripts/run_task031.py", "scripts/verify_task031.py",
    "scripts/run_exp026.py", "src/st3/vlm_supervised.py", "src/st3/vlm_supervised_training.py",
    "src/st3/vlm_semantics.py", "requirements-exp026.lock.txt",
    "docs/exp027_closure_evidence.json", "docs/task031_protocol.md",
    "configs/exp026.json", "configs/exp027.json", "src/st3/caption_data.py", "src/st3/data.py",
]


def load_saved(backend, head, state):
    """Require exactly the existing decoder adapter/head topology before copying."""
    expected = {name + "." + part: value for name, module in backend.model.named_modules()
                if isinstance(module, DecoderLoRA) for part, value in (("A", module.A), ("B", module.B))}
    if set(expected) != set(state["adapters"]):
        raise ValueError("Saved decoder adapter names differ from pinned model")
    if any(expected[k].shape != v.shape or not torch.isfinite(v).all() for k,v in state["adapters"].items()):
        raise ValueError("Saved decoder adapter shape/value mismatch")
    if list(head.languages) != state["languages"]:
        raise ValueError("Saved sentiment-head language order differs")
    if any(not torch.isfinite(v).all() for v in state["head"].values()):
        raise ValueError("Nonfinite saved sentiment head")
    load_state(backend, head, state)
    backend.model.requires_grad_(False); backend.model.eval()
    head.requires_grad_(False); head.eval()
    if any(p.requires_grad for p in list(backend.model.parameters()) + list(head.parameters())):
        raise ValueError("Inference model contains trainable parameters")


@torch.inference_mode()
def query_image(backend, head, vision, queries, adapter_state):
    """One actual-image decoder batch; reuse states across explicit head interventions."""
    prompts = [None, *sorted({q["prompt_language"] for q in queries if q["prompt_language"] is not None})]
    context = adapters_disabled(backend.model) if adapter_state == "disabled_same_head_OOD" else nullcontext()
    with context, torch.autocast("cuda", dtype=torch.float16, enabled=torch.device(backend.device).type=="cuda"):
        hidden = backend.hidden([vision] * len(prompts), prompts, train=False)
    if hidden.requires_grad or not torch.isfinite(hidden).all():
        raise ValueError("Invalid inference-only decoder output")
    selected = hidden[[prompts.index(q["prompt_language"]) for q in queries]]
    indices = head.indices([q["head_language"] for q in queries], selected.device)
    if (indices < 0).any():raise ValueError("Unknown intervention head language")
    with torch.autocast("cuda", dtype=torch.float16, enabled=selected.is_cuda):
        logits = head(selected, indices).float()
        values = head.visual(selected.float())
        shared = head.shared(values).squeeze(-1)
        bias = head.language_bias(indices).squeeze(-1)
        interaction = (values * head.language_visual(indices)).sum(-1) / head.scale
    if not torch.equal(logits, shared + (bias + interaction)):
        raise ValueError("Raw authentic head/component reconstruction differs")
    if not torch.isfinite(logits).all():raise ValueError("Nonfinite raw logits")
    return [dict(q, logit=float(z), shared=float(s), language_bias=float(b), head_interaction=float(i))
            for q,z,s,b,i in zip(queries, logits.cpu(), shared.cpu(), bias.cpu(), interaction.cpu())]


def read_batch(path, identity):
    path = Path(path); seal = path.with_suffix(".sha256.json")
    if not path.exists() and not seal.exists():return None
    if not path.exists() or not seal.exists() or json.loads(seal.read_text())["sha256"] != file_hash(path):
        raise ValueError("Incomplete/corrupt probe cache; preserve it and choose a fresh run directory: " + str(path))
    payload = json.loads(path.read_text())
    if payload["identity"] != identity:raise ValueError("Incompatible probe cache identity")
    return payload["rows"]


def write_batch(path, identity, rows):
    atomic_json(path, {"identity":identity, "rows":rows})
    atomic_json(Path(path).with_suffix(".sha256.json"), {"sha256":file_hash(path)})
    if read_batch(path,identity) != rows:raise IOError("Published logit batch read-back failed")


def batch_identity(identity, model, state, painting):
    return dict(run=fingerprint(identity), model=model, adapter_state=state, painting=painting)


def verify_outputs(run_dir, manifest, targets, root):
    """Offline key/hash/analysis replay, explicitly not a second encoder inference."""
    run_dir=Path(run_dir); root=Path(root)
    atomic_json(run_dir/"verification.json", {"status":"VERIFYING", "mechanism_decision":"INVALID_UNTIL_PASS"})
    identity=json.loads((run_dir/"identity.json").read_text())
    if identity["manifest_sha256"] != fingerprint(manifest):raise ValueError("Frozen probe manifest differs")
    for name,digest in identity["source_sha256"].items():
        if file_hash(root/name)!=digest:raise ValueError("Probe source identity differs: "+name)
    if identity["checkpoint_sha256"]!=manifest["checkpoint_sha256"]:raise ValueError("Checkpoint identity differs")
    rows=[]
    for model in MODELS:
        for state in manifest["config"]["adapter_states"]:
            for painting in manifest["images"]:
                part=read_batch(run_dir/"logits"/model/state/(painting+".json"), batch_identity(identity,model,state,painting))
                if part is None:raise ValueError("Incomplete probe coverage")
                expected=[q for q in manifest["queries"] if q["painting"]==painting]
                if len(part)!=len(expected) or any(r["model"]!=model or r["adapter_state"]!=state for r in part):
                    raise ValueError("Misfiled probe batch")
                rows.extend(part)
    validate_rows(rows,manifest)
    diagnostics=json.loads((run_dir/"inference_diagnostics.json").read_text())
    if len(diagnostics["exact_determinism_checks"]) != 6 or not all(diagnostics["exact_determinism_checks"].values()):
        raise ValueError("Missing/failed deterministic inference checks")
    if diagnostics["official_cache_parity"]["status"]!="PASS" or diagnostics["fresh_vision_cache_exact"] is not True:
        raise ValueError("Missing/failed official processor and image cache parity")
    frame,summary=analyze(rows,manifest,targets)
    frame.to_csv(run_dir/"rectangles.csv",index=False,lineterminator="\n")
    atomic_json(run_dir/"analysis.json",summary)
    atomic_json(run_dir/"verification.json", {"status":"PASS", "raw_logits":len(rows),
        "verification_scope":"Offline source, checkpoint identity, cache SHA, keys and metric reconstruction; no GPU replay",
        "files_sha256":{p.name:file_hash(p) for p in (run_dir/"rectangles.csv",run_dir/"analysis.json",run_dir/"inference_diagnostics.json")}})
    return summary


def run_inference(backend, models, config, manifest, inputs, cache_root, run_dir, identity, check):
    inputs=Path(inputs);run_dir=Path(run_dir)
    allowed=pd.DataFrame(list(manifest["images"].values()))
    # Verify real pixel bytes even when an old frozen-vision cache is available.
    for row in allowed.to_dict("records"):
        path=(inputs/"artwork/images"/row["image"]).resolve()
        if not path.is_relative_to((inputs/"artwork/images").resolve()) or file_hash(path)!=row["image_sha256"]:
            raise ValueError("Selected artwork pixels differ")
    cache=FeatureCache(cache_root,inputs,config,allowed)
    first=next(iter(manifest["images"])); image=manifest["images"][first]["image"]
    backend.model.requires_grad_(False);backend.model.eval()
    check()
    with torch.inference_mode(), Image.open(inputs/"artwork/images"/image) as opened:
        vision=cache.vision(image,backend);fresh=backend.vision(opened.convert("RGB"))
        if any(not torch.equal(fresh[k],vision[k]) for k in fresh):raise ValueError("Fresh/cached original image vision tokens differ")
        parity=backend.parity(opened.convert("RGB"),vision)
    diagnostics={"official_cache_parity":parity,"fresh_vision_cache_exact":True,"exact_determinism_checks":{},
                 "batch_seconds":[],"resumed_image_batches":0,"new_image_batches":0}
    for model in MODELS:
        state=models[model]["state"];head=make_head(state["languages"],config,backend.device)
        load_saved(backend,head,state)
        for adapter_state in manifest["config"]["adapter_states"]:
            for painting,meta in manifest["images"].items():
                check();queries=[q for q in manifest["queries"] if q["painting"]==painting]
                path=run_dir/"logits"/model/adapter_state/(painting+".json")
                key=batch_identity(identity,model,adapter_state,painting)
                saved=read_batch(path,key)
                if saved is not None and painting!=first:
                    diagnostics["resumed_image_batches"]+=1;continue
                vision=cache.vision(meta["image"],backend)
                torch.cuda.synchronize();start=time.monotonic()
                rows=query_image(backend,head,vision,queries,adapter_state)
                torch.cuda.synchronize();diagnostics["batch_seconds"].append(time.monotonic()-start)
                if painting==first:
                    repeated=query_image(backend,head,vision,queries,adapter_state)
                    if rows!=repeated:raise ValueError("Repeated inference is not exactly deterministic")
                    diagnostics["exact_determinism_checks"][model+"/"+adapter_state]=True
                rows=[dict(r,model=model,adapter_state=adapter_state) for r in rows]
                if saved is not None:
                    if saved!=rows:raise ValueError("Resumed counterfactual logits fail exact replay")
                    diagnostics["resumed_image_batches"]+=1
                else:
                    write_batch(path,key,rows);diagnostics["new_image_batches"]+=1
                atomic_json(run_dir/"inference_diagnostics.json",diagnostics)
                print(model,adapter_state,painting,"cached" if saved is not None else "published",flush=True)
    return diagnostics
