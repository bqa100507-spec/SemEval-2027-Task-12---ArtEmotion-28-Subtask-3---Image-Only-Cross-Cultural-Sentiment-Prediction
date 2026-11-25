#!/usr/bin/env python3
"""Train and publish the three EXP-037 stages or replay a sealed image-only candidate."""

import argparse
import math
import os
import random
import shutil
import sys
import time
from importlib.metadata import version
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from run_exp036 import restore_rng, rng_state
from st3 import data, submission
from st3.contrast_study import new_output
from st3.e50_inference import read_input, safe_image_path
from st3.exp036 import rounded
from st3.exp036_io import check, export, read, relative, restore, sha, write
from st3.exp036_train import Paintings, amp, budget, panels
from st3.exp037 import STAGES, blends, counts_for, diagnostics, losses
from st3.exp037_io import verify_inputs
from st3.exp037_runtime import BatchMemoryError, TrainingPaintings, optimizer, pinned_model, train_step
from st3.metrics import score

DEVICE = torch.device("cuda:0")
FIELDS = ("sentiment", "emotion", "caption", "total", "fusion_grad", "vision_grad", "qwen_grad", "clipped")


def load_configs():
    return tuple(read(ROOT / "configs" / name)
                 for name in ("exp037.json", "exp036.json", "competition_semantic_vlm.json"))


def predict(model, panel, processor, micro, deadline):
    q = np.zeros(len(panel.pairs))
    model.eval()
    with torch.inference_mode():
        for i in range(0, len(panel.members), micro):
            budget(deadline)
            indices = list(range(i, min(i+micro, len(panel.members))))
            ids, groups = panel.groups(indices)
            pixels, encoded = processor.batch([panel.images[k] for k in indices], DEVICE)
            with amp(DEVICE):
                hidden = model.features(pixels, encoded)
                logits = model.heads.logits(hidden, torch.as_tensor(groups, device=DEVICE),
                                           torch.as_tensor(panel.languages[ids], device=DEVICE))
            q[ids] = logits.float().sigmoid().cpu().numpy()
    return rounded(panel.pairs[data.KEYS].assign(p_positive=q), panel.pairs)


def state_file(directory, filename="state.pt"):
    return torch.load(directory / filename, weights_only=True, map_location="cpu")


def seal(folder, identity):
    write(folder / "seal.json", {"identity": identity, "files": {p.relative_to(folder).as_posix(): sha(p)
          for p in sorted(folder.rglob("*")) if p.is_file() and p.name != "seal.json"}})


def verify_seal(folder, identity=None):
    meta = read(folder / "seal.json")
    if identity is not None and meta["identity"] != identity:
        raise ValueError(f"Stage or runtime identity changed at {folder}")
    check(folder, meta["files"])
    return meta


def save_boundary(root, identity, model, opt, schedule, scaler, boundary):
    name = f"{boundary['stage']}-{boundary['epoch']:02d}-{boundary['step']:05d}"
    folder = root / "checkpoints" / name
    if not folder.exists():
        with new_output(folder) as stage:
            torch.save({"model": model.snapshot(), "optimizer": opt.state_dict(), "scheduler": schedule.state_dict(),
                        "scaler": scaler.state_dict(), "rng": rng_state(DEVICE), "boundary": boundary}, stage / "state.pt")
            seal(stage, identity)
    else:
        verify_seal(folder, identity)
    write(root / "latest.json", {"checkpoint": name})
    # The new pointer is already published. Retain only the latest optimizer boundary.
    for old in folder.parent.iterdir():
        if old != folder and (old / "seal.json").is_file():
            shutil.rmtree(old)


def latest(root, identity):
    if not (root / "latest.json").exists():
        return None
    name = relative(read(root / "latest.json")["checkpoint"])
    if len(name.parts) != 1:
        raise ValueError("Unexpected optimizer checkpoint path")
    path = root / "checkpoints" / name
    verify_seal(path, identity)
    return state_file(path)


def exportable(root):
    for p in root.rglob("*"):
        rel = p.relative_to(root)
        if p.is_file() and not any(part.startswith(("tmp", ".")) for part in rel.parts) and not p.name.endswith(".tmp"):
            yield p


def progress(root, digest, name="exp037_progress.zip"):
    export(root.parent / name, root, list(exportable(root)), digest)


def stage_preflight(model, train, processor, config, stage, steps, deadline):
    before = model.snapshot()
    rng = rng_state(DEVICE)
    mutable = {id(p) for p in model.learned().values()}
    protected = [(p, p._version) for p in model.parameters() if id(p) not in mutable]
    order = np.random.default_rng(config["seed"]).permutation(len(train.members))
    width = config["effective_paintings"]
    trials = []
    model.train()
    for micro in config["microbatches"]:
        budget(deadline)
        opt, _ = optimizer(model, config, stage, steps)
        scaler = torch.amp.GradScaler("cuda", init_scale=128.)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        start = time.monotonic()
        error, first, second, changes, scales = None, None, None, {}, {}
        duration = None
        try:
            first = train_step(model, train, order[:width], processor, micro, config, opt, scaler, DEVICE)
            second = train_step(model, train, order[width:2*width], processor, micro, config, opt, scaler, DEVICE)
            torch.cuda.synchronize()
            duration = time.monotonic() - start
            ids, groups = train.groups(order[:1])
            pixels, encoded = processor.batch([train.images[order[0]]], DEVICE)
            with amp(DEVICE):
                features = model.features(pixels, encoded)
                terms = losses(model.heads, features, train.targets(ids, groups, DEVICE),
                               float(train.n[ids].sum()), max(1., float((train.mask[ids]*train.n[ids]).sum())))
            shared = (model.heads.visual_projection.weight, model.heads.qwen_projection.weight)
            for i, (label, weight) in enumerate(zip(FIELDS[:3], (1., config["emotion_lambda"],
                                                             config["caption_lambda"]), strict=True)):
                grad = torch.autograd.grad(terms[i]*weight, shared, retain_graph=i<2)
                scales[label] = math.sqrt(sum(g.float().square().sum().item() for g in grad))
            del features, terms, pixels, encoded, grad
            if not all(math.isfinite(v) for v in scales.values()):
                raise FloatingPointError("Nonfinite component gradient diagnostic")
            for group, parameters in model.groups().items():
                ids = {id(p) for p in parameters}
                changes[group] = math.sqrt(sum((p.detach().cpu()-before[n]).square().sum().item()
                                               for n,p in model.learned().items() if id(p) in ids))
                if f"{group}_lr" in config["stages"][stage]:
                    index = FIELDS.index(f"{group}_grad")
                    if changes[group] <= 0 or max(first[index], second[index]) <= 0:
                        raise RuntimeError(f"Preflight did not update the {group} parameters")
                elif changes[group] != 0 or any(p.grad is not None for p in parameters):
                    raise RuntimeError(f"Preflight updated frozen {group} parameters")
            if any(p.grad is not None or p._version != version for p, version in protected):
                raise RuntimeError("A pretrained backbone parameter changed")
        except (BatchMemoryError, torch.cuda.OutOfMemoryError):
            error = "CUDA OOM"
        peak = torch.cuda.max_memory_reserved()
        total = torch.cuda.get_device_properties(DEVICE).total_memory
        if peak + config["memory_margin_mib"]*2**20 > total:
            error = "Reserved memory exceeded the headroom check"
        elapsed = time.monotonic() - start
        trials.append({"microbatch": micro, "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                       "peak_reserved_bytes": peak, "seconds": elapsed, "timed_step_seconds": duration, "failure": error,
                       "probe_first": dict(zip(FIELDS, first)) if first else None,
                       "probe_second": dict(zip(FIELDS, second)) if second else None,
                       "weighted_projection_gradient_l2": scales, "parameter_update_l2": changes})
        del opt, scaler
        model.restore(before)
        model.zero_grad(set_to_none=True)
        restore_rng(rng, DEVICE)
        torch.cuda.empty_cache()
        if error is None:
            rate = 2*width/duration
            return {"stage": stage, "trials": trials, "microbatch": micro, "paintings_per_second": rate,
                    "estimated_fit_seconds": len(train.members)*config["stages"][stage]["epochs"]/rate,
                    "trainable": sum(p.numel() for p in model.parameters() if p.requires_grad)}
    return {"stage": stage, "trials": trials, "microbatch": None}


def choose(rows):
    return max(rows, key=lambda r: (r["metrics"]["full"]["cultural_contrast"],
                                    -r["metrics"]["full"]["log_loss"], -r["epoch"]))


def publish_stage(folder, identity, model, development, processor, incumbent, gold, config, visual, qwen,
                  languages, deadline):
    history = [read(p) for p in sorted(folder.glob("epoch_*/metrics.json"))]
    selected = choose(history)
    selected_dir = folder / f"epoch_{selected['epoch']:02d}"
    verify_seal(selected_dir, identity)
    model.restore(state_file(selected_dir))
    prediction = predict(model, development, processor, 1, deadline)
    original = pd.read_csv(selected_dir / "predictions.csv", float_precision="round_trip")
    pd.testing.assert_frame_equal(prediction, original, check_exact=True)
    dev = development.pairs
    table, winner, ensemble = blends(gold, dev, incumbent, prediction, config)
    stage = folder.name
    with new_output(folder / "selected") as final:
        torch.save(model.snapshot(inference=True), final / "inference.pt")
        write(final / "bundle.json", {"stage": stage, "epoch": selected["epoch"], "languages": languages,
              "microbatch": 1, "model": config, "visual": visual, "qwen": qwen,
              "blend_alpha": winner["alpha"], "parent": "EXP-036 frozen 75/25 incumbent",
              "runtime": identity["runtime"], "identity": identity})
        write(final / "summary.json", {"standalone": selected, "grid": table, "ensemble": winner})
        for name, frame in (("standalone", prediction), ("ensemble", ensemble)):
            dest = final / name
            dest.mkdir()
            csv = dest / "predictions.csv"
            frame.to_csv(csv,index=False,lineterminator="\n",float_format="%.6f")
            archive = final / f"exp037{stage.lower()}_{name}_submission.zip"
            submission.pack(csv, archive, dev)
            pd.testing.assert_frame_equal(submission.validate_archive(archive, dev), frame, check_exact=True)
        seal(final, identity)
    print(f"Stage {stage} complete; selected epoch {selected['epoch']}, alpha {winner['alpha']}.", flush=True)


def train_stage(root, stage, model, train, development, processor, incumbent, gold, config, visual, qwen,
                languages, identity, saved, deadline):
    folder = root / stage
    folder.mkdir(exist_ok=True)
    model.stage(stage)
    steps = math.ceil(len(train.members)/config["effective_paintings"])
    resume = saved if saved is not None and saved["boundary"]["stage"] == stage else None
    if resume:
        probe = read(folder / "preflight.json")
    else:
        torch.manual_seed(config["seed"] + 100 * STAGES.index(stage))
        probe = stage_preflight(model, train, processor, config, stage, steps, deadline)
        write(folder / "preflight.json", probe)
        print(f"Stage {stage} preflight {probe}", flush=True)
    micro = probe["microbatch"]
    if micro is None:
        raise RuntimeError(f"Stage {stage} did not fit microbatch 1. See preflight.json.")
    opt, schedule = optimizer(model, config, stage, steps)
    scaler = torch.amp.GradScaler("cuda",init_scale=128.)
    state = {"stage": stage, "epoch": 1, "step": 0, "sums": [0.]*len(FIELDS), "fit_seconds": 0., "micro": micro}
    if resume:
        state = resume["boundary"]
        model.restore(resume["model"])
        opt.load_state_dict(resume["optimizer"])
        schedule.load_state_dict(resume["scheduler"])
        scaler.load_state_dict(resume["scaler"])
        restore_rng(resume["rng"], DEVICE)
    while state["epoch"] <= config["stages"][stage]["epochs"]:
        epoch = state["epoch"]
        order = np.random.default_rng(config["seed"]+100*STAGES.index(stage)+epoch).permutation(len(train.members))
        width = config["effective_paintings"]
        torch.cuda.reset_peak_memory_stats()
        model.train()
        for i in range(state["step"], steps):
            if time.time() >= deadline:
                save_boundary(root, identity, model, opt, schedule, scaler, state)
                budget(deadline)
            tick = time.monotonic()
            ids = order[i*width:(i+1)*width]
            random_state = rng_state(DEVICE)
            while True:
                micro = state["micro"]
                try:
                    terms = train_step(model, train, ids, processor, micro, config, opt, scaler, DEVICE)
                    break
                except BatchMemoryError:
                    remaining = [m for m in config["microbatches"] if m < micro]
                    if not remaining:
                        raise
                # Retry the same effective batch before any update, with the original RNG.
                opt.zero_grad(set_to_none=True)
                scaler = torch.amp.GradScaler("cuda", init_scale=scaler.get_scale())
                restore_rng(random_state, DEVICE)
                torch.cuda.empty_cache()
                state["micro"] = remaining[0]
                print(f"Stage {stage} CUDA OOM; retrying the optimizer batch at microbatch {state['micro']}", flush=True)
            schedule.step()
            state["fit_seconds"] += time.monotonic()-tick
            state["sums"] = [a+b for a,b in zip(state["sums"], terms, strict=True)]
            state["step"] = i+1
            if state["step"] % config["checkpoint_steps"] == 0 or state["step"] == steps:
                save_boundary(root, identity, model, opt, schedule, scaler, state)
                print(f"Stage {stage} epoch {epoch}/{config['stages'][stage]['epochs']} step {i+1}/{steps}", flush=True)
            if state["step"] % config["progress_steps"] == 0:
                progress(root, identity["handoff"])
        epoch_dir = folder / f"epoch_{epoch:02d}"
        if epoch_dir.exists():
            verify_seal(epoch_dir, identity)
        else:
            validation_start = time.monotonic()
            prediction = predict(model, development, processor, 1, deadline)
            info = {"epoch": epoch, "train": dict(zip(FIELDS, [x/steps for x in state["sums"]])),
                    "fit_seconds": state["fit_seconds"], "validation_seconds": time.monotonic()-validation_start,
                    "microbatch": state["micro"], "train_paintings_per_second":
                    len(train.members)/state["fit_seconds"], "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                    "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
                    "metrics": diagnostics(gold, prediction, config["test_like_excluded"])}
            with new_output(epoch_dir) as saved_epoch:
                prediction.to_csv(saved_epoch / "predictions.csv", index=False, lineterminator="\n", float_format="%.6f")
                write(saved_epoch / "metrics.json", info)
                torch.save(model.snapshot(), saved_epoch / "state.pt")
                seal(saved_epoch, identity)
            print(f"Stage {stage} epoch {epoch} Development {info['metrics']['full']['cultural_contrast']:.6f}", flush=True)
        state = state | {"epoch": epoch+1, "step": 0, "sums": [0.]*len(FIELDS), "fit_seconds": 0.}
        save_boundary(root, identity, model, opt, schedule, scaler, state)
        progress(root, identity["handoff"])
    publish_stage(folder, identity, model, development, processor, incumbent, gold, config, visual, qwen,
                  languages, deadline)
    progress(root, identity["handoff"])


def run(args):
    config, visual, qwen = load_configs()
    if torch.cuda.device_count() != 1:
        raise RuntimeError("EXP-037 needs one visible CUDA GPU")
    args.output.mkdir(parents=True, exist_ok=True)
    root = args.output
    if args.resume:
        restore(args.resume, root, args.handoff)
    manifest = verify_inputs(args.inputs, args.handoff)
    identity = {"handoff": args.handoff, "source": manifest["source"],
                "gpu": torch.cuda.get_device_name(), "python": sys.version.split()[0],
                "runtime": {name: version(name) for name in [*qwen["runtime"], "numpy", "pandas",
                                                              "scipy", "safetensors", "sentencepiece"]},
                "cuda": torch.version.cuda}
    if (root / "run.json").is_file() and read(root / "run.json") != identity:
        raise ValueError("Resume requires the original handoff and runtime/GPU type")
    write(root / "run.json", identity)
    write(root / "config.json", {"model": config, "visual": visual, "qwen": qwen})
    for name in ("requirements-exp035-kaggle.lock.txt","requirements-exp036-kaggle.txt"):
        shutil.copyfile(ROOT / name, root / name)
    write(root / "status.json", {"state": "IN_PROGRESS"})
    try:
        torch.manual_seed(config["seed"])
        np.random.seed(config["seed"])
        random.seed(config["seed"])
        pairs, dev, _ = panels(args.inputs, visual)
        annotations = pd.read_csv(args.inputs / "training/annotations.csv", dtype=str, keep_default_na=False)
        counts = counts_for(annotations, pairs)
        del annotations
        with np.load(args.inputs / "training/targets.npz", allow_pickle=False) as arrays:
            values, mask = arrays["values"], arrays["mask"]
        if values.shape != (len(pairs),visual["hidden_size"]) or mask.shape != (len(pairs),) or not np.isfinite(values).all():
            raise ValueError("The sealed TRAIN-only EXP-036 caption targets differ")
        gold = pd.read_csv(args.inputs / "scoring/dev_gold.csv")
        incumbent = submission.read_predictions(args.inputs / "incumbent/predictions.csv", dev)
        baseline = score(gold, incumbent)
        if any(not math.isclose(baseline[key], value, abs_tol=1e-12)
               for key,value in manifest["incumbent_scores"].items()):
            raise ValueError("EXP-036 incumbent replay differs")
        write(root / "incumbent.json", diagnostics(gold, incumbent, config["test_like_excluded"]))
        languages = {lang:i for i,lang in enumerate(sorted(pairs.language.unique()))}
        train = TrainingPaintings(pairs, languages, counts, values, mask)
        development = Paintings(dev, languages)
        model, processor, hardware = pinned_model(visual,qwen,config,args.siglip,args.qwen,languages,
                            auxiliary=True,root=args.inputs,images=manifest["images"])
        write(root / "hardware.json", hardware)
        saved = latest(root, identity)
        for stage in STAGES:
            completed = root / stage / "selected"
            if completed.exists():
                verify_seal(completed, identity)
            else:
                previous = STAGES.index(stage) - 1
                if previous >= 0:
                    parent = root / STAGES[previous]
                    selected = read(parent / "selected/summary.json")["standalone"]["epoch"]
                    parent_epoch = parent / f"epoch_{selected:02d}"
                    verify_seal(parent_epoch, identity)
                    model.restore(state_file(parent_epoch))
                train_stage(root, stage, model, train, development, processor, incumbent, gold, config, visual,
                            qwen, languages, identity, saved, args.deadline)
        candidates = []
        for stage in STAGES:
            summary = read(root / stage / "selected/summary.json")
            for kind, value, alpha in (("standalone", summary["standalone"]["metrics"], 1.),
                                       ("ensemble", summary["ensemble"],summary["ensemble"]["alpha"])):
                candidates.append({"stage":stage,"kind":kind,"alpha":alpha,"metrics":value["full"],
                                   "epoch":summary["standalone"]["epoch"]})
        best = max(candidates,key=lambda x:(x["metrics"]["cultural_contrast"],-x["metrics"]["log_loss"]))
        shutil.copyfile(root / best["stage"] / "selected" /
                       f"exp037{best['stage'].lower()}_{best['kind']}_submission.zip", root / "exp037_best_submission.zip")
        submission.validate_archive(root / "exp037_best_submission.zip",dev)
        write(root / "selection.json", {"winner":best,"candidates":candidates,
              "experimental_only": config["eligibility"], "best_sha256":sha(root / "exp037_best_submission.zip")})
        write(root / "status.json", {"state":"COMPLETE"})
    except (Exception, KeyboardInterrupt) as error:
        write(root / "status.json", {"state":"RESUMABLE_STOP" if isinstance(error,TimeoutError) else "FAILED",
                                     "reason":f"{type(error).__name__}: {error}"})
        raise
    finally:
        if args.log.is_file():
            shutil.copyfile(args.log, root / "worker.log")
        try:
            progress(root, args.handoff, "exp037_results.zip")
        except Exception as error:
            write(root / "status.json", {"state":"EXPORT_FAILED","reason":str(error)})
            raise


def infer(args):
    verify_seal(args.bundle)
    meta = read(args.bundle / "bundle.json")
    inputs = read_input(args.input, read(ROOT / "configs/e50_inference.json")["official_languages"])
    for image in inputs.image.unique():
        safe_image_path(args.images,image)
    model, processor, _ = pinned_model(meta["visual"],meta["qwen"],meta["model"],args.siglip,args.qwen,meta["languages"],
                                        auxiliary=False,root=args.images,images=None)
    model.stage(meta["stage"])
    model.restore(state_file(args.bundle,"inference.pt"), inference=True)
    pred = predict(model,Paintings(inputs,meta["languages"]),processor,meta["microbatch"],float("inf"))
    with new_output(args.output) as out:
        versions = {"standalone":pred}
        if args.incumbent:
            parent = submission.validate_archive(args.incumbent, inputs)
            w = meta["blend_alpha"]
            parent = rounded(parent,inputs)
            versions["ensemble"] = pred.assign(p_positive=((1-w)*parent.p_positive+w*pred.p_positive).round(6))
        for kind, frame in versions.items():
            sub = out / kind
            sub.mkdir()
            csv = sub / "predictions.csv"
            frame.to_csv(csv,index=False,lineterminator="\n",float_format="%.6f")
            submission.pack(csv,out / f"exp037{meta['stage'].lower()}_{kind}_submission.zip",inputs)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    choices = p.add_subparsers(dest="mode",required=True)
    t = choices.add_parser("train")
    i = choices.add_parser("infer")
    for field in ("inputs","output","log"):
        t.add_argument(f"--{field}",type=Path,required=True)
    t.add_argument("--resume",type=Path)
    t.add_argument("--handoff",required=True)
    t.add_argument("--deadline",type=float,required=True)
    for field in ("bundle","input","images","output"):
        i.add_argument(f"--{field}",type=Path,required=True)
    i.add_argument("--incumbent",type=Path)
    for command in (t,i):
        command.add_argument("--siglip",type=Path)
        command.add_argument("--qwen",type=Path)
    args = p.parse_args()
    (run if args.mode == "train" else infer)(args)
