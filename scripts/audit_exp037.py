"""Replay the sealed EXP-037 public Development artifacts without model training."""

import argparse
import io
import math
import re
import shutil
import subprocess
import sys
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from package_exp037 import PRIOR_RESULTS, SOURCES
from run_exp037 import choose
from st3 import data
from st3.exp036_io import sha, write
from st3.exp036_train import panels
from st3.exp037 import FusionHeads, counts_for, diagnostics
from st3.exp037_closure import (
    checked_metrics, digest, json_member, mixed, prediction_bytes, require, verify_export, zip_prediction_bytes,
)
from st3.exp037_io import safe_members, unpack_inputs
from st3.exp037_runtime import TrainingPaintings

HANDOFF = "026720f406c02236aeaf5bfdc806853816323b66267f81f0e3e2dbbfe1ce7e0e"
INPUT_ZIP = "eaf16b9301cb131fbdf61a1b86fb191d3617b2141e39398f0c3b12bcf57ffce2"
RESULTS = "852154e257a198f575dde6d47475aa80588aa48348b50792d47e34603bacb194"
BEST_ZIP = "b50861a1414fd863b4293bb6a8b3dd336bd6f2bc9071f8e20559b2774f4e1f33"
SOURCE = "38b1e381127c5f4fdc02dfc9845caeba90d9667e"
NOTEBOOK = "8ec141a304a5031ba76fac6c2a9541c696bb6a30"
STAGES = {"A": (4, 4, 0., 2386793), "B": (6, 6, .1, 5366633), "C": (8, 6, .2, 6677353)}
LOCKS = ("requirements-exp035-kaggle.lock.txt", "requirements-exp036-kaggle.txt")


def require_hash(path, expected):
    require(sha(path) == expected, f"Artifact checksum differs: {path}")


def read_tensor(archive, path):
    return torch.load(io.BytesIO(archive.read(path)), weights_only=True, map_location="cpu")


def group(name):
    return name.split(".", 1)[0]


def amount(state, group_name):
    return sum(t.numel() for name, t in state.items() if group(name) == group_name)


def delta(state, reference, names):
    return math.sqrt(sum((state[name]-reference[name]).float().square().sum().item() for name in names))


def parallel_to_previous(value, previous):
    """Residual to uniform scalar decay. Weight decay alone cannot rotate a tensor."""
    a, b = value.flatten().double(), previous.flatten().double()
    return (a - (a @ b / (b @ b).clamp_min(1e-40))*b).norm().item()


def expected_shapes(config, visual, n_languages):
    hidden = visual["hidden_size"]
    with torch.device("meta"):
        heads = FusionHeads(hidden,n_languages,config,auxiliary=True)
    shapes = {"heads."+name:tuple(p.shape) for name,p in heads.named_parameters()}
    start = visual["layers"]-visual["adapter_layers"]
    # Dimensions are from the hash-pinned SigLIP2 and Qwen architecture configs.
    for i in range(start,visual["layers"]):
        for name,(inp,out) in {"self_attn.q_proj":(hidden,hidden), "self_attn.v_proj":(hidden,hidden),
                               "mlp.fc1":(hidden,4304),"mlp.fc2":(4304,hidden)}.items():
            target = f"vision.encoder.layers.{i}.{name}"
            shapes[target+".left"] = (visual["rank"],inp)
            shapes[target+".right"] = (out,visual["rank"])
    for i in range(config["qwen_layers"]-config["qwen_tail"],config["qwen_layers"]):
        for name,out in (("q_proj",4096),("v_proj",1024)):
            target = f"qwen.language_model.layers.{i}.self_attn.{name}"
            shapes[target+".left"] = (config["qwen_rank"],config["qwen_width"])
            shapes[target+".right"] = (out,config["qwen_rank"])
    return shapes


def verify_seal(z, directory, identity, exported):
    seal = json_member(z, directory+"/seal.json")
    require(seal["identity"] == identity, f"Checkpoint identity differs: {directory}")
    prefix = directory+"/"
    require(seal["files"] == {n[len(prefix):]: h for n,h in exported.items()
                              if n.startswith(prefix) and n != prefix+"seal.json"},
            f"Checkpoint seal differs: {directory}")


def git_at(root, *args):
    return subprocess.check_output(["git", "-C", str(root), *args])


def input_provenance(args):
    import tempfile
    require_hash(args.inputs, INPUT_ZIP)
    with ZipFile(args.inputs) as original:
        manifest = verify_export(original, HANDOFF, name="exp037_manifest.json", prefix="exp037_inputs/")
        safe_members(original.namelist())
        require(digest(original.read("exp037_inputs/exp037_manifest.json")) == HANDOFF
                and manifest["source"] == SOURCE, "Handoff fingerprint differs")
        tree = git_at(ROOT, "archive", "--format=zip", SOURCE, *SOURCES)
        with ZipFile(io.BytesIO(tree)) as source:
            runtime = {"runtime/"+n: digest(source.read(n)) for n in source.namelist() if not n.endswith("/")}
        require(runtime == {k:v for k,v in manifest["files"].items() if k.startswith("runtime/")},
                "Packaged worker differs from immutable Git source")
        notebook = git_at(ROOT, "show", f"{NOTEBOOK}:notebooks/EXP037_Kaggle.ipynb")
        require(all(value.encode() in notebook for value in (SOURCE, INPUT_ZIP, HANDOFF)),
                "Pinned notebook does not reference this handoff")
        visual = json_member(original, "exp037_inputs/runtime/configs/exp036.json")
        config = json_member(original, "exp037_inputs/runtime/configs/exp037.json")
        qwen = json_member(original, "exp037_inputs/runtime/configs/competition_semantic_vlm.json")
        with tempfile.TemporaryDirectory() as scratch:
            unpacked = unpack_inputs(args.inputs, Path(scratch), HANDOFF, INPUT_ZIP)
            pairs, dev, _ = panels(unpacked, visual)
            train = pd.read_csv(unpacked / "training/annotations.csv", dtype=str, keep_default_na=False)
            votes = counts_for(train, pairs)
            with np.load(unpacked / "training/targets.npz", allow_pickle=False) as target:
                values, mask = target["values"], target["mask"]
            gold = pd.read_csv(unpacked / "scoring/dev_gold.csv")
            incumbent = prediction_bytes((unpacked / "incumbent/predictions.csv").read_bytes(), dev)
    require_hash(args.upstream / data.ANNOTATIONS, visual["annotations_sha256"])
    require_hash(args.upstream / data.DEV_INPUT, visual["dev_input_sha256"])
    require(git_at(args.upstream, "rev-parse", "HEAD").decode().strip() == manifest["upstream"],
            "Expected the pinned organizer checkout")
    public = data.read_annotations(args.upstream)
    data.check_dev_isolation(public, dev)
    from_public = data.validation_gold(public, dev)[[*data.KEYS, "k", "n"]]
    pd.testing.assert_frame_equal(data.read_dev_input(args.upstream), dev)
    pd.testing.assert_frame_equal(from_public.reset_index(drop=True), gold.reset_index(drop=True))
    pd.testing.assert_frame_equal(data.training_pairs(public).sort_values(data.KEYS, ignore_index=True), pairs)
    require(set(manifest["images"]) == set(pairs.image) | set(dev.image), "Artwork mapping coverage differs")
    require(all(manifest["images"][name] == f"artwork/{manifest['files'][manifest['images'][name]]}.jpg"
                for name in manifest["images"]), "Artwork mapping digests differ")
    require(values.shape == (len(pairs), visual["hidden_size"]) and np.isfinite(values).all()
            and mask.dtype == np.bool_ and int(mask.sum()) == 16001, "Unexpected caption-target matrix")
    norms = np.linalg.norm(values, axis=1)
    require(np.allclose(norms[mask], 1., atol=1e-6) and np.count_nonzero(values[~mask]) == 0,
            "Caption target validity differs")
    languages = {l:i for i,l in enumerate(sorted(pairs.language.unique()))}
    panel = TrainingPaintings(pairs, languages, votes, values, mask)
    ids = np.random.default_rng(config["seed"]).permutation(len(panel.members))[:4]
    samples = [{"painting": panel.pairs.painting.iloc[panel.members[i][0]], "pairs": len(panel.members[i]),
                "valid": int(panel.mask[panel.members[i]].sum()),
                "caption_vote_mass": int((panel.mask[panel.members[i]]*panel.n[panel.members[i]]).sum()),
                "min_target_norm": float(norms[panel.members[i]].min()),
                "max_target_norm": float(norms[panel.members[i]].max())} for i in ids]
    return manifest, config, visual, qwen, panel, dev, gold, incumbent, samples


def check_epochs(z, exported, identity, config, visual, panel, dev, gold):
    steps = math.ceil(len(panel.members)/config["effective_paintings"])
    log = z.read("worker.log").decode("utf-8", errors="replace")
    logged = [(m[1], int(m[2]), int(m[3]), int(m[4]), int(m[5])) for m in
              re.finditer(r"^Stage ([ABC]) epoch (\d+)/(\d+) step (\d+)/(\d+)$", log, re.M)]
    expected = [(stage, e, limit, k, steps) for stage, (limit, *_) in STAGES.items()
                for e in range(1,limit+1) for k in [*range(config["checkpoint_steps"],steps,config["checkpoint_steps"]), steps]]
    require(logged == expected, "Incomplete or out-of-order optimizer boundary logs")
    records, frame_by_epoch, states, gates = [], {}, {}, {}
    prior = None
    shapes = expected_shapes(config,visual,int(panel.pairs.language.nunique()))
    for stage, (limit, selected, alpha, trainable) in STAGES.items():
        probe = json_member(z, f"{stage}/preflight.json")
        require(probe["stage"] == stage and probe["microbatch"] == 4 and probe["trainable"] == trainable,
                f"Invalid preflight for {stage}")
        gate = probe["trials"][-1]
        require(gate["failure"] is None, f"Failed stage {stage} preflight gate")
        expected_trainable = {"A":{"heads"}, "B":{"heads","vision"}, "C":{"heads","vision","qwen"}}[stage]
        updates = gate["parameter_update_l2"]
        expected_probes = {"heads": "fusion", "vision": "vision", "qwen": "qwen"}
        require(all((updates[expected_probes[name]] > 0) == (name in expected_trainable)
                    for name in expected_probes), f"Stage {stage} trainable updates differ")
        gates[stage] = probe
        for epoch in range(1,limit+1):
            candidate = f"{stage}{epoch}"
            path = f"{stage}/epoch_{epoch:02d}"
            verify_seal(z, path, identity, exported)
            prediction = prediction_bytes(z.read(path+"/predictions.csv"), dev)
            recorded = json_member(z, path+"/metrics.json")
            recompute = diagnostics(gold, prediction, config["test_like_excluded"])
            checked_metrics(recorded["metrics"], recompute)
            require(recorded["epoch"] == epoch and recorded["microbatch"] == 4, "Unexpected epoch counter or microbatch")
            train = recorded["train"]
            require(math.isclose(train["total"], train["sentiment"]+config["emotion_lambda"]*train["emotion"]
                                +config["caption_lambda"]*train["caption"], abs_tol=1e-6), "Training loss differs")
            state = read_tensor(z, path+"/state.pt")
            require({n:tuple(t.shape) for n,t in state.items()} == shapes
                    and all(t.dtype == torch.float32 and torch.isfinite(t).all() for t in state.values())
                    and sum(amount(state, key) for key in expected_trainable) == trainable, "Invalid learned state")
            if prior is not None:
                require(state.keys() == prior.keys() and all(state[n].shape == prior[n].shape for n in state),
                        "Checkpoint topology differs")
            changes = {key:delta(state, prior, [n for n in state if group(n)==key]) if prior is not None else None
                       for key in ("heads", "vision", "qwen")}
            if prior is not None:
                require(all((v > 0) == (key in expected_trainable) for key,v in changes.items()),
                        f"Frozen vs updated tensors differ at {candidate}")
            else:
                require(all(torch.count_nonzero(t) == 0 for n,t in state.items()
                            if group(n) in {"vision","qwen"} and n.endswith(".right")),
                        "Frozen adapters should retain their initial zero-output matrices in A1")
            caption_rotation = (parallel_to_previous(state["heads.caption_language.weight"],
                                prior["heads.caption_language.weight"]) if prior is not None else None)
            records.append({**recorded, "id":candidate, "adapter_delta_l2":changes,
                            "caption_language_weight_nondecay_l2":caption_rotation,
                            "state_sha256": exported[path+"/state.pt"]})
            frame_by_epoch[candidate] = prediction
            if epoch == selected or epoch == limit:
                states[candidate] = state
            prior = state
        require(choose([r for r in records if r["id"][0]==stage])["epoch"] == selected,
                f"Standalone checkpoint selection differs: {stage}")
        prior = states[f"{stage}{selected}"]
    checkpoint_name = json_member(z, "latest.json")["checkpoint"]
    require(checkpoint_name == "C-09-00000", "Unexpected final optimization boundary")
    checkpoint_path = "checkpoints/"+checkpoint_name
    verify_seal(z, checkpoint_path, identity, exported)
    final = read_tensor(z, checkpoint_path+"/state.pt")
    require(final["boundary"] == {"stage":"C", "epoch":9, "step":0, "sums":[0.]*8, "fit_seconds":0., "micro":4},
            "Unexpected final stage state")
    require(final["scheduler"]["last_epoch"] == 8*steps and
            all(o["step"].item() == 8*steps for o in final["optimizer"]["state"].values()) and
            set(final["rng"]) == {"torch","cuda","numpy","python"} and bool(final["scaler"]),
            "Stage C optimizer, schedule, scaler or RNG state failed verification")
    require(final["model"].keys() == states["C8"].keys() and
            all(t.equal(states["C8"][name]) for name,t in final["model"].items()), "Final state differs from C8")
    # The worker builds optimizer groups from module order and each group's parameter iterator.
    # Confirm the mapping before associating optimizer moments with caption-only parameters.
    names = {
        "heads": [name for name in final["model"] if name.startswith("heads.")],
        "vision": [name for name in final["model"] if name.startswith("vision.")],
        "qwen": [name for name in final["model"] if name.startswith("qwen.")],
    }
    moments = {}
    for labelled, optimizer_group in zip(names, final["optimizer"]["param_groups"], strict=True):
        for name, index in zip(names[labelled], optimizer_group["params"], strict=True):
            moment = final["optimizer"]["state"][index]["exp_avg"]
            require(moment.shape == final["model"][name].shape and torch.isfinite(moment).all(),
                    "Cannot align optimizer moments with their tensor shapes")
            if name.startswith("heads.caption"):
                moments[name] = moment.norm().item()
    require(len(moments) == 5 and all(v>0 for v in moments.values()), "Unexpected caption-only optimizer states")
    return records, frame_by_epoch, states, gates, steps, moments


def verify_selected(z, exported, identity, configs, gold, dev, incumbent, frames, states):
    config, visual, qwen = configs
    selections, grids, hashes = {}, [], {}
    for stage, (_, selected, alpha, _) in STAGES.items():
        dest = f"{stage}/selected"
        verify_seal(z, dest, identity, exported)
        bundle = json_member(z, dest+"/bundle.json")
        require((bundle["stage"],bundle["epoch"],bundle["blend_alpha"],bundle["microbatch"]) == (stage,selected,alpha,1)
                and bundle["model"] == config and bundle["visual"] == visual and bundle["qwen"] == qwen
                and bundle["identity"] == identity and bundle["runtime"] == identity["runtime"], "Bundle provenance differs")
        saved = read_tensor(z, dest+"/inference.pt")
        expected = {n:t for n,t in states[f"{stage}{selected}"].items() if not n.startswith(("heads.caption","heads.emotion"))}
        require(saved.keys() == expected.keys() and all(t.equal(expected[n]) for n,t in saved.items()),
                "Selected inference tensor replay differs")
        archived = json_member(z, dest+"/summary.json")
        checked_metrics(archived["standalone"], json_member(z, f"{stage}/epoch_{selected:02d}/metrics.json"))
        standalone = frames[f"{stage}{selected}"]
        epoch_grid = []
        for a in config["ensemble_grid"]:
            pred = mixed(dev, [(1-a,incumbent),(a,standalone)])
            row = {"alpha": a, **diagnostics(gold,pred,config["test_like_excluded"])}
            epoch_grid.append(row)
            if a == alpha:
                selected_prediction = pred
        winner = max(epoch_grid,key=lambda r:(r["full"]["cultural_contrast"],-r["full"]["log_loss"],-r["alpha"]))
        require(winner["alpha"] == alpha, "Recorded blend coefficient does not win original grid")
        checked_metrics(epoch_grid[0], archived["grid"][0])
        for observed, expected in zip(epoch_grid, archived["grid"], strict=True):
            checked_metrics(observed, expected)
        checked_metrics(winner, archived["ensemble"])
        grids.extend({"checkpoint":f"{stage}{selected}",**v} for v in epoch_grid)
        for label, prediction in (("standalone", standalone), ("ensemble",selected_prediction)):
            archive_path = f"{dest}/exp037{stage.lower()}_{label}_submission.zip"
            pd.testing.assert_frame_equal(prediction, zip_prediction_bytes(z.read(archive_path),dev))
            pd.testing.assert_frame_equal(prediction, prediction_bytes(z.read(f"{dest}/{label}/predictions.csv"),dev))
            hashes[archive_path] = exported[archive_path]
        selections[stage] = selected_prediction
    return selections, grids, hashes


def audit(args):
    torch.set_num_threads(2)
    require_hash(args.results, RESULTS)
    require_hash(args.best, BEST_ZIP)
    require_hash(args.exp036_results, PRIOR_RESULTS)
    manifest, cfg, visual, qwen, panel, dev, gold, incumbent, samples = input_provenance(args)
    excluded = cfg["test_like_excluded"]
    require_hash(args.ridge, visual["ridge_zip_sha256"])
    args.output.mkdir(parents=True, exist_ok=False)
    with ZipFile(args.results) as z, ZipFile(args.exp036_results) as prior:
        exported = verify_export(z, HANDOFF)["files"]
        identity = json_member(z, "run.json")
        runtime = json_member(z, "hardware.json")
        require(identity["handoff"] == HANDOFF and identity["source"] == SOURCE and
                identity["gpu"] == runtime["gpu"] == "Tesla T4" and identity["python"] == "3.12.10"
                and identity["cuda"] == runtime["cuda_runtime"] == "12.6"
                and runtime["runtime"] == qwen["runtime"] and runtime["device_map"] == {"":0},
                "Worker runtime differs")
        require(json_member(z,"status.json")=={"state":"COMPLETE"}
                and json_member(z,"config.json")=={"model":cfg,"visual":visual,"qwen":qwen}, "Run config or status differs")
        for path in LOCKS:
            lock = z.read(path)
            require(digest(lock)==manifest["files"]["runtime/"+path], "Run dependency lock differs")
            for line in lock.decode("utf-8").splitlines():
                if not line or line.startswith("#"):
                    continue
                name, version = line.split("==")
                if name.lower() in {n.lower() for n in identity["runtime"]}:
                    observed = {n.lower():v for n,v in identity["runtime"].items()}[name.lower()]
                    require(version==observed, f"Installed dependency differs: {name}")
        checked_metrics(diagnostics(gold,incumbent,excluded), json_member(z,"incumbent.json"))
        records, frames, states, gates, steps, moments = check_epochs(z,exported,identity,cfg,visual,panel,dev,gold)
        selections, grids, zip_hashes = verify_selected(z,exported,identity,(cfg,visual,qwen),gold,dev,incumbent,frames,states)
        winners = json_member(z,"selection.json")
        expected_winner = max(winners["candidates"],key=lambda x:(x["metrics"]["cultural_contrast"],-x["metrics"]["log_loss"]))
        for candidate in winners["candidates"]:
            stage = candidate["stage"]
            _, selected, alpha, _ = STAGES[stage]
            standalone = candidate["kind"] == "standalone"
            require(candidate["epoch"] == selected and candidate["alpha"] == (1. if standalone else alpha),
                    "Best-selection candidate provenance differs")
            checked_metrics(candidate["metrics"],diagnostics(gold, frames[f"{stage}{selected}"] if standalone
                                                               else selections[stage],excluded)["full"])
        require(winners["winner"] == expected_winner and (expected_winner["stage"],expected_winner["kind"])==("C","ensemble")
                and winners["best_sha256"]==BEST_ZIP, "Global selection differs")
        best = z.read("exp037_best_submission.zip")
        require(best == z.read("C/selected/exp037c_ensemble_submission.zip") == args.best.read_bytes(),
                "Best archive is not byte-identical to selected C or owner download")
        if args.stage_c_ensemble:
            require(best == args.stage_c_ensemble.read_bytes(), "The separate Stage C download differs")
        ridge = zip_prediction_bytes(args.ridge.read_bytes(), dev)
        cscva = zip_prediction_bytes(prior.read("exp036_standalone_submission.zip"),dev)
        pd.testing.assert_frame_equal(incumbent,mixed(dev,[(.75,ridge),(.25,cscva)]))
        archive_info = {"hashes": {"results":RESULTS, "best": BEST_ZIP, **zip_hashes},
                        "files_verified":len(exported), "input_files_verified":len(manifest["files"]),
                        "source":SOURCE,"notebook":NOTEBOOK,"handoff":HANDOFF,"identity":identity,"runtime":runtime,
                        "population":manifest["population"],"train_votes":int(panel.n.sum()),"caption_samples":samples,
                        "caption_final_adam_moments":moments,"steps_per_epoch":steps,"preflight":gates,
                        "history":records,"selected_original":{s:diagnostics(gold,p,excluded) for s,p in selections.items()},
                        "selected_grids":grids,"best":winners,"fit_seconds":sum(r["fit_seconds"] for r in records)}
        write(args.output/"audit.json",archive_info)
        for model,prediction in {"ridge":ridge,"cscva":cscva,"incumbent":incumbent,"selected":selections["C"],**frames}.items():
            where = args.output/"predictions"
            where.mkdir(exist_ok=True)
            prediction.to_csv(where/f"{model}.csv",index=False,float_format="%.6f")
        gold.to_csv(args.output/"development_counts.csv",index=False)
        panel.pairs.to_csv(args.output/"training_counts.csv",index=False)
        write(args.output/"intercepts.json",{f"{stage}{e}":states[f"{stage}{e}"]["heads.interaction.bias"].tolist()
              for stage,(_,e,_,_) in STAGES.items()})
        for path in filter(None, (args.results,args.best,args.stage_c_ensemble)):
            shutil.copyfile(path,args.output/path.name)
            require_hash(args.output/path.name,sha(path))
        print("Verified supplied archives, 18 epochs, all seven submission ZIPs and the official Development replay.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for field in ("inputs","results","best","upstream","ridge","exp036_results","output"):
        parser.add_argument("--"+field.replace("_","-"),type=Path,required=True)
    parser.add_argument("--stage-c-ensemble",type=Path)
    audit(parser.parse_args())
