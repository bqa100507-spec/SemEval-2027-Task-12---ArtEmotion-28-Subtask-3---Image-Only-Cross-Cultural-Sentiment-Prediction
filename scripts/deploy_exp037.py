"""Prepare sealed EXP-037 inference bundles and compose exact image-only predictions."""

import argparse
import subprocess
import sys
from pathlib import Path
from zipfile import ZipFile

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"src"))
from audit_exp037 import HANDOFF, RESULTS, STAGES, require_hash, verify_seal
from st3 import data, submission
from st3.contrast_study import new_output
from st3.e50_inference import read_input
from st3.exp036_io import check, read, sha, write
from st3.exp037_closure import json_member, recipe_leaves, recipe_prediction, require, verify_export


def prepare(args):
    require_hash(args.results,RESULTS)
    selection = read(args.diagnostics/"candidate_recipes.json")
    best = selection["best"]["id"]
    output_recipes = {"A_standalone":"A4", "B_standalone":"B6","C_standalone":"C6",
                      "A_original_blend":"incumbent","B_original_blend":selection["recipes"]["B6_a010"],
                      "C_original_blend":selection["recipes"]["selected"]}
    if best!="selected":
        output_recipes |= {"posthoc_"+best:selection["recipes"][best]}
    checkpoints = set.union(*(recipe_leaves(r) & {"A4","B6","C6"} for r in output_recipes.values()))
    required = set.union(*(recipe_leaves(r) for r in output_recipes.values()))
    require(required <= checkpoints | {"ridge","cscva","incumbent"}, "A recipe requires an unprepared checkpoint")
    with ZipFile(args.results) as z, new_output(args.output) as scratch:
        exported = verify_export(z,HANDOFF)["files"]
        identity = json_member(z,"run.json")
        verified = {}
        for label in sorted(checkpoints):
            stage = label[0]
            folder = f"{stage}/selected"
            verify_seal(z,folder,identity,exported)
            prefix = folder+"/"
            dest = scratch/label
            dest.mkdir()
            for name in ("inference.pt","bundle.json","seal.json"):
                (dest/name).write_bytes(z.read(prefix+name))
            # Include the full selected seal and its members so the immutable worker can verify it.
            for name in json_member(z,prefix+"seal.json")["files"]:
                target = dest/name
                if not target.exists():
                    target.parent.mkdir(parents=True,exist_ok=True)
                    target.write_bytes(z.read(prefix+name))
            verify_epoch = STAGES[stage][1]
            meta = json_member(z,prefix+"bundle.json")
            require(verify_epoch == meta["epoch"] == int(label[1:]), "Candidate does not match verified selection")
            verified[label] = {"checkpoint_sha256":exported[f"{stage}/epoch_{verify_epoch:02d}/state.pt"],
                               "bundle_sha256":exported[prefix+"inference.pt"],
                               "selected_standalone_zip_sha256":exported[prefix+f"exp037{stage.lower()}_standalone_submission.zip"]}
        for name in ("requirements-exp035-kaggle.lock.txt","requirements-exp036-kaggle.txt"):
            (scratch/name).write_bytes(z.read(name))
        index = {"identity":identity,"results_zip_sha256":RESULTS,"epochs":verified,
                 "recipes":output_recipes, "eligibility":"UNCONFIRMED_QWEN_ST3",
                 "probability_rounding":6, "posthoc_selection":"public Development",
                 "files":{p.relative_to(scratch).as_posix():sha(p) for p in sorted(scratch.rglob("*")) if p.is_file()}}
        write(scratch/"deployment_index.json",index)


def compose(args):
    index = read(args.bundle/"deployment_index.json")
    check(args.bundle,index["files"])
    require(args.candidate in index["recipes"], f"Unknown candidate. Choose {', '.join(index['recipes'])}.")
    recipe = index["recipes"][args.candidate]
    if args.replay:
        dev = data.read_dev_input(args.data_dir)
    else:
        if args.input is None:
            raise ValueError("Supply the official input-only CSV with --input")
        official = read(ROOT/"configs/e50_inference.json")["official_languages"]
        dev = read_input(args.input,official)
    overrides = {}
    for item in args.component:
        label,path = item.split("=",1)
        require(label not in overrides and label in recipe_leaves(recipe), f"Unexpected component: {label}")
        overrides[label] = Path(path)
    with new_output(args.output) as staged:
        frames, components = {},{}
        for name in sorted(recipe_leaves(recipe)):
            path = overrides.get(name)
            if not path and args.replay and name in index["epochs"]:
                path = args.bundle/name/f"exp037{name[0].lower()}_standalone_submission.zip"
            if not path:
                if name not in index["epochs"] or args.replay:
                    raise ValueError(f"Supply --component {name}=PATH_TO_MATCHED_ZIP")
                inference_output = staged/"component_inference"/name
                cmd = [sys.executable,str(ROOT/"scripts/run_exp037.py"),"infer","--bundle",str(args.bundle/name),
                       "--input",str(args.input),"--images",str(args.data_dir),"--output",str(inference_output)]
                for field in ("siglip","qwen"):
                    if (value := getattr(args,field)) is not None:
                        cmd.extend((f"--{field}",str(value)))
                subprocess.run(cmd,check=True)
                path = inference_output/f"exp037{name[0].lower()}_standalone_submission.zip"
            frames[name] = submission.validate_archive(path,dev)
            components[name] = sha(path)
        pred = recipe_prediction(dev,frames,recipe)
        csv = staged/"predictions.csv"
        pred.to_csv(csv,index=False,float_format="%.6f")
        archive = staged/f"exp037_{args.candidate}_submission.zip"
        submission.pack(csv,archive,dev)
        pd.testing.assert_frame_equal(submission.validate_archive(archive,dev),pred)
        write(staged/"inference_receipt.json",{"candidate":args.candidate,"recipe":recipe,
              "archived_development_replay":args.replay,"provenance":index["identity"],
              "components_sha256":components,"bundle_sha256":sha(args.bundle/"deployment_index.json"),
              "submission_sha256":sha(archive),"pairs":len(dev),"paintings":int(dev.painting.nunique())})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command",required=True)
    p = sub.add_parser("prepare")
    for name in ("results","diagnostics","output"):
        p.add_argument(f"--{name}",type=Path,required=True)
    p = sub.add_parser("infer")
    for name in ("bundle","data-dir","output"):
        p.add_argument(f"--{name}",type=Path,required=True)
    for name in ("input","siglip","qwen"):
        p.add_argument(f"--{name}",type=Path)
    p.add_argument("--replay",action="store_true",help="Use archived Development standalone ZIPs for CPU parity")
    p.add_argument("--candidate",required=True)
    p.add_argument("--component",action="append",default=[],metavar="NAME=ARCHIVE")
    args = parser.parse_args()
    (prepare if args.command=="prepare" else compose)(args)
