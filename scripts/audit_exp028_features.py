"""Label-free local cache inventory for a proposed experiment; no extraction/fitting."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/"src"))
from st3.vlm_semantics import file_hash,fingerprint,atomic_json


def audit(workspace,inputs,cloud_identity=None):
    workspace=Path(workspace);inputs=Path(inputs)
    config=json.loads((ROOT/"configs/exp026.json").read_text())
    for name in ("training/pairs.csv","training/protocol.csv"):
        if file_hash(inputs/name)!=config["input_sha256"][name]:raise ValueError("Original metadata checksum differs")
    # No k/n, p, captions or sealed annotations are read into the audit.
    protocol=pd.read_csv(inputs/"training/protocol.csv")
    p2=protocol[protocol.role.eq("development")]
    pairs=pd.read_csv(inputs/"training/pairs.csv",usecols=["painting","language","image"])
    if len(p2)!=1335 or len(pairs)!=13600 or not set(pairs.painting)<=set(p2.painting):raise ValueError("P2 key coverage differs")
    images=set(p2.image);languages=sorted(pairs.language.unique());records=[]
    baseline=json.loads((workspace/"runs/exp002_baseline_ablation/run.json").read_text())["embedding"]
    sources=[("CLIP-B",workspace/"cache"/Path(baseline["path"].replace("\\","/")).name,baseline["sha256"],{k:v for k,v in baseline.items() if k!="path"})]
    for label,directory in (("CLIP-L","clip-large-b73b0a11250c507e9f1ec20079c65cfa34d33ee39f804304d0ea8f7a14cf7c81"),
                            ("SigLIP2-Base","siglip2-2e7f999fbf7125bc04084a576cb358fe760dfa2cc12ed6fa5e3fc1d8c790b3fa")):
        path=workspace/"cache"/directory
        manifest=json.loads((path/"manifest.json").read_text())
        sources.append((label,path/"features.npz",manifest["features_sha256"],manifest["identity"]))
    for label,path,expected,identity in sources:
        if file_hash(path)!=expected:raise ValueError("Existing visual cache checksum differs: "+label)
        with np.load(path,allow_pickle=False) as saved:
            names=saved["names"].tolist();lookup={name:i for i,name in enumerate(names)}
            if len(lookup)!=len(names):raise ValueError("Duplicate cache image keys")
            missing=sorted(images-set(names));values=saved["features"][[lookup[x] for x in sorted(images-set(missing))]]
            if not np.isfinite(values).all():raise ValueError("Nonfinite existing visual cache")
            norms=np.linalg.norm(values,axis=1)
            records.append(dict(model=label,path=path.relative_to(workspace).as_posix(),sha256=expected,identity=identity,
                                stored_rows=len(names),P2_covered=len(values),P2_missing=len(missing),width=values.shape[1],
                                P2_max_abs_L2_norm_error=float(np.max(np.abs(norms-1))),bytes=path.stat().st_size))
    qroot=workspace/"cache/exp026_states";legacy=[]
    expected_identity={k:config[k] for k in ("vlm","runtime","neutral_prompt","language_prompt","hidden_size")}
    expected_identity.update(cache_format="qwen-state-arrays-v1",extraction_source_sha256=hashlib.sha256((ROOT/"src/st3/vlm_supervised.py").read_bytes().replace(b"\r\n",b"\n")).hexdigest())
    for path in sorted(qroot.rglob("identity.json")):
        identity=json.loads(path.read_text());counts=dict(vision_files=0,language_states=0,neutral_states=0)
        for arrayfile in sorted(path.parent.glob("*.npz")):
            with np.load(arrayfile,allow_pickle=False) as s:
                arrays={k:s[k] for k in s.files if k!="__metadata"}
                meta=json.loads(str(s["__metadata"]))
                expected_arrays={k:dict(sha256=hashlib.sha256(v.tobytes()).hexdigest(),shape=list(v.shape),dtype=str(v.dtype)) for k,v in arrays.items()}
                if meta!={"identity":fingerprint(identity),"filename":arrayfile.name,"arrays":expected_arrays}:raise ValueError("Qwen cache array integrity differs")
                if any(not np.isfinite(v).all() for v in arrays.values()):raise ValueError("Nonfinite Qwen cache")
                counts["vision_files"]+=int("tokens" in arrays);counts["neutral_states"]+=int("neutral" in arrays)
                counts["language_states"]+=sum(k.startswith("language_") for k in arrays)
        legacy.append(dict(path=path.parent.relative_to(workspace).as_posix(),identity_sha256=file_hash(path),
                           exact_current_identity=identity==expected_identity,counts=counts,
                           reuse="REQUIRES_ORIGINAL_SOURCE_PARITY_REVIEW" if identity!=expected_identity else "CONTENT_COVERAGE_REVIEW_REQUIRED"))
    cold_batches=int(sum((len(g)+config["microbatch_pairs"]-1)//config["microbatch_pairs"] for _,g in pairs.groupby("painting")))
    result=dict(status="DESIGN_ONLY_NO_EXTRACTION",P2_paintings=len(p2),P2_pairs=len(pairs),languages=languages,
                input_sha256={n:file_hash(inputs/n) for n in ("training/pairs.csv","training/protocol.csv")},
                visual_caches=records,local_Qwen_caches=legacy,
                exact_cold_extraction=dict(vision_forwards=1335,language_decoder_sequences=13600,
                  per_painting_decoder_batches_at_8=cold_batches,optional_neutral_sequences=1335,optional_neutral_batches=1335),
                checkpoint_scope="Fresh pretrained frozen Qwen / adapters disabled; EXP026/027 fitted decoder states are not reusable across folds")
    if cloud_identity:
        actual=json.loads(Path(cloud_identity).read_text())
        result["Drive_Qwen_cache"]=dict(identity_sha256=file_hash(cloud_identity),exact_identity=actual==expected_identity,
                directory="MyDrive/exp026/features/8f1e57257da52d7212b0",metadata_listing="PARTIAL_FIRST_100_FILES_ONLY",
                full_payload_coverage="UNVERIFIED; do not infer complete P2 state coverage from a folder/identity",
                new_decoder_sequences="Unknown until all payload keys/checksums are audited; exact cold ceiling13600, lower bound0")
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--workspace",type=Path,required=True);p.add_argument("--inputs",type=Path,required=True)
    p.add_argument("--cloud-cache-identity",type=Path);p.add_argument("--output",type=Path,required=True);a=p.parse_args()
    if a.output.exists():p.error("Choose a new output filename")
    result=audit(a.workspace,a.inputs,a.cloud_cache_identity);atomic_json(a.output,result)
    print(json.dumps({k:v for k,v in result.items() if k not in ("visual_caches","input_sha256","local_Qwen_caches")},indent=2))


if __name__=="__main__":main()
