"""Preregistered, training-only logit mechanism probe. No model training."""
import csv,json,itertools
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from .vlm_semantics import file_hash,fingerprint,atomic_json

MODELS=("BCE","P_low","P_high")
KEYS=["painting","target_language","condition"]


def checkpoints(directory,root):
    directory=Path(directory);root=Path(root)
    expected={k.replace("\\","/"):v for k,v in json.loads((root/"docs/exp027_closure_evidence.json").read_text())["files_sha256"].items()}
    identity=json.loads((directory/"identity.json").read_text())
    if file_hash(directory/"identity.json")!=expected["identity.json"]:raise ValueError("Pinned original run identity differs")
    if identity["source_revision"]!="4bc703af144e9121e58ed4cb75c94306c318dff1":raise ValueError("Wrong original checkpoint source")
    for name,digest in identity["source_sha256"].items():
        import subprocess,hashlib
        raw=subprocess.check_output(["git","show",identity["source_revision"]+":"+name],cwd=root)
        if hashlib.sha256(raw).hexdigest()!=digest:raise ValueError("Original scientific source differs: "+name)
        if hashlib.sha256((root/name).read_bytes().replace(b"\r\n",b"\n")).hexdigest()!=digest:
            raise ValueError("Reused scientific source changed: "+name)
    result={}
    for name in MODELS:
        path=directory/"models"/(name+".pt")
        if file_hash(path)!=expected["models/"+name+".pt"]:raise ValueError("Pinned checkpoint SHA differs: "+name)
        seal=json.loads(path.with_suffix(".sha256.json").read_text())
        if seal["sha256"]!=file_hash(path):raise ValueError("Checkpoint receipt differs")
        p=torch.load(path,map_location="cpu",weights_only=True)
        if not p["complete"] or p["identity"]["run"]!=identity:raise ValueError("Checkpoint config/run identity differs")
        best=min(p["curves"],key=lambda r:(r["validation_BCE_nats"],r["epoch"]))
        if p["selected_epoch"]!=best["epoch"]:raise ValueError("Original selection differs")
        result[name]=p
    if any(result[n]["state"]["languages"]!=result["BCE"]["state"]["languages"] for n in MODELS):raise ValueError("Head language vocabularies differ")
    return identity,result


def make_manifest(inputs,checkpoint_dir,root,config):
    identity,models=checkpoints(checkpoint_dir,root);inputs=Path(inputs)
    for name,digest in identity["inputs"].items():
        if file_hash(inputs/name)!=digest:raise ValueError("Original input SHA differs")
    metadata=pd.read_csv(inputs/"training/protocol.csv")
    fit=set(models["BCE"]["identity"]["fit"])
    if any(set(models[n]["identity"]["fit"])!=fit for n in MODELS):raise ValueError("Training identities differ")
    allowed=metadata[metadata.painting.isin(fit)&metadata.role.eq("development")&metadata.fold.ne(1)&metadata.fold.ne(2)]
    images=allowed.set_index("painting")[["image","image_sha256"]].to_dict("index");languages={p:set() for p in images}
    with (inputs/"training/pairs.csv").open(encoding="utf-8",newline="") as f:
        for row in csv.DictReader(f):
            if row["painting"] in languages:languages[row["painting"]].add(row["language"])  # No targets indexed for sampling.
    ranked=sorted(images,key=lambda p:fingerprint([config["seed"],p]));used=set();rectangles=[];pairs=[]
    for i in ranked:
        if i in used:continue
        partners=[j for j in ranked if j!=i and j not in used and len(languages[i]&languages[j])>=config["minimum_shared_languages"]]
        if not partners:continue
        j=partners[0];pair_id=f"pair{len(pairs):02d}"
        options=sorted(itertools.combinations(sorted(languages[i]&languages[j]),2),key=lambda ab:fingerprint([config["seed"],i,j,*ab]))
        for a,b in options[:config["language_pairs_per_rectangle_group"]]:rectangles.append(dict(pair_id=pair_id,I=i,J=j,a=a,b=b))
        pairs.append(pair_id);used.update([i,j])
        if len(pairs)==config["painting_pairs"]:break
    if len(pairs)!=config["painting_pairs"]:raise ValueError("Insufficient fully observed disjoint painting pairs")
    vocab=sorted(models["BCE"]["state"]["languages"]);ref=sorted(vocab,key=lambda l:(-sum(l in s for s in languages.values()),l))[0]
    if any(not s<=set(vocab) for s in languages.values()):raise ValueError("Unknown language in sampling population")
    permutation={l:vocab[(vocab.index(l)+1)%len(vocab)] for l in vocab}
    selected=sorted({(r[p],r[l]) for r in rectangles for p in ("I","J") for l in ("a","b")})
    targets=[]
    with (inputs/"training/pairs.csv").open(encoding="utf-8",newline="") as f:
        wanted=set(selected)
        for row in csv.DictReader(f):
            if (row["painting"],row["language"]) in wanted:
                n=int(row["n"]);k=int(row["k"])
                if n<5 or not 0<=k<=n:raise ValueError("Invalid selected support")
                targets.append(dict(painting=row["painting"],language=row["language"],k=k,n=n))
    targets=sorted(targets,key=lambda r:(r["painting"],r["language"]))
    if len(targets)!=len(selected):raise ValueError("Incomplete/duplicate selected targets")
    queries=[]
    for painting,language in selected:
        for condition in config["conditions"]:
            queries.append(dict(painting=painting,target_language=language,condition=condition,
               prompt_language=None if condition=="B_neutral_prompt" else language,
               head_language=ref if condition=="C_fixed_head" else permutation[language] if condition=="D_crossed_head" else language))
    decoder_sequences=sum(1+len({l for p,l in selected if p==image}) for image in used)*len(MODELS)*2
    repeat_sequences=(1+len({l for p,l in selected if p==sorted(used)[0]}))*len(MODELS)*2
    manifest={"task":"TASK-031","config":config,"checkpoint_run_identity":fingerprint(identity),
              "checkpoint_sha256":{n:file_hash(Path(checkpoint_dir)/"models"/(n+".pt")) for n in MODELS},
              "images":{p:images[p] for p in sorted(used)},"rectangles":rectangles,"queries":queries,
              "reference_head_language":ref,"head_permutation":permutation,"targets_sha256":fingerprint(targets),
              "counts":{"paintings":len(used),"rectangles":len(rectangles),"unique_cells":len(selected),
                        "decoder_sequences":decoder_sequences,"decoder_batch_calls":len(used)*len(MODELS)*2,
                        "repeat_batch_calls":6,"repeat_decoder_sequences":repeat_sequences,
                        "official_parity_decoder_calls":2,"vision_calls_if_cold":len(used)+2,
                        "logical_logits":len(queries)*len(MODELS)*2}}
    return manifest,targets


def mixed(values,I,J,a,b):return values[I,a]-values[I,b]-values[J,a]+values[J,b]


def validate_rows(rows,manifest):
    expected={(q["painting"],q["target_language"],q["condition"],m,s) for q in manifest["queries"] for m in MODELS for s in manifest["config"]["adapter_states"]}
    actual=[(r["painting"],r["target_language"],r["condition"],r["model"],r["adapter_state"]) for r in rows]
    if len(set(actual))!=len(actual) or set(actual)!=expected:raise ValueError("Logit keys missing/duplicated/extra")
    query={(q["painting"],q["target_language"],q["condition"]):q for q in manifest["queries"]}
    for r in rows:
        q=query[r["painting"],r["target_language"],r["condition"]]
        if r["prompt_language"]!=q["prompt_language"] or r["head_language"]!=q["head_language"]:raise ValueError("Prompt/head key mismatch")
        if not np.isfinite(r["logit"]):raise ValueError("Nonfinite raw logit")


def correlation(a,b):return float(np.corrcoef(a,b)[0,1]) if len(a)>=3 and np.ptp(a)>0 and np.ptp(b)>0 else None


def analyze(rows,manifest,targets):
    validate_rows(rows,manifest)
    if fingerprint(targets)!=manifest["targets_sha256"]:raise ValueError("Selected training targets changed")
    target={(r["painting"],r["language"]):r for r in targets};details=[];c=manifest["config"]
    for model,state,condition in itertools.product(MODELS,c["adapter_states"],c["conditions"]):
        z={(r["painting"],r["target_language"]):r["logit"] for r in rows if (r["model"],r["adapter_state"],r["condition"])==(model,state,condition)}
        for r in manifest["rectangles"]:
            I,J,a,b=[r[k] for k in ("I","J","a","b")];gold={key:t["k"]/t["n"] for key,t in target.items()}
            variance=0.
            for key in ((I,a),(I,b),(J,a),(J,b)):
                t=target[key];A=t["k"]+.5;B=t["n"]-t["k"]+.5
                variance+=A*B/((A+B)**2*(A+B+1))
            D=mixed(z,I,J,a,b);T=mixed(gold,I,J,a,b)
            details.append({**r,"model":model,"adapter_state":state,"condition":condition,"D":D,"target_D":T,
              "image_effect":.5*(z[I,a]+z[I,b]-z[J,a]-z[J,b]),"language_effect":.5*(z[I,a]+z[J,a]-z[I,b]-z[J,b]),
              "gap_I":z[I,a]-z[I,b],"target_gap_I":gold[I,a]-gold[I,b],"gap_J":z[J,a]-z[J,b],"target_gap_J":gold[J,a]-gold[J,b],
              "target_tie":abs(T)<=c["target_tie_threshold"],"target_resolved_approx":abs(T)>c["noise_multiplier"]*np.sqrt(variance),
              "prediction_tie":abs(D)<=c["near_zero_logit_threshold"]})
    frame=pd.DataFrame(details);summary=[]
    for keys,g in frame.groupby(["model","adapter_state","condition"],sort=True):
        d=g.D.to_numpy();t=g.target_D.to_numpy();clusters=sorted(g.pair_id.unique());rng=np.random.default_rng(c["bootstrap_seed"])
        means=g.assign(absD=g.D.abs()).groupby("pair_id").absD.mean().reindex(clusters).to_numpy()
        boot=np.array([rng.choice(means,len(means),replace=True).mean() for _ in range(c["bootstrap_replicates"])])
        eligible=~g.prediction_tie&~g.target_tie;resolved=eligible&g.target_resolved_approx
        image_rms=float(np.sqrt(np.mean(g.image_effect.to_numpy()**2)));language_rms=float(np.sqrt(np.mean(g.language_effect.to_numpy()**2)))
        order_known=(g.gap_I.abs()>c["near_zero_logit_threshold"])&(g.gap_J.abs()>c["near_zero_logit_threshold"])
        summary.append(dict(zip(["model","adapter_state","condition"],keys))|{
          "mean_abs_D":float(np.mean(np.abs(d))),"median_abs_D":float(np.median(np.abs(d))),"mean_D":float(d.mean()),
          "D_quantiles":np.quantile(d,[0,.25,.5,.75,1]).tolist(),"near_zero_fraction":float(g.prediction_tie.mean()),
          "image_main_effect_RMS":image_rms,"language_main_effect_RMS":language_rms,
          "interaction_to_image_scale":float(np.sqrt(np.mean(d*d))/(2*image_rms)) if image_rms else None,
          "interaction_to_language_scale":float(np.sqrt(np.mean(d*d))/(2*language_rms)) if language_rms else None,
          "D_target_correlation":correlation(d,t),"gap_target_correlation":correlation(np.r_[g.gap_I,g.gap_J],np.r_[g.target_gap_I,g.target_gap_J]),
          "language_order_reversal_fraction":float((np.sign(g.loc[order_known,"gap_I"])!=np.sign(g.loc[order_known,"gap_J"])).mean()) if order_known.any() else None,
          "language_order_non_tied_rectangles":int(order_known.sum()),
          "sign_agreement_non_ties":float((np.sign(g.loc[eligible,"D"])==np.sign(g.loc[eligible,"target_D"])).mean()) if eligible.any() else None,
          "resolved_target_sign_agreement":float((np.sign(g.loc[resolved,"D"])==np.sign(g.loc[resolved,"target_D"])).mean()) if resolved.any() else None,
          "non_ties":int(eligible.sum()),"resolved_non_ties":int(resolved.sum()),"pair_cluster_95_CI_mean_abs_D":np.quantile(boot,[.025,.975]).tolist()})
    # Paired rectangle/cluster comparisons hold the images and language pairs fixed.
    comparisons=[]
    base_columns=["pair_id","I","J","a","b"]
    for model,state,condition in itertools.product(MODELS,c["adapter_states"],c["conditions"]):
        current=frame[(frame.model==model)&(frame.adapter_state==state)&(frame.condition==condition)]
        references=[("BCE_same_condition","BCE",state,condition),("matching_same_model",model,state,"A_matching"),
                    ("adapter_off_same_head",model,"disabled_same_head_OOD",condition)]
        for label,m,s,v in references:
            if (m,s,v)==(model,state,condition):continue
            ref=frame[(frame.model==m)&(frame.adapter_state==s)&(frame.condition==v)]
            joined=current.merge(ref,on=base_columns,validate="one_to_one",suffixes=("","_reference"))
            delta=joined.D.abs()-joined.D_reference.abs()
            cluster=joined.assign(delta=delta).groupby("pair_id").delta.mean().to_numpy()
            rng=np.random.default_rng(c["bootstrap_seed"])
            boot=rng.choice(cluster,size=(c["bootstrap_replicates"],len(cluster)),replace=True).mean(axis=1)
            comparisons.append(dict(model=model,adapter_state=state,condition=condition,reference=label,
                mean_abs_D_difference=float(delta.mean()),pair_cluster_95_CI=np.quantile(boot,[.025,.975]).tolist(),
                greater_abs_D=int((delta>c["target_tie_threshold"]).sum()),smaller_abs_D=int((delta<-c["target_tie_threshold"]).sum())))
    return frame,{"status":"DESCRIPTIVE_TRAINING_MECHANISM_ONLY","outcome":"INCONCLUSIVE_REQUIRES_INTERPRETATION",
                  "paired_interaction_comparisons":comparisons,
                  "summaries":summary,"bootstrap_unit":"disjoint painting pair, retaining all language rectangles",
                  "limitations":"OOD interventions and adapter-off same adapted head are not unbiased deployment or independent frozen-model comparisons; noisy targets use an approximate Jeffreys variance assuming independent votes; not causal proof"}
