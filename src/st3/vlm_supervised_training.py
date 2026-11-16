"""Painting-grouped count-weighted training and small persistent Qwen feature caches."""

import json
import hashlib
from collections import OrderedDict
from pathlib import Path
import time

import numpy as np
import pandas as pd
from PIL import Image
import torch

from . import data
from .caption_data import inner_partition, require_isolation
from .vlm_semantics import atomic_json, file_hash, fingerprint
from .vlm_supervised import (adapter_parameters, adapters_disabled, frozen_digest,
                             make_head, vote_loss)


def load_training(inputs, config):
    inputs = Path(inputs)
    for name, expected in config["input_sha256"].items():
        if file_hash(inputs/name) != expected:raise ValueError("Pinned training input differs: " + name)
    train = pd.read_csv(inputs/"training/pairs.csv", float_precision="round_trip")
    protocol = pd.read_csv(inputs/"training/protocol.csv")
    actual = train[["painting","image","image_sha256","fold"]].drop_duplicates().sort_values("painting").reset_index(drop=True)
    expected = protocol[protocol.role.eq("development")][actual.columns].sort_values("painting").reset_index(drop=True)
    if (len(train)!=13600 or len(actual)!=1335 or not actual.equals(expected) or not train.split.eq("train").all()
            or (train.n<5).any() or (train.k<0).any() or (train.k>train.n).any() or train[data.KEYS].duplicated().any()):
        raise ValueError("Original P2 training pairs/folds/counts differ")
    require_isolation(train,protocol[protocol.role.eq("holdout")])
    reference = pd.read_csv(inputs/"history/exp022_R0.csv",float_precision="round_trip")
    joined = train[data.KEYS+["fold"]].merge(reference,on=data.KEYS,suffixes=("_train","_reference"),validate="one_to_one")
    if len(joined)!=len(train) or len(reference)!=len(train) or not joined.fold_train.equals(joined.fold_reference):
        raise ValueError("P2 reference keys/folds differ")
    return train,reference


class FeatureCache:
    def __init__(self, root, inputs, config, allowed):
        self.config,self.inputs=config,Path(inputs)
        identity={key:config[key] for key in ("vlm","runtime","neutral_prompt","language_prompt","hidden_size")}
        identity["cache_format"]="qwen-state-arrays-v1"
        identity["extraction_source_sha256"]=file_hash(Path(__file__).with_name("vlm_supervised.py"))
        self.identity=identity
        self.vision_memory=OrderedDict();self.hidden_memory={};self.vision_bytes=0
        self.root=Path(root)/fingerprint(identity)[:20];self.root.mkdir(parents=True,exist_ok=True)
        path=self.root/"identity.json"
        if path.exists() and json.loads(path.read_text())!=identity:raise ValueError("Feature-cache identity differs")
        atomic_json(path,identity)
        images=allowed[["image","image_sha256"]].drop_duplicates()
        self.digests=dict(zip(images.image,images.image_sha256))
        if len(self.digests)!=len(images):raise ValueError("Ambiguous image checksum mapping")

    def path(self,image,language="vision"):
        if image not in self.digests:raise ValueError("Image outside allowed original training population")
        return self.root/(self.digests[image]+("_vision.npz" if language=="vision" else "_states.npz"))

    @staticmethod
    def state_key(language):return "neutral" if language is None else "language_"+fingerprint(language)[:20]

    def read(self,path):
        if not path.exists():return None
        with np.load(path,allow_pickle=False) as saved:values={k:saved[k] for k in saved.files}
        metadata=json.loads(str(values.pop("__metadata")))
        actual={k:{"sha256":hashlib.sha256(v.tobytes()).hexdigest(),"shape":list(v.shape),"dtype":str(v.dtype)} for k,v in values.items()}
        if metadata!={"identity":fingerprint(self.identity),"filename":path.name,"arrays":actual}:
            raise ValueError("Corrupt/incompatible feature arrays: "+str(path))
        if any(not np.isfinite(v).all() for v in values.values()):raise ValueError("Nonfinite cached feature")
        for key,value in values.items():
            if key=="neutral" or key.startswith("language_"):
                if value.shape!=(self.config["hidden_size"],):raise ValueError("Hidden width mismatch")
        if "tokens" in values and (values["tokens"].ndim!=2 or values["tokens"].shape[1]!=self.config["hidden_size"] or values["grid"].shape!=(3,)):
            raise ValueError("Vision-token width/grid mismatch")
        return values

    def write(self,path,**values):
        arrays={k:{"sha256":hashlib.sha256(v.tobytes()).hexdigest(),"shape":list(v.shape),"dtype":str(v.dtype)} for k,v in values.items()}
        metadata=np.array(json.dumps({"identity":fingerprint(self.identity),"filename":path.name,"arrays":arrays}))
        temporary=path.with_suffix(".pending.npz");np.savez(temporary,**values,__metadata=metadata);temporary.replace(path)

    def vision(self,image,backend):
        if image in self.vision_memory:
            self.vision_memory.move_to_end(image);return self.vision_memory[image]
        path=self.path(image);saved=self.read(path)
        if saved is None:
            if backend is None:raise ValueError("Missing vision cache; run authorized extraction")
            source=(self.inputs/"artwork/images"/image).resolve()
            if not source.is_relative_to((self.inputs/"artwork/images").resolve()) or file_hash(source)!=self.digests[image]:
                raise ValueError("Artwork path/checksum differs")
            with Image.open(source) as opened:vision=backend.vision(opened.convert("RGB"))
            self.write(path,tokens=vision["tokens"].numpy(),grid=vision["grid"].numpy());saved=self.read(path)
        result={k:torch.from_numpy(v) for k,v in saved.items()}
        size=sum(v.numel()*v.element_size() for v in result.values())
        while self.vision_memory and self.vision_bytes+size>2*1024**3:
            _,old=self.vision_memory.popitem(last=False);self.vision_bytes-=sum(v.numel()*v.element_size() for v in old.values())
        if size<=2*1024**3:self.vision_memory[image]=result;self.vision_bytes+=size
        return result

    def hidden(self,image,language,backend):
        key=(image,language)
        if key in self.hidden_memory:return self.hidden_memory[key]
        path=self.path(image,language);saved=self.read(path) or {};state=self.state_key(language)
        if state not in saved:
            if backend is None:raise ValueError("Missing frozen decoder feature; run authorized cache extraction")
            self.prepare_image(image,[language],backend,lambda:None);saved=self.read(path)
        result=torch.from_numpy(saved[state]);self.hidden_memory[key]=result
        return result

    def prepare_image(self,image,languages,backend,check):
        path=self.path(image,None);states=self.read(path) or {}
        missing=[language for language in languages if self.state_key(language) not in states]
        if missing:
            vision=self.vision(image,backend)
            for first in range(0,len(missing),backend.config["microbatch_pairs"]):
                check();batch=missing[first:first+backend.config["microbatch_pairs"]]
                with torch.no_grad(),adapters_disabled(backend.model):
                    values=backend.hidden([vision]*len(batch),batch).float().cpu().numpy()
                for language,value in zip(batch,values):states[self.state_key(language)]=value
            self.write(path,**states)
        for language in languages:self.hidden_memory[(image,language)]=torch.from_numpy(states[self.state_key(language)])

    def prepare(self,frame,backend,check,*,neutral=True):
        # Persist at painting boundaries, rather than thousands of tiny pair files.
        mode=backend.model.model.training;backend.model.model.eval()
        try:
            for image,rows in frame.groupby("image",sort=True):
                check();languages=sorted(rows.language.unique())
                if neutral:languages=[None,*languages]
                self.prepare_image(image,languages,backend,check)
        finally:backend.model.model.train(mode)


def windows(frame,config,seed):
    names=np.array(sorted(frame.painting.unique()));np.random.default_rng(seed).shuffle(names)
    for first in range(0,len(names),config["effective_painting_batch"]):
        selected=names[first:first+config["effective_painting_batch"]]
        # Canonical language-row ordering is shared by frozen and live conditions.
        yield pd.concat([frame[frame.painting.eq(name)].sort_values("language") for name in selected])


def head_state(head):return {name:value.detach().cpu().clone() for name,value in head.state_dict().items()}


def model_state(backend,head):
    return {"head":head_state(head),"adapters":{name:p.detach().cpu().clone() for name,p in backend.model.named_parameters() if p.requires_grad},
            "languages":list(head.languages)}


def load_state(backend,head,state):
    head.load_state_dict(state["head"],strict=True)
    current=dict(backend.model.named_parameters())
    with torch.no_grad():
        for name,values in state["adapters"].items():current[name].copy_(values.to(current[name].device))


class Trainer:
    def __init__(self,backend,cache,config,languages,condition,*,device=None):
        self.backend,self.cache,self.config,self.condition=backend,cache,config,condition
        self.device=torch.device(device or ("cuda" if condition=="L0" else "cpu"))
        self.head=make_head(languages,config,self.device)
        groups=[{"params":self.head.parameters(),"lr":config["optimizer"]["head_lr"]}]
        if condition=="L0":groups.append({"params":adapter_parameters(backend.model),"lr":config["optimizer"]["adapter_lr"]})
        self.optimizer=torch.optim.AdamW(groups,weight_decay=config["optimizer"]["weight_decay"])
        self.parameters=[p for group in self.optimizer.param_groups for p in group["params"]]
        self.scaler=torch.amp.GradScaler("cuda",enabled=condition=="L0",init_scale=1024.)
        self.clipped,self.steps=0,0

    def hidden(self,rows,train=False):
        if self.condition=="L0":
            bank={image:self.cache.vision(image,self.backend) for image in rows.image.unique()}
            return self.backend.hidden([bank[image] for image in rows.image],rows.language.tolist(),train=train)
        return torch.stack([self.cache.hidden(r.image,None if self.condition=="F0_image" else r.language,None) for r in rows.itertuples()]).to(self.device)

    def epoch(self,frame,epoch,check):
        self.head.train()
        if self.condition=="L0":self.backend.model.model.train()
        tick=time.perf_counter();loss_sum=0.;votes=0.;pairs=0;head_gradients=[];lora_gradients=[]
        for batch in windows(frame,self.config,self.config["seed"]+epoch):
            check();self.optimizer.zero_grad(set_to_none=True)
            denominator=float(batch.n.sum())
            micro=self.config["microbatch_pairs"]
            for first in range(0,len(batch),micro):
                rows=batch.iloc[first:first+micro]
                with torch.autocast("cuda",dtype=torch.float16,enabled=self.condition=="L0"):
                    hidden=self.hidden(rows,train=self.condition=="L0")
                    if self.condition=="L0" and not hidden.requires_grad:raise ValueError("LoRA hidden-state graph disconnected")
                    logits=self.head(hidden,self.head.indices(rows.language,self.device))
                    loss=vote_loss(logits,torch.tensor(rows.k.to_numpy(),device=self.device),torch.tensor(rows.n.to_numpy(),device=self.device),denominator=denominator)
                self.scaler.scale(loss).backward();loss_sum+=float(loss.detach())*denominator;votes+=float(rows.n.sum());pairs+=len(rows)
            self.scaler.unscale_(self.optimizer)
            norm_of=lambda parameters:float(torch.sqrt(sum((p.grad.float().square().sum() for p in parameters if p.grad is not None),torch.zeros((),device=self.device))))
            head_gradients.append(norm_of(self.head.parameters()))
            if self.condition=="L0":lora_gradients.append(norm_of(adapter_parameters(self.backend.model)))
            norm=torch.nn.utils.clip_grad_norm_(self.parameters,self.config["optimizer"]["gradient_clip"])
            if not torch.isfinite(norm):raise ValueError("Nonfinite optimizer gradient")
            self.clipped+=int(float(norm)>self.config["optimizer"]["gradient_clip"])
            self.scaler.step(self.optimizer);self.scaler.update();self.steps+=1
        if self.condition=="L0":torch.cuda.synchronize()
        elapsed=time.perf_counter()-tick
        return {"binary_BCE_nats":loss_sum/votes,"pairs":pairs,"paintings":frame.painting.nunique(),"seconds":elapsed,"pairs_per_second":pairs/elapsed,
                "head_gradient_norm_min":min(head_gradients),"LoRA_gradient_norm_min":min(lora_gradients) if lora_gradients else None,
                "clipping_frequency":self.clipped/self.steps,"AMP_scale":self.scaler.get_scale()}

    def predict(self,frame):
        self.head.eval()
        if self.condition=="L0":self.backend.model.model.eval()
        values=[]
        with torch.no_grad():
            for first in range(0,len(frame),self.config["microbatch_pairs"]):
                rows=frame.iloc[first:first+self.config["microbatch_pairs"]]
                with torch.autocast("cuda",dtype=torch.float16,enabled=self.condition=="L0"):
                    hidden=self.hidden(rows)
                    logits=self.head(hidden,self.head.indices(rows.language,self.device))
                values.extend(torch.sigmoid(logits.float()).cpu().tolist())
        result=np.array(values)
        if not np.isfinite(result).all():raise ValueError("Nonfinite sentiment probabilities")
        return result

    def validate(self,frame):
        prediction=torch.tensor(self.predict(frame),dtype=torch.float64).clamp(1e-7,1-1e-7)
        k=torch.tensor(frame.k.to_numpy());n=torch.tensor(frame.n.to_numpy())
        value=float((-(k*prediction.log()+(n-k)*torch.log1p(-prediction))).sum()/n.sum())
        if not np.isfinite(value):raise ValueError("Nonfinite validation loss")
        return value
