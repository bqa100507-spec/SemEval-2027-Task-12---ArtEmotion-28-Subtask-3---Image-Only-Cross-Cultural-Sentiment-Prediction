"""Measure isolated caption gradients with sealed targets and saved heads on bounded proxy activations."""

import argparse
import io
import json
import sys
from contextlib import nullcontext
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pandas as pd
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"src"))
from audit_exp037 import HANDOFF, INPUT_ZIP, RESULTS, require_hash
from st3.competition_direct_qwen import LowRankUpdate
from st3.data import KEYS, training_pairs
from st3.exp036 import LoRA
from st3.exp036_io import write
from st3.exp037 import FusionHeads, counts_for, losses
from st3.exp037_io import unpack_inputs
from st3.exp037_runtime import TrainingPaintings, pinned_model


def native_probe(args, z, panel, sample, device):
    """Use two real Train images and the selected states with one caption-only backward each."""
    import tempfile
    configs = json.loads(z.read("config.json"))
    visual, qwen, config = (configs[n] for n in ("visual","qwen","model"))
    languages = {l:i for i,l in enumerate(sorted(panel.pairs.language.unique()))}
    manifest = json.loads(z.read("run.json"))
    if manifest["handoff"] != HANDOFF or device.type != "cuda" or not args.siglip or not args.qwen:
        raise ValueError("Native probe requires CUDA and both exact local snapshots")
    records = []
    with tempfile.TemporaryDirectory() as scratch:
        root = unpack_inputs(args.inputs,Path(scratch),HANDOFF,INPUT_ZIP)
        mapping = json.loads((root/"exp037_manifest.json").read_text())["images"]
        model,processor,_ = pinned_model(visual,qwen,config,args.siglip,args.qwen,languages,auxiliary=True,root=root,images=mapping)
        torch.cuda.reset_peak_memory_stats()
        for stage,epoch in (("A",4),("B",6),("C",6)):
            state = torch.load(io.BytesIO(z.read(f"{stage}/epoch_{epoch:02d}/state.pt")),weights_only=True,map_location="cpu")
            model.stage(stage)
            model.restore(state)
            model.train()
            for image in sample[:2]:
                ids,groups = panel.groups([image])
                target = panel.targets(ids,groups,device)
                votes = float(panel.n[ids].sum())
                mass = float((panel.mask[ids]*panel.n[ids]).sum())
                pixels,encoded = processor.batch([panel.images[image]],device)
                shared = {"visual_projection":[model.heads.visual_projection.weight],
                          "qwen_projection":[model.heads.qwen_projection.weight],
                          "caption_language":[model.heads.caption_language.weight]}
                shared.update({n:p for n,p in model.groups().items() if n!="fusion" and p[0].requires_grad})
                for scale in (1.,128.,8192.):
                    torch.manual_seed(203038)
                    with torch.autocast("cuda",dtype=torch.float16):
                        features = model.features(pixels,encoded)
                        loss = losses(model.heads,features,target,votes,mass)[2]*config["caption_lambda"]
                    grads = iter(torch.autograd.grad(loss*scale,[p for group in shared.values() for p in group]))
                    norms = {n:float(torch.stack([next(grads).float().norm() for _ in group]).norm().item()/scale)
                             for n,group in shared.items()}
                    if not np.isfinite(list(norms.values())).all():
                        raise FloatingPointError(f"Native caption gradient at scale {scale} is nonfinite")
                    records.append({"stage":stage,"painting":panel.pairs.painting.iloc[ids[0]],"caption_mass":mass,
                                    "weighted_caption_loss":loss.item(),"loss_scale":scale,"rescaled_grad_l2":norms})
        return {"measurements":records,"gpu":torch.cuda.get_device_name(device),
                "peak_allocated_bytes":torch.cuda.max_memory_allocated(),"peak_reserved_bytes":torch.cuda.max_memory_reserved()}


def probe(args):
    require_hash(args.inputs, INPUT_ZIP)
    require_hash(args.results, RESULTS)
    torch.set_num_threads(2)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable for the optional bounded FP16 diagnostic")
    with ZipFile(args.inputs) as handoff, ZipFile(args.results) as z:
        annotations = pd.read_csv(handoff.open("exp037_inputs/training/annotations.csv"),dtype=str,keep_default_na=False)
        pairs = training_pairs(annotations).sort_values(KEYS,ignore_index=True)
        counts = counts_for(annotations,pairs)
        with np.load(io.BytesIO(handoff.read("exp037_inputs/training/targets.npz")),allow_pickle=False) as targets:
            values,mask = targets["values"],targets["mask"]
        config = json.loads(z.read("config.json"))["model"]
        visual = json.loads(z.read("config.json"))["visual"]
        languages = {name:i for i,name in enumerate(sorted(pairs.language.unique()))}
        panel = TrainingPaintings(pairs,languages,counts,values,mask)
        sample = np.random.default_rng(config["seed"]).permutation(len(panel.members))[:4]
        if args.native:
            write(args.output,native_probe(args,z,panel,sample,device))
            return
        measurements = []
        for stage,epoch in (("A",4),("B",6),("C",6)):
            snapshot = torch.load(io.BytesIO(z.read(f"{stage}/epoch_{epoch:02d}/state.pt")),
                                  weights_only=True,map_location="cpu")
            heads = FusionHeads(visual["hidden_size"],len(languages),config,auxiliary=True).to(device).train()
            heads.load_state_dict({n.removeprefix("heads."):adj for n,adj in snapshot.items() if n.startswith("heads.")})
            # Proxy encoder outputs isolate exact saved fusion/heads from multi-gigabyte checkpoints.
            # The two adapters use the historical low-rank classes with synthetic base layers.
            for batch in ([sample[0]],[sample[1]],[sample[2]],sample.tolist()):
                ids, groups = panel.groups(batch)
                target = panel.targets(ids,groups,device)
                vote_mass = float(panel.n[ids].sum())
                mass = max(1.,float((panel.n[ids]*panel.mask[ids]).sum()))
                for mode in ("fp32","fp16_unscaled","fp16_scaled_128") if device.type=="cuda" else ("fp32",):
                    torch.manual_seed(203038)
                    visual_width,qwen_width = visual["hidden_size"],config["qwen_width"]
                    vis = LoRA(nn.Linear(visual_width,visual_width).requires_grad_(False),4,4).to(device)
                    vlm = LowRankUpdate(nn.Linear(qwen_width,qwen_width).requires_grad_(False),4,8,
                                        torch.Generator().manual_seed(203038)).to(device)
                    p = torch.randn(len(batch),25,visual_width,device=device)
                    q = torch.randn(len(batch),17,qwen_width,device=device)
                    pad = torch.zeros(len(batch),17,dtype=torch.bool,device=device)
                    context = torch.autocast("cuda",dtype=torch.float16) if mode.startswith("fp16") else nullcontext()
                    with context:
                        vfeats = vis(p)
                        features = heads(vfeats,vfeats.mean(1),vlm(q),pad)
                        caption = losses(heads,features,target,vote_mass,mass)[2]*config["caption_lambda"]
                    scale = 128 if mode.endswith("128") else 1
                    wrt = [heads.visual_projection.weight, heads.qwen_projection.weight, heads.caption[0].weight,
                           heads.caption_language.weight,vis.right,vlm.right]
                    grads = torch.autograd.grad(caption*scale,wrt)
                    names = ["visual_projection","qwen_projection","caption_head","caption_language",
                             "proxy_visual_lora","proxy_qwen_qlora"]
                    measurements.append({"stage":stage,"mode":mode,"paintings":len(batch),
                        "sample_paintings":[panel.pairs.painting.iloc[panel.members[i][0]] for i in batch],
                        "caption_mass":mass,"weighted_loss":caption.detach().item(),
                        "grad_l2":{n:float(g.float().norm().item()/scale) for n,g in zip(names,grads,strict=True)}})
    write(args.output,{"device":torch.cuda.get_device_name(device) if device.type=="cuda" else str(device),
          "torch":torch.__version__, "encoder_inputs":"deterministic synthetic activations",
          "saved_heads":"A4, B6 and C6", "measurements":measurements})
    print(f"Measured {len(measurements)} fixed caption-only backward evaluations. No optimizer steps.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("inputs","results","output"):
        parser.add_argument("--"+name,type=Path,required=True)
    parser.add_argument("--device", choices=("cpu","cuda:0"),default="cpu")
    parser.add_argument("--native",action="store_true")
    parser.add_argument("--siglip",type=Path)
    parser.add_argument("--qwen",type=Path)
    probe(parser.parse_args())
