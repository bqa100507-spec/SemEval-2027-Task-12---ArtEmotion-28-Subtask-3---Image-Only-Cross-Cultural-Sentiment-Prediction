"""Original CLIP-B visual projection with exact, zero-initialized q/v LoRA."""

import hashlib
import json
import math

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from .baseline import file_sha256


class LoRALinear(nn.Module):
    def __init__(self,base,rank,alpha):
        super().__init__();self.base=base.requires_grad_(False)
        self.a=nn.Parameter(torch.empty(rank,base.in_features,dtype=torch.float32,device=base.weight.device))
        self.b=nn.Parameter(torch.zeros(base.out_features,rank,dtype=torch.float32,device=base.weight.device))
        self.scale=alpha/rank

    def reset(self,generator):
        nn.init.kaiming_uniform_(self.a,a=math.sqrt(5),generator=generator);nn.init.zeros_(self.b)

    def forward(self,value):
        return self.base(value)+F.linear(F.linear(value,self.a),self.b)*self.scale


class VisualLoRA(nn.Module):
    def __init__(self,model,config):
        super().__init__();self.model=model.requires_grad_(False);self.config=config;self.targets=[]
        layers=model.vision_model.encoder.layers
        if len(layers)!=config['clip']['vision_blocks']:raise ValueError('Unexpected visual block count')
        for i,layer in enumerate(layers):
            for name in config['targets']:
                base=getattr(layer.self_attn,name)
                if not isinstance(base,nn.Linear) or base.in_features!=config['clip']['vision_width']:
                    raise ValueError('Unexpected visual attention implementation')
                setattr(layer.self_attn,name,LoRALinear(base,config['rank'],config['alpha']))
                self.targets.append(f'vision_model.encoder.layers.{i}.self_attn.{name}')
        expected={f'vision_model.encoder.layers.{i}.self_attn.{name}' for i in range(config['clip']['vision_blocks']) for name in ('q_proj','v_proj')}
        if set(self.targets)!=expected or len(self.targets)!=config['expected_target_modules']:
            raise ValueError('LoRA target matching must cover only the 24 visual q/v modules')
        if any('text' in n for n,_ in model.named_modules()):raise ValueError('Text tower must not be instantiated')
        if sum(p.numel() for p in self.adapters().values())!=config['expected_adapter_parameters']:
            raise ValueError('Unexpected LoRA parameter count')
        self.reset(2027,True)

    def adapters(self):
        return {name:p for name,p in self.named_parameters() if name.endswith(('.a','.b'))}

    def reset(self,seed,trainable):
        device=next(self.model.parameters()).device
        generator=torch.Generator(device=device).manual_seed(seed+22000)
        for module in self.modules():
            if isinstance(module,LoRALinear):
                module.reset(generator);module.a.requires_grad_(trainable);module.b.requires_grad_(trainable)
        adapters=set(self.adapters())
        for name,p in self.named_parameters():
            if name not in adapters:p.requires_grad_(False)
            p.grad=None
        self.model.eval()

    def forward(self,pixels):
        # FP32 vision keeps the numerical path matched to historical FP32 caches.
        with torch.autocast(device_type=pixels.device.type,enabled=False):
            vision=self.model.vision_model
            hidden=vision.pre_layrnorm(vision.embeddings(pixels.float()))
            for layer in vision.encoder.layers:
                if self.config['gradient_checkpointing'] and torch.is_grad_enabled():
                    hidden=checkpoint(layer,hidden,None,None,use_reentrant=False)[0]
                else:hidden=layer(hidden,None,None)[0]
            projected=self.model.visual_projection(vision.post_layernorm(hidden[:,0]))
            return F.normalize(projected.float(),dim=-1)

    def frozen_digest(self):
        digest=hashlib.sha256();adapters=set(self.adapters())
        for name,p in self.named_parameters():
            if name not in adapters:
                digest.update(name.encode());digest.update(p.detach().cpu().contiguous().numpy().tobytes())
        return digest.hexdigest()


def adapter_state(vision):
    return {name:p.detach().cpu().clone() for name,p in vision.adapters().items()}


def load_adapter(vision,state):
    if set(state)!=set(vision.adapters()):raise ValueError('Unexpected adapter checkpoint keys')
    with torch.no_grad():
        for name,p in vision.adapters().items():
            value=state[name]
            if value.shape!=p.shape or value.dtype!=torch.float32 or not bool(torch.isfinite(value).all()):
                raise ValueError('Invalid adapter checkpoint tensor')
            p.copy_(value.to(p.device))


def load_visual(cache_dir,config,device):
    from transformers import CLIPImageProcessor,CLIPVisionModelWithProjection
    from transformers.utils import logging
    pin=config['clip'];directory=cache_dir/'huggingface'/('models--'+pin['model'].replace('/','--'))/'snapshots'/pin['revision']
    for name,digest in pin['assets_sha256'].items():
        if not (directory/name).is_file():raise FileNotFoundError(f'Existing pinned CLIP-B asset missing: {directory/name}')
        if file_sha256(directory/name)!=digest:raise ValueError('Pinned CLIP-B asset changed: '+name)
    logging.set_verbosity_error()
    model,info=CLIPVisionModelWithProjection.from_pretrained(directory,use_safetensors=False,
        local_files_only=True,attn_implementation=pin['attention'],output_loading_info=True,low_cpu_mem_usage=True)
    if info['missing_keys'] or info['mismatched_keys'] or any(not n.startswith(('text_model.','text_projection.','logit_scale')) for n in info['unexpected_keys']):
        raise ValueError('Incomplete visual-only CLIP checkpoint loading')
    model.eval().requires_grad_(False).to(device)
    processor=CLIPImageProcessor.from_pretrained(directory,local_files_only=True)
    if processor.crop_size!={'height':224,'width':224} or processor.size!={'shortest_edge':224}:
        raise ValueError('Original 224px preprocessing changed')
    return model,processor,{'checkpoint_revision':pin['revision'],'assets_sha256':pin['assets_sha256'],
        'processor':processor.to_dict(),'attention_class':type(model.vision_model.encoder.layers[0].self_attn).__name__,
        'ignored_nonvisual_checkpoint_keys':len(info['unexpected_keys'])}
