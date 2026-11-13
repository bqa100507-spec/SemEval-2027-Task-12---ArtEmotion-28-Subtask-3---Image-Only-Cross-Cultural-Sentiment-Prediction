"""Pinned EXP-021 evidence and portable, annotation-only image-feature inputs."""

import importlib.metadata
import json
import platform
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch

from . import data
from .baseline import file_sha256
from .caption_comparison import image_banks
from .cli import ROOT,json_output
from .contrast_study import new_output
from .polarity_workflow import StageBlocked,read_sealed,seal
from .vote_inputs import empirical_distributions,input_context


def config022():
    cfg=json.loads((ROOT/'configs/exp022.json').read_text())
    if (cfg['rank']!=8 or cfg['alpha']!=16 or cfg['targets']!=['q_proj','v_proj'] or
        cfg['adapter_dropout']!=0 or cfg['clip']['revision']!='3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268'):
        raise ValueError('Frozen visual adapter design changed')
    return cfg


def verify_parent(args,cfg):
    if file_sha256(ROOT/'configs/exp021.json')!=cfg['parent_config_sha256']:raise ValueError('EXP-021 configuration changed')
    path=args.exp021_dir/'decision/run.json'
    if not path.exists():raise StageBlocked('Completed EXP-021 CV/Verify required on its original branch')
    identity=json.loads(path.read_text())['identity']
    for name,digest in identity['source_sha256'].items():
        if file_sha256(ROOT/name)!=digest:raise ValueError('Historical source changed: '+name)
    for name,digest in cfg['parent_manifests'].items():
        path=args.exp021_dir/name
        if file_sha256(path)!=digest:raise ValueError('EXP-021 manifest changed: '+name)
        read_sealed(path.parent,identity['audit_identity'] if name.startswith('audit/') else identity)
    decision=read_sealed(args.exp021_dir/'decision',identity);verification=read_sealed(args.exp021_dir/'verification',identity)
    if decision['status']!='COMPETITION_NO_GO' or not decision['all_checkpoints_replayed'] or verification['checkpoints_replayed']!=30:
        raise ValueError('Unexpected EXP-021 completion state')
    for key,directory in (('study_manifest_sha256','cv'),('verification_manifest_sha256','verification')):
        if decision[key]!=file_sha256(args.exp021_dir/directory/'run.json'):raise ValueError('Parent decision evidence changed')
    attempts=sorted((args.exp021_dir/'cv_verification_attempts').glob('*.json'))
    latest=json.loads(attempts[-1].read_text()) if attempts else {}
    if latest.get('status')!='PASS' or latest.get('study_manifest_sha256')!=file_sha256(args.exp021_dir/'cv/run.json'):
        raise StageBlocked('Latest parent replay failed or incomplete')
    if any((args.exp021_dir/name).exists() for name in ('confirm','dev')):raise ValueError('Expected locked/public evaluation sealed')
    return decision


def input_identity(cfg):
    return {'parent_config_sha256':cfg['parent_config_sha256'],'parent_targets_sha256':cfg['parent_targets_sha256'],
        'parent_predictions_sha256':cfg['parent_predictions_sha256'],'clip_revision':cfg['clip']['revision']}


def export_inputs(args,cfg):
    directory=args.run_dir/'inputs'
    if directory.exists():read_sealed(directory,input_identity(cfg));return directory
    decision=verify_parent(args,cfg)
    original=SimpleNamespace(**{**vars(args),'run_dir':args.exp021_dir})
    ctx=input_context(original)
    if file_sha256(args.exp021_dir/'audit/targets.csv')!=cfg['parent_targets_sha256'] or file_sha256(args.exp021_dir/'cv/predictions.csv')!=cfg['parent_predictions_sha256']:
        raise ValueError('Pinned annotation targets or parent OOF predictions changed')
    empirical_distributions(ctx.train)
    banks=image_banks(args,ctx.train);names=sorted(banks[0])
    histories=json.loads((args.exp021_dir/'cv/fits.json').read_text())
    with new_output(directory) as stage:
        # Preserve exact parent CSV bytes rather than resampling/reweighting.
        (stage/'targets.csv').write_bytes((args.exp021_dir/'audit/targets.csv').read_bytes())
        (stage/'parent_predictions.csv').write_bytes((args.exp021_dir/'cv/predictions.csv').read_bytes())
        np.savez(stage/'features.npz',names=np.array(names),clip=np.stack([banks[0][n] for n in names]),
                 siglip=np.stack([banks[1][n] for n in names]),large=np.stack([banks[2][n] for n in names]))
        json_output(histories,stage/'historical_fits.json')
        historical=[]
        for fold in range(1,6):
            for condition in ('D0','D8'):
                for seed in cfg['training']['seeds']:
                    with np.load(args.exp021_dir/'work'/f'fold_{fold}'/f'{condition}_{seed}'/'predictions.npz',allow_pickle=False) as a:
                        historical.append(pd.DataFrame({'painting':a['painting'],'language':a['language'],
                            'p_positive':a['probabilities'],'fold':fold,'model':condition,'seed':seed}))
        pd.concat(historical).to_csv(stage/'historical_seeds.csv',index=False,lineterminator='\n')
        json_output({'verified_parent':decision,'parent_manifest_sha256':cfg['parent_manifests'],
                     'locked_or_caption_targets_included':False},stage/'parent_proof.json')
        seal(stage,input_identity(cfg),{'paintings':ctx.train.painting.nunique(),'pairs':len(ctx.train),
            'frozen_widths':[512,768,768],'parent_verified_locally':True})
    return directory


def inputs(args):
    cfg=config022()
    directory=args.portable_dir/'inputs' if args.portable_dir else export_inputs(args,cfg)
    if args.portable_dir:
        if not args.input_manifest_sha256:raise StageBlocked('Portable mode requires the published --input-manifest-sha256')
        if file_sha256(directory/'run.json')!=args.input_manifest_sha256:raise ValueError('Portable input manifest hash mismatch')
    read_sealed(directory,input_identity(cfg))
    if file_sha256(directory/'targets.csv')!=cfg['parent_targets_sha256'] or file_sha256(directory/'parent_predictions.csv')!=cfg['parent_predictions_sha256']:
        raise ValueError('Portable original annotation/reference identity differs')
    train=pd.read_csv(directory/'targets.csv',float_precision='round_trip');empirical_distributions(train)
    if not train.role.eq('development').all() or train.painting.nunique()!=1335:raise ValueError('Locked or incomplete training population')
    parent=pd.read_csv(directory/'parent_predictions.csv',float_precision='round_trip')
    reference=parent[parent.model.eq('R0')][data.KEYS+['p_positive','fold']]
    data.require_same_pairs(train,reference)
    joined=train[data.KEYS+['fold']].merge(reference,on=data.KEYS,suffixes=('_expected',''),validate='one_to_one')
    if not joined.fold.eq(joined.fold_expected).all():raise ValueError('Original painting folds changed')
    banks=[]
    with np.load(directory/'features.npz',allow_pickle=False) as archive:
        names=archive['names'].tolist()
        for name,width in (('clip',512),('siglip',768),('large',768)):
            values=archive[name]
            if values.shape!=(len(names),width) or values.dtype!=np.float32 or not np.allclose(np.linalg.norm(values,axis=1),1.,atol=3e-6,rtol=0.):
                raise ValueError('Invalid frozen feature normalization/dimensions')
            banks.append(dict(zip(names,values.copy())))
    if set(names)!=set(train.image):raise ValueError('Image feature coverage differs')
    device=('cuda' if torch.cuda.is_available() else 'cpu') if args.device=='auto' else args.device
    if device=='cuda' and not torch.cuda.is_available():raise StageBlocked('CUDA unavailable: use CPU feasibility or Colab')
    source=sorted((ROOT/'src/st3').glob('*.py'))+[ROOT/'scripts/run_exp022.py',ROOT/'scripts/run_exp022.ps1']
    identity={'config_sha256':file_sha256(ROOT/'configs/exp022.json'),'input_manifest_sha256':file_sha256(directory/'run.json'),
        'source_sha256':{str(p.relative_to(ROOT)):file_sha256(p) for p in source},'device':device,'python':platform.python_version(),
        'versions':{n:importlib.metadata.version(n) for n in ('torch','transformers','numpy','pandas','scikit-learn','Pillow')},
        'gpu':torch.cuda.get_device_name() if device=='cuda' else None,'cuda':torch.version.cuda,
        'allow_tf32':torch.backends.cuda.matmul.allow_tf32,'cudnn_allow_tf32':torch.backends.cudnn.allow_tf32,
        'matmul_precision':torch.get_float32_matmul_precision()}
    return SimpleNamespace(config=cfg,identity=identity,train=train,reference=reference,parent_predictions=parent,banks=banks,
        device=torch.device(device),data_root=args.portable_dir/'data' if args.portable_dir else args.data_dir,
        input_dir=directory,historical=json.loads((directory/'historical_fits.json').read_text()))


def historical_snapshot(args,output):
    if args.portable_dir:
        json_output({},output);return 0
    own=args.run_dir.resolve();paths=[]
    for p in (ROOT/'runs').rglob('*'):
        if p.is_file() and own not in p.resolve().parents:paths.append(p)
    for p in args.cache_dir.rglob('*'):
        if p.is_file() and 'huggingface' not in p.parts:paths.append(p)
    values={str(p.resolve()):file_sha256(p) for p in paths};json_output(values,output);return len(values)


def preserve_history(directory):
    values=json.loads((directory/'historical_sha256.json').read_text())
    for name,digest in values.items():
        if file_sha256(Path(name))!=digest:raise ValueError('Historical cache/artifact changed: '+name)
    return len(values)
