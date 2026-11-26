"""Independent checkpoint replay; any failed attempt revokes the current gate."""

import json
import time

import numpy as np
import pandas as pd
import torch

from . import data
from .baseline import file_sha256
from .cli import json_output
from .contrast_study import new_output
from .emotion_verification import compare_diagnostics
from .gradient_training import state_hash
from .metrics import score
from .polarity_workflow import ExecutionBudget,StageBlocked,read_sealed,seal
from .visual_experiment import aggregate_predictions,checkpoint_path,scope
from .visual_feasibility import load_context
from .visual_gates import competition_gate,screening_gate
from .visual_groups import PaintingGroups,make_head
from .visual_lora import VisualLoRA,adapter_state
from .visual_lora_audit import feature_comparison,gradient_audit
from .visual_lora_inputs import inputs,preserve_history
from .visual_training import CONDITIONS,load_checkpoint,predict,update_norm


def decision_from_files(args,ctx,phase,healthy):
    summary=pd.read_csv(args.run_dir/phase/'summary.csv',index_col='model',float_precision='round_trip')
    folds=pd.read_csv(args.run_dir/phase/'fold_metrics.csv',float_precision='round_trip')
    return competition_gate(summary,ctx.config['training'],verified=True,healthy=healthy) if phase=='cv' else screening_gate(
        summary,folds,ctx.config,verified=True,healthy=healthy,mini=phase=='mini')


def verified_decision(args,phase,ctx=None):
    ctx=ctx or inputs(args);study=read_sealed(args.run_dir/phase,ctx.identity)
    verification=read_sealed(args.run_dir/(phase+'_verification'),ctx.identity)
    decision=read_sealed(args.run_dir/(phase+'_decision'),ctx.identity)
    attempts=sorted((args.run_dir/(phase+'_verification_attempts')).glob('*.json'))
    last=json.loads(attempts[-1].read_text()) if attempts else {};digest=file_sha256(args.run_dir/phase/'run.json')
    if last.get('status')!='PASS' or last.get('study_sha256')!=digest:raise StageBlocked('Latest replay failed/interrupted; current successful decision is invalid until replay passes')
    if verification['study_sha256']!=digest or decision['study_sha256']!=digest or decision['verification_sha256']!=file_sha256(args.run_dir/(phase+'_verification')/'run.json'):
        raise ValueError('Decision evidence changed')
    for name in ('preflight','pilot'):
        if study[name+'_sha256']!=file_sha256(args.run_dir/name/'run.json'):raise ValueError('Study prerequisite changed')
        read_sealed(args.run_dir/name,ctx.identity)
    if phase!='mini':
        parent='mini' if phase=='screen' else 'screen'
        prerequisite=verified_decision(args,parent,ctx)
        if study['parent_decision_sha256']!=file_sha256(args.run_dir/(parent+'_decision')/'run.json'):raise ValueError('Study parent decision changed')
        if (parent=='mini' and prerequisite['status']!='MINISCREEN_SANITY_PASS') or (parent=='screen' and not (prerequisite['lora_promising'] or prerequisite['auxiliary_promising'])):
            raise StageBlocked('Parent decision no longer permits this stage')
    for key,value in study['seed_manifests'].items():
        fold,name=key.split('/');path=args.run_dir/'work'/f'fold_{fold}'/name
        read_sealed(path,ctx.identity)
        if file_sha256(path/'run.json')!=value:raise ValueError('Verified checkpoint changed')
    expected=decision_from_files(args,ctx,phase,verification['healthy'])
    if any(decision.get(k)!=v for k,v in expected.items()):raise ValueError('Frozen decision gate differs')
    return decision


def replay_seed(args,ctx,vision,fold,seed,condition,pre):
    path=checkpoint_path(args,fold,condition,seed);info=read_sealed(path,ctx.identity)
    fit=ctx.train[ctx.train.fold.ne(fold)];held=ctx.train[ctx.train.fold.eq(fold)]
    from .caption_data import inner_partition,require_isolation
    inner,stop=inner_partition(fit,ctx.config['training'],fold);require_isolation(inner,stop,held)
    for name,frame in (('fit',fit),('inner',inner),('stopping',stop),('held',held)):
        if info[name+'_paintings']!=sorted(frame.painting.unique()):raise ValueError('Replay painting identity mismatch')
    if info['preflight_manifest_sha256']!=file_sha256(args.run_dir/'preflight/run.json'):raise ValueError('Replay preflight differs')
    if not 1<=info['selected_epoch']<=ctx.config['training']['max_epochs']:raise ValueError('Invalid stopping epoch')
    curves=pd.read_csv(path/'curves.csv',float_precision='round_trip');selection=curves[curves.phase.eq('selection')]
    if int(selection.loc[selection.validation_bce.idxmin(),'epoch'])!=info['selected_epoch'] or len(curves[curves.phase.eq('refit')])!=info['selected_epoch']:
        raise ValueError('Nested stopping/refit curves differ')
    head,optimizer,scaler=make_head(ctx,vision,sorted(inner.language.unique()),seed,condition)
    if state_hash(head)!=info['initialization_sha256']['selection']:raise ValueError('Selection initialization mismatch')
    del head,optimizer,scaler
    head,optimizer,scaler=make_head(ctx,vision,sorted(fit.language.unique()),seed,condition)
    if state_hash(head)!=info['initialization_sha256']['refit']:raise ValueError('Refit initialization mismatch')
    del head,optimizer,scaler
    initial=adapter_state(vision);head=load_checkpoint(ctx,vision,path/'student.pt',seed,condition)
    if vision.frozen_digest()!=pre['base_sha256'] or info['base_sha256']!=pre['base_sha256']:raise ValueError('Pretrained base changed')
    delta=update_norm(vision,initial)
    np.testing.assert_allclose(delta,info['adapter_update_norm'],atol=1e-12,rtol=0.)
    if (condition.startswith('L') and delta<=0) or (condition.startswith('F') and delta!=0):raise ValueError('Invalid adapter update')
    predicted,_=predict(ctx,vision,head,held,condition,pre['microbatch'])
    expected=pd.read_csv(path/'predictions.csv',float_precision='round_trip').sort_values(data.KEYS).reset_index(drop=True)
    if not predicted[data.KEYS].equals(expected[data.KEYS]):raise ValueError('Replay prediction keys differ')
    if not np.array_equal(predicted.p_positive.to_numpy(),expected.p_positive.to_numpy()):raise ValueError('Replay prediction mismatch')
    rows=PaintingGroups(fit,head.languages);indices=np.array(info['refit_diagnostic_indices'])
    diagnostic=gradient_audit(ctx,vision,head,rows,indices,condition)
    saved=json.loads((path/'gradients.json').read_text())
    compare_diagnostics(diagnostic,saved['refit_final'])
    names=rows.batch(indices,ctx.device)['names']
    with torch.no_grad():comparison=feature_comparison(vision(ctx.pixels(names,ctx.device)),np.stack([ctx.banks[0][n] for n in names]))
    compare_diagnostics(comparison,info['representation_shift'])
    if condition.startswith('F') and comparison['max_abs_error']>ctx.config['cache_feature_atol']:raise ValueError('Frozen live/cache parity failed during replay')
    constants=predicted.groupby('language').p_positive.agg(['count','std'])
    eligible=constants['count'].ge(2)
    collapse=float(constants.loc[eligible,'std'].fillna(0).lt(1e-8).mean())
    return {'fold':fold,'seed':seed,'condition':condition,'adapter_update_norm':delta,
        'constant_language_fraction':collapse,'healthy':collapse<ctx.config['major_constant_prediction_fraction'],
        'initialization_sha256':info['initialization_sha256'],'gradients':diagnostic}


def run_verify(args,phase):
    # STARTED is durable before context/loading; even a source mismatch revokes PASS.
    attempts=args.run_dir/(phase+'_verification_attempts');attempts.mkdir(parents=True,exist_ok=True)
    previous=sorted(attempts.glob('*.json'));number=int(previous[-1].stem)+1 if previous else 1
    attempt=attempts/f'{number:08}.json';json_output({'status':'STARTED'},attempt)
    start=time.perf_counter()
    try:
        ctx,model=load_context(args);study=read_sealed(args.run_dir/phase,ctx.identity)
        if phase!='mini':verified_decision(args,'mini' if phase=='screen' else 'screen',ctx)
        pre=read_sealed(args.run_dir/'preflight',ctx.identity);vision=VisualLoRA(model,ctx.config)
        folds,seeds=scope(ctx,phase);checks=[];budget=ExecutionBudget(args.max_seconds or 900.)
        if study['folds']!=folds or study['seeds']!=seeds:raise ValueError('Incomplete screening seed/fold coverage')
        for fold in folds:
            for seed in seeds:
                matching=[]
                for condition in CONDITIONS:
                    budget.check(f'Replay {fold}/{seed}/{condition}',-1)
                    key=f'{fold}/{condition}_{seed}'
                    if file_sha256(checkpoint_path(args,fold,condition,seed)/'run.json')!=study['seed_manifests'][key]:raise ValueError('Study checkpoint changed')
                    check=replay_seed(args,ctx,vision,fold,seed,condition,pre);checks.append(check);matching.append(check['initialization_sha256'])
                if any(v!=matching[0] for v in matching):raise ValueError('Unmatched four-condition starting heads')
        rebuilt,_,manifests=aggregate_predictions(args,ctx,phase)
        if manifests!=study['seed_manifests']:raise ValueError('Checkpoint inventory mismatch')
        saved=pd.read_csv(args.run_dir/phase/'predictions.csv',float_precision='round_trip')
        sort=['fold','model',*data.KEYS]
        pd.testing.assert_frame_equal(rebuilt.sort_values(sort).reset_index(drop=True),saved[rebuilt.columns].sort_values(sort).reset_index(drop=True),check_exact=True)
        summary=pd.read_csv(args.run_dir/phase/'summary.csv',index_col='model',float_precision='round_trip')
        metrics=pd.read_csv(args.run_dir/phase/'fold_metrics.csv',float_precision='round_trip')
        seed_metrics=[]
        for name,pred in rebuilt.groupby('model'):
            subset=ctx.train[ctx.train.fold.isin(folds)];r=score(subset,pred[data.KEYS+['p_positive']])
            np.testing.assert_allclose([r['cultural_contrast'],r['log_loss']],summary.loc[name,['pooled_contrast','pooled_log_loss']].astype(float),atol=1e-14,rtol=0.)
            actual=[]
            for fold in folds:
                s=score(subset[subset.fold.eq(fold)],pred[pred.fold.eq(fold)][data.KEYS+['p_positive']]);actual.append(s)
                original=metrics[metrics.model.eq(name)&metrics.fold.eq(fold)].iloc[0]
                np.testing.assert_allclose([s['cultural_contrast'],s['log_loss']],[original.contrast,original.log_loss],atol=1e-14,rtol=0.)
            np.testing.assert_allclose([np.mean([v['cultural_contrast'] for v in actual]),np.mean([v['log_loss'] for v in actual])],summary.loc[name,['contrast_mean','log_loss_mean']].astype(float),atol=1e-14,rtol=0.)
        for fold in folds:
            held=ctx.train[ctx.train.fold.eq(fold)];base=ctx.reference[ctx.reference.fold.eq(fold)][data.KEYS+['p_positive']]
            from .fusion import blend
            for seed in seeds:
                for condition in CONDITIONS:
                    pred=pd.read_csv(checkpoint_path(args,fold,condition,seed)/'predictions.csv',float_precision='round_trip')
                    for name,p in ((condition,pred.assign(p_positive=pred.p_positive.round(6))),('E'+condition,blend(base,pred.assign(p_positive=pred.p_positive.round(6)),.5))):
                        r=score(held,p);seed_metrics.append({'fold':fold,'seed':seed,'model':name,'contrast':r['cultural_contrast'],'log_loss':r['log_loss']})
        count=preserve_history(args.run_dir/'preflight');healthy=all(c['healthy'] for c in checks)
        verification=args.run_dir/(phase+'_verification');digest=file_sha256(args.run_dir/phase/'run.json')
        if not verification.exists():
            with new_output(verification) as stage:
                json_output(checks,stage/'replay_checks.json');pd.DataFrame(seed_metrics).to_csv(stage/'seed_metrics.csv',index=False,lineterminator='\n')
                pd.concat([pd.read_csv(checkpoint_path(args,c['fold'],c['condition'],c['seed'])/'curves.csv').assign(fold=c['fold'],seed=c['seed'],condition=c['condition']) for c in checks]).to_csv(stage/'training_curves.csv',index=False,lineterminator='\n')
                seal(stage,ctx.identity,{'status':'PASS','healthy':healthy,'checkpoints_replayed':len(checks),'protected_files_matched':count,
                    'study_sha256':digest,'seconds':time.perf_counter()-start,'locked_data_accessed':False})
        else:read_sealed(verification,ctx.identity)
        decision=decision_from_files(args,ctx,phase,healthy);output=args.run_dir/(phase+'_decision')
        if not output.exists():
            with new_output(output) as stage:
                seal(stage,ctx.identity,{**decision,'study_sha256':digest,'verification_sha256':file_sha256(verification/'run.json')})
        json_output({'status':'PASS','study_sha256':digest},attempt);verified_decision(args,phase,ctx)
        print(f"{phase}: independent replay PASS; {decision['status']}",flush=True)
    except BaseException as error:
        json_output({'status':'FAILED','error':str(error)},attempt);raise
