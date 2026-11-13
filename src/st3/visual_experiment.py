"""Four-condition nested screen; completed model boundaries are immutable."""

import json
import time

import numpy as np
import pandas as pd
import torch

from . import data
from .ablation import paired_bootstrap
from .baseline import file_sha256
from .caption_data import inner_partition,require_isolation
from .cli import json_output
from .contrast_study import new_output
from .fusion import blend
from .fusion_study import write_cv_analysis
from .low_rank_study import score_result
from .polarity_workflow import ExecutionBudget,StageBlocked,read_sealed,seal
from .visual_feasibility import load_context
from .visual_lora import VisualLoRA
from .visual_training import CONDITIONS,fit_model,predict,save_checkpoint


def scope(ctx,phase):
    if phase=='mini':return ctx.config['mini_folds'],ctx.config['mini_seeds']
    if phase=='screen':return ctx.config['screen_folds'],ctx.config['screen_seeds']
    return list(range(1,6)),ctx.config['training']['seeds']


def checkpoint_path(args,fold,condition,seed):return args.run_dir/'work'/f'fold_{fold}'/f'{condition}_{seed}'


def fit_seed(args,ctx,vision,fold,seed,condition,micro,budget):
    fit=ctx.train[ctx.train.fold.ne(fold)];held=ctx.train[ctx.train.fold.eq(fold)].sort_values(data.KEYS)
    inner,stopping=inner_partition(fit,ctx.config['training'],fold);require_isolation(inner,stopping,held)
    directory=checkpoint_path(args,fold,condition,seed)
    if not directory.exists():
        with new_output(directory) as stage:
            head,curves,gradients,info=fit_model(ctx,vision,fit,held,fold,seed,condition,micro,budget)
            prediction,pred_info=predict(ctx,vision,head,held,condition,micro)
            checkpoint_tick=time.perf_counter()
            save_checkpoint(stage/'student.pt',head,vision)
            prediction.to_csv(stage/'predictions.csv',index=False,lineterminator='\n')
            pd.DataFrame(curves).to_csv(stage/'curves.csv',index=False,lineterminator='\n')
            json_output(gradients,stage/'gradients.json')
            seal(stage,ctx.identity,{**info,'fold':fold,'seed':seed,'condition':condition,'microbatch':micro,
                'prediction_seconds':pred_info['seconds'],'checkpoint_seconds':time.perf_counter()-checkpoint_tick,
                'preflight_manifest_sha256':file_sha256(args.run_dir/'preflight/run.json')})
            del head
    info=read_sealed(directory,ctx.identity)
    for key,frame in (('fit',fit),('inner',inner),('stopping',stopping),('held',held)):
        if info[key+'_paintings']!=sorted(frame.painting.unique()):raise ValueError('Stored training isolation mismatch')
    if (info['fold'],info['seed'],info['condition'],info['microbatch'])!=(fold,seed,condition,micro):raise ValueError('Stored checkpoint identity mismatch')
    if info['preflight_manifest_sha256']!=file_sha256(args.run_dir/'preflight/run.json'):raise ValueError('Preflight identity changed')
    prediction=pd.read_csv(directory/'predictions.csv',float_precision='round_trip')
    data.require_same_pairs(held,prediction)
    print(f"Completed/reused fold {fold}, {condition}, seed {seed}, selected epoch {info['selected_epoch']}",flush=True)
    return prediction,info


def aggregate_predictions(args,ctx,phase):
    folds,seeds=scope(ctx,phase);all_predictions=[];infos=[];manifests={}
    historical=pd.read_csv(ctx.input_dir/'historical_seeds.csv',float_precision='round_trip')
    for fold in folds:
        base=ctx.reference[ctx.reference.fold.eq(fold)][data.KEYS+['p_positive']].sort_values(data.KEYS).reset_index(drop=True)
        predictions={'R0':base}
        for condition in CONDITIONS:
            values=[]
            for seed in seeds:
                directory=checkpoint_path(args,fold,condition,seed);info=read_sealed(directory,ctx.identity)
                frame=pd.read_csv(directory/'predictions.csv',float_precision='round_trip').sort_values(data.KEYS).reset_index(drop=True)
                data.require_same_pairs(base,frame)
                if not frame[data.KEYS].equals(base[data.KEYS]):raise ValueError('Checkpoint prediction ordering differs')
                values.append(frame.p_positive.to_numpy());infos.append(info)
                manifests[f'{fold}/{condition}_{seed}']=file_sha256(directory/'run.json')
            pred=base[data.KEYS].assign(p_positive=np.round(np.mean(values,axis=0),6))
            predictions[condition]=pred;predictions['E'+condition]=blend(base,pred,.5)
        # Same folds AND seed population; historical full-five-fold aggregates are not used.
        for name in ('D0','D8'):
            old=historical[historical.fold.eq(fold)&historical.seed.isin(seeds)&historical.model.eq(name)]
            if old.groupby(data.KEYS).size().ne(len(seeds)).any():raise ValueError('Incomplete same-seed historical context')
            pred=old.groupby(data.KEYS,sort=True).p_positive.mean().round(6).reset_index()
            data.require_same_pairs(base,pred);predictions['H'+name]=pred
        for name,pred in predictions.items():all_predictions.append(pred.assign(fold=fold,model=name))
    return pd.concat(all_predictions,ignore_index=True),infos,manifests


def write_analysis(stage,ctx,phase,predictions,infos):
    folds,_=scope(ctx,phase);subset=ctx.train[ctx.train.fold.isin(folds)];records={n:[] for n in ('predictions','paintings','languages')};rows=[]
    for (fold,name),pred in predictions.groupby(['fold','model'],sort=True):
        c,l=score_result(subset[subset.fold.eq(fold)],pred[data.KEYS+['p_positive']],{'fold':fold,'model':name},records)
        own=[i for i in infos if i['fold']==fold and i['condition']==name]
        rows.append({'fold':fold,'model':name,'contrast':c,'log_loss':l,'fit_seconds':sum(i['fit_seconds'] for i in own),
                     'predict_seconds':sum(i['prediction_seconds'] for i in own),'converged':True,
                     'seconds':sum(i['fit_seconds']+i['prediction_seconds'] for i in own)})
    summary,_=write_cv_analysis(stage,subset,records,pd.DataFrame(rows),ctx.config['training'],'R0','R0')
    painting=pd.concat(records['paintings']);wide=painting[painting.status.eq('scored')].pivot(index=['fold','painting'],columns='model',values='contrast').reset_index()
    effects={f'{a}-{b}':wide[a]-wide[b] for a,b in (('L0','F0'),('L8','F8'),('F8','F0'),('L8','L0'),('EL0','EF0'),('EL8','EF8'))}
    effects['factorial_interaction']=effects['L8-L0']-effects['F8-F0']
    paired=wide[['fold','painting']].assign(status='scored',**effects)
    draws,intervals=paired_bootstrap(paired,ctx.config['bootstrap_replicates'],ctx.config['bootstrap_seed'],columns=list(effects))
    paired.to_csv(stage/'factorial_paintings.csv',index=False,lineterminator='\n')
    draws.to_csv(stage/'factorial_bootstrap.csv',index=False,lineterminator='\n');json_output(intervals,stage/'factorial_bootstrap.json')
    prediction_variance=predictions.groupby(['model','language']).p_positive.agg(['mean','std','min','max'])
    prediction_variance.to_csv(stage/'prediction_variance.csv',lineterminator='\n')
    json_output(infos,stage/'fits.json')
    language=pd.concat(records['languages']).pivot(index=['fold','language'],columns='model',values='log_loss')
    for a,b in (('L0','F0'),('L8','F8'),('F8','F0'),('L8','L0')):language[a+'-'+b]=language[a]-language[b]
    language.to_csv(stage/'factorial_languages.csv',lineterminator='\n')
    return summary


def run_study(args,phase):
    # Gate before loading the expensive visual model.
    from .visual_lora_inputs import inputs
    from .visual_verification import verified_decision
    ctx=inputs(args);pre=read_sealed(args.run_dir/'preflight',ctx.identity);pilot=read_sealed(args.run_dir/'pilot',ctx.identity)
    if pre['status']!='PASS' or pilot['status']!='PILOT_PASS':raise StageBlocked('Successful preflight and pilot required')
    projection=pilot['runtime_projection']
    if phase in ('screen','cv'):
        mini=verified_decision(args,'mini',ctx)
        if mini['status']!='MINISCREEN_SANITY_PASS':raise StageBlocked('MiniScreen requires review')
    if phase=='screen' and not projection['local_screen_within_gpu_budget']:
        raise StageBlocked('Measured Screen forecast exceeds four GPU-hours on this runtime; use the Colab option and rerun Preflight/Pilot there')
    if phase=='cv':
        result=verified_decision(args,'screen',ctx)
        if not args.allow_full_cv or not (result['lora_promising'] or result['auxiliary_promising']):
            raise StageBlocked('CV requires verified promising Screen AND explicit --allow-full-cv owner invocation')
    if (args.run_dir/phase).exists():read_sealed(args.run_dir/phase,ctx.identity);print('Reused completed '+phase);return
    estimate=projection['miniscreen_seconds'] if phase=='mini' else projection['screen_seconds']*(3.75 if phase=='cv' else 1.)
    print(f'Frozen forecast for {phase}: {estimate/3600:.2f} GPU-hours; budget {args.max_seconds or 900.}s',flush=True)
    if args.max_seconds==0 and estimate>ctx.config['automatic_seconds_limit']:
        raise StageBlocked('Forecast exceeds automatic 15-minute limit; owner must invoke with an explicit --max-seconds budget')
    budget=ExecutionBudget(args.max_seconds or 900.);ctx,model=load_context(args);vision=VisualLoRA(model,ctx.config)
    folds,seeds=scope(ctx,phase)
    for fold in folds:
        for seed in seeds:
            for condition in CONDITIONS:
                budget.check(f'{fold}/{seed}/{condition}',-1)
                fit_seed(args,ctx,vision,fold,seed,condition,pre['microbatch'],budget)
    predictions,infos,manifests=aggregate_predictions(args,ctx,phase)
    with new_output(args.run_dir/phase) as stage:
        write_analysis(stage,ctx,phase,predictions,infos)
        seal(stage,ctx.identity,{'status':'UNVERIFIED','folds':folds,'seeds':seeds,'seed_manifests':manifests,
            'preflight_sha256':file_sha256(args.run_dir/'preflight/run.json'),'pilot_sha256':file_sha256(args.run_dir/'pilot/run.json'),
            'parent_decision_sha256':file_sha256(args.run_dir/('mini_decision' if phase=='screen' else 'screen_decision')/'run.json') if phase!='mini' else None})
    print(f'{phase} complete; independent verification required before any decision',flush=True)
