"""Bounded real-image numerical, gradient and RTX memory preflight/pilot."""

import gc
import time

import numpy as np
import torch

from .caption_data import inner_partition,require_isolation
from .baseline import file_sha256
from .data import image_path
from .cli import json_output
from .contrast_study import new_output
from .emotion_experiment import memory_report
from .gradient_training import state_hash
from .polarity_workflow import ExecutionBudget,StageBlocked,read_sealed,seal
from .visual_groups import PaintingGroups,PixelReader,features,grouped_step,make_head,train_epoch
from .visual_lora import VisualLoRA,adapter_state,load_visual
from .visual_lora_audit import feature_comparison,gradient_audit
from .visual_lora_inputs import historical_snapshot,inputs,preserve_history
from .visual_training import CONDITIONS,predict,update_norm


def load_context(args):
    ctx=inputs(args)
    ctx.free_before_load=torch.cuda.mem_get_info()[0] if ctx.device.type=='cuda' else None
    ctx.total_vram=torch.cuda.get_device_properties(ctx.device).total_memory if ctx.device.type=='cuda' else None
    cache=args.portable_dir/'cache' if args.portable_dir else args.cache_dir
    for image,digest in ctx.train[['image','image_sha256']].drop_duplicates().itertuples(index=False,name=None):
        if file_sha256(image_path(ctx.data_root,image))!=digest:raise ValueError('Training image bytes changed: '+image)
    model,processor,ctx.encoder_info=load_visual(cache,ctx.config,ctx.device)
    ctx.pixels=PixelReader(ctx.data_root,processor)
    if getattr(args,'pixel_cache','off')=='ram':
        from .pixel_cache import RamPixels
        ctx.pixels=RamPixels(ctx.pixels,ctx.train.image.unique(),max_bytes=int(args.pixel_cache_max_gib*1024**3),margin_bytes=int(args.pixel_cache_margin_gib*1024**3))
    return ctx,model


def choose_microbatch(candidates,attempt,free_bytes,margin):
    """Try each predefined size once; failed/OOM candidates never become PASS."""
    trials=[]
    for size in candidates:
        try:
            result=attempt(size)
            safe=free_bytes is None or result['peak_cuda_reserved_bytes']<=free_bytes-margin
            trials.append({'microbatch':size,'status':'PASS' if safe else 'INSUFFICIENT_MARGIN',**result})
            if safe:return size,trials
        except torch.cuda.OutOfMemoryError:
            gc.collect()
            if torch.cuda.is_available():torch.cuda.empty_cache()
            trials.append({'microbatch':size,'status':'CUDA_OOM'})
    return None,trials


def representative(ctx):
    fit=ctx.train[ctx.train.fold.ne(1)];held=ctx.train[ctx.train.fold.eq(1)]
    inner,stopping=inner_partition(fit,ctx.config['training'],1)
    require_isolation(inner,stopping,held)
    return inner,stopping


def optimizer_bytes(optimizer):
    return sum(v.numel()*v.element_size() for state in optimizer.state.values() for v in state.values() if torch.is_tensor(v))


def run_preflight(args):
    start=time.perf_counter();ctx,model=load_context(args);directory=args.run_dir/'preflight'
    if directory.exists():
        result=read_sealed(directory,ctx.identity)
        if result['status']!='PASS':raise StageBlocked('Preflight blocked; read the feasibility report before changing hardware')
        print('Reused sealed preflight',flush=True);return
    budget=ExecutionBudget(args.max_seconds or 600.);inner,_=representative(ctx)
    population=inner[['painting','image']].drop_duplicates().sort_values('painting').image.tolist()
    names=np.random.default_rng(222027).choice(population,size=ctx.config['parity_images'],replace=False).tolist()
    cached=np.stack([ctx.banks[0][name] for name in names]);original=[]
    for at in range(0,len(names),4):
        with torch.no_grad():original.append(torch.nn.functional.normalize(model(ctx.pixels(names[at:at+4],ctx.device)).image_embeds.float(),dim=-1).cpu())
    original=torch.cat(original);vision=VisualLoRA(model,ctx.config);base=vision.frozen_digest();zero=[]
    for at in range(0,len(names),4):
        with torch.no_grad():zero.append(vision(ctx.pixels(names[at:at+4],ctx.device)).cpu())
    parity={'frozen_vs_cache':feature_comparison(original,cached),'zero_lora_vs_frozen':feature_comparison(torch.cat(zero),original),
        'image_mapping':dict(zip(names,inner.drop_duplicates('image').set_index('image').loc[names].image_sha256)),
        'encoder':ctx.encoder_info,'fusion_width':2048,'fusion_normalization':'Each block unit L2; concatenate / float32(sqrt(3)); no second block normalization'}
    for key,tolerance in (('frozen_vs_cache','cache_feature_atol'),('zero_lora_vs_frozen','zero_adapter_atol')):
        value=parity[key]
        if value['max_abs_error']>ctx.config[tolerance] or value['cosine_min']<ctx.config['feature_cosine_min'] or value['norm_max_error']>3e-6:
            # Numerical failures are evidence, but are not valid stage manifests.
            json_output(parity,args.run_dir/'failed_preprocessing_parity.json')
            raise ValueError('Preprocessing parity failed: '+key)
    rows=PaintingGroups(inner,sorted(inner.language.unique()));sample=rows.order(2027,0)[:4]
    def attempt(size):
        head,optimizer,scaler=make_head(ctx,vision,sorted(inner.language.unique()),2027,'L8')
        if ctx.device.type=='cuda':torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats()
        tick=time.perf_counter();timings=[]
        try:
            for _ in range(ctx.config['probe_steps']):timings.append(grouped_step(ctx,vision,head,rows,sample,size,'L8',optimizer,scaler))
            elapsed=time.perf_counter()-tick;count=len(sample)*len(timings)
            pairs=len(rows.batch(sample,ctx.device)['rows'])*len(timings)
            return {**memory_report(),'seconds':elapsed,'iteration_seconds':elapsed/len(timings),
                'images_per_second':count/elapsed,'pairs_per_second':pairs/elapsed,'optimizer_bytes':optimizer_bytes(optimizer),
                'timing':{key:sum(t[key] for t in timings) for key in ('image_loading','encoder_forward','head_forward','head_backward','encoder_backward','backward')}}
        finally:del head,optimizer,scaler
    selected,trials=choose_microbatch(ctx.config['microbatch_candidates'],attempt,ctx.free_before_load,ctx.config['memory_margin_bytes'])
    diagnostics={};initial_states={};initial_logits={}
    if selected:
        for condition in CONDITIONS:
            budget.check(condition+' probe',-1)
            head,optimizer,scaler=make_head(ctx,vision,sorted(inner.language.unique()),2027,condition)
            initial_states[condition]=state_hash(head);initial_adapters=adapter_state(vision)
            initial_logits[condition]=predict(ctx,vision,head,rows.frame.iloc[rows.batch(sample,ctx.device)['rows']],condition,selected)[0].p_positive.to_numpy()
            before=gradient_audit(ctx,vision,head,rows,sample[:selected],condition)
            traces=[grouped_step(ctx,vision,head,rows,sample,selected,condition,optimizer,scaler) for _ in range(ctx.config['probe_steps'])]
            after=gradient_audit(ctx,vision,head,rows,sample[:selected],condition);delta=update_norm(vision,initial_adapters)
            if condition.startswith('L') and delta<=0:raise ValueError('No LoRA update after multiple diagnostic steps')
            if condition.startswith('F') and delta!=0:raise ValueError('Frozen control adapters updated')
            if base!=vision.frozen_digest():raise ValueError('Base weights changed during probe')
            names_probe=rows.batch(sample[:selected],ctx.device)['names']
            with torch.no_grad():shift=feature_comparison(vision(ctx.pixels(names_probe,ctx.device)),np.stack([ctx.banks[0][n] for n in names_probe]))
            # A common target language isolates image sensitivity from language offsets.
            with torch.no_grad():
                z,_,_=features(ctx,vision,rows.batch(sample,ctx.device)['names'],condition.startswith('L'))
                head.eval();logits,_=head(z,torch.zeros(len(sample),dtype=torch.long,device=ctx.device))
                variation=float(logits.sigmoid().std())
            if variation<=0:raise ValueError('Image-insensitive sentiment inference')
            diagnostics[condition]={'initial':before,'after_steps':after,'adapter_update_norm':delta,
                'representation_shift':shift,'same_language_prediction_std':variation,'steps':traces,
                'trainable_head_parameters':sum(p.numel() for p in head.parameters()),
                'trainable_adapter_parameters':sum(p.numel() for p in vision.adapters().values() if p.requires_grad)}
            del head,optimizer,scaler
        if len(set(initial_states.values()))!=1:raise ValueError('Four conditions have unmatched head initialization')
        for name,values in initial_logits.items():
            if np.max(np.abs(values-initial_logits['F0']))>ctx.config['initial_logit_atol']:raise ValueError('Initial sentiment prediction mismatch: '+name)
    with new_output(directory) as stage:
        count=historical_snapshot(args,stage/'historical_sha256.json')
        json_output(parity,stage/'preprocessing.json');json_output(diagnostics,stage/'gradients.json')
        json_output(trials,stage/'memory_trials.json');json_output(vision.targets,stage/'target_modules.json')
        seal(stage,ctx.identity,{'status':'PASS' if selected else 'BLOCKED_MEMORY','microbatch':selected,'effective_paintings':4,
            'gradient_accumulation':4//selected if selected else None,'gpu_name':ctx.identity['gpu'],'total_vram_bytes':ctx.total_vram,
            'free_before_model_load_bytes':ctx.free_before_load,'base_sha256':base,'adapter_parameters':sum(p.numel() for p in vision.adapters().values()),
            'initial_head_sha256':initial_states,'protected_files':count,'seconds':time.perf_counter()-start,
            'frozen_visual_parameters':sum(p.numel() for n,p in vision.named_parameters() if n not in vision.adapters()),
            'frozen_visual_bytes':sum(p.numel()*p.element_size() for n,p in vision.named_parameters() if n not in vision.adapters()),
            'locked_annotation_rows_retained':0,'development_gold_accessed':False,'vision_precision':'fp32','head_precision':ctx.config['head_precision']})
    print(f'Preflight: {"PASS" if selected else "BLOCKED_MEMORY"}; microbatch {selected}; {time.perf_counter()-start:.1f}s',flush=True)
    if not selected:raise StageBlocked('None of 4/2/1 fits the fixed memory margin; use Colab handoff')


def runtime_projection(results,historical,cfg):
    # Prior stopping epochs are a scheduling prior, not a score or selection rule.
    q=cfg['runtime_quantile'];selected=int(np.ceil(np.quantile([r['selected_epoch'] for r in historical],q)))
    selected=min(cfg['training']['max_epochs'],selected)
    stop=min(cfg['training']['max_epochs'],selected+cfg['training']['patience'])
    costs={};ceilings={}
    for name,value in results.items():
        epoch=value['epoch'];per_image=epoch['seconds']/epoch['paintings']
        costs[name]=cfg['runtime_margin']*(stop*(epoch['seconds']+value['validation']['seconds'])+selected*per_image*1068+
            2*value['validation']['seconds']+value['diagnostic_seconds']+5.)
        maximum=cfg['training']['max_epochs']
        ceilings[name]=cfg['runtime_margin']*(maximum*(epoch['seconds']+value['validation']['seconds'])+maximum*per_image*1068+
            2*value['validation']['seconds']+value['diagnostic_seconds']+5.)
    mini=sum(costs.values());screen=4*mini
    return {'method':'p90 historical selected epoch + fixed patience; measured full inner epoch / validation; 25% timing margin',
        'planning_selected_epochs':selected,'planning_selection_epochs':stop,'per_model_seconds':costs,
        'miniscreen_seconds':mini,'screen_seconds':screen,'screen_gpu_hours':screen/3600.,
        'maximum_epoch_miniscreen_seconds':sum(ceilings.values()),'maximum_epoch_screen_gpu_hours':4*sum(ceilings.values())/3600.,
        'screen_incremental_after_mini_seconds':screen-mini,'miniscreen_automatic_allowed':mini<=cfg['automatic_seconds_limit'],
        'local_screen_within_gpu_budget':screen/3600.<=cfg['screen_gpu_hours_limit'],
        'warning':'Matched grouped optimization may stop differently; this is a scheduling estimate, not a runtime guarantee.'}


def run_pilot(args):
    start=time.perf_counter();ctx,model=load_context(args);pre=read_sealed(args.run_dir/'preflight',ctx.identity)
    if pre['status']!='PASS':raise StageBlocked('Preflight must pass')
    output=args.run_dir/'pilot'
    if output.exists():read_sealed(output,ctx.identity);print('Reused sealed pilot',flush=True);return
    vision=VisualLoRA(model,ctx.config)
    if vision.frozen_digest()!=pre['base_sha256']:raise ValueError('Pinned base changed')
    micro=pre['microbatch'];inner,stopping=representative(ctx);results={};budget=ExecutionBudget(args.max_seconds or 900.)
    for condition in CONDITIONS:
        head,optimizer,scaler=make_head(ctx,vision,sorted(inner.language.unique()),2027,condition)
        rows=PaintingGroups(inner,head.languages);initial=adapter_state(vision)
        if ctx.device.type=='cuda':torch.cuda.reset_peak_memory_stats()
        metric=train_epoch(ctx,vision,head,rows,micro,condition,optimizer,scaler,2027,1,budget)
        pred,validation=predict(ctx,vision,head,stopping,condition,micro,with_loss=True)
        tick=time.perf_counter();grad=gradient_audit(ctx,vision,head,rows,rows.order(2027,0)[:micro],condition)
        diagnostic_seconds=time.perf_counter()-tick;delta=update_norm(vision,initial)
        if condition.startswith('L') and delta<=0:raise ValueError('Pilot adapter failed to update')
        if condition.startswith('F') and delta!=0:raise ValueError('Frozen pilot changed adapters')
        if vision.frozen_digest()!=pre['base_sha256']:raise ValueError('Pilot changed pretrained base')
        results[condition]={'epoch':metric,'validation':validation,'gradients':grad,'diagnostic_seconds':diagnostic_seconds,
            'adapter_update_norm':delta,'memory':memory_report(),'optimizer_bytes':optimizer_bytes(optimizer),
            'images_per_second':metric['paintings']/metric['seconds'],'pairs_per_second':metric['language_pairs']/metric['seconds'],
            'inner_validation_probability_std':float(pred.p_positive.std()),'head_sha256':state_hash(head)}
        print(f"Pilot {condition}: {metric['seconds']:.1f}s/epoch, {results[condition]['images_per_second']:.2f} images/s",flush=True)
        del head,optimizer,scaler
    projection=runtime_projection(results,ctx.historical,ctx.config)
    count=preserve_history(args.run_dir/'preflight')
    with new_output(output) as stage:
        json_output(results,stage/'conditions.json');json_output(projection,stage/'runtime_projection.json')
        seal(stage,ctx.identity,{'status':'PILOT_PASS','seconds':time.perf_counter()-start,'epochs_per_condition':1,
            'held_out_performance_claim':False,'protected_files_matched':count,'runtime_projection':projection})
    print(f"Projected MiniScreen {projection['miniscreen_seconds']/60:.1f}min; Screen {projection['screen_gpu_hours']:.2f} GPU-hours",flush=True)
