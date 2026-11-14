"""Predeclared resource-allocation gates, never promotion from a single fold."""

def screening_gate(summary,fold_metrics,cfg,*,verified,healthy,mini=False):
    if not verified or not healthy:return {'status':'SCREEN_INCONCLUSIVE','lora_promising':False,'auxiliary_promising':False,'verified_healthy':False}
    def compare(a,b):
        x,y=summary.loc[a],summary.loc[b]
        folds=fold_metrics.pivot(index='fold',columns='model',values='contrast')
        return {'gain':float(x.pooled_contrast-y.pooled_contrast),'loss_delta':float(x.pooled_log_loss-y.pooled_log_loss),
            'fold_wins':int((folds[a]>folds[b]).sum())}
    effects={f'{a}-{b}':compare(a,b) for a,b in (('L0','F0'),('L8','F8'),('F8','F0'),('L8','L0'))}
    if mini:
        bad=all(effects[n]['gain']<=cfg['mini_catastrophic_gain'] and effects[n]['loss_delta']>cfg['mini_catastrophic_loss'] for n in ('L0-F0','L8-F8'))
        return {'status':'MINISCREEN_STOP_REVIEW' if bad else 'MINISCREEN_SANITY_PASS','effects':effects,'verified_healthy':True,
                'lora_promising':False,'auxiliary_promising':False,'single_fold_is_not_selection':True}
    def good(value):return value['gain']>=cfg['screen_gain'] and value['fold_wins']>=1 and value['loss_delta']<=cfg['screen_loss_guardrail']
    lora=any(good(effects[n]) for n in ('L0-F0','L8-F8'));aux=good(effects['L8-L0'])
    bad=all(effects[n]['gain']<=cfg['screen_unpromising_gain'] and effects[n]['loss_delta']>cfg['screen_unpromising_loss'] for n in ('L0-F0','L8-F8'))
    labels=[n for n,yes in (('LORA_PROMISING',lora),('AUXILIARY_PROMISING',aux)) if yes]
    return {'status':'+'.join(labels) if labels else ('SCREEN_UNPROMISING' if bad else 'SCREEN_INCONCLUSIVE'),
        'lora_promising':lora,'auxiliary_promising':aux,'verified_healthy':True,'effects':effects,
        'factorial_interaction':effects['L8-L0']['gain']-effects['F8-F0']['gain'],
        'interpretation':'Resource-allocation heuristics on repeatedly examined training folds; not independent evidence of generalization'}


def competition_gate(summary,cfg,*,verified,healthy):
    candidates=[];checks={}
    for name in ('EF8','EL0','EL8'):
        row=summary.loc[name];base=summary.loc['R0'];control=summary.loc['EF0']
        flags={'gain_r0':row.pooled_contrast-base.pooled_contrast>=cfg['minimum_pooled_contrast_gain'],
            'gain_frozen_bce':row.pooled_contrast-control.pooled_contrast>=cfg['minimum_incremental_gain'],
            'fold_wins':row.fold_wins>=cfg['minimum_fold_wins'],
            'loss':row.log_loss_mean<=base.log_loss_mean+cfg['log_loss_guardrail_bits'],'verification':verified and healthy}
        checks[name]={k:bool(v) for k,v in flags.items()}
        if all(flags.values()):candidates.append(name)
    selected=max(candidates,key=lambda n:summary.loc[n].pooled_contrast) if candidates else None
    return {'status':'CV_GATE_QUALIFIED_NO_DEVELOPMENT_AUTHORIZED' if selected else 'COMPETITION_NO_GO',
        'selected':selected,'checks':checks,'production_model':'E50+L','locked_confirmation_opened':False,'development_scored':False}
