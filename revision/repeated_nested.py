"""Repeated subject-nested CV with frozen search spaces; not new biological data.
Reference seed may reuse verified legacy OOF; all other seeds require actual fits.
Never resample saved predictions to simulate retraining on another split.
"""
from pathlib import Path
import argparse,csv,json,traceback,importlib.util,importlib.metadata,platform
import numpy as np
import joblib
from threadpoolctl import threadpool_limits
from . import ROOT
from .storage import Store,run_lock,csv_write,json_write,sha,digest
from .protocol import split_subjects,subject_rows


def read_csv(path):
    with Path(path).open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))

def legacy_config(source):
    identity=json.loads((source/'run_identity.json').read_text());expected=json.loads((ROOT/'LEGACY_COMPATIBILITY.json').read_text())
    if identity['sources']!=expected['sources']:raise ValueError('Unknown legacy computational source')
    for rel,h in expected['files'].items():
        if sha(ROOT/rel)!=h:raise ValueError('Changed legacy source: '+rel)
    return identity['config']

def frozen_spaces(source,arms):
    spaces=json.loads((source/'search_space.json').read_text())
    if not arms or len(arms)!=len(set(arms)):raise ValueError('Invalid arm list')
    for arm in arms:
        for c in spaces[arm]:
            if c['candidate_id']!=digest({k:v for k,v in c.items() if k!='candidate_id'})[:12]:raise ValueError('Candidate ID hash mismatch')
    return {arm:spaces[arm] for arm in arms}

def validate_manifest(source,cfg):
    rows=read_csv(source/'segment_manifest.csv');ids=[r['id'] for r in rows]
    if not rows or len(ids)!=len(set(ids)):raise ValueError('Invalid segment IDs')
    if cfg.get('expected_subjects') is not None and len({r['subject'] for r in rows})!=cfg['expected_subjects']:raise ValueError('Unexpected biological sample size')
    return rows

def plan(source,old,cfg,folder):
    manifest=validate_manifest(source,old)
    frames=[{'id':r['id'],'subject':r['subject'],'y':r.get('species') or r.get('y')} for r in manifest]
    out=[]
    for seed in cfg['split_seeds']:
        for fold,(tr,te) in enumerate(split_subjects(frames,np.arange(len(frames)),old['outer_folds'],seed)):
            inner=split_subjects(frames,tr,old['inner_folds'],seed+fold+1)
            for level,splits in [('outer',[(tr,te)]),('inner',inner)]:
                for j,(a,b) in enumerate(splits):
                    if set(frames[i]['subject'] for i in a)&set(frames[i]['subject'] for i in b):raise AssertionError('Subject leakage')
                    for role,ix in [('train',a),('test',b)]:
                        for subject in sorted({frames[i]['subject'] for i in ix}):out.append({'repeat_seed':seed,'level':level,'outer_fold':fold,'inner_fold':j if level=='inner' else '', 'role':role,'subject':subject,'species':next(frames[i]['y'] for i in ix if frames[i]['subject']==subject)})
    csv_write(folder/'planned_splits.csv',out)
    return frames,out

def preflight(source,old,folder,data_root=None):
    files=sorted((source/'checkpoints').glob('prepare__record_*.joblib'));checked=[];broken=[]
    for p in files:
        receipt=p.with_suffix('.done.json')
        try:
            if not receipt.exists() or json.loads(receipt.read_text())['sha256']!=sha(p):raise ValueError('missing/invalid receipt')
            checked.append(p)
        except Exception as e:broken.append({'path':str(p),'reason':str(e)})
    expected_records=len(read_csv(source/'preparation_audit.csv')) if (source/'preparation_audit.csv').exists() else (len(read_csv(source/'input_checksums.csv')) if old['data_mode']=='polliradar' and (source/'input_checksums.csv').exists() else None)
    raw_valid=False;raw_error=None
    if data_root:
        try:
            expected={r['name']:r['sha256'] for r in read_csv(source/'input_checksums.csv')}
            actual={p.name:sha(p) for p in Path(data_root).glob('*.csv')}
            if actual!=expected:raise ValueError('Raw CSV names/hashes differ from frozen source')
            raw_valid=True
        except Exception as e:raw_error=str(e)
    prepared_valid=bool(checked) and not broken and (expected_records is None or len(checked)==expected_records)
    deps={n:importlib.util.find_spec(n) is not None for n in ['librosa','sklearn','joblib','scipy','numpy']}
    ready=bool((prepared_valid or raw_valid) and all(deps.values()))
    result={'ready_for_refitting':ready,'prepared_checkpoint_count':len(files),'valid_prepared_checkpoint_count':len(checked),'expected_recordings':expected_records,'broken':broken,'raw_data_supplied':bool(data_root),'raw_data_valid':raw_valid,'raw_error':raw_error,'dependencies':deps,'block_reason':None if ready else 'Actual prepared radar frames with checksum receipts or matching original raw CSV are required, plus runtime dependencies. Saved prediction/summary CSV cannot train a new split.'}
    json_write(folder/'input_preflight.json',result);return result

def load_frames(source,old,store,data_root=None):
    if data_root:
        from .data import input_identity,prepare
        c=dict(old,data_root=str(Path(data_root).resolve()));input_identity(c);frames,_=prepare(c,store)
    else:
        frames=[]
        for p in sorted((source/'checkpoints').glob('prepare__record_*.joblib')):
            r=p.with_suffix('.done.json')
            if not r.exists() or json.loads(r.read_text())['sha256']!=sha(p):raise ValueError('Invalid preparation commit')
            f,_=joblib.load(p);frames.extend(f)
    wanted={r['id']:r for r in validate_manifest(source,old)}
    if len(frames)!=len(wanted) or {f['id'] for f in frames}!=set(wanted):raise ValueError('Prepared frames do not match original cohort')
    for f in frames:
        r=wanted[f['id']]
        if f['subject']!=r['subject'] or f['y']!=(r.get('species') or r.get('y')):raise ValueError('Metadata mismatch')
        if not all(np.isfinite(f[k]).all() for k in ['amp','cpx','window']):raise ValueError('Non-finite frame')
    return frames

def legacy_reference(source,old,cfg,planned):
    seed=old['seed'];old_splits=read_csv(source/'splits.csv')
    normalize=lambda r:(r['level'],int(r['outer_fold']),str(r['inner_fold']),r['role'],r['subject'])
    expected={normalize(r) for r in planned if r['repeat_seed']==seed}
    if {normalize(r) for r in old_splits}!=expected:raise ValueError('Reference folds differ from planned original seed')
    pred=[r for r in read_csv(source/'oof_predictions.csv') if r['arm'] in cfg['arms']]
    manifest=validate_manifest(source,old);ids={r['id'] for r in manifest}
    for arm in cfg['arms']:
        rr=[r for r in pred if r['arm']==arm]
        if len(rr)!=len(ids) or {r['id'] for r in rr}!=ids:raise ValueError('Incomplete legacy OOF')
        testfold={r['subject']:r['outer_fold'] for r in planned if r['repeat_seed']==seed and r['level']=='outer' and r['role']=='test'}
        for r in rr:
            if int(r['fold'])!=int(testfold[r['subject']]):raise ValueError('Reference OOF test fold mismatch')
    return [dict(r,repeat_seed=seed,prediction_source='verified_legacy_OOF') for r in pred]

def macro(cm):
    den=cm.sum(-1)+cm.sum(-2);diag=np.diagonal(cm,axis1=-2,axis2=-1)
    return np.divide(2*diag,den,out=np.zeros_like(diag,dtype=float),where=den>0).mean(-1)

def summarize(pred,cfg,folder,complete):
    seeds=sorted({int(r['repeat_seed']) for r in pred});arms=cfg['arms'];subjects=sorted({r['subject'] for r in pred});labels=sorted({r['y'] for r in pred});C=len(labels)
    label={c:j for j,c in enumerate(labels)};subject={s:j for j,s in enumerate(subjects)};metric=[];deltas=[];agg=[];summ=[]
    ys={r['subject']:r['y'] for r in pred};rng=np.random.default_rng(cfg['bootstrap_seed'])
    # The same subject draw is used for every repeat and arm. Repeats never
    # increase biological n; no bootstrap of folds or treating seed*subject as n.
    weights=np.zeros((cfg['bootstrap'],len(subjects)),int)
    for c in labels:
        ix=np.array([subject[s] for s in subjects if ys[s]==c]);weights[:,ix]=rng.multinomial(len(ix),np.ones(len(ix))/len(ix),size=cfg['bootstrap'])
    for level in ['segment','subject']:
        cms=np.zeros((len(seeds),len(arms),len(subjects),C,C))
        for i,seed in enumerate(seeds):
            for j,arm in enumerate(arms):
                rr=[r for r in pred if int(r['repeat_seed'])==seed and r['arm']==arm]
                if len({r['id'] for r in rr})!=len(rr):raise ValueError('Duplicate OOF within repeat/arm')
                if level=='subject':rr=subject_rows(rr)
                for r in rr:cms[i,j,subject[r['subject']],label[r['y']],label[r['pred']]]+=1
        point=macro(cms.sum(2));boots=macro(np.einsum('bn,ianxy->biaxy',weights,cms,optimize=True))
        for j,arm in enumerate(arms):
            for i,seed in enumerate(seeds):
                lo,hi=np.quantile(boots[:,i,j],[.025,.975]);metric.append({'repeat_seed':seed,'arm':arm,'level':level,'macro_f1':point[i,j],'lo':lo,'hi':hi})
            values=point[:,j];avgdraw=boots[:,:,j].mean(1);lo,hi=np.quantile(avgdraw,[.025,.975])
            summ.append({'arm':arm,'level':level,'repeats_available':len(seeds),'repeats_planned':len(cfg['split_seeds']),'mean_macro_f1':values.mean(),'sd_across_splits':values.std(ddof=1) if len(seeds)>1 else None,'min':values.min(),'max':values.max(),'lo':lo,'hi':hi,'biological_subjects':len(subjects),'complete':complete,'interval_scope':'species-stratified subject bootstrap; shared draws across repeats; conditional on fitted models'})
        for a,b in cfg['contrasts']:
            ia,ib=arms.index(a),arms.index(b);delta=point[:,ia]-point[:,ib];bd=boots[:,:,ia]-boots[:,:,ib]
            for i,seed in enumerate(seeds):
                lo,hi=np.quantile(bd[:,i],[.025,.975]);deltas.append({'repeat_seed':seed,'a':a,'b':b,'level':level,'delta_macro_f1':delta[i],'lo':lo,'hi':hi})
            lo,hi=np.quantile(bd.mean(1),[.025,.975]);agg.append({'a':a,'b':b,'level':level,'repeats_available':len(seeds),'repeats_planned':len(cfg['split_seeds']),'mean_delta_macro_f1':delta.mean(),'sd_across_splits':delta.std(ddof=1) if len(seeds)>1 else None,'min_delta':delta.min(),'max_delta':delta.max(),'lo':lo,'hi':hi,'biological_subjects':len(subjects),'complete':complete,'interval_scope':'shared subject-cluster draws across fitted repeat models; not unconditional retraining CI','analysis_scope':'exploratory split sensitivity; no best-seed selection; no multiplicity-adjusted superiority claim'})
    for name,rows in [('repeat_metrics',metric),('repeat_paired_deltas',deltas),('repeated_metrics_summary',summ),('repeated_paired_summary',agg)]:csv_write(folder/(name+'.csv'),rows)

def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source-results',required=True);p.add_argument('--config',default='configs/repeated_nested_full.json');p.add_argument('--stage',choices=['plan','run'],default='run');p.add_argument('--data-root');p.add_argument('--stop-after',type=int,default=0);a=p.parse_args(argv)
    source=Path(a.source_results).resolve();cfg=json.loads(Path(a.config).read_text());old=legacy_config(source)
    if len(cfg['split_seeds'])!=len(set(cfg['split_seeds'])):raise ValueError('Duplicate seeds')
    spaces=frozen_spaces(source,cfg['arms']);inputs={n:sha(source/n) for n in ['run_identity.json','search_space.json','segment_manifest.csv','splits.csv','oof_predictions.csv','input_checksums.csv']}
    checkpoints=[{'name':p.name,'sha256':sha(p)} for p in sorted((source/'checkpoints').glob('prepare__record_*.joblib'))]
    identity={'version':'1.3.0','config':cfg,'source_config':old,'inputs':inputs,'prepared':checkpoints,'data_root':str(Path(a.data_root).resolve()) if a.data_root else None,'sources':{n:sha(ROOT/'revision'/n) for n in ['repeated_nested.py','nested_fixed.py','storage.py','protocol.py']},'runtime':{'python':platform.python_version(),'numpy':np.__version__,**{n:importlib.metadata.version(n) for n in ['scipy','scikit-learn','joblib','threadpoolctl']}}}
    folder=ROOT/cfg['output_root']/('repeated_nested-'+digest(identity)[:16]);print('RUN DIRECTORY',folder,flush=True)
    from . import storage
    storage.CSV_CONTEXT={'run_id':folder.name,'data_mode':old['data_mode']}
    with run_lock(folder),threadpool_limits(limits=cfg['threads']):
        (folder/'complete.json').unlink(missing_ok=True);json_write(folder/'run_identity.json',identity);json_write(folder/'frozen_search_space.json',spaces)
        _,planned=plan(source,old,cfg,folder);check=preflight(source,old,folder,a.data_root)
        json_write(folder/'SCOPE.json',{'seeds_chosen_before_followup_fit':cfg['split_seeds'],'candidate_policy':'reuse exact legacy candidate dictionaries/IDs, same model seeds; vary only subject-split seeds','arms_scope':cfg['arms'],'primary_contrast':cfg['contrasts'][0],'baseline_seed_reuse':'validated legacy OOF only; no invented refitting','limits':'one previously developed corpus, same 33 insects; repeated CV measures split sensitivity; fixed-model subject CIs do not quantify all model-development/retraining uncertainty'})
        allpred=[];done=[]
        if cfg.get('reuse_reference',True) and old['seed'] in cfg['split_seeds']:
            allpred=legacy_reference(source,old,cfg,planned);done.append(old['seed']);csv_write(folder/'repeated_oof_predictions.csv',allpred)
        if a.stage=='plan':
            json_write(folder/'plan_complete.json',{'planned_seeds':cfg['split_seeds'],'available_reference_seeds':done,'ready_for_refitting':check['ready_for_refitting']});print('PLAN COMPLETE',check);return
        if not check['ready_for_refitting']:
            if allpred:summarize(allpred,cfg,folder,False)
            json_write(folder/'blocked.json',{'missing_seeds':[s for s in cfg['split_seeds'] if s not in done],**check,'reference_only_no_new_retraining':True});print('BLOCKED: actual radar training inputs are absent; no synthetic results substituted.',flush=True);raise SystemExit(2)
        try:
            from .nested_fixed import nested_fixed
            store=Store(folder/'checkpoints',a.stop_after);frames=load_frames(source,old,store,a.data_root)
            for seed in cfg['split_seeds']:
                if seed in done:continue
                sub=folder/f'seed_{seed}';sub.mkdir(exist_ok=True);options=dict(old,threads=cfg['threads'])
                _,pred=nested_fixed(frames,options,Store(sub/'checkpoints',a.stop_after),sub,spaces,seed)
                for r in pred:
                    if r['arm'] in cfg['arms']:allpred.append(dict(r,repeat_seed=seed,prediction_source='new_nested_refit'))
                done.append(seed);csv_write(folder/'repeated_oof_predictions.csv',allpred);json_write(folder/'progress.json',{'completed_seeds':done,'planned':cfg['split_seeds']})
            complete=set(done)==set(cfg['split_seeds']);summarize(allpred,cfg,folder,complete)
            json_write(folder/'complete.json',{'completed_seeds':done,'planned_seeds':cfg['split_seeds'],'biological_subjects':len({f['subject'] for f in frames}),'no_new_cohort':True});print('COMPLETE',done,flush=True)
        except InterruptedError as e:json_write(folder/'interrupted.json',{'reason':str(e)});raise SystemExit(75)
        except Exception as e:json_write(folder/'last_error.json',{'type':type(e).__name__,'message':str(e),'traceback':traceback.format_exc()});raise
if __name__=='__main__':main()
