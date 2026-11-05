"""Reviewer closure v3. Run with python -m revision.final_three_checks --stage ...
Scientific gate: no modification of historical five-seed artifacts; evidence only.
"""
import argparse,csv,hashlib,json,os,platform,subprocess,sys,time,traceback
from pathlib import Path
import numpy as np
import joblib
from sklearn.metrics import f1_score
from threadpoolctl import threadpool_limits
from . import ROOT
from .models import Model,get_cycles
from .raw_baselines import load_prepared
from .reviewer_closure import tensor
from .protocol import split_subjects,paired
from .storage import Store,sha,digest,csv_write,json_write,run_lock,atomic


def rows(p):
    with Path(p).open(newline='',encoding='utf-8-sig') as f:return list(csv.DictReader(f))
def harray(x):
    a=np.ascontiguousarray(x)
    h=hashlib.sha256(json.dumps({'shape':a.shape,'dtype':a.dtype.str},sort_keys=True).encode());h.update(a.tobytes());return h.hexdigest()
def identity(a):return {'size':Path(a).stat().st_size,'sha256':sha(a)}
def metric(rs,labels):return float(f1_score([r['y'] for r in rs],[r['pred'] for r in rs],labels=labels,average='macro',zero_division=0))
def index_unique(rs,key=('id',)):
    result={}
    for r in rs:
        k=tuple(str(r[x]) for x in key)
        if k in result:raise ValueError('Duplicated key '+str(k))
        result[k]=r
    return result

def validate_reference(frames,locked,v2):
    gold=index_unique([{'id':f['id'],'subject':f['subject'],'y':f['y']} for f in frames])
    for tag,data in [('locked',locked),('v2',v2)]:
        for r in data:
            k=(r['id'],)
            if k not in gold or any(str(r[x])!=str(gold[k][x]) for x in ('subject','y')):
                raise ValueError(f'{tag}: ID, subject or label mismatch: {k}')
    return gold

def file_evidence(paths,out):
    records=[]
    for name,p in paths.items():
        p=Path(p)
        records.append({'artifact':name,'path':str(p),'present':int(p.is_file()),'sha256':sha(p) if p.is_file() else '', 'size_bytes':p.stat().st_size if p.is_file() else 0})
    csv_write(out/'source_hash_manifest.csv',records)
    return records

def load_context(a):
    src=Path(a.source).resolve(); v2=Path(a.v2).resolve();locked=Path(a.locked).resolve()
    frames,origin,receipts=load_prepared(src)
    v2_identity=json.loads((v2/'run_manifest.json').read_text(encoding='utf-8'))
    if sha(src/'run_identity.json')!=v2_identity['source_identity_sha256']:
        raise ValueError('Original run identity differs from v2; cannot compare across source runs')
    folds=split_subjects(frames,np.arange(len(frames)),origin['config']['outer_folds'],origin['config']['seed'])
    oldpred=rows(locked/'oof_predictions.csv'); oldsel=rows(locked/'selected_configs.csv'); oldinner=rows(locked/'inner_search.csv')
    v2pred=rows(v2/'ha_outer_predictions.csv')
    validate_reference(frames,oldpred,v2pred)
    return src,v2,locked,frames,origin,receipts,folds,oldpred,oldsel,oldinner,v2pred

def _available_version(pkg):
    from importlib.metadata import version,PackageNotFoundError
    try:version(pkg);return True
    except PackageNotFoundError:return False

def check_source(a,ctx,out):
    src,v2,locked,frames,origin,receipts,folds,oldpred,oldsel,oldinner,v2pred=ctx
    paths={'run_identity':src/'run_identity.json','segment_manifest':src/'segment_manifest.csv','splits':src/'splits.csv',
           'locked_oof':locked/'oof_predictions.csv','locked_selected':locked/'selected_configs.csv','locked_inner':locked/'inner_search.csv',
           'v2_ha_oof':v2/'ha_outer_predictions.csv','v2_baseline_oof':v2/'baseline_outer_predictions.csv','v2_manifest':v2/'run_manifest.json'}
    file_evidence(paths,out)
    env={'python':sys.version,'platform':platform.platform(),'host':platform.node(),'source_module_sha256':sha(Path(__file__).with_name('models.py')),'v2_tensor_dtype':'float64 (documented source_change_float64.diff)'}
    try:
        from importlib.metadata import version
        env['packages']={x:(version(x) if True else '') for x in ['numpy','scipy','scikit-learn','joblib','aeon','psutil'] if _available_version(x)}
    except Exception as e:env['package_capture_error']=repr(e)
    json_write(out/'environment.json',env)
    # Cross-check actual subject split assignments against the locked split file.
    split_rows=rows(src/'splits.csv')
    for fold,(train_ix,test_ix) in enumerate(folds):
        old_test={r['subject'] for r in split_rows if r.get('level')=='outer' and int(r['outer_fold'])==fold and r['role']=='test'}
        computed={frames[i]['subject'] for i in test_ix}
        if not old_test or old_test!=computed:raise ValueError(f'Outer-fold subject split mismatch fold={fold}')
    check=[]
    for r in receipts:check.append({'artifact':'prepared/'+r['name'],'sha256':r['sha256'],'status':'PASS'})
    csv_write(out/'prepared_checkpoint_audit.csv',check)
    # Never assert raw CSV bytes verified unless original raw recordings supplied.
    rawroot=Path(a.raw_root).resolve() if a.raw_root else None
    raw=[]
    for spec in origin.get('inputs',[]):
        p=rawroot/spec['name'] if rawroot else None
        actual=sha(p) if p and p.exists() else ''
        raw.append({'name':spec['name'],'recorded_sha256':spec.get('sha256',''),'actual_sha256':actual,
                    'status':'PASS' if actual and actual==spec.get('sha256') else 'FAIL' if actual else 'NOT_AVAILABLE'})
    csv_write(out/'raw_source_audit.csv',raw)
    if rawroot and any(r['status']!='PASS' for r in raw):raise ValueError('Raw source mismatches or missing recordings; fix source root')
    return raw

def reconcile(ctx,out):
    src,v2,locked,frames,origin,receipts,folds,oldpred,oldsel,oldinner,v2pred=ctx
    labels=sorted({f['y'] for f in frames}); old=index_unique([r for r in oldpred if r['arm']=='combined']); va=index_unique([r for r in v2pred if r['view']=='combined' and r['mapping']=='ha_soft'])
    if set(old)!=set(va):raise ValueError('Locked vs v2 combined OOF cohort differs')
    original_cfg=origin['config']; v2c=json.loads((v2/'run_manifest.json').read_text())['config']
    choice=[]; scores=[]
    for fold,(tr,te) in enumerate(folds):
        x=next(r for r in oldsel if str(r.get('outer_fold'))==str(fold) and r['arm']=='combined')
        c=json.loads(x['config']); s=next(r for r in rows(v2/'ha_selected.csv') if r['view']=='combined' and r['mapping']=='ha_soft' and int(r['fold'])==fold)
        oo=[old[(frames[i]['id'],)] for i in te];vv=[va[(frames[i]['id'],)] for i in te]
        choice.append({'fold':fold,'locked_B':c['B'],'v2_B':v2c['B'],'locked_states':c['states'],'v2_states':v2c['states'],
                       'locked_max_cycles':original_cfg['max_cycles'],'v2_max_cycles':v2c['max_cycles'],
                       'locked_descriptor_count':x['input_features'],'v2_descriptor_count':s['n_descriptors'],
                       'locked_candidate_id':c.get('candidate_id',''),'v2_candidate_id':s['candidate'],
                       'locked_penalty':c['l1'],'v2_selected_penalty':'see ha_inner_search.csv',
                       'same_configuration':int(c['B']==v2c['B'] and c['states']==v2c['states'] and original_cfg['max_cycles']==v2c['max_cycles'])})
        scores.append({'fold':fold,'locked_combined_macro_f1':metric(oo,labels),'v2_ha_combined_macro_f1':metric(vv,labels),
                       'changed_decisions':sum(u['pred']!=v['pred'] for u,v in zip(oo,vv)), 'n_samples':len(te)})
    csv_write(out/'configuration_delta.csv',choice);csv_write(out/'fold_metric_reconciliation.csv',scores)
    stats={'locked_macro_f1':metric(list(old.values()),labels),'v2_macro_f1':metric(list(va.values()),labels),'absolute_gap':metric(list(old.values()),labels)-metric(list(va.values()),labels)}
    json_write(out/'reconciliation_descriptive.json',stats)
    return stats

def do_ablation(ctx,out,store,stop_guard=True):
    """Re-run original full combined grid for both mappings with identical split/selection.
    All settings apart from partition are paired. Do not use test metrics to select models.
    """
    src,v2,locked,frames,origin,receipts,folds,oldpred,oldsel,oldinner,v2pred=ctx
    cfg=origin['config'];labels=sorted({f['y'] for f in frames});sel=[];inn=[];pred=[];runtime=[];diagnostics=[]
    # Legacy search grid is the authoritative parameter space, not the restricted v2 grid.
    original_candidates={}
    for r in oldinner:
        if r['arm']!='combined':continue
        c=json.loads(r['config']); c.pop('candidate_id',None)
        original_candidates[digest(c)]=c
    settings=sorted({(v['B'],v['states'],v['l1'],v['lg']) for v in original_candidates.values()})
    if not settings:raise ValueError('No locked combined search settings')
    expected_set={f['id'] for f in frames}
    for fold,(tr,te) in enumerate(folds):
        inner=split_subjects(frames,tr,cfg['inner_folds'],cfg['seed']+fold+1)
        for mapping in ('ha_soft','quantile'):
            ranked=[]
            for B,states,l1,lg in settings:
                c={'view':'combined','method':'linguistic','partition':mapping,'states':states,'B':B,'l1':l1,'lg':lg}
                key=digest(c)[:12];yy=[];pp=[]
                for j,(it,iv) in enumerate(inner):
                    def work(it=it,iv=iv,c=c):
                        t=time.perf_counter();m=Model(c,cfg).fit([frames[i] for i in it]);fit=time.perf_counter()-t
                        t=time.perf_counter();p=list(map(str,m.predict([frames[i] for i in iv])));predict=time.perf_counter()-t
                        return {'pred':p,'fit_seconds':fit,'predict_seconds':predict,'n_desc':len(m.names),'active':m.active(),'converged':m.converged}
                    r=store.task(f'reconcile_inner/{fold}/{mapping}/{key}/{j}',work)
                    yy.extend(frames[i]['y'] for i in iv);pp.extend(r['pred'])
                    inn.append({'fold':fold,'mapping':mapping,'candidate':key,'inner_fold':j,'B':B,'states':states,'l1':l1,'n_train':len(it),'n_valid':len(iv),'fit_s':r['fit_seconds'],'predict_s':r['predict_seconds'],'n_descriptors':r['n_desc']})
                sc=float(f1_score(yy,pp,labels=labels,average='macro',zero_division=0));ranked.append((sc,key,c))
                csv_write(out/'reconcile_inner_progress.csv',inn)
            sc,key,c=sorted(ranked,key=lambda x:(-x[0],x[1]))[0]
            def fit_outer():
                t=time.perf_counter();m=Model(c,cfg).fit([frames[i] for i in tr]);return {'model':m,'fit_s':time.perf_counter()-t}
            rec=store.task(f'reconcile_outer/{fold}/{mapping}',fit_outer)
            m=rec['model'];t=time.perf_counter();pr=list(map(str,m.predict([frames[i] for i in te])));pred_s=time.perf_counter()-t
            cp=store.folder/(f'reconcile_outer__{fold}__{mapping}.joblib')
            sel.append({'fold':fold,'mapping':mapping,'candidate':key,'inner_macro_f1':sc,'config':json.dumps(c,sort_keys=True),'n_descriptors':len(m.names),'active_descriptors':m.active(),'checkpoint_sha256':sha(cp)})
            runtime.append({'fold':fold,'mapping':mapping,'fit_seconds':rec['fit_s'],'predict_seconds':pred_s,'n_train':len(tr),'n_test':len(te)})
            pred.extend({'fold':fold,'mapping':mapping,'view':'combined','id':frames[i]['id'],'subject':frames[i]['subject'],'y':frames[i]['y'],'pred':p} for i,p in zip(te,pr))
            for name,rr in [('reconcile_selected.csv',sel),('reconcile_runtime.csv',runtime),('reconcile_outer_predictions.csv',pred)]:csv_write(out/name,rr)
    a=index_unique([r for r in pred if r['mapping']=='ha_soft']);b=index_unique([r for r in pred if r['mapping']=='quantile'])
    if set(a)!=set(b) or set(a)!= {(x,) for x in expected_set}:raise AssertionError('Unmatched OOF samples')
    old=index_unique([r for r in oldpred if r['arm']=='combined'])
    contrasts=[]
    for level in ('segment','subject'):
        contrasts.append({'comparison':'HA - quantile','level':level,**paired(list(a.values()),list(b.values()),5000,20261103,level)})
        contrasts.append({'comparison':'replayed HA - original locked','level':level,**paired(list(a.values()),list(old.values()),5000,20261103,level)})
    csv_write(out/'reconcile_paired_contrasts.csv',contrasts)
    diag=[]
    for k,r in a.items():
        diag.append({'id':r['id'],'fold':r['fold'],'y':r['y'],'original_pred':old[k]['pred'],'replay_ha_pred':r['pred'],'quantile_pred':b[k]['pred'],
                     'ha_matches_original':int(r['pred']==old[k]['pred']),'ha_correct':int(r['pred']==r['y']),'quantile_correct':int(b[k]['pred']==r['y'])})
    csv_write(out/'sample_prediction_diagnostics.csv',diag)
    json_write(out/'ablation_COMPLETE.json',{'n':len(diag),'full_legacy_grid_used':True,'nested_train_only_selection':True})

def tensor_v2_exact(frames,B=8,max_cycles=32):
    """Exact documented v2 local float64 patch, never silently use code default."""
    result=[]
    for f in frames:
        b=get_cycles(f,B,max_cycles)
        if len(b)<2:raise ValueError('Insufficient cycles')
        n=min(len(b),max_cycles)
        x=np.zeros((4,max_cycles*B),dtype=np.float64)
        x[:3,:n*B]=b[:n].transpose(2,0,1).reshape(3,n*B)
        x[3,:n*B]=1.
        result.append(x)
    return np.stack(result)

def do_audit(ctx,out,store,raw):
    src,v2,locked,frames,origin,receipts,folds,oldpred,oldsel,oldinner,v2pred=ctx
    b=rows(v2/'baseline_outer_predictions.csv');expected={f['id']:f for f in frames};audit=[];oof=[];patch=[]
    if a_num:=len([r for r in b if r['method']=='DrCIF']):
        if a_num!=len(frames):raise ValueError('Incomplete baseline OOF')
    # Train-independent tensors are reconstructed from original prepared frames,
    # then compared to actual saved outer model predictions (no refit).
    for fold,(tr,te) in enumerate(folds):
        xx=store.task(f'audit_tensors/{fold}',lambda: {'train':tensor_v2_exact([frames[i] for i in tr]),'test':tensor_v2_exact([frames[i] for i in te])})
        for role,idx in [('train',tr),('test',te)]:
            arr=xx[role];p=out/'tensors'/f'fold_{fold}_{role}.npy';p.parent.mkdir(parents=True,exist_ok=True)
            if not p.exists():atomic(p,lambda q:np.save(q.open('wb'),arr,allow_pickle=False))
            if harray(np.load(p,mmap_mode='r'))!=harray(arr):raise ValueError('Tensor file mismatch')
            audit.append({'fold':fold,'role':role,'file':str(p.relative_to(out)),'sha256':sha(p),'array_sha256':harray(arr),
                          'shape':str(arr.shape),'dtype':str(arr.dtype),'ids_sha256':digest([frames[i]['id'] for i in idx])})
            csv_write(out/'tensor_manifest.csv',audit)
        for method in ('DrCIF','RDST'):
            checkpoint=Path(a_global.v2_checkpoints)/f'baseline_outer__{fold}__{method}.joblib'
            receipt=checkpoint.with_suffix('.done.json')
            if not checkpoint.exists() or not receipt.exists():raise FileNotFoundError(f'Missing baseline model and receipt: {checkpoint}')
            record=json.loads(receipt.read_text())
            if sha(checkpoint)!=record['sha256']:raise ValueError('Checkpoint hash mismatch')
            state=joblib.load(checkpoint)
            if not isinstance(state,dict) or 'model' not in state:raise ValueError('Unknown checkpoint schema')
            model=state['model']
            # Fail closed on environment incompatibility; report rather than overriding silently.
            actual=list(map(str,model.predict(xx['test'])))
            expected_b=index_unique([r for r in b if int(r['fold'])==fold and r['method']==method]);items=[frames[i]['id'] for i in te]
            if len(expected_b)!=len(items) or set(expected_b)!={(i,) for i in items}:raise ValueError('Baseline OOF cohort mismatch')
            for i,pred in zip(te,actual):
                old=expected_b[(frames[i]['id'],)];match=pred==old['pred']
                oof.append({'fold':fold,'method':method,'id':frames[i]['id'],'subject':frames[i]['subject'],'y':frames[i]['y'],'saved_pred':old['pred'],'replay_pred':pred,'pass':int(match),'checkpoint_sha256':sha(checkpoint)})
                if not match:raise AssertionError(f'OOF prediction mismatch {fold} {method} {frames[i]["id"]}')
            csv_write(out/'baseline_oof_replay.csv',oof)
    json_write(out/'audit_COMPLETE.json',{'tensor_files':len(audit),'oof_replays':len(oof),'all_replay_pass':True,'raw_source_fully_verified':bool(raw) and all(r['status']=='PASS' for r in raw),
          'tensor_dtype':'float64 per documented v2 local compatibility patch','notice':'Numeric binary tensor hashes document current execution. Identical prepared samples do not establish identical historical fitted baseline.'})

class TimedModel(Model):
    def __init__(self,c,options):
        super().__init__(c,options)
        self.feature_train_s=0.0;self.feature_test_s=0.0
    def _describe(self,*args,**kwargs):
        t=time.perf_counter();z=super()._describe(*args,**kwargs)
        if kwargs.get('fit',False) or (len(args)>1 and bool(args[1])):
            self.feature_train_s+=time.perf_counter()-t
        else:self.feature_test_s+=time.perf_counter()-t
        return z

def timed_model(c,cfg,train,test):
    import psutil,threading
    m=TimedModel(c,cfg);process=psutil.Process();samples=[];stop=threading.Event()
    def watcher():
        while not stop.wait(.02):
            try:samples.append(process.memory_info().rss)
            except psutil.Error:break
    samples.append(process.memory_info().rss)
    worker=threading.Thread(target=watcher,daemon=True);worker.start()
    try:
        t=time.perf_counter();m.fit(train);totalfit=time.perf_counter()-t
        t=time.perf_counter();pred=list(map(str,m.predict(test)));testtot=time.perf_counter()-t
    finally:
        stop.set();worker.join(timeout=1);samples.append(process.memory_info().rss)
    return {'model':m,'pred':pred,'fit_total_s':totalfit,'feature_train_s':m.feature_train_s,'train_solver_and_scaling_s':max(0,totalfit-m.feature_train_s),
            'predict_total_s':testtot,'feature_test_s':m.feature_test_s,'classifier_inference_s':max(0,testtot-m.feature_test_s),
            'sampled_peak_rss_mb':max(samples)/1048576,'rss_samples':len(samples)}

def do_runtime(ctx,out,store,repeats):
    src,v2,locked,frames,origin,receipts,folds,oldpred,oldsel,oldinner,v2pred=ctx
    import psutil
    proc=psutil.Process();rows_out=[];pred=[]
    for fold,(tr,te) in enumerate(folds):
        cfgrow=next(r for r in oldsel if str(r.get('outer_fold'))==str(fold) and r['arm']=='combined')
        c=json.loads(cfgrow['config']);cfg=origin['config'];
        for repeat in range(repeats):
            t=time.perf_counter();base=proc.memory_info().rss;obj=store.task(f'pg_runtime/{fold}/{repeat}',lambda:timed_model(c,cfg,[frames[i] for i in tr],[frames[i] for i in te]));wall=time.perf_counter()-t
            peak=proc.memory_info().rss
            from .models import _cached_cycles
            _cached_cycles.cache_clear()
            rows_out.append({'fold':fold,'repeat':repeat,'n_train':len(tr),'n_test':len(te),'B':c['B'],'states':c['states'],'features':len(obj['model'].names),
               'feature_train_s':obj['feature_train_s'],'training_s':obj['train_solver_and_scaling_s'],'feature_test_s':obj['feature_test_s'],
               'prediction_s':obj['classifier_inference_s'],'fit_total_s':obj['fit_total_s'],'predict_total_s':obj['predict_total_s'],
               'process_rss_after_mb':peak/1048576,'process_rss_before_mb':base/1048576, 'sampled_peak_rss_mb':obj['sampled_peak_rss_mb'],
               'note':'Sampled peak RSS at 20 ms intervals; includes Python process and background imports. Checkpoint-reused runs are not fresh timings.'})
            if repeat==0:pred.extend({'id':frames[i]['id'],'subject':frames[i]['subject'],'y':frames[i]['y'],'fold':fold,'pred':p} for i,p in zip(te,obj['pred']))
            csv_write(out/'pg_runtime_segments.csv',rows_out);csv_write(out/'pg_runtime_oof.csv',pred)
    old=index_unique([r for r in oldpred if r['arm']=='combined']); rr=index_unique(pred)
    if old.keys()!=rr.keys():raise ValueError('Runtime replay incomplete')
    diagnostic=[{'id':k[0],'match':int(old[k]['pred']==rr[k]['pred']),'old':old[k]['pred'],'replay':rr[k]['pred']} for k in old]
    csv_write(out/'pg_runtime_prediction_match.csv',diagnostic)
    json_write(out/'runtime_COMPLETE.json',{'n_prediction_comparisons':len(diagnostic),'matching':sum(r['match'] for r in diagnostic),'n_repeats':repeats,
      'warning':'Store.task returns cached fit timings on restart: for valid on-device runtime measurements use new --out or --fresh-runtime-run and do not reuse checkpointed timings.'})

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--stage',choices=['reconcile','ablation','audit','runtime','all'],required=True)
    ap.add_argument('--source',required=True,help='source_original_run with prepared checkpoints')
    ap.add_argument('--locked',default='../locked_reference');ap.add_argument('--v2',default='../v2_reference')
    ap.add_argument('--v2-checkpoints',default=None,help='results_PG_LTS_v2/run/checkpoints')
    ap.add_argument('--raw-root',default=None);ap.add_argument('--out',default='runs_closure_v3');ap.add_argument('--stop-after',type=int,default=0);ap.add_argument('--repeats',type=int,default=3)
    global a_global
    a=ap.parse_args();a_global=a
    if a.v2_checkpoints is None:a.v2_checkpoints=str(Path(a.v2)/'checkpoints')
    ctx=load_context(a);src,v2,locked,*_=ctx
    root=Path(a.out).resolve();root.mkdir(parents=True,exist_ok=True)
    with run_lock(root),threadpool_limits(limits=1):
        manifest={'version':'3.0','source':sha(src/'run_identity.json'),'locked':sha(locked/'oof_predictions.csv'),
          'v2':sha(v2/'ha_outer_predictions.csv'),'code':sha(__file__),'python':sys.version,'platform':platform.platform(),'hostname':platform.node(),'pipeline_module_sha256':sha(Path(__file__).with_name('models.py')),'argv':vars(a)}
        mpath=root/'run_manifest.json'
        if mpath.exists():
            prev=json.loads(mpath.read_text())
            for k in ('source','locked','v2','code'):
                if prev[k]!=manifest[k]:raise ValueError('Source or code changed. Use a NEW --out directory; do not mix runs.')
        json_write(mpath,manifest);raw=check_source(a,ctx,root)
        store=Store(root/'checkpoints',a.stop_after)
        stages=['reconcile','ablation','audit','runtime'] if a.stage=='all' else [a.stage]
        try:
            for stage in stages:
                if stage=='reconcile':reconcile(ctx,root)
                if stage=='ablation':do_ablation(ctx,root,store)
                if stage=='audit':do_audit(ctx,root,store,raw)
                if stage=='runtime':do_runtime(ctx,root,store,a.repeats)
                json_write(root/(stage+'_DONE.json'),{'stage':stage,'time':time.strftime('%Y-%m-%dT%H:%M:%S'), 'checkpoints_reused':store.reused,'new_checkpoints':store.completed})
            json_write(root/'status.json',{'completed_stages':[s for s in ('reconcile','ablation','audit','runtime') if (root/(s+'_DONE.json')).exists()],
                                          'historical_five_seed_never_modified':True})
        except InterruptedError:
            json_write(root/'INTERRUPTED.json',{'resume':'Repeat command with same --out and --source'});raise SystemExit(75)
        except Exception as exc:
            json_write(root/'ERROR.json',{'error':str(exc),'traceback':traceback.format_exc()});raise
    print('DONE:',root)
if __name__=='__main__':main()
