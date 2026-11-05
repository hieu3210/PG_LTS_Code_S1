"""Locked independent baseline tensor audit and HA-vs-quantile ablation.

No labels are used to construct input tensors; only training folds fit mappings.
Re-runs from the authenticated original prepared source, never from summary CSVs.
"""
import argparse,csv,hashlib,json,platform,sys,time
from pathlib import Path
import numpy as np
from sklearn.metrics import f1_score
from threadpoolctl import threadpool_limits
from . import ROOT
from .raw_baselines import load_prepared
from .reviewer_closure import tensor,aeon_model,seed_of
from .models import Model,get_cycles
from .protocol import split_subjects,paired
from .storage import Store,run_lock,sha,digest,json_write,csv_write,atomic


def hash_array(x):
    x=np.ascontiguousarray(x)
    h=hashlib.sha256();h.update(json.dumps({'shape':x.shape,'dtype':x.dtype.str},sort_keys=True).encode());h.update(x.tobytes())
    return h.hexdigest()


def write_tensor(path, x, metadata):
    path.parent.mkdir(parents=True,exist_ok=True)
    atomic(path,lambda p:np.savez_compressed(p.open('wb'),X=x))
    record={'path':str(path.name),'sha256_file':sha(path),'tensor_sha256':hash_array(x),'shape':list(x.shape),'dtype':str(x.dtype),**metadata}
    json_write(path.with_suffix('.json'),record)
    return record


def f1(y,p,labels):return float(f1_score(y,p,labels=labels,average='macro',zero_division=0))

def audit_tensors_only(frames,folds,cfg,out):
    """Deterministically reconstruct exact baseline tensor and compare with PG cycles."""
    rows=[];B=cfg['B'];N=cfg['max_cycles']
    for fold,(train,test) in enumerate(folds):
        for role,ids in [('outer_train',train),('outer_test',test)]:
            block=tensor([frames[i] for i in ids],B,N)
            # This assertion compares the baseline's flattened samples, not guessed file metadata.
            for k,i in enumerate(ids):
                c=get_cycles(frames[i],B,N)
                n=min(len(c),N)
                reconstructed=c[:n].transpose(2,0,1).reshape(3,n*B)
                assert np.array_equal(block[k,:3,:n*B],reconstructed)
                assert np.all(block[k,3,:n*B]==1)
                assert np.all(block[k,:,:][3,n*B:]==0)
                assert np.all(block[k,:3,n*B:]==0)
            record=write_tensor(out/'tensor_audit'/(f'fold_{fold}_{role}.npz'),block,{'fold':fold,'role':role,'ids_sha256':digest([frames[i]['id'] for i in ids]),'subject_sha256':digest([frames[i]['subject'] for i in ids]),'pg_cycle_elementwise_equal':True})
            rows.append(record)
            csv_write(out/'tensor_verified.csv',rows)
    json_write(out/'tensor_audit_COMPLETE.json',{'n_folds':len(folds),'all_pg_cycles_match_baseline_tensor':True,'note':'Tensor construction replay only; no DrCIF/RDST fitting claim'})
    return rows


def evaluate_baselines(frames,folds,original,cfg,store,out):
    B=cfg['B'];N=cfg['max_cycles'];labels=sorted({f['y'] for f in frames})
    audit=[];search=[];selected=[];predictions=[];runtime=[]
    for fold,(tr,te) in enumerate(folds):
        inner=split_subjects(frames,tr,original['inner_folds'],original['seed']+fold+1)
        ix=sorted(set(tr.tolist()+te.tolist()))
        t0=time.perf_counter();by_id={i:tensor([frames[i]],B,N)[0] for i in ix};tensor_seconds=time.perf_counter()-t0
        for method,grid in cfg['baseline_grid'].items():
            ranked=[]
            for param in grid:
                cand=digest({'method':method,'params':param})[:12];ys=[];ps=[]
                for j,(it,iv) in enumerate(inner):
                    key=f'baseline_inner/{fold}/{method}/{cand}/{j}'
                    train=np.stack([by_id[i] for i in it]);valid=np.stack([by_id[i] for i in iv])
                    def build():
                        m=aeon_model(method,param,seed_of(cfg['seed'],fold,method,cand,j))
                        start=time.perf_counter();m.fit(train,np.array([frames[i]['y'] for i in it]));fit_sec=time.perf_counter()-start
                        pred=m.predict(valid)
                        return {'pred':list(map(str,pred)),'fit_s':fit_sec}
                    r=store.task(key,build)
                    ys.extend([frames[i]['y'] for i in iv]);ps.extend(r['pred'])
                    for role,ind,x in [('inner_train',it,train),('inner_valid',iv,valid)]:
                        k=f'{fold}_{method}_{cand}_{j}_{role}'
                        audit.append(write_tensor(out/'tensor_audit'/(k+'.npz'),x,{'fold':fold,'method':method,'candidate':cand,'inner_fold':j,'role':role,'ids_sha256':digest([frames[i]['id'] for i in ind]),'n_subjects':len({frames[i]['subject'] for i in ind})}))
                value=f1(ys,ps,labels);ranked.append((value,cand,param));search.append({'fold':fold,'method':method,'candidate':cand,'inner_macro_f1':value,'params':json.dumps(param,sort_keys=True)})
            value,cand,best=sorted(ranked,key=lambda z:(-z[0],z[1]))[0]
            train=np.stack([by_id[i] for i in tr]);test=np.stack([by_id[i] for i in te]);key=f'baseline_outer/{fold}/{method}'
            for role,ind,x in [('outer_train',tr,train),('outer_test',te,test)]:
                k=f'{fold}_{method}_{role}';audit.append(write_tensor(out/'tensor_audit'/(k+'.npz'),x,{'fold':fold,'method':method,'candidate':cand,'role':role,'ids_sha256':digest([frames[i]['id'] for i in ind]),'n_subjects':len({frames[i]['subject'] for i in ind})}))
            def build():
                m=aeon_model(method,best,seed_of(cfg['seed'],fold,method,'outer'))
                start=time.perf_counter();m.fit(train,np.array([frames[i]['y'] for i in tr]));fit_sec=time.perf_counter()-start
                start=time.perf_counter();pr=m.predict(test);pred_sec=time.perf_counter()-start
                return {'model':m,'pred':list(map(str,pr)),'fit_s':fit_sec,'predict_s':pred_sec}
            result=store.task(key,build)
            # Replay inference on the fitted saved model, rather than trusting saved predictions.
            repeat=list(map(str,result['model'].predict(test)))
            if repeat!=result['pred']:raise AssertionError(f'Outer inference replay mismatch: {key}')
            model_file=store.folder/(key.replace('/','__')+'.joblib')
            model_hash=sha(model_file)
            selected.append({'fold':fold,'method':method,'candidate':cand,'best_params':json.dumps(best,sort_keys=True),'inner_macro_f1':value,'checkpoint_sha256':model_hash,'outer_prediction_replay':'PASS'})
            runtime.append({'fold':fold,'method':method,'fit_seconds':result['fit_s'],'predict_seconds':result['predict_s'],'tensor_construction_seconds_all_fold_samples':tensor_seconds,'n_outer_test':len(te)})
            predictions.extend([{'fold':fold,'method':method,'id':frames[i]['id'],'subject':frames[i]['subject'],'y':frames[i]['y'],'pred':p} for i,p in zip(te,repeat)])
            for name,rs in [('tensor_audit_manifest',audit),('baseline_inner_search',search),('baseline_selected',selected),('baseline_outer_predictions',predictions),('baseline_runtime',runtime)]:csv_write(out/(name+'.csv'),rs)
    # Historical OOF comparison is diagnostic, never silently replaces rerun data.
    historical=source_hist_csv(out)
    if historical is not None:
        old={(r['fold'],r['method'],r['id']):r['pred'] for r in historical}
        rows=[{'fold':r['fold'],'method':r['method'],'id':r['id'],'replayed':r['pred'],'historical':old.get((str(r['fold']),r['method'],r['id']),''),'match':int(old.get((str(r['fold']),r['method'],r['id']))==r['pred'])} for r in predictions]
        csv_write(out/'historical_baseline_comparison.csv',rows)
    return predictions


def source_hist_csv(out):
    p=out/'historical_matched_oof_predictions.csv'
    if not p.exists():return None
    with p.open(newline='',encoding='utf-8') as f:return [r for r in csv.DictReader(f) if r.get('arm') in ('DrCIF','RDST')] if 'arm' in p.read_text(encoding='utf-8').splitlines()[0] else list(csv.DictReader(p.open(newline='',encoding='utf-8')))


def evaluate_mapping(frames,folds,original,cfg,store,out):
    """Identical number of bins, descriptors, penalty grid, folds; vary only mapping."""
    labels=sorted({f['y'] for f in frames});search=[];selected=[];preds=[]
    for fold,(tr,te) in enumerate(folds):
        inner=split_subjects(frames,tr,original['inner_folds'],original['seed']+fold+1)
        for view in cfg['ha_views']:
            for mapping in cfg['mappings']:
                ranked=[]
                for penalty in cfg['pg_penalties']:
                    c={'view':view,'method':'linguistic','partition':mapping,'states':cfg['states'],'B':cfg['B'],'l1':penalty,'lg':penalty/2}
                    cand=digest(c)[:12];ys=[];ps=[]
                    for j,(it,iv) in enumerate(inner):
                        def build():
                            m=Model(c,dict(original,max_cycles=cfg['max_cycles'],threads=cfg['threads'])).fit([frames[i] for i in it]);p=m.predict([frames[i] for i in iv]);return list(map(str,p))
                        predictions=store.task(f'ha_inner/{fold}/{view}/{mapping}/{cand}/{j}',build)
                        ys.extend([frames[i]['y'] for i in iv]);ps.extend(predictions)
                    score=f1(ys,ps,labels);ranked.append((score,cand,c))
                    search.append({'fold':fold,'view':view,'mapping':mapping,'candidate':cand,'inner_macro_f1':score,'penalty':penalty})
                score,cand,c=sorted(ranked,key=lambda r:(-r[0],r[1]))[0]
                key=f'ha_outer/{fold}/{view}/{mapping}'
                m=store.task(key,lambda:Model(c,dict(original,max_cycles=cfg['max_cycles'],threads=cfg['threads'])).fit([frames[i] for i in tr]))
                p=list(map(str,m.predict([frames[i] for i in te])))
                selected.append({'fold':fold,'view':view,'mapping':mapping,'candidate':cand,'inner_macro_f1':score,'n_descriptors':len(m.names),'active_descriptors':m.active(),'checkpoint_sha256':sha(store.folder/(key.replace('/','__')+'.joblib'))})
                preds.extend([{'fold':fold,'view':view,'mapping':mapping,'id':frames[i]['id'],'subject':frames[i]['subject'],'y':frames[i]['y'],'pred':v} for i,v in zip(te,p)])
                for name,rs in [('ha_inner_search',search),('ha_selected',selected),('ha_outer_predictions',preds)]:csv_write(out/(name+'.csv'),rs)
    counts={}
    for r in selected:counts.setdefault((r['fold'],r['view']),{})[r['mapping']]=r['n_descriptors']
    for k,values in counts.items():
        if set(values)!={'ha_soft','quantile'} or values['ha_soft']!=values['quantile']:
            raise AssertionError(f'Non-matched descriptor dimension for {k}: {values}')
    # Paired split-locked difference; only accept as superior when the CI excludes zero.
    contrasts=[]
    for view in cfg['ha_views']:
        a=[r for r in preds if r['view']==view and r['mapping']=='ha_soft']
        b=[r for r in preds if r['view']==view and r['mapping']=='quantile']
        assert [(r['fold'],r['id'],r['subject'],r['y']) for r in a]==[(r['fold'],r['id'],r['subject'],r['y']) for r in b]
        for level in ('segment','subject'):
            result=paired([dict(r,arm='HA') for r in a],[dict(r,arm='quantile') for r in b],cfg['bootstrap'],cfg['seed'],level)
            contrasts.append({'view':view,'level':level,**result})
    csv_write(out/'ha_paired_contrasts.csv',contrasts)
    return contrasts


def main():
    p=argparse.ArgumentParser();p.add_argument('--source-results',required=True);p.add_argument('--config',default='configs/independent_validation.json');p.add_argument('--out',default='runs_independent');p.add_argument('--stage',choices=['tensor','baseline','ha','all'],default='all');p.add_argument('--stop-after',type=int,default=0);a=p.parse_args()
    source=Path(a.source_results).resolve();cfg=json.loads(Path(a.config).read_text(encoding='utf-8'));frames,ident,receipts=load_prepared(source)
    original=ident['config'];assert original['data_mode']=='polliradar'
    assert len({f['subject'] for f in frames})==original['expected_subjects']
    assert cfg['max_cycles']<=original['max_cycles']
    folds=split_subjects(frames,np.arange(len(frames)),original['outer_folds'],original['seed'])
    manifest={'source_identity_sha256':sha(source/'run_identity.json'),'config':cfg,'source_receipts':receipts,'code_sha256':sha(__file__),'python':platform.python_version()}
    out=(ROOT/a.out/('locked_'+digest(manifest)[:16])).resolve()
    with run_lock(out),threadpool_limits(limits=cfg['threads']):
        json_write(out/'run_manifest.json',manifest);store=Store(out/'checkpoints',a.stop_after)
        done=out/'COMPLETE.json';done.unlink(missing_ok=True)
        stages=['baseline','ha'] if a.stage=='all' else [a.stage]
        try:
            for stage in stages:
                if stage=='tensor':audit_tensors_only(frames,folds,cfg,out)
                elif stage=='baseline':evaluate_baselines(frames,folds,original,cfg,store,out)
                else:evaluate_mapping(frames,folds,original,cfg,store,out)
                json_write(out/(stage+'_COMPLETE.json'),{'stage':stage,'n_subjects':len({f['subject'] for f in frames}),'new_jobs':store.completed,'reused_jobs':store.reused})
            json_write(done,{'stages':stages,'completed':True,'no_test_set_model_selection':True})
            print('DONE',out)
        except InterruptedError:
            json_write(out/'INTERRUPTED.json',{'resume':'Repeat identical command'});raise SystemExit(75)
        except Exception as e:
            json_write(out/'ERROR.json',{'error':str(e),'type':type(e).__name__});raise
if __name__=='__main__':main()
